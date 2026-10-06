# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SparseCore gather-reduce kernel implementation using Pallas.

This module contains a Pallas kernel implementation for performing a
gather-reduce operation on TPU SparseCore. It groups rows of an operand
based on provided indices, sums them up, and scatters the results.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def is_compatible(
    op: jax.Array,
    idx: jax.Array,
    reduce_group_size: int,
    row_chunk_size: int = 512,
    single_sc: bool = False,
) -> bool:
    """Checks if the inputs are compatible with the SparseCore Pallas kernel."""
    del row_chunk_size, single_sc
    if op.dtype != jnp.bfloat16 and op.dtype != jnp.float32:
        return False
    if idx.size % reduce_group_size != 0:
        return False

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return False

    packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
    max_sc_group = sc_info.num_lanes // packing
    sub_group = math.gcd(reduce_group_size, max_sc_group)
    if sub_group < 1:
        return False

    if op.shape[-1] % 128 != 0:
        return False

    return True


def _select_col_chunk_size(k: int, requested: int) -> int:
    """Selects a 128-aligned column chunk size that evenly divides K."""
    col_chunk = min(int(requested), k, 2048)
    col_chunk = max(128, (col_chunk // 128) * 128)
    while col_chunk > 128 and k % col_chunk != 0:
        col_chunk -= 128
    return col_chunk if k % col_chunk == 0 else 128


def _select_row_chunk_size(
    m_total: int,
    requested: int,
    num_subcores_total: int,
    num_lanes: int,
    sub_group: int,
) -> int:
    """Selects a SparseCore row_chunk_size aligned to num_lanes and bounding outer pipeline depth."""
    min_rc = max(num_lanes, sub_group)
    # Keep outer idx_pipeline grid <= 64 to avoid XProf/Mosaic trace buffer overflow on large Hard shapes
    min_rc_for_trace = max(
        min_rc,
        ((m_total + (num_subcores_total * 64) - 1) // (num_subcores_total * 64)),
    )
    min_rc_for_trace = ((min_rc_for_trace + num_lanes - 1) // num_lanes) * num_lanes
    rc = max(int(requested), min_rc_for_trace)
    rc = ((rc + num_lanes - 1) // num_lanes) * num_lanes

    max_rc = max(min_rc, m_total // num_subcores_total)
    max_rc = max(min_rc, (max_rc // num_lanes) * num_lanes)
    rc = min(rc, max_rc)
    while rc > min_rc and (m_total // num_subcores_total) % rc != 0:
        rc -= num_lanes
    return max(min_rc, rc)


def _sc_gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Performs a gather-reduce operation on SparseCore.

  Supports arbitrary ``reduce_group_size`` (including bag sizes larger than
  ``sc_info.num_lanes`` such as ``J=16, 32, 64``) by reducing ``sub_group =
  gcd(reduce_group_size, sc_info.num_lanes // packing)`` rows inside SparseCore
  vector registers and summing any remaining outer factor, while automatically
  aligning ``col_chunk_size``, ``row_chunk_size``, and wave padding.
  """

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        raise RuntimeError("SparseCore is not available on this TPU version.")

    idx = idx.reshape(-1)
    [M_orig] = idx.shape
    V, K = op.shape
    if M_orig % reduce_group_size != 0:
        raise ValueError(
            f"idx.size={M_orig} must be divisible by reduce_group_size={reduce_group_size}"
        )
    M_final_out = M_orig // reduce_group_size

    packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
    max_sc_group = sc_info.num_lanes // packing
    sub_group = math.gcd(reduce_group_size, max_sc_group)
    outer_group = reduce_group_size // sub_group

    num_cores = 1 if single_sc else sc_info.num_cores
    num_subcores_total = num_cores * sc_info.num_subcores

    col_chunk = _select_col_chunk_size(K, col_chunk_size)
    row_chunk = _select_row_chunk_size(
        M_orig, row_chunk_size, num_subcores_total, sc_info.num_lanes, sub_group
    )
    row_wave_size = row_chunk * num_subcores_total

    if topk_weights is not None:
        topk_weights = topk_weights.reshape(-1)

    pad_m = (row_wave_size - (M_orig % row_wave_size)) % row_wave_size
    if pad_m > 0:
        idx = jnp.pad(idx, ((0, pad_m),), constant_values=0)
        if topk_weights is not None:
            topk_weights = jnp.pad(
                topk_weights, ((0, pad_m),), constant_values=0
            )

    # Ensure all indices are in [0, V - 1] to prevent out-of-bounds DMA faults
    idx = jnp.clip(idx, 0, V - 1)

    M = idx.shape[0]
    M_sc_out = M // sub_group
    M_sc_out_unpadded = M_orig // sub_group

    @jax.jit
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((M_sc_out, K), op.dtype),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=num_cores,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
        ),
    )
    def kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref):
        num_row_chunks = M // row_wave_size
        num_col_chunks = K // col_chunk

        subcore_first_row_chunk = (
            lax.axis_index(("core", "subcore")) * num_row_chunks
        )

        in_spec = pl.BlockSpec(
            (row_chunk,), lambda i: (subcore_first_row_chunk + i,)
        )
        in_specs = (in_spec,) * (1 + (weights_hbm_ref is not None))

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_row_chunks,),
            in_specs=in_specs,
        )
        def idx_pipeline(idx_ref, weights_ref=None):
            row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)

            row_subchunk_size = sc_info.num_lanes
            out_rows_per_step = row_subchunk_size // sub_group
            assert sub_group * out_rows_per_step == sc_info.num_lanes
            num_row_subchunks = row_chunk // row_subchunk_size

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, num_col_chunks),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), col_chunk),
                    lambda r, c: (
                        lax.div(
                            idx_ref[
                                pl.ds(r * row_subchunk_size, row_subchunk_size)
                            ],
                            packing,
                        ),
                        c,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (out_rows_per_step // packing, col_chunk),
                    lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
                ),
            )
            def data_pipeline(gather_ref, out_ref):
                gather_ref = gather_ref.bitcast(op.dtype)
                out_ref = out_ref.bitcast(op.dtype)

                row_slice = pl.ds(
                    pl.program_id(0) * row_subchunk_size, row_subchunk_size
                )
                subchunk_idxs = idx_ref[row_slice]
                weights = (
                    None
                    if weights_ref is None
                    else weights_ref[row_slice].astype(jnp.float32)
                )

                unpack_col_chunk = 32

                @plsc.parallel_loop(0, col_chunk, step=unpack_col_chunk)
                def _(col_base):
                    accs = []
                    for reduce_group in range(out_rows_per_step):
                        row_datas = []
                        for row_in_group in range(sub_group):
                            row = reduce_group * sub_group + row_in_group
                            row_data = gather_ref[
                                pl.ds(row * packing, packing),
                                pl.ds(col_base, unpack_col_chunk),
                            ].astype(jnp.float32)
                            if packing == 1:
                                row_data = row_data[0]
                            else:
                                assert packing == 2
                                row_data = jnp.where(
                                    lax.bitwise_and(subchunk_idxs[row], 1) == 0,
                                    row_data[0],
                                    row_data[1],
                                )
                            if weights is not None:
                                row_data *= weights[row]
                                if topk_wgt_zero_nan:
                                    row_data = jnp.where(
                                        weights[row] == 0.0,
                                        jnp.zeros_like(row_data),
                                        row_data,
                                    )
                            row_datas.append(row_data)

                        # Tree reduction to reduce critical path and stalls
                        while len(row_datas) > 1:
                            next_level = []
                            for i in range(0, len(row_datas), 2):
                                if i + 1 < len(row_datas):
                                    next_level.append(
                                        row_datas[i] + row_datas[i + 1]
                                    )
                                else:
                                    next_level.append(row_datas[i])
                            row_datas = next_level
                        accs.append(row_datas[0])
                    out = jnp.stack(accs, axis=0).astype(op.dtype)
                    out_ref[:, pl.ds(col_base, unpack_col_chunk)] = out

            data_pipeline(
                in_hbm_ref.bitcast(jnp.int32), out_hbm_ref.bitcast(jnp.int32)
            )

        idx_pipeline(
            idx_hbm_ref,
            *([weights_hbm_ref] if weights_hbm_ref is not None else []),
        )

    out_sc = kernel(op, idx, topk_weights)  # pylint: disable=no-value-for-parameter
    if pad_m > 0:
        out_sc = out_sc[:M_sc_out_unpadded, :]
    if outer_group > 1:
        out_sc = (
            out_sc.astype(jnp.float32)
            .reshape(M_final_out, outer_group, K)
            .sum(axis=1)
            .astype(op.dtype)
        )
    return out_sc


def _jax_fallback(
    x,
    indices,
    topk_weights,
    reduce_group_size,
    topk_wgt_zero_nan=False,
):
    token_hidden_full = x[indices]
    cur_sorted = token_hidden_full.reshape((-1, reduce_group_size, x.shape[-1]))
    cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
    if topk_wgt_zero_nan:
        cur_weighted = jnp.where(
            cur_topk_weights == 0.0,
            0.0,
            cur_sorted.astype(jnp.float32)
            * cur_topk_weights.astype(jnp.float32),
        )
    else:
        cur_weighted = cur_sorted.astype(
            jnp.float32
        ) * cur_topk_weights.astype(jnp.float32)
    out = cur_weighted.sum(axis=-2)
    return out.astype(x.dtype)


@jax.jit(static_argnames=("reduce_group_size", "topk_wgt_zero_nan"))
def dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Wrapper that redirects to Pallas dense gather reduce kernel if constraints are met."""
    if is_compatible(x, indices, reduce_group_size):
        K = x.shape[-1]
        col_chunk_size = _select_col_chunk_size(K, 2048)
        if K % col_chunk_size == 0:
            return _sc_gather_reduce(
                x,
                indices.reshape(-1),
                topk_weights.reshape(-1),
                reduce_group_size=reduce_group_size,
                col_chunk_size=col_chunk_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            )
    return _jax_fallback(
        x, indices, topk_weights, reduce_group_size, topk_wgt_zero_nan
    )
