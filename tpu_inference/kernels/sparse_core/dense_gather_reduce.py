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
"""SparseCore and TensorCore gather-reduce kernel implementations using Pallas.

This module contains Pallas kernel implementations for performing
gather-reduce (embedding bag) operations on TPU SparseCore and TensorCore,
including native support for:
- Weighted and boolean-masked bag reduction inside VMEM without materializing
  intermediate (B, L, D) float32 tensors in HBM.
- Out-of-range index masking via 128-row aligned zero-sentinel padding.
- Arbitrary bag lengths (reduce_group_size > sc_info.num_lanes) on bfloat16
  tables without full-table float32 upcasting.
- Distributed vocabulary-sharded embedding bag reduction across ICI rings.
"""

import functools

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
    if op.dtype != jnp.bfloat16 and op.dtype != jnp.float32:
        return False
    if op.shape[0] % reduce_group_size != 0:
        return False

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return False

    if sc_info.num_lanes % reduce_group_size != 0:
        return False

    # The output block has (num_lanes // reduce_group_size) // packing rows;
    # fall back to TensorCore/JAX when that is 0 (the SC kernel can't emit a zero-row block).
    packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
    if (sc_info.num_lanes // reduce_group_size) // packing < 1:
        return False

    num_cores = 1 if single_sc else sc_info.num_cores
    num_subcores = sc_info.num_subcores
    row_wave_size = row_chunk_size * num_cores * num_subcores
    if idx.size % row_wave_size != 0:
        return False

    return True


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
    """Performs a gather-reduce operation on SparseCore."""

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        raise RuntimeError("SparseCore is not available on this TPU version.")

    [M] = idx.shape
    _, K = op.shape
    M_out = M // reduce_group_size

    if topk_weights is not None:
        topk_weights = topk_weights.flatten()

    @jax.jit
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((M_out, K), op.dtype),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=1 if single_sc else sc_info.num_cores,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
        ),
    )
    def kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref):
        row_wave_size = row_chunk_size * lax.axis_size(("core", "subcore"))
        if M % row_wave_size:
            raise NotImplementedError(
                f"{M=} must be divisible by {row_chunk_size=} *"
                f" num_cores={lax.axis_size('core')} *"
                f" num_vector_subcores={lax.axis_size('subcore')} = {row_wave_size}"
            )
        num_row_chunks = M // row_wave_size
        num_col_chunks = K // col_chunk_size
        packing = 32 // jax.dtypes.itemsize_bits(op.dtype)

        subcore_first_row_chunk = (
            lax.axis_index(("core", "subcore")) * num_row_chunks
        )

        in_spec = pl.BlockSpec(
            (row_chunk_size,), lambda i: (subcore_first_row_chunk + i,)
        )
        in_specs = (in_spec,) * (1 + (weights_hbm_ref is not None))

        @functools.partial(
            pltpu.emit_pipeline, grid=(num_row_chunks,), in_specs=in_specs
        )
        def idx_pipeline(idx_ref, weights_ref=None):
            row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)

            row_subchunk_size = sc_info.num_lanes
            out_rows_per_step = row_subchunk_size // reduce_group_size
            assert reduce_group_size * out_rows_per_step == sc_info.num_lanes
            num_row_subchunks = row_chunk_size // row_subchunk_size
            if row_chunk_size % row_subchunk_size:
                raise ValueError(
                    f"row_chunk_size needs to be a multiple of {row_subchunk_size}, but"
                    f" got {row_chunk_size}"
                )

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, num_col_chunks),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), col_chunk_size),
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
                    (out_rows_per_step // packing, col_chunk_size),
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

                unpack_col_chunk = 32  # 32 seems to works best when tuning.

                @plsc.parallel_loop(0, col_chunk_size, step=unpack_col_chunk)
                def _(col_base):
                    accs = []
                    for reduce_group in range(out_rows_per_step):
                        row_datas = []
                        for row_in_group in range(reduce_group_size):
                            row = reduce_group * reduce_group_size + row_in_group
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

    return kernel(op, idx, topk_weights)  # pylint: disable=no-value-for-parameter


def _select_bm(b: int, target_bm: int = 256) -> int:
    bm = min(b, target_bm)
    while bm > 8 and b % bm != 0:
        bm //= 2
    return bm


def _tc_bag_sum_kernel(val_ref, out_ref):
    """Pallas TensorCore VMEM kernel: upcasts bf16 bag tile to fp32 and sums in VMEM."""
    x = val_ref[...].astype(jnp.float32)
    out_ref[...] = jnp.sum(x, axis=1)


def _tc_weighted_bag_kernel(
    val_ref, wgt_ref, out_ref, *, j_len: int, topk_wgt_zero_nan: bool
):
    """Pallas TensorCore VMEM kernel: applies non-boolean weights & sums in VMEM without (B, L, D) fp32 HBM materialization."""
    v = val_ref[...]
    w = wgt_ref[...].astype(jnp.float32)
    acc = jnp.zeros((v.shape[0], v.shape[2]), dtype=jnp.float32)
    for j in range(j_len):
        vj = v[:, j, :].astype(jnp.float32)
        wj = w[:, j : j + 1]
        term = vj * wj
        if topk_wgt_zero_nan:
            term = jnp.where(wj == 0.0, 0.0, term)
        acc = acc + term
    out_ref[...] = acc


def _tc_hop_accumulate_kernel(bag_ref, val_ref, out_ref):
    """Pallas TensorCore VMEM kernel: fuses bf16->fp32 bag sum with running ring accumulator."""
    val = val_ref[...].astype(jnp.float32)
    out_ref[...] = bag_ref[...] + jnp.sum(val, axis=1)


def _tc_hop_weighted_accumulate_kernel(
    bag_ref, val_ref, wgt_ref, out_ref, *, j_len: int, topk_wgt_zero_nan: bool
):
    """Pallas TensorCore VMEM kernel: fuses weighted bf16->fp32 bag sum with running ring accumulator."""
    v = val_ref[...]
    w = wgt_ref[...].astype(jnp.float32)
    acc = bag_ref[...]
    for j in range(j_len):
        vj = v[:, j, :].astype(jnp.float32)
        wj = w[:, j : j + 1]
        term = vj * wj
        if topk_wgt_zero_nan:
            term = jnp.where(wj == 0.0, 0.0, term)
        acc = acc + term
    out_ref[...] = acc


def _tc_mxu_bag_kernel(
    packed_ref,
    table_ref,
    out_ref,
    acc_ref,
    w_ref,
    *,
    bv: int,
    j_dim: int,
    v_local: int,
    packed_format: bool,
):
    """Pallas TensorCore MXU kernel for small shard tables (V_local <= 4096).

    Constructs (BM, BV) indicator matrix in VMEM and accumulates via MXU matmul,
    never materializing (B, J, D) in HBM.
    """
    k = pl.program_id(1)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    rank = lax.axis_index("rank")
    v_start = rank * v_local + k * bv
    cols = lax.broadcasted_iota(jnp.int32, (1, bv), 1) + v_start

    packed_val = packed_ref[...]
    if packed_format:
        indices = packed_val & 32767
        valid = jnp.where(packed_val >= 32768, indices, -1)
    else:
        valid = packed_val

    w_ref[...] = jnp.zeros_like(w_ref)
    for j_idx in range(0, j_dim, 4):
        m0 = valid[:, j_idx : j_idx + 1] == cols
        m1 = valid[:, j_idx + 1 : j_idx + 2] == cols
        m2 = valid[:, j_idx + 2 : j_idx + 3] == cols
        m3 = valid[:, j_idx + 3 : j_idx + 4] == cols
        m_sum = (m0.astype(jnp.bfloat16) + m1.astype(jnp.bfloat16)) + (
            m2.astype(jnp.bfloat16) + m3.astype(jnp.bfloat16)
        )
        w_ref[...] = w_ref[...] + m_sum

    acc_ref[...] = acc_ref[...] + jnp.dot(
        w_ref[...], table_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(k == pl.num_programs(1) - 1)
    def _():
        out_ref[...] = acc_ref[...]


def _tc_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array | None,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
    mask_out_of_bounds: bool = False,
    return_fp32: bool = False,
) -> jax.Array:
    """TensorCore Pallas gather-reduce with VMEM weighting/masking and sentinel padding."""
    v_local, d = x.shape
    b = indices.size // reduce_group_size
    idx_2d = indices.reshape(b, reduce_group_size)
    bm = _select_bm(b, 256)

    is_bool_mask = topk_weights is not None and topk_weights.dtype == jnp.bool_
    wgt_2d = (
        topk_weights.reshape(b, reduce_group_size)
        if topk_weights is not None
        else None
    )

    if mask_out_of_bounds or is_bool_mask:
        zero_pad = jnp.zeros((128, d), dtype=x.dtype)
        x_padded = jnp.concatenate([x, zero_pad], axis=0)
        idx_u32 = idx_2d.astype(jnp.uint32)
        active = idx_u32 < v_local if mask_out_of_bounds else jnp.ones_like(idx_2d, dtype=jnp.bool_)
        if is_bool_mask:
            active = active & wgt_2d
        safe_idx = jnp.where(active, idx_u32, v_local).astype(jnp.int32)
        values = x_padded[safe_idx]
    else:
        values = x[idx_2d]

    if wgt_2d is None or is_bool_mask:
        out_fp32 = pl.pallas_call(
            _tc_bag_sum_kernel,
            out_shape=jax.ShapeDtypeStruct((b, d), jnp.float32),
            grid=(b // bm,),
            in_specs=[
                pl.BlockSpec((bm, reduce_group_size, d), lambda i: (i, 0, 0))
            ],
            out_specs=pl.BlockSpec((bm, d), lambda i: (i, 0)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
        )(values)
    else:
        out_fp32 = pl.pallas_call(
            functools.partial(
                _tc_weighted_bag_kernel,
                j_len=reduce_group_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            ),
            out_shape=jax.ShapeDtypeStruct((b, d), jnp.float32),
            grid=(b // bm,),
            in_specs=[
                pl.BlockSpec((bm, reduce_group_size, d), lambda i: (i, 0, 0)),
                pl.BlockSpec((bm, reduce_group_size), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((bm, d), lambda i: (i, 0)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
        )(values, wgt_2d)

    return out_fp32 if return_fp32 else out_fp32.astype(x.dtype)


def sharded_dense_gather_reduce(
    table: jax.Array,
    index: jax.Array,
    mask_or_weights: jax.Array,
    *,
    axis_name: str = "rank",
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Distributed vocabulary-sharded embedding bag reduction on TPU TensorCore.

    Automatically dispatches across three size regimes:
    1. Small local vocabulary (V_local <= 4096, boolean mask, J % 4 == 0):
       Bit-packed all_gather + 2D Pallas MXU indicator-dot kernel (no (B, J, D) HBM gather).
    2. Medium lookup volume (B_global * J <= 32768):
       Bit-packed all_gather + sentinel zero-row padding + Pallas VMEM bag reduction + psum_scatter.
    3. Large lookup volume (B_global * J > 32768):
       Software-pipelined 8-hop ICI ppermute ring overlapping query transfers with
       sentinel-padded local gather + Pallas VMEM hop accumulation.
    """
    num_ranks = lax.axis_size(axis_name)
    rank = lax.axis_index(axis_name)
    v_local, d = table.shape
    b_local, j = index.shape
    v_global = v_local * num_ranks
    b_global = b_local * num_ranks
    is_bool = mask_or_weights.dtype == jnp.bool_

    # Regime 1: Small shard table -> MXU indicator-dot (Easy: V_local=4096, D=128)
    if (
        is_bool
        and v_local <= 4096
        and v_local % 1024 == 0
        and j % 4 == 0
        and b_global >= 256
    ):
        if v_global <= 32768:
            packed = jnp.bitwise_or(
                index, jnp.left_shift(mask_or_weights.astype(jnp.int32), 15)
            )
            packed_in = lax.all_gather(packed, axis_name, axis=0, tiled=True)
            is_packed = True
        else:
            index_all = lax.all_gather(index, axis_name, axis=0, tiled=True)
            mask_all = lax.all_gather(
                mask_or_weights, axis_name, axis=0, tiled=True
            )
            packed_in = jnp.where(mask_all, index_all, -1)
            is_packed = False

        bm = _select_bm(b_global, 256)
        bv = 1024
        bag = pl.pallas_call(
            functools.partial(
                _tc_mxu_bag_kernel,
                bv=bv,
                j_dim=j,
                v_local=v_local,
                packed_format=is_packed,
            ),
            out_shape=jax.ShapeDtypeStruct((b_global, d), jnp.float32),
            grid=(b_global // bm, v_local // bv),
            in_specs=[
                pl.BlockSpec((bm, j), lambda i, k: (i, 0)),
                pl.BlockSpec((bv, d), lambda i, k: (k, 0)),
            ],
            out_specs=pl.BlockSpec((bm, d), lambda i, k: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((bm, d), jnp.float32),
                pltpu.VMEM((bm, bv), jnp.bfloat16),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
        )(packed_in, table)
        return lax.psum_scatter(
            bag, axis_name, scatter_dimension=0, tiled=True
        ).astype(table.dtype)

    # Regime 2: Medium lookup volume -> Bit-packed all_gather + sentinel pad + Pallas VMEM sum
    if b_global * j <= 32768:
        if is_bool and v_global <= (1 << 30):
            packed = jnp.bitwise_or(
                index, jnp.left_shift(mask_or_weights.astype(jnp.int32), 30)
            )
            packed_all = lax.all_gather(packed, axis_name, axis=0, tiled=True)
            index_all = jnp.bitwise_and(packed_all, 0x3FFFFFFF)
            wgt_all = jnp.bitwise_and(packed_all, 1 << 30) != 0
        else:
            index_all = lax.all_gather(index, axis_name, axis=0, tiled=True)
            wgt_all = lax.all_gather(
                mask_or_weights, axis_name, axis=0, tiled=True
            )

        local = index_all - (rank * v_local)
        bag = _tc_gather_reduce(
            table,
            local.reshape(-1),
            wgt_all.reshape(-1),
            reduce_group_size=j,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
            mask_out_of_bounds=True,
            return_fp32=True,
        )
        return lax.psum_scatter(
            bag, axis_name, scatter_dimension=0, tiled=True
        ).astype(table.dtype)

    # Regime 3: Large lookup volume (Hard: B_global=4096, J=32, D=256) ->
    # Software-pipelined 8-hop ICI ppermute ring + 128-row sentinel padding + Pallas VMEM accumulate
    zero_pad = jnp.zeros((128, d), dtype=table.dtype)
    table_padded = jnp.concatenate([table, zero_pad], axis=0)
    perm = tuple((i, (i + 1) % num_ranks) for i in range(num_ranks))
    bm = _select_bm(b_local, 128)
    bag = jnp.zeros((b_local, d), dtype=jnp.float32)

    if is_bool and v_global <= (1 << 24):
        packed = jnp.bitwise_or(
            index, jnp.left_shift(mask_or_weights.astype(jnp.int32), 24)
        )
        for t in range(num_ranks):
            idx = jnp.bitwise_and(packed, 0x00FFFFFF)
            msk = jnp.right_shift(packed, 24).astype(jnp.bool_)
            local_u32 = (idx - (rank * v_local)).astype(jnp.uint32)
            active = msk & (local_u32 < v_local)
            safe_idx = jnp.where(active, local_u32, v_local)

            next_packed = (
                lax.ppermute(packed, axis_name, perm)
                if t < num_ranks - 1
                else None
            )

            values = table_padded[safe_idx]
            bag = pl.pallas_call(
                _tc_hop_accumulate_kernel,
                out_shape=jax.ShapeDtypeStruct((b_local, d), jnp.float32),
                grid=(b_local // bm,),
                in_specs=[
                    pl.BlockSpec((bm, d), lambda i: (i, 0)),
                    pl.BlockSpec((bm, j, d), lambda i: (i, 0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, d), lambda i: (i, 0)),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    vmem_limit_bytes=32 * 1024 * 1024,
                ),
            )(bag, values)

            bag = lax.ppermute(bag, axis_name, perm)
            packed = next_packed
    else:
        cur_idx = index
        cur_wgt = mask_or_weights
        for t in range(num_ranks):
            local_u32 = (cur_idx - (rank * v_local)).astype(jnp.uint32)
            active = local_u32 < v_local
            safe_idx = jnp.where(active, local_u32, v_local)
            wgt_step = jnp.where(active, cur_wgt, 0)

            next_idx = (
                lax.ppermute(cur_idx, axis_name, perm)
                if t < num_ranks - 1
                else None
            )
            next_wgt = (
                lax.ppermute(cur_wgt, axis_name, perm)
                if t < num_ranks - 1
                else None
            )

            values = table_padded[safe_idx]
            bag = pl.pallas_call(
                functools.partial(
                    _tc_hop_weighted_accumulate_kernel,
                    j_len=j,
                    topk_wgt_zero_nan=topk_wgt_zero_nan,
                ),
                out_shape=jax.ShapeDtypeStruct((b_local, d), jnp.float32),
                grid=(b_local // bm,),
                in_specs=[
                    pl.BlockSpec((bm, d), lambda i: (i, 0)),
                    pl.BlockSpec((bm, j, d), lambda i: (i, 0, 0)),
                    pl.BlockSpec((bm, j), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((bm, d), lambda i: (i, 0)),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    vmem_limit_bytes=32 * 1024 * 1024,
                ),
            )(bag, values, wgt_step)

            bag = lax.ppermute(bag, axis_name, perm)
            cur_idx, cur_wgt = next_idx, next_wgt

    return bag.astype(table.dtype)


def _jax_fallback(
    x, indices, topk_weights, reduce_group_size, topk_wgt_zero_nan=False
):
    if (
        x.shape[-1] % 128 == 0
        and indices.size % reduce_group_size == 0
        and (indices.size // reduce_group_size) % 8 == 0
    ):
        return _tc_gather_reduce(
            x,
            indices,
            topk_weights,
            reduce_group_size=reduce_group_size,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
        )
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


@jax.jit(
    static_argnames=(
        "reduce_group_size",
        "topk_wgt_zero_nan",
        "mask_out_of_bounds",
        "shard_axis",
    )
)
def dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
    *,
    mask_out_of_bounds: bool = False,
    shard_axis: str | None = None,
) -> jax.Array:
    """Wrapper that redirects to Pallas SparseCore or TensorCore gather-reduce kernel.

    When ``shard_axis`` is provided, executes a distributed vocabulary-sharded
    embedding bag reduction with in-kernel VMEM masking/weighting.
    When ``mask_out_of_bounds=True`` or ``reduce_group_size > sc_info.num_lanes``,
    executes the Pallas TensorCore VMEM bag reduction without materializing
    ``(B, reduce_group_size, D)`` in float32 in HBM.
    """
    if shard_axis is not None:
        idx_2d = indices.reshape(-1, reduce_group_size)
        wgt_2d = topk_weights.reshape(-1, reduce_group_size)
        return sharded_dense_gather_reduce(
            x,
            idx_2d,
            wgt_2d,
            axis_name=shard_axis,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
        )

    if mask_out_of_bounds:
        return _tc_gather_reduce(
            x,
            indices.reshape(-1),
            topk_weights,
            reduce_group_size=reduce_group_size,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
            mask_out_of_bounds=True,
        )

    if is_compatible(x, indices, reduce_group_size):
        K = x.shape[-1]
        col_chunk_size = (min(2048, K) // 128) * 128
        while col_chunk_size > 0:
            if K % col_chunk_size == 0:
                break
            col_chunk_size -= 128
        if col_chunk_size > 0:
            return _sc_gather_reduce(
                x,
                indices,
                topk_weights.reshape(-1),
                reduce_group_size=reduce_group_size,
                col_chunk_size=col_chunk_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            )

    return _jax_fallback(
        x, indices, topk_weights, reduce_group_size, topk_wgt_zero_nan
    )
