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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def find_valid_row_chunk_size(
    idx_size: int,
    sc_info,
    single_sc: bool = False,
    preferred_row_chunk_size: int = 512,
) -> int:
    """Finds a tiled row chunk that evenly divides ``idx_size``.

    SparseCore stages the bf16 weights through a 256-element tiled VMEM
    reference.  Smaller chunks can satisfy the arithmetic row-wave check but
    fail Mosaic verification when the pipeline slices that reference.
    """
    num_cores = 1 if single_sc else sc_info.num_cores
    num_subcores = sc_info.num_subcores
    total_subcores = num_cores * num_subcores
    num_lanes = sc_info.num_lanes
    for chunk_size in dict.fromkeys([preferred_row_chunk_size, 512, 256]):
        if chunk_size < 256 or chunk_size % num_lanes != 0:
            continue
        row_wave_size = chunk_size * total_subcores
        if idx_size % row_wave_size == 0:
            return chunk_size
    return 0


def _select_sc_output_dtype(op_dtype, reduce_group_size: int, sc_info):
    """Selects a legal SparseCore output dtype for the reduction topology.

    SparseCore HBM blocks are addressed in 32-bit units.  On v6e, a top-k-8
    reduction over its eight SIMD lanes produces one output row per step.  A
    bf16 output block would need half of a packed 32-bit row and therefore has
    an invalid zero-row BlockSpec.  Keep the bf16 input gather, but emit the
    fp32 accumulator and cast the much smaller reduced result back to bf16 in
    the wrapper.
    """
    out_rows_per_step = sc_info.num_lanes // reduce_group_size
    output_packing = 32 // jax.dtypes.itemsize_bits(op_dtype)
    if out_rows_per_step // output_packing >= 1:
        return op_dtype
    if out_rows_per_step >= 1:
        return jnp.float32
    return None


def is_compatible(
    op: jax.Array,
    idx: jax.Array,
    reduce_group_size: int,
    row_chunk_size: int | None = None,
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

    # A bf16 input need not force a bf16 SparseCore output.  In particular,
    # v6e top-k-8 emits one row per SIMD step, which cannot be represented as
    # half of a packed bf16 row.  It can legally emit one fp32 accumulator row
    # and cast the reduced output in the wrapper.
    if _select_sc_output_dtype(op.dtype, reduce_group_size, sc_info) is None:
        return False

    if row_chunk_size is None:
        chunk = find_valid_row_chunk_size(idx.size, sc_info, single_sc)
        if chunk == 0 and not single_sc:
            chunk = find_valid_row_chunk_size(idx.size, sc_info, True)
        if chunk == 0:
            return False
    else:
        num_cores = 1 if single_sc else sc_info.num_cores
        num_subcores = sc_info.num_subcores
        if (row_chunk_size < 256
                or row_chunk_size % sc_info.num_lanes != 0):
            return False
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
    output_dtype: jnp.dtype | None = None,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand ``op`` based on ``idx``, sums them
  up, and scatters the results. The gather and add operations are performed
  in fp32, and the results are written back in bf16.

  Equivalent JAX code::

    gathered = op[idx, :]
    if topk_weights is not None:
      flat_weights = topk_weights.flatten()
      gathered = gathered * flat_weights[:, None].astype(jnp.float32)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(jnp.float32), axis=1).astype(jnp.bfloat16)

  Args:
    op: The operand matrix [B, K] in f32 or bf16 to gather from and reduce.
    idx: The indices [M,] in int32 guiding the gather.
    topk_weights: Optional weights [M // 128, 128] in bf16 to apply to the
      gathered rows before reduction.
    reduce_group_size: The number of gathered rows to sum per output row.
    output_dtype: Optional HBM output dtype. The accumulation remains fp32.
      v6e bf16 top-k-8 uses fp32 output because a single bf16 result row cannot
      form a complete 32-bit packed SparseCore output row.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing. Must be a
      multiple of the SparseCore SIMD lane count.
    topk_wgt_zero_nan: If True, treat zero ``topk_weights`` as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The reduced result with ``output_dtype`` and shape
    [M / reduce_group_size, K].
  """

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        raise RuntimeError("SparseCore is not available on this TPU version.")

    [M] = idx.shape
    _, K = op.shape
    M_out = M // reduce_group_size
    output_dtype = op.dtype if output_dtype is None else output_dtype

    if topk_weights is not None:
        topk_weights = topk_weights.flatten()

    @jax.jit
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((M_out, K), output_dtype),
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
        input_packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
        output_packing = 32 // jax.dtypes.itemsize_bits(output_dtype)

        subcore_first_row_chunk = (lax.axis_index(
            ("core", "subcore")) * num_row_chunks)

        in_spec = pl.BlockSpec((row_chunk_size, ), lambda i:
                               (subcore_first_row_chunk + i, ))
        in_specs = (in_spec, ) * (1 + (weights_hbm_ref is not None))

        @functools.partial(pltpu.emit_pipeline,
                           grid=(num_row_chunks, ),
                           in_specs=in_specs)
        def idx_pipeline(idx_ref, weights_ref=None):
            row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)

            row_subchunk_size = sc_info.num_lanes
            out_rows_per_step = row_subchunk_size // reduce_group_size
            assert reduce_group_size * out_rows_per_step == sc_info.num_lanes
            num_row_subchunks = row_chunk_size // row_subchunk_size
            if row_chunk_size % row_subchunk_size:
                raise ValueError(
                    f"row_chunk_size needs to be a multiple of {row_subchunk_size}, but"
                    f" got {row_chunk_size}")

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, num_col_chunks),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), col_chunk_size),
                    lambda r, c: (
                        lax.div(
                            idx_ref[pl.ds(r * row_subchunk_size,
                                          row_subchunk_size)],
                            input_packing,
                        ),
                        c,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (out_rows_per_step // output_packing, col_chunk_size),
                    lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
                ),
            )
            def data_pipeline(gather_ref, out_ref):
                gather_ref = gather_ref.bitcast(op.dtype)
                out_ref = out_ref.bitcast(output_dtype)

                row_slice = pl.ds(
                    pl.program_id(0) * row_subchunk_size, row_subchunk_size)
                subchunk_idxs = idx_ref[row_slice]
                weights = (None if weights_ref is None else
                           weights_ref[row_slice].astype(jnp.float32))

                unpack_col_chunk = 32  # 32 seems to works best when tuning.

                @plsc.parallel_loop(0, col_chunk_size, step=unpack_col_chunk)
                def _(col_base):
                    accs = []
                    for reduce_group in range(out_rows_per_step):
                        row_datas = []
                        for row_in_group in range(reduce_group_size):
                            row = reduce_group * reduce_group_size + row_in_group
                            row_data = gather_ref[
                                pl.ds(row * input_packing, input_packing),
                                pl.ds(col_base, unpack_col_chunk),
                            ].astype(jnp.float32)
                            if input_packing == 1:
                                row_data = row_data[0]
                            else:
                                assert input_packing == 2
                                row_data = jnp.where(
                                    lax.bitwise_and(subchunk_idxs[row],
                                                    1) == 0,
                                    row_data[0],
                                    row_data[1],
                                )
                            if weights is not None:
                                row_data *= weights[row]
                                if topk_wgt_zero_nan:
                                    row_data = jnp.where(
                                        weights[row] == 0.0,
                                        jnp.zeros_like(row_data), row_data)
                            row_datas.append(row_data)

                        # Tree reduction to reduce critical path and stalls
                        while len(row_datas) > 1:
                            next_level = []
                            for i in range(0, len(row_datas), 2):
                                if i + 1 < len(row_datas):
                                    next_level.append(row_datas[i] +
                                                      row_datas[i + 1])
                                else:
                                    next_level.append(row_datas[i])
                            row_datas = next_level
                        accs.append(row_datas[0])
                    out = jnp.stack(accs, axis=0).astype(output_dtype)
                    out_ref[:, pl.ds(col_base, unpack_col_chunk)] = out

            data_pipeline(in_hbm_ref.bitcast(jnp.int32),
                          out_hbm_ref.bitcast(jnp.int32))

        idx_pipeline(
            idx_hbm_ref,
            *([weights_hbm_ref] if weights_hbm_ref is not None else []))

    return kernel(op, idx, topk_weights)  # pylint: disable=no-value-for-parameter


def _jax_fallback(x,
                  indices,
                  topk_weights,
                  reduce_group_size,
                  topk_wgt_zero_nan=False):
    token_hidden_full = x[indices]
    cur_sorted = token_hidden_full.reshape(
        (-1, reduce_group_size, x.shape[-1]))
    # topk_weights is already 2D [tokens, reduce_group_size]
    cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
    # Accumulate in float32 to match reference precision and Pallas kernel
    # behavior.
    if topk_wgt_zero_nan:
        cur_weighted = jnp.where(
            cur_topk_weights == 0.0,
            0.0,
            cur_sorted.astype(jnp.float32) *
            cur_topk_weights.astype(jnp.float32),
        )
    else:
        cur_weighted = cur_sorted.astype(
            jnp.float32) * cur_topk_weights.astype(jnp.float32)
    out = cur_weighted.sum(axis=-2)
    return out.astype(x.dtype)


@jax.jit(static_argnames=("reduce_group_size", "topk_wgt_zero_nan",
                          "row_chunk_size"))
def dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
    row_chunk_size: int | None = None,
) -> jax.Array:
    """Wrapper that redirects to Pallas dense gather reduce kernel if constraints are met.

  Otherwise, it falls back to the JAX baseline.

  Args:
    x: Input array [out_size, hidden_size].
    indices: Gather indices [out_size].
    topk_weights: 2D weights [tokens, reduce_group_size], where tokens *
      reduce_group_size = out_size.
    reduce_group_size: Group size for reduction (topk).
    topk_wgt_zero_nan: If True, treat zero weights as indicators of NaN during
      multiplication, resulting in zero output.
    row_chunk_size: Optional row chunk size for SparseCore. If None, dynamically selected.
  """
    if is_compatible(x,
                     indices,
                     reduce_group_size,
                     row_chunk_size=row_chunk_size):
        sc_info = pltpu.get_tpu_info().sparse_core
        output_dtype = _select_sc_output_dtype(x.dtype, reduce_group_size,
                                               sc_info)
        chosen_single_sc = False
        if row_chunk_size is None:
            chosen_row_chunk_size = find_valid_row_chunk_size(
                indices.size, sc_info)
            if chosen_row_chunk_size == 0:
                chosen_single_sc = True
                chosen_row_chunk_size = find_valid_row_chunk_size(
                    indices.size, sc_info, single_sc=True)
        else:
            chosen_row_chunk_size = row_chunk_size
        K = x.shape[-1]
        col_chunk_size = (min(2048, K) // 128) * 128
        while col_chunk_size > 0:
            if K % col_chunk_size == 0:
                break
            col_chunk_size -= 128
        if col_chunk_size > 0 and chosen_row_chunk_size > 0:
            # Pallas kernel expects 1D weights
            out = _sc_gather_reduce(
                x,
                indices,
                topk_weights.reshape(-1),
                reduce_group_size=reduce_group_size,
                output_dtype=output_dtype,
                single_sc=chosen_single_sc,
                col_chunk_size=col_chunk_size,
                row_chunk_size=chosen_row_chunk_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            )
            return out.astype(x.dtype)
    # Fallback to JAX baseline
    return _jax_fallback(x, indices, topk_weights, reduce_group_size,
                         topk_wgt_zero_nan)
