# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import core_map_helper


def _tc_fused_init_kernel(val_ref, valid_ref, out_ref):
    mask = valid_ref[...][:, None] > 0
    out_ref[...] = jnp.where(mask, val_ref[...], 0.0).astype(out_ref.dtype)


def _tc_fused_add_kernel(val_ref, valid_ref, buf_ref, out_ref):
    mask = valid_ref[...][:, None] > 0
    val = jnp.where(mask, val_ref[...], 0.0)
    out_ref[...] = (val + buf_ref[...]).astype(out_ref.dtype)


@functools.lru_cache(maxsize=None)
def _get_tc_init_call(
    num_rows: int, hidden_size: int, bm: int, dtype: jnp.dtype
):
    return pl.pallas_call(
        _tc_fused_init_kernel,
        out_shape=jax.ShapeDtypeStruct((num_rows, hidden_size), dtype),
        grid=(num_rows // bm,),
        in_specs=[
            pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((bm,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
    )


@functools.lru_cache(maxsize=None)
def _get_tc_add_call(
    num_rows: int, hidden_size: int, bm: int, dtype: jnp.dtype
):
    return pl.pallas_call(
        _tc_fused_add_kernel,
        out_shape=jax.ShapeDtypeStruct((num_rows, hidden_size), dtype),
        grid=(num_rows // bm,),
        in_specs=[
            pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
            pl.BlockSpec((bm,), lambda i: (i,)),
            pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
    )


def _select_block_m(num_rows: int, target_bm: int = 1024) -> int:
    bm = min(target_bm, num_rows)
    if num_rows % bm == 0:
        return bm
    for cand in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 1):
        if cand <= num_rows and num_rows % cand == 0:
            return cand
    return 1


def _local_shard_lookup(
    table: jax.Array, idx: jax.Array, rank_offset: jax.Array, table_size: int
) -> tuple[jax.Array, jax.Array]:
    local_idx = idx - rank_offset
    valid = ((local_idx >= 0) & (local_idx < table_size)).astype(jnp.int32)
    clipped = jnp.clip(local_idx, 0, table_size - 1)
    return table[clipped], valid


def sharded_ragged_gather_v2(
    table: jax.Array,
    index: jax.Array,
    *,
    axis_name: str = "rank",
    bm: int = 1024,
) -> jax.Array:
    """Distributed vocabulary-sharded embedding lookup with fused VMEM masking.

    Avoids the 3-kernel launch penalty (`jnp.clip` -> SparseCore `ragged_gather_v2`
    -> TensorCore `jnp.where` -> `psum_scatter`) when `table` is sharded across
    `axis_name` and out-of-shard queries must be zeroed and reduce-scattered:
    - Small gather payloads (`L_total * D <= 8192 * 256`): runs `all_gather` +
      local gather + fused Pallas VMEM mask + `psum_scatter`.
    - Medium/Large gather payloads (`L_total >= 16384`): overlaps bidirectional
      ICI ring `ppermute` with double-buffered local HBM lookup and fused Pallas
      TensorCore VMEM mask-and-accumulate.
    """
    v_local, d = table.shape
    l_local = index.shape[0]
    n_devices = lax.axis_size(axis_name)
    rank = lax.axis_index(axis_name)
    rank_offset = rank * v_local

    index_all = lax.all_gather(index, axis_name, axis=0, tiled=True)

    if l_local * n_devices * d <= 8192 * 256 or l_local % 2 != 0:
        val, valid = _local_shard_lookup(table, index_all, rank_offset, v_local)
        l_total = index_all.shape[0]
        step_bm = _select_block_m(l_total, bm)
        masked = _get_tc_init_call(l_total, d, step_bm, table.dtype)(val, valid)
        return lax.psum_scatter(
            masked, axis_name, scatter_dimension=0, tiled=True
        )

    indices = index_all.reshape(n_devices, l_local)
    l_half = l_local // 2
    step_bm = _select_block_m(l_half, bm)
    init_fn = _get_tc_init_call(l_half, d, step_bm, table.dtype)
    add_fn = _get_tc_add_call(l_half, d, step_bm, table.dtype)

    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1 + n_devices) % n_devices) for i in range(n_devices)]

    c_0 = (rank - 1 + n_devices) % n_devices
    val_0, valid_0 = _local_shard_lookup(
        table, indices[c_0, :l_half], rank_offset, v_local
    )
    buf_0 = init_fn(val_0, valid_0)

    c_1 = (rank + 1) % n_devices
    val_1, valid_1 = _local_shard_lookup(
        table, indices[c_1, l_half:], rank_offset, v_local
    )
    buf_1 = init_fn(val_1, valid_1)

    c_next_0 = (rank - 2 + n_devices) % n_devices
    val_next_0, valid_next_0 = _local_shard_lookup(
        table, indices[c_next_0, :l_half], rank_offset, v_local
    )
    c_next_1 = (rank + 2) % n_devices
    val_next_1, valid_next_1 = _local_shard_lookup(
        table, indices[c_next_1, l_half:], rank_offset, v_local
    )

    for s in range(n_devices - 1):
        buf_recv_0 = lax.ppermute(buf_0, axis_name, perm_right)
        buf_recv_1 = lax.ppermute(buf_1, axis_name, perm_left)

        val_cur_0, valid_cur_0 = val_next_0, valid_next_0
        val_cur_1, valid_cur_1 = val_next_1, valid_next_1

        if s < n_devices - 2:
            c_next_0 = (rank - 3 - s + 10 * n_devices) % n_devices
            val_next_0, valid_next_0 = _local_shard_lookup(
                table, indices[c_next_0, :l_half], rank_offset, v_local
            )
            c_next_1 = (rank + 3 + s) % n_devices
            val_next_1, valid_next_1 = _local_shard_lookup(
                table, indices[c_next_1, l_half:], rank_offset, v_local
            )

        buf_0 = add_fn(val_cur_0, valid_cur_0, buf_recv_0)
        buf_1 = add_fn(val_cur_1, valid_cur_1, buf_recv_1)

    return jnp.concatenate([buf_0, buf_1], axis=0)


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and hidden_size divisibility."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            target_bytes = (256 * 1024) * 0.8
        case 7:
            target_bytes = (512 * 1024) * 0.8
        case _:
            target_bytes = (128 * 1024) * 0.8

    # Calculate max safe column size based on VMEM budget and aligne to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = int((target_bytes // bytes_per_col) // 128) * 128

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


def main_kernel_v2(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    col_size: int | None = None,
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    dtype_bits = jax.dtypes.itemsize_bits(out_hbm_ref.dtype)
    packing = 32 // dtype_bits
    if col_size is None:
        col_size = calculate_col_size(hidden_size, packing)

    assert isinstance(hidden_size,
                      int), f"hidden_size must be int, got {type(hidden_size)}"
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    recv_sem = sem_ref.at[0]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1],
                                       recv_sem)
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()

    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    block_start = start // block_size
    block_end = pl.cdiv(end, block_size)
    num_blocks = block_end - block_start
    num_blocks = jnp.where(end <= start, 0, num_blocks)

    num_cols = pl.cdiv(hidden_size, col_size)

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore `.bitcast()` leverages hardware Row-Packing for 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)

    num_phys_cols = col_size

    def col_loop(col_base, gather_ref, out_ref, idx_rem, unpack_col_chunk):
        col_slice = pl.ds(col_base, unpack_col_chunk)
        if packing == 1:
            gather_dt = gather_ref.bitcast(dtype)
            out_dt = out_ref.bitcast(dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        else:
            # Manual bitwise extraction and packing for packing >= 2 (bfloat16, int8, int4)
            # bf16: 0xFFFF, int8: 0xFF, int4: 0xF
            mask = (1 << dtype_bits) - 1
            shift_multiplier = dtype_bits.bit_length() - 1
            for i in range(num_simd_lanes // packing):
                packed_row = jnp.zeros((1, unpack_col_chunk), dtype=jnp.int32)
                for j in range(packing):
                    k = i * packing + j
                    dynamic_shift = jnp.left_shift(idx_rem[k],
                                                   shift_multiplier)
                    val = jnp.bitwise_right_shift(
                        gather_ref[pl.ds(k, 1), col_slice], dynamic_shift)
                    val = jnp.bitwise_and(val, mask)
                    pack_shift = j * dtype_bits
                    packed_row = jnp.bitwise_or(
                        packed_row, jnp.left_shift(val, pack_shift))
                out_ref[pl.ds(i, 1), col_slice] = packed_row

    def inner_pipeline(gather_ref, out_ref, idx_ref, unpack_col_chunk):
        row_slice = pl.ds(
            pl.program_id(0) * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        if packing > 1:
            # Equivalent to `subchunk_idxs % packing`
            idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)

        col_loop_fn = functools.partial(
            col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            unpack_col_chunk=unpack_col_chunk,
        )
        plsc.parallel_loop(0, num_phys_cols,
                           step=unpack_col_chunk)(col_loop_fn)

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        b_global = b + block_start

        unpack_col_chunk = 128
        assert num_phys_cols % unpack_col_chunk == 0
        shift_amount = packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(inner_pipeline,
                              idx_ref=idx_ref,
                              unpack_col_chunk=unpack_col_chunk),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)
                                ],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (row_subchunk_size // packing, num_phys_cols),
                lambda r, col_id: (
                    (b_global * num_cores + core_index) * num_row_subchunks +
                    r,
                    col_id,
                ),
            ),
        )(in_hbm_i32, out_hbm_i32)

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks, ),
        in_specs=pl.BlockSpec(
            (row_chunk_size, ),
            lambda b: ((b + block_start) * num_cores + core_index, ),
        ),
    )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=("num_row_subchunks", "col_size", "reduce_scatter_axis"),
)
def ragged_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    *,
    num_row_subchunks: int | None = None,
    col_size: int | None = None,
    invalid_mask: jax.Array | None = None,
    reduce_scatter_axis: str | None = None,
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec.

    When `invalid_mask` or `reduce_scatter_axis` is provided (e.g., distributed
    vocabulary-sharded embedding lookup with out-of-shard queries), uses the
    fused TensorCore/VPU gather + VMEM masking fast path to avoid launching 3
    separate kernels (`jnp.clip` -> SparseCore `ragged_gather_v2` -> TensorCore
    `jnp.where`).
    """

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if invalid_mask is not None or reduce_scatter_axis is not None:
        vocab_size, hidden_size = x.shape
        out_size = indices.size
        valid = (
            ~invalid_mask
            if invalid_mask is not None
            else ((indices >= 0) & (indices < vocab_size))
        ).astype(jnp.int32)
        clipped = jnp.clip(indices, 0, vocab_size - 1)
        raw = x[clipped]
        bm = _select_block_m(out_size, 1024)
        masked = _get_tc_init_call(out_size, hidden_size, bm, x.dtype)(
            raw, valid
        )
        if reduce_scatter_axis is not None:
            return lax.psum_scatter(
                masked, reduce_scatter_axis, scatter_dimension=0, tiled=True
            )
        return masked

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    # any data type with a size of {4,8,16,32} should be fine
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    if dtype_bits not in (4, 8, 16, 32):
        raise ValueError(
            f"dtype bit width must be one of 4, 8, 16, or 32, but got {dtype_bits} ({dtype})"
        )

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    packing = 32 // dtype_bits
    max_col_size = calculate_col_size(hidden_size, packing)
    if col_size is None:
        col_size = max_col_size
    else:
        col_size = max(
            128, (min(col_size, hidden_size, max_col_size) // 128) * 128
        )

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    # Calculate ideal num_row_subchunks to avoid too much padding overhead.
    if num_row_subchunks is None:
        num_row_subchunks = max(
            1, min(4, (out_size + base_block_size - 1) // base_block_size)
        )
    else:
        num_row_subchunks = max(1, int(num_row_subchunks))

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = (
        (out_size + block_size - 1) // block_size
    ) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_helper.kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            col_size=col_size,
        ),
        out_type=jax.ShapeDtypeStruct(
            (out_size + out_pad_size, aligned_hidden_size), dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)[:out_size, :hidden_size]
