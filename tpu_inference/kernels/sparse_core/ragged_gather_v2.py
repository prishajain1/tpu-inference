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


def _tc_paged_block_gather_kernel(
    pt_ref,
    len_ref,
    *args,
    block: int,
    p: int,
    K: int,
    has_lengths: bool,
):
    c_refs = args[:K]
    out_ref = args[K]

    i = pl.program_id(0)
    j = pl.program_id(1)
    len_i = len_ref[i]
    rem = len_i - j * (K * block)

    token_idx = jnp.arange(block, dtype=jnp.int32).reshape(block, 1)

    for k in range(K):
        page_val = c_refs[k][0]
        if has_lengths:
            val = jnp.clip(rem - k * block, 0, block)
            mask = token_idx < val
            page_val = jnp.where(mask, page_val, 0).astype(out_ref.dtype)
        out_ref[0, pl.ds(k * block, block), :] = page_val


def _make_tc_cache_index_map(k: int, pages: int, p: int, K: int):
    def index_map(i, j, pt_ref, len_ref):
        idx = i * p + K * j + k
        safe_page = jnp.clip(pt_ref[idx], 0, pages - 1)
        return (safe_page, 0, 0)

    return index_map


def _tc_out_index_map(i, j, pt_ref, len_ref):
    return (i, j, 0)


@functools.partial(
    jax.jit,
    static_argnames=("page_size", "pages_per_step"),
)
def paged_block_gather_tc(
    x: jax.Array,
    page_table: jax.Array,
    lengths: jax.Array | None = None,
    *,
    page_size: int = 16,
    pages_per_step: int = 16,
) -> jax.Array:
    """TensorCore DMA block-gather fast path for contiguous (page_size, head_dim) pages."""
    orig_x_shape = x.shape
    pages = orig_x_shape[0]
    if x.ndim == 2:
        assert x.shape[1] % page_size == 0
        hd = x.shape[1] // page_size
        cache_3d = x.reshape(pages, page_size, hd)
    else:
        page_size = orig_x_shape[1]
        hd = int(functools.reduce(lambda a, b: a * b, orig_x_shape[2:], 1))
        cache_3d = x.reshape(pages, page_size, hd)

    if page_table.ndim == 1:
        b, p = 1, page_table.shape[0]
    else:
        b, p = page_table.shape

    K = min(pages_per_step, p)
    while p % K != 0 and K > 1:
        K //= 2

    flat_pt = page_table.reshape(-1).astype(jnp.int32)
    has_lengths = lengths is not None
    if lengths is None:
        lengths_arr = jnp.full((b,), p * page_size, dtype=jnp.int32)
    else:
        lengths_arr = lengths.reshape(b).astype(jnp.int32)

    grid = (b, p // K)
    in_specs = [
        pl.BlockSpec(
            (1, page_size, hd), _make_tc_cache_index_map(k, pages, p, K)
        )
        for k in range(K)
    ]
    out_specs = pl.BlockSpec((1, K * page_size, hd), _tc_out_index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    kernel_fn = functools.partial(
        _tc_paged_block_gather_kernel,
        block=page_size,
        p=p,
        K=K,
        has_lengths=has_lengths,
    )

    out = pl.pallas_call(
        kernel_fn,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((b, p * page_size, hd), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
    )(flat_pt, lengths_arr, *([cache_3d] * K))

    if x.ndim == 2 and page_table.ndim == 1:
        return out.reshape(p, page_size * hd)
    return out.reshape(b, p * page_size, *orig_x_shape[2:])


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
    static_argnames=("num_row_subchunks", "col_size", "page_size"),
)
def ragged_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | None = None,
    end: jax.Array | None = None,
    *,
    num_row_subchunks: int | None = None,
    col_size: int | None = None,
    page_size: int | None = None,
    lengths: jax.Array | None = None,
) -> jax.Array:
    """Perform gather on indices, using TensorCore DMA block gather for contiguous page blocks."""
    if x.ndim >= 3 or page_size is not None or lengths is not None:
        return paged_block_gather_tc(
            x,
            indices,
            lengths=lengths,
            page_size=page_size or (x.shape[1] if x.ndim >= 3 else 16),
        )

    # Fast path for flattened contiguous page blocks (e.g., (pages, block * h * d))
    # where row width is a multiple of (16 * 128) = 2048 and TensorCore DMA block gather
    # avoids SparseCore 2x bitcast read amplification and nested pipeline bubbles.
    if (
        x.ndim == 2
        and indices.ndim == 1
        and start is None
        and end is None
        and x.shape[1] >= 2048
        and x.shape[1] % (16 * 128) == 0
    ):
        return paged_block_gather_tc(x, indices, page_size=16)

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if start is None:
        start = jnp.array([0], jnp.int32)
    if end is None:
        end = jnp.array([indices.size], jnp.int32)
    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
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
        col_size = max(128, (min(col_size, hidden_size, max_col_size) // 128) * 128)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

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
