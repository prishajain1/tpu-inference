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


@functools.lru_cache(maxsize=64)
def _build_vector_indexed_gather_kernel(
    l_work: int,
    d_padded: int,
    b_size: int,
    num_workers: int,
    num_subcores: int,
    dtype: jnp.dtype,
):
    """Builds a zero-copy vector indirect DMA SparseCore kernel for 32-bit rows."""
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_workers // num_subcores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    items_per_subcore = l_work // num_workers
    num_steps = items_per_subcore // b_size

    def sc_kernel(table_hbm, index_hbm, out_hbm, idx_vmem, rows_vmem):
        wid = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
        subcore_start = wid * items_per_subcore

        if num_steps <= 4:
            for step in range(num_steps):
                offset = subcore_start + step * b_size
                pltpu.sync_copy(index_hbm.at[pl.ds(offset, b_size)], idx_vmem)
                pltpu.sync_copy(table_hbm.at[idx_vmem], rows_vmem)
                pltpu.sync_copy(rows_vmem, out_hbm.at[pl.ds(offset, b_size), :])
        else:

            def step_fn(step, _):
                offset = subcore_start + step * b_size
                pltpu.sync_copy(index_hbm.at[pl.ds(offset, b_size)], idx_vmem)
                pltpu.sync_copy(table_hbm.at[idx_vmem], rows_vmem)
                pltpu.sync_copy(rows_vmem, out_hbm.at[pl.ds(offset, b_size), :])
                return None

            lax.fori_loop(0, num_steps, step_fn, None)

    return pl.kernel(
        sc_kernel,
        out_type=jax.ShapeDtypeStruct((l_work, d_padded), dtype),
        mesh=mesh,
        scratch_types=[
            pltpu.VMEM((b_size,), jnp.int32),
            pltpu.VMEM((b_size, d_padded), dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        name="sc_vector_indexed_gather_v2",
    )


def indexed_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    *,
    block_rows: int | None = None,
) -> jax.Array:
    """Full-range SparseCore vector indirect gather with ~128 KiB VMEM tiles."""
    assert x.ndim == 2, "Indexed gather only supports 2d inputs."
    assert indices.ndim == 1, "Indexed gather only supports 1d indices."

    orig_l = indices.shape[0]
    orig_d = x.shape[1]
    dtype = x.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None or (orig_l <= 4096 and orig_d <= 128):
        d_num = lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        )
        return lax.gather(
            x,
            indices[:, None],
            dimension_numbers=d_num,
            slice_sizes=(1, orig_d),
            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        )

    if dtype_bits != 32:
        return ragged_gather_v2(
            x,
            indices,
            jnp.array([0], jnp.int32),
            jnp.array([orig_l], jnp.int32),
            use_vector_dma=False,
        )

    num_cores = int(sc_info.num_cores)
    num_subcores = int(sc_info.num_subcores)
    num_workers = num_cores * num_subcores

    pad_d = (-orig_d) % 128
    if pad_d > 0:
        x_in = jnp.pad(x, ((0, 0), (0, pad_d)))
        d_padded = orig_d + pad_d
    else:
        x_in = x
        d_padded = orig_d

    b_cfg = (
        block_rows
        if block_rows is not None
        else (128 if d_padded >= 256 else 256)
    )
    max_b_vmem = (128 * 1024) // (d_padded * 4)
    b_size = max(8, min(b_cfg, max_b_vmem, max(32, orig_l // num_workers)))

    tile_quantum = num_workers * b_size
    pad_l = (-orig_l) % tile_quantum
    if pad_l > 0:
        index_in = jnp.pad(indices, (0, pad_l), constant_values=0)
        l_work = orig_l + pad_l
    else:
        index_in = indices
        l_work = orig_l

    items_per_subcore = l_work // num_workers
    while items_per_subcore % b_size != 0 and b_size > 8:
        b_size //= 2

    kernel = _build_vector_indexed_gather_kernel(
        l_work, d_padded, b_size, num_workers, num_subcores, dtype
    )
    out = kernel(x_in, index_in)
    if pad_l > 0 or pad_d > 0:
        out = out[:orig_l, :orig_d]
    return out


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
    static_argnames=(
        "num_row_subchunks",
        "col_size",
        "block_rows",
        "use_vector_dma",
    ),
)
def ragged_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | None = None,
    end: jax.Array | None = None,
    *,
    num_row_subchunks: int | None = None,
    col_size: int | None = None,
    block_rows: int | None = None,
    use_vector_dma: bool = True,
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec."""

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    dtype = x.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    if dtype_bits not in (4, 8, 16, 32):
        raise ValueError(
            f"dtype bit width must be one of 4, 8, 16, or 32, but got {dtype_bits} ({dtype})"
        )

    if use_vector_dma and dtype_bits == 32 and (start is None or end is None):
        return indexed_gather_v2(x, indices, block_rows=block_rows)

    if start is None:
        start = jnp.array([0], jnp.int32)
    elif jnp.isscalar(start):
        start = start[None]
    if end is None:
        end = jnp.array([indices.size], jnp.int32)
    elif jnp.isscalar(end):
        end = end[None]

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
    if out_pad_size > 0:
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
