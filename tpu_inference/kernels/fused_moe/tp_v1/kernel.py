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
"""Expert-major BF16 tensor-parallel fused MoE prototype.

This is a serving-inactive performance checkpoint for Gemma 4 TP=8. Routes
are packed by expert before entering Pallas. One program then evaluates a
block of routed rows for one expert, reusing each streamed W1/W2 tile across
the whole row block. The kernel produces TP-local routed-row outputs; JAX
performs the exact weighted unpermute and the final TP psum.

The fixed route capacity deliberately makes this a balanced-routing benchmark
prototype. Production integration requires a variable-length expert schedule
and an overflow-free fallback.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
_ROUTE_WEIGHT_STORAGE = 128


def _expert_block_kernel(
    packed_tokens_ref,
    w1_ref,
    w2_ref,
    packed_weights_ref,
    output_ref,
    w1_tile_ref,
    w2_tile_ref,
    projection_ref,
    output_acc_ref,
    dma_sem,
    *,
    hidden_size: int,
    local_intermediate_size: int,
    w2_intermediate_size: int,
    hidden_tile: int,
    route_block: int,
):
    """Evaluates one block of rows routed to one expert."""
    expert_id = pl.program_id(0)
    output_acc_ref[...] = jnp.zeros_like(output_acc_ref)
    projection_ref[...] = jnp.zeros_like(projection_ref)

    num_hidden_tiles = hidden_size // hidden_tile
    w1_copies = []
    for hidden_tile_id in range(num_hidden_tiles):
        hidden_slice = pl.ds(hidden_tile_id * hidden_tile, hidden_tile)
        buffer_id = hidden_tile_id % 2
        w1_copies.append(
            pltpu.make_async_copy(
                src_ref=w1_ref.at[
                    expert_id,
                    hidden_slice,
                    pl.ds(0, 2 * local_intermediate_size),
                ],
                dst_ref=w1_tile_ref.at[buffer_id],
                sem=dma_sem.at[buffer_id],
            ))

    w1_copies[0].start()
    for hidden_tile_id in range(num_hidden_tiles):
        hidden_slice = pl.ds(hidden_tile_id * hidden_tile, hidden_tile)
        buffer_id = hidden_tile_id % 2
        w1_copies[hidden_tile_id].wait()
        if hidden_tile_id + 1 < num_hidden_tiles:
            w1_copies[hidden_tile_id + 1].start()
        token_tile = packed_tokens_ref[0, :, hidden_slice]
        projection_ref[...] += jnp.dot(
            token_tile,
            w1_tile_ref[buffer_id],
            preferred_element_type=jnp.float32,
        )
    gate = projection_ref[:, :local_intermediate_size]
    up = projection_ref[:, local_intermediate_size:]
    activated = (jax.nn.silu(gate) * up).astype(packed_tokens_ref.dtype)
    route_weights = packed_weights_ref[0, :, 0].astype(jnp.float32)

    # W1 is padded to the TensorCore-friendly width 128, but Gemma 4 TP=8 has
    # only 88 real W2 contraction rows. Keep the VMEM dot tile at 128 while
    # transferring only those 88 rows from HBM.
    padded_w2_rows = local_intermediate_size - w2_intermediate_size
    if padded_w2_rows:
        w2_tile_ref[:,
                    pl.ds(w2_intermediate_size,
                          padded_w2_rows), :] = jnp.zeros_like(
                              w2_tile_ref[:,
                                          pl.ds(w2_intermediate_size,
                                                padded_w2_rows), :])
    w2_copies = []
    for hidden_tile_id in range(num_hidden_tiles):
        hidden_slice = pl.ds(hidden_tile_id * hidden_tile, hidden_tile)
        buffer_id = hidden_tile_id % 2
        w2_copies.append(
            pltpu.make_async_copy(
                src_ref=w2_ref.at[
                    expert_id,
                    pl.ds(0, w2_intermediate_size),
                    hidden_slice,
                ],
                dst_ref=w2_tile_ref.at[
                    buffer_id,
                    pl.ds(0, w2_intermediate_size),
                    pl.ds(0, hidden_tile),
                ],
                sem=dma_sem.at[buffer_id],
            ))

    w2_copies[0].start()
    for hidden_tile_id in range(num_hidden_tiles):
        hidden_slice = pl.ds(hidden_tile_id * hidden_tile, hidden_tile)
        buffer_id = hidden_tile_id % 2
        w2_copies[hidden_tile_id].wait()
        if hidden_tile_id + 1 < num_hidden_tiles:
            w2_copies[hidden_tile_id + 1].start()
        partial = jnp.dot(
            activated,
            w2_tile_ref[buffer_id],
            preferred_element_type=jnp.float32,
        )
        output_acc_ref[:, hidden_slice] = partial * route_weights[:, None]

    output_ref[0, ...] = output_acc_ref[...].astype(output_ref.dtype)


def _expert_major_pallas_call(packed_tokens, w1, w2, packed_weights,
                              block_experts, actual_blocks, *, hidden_tile,
                              route_block, activation):
    _, hidden_size = packed_tokens.shape
    local_intermediate_size = w1.shape[-1] // 2
    w2_intermediate_size = w2.shape[1]
    if packed_weights.shape[0] % route_block:
        raise ValueError("Packed route rows must divide into route blocks")
    if hidden_size % hidden_tile:
        raise ValueError(
            f"Hidden size {hidden_size} must divide by {hidden_tile}")
    if local_intermediate_size != 128:
        raise ValueError("TP=8 checkpoint requires local intermediate 128")
    if w2_intermediate_size > local_intermediate_size:
        raise ValueError(
            f"W2 width {w2_intermediate_size} exceeds {local_intermediate_size}")
    if w2_intermediate_size % 8:
        raise ValueError("W2 width must be divisible by 8")

    del hidden_tile

    def expert_body(tokens_ref, w1_ref, w2_ref, weights_ref, out_ref):
        tokens = tokens_ref[...]
        projection = jnp.dot(
            tokens,
            w1_ref[0],
            preferred_element_type=jnp.float32,
        )
        chunks = jnp.split(projection,
                           projection.shape[-1] // 8,
                           axis=-1)
        gate = jnp.concatenate(chunks[0::2], axis=-1)
        up = jnp.concatenate(chunks[1::2], axis=-1)
        if activation == "silu":
            activated = jax.nn.silu(gate) * up
        elif activation == "gelu":
            activated = jax.nn.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported TP MoE activation: {activation}")
        activated = activated.astype(tokens.dtype)
        partial = jnp.dot(
            activated[:, :w2_intermediate_size],
            w2_ref[0],
            preferred_element_type=jnp.float32,
        )
        route_weights = weights_ref[:, 0].astype(jnp.float32)
        out_ref[...] = (partial * route_weights[:, None]).astype(out_ref.dtype)

    def persistent_kernel(block_experts_ref, actual_blocks_ref, tokens_ref,
                          w1_ref, w2_ref, weights_ref, out_ref):
        pipeline = pltpu.emit_pipeline(
            expert_body,
            grid=(actual_blocks_ref[0], ),
            in_specs=(
                pl.BlockSpec(
                    (route_block, hidden_size),
                    lambda block: (block, 0),
                ),
                pl.BlockSpec(
                    (1, hidden_size, 2 * local_intermediate_size),
                    lambda block: (block_experts_ref[block], 0, 0),
                ),
                pl.BlockSpec(
                    (1, w2_intermediate_size, hidden_size),
                    lambda block: (block_experts_ref[block], 0, 0),
                ),
                pl.BlockSpec(
                    (route_block, _ROUTE_WEIGHT_STORAGE),
                    lambda block: (block, 0),
                ),
            ),
            out_specs=pl.BlockSpec(
                (route_block, hidden_size),
                lambda block: (block, 0),
            ),
        )
        pipeline(tokens_ref, w1_ref, w2_ref, weights_ref, out_ref)

    return pl.pallas_call(
        persistent_kernel,
        out_shape=jax.ShapeDtypeStruct(packed_tokens.shape,
                                      packed_tokens.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
        ),
        name="fused-tp-moe-persistent-v1",
    )(block_experts, actual_blocks, packed_tokens, w1, w2, packed_weights)


def _sorted_expert_pallas_call(sorted_tokens, w1, w2, block_weights,
                               block_starts, block_experts,
                               actual_blocks, *, route_block, activation):
    """Runs fused experts directly over contiguous variable-length routes."""
    num_routes, hidden_size = sorted_tokens.shape
    local_intermediate_size = w1.shape[-1] // 2
    w2_intermediate_size = w2.shape[1]

    def expert_body(tokens_ref, w1_ref, w2_ref, weights_ref, out_ref):
        projection = jnp.dot(tokens_ref[...],
                             w1_ref[0],
                             preferred_element_type=jnp.float32)
        chunks = jnp.split(projection, projection.shape[-1] // 8, axis=-1)
        gate = jnp.concatenate(chunks[0::2], axis=-1)
        up = jnp.concatenate(chunks[1::2], axis=-1)
        if activation == "silu":
            activated = jax.nn.silu(gate) * up
        elif activation == "gelu":
            activated = jax.nn.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported TP MoE activation: {activation}")
        partial = jnp.dot(
            activated[:, :w2_intermediate_size].astype(tokens_ref.dtype),
            w2_ref[0],
            preferred_element_type=jnp.float32)
        out_ref[0] = (partial *
                      weights_ref[0, :, :1].astype(jnp.float32)).astype(
                          out_ref.dtype)

    def persistent_kernel(block_starts_ref, block_experts_ref,
                          actual_blocks_ref, tokens_ref,
                          w1_ref, w2_ref, weights_ref, out_ref):
        pipeline = pltpu.emit_pipeline(
            expert_body,
            grid=(actual_blocks_ref[0], ),
            in_specs=(
                pl.BlockSpec((route_block, hidden_size), lambda block:
                             (block_starts_ref[block] // route_block, 0)),
                pl.BlockSpec(
                    (1, hidden_size, 2 * local_intermediate_size),
                    lambda block: (block_experts_ref[block], 0, 0)),
                pl.BlockSpec(
                    (1, w2_intermediate_size, hidden_size),
                    lambda block: (block_experts_ref[block], 0, 0)),
                pl.BlockSpec((1, route_block, _ROUTE_WEIGHT_STORAGE),
                             lambda block: (block, 0, 0)),
            ),
            out_specs=pl.BlockSpec(
                (1, route_block, hidden_size), lambda block: (block, 0, 0)),
        )
        pipeline(tokens_ref, w1_ref, w2_ref, weights_ref, out_ref)

    return pl.pallas_call(
        persistent_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (block_starts.shape[0], route_block, hidden_size),
            sorted_tokens.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 *
                                              1024),
        name="fused-tp-moe-sorted-v1",
    )(block_starts, block_experts, actual_blocks, sorted_tokens, w1, w2,
      block_weights)


def _pack_routes(tokens, route_weights, route_indices, *, num_experts,
                 route_block):
    """Packs arbitrary routes into non-overlapping expert row blocks."""
    num_tokens, top_k = route_indices.shape
    flat_experts = route_indices.reshape(-1)
    flat_weights = route_weights.reshape(-1)
    flat_tokens = jnp.repeat(tokens, top_k, axis=0)
    one_hot = jax.nn.one_hot(flat_experts, num_experts, dtype=jnp.int32)
    ranks = jnp.cumsum(one_hot, axis=0) - 1
    slots = jnp.take_along_axis(ranks, flat_experts[:, None], axis=1)[:, 0]

    expert_counts = jnp.sum(one_hot, axis=0)
    expert_blocks = (expert_counts + route_block - 1) // route_block
    block_offsets = jnp.concatenate((
        jnp.zeros((1, ), dtype=jnp.int32),
        jnp.cumsum(expert_blocks[:-1]),
    ))
    packed_slots = block_offsets[flat_experts] * route_block + slots

    # At most route_block-1 padding rows are needed per expert. Static storage
    # keeps compilation shapes fixed; the pipeline executes only actual_blocks.
    max_padded_routes = flat_experts.size + (route_block - 1) * num_experts
    max_padded_routes = (
        (max_padded_routes + route_block - 1) // route_block) * route_block
    max_blocks = max_padded_routes // route_block
    actual_blocks = jnp.sum(expert_blocks, keepdims=True)
    block_experts = jnp.repeat(
        jnp.arange(num_experts, dtype=jnp.int32),
        expert_blocks,
        total_repeat_length=max_blocks,
    )
    packed_tokens = jnp.zeros(
        (max_padded_routes, tokens.shape[-1]), tokens.dtype)
    # The trailing 128-wide dimension gives Mosaic a legal tiled layout while
    # keeping the eight route weights in a statically indexed expert block.
    packed_weights = jnp.zeros(
        (max_padded_routes, _ROUTE_WEIGHT_STORAGE),
        route_weights.dtype)
    packed_tokens = packed_tokens.at[packed_slots].set(flat_tokens)
    packed_weights = packed_weights.at[packed_slots, 0].set(flat_weights)
    return (packed_tokens, packed_weights, packed_slots, block_experts,
            actual_blocks)


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "top_k", "tp_axis_name", "hidden_tile",
                     "route_block", "activation"),
)
def fused_tp_moe_from_routes(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    routing_weights: jax.Array,
    routing_indices: jax.Array,
    top_k: int,
    *,
    activation: str = "silu",
    tp_axis_name: str = "model",
    hidden_tile: int = 256,
    route_block: int = 8,
):
    """Runs the persistent TP kernel from exact precomputed routes."""
    tp_axes = ((tp_axis_name, )
               if isinstance(tp_axis_name, str) else tuple(tp_axis_name))
    missing_axes = tuple(axis for axis in tp_axes if axis not in mesh.axis_names)
    if missing_axes:
        raise ValueError(f"Missing TP axes {missing_axes!r} in {mesh.axis_names}")
    num_experts = w1.shape[0]
    if routing_weights.shape != routing_indices.shape:
        raise ValueError("Route weight and index shapes must match")
    if routing_weights.shape != (tokens.shape[0], top_k):
        raise ValueError("Routes must have shape [num_tokens, top_k]")

    flat_experts = routing_indices.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    route_order = jnp.argsort(flat_experts)
    inverse_order = jnp.empty_like(route_order)
    inverse_order = inverse_order.at[route_order].set(
        jnp.arange(route_order.size, dtype=route_order.dtype))
    token_ids = jnp.repeat(jnp.arange(tokens.shape[0], dtype=jnp.int32), top_k)
    sorted_tokens = tokens[token_ids[route_order]]
    sorted_weights = flat_weights[route_order]
    sorted_experts = flat_experts[route_order]
    expert_counts = jax.nn.one_hot(flat_experts,
                                   num_experts,
                                   dtype=jnp.int32).sum(axis=0)
    expert_starts = jnp.cumsum(expert_counts) - expert_counts
    expert_lane_offsets = expert_starts % route_block
    expert_aligned_starts = expert_starts - expert_lane_offsets
    expert_blocks = jnp.where(
        expert_counts > 0,
        (expert_lane_offsets + expert_counts + route_block - 1) // route_block,
        0)
    block_offsets = jnp.concatenate((
        jnp.zeros((1, ), dtype=jnp.int32),
        jnp.cumsum(expert_blocks[:-1]),
    ))
    # The routes occupy ceil(M / route_block) global lane windows. Every
    # boundary between two non-empty experts can make one window get processed
    # once by each expert, adding at most one block. This is both safe for
    # arbitrary routing and substantially tighter than one block per route.
    max_nonempty_experts = min(num_experts, flat_experts.size)
    max_blocks = ((flat_experts.size + route_block - 1) // route_block +
                  max_nonempty_experts - 1)
    block_experts = jnp.repeat(
        jnp.arange(num_experts, dtype=jnp.int32),
        expert_blocks,
        total_repeat_length=max_blocks)
    block_ids = jnp.arange(max_blocks, dtype=jnp.int32)
    block_ranks = block_ids - block_offsets[block_experts]
    block_starts = (expert_aligned_starts[block_experts] +
                    block_ranks * route_block)
    actual_blocks = jnp.sum(expert_blocks, keepdims=True)
    sorted_rows = jnp.arange(flat_experts.size, dtype=jnp.int32)
    sorted_block_ids = (block_offsets[sorted_experts] +
                        (sorted_rows -
                         expert_aligned_starts[sorted_experts]) // route_block)
    sorted_block_slots = sorted_rows % route_block
    sorted_block_positions = (sorted_block_ids * route_block +
                              sorted_block_slots)
    block_weights = jnp.zeros(
        (max_blocks * route_block, _ROUTE_WEIGHT_STORAGE),
        sorted_weights.dtype)
    block_weights = block_weights.at[sorted_block_positions, 0].set(
        sorted_weights)
    block_weights = block_weights.reshape(max_blocks, route_block,
                                          _ROUTE_WEIGHT_STORAGE)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(None, None, tp_axis_name),
                  P(None, tp_axis_name, None), P(), P(), P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def run_local(sorted_tokens, w1, w2, block_weights, block_starts,
                  block_experts, actual_blocks, sorted_block_positions,
                  inverse_order):
        block_partial = _sorted_expert_pallas_call(
            sorted_tokens,
            w1,
            w2,
            block_weights,
            block_starts,
            block_experts,
            actual_blocks,
            activation=activation,
            route_block=route_block,
        )
        sorted_partial = block_partial.reshape(-1, tokens.shape[-1])[
            sorted_block_positions]
        flat_partial = sorted_partial[inverse_order]
        flat_partial = flat_partial.astype(jnp.float32)
        # packed_slots restores the original token-major route order, where
        # every token owns exactly ``top_k`` consecutive rows.  A direct
        # reduction is equivalent to scattering those rows to repeated token
        # IDs, but avoids materializing and updating an output-sized scatter.
        token_partial = flat_partial.reshape(tokens.shape[0], top_k,
                                             tokens.shape[-1]).sum(axis=1)
        return lax.psum(token_partial,
                        tp_axis_name).astype(tokens.dtype)

    return run_local(sorted_tokens, w1, w2, block_weights, block_starts,
                     block_experts, actual_blocks, sorted_block_positions,
                     inverse_order)


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "top_k", "tp_axis_name", "hidden_tile",
                     "route_block", "renormalize_topk_logits", "activation"),
)
def fused_tp_moe(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating_output: jax.Array,
    top_k: int,
    *,
    activation: str = "silu",
    tp_axis_name: str = "model",
    hidden_tile: int = 256,
    route_block: int = 8,
    renormalize_topk_logits: bool = True,
):
    """Benchmark wrapper that computes softmax top-k routes."""
    routing_weights = jax.nn.softmax(gating_output, axis=-1)
    routing_weights, routing_indices = lax.top_k(routing_weights, top_k)
    if renormalize_topk_logits:
        routing_weights /= routing_weights.sum(axis=-1, keepdims=True)
    return fused_tp_moe_from_routes(
        mesh,
        tokens,
        w1,
        w2,
        routing_weights,
        routing_indices,
        top_k,
        activation=activation,
        tp_axis_name=tp_axis_name,
        hidden_tile=hidden_tile,
        route_block=route_block,
    )
