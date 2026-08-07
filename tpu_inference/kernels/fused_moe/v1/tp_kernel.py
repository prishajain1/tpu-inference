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
"""Tensor-parallel building blocks for the persistent fused MoE kernel.

This module intentionally starts with a correctness implementation.  It makes
the TP contract explicit before the local expert calculation is replaced by a
Pallas kernel:

* tokens and routing logits are replicated across TP ranks;
* W1 is sharded on its intermediate output dimension;
* W2 is sharded on its intermediate contraction dimension; and
* each rank produces a partial hidden output which is combined by one psum.

The implementation is not used by serving and is not intended as a performance
path.  It is the numerical oracle for the native (including non-128-aligned)
TP Pallas kernel.
"""

import jax
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.fused_moe.v1.kernel import (apply_act_fn,
                                                       apply_scoring_fn)

P = jax.sharding.PartitionSpec


def _local_tp_moe(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating_output: jax.Array,
    top_k: int,
    *,
    renormalize_topk_logits: bool,
    act_fn: str,
    scoring_fn: str,
) -> jax.Array:
    """Computes one rank's partial MoE output using its intermediate shard."""
    scores = apply_scoring_fn(scoring_fn, gating_output)
    top_k_weights, top_k_indices = lax.top_k(scores, top_k)
    if renormalize_topk_logits:
        top_k_weights /= jnp.sum(top_k_weights, axis=-1, keepdims=True)
    # Expert matmuls accumulate in F32. Keep the weighting explicit so this
    # path also works when JAX's implicit dtype promotion policy is strict.
    top_k_weights = top_k_weights.astype(jnp.float32)

    # Selected local weights:
    #   w1: [T, K, 2, D, F_local]
    #   w2: [T, K, F_local, D]
    selected_w1 = w1[top_k_indices].astype(jnp.float32)
    selected_w2 = w2[top_k_indices].astype(jnp.float32)
    tokens_f32 = tokens.astype(jnp.float32)

    gate = jnp.einsum("td,tkdf->tkf", tokens_f32, selected_w1[:, :, 0])
    up = jnp.einsum("td,tkdf->tkf", tokens_f32, selected_w1[:, :, 1])
    activated = apply_act_fn(gate, up, act_fn)
    expert_output = jnp.einsum("tkf,tkfd->tkd", activated, selected_w2)
    return jnp.sum(expert_output * top_k_weights[..., None], axis=1)


def tp_moe_reference(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating_output: jax.Array,
    top_k: int,
    *,
    tp_axis_name: str = "model",
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
) -> jax.Array:
    """Runs the unquantized MoE TP contract for correctness testing.

    Unlike the existing EP kernel, this path does not require the local
    intermediate dimension to be 128-aligned.  Gemma 4 therefore exercises
    native local widths of 352, 176, and 88 for TP=2, 4, and 8 respectively.
    """
    if tp_axis_name not in mesh.axis_names:
        raise ValueError(f"TP axis {tp_axis_name!r} is not in {mesh.axis_names=}")

    num_tokens, hidden_size = tokens.shape
    num_experts, two, w1_hidden_size, intermediate_size = w1.shape
    if two != 2 or w1_hidden_size != hidden_size:
        raise ValueError(
            "Expected w1 shape (num_experts, 2, hidden_size, intermediate_size), "
            f"got {w1.shape}")
    if w2.shape != (num_experts, intermediate_size, hidden_size):
        raise ValueError(
            "Expected w2 shape (num_experts, intermediate_size, hidden_size), "
            f"got {w2.shape}")
    if gating_output.shape != (num_tokens, num_experts):
        raise ValueError(
            f"Expected gating output {(num_tokens, num_experts)}, got "
            f"{gating_output.shape}")
    if not 0 < top_k <= num_experts:
        raise ValueError(f"Expected 0 < top_k <= {num_experts}, got {top_k}")

    tp_size = mesh.shape[tp_axis_name]
    if intermediate_size % tp_size:
        raise ValueError(
            f"Intermediate size {intermediate_size} must be divisible by "
            f"TP size {tp_size}")

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P(),
            P(None, None, None, tp_axis_name),
            P(None, tp_axis_name, None),
            P(),
        ),
        out_specs=P(),
        check_vma=False,
    )
    def run_local(tokens, w1, w2, gating_output):
        partial_output = _local_tp_moe(
            tokens,
            w1,
            w2,
            gating_output,
            top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            scoring_fn=scoring_fn,
        )
        return lax.psum(partial_output, tp_axis_name).astype(tokens.dtype)

    return run_local(tokens, w1, w2, gating_output)
