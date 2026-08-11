# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serving-inactive fused BF16 Dense MLP checkpoint for TPU.

The kernel consumes one TP-local intermediate tile at a time:

  gate/up projection -> approximate GELU * up -> down projection

Only the final hidden output is written to HBM.  This is intentionally a
small performance checkpoint, not a general linear-layer backend.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, gate_up_ref, down_ref, out_ref, gate_up_tile_ref,
            down_tile_ref, out_acc_ref, dma_sem, *,
            hidden_size: int, local_intermediate_size: int,
            intermediate_tile: int):
    out_acc_ref[...] = jnp.zeros_like(out_acc_ref)
    gate_up_copy = pltpu.make_async_copy(gate_up_ref, gate_up_tile_ref,
                                         dma_sem.at[0])
    down_copy = pltpu.make_async_copy(down_ref, down_tile_ref, dma_sem.at[1])
    gate_up_copy.start()
    down_copy.start()
    gate_up_copy.wait()

    # Match the existing two-matmul path's materialized BF16 boundary.
    projection = jax.lax.dot_general(
        x_ref[...],
        gate_up_tile_ref[...],
        dimension_numbers=(((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x_ref.dtype)
    gate = projection[:, :local_intermediate_size]
    up = projection[:, local_intermediate_size:]
    activated = jax.nn.gelu(gate, approximate=True) * up

    down_copy.wait()
    out_acc_ref[...] = jnp.dot(
        activated,
        down_tile_ref[...],
        preferred_element_type=jnp.float32,
    )

    out_ref[...] = out_acc_ref[...].astype(out_ref.dtype)


def fused_dense_mlp_local(x: jax.Array,
                          gate_up_weight: jax.Array,
                          down_weight: jax.Array,
                          *,
                          token_tile: int = 8,
                          intermediate_tile: int = 128) -> jax.Array:
    """Runs a fused TP-local Gemma Dense MLP without its final TP psum.

    Args:
      x: BF16 activations ``[tokens, hidden]``.
      gate_up_weight: TP-local, gate then up, ``[hidden, 2 * local_ffn]``.
      down_weight: TP-local ``[local_ffn, hidden]``.
    """
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"Expected BF16 activations, got {x.dtype}")
    if gate_up_weight.dtype != x.dtype or down_weight.dtype != x.dtype:
        raise ValueError("Activations and weights must have the same dtype")
    if x.ndim != 2 or gate_up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("Fused Dense MLP expects rank-two inputs")

    num_tokens, hidden_size = x.shape
    local_intermediate_size = down_weight.shape[0]
    expected_gate_up = (hidden_size, 2 * local_intermediate_size)
    if gate_up_weight.shape != expected_gate_up:
        raise ValueError(
            f"Expected gate/up shape {expected_gate_up}, got "
            f"{gate_up_weight.shape}")
    if down_weight.shape[1] != hidden_size:
        raise ValueError("Down-projection output must equal hidden size")
    if num_tokens % token_tile:
        raise ValueError("Token count must be divisible by token_tile")
    if hidden_size % 128:
        raise ValueError("Hidden size must be divisible by 128")
    if local_intermediate_size != intermediate_tile:
        raise ValueError("The fused kernel requires one full intermediate tile")

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            hidden_size=hidden_size,
            local_intermediate_size=local_intermediate_size,
            intermediate_tile=intermediate_tile,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((token_tile, hidden_size),
                             lambda token_block: (token_block, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (token_tile, hidden_size),
                lambda token_block: (token_block, 0),
            ),
            grid=(num_tokens // token_tile, ),
            scratch_shapes=[
                pltpu.VMEM((2 * intermediate_tile, hidden_size), x.dtype),
                pltpu.VMEM((intermediate_tile, hidden_size), x.dtype),
                pltpu.VMEM((token_tile, hidden_size), jnp.float32),
                pltpu.SemaphoreType.DMA((2, )),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=16 * 1024 *
                                             1024),
        name="fused-dense-mlp-v1",
    )(x, gate_up_weight.T, down_weight)
