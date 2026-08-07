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
"""Gemma 4 TP=8 benchmark for the serving-inactive fused TP MoE kernel."""

import argparse

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_moe.tp_v1.kernel import fused_tp_moe
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func


def _device_ones(shape, sharding):
    return jax.jit(
        lambda: jnp.ones(shape, dtype=jnp.bfloat16),
        out_shardings=sharding,
    )()


def _measure(name, fn, warmup=3, iterations=20):
    for _ in range(warmup):
        jax.block_until_ready(fn())

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn())
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    print(
        f"{name}: median={statistics.median(samples_ms):.3f} ms, "
        f"min={min(samples_ms):.3f} ms, max={max(samples_ms):.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=32,
        choices=(32, 64, 128),
        help="Balanced token count; 128 fills all eight expert route rows.",
    )
    args = parser.parse_args()
    tp_size = 8
    if len(jax.devices()) < tp_size:
        raise RuntimeError(f"Benchmark requires {tp_size} TPU devices")
    if jax.devices()[0].platform != "tpu":
        raise RuntimeError("Benchmark must run on TPU")

    mesh = Mesh(
        np.array(jax.devices()[:tp_size]).reshape(1, tp_size),
        axis_names=("data", "model"),
    )
    num_tokens = args.num_tokens
    hidden_size = 2816
    intermediate_size = 704
    padded_intermediate_size = 128 * tp_size
    num_experts = 128
    top_k = 8

    replicated = NamedSharding(mesh, P())
    gmm_w1_sharding = NamedSharding(mesh, P(None, None, "model"))
    w2_sharding = NamedSharding(mesh, P(None, "model", None))

    tokens = _device_ones((num_tokens, hidden_size), replicated)
    selected = ((jnp.arange(num_tokens)[:, None] * top_k +
                 jnp.arange(top_k)[None, :]) % num_experts)
    router_logits = jnp.full((num_tokens, num_experts), -20, jnp.bfloat16)
    router_logits = router_logits.at[
        jnp.arange(num_tokens)[:, None], selected].set(20)
    router_logits = jax.device_put(router_logits, replicated)

    # This is the layout produced for Gemma by the current GMM TP path: each
    # local gate/up output is padded from 88 to 128, while W2 keeps 88 columns.
    gmm_w1 = _device_ones(
        (num_experts, hidden_size, 2 * padded_intermediate_size),
        gmm_w1_sharding,
    )
    gmm_w2 = _device_ones(
        (num_experts, intermediate_size, hidden_size),
        w2_sharding,
    )

    fused_w1 = gmm_w1
    fused_w2 = _device_ones(
        (num_experts, intermediate_size, hidden_size),
        w2_sharding,
    )

    @jax.jit
    def run_gmm():
        return fused_moe_func(
            hidden_states=tokens,
            w1=gmm_w1,
            w2=gmm_w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            gating_output=router_logits,
            topk=top_k,
            renormalize=True,
            mesh=mesh,
            use_ep=False,
            activation="gelu",
            scoring_fn="softmax",
        )

    def run_fused():
        return fused_tp_moe(
            mesh,
            tokens,
            fused_w1,
            fused_w2,
            router_logits,
            top_k,
            activation="gelu",
        )

    routes_per_expert = num_tokens * top_k // num_experts
    print(
        f"Gemma 4 MoE: T={num_tokens}, D=2816, F=704, E=128, K=8, "
        f"TP=8, routes/expert={routes_per_expert}")
    _measure("GMM TP", run_gmm)
    _measure("TP-specific fused Pallas", run_fused)


if __name__ == "__main__":
    main()
