# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Gemma 4 TP=8 benchmark for the fused Dense MLP checkpoint."""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_dense_mlp import fused_dense_mlp_local


def _random(shape, sharding, seed, scale=1.0):
    def initialize(key):
        values = jax.random.normal(key, shape, dtype=jnp.bfloat16)
        return values * jnp.asarray(scale, dtype=jnp.bfloat16)

    return jax.jit(initialize,
                   out_shardings=sharding)(jax.random.key(seed))


def _measure(name, fn, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    print(f"{name}: median={statistics.median(samples):.3f} ms, "
          f"min={min(samples):.3f} ms, max={max(samples):.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--token-tile", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    tp_size = 8
    if len(jax.devices()) < tp_size or jax.devices()[0].platform != "tpu":
        raise RuntimeError("Benchmark requires a v6e-8 TPU")

    mesh = Mesh(np.asarray(jax.devices()[:tp_size]), ("model", ))
    hidden_size = 2816
    local_intermediate = 264  # Gemma dense width 2112 / TP=8.
    x = _random((args.tokens, hidden_size), NamedSharding(mesh, P()), 0)
    gate_up = _random(
        (tp_size, hidden_size, 2 * local_intermediate),
        NamedSharding(mesh, P("model", None, None)), 1,
        hidden_size**-0.5)
    down = _random(
        (tp_size, local_intermediate, hidden_size),
        NamedSharding(mesh, P("model", None, None)), 2,
        local_intermediate**-0.5)
    specs = (P(), P("model", None, None), P("model", None, None))

    def baseline_local(x_local, gate_up_local, down_local):
        projection = x_local @ gate_up_local[0]
        gate, up = jnp.split(projection, 2, axis=-1)
        hidden = (jax.nn.gelu(gate, approximate=True) * up).astype(x.dtype)
        return jax.lax.psum(hidden @ down_local[0], "model")

    def fused_local(x_local, gate_up_local, down_local):
        partial = fused_dense_mlp_local(x_local, gate_up_local[0],
                                        down_local[0],
                                        token_tile=args.token_tile,
                                        intermediate_tile=local_intermediate)
        return jax.lax.psum(partial, "model")

    baseline = jax.jit(
        jax.shard_map(baseline_local,
                      mesh=mesh,
                      in_specs=specs,
                      out_specs=P(),
                      check_vma=False))
    fused = jax.jit(
        jax.shard_map(fused_local,
                      mesh=mesh,
                      in_specs=specs,
                      out_specs=P(),
                      check_vma=False))

    print(f"Gemma Dense MLP: T={args.tokens}, D={hidden_size}, "
          f"F_local={local_intermediate}, TP={tp_size}, "
          f"token_tile={args.token_tile}")
    _measure("XLA Dense MLP", lambda: baseline(x, gate_up, down), args.warmup,
             args.iterations)
    _measure("fused Pallas Dense MLP", lambda: fused(x, gate_up, down),
             args.warmup, args.iterations)


if __name__ == "__main__":
    main()
