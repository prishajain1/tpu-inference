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
"""TP all-reduce scheduling experiment for Gemma 4's parallel FFNs.

The two functions below perform the same two independent MLPs.  The first
puts each MLP in its own manual-sharding region, like the current Gemma 4
dense/MoE branches.  The second puts both MLPs in one region so XLA's
latency-hiding scheduler can see useful work following the first psum.

Run this on a v6e-8 with the latency-hiding flags enabled and inspect the
optional trace for overlap between ``moe_psum`` and ``dense_mlp``.
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _measure(name, fn, warmup, iterations):
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


def _local_mlp(x, gate_up_weight, down_weight, label):
    with jax.named_scope(label):
        gate_up = x @ gate_up_weight
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.gelu(gate, approximate=True) * up
        return hidden @ down_weight


def _rms_norm(x, label):
    """Gemma-style branch norm that prevents combining the two psums."""
    with jax.named_scope(label):
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1,
                            keepdims=True)
        return x * jax.lax.rsqrt(variance + 1e-6).astype(x.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--profile-dir",
        help="Write a short XProf trace to this directory when specified.",
    )
    parser.add_argument(
        "--profile-variant",
        choices=("separate", "shared"),
        help=("Profile only this variant. Use separate processes and trace "
              "directories to avoid executable-name deduplication in XProf."),
    )
    args = parser.parse_args()

    tp_size = 8
    devices = jax.devices()
    if len(devices) < tp_size or devices[0].platform != "tpu":
        raise RuntimeError("This benchmark requires eight TPU devices")

    mesh = Mesh(
        np.asarray(devices[:tp_size]).reshape(tp_size),
        axis_names=("model", ),
    )
    hidden_size = 2816
    # Gemma 4's dense branch is much wider than one MoE expert. Keeping its
    # real width makes the candidate overlap window representative.
    dense_width = 9216
    moe_width = 704

    replicated = NamedSharding(mesh, P())
    column = NamedSharding(mesh, P(None, "model"))
    row = NamedSharding(mesh, P("model", None))

    def random_array(shape, sharding, seed):
        key = jax.random.key(seed)
        return jax.jit(
            lambda rng: jax.random.normal(rng, shape, dtype=jnp.bfloat16),
            out_shardings=sharding,
        )(key)

    x = random_array((args.tokens, hidden_size), replicated, 0)
    moe_gate_up = random_array((hidden_size, 2 * moe_width), column, 1)
    moe_down = random_array((moe_width, hidden_size), row, 2)
    dense_gate_up = random_array((hidden_size, 2 * dense_width), column, 3)
    dense_down = random_array((dense_width, hidden_size), row, 4)

    in_specs = (P(), P(None, "model"), P("model", None))

    def one_branch(x_arg, gate_up_arg, down_arg, label):
        def local_fn(x_local, gate_up_local, down_local):
            partial = _local_mlp(x_local, gate_up_local, down_local, label)
            with jax.named_scope(f"{label}_psum"):
                reduced = jax.lax.psum(partial, "model")
            return _rms_norm(reduced, f"{label}_post_rmsnorm")

        return jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(),
            check_vma=False,
        )(x_arg, gate_up_arg, down_arg)

    @jax.jit
    def separate_regions():
        moe = one_branch(x, moe_gate_up, moe_down, "moe_mlp")
        dense = one_branch(x, dense_gate_up, dense_down, "dense_mlp")
        # Keep both branch results observable so XLA cannot discard a branch.
        return moe, dense

    @jax.jit
    def shared_region():
        def local_fn(x_local, moe_gate_up_local, moe_down_local,
                     dense_gate_up_local, dense_down_local):
            moe_partial = _local_mlp(x_local, moe_gate_up_local,
                                     moe_down_local, "moe_mlp")
            with jax.named_scope("moe_psum"):
                moe = jax.lax.psum(moe_partial, "model")
            moe = _rms_norm(moe, "moe_post_rmsnorm")
            dense_partial = _local_mlp(x_local, dense_gate_up_local,
                                       dense_down_local, "dense_mlp")
            with jax.named_scope("dense_psum"):
                dense = jax.lax.psum(dense_partial, "model")
            dense = _rms_norm(dense, "dense_post_rmsnorm")
            return moe, dense

        return jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=(P(), P(None, "model"), P("model", None),
                      P(None, "model"), P("model", None)),
            out_specs=(P(), P()),
            check_vma=False,
        )(x, moe_gate_up, moe_down, dense_gate_up, dense_down)

    _measure("separate shard_map regions", separate_regions, args.warmup,
             args.iterations)
    _measure("shared shard_map region", shared_region, args.warmup,
             args.iterations)

    if args.profile_dir:
        if args.profile_variant is None:
            raise ValueError("--profile-dir requires --profile-variant")
        profile_fn = (separate_regions
                      if args.profile_variant == "separate" else shared_region)
        jax.profiler.start_trace(args.profile_dir)
        for _ in range(10):
            jax.block_until_ready(profile_fn())
        jax.profiler.stop_trace()
        print(f"{args.profile_variant} trace written to {args.profile_dir}")


if __name__ == "__main__":
    main()
