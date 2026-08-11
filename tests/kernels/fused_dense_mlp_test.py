# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_dense_mlp import fused_dense_mlp_local


def _random(shape, sharding, seed, scale=1.0):
    def initialize(key):
        values = jax.random.normal(key, shape, dtype=jnp.bfloat16)
        return values * jnp.asarray(scale, dtype=jnp.bfloat16)

    return jax.jit(initialize,
                   out_shardings=sharding)(jax.random.key(seed))


def test_fused_dense_mlp_tp_checkpoint():
    if len(jax.devices()) < 8 or jax.devices()[0].platform != "tpu":
        pytest.skip("Test requires a v6e-8 TPU")

    tp_size = 8
    mesh = Mesh(np.asarray(jax.devices()[:tp_size]), ("model", ))
    hidden_size = 256
    local_intermediate = 128
    num_tokens = 8

    x = _random((num_tokens, hidden_size), NamedSharding(mesh, P()), 0)
    gate_up = _random(
        (tp_size, hidden_size, 2 * local_intermediate),
        NamedSharding(mesh, P("model", None, None)), 1,
        hidden_size**-0.5)
    down = _random(
        (tp_size, local_intermediate, hidden_size),
        NamedSharding(mesh, P("model", None, None)), 2,
        local_intermediate**-0.5)

    def reference_local(x_local, gate_up_local, down_local):
        projection = x_local @ gate_up_local[0]
        gate, up = jnp.split(projection, 2, axis=-1)
        hidden = (jax.nn.gelu(gate, approximate=True) * up).astype(x.dtype)
        return jax.lax.psum(hidden @ down_local[0], "model")

    def fused_local(x_local, gate_up_local, down_local):
        partial = fused_dense_mlp_local(x_local, gate_up_local[0],
                                        down_local[0])
        return jax.lax.psum(partial, "model")

    specs = (P(), P("model", None, None), P("model", None, None))
    expected = jax.jit(
        jax.shard_map(reference_local,
                      mesh=mesh,
                      in_specs=specs,
                      out_specs=P(),
                      check_vma=False))(x, gate_up, down)
    actual = jax.jit(
        jax.shard_map(fused_local,
                      mesh=mesh,
                      in_specs=specs,
                      out_specs=P(),
                      check_vma=False))(x, gate_up, down)

    # Pallas and XLA use different BF16 MXU/GELU reduction orders, so the
    # algebraically identical results need not be bit-identical.
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               rtol=0.05,
                               atol=0.05)
