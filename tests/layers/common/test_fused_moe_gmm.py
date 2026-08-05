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

import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.sparse_core.dense_gather_reduce import \
    dense_gather_reduce
from tpu_inference.layers.common.fused_moe_gmm import _invert_permutation


def test_invert_permutation_matches_argsort():
    rng = np.random.default_rng(1234)

    for size in (1, 16, 256, 1024):
        permutation = jnp.asarray(rng.permutation(size), dtype=jnp.int32)

        actual = _invert_permutation(permutation)
        expected = jnp.argsort(permutation)

        np.testing.assert_array_equal(actual, expected)


def test_invert_permutation_restores_original_order():
    permutation = jnp.array([1, 3, 0, 2], dtype=jnp.int32)
    original = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    reordered = original[permutation]

    restored = reordered[_invert_permutation(permutation)]

    np.testing.assert_array_equal(restored, original)


def test_chunked_dense_gather_reduce_matches_full_batch():
    """TP MoE chunking must gather only the rows for the current chunk."""
    rng = np.random.default_rng(1234)
    num_tokens = 32
    topk = 8
    hidden_size = 128
    chunk_size = 16
    num_routed_rows = num_tokens * topk

    expert_outputs = jnp.asarray(
        rng.standard_normal((num_routed_rows, hidden_size)),
        dtype=jnp.float32,
    )
    inverse_permutation = jnp.asarray(
        rng.permutation(num_routed_rows),
        dtype=jnp.int32,
    )
    routing_weights = jnp.asarray(
        rng.random((num_tokens, topk)),
        dtype=jnp.float32,
    )

    expected = dense_gather_reduce(
        expert_outputs,
        inverse_permutation,
        routing_weights,
        topk,
    )

    chunks = []
    for start_token in range(0, num_tokens, chunk_size):
        end_token = start_token + chunk_size
        start_row = start_token * topk
        end_row = end_token * topk
        chunks.append(
            dense_gather_reduce(
                expert_outputs,
                inverse_permutation[start_row:end_row],
                routing_weights[start_token:end_token],
                topk,
            ))

    actual = jnp.concatenate(chunks, axis=0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
