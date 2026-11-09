from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

hk = pytest.importorskip("haiku")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from graphcast_latent.extraction import capture_latents


class Node(NamedTuple):
    features: object


class Graph(NamedTuple):
    nodes: dict


class Encoder(hk.Module):
    def __call__(self, values):
        return Graph({"mesh_nodes": Node(values + 1)})


class Processor(hk.Module):
    def __init__(self):
        super().__init__()
        self._num_message_passing_steps = 2
        self._num_processor_repetitions = 1

    def _process_step(self, graph):
        return Graph({"mesh_nodes": Node(2 * graph.nodes["mesh_nodes"].features)})

    def __call__(self, graph):
        for _ in range(self._num_message_passing_steps):
            graph = self._process_step(graph)
        return graph


def test_interception_survives_cached_jit_and_preserves_result():
    captured = []

    def sink(index, values):
        captured.append((int(index), np.asarray(values)))

    def forward(values):
        model = SimpleNamespace(_grid2mesh_gnn=Encoder(), _mesh_gnn=Processor())
        with capture_latents(model, sink):
            graph = model._mesh_gnn(model._grid2mesh_gnn(values))
        return graph.nodes["mesh_nodes"].features

    transformed = hk.without_apply_rng(hk.transform(forward))
    apply = jax.jit(transformed.apply)
    for offset in (0, 5):
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) + offset
        output = apply({}, inputs)
        jax.block_until_ready(output)
        jax.effects_barrier()
        np.testing.assert_array_equal(output, 4 * (inputs + 1))
    assert [index for index, _ in captured] == [0, 1, 2, 0, 1, 2]
    np.testing.assert_array_equal(captured[0][1] * 4, captured[2][1])
    np.testing.assert_array_equal(captured[3][1] * 4, captured[5][1])


def test_exception_removes_interceptor():
    captured = []

    def forward(values):
        model = SimpleNamespace(_grid2mesh_gnn=Encoder(), _mesh_gnn=Processor())
        with pytest.raises(ValueError, match="stop"):
            with capture_latents(model, lambda *args: captured.append(args)):
                raise ValueError("stop")
        return model._mesh_gnn(model._grid2mesh_gnn(values)).nodes["mesh_nodes"].features

    apply = hk.without_apply_rng(hk.transform(forward)).apply
    output = jax.jit(apply)({}, jnp.zeros((2, 1, 3)))
    jax.block_until_ready(output)
    jax.effects_barrier()
    assert captured == []
    np.testing.assert_array_equal(output, np.full((2, 1, 3), 4))


def test_missing_forward_is_rejected():
    def forward():
        model = SimpleNamespace(_grid2mesh_gnn=Encoder(), _mesh_gnn=Processor())
        with capture_latents(model, lambda *args: None):
            pass

    transformed = hk.without_apply_rng(hk.transform(forward))
    with pytest.raises(RuntimeError, match="Unsupported GraphCast execution"):
        transformed.apply({})
