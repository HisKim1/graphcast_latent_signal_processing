"""Float64 graph-signal diagnostics with explicit channel reductions."""

from __future__ import annotations

import numpy as np
from pygsp.graphs import Graph


def _signal(value: np.ndarray, nodes: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("latent signals must be real")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("signal must have shape (nodes, channels) with nonempty axes")
    if nodes is not None and array.shape[0] != nodes:
        raise ValueError("signal node axis does not match the graph or basis")
    if not np.isfinite(array).all():
        raise ValueError("signal must be finite")
    return array


def channel_dirichlet_energy(signal: np.ndarray, graph: Graph) -> np.ndarray:
    """Return ``z_c.T @ L @ z_c`` per channel, without an eigendecomposition.

    The graph determines Laplacian normalization. Channels are not centered,
    standardized, or averaged. PyGSP's matrix-signal method returns a Gram
    matrix; this diagonal-only product avoids computing cross-channel terms.
    """
    values = _signal(signal, graph.N)
    return np.sum(values * (graph.L @ values), axis=0)


def dirichlet_energy(signal: np.ndarray, graph: Graph) -> float:
    """Return total raw Dirichlet energy, summed over channels and vertices."""
    return float(channel_dirichlet_energy(signal, graph).sum())


def local_dirichlet_energy(signal: np.ndarray, graph: Graph) -> np.ndarray:
    """Split each undirected edge's channel-summed energy equally at its ends.

    The nonnegative node contributions sum to total Dirichlet energy. These
    contributions are not the possibly negative node terms of ``z * (L @ z)``.
    """
    values = _signal(signal, graph.N)
    if graph.is_directed() or graph.has_loops() or (graph.W.data < 0).any():
        raise ValueError("local energy requires undirected nonnegative loop-free weights")
    graph.compute_differential_operator()
    edge_energy = np.sum(graph.grad(values) ** 2, axis=1)
    source, target, _ = graph.get_edge_list()
    local = np.zeros(graph.N, dtype=np.float64)
    np.add.at(local, source, 0.5 * edge_energy)
    np.add.at(local, target, 0.5 * edge_energy)
    return local


def spectral_power(coefficients: np.ndarray) -> np.ndarray:
    """Return squared coefficients summed over channels for each mode."""
    values = _signal(coefficients)
    return np.sum(values ** 2, axis=1)
