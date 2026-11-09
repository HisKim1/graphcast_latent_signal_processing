"""Construct the undirected analysis graph without importing a forecast model."""

from __future__ import annotations

import numpy as np
from pygsp import graphs
from scipy import sparse


def graph_from_adjacency(
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    lap_type: str = "normalized",
) -> graphs.Graph:
    """Return a PyGSP graph from symmetric, nonnegative, loop-free weights.

    Isolated vertices are allowed; PyGSP gives them a zero Laplacian row.
    Node order is exactly the row order of ``adjacency``.
    """
    if lap_type not in {"normalized", "combinatorial"}:
        raise ValueError("lap_type must be 'normalized' or 'combinatorial'")
    if np.iscomplexobj(adjacency):
        raise ValueError("adjacency must be real")
    weights = sparse.csr_matrix(adjacency, dtype=np.float64, copy=True)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or not weights.shape[0]:
        raise ValueError("adjacency must be a nonempty square matrix")
    weights.sum_duplicates()
    weights.eliminate_zeros()
    weights.sort_indices()
    if not np.isfinite(weights.data).all() or (weights.data < 0).any():
        raise ValueError("edge weights must be finite and nonnegative")
    if np.any(weights.diagonal() != 0):
        raise ValueError("self-loops are not part of the analysis graph")
    if (weights != weights.T).nnz:
        raise ValueError("adjacency must be symmetric; symmetrize explicitly")
    return graphs.Graph(weights, lap_type=lap_type)


def graph_from_mesh(
    vertices: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    *,
    lap_type: str = "normalized",
) -> graphs.Graph:
    """Build inverse-angular-distance weights from a model's unit-sphere mesh.

    Preserve the model's node order and supplied edge multiplicities. Directed
    weights are averaged with their transpose after duplicate edges are summed.
    Angular distances are in radians; vertex coordinates are not renormalized.
    """
    coordinates = np.asarray(vertices)
    if np.iscomplexobj(coordinates):
        raise ValueError("vertices must be real")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError("vertices must have shape (nodes, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("vertices must be finite")
    if not np.allclose(np.linalg.norm(coordinates, axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("vertices must lie on the unit sphere")
    source, target = np.asarray(senders), np.asarray(receivers)
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("senders and receivers must be equally sized vectors")
    if source.dtype.kind not in "iu" or target.dtype.kind not in "iu":
        raise ValueError("edge indices must be integers")
    if ((source < 0).any() or (target < 0).any()
            or (source >= len(coordinates)).any() or (target >= len(coordinates)).any()):
        raise ValueError("edge index outside the vertex array")
    if np.any(source == target):
        raise ValueError("self-loops are not part of the analysis graph")
    cosine = np.clip((coordinates[source] * coordinates[target]).sum(axis=1), -1, 1)
    distance = np.arccos(cosine)
    if np.any(distance <= 0):
        raise ValueError("an edge joins coincident vertices")
    weights = sparse.csr_matrix(
        (1 / distance, (source, target)), shape=(len(coordinates), len(coordinates))
    )
    weights = (weights + weights.T) * 0.5
    graph = graph_from_adjacency(weights, lap_type=lap_type)
    graph.coords = coordinates
    return graph
