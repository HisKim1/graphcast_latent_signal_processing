"""GraphCast latent extraction and graph spectral diagnostics."""

from .graph import graph_from_adjacency, graph_from_mesh
from .spectral import (
    channel_dirichlet_energy,
    dirichlet_energy,
    local_dirichlet_energy,
    spectral_power,
)
from .storage import LatentWriter

__all__ = [
    "LatentWriter",
    "channel_dirichlet_energy",
    "dirichlet_energy",
    "graph_from_adjacency",
    "graph_from_mesh",
    "local_dirichlet_energy",
    "spectral_power",
]
