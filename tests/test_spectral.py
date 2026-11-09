"""Independent small-graph checks of the published mathematical conventions."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from graphcast_latent.graph import graph_from_adjacency, graph_from_mesh
from graphcast_latent.spectral import (
    channel_dirichlet_energy,
    dirichlet_energy,
    local_dirichlet_energy,
    spectral_power,
)


@pytest.mark.parametrize("lap_type", ["normalized", "combinatorial"])
def test_energies_match_independent_edge_sum(lap_type):
    weights = np.array([[0., 2., 0.], [2., 0., 5.], [0., 5., 0.]])
    graph = graph_from_adjacency(weights, lap_type=lap_type)
    signal = np.array([[1., 4.], [2., -3.], [4., 2.]])
    normalized = signal / np.sqrt(weights.sum(axis=1))[:, None] if lap_type == "normalized" else signal
    per_edge_channel = np.stack([2 * (normalized[0] - normalized[1]) ** 2,
                                 5 * (normalized[1] - normalized[2]) ** 2])
    expected_channels = per_edge_channel.sum(axis=0)
    expected_local = np.array([per_edge_channel[0].sum(), per_edge_channel.sum(),
                               per_edge_channel[1].sum()]) / 2
    np.testing.assert_allclose(channel_dirichlet_energy(signal, graph), expected_channels, rtol=2e-12)
    np.testing.assert_allclose(local_dirichlet_energy(signal, graph), expected_local, rtol=3e-13, atol=1e-12)
    np.testing.assert_allclose(dirichlet_energy(signal, graph), expected_channels.sum(), rtol=2e-12)
    graph.compute_fourier_basis()
    np.testing.assert_allclose(graph.e @ spectral_power(graph.gft(signal)), expected_channels.sum(), rtol=2e-12)


def test_isolated_vertex_and_null_signal_have_zero_energy():
    graph = graph_from_adjacency([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
    signal = np.sqrt(graph.dw)[:, None]
    signal[-1] = 12
    np.testing.assert_allclose(local_dirichlet_energy(signal, graph), 0, atol=1e-28)
    np.testing.assert_allclose(dirichlet_energy(signal, graph), 0, atol=1e-14)
    assert dirichlet_energy(np.zeros((3, 2)), graph) == 0


def test_mesh_preserves_edge_multiplicity_and_averages_orientation():
    vertices = np.eye(3)
    graph = graph_from_mesh(vertices, np.array([0, 0, 1]), np.array([1, 1, 2]))
    expected = np.array([[0., 2., 0.], [2., 0., 1.], [0., 1., 0.]]) / np.pi
    np.testing.assert_array_equal(graph.W.toarray(), expected)
    np.testing.assert_array_equal(graph.coords, vertices)


@pytest.mark.parametrize("weights", [
    [[0, 1], [0, 0]], [[0, -1], [-1, 0]], [[1, 0], [0, 0]],
    [[0, np.nan], [np.nan, 0]], [[0, np.inf], [np.inf, 0]],
    [[0, 1j], [1j, 0]], np.zeros((2, 3)), np.zeros((0, 0)),
])
def test_invalid_adjacency_is_rejected(weights):
    with pytest.raises(ValueError):
        graph_from_adjacency(weights)


def test_graph_construction_does_not_mutate_sparse_input():
    weights = sparse.csr_matrix([[0., 2.], [2., 0.]])
    original = weights.copy()
    graph = graph_from_adjacency(weights)
    graph.W.data[:] = 7
    np.testing.assert_array_equal(weights.toarray(), original.toarray())


@pytest.mark.parametrize("signal", [np.zeros(3), np.zeros((4, 2)),
    np.zeros((3, 0)), np.full((3, 2), np.nan), np.full((3, 2), np.inf),
    np.full((3, 2), 1j)])
def test_invalid_signal_is_rejected(signal):
    graph = graph_from_adjacency([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    with pytest.raises(ValueError):
        dirichlet_energy(signal, graph)


@pytest.mark.parametrize("senders,receivers", [
    (np.array([0.]), np.array([1.])), (np.array([0]), np.array([0])),
    (np.array([-1]), np.array([1])), (np.array([0]), np.array([3])),
    (np.array([0, 1]), np.array([1])),
])
def test_invalid_mesh_edges_are_rejected(senders, receivers):
    with pytest.raises(ValueError):
        graph_from_mesh(np.eye(3), senders, receivers)


def test_nonunit_mesh_is_rejected():
    with pytest.raises(ValueError, match="unit sphere"):
        graph_from_mesh(np.eye(3) * 2, np.array([0]), np.array([1]))


def test_native_pygsp_transform_preserves_signal_and_energy():
    graph = graph_from_adjacency([[0, 2, 0], [2, 0, 5], [0, 5, 0]])
    graph.compute_fourier_basis()
    signal = np.array([[1., 4.], [2., -3.], [4., 2.]])
    coefficients = graph.gft(signal)
    np.testing.assert_allclose(graph.igft(coefficients), signal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(spectral_power(coefficients).sum(), np.sum(signal ** 2), rtol=5e-12)


def test_comparator_accepts_valid_rotation_without_claiming_mode_identity(tmp_path):
    graph = graph_from_adjacency([[0, 1, 0, 0], [1, 0, 0, 0],
                                 [0, 0, 0, 1], [0, 0, 1, 0]])
    graph.compute_fourier_basis()
    original = graph.U.copy()
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    original[:, 2:] = original[:, 2:] @ rotation
    np.testing.assert_allclose(graph.L @ original, original * graph.e, atol=1e-14)
    signals = np.arange(16, dtype=np.float64).reshape(2, 4, 2) + .3
    sparse.save_npz(tmp_path / "adjacency.npz", graph.W)
    for name, values in {
        "signals": signals, "basis": original, "eigenvalues": graph.e,
        "reference_local": [local_dirichlet_energy(z, graph) for z in signals],
        "reference_channel": [channel_dirichlet_energy(z, graph) for z in signals],
    }.items():
        np.save(tmp_path / f"{name}.npy", values)
    script = Path(__file__).resolve().parents[1] / "validation/compare_backends.py"
    result = subprocess.run(
        [sys.executable, str(script), "--fixtures", str(tmp_path),
         "--basis", str(tmp_path / "basis.npy"),
         "--eigenvalues", str(tmp_path / "eigenvalues.npy"),
         "--output", str(tmp_path / "output")], capture_output=True, text=True)
    report = json.loads((tmp_path / "output/comparison.json").read_text())
    assert result.returncode == 0, result.stderr
    assert report["status"] == "PASS" and report["mathematical_compatibility_pass"]
    assert not report["strict_pointwise_replacement_pass"]
    assert not report["individual_mode_gate_pass"]
    assert report["scientific_claim_evaluation"].startswith("Not performed")
