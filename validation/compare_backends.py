"""Compare PyGSP with an existing graph, basis and finite latent fixtures.

The fixture directory contains adjacency.npz (SciPy sparse), signals.npy
(case, node, channel), reference_local.npy and reference_channel.npy. Output
artifacts belong in a separate directory, never in the original data directory.
Exit status checks graph, energy, and transform compatibility. Strict individual-
mode equality is reported as a diagnostic; scientific claims need separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from pygsp import graphs
from scipy import sparse

from graphcast_latent.spectral import (
    channel_dirichlet_energy,
    local_dirichlet_energy,
    spectral_power,
)


TOLERANCES = {
    "laplacian_scaled_absolute": 1e-14,
    "eigenvalue_relative": 1e-12,
    "eigenvalue_absolute": 1e-12,
    "energy_relative": 2e-12,
    "local_relative": 3e-13,
    "local_absolute": 1e-12,
    "coefficient_relative": 3e-12,
    "coefficient_absolute": 3e-12,
    "reconstruction_relative_norm": 5e-12,
    "cluster_gap_scaled": 5e-12,
    "subspace_relative_norm": 5e-9,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def comparison(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> dict:
    difference = np.abs(actual - expected)
    scale = float(np.max(np.abs(expected)))
    nonzero = expected != 0
    return {
        "pass": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_absolute_error": float(difference.max(initial=0)),
        "max_relative_error_nonzero": float(
            (difference[nonzero] / np.abs(expected[nonzero])).max(initial=0)
        ),
        "max_error_over_reference_max": float(difference.max(initial=0) / scale) if scale else 0.0,
        "rtol": rtol,
        "atol": atol,
    }


def toy_checks() -> dict:
    """Independent formulas catch normalization, channel and edge-count errors."""
    results = {}
    for lap_type in ("normalized", "combinatorial"):
        weights = np.array([[0., 2., 0., 0.], [2., 0., 3., 0.],
                            [0., 3., 0., 0.], [0., 0., 0., 0.]])
        graph = graphs.Graph(weights, lap_type=lap_type)
        degree = weights.sum(axis=1)
        manual = np.diag(degree) - weights
        if lap_type == "normalized":
            inverse = np.zeros_like(degree)
            np.divide(1, np.sqrt(degree), out=inverse, where=degree > 0)
            manual = inverse[:, None] * manual * inverse[None, :]
        np.testing.assert_allclose(graph.L.toarray(), manual, rtol=0, atol=1e-14)
        signal = np.array([[1., 2.], [3., -1.], [-2., 4.], [9., 1.]])
        graph.compute_fourier_basis()
        graph.compute_differential_operator()
        coefficients = graph.gft(signal)
        np.testing.assert_allclose(graph.igft(coefficients), signal, rtol=1e-12, atol=1e-12)
        per_channel = np.sum(signal * (manual @ signal), axis=0)
        gram = graph.dirichlet_energy(signal)
        assert gram.shape == (2, 2)
        np.testing.assert_allclose(np.diag(gram), per_channel, rtol=1e-12)
        np.testing.assert_allclose(np.sum(graph.grad(signal) ** 2, axis=0), per_channel, rtol=1e-12)
        np.testing.assert_allclose(graph.e @ (coefficients ** 2), per_channel, rtol=1e-12)
        assert graph.dirichlet_energy(np.zeros(4)) == 0
        null = np.sqrt(degree) if lap_type == "normalized" else np.ones(4)
        np.testing.assert_allclose(graph.L @ null, 0, atol=1e-14)
        results[lap_type] = {"pass": True, "matrix_energy_shape": list(gram.shape)}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--eigenvalues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "tolerances": TOLERANCES,
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pygsp")},
              "toy": toy_checks(), "cases": []}

    def save() -> None:
        target = args.output / "comparison.json"
        temporary = target.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(target)

    inputs = {name: args.fixtures / name for name in (
        "adjacency.npz", "signals.npy", "reference_local.npy", "reference_channel.npy")}
    inputs.update(reference_basis=args.basis, reference_eigenvalues=args.eigenvalues)
    report["input_sha256"] = {name: digest(path) for name, path in inputs.items()}
    weights = sparse.load_npz(inputs["adjacency.npz"]).tocsr()
    signals = np.load(inputs["signals.npy"], mmap_mode="r")
    reference_local = np.load(inputs["reference_local.npy"], mmap_mode="r")
    reference_channel = np.load(inputs["reference_channel.npy"], mmap_mode="r")
    original = np.load(args.basis, mmap_mode="r")
    original_values = np.load(args.eigenvalues)
    n = weights.shape[0]
    assert weights.shape == (n, n) and original.shape == (n, n)
    assert original.dtype == np.float64 and original_values.shape == (n,)
    assert signals.ndim == 3 and signals.shape[1] == n
    assert np.isfinite(signals).all() and np.isfinite(original_values).all()
    assert (weights != weights.T).nnz == 0 and np.all(weights.diagonal() == 0)
    degree = np.asarray(weights.sum(axis=1)).ravel()
    assert np.all(degree > 0)
    inverse = sparse.diags(degree ** -.5)
    manual_laplacian = sparse.eye(n, format="csr") - inverse @ weights @ inverse
    graph = graphs.Graph(weights.copy(), lap_type="normalized")
    delta = graph.L - manual_laplacian
    scale = max(1., float(np.abs(manual_laplacian.data).max()))
    operator_error = float(np.abs(delta.data).max(initial=0)) / scale
    report["graph"] = {"nodes": int(n), "undirected_edges": int(graph.n_edges),
                       "adjacency_exact": bool((graph.W != weights).nnz == 0),
                       "laplacian_scaled_max_error": operator_error,
                       "pass": operator_error <= TOLERANCES["laplacian_scaled_absolute"]}
    save()
    print("toy and sparse-operator gates PASS; full basis starting", flush=True)
    started = time.monotonic()
    graph.compute_fourier_basis()
    report["eigensolve_seconds"] = time.monotonic() - started
    np.save(args.output / "pygsp_eigenvalues.npy", graph.e)
    np.save(args.output / "pygsp_eigenvectors.npy", graph.U)
    report["eigenvalues"] = comparison(graph.e, original_values, rtol=1e-12, atol=1e-12)
    correlation = np.sum(original * graph.U, axis=0)
    signs = np.where(correlation < 0, -1., 1.)
    report["basis"] = {"minimum_absolute_column_correlation": float(np.min(np.abs(correlation))),
                       "median_absolute_column_correlation": float(np.median(np.abs(correlation)))}
    cluster_gap = TOLERANCES["cluster_gap_scaled"] * max(1., float(np.max(np.abs(graph.e))))
    boundaries = np.r_[0, np.flatnonzero(np.maximum(np.diff(graph.e), np.diff(original_values)) > cluster_gap) + 1, n]
    subspace_errors = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if end - begin < 2:
            continue
        old, new = original[:, begin:end], graph.U[:, begin:end]
        residual = new - old @ (old.T @ new)
        subspace_errors.append(float(np.linalg.norm(residual) / np.linalg.norm(new)))
    report["degenerate_groups"] = {
        "gap_threshold": cluster_gap, "group_count": len(subspace_errors),
        "max_relative_projection_residual": max(subspace_errors, default=0.),
        "pass": max(subspace_errors, default=0.) <= TOLERANCES["subspace_relative_norm"],
        "meaning": "Subspace agreement does not certify individual mode equivalence.",
    }
    report["selected_modes"] = [
        {"index": mode, "eigenvalue": float(graph.e[mode]),
         "absolute_column_correlation": float(abs(correlation[mode]))}
        for mode in args.modes
    ]
    save()
    print("basis complete", report["basis"], flush=True)
    graph.compute_differential_operator()
    source, target, _ = graph.get_edge_list()
    for index in range(len(signals)):
        signal = np.asarray(signals[index], dtype=np.float64)
        old_coefficients = original.T @ signal
        new_coefficients = graph.gft(signal)
        old_power = np.sum(old_coefficients ** 2, axis=1)
        new_power = np.sum(new_coefficients ** 2, axis=1)
        channel_energy = np.sum(signal * (graph.L @ signal), axis=0)
        edge_energy = np.sum(graph.grad(signal) ** 2, axis=1)
        local = np.zeros(n)
        np.add.at(local, source, .5 * edge_energy)
        np.add.at(local, target, .5 * edge_energy)
        reconstructed = graph.igft(new_coefficients)
        relative_reconstruction = float(np.linalg.norm(reconstructed - signal) / np.linalg.norm(signal))
        direct_signal_energy = float(np.sum(signal ** 2))
        parseval_error = float(abs(new_power.sum() - direct_signal_energy) / direct_signal_energy)
        retained = original.T @ signal
        retained_power = spectral_power(retained)
        retained_reconstruction = float(np.linalg.norm(original @ retained - signal) / np.linalg.norm(signal))
        relative_power_difference = np.abs(new_power - old_power) / old_power
        worst = int(np.argmax(relative_power_difference))
        item = {
            "case": index,
            "channel_energy": comparison(channel_energy, reference_channel[index], rtol=2e-12, atol=0),
            "local_energy": comparison(local, reference_local[index], rtol=3e-13, atol=1e-12),
            "global_energy": comparison(np.asarray(graph.e @ new_power), np.asarray(reference_channel[index].sum()), rtol=2e-12, atol=0),
            "mode_power": comparison(new_power, old_power, rtol=3e-12, atol=0),
            "signed_aligned_coefficients": comparison(signs[:, None] * new_coefficients, old_coefficients, rtol=3e-12, atol=3e-12),
            "parseval_relative_error": parseval_error,
            "reconstruction_relative_norm": relative_reconstruction,
            "transform_identity_pass": max(parseval_error, relative_reconstruction) <= 5e-12,
            "selected_mode_power": comparison(new_power[args.modes], old_power[args.modes], rtol=3e-12, atol=0) if args.modes else None,
            "selected_mode_values": [
                {"mode": mode, "reference_power": float(old_power[mode]),
                 "pygsp_power": float(new_power[mode])} for mode in args.modes],
            "worst_relative_power_mode": {
                "mode": worst, "reference_power": float(old_power[worst]),
                "pygsp_power": float(new_power[worst]),
                "reference_fraction_of_total": float(old_power[worst] / old_power.sum()),
            },
            "retained_basis": {
                "coefficients_exact": bool(np.array_equal(retained, old_coefficients)),
                "power_exact": bool(np.array_equal(retained_power, old_power)),
                "reconstruction_relative_norm": retained_reconstruction,
                "reconstruction_pass": retained_reconstruction <= 5e-12,
            },
            "released_channel_energy": comparison(channel_dirichlet_energy(signal, graph), reference_channel[index], rtol=2e-12, atol=0),
            "released_local_energy": comparison(local_dirichlet_energy(signal, graph), reference_local[index], rtol=3e-13, atol=1e-12),
        }
        report["cases"].append(item)
        save()
        print("case", index, "energy", item["global_energy"]["pass"],
              "individual modes", item["mode_power"]["pass"], flush=True)
    report["energy_gate_pass"] = all(
        case[name]["pass"] for case in report["cases"]
        for name in ("channel_energy", "local_energy", "global_energy"))
    report["individual_mode_gate_pass"] = all(
        case["mode_power"]["pass"] and case["signed_aligned_coefficients"]["pass"]
        for case in report["cases"])
    report["strict_pointwise_replacement_pass"] = bool(report["graph"]["pass"] and report["eigenvalues"]["pass"]
        and report["energy_gate_pass"] and report["individual_mode_gate_pass"]
        and all(case["transform_identity_pass"] for case in report["cases"]))
    report["retained_basis_gate_pass"] = all(
        case["retained_basis"]["coefficients_exact"] and case["retained_basis"]["power_exact"]
        and case["retained_basis"]["reconstruction_pass"]
        and case["released_channel_energy"]["pass"] and case["released_local_energy"]["pass"]
        for case in report["cases"])
    report["mathematical_compatibility_pass"] = bool(
        report["graph"]["pass"] and report["eigenvalues"]["pass"]
        and report["energy_gate_pass"] and report["retained_basis_gate_pass"]
        and all(case["transform_identity_pass"] for case in report["cases"]))
    report["scientific_claim_evaluation"] = "Not performed by this generic fixture comparator"
    report["status"] = "PASS" if report["mathematical_compatibility_pass"] else "FAIL"
    save()
    return 0 if report["mathematical_compatibility_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
