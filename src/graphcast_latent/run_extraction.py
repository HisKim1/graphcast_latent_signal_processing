"""Run pinned official GraphCast and export its encoder/processor states.

The CLI accepts two six-hourly initialization frames, not future weather truth.
Inference dependencies are imported only when ``run_extraction`` is called.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np

from .extraction import capture_latents
from .storage import LatentWriter

UPSTREAM_COMMIT = "7077d40a36db6541e3ed72ccaed1c0d202fa6014"
STATISTICS_FILES = (
    "stats_mean_by_level.nc",
    "stats_stddev_by_level.nc",
    "stats_diffs_stddev_by_level.nc",
)


def configure_runtime() -> str:
    """Set the validated compiler options before JAX is imported.

Already-imported JAX requires these options to have been set by the caller.
This function never changes GPU visibility or chooses a device for the caller.
"""
    tokens = shlex.split(os.environ.get("XLA_FLAGS", ""))
    options = dict(token.split("=", 1) for token in tokens if "=" in token)
    deterministic = "--xla_gpu_deterministic_ops"
    disabled = "--xla_disable_hlo_passes"
    required_pass = "fusion-dynamic-memcpy-rewriter"
    if options.get(deterministic, "true") != "true":
        raise ValueError(f"{deterministic}=true is required for this runner")
    passes = options.get(disabled, "").split(",") if options.get(disabled) else []
    ready = options.get(deterministic) == "true" and required_pass in passes
    if "jax" in sys.modules and not ready:
        raise RuntimeError("Set the documented XLA_FLAGS before importing JAX")
    if deterministic not in options:
        tokens.append(f"{deterministic}=true")
    if required_pass not in passes:
        passes.append(required_pass)
        tokens = [token for token in tokens if not token.startswith(disabled + "=")]
        tokens.append(f"{disabled}={','.join(passes)}")
    os.environ["XLA_FLAGS"] = " ".join(tokens)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return os.environ["XLA_FLAGS"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_paths(output: Path, forecast_output: Path | None) -> list[Path]:
    if output.suffix != ".npy":
        raise ValueError("--output must use the .npy suffix")
    paths = [output, output.with_suffix(".json"), output.with_suffix(".npy.partial"),
             output.with_suffix(".json.partial"), output.with_suffix(".graph.npz"),
             output.with_suffix(".graph.npz.partial")]
    if forecast_output is not None:
        if forecast_output.suffix != ".nc":
            raise ValueError("--forecast-output must use the .nc suffix")
        paths += [forecast_output, forecast_output.with_suffix(".nc.partial")]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Output paths must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    return paths


def _finite(dataset: Any, label: str) -> None:
    for name, values in dataset.data_vars.items():
        if not np.isfinite(values.values).all():
            raise ValueError(f"{label} variable {name!r} contains nonfinite values")


def prepare_inputs(
    dataset: Any, task_config: Any, model_config: Any, steps: int,
) -> tuple[Any, Any, Any, np.datetime64]:
    """Validate two initialization frames and build official forcing/templates.

    Returns ``(inputs, targets_template, forcings, initialization_time)``.
    Input ``time`` contains UTC datetime64 values. This runner supports one
    initialization per call; the lower-level capture API supports native batches.
    """
    import pandas as pd
    import xarray as xr
    from graphcast import data_utils, graphcast

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if dataset.sizes.get("time") != 2 or not np.issubdtype(dataset.time.dtype, np.datetime64):
        raise ValueError("Input time must contain exactly two UTC datetime64 frames")
    absolute = dataset.time.values.astype("datetime64[ns]")
    interval = np.timedelta64(6, "h")
    if np.isnat(absolute).any() or absolute[1] - absolute[0] != interval:
        raise ValueError("The two initialization frames must be exactly six hours apart")
    if pd.Timedelta(task_config.input_duration) != pd.Timedelta("12h"):
        raise ValueError("This runner requires the official two-frame, 12h input duration")
    if dataset.sizes.get("batch", 1) != 1:
        raise ValueError("This runner accepts one initialization (batch size 1)")
    for name, extent in (("lat", 180), ("lon", 360)):
        if name not in dataset.coords or dataset[name].dims != (name,):
            raise ValueError(f"A one-dimensional {name} coordinate is required")
        values = dataset[name].values
        resolution = model_config.resolution
        expected = int(round(extent / resolution)) + (name == "lat")
        if values.size != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} does not match checkpoint resolution {resolution}")
        reference = (np.linspace(-90, 90, expected) if name == "lat"
                     else np.arange(expected) * resolution)
        if name == "lat" and values[0] > values[-1]:
            reference = reference[::-1]
        if not np.allclose(values, reference, rtol=0, atol=1e-5):
            raise ValueError(f"{name} must span the regular global checkpoint grid")
    levels = list(task_config.pressure_levels)
    if "level" not in dataset.coords or not np.isin(levels, dataset.level.values).all():
        raise ValueError("Input level coordinate is missing checkpoint pressure levels (hPa)")
    if len(np.unique(dataset.level.values)) != dataset.sizes["level"]:
        raise ValueError("Pressure levels must be unique")
    source = dataset.sel(level=levels).drop_vars("datetime", errors="ignore")
    # Expand weather fields, but retain upstream's unbatched static fields.
    if "batch" not in source.dims:
        source = source.expand_dims(batch=[0])
    for name in graphcast.STATIC_VARS:
        if name in source:
            if "time" in source[name].dims:
                raise ValueError(f"Static field {name!r} must not have a time dimension")
            if "batch" in source[name].dims:
                source[name] = source[name].isel(batch=0, drop=True)
    required = set(task_config.input_variables) - set(task_config.forcing_variables)
    missing = required - set(source.data_vars)
    if missing:
        raise ValueError(f"Missing checkpoint input variables: {sorted(missing)}")
    for name in required:
        spatial = ("lat", "lon", "level") if name in graphcast.TARGET_ATMOSPHERIC_VARS else ("lat", "lon")
        expected = set(spatial) | (set() if name in graphcast.STATIC_VARS else {"batch", "time"})
        if set(source[name].dims) != expected:
            raise ValueError(f"Input {name!r} must have dimensions {sorted(expected)}")
    offsets = np.arange(-1, steps + 1) * interval
    forcing_data = xr.Dataset(coords={
        "batch": source.batch, "time": offsets, "lat": source.lat, "lon": source.lon,
        "datetime": (("batch", "time"), (absolute[-1] + offsets)[None]),
    })
    if set(task_config.forcing_variables) & set(graphcast.GENERATED_FORCING_VARS):
        data_utils.add_derived_vars(forcing_data)
    if data_utils.TISR in task_config.forcing_variables:
        data_utils.add_tisr_var(forcing_data)
    missing_forcings = set(task_config.forcing_variables) - set(forcing_data.data_vars)
    if missing_forcings:
        raise ValueError(f"Unsupported forcing variables: {sorted(missing_forcings)}")
    forcing_data = forcing_data[list(task_config.forcing_variables)].drop_vars("datetime")
    source = source.assign_coords(time=offsets[:2])
    for name in task_config.forcing_variables:
        if name not in source:
            source[name] = forcing_data[name].isel(time=slice(0, 2))
    inputs = source[list(task_config.input_variables)].astype(np.float32)
    forcings = forcing_data.isel(time=slice(2, None)).astype(np.float32)
    targets = xr.Dataset()
    for name in task_config.target_variables:
        if name not in graphcast.TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS:
            raise ValueError(f"Unsupported GraphCast target variable: {name}")
        dims = ("batch", "time", "lat", "lon")
        if name in graphcast.TARGET_ATMOSPHERIC_VARS:
            dims += ("level",)
        coords = {dim: (offsets[2:] if dim == "time" else source[dim]) for dim in dims}
        shape = tuple(len(coords[dim]) for dim in dims)
        # Broadcast a scalar: future truth is never loaded or allocated here.
        targets[name] = xr.DataArray(np.broadcast_to(np.float32(np.nan), shape), dims=dims, coords=coords)
    _finite(inputs, "Input")
    _finite(forcings, "Forcing")
    return inputs, targets, forcings, absolute[-1]


def run_extraction(
    *, checkpoint: str | Path, stats_dir: str | Path, input: str | Path,
    steps: int, output: str | Path, forecast_output: str | Path | None = None,
) -> Path:
    """Run autoregressive inference and return the completed latent NPY path.

    All output names must be new. Failure retains partial files for inspection;
    the JSON completion record is written only after all outputs are ready.
    """
    checkpoint_path, stats_dir, input_path = Path(checkpoint), Path(stats_dir), Path(input)
    output = Path(output)
    forecast_output = None if forecast_output is None else Path(forecast_output)
    _output_paths(output, forecast_output)
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    sources = [checkpoint_path, input_path] + [stats_dir / name for name in STATISTICS_FILES]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    flags = configure_runtime()
    import haiku as hk
    import jax
    import xarray as xr
    from graphcast import casting, checkpoint as checkpoint_io, graphcast, icosahedral_mesh, normalization, rollout

    direct = json.loads(importlib.metadata.distribution("graphcast").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != UPSTREAM_COMMIT:
        raise RuntimeError(f"Install official GraphCast at pinned commit {UPSTREAM_COMMIT}")
    with checkpoint_path.open("rb") as stream:
        ckpt = checkpoint_io.load(stream, graphcast.CheckPoint)
    mean, std, diff = [xr.load_dataset(stats_dir / name) for name in STATISTICS_FILES]
    with xr.open_dataset(input_path) as dataset:
        inputs, targets, forcings, initial = prepare_inputs(dataset.load(), ckpt.task_config, ckpt.model_config, steps)
    for stats, names, label in (
        (mean, set(ckpt.task_config.input_variables) | set(ckpt.task_config.target_variables), "mean"),
        (std, set(ckpt.task_config.input_variables) | set(ckpt.task_config.target_variables), "stddev"),
        (diff, set(ckpt.task_config.target_variables) & set(ckpt.task_config.input_variables), "diffs_stddev"),
    ):
        if names - set(stats):
            raise ValueError(f"Missing {label} statistics: {sorted(names - set(stats))}")
        selected = stats[list(names)].sel(level=list(ckpt.task_config.pressure_levels))
        _finite(selected, label)
        if label != "mean" and any((value.values <= 0).any() for value in selected.data_vars.values()):
            raise ValueError(f"{label} statistics must be strictly positive")
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=ckpt.model_config.mesh_size)
    vertices = meshes[-1].vertices
    senders, receivers = icosahedral_mesh.faces_to_edges(icosahedral_mesh.merge_meshes(meshes).faces)
    graph_path = output.with_suffix(".graph.npz")
    graph_partial = output.with_suffix(".graph.npz.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with graph_partial.open("xb") as stream:
        np.savez(stream, vertices=vertices, senders=senders, receivers=receivers)
        stream.flush()
        os.fsync(stream.fileno())
    graph_hash = _sha256(graph_partial)
    metadata = {
        "upstream_commit": UPSTREAM_COMMIT,
        "model_config": dataclasses.asdict(ckpt.model_config),
        "task_config": dataclasses.asdict(ckpt.task_config),
        "checkpoint_sha256": _sha256(checkpoint_path), "input_sha256": _sha256(input_path),
        "statistics_sha256": {name: _sha256(stats_dir / name) for name in STATISTICS_FILES},
        "initialization_time_utc": str(initial),
        "history_times_utc": [str(initial + np.timedelta64(hours, "h")) for hours in (-6, 0)],
        "history_offsets_hours": [-6, 0], "input_leads_hours": list(range(0, steps * 6, 6)),
        "forecast_leads_hours": list(range(6, (steps + 1) * 6, 6)),
        "forecast_times_utc": [str(initial + np.timedelta64(hours, "h"))
                               for hours in range(6, (steps + 1) * 6, 6)],
        "latent_time_semantics": "state at input lead, used to predict input lead + 6 hours",
        "batch_identifiers": inputs.batch.values.tolist(),
        "inference_dtype": "bfloat16", "export_dtype": "float32",
        "node_order": "official finest_mesh.vertices order; merged hierarchy face edges",
        "graph_file": graph_path.name, "graph_sha256": graph_hash,
        "xla_flags": flags,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("graphcast", "jax", "jaxlib", "dm-haiku", "xarray", "numpy")},
    }
    shape = (steps, ckpt.model_config.gnn_msg_steps + 1, len(vertices), 1, ckpt.model_config.latent_size)
    with LatentWriter(output, shape, metadata) as writer:
        def forward(inputs, targets_template, forcings):
            model = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
            if not np.array_equal(model._finest_mesh.vertices, vertices):
                raise RuntimeError("Exported graph does not match model node order")
            predictor = normalization.InputsAndResiduals(
                casting.Bfloat16Cast(model), diffs_stddev_by_level=diff,
                mean_by_level=mean, stddev_by_level=std)
            with capture_latents(model, writer.append):
                return predictor(inputs, targets_template=targets_template, forcings=forcings)

        apply = jax.jit(hk.transform_with_state(forward).apply)
        params = jax.device_put(ckpt.params)
        state = {}

        def predict(rng, inputs, targets_template, forcings):
            nonlocal state
            forecast, state = apply(params, state, rng, inputs, targets_template, forcings)
            try:
                jax.block_until_ready((forecast, state))
            finally:
                jax.effects_barrier()
            forecast = jax.device_get(forecast)
            _finite(forecast, "Forecast")
            return forecast

        forecasts = []
        for index, forecast in enumerate(rollout.chunked_prediction_generator(
            predict, jax.random.PRNGKey(0), inputs, targets, forcings, num_steps_per_chunk=1
        )):
            if forecast_output is not None:
                forecasts.append(forecast)
            print(json.dumps({"completed_step": index + 1, "steps": steps}), flush=True)
        jax.effects_barrier()
        if forecast_output is not None:
            forecast_output.parent.mkdir(parents=True, exist_ok=True)
            partial = forecast_output.with_suffix(".nc.partial")
            with partial.open("xb"):
                pass
            combined = xr.concat(forecasts, dim="time")
            combined = combined.assign_coords(datetime=("time", initial + combined.time.values))
            combined.to_netcdf(partial, engine="h5netcdf")
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            writer.metadata["forecast_file"] = os.path.relpath(forecast_output, output.parent)
            writer.metadata["forecast_sha256"] = _sha256(partial)
            os.link(partial, forecast_output)
            partial.unlink()
        os.link(graph_partial, graph_path)
        graph_partial.unlink()
        # The writer publishes its completion JSON only after these sidecars.
    return output


def main() -> None:
    """Command-line entry point; --help does not import inference libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NetCDF with two six-hourly UTC initialization frames")
    parser.add_argument("--steps", type=int, required=True, help="Number of six-hour forecast steps")
    parser.add_argument("--output", type=Path, required=True, help="New latent .npy path")
    parser.add_argument("--forecast-output", type=Path, help="Optional new forecast .nc path")
    args = parser.parse_args()
    result = run_extraction(**vars(args))
    print(json.dumps({"complete": True, "output": str(result)}), flush=True)


if __name__ == "__main__":
    main()
