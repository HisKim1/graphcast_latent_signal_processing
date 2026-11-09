"""Runner boundary checks; real checkpoint execution is recorded in validation."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from graphcast_latent.run_extraction import _output_paths, prepare_inputs, run_extraction


def test_help_does_not_import_inference_libraries():
    code = """
import sys
from graphcast_latent.run_extraction import main
sys.argv = ['graphcast-latent-extract', '--help']
try:
    main()
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('jax', 'haiku', 'graphcast', 'xarray'))
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)


def test_compiler_defaults_preserve_other_flags_and_require_early_setup():
    code = """
import os, sys
from graphcast_latent.run_extraction import configure_runtime
os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=another-pass'
flags = configure_runtime()
assert '--xla_gpu_deterministic_ops=true' in flags
assert 'another-pass,fusion-dynamic-memcpy-rewriter' in flags
assert configure_runtime() == flags
sys.modules['jax'] = object()
os.environ['XLA_FLAGS'] = ''
try:
    configure_runtime()
except RuntimeError:
    pass
else:
    raise AssertionError('must not configure after JAX import')
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


@pytest.mark.parametrize("suffix", [".npy", ".json", ".npy.partial", ".json.partial", ".graph.npz", ".graph.npz.partial"])
def test_existing_latent_output_blocks_execution_before_loading(tmp_path, suffix):
    output = tmp_path / "latents.npy"
    existing = output.with_suffix(suffix)
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        run_extraction(checkpoint="missing", stats_dir="missing", input="missing", steps=1, output=output)
    assert existing.read_bytes() == b"preserve"


def test_forecast_collision_and_broken_symlinks_are_rejected(tmp_path):
    forecast = tmp_path / "forecast.nc"
    forecast.with_suffix(".nc.partial").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        _output_paths(tmp_path / "latent.npy", forecast)
    output = tmp_path / "other.npy"
    output.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(FileExistsError):
        _output_paths(output, None)


@pytest.fixture
def prepared_source():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("graphcast")
    coordinates = {
        "time": np.array(["2022-01-01T18", "2022-01-02T00"], dtype="datetime64[ns]"),
        "lat": [-90.0, 0.0, 90.0], "lon": [0.0, 90.0, 180.0, 270.0], "level": [500, 850],
    }
    dataset = xr.Dataset(coords=coordinates)
    dataset["temperature"] = (("time", "level", "lat", "lon"), np.ones((2, 2, 3, 4), dtype=np.float32) * 270)
    dataset["2m_temperature"] = (("time", "lat", "lon"), np.ones((2, 3, 4), dtype=np.float32) * 280)
    dataset["land_sea_mask"] = (("lat", "lon"), np.ones((3, 4), dtype=np.float32))
    task = SimpleNamespace(
        input_variables=("temperature", "2m_temperature", "land_sea_mask", "day_progress_sin"),
        target_variables=("temperature", "2m_temperature", "total_precipitation_6hr"),
        forcing_variables=("day_progress_sin",), pressure_levels=(850, 500), input_duration="12h")
    return dataset, task, SimpleNamespace(resolution=90.0)


def test_initialization_only_builds_nan_targets_and_future_forcings(prepared_source):
    dataset, task, model = prepared_source
    inputs, targets, forcings, initial = prepare_inputs(dataset, task, model, 2)
    assert initial == np.datetime64("2022-01-02T00")
    np.testing.assert_array_equal(inputs.time.values, np.array([-6, 0], dtype="timedelta64[h]"))
    np.testing.assert_array_equal(targets.time.values, np.array([6, 12], dtype="timedelta64[h]"))
    np.testing.assert_array_equal(inputs.level, [850, 500])
    assert inputs.land_sea_mask.dims == ("lat", "lon")
    assert targets.temperature.shape == (1, 2, 3, 4, 2)
    assert targets.total_precipitation_6hr.shape == (1, 2, 3, 4)
    assert all(np.isnan(value.values).all() for value in targets.data_vars.values())
    assert all(np.isfinite(value.values).all() for value in forcings.data_vars.values())
    assert dataset.sizes["time"] == 2 and "batch" not in dataset.dims


@pytest.mark.parametrize("failure", ["time", "grid", "levels", "nonfinite", "variable", "static"])
def test_invalid_inputs_fail_clearly(prepared_source, failure):
    dataset, task, model = prepared_source
    if failure == "time":
        dataset = dataset.assign_coords(time=dataset.time.values + np.array([0, 1], dtype="timedelta64[h]"))
    elif failure == "grid":
        dataset = dataset.assign_coords(lon=[0, 90, 180, 360])
    elif failure == "levels":
        dataset = dataset.sel(level=[500])
    elif failure == "nonfinite":
        dataset["temperature"].values[0, 0, 0, 0] = np.nan
    elif failure == "variable":
        dataset = dataset.drop_vars("temperature")
    elif failure == "static":
        dataset["land_sea_mask"] = dataset.land_sea_mask.expand_dims(time=dataset.time)
    with pytest.raises(ValueError):
        prepare_inputs(dataset, task, model, 2)
