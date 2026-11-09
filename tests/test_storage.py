import json

import numpy as np
import pytest

from graphcast_latent.storage import LatentWriter


def test_round_trip_and_completion(tmp_path):
    path = tmp_path / "latent.npy"
    expected = np.arange(2 * 3 * 4 * 2 * 5, dtype=np.float32).reshape(2, 3, 4, 2, 5)
    with LatentWriter(path, expected.shape, {"input_lead_hours": [0, 6]}) as writer:
        for lead in expected:
            for state_index, values in enumerate(lead):
                writer.append(state_index, values)
        assert not path.exists()
    np.testing.assert_array_equal(np.load(path), expected)
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["state_names"] == ["encoder", "processor_step_01", "processor_step_02"]
    assert record["complete"] and record["metadata"]["input_lead_hours"] == [0, 6]
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        LatentWriter(path, expected.shape)
    np.testing.assert_array_equal(np.load(path), expected)


def test_incomplete_export_stays_partial(tmp_path):
    path = tmp_path / "latent.npy"
    with pytest.raises(RuntimeError, match="incomplete"):
        with LatentWriter(path, (1, 2, 3, 1, 2)) as writer:
            writer.append(0, np.zeros((3, 1, 2)))
    assert not path.exists() and not path.with_suffix(".json").exists()
    assert path.with_suffix(".npy.partial").exists()


@pytest.mark.parametrize("problem", ["order", "shape", "nonfinite"])
def test_rejects_bad_states_without_publishing(tmp_path, problem):
    path = tmp_path / "latent.npy"
    with pytest.raises(ValueError):
        with LatentWriter(path, (1, 2, 3, 1, 2)) as writer:
            if problem == "order":
                writer.append(1, np.zeros((3, 1, 2)))
            elif problem == "shape":
                writer.append(0, np.zeros((2, 1, 3)))
            else:
                writer.append(0, np.full((3, 1, 2), np.nan))
    assert not path.exists() and not path.with_suffix(".json").exists()


def test_publication_does_not_overwrite_concurrent_file(tmp_path):
    path = tmp_path / "latent.npy"
    with pytest.raises(FileExistsError):
        with LatentWriter(path, (1, 1, 2, 1, 1)) as writer:
            writer.append(0, np.zeros((2, 1, 1)))
            path.write_bytes(b"another writer")
    assert path.read_bytes() == b"another writer"
    assert not path.with_suffix(".json").exists()


def test_metadata_collision_unpublishes_only_owned_data(tmp_path):
    path = tmp_path / "latent.npy"
    with pytest.raises(FileExistsError):
        with LatentWriter(path, (1, 1, 2, 1, 1)) as writer:
            writer.append(0, np.zeros((2, 1, 1)))
            path.with_suffix(".json").write_text("another writer")
    assert not path.exists()
    assert path.with_suffix(".json").read_text() == "another writer"
    assert path.with_suffix(".npy.partial").exists()


def test_rejects_fractional_state_index(tmp_path):
    with pytest.raises(TypeError):
        with LatentWriter(tmp_path / "latent.npy", (1, 1, 2, 1, 1)) as writer:
            writer.append(0.9, np.zeros((2, 1, 1)))


def test_rejects_complex_latents(tmp_path):
    with pytest.raises(ValueError, match="real-valued"):
        with LatentWriter(tmp_path / "latent.npy", (1, 1, 2, 1, 1)) as writer:
            writer.append(0, np.full((2, 1, 1), 1 + 2j))
