"""Sequential, non-overwriting NPY export for extracted latent states."""

import json
import os
import operator
from pathlib import Path
import shutil
from typing import Any

import numpy as np


class LatentWriter:
    """Write float32 arrays with axes (lead, state, node, batch, channel).

    ``append`` accepts one (node, batch, channel) state at a time, ordered by
    lead then state. A normal context exit checks completeness and publishes
    the NPY and JSON completion record. Failed exports retain ``.partial``
    files and never replace existing output. Metadata should include input
    leads, batch identifiers, model/checkpoint provenance and node ordering.
    """

    def __init__(
        self,
        path: str | Path,
        shape: tuple[int, int, int, int, int],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.suffix != ".npy":
            raise ValueError("Latent output must use the .npy suffix")
        if len(shape) != 5 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("shape must contain five positive integers")
        self.shape = shape
        self.metadata = json.loads(json.dumps({} if metadata is None else metadata, allow_nan=False))
        self.metadata_path = self.path.with_suffix(".json")
        self.partial_path = self.path.with_suffix(".npy.partial")
        self.partial_metadata_path = self.path.with_suffix(".json.partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in (
            self.path, self.metadata_path, self.partial_path, self.partial_metadata_path
        ):
            if candidate.exists():
                raise FileExistsError(candidate)
        if shutil.disk_usage(self.path.parent).free < np.prod(shape, dtype=object) * 4 + 1024**2:
            raise OSError("Insufficient disk space for the declared latent array")
        self._file = self.partial_path.open("xb")
        np.lib.format.write_array_header_1_0(
            self._file, {"descr": np.dtype("float32").str, "fortran_order": False, "shape": shape}
        )
        self._written = 0
        self._finished = False

    def append(self, state_index: int, values: np.ndarray) -> None:
        """Append exactly the next state, rejecting missing or repeated states."""
        if self._file.closed or self._written >= self.shape[0] * self.shape[1]:
            raise RuntimeError("Latent output is closed or already full")
        if operator.index(state_index) != self._written % self.shape[1]:
            raise ValueError("Latent states arrived out of order")
        if np.iscomplexobj(values):
            raise ValueError("GraphCast latent states must be real-valued")
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.shape != self.shape[2:] or not np.isfinite(array).all():
            raise ValueError("Latent state has an unexpected shape or nonfinite values")
        self._file.write(memoryview(array).cast("B"))
        self._written += 1

    def finish(self) -> None:
        """Publish complete output; the JSON is the completion record."""
        if self._finished:
            return
        if self._written != self.shape[0] * self.shape[1]:
            raise RuntimeError("Cannot publish an incomplete latent array")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        record = {
            "format_version": 1,
            "complete": True,
            "axes": ["lead", "state", "node", "batch", "channel"],
            "shape": list(self.shape),
            "dtype": "float32",
            "state_names": ["encoder"] + [
                f"processor_step_{step:02d}" for step in range(1, self.shape[1])
            ],
            "metadata": self.metadata,
        }
        with self.partial_metadata_path.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard links publish atomically without overwriting a competing writer.
        os.link(self.partial_path, self.path)
        try:
            os.link(self.partial_metadata_path, self.metadata_path)
        except BaseException:
            if self.path.exists() and self.path.samefile(self.partial_path):
                self.path.unlink()
            raise
        self.partial_path.unlink()
        self.partial_metadata_path.unlink()
        self._finished = True

    def __enter__(self) -> "LatentWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.finish()
        finally:
            self._file.close()
