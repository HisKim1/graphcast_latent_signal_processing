# graphcast-latent

A small Python package for extracting GraphCast mesh states and computing graph Fourier and Dirichlet diagnostics from stored latent arrays. Computation is independent of plotting. The analysis package does not import GraphCast, JAX, Matplotlib, or Cartopy.

## Installation

Python 3.12 is the validated interpreter. Clone the repository and install the locked analysis environment:

```bash
git clone https://github.com/HisKim1/graphcast_latent_signal_processing.git
cd graphcast_latent_signal_processing
uv sync --locked
```

Install the optional extraction dependencies, CUDA 12 support, and tests when needed:

```bash
uv sync --locked --extra extraction --extra gpu --extra test
JAX_PLATFORMS=cpu uv run --no-sync pytest
```

The extraction extra pins the official GraphCast revision `7077d40a36db6541e3ed72ccaed1c0d202fa6014`. It does not include model weights, normalization statistics, input weather data, or GPU drivers. The official [GraphCast README](https://raw.githubusercontent.com/google-deepmind/graphcast/7077d40a36db6541e3ed72ccaed1c0d202fa6014/README.md) identifies the available checkpoints and associated assets. The `gpu` extra installs the pinned JAX CUDA 12 runtime; omit it for CPU-only extraction. A compatible NVIDIA driver is provided by the host, not this package. The prepared environment was verified to detect a GPU.

## Extract from a checkpoint

The installed command loads the checkpoint and normalization files, builds the
unchanged official model, runs six-hour autoregressive forecast steps, and saves
encoder/processor states. No model-forward edits or caller-written Haiku function
are required. After the locked extraction installation above, run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync graphcast-latent-extract \
  --checkpoint checkpoint.npz \
  --stats-dir normalization \
  --input inputs.nc \
  --steps 2 \
  --output output/latents.npy \
  --forecast-output output/forecast.nc
```

All arguments except `--forecast-output` are required. `--steps` is a positive
integer; it counts six-hour forecast steps. Omit `--forecast-output` to save only
latents and graph geometry. Keep `--no-sync` when using `uv run` after installing
extras so an implicit sync does not remove optional dependencies.
`uv run --no-sync graphcast-latent-extract --help` does not import inference
libraries or initialize a device.

The checkpoint must use the pinned official GraphCast checkpoint schema,
including parameters, model configuration, and task configuration. Download
checkpoint and normalization assets from the official source linked above;
this command does not download them. `--stats-dir` must contain:

```text
stats_mean_by_level.nc
stats_stddev_by_level.nc
stats_diffs_stddev_by_level.nc
```

The pinned official [demo's normalization-loading cell](https://github.com/google-deepmind/graphcast/blob/7077d40a36db6541e3ed72ccaed1c0d202fa6014/graphcast_demo.ipynb)
uses shorter download names under the GraphCast bucket's `stats/` directory.
Copy those files into your local statistics directory with this mapping:

| Official download filename | Required local filename |
| --- | --- |
| `mean_by_level.nc` | `stats_mean_by_level.nc` |
| `stddev_by_level.nc` | `stats_stddev_by_level.nc` |
| `diffs_stddev_by_level.nc` | `stats_diffs_stddev_by_level.nc` |

Their variables and pressure levels must cover the checkpoint task. Selected
statistics must be finite, and standard deviations must be strictly positive.
Model dimensions and required input variables come from the checkpoint, rather
than resolution or model-name defaults in this package.

### Input NetCDF contract

Supply one initialization per call, with these coordinates and variables:

| Item | Required content |
| --- | --- |
| `time` | Exactly two UTC `datetime64` values, in increasing order and six hours apart. The later value is initialization. |
| `batch` | Optional; when present its size must be one. Otherwise a single batch member is added. The lower-level capture API supports native batches. |
| `lat` | One-dimensional regular global latitude grid, including both poles, at checkpoint resolution. Ascending or descending order is accepted. |
| `lon` | One-dimensional ascending global grid from 0 to less than 360 degrees, at checkpoint resolution; no duplicate 360-degree endpoint. |
| `level` | Unique pressure levels in hPa containing every checkpoint task level; extra levels are selected out. |
| Weather fields | Every non-forcing variable in the checkpoint's `task_config.input_variables`, in physical units expected by GraphCast. Surface fields have `(batch, time, lat, lon)` dimensions; atmospheric fields additionally have `level`. Dimension order may vary. |
| Static fields | Checkpoint static variables, including `geopotential_at_surface` and `land_sea_mask`, on `(lat, lon)`, with no `time` dimension. A singleton `batch` dimension is accepted and removed. |

The `batch` dimension may be omitted from weather fields when omitted from the
dataset. Required data must remain finite after conversion to float32.
Atmospheric geopotential is not geopotential height, and the command does not
convert weather units or regrid input fields. Errors list missing checkpoint
variables or incompatible coordinates/dimensions.

Future weather truth is unnecessary. Target shapes are constructed from the
checkpoint task using NaN placeholders. Official GraphCast utilities generate
future time/solar forcings from the initialization, grid, and six-hour leads.
Historical forcings supplied in the file are retained; missing historical
forcings are generated. Supplied forcings must follow GraphCast's coordinate and
physical conventions. An input `datetime` coordinate is ignored in favor of the
validated `time` coordinate.

### Outputs and failure behavior

For an output named `latents.npy`, the command writes:

| File | Content |
| --- | --- |
| `latents.npy` | Float32 array with axes `(lead, state, node, batch, channel)`. State zero is the encoder; subsequent states follow processor residual updates. Sizes follow the checkpoint and requested steps. |
| `latents.graph.npz` | Unit-sphere `vertices` and integer `senders`/`receivers` from the official merged mesh hierarchy, in the captured latent node order. |
| `latents.json` | Completion record with `complete: true`, axes, shape, dtype, state names, and provenance metadata. |
| Optional forecast `.nc` | Forecast variables with six-hour lead coordinates and absolute forecast times, when `--forecast-output` is supplied. |

Latent lead zero is the state produced from the two initialization frames and
used to predict +6 h. Subsequent latent leads are 6 h, 12 h, and so on. Metadata
records input/forecast leads and UTC times, checkpoint/task configurations,
batch identifiers, bfloat16 inference and float32 export dtypes, package
versions, compiler flags, SHA-256 values for the checkpoint/input/statistics,
and the graph sidecar filename/hash. Optional forecast filename/hash are also
recorded. The mesh vertices are checked against the actual model during tracing.
Pass the saved geometry to `graph_from_mesh` to construct the analysis graph.

Every output and its partial-file name must be unused, including graph and
optional forecast sidecars. Existing files and symlinks are rejected. Inputs,
statistics, captured latents and forecasts are checked for finite values;
future target placeholders are deliberately NaN. Failed runs retain partial
files, and may leave already published sidecars, for inspection. Consumers must
require the final matching JSON completion record and verify sidecar hashes.
Reruns need a new output name or deliberate removal of the failed run's files;
the command never overwrites them.

Before importing JAX, the command adds
`--xla_gpu_deterministic_ops=true` and disables the
`fusion-dynamic-memcpy-rewriter` HLO pass, preserving other supplied XLA options.
It rejects a conflicting deterministic setting. If JAX was already imported by
a Python caller, both settings must already be present. GPU visibility is left
to the caller, and GPU preallocation defaults to false. These settings support
the recorded compatibility checks; they do not establish equality across
untested hardware, library versions, or compiler configurations.

## Analyze the extracted output

Continue from the command above, using its `output/latents.npy` and matching
sidecars. For the first complete Fourier walkthrough, use the GraphCast Small
checkpoint. This block checks the completion record, saved array dimensions,
and graph SHA-256 before selecting **lead 0, encoder state, batch 0, all
channels**. The resulting signal has shape `(node, channel)`; no channel
averaging or normalization is applied.

Sparse Dirichlet calculations run first and do not need a Fourier basis.
The final section uses native PyGSP methods for a full GFT and inverse GFT,
checks reconstruction and Parseval's identity, and compares the spectral and
sparse forms of Dirichlet energy. Its tolerances apply to float64 arithmetic,
not to equality of individual eigenvectors from different eigensolvers.

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run --no-sync python - <<'PY'
import hashlib
import json
from pathlib import Path

import numpy as np
from graphcast_latent import (
    channel_dirichlet_energy, dirichlet_energy, graph_from_mesh,
    local_dirichlet_energy, spectral_power,
)

path = Path("output/latents.npy")
record = json.loads(path.with_suffix(".json").read_text())
assert record["complete"] is True and record["format_version"] == 1
assert record["axes"] == ["lead", "state", "node", "batch", "channel"]
metadata = record["metadata"]
graph_path = path.parent / metadata["graph_file"]
assert hashlib.sha256(graph_path.read_bytes()).hexdigest() == metadata["graph_sha256"]

latents = np.load(path, mmap_mode="r", allow_pickle=False)
assert list(latents.shape) == record["shape"]
assert latents.dtype == np.dtype(record["dtype"]) == np.dtype("float32")
with np.load(graph_path, allow_pickle=False) as mesh:
    graph = graph_from_mesh(mesh["vertices"], mesh["senders"], mesh["receivers"],
                            lap_type="normalized")
assert graph.N == latents.shape[2]
lead, state, batch = 0, record["state_names"].index("encoder"), 0
signal = np.asarray(latents[lead, state, :, batch, :], dtype=np.float64)
assert signal.shape == (graph.N, latents.shape[4]) and np.isfinite(signal).all()

energy = dirichlet_energy(signal, graph)
channel_energy = channel_dirichlet_energy(signal, graph)
local_energy = local_dirichlet_energy(signal, graph)
np.testing.assert_allclose(channel_energy.sum(), energy, rtol=1e-9, atol=1e-8)
np.testing.assert_allclose(local_energy.sum(), energy, rtol=1e-9, atol=1e-8)
assert (local_energy >= 0).all()
signal_energy = float(np.sum(signal**2))
roughness = energy / signal_energy if signal_energy else float("nan")
print({"input_lead_hours": metadata["input_leads_hours"][lead],
       "state": record["state_names"][state], "signal_shape": signal.shape,
       "raw_dirichlet_energy": energy, "signal_energy": signal_energy,
       "normalized_roughness": roughness})

# Keep this walkthrough's dense eigensolve bounded to the Small mesh.
if graph.N > 10_242:
    print("Sparse energy checks passed; full GFT skipped for this larger mesh.")
    raise SystemExit(0)
graph.compute_fourier_basis()
coefficients = graph.gft(signal)
reconstructed = graph.igft(coefficients)
power = spectral_power(coefficients)  # One value per mode, summed over channels.
np.testing.assert_allclose(reconstructed, signal, rtol=1e-9, atol=1e-9)
np.testing.assert_allclose(power.sum(), signal_energy, rtol=1e-9, atol=1e-8)
spectral_energy = float(graph.e @ power)
np.testing.assert_allclose(spectral_energy, energy, rtol=1e-9, atol=1e-8)
print({"gft_shape": coefficients.shape, "spectral_dirichlet_energy": spectral_energy,
       "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - signal))),
       "checks": "reconstruction, Parseval, spectral/sparse energy passed"})
PY
```

`lap_type="normalized"` selects the normalized graph Laplacian. The reported
`raw_dirichlet_energy` is still the unnormalized quadratic energy
`sum(signal * (L @ signal))`; `normalized_roughness` explicitly divides it by
`sum(signal**2)`. Local energy is the nonnegative half-edge allocation, whose
node sum equals the global energy.

Full Fourier bases are expensive: one float64 square array uses about 0.78 GiB
for 10,242 nodes and 12.50 GiB for 40,962 nodes. The eigensolver needs multiple
such arrays plus workspace, has cubic computational cost, and can take minutes
or longer. The block therefore limits CPU threads and skips the full solve for
larger meshes; it still computes their sparse energies. Do not remove that
guard without planning the memory and runtime. Reuse a computed graph basis
for additional states in the same Python session instead of solving it again.

## Modules

| Module | Responsibility |
| --- | --- |
| `graphcast_latent.graph` | Validate adjacency matrices or construct inverse-arc-weighted analysis graphs from supplied mesh vertices and edges. |
| `graphcast_latent.spectral` | Spectral power and global, channel, and local Dirichlet energies. |
| `graphcast_latent.extraction` | Observe an unchanged GraphCast forward pass through Haiku interception. Requires the extraction extra. |
| `graphcast_latent.storage` | Write complete, non-overwriting NPY exports with a JSON completion record. |
| `graphcast_latent.run_extraction` | Load checkpoint/NetCDF assets, prepare forcing/templates, execute forecasts, and coordinate latent/geometry export. Requires the extraction extra. |

Graph construction returns a PyGSP `Graph`; no duplicate graph class is introduced. Supply vertices and edges in exactly the latent node order. The published multimesh diagnostic requires edges from the merged hierarchy containing all used mesh levels, including repeated edges. The analysis graph is a symmetric inverse-arc-weighted diagnostic graph, separate from the model's learned message-passing operator.

## Array and energy contracts

A signal has shape `(node, channel)`. Energy calculations convert real input to float64 and do not center, average, or normalize channels. A stored export has axes `(lead, state, node, batch, channel)`; select a lead, state, and batch member before passing a signal to the analysis API.

For positive-degree vertices, the normalized Laplacian is `L = I - D^(-1/2) W D^(-1/2)`; isolated vertices have zero rows, following PyGSP. Global Dirichlet energy is `sum(signal * (L @ signal))`; channel energies partition that total. Local energy splits each undirected edge contribution equally between its two endpoints, using `signal / sqrt(degree)` for the normalized operator. Combinatorial graphs instead use `L = D - W` and unscaled signals. Sparse energy evaluation does not require an eigendecomposition.

Signal energy, `sum(signal**2)`, differs from Dirichlet energy. Dividing Dirichlet energy by signal energy gives normalized roughness; the normalized Laplacian alone does not perform that division. Case averaging, paired ratios, spatial display interpolation, and plotting are outside this package.

## Analysis API

| Function | Inputs and result |
| --- | --- |
| `graph_from_adjacency(adjacency, *, lap_type="normalized")` | Symmetric, nonnegative, loop-free adjacency to a PyGSP `Graph`. |
| `graph_from_mesh(vertices, senders, receivers, *, lap_type="normalized")` | Unit-sphere vertices and indexed edges to a graph; preserves node order and edge multiplicity. |
| `spectral_power(coefficients)` | Sum squared coefficients across channels; returns one value per mode. |
| `dirichlet_energy(signal, graph)` | Scalar raw energy, summed across channels. |
| `channel_dirichlet_energy(signal, graph)` | One raw energy per channel. |
| `local_dirichlet_energy(signal, graph)` | Nonnegative half-edge contributions, one per node, summing to global energy. |

For new analyses, call the returned graph's `compute_fourier_basis()`, then use its `gft(signal)` and `igft(coefficients)` methods. `graph.e` gives the ordered eigenvalues, and `graph.U` gives their eigenvectors. These are the native PyGSP methods. Full eigensystems require quadratic storage; sparse energy evaluation avoids that cost.

A fresh eigensolve can change individual modes at floating-point precision, especially near repeated eigenvalues. Validate eigenpairs, orthogonality, and node ordering when preparing a basis. The validation record preserves the failed strict mode-equality check and separately evaluates whether those differences affect the measured research conclusions. No private PyGSP basis attributes are assigned.

## Extraction and storage API

`graphcast_latent.extraction.capture_latents(model, sink)` is a context manager used inside the Haiku-transformed forward function, before JIT tracing. `sink(state_index, values)` receives a host array with shape `(node, batch, channel)`. State zero is the post-encoder mesh state; subsequent states follow each processor residual update. The number of states follows the supplied model.

The lower-level capture API leaves compiler settings to its caller. Validated GPU comparisons set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` before importing JAX; larger checkpoints additionally disable `fusion-dynamic-memcpy-rewriter`. The CLI applies both settings as described above. Exact reference-capture agreement requires the recorded settings.

The callback path is for inference with the model's native batch dimension; it does not support automatic differentiation or an outer `vmap`. Call `jax.effects_barrier()` before finishing the writer so asynchronous callbacks have completed. Recreate the transformed/JIT function when changing its captured sink.

`graphcast_latent.storage.LatentWriter(path, shape, metadata=None)` is a context manager. Its `append(state_index, values)` method accepts one state in lead-then-state order. The declared five-dimensional shape determines all sizes; there are no fixed cohort, date, resolution, or processor-count defaults. Exports use float32.

Only a complete export receives the final NPY filename and JSON completion record. Existing files are never overwritten; failed exports retain partial files. Consumers should require the matching JSON record with `complete: true`. Record input leads, batch identifiers, checkpoint identity, normalization assets, and mesh/node ordering in `metadata`; these are provided by the caller rather than inferred from filenames.

## Validation and distribution

[VALIDATION.md](VALIDATION.md) records the checks actually performed and their limits. Tests use synthetic data and do not require research datasets or pretrained weights. Private compatibility fixtures and model assets are not distributed.

The source repository is [HisKim1/graphcast_latent_signal_processing](https://github.com/HisKim1/graphcast_latent_signal_processing). The project license will be decided later; [NOTICE.md](NOTICE.md) records the independent upstream terms. No archival DOI or manuscript citation is assigned here.
