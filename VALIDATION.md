# Validation

The package uses native PyGSP graphs and Fourier transforms. [Numerical diagnostics](validation/numerical_comparison.json), [research comparisons](validation/research_comparison.json), original [capture-adapter checks](validation/extraction_compatibility.json), and current [console-runner checks](validation/cli_compatibility.json) distinguish measured agreement, mathematical bounds, and untested scope. Original research arrays and figures are preserved. Earlier records remain separate from the executable-runner validation.

## Executable checkpoint runner

The installed `graphcast-latent-extract` command was executed in the project
environment against three real checkpoints on NVIDIA H200 NVL GPUs. Each run
loaded its checkpoint, normalization files and two initialization frames from
NetCDF, generated future forcings and NaN target templates, and completed
without future weather truth or caller-written model-forward code.

| Checkpoint | Forecast steps | Saved float32 shape | Captured states |
| --- | --- | --- | --- |
| Small, 1 degree | 2 | `(2, 17, 10242, 1, 512)` | 34 |
| Operational, 0.25 degree | 1 | `(1, 17, 40962, 1, 512)` | 17 |
| AMSE fine-tuned, 0.25 degree | 1 | `(1, 17, 40962, 1, 512)` | 17 |

All saved states exactly match the independent legacy-method capture. All 11
forecast variables per checkpoint exactly match independent plain predictions
after matching named dimension order. Batch, latitude, longitude, pressure-level
and lead coordinates match. Absolute UTC valid times are checked independently
against initialization plus the requested forecast leads: the reference
forecast's inherited datetime metadata was unsuitable for that comparison.
This is value/coordinate agreement, not a claim that NetCDF files have identical
serialized bytes or that the reference's absolute datetime metadata is correct.

Each completed NPY/JSON was reloaded; finite values, state order and metadata
were checked. Exported vertices and merged-hierarchy edge indices exactly match
the official mesh and captured node order. A four-channel integration check
with the public analysis API confirms that local Dirichlet energies sum to the
global energy within `1e-12` relative tolerance. Checkpoint/input/statistics and
geometry fingerprints are recorded. Input assets are checked again after the
runs; original weather archives are read only.

The runner source hash is recorded in the CLI report, together with the pinned
upstream revision and exact compiler flags. The accepted record uses the frozen
source and validates absolute UTC clocks independently from the reference
metadata. Earlier capture diagnostics remain documented separately. Real CLI
coverage is one initialization and batch one per
checkpoint, with the step counts above; longer rollouts, other checkpoints and
other hardware are not established by these checks.

## Mathematical compatibility

Eight representative encoder/final-processor states were compared using float64 analysis of stored float32 latents. Adjacency is exact and the maximum normalized-Laplacian difference is `8.33e-17`. Maximum relative global, channel, and local energy differences are `2.79e-14`, `3.07e-15`, and `1.32e-15`.

A fresh PyGSP basis passes eigenvalue, reconstruction, and Parseval checks. Individual-mode power differs by up to `1.39e-6`, exceeding the strict `3e-12` pointwise criterion. That failed criterion remains recorded. Scientific adoption instead requires the measured conclusions to remain stable. Supplied-basis projection is retained only as a reference calculation in the comparison tool; the public API uses PyGSP directly.

## Scientific coverage

Full-cohort spectral projections use float64 GPU multiplication with a fresh PyGSP basis. Native CPU PyGSP transforms were checked on the eight representative states. The full thermodynamic-selection comparison calls the public local-energy function directly on the CPU.

| Evidence | Check and result |
| --- | --- |
| Main encoder spectra | Both streams, all 208 cases, all 21 displayed leads through 120 h, plus 240 h. Maximum spectrum relative L2 change `2.41e-9`; maximum mean-mode relative change `3.00e-7`. |
| Energy deficits and normalized spectra | Ratio changes at most `3.45e-15`; one-decimal deficits are unchanged. Smoothed normalized spectral-ratio maximum absolute change is `2.81e-9`. All checked frequency counts and selected peak locations are preserved. |
| Signed peak maps | Full-cohort maps at 120 h differ by at most `6.39e-10` in relative L2 norm and `2.14e-12` in absolute value. Two nearly zero vertices change sign for one mode, with reference magnitudes below `5.78e-13`. |
| Thermodynamic selection and profiles | Direct local-energy evaluation on 416 states at 24 h gives maximum relative change `9.56e-14`. Both domains retain every vertex rank, tail ID/order, and selection mask exactly. Profiles and valid counts recomputed from frozen physical predictors match exactly, preserving inputs to weekly confidence bands, paired tests, recurrence maps, and advection comparisons. |
| Main processor and global diagnostics | The same graph functional and an operator perturbation bound cover 44,512 finite cached energies, with relative real-arithmetic bound `5.82e-15`. The encoder bound covers both streams, all cases, and all 41 saved leads, with bound `5.89e-15`. |
| Channel energies, remaining local maps, and physical diagnostics | Basis-independent definitions and representative implementation checks apply. These complete historical arrays and physical fields were not regenerated. |
| Operational/fine-tuned spectra | Both streams and all 208 cases per model were projected at 0, 6, 24, 72, and 120 h. Maximum relative changes in smoothed spectra are `1.89e-8` and `1.73e-8`. Paired energy ratios change by at most `2.69e-14`; 5th/95th percentile bands change by at most `4.82e-14`. Deficits, rounding, and intermodel ordering are preserved. All 21 displayed leads additionally have real-arithmetic relative spectral bounds below 0.052%. |

For the larger mesh, the generic mesh factory's dot-product reduction order changes the stored normalized Laplacian by at most `6.83e-14`. Real-arithmetic bounds preserve both models' reference-relative energy deficits and the fine-tuned model's higher paired ratio at every nonzero displayed lead. Bounds exclude floating-point accumulation error and are reported separately from direct projections.

## Checkpoint extraction

The standalone environment executes the pinned official GraphCast source without model modifications or an xarray shim. On an NVIDIA H200 NVL, the 1-degree checkpoint passes two autoregressive steps: all 11 forecast variables and 34 captured latent arrays match the unchanged forward pass or reference capture exactly, including coordinates, dtypes, Haiku states, and NPY readback.

Operational and fine-tuned 0.25-degree checkpoints each pass one forecast step and an identical-input repeat. Forecasts, coordinates, Haiku states, and repeated saved latents match exactly; each JIT cache remains at one entry. Every saved latent is finite. Independent plain forwards confirm all 86,173,920 forecast values per checkpoint are finite and all coordinates match the target template. Real inference uses batch size one; synthetic tests also exercise native batch size two.

The compiler settings in the README are required for these exact comparisons. Default GPU compilation produced a reference-latent discrepancy up to `0.125`; exact equality is not asserted for that configuration.

## Package verification and limits

Source and independently installed-wheel suites pass 56 tests on Python 3.12.12: the original 40 component tests plus 16 runner checks. Coverage includes graph validation, independent edge-energy formulas, native PyGSP transform identities, invalid signals, cached extraction, safe storage, analysis-import isolation, lazy command help, compiler setup, initialization-only forcing/templates, invalid input contracts, and output collisions. A valid rotation within a repeated-eigenvalue subspace verifies that the comparison tool distinguishes mathematical compatibility from strict mode identity and scientific adoption. An analysis-only wheel environment passes 46 tests and skips 8 optional-dependency cases; its actual console help works without JAX, GraphCast, Haiku, xarray, Matplotlib or Cartopy installed. Tests run outside the source tree against installed code. Wheel and source archives exclude private fixtures, research arrays, environments, examples, and credentials.

Raw weather data, weights, normalization assets, saved latents, and private validation fixtures are not distributed. The generic comparison tool accepts supplied fixtures and exits on mathematical compatibility; claim-specific checks determine adoption. Validation does not repeat every historical rollout, atmospheric diagnostic, notebook, or figure renderer, and does not establish GPU/CPU throughput equivalence or compatibility with other upstream revisions.
