# Changelog

All notable changes to ThermoFlow are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.5.0] - 2026-07-06
### Changed (affects numerical results)
- **Weighted global exponential fit.** `_fit_global_exponential` now weights each
  residual by 1/standard-error (bootstrap SE when available, else 1/|y| relative
  weighting). The shared background `C` is retained and is physically correct (fixed
  autofluorescence/non-specific floor, same absolute value for every sample), but the
  previous **unweighted** fit let the brightest sample dominate the shared-`C` estimate
  and bias the `k`/`t_half` of dimmer samples. In a controlled simulation (two samples,
  identical true `k`, 20× expression gap) this reduced the dim sample's `k` error from
  ~55% to ~30%. **Existing analyses will produce slightly different fit parameters.**
### Added
- **Per-plate normalization** — `run_pri_analysis(..., per_plate=True)`. When a
  `dataset` column is present, each plate is analysed independently: its own control
  sets the threshold, its own `reference_sample` (WT) sets the baseline, and it gets
  its own shared `C` and ΔΔG‡. Removes plate-to-plate batch effects (instrument gain,
  staining day). `pri_table`/`pri_fits_*` gain a `dataset` column; `plot_pri` and
  `plot_pri_bars` render one panel/bar per (sample, plate). Exposed in the Streamlit GUI.
- **Bootstrap for both metrics** — `PRI_norm_ci_low/high`, `PRI_norm_se`, and
  `PRI_abs_se` columns added to `pri_table`. Bootstrap now uses a single seeded RNG
  stream (`random_state`, default 42) instead of re-seeding per well.
- `flatline_z` parameter: a sample is only called `hyperstable` when its decay is both
  small **and** statistically insignificant (slope within `flatline_z` SEs of zero),
  preventing genuine slow decays from being erased.
### Fixed
- Sane upper bound on `k` derived from the time resolution (was a meaningless `1e7`).
- NaN/inf PRI points get zero weight instead of corrupting the fit.
- Warning emitted when a sample lacks the baseline timepoint (self-normalized PRI → NaN).
- `mfi_metric`/`flatline_threshold` validation moved to the top of `run_pri_analysis`.

---

## [0.4.0] - 2026-04-19
### Added
- `median_mfi` helper and `mfi_metric` param in `run_pri_analysis` (`'geometric_mean'` | `'median'`) — matches manuscript PRI definition
- Flatline / hyperstable detection in `_fit_global_exponential`: samples with negligible signal decay get `fit_quality='hyperstable'` and `t_half=inf`
- ΔΔG‡ calculation (`ddG_kin`, `ddG_kin_err` columns in `pri_fits_norm`) via Transition State Theory when `wt_sample` is supplied
- Input validation for `mfi_metric` and `flatline_threshold` parameters
- Streamlit GUI: MFI metric radio, WT reference selector, assay temperature input in PRI Analysis tab
- README: Advanced PRI Analysis code examples

---

## [0.3.0] - 2026-04-03

### Added
- **`FlowExperiment.rename_sample(old_name, new_name)`** — renames a sample across all
  populations, `pri_table`, `pri_fits_abs`, `pri_fits_norm`, and `pri_control_sample` in one call.
- **`run_pri_analysis(..., threshold_log=None)`** — new keyword argument to supply a
  user-defined positive/negative gate boundary in log1p space, bypassing the
  automatic control-quantile computation. Useful when the control distribution is
  atypical or a fixed gate is required for cross-experiment reproducibility.
- **`__version__`** attribute exposed at module level (`thermoflow_app.__version__`).

### Fixed
- **Interactive gating in Jupyter** (`run_gating_ui`): the method now automatically
  switches matplotlib to the `ipympl` widget backend via
  `get_ipython().run_line_magic('matplotlib', 'widget')`, so interactive selectors
  (Polygon, Rectangle, Ellipse, Span) work without placing `%matplotlib widget` in
  the notebook cell.
- **Error bars below zero in bar plots** (`plot_pri_bars`, `plot_pri_summary_grid`):
  lower error caps are clipped to the bar height so bars never shift below the
  baseline. Upper caps remain symmetric. Implemented via matplotlib's asymmetric
  `yerr = [lower, upper]` format.

---

## [0.2.0] - 2026-03-31

### Added
- Global exponential fit with per-sample A / k / t½ and bootstrap 95 % CIs.
- `fit_quality` flag (`good` / `poor` / `insufficient_data`) per sample.
- `FlowReport` with `export_html` and `export_pdf` (stdlib + matplotlib PDF backend only).
- `plot_pri_summary_grid` — combined decay-curve + t½ bar figure at Nature double-column width.
- `plot_density` gate overlay with per-gate event statistics.
- `export_gates` / `load_gates` JSON persistence for reproducible gating.
- Multiple-dataset support via `dataset_id` kwarg on `load_fcs_files`.
- `QuadrantGate` and boolean logic expressions in `GateSet`.
- `reference_sample` normalisation in `run_pri_analysis`.

---

## [0.1.0] - initial release

- `FlowExperiment` with FCS loading, interactive gating widget, PRI analysis, and
  publication-quality figures.
