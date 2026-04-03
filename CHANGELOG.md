# Changelog

All notable changes to ThermoFlow are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

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
