# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable/dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run tests directly (used in the test file comments)
python tests/test_thermoflow.py

# Run a single test section (the test file is a script, not pytest-style)
# Tests print PASS/FAIL and exit with code 0/1
```

## Architecture

This is a **single-file scientific library** — all code lives in `thermoflow_app.py`. There is no package structure; the module is installed as a top-level module via `pyproject.toml`'s `py-modules`.

### Core classes (in order of dependency)

1. **Gate classes** (`BaseGate`, `RectangleGate`, `PolygonGate`, `EllipseGate`, `ThresholdGate`) — dataclasses with `evaluate(df)` returning a boolean Series. All gates operate on `log1p`-transformed data by default. Each has a `validate()` method that raises `ValueError` on bad parameters.

2. **`GateSet`** — combines multiple gate instances with a boolean logic string (e.g. `'live & positive'`). Serialises to/from JSON via `to_dict()` / `gate_from_dict()`.

3. **`FlowExperiment`** — the central analysis object:
   - `populations: Dict[str, pd.DataFrame]` — named event tables (always starts with `'raw'`)
   - `gatesets: Dict[str, GateSet]` — named gate collections
   - `pri_table`, `pri_fits_norm`, `pri_fits_abs` — PRI analysis results (DataFrames)
   - Key methods: `load_fcs_files()`, `run_gating_ui()`, `apply_gateset()`, `export_gates()` / `load_gates()`, `run_pri_analysis()`, `plot_pri()`, `plot_pri_bars()`, `plot_density()`

4. **`FlowReport`** — wraps a `FlowExperiment` and generates HTML/PDF reports via `export_html()` / `export_pdf()` (stdlib + matplotlib PDF backend only, no extra deps).

### PRI (Prefusion Retention Index)

`run_pri_analysis(channel, control_sample, pop_name='raw')` fits a global exponential decay model to median fluorescence vs. time per sample. Produces:
- `pri_table` — per-well medians
- `pri_fits_norm` / `pri_fits_abs` — per-sample fit params (`A`, `k`, `t_half`, `r2`, `fit_quality`, bootstrap CIs)
- `fit_quality` is `'good'`, `'poor'`, or `'insufficient_data'` (< 3 time points triggers a `UserWarning`)

### Plotting globals

Set at module load via `mpl.rcParams`. Figure widths follow Nature column widths: `FIG_1COL = 3.46"` (88 mm), `FIG_2COL = 7.09"` (180 mm). The custom palette `_PALETTE` cycles 10 NPG-inspired colours. Never use `'jet'` colormap — the test suite checks for it.

### Data conventions

- Event DataFrames must have at minimum: `sample`, `time`, channel column(s), `well`
- `dataset` column is optional; used by the `datasets` property to report per-dataset stats
- Gates store coordinates in `log1p` space when `log1p=True` (default); raw FCS values are transformed before gating

### Test file

`tests/test_thermoflow.py` is a plain script (not pytest classes). It uses a global `check(name, condition)` helper and prints a summary. Tests use synthetic data built by `make_experiment()` — no real FCS files required. Matplotlib backend is forced to `'Agg'` at the top.
