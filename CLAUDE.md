# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python commands must go through the conda environment — bare `python`/`pip` will fail with `ModuleNotFoundError`.

```bash
# Install in editable/dev mode
conda run -n e1_esm pip install -e ".[dev]"

# Run the test suite (plain script — prints PASS/FAIL, exits 0/1)
conda run -n e1_esm python tests/test_thermoflow.py
```

## Versioning

When bumping the version, update **three places together**:
1. `__version__` at the top of `thermoflow_app.py`
2. `version` in `pyproject.toml`
3. A new entry in `CHANGELOG.md`

Follow semver: patch for bug fixes, minor for new features/parameters, major for breaking API changes.

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

## Streamlit GUI

Located in `gui/streamlit_app.py`. Run with:
```bash
conda run -n e1_esm streamlit run gui/streamlit_app.py
```

Key design decisions:
- `FlowExperiment` is stored in `st.session_state["exp"]` and survives reruns.
- `_patch_exp()` monkey-patches `_show_static_fig` so matplotlib figures are captured as PNG bytes into `st.session_state["last_fig"]` instead of being sent to IPython display. Call `_render(exp.plot_*)` to invoke any plot method and display the result.
- Interactive gating uses a Plotly density heatmap (via `points_to_density_image`) + numeric inputs for gate bounds — no `ipympl` needed.
- FCS files are written to a `tempfile.mkdtemp` directory stored in `st.session_state["tmpdir"]`; the glob pattern `tmpdir/*.fcs` is then passed to `load_fcs_files`.

## Gotchas

- **`jet` colormap is banned** — the test suite asserts zero occurrences of `'jet'`/`"jet"` in source. Use `'viridis'` or `_PALETTE`.
- **Asymmetric error bars** — to clip lower caps at zero without touching upper caps, pass `yerr=np.array([lower_arr, upper_arr])` to matplotlib bar/errorbar.
- **Interactive backend in `.py` files** — `%matplotlib widget` is a notebook magic and cannot go in a `.py` file. The programmatic equivalent is `get_ipython().run_line_magic('matplotlib', 'widget')`; requires `ipympl` installed in the environment.
- **Global C in PRI fit** — `_fit_global_exponential` fits a single shared baseline `C` across all samples simultaneously via `least_squares`. Per-sample free params are `A` and `k` only. Changing this would break the global-fit design.
- **Gate coordinates are in log1p space** — `threshold_log` and all gate boundaries stored in `RectangleGate`/`PolygonGate`/etc. are in `log1p(raw_value)` space, matching what histogram/density plots show on their axes.
