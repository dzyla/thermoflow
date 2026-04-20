# ThermoFlow

Flow cytometry analysis library for measuring protein thermal stability via the **Prefusion Retention Index (PRI)** — a fluorescence-decay kinetics assay.

![version](https://img.shields.io/badge/version-0.3.0-blue)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
pip install git+https://github.com/dzyla/thermoflow.git
```

Or clone and install locally:

```bash
git clone https://github.com/dzyla/thermoflow.git
cd thermoflow
pip install .
```

> **Conda users:** install into your environment first, then pip-install ThermoFlow on top:
> ```bash
> conda activate your_env
> pip install git+https://github.com/dzyla/thermoflow.git
> ```

For interactive gating in Jupyter notebooks, also install `ipympl`:

```bash
pip install ipympl
```

---

## Quick start

```python
from thermoflow_app import FlowExperiment, FlowReport

exp = FlowExperiment()
exp.load_fcs_files("data/*.fcs")

# Run PRI kinetics analysis
exp.run_pri_analysis("RL1-H", control_sample="Ctrl")

# Plot decay curves with 95% CI
exp.plot_pri(which="PRI_norm", show_ci=True, show_params=["t_half", "r2"])

# Bar chart of half-lives
exp.plot_pri_bars()

# Export report
report = FlowReport(exp)
report.export_html("report.html")
report.export_pdf("report.pdf")
```

---

## Gating

### Interactive gating (Jupyter)

Launch the interactive widget to draw gates on any 2D scatter plot.
Supports Rectangle, Polygon, and Ellipse gate types with undo/redo.

> **No magic command needed.** `run_gating_ui` automatically switches matplotlib to
> the interactive widget backend. Just call it directly — no `%matplotlib widget`
> required in the notebook.

```python
# Open the gating UI on the raw population, save result as 'cells'
exp.run_gating_ui(parent_pop='raw', new_pop_name='cells')
```

Once you confirm a gate in the UI it is added to `exp.gatesets` and the
filtered population is stored in `exp.populations['cells']`.

### Saving and loading gates

Gates are fully serialisable to JSON for reproducible analysis across sessions.

```python
# Save all drawn gatesets to disk
exp.export_gates('gates.json')

# Reload gates in a future session (no need to redraw)
exp.load_gates('gates.json')
print(exp.gatesets)   # {'ManualGate_cells': <GateSet ...>, ...}
```

### Applying a saved gateset

After loading, apply any gateset to create a gated population:

```python
exp.apply_gateset('ManualGate_cells', parent_pop='raw', new_pop_name='cells')
```

The gated events are stored as `exp.populations['cells']` and can be used
in all downstream methods via `pop_name='cells'`.

### Programmatic gates

Gates can also be built in code without the UI:

```python
from thermoflow_app import RectangleGate, PolygonGate, EllipseGate, ThresholdGate, GateSet

# Rectangle gate on FSC/SSC
rect = RectangleGate('live', x='FSC-H', y='SSC-H',
                     xmin=2.0, xmax=8.0, ymin=1.5, ymax=7.5)

# Threshold gate on a fluorescence channel
thresh = ThresholdGate('positive', column='RL1-H', op='>', threshold=4.5)

# Combine with boolean logic
gs = GateSet('my_gate', gates=[rect, thresh], logic='live & positive')
exp.gatesets['my_gate'] = gs
exp.apply_gateset('my_gate', parent_pop='raw', new_pop_name='live_positive')
```

### Visualising gates

Overlay any gateset on a density plot to inspect coverage:

```python
exp.plot_density(
    x_col='FSC-H', y_col='SSC-H',
    pop_name='raw',
    gates_to_overlay=[exp.gatesets['ManualGate_cells']],
    gate_colors=['#FF5733'],
    show_stats=True,
)
```

---

## PRI analysis

### Basic usage

```python
exp.run_pri_analysis('RL1-H', control_sample='Ctrl',
                     pop_name='cells', reference_sample='Fwt')
```

### Custom threshold

By default the positive/negative gate is set from the top `pos_frac` quantile of
the control sample. Pass `threshold_log` to override with a fixed value in log1p
space (the same scale shown on histogram x-axes):

```python
# Inspect where the gate should fall
exp.plot_sliced_histogram('RL1-H', slice_by='sample', filter_col='time', filter_val=0)

# Supply the gate position directly
exp.run_pri_analysis('RL1-H', control_sample='Ctrl', threshold_log=5.5)
```

### Advanced PRI Analysis

```python
# Median MFI metric (matches manuscript definition) + ΔΔG‡ relative to WT
exp.run_pri_analysis(
    channel="APC-A",
    control_sample="untransfected",
    mfi_metric="median",          # log-space median of positive events (alt. to geometric mean)
    wt_sample="WT",               # reference for ΔΔG‡ (kcal/mol)
    temperature_c=55.0,           # assay temperature in °C
)

# pri_fits_norm now contains ddG_kin and ddG_kin_err columns
print(exp.pri_fits_norm[["sample", "t_half", "ddG_kin", "ddG_kin_err"]])
```

```python
# Hyper-stable / flatline variant detection
exp.run_pri_analysis(
    channel="APC-A",
    control_sample="untransfected",
    wt_sample="WT",
    temperature_c=55.0,
    flatline_threshold=0.10,      # <10% signal drop → fit_quality='hyperstable', t_half=inf
)

# Hyper-stable samples appear with t_half=inf and ddG_kin=inf
stable = exp.pri_fits_norm[exp.pri_fits_norm["fit_quality"] == "hyperstable"]
print(stable[["sample", "t_half", "ddG_kin"]])
```

---

## Sample management

### Rename a sample

If a sample name was typed incorrectly, rename it across all populations and PRI
tables in one call:

```python
exp.rename_sample('FWT', 'Fwt')   # fixes a typo everywhere
```

---

## Full workflow example

```python
import re
from thermoflow_app import FlowExperiment, FlowReport

exp = FlowExperiment()
path = 'data/20260107/*.fcs'
exp.load_fcs_files(path, dataset_id='Plate_1')

experiment = re.search(r'(\d{6})', path).group(1)

# Fix any sample name typos before analysis
exp.rename_sample('FWT', 'Fwt')

# Load previously saved gates and apply
exp.load_gates('gates.json')
exp.apply_gateset('ManualGate_cells', 'raw', 'cells')

# Visualise gate on raw data
exp.plot_density(x_col='FSC-H', y_col='SSC-H', pop_name='raw',
                 gates_to_overlay=[exp.gatesets['ManualGate_cells']],
                 show_stats=True, gate_colors=['#FF5733'])

# PRI analysis on gated population
exp.run_pri_analysis('RL1-H', control_sample='Ctrl',
                     pop_name='cells', reference_sample='Fwt')

# Publication figures
exp.plot_pri(which='PRI_norm', cols=3, show_ci=True,
             show_params=['t_half', 'r2'], spine_width=0.8,
             save_path=f'{experiment}_pri_norm.png')

exp.plot_pri_bars(which='t_half', norm_sample='Fwt',
                  save_path=f'{experiment}_pri_half_life.png')

# Export PRI data and full report
exp.export_pri(out_dir='pri_export', file_prefix=experiment)
report = FlowReport(exp)
report.export_html('report.html')
report.export_pdf('report.pdf')
```

---

## Key features

| Feature | Details |
|---|---|
| FCS loading | Multi-file, channel normalisation, NaN audit, load-report dict |
| Interactive gating | ipywidgets UI — Rectangle, Polygon, Ellipse; undo/redo; auto widget backend |
| Programmatic gates | Rectangle, Polygon, Ellipse, Threshold; boolean logic expressions |
| Gate persistence | Save/load gatesets as JSON (`export_gates` / `load_gates`) |
| Gate overlay | Overlay any gateset on density plots with per-gate event statistics |
| PRI analysis | Global exponential fit, per-sample A/k/t½, bootstrap CIs, R², fit quality |
| Custom threshold | `threshold_log` overrides auto control-quantile gate in `run_pri_analysis` |
| Error bars | Asymmetric — lower cap clipped at zero, upper cap unaffected |
| Sample rename | `rename_sample(old, new)` updates all populations and PRI tables atomically |
| Figures | Nature/Science column widths, publication-ready axes, optional 95% CI bands |
| Reports | Auto-generated HTML + PDF with density, histogram, and PRI panels |
| Multiple datasets | `dataset=` kwarg on all plot methods; `FlowExperiment.datasets` property |

## Dependencies

- Python ≥ 3.9
- numpy, pandas, matplotlib, scipy
- ipython, ipywidgets (Jupyter notebook display)
- ipympl (interactive gating — `pip install ipympl`)
- [flowio](https://github.com/whitews/FlowIO) (FCS file parsing)

## Streamlit GUI

A browser-based GUI covering the full workflow is available in `gui/streamlit_app.py`.

```bash
# install extra GUI deps (streamlit + plotly — may already be present)
pip install streamlit plotly

# launch
streamlit run gui/streamlit_app.py
```

**Workflow tabs:**

| Tab | What you can do |
|---|---|
| 📁 Load Data | Upload FCS files + CSV annotation, rename samples |
| 🔲 Gate | Interactive density/histogram viewer; Rectangle, Ellipse, Threshold gates; save/load gates JSON |
| 📊 PRI Analysis | Configure and run PRI fitting; live threshold preview |
| 📈 Visualize | Decay curves, bar charts, summary grid, density plots, histograms |
| 💾 Export | Download PRI CSVs, gates JSON, HTML/PDF report |

The annotation CSV must have at minimum the columns `well`, `sample`, `time`.
A template can be downloaded from the **Export** tab.

## Running tests

```bash
conda run -n your_env python tests/test_thermoflow.py
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## License

MIT
