# ThermoFlow

Flow cytometry analysis library for measuring protein thermal stability via the **Prefusion Retention Index (PRI)** — a fluorescence-decay kinetics assay.

## Install

```bash
pip install git+https://github.com/dzyla/ThermoFlow.git
```

Or clone and install locally:

```bash
git clone https://github.com/dzyla/ThermoFlow.git
cd ThermoFlow
pip install .
```

> **Conda users:** install into your environment first, then pip-install ThermoFlow on top:
> ```bash
> conda activate your_env
> pip install git+https://github.com/dzyla/ThermoFlow.git
> ```

## Quick start

```python
from thermoflow_app import FlowExperiment, FlowReport

exp = FlowExperiment()
exp.load_fcs_files("data/*.fcs")

# Gate and select population
exp.active_pop = "live"

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

## Key features

| Feature | Details |
|---|---|
| FCS loading | Multi-file, channel normalisation, NaN audit |
| Gating | Rectangle, Polygon, Ellipse, Threshold; boolean logic; interactive widget |
| PRI analysis | Global exponential fit, per-sample A/k/t½, bootstrap CIs, R², fit quality |
| Figures | Nature/Science column widths, publication-ready axes, optional 95% CI bands |
| Reports | Self-contained HTML (base64 PNG) and multi-page PDF |
| Multiple datasets | `dataset=` kwarg on all plot methods; `FlowExperiment.datasets` property |

## Dependencies

- Python ≥ 3.9
- numpy, pandas, matplotlib, scipy
- ipython, ipywidgets (Jupyter notebook display)
- [flowio](https://github.com/whitews/FlowIO) (FCS file parsing)

## Running tests

```bash
pip install pytest
pytest tests/
```

## License

MIT
