# ThermoFlow App — Optimization Design Spec
**Date:** 2026-03-31  
**Approach:** Option A — In-place refactor of `thermoflow_app.py`  
**Goal:** Production-ready, Jupyter-importable scientific library with Nature/Science-quality figures, robust error handling, multiple-dataset support, and PRI reporting.

---

## 1. Scope & Constraints

- **Single file:** `thermoflow_app.py` remains one importable module. No package restructuring.
- **Backward compatibility:** All existing public APIs (`FlowExperiment`, `GateSet`, `BaseGate`, gate subclasses, `GateTemplate`, `gate_from_dict`, `extract_gated_events`) keep the same signatures. New parameters are keyword-only with safe defaults.
- **No new hard dependencies:** Only packages already present in the `e1_esm` conda environment. HTML report uses stdlib only. PDF export uses `matplotlib.backends.backend_pdf`.
- **Target environment:** Jupyter notebook with `%matplotlib widget` or `%matplotlib inline`.

---

## 2. Plot Style System

### 2.1 rcParams Overhaul

Replace the existing `mpl.rcParams` block with a Nature/Science-compliant set:

| Parameter | Value | Rationale |
|---|---|---|
| `font.family` | `sans-serif` | Nature house style |
| `font.sans-serif` | `['Arial', 'Helvetica', 'DejaVu Sans']` | Editable vectors in PDF |
| `pdf.fonttype` / `ps.fonttype` | `42` | TrueType in vector exports |
| `axes.linewidth` | `0.5` | Nature: thin axis lines |
| `axes.labelsize` | `7` | Nature: 7pt axis labels |
| `axes.titlesize` | `7` | Consistent with labels |
| `xtick.labelsize` / `ytick.labelsize` | `6` | 6pt tick labels |
| `xtick.major.size` / `ytick.major.size` | `3` | 3pt major ticks |
| `xtick.minor.size` / `ytick.minor.size` | `2` | 2pt minor ticks |
| `xtick.major.width` / `ytick.major.width` | `0.5` | Match axis linewidth |
| `xtick.direction` / `ytick.direction` | `in` | Inward ticks (Nature standard) |
| `legend.fontsize` | `6` | |
| `legend.frameon` | `False` | No legend box |
| `figure.dpi` | `150` | Crisp in-notebook display |
| `savefig.dpi` | `300` | Publication-quality on save |

### 2.2 Figure Width Constants

```python
FIG_1COL = 3.46   # 88 mm — Nature single column
FIG_2COL = 7.09   # 180 mm — Nature double column
FIG_1COL_TALL = (3.46, 3.46)
FIG_2COL_WIDE = (7.09, 3.5)
```

### 2.3 Colormap Replacements

| Plot type | Old | New |
|---|---|---|
| 2D density (pcolormesh) | `jet` | `viridis` |
| Contour lines | `jet` | `viridis` |
| Sliced histograms | `viridis` (kept) | `viridis` |
| PRI curves | `tab10` (kept) | muted `tab10` with alpha=0.85 |
| Bar charts | viridis slice | muted qualitative palette |

### 2.4 `_style_ax()` Signature Extension

```python
def _style_ax(self, ax, xlabel=None, ylabel=None, title=None,
              is_2d=False, spine_style='open'):
```

- `spine_style='open'`: remove top + right spines (all 1D plots)
- `spine_style='box'`: keep all four spines (density/2D plots)
- `spine_style='none'`: remove all spines (reserved, unused in this release)

---

## 3. Error Handling & Data Processing

### 3.1 `load_fcs_files` Improvements

- **Pre-flight check:** if `glob.glob(data_path_pattern)` returns empty, raise `FileNotFoundError(f"No FCS files matched: {data_path_pattern}")` immediately.
- **Channel name normalization:** after loading `pnn_labels`, apply `.strip().replace('\x00', '')` to each label to handle cytometer encoding artifacts.
- **Numeric coercion:** after building `df`, call `pd.to_numeric(df[col], errors='coerce')` on all channel columns; track NaN counts and include in load report.
- **Return value:** method now returns a load-report dict `{file: status, unmatched_wells: [...], unmatched_files: [...], nan_summary: {col: count}}` instead of `None`. Callers that ignore the return value are unaffected.

### 3.2 Gate Geometry Validation

Add `validate()` method to each gate class, called inside `evaluate()` before computation:
- `PolygonGate.validate()`: require `len(vertices) >= 3`
- `RectangleGate.validate()`: require `xmin < xmax` and `ymin < ymax`
- `EllipseGate.validate()`: require `width > 0` and `height > 0`
- `ThresholdGate.validate()`: for `between`, require `lo < hi`

Validation failures raise `ValueError` with a descriptive message naming the gate.

### 3.3 `GateSet.evaluate()` Error Context

Wrap AST evaluation failures with: `ValueError(f"Logic expression '{self.logic}' failed at gate '{node.id}': {original_error}")`.

### 3.4 `run_pri_analysis` Guards

- Validate `channel in df.columns` before any computation; raise `KeyError` with available channels listed.
- Per-sample: if `len(times) < 3`, skip fitting, set fit params to `NaN`, add `fit_quality='insufficient_data'` to fits table.
- Post-fit: compute residual RMSE; if `rmse > 0.5 * np.nanmean(values)`, set `fit_quality='poor'`; otherwise `fit_quality='good'`.
- Bootstrap RNG: replace `np.random.choice` with `rng = np.random.default_rng(seed=42)` then `rng.choice(...)` for reproducibility.

### 3.5 `_fit_global_exponential` Warnings

- Define `FitConvergenceWarning(UserWarning)` at module level.
- When Jacobian is rank-deficient (`np.linalg.matrix_rank(J.T @ J) < len(res.x)`), emit `warnings.warn(..., FitConvergenceWarning)` instead of silently filling NaN.
- Add R² calculation: `r2 = 1 - SS_res / SS_tot` per sample; add `r2` column to output DataFrame.

---

## 4. Multiple Datasets

### 4.1 `datasets` Property

```python
@property
def datasets(self) -> Dict[str, Dict]:
```
Returns `{dataset_id: {'n_events': int, 'samples': list, 'n_wells': int}}` derived from `self.populations['raw']` if a `dataset` column exists; returns `{'default': {...}}` if not.

### 4.2 Plot Filtering

`plot_density`, `plot_sliced_histogram`, and `plot_pri` each gain an optional `dataset: str = None` kwarg. When provided, the data is pre-filtered to `df[df['dataset'] == dataset]` before all existing logic runs. Fully backward-compatible.

---

## 5. FlowReport Class

```python
class FlowReport:
    def __init__(self, experiment: FlowExperiment): ...
    def summary(self) -> str: ...           # printed table, also returned
    def pri_summary(self) -> pd.DataFrame: ... # per-sample: A, k, t½±err, R², fit_quality
    def export_html(self, path: str) -> None: ...   # self-contained HTML with embedded PNGs
    def export_pdf(self, path: str) -> None: ...    # multi-page PDF via PdfPages
```

**`summary()`** columns: Population | N Events | % of Raw | Channels  
**`pri_summary()`** columns: Sample | A | k | t½ | t½_err | R² | fit_quality | CI_low | CI_high  
**`export_html()`**: stdlib `string.Template`; embeds figures as base64 PNG; no external deps.  
**`export_pdf()`**: renders `plot_pri_summary_grid` + `plot_pri_bars` + one density plot per population (using the first two numeric channels) to a single multi-page PDF via `PdfPages`.

---

## 6. `plot_pri_summary_grid` Utility

New method on `FlowExperiment`:

```python
def plot_pri_summary_grid(self, which='PRI_norm', save_path=None, **kwargs):
```

Produces a single `FIG_2COL`-width figure:
- **Top row:** individual PRI decay curves (same as `plot_pri` but sized for double column; accepts same `cols` kwarg, default 4)
- **Bottom row:** single bar chart of t½ values with error bars (same as `plot_pri_bars`)
- Uses `gridspec` with `height_ratios=[3, 1]`
- Shared x-axis label "Time (min)" on the bottom of the top row
- Suitable as a single multi-panel figure for a manuscript

---

## 7. Files Changed

| File | Change |
|---|---|
| `thermoflow_app.py` | All changes above — in-place refactor; `import warnings` added to stdlib imports |
| `docs/superpowers/specs/2026-03-31-thermoflow-optimization-design.md` | This file |

No new files created. No notebook changes required.

---

## 8. Testing

After implementation, verify in the `e1_esm` conda environment:
1. `python -c "from thermoflow_app import FlowExperiment, FlowReport, GateTemplate"` — clean import
2. `FlowExperiment()` instantiates without errors
3. Load a test FCS file and confirm channel name normalization
4. Run `plot_density` with `viridis` — confirm no `jet` appears
5. Run `run_pri_analysis` on synthetic data with < 3 time points — confirm graceful skip
6. `FlowReport(exp).export_html('test_report.html')` — confirm file is created

---

## 9. Out of Scope

- Splitting into multiple modules
- Adding new gating algorithms
- FCS file writing (already noted as stub)
- GUI enhancements beyond current ipywidgets
