# ThermoFlow Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `thermoflow_app.py` in-place to produce Nature/Science-quality figures, robust error handling, multiple-dataset support, R²/fit-quality metrics, and a `FlowReport` class — while keeping all existing public APIs backward-compatible.

**Architecture:** Single-file in-place refactor. No new files except the test script. All changes are additive (new kwargs default to `None`/safe values) or drop-in replacements (rcParams, colormap strings). New `FlowReport` class appended at the bottom of the file.

**Tech Stack:** Python 3.10+, numpy, pandas, matplotlib, scipy, ipywidgets, flowio — all present in `e1_esm` conda environment. Test runner: plain `python` (no pytest needed since this is a scientific notebook library; tests are a runnable script).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `thermoflow_app.py` | Modify | All production changes |
| `tests/test_thermoflow.py` | Create | Smoke + unit tests run via `python tests/test_thermoflow.py` |

---

## Task 1: rcParams & Figure Constants

**Files:**
- Modify: `thermoflow_app.py:37-52` (the existing rcParams block)

- [ ] **Step 1: Replace the rcParams block**

Find the block starting at line 37 (`mpl.rcParams['font.family']`) and ending at `mpl.rcParams['legend.frameon'] = False`. Replace it entirely with:

```python
import warnings

# ==============================================================================
# 0. NATURE/SCIENCE PUBLICATION-QUALITY PLOTTING GLOBALS
# ==============================================================================
# Figure width constants (inches) — Nature single/double column
FIG_1COL = 3.46   # 88 mm
FIG_2COL = 7.09   # 180 mm

mpl.rcParams.update({
    'font.family':           'sans-serif',
    'font.sans-serif':       ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':          42,   # TrueType — editable in Illustrator
    'ps.fonttype':           42,
    'axes.linewidth':        0.5,
    'axes.labelsize':        7,
    'axes.titlesize':        7,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'xtick.labelsize':       6,
    'ytick.labelsize':       6,
    'xtick.major.size':      3,
    'ytick.major.size':      3,
    'xtick.minor.size':      2,
    'ytick.minor.size':      2,
    'xtick.major.width':     0.5,
    'ytick.major.width':     0.5,
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    'legend.fontsize':       6,
    'legend.frameon':        False,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'lines.linewidth':       1.0,
    'patch.linewidth':       0.5,
})

# Custom warning for fit convergence issues
class FitConvergenceWarning(UserWarning):
    pass

Number = Union[int, float]
```

Note: `import warnings` moves to the stdlib imports block at the top of the file (line ~8, with the other stdlib imports). Remove the `Number = Union[int, float]` line from its old location (after the old rcParams block).

- [ ] **Step 2: Add `import warnings` to stdlib imports**

In the stdlib import block (lines 3-13), add `import warnings` after `import re`.

- [ ] **Step 3: Verify clean import**

```bash
cd /home/dzyla/thermoflow/ThermoFlow/app
conda run -n e1_esm python -c "import thermoflow_app; print('FIG_1COL =', thermoflow_app.FIG_1COL)"
```
Expected output: `FIG_1COL = 3.46`

---

## Task 2: `_style_ax` Overhaul

**Files:**
- Modify: `thermoflow_app.py` — `_style_ax` method in `FlowExperiment` class

- [ ] **Step 1: Replace `_style_ax` method**

Find the existing `_style_ax` method (around line 899) and replace it:

```python
def _style_ax(self, ax, xlabel=None, ylabel=None, title=None,
              is_2d=False, spine_style='open'):
    """Apply Nature/Science spine and label style to an axis.
    
    spine_style: 'open' = remove top+right (1D plots)
                 'box'  = keep all four (2D density plots)
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=4)

    if spine_style == 'open':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    elif spine_style == 'box':
        for s in ['top', 'right', 'bottom', 'left']:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=3)
```

- [ ] **Step 2: Update all `_style_ax` callers that pass `is_2d=True`**

Search for all `self._style_ax(` calls that include `is_2d=True` and replace `is_2d=True` with `spine_style='box'`. Replace `is_2d=False` with `spine_style='open'` (or just remove it since `'open'` is the default).

```bash
grep -n "is_2d=" /home/dzyla/thermoflow/ThermoFlow/app/thermoflow_app.py
```

For each occurrence: replace `is_2d=True` → `spine_style='box'` and `is_2d=False` → remove the kwarg entirely.

- [ ] **Step 3: Verify import still clean**

```bash
conda run -n e1_esm python -c "from thermoflow_app import FlowExperiment; e = FlowExperiment(); print('_style_ax OK')"
```
Expected: `_style_ax OK`

---

## Task 3: Colormap & Plot Style Replacements

**Files:**
- Modify: `thermoflow_app.py` — all plot methods

- [ ] **Step 1: Replace all `cmap='jet'` defaults and literals**

```bash
grep -n "jet" /home/dzyla/thermoflow/ThermoFlow/app/thermoflow_app.py
```

For each occurrence:
- Default parameter `cmap: str = 'jet'` → `cmap: str = 'viridis'`
- Hardcoded `cmap='jet'` inside `plot_population_comparison` → `cmap='viridis'`
- Contour `cmap='jet'` in `run_gating_ui` → `cmap='viridis'`

- [ ] **Step 2: Update `plot_histogram` color**

Find `plot_histogram` method. Replace:
```python
ax.hist(np.log1p(df[col].clip(lower=0)), bins=100, color='dodgerblue', alpha=0.8, density=True)
```
With:
```python
ax.hist(np.log1p(df[col].clip(lower=0)), bins=100, color='#2166ac', alpha=0.85,
        density=True, linewidth=0)
```

- [ ] **Step 3: Update `plot_sliced_histogram` colormap**

Find the line:
```python
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(slice_vals)))
```
Replace with a more distinguishable qualitative palette when ≤10 slices, falling back to viridis:
```python
if len(slice_vals) <= 10:
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(slice_vals)))
else:
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(slice_vals)))
```

- [ ] **Step 4: Update `plot_pri_bars` colormap call**

Find:
```python
try:
    cmap = cm.get_cmap(color_palette)
except:
    cmap = cm.viridis
```
Replace with (avoids deprecation warning in matplotlib ≥3.7):
```python
try:
    cmap = mpl.colormaps.get_cmap(color_palette)
except (ValueError, KeyError):
    cmap = mpl.colormaps['viridis']
```

- [ ] **Step 5: Update `plot_pri` data marker style**

Find the scatter marker line in `plot_pri`:
```python
ax.plot(g["time"].values, g[which].values, "o", color="#1f77b4", markeredgecolor='white', markeredgewidth=0.5, markersize=7, alpha=0.8, label="Data")
```
Replace with:
```python
ax.plot(g["time"].values, g[which].values, "o", color="#2166ac",
        markeredgecolor='white', markeredgewidth=0.3,
        markersize=4, alpha=0.9, label="Data")
```

And the fit line:
```python
ax.plot(tgrid, A * np.exp(-k * tgrid) + C, "--", color="#333333", linewidth=1.5, zorder=0, label="Fit")
```
Replace with:
```python
ax.plot(tgrid, A * np.exp(-k * tgrid) + C, "-", color="#d6604d",
        linewidth=1.0, zorder=0, label="Fit")
```

- [ ] **Step 6: Verify no remaining `jet` references**

```bash
grep -n "jet" /home/dzyla/thermoflow/ThermoFlow/app/thermoflow_app.py
```
Expected: zero matches.

---

## Task 4: Gate Validation Methods

**Files:**
- Modify: `thermoflow_app.py` — gate dataclasses (lines ~58-178)

- [ ] **Step 1: Add `validate()` to `ThresholdGate`**

Inside the `ThresholdGate` class, add before `evaluate()`:
```python
def validate(self):
    if self.op == 'between' and self.lo is not None and self.hi is not None:
        if self.lo >= self.hi:
            raise ValueError(f"ThresholdGate '{self.name}': lo ({self.lo}) must be < hi ({self.hi})")
```

- [ ] **Step 2: Add `validate()` to `RectangleGate`**

Inside `RectangleGate`, add before `evaluate()`:
```python
def validate(self):
    if self.xmin >= self.xmax:
        raise ValueError(f"RectangleGate '{self.name}': xmin ({self.xmin}) must be < xmax ({self.xmax})")
    if self.ymin >= self.ymax:
        raise ValueError(f"RectangleGate '{self.name}': ymin ({self.ymin}) must be < ymax ({self.ymax})")
```

- [ ] **Step 3: Add `validate()` to `PolygonGate`**

Inside `PolygonGate`, add before `evaluate()`:
```python
def validate(self):
    if len(self.vertices) < 3:
        raise ValueError(f"PolygonGate '{self.name}': requires at least 3 vertices, got {len(self.vertices)}")
```

- [ ] **Step 4: Add `validate()` to `EllipseGate`**

Inside `EllipseGate`, add before `evaluate()`:
```python
def validate(self):
    if self.width <= 0:
        raise ValueError(f"EllipseGate '{self.name}': width must be > 0, got {self.width}")
    if self.height <= 0:
        raise ValueError(f"EllipseGate '{self.name}': height must be > 0, got {self.height}")
```

- [ ] **Step 5: Call `validate()` at top of each `evaluate()` method**

In each gate class's `evaluate()` method, add as the first line (after the docstring if any):
```python
self.validate()
```

For `PolygonGate.evaluate()`, the existing early-return for `len(vertices) < 3` should be replaced by the validate call (which now raises instead of silently returning empty).

- [ ] **Step 6: Improve `GateSet.evaluate()` error message**

Find the except clause inside `GateSet.evaluate()`:
```python
        except Exception as e:
            raise ValueError(f"Invalid logic expression '{self.logic}': {e}")
```
Replace with:
```python
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"GateSet '{self.name}': logic expression '{self.logic}' failed — {e}. "
                f"Available gate IDs: {list(masks.keys())}"
            ) from e
```

- [ ] **Step 7: Verify import clean**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import RectangleGate
import pandas as pd, numpy as np
try:
    g = RectangleGate('bad', x='A', y='B', xmin=5, xmax=1)
    g.validate()
    print('FAIL: should have raised')
except ValueError as e:
    print('PASS:', e)
"
```
Expected: `PASS: RectangleGate 'bad': xmin (5) must be < xmax (1)`

---

## Task 5: `load_fcs_files` Hardening

**Files:**
- Modify: `thermoflow_app.py` — `load_fcs_files` method

- [ ] **Step 1: Add pre-flight check and channel normalization**

Find `load_fcs_files`. Replace the section:
```python
        data_files = sorted(glob.glob(data_path_pattern))
        if not data_files: return
```
With:
```python
        data_files = sorted(glob.glob(data_path_pattern))
        if not data_files:
            raise FileNotFoundError(
                f"No FCS files found matching pattern: '{data_path_pattern}'"
            )
```

- [ ] **Step 2: Add channel name normalization after reading pnn_labels**

Find inside the `try` block:
```python
                df = pd.DataFrame(events, columns=fcs_file.pnn_labels)
```
Replace with:
```python
                clean_labels = [
                    lbl.strip().replace('\x00', '') 
                    for lbl in fcs_file.pnn_labels
                ]
                df = pd.DataFrame(events, columns=clean_labels)
                # Coerce all channel columns to numeric, tracking NaNs
                nan_counts = {}
                for col in clean_labels:
                    original = df[col].copy()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    n_nan = df[col].isna().sum()
                    if n_nan > 0:
                        nan_counts[col] = int(n_nan)
```

- [ ] **Step 3: Build and return a load report**

At the start of `load_fcs_files`, add:
```python
        load_report = {
            'matched_files': [],
            'unmatched_wells': [],
            'unmatched_files': [],
            'errors': {},
            'nan_summary': {},
        }
```

Update the per-file error handler:
```python
            except Exception as e:
                load_report['errors'][matching_files_for_well[0]] = str(e)
                print(f"  ❌ Error processing {os.path.basename(matching_files_for_well[0])}: {e}")
```

After `unmatched_wells.append(well_id)` and the file-matching logic, populate:
```python
        load_report['unmatched_wells'] = unmatched_wells
        load_report['unmatched_files'] = unmatched_files
        load_report['matched_files'] = list(matched_files)
```

Change the final `return` (currently implicit `None`) to `return load_report`.

- [ ] **Step 4: Verify the return value is a dict**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import FlowExperiment
e = FlowExperiment()
try:
    e.load_fcs_files('/no/such/path/*.fcs')
except FileNotFoundError as ex:
    print('PASS pre-flight:', ex)
"
```
Expected: `PASS pre-flight: No FCS files found matching pattern: '/no/such/path/*.fcs'`

---

## Task 6: `run_pri_analysis` & `_fit_global_exponential` Hardening

**Files:**
- Modify: `thermoflow_app.py` — `run_pri_analysis` and `_fit_global_exponential`

- [ ] **Step 1: Add channel validation at start of `run_pri_analysis`**

After `df = self.get_data(pop_name)`, add:
```python
        if df.empty:
            raise ValueError(f"Population '{pop_name or self.active_pop}' is empty or does not exist.")
        if channel not in df.columns:
            raise KeyError(
                f"Channel '{channel}' not found. Available columns: {list(df.columns)}"
            )
```

- [ ] **Step 2: Skip fitting for samples with < 3 time points**

In `_fit_global_exponential`, after building `times_list` and `values_list`, add a pre-filter:
```python
        skipped = []
        valid_samples, valid_times, valid_values = [], [], []
        for s, t, v in zip(samples, times_list, values_list):
            if len(t) < 3:
                skipped.append(s)
            else:
                valid_samples.append(s)
                valid_times.append(t)
                valid_values.append(v)
        
        if skipped:
            warnings.warn(
                f"Skipping fit for samples with < 3 time points: {skipped}",
                UserWarning, stacklevel=3
            )
        
        # Build NaN rows for skipped samples
        skipped_rows = [
            dict(sample=s, A=np.nan, A_err=np.nan, k=np.nan, k_err=np.nan,
                 C=np.nan, C_err=np.nan, t_half=np.nan, t_half_err=np.nan,
                 r2=np.nan, fit_quality='insufficient_data')
            for s in skipped
        ]
        
        # Replace working variables with filtered lists
        samples, times_list, values_list = valid_samples, valid_times, valid_values
        
        if not samples:
            return pd.DataFrame(skipped_rows)
```

- [ ] **Step 3: Add R² calculation per sample in `_fit_global_exponential`**

Inside the `for i, s in enumerate(samples):` loop, after computing `t_half_err`, add:
```python
            # R² for this sample
            y_obs = values_list[i]
            y_pred_s = A * np.exp(-k * times_list[i]) + global_C
            valid = np.isfinite(y_obs) & np.isfinite(y_pred_s)
            if valid.sum() > 1:
                ss_res = np.sum((y_obs[valid] - y_pred_s[valid]) ** 2)
                ss_tot = np.sum((y_obs[valid] - np.mean(y_obs[valid])) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            else:
                r2 = np.nan
```

Add `r2=r2` to the `out.append(dict(...))` call.

- [ ] **Step 4: Add `fit_quality` flag based on RMSE**

In the same loop, add after computing `r2`:
```python
            y_obs_valid = y_obs[valid] if valid.sum() > 0 else np.array([])
            if len(y_obs_valid) > 0:
                rmse = np.sqrt(np.mean((y_obs[valid] - y_pred_s[valid]) ** 2))
                median_val = np.nanmedian(np.abs(y_obs_valid))
                fit_quality = 'poor' if (median_val > 1e-12 and rmse > 0.5 * median_val) else 'good'
            else:
                fit_quality = 'unknown'
```

Add `fit_quality=fit_quality` to the `out.append(dict(...))` call.

- [ ] **Step 5: Emit `FitConvergenceWarning` for rank-deficient Jacobian**

Find the except block in `_fit_global_exponential`:
```python
        except Exception:
            param_errors = np.full_like(res.x, np.nan)
```
Replace with:
```python
        except Exception as cov_err:
            param_errors = np.full_like(res.x, np.nan)
            warnings.warn(
                f"Jacobian is rank-deficient or singular; parameter errors set to NaN. "
                f"Detail: {cov_err}",
                FitConvergenceWarning, stacklevel=3
            )
```

Also add a rank check before the `try`:
```python
        J = res.jac
        if np.linalg.matrix_rank(J.T @ J) < len(res.x):
            warnings.warn(
                "Jacobian rank deficiency detected; covariance estimation may be unreliable.",
                FitConvergenceWarning, stacklevel=3
            )
```

- [ ] **Step 6: Fix bootstrap RNG for reproducibility**

Find the bootstrap loop in `run_pri_analysis`:
```python
                    for _ in range(n_bootstrap):
                        boot_idx = np.random.choice(len(vals), len(vals), replace=True)
```
Replace with (add rng instantiation before the loop):
```python
                    rng = np.random.default_rng(seed=42)
                    for _ in range(n_bootstrap):
                        boot_idx = rng.choice(len(vals), len(vals), replace=True)
```

- [ ] **Step 7: Append skipped_rows to final DataFrame**

At the end of `_fit_global_exponential`, replace:
```python
        return pd.DataFrame(out).sort_values("sample").reset_index(drop=True)
```
With:
```python
        all_rows = out + skipped_rows
        return pd.DataFrame(all_rows).sort_values("sample").reset_index(drop=True)
```

- [ ] **Step 8: Verify channel validation raises correctly**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import FlowExperiment
import pandas as pd
e = FlowExperiment()
e.populations['raw'] = pd.DataFrame({'sample':['A'], 'time':[0], 'FSC-A':[100.0]})
e.active_pop = 'raw'
try:
    e.run_pri_analysis('NONEXISTENT', control_sample='A')
except KeyError as ex:
    print('PASS:', ex)
"
```
Expected: `PASS: "Channel 'NONEXISTENT' not found. Available columns: [...]"`

---

## Task 7: Multiple Datasets — `datasets` Property & Plot Filter

**Files:**
- Modify: `thermoflow_app.py` — `FlowExperiment` class

- [ ] **Step 1: Add `datasets` property**

After the `channels` property (around line 492), add:

```python
@property
def datasets(self) -> Dict[str, Dict]:
    """Summary of loaded datasets. Returns {dataset_id: {n_events, samples, n_wells}}."""
    raw = self.populations.get('raw', pd.DataFrame())
    if raw.empty:
        return {}
    if 'dataset' not in raw.columns:
        return {'default': {
            'n_events': len(raw),
            'samples': sorted(raw['sample'].dropna().unique().tolist()) if 'sample' in raw.columns else [],
            'n_wells': raw['well'].nunique() if 'well' in raw.columns else 0,
        }}
    result = {}
    for ds_id, grp in raw.groupby('dataset'):
        result[str(ds_id)] = {
            'n_events': len(grp),
            'samples': sorted(grp['sample'].dropna().unique().tolist()) if 'sample' in grp.columns else [],
            'n_wells': grp['well'].nunique() if 'well' in grp.columns else 0,
        }
    return result
```

- [ ] **Step 2: Add `dataset` filter kwarg to `plot_density`**

In `plot_density` signature, add `dataset: str = None` after `pop_name: str = None`.

At the start of `plot_density`, after `df = self.get_data(pop_name)`, add:
```python
        if dataset is not None and 'dataset' in df.columns:
            df = df[df['dataset'] == dataset]
            if df.empty:
                print(f"⚠️ No data for dataset='{dataset}'")
                return
```

- [ ] **Step 3: Add `dataset` filter kwarg to `plot_sliced_histogram`**

In `plot_sliced_histogram` signature, add `dataset: str = None` after `pop_name: str = None`.

After `df = self.get_data(pop_name)`, add:
```python
        if dataset is not None and 'dataset' in df.columns:
            df = df[df['dataset'] == dataset]
            if df.empty:
                print(f"⚠️ No data for dataset='{dataset}'")
                return
```

- [ ] **Step 4: Add `dataset` filter kwarg to `plot_pri`**

In `plot_pri` signature, add `dataset: str = None` after `save_path: str = None`.

After `data = self.pri_table.copy()`, add:
```python
        if dataset is not None and 'dataset' in data.columns:
            data = data[data['dataset'] == dataset]
```

- [ ] **Step 5: Verify `datasets` property on a synthetic experiment**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import FlowExperiment
import pandas as pd
e = FlowExperiment()
e.populations['raw'] = pd.DataFrame({
    'sample': ['A','A','B'], 'time': [0,1,0],
    'FSC-A': [1.0,2.0,3.0], 'well': ['A1','A1','B2'],
    'dataset': ['exp1','exp1','exp2']
})
print(e.datasets)
"
```
Expected: `{'exp1': {'n_events': 2, 'samples': ['A'], 'n_wells': 1}, 'exp2': {'n_events': 1, 'samples': ['B'], 'n_wells': 1}}`

---

## Task 8: `FlowReport` Class

**Files:**
- Modify: `thermoflow_app.py` — append new class after the last line

- [ ] **Step 1: Append `FlowReport` class**

Add the following at the very end of `thermoflow_app.py`:

```python
# ==============================================================================
# 5. FLOW REPORT
# ==============================================================================
import base64
import io as _io
from string import Template as _Template
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages

class FlowReport:
    """Generate summary reports from a FlowExperiment."""

    def __init__(self, experiment: 'FlowExperiment'):
        self.exp = experiment

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Print and return a formatted text summary of all populations."""
        lines = [
            "=" * 60,
            f"  FlowExperiment Summary",
            "=" * 60,
        ]
        raw = self.exp.populations.get('raw', pd.DataFrame())
        n_raw = len(raw)
        lines.append(f"  Raw events : {n_raw:,}")
        lines.append(f"  Channels   : {', '.join(self.exp.channels)}")
        ds = self.exp.datasets
        if len(ds) > 1:
            lines.append(f"  Datasets   : {', '.join(ds.keys())}")
        lines.append("")
        lines.append(f"  {'Population':<20} {'N Events':>10} {'% of Raw':>10}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*10}")
        for name, df in self.exp.populations.items():
            pct = f"{100 * len(df) / n_raw:.1f}" if n_raw > 0 else "—"
            active = " ◀" if name == self.exp.active_pop else ""
            lines.append(f"  {name:<20} {len(df):>10,} {pct:>10}{active}")
        lines.append("=" * 60)
        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    def pri_summary(self) -> pd.DataFrame:
        """Return per-sample PRI fit summary as a DataFrame and print it."""
        if self.exp.pri_fits_norm.empty and self.exp.pri_fits_abs.empty:
            print("⚠️ No PRI fits available. Run run_pri_analysis first.")
            return pd.DataFrame()

        fits = self.exp.pri_fits_norm if not self.exp.pri_fits_norm.empty else self.exp.pri_fits_abs

        cols = ['sample', 'A', 'A_err', 'k', 'k_err', 't_half', 't_half_err']
        if 'r2' in fits.columns:
            cols.append('r2')
        if 'fit_quality' in fits.columns:
            cols.append('fit_quality')

        # Bootstrap CI from pri_table if available
        display_df = fits[[c for c in cols if c in fits.columns]].copy()

        if not self.exp.pri_table.empty and 'PRI_abs_ci_low' in self.exp.pri_table.columns:
            ci = (self.exp.pri_table
                  .groupby('sample')[['PRI_abs_ci_low', 'PRI_abs_ci_high']]
                  .mean()
                  .reset_index())
            display_df = display_df.merge(ci, on='sample', how='left')

        print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return display_df

    # ------------------------------------------------------------------
    def _fig_to_b64(self, fig) -> str:
        """Render a matplotlib figure to a base64-encoded PNG string."""
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    # ------------------------------------------------------------------
    def export_html(self, path: str) -> None:
        """Write a self-contained HTML report with embedded figures and tables."""
        figs_b64 = []

        # 1. Density plot for each population (first two numeric channels)
        chans = self.exp.channels
        if len(chans) >= 2:
            for pop_name, df in self.exp.populations.items():
                if df.empty or len(df) < 10:
                    continue
                fig, ax = plt.subplots(figsize=(FIG_1COL, FIG_1COL))
                x = np.log1p(df[chans[0]].clip(lower=0))
                y = np.log1p(df[chans[1]].clip(lower=0))
                density, xe, ye = points_to_density_image(x, y, bins=128)
                density_plot = np.where(density == 0, np.nan, density)
                ax.pcolormesh(xe, ye, density_plot, norm=LogNorm(clip=True),
                              cmap='viridis', shading='auto')
                self.exp._style_ax(ax,
                    xlabel=f"{chans[0]} (log)",
                    ylabel=f"{chans[1]} (log)",
                    title=f"{pop_name} (n={len(df):,})",
                    spine_style='box')
                fig.tight_layout()
                figs_b64.append(('density', pop_name, self._fig_to_b64(fig)))

        # 2. PRI summary grid if available
        if not self.exp.pri_table.empty:
            fig_pri, _ = plt.subplots(figsize=(FIG_2COL, 3))
            plt.close(fig_pri)
            # Use the experiment's own plot method, capture output
            import io as _sysio
            old_show = plt.show
            captured_fig = [None]
            def _capture_show():
                captured_fig[0] = plt.gcf()
            plt.show = _capture_show
            try:
                self.exp.plot_pri(which='PRI_norm', cols=4)
            finally:
                plt.show = old_show
            if captured_fig[0] is not None:
                buf = _io.BytesIO()
                captured_fig[0].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(captured_fig[0])
                buf.seek(0)
                figs_b64.append(('pri', 'PRI Curves', base64.b64encode(buf.getvalue()).decode('ascii')))

        # Build HTML
        summary_text = self.summary().replace('\n', '<br>')
        pri_df = self.pri_summary()

        img_tags = ''.join(
            f'<div class="fig-block"><h3>{label}</h3>'
            f'<img src="data:image/png;base64,{b64}" alt="{label}"/></div>'
            for _, label, b64 in figs_b64
        )

        table_html = pri_df.to_html(index=False, float_format=lambda x: f"{x:.4f}",
                                     border=0, classes='pri-table') if not pri_df.empty else ''

        html_template = _Template('''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>FlowReport</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; font-size: 11px; margin: 24px; color: #222; }
  h1 { font-size: 14px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h2 { font-size: 12px; color: #444; margin-top: 20px; }
  h3 { font-size: 11px; color: #555; }
  pre { background: #f5f5f5; padding: 10px; font-size: 10px; border-radius: 4px; }
  .fig-block { display: inline-block; margin: 8px; vertical-align: top; }
  .fig-block img { max-width: 400px; border: 1px solid #ddd; }
  .pri-table { border-collapse: collapse; font-size: 10px; }
  .pri-table th, .pri-table td { border: 1px solid #ddd; padding: 4px 8px; }
  .pri-table th { background: #f0f0f0; }
</style>
</head>
<body>
<h1>FlowReport</h1>
<h2>Experiment Summary</h2>
<pre>$summary</pre>
<h2>Figures</h2>
$figures
<h2>PRI Fit Summary</h2>
$table
</body>
</html>''')

        html = html_template.substitute(
            summary=summary_text,
            figures=img_tags,
            table=table_html,
        )

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 HTML report saved to: {path}")

    # ------------------------------------------------------------------
    def export_pdf(self, path: str) -> None:
        """Write a multi-page PDF report using matplotlib PdfPages."""
        chans = self.exp.channels

        with _PdfPages(path) as pdf:
            # Page 1+: density plot per population
            if len(chans) >= 2:
                for pop_name, df in self.exp.populations.items():
                    if df.empty or len(df) < 10:
                        continue
                    fig, ax = plt.subplots(figsize=(FIG_1COL, FIG_1COL))
                    x = np.log1p(df[chans[0]].clip(lower=0))
                    y = np.log1p(df[chans[1]].clip(lower=0))
                    density, xe, ye = points_to_density_image(x, y, bins=128)
                    density_plot = np.where(density == 0, np.nan, density)
                    ax.pcolormesh(xe, ye, density_plot, norm=LogNorm(clip=True),
                                  cmap='viridis', shading='auto')
                    self.exp._style_ax(ax,
                        xlabel=f"{chans[0]} (log)",
                        ylabel=f"{chans[1]} (log)",
                        title=f"{pop_name} (n={len(df):,})",
                        spine_style='box')
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            # PRI page
            if not self.exp.pri_table.empty:
                self.exp.plot_pri(which='PRI_norm', cols=4, save_path=None)
                pdf.savefig(plt.gcf(), bbox_inches='tight')
                plt.close()

                self.exp.plot_pri_bars(which='t_half', use_norm=True, save_path=None)
                pdf.savefig(plt.gcf(), bbox_inches='tight')
                plt.close()

        print(f"📄 PDF report saved to: {path}")
```

- [ ] **Step 2: Verify `FlowReport` instantiates**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import FlowExperiment, FlowReport
import pandas as pd
e = FlowExperiment()
e.populations['raw'] = pd.DataFrame({'sample':['A'], 'time':[0], 'FSC-A':[1.0], 'FSC-H':[0.9]})
e.active_pop = 'raw'
r = FlowReport(e)
r.summary()
print('FlowReport OK')
"
```
Expected: prints summary table and `FlowReport OK`.

---

## Task 9: `plot_pri_summary_grid`

**Files:**
- Modify: `thermoflow_app.py` — add method to `FlowExperiment` after `plot_pri`

- [ ] **Step 1: Add `plot_pri_summary_grid` method**

After the closing line of `plot_pri` (the `self._show_static_fig(fig, save_path)` call), add:

```python
def plot_pri_summary_grid(self, which: str = 'PRI_norm', cols: int = 4,
                           save_path: str = None, title: str = None):
    """
    Publication-ready combined figure: PRI decay curves (top) + t½ bar chart (bottom).
    Sized at FIG_2COL width. Suitable as a single multi-panel manuscript figure.
    """
    if self.pri_table.empty:
        print("⚠️ No PRI data. Run run_pri_analysis first.")
        return

    data = self.pri_table.copy()
    fits_df = self.pri_fits_norm if which == 'PRI_norm' else self.pri_fits_abs
    samples = sorted(data['sample'].unique())
    n_samples = len(samples)
    rows_top = int(np.ceil(n_samples / cols))

    fig = plt.figure(figsize=(FIG_2COL, rows_top * 2.0 + 1.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[rows_top * 2.0, 1.8],
                             hspace=0.45)

    # --- Top: decay curves ---
    inner_top = outer[0].subgridspec(rows_top, cols, hspace=0.55, wspace=0.35)
    tgrid = np.linspace(np.nanmin(data['time']), np.nanmax(data['time']), 200)

    y_vals = data[which].dropna().values
    y_pad = (y_vals.max() - y_vals.min()) * 0.08 if len(y_vals) > 0 else 0.1
    global_ylim = (min(y_vals.min() - y_pad, -0.02), y_vals.max() + y_pad)

    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(inner_top[r, c])
        g = data[data['sample'] == s].sort_values('time')

        ax.plot(g['time'].values, g[which].values, 'o',
                color='#2166ac', markeredgecolor='white',
                markeredgewidth=0.3, markersize=3.5, alpha=0.9)

        fit_row = fits_df[fits_df['sample'] == s]
        if not fit_row.empty and np.isfinite(fit_row['t_half'].values[0]):
            A = fit_row['A'].values[0]
            k = fit_row['k'].values[0]
            C = fit_row['C'].values[0]
            thalf = fit_row['t_half'].values[0]
            thalf_err = fit_row['t_half_err'].values[0]
            ax.plot(tgrid, A * np.exp(-k * tgrid) + C, '-',
                    color='#d6604d', linewidth=0.8, zorder=0)
            err_str = f"±{thalf_err:.1f}" if pd.notna(thalf_err) else ""
            label = f"$t_{{1/2}}={thalf:.1f}${err_str}"
        else:
            label = "$t_{1/2}=$ N/A"

        box_kw = dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='lightgray', linewidth=0.4)
        ax.text(0.97, 0.97, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=5.5, bbox=box_kw)
        ax.set_ylim(global_ylim)
        ax.grid(True, alpha=0.15, linewidth=0.3)
        self._style_ax(ax, title=s,
                       ylabel=which.replace('_', ' ') if c == 0 else None,
                       xlabel='Time (min)' if r == rows_top - 1 else None)

    # Hide unused top subplots
    for j in range(n_samples, rows_top * cols):
        r, c = divmod(j, cols)
        fig.add_subplot(inner_top[r, c]).axis('off')

    # --- Bottom: t½ bar chart ---
    ax_bar = fig.add_subplot(outer[1])
    valid_fits = fits_df.dropna(subset=['t_half'])
    x = np.arange(len(valid_fits))
    y = valid_fits['t_half'].values
    yerr = valid_fits['t_half_err'].fillna(0).values if 't_half_err' in valid_fits.columns else np.zeros_like(y)

    cmap_bar = mpl.colormaps['viridis']
    bar_colors = cmap_bar(np.linspace(0.15, 0.85, len(valid_fits)))
    ax_bar.bar(x, y, yerr=yerr, capsize=2.5, color=bar_colors,
               edgecolor='black', linewidth=0.5, alpha=0.9,
               error_kw={'linewidth': 0.5})
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(valid_fits['sample'], rotation=40, ha='right', fontsize=6)
    ax_bar.yaxis.grid(True, linestyle='--', linewidth=0.3, color='grey', alpha=0.4)
    ax_bar.set_axisbelow(True)
    self._style_ax(ax_bar, ylabel='$t_{1/2}$ (min)')

    if title:
        fig.suptitle(title, fontsize=8, fontweight='bold', y=1.01)

    fig.tight_layout()
    self._show_static_fig(fig, save_path)
```

- [ ] **Step 2: Verify method exists**

```bash
conda run -n e1_esm python -c "
from thermoflow_app import FlowExperiment
e = FlowExperiment()
assert hasattr(e, 'plot_pri_summary_grid'), 'method missing'
print('plot_pri_summary_grid OK')
"
```
Expected: `plot_pri_summary_grid OK`

---

## Task 10: Write & Run Test Script

**Files:**
- Create: `tests/test_thermoflow.py`

- [ ] **Step 1: Create test directory and script**

```bash
mkdir -p /home/dzyla/thermoflow/ThermoFlow/app/tests
```

- [ ] **Step 2: Write test script**

Create `/home/dzyla/thermoflow/ThermoFlow/app/tests/test_thermoflow.py`:

```python
"""
Smoke + unit tests for thermoflow_app.py
Run with: conda run -n e1_esm python tests/test_thermoflow.py
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import thermoflow_app as tf

PASS = 0
FAIL = 0

def check(name, condition, msg=''):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}: {msg}")
        FAIL += 1

def make_experiment(n_times=5, n_events=50):
    """Helper: synthetic FlowExperiment with PRI-able data."""
    rng = np.random.default_rng(0)
    rows = []
    for s in ['Ctrl', 'SampleA', 'SampleB']:
        for t in np.linspace(0, 30, n_times):
            vals = rng.exponential(scale=max(100 * np.exp(-0.05 * t), 5), size=n_events)
            for v in vals:
                rows.append({'sample': s, 'time': float(t),
                             'RL1-H': v, 'FSC-A': rng.uniform(1000, 5000),
                             'SSC-A': rng.uniform(500, 3000),
                             'well': f"{s}_{t}"})
    e = tf.FlowExperiment()
    e.populations['raw'] = pd.DataFrame(rows)
    e.active_pop = 'raw'
    return e

# ── 1. Import & constants ──────────────────────────────────────────────────────
print("\n[1] Import & constants")
check("FIG_1COL defined", hasattr(tf, 'FIG_1COL'))
check("FIG_2COL defined", hasattr(tf, 'FIG_2COL'))
check("FIG_1COL value", abs(tf.FIG_1COL - 3.46) < 0.01)
check("FIG_2COL value", abs(tf.FIG_2COL - 7.09) < 0.01)
check("FitConvergenceWarning defined", hasattr(tf, 'FitConvergenceWarning'))
check("FlowReport defined", hasattr(tf, 'FlowReport'))

# ── 2. Gate validation ─────────────────────────────────────────────────────────
print("\n[2] Gate validation")
try:
    g = tf.RectangleGate('bad', x='A', y='B', xmin=5, xmax=1)
    g.validate()
    check("RectangleGate bad xmin raises", False, "no error raised")
except ValueError:
    check("RectangleGate bad xmin raises", True)

try:
    g = tf.PolygonGate('bad', x='A', y='B', vertices=[(0,0),(1,1)])
    g.validate()
    check("PolygonGate <3 vertices raises", False, "no error raised")
except ValueError:
    check("PolygonGate <3 vertices raises", True)

try:
    g = tf.EllipseGate('bad', x='A', y='B', center=(0,0), width=0, height=1)
    g.validate()
    check("EllipseGate zero width raises", False, "no error raised")
except ValueError:
    check("EllipseGate zero width raises", True)

try:
    g = tf.ThresholdGate('bad', column='A', op='between', lo=5.0, hi=1.0)
    g.validate()
    check("ThresholdGate lo>=hi raises", False, "no error raised")
except ValueError:
    check("ThresholdGate lo>=hi raises", True)

# ── 3. load_fcs_files pre-flight ───────────────────────────────────────────────
print("\n[3] load_fcs_files pre-flight")
e = tf.FlowExperiment()
try:
    e.load_fcs_files('/no/such/path/*.fcs')
    check("load_fcs_files bad path raises FileNotFoundError", False, "no error raised")
except FileNotFoundError:
    check("load_fcs_files bad path raises FileNotFoundError", True)

# ── 4. run_pri_analysis guards ─────────────────────────────────────────────────
print("\n[4] run_pri_analysis guards")
e = make_experiment()
try:
    e.run_pri_analysis('NONEXISTENT', control_sample='Ctrl')
    check("bad channel raises KeyError", False, "no error raised")
except KeyError:
    check("bad channel raises KeyError", True)

# Run with good data
e.run_pri_analysis('RL1-H', control_sample='Ctrl')
check("pri_table non-empty", not e.pri_table.empty)
check("pri_fits_norm has r2 column", 'r2' in e.pri_fits_norm.columns)
check("pri_fits_norm has fit_quality column", 'fit_quality' in e.pri_fits_norm.columns)
check("pri_fits_abs has r2 column", 'r2' in e.pri_fits_abs.columns)

# ── 5. <3 time points skipped ─────────────────────────────────────────────────
print("\n[5] Insufficient data skip")
e2 = tf.FlowExperiment()
rows = []
rng = np.random.default_rng(1)
for t in [0, 5]:  # only 2 time points
    vals = rng.exponential(100, 30)
    for v in vals:
        rows.append({'sample': 'Short', 'time': float(t), 'RL1-H': v,
                     'FSC-A': 1000.0, 'well': f"X{t}"})
for t in [0, 5, 10, 20, 30]:
    vals = rng.exponential(100, 30)
    for v in vals:
        rows.append({'sample': 'Long', 'time': float(t), 'RL1-H': v,
                     'FSC-A': 1000.0, 'well': f"Y{t}"})
e2.populations['raw'] = pd.DataFrame(rows)
e2.active_pop = 'raw'
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    e2.run_pri_analysis('RL1-H', control_sample='Long')
    short_warned = any('Skipping fit' in str(x.message) for x in w)
check("short sample triggers UserWarning", short_warned)
short_fit = e2.pri_fits_norm[e2.pri_fits_norm['sample'] == 'Short']
check("short sample fit_quality='insufficient_data'",
      not short_fit.empty and short_fit['fit_quality'].iloc[0] == 'insufficient_data')

# ── 6. datasets property ───────────────────────────────────────────────────────
print("\n[6] datasets property")
e3 = tf.FlowExperiment()
e3.populations['raw'] = pd.DataFrame({
    'sample': ['A','A','B'], 'time': [0,1,0],
    'FSC-A': [1.0,2.0,3.0], 'well': ['A1','A1','B2'],
    'dataset': ['d1','d1','d2']
})
ds = e3.datasets
check("datasets returns dict", isinstance(ds, dict))
check("datasets has two keys", len(ds) == 2)
check("d1 n_events=2", ds.get('d1', {}).get('n_events') == 2)

# ── 7. FlowReport.summary ─────────────────────────────────────────────────────
print("\n[7] FlowReport")
e4 = make_experiment()
e4.run_pri_analysis('RL1-H', control_sample='Ctrl')
r = tf.FlowReport(e4)
s = r.summary()
check("summary is string", isinstance(s, str))
check("summary contains 'raw'", 'raw' in s)
pri_df = r.pri_summary()
check("pri_summary returns DataFrame", isinstance(pri_df, pd.DataFrame))
check("pri_summary has sample column", 'sample' in pri_df.columns)

# ── 8. plot_pri_summary_grid ──────────────────────────────────────────────────
print("\n[8] plot_pri_summary_grid")
e5 = make_experiment()
e5.run_pri_analysis('RL1-H', control_sample='Ctrl')
try:
    e5.plot_pri_summary_grid(which='PRI_norm', cols=3)
    check("plot_pri_summary_grid runs without error", True)
except Exception as ex:
    check("plot_pri_summary_grid runs without error", False, str(ex))

# ── 9. No jet colormap ────────────────────────────────────────────────────────
print("\n[9] Colormap audit")
with open(os.path.join(os.path.dirname(__file__), '..', 'thermoflow_app.py')) as f:
    src = f.read()
jet_count = src.count("'jet'") + src.count('"jet"')
check("no jet colormap in source", jet_count == 0, f"found {jet_count} occurrences")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
```

- [ ] **Step 3: Run the full test suite**

```bash
cd /home/dzyla/thermoflow/ThermoFlow/app
conda run -n e1_esm python tests/test_thermoflow.py
```
Expected: all tests PASS, exit 0.

- [ ] **Step 4: Fix any failures before continuing**

If any test fails, read the error message, trace it to the relevant task, and fix the implementation. Re-run until all pass.

---

## Task 11: Export HTML Report Smoke Test

**Files:**
- No new files — uses existing `tests/test_thermoflow.py` experiment fixture

- [ ] **Step 1: Run HTML export test**

```bash
cd /home/dzyla/thermoflow/ThermoFlow/app
conda run -n e1_esm python -c "
import matplotlib
matplotlib.use('Agg')
import sys; sys.path.insert(0, '.')
from thermoflow_app import FlowExperiment, FlowReport
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
rows = []
for s in ['Ctrl', 'SampleA']:
    for t in [0, 5, 10, 20, 30]:
        vals = rng.exponential(max(100 * np.exp(-0.05*t), 5), 40)
        for v in vals:
            rows.append({'sample':s,'time':float(t),'RL1-H':v,'FSC-A':rng.uniform(1000,5000),'SSC-A':rng.uniform(500,3000),'well':f'{s}_{t}'})

e = FlowExperiment()
e.populations['raw'] = pd.DataFrame(rows)
e.active_pop = 'raw'
e.run_pri_analysis('RL1-H', control_sample='Ctrl')

r = FlowReport(e)
r.export_html('/tmp/test_report.html')

import os
size = os.path.getsize('/tmp/test_report.html')
print(f'HTML size: {size:,} bytes')
assert size > 5000, f'HTML too small: {size}'
print('PASS: HTML report generated')
"
```
Expected: `PASS: HTML report generated` with size > 5000 bytes.

- [ ] **Step 2: Run PDF export test**

```bash
conda run -n e1_esm python -c "
import matplotlib
matplotlib.use('Agg')
import sys; sys.path.insert(0, '.')
from thermoflow_app import FlowExperiment, FlowReport
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
rows = []
for s in ['Ctrl', 'SampleA']:
    for t in [0, 5, 10, 20, 30]:
        vals = rng.exponential(max(100 * np.exp(-0.05*t), 5), 40)
        for v in vals:
            rows.append({'sample':s,'time':float(t),'RL1-H':v,'FSC-A':rng.uniform(1000,5000),'SSC-A':rng.uniform(500,3000),'well':f'{s}_{t}'})

e = FlowExperiment()
e.populations['raw'] = pd.DataFrame(rows)
e.active_pop = 'raw'
e.run_pri_analysis('RL1-H', control_sample='Ctrl')

r = FlowReport(e)
r.export_pdf('/tmp/test_report.pdf')

import os
size = os.path.getsize('/tmp/test_report.pdf')
print(f'PDF size: {size:,} bytes')
assert size > 1000, f'PDF too small'
print('PASS: PDF report generated')
"
```
Expected: `PASS: PDF report generated`.

---

## Self-Review Checklist

**Spec coverage:**
- [x] §2.1 rcParams overhaul → Task 1
- [x] §2.2 Figure width constants → Task 1
- [x] §2.3 Colormap replacements → Task 3
- [x] §2.4 `_style_ax` extension → Task 2
- [x] §3.1 `load_fcs_files` hardening → Task 5
- [x] §3.2 Gate geometry validation → Task 4
- [x] §3.3 `GateSet` error context → Task 4
- [x] §3.4 `run_pri_analysis` guards → Task 6
- [x] §3.5 `FitConvergenceWarning`, R² → Task 6
- [x] §4.1 `datasets` property → Task 7
- [x] §4.2 `dataset` plot filter → Task 7
- [x] §5 `FlowReport` class → Task 8
- [x] §6 `plot_pri_summary_grid` → Task 9
- [x] §8 Testing in e1_esm → Tasks 10, 11

**No placeholders:** All steps contain exact code.

**Type consistency:**
- `FitConvergenceWarning` defined in Task 1, referenced in Task 6 — consistent.
- `FIG_1COL`, `FIG_2COL` defined in Task 1, used in Tasks 8, 9 — consistent.
- `_style_ax(spine_style=...)` defined in Task 2, all callers updated in Task 2 — consistent.
- `r2`, `fit_quality` columns added in Task 6, read in Task 8 (`FlowReport.pri_summary`) — consistent.
- `datasets` property returns `Dict[str, Dict]` in Task 7, tested in Task 10 — consistent.
