# ThermoFlow Tier 1 Improvements — Design Spec

**Date:** 2026-04-19  
**Scope:** `thermoflow_app.py` + `gui/streamlit_app.py`  
**Backward compatible:** Yes — all new parameters use defaults that reproduce existing behaviour exactly.

---

## 1. Goal

Align the ThermoFlow library with the published methodology in the manuscript:

1. **Median MFI** — offer `mfi_metric='median'` to match the paper's definition of PRI (log-space median × f_plus).
2. **Flatline / hyperstable detection** — prevent optimizer distortion for samples whose signal does not decay (e.g. MuV F 6M); report `t_half=inf` and `fit_quality='hyperstable'`.
3. **ΔΔG‡ calculation** — compute kinetic stabilization energy `ddG_kin` (kcal/mol) relative to a WT reference using Transition State Theory, and propagate errors.

---

## 2. API Changes — `run_pri_analysis`

### New keyword parameters (all optional, backward-compatible defaults)

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `mfi_metric` | `str` | `'geometric_mean'` | `'geometric_mean'` or `'median'` |
| `wt_sample` | `str` | `None` | WT reference for ΔΔG‡; if `None`, columns are not added |
| `temperature_c` | `float` | `55.0` | Assay temperature in °C |

Full updated signature:

```python
def run_pri_analysis(
    self, channel: str, control_sample: str, samples: list = None,
    pos_frac: float = 0.01, baseline_time: int = 0, pop_name: str = None,
    n_bootstrap: int = 100, confidence: float = 0.95, ctrl_sample_list: list = None,
    reference_sample: str = None, threshold_log: float = None,
    mfi_metric: str = 'geometric_mean',
    wt_sample: str = None,
    temperature_c: float = 55.0,
):
```

---

## 3. Feature: Median MFI (`mfi_metric`)

### New module-level helper

```python
def median_mfi(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= 0)]
    return float(np.expm1(np.median(np.log1p(x + eps)))) if x.size > 0 else np.nan
```

This computes the median in log1p space and converts back to linear, matching the manuscript's definition.

### Dispatch inside `run_pri_analysis`

Wherever `geometric_mfi(...)` is called (main loop + `reference_sample` baseline path), replace with:

```python
_mfi_fn = median_mfi if mfi_metric == 'median' else geometric_mfi
```

Both paths use the same function for consistency.

### Backward compatibility

`mfi_metric='geometric_mean'` (default) → identical results to current code.

---

## 4. Feature: Flatline / Hyperstable Detection

### Location

Inside `_fit_global_exponential`, before building the optimizer parameter list, iterate over samples and classify each as "flatline" or "normal".

### Heuristic

```python
flatline_threshold: float = 0.10   # new keyword arg on _fit_global_exponential
```

For each sample's value series `y` (sorted by time):

```python
signal_drop = (y[0] - y[-1]) / (abs(y[0]) + eps)
is_flatline = signal_drop < flatline_threshold and len(y) >= 3
```

### Outcome for flatline samples

- Excluded from the global `least_squares` call (so they don't pull the shared `C` baseline).
- Result row: `A=np.nan, k=0.0, k_err=np.nan, t_half=np.inf, t_half_err=np.nan, r2=np.nan, fit_quality='hyperstable'`, `C` set to the global `C` from the fitted subset (or `np.nan` if no non-flatline samples remain).

### New parameter exposure

`_fit_global_exponential` gains `flatline_threshold=0.10`. `run_pri_analysis` passes it through (also as a keyword arg with the same default). No change to existing call sites.

### Backward compatibility

Default threshold 0.10 — samples with >10% signal drop are fitted normally, identical to current behaviour.

---

## 5. Feature: ΔΔG‡ Calculation

### Location

In `run_pri_analysis`, immediately after `self.pri_fits_norm` is assigned, if `wt_sample` is not `None`.

### Constants

```python
R_KCAL = 0.001987204258   # kcal / (mol · K) — ideal gas constant
```

### Algorithm

```python
T_K = temperature_c + 273.15
wt_row = pri_fits_norm[pri_fits_norm['sample'] == wt_sample]

# Guard: missing or invalid WT
if wt_row.empty or np.isnan(wt_row['t_half'].values[0]):
    warn and return without adding columns

t_half_wt = wt_row['t_half'].values[0]
t_half_wt_err = wt_row['t_half_err'].values[0]  # may be NaN

# ddG_kin = R·T·ln(t_half / t_half_wt)
pri_fits_norm['ddG_kin'] = R_KCAL * T_K * np.log(pri_fits_norm['t_half'] / t_half_wt)

# Error propagation: err(ln(A/B)) = sqrt((dA/A)^2 + (dB/B)^2)
rel_err_wt = (t_half_wt_err / t_half_wt)**2 if pd.notna(t_half_wt_err) else 0.0
rel_err_mut = (pri_fits_norm['t_half_err'] / pri_fits_norm['t_half'])**2
pri_fits_norm['ddG_kin_err'] = R_KCAL * T_K * np.sqrt(rel_err_mut + rel_err_wt)
```

- **WT row** → `ddG_kin = 0.0`, `ddG_kin_err = propagated`.
- **Hyperstable samples** (`t_half = inf`) → `ddG_kin = inf`.
- **Samples with `t_half = NaN`** → `ddG_kin = NaN`.

### Backward compatibility

`wt_sample=None` (default) → columns are never added; all existing code reading `pri_fits_norm` is unaffected.

---

## 6. Streamlit GUI Updates

### Location

`gui/streamlit_app.py`, inside the existing `Advanced` expander in the PRI Analysis tab.

### New controls

```python
mfi_metric    = st.radio("MFI metric", ["geometric_mean", "median"], horizontal=True)
wt_opts       = ["(none)"] + samples_in_pop
wt_sample_sel = st.selectbox("WT reference (for ΔΔG‡)", wt_opts, key="pri_wt")
wt_sample     = None if wt_sample_sel == "(none)" else wt_sample_sel
temperature_c = st.number_input("Assay temperature (°C)", value=55.0, step=0.5)
```

### Wiring

Pass all three to `exp.run_pri_analysis(...)`.

### Display

After analysis, `ddG_kin` and `ddG_kin_err` appear automatically in the existing `pri_fits_norm` dataframe display (no extra code needed — the table shows all columns).

---

## 7. Test Coverage

Add to `tests/test_thermoflow.py` (using the existing `make_experiment()` helper and `check()` function):

1. `mfi_metric='median'` produces a finite, positive `PRI_norm` value.
2. `mfi_metric='geometric_mean'` (default) result matches pre-change baseline.
3. A synthetic flat series is classified as `fit_quality='hyperstable'` with `t_half=inf`.
4. `ddG_kin` for the WT sample is `0.0`; a faster-decaying sample has `ddG_kin < 0`.
5. Calling with no new args (all defaults) produces identical output structure to the current code.

---

## 8. Files Changed

| File | Change |
|---|---|
| `thermoflow_app.py` | Add `median_mfi`; update `run_pri_analysis` signature + dispatch; update `_fit_global_exponential` with flatline detection; add ΔΔG‡ block |
| `gui/streamlit_app.py` | Add 3 controls in Advanced expander; wire to `run_pri_analysis` |
| `tests/test_thermoflow.py` | Add 5 new `check()` assertions |

No new files. No version bump in this spec (defer to implementer per semver rules).
