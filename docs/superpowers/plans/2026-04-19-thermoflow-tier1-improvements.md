# ThermoFlow Tier 1 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add median MFI toggle, hyperstable flatline detection, and ΔΔG‡ (kcal/mol) calculation to `run_pri_analysis`, wire them into the Streamlit GUI, and document them in README — all fully backward-compatible.

**Architecture:** All library changes are in `thermoflow_app.py` (single-file library). New parameters use keyword-only defaults that reproduce existing behaviour when omitted. `_fit_global_exponential` gains a `flatline_threshold` parameter; ΔΔG‡ columns are appended to `pri_fits_norm` in-place after fitting. GUI changes are isolated to the `Advanced` expander in `gui/streamlit_app.py`.

**Tech Stack:** Python 3, NumPy, SciPy (`least_squares`), Pandas, Streamlit, Matplotlib. Test runner: plain script `conda run -n e1_esm python tests/test_thermoflow.py`.

---

## File Map

| File | What changes |
|---|---|
| `thermoflow_app.py:365-368` | Add `median_mfi` helper after `geometric_mfi` |
| `thermoflow_app.py:1695-1698` | Extend `run_pri_analysis` signature |
| `thermoflow_app.py:1762` | Add `_mfi_fn` dispatch variable |
| `thermoflow_app.py:1754,1773,1781,1811` | Replace `geometric_mfi(` with `_mfi_fn(` |
| `thermoflow_app.py:1821-1822` | Pass `flatline_threshold` to both `_fit_global_exponential` calls |
| `thermoflow_app.py:1823-1832` | Add ΔΔG‡ block after fits are assigned |
| `thermoflow_app.py:1846` | Add `flatline_threshold=0.10` to `_fit_global_exponential` signature |
| `thermoflow_app.py:1877-1879` | Insert flatline-detection block; update early-return to include flatline rows |
| `thermoflow_app.py:1964` | Include `flat_rows` in final `return` |
| `gui/streamlit_app.py:759-771` | Add 3 new controls in Advanced expander |
| `gui/streamlit_app.py:813-822` | Pass 3 new args to `run_pri_analysis` |
| `tests/test_thermoflow.py` | Add `make_experiment_flat()` + sections 10, 11, 12 |
| `README.md:163` | Insert `### Advanced PRI Analysis` section before `---` |

---

## Task 1: Add `median_mfi` helper and `mfi_metric` dispatch

### Files
- Modify: `thermoflow_app.py` (lines 365–368, 1695–1698, 1754, 1762, 1773, 1781, 1811)
- Test: `tests/test_thermoflow.py`

- [ ] **Step 1.1: Write the failing test — section [10]**

Append to `tests/test_thermoflow.py`:

```python
# ── 10. Tier 1 — median MFI ───────────────────────────────────────────────────
print("\n[10] Tier 1 — median MFI (mfi_metric)")
check("median_mfi helper exists", hasattr(tf, 'median_mfi'))

e10 = make_experiment()
e10.run_pri_analysis('RL1-H', control_sample='Ctrl', mfi_metric='median')
check("median: pri_table non-empty", not e10.pri_table.empty)
check("median: PRI_norm all finite",
      e10.pri_table['PRI_norm'].dropna().apply(np.isfinite).all())
check("median: pri_fits_norm has t_half", 't_half' in e10.pri_fits_norm.columns)

# Default (geometric_mean) must produce identical columns — backward compat
e10b = make_experiment()
e10b.run_pri_analysis('RL1-H', control_sample='Ctrl')  # no mfi_metric arg
check("geom_mean default: ddG_kin absent (no wt_sample)",
      'ddG_kin' not in e10b.pri_fits_norm.columns)
check("geom_mean default: pri_table non-empty", not e10b.pri_table.empty)
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)\s+\[10\]|^\[10\]|AttributeError|TypeError"
```

Expected: `FAIL  median_mfi helper exists` (attribute not yet defined).

- [ ] **Step 1.3: Add `median_mfi` after `geometric_mfi` in `thermoflow_app.py`**

At line 369, after the `geometric_mfi` definition, insert:

```python
def median_mfi(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= 0)]
    return float(np.expm1(np.median(np.log1p(x + eps)))) if x.size > 0 else np.nan
```

- [ ] **Step 1.4: Extend `run_pri_analysis` signature**

Current signature (lines 1695–1698):
```python
def run_pri_analysis(self, channel: str, control_sample: str, samples: list = None,
                     pos_frac: float = 0.01, baseline_time: int = 0, pop_name: str = None,
                     n_bootstrap: int = 100, confidence: float = 0.95, ctrl_sample_list: list = None,
                     reference_sample: str = None, threshold_log: float = None):
```

Replace with:
```python
def run_pri_analysis(self, channel: str, control_sample: str, samples: list = None,
                     pos_frac: float = 0.01, baseline_time: int = 0, pop_name: str = None,
                     n_bootstrap: int = 100, confidence: float = 0.95, ctrl_sample_list: list = None,
                     reference_sample: str = None, threshold_log: float = None,
                     mfi_metric: str = 'geometric_mean',
                     wt_sample: str = None,
                     temperature_c: float = 55.0,
                     flatline_threshold: float = 0.10):
```

- [ ] **Step 1.5: Add `_mfi_fn` dispatch and replace all `geometric_mfi` calls**

After line 1762 (the `print(f"✅ Normalizing all samples...")` block, just before `tables = []`), add:

```python
        _mfi_fn = median_mfi if mfi_metric == 'median' else geometric_mfi
```

Then replace the four `geometric_mfi(` occurrences inside `run_pri_analysis` with `_mfi_fn(`:

- Line ~1754 (reference_sample path):
  `ref_gmfi = geometric_mfi(ref_vals[ref_log >= thr_log]) if ref_n_pos else 0.0`
  → `ref_gmfi = _mfi_fn(ref_vals[ref_log >= thr_log]) if ref_n_pos else 0.0`

- Line ~1773 (self-baseline):
  `gmfi0 = geometric_mfi(t0_vals[t0_pos_mask])`
  → `gmfi0 = _mfi_fn(t0_vals[t0_pos_mask])`

- Line ~1781 (per-timepoint MFI):
  `gmfi_pos = geometric_mfi(vals[log_vals >= thr_log]) if n_pos else 0.0`
  → `gmfi_pos = _mfi_fn(vals[log_vals >= thr_log]) if n_pos else 0.0`

- Line ~1811 (bootstrap):
  `boot_gmfi = geometric_mfi(boot_vals[boot_log >= thr_log]) if boot_n_pos else 0.0`
  → `boot_gmfi = _mfi_fn(boot_vals[boot_log >= thr_log]) if boot_n_pos else 0.0`

- [ ] **Step 1.6: Run tests — section [10] must now pass**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)|\[10\]"
```

Expected: all `[10]` checks PASS, no regressions in sections [1]–[9].

- [ ] **Step 1.7: Commit**

```bash
git add thermoflow_app.py tests/test_thermoflow.py
git commit -m "feat: add median_mfi helper and mfi_metric param to run_pri_analysis"
```

---

## Task 2: Flatline / hyperstable detection in `_fit_global_exponential`

### Files
- Modify: `thermoflow_app.py` (lines 1821–1822, 1846, 1877–1879, 1964)
- Test: `tests/test_thermoflow.py`

- [ ] **Step 2.1: Write the failing test — section [11]**

Before the section [10] block added in Task 1, add a new helper at the top of the test additions area, and then add section [11]:

```python
def make_experiment_flat():
    """Experiment with a flat/hyperstable sample (no signal decay over time)."""
    rng = np.random.default_rng(1)
    rows = []
    times = np.linspace(0, 30, 5)
    for t in times:
        # Ctrl — low signal (untransfected)
        for v in rng.exponential(scale=5, size=50):
            rows.append({'sample': 'Ctrl', 'time': float(t), 'RL1-H': float(v),
                         'FSC-A': 1000., 'SSC-A': 500., 'well': f'Ctrl_{t}'})
        # WT — decaying signal
        for v in rng.exponential(scale=max(300 * np.exp(-0.15 * t), 5), size=50):
            rows.append({'sample': 'WT', 'time': float(t), 'RL1-H': float(v),
                         'FSC-A': 1000., 'SSC-A': 500., 'well': f'WT_{t}'})
        # HyperStable — constant high signal (no decay)
        for v in rng.exponential(scale=300, size=50):
            rows.append({'sample': 'HyperStable', 'time': float(t), 'RL1-H': float(v),
                         'FSC-A': 1000., 'SSC-A': 500., 'well': f'HS_{t}'})
    e = tf.FlowExperiment()
    e.populations['raw'] = pd.DataFrame(rows)
    e.active_pop = 'raw'
    return e
```

Then:

```python
# ── 11. Tier 1 — flatline / hyperstable detection ─────────────────────────────
print("\n[11] Tier 1 — flatline detection")
e11 = make_experiment_flat()
e11.run_pri_analysis('RL1-H', control_sample='Ctrl', flatline_threshold=0.10)

hs = e11.pri_fits_norm[e11.pri_fits_norm['sample'] == 'HyperStable']
check("hyperstable: fit_quality='hyperstable'",
      not hs.empty and hs['fit_quality'].iloc[0] == 'hyperstable',
      f"got '{hs['fit_quality'].iloc[0] if not hs.empty else 'missing'}'")
check("hyperstable: t_half is inf",
      not hs.empty and hs['t_half'].iloc[0] == np.inf)
check("hyperstable: k == 0.0",
      not hs.empty and hs['k'].iloc[0] == 0.0)

wt = e11.pri_fits_norm[e11.pri_fits_norm['sample'] == 'WT']
check("WT sample fitted normally (not hyperstable)",
      not wt.empty and wt['fit_quality'].iloc[0] != 'hyperstable')
check("WT sample has finite t_half",
      not wt.empty and np.isfinite(wt['t_half'].iloc[0]))
```

- [ ] **Step 2.2: Run to confirm it fails**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)|\[11\]"
```

Expected: `FAIL  hyperstable: fit_quality='hyperstable'` (detection not yet implemented).

- [ ] **Step 2.3: Add `flatline_threshold` param to `_fit_global_exponential` signature**

Current (line 1846):
```python
    def _fit_global_exponential(self, df_source: pd.DataFrame, which: str) -> pd.DataFrame:
```

Replace with:
```python
    def _fit_global_exponential(self, df_source: pd.DataFrame, which: str,
                                 flatline_threshold: float = 0.10) -> pd.DataFrame:
```

- [ ] **Step 2.4: Insert flatline detection block after `skipped_rows` is built**

After line 1876 (the closing `]` of `skipped_rows`), before the `if not samples:` guard, insert:

```python
        # --- Flatline / hyperstable detection ---
        _eps = 1e-12
        flatline_samples = []
        normal_samples, normal_times, normal_values = [], [], []
        for _s, _t, _v in zip(samples, times_list, values_list):
            _vf = _v[np.isfinite(_v)]
            if _vf.size > 0 and abs(_vf[0]) > _eps:
                _drop = (_vf[0] - _vf[-1]) / (abs(_vf[0]) + _eps)
            else:
                _drop = 1.0  # unknown — treat as normal decaying
            if _drop < flatline_threshold:
                flatline_samples.append(_s)
            else:
                normal_samples.append(_s)
                normal_times.append(_t)
                normal_values.append(_v)
        samples, times_list, values_list = normal_samples, normal_times, normal_values
```

- [ ] **Step 2.5: Update the early-return guard to include flatline rows**

Current (lines 1878–1879):
```python
        if not samples:
            return pd.DataFrame(skipped_rows)
```

Replace with:
```python
        if not samples:
            _flat_rows = [
                dict(sample=_s, A=np.nan, A_err=np.nan, k=0.0, k_err=np.nan,
                     C=np.nan, C_err=np.nan, t_half=np.inf, t_half_err=np.nan,
                     r2=np.nan, fit_quality='hyperstable')
                for _s in flatline_samples
            ]
            return pd.DataFrame(skipped_rows + _flat_rows)
```

- [ ] **Step 2.6: Build `flat_rows` after `global_C` is known and include in return**

After line 1903 (`global_C = res.x[0]`), insert:

```python
        flat_rows = [
            dict(sample=_s, A=np.nan, A_err=np.nan, k=0.0, k_err=np.nan,
                 C=global_C, C_err=param_errors[0] if not np.all(np.isnan(param_errors)) else np.nan,
                 t_half=np.inf, t_half_err=np.nan, r2=np.nan, fit_quality='hyperstable')
            for _s in flatline_samples
        ]
```

Note: `param_errors` is defined at line 1919. Move `flat_rows` construction to after line 1925 (after `param_errors` is computed):

```python
        flat_rows = [
            dict(sample=_s, A=np.nan, A_err=np.nan, k=0.0, k_err=np.nan,
                 C=global_C, C_err=param_errors[0],
                 t_half=np.inf, t_half_err=np.nan, r2=np.nan, fit_quality='hyperstable')
            for _s in flatline_samples
        ]
```

- [ ] **Step 2.7: Update the final return to include `flat_rows`**

Current (line 1964–1965):
```python
        all_rows = out + skipped_rows
        return pd.DataFrame(all_rows).sort_values("sample").reset_index(drop=True)
```

Replace with:
```python
        all_rows = out + skipped_rows + flat_rows
        return pd.DataFrame(all_rows).sort_values("sample").reset_index(drop=True)
```

- [ ] **Step 2.8: Pass `flatline_threshold` from `run_pri_analysis` to both fit calls**

Current (lines 1821–1822):
```python
        self.pri_fits_abs = self._fit_global_exponential(self.pri_table, "PRI_abs")
        self.pri_fits_norm = self._fit_global_exponential(self.pri_table, "PRI_norm")
```

Replace with:
```python
        self.pri_fits_abs = self._fit_global_exponential(self.pri_table, "PRI_abs",
                                                          flatline_threshold=flatline_threshold)
        self.pri_fits_norm = self._fit_global_exponential(self.pri_table, "PRI_norm",
                                                           flatline_threshold=flatline_threshold)
```

- [ ] **Step 2.9: Run tests — sections [10] and [11] must pass, no regressions**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)|\[10\]|\[11\]"
```

Expected: all checks PASS. Full suite:

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | tail -5
```

Expected: `PASS: N  FAIL: 0`.

- [ ] **Step 2.10: Commit**

```bash
git add thermoflow_app.py tests/test_thermoflow.py
git commit -m "feat: add flatline/hyperstable detection to _fit_global_exponential"
```

---

## Task 3: ΔΔG‡ calculation in `run_pri_analysis`

### Files
- Modify: `thermoflow_app.py` (after line 1822)
- Test: `tests/test_thermoflow.py`

- [ ] **Step 3.1: Write the failing test — section [12]**

Append to `tests/test_thermoflow.py`:

```python
# ── 12. Tier 1 — ΔΔG‡ calculation ────────────────────────────────────────────
print("\n[12] Tier 1 — ΔΔG\u2021 (ddG_kin)")
e12 = make_experiment_flat()
e12.run_pri_analysis('RL1-H', control_sample='Ctrl',
                     wt_sample='WT', temperature_c=55.0)
check("ddG_kin column present in pri_fits_norm",
      'ddG_kin' in e12.pri_fits_norm.columns)
check("ddG_kin_err column present in pri_fits_norm",
      'ddG_kin_err' in e12.pri_fits_norm.columns)

wt_ddg = e12.pri_fits_norm.loc[e12.pri_fits_norm['sample'] == 'WT', 'ddG_kin']
check("WT ddG_kin == 0.0",
      not wt_ddg.empty and abs(wt_ddg.values[0]) < 1e-10,
      f"got {wt_ddg.values[0] if not wt_ddg.empty else 'missing'}")

hs_ddg = e12.pri_fits_norm.loc[e12.pri_fits_norm['sample'] == 'HyperStable', 'ddG_kin']
check("hyperstable ddG_kin is inf",
      not hs_ddg.empty and hs_ddg.values[0] == np.inf,
      f"got {hs_ddg.values[0] if not hs_ddg.empty else 'missing'}")

# No wt_sample → columns must NOT appear (backward compat)
e12b = make_experiment()
e12b.run_pri_analysis('RL1-H', control_sample='Ctrl')
check("no wt_sample → ddG_kin absent",
      'ddG_kin' not in e12b.pri_fits_norm.columns)
check("no wt_sample → ddG_kin_err absent",
      'ddG_kin_err' not in e12b.pri_fits_norm.columns)
```

- [ ] **Step 3.2: Run to confirm it fails**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)|\[12\]"
```

Expected: `FAIL  ddG_kin column present in pri_fits_norm`.

- [ ] **Step 3.3: Add ΔΔG‡ block to `run_pri_analysis`**

After the two `_fit_global_exponential` calls (after the updated lines 1821–1822 from Task 2), add:

```python
        # ΔΔG‡ via Transition State Theory (Eyring equation, pre-exponential terms cancel)
        if wt_sample is not None:
            _R_KCAL = 0.001987204258  # kcal / (mol·K)
            _T_K = temperature_c + 273.15
            _wt = self.pri_fits_norm[self.pri_fits_norm['sample'] == wt_sample]
            if _wt.empty:
                print(f"⚠️ wt_sample '{wt_sample}' not found in pri_fits_norm; skipping ΔΔG‡.")
            else:
                _t_half_wt = _wt['t_half'].values[0]
                _t_half_wt_err = _wt['t_half_err'].values[0]
                if not np.isfinite(_t_half_wt) or _t_half_wt <= 0:
                    print(f"⚠️ wt_sample '{wt_sample}' has invalid t_half={_t_half_wt}; skipping ΔΔG‡.")
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.pri_fits_norm['ddG_kin'] = (
                            _R_KCAL * _T_K
                            * np.log(self.pri_fits_norm['t_half'] / _t_half_wt)
                        )
                    _rel_err_wt = ((_t_half_wt_err / _t_half_wt) ** 2
                                   if pd.notna(_t_half_wt_err) and _t_half_wt_err > 0 else 0.0)
                    _rel_err_mut = (self.pri_fits_norm['t_half_err']
                                    / self.pri_fits_norm['t_half']) ** 2
                    self.pri_fits_norm['ddG_kin_err'] = (
                        _R_KCAL * _T_K * np.sqrt(_rel_err_mut + _rel_err_wt)
                    )
                    print(f"✅ ΔΔG‡ calculated relative to '{wt_sample}' at {temperature_c}°C "
                          f"(R={_R_KCAL} kcal/mol/K, T={_T_K:.2f} K).")
```

- [ ] **Step 3.4: Run tests — all sections must pass**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | grep -E "^\s*(FAIL|PASS)|\[10\]|\[11\]|\[12\]"
```

Expected: all checks PASS. Then run full suite:

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1 | tail -5
```

Expected: `PASS: N  FAIL: 0`.

- [ ] **Step 3.5: Commit**

```bash
git add thermoflow_app.py tests/test_thermoflow.py
git commit -m "feat: compute ddG_kin / ddG_kin_err in run_pri_analysis via Transition State Theory"
```

---

## Task 4: Streamlit GUI — wire new parameters

### Files
- Modify: `gui/streamlit_app.py` (Advanced expander ~lines 759–771, run call ~lines 813–822)

- [ ] **Step 4.1: Add three controls inside the `Advanced` expander**

Locate the `with st.expander("Advanced"):` block (around line 759). After the existing `custom_thr` block (after the `if use_thr:` section), append:

```python
                st.divider()
                mfi_metric = st.radio(
                    "MFI metric", ["geometric_mean", "median"],
                    horizontal=True, key="pri_mfi_metric",
                    help="'median' matches the published PRI definition (log-space median × f_plus).",
                )
                wt_opts = ["(none)"] + samples_in_pop
                wt_sample_sel = st.selectbox(
                    "WT reference (for ΔΔG‡)", wt_opts, key="pri_wt",
                    help="If set, adds ddG_kin and ddG_kin_err columns (kcal/mol) to fit results.",
                )
                wt_sample = None if wt_sample_sel == "(none)" else wt_sample_sel
                temperature_c = st.number_input(
                    "Assay temperature (°C)", value=55.0, step=0.5,
                    format="%.1f", key="pri_temp",
                )
```

Note: `mfi_metric`, `wt_sample`, and `temperature_c` must be defined in the same `with tab_pri:` scope so they are accessible at the `run_pri_analysis` call below.

- [ ] **Step 4.2: Pass new parameters to `run_pri_analysis`**

Current call (lines 813–822):
```python
                        exp.run_pri_analysis(
                            channel=pri_channel,
                            control_sample=ctrl_sample,
                            pop_name=pri_pop,
                            reference_sample=ref_sample,
                            pos_frac=pos_frac,
                            n_bootstrap=n_boot,
                            baseline_time=baseline_time,
                            threshold_log=custom_thr if use_thr else None,
                        )
```

Replace with:
```python
                        exp.run_pri_analysis(
                            channel=pri_channel,
                            control_sample=ctrl_sample,
                            pop_name=pri_pop,
                            reference_sample=ref_sample,
                            pos_frac=pos_frac,
                            n_bootstrap=n_boot,
                            baseline_time=baseline_time,
                            threshold_log=custom_thr if use_thr else None,
                            mfi_metric=mfi_metric,
                            wt_sample=wt_sample,
                            temperature_c=temperature_c,
                        )
```

- [ ] **Step 4.3: Verify controls are accessible before the run button**

The `mfi_metric`, `wt_sample`, and `temperature_c` variables are created inside `with st.expander("Advanced"):`, which is nested inside `with p_left:`. The `run_pri_analysis` call is inside a button callback also in `with p_left:`. Confirm both are in the same `with tab_pri:` block scope — no indentation fix should be needed, but double-check visually.

- [ ] **Step 4.4: Smoke-test the Streamlit app**

```bash
conda run -n e1_esm streamlit run gui/streamlit_app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | grep -c "ThermoFlow"
kill %1
```

Expected: output ≥ 1 (page loads). If Playwright is available, additionally navigate to the PRI Analysis tab and confirm the three new controls appear in the Advanced expander.

- [ ] **Step 4.5: Commit**

```bash
git add gui/streamlit_app.py
git commit -m "feat: expose mfi_metric, wt_sample, temperature_c in Streamlit PRI tab"
```

---

## Task 5: README — add Advanced PRI Analysis examples

### Files
- Modify: `README.md` (after the `### Custom threshold` section, before `---`)

- [ ] **Step 5.1: Insert the new section**

In `README.md`, locate the line `---` that follows the `### Custom threshold` code block (line 164). Insert before it:

```markdown
### Advanced PRI Analysis

```python
# Median MFI metric (matches manuscript definition) + ΔΔG‡ relative to WT
exp.run_pri_analysis(
    channel="APC-A",
    control_sample="untransfected",
    mfi_metric="median",          # log-space median × f_plus
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
```

- [ ] **Step 5.2: Verify README renders correctly**

```bash
grep -n "Advanced PRI Analysis\|flatline_threshold\|ddG_kin\|mfi_metric" README.md
```

Expected: all four strings appear, each at least once.

- [ ] **Step 5.3: Commit**

```bash
git add README.md
git commit -m "docs: add Advanced PRI Analysis examples (median MFI, flatline, ΔΔG‡)"
```

---

## Task 6: Final verification + version bump

- [ ] **Step 6.1: Run the full test suite**

```bash
conda run -n e1_esm python tests/test_thermoflow.py 2>&1
```

Expected final line: `PASS: N  FAIL: 0` (zero failures).

- [ ] **Step 6.2: Verify no `jet` colormap in source**

```bash
grep -n '"jet"\|'"'jet'" thermoflow_app.py && echo "FOUND JET - BAD" || echo "jet-free OK"
```

Expected: `jet-free OK`.

- [ ] **Step 6.3: Bump version (minor — new features added)**

Update these three places together:

1. `thermoflow_app.py` top: `__version__ = "0.4.0"`
2. `pyproject.toml`: `version = "0.4.0"`
3. `CHANGELOG.md`: add entry:

```markdown
## [0.4.0] - 2026-04-19
### Added
- `median_mfi` helper and `mfi_metric` param in `run_pri_analysis` (`'geometric_mean'` | `'median'`)
- Flatline / hyperstable detection in `_fit_global_exponential`; `fit_quality='hyperstable'`, `t_half=inf`
- ΔΔG‡ calculation (`ddG_kin`, `ddG_kin_err` columns) via Transition State Theory when `wt_sample` is supplied
- Streamlit GUI: MFI metric radio, WT reference selector, assay temperature input in PRI tab
- README: Advanced PRI Analysis examples
```

- [ ] **Step 6.4: Final commit**

```bash
git add thermoflow_app.py pyproject.toml CHANGELOG.md
git commit -m "chore: bump to v0.4.0 — Tier 1 biophysics improvements"
```
