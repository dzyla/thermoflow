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
matplotlib.use('Agg')  # non-interactive backend — must come before pyplot import
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


def make_experiment(n_times=5, n_events=50):
    """Synthetic FlowExperiment with PRI-able data."""
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
    g = tf.PolygonGate('bad', x='A', y='B', vertices=[(0, 0), (1, 1)])
    g.validate()
    check("PolygonGate <3 vertices raises", False, "no error raised")
except ValueError:
    check("PolygonGate <3 vertices raises", True)

try:
    g = tf.EllipseGate('bad', x='A', y='B', center=(0, 0), width=0, height=1)
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
    'sample': ['A', 'A', 'B'], 'time': [0, 1, 0],
    'FSC-A': [1.0, 2.0, 3.0], 'well': ['A1', 'A1', 'B2'],
    'dataset': ['d1', 'd1', 'd2']
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
src_path = os.path.join(os.path.dirname(__file__), '..', 'thermoflow_app.py')
with open(src_path) as f:
    src = f.read()
jet_count = src.count("'jet'") + src.count('"jet"')
check("no jet colormap in source", jet_count == 0, f"found {jet_count} occurrences")

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

e10c = make_experiment()
e10c.run_pri_analysis('RL1-H', control_sample='Ctrl', mfi_metric='geometric_mean')
check("median != geometric_mean PRI values (metrics differ)",
      not np.allclose(
          e10.pri_table['PRI_norm'].dropna().values,
          e10c.pri_table['PRI_norm'].dropna().values,
      ))

# ── 11. export_html ───────────────────────────────────────────────────────────
print("\n[11] HTML/PDF export")
import tempfile
e6 = make_experiment()
e6.run_pri_analysis('RL1-H', control_sample='Ctrl')
r6 = tf.FlowReport(e6)

with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
    html_path = f.name
r6.export_html(html_path)
html_size = os.path.getsize(html_path)
check("HTML file created", os.path.exists(html_path))
check("HTML file > 5 KB", html_size > 5000, f"size={html_size}")
os.unlink(html_path)

with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
    pdf_path = f.name
r6.export_pdf(pdf_path)
pdf_size = os.path.getsize(pdf_path)
check("PDF file created", os.path.exists(pdf_path))
check("PDF file > 1 KB", pdf_size > 1000, f"size={pdf_size}")
os.unlink(pdf_path)

# ── 12. Tier 1 — flatline / hyperstable detection ─────────────────────────────
print("\n[12] Tier 1 — flatline detection")
e12 = make_experiment_flat()
e12.run_pri_analysis('RL1-H', control_sample='Ctrl', flatline_threshold=0.10)

hs = e12.pri_fits_norm[e12.pri_fits_norm['sample'] == 'HyperStable']
check("hyperstable: fit_quality='hyperstable'",
      not hs.empty and hs['fit_quality'].iloc[0] == 'hyperstable',
      f"got '{hs['fit_quality'].iloc[0] if not hs.empty else 'missing'}'")
check("hyperstable: t_half is inf",
      not hs.empty and hs['t_half'].iloc[0] == np.inf)
check("hyperstable: k == 0.0",
      not hs.empty and hs['k'].iloc[0] == 0.0)

wt = e12.pri_fits_norm[e12.pri_fits_norm['sample'] == 'WT']
check("WT sample fitted normally (not hyperstable)",
      not wt.empty and wt['fit_quality'].iloc[0] != 'hyperstable')
check("WT sample has finite t_half",
      not wt.empty and np.isfinite(wt['t_half'].iloc[0]))

# ── 13. Tier 1 — ΔΔG‡ calculation ────────────────────────────────────────────
print("\n[13] Tier 1 \u2014 \u0394\u0394G\u2021 (ddG_kin)")
e13 = make_experiment_flat()
e13.run_pri_analysis('RL1-H', control_sample='Ctrl',
                     wt_sample='WT', temperature_c=55.0)
check("ddG_kin column present in pri_fits_norm",
      'ddG_kin' in e13.pri_fits_norm.columns)
check("ddG_kin_err column present in pri_fits_norm",
      'ddG_kin_err' in e13.pri_fits_norm.columns)

wt_ddg = e13.pri_fits_norm.loc[e13.pri_fits_norm['sample'] == 'WT', 'ddG_kin']
check("WT ddG_kin == 0.0",
      not wt_ddg.empty and abs(wt_ddg.values[0]) < 1e-10,
      f"got {wt_ddg.values[0] if not wt_ddg.empty else 'missing'}")

hs_ddg = e13.pri_fits_norm.loc[e13.pri_fits_norm['sample'] == 'HyperStable', 'ddG_kin']
check("hyperstable ddG_kin is inf",
      not hs_ddg.empty and hs_ddg.values[0] == np.inf,
      f"got {hs_ddg.values[0] if not hs_ddg.empty else 'missing'}")

wt_err = e13.pri_fits_norm.loc[e13.pri_fits_norm['sample'] == 'WT', 'ddG_kin_err']
check("WT ddG_kin_err is finite and non-negative",
      not wt_err.empty and np.isfinite(wt_err.values[0]) and wt_err.values[0] >= 0)

# No wt_sample → columns must NOT appear (backward compat)
e13b = make_experiment()
e13b.run_pri_analysis('RL1-H', control_sample='Ctrl')
check("no wt_sample: ddG_kin absent",
      'ddG_kin' not in e13b.pri_fits_norm.columns)
check("no wt_sample: ddG_kin_err absent",
      'ddG_kin_err' not in e13b.pri_fits_norm.columns)

# ── 14. Weighted fit removes expression-scale bias ────────────────────────────
print("\n[14] Weighted fit — scale-domination regression")
# Two samples, IDENTICAL true k and shared absolute background, 20x expression gap.
# A correct fit must recover nearly the same k for both despite the shared C.
def _make_scale_experiment():
    rng = np.random.default_rng(7)
    times = [0, 1, 3, 5, 10, 20, 30]
    ktrue, C_abs = 0.05, 30.0
    rows = []
    # control to set threshold well below signal
    for t in times:
        for v in rng.exponential(5, 200):
            rows.append({'sample': 'Ctrl', 'time': float(t), 'RL1-H': float(v), 'well': f'C{t}'})
    for name, A in [('Hi', 4000.0), ('Lo', 200.0)]:
        for t in times:
            mu = A * np.exp(-ktrue * t) + C_abs
            for v in rng.lognormal(np.log(mu), 0.10, 400):
                rows.append({'sample': name, 'time': float(t), 'RL1-H': float(v), 'well': f'{name}{t}'})
    ex = tf.FlowExperiment()
    ex.populations['raw'] = pd.DataFrame(rows); ex.active_pop = 'raw'
    return ex

e14 = _make_scale_experiment()
e14.run_pri_analysis('RL1-H', control_sample='Ctrl', reference_sample='Hi', n_bootstrap=0)
k_hi = e14.pri_fits_abs.loc[e14.pri_fits_abs['sample'] == 'Hi', 'k'].values[0]
k_lo = e14.pri_fits_abs.loc[e14.pri_fits_abs['sample'] == 'Lo', 'k'].values[0]
check("weighted fit: Hi k near 0.05", abs(k_hi - 0.05) / 0.05 < 0.25, f"k_hi={k_hi:.4f}")
check("weighted fit: Lo k near 0.05 (no scale domination)",
      abs(k_lo - 0.05) / 0.05 < 0.35, f"k_lo={k_lo:.4f}")

# Bootstrap SE columns now present for BOTH metrics
e14b = _make_scale_experiment()
e14b.run_pri_analysis('RL1-H', control_sample='Ctrl', reference_sample='Hi', n_bootstrap=50)
check("PRI_abs_se column present", 'PRI_abs_se' in e14b.pri_table.columns)
check("PRI_norm_se column present", 'PRI_norm_se' in e14b.pri_table.columns)
check("PRI_norm CI columns present",
      'PRI_norm_ci_low' in e14b.pri_table.columns and 'PRI_norm_ci_high' in e14b.pri_table.columns)

# Deterministic given random_state
e14c = _make_scale_experiment()
e14c.run_pri_analysis('RL1-H', control_sample='Ctrl', reference_sample='Hi', n_bootstrap=50)
check("bootstrap reproducible with fixed random_state",
      np.allclose(e14b.pri_table['PRI_abs_se'].fillna(0).values,
                  e14c.pri_table['PRI_abs_se'].fillna(0).values))

# ── 15. Significance-aware flatline ────────────────────────────────────────────
print("\n[15] Flatline requires statistical insignificance")
# A slow but clearly real decay must NOT be flagged hyperstable.
rng = np.random.default_rng(3)
rows = []
times = [0, 1, 3, 5, 10, 20, 30]
for t in times:
    for v in rng.exponential(5, 100):
        rows.append({'sample': 'Ctrl', 'time': float(t), 'RL1-H': float(v), 'well': f'C{t}'})
    # gentle but monotonic decay ~ e^{-0.012 t}: ~30% total drop, low noise
    mu = 500 * np.exp(-0.012 * t) + 20
    for v in rng.lognormal(np.log(mu), 0.03, 300):
        rows.append({'sample': 'SlowReal', 'time': float(t), 'RL1-H': float(v), 'well': f'S{t}'})
e15 = tf.FlowExperiment()
e15.populations['raw'] = pd.DataFrame(rows); e15.active_pop = 'raw'
e15.run_pri_analysis('RL1-H', control_sample='Ctrl', n_bootstrap=0, flatline_threshold=0.5)
sr = e15.pri_fits_abs[e15.pri_fits_abs['sample'] == 'SlowReal']
check("significant slow decay NOT flagged hyperstable",
      not sr.empty and sr['fit_quality'].iloc[0] != 'hyperstable',
      f"got '{sr['fit_quality'].iloc[0] if not sr.empty else 'missing'}'")
check("significant slow decay has finite t_half",
      not sr.empty and np.isfinite(sr['t_half'].iloc[0]))

# ── 16. Per-plate normalization ────────────────────────────────────────────────
print("\n[16] Per-plate (multi-dataset) normalization")
rng = np.random.default_rng(11)
rows = []
# Two plates with different instrument gain; each has its own Ctrl + Fwt.
for plate, gain in [('P1', 1.0), ('P2', 3.0)]:
    for name, A, k in [('Fwt', 400.0, 0.04), ('Mut', 300.0, 0.10)]:
        for t in times:
            mu = gain * (A * np.exp(-k * t) + 15)
            for v in rng.lognormal(np.log(mu), 0.08, 300):
                rows.append({'sample': name, 'time': float(t), 'RL1-H': float(v),
                             'well': f'{plate}{name}{t}', 'dataset': plate})
    for t in times:
        for v in rng.exponential(5 * gain, 150):
            rows.append({'sample': 'Ctrl', 'time': float(t), 'RL1-H': float(v),
                         'well': f'{plate}C{t}', 'dataset': plate})
e16 = tf.FlowExperiment()
e16.populations['raw'] = pd.DataFrame(rows); e16.active_pop = 'raw'
e16.run_pri_analysis('RL1-H', control_sample='Ctrl', reference_sample='Fwt',
                     per_plate=True, n_bootstrap=0)
check("per_plate: pri_table has dataset column", 'dataset' in e16.pri_table.columns)
check("per_plate: fits have dataset column", 'dataset' in e16.pri_fits_norm.columns)
check("per_plate: Fwt appears once per plate (2 rows)",
      (e16.pri_fits_norm['sample'] == 'Fwt').sum() == 2)
# The mutant's k should be recovered on BOTH plates despite 3x gain difference
mut = e16.pri_fits_abs[e16.pri_fits_abs['sample'] == 'Mut'].sort_values('dataset')
check("per_plate: Mut k consistent across plates (gain-independent)",
      len(mut) == 2 and abs(mut['k'].values[0] - mut['k'].values[1]) < 0.03,
      f"k={mut['k'].values}")
# Plotting must not crash with duplicate sample names across plates
try:
    e16.plot_pri(which='PRI_norm', cols=2)
    e16.plot_pri_bars(which='t_half', use_norm=True, norm_sample='Fwt')
    check("per_plate: plotting runs without error", True)
except Exception as ex:
    check("per_plate: plotting runs without error", False, str(ex))

# ── 17. Robust covariance — no negative-sqrt, non-negative errors ──────────────
print("\n[17] Robust covariance (no negative sqrt)")
# Rank-deficient Jacobians (parameters pinned at bounds) previously made
# pinv(JᵀJ) return tiny-negative variances → 'invalid value in sqrt' → NaN errors.
# The per-plate multi-gain layout reliably reproduces that pathology.
_rng17 = np.random.default_rng(11)
_rows17 = []
_times17 = [0, 1, 3, 5, 10, 20, 30]
for _plate, _gain in [('P1', 1.0), ('P2', 3.0)]:
    for _name, _A, _k in [('Fwt', 400.0, 0.04), ('Mut', 300.0, 0.10)]:
        for _t in _times17:
            _mu = _gain * (_A * np.exp(-_k * _t) + 15)
            for _v in _rng17.lognormal(np.log(_mu), 0.08, 300):
                _rows17.append({'sample': _name, 'time': float(_t), 'RL1-H': float(_v),
                                'well': f'{_plate}{_name}{_t}', 'dataset': _plate})
    for _t in _times17:
        for _v in _rng17.exponential(5 * _gain, 150):
            _rows17.append({'sample': 'Ctrl', 'time': float(_t), 'RL1-H': float(_v),
                            'well': f'{_plate}C{_t}', 'dataset': _plate})
e17 = tf.FlowExperiment()
e17.populations['raw'] = pd.DataFrame(_rows17); e17.active_pop = 'raw'
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    e17.run_pri_analysis('RL1-H', control_sample='Ctrl', reference_sample='Fwt',
                         wt_sample='Fwt', per_plate=True)
    bad_sqrt = any('invalid value encountered in sqrt' in str(x.message) for x in w)
    cov_failed = any('Covariance estimation failed' in str(x.message) for x in w)
check("no 'invalid value encountered in sqrt' warning", not bad_sqrt)
check("covariance estimation did not fall back to NaN", not cov_failed)
for _col in ('A_err', 'k_err', 'C_err', 't_half_err'):
    _vals = e17.pri_fits_norm[_col].values
    _finite = _vals[np.isfinite(_vals)]
    check(f"{_col}: all finite values >= 0", np.all(_finite >= 0),
          f"min={_finite.min() if _finite.size else 'n/a'}")

# ── 18. absolute_sigma — parameter error scales with input SE ──────────────────
print("\n[18] absolute_sigma covariance scaling")
_ex = tf.FlowExperiment()
_t = np.array([0, 1, 3, 5, 10, 20, 30], float)
_y = 1000.0 * np.exp(-0.05 * _t) + 30.0  # exact exponential → residuals ~ 0


def _fit_kerr(se_val):
    _df = pd.DataFrame([dict(sample='S', time=t, PRI_abs=yv, PRI_abs_se=se_val)
                        for t, yv in zip(_t, _y)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _f = _ex._fit_global_exponential(_df, 'PRI_abs')
    return float(_f['k_err'].iloc[0])


_ke1, _ke10 = _fit_kerr(20.0), _fit_kerr(200.0)
check("k_err reflects input SE (not ~0 from mse rescale)", _ke1 > 1e-6,
      f"k_err={_ke1:.3e}")
check("k_err scales ~linearly with SE (absolute_sigma)",
      0.7 < (_ke10 / _ke1) / 10.0 < 1.4 if _ke1 else False,
      f"ratio={_ke10/_ke1 if _ke1 else float('nan'):.3f} (want ~10)")

# ── 19. Weight clipping — one tiny-SE point cannot dominate the fit ────────────
print("\n[19] Fit weights are bounded")
# One off-curve time-point (t=5, value tripled) carrying an absurdly small SE.
# Uncapped, its ~1e8x weight dominates and drags the shared-C fit's k badly off.
_exc = tf.FlowExperiment()
_yo = _y.copy(); _yo[3] = _y[3] * 3.0
_seo = np.full_like(_t, 20.0); _seo[3] = 1e-6
_dfo = pd.DataFrame([dict(sample='S', time=t, PRI_abs=yv, PRI_abs_se=s)
                     for t, yv, s in zip(_t, _yo, _seo)])


def _fit_k_with_cap(cap):
    _saved = tf._WEIGHT_CAP_FACTOR
    tf._WEIGHT_CAP_FACTOR = cap
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(_exc._fit_global_exponential(_dfo, 'PRI_abs')['k'].iloc[0])
    finally:
        tf._WEIGHT_CAP_FACTOR = _saved


_k_unclipped = _fit_k_with_cap(np.inf)
_k_clipped = _fit_k_with_cap(tf._WEIGHT_CAP_FACTOR)  # default cap
check("weight cap reduces the outlier's bias on k",
      abs(_k_clipped - 0.05) < abs(_k_unclipped - 0.05),
      f"clipped={_k_clipped:.4f} vs unclipped={_k_unclipped:.4f} (true 0.05)")
check("clipped k within 40% of truth", abs(_k_clipped - 0.05) / 0.05 < 0.40,
      f"k={_k_clipped:.4f}")

# ── 20. Self-normalized PRI_norm baseline == 1 (matches reference mode) ────────
print("\n[20] Self-normalized PRI_norm baseline")
e20 = make_experiment(n_events=200)
e20.run_pri_analysis('RL1-H', control_sample='Ctrl', n_bootstrap=0)  # no reference
_t0 = e20.pri_table[e20.pri_table['time'] == 0.0]
_pn0 = _t0['PRI_norm'].dropna().values
check("self-norm: every sample's PRI_norm(t0) ~ 1.0",
      _pn0.size > 0 and np.allclose(_pn0, 1.0, atol=1e-9),
      f"values={_pn0}")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
