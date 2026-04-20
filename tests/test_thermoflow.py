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

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
