"""
ThermoFlow — Streamlit GUI
Full analysis workflow: load FCS data → gate → PRI analysis → visualise → export.

Run with:
    streamlit run gui/streamlit_app.py
"""
import io
import json
import os
import sys
import tempfile
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import streamlit as st

# ── resolve thermoflow_app from parent directory ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from thermoflow_app import (
    EllipseGate,
    FlowExperiment,
    FlowReport,
    GateSet,
    PolygonGate,
    RectangleGate,
    ThresholdGate,
    __version__,
    points_to_density_image,
)

# ═════════════════════════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ThermoFlow",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# Session state
# ═════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "exp": None,
    "tmpdir": None,
    "annotation_df": pd.DataFrame(columns=["well", "sample", "time"]),
    "last_fig": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _patch_exp(exp: FlowExperiment) -> FlowExperiment:
    """Override _show_static_fig so plots are captured as PNG bytes rather than
    sent to IPython display."""
    def _show(self, fig, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        st.session_state["last_fig"] = buf.getvalue()

    exp._show_static_fig = types.MethodType(_show, exp)
    return exp


def get_exp() -> FlowExperiment:
    if st.session_state["exp"] is None:
        st.session_state["exp"] = _patch_exp(FlowExperiment())
    return st.session_state["exp"]


def _render(plot_fn, *args, **kwargs) -> None:
    """Call a plot method and display the captured PNG with a download button."""
    st.session_state["last_fig"] = None
    try:
        plot_fn(*args, **kwargs)
    except Exception as exc:
        st.error(f"Plot error: {exc}")
        return
    if st.session_state.get("last_fig"):
        st.image(st.session_state["last_fig"], use_container_width=True)
        st.download_button(
            "⬇️ Download PNG",
            data=st.session_state["last_fig"],
            file_name="thermoflow_plot.png",
            mime="image/png",
        )


def _plotly_density(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None = None,
    gate=None,
    sample_filter: str = "All",
    time_filter=None,
    title: str = "",
) -> go.Figure:
    """Interactive Plotly density / histogram used in the gating tab."""
    sub = df.copy()
    if sample_filter != "All" and "sample" in sub.columns:
        sub = sub[sub["sample"] == sample_filter]
    if time_filter is not None and time_filter != "All" and "time" in sub.columns:
        sub = sub[sub["time"] == time_filter]

    if sub.empty or x_col not in sub.columns:
        return go.Figure().update_layout(title="No data for selection")

    x = np.log1p(sub[x_col].clip(lower=0)).values

    if y_col and y_col in sub.columns:
        y = np.log1p(sub[y_col].clip(lower=0)).values
        density, xe, ye = points_to_density_image(x, y, bins=200)
        smoothed = gaussian_filter(np.nan_to_num(density, nan=0), sigma=1.2)
        smoothed = np.where(density == 0, np.nan, smoothed)
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2

        fig = go.Figure(go.Heatmap(
            z=smoothed, x=xc, y=yc,
            colorscale="Viridis",
            colorbar=dict(title="Density", thickness=14, len=0.8),
            hoverongaps=False,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>density=%{z:.0f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=f"{x_col} (log1p)",
            yaxis_title=f"{y_col} (log1p)",
        )

        # Gate overlays
        if gate is not None:
            if isinstance(gate, RectangleGate):
                fig.add_shape(
                    type="rect",
                    x0=gate.xmin, x1=gate.xmax,
                    y0=gate.ymin, y1=gate.ymax,
                    line=dict(color="#FF5733", width=2, dash="dash"),
                )
            elif isinstance(gate, EllipseGate):
                theta = np.linspace(0, 2 * np.pi, 120)
                rad = np.radians(gate.angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                raw_x = (gate.width / 2) * np.cos(theta)
                raw_y = (gate.height / 2) * np.sin(theta)
                ex = gate.center[0] + raw_x * cos_a - raw_y * sin_a
                ey = gate.center[1] + raw_x * sin_a + raw_y * cos_a
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines",
                    line=dict(color="#FF5733", width=2, dash="dash"),
                    showlegend=False,
                ))
    else:
        # 1-D histogram
        fig = go.Figure(go.Histogram(
            x=x, nbinsx=200,
            marker_color="#4DBBD5", opacity=0.85,
            hovertemplate="log1p=%{x:.3f}<br>count=%{y}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=f"{x_col} (log1p)",
            yaxis_title="Count",
            bargap=0,
        )
        if gate is not None and isinstance(gate, ThresholdGate):
            v = gate.value if gate.value is not None else gate.lo
            if v is not None:
                fig.add_vline(
                    x=float(v), line_color="#FF5733",
                    line_dash="dash", line_width=2,
                    annotation_text=f"  thr = {float(v):.3f}",
                    annotation_position="top right",
                )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=440,
        margin=dict(l=60, r=20, t=50, b=55),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🌡️ ThermoFlow")
    st.caption(f"v{__version__}")
    st.divider()

    exp = get_exp()
    if exp.populations:
        st.success(f"**{sum(len(v) for v in exp.populations.values()):,}** events loaded")
        for p, df in exp.populations.items():
            marker = "◀ " if p == exp.active_pop else "   "
            st.caption(f"{marker}`{p}`: {len(df):,}")

        if not exp.pri_fits_norm.empty:
            st.divider()
            st.success("✅ PRI analysis complete")
            good = (exp.pri_fits_norm["fit_quality"] == "good").sum()
            st.caption(f"{good}/{len(exp.pri_fits_norm)} good fits")
    else:
        st.info("No data loaded yet.")

    st.divider()
    if st.button("🗑️ Reset experiment", use_container_width=True):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Main tabs
# ═════════════════════════════════════════════════════════════════════════════
(
    tab_load,
    tab_gate,
    tab_pri,
    tab_viz,
    tab_export,
) = st.tabs(["📁 Load Data", "🔲 Gate", "📊 PRI Analysis", "📈 Visualize", "💾 Export"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
with tab_load:
    st.header("Load FCS Data")

    col_fcs, col_ann = st.columns(2, gap="large")

    # ── FCS upload ──────────────────────────────────────────────────────────
    with col_fcs:
        st.subheader("FCS Files")
        fcs_files = st.file_uploader(
            "Upload FCS files",
            type=["fcs"],
            accept_multiple_files=True,
            help="Upload one or more FCS files. They will be matched to wells via the annotation table.",
        )
        dataset_id_input = st.text_input(
            "Dataset ID (optional)",
            placeholder="e.g. Plate_1",
            help="Optional tag appended to every event row (useful for multi-plate experiments).",
        )

    # ── Annotation ──────────────────────────────────────────────────────────
    with col_ann:
        st.subheader("Sample Annotation")
        st.caption(
            "Upload a CSV/Excel **or** fill in the table below.  \n"
            "Required columns: `well`, `sample`, `time`."
        )

        ann_file = st.file_uploader(
            "Upload annotation file",
            type=["csv", "xlsx", "xls"],
            key="ann_upload",
        )
        if ann_file:
            try:
                ann_read = (
                    pd.read_excel(ann_file)
                    if ann_file.name.endswith((".xlsx", ".xls"))
                    else pd.read_csv(ann_file)
                )
                ann_read.columns = [c.strip().lower() for c in ann_read.columns]
                missing = [c for c in ("well", "sample", "time") if c not in ann_read.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    st.session_state["annotation_df"] = ann_read
                    st.success(f"Loaded {len(ann_read)} annotation rows.")
            except Exception as exc:
                st.error(f"Could not read annotation file: {exc}")

        edited_ann = st.data_editor(
            st.session_state["annotation_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="ann_editor",
            column_config={
                "well": st.column_config.TextColumn("Well ID", help="Must match part of the FCS filename"),
                "sample": st.column_config.TextColumn("Sample Name"),
                "time": st.column_config.NumberColumn("Time (min)", format="%.1f"),
            },
        )
        st.session_state["annotation_df"] = edited_ann

    st.divider()

    can_load = bool(fcs_files) and not st.session_state["annotation_df"].dropna(how="all").empty

    if st.button("⬆️ Load into experiment", type="primary", disabled=not can_load,
                  use_container_width=True):
        ann = st.session_state["annotation_df"].dropna(how="all").copy()
        missing_cols = [c for c in ("well", "sample", "time") if c not in ann.columns]
        if missing_cols:
            st.error(f"Annotation missing columns: {missing_cols}")
        else:
            # Build setup dict
            setup_dict = {}
            for _, row in ann.iterrows():
                entry = {"sample": str(row["sample"]), "time": float(row["time"])}
                for col in ann.columns:
                    if col not in ("well", "sample", "time"):
                        entry[col] = row[col]
                setup_dict[str(row["well"])] = entry

            # Write FCS files to temp dir
            tmpdir = st.session_state["tmpdir"]
            if tmpdir is None or not os.path.isdir(tmpdir):
                tmpdir = tempfile.mkdtemp(prefix="thermoflow_")
                st.session_state["tmpdir"] = tmpdir

            for uf in fcs_files:
                dest = os.path.join(tmpdir, uf.name)
                with open(dest, "wb") as fh:
                    fh.write(uf.getbuffer())

            exp = get_exp()
            with st.spinner("Loading FCS files…"):
                try:
                    did = dataset_id_input.strip() or None
                    report = exp.load_fcs_files(
                        os.path.join(tmpdir, "*.fcs"),
                        data_setup_dict=setup_dict,
                        dataset_id=did,
                    )
                    n_raw = len(exp.populations.get("raw", []))
                    st.success(f"✅ Loaded {n_raw:,} events into 'raw'.")
                    if report.get("errors"):
                        st.warning(f"File errors: {list(report['errors'].keys())}")
                    if report.get("unmatched_files"):
                        st.warning(f"Unmatched files: {report['unmatched_files']}")
                    if report.get("unmatched_wells"):
                        st.warning(f"Unmatched wells: {report['unmatched_wells']}")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")
            st.rerun()

    # ── Preview ──────────────────────────────────────────────────────────────
    exp = get_exp()
    if "raw" in exp.populations and not exp.populations["raw"].empty:
        st.divider()
        raw = exp.populations["raw"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total events", f"{len(raw):,}")
        m2.metric("Samples", raw["sample"].nunique() if "sample" in raw.columns else "—")
        m3.metric("Time points", raw["time"].nunique() if "time" in raw.columns else "—")
        m4.metric("Channels", len(exp.channels))

        with st.expander("Raw data preview (first 500 rows)"):
            st.dataframe(raw.head(500), use_container_width=True, height=220)

        # Rename sample utility
        st.subheader("Rename Sample")
        samples_avail = sorted(raw["sample"].dropna().unique()) if "sample" in raw.columns else []
        rn1, rn2, rn3 = st.columns([2, 2, 1])
        old_name = rn1.selectbox("Current name", samples_avail, key="rn_old")
        new_name = rn2.text_input("New name", key="rn_new", placeholder="New sample name")
        if rn3.button("Rename", use_container_width=True) and new_name.strip():
            exp.rename_sample(old_name, new_name.strip())
            st.rerun()

        # Annotation template download
        st.divider()
        template_df = pd.DataFrame([
            {"well": "A01", "sample": "Ctrl", "time": 0.0},
            {"well": "A02", "sample": "Ctrl", "time": 5.0},
            {"well": "B01", "sample": "SampleA", "time": 0.0},
            {"well": "B02", "sample": "SampleA", "time": 5.0},
        ])
        st.download_button(
            "⬇️ Download annotation template (CSV)",
            data=template_df.to_csv(index=False).encode(),
            file_name="annotation_template.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — GATE
# ═════════════════════════════════════════════════════════════════════════════
with tab_gate:
    st.header("Interactive Gating")
    exp = get_exp()

    if not exp.populations:
        st.info("Load data first in the **Load Data** tab.")
    else:
        pops = list(exp.populations.keys())
        chans = exp.channels

        g_ctrl, g_plot = st.columns([1, 2], gap="large")

        with g_ctrl:
            st.subheader("Gate Controls")

            parent_pop = st.selectbox("Parent population", pops, key="g_parent")
            pop_df = exp.get_data(parent_pop)

            all_samples = ["All"] + sorted(pop_df["sample"].dropna().unique().tolist()) \
                if "sample" in pop_df.columns else ["All"]
            all_times = ["All"] + [str(t) for t in sorted(pop_df["time"].dropna().unique())] \
                if "time" in pop_df.columns else ["All"]

            g_sample = st.selectbox("Filter by sample", all_samples, key="g_sample")
            g_time = st.selectbox("Filter by time", all_times, key="g_time")

            x_col = st.selectbox("X channel", chans, key="g_x")
            y_opts = ["(histogram — 1D)"] + chans
            y_sel = st.selectbox(
                "Y channel", y_opts,
                index=min(2, len(y_opts) - 1),
                key="g_y",
            )
            y_col = None if y_sel == "(histogram — 1D)" else y_sel

            gate_types = ["Rectangle", "Ellipse"] if y_col else ["Threshold"]
            if y_col:
                gate_types.append("Threshold")
            gate_type = st.radio("Gate type", gate_types, horizontal=True, key="g_type")

            st.divider()

            # Compute data range for sensible defaults
            sub_df = pop_df.copy()
            if g_sample != "All":
                sub_df = sub_df[sub_df["sample"] == g_sample]
            g_time_val = None if g_time == "All" else float(g_time)
            if g_time_val is not None:
                sub_df = sub_df[sub_df["time"] == g_time_val]

            gate_obj = None

            if gate_type == "Rectangle" and y_col:
                st.markdown("**Bounds (log1p scale)** — hover over the density plot to read coordinates.")
                if not sub_df.empty:
                    xd = np.log1p(sub_df[x_col].clip(lower=0))
                    yd = np.log1p(sub_df[y_col].clip(lower=0))
                    xlo, xhi = float(xd.min()), float(xd.max())
                    ylo, yhi = float(yd.min()), float(yd.max())
                else:
                    xlo, xhi, ylo, yhi = 0.0, 10.0, 0.0, 10.0

                c1, c2 = st.columns(2)
                xmin_v = c1.number_input("X min", value=round((xlo + xhi) * 0.25, 3),
                                          step=0.05, format="%.3f", key="r_xmin")
                xmax_v = c2.number_input("X max", value=round(xhi * 0.95, 3),
                                          step=0.05, format="%.3f", key="r_xmax")
                ymin_v = c1.number_input("Y min", value=round((ylo + yhi) * 0.25, 3),
                                          step=0.05, format="%.3f", key="r_ymin")
                ymax_v = c2.number_input("Y max", value=round(yhi * 0.95, 3),
                                          step=0.05, format="%.3f", key="r_ymax")

                if xmin_v < xmax_v and ymin_v < ymax_v:
                    gate_obj = RectangleGate(
                        "rect", x=x_col, y=y_col,
                        xmin=xmin_v, xmax=xmax_v,
                        ymin=ymin_v, ymax=ymax_v, log1p=True,
                    )

            elif gate_type == "Ellipse" and y_col:
                st.markdown("**Ellipse parameters (log1p scale)**")
                if not sub_df.empty:
                    xd = np.log1p(sub_df[x_col].clip(lower=0))
                    yd = np.log1p(sub_df[y_col].clip(lower=0))
                    cx_def = float(xd.median())
                    cy_def = float(yd.median())
                    w_def = float((xd.max() - xd.min()) * 0.4)
                    h_def = float((yd.max() - yd.min()) * 0.4)
                else:
                    cx_def, cy_def, w_def, h_def = 5.0, 5.0, 2.0, 2.0

                c1, c2 = st.columns(2)
                ec_x = c1.number_input("Center X", value=round(cx_def, 3), step=0.05,
                                        format="%.3f", key="e_cx")
                ec_y = c2.number_input("Center Y", value=round(cy_def, 3), step=0.05,
                                        format="%.3f", key="e_cy")
                ew = c1.number_input("Width", value=round(w_def, 3), step=0.05,
                                      min_value=0.01, format="%.3f", key="e_w")
                eh = c2.number_input("Height", value=round(h_def, 3), step=0.05,
                                      min_value=0.01, format="%.3f", key="e_h")
                ea = st.slider("Rotation angle (°)", -90.0, 90.0, 0.0, 1.0, key="e_a")

                gate_obj = EllipseGate(
                    "ellipse", x=x_col, y=y_col,
                    center=(ec_x, ec_y), width=ew, height=eh,
                    angle=ea, log1p=True,
                )

            elif gate_type == "Threshold":
                st.markdown("**Threshold (log1p scale)**")
                thr_col = x_col  # always X in 1-D; works for 2D too
                if not sub_df.empty and thr_col in sub_df.columns:
                    xd = np.log1p(sub_df[thr_col].clip(lower=0))
                    xlo, xhi = float(xd.min()), float(xd.max())
                    xmid = float(xd.median())
                else:
                    xlo, xhi, xmid = 0.0, 10.0, 5.0

                thr_val = st.slider(
                    "Threshold value", min_value=round(xlo, 3), max_value=round(xhi, 3),
                    value=round(xmid, 3), step=0.01, key="thr_v",
                )
                thr_dir = st.radio(
                    "Keep events", ["≥ threshold (positive)", "< threshold (negative)"],
                    horizontal=True, key="thr_dir",
                )
                op = ">=" if "≥" in thr_dir else "<"
                gate_obj = ThresholdGate(
                    "threshold", column=thr_col, op=op, value=thr_val, log1p=True,
                )

            # Live stats
            if gate_obj is not None and not pop_df.empty:
                try:
                    mask = gate_obj.evaluate(pop_df)
                    n_in = int(mask.sum())
                    n_tot = len(pop_df)
                    pct = 100 * n_in / n_tot if n_tot else 0
                    st.metric(
                        "Events inside gate",
                        f"{n_in:,}",
                        delta=f"{pct:.1f}% of {n_tot:,} total",
                    )
                except Exception:
                    pass

            st.divider()
            new_pop_name = st.text_input("New population name", value="cells", key="g_newpop")

            apply_col, _ = st.columns([1, 1])
            if apply_col.button(
                "✅ Apply Gate", type="primary",
                disabled=gate_obj is None or not new_pop_name.strip(),
                use_container_width=True,
            ):
                try:
                    gs_name = f"ManualGate_{new_pop_name.strip()}"
                    gs = GateSet(name=gs_name, gates=[gate_obj])
                    exp.apply_gateset(gs, parent_pop=parent_pop,
                                      new_pop_name=new_pop_name.strip())
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gate error: {exc}")

            # Gate save / load
            st.divider()
            st.subheader("Gate Persistence")

            if exp.gatesets:
                gates_json_str = json.dumps(
                    {"gatesets": {k: v.to_dict() for k, v in exp.gatesets.items()}},
                    indent=2, default=str,
                )
                st.download_button(
                    "💾 Save gates (JSON)",
                    data=gates_json_str.encode(),
                    file_name="gates.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander(f"Saved gatesets ({len(exp.gatesets)})"):
                    for gname, gs in exp.gatesets.items():
                        st.caption(f"**{gname}** — {len(gs.gates)} gate(s)")

            gates_upload = st.file_uploader("Load gates JSON", type=["json"],
                                             key="gates_up")
            if gates_upload:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    tf.write(gates_upload.read())
                    tpath = tf.name
                try:
                    exp.load_gates(tpath)
                    st.success(f"Loaded {len(exp.gatesets)} gatesets.")
                    st.rerun()
                finally:
                    os.unlink(tpath)

        with g_plot:
            st.subheader("Preview")
            st.caption(
                "💡 Hover over the plot to read log1p coordinates, then type them into the "
                "gate controls on the left."
            )
            fig_d = _plotly_density(
                pop_df,
                x_col=x_col,
                y_col=y_col,
                gate=gate_obj,
                sample_filter=g_sample,
                time_filter=g_time_val,
                title=f"[{parent_pop}] "
                      + (f"{x_col} vs {y_col}" if y_col else f"{x_col}"),
            )
            st.plotly_chart(fig_d, use_container_width=True)

            # Population summary table
            if len(exp.populations) > 1:
                st.subheader("All Populations")
                pop_rows = []
                raw_n = len(exp.populations.get("raw", []))
                for pname, pdata in exp.populations.items():
                    pct = f"{100 * len(pdata) / raw_n:.1f}%" if raw_n else "—"
                    pop_rows.append({
                        "Population": pname,
                        "Events": f"{len(pdata):,}",
                        "% of raw": pct,
                        "Active": "◀" if pname == exp.active_pop else "",
                    })
                st.dataframe(pd.DataFrame(pop_rows), use_container_width=True,
                             hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRI ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pri:
    st.header("PRI Analysis")
    exp = get_exp()

    if not exp.populations:
        st.info("Load data first.")
    else:
        pops = list(exp.populations.keys())
        chans = exp.channels

        p_left, p_right = st.columns(2, gap="large")

        with p_left:
            st.subheader("Parameters")

            pri_pop = st.selectbox("Population", pops, key="pri_pop")
            pri_df = exp.get_data(pri_pop)
            samples_in_pop = (
                sorted(pri_df["sample"].dropna().unique().tolist())
                if "sample" in pri_df.columns else []
            )

            pri_channel = st.selectbox("Fluorescence channel", chans, key="pri_ch")
            ctrl_sample = st.selectbox("Control sample", samples_in_pop, key="pri_ctrl")

            ref_opts = ["(none)"] + samples_in_pop
            ref_sample = st.selectbox(
                "Reference sample for cross-normalisation (optional)",
                ref_opts, key="pri_ref",
            )
            ref_sample = None if ref_sample == "(none)" else ref_sample

            with st.expander("Advanced"):
                pos_frac = st.slider(
                    "pos_frac — top fraction counted as positive",
                    0.001, 0.5, 0.01, 0.001, format="%.3f", key="pri_pf",
                )
                n_boot = st.slider("Bootstrap iterations", 0, 500, 100, 10, key="pri_boot")
                baseline_time = st.number_input("Baseline time (min)", value=0, step=1,
                                                 key="pri_bt")
                use_thr = st.checkbox("Override threshold (log1p space)", key="pri_thr_cb")
                custom_thr = None
                if use_thr:
                    custom_thr = st.number_input(
                        "threshold_log", value=5.0, step=0.05, format="%.4f",
                        help="Fixed positive/negative gate boundary in log1p space.",
                        key="pri_thr_val",
                    )

        with p_right:
            st.subheader("Threshold Preview")
            if pri_channel in pri_df.columns and not pri_df.empty:
                ctrl_rows = pri_df[pri_df["sample"] == ctrl_sample] if ctrl_sample else pri_df
                if not ctrl_rows.empty:
                    xv = np.log1p(ctrl_rows[pri_channel].clip(lower=0))
                    auto_thr = (
                        custom_thr
                        if (use_thr and custom_thr is not None)
                        else float(np.quantile(xv, np.clip(1.0 - pos_frac, 0, 1)))
                    )
                    n_pos = int((xv >= auto_thr).sum())
                    n_total = len(xv)

                    fig_thr = go.Figure()
                    fig_thr.add_trace(go.Histogram(
                        x=xv, nbinsx=150,
                        marker_color="#4DBBD5", opacity=0.85, name=ctrl_sample,
                        hovertemplate="log1p=%{x:.3f}<br>count=%{y}<extra></extra>",
                    ))
                    fig_thr.add_vline(
                        x=auto_thr, line_color="#E64B35",
                        line_dash="dash", line_width=2,
                        annotation_text=f"  thr={auto_thr:.3f}  ({100*n_pos/n_total:.1f}% positive)",
                        annotation_position="top right",
                        annotation_font_color="#E64B35",
                    )
                    fig_thr.update_layout(
                        xaxis_title=f"{pri_channel} (log1p)",
                        yaxis_title="Count",
                        height=280, showlegend=False, bargap=0,
                        margin=dict(l=50, r=20, t=30, b=45),
                    )
                    st.plotly_chart(fig_thr, use_container_width=True)
                    st.caption(
                        f"Control: **{ctrl_sample}** — {n_pos:,} / {n_total:,} events "
                        f"({100*n_pos/n_total:.1f}%) above threshold"
                    )

        if st.button("▶️ Run PRI Analysis", type="primary", use_container_width=True):
            with st.spinner("Running PRI analysis with bootstrap…"):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
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
                    for w in caught:
                        st.warning(str(w.message))
                    st.success("✅ PRI analysis complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"PRI analysis failed: {exc}")

        if not exp.pri_fits_norm.empty:
            st.divider()
            st.subheader("Fit Results")

            r_norm, r_abs = st.tabs(["Normalised fits", "Absolute fits"])

            def _fmt(df):
                float_cols = df.select_dtypes(float).columns
                return df.style.format({c: "{:.4f}" for c in float_cols})

            with r_norm:
                st.dataframe(_fmt(exp.pri_fits_norm), use_container_width=True)
            with r_abs:
                st.dataframe(_fmt(exp.pri_fits_abs), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — VISUALIZE
# ═════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.header("Visualize")
    exp = get_exp()

    if not exp.populations:
        st.info("Load data first.")
    else:
        pops = list(exp.populations.keys())
        chans = exp.channels

        v1, v2, v3, v4, v5 = st.tabs([
            "PRI Decay Curves",
            "PRI Bar Chart",
            "Summary Grid",
            "Density Plot",
            "Histogram",
        ])

        # ── PRI decay curves ────────────────────────────────────────────────
        with v1:
            if exp.pri_table.empty:
                st.info("Run PRI analysis first.")
            else:
                a, b = st.columns(2)
                which_v = a.radio("Metric", ["PRI_norm", "PRI_abs"],
                                   horizontal=True, key="v1_which")
                cols_v = b.slider("Columns", 1, 6, 3, key="v1_cols")
                ci_v = a.checkbox("Show 95% CI band", key="v1_ci")
                params_v = b.multiselect(
                    "Show parameters",
                    ["t_half", "y0", "r2"],
                    default=["t_half", "r2"],
                    key="v1_params",
                )
                spine_v = a.slider("Spine width", 0.3, 2.0, 0.8, 0.1, key="v1_spine")
                res_v = b.checkbox("Show residuals panel", key="v1_res")

                if st.button("Plot decay curves", key="btn_v1", use_container_width=True):
                    _render(
                        exp.plot_pri,
                        which=which_v, cols=cols_v,
                        show_ci=ci_v, show_params=params_v,
                        spine_width=spine_v, plot_residuals=res_v,
                    )

        # ── PRI bar chart ───────────────────────────────────────────────────
        with v2:
            if exp.pri_fits_norm.empty:
                st.info("Run PRI analysis first.")
            else:
                samples_list = sorted(
                    exp.pri_fits_norm["sample"].dropna().unique().tolist()
                )
                a, b = st.columns(2)
                which_b = a.radio("Metric", ["t_half", "A"],
                                   horizontal=True, key="v2_which")
                use_norm_b = a.checkbox("Use normalised fits", value=True, key="v2_norm")
                norm_opts = ["(none)"] + samples_list
                norm_samp = b.selectbox("Normalise bars to sample",
                                         norm_opts, key="v2_normsamp")
                norm_samp = None if norm_samp == "(none)" else norm_samp
                cpal = b.selectbox(
                    "Colour palette",
                    ["viridis", "plasma", "cividis", "magma", "Blues", "RdBu"],
                    key="v2_cpal",
                )
                filter_samps = st.multiselect(
                    "Restrict to samples (leave empty = all)",
                    samples_list, key="v2_filter",
                )

                if st.button("Plot bar chart", key="btn_v2", use_container_width=True):
                    _render(
                        exp.plot_pri_bars,
                        which=which_b,
                        use_norm=use_norm_b,
                        norm_sample=norm_samp,
                        color_palette=cpal,
                        samples=filter_samps if filter_samps else None,
                    )

        # ── Summary grid ────────────────────────────────────────────────────
        with v3:
            if exp.pri_table.empty:
                st.info("Run PRI analysis first.")
            else:
                a, b = st.columns(2)
                sg_which = a.radio("Metric", ["PRI_norm", "PRI_abs"],
                                    horizontal=True, key="v3_which")
                sg_cols = b.slider("Columns", 2, 6, 4, key="v3_cols")
                sg_ci = a.checkbox("Show 95% CI band", key="v3_ci")
                sg_params = b.multiselect(
                    "Show parameters",
                    ["t_half", "y0", "r2"],
                    default=["t_half", "r2"],
                    key="v3_params",
                )

                if st.button("Plot summary grid", key="btn_v3", use_container_width=True):
                    _render(
                        exp.plot_pri_summary_grid,
                        which=sg_which, cols=sg_cols,
                        show_ci=sg_ci, show_params=sg_params,
                    )

        # ── Density plot ────────────────────────────────────────────────────
        with v4:
            a, b = st.columns(2)
            dp_pop = a.selectbox("Population", pops, key="v4_pop")
            dp_x = a.selectbox("X channel", chans, key="v4_x")
            dp_y = b.selectbox(
                "Y channel",
                chans, index=min(1, len(chans) - 1), key="v4_y",
            )
            dp_df = exp.get_data(dp_pop)
            dp_samples = ["All"] + (
                sorted(dp_df["sample"].dropna().unique().tolist())
                if "sample" in dp_df.columns else []
            )
            dp_filter_samp = b.selectbox("Filter sample", dp_samples, key="v4_samp")

            with st.expander("More options"):
                e1, e2 = st.columns(2)
                dp_slice = e1.checkbox("Facet by sample", key="v4_slice")
                dp_contours = e1.checkbox("Show contours", key="v4_cnt")
                dp_stats = e2.checkbox("Gate stats overlay", value=True, key="v4_stats")
                gate_keys = list(exp.gatesets.keys())
                dp_overlay = e2.multiselect("Overlay gatesets", gate_keys, key="v4_ov")
                dp_sub_samp = None if dp_filter_samp == "All" else dp_filter_samp

            if st.button("Plot density", key="btn_v4", use_container_width=True):
                overlay_gs = [exp.gatesets[k] for k in dp_overlay]
                _render(
                    exp.plot_density,
                    x_col=dp_x, y_col=dp_y,
                    pop_name=dp_pop,
                    subset_sample=dp_sub_samp,
                    slice_by="sample" if dp_slice else None,
                    show_contours=dp_contours,
                    gates_to_overlay=overlay_gs or None,
                    show_stats=dp_stats,
                )

        # ── Histogram ───────────────────────────────────────────────────────
        with v5:
            a, b = st.columns(2)
            h_pop = a.selectbox("Population", pops, key="v5_pop")
            h_col = a.selectbox("Channel", chans, key="v5_col")
            h_slice = b.checkbox("Facet by sample", key="v5_facet")
            h_overlay = b.checkbox("Overlay all samples", key="v5_ov")
            h_horiz = b.checkbox("Horizontal layout", key="v5_h")
            h_vline = b.number_input(
                "Threshold line (log1p, 0 = off)",
                value=0.0, step=0.05, format="%.3f", key="v5_vl",
            )

            if st.button("Plot histogram", key="btn_v5", use_container_width=True):
                _render(
                    exp.plot_sliced_histogram,
                    col=h_col,
                    pop_name=h_pop,
                    slice_by="sample" if h_slice else "time",
                    overlay=h_overlay,
                    orientation="horizontal" if h_horiz else "vertical",
                    vline=h_vline if h_vline != 0.0 else None,
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("Export")
    exp = get_exp()

    e_left, e_right = st.columns(2, gap="large")

    with e_left:
        st.subheader("PRI Data")
        if not exp.pri_table.empty:
            st.download_button(
                "⬇️ PRI event table (CSV)",
                data=exp.pri_table.to_csv(index=False).encode(),
                file_name="pri_table.csv", mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Normalised fits (CSV)",
                data=exp.pri_fits_norm.to_csv(index=False).encode(),
                file_name="pri_fits_norm.csv", mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Absolute fits (CSV)",
                data=exp.pri_fits_abs.to_csv(index=False).encode(),
                file_name="pri_fits_abs.csv", mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("No PRI data yet — run analysis first.")

        st.divider()
        st.subheader("Gates")
        if exp.gatesets:
            gates_bytes = json.dumps(
                {"gatesets": {k: v.to_dict() for k, v in exp.gatesets.items()}},
                indent=2, default=str,
            ).encode()
            st.download_button(
                "⬇️ Gates (JSON)",
                data=gates_bytes,
                file_name="gates.json", mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("No gates drawn yet.")

        st.divider()
        st.subheader("Annotation Template")
        template = pd.DataFrame([
            {"well": "A01", "sample": "Ctrl", "time": 0.0},
            {"well": "A02", "sample": "Ctrl", "time": 5.0},
            {"well": "B01", "sample": "SampleA", "time": 0.0},
            {"well": "B02", "sample": "SampleA", "time": 5.0},
        ])
        st.download_button(
            "⬇️ Annotation template (CSV)",
            data=template.to_csv(index=False).encode(),
            file_name="annotation_template.csv", mime="text/csv",
            use_container_width=True,
        )

    with e_right:
        st.subheader("Full Report")
        if exp.pri_table.empty:
            st.info("Run PRI analysis before generating a report.")
        else:
            fmt = st.radio("Format", ["HTML", "PDF"], horizontal=True, key="rpt_fmt")

            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report…"):
                    try:
                        rpt = FlowReport(exp)
                        suffix = ".html" if fmt == "HTML" else ".pdf"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                            rpath = tf.name
                        if fmt == "HTML":
                            rpt.export_html(rpath)
                        else:
                            rpt.export_pdf(rpath)
                        with open(rpath, "rb") as fh:
                            rpt_bytes = fh.read()
                        os.unlink(rpath)
                        mime = "text/html" if fmt == "HTML" else "application/pdf"
                        fname = f"thermoflow_report{suffix}"
                        st.download_button(
                            f"⬇️ Download {fmt} report",
                            data=rpt_bytes, file_name=fname, mime=mime,
                            use_container_width=True,
                        )
                        st.success("Report ready.")
                    except Exception as exc:
                        st.error(f"Report generation failed: {exc}")

        st.divider()
        st.subheader("Experiment Summary")
        if exp.populations:
            rpt = FlowReport(exp)
            st.text(rpt.summary())
        else:
            st.caption("No data loaded.")
