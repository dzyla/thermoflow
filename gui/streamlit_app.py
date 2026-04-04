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
_DEFAULTS: dict = {
    "exp": None,
    "tmpdir": None,
    "annotation_df": pd.DataFrame(columns=["well", "sample", "time"]),
    "last_fig": None,
    # gating
    "g_pending": None,       # {gate, x_col, y_col, plot_id, n_in, n_tot}
    "g_plot_ver": 0,         # incremented after Apply to reset chart selections
    "g_plot_cfgs": [],       # [{x, y}, …] one per plot slot
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — experiment
# ═════════════════════════════════════════════════════════════════════════════
def _patch_exp(exp: FlowExperiment) -> FlowExperiment:
    """Redirect _show_static_fig so matplotlib figures are captured as PNG bytes."""
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
    """Call any FlowExperiment plot method and display the captured PNG."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — gating plots
# ═════════════════════════════════════════════════════════════════════════════
def _density_colors(x: np.ndarray, y: np.ndarray, bins: int = 60) -> np.ndarray:
    """Assign each point a log-density color via 2-D histogram (fast, O(n))."""
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.searchsorted(xe[1:], x), 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(ye[1:], y), 0, H.shape[1] - 1)
    return np.log1p(H[xi, yi])


def _add_gate_shapes(fig: go.Figure, gate, color: str = "lime") -> None:
    """Overlay a single gate object as a Plotly shape / trace."""
    if isinstance(gate, RectangleGate):
        fig.add_shape(
            type="rect",
            x0=gate.xmin, x1=gate.xmax,
            y0=gate.ymin, y1=gate.ymax,
            line=dict(color=color, width=2, dash="dash"),
            fillcolor=f"rgba(0,255,0,0.04)" if color == "lime" else "rgba(255,87,51,0.08)",
        )
    elif isinstance(gate, PolygonGate) and len(gate.vertices) >= 3:
        verts = list(gate.vertices) + [gate.vertices[0]]
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts],
            y=[v[1] for v in verts],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
    elif isinstance(gate, EllipseGate):
        theta = np.linspace(0, 2 * np.pi, 100)
        rad = np.radians(gate.angle)
        ex = (gate.center[0]
              + (gate.width / 2) * np.cos(theta) * np.cos(rad)
              - (gate.height / 2) * np.sin(theta) * np.sin(rad))
        ey = (gate.center[1]
              + (gate.width / 2) * np.cos(theta) * np.sin(rad)
              + (gate.height / 2) * np.sin(theta) * np.cos(rad))
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))


def _make_gate_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    gatesets: dict,
    pending_gate=None,
    draw_mode: str = "box",   # "box" | "lasso"
    height: int = 370,
) -> go.Figure:
    """
    Density-coloured scatter (2-D) or histogram (1-D) with:
    - existing gate overlays (green dashed)
    - pending gate preview (orange)
    - Plotly drag mode set for box or lasso selection
    """
    if df.empty or x_col not in df.columns:
        return go.Figure().update_layout(
            title="No data", height=height,
            xaxis_title=x_col, yaxis_title=y_col or "Count",
        )

    x = np.log1p(df[x_col].clip(lower=0).values)

    if y_col and y_col in df.columns:
        # ── 2D scatter ────────────────────────────────────────────────────
        y = np.log1p(df[y_col].clip(lower=0).values)
        color = _density_colors(x, y)

        fig = go.Figure(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=2,
                color=color,
                colorscale="Viridis",
                showscale=False,
                opacity=0.75,
            ),
            selected=dict(marker=dict(color="#FF5733", size=4, opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.25)),
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
        ))
        fig.update_layout(
            dragmode="lasso" if draw_mode == "lasso" else "select",
            xaxis_title=f"{x_col}  (log1p)",
            yaxis_title=f"{y_col}  (log1p)",
        )

        # existing gate overlays
        for gs in gatesets.values():
            for g in gs.gates:
                if getattr(g, "x", None) == x_col and getattr(g, "y", None) == y_col:
                    _add_gate_shapes(fig, g, color="lime")

        # pending gate preview
        if pending_gate is not None:
            if (getattr(pending_gate, "x", None) == x_col
                    and getattr(pending_gate, "y", None) == y_col):
                _add_gate_shapes(fig, pending_gate, color="#FF5733")

    else:
        # ── 1D histogram ──────────────────────────────────────────────────
        fig = go.Figure(go.Histogram(
            x=x, nbinsx=200,
            marker_color="#4DBBD5", opacity=0.85,
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>count: %{{y}}<extra></extra>",
        ))
        fig.update_layout(
            dragmode="select",
            selectdirection="h",
            bargap=0,
            xaxis_title=f"{x_col}  (log1p)",
            yaxis_title="Count",
        )

        # existing threshold overlays
        for gs in gatesets.values():
            for g in gs.gates:
                if isinstance(g, ThresholdGate) and g.column == x_col:
                    v = g.value if g.value is not None else g.lo
                    if v is not None:
                        fig.add_vline(x=float(v), line_color="lime",
                                       line_dash="dash", line_width=2)

        # pending threshold preview
        if pending_gate is not None and isinstance(pending_gate, ThresholdGate):
            if pending_gate.column == x_col:
                lo = pending_gate.lo if pending_gate.lo is not None else pending_gate.value
                hi = pending_gate.hi
                if lo is not None and hi is not None:
                    fig.add_vrect(x0=lo, x1=hi,
                                   fillcolor="rgba(255,87,51,0.2)",
                                   line_width=0)
                elif lo is not None:
                    fig.add_vline(x=float(lo), line_color="#FF5733",
                                   line_dash="dash", line_width=2)

    fig.update_layout(
        height=height,
        margin=dict(l=55, r=10, t=28, b=50),
        showlegend=False,
        plot_bgcolor="rgba(20,20,30,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    )
    return fig


def _derive_gate(
    sel: dict,
    x_col: str,
    y_col: str | None,
    pop_df: pd.DataFrame,
) -> tuple:
    """
    Parse a Streamlit Plotly selection event into a gate object.
    Returns (gate, n_in, n_tot) or (None, 0, 0) on failure.
    """
    gate = None

    if y_col:
        # 2D
        if sel.get("box"):
            b = sel["box"][0]
            xs = sorted(b.get("x", [0, 0]))
            ys = sorted(b.get("y", [0, 0]))
            if xs[1] > xs[0] and ys[1] > ys[0]:
                gate = RectangleGate(
                    "rect", x=x_col, y=y_col,
                    xmin=xs[0], xmax=xs[1],
                    ymin=ys[0], ymax=ys[1],
                    log1p=True,
                )
        elif sel.get("lasso"):
            lasso = sel["lasso"][0]
            verts = list(zip(lasso.get("x", []), lasso.get("y", [])))
            if len(verts) >= 3:
                gate = PolygonGate(
                    "poly", x=x_col, y=y_col,
                    vertices=verts, log1p=True,
                )
    else:
        # 1D — horizontal box selection gives x range
        if sel.get("box"):
            b = sel["box"][0]
            xs = sorted(b.get("x", [0, 0]))
            if xs[1] > xs[0]:
                gate = ThresholdGate(
                    "threshold", column=x_col,
                    op="between", lo=xs[0], hi=xs[1],
                    log1p=True,
                )

    if gate is None:
        return None, 0, 0

    try:
        mask = gate.evaluate(pop_df)
        return gate, int(mask.sum()), len(pop_df)
    except Exception:
        return None, 0, 0


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
(tab_load, tab_gate, tab_pri, tab_viz, tab_export) = st.tabs(
    ["📁 Load Data", "🔲 Gate", "📊 PRI Analysis", "📈 Visualize", "💾 Export"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
with tab_load:
    st.header("Load FCS Data")

    col_fcs, col_ann = st.columns(2, gap="large")

    with col_fcs:
        st.subheader("FCS Files")
        fcs_files = st.file_uploader(
            "Upload FCS files", type=["fcs"], accept_multiple_files=True,
            help="Upload one or more FCS files matched to wells via the annotation table.",
        )
        dataset_id_input = st.text_input(
            "Dataset ID (optional)", placeholder="e.g. Plate_1",
            help="Tag appended to every event row — useful for multi-plate experiments.",
        )

    with col_ann:
        st.subheader("Sample Annotation")
        st.caption(
            "Upload a CSV/Excel **or** fill in the table.  \n"
            "Required columns: `well`, `sample`, `time`."
        )
        ann_file = st.file_uploader(
            "Upload annotation file", type=["csv", "xlsx", "xls"], key="ann_upload",
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
                "well": st.column_config.TextColumn("Well ID"),
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
            setup_dict = {}
            for _, row in ann.iterrows():
                entry = {"sample": str(row["sample"]), "time": float(row["time"])}
                for col in ann.columns:
                    if col not in ("well", "sample", "time"):
                        entry[col] = row[col]
                setup_dict[str(row["well"])] = entry

            tmpdir = st.session_state["tmpdir"]
            if tmpdir is None or not os.path.isdir(tmpdir):
                tmpdir = tempfile.mkdtemp(prefix="thermoflow_")
                st.session_state["tmpdir"] = tmpdir

            for uf in fcs_files:
                with open(os.path.join(tmpdir, uf.name), "wb") as fh:
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
                except Exception as exc:
                    st.error(f"Load failed: {exc}")
            st.rerun()

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

        st.subheader("Rename Sample")
        samples_avail = sorted(raw["sample"].dropna().unique()) if "sample" in raw.columns else []
        rn1, rn2, rn3 = st.columns([2, 2, 1])
        old_name = rn1.selectbox("Current name", samples_avail, key="rn_old")
        new_name_v = rn2.text_input("New name", key="rn_new", placeholder="Replacement name")
        if rn3.button("Rename", use_container_width=True) and new_name_v.strip():
            exp.rename_sample(old_name, new_name_v.strip())
            st.rerun()

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
            file_name="annotation_template.csv", mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — GATE  (FlowJo-style multi-plot interactive gating)
# ═════════════════════════════════════════════════════════════════════════════
with tab_gate:
    exp = get_exp()

    if not exp.populations:
        st.info("Load data first in the **Load Data** tab.")
    else:
        pops  = list(exp.populations.keys())
        chans = exp.channels

        # ── Top controls ─────────────────────────────────────────────────────
        tc1, tc2, tc3, tc4, tc5 = st.columns([2, 1, 1, 1, 1])
        parent_pop = tc1.selectbox("Gate population", pops, key="g_parent")
        pop_df     = exp.get_data(parent_pop)

        all_samples = (["All"] + sorted(pop_df["sample"].dropna().unique().tolist())
                       if "sample" in pop_df.columns else ["All"])
        all_times   = (["All"] + [str(t) for t in sorted(pop_df["time"].dropna().unique())]
                       if "time"   in pop_df.columns else ["All"])

        g_sample = tc2.selectbox("Sample", all_samples,  key="g_samp")
        g_time   = tc3.selectbox("Time",   all_times,    key="g_time")

        draw_mode  = tc4.radio("Draw mode", ["Box", "Lasso"], horizontal=True, key="g_draw")
        n_slots    = tc5.select_slider("Plots", [1, 2, 4, 6], value=4, key="g_nplots")
        subsample  = tc5.select_slider("Max pts", [1_000, 5_000, 10_000, 30_000],
                                       value=5_000, key="g_sub")

        # ── Population hierarchy ─────────────────────────────────────────────
        raw_n = len(exp.populations.get("raw", []))
        with st.container():
            hier_cols = st.columns(min(len(pops), 6))
            for i, (pname, pdata) in enumerate(exp.populations.items()):
                pct = f"{100*len(pdata)/raw_n:.0f}% of raw" if raw_n and pname != "raw" else ""
                hier_cols[i % len(hier_cols)].metric(
                    f"`{pname}`", f"{len(pdata):,}", delta=pct or None
                )

        st.divider()

        # ── Init / resize plot configs ───────────────────────────────────────
        cfgs = st.session_state["g_plot_cfgs"]
        if len(cfgs) != n_slots:
            new_cfgs = []
            for i in range(n_slots):
                if i < len(cfgs):
                    new_cfgs.append(cfgs[i])
                else:
                    x_i = chans[i % len(chans)]
                    y_i = chans[(i + 1) % len(chans)] if len(chans) > 1 and i < n_slots - 1 else None
                    new_cfgs.append({"x": x_i, "y": y_i})
            st.session_state["g_plot_cfgs"] = new_cfgs
            cfgs = st.session_state["g_plot_cfgs"]

        # ── Build filtered / subsampled view DataFrame ───────────────────────
        view_df = pop_df.copy()
        if g_sample != "All" and "sample" in view_df.columns:
            view_df = view_df[view_df["sample"] == g_sample]
        if g_time != "All" and "time" in view_df.columns:
            view_df = view_df[view_df["time"] == float(g_time)]
        if len(view_df) > subsample:
            view_df = view_df.sample(n=subsample, random_state=42)

        pending = st.session_state["g_pending"]

        # ── Plot grid ─────────────────────────────────────────────────────────
        cols_per_row = 2 if n_slots > 1 else 1
        n_rows = (n_slots + cols_per_row - 1) // cols_per_row

        # pre-allocate column containers
        grid: list = []
        for _r in range(n_rows):
            grid.append(st.columns(cols_per_row))

        new_pending = None  # will be set if any plot produces a selection this rerun

        for slot_i, cfg in enumerate(cfgs):
            row_i = slot_i // cols_per_row
            col_i = slot_i % cols_per_row

            with grid[row_i][col_i]:
                # channel selectors — compact, side by side
                cx_col, cy_col = st.columns(2)
                x_sel = cx_col.selectbox(
                    "X", chans,
                    index=chans.index(cfg["x"]) if cfg["x"] in chans else 0,
                    key=f"gx_{slot_i}", label_visibility="collapsed",
                )
                y_opts = ["— histogram —"] + chans
                y_def  = "— histogram —" if cfg.get("y") is None else cfg.get("y", chans[0])
                y_idx  = y_opts.index(y_def) if y_def in y_opts else 0
                y_sel  = cy_col.selectbox(
                    "Y", y_opts, index=y_idx,
                    key=f"gy_{slot_i}", label_visibility="collapsed",
                )
                y_col_sel = None if y_sel == "— histogram —" else y_sel

                # update stored config
                cfgs[slot_i] = {"x": x_sel, "y": y_col_sel}

                # pending gate for overlay (only if channels match)
                plot_pending = None
                if pending is not None and pending.get("gate") is not None:
                    g = pending["gate"]
                    gx = getattr(g, "x",      getattr(g, "column", None))
                    gy = getattr(g, "y",      None)
                    if gx == x_sel and (gy == y_col_sel or (gy is None and y_col_sel is None)):
                        plot_pending = g

                fig = _make_gate_plot(
                    view_df, x_sel, y_col_sel,
                    exp.gatesets,
                    pending_gate=plot_pending,
                    draw_mode=draw_mode.lower(),
                )

                # Versioned key so incrementing g_plot_ver resets all selections
                chart_key = f"g_chart_{slot_i}_v{st.session_state['g_plot_ver']}"
                sel_mode  = ["lasso"] if draw_mode == "Lasso" else ["box"]

                event = st.plotly_chart(
                    fig,
                    on_select="rerun",
                    selection_mode=sel_mode,
                    use_container_width=True,
                    key=chart_key,
                )

                # Parse selection — only adopt if it contains actual region data
                if event and event.selection:
                    sel = event.selection
                    has_region = bool(sel.get("box")) or bool(sel.get("lasso"))
                    if has_region and new_pending is None:
                        gate, n_in, n_tot = _derive_gate(sel, x_sel, y_col_sel, pop_df)
                        if gate is not None:
                            new_pending = {
                                "gate": gate,
                                "x_col": x_sel, "y_col": y_col_sel,
                                "slot": slot_i,
                                "n_in": n_in, "n_tot": n_tot,
                            }

        # Commit new pending (only when a real selection was drawn this rerun)
        if new_pending is not None:
            st.session_state["g_pending"] = new_pending
            pending = new_pending

        st.session_state["g_plot_cfgs"] = cfgs  # persist any channel changes

        # ── Pending gate confirmation banner ──────────────────────────────────
        st.divider()

        if pending is not None:
            gate    = pending["gate"]
            n_in    = pending["n_in"]
            n_tot   = pending["n_tot"]
            pct     = 100 * n_in / n_tot if n_tot else 0
            gtype   = type(gate).__name__.replace("Gate", "")
            chan_str = pending["x_col"] + (
                f" × {pending['y_col']}" if pending.get("y_col") else " (1D)"
            )

            ba, bb, bc = st.columns([4, 2, 1])
            ba.info(
                f"**{gtype} gate** on {chan_str} — "
                f"**{n_in:,}** / {n_tot:,} events  ({pct:.1f}%)"
            )
            confirm_name = bb.text_input(
                "New population name", value="cells", key="g_confirm_name",
                label_visibility="collapsed",
            )
            btn_apply, btn_discard = bc.columns(2)

            if btn_apply.button("✅", help="Apply gate", use_container_width=True):
                if confirm_name.strip():
                    gs = GateSet(
                        name=f"ManualGate_{confirm_name.strip()}",
                        gates=[gate],
                    )
                    exp.apply_gateset(gs, parent_pop=parent_pop,
                                      new_pop_name=confirm_name.strip())
                    st.session_state["g_pending"]  = None
                    st.session_state["g_plot_ver"] += 1   # resets chart selections
                    st.rerun()
                else:
                    st.warning("Enter a population name before applying.")

            if btn_discard.button("✕", help="Discard gate", use_container_width=True):
                st.session_state["g_pending"]  = None
                st.session_state["g_plot_ver"] += 1
                st.rerun()
        else:
            st.caption(
                "💡 **Draw a gate** on any plot above using the box or lasso tool in the "
                "Plotly toolbar (top-right of each chart).  The confirmation bar will "
                "appear here."
            )

        # ── Gate persistence ──────────────────────────────────────────────────
        with st.expander("Gate save / load", expanded=False):
            gs_col1, gs_col2 = st.columns(2)
            if exp.gatesets:
                gates_json = json.dumps(
                    {"gatesets": {k: v.to_dict() for k, v in exp.gatesets.items()}},
                    indent=2, default=str,
                )
                gs_col1.download_button(
                    "💾 Save gates (JSON)",
                    data=gates_json.encode(),
                    file_name="gates.json", mime="application/json",
                    use_container_width=True,
                )
                for gname, gs in exp.gatesets.items():
                    st.caption(f"  **{gname}** — {len(gs.gates)} gate(s), "
                               f"logic: {gs.logic}")
            else:
                gs_col1.caption("No gates yet.")

            gates_up = gs_col2.file_uploader(
                "Load gates JSON", type=["json"], key="g_gates_up",
            )
            if gates_up:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    tf.write(gates_up.read())
                    tpath = tf.name
                try:
                    exp.load_gates(tpath)
                    st.success(f"Loaded {len(exp.gatesets)} gatesets.")
                    st.rerun()
                finally:
                    os.unlink(tpath)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRI ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pri:
    st.header("PRI Analysis")
    exp = get_exp()

    if not exp.populations:
        st.info("Load data first.")
    else:
        pops  = list(exp.populations.keys())
        chans = exp.channels
        p_left, p_right = st.columns(2, gap="large")

        with p_left:
            st.subheader("Parameters")
            pri_pop = st.selectbox("Population", pops, key="pri_pop")
            pri_df  = exp.get_data(pri_pop)
            samples_in_pop = (sorted(pri_df["sample"].dropna().unique().tolist())
                              if "sample" in pri_df.columns else [])

            pri_channel = st.selectbox("Fluorescence channel", chans, key="pri_ch")
            ctrl_sample = st.selectbox("Control sample", samples_in_pop, key="pri_ctrl")

            ref_opts   = ["(none)"] + samples_in_pop
            ref_sample = st.selectbox("Reference sample (optional)", ref_opts, key="pri_ref")
            ref_sample = None if ref_sample == "(none)" else ref_sample

            with st.expander("Advanced"):
                pos_frac      = st.slider("pos_frac", 0.001, 0.5, 0.01, 0.001,
                                           format="%.3f", key="pri_pf")
                n_boot        = st.slider("Bootstrap iterations", 0, 500, 100, 10, key="pri_boot")
                baseline_time = st.number_input("Baseline time (min)", value=0, step=1,
                                                 key="pri_bt")
                use_thr       = st.checkbox("Override threshold (log1p)", key="pri_thr_cb")
                custom_thr    = None
                if use_thr:
                    custom_thr = st.number_input(
                        "threshold_log", value=5.0, step=0.05, format="%.4f",
                        key="pri_thr_val",
                    )

        with p_right:
            st.subheader("Threshold Preview")
            if pri_channel in pri_df.columns and not pri_df.empty:
                ctrl_rows = pri_df[pri_df["sample"] == ctrl_sample] if ctrl_sample else pri_df
                if not ctrl_rows.empty:
                    xv       = np.log1p(ctrl_rows[pri_channel].clip(lower=0))
                    auto_thr = (custom_thr if (use_thr and custom_thr is not None)
                                else float(np.quantile(xv, np.clip(1.0 - pos_frac, 0, 1))))
                    n_pos    = int((xv >= auto_thr).sum())
                    n_total  = len(xv)

                    fig_thr = go.Figure(go.Histogram(
                        x=xv, nbinsx=150, marker_color="#4DBBD5",
                        opacity=0.85,
                        hovertemplate="log1p=%{x:.3f}<br>count=%{y}<extra></extra>",
                    ))
                    fig_thr.add_vline(
                        x=auto_thr, line_color="#E64B35",
                        line_dash="dash", line_width=2,
                        annotation_text=f"  thr={auto_thr:.3f}  ({100*n_pos/n_total:.1f}% +)",
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
                        f"{ctrl_sample} — {n_pos:,} / {n_total:,} events "
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
        pops  = list(exp.populations.keys())
        chans = exp.channels

        v1, v2, v3, v4, v5 = st.tabs([
            "PRI Decay Curves", "PRI Bar Chart", "Summary Grid",
            "Density Plot", "Histogram",
        ])

        with v1:
            if exp.pri_table.empty:
                st.info("Run PRI analysis first.")
            else:
                a, b = st.columns(2)
                which_v  = a.radio("Metric", ["PRI_norm", "PRI_abs"], horizontal=True, key="v1_w")
                cols_v   = b.slider("Columns", 1, 6, 3, key="v1_c")
                ci_v     = a.checkbox("Show 95% CI band", key="v1_ci")
                params_v = b.multiselect("Show params", ["t_half", "y0", "r2"],
                                          default=["t_half", "r2"], key="v1_p")
                spine_v  = a.slider("Spine width", 0.3, 2.0, 0.8, 0.1, key="v1_s")
                res_v    = b.checkbox("Show residuals panel", key="v1_r")
                if st.button("Plot decay curves", key="btn_v1", use_container_width=True):
                    _render(exp.plot_pri, which=which_v, cols=cols_v,
                            show_ci=ci_v, show_params=params_v,
                            spine_width=spine_v, plot_residuals=res_v)

        with v2:
            if exp.pri_fits_norm.empty:
                st.info("Run PRI analysis first.")
            else:
                samples_list = sorted(exp.pri_fits_norm["sample"].dropna().unique().tolist())
                a, b = st.columns(2)
                which_b  = a.radio("Metric", ["t_half", "A"], horizontal=True, key="v2_w")
                use_nb   = a.checkbox("Use normalised fits", value=True, key="v2_n")
                norm_s   = b.selectbox("Normalise to sample", ["(none)"] + samples_list, key="v2_ns")
                norm_s   = None if norm_s == "(none)" else norm_s
                cpal     = b.selectbox("Colour palette",
                                        ["viridis", "plasma", "cividis", "magma"], key="v2_cp")
                filt_s   = st.multiselect("Restrict to samples", samples_list, key="v2_f")
                if st.button("Plot bar chart", key="btn_v2", use_container_width=True):
                    _render(exp.plot_pri_bars, which=which_b, use_norm=use_nb,
                            norm_sample=norm_s, color_palette=cpal,
                            samples=filt_s if filt_s else None)

        with v3:
            if exp.pri_table.empty:
                st.info("Run PRI analysis first.")
            else:
                a, b = st.columns(2)
                sg_w  = a.radio("Metric", ["PRI_norm", "PRI_abs"], horizontal=True, key="v3_w")
                sg_c  = b.slider("Columns", 2, 6, 4, key="v3_c")
                sg_ci = a.checkbox("Show 95% CI band", key="v3_ci")
                sg_p  = b.multiselect("Show params", ["t_half", "y0", "r2"],
                                       default=["t_half", "r2"], key="v3_p")
                if st.button("Plot summary grid", key="btn_v3", use_container_width=True):
                    _render(exp.plot_pri_summary_grid,
                            which=sg_w, cols=sg_c, show_ci=sg_ci, show_params=sg_p)

        with v4:
            a, b = st.columns(2)
            dp_pop = a.selectbox("Population", pops, key="v4_pop")
            dp_x   = a.selectbox("X channel", chans, key="v4_x")
            dp_y   = b.selectbox("Y channel", chans,
                                  index=min(1, len(chans) - 1), key="v4_y")
            dp_df  = exp.get_data(dp_pop)
            dp_samps = ["All"] + (sorted(dp_df["sample"].dropna().unique().tolist())
                                   if "sample" in dp_df.columns else [])
            dp_fs = b.selectbox("Filter sample", dp_samps, key="v4_fs")
            with st.expander("More options"):
                e1, e2 = st.columns(2)
                dp_slice   = e1.checkbox("Facet by sample", key="v4_sl")
                dp_cnt     = e1.checkbox("Show contours",   key="v4_cn")
                dp_stats   = e2.checkbox("Gate stats",      value=True, key="v4_st")
                gate_keys  = list(exp.gatesets.keys())
                dp_ov      = e2.multiselect("Overlay gatesets", gate_keys, key="v4_ov")
            if st.button("Plot density", key="btn_v4", use_container_width=True):
                _render(exp.plot_density,
                        x_col=dp_x, y_col=dp_y, pop_name=dp_pop,
                        subset_sample=None if dp_fs == "All" else dp_fs,
                        slice_by="sample" if dp_slice else None,
                        show_contours=dp_cnt,
                        gates_to_overlay=[exp.gatesets[k] for k in dp_ov] or None,
                        show_stats=dp_stats)

        with v5:
            a, b = st.columns(2)
            h_pop  = a.selectbox("Population", pops, key="v5_pop")
            h_col  = a.selectbox("Channel",    chans, key="v5_col")
            h_sl   = b.checkbox("Facet by sample",   key="v5_sl")
            h_ov   = b.checkbox("Overlay all samples", key="v5_ov")
            h_hz   = b.checkbox("Horizontal layout",   key="v5_hz")
            h_vl   = b.number_input("Threshold line (0 = off)", value=0.0,
                                     step=0.05, format="%.3f", key="v5_vl")
            if st.button("Plot histogram", key="btn_v5", use_container_width=True):
                _render(exp.plot_sliced_histogram,
                        col=h_col, pop_name=h_pop,
                        slice_by="sample" if h_sl else "time",
                        overlay=h_ov,
                        orientation="horizontal" if h_hz else "vertical",
                        vline=h_vl if h_vl != 0.0 else None)


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
            st.download_button("⬇️ PRI event table (CSV)",
                                data=exp.pri_table.to_csv(index=False).encode(),
                                file_name="pri_table.csv", mime="text/csv",
                                use_container_width=True)
            st.download_button("⬇️ Normalised fits (CSV)",
                                data=exp.pri_fits_norm.to_csv(index=False).encode(),
                                file_name="pri_fits_norm.csv", mime="text/csv",
                                use_container_width=True)
            st.download_button("⬇️ Absolute fits (CSV)",
                                data=exp.pri_fits_abs.to_csv(index=False).encode(),
                                file_name="pri_fits_abs.csv", mime="text/csv",
                                use_container_width=True)
        else:
            st.caption("Run PRI analysis first.")

        st.divider()
        st.subheader("Gates")
        if exp.gatesets:
            st.download_button(
                "⬇️ Gates (JSON)",
                data=json.dumps(
                    {"gatesets": {k: v.to_dict() for k, v in exp.gatesets.items()}},
                    indent=2, default=str,
                ).encode(),
                file_name="gates.json", mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("No gates drawn yet.")

        st.divider()
        template = pd.DataFrame([
            {"well": "A01", "sample": "Ctrl",    "time": 0.0},
            {"well": "A02", "sample": "Ctrl",    "time": 5.0},
            {"well": "B01", "sample": "SampleA", "time": 0.0},
        ])
        st.download_button("⬇️ Annotation template (CSV)",
                            data=template.to_csv(index=False).encode(),
                            file_name="annotation_template.csv", mime="text/csv",
                            use_container_width=True)

    with e_right:
        st.subheader("Full Report")
        if exp.pri_table.empty:
            st.info("Run PRI analysis first.")
        else:
            fmt = st.radio("Format", ["HTML", "PDF"], horizontal=True, key="rpt_fmt")
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report…"):
                    try:
                        rpt    = FlowReport(exp)
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
                        mime  = "text/html" if fmt == "HTML" else "application/pdf"
                        fname = f"thermoflow_report{suffix}"
                        st.download_button(f"⬇️ Download {fmt} report",
                                            data=rpt_bytes, file_name=fname, mime=mime,
                                            use_container_width=True)
                        st.success("Report ready.")
                    except Exception as exc:
                        st.error(f"Report generation failed: {exc}")

        st.divider()
        st.subheader("Experiment Summary")
        if exp.populations:
            st.text(FlowReport(exp).summary())
        else:
            st.caption("No data loaded.")
