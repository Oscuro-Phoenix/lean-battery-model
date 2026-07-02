"""Community Database — universal Damkohler-space map of every submitted cell."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from lean_model import database
    from lean_model.uistyle import CHEM_COLORS, MARKER_LINE, style_fig
except ImportError:
    import importlib
    import sys as _sys
    for _m in [m for m in list(_sys.modules) if m.startswith("lean_model")]:
        del _sys.modules[_m]
    importlib.invalidate_caches()
    from lean_model import database
    from lean_model.uistyle import CHEM_COLORS, MARKER_LINE, style_fig

st.set_page_config(page_title="Community Database — Lean Battery Model",
                   page_icon="🌐", layout="wide")

st.title("🌐 Community Database")
st.caption(
    "Every cell characterized with this tool — literature references plus "
    "community submissions — on the same Damköhler map from *Scaling and "
    "Analytical Approximation of Porous Electrode Theory for "
    "Reaction-limited Batteries* (Pathak & Bazant)."
)

if not database.db_configured():
    st.info(
        "ℹ️ No shared database is configured for this deployment, so "
        "community submissions are stored locally to this running instance "
        "only (they reset on the next redeploy). See `lean_model/database.py` "
        "for a two-minute setup with a free Postgres database.")

df = database.load_database()

with st.sidebar:
    st.markdown("### Filters")
    chems = sorted(df["chemistry"].dropna().unique().tolist())
    pick = st.multiselect("Chemistry / type", chems, default=chems)
    show_ref = st.checkbox("Literature reference electrodes", True)
    show_community = st.checkbox("Community submissions", True)

mask = df["chemistry"].isin(pick)
if not show_ref:
    mask &= ~df["is_reference"].astype(bool)
if not show_community:
    mask &= df["is_reference"].astype(bool)
view = df[mask].copy()

n_ref = int(df["is_reference"].astype(bool).sum())
n_com = int((~df["is_reference"].astype(bool)).sum())
st.markdown(f"**{len(view)}** cell(s) shown — {n_ref} reference, {n_com} community.")


PANELS = [
    dict(y="Da_w", ytitle="Da_w  (wiring)", title="Wiring vs. process",
        region=lambda x: 1e2 * x, region_label="lean-model validity (Da_w ≲ 10² Da_p)"),
    dict(y="Da", ytitle="Da  (electrolyte)", title="Electrolyte vs. process",
        region=lambda x: x, region_label="electrolyte not limiting (Da ≲ Da_p)"),
    dict(y="Da_c", ytitle="Da_c  (capacitive)", title="Capacitive vs. process",
        region=None, region_label=None),
]

valid = view.dropna(subset=["Da_p"])
if not valid.empty:
    xlo = max(valid["Da_p"].min() * 0.5, 1e-4)
    xhi = valid["Da_p"].max() * 2
else:
    xlo, xhi = 1e-2, 1e3
xrange = [np.log10(xlo), np.log10(xhi)]
xx = np.logspace(*xrange, 60)

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                    subplot_titles=[p["title"] for p in PANELS])

for col, panel in enumerate(PANELS, start=1):
    if panel["region"] is not None:
        yy = panel["region"](xx)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xx, xx[::-1]]),
            y=np.concatenate([yy, np.full_like(xx, 1e-6)]),
            fill="toself", fillcolor="rgba(46,204,154,0.14)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=1, col=col)
    for chem in sorted(view["chemistry"].dropna().unique()):
        sub = view[view["chemistry"] == chem].dropna(subset=["Da_p", panel["y"]])
        if sub.empty:
            continue
        is_ref = sub["is_reference"].astype(bool)
        symbols = np.where(is_ref, "diamond", "circle")
        sizes = np.where(is_ref, 13, 10)
        fig.add_trace(go.Scatter(
            x=sub["Da_p"], y=sub[panel["y"]], mode="markers", name=chem,
            legendgroup=chem, showlegend=(col == 1),
            marker=dict(color=CHEM_COLORS.get(chem, "#9aa4b2"), size=sizes,
                        symbol=symbols, line=dict(color=MARKER_LINE, width=0.8)),
            customdata=np.stack([sub["cell_name"], sub["contributor"].fillna(""),
                                sub["source_note"].fillna("")], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Da_p=%{x:.3g}<br>" + panel["y"] +
                "=%{y:.3g}<br>%{customdata[1]}<br>%{customdata[2]}"
                "<extra></extra>")),
            row=1, col=col)
    fig.update_xaxes(title="Da_p  (process)", type="log", range=xrange,
                     row=1, col=col)
    fig.update_yaxes(title=panel["ytitle"], type="log",
                     scaleanchor=f"x{col if col > 1 else ''}", scaleratio=1,
                     row=1, col=col)

style_fig(fig, height=440, legend=dict(orientation="h", yanchor="bottom", y=1.16,
                                       groupclick="togglegroup"))
for ann in fig.layout.annotations:
    ann.font = dict(color="#e6edf3", size=14)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "◆ literature reference · ● community submission — shaded bands mark "
    "the lean-model validity region (left) and where the electrolyte is "
    "not rate-limiting (middle), from the manuscript's design envelope."
)

st.markdown("#### All entries")
display_cols = ["cell_name", "chemistry", "method", "crate_ref", "Da_w", "Da",
                "Da_p", "Da_c", "R_s", "frac", "rms_mV", "contributor",
                "source_note", "timestamp", "is_reference"]
st.dataframe(view[display_cols].sort_values("is_reference", ascending=False),
            hide_index=True, use_container_width=True)
st.download_button("Download full database (CSV)",
                   df[display_cols].to_csv(index=False).encode(),
                   "lean_model_community_database.csv", "text/csv")

st.markdown("---")
st.caption(
    "Contribute your own cell from the **Predict discharge**, **Fit "
    "discharge data**, **Fit EIS**, or **Damköhler calculator** pages — "
    "look for the *\"Add this cell to the public community database\"* "
    "panel after computing your groups. Submissions are public, attributed "
    "to the contributor name you provide (or \"anonymous\"), and limited "
    "to dimensionless descriptors — no raw curves are shared."
)
