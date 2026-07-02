"""Community Database — universal Damkohler-space plot of every submitted cell."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from lean_model import database
except ImportError:
    import importlib
    import sys as _sys
    for _m in [m for m in list(_sys.modules) if m.startswith("lean_model")]:
        del _sys.modules[_m]
    importlib.invalidate_caches()
    from lean_model import database

st.set_page_config(page_title="Community Database — Lean Battery Model",
                   page_icon="🌐", layout="wide")

st.title("🌐 Community Database")
st.caption(
    "Every cell characterized with this tool — literature references plus "
    "community submissions — placed on the same Damköhler map from "
    "*Scaling and Analytical Approximation of Porous Electrode Theory for "
    "Reaction-limited Batteries* (Pathak & Bazant)."
)

if not database.github_configured():
    st.info(
        "ℹ️ This deployment doesn't have public GitHub persistence "
        "configured, so community submissions are stored locally to this "
        "running instance only (they will reset on the next redeploy). "
        "See `lean_model/database.py` for the two-minute setup.")

df = database.load_database()

CHEM_COLORS = {
    "cathode": "#d62728", "anode": "#1f77b4", "full cell": "#2ca02c",
    "pseudocapacitor": "#9467bd", "other": "#7f7f7f",
}

with st.sidebar:
    st.markdown("### Filters")
    chems = sorted(df["chemistry"].dropna().unique().tolist())
    pick = st.multiselect("Chemistry / type", chems, default=chems)
    show_ref = st.checkbox("Show literature reference electrodes", True)
    show_community = st.checkbox("Show community submissions", True)

mask = df["chemistry"].isin(pick)
if not show_ref:
    mask &= ~df["is_reference"].astype(bool)
if not show_community:
    mask &= df["is_reference"].astype(bool)
view = df[mask].copy()

st.markdown(f"**{len(view)}** cell(s) shown "
           f"({int(df['is_reference'].astype(bool).sum())} reference, "
           f"{int((~df['is_reference'].astype(bool)).sum())} community).")


def _scatter_by_chem(fig, x, y, name_suffix, symbol_ref="diamond",
                     symbol_com="circle"):
    for chem in sorted(view["chemistry"].dropna().unique()):
        sub = view[view["chemistry"] == chem]
        for is_ref, symbol in ((True, symbol_ref), (False, symbol_com)):
            s = sub[sub["is_reference"].astype(bool) == is_ref]
            s = s.dropna(subset=[x, y])
            if s.empty:
                continue
            label = f"{chem} ({'ref' if is_ref else 'community'})"
            fig.add_trace(go.Scatter(
                x=s[x], y=s[y], mode="markers", name=label,
                marker=dict(color=CHEM_COLORS.get(chem, "#7f7f7f"), size=11,
                            symbol=symbol,
                            line=dict(color="black", width=0.8)),
                customdata=np.stack([s["cell_name"], s["contributor"].fillna(""),
                                    s["source_note"].fillna("")], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" + x + "=%{x:.3g}<br>" +
                    y + "=%{y:.3g}<br>%{customdata[1]}<br>%{customdata[2]}"
                    "<extra></extra>")))


st.markdown("#### Wiring vs. process — lean-model validity")
st.caption(
    "The lean model's leading-order solution is accurate while the wiring "
    "group stays modest relative to the process group; literature cells and "
    "submissions falling well above the shaded band are wiring-limited and "
    "need the full (higher-order) theory."
)
fig1 = go.Figure()
if not view.dropna(subset=["Da_p", "Da_w"]).empty:
    xx = np.logspace(
        np.log10(max(view["Da_p"].min() * 0.5, 1e-3)),
        np.log10(view["Da_p"].max() * 2), 50)
    fig1.add_trace(go.Scatter(
        x=np.concatenate([xx, xx[::-1]]),
        y=np.concatenate([1e2 * xx, np.full_like(xx, 1e-2)]),
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(width=0), name="lean-model validity (Da_w ≲ 10² Da_p)",
        hoverinfo="skip"))
_scatter_by_chem(fig1, "Da_p", "Da_w", "wiring")
fig1.update_layout(
    xaxis=dict(title="Da_p  (process)", type="log"),
    yaxis=dict(title="Da_w  (wiring)", type="log"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=10, r=10, t=30, b=10), height=480, template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("#### Electrolyte vs. process — salt-depletion onset")
st.caption(
    "Below the shaded diagonal (Da ≲ Da_p) electrolyte transport keeps up "
    "with the applied rate; above it the electrolyte is expected to be "
    "rate-limiting at the reference C-rate."
)
fig2 = go.Figure()
if not view.dropna(subset=["Da_p", "Da"]).empty:
    xx2 = np.logspace(
        np.log10(max(view["Da_p"].min() * 0.5, 1e-3)),
        np.log10(view["Da_p"].max() * 2), 50)
    fig2.add_trace(go.Scatter(
        x=np.concatenate([xx2, xx2[::-1]]),
        y=np.concatenate([xx2, np.full_like(xx2, 1e-2)]),
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(width=0), name="electrolyte not limiting (Da ≲ Da_p)",
        hoverinfo="skip"))
_scatter_by_chem(fig2, "Da_p", "Da", "electrolyte")
fig2.update_layout(
    xaxis=dict(title="Da_p  (process)", type="log"),
    yaxis=dict(title="Da  (electrolyte)", type="log"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=10, r=10, t=30, b=10), height=480, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Capacitive vs. process — double-layer charging")
fig3 = go.Figure()
_scatter_by_chem(fig3, "Da_p", "Da_c", "capacitive")
fig3.update_layout(
    xaxis=dict(title="Da_p  (process)", type="log"),
    yaxis=dict(title="Da_c  (capacitive)", type="log"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=10, r=10, t=30, b=10), height=440, template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

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
    "Contribute your own cell from the **Fit discharge data**, **Fit EIS**, "
    "**Damköhler calculator**, or **Predict discharge** pages — look for "
    "the *\"Add this cell to the public community database\"* panel after "
    "computing your groups. Submitted data is public, attributed to the "
    "contributor name you provide (or \"anonymous\"), and limited to "
    "dimensionless descriptors — no raw curves are shared."
)
