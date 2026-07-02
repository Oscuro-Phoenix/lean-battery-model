"""Lean Porous-Electrode Model — interactive predictor and data fitter.

Streamlit app around the analytical "lean model" of Pathak & Bazant
(Scaling and Analytical Approximation of Porous Electrode Theory for
Reaction-limited Batteries, J. Electrochem. Soc.).

Run locally:   streamlit run app.py
"""

import io
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import to_hex

from lean_model import (
    compute_da_numbers, ecd_mhc, ecd_mhc_learnable, eis_model,
    fit_descriptors, fit_eis, model_V, nmc532_ocv, nmc532_ocv_deriv,
    ocv_derivative, ocv_from_table,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")

st.set_page_config(page_title="Lean Battery Model", page_icon="🔋",
                   layout="wide")

st.title("Lean Porous-Electrode Model")
st.caption(
    "Analytical prediction and characterization of battery discharge from "
    "four dimensionless Damköhler groups — no PDE solver required. "
    "Based on Pathak & Bazant, *Scaling and Analytical Approximation of "
    "Porous Electrode Theory for Reaction-limited Batteries*."
)


# ── helpers ────────────────────────────────────────────────────────────────

def parse_two_col_csv(uploaded) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read a 2-column CSV (capacity/SoC, voltage), with or without header.

    Returns (x, V, normalized) where `normalized` flags that the capacity
    column was rescaled by its maximum (input looked like mAh/g, not SoC).
    """
    raw = uploaded.read() if hasattr(uploaded, "read") else open(uploaded, "rb").read()
    df = pd.read_csv(io.BytesIO(raw), header=None)
    # drop a header row if the first row is not numeric
    if df.iloc[0].apply(lambda v: not str(v).replace(".", "").replace("-", "")
                        .replace("e", "").replace("E", "").replace("+", "")
                        .isdigit()).any():
        df = df.iloc[1:]
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    x = df.iloc[:, 0].to_numpy(float)
    v = df.iloc[:, 1].to_numpy(float)
    normalized = False
    if x.max() > 1.5:  # looks like raw capacity, not normalized SoC
        x = x / x.max()
        normalized = True
    m = (x >= 0) & (x <= 1)
    order = np.argsort(x[m])
    return x[m][order], v[m][order], normalized


def parse_eis_csv(uploaded):
    """Read an EIS CSV: frequency (Hz), Z' (Ohm), Z'' (Ohm), header optional.

    Returns (freq, z_re, z_im, flipped); if the imaginary column is mostly
    positive it is assumed to be -Z'' and flipped (flagged via `flipped`).
    """
    raw = uploaded.read() if hasattr(uploaded, "read") else open(uploaded, "rb").read()
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if df.iloc[0].apply(lambda v: not str(v).replace(".", "").replace("-", "")
                        .replace("e", "").replace("E", "").replace("+", "")
                        .isdigit()).any():
        df = df.iloc[1:]
    df = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna()
    freq = df.iloc[:, 0].to_numpy(float)
    z_re = df.iloc[:, 1].to_numpy(float)
    z_im = df.iloc[:, 2].to_numpy(float)
    flipped = False
    if np.nanmedian(z_im) > 0:  # user supplied -Z''
        z_im, flipped = -z_im, True
    order = np.argsort(freq)
    return freq[order], z_re[order], z_im[order], flipped


def crate_from_name(name: str) -> float | None:
    """Guess the C-rate from a filename like '0.5C.csv' or 'discharge_2C.csv'."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*C", name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def rate_colors(crates):
    cmap = plt.cm.coolwarm
    lo, hi = min(crates), max(crates)
    span = (hi - lo) or 1.0
    return {c: to_hex(cmap(0.05 + 0.9 * (c - lo) / span)) for c in crates}


def plot_curves(curves, params, ocv, title="", v_min=None, kin_shape=None):
    """Interactive overlay of measured points (if any) and lean-model curves."""
    fig = go.Figure()
    colors = rate_colors(list(curves.keys()))
    frac = params[6]
    socg = np.linspace(1e-3, min(frac, 0.999), 400)
    socf = np.linspace(1e-3, 0.999, 400)
    fig.add_trace(go.Scatter(
        x=socf, y=ocv(socf), mode="lines", name="OCV",
        line=dict(color="#8a8a8a", width=1.4, dash="dash"),
        hovertemplate="q=%{x:.3f}<br>U=%{y:.3f} V<extra>OCV</extra>"))
    for crate, data in sorted(curves.items()):
        col = colors[crate]
        if data is not None:
            soc, V = data
            fig.add_trace(go.Scatter(
                x=soc, y=V, mode="markers", name=f"{crate:g}C data",
                marker=dict(color=col, size=7,
                            line=dict(color="black", width=0.5)),
                hovertemplate="q=%{x:.3f}<br>V=%{y:.3f} V"
                              f"<extra>{crate:g}C data</extra>"))
        fig.add_trace(go.Scatter(
            x=socg,
            y=model_V(socg, params, crate, ocv, drop_saturated=True,
                      v_min=v_min, kin_shape=kin_shape),
            mode="lines", name=f"{crate:g}C model",
            line=dict(color=col, width=2.5),
            hovertemplate="q=%{x:.3f}<br>V=%{y:.3f} V"
                          f"<extra>{crate:g}C model</extra>"))
    if v_min is not None:
        fig.add_hline(y=v_min, line=dict(color="#aaaaaa", width=1,
                                         dash="dot"))
    fig.update_layout(
        title=title or None,
        xaxis=dict(title="Normalized capacity (–)", range=[0, 1]),
        yaxis=dict(title="Voltage (V)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10), height=480,
        template="plotly_white")
    return fig


def nyquist_figure(freq, z_re, z_im, z_fit=None):
    """Interactive Nyquist plot (frequency shown on hover)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_re, y=-np.asarray(z_im), mode="markers", name="data",
        customdata=np.asarray(freq),
        marker=dict(color="#1f77b4", size=7,
                    line=dict(color="black", width=0.5)),
        hovertemplate="f=%{customdata:.3g} Hz<br>Z'=%{x:.3g} Ω<br>"
                      "-Z''=%{y:.3g} Ω<extra>data</extra>"))
    if z_fit is not None:
        fig.add_trace(go.Scatter(
            x=np.real(z_fit), y=-np.imag(z_fit), mode="lines",
            name="lean model", customdata=np.asarray(freq),
            line=dict(color="#d62728", width=2.5),
            hovertemplate="f=%{customdata:.3g} Hz<br>Z'=%{x:.3g} Ω<br>"
                          "-Z''=%{y:.3g} Ω<extra>model</extra>"))
    fig.update_layout(
        xaxis=dict(title="Z' (Ω)"),
        yaxis=dict(title="-Z'' (Ω)", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10), height=480,
        template="plotly_white")
    return fig


def predictions_csv(params, crates, ocv, v_min=None, kin_shape=None) -> bytes:
    frac = params[6]
    socg = np.linspace(1e-3, min(frac, 0.999), 400)
    out = {"normalized_capacity": socg, "OCV_V": ocv(socg)}
    for c in crates:
        out[f"V_{c:g}C"] = model_V(socg, params, c, ocv, drop_saturated=True,
                                   v_min=v_min, kin_shape=kin_shape)
    return pd.DataFrame(out).to_csv(index=False).encode()


def kinetics_figure(kin_shape=None):
    """f(c) preview: pure CIET/MHC vs custom/learned shape."""
    c = np.linspace(1e-3, 1 - 1e-3, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c, y=ecd_mhc(c), mode="lines", name="CIET/MHC",
        line=dict(color="#8a8a8a", width=2, dash="dash"),
        hovertemplate="c=%{x:.3f}<br>f=%{y:.4f}<extra>CIET/MHC</extra>"))
    if kin_shape is not None:
        g, p1, p2 = kin_shape
        fig.add_trace(go.Scatter(
            x=c, y=ecd_mhc_learnable(c, gamma=g, p1=p1, p2=p2),
            mode="lines", name="custom f(c)",
            line=dict(color="#d62728", width=2.5),
            hovertemplate="c=%{x:.3f}<br>f=%{y:.4f}<extra>custom</extra>"))
    fig.update_layout(
        xaxis=dict(title="Filling fraction c", range=[0, 1]),
        yaxis=dict(title="Exchange-current factor f(c)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10), height=320,
        template="plotly_white")
    return fig


def ocv_selector(key: str):
    """Shared OCV-source widget. Returns an OCV(soc) callable."""
    src = st.radio("Open-circuit voltage (OCV) source",
                   ["Built-in NMC532 (Colclasure 2020)", "Upload OCV CSV"],
                   key=f"ocv_src_{key}", horizontal=True)
    if src.startswith("Built-in"):
        return nmc532_ocv
    up = st.file_uploader(
        "OCV CSV — two columns: normalized capacity (0–1), voltage (V)",
        type="csv", key=f"ocv_file_{key}")
    if up is None:
        st.info("Upload an OCV file (or switch to the built-in OCV) to continue.")
        st.stop()
    soc, v, normed = parse_two_col_csv(up)
    if normed:
        st.warning("OCV capacity column was rescaled by its maximum.")
    return ocv_from_table(soc, v)


# ── tabs ───────────────────────────────────────────────────────────────────

tab_predict, tab_fit, tab_eis, tab_calc, tab_about = st.tabs(
    ["Predict discharge", "Fit discharge data", "Fit EIS",
     "Damköhler calculator", "About"])


# ── 1. Predict ─────────────────────────────────────────────────────────────

with tab_predict:
    st.subheader("Predict galvanostatic discharge curves")
    mode = st.radio("Specify the electrode by",
                    ["Dimensionless groups", "Dimensional cell parameters"],
                    horizontal=True)

    ocv = ocv_selector("predict")

    cA, cB = st.columns([1, 1.4])
    with cA:
        rates_text = st.text_input("C-rates (comma-separated)", "0.5, 1, 2")
        try:
            crates = sorted({float(s) for s in rates_text.split(",") if s.strip()})
        except ValueError:
            st.error("Could not parse the C-rate list.")
            st.stop()
        a = st.slider("Start-of-discharge stoichiometry a", 0.0, 0.8, 0.30, 0.01)
        frac = st.slider("Usable capacity fraction", 0.70, 1.00, 1.00, 0.01)
        R_s = st.number_input("Series resistance R_s (V per C-rate)",
                              0.0, 0.5, 0.0, 0.005, format="%.3f")
        v_cut = st.number_input(
            "Lower cutoff voltage (V)", 0.0, 5.0, 3.0, 0.05,
            help="Discharge terminates here, as in a real cycler; the model "
                 "voltage is masked below this value.")

    if mode == "Dimensionless groups":
        with cB:
            c1, c2 = st.columns(2)
            Da_w = c1.number_input("Wiring group Da_w", 1e-3, 1e6, 200.0,
                                   format="%.4g")
            beta = c2.number_input("Electronic split β = Da_w,σ/Da_w",
                                   0.0, 0.5, 0.0, 0.05)
            Da = c1.number_input("Electrolyte group Da", 1e-4, 1e6, 1000.0,
                                 format="%.4g")
            Da_p1 = c2.number_input("Process group Da_p at 1C", 1e-3, 1e6,
                                    300.0, format="%.4g")
        params = (a, Da_w, beta, Da, Da_p1, R_s, frac)
    else:
        with cB:
            c1, c2, c3 = st.columns(3)
            L = c1.number_input("Thickness L (µm)", 1.0, 1000.0, 75.0) * 1e-6
            Rp = c2.number_input("Particle radius R_p (µm)", 0.01, 100.0,
                                 5.0) * 1e-6
            eps_p = c3.number_input("Porosity ε_p", 0.05, 0.9, 0.335)
            eps_am = c1.number_input("Active fraction ε_am", 0.05, 0.95, 0.665)
            sigma_s = c2.number_input("σ_s (S/m)", 1e-4, 1e4, 0.18,
                                      format="%.4g")
            kappa_l = c3.number_input("κ_l (S/m)", 1e-3, 1e2, 0.95,
                                      format="%.4g")
            D_ref = c1.number_input("D_ref (m²/s)", 1e-13, 1e-8, 2.5e-10,
                                    format="%.3g")
            t_plus = c2.number_input("t₊", 0.0, 1.0, 0.38)
            j0 = c3.number_input("j₀ (A/m²)", 1e-6, 1e3, 2.8, format="%.4g")
            csmax = c1.number_input("c_s,max (mol/m³)", 1e3, 1e5, 51765.0,
                                    format="%.5g")
            clref = c2.number_input("c_l,ref (mol/m³)", 100.0, 5000.0, 1000.0)
            CDL = c3.number_input("C_DL (F/m²)", 1e-3, 10.0, 0.2,
                                  format="%.3g")
        ap = 3.0 * eps_am / Rp
        groups = compute_da_numbers(L=L, ap=ap, eps_p=eps_p, eps_am=eps_am,
                                    D_ref=D_ref, sigma_s=sigma_s,
                                    kappa_l=kappa_l, t_plus=t_plus, j0=j0,
                                    C_DL=CDL, clref=clref, csmax=csmax,
                                    crate=1.0)
        st.markdown("**Computed dimensionless groups (at 1C):**")
        st.dataframe(pd.DataFrame({
            "Group": ["Da (electrolyte)", "Da_p (process)", "Da_w (wiring)",
                      "Da_c (capacitive)", "a_p (1/m)"],
            "Value": [f"{groups['Da']:.3g}", f"{groups['Da_p']:.3g}",
                      f"{groups['Da_w_sigma'] + groups['Da_w_kappa']:.3g}",
                      f"{groups['Da_c']:.3g}", f"{ap:.3g}"],
        }), hide_index=True, use_container_width=True)
        Da_w_tot = groups["Da_w_sigma"] + groups["Da_w_kappa"]
        beta = groups["Da_w_sigma"] / Da_w_tot if Da_w_tot else 0.5
        params = (a, Da_w_tot, min(beta, 0.5), groups["Da"], groups["Da_p"],
                  R_s, frac)

    with st.expander("Exchange-current kinetics f(c) — advanced"):
        st.markdown(
            "By default the exchange-current prefactor is the pure CIET/MHC "
            "form. Enable a custom shape to explore non-typical kinetics "
            "$f(\\tilde c_s)$: a symmetric sharpness exponent plus "
            "asymmetry and width corrections (identical to the *learnable* "
            "form on the fitting tab)."
        )
        custom_kin = st.checkbox("Use custom (non-CIET) f(c)",
                                 key="pred_custom_kin")
        kin_shape = None
        if custom_kin:
            k1, k2, k3 = st.columns(3)
            g_ = k1.slider("γ — endpoint sharpness", 0.3, 3.0, 1.0, 0.05,
                           help="γ<1: flatter near c=0,1; γ>1: sharper decay.")
            p1_ = k2.slider("p₁ — asymmetry", -3.0, 3.0, 0.0, 0.1,
                            help="Skews the peak toward low/high filling.")
            p2_ = k3.slider("p₂ — peak width", -4.0, 4.0, 0.0, 0.1,
                            help="Narrows (<0) or broadens (>0) the peak.")
            kin_shape = (g_, p1_, p2_)
        st.plotly_chart(kinetics_figure(kin_shape), use_container_width=True)

    fig = plot_curves({c: None for c in crates}, params, ocv, v_min=v_cut,
                      kin_shape=kin_shape)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download predicted curves (CSV)",
                       predictions_csv(params, crates, ocv, v_min=v_cut,
                                       kin_shape=kin_shape),
                       "lean_model_prediction.csv", "text/csv")


# ── 2. Fit ─────────────────────────────────────────────────────────────────

with tab_fit:
    st.subheader("Characterize an electrode from measured discharge curves")
    st.markdown(
        "Upload one CSV per C-rate (two columns: normalized capacity 0–1, "
        "voltage). The fit returns the electrode's dimensionless fingerprint: "
        "wiring group $Da_w$, electrolyte group $Da$, process group $Da_p$ "
        "(at 1C), series resistance, and usable-capacity fraction."
    )

    demo = st.toggle("Use demo data (HP-NMC111 half-cell, Ren et al. 2019)")

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if demo:
        ocv_soc, ocv_v, _ = parse_two_col_csv(os.path.join(SAMPLE_DIR, "OCV.csv"))
        ocv_fit = ocv_from_table(ocv_soc, ocv_v)
        for fn in ("0.5C.csv", "1C.csv", "2C.csv"):
            soc, v, _ = parse_two_col_csv(os.path.join(SAMPLE_DIR, fn))
            curves[crate_from_name(fn)] = (soc, v)
        st.success("Loaded demo data: OCV + discharge at 0.5C, 1C, 2C.")
    else:
        ocv_fit = ocv_selector("fit")
        ups = st.file_uploader(
            "Discharge CSVs (one per C-rate; name them like `0.5C.csv` for "
            "automatic rate detection)", type="csv", accept_multiple_files=True)
        for up in ups or []:
            guess = crate_from_name(up.name) or 1.0
            c = st.number_input(f"C-rate for `{up.name}`", 0.01, 100.0,
                                guess, key=f"rate_{up.name}")
            soc, v, normed = parse_two_col_csv(up)
            if normed:
                st.warning(f"`{up.name}`: capacity column rescaled by its max.")
            curves[c] = (soc, v)

    if curves:
        cM, cK = st.columns([2, 1])
        maxiter = cM.slider("Fit effort (dual-annealing iterations)",
                            100, 2000, 600, 100,
                            help="More iterations = more robust fit, slower.")
        learn_kin = cK.checkbox(
            "Learn kinetics shape f(c)",
            help="Also fit a 3-parameter deformation (γ, p₁, p₂) of the "
                 "CIET/MHC exchange-current prefactor instead of assuming "
                 "its pure form. Use when the residual is systematic in "
                 "mid-discharge. Slower; needs 3+ rates to be meaningful.")
        if st.button("Run fit", type="primary"):
            with st.spinner(f"Fitting {len(curves)} curve(s)…"):
                result = fit_descriptors(curves, ocv_fit, maxiter=maxiter,
                                         learn_kinetics=learn_kin)
            st.session_state["fit_result"] = result
            st.session_state["fit_curves"] = curves
            st.session_state["fit_ocv"] = ocv_fit

    if "fit_result" in st.session_state and curves:
        r = st.session_state["fit_result"]
        st.markdown("#### Fitted lean-model descriptors")
        names = ["Wiring group Da_w",
                 "Electrolyte group Da",
                 "Process group Da_p (at 1C)",
                 "Da/Da_p (at 1C)",
                 "Series resistance R_s (mV per C)",
                 "Start stoichiometry a",
                 "Usable capacity fraction",
                 "Voltage RMS (mV)"]
        vals = [f"{r['Da_w']:.3g}", f"{r['Da']:.3g}",
                f"{r['Da_p1']:.3g}", f"{r['Da'] / r['Da_p1']:.3g}",
                f"{r['R_s'] * 1e3:.1f}", f"{r['a']:.3f}",
                f"{r['frac']:.3f}", f"{r['rms'] * 1e3:.1f}"]
        if r.get("kin_shape") is not None:
            names += ["Kinetics γ (sharpness)", "Kinetics p₁ (asymmetry)",
                      "Kinetics p₂ (width)"]
            vals += [f"{r['gamma']:.3f}", f"{r['p1']:.3f}", f"{r['p2']:.3f}"]
        st.dataframe(pd.DataFrame({"Descriptor": names, "Value": vals}),
                     hide_index=True, use_container_width=True)

        # end model curves at the lowest measured voltage (the cell's cutoff)
        v_cut_fit = min(float(v.min())
                        for _, v in st.session_state["fit_curves"].values())
        fig = plot_curves(st.session_state["fit_curves"], tuple(r["x"]),
                          st.session_state["fit_ocv"],
                          title="Measured (points) vs lean model (lines)",
                          v_min=v_cut_fit, kin_shape=r.get("kin_shape"))
        st.plotly_chart(fig, use_container_width=True)
        if r.get("kin_shape") is not None:
            st.markdown("##### Learned exchange-current shape vs CIET/MHC")
            st.plotly_chart(kinetics_figure(r["kin_shape"]),
                            use_container_width=True)

        payload = {k: (float(v) if np.isscalar(v) else list(map(float, v)))
                   for k, v in r.items() if v is not None}
        st.download_button("Download fit results (JSON)",
                           json.dumps(payload, indent=2),
                           "lean_model_fit.json", "application/json")


# ── 3. Fit EIS ─────────────────────────────────────────────────────────────

with tab_eis:
    st.subheader("Fit an impedance spectrum with the analytical lean model")
    st.markdown(
        "Upload a CSV with three columns: **frequency (Hz)**, **Z′ (Ω)**, "
        "**Z″ (Ω)**. The analytical porous-electrode impedance is fit "
        "directly — no equivalent circuit — returning the ohmic resistance, "
        "the impedance scale, and the wiring ($Da_w$), process ($Da_p$), and "
        "capacitive ($Da_c$) groups."
    )

    demo_eis = st.toggle("Use demo data (synthetic lean-model spectrum)",
                         key="demo_eis")

    eis_data = None
    if demo_eis:
        f_, zr_, zi_, _ = parse_eis_csv(os.path.join(SAMPLE_DIR,
                                                     "EIS_demo.csv"))
        eis_data = (f_, zr_, zi_)
        st.success(f"Loaded demo spectrum ({len(f_)} points, "
                   f"{f_.min():.3g}–{f_.max():.3g} Hz).")
    else:
        up_eis = st.file_uploader("EIS CSV (frequency Hz, Z′ Ω, Z″ Ω)",
                                  type="csv", key="eis_file")
        if up_eis is not None:
            f_, zr_, zi_, flipped = parse_eis_csv(up_eis)
            if flipped:
                st.warning("Imaginary column was mostly positive — "
                           "interpreted as −Z″ and flipped.")
            eis_data = (f_, zr_, zi_)

    if eis_data is not None:
        c1, c2, c3 = st.columns(3)
        stoich = c1.slider("Filling fraction at measurement", 0.05, 0.95,
                           0.30, 0.01,
                           help="Local state of charge of the electrode "
                                "when the spectrum was taken.")
        ocv_src = c2.radio("OCV slope dU/dx from",
                           ["Built-in NMC532", "Upload OCV CSV"],
                           key="eis_ocv_src")
        maxiter_eis = c3.slider("Fit effort", 100, 1000, 300, 50,
                                key="eis_maxiter")
        if ocv_src.startswith("Built-in"):
            dudx = float(nmc532_ocv_deriv(stoich))
        else:
            up_ocv = st.file_uploader("OCV CSV (SoC 0–1, voltage V)",
                                      type="csv", key="eis_ocv_file")
            if up_ocv is None:
                st.info("Upload an OCV file to set dU/dx, or use the "
                        "built-in OCV.")
                st.stop()
            s_, v_, _ = parse_two_col_csv(up_ocv)
            dudx = float(ocv_derivative(ocv_from_table(s_, v_), stoich))
        st.caption(f"OCV slope at x = {stoich:.2f}:  dU/dx = {dudx:.3f} V")

        if st.button("Run EIS fit", type="primary", key="run_eis"):
            with st.spinner("Fitting impedance spectrum…"):
                r = fit_eis(*eis_data, stoich=stoich, dudx=dudx,
                            maxiter=maxiter_eis)
            st.session_state["eis_result"] = r
            st.session_state["eis_data"] = eis_data

        if "eis_result" in st.session_state:
            r = st.session_state["eis_result"]
            f_, zr_, zi_ = st.session_state["eis_data"]
            st.markdown("#### Fitted impedance descriptors")
            st.dataframe(pd.DataFrame({
                "Descriptor": ["Ohmic resistance R_Ω (Ω)",
                               "Impedance scale R_z (Ω)",
                               "Wiring group Da_w",
                               "Process group Da_p (at 1C)",
                               "Capacitive group Da_c (at 1C)",
                               "Conductivity ratio R_hf",
                               "Relative RMS (%)"],
                "Value": [f"{r['R_ohm']:.3g}", f"{r['R_z']:.3g}",
                          f"{r['Da_w']:.3g}", f"{r['Da_p1']:.3g}",
                          f"{r['Da_c1']:.3g}", f"{r['R_hf']:.3g}",
                          f"{100 * r['rel_rms']:.2f}"],
            }), hide_index=True, use_container_width=True)

            st.plotly_chart(nyquist_figure(f_, zr_, zi_, r["Z_fit"]),
                            use_container_width=True)

            payload = {k: float(v) for k, v in r.items() if np.isscalar(v)}
            st.download_button("Download EIS fit (JSON)",
                               json.dumps(payload, indent=2),
                               "lean_model_eis_fit.json", "application/json")


# ── 4. Damköhler calculator ────────────────────────────────────────────────

with tab_calc:
    st.subheader("Damköhler-number calculator")
    st.markdown("Convert dimensional cell parameters into the four "
                "lean-model groups at a chosen C-rate.")
    c1, c2, c3 = st.columns(3)
    L = c1.number_input("Thickness L (µm)", 1.0, 1000.0, 100.0,
                        key="calc_L") * 1e-6
    ap_mode = c2.radio("Interfacial area from", ["R_p (spheres)", "a_p direct"],
                       key="calc_apmode")
    if ap_mode.startswith("R_p"):
        Rp = c3.number_input("Particle radius R_p (µm)", 0.01, 100.0, 5.0,
                             key="calc_Rp") * 1e-6
        eps_am = c1.number_input("Active fraction ε_am", 0.05, 0.95, 0.665,
                                 key="calc_epsam")
        ap = 3.0 * eps_am / Rp
    else:
        ap = c3.number_input("a_p (1/m)", 1e3, 1e10, 1e6, format="%.3g",
                             key="calc_ap")
        eps_am = c1.number_input("Active fraction ε_am", 0.05, 0.95, 0.665,
                                 key="calc_epsam2")
    eps_p = c2.number_input("Porosity ε_p", 0.05, 0.9, 0.335, key="calc_epsp")
    sigma_s = c3.number_input("σ_s (S/m)", 1e-4, 1e4, 1.0, format="%.4g",
                              key="calc_sig")
    kappa_l = c1.number_input("κ_l (S/m)", 1e-3, 1e2, 0.95, format="%.4g",
                              key="calc_kap")
    D_ref = c2.number_input("D_ref (m²/s)", 1e-13, 1e-8, 2.5e-10,
                            format="%.3g", key="calc_D")
    t_plus = c3.number_input("t₊", 0.0, 1.0, 0.38, key="calc_tp")
    j0 = c1.number_input("j₀ (A/m²)", 1e-6, 1e3, 1.0, format="%.4g",
                         key="calc_j0")
    csmax = c2.number_input("c_s,max (mol/m³)", 1e3, 1e5, 50000.0,
                            format="%.5g", key="calc_cs")
    clref = c3.number_input("c_l,ref (mol/m³)", 100.0, 5000.0, 1000.0,
                            key="calc_cl")
    CDL = c1.number_input("C_DL (F/m²)", 1e-3, 10.0, 0.2, format="%.3g",
                          key="calc_cdl")
    T = c2.number_input("Temperature (K)", 250.0, 350.0, 298.15, key="calc_T")
    crate = c3.number_input("C-rate", 0.01, 100.0, 1.0, key="calc_crate")

    g = compute_da_numbers(L=L, ap=ap, eps_p=eps_p, eps_am=eps_am,
                           D_ref=D_ref, sigma_s=sigma_s, kappa_l=kappa_l,
                           t_plus=t_plus, j0=j0, C_DL=CDL, clref=clref,
                           csmax=csmax, crate=crate, T=T)
    st.dataframe(pd.DataFrame({
        "Quantity": ["Da (reaction/diffusion)", "Da_p (reaction/C-rate)",
                     "Da_w (reaction/wiring)", "Da_c (reaction/DL charging)",
                     "Da/Da_p (electrolyte limitation)",
                     "σ_eff (S/m)", "t_p (s)"],
        "Value": [f"{g['Da']:.4g}", f"{g['Da_p']:.4g}", f"{g['Da_w']:.4g}",
                  f"{g['Da_c']:.4g}", f"{g['Da'] / g['Da_p']:.4g}",
                  f"{g['sigma_eff']:.4g}", f"{g['t_p']:.4g}"],
    }), hide_index=True, use_container_width=True)
    if g["Da"] > g["Da_p"]:
        st.warning("Da > Da_p: the electrolyte is expected to be "
                   "rate-limiting at this C-rate.")
    else:
        st.success("Da ≤ Da_p: electrolyte transport is not rate-limiting "
                   "at this C-rate.")


# ── 5. About ───────────────────────────────────────────────────────────────

with tab_about:
    st.subheader("About the lean model")
    st.markdown(r"""
The **lean model** reduces porous-electrode theory (P2D/DFN) to a set of
closed-form expressions valid for *reaction-limited*, high-performance
electrodes. Four dimensionless Damköhler groups govern the response:

| Group | Meaning |
|---|---|
| $Da$ | reaction rate vs. electrolyte diffusion (electrolyte group) |
| $Da_p$ | reaction rate vs. applied C-rate (process group) |
| $Da_w$ | reaction rate vs. ionic+electronic wiring (wiring group) |
| $Da_c$ | reaction rate vs. double-layer charging (capacitive group) |

The galvanostatic discharge voltage follows the leading-order solution

$$ V = U(x) \;-\; V_T\,\frac{\Lambda^2\,Z(\Lambda,\beta)}{Da_p\, f(x)} \;-\; R_s\,\mathrm{C}, \qquad \Lambda^2 = f\,Da_w + \alpha\,\frac{Da}{Da_p}, $$

with the exchange-current prefactor $f(x)$ from coupled ion–electron
transfer (CIET/MHC) kinetics.

**Fitting** returns an electrode's dimensionless fingerprint from routine
rate data ($V$–$Q$ curves at 2–3 C-rates plus a low-rate OCV), separating
wiring, electrolyte, kinetic, and capacity-utilization limitations without
running a PDE solver.

**EIS** is fit with the same physics: the linearized lean model yields an
analytical porous-electrode impedance (no equivalent circuit), whose arc is
set by $Da_w$ and $Da_c$ and whose low-frequency tail follows from $Da_p$
and the local OCV slope $dU/dx$.

**Reference:** S. Pathak and M. Z. Bazant, *Scaling and Analytical
Approximation of Porous Electrode Theory for Reaction-limited Batteries*,
Journal of The Electrochemical Society (2026).

Demo dataset: NMC-111 half-cell discharge digitized from Ren et al.,
*ACS Appl. Mater. Interfaces* 11, 41178 (2019).
""")
