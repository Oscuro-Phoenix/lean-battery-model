"""Analytical lean-model voltage-capacity (V-Q) prediction."""

from __future__ import annotations

import numpy as np

from lean_model.kinetics import V_T, ecd_mhc, ecd_mhc_df_dclyte


def predict_vq(ocv_function, X, Da_w, Da_w_sigma, Da_w_kappa, Da, Da_p,
               drop_saturated=False):
    """Leading-order galvanostatic discharge voltage of the lean model.

    Parameters
    ----------
    ocv_function : callable
        OCV(x) in volts for stoichiometry x in (0, 1).
    X : array_like
        Stoichiometry (filling-fraction) grid.
    Da_w, Da_w_sigma, Da_w_kappa : float
        Total wiring Damkohler number and its electronic/ionic split
        (Da_w = Da_w_sigma + Da_w_kappa).
    Da : float
        Electrolyte (traditional) Damkohler number.
    Da_p : float
        Process Damkohler number at the applied C-rate.
    drop_saturated : bool
        If True, return NaN where the surface has saturated (ecd below
        threshold) instead of snapping back to the OCV; use for plotting.

    Returns
    -------
    V : ndarray
        Cell voltage [V] at each stoichiometry.
    """
    X = np.asarray(X, dtype=float)
    ec = ecd_mhc(X)
    alpha = ecd_mhc_df_dclyte(X)

    Lambda = np.sqrt(Da_w * ec + alpha * Da / Da_p)
    beta = Da_w_sigma / Da_w if Da_w != 0 else 0.5

    Xi = np.zeros_like(X)
    mask = ec > 1e-10
    Lam = Lambda[mask]
    Z = (Lam ** 2) * (
        2.0 * beta * (1.0 - beta) * (0.5 + 1.0 / (Lam * np.sinh(Lam)))
        + (beta ** 2 + (1.0 - beta) ** 2) * np.cosh(Lam) / (Lam * np.sinh(Lam))
    )
    Xi[mask] = Z / (Da_p * ec[mask])

    V = ocv_function(X) - np.abs(Xi) * V_T
    if drop_saturated:
        V = np.where(mask, V, np.nan)
    return V


def model_V(soc, params, crate, ocv, drop_saturated=False, v_min=None):
    """Lean-model voltage at normalized capacity points `soc` for C-rate `crate`.

    ``params`` is the 7-tuple used throughout the app and the fitter:

        a      start-of-discharge stoichiometry (soc -> cs = a + span*soc)
        Da_w   total wiring Damkohler number
        beta   electronic split, beta = Da_w_sigma / Da_w  (in [0, 1/2])
        Da     electrolyte Damkohler number
        Da_p1  process Damkohler number at 1C (Da_p = Da_p1 / crate)
        R_s    lumped series resistance, ohmic drop proportional to C [V/C]
        frac   usable fraction of the measured capacity (cs -> 1 at soc = frac)

    If ``v_min`` is given, voltages below the cutoff are masked to NaN so the
    discharge terminates at the cutoff instead of diving indefinitely.
    """
    a, Da_w, beta, Da, Da_p1, R_s, frac = params
    span = (1.0 - a) / frac
    cs = np.clip(a + span * np.asarray(soc, float), 1e-9, 1.0 - 1e-9)
    Da_p = Da_p1 / crate
    Da_w_s, Da_w_k = beta * Da_w, (1.0 - beta) * Da_w

    def ocv_of_cs(c):
        return ocv(np.clip((c - a) / span, 0.0, 1.0))

    V = predict_vq(ocv_of_cs, cs, Da_w, Da_w_s, Da_w_k, Da, Da_p,
                   drop_saturated=drop_saturated)
    V = V - R_s * crate
    if v_min is not None:
        V = np.where(V >= v_min, V, np.nan)
    return V
