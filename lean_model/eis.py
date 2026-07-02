"""Analytical lean-model electrochemical impedance and EIS fitting.

The impedance follows the linearized lean-model solution: a porous-electrode
(transmission-line-like) response with CIET/MHC charge-transfer kinetics,
double-layer charging through the capacitive group Da_c, and a low-frequency
differential-capacity tail through Da_p and the local OCV slope dU/dx.

Da_p and Da_c are referenced to t_p = 3600 s (i.e. "at 1C"), consistent with
the V-Q fitting convention; the physical impedance is independent of this
reference because only t_p-free combinations enter.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import dual_annealing, least_squares

from lean_model.kinetics import V_T, ecd_mhc

T_P_REF = 3600.0  # reference process time [s] (1C)

EIS_PARAM_NAMES = ["R_ohm", "R_z", "Da_w", "Da_p1", "Da_c1", "R_hf"]


def _coth(x):
    return 1.0 / np.tanh(x)


def _inv_x_sinh(x):
    """1/(x sinh x), safe against overflow for large |Re x| (limit -> 0)."""
    x = np.asarray(x, dtype=complex)
    out = np.zeros_like(x)
    ok = np.abs(np.real(x)) < 300.0
    out[ok] = 1.0 / (x[ok] * np.sinh(x[ok]))
    return out


def eis_model(freq_hz, R_ohm, R_z, Da_w, Da_p1, Da_c1, R_hf, stoich, dudx):
    """Complex lean-model impedance Z(f) [Ohm].

    Parameters
    ----------
    freq_hz : array_like
        Frequencies [Hz].
    R_ohm : float
        High-frequency series (ohmic) resistance [Ohm].
    R_z : float
        Impedance scale of the porous-electrode response [Ohm]
        (lumps geometry, area, and exchange current).
    Da_w, Da_p1, Da_c1 : float
        Wiring group, process group at 1C, capacitive group at 1C.
    R_hf : float
        Conductivity ratio sigma_eff/kappa_eff entering the high-frequency
        distribution (0 = one phase perfectly conducting).
    stoich : float
        Local filling fraction (state of charge) of the measurement.
    dudx : float
        OCV slope dU/dx [V] at `stoich` (negative for a discharging cathode).
    """
    omega_t = 2.0 * np.pi * np.asarray(freq_hz, dtype=float) * T_P_REF
    f = float(ecd_mhc(stoich))
    dphidc = dudx / V_T

    Lam = np.sqrt(
        Da_w * (f + 1j * omega_t / Da_c1 - dphidc * Da_p1 * f / Da_c1)
        / (1.0 - dphidc * Da_p1 * f / (1j * omega_t))
    )
    G = (
        R_hf * (1.0 + 2.0 * _inv_x_sinh(Lam))
        + (1.0 + R_hf ** 2) * _coth(Lam) / Lam
    ) / (1.0 + R_hf) ** 2
    return R_ohm + R_z * G


def fit_eis(freq_hz, z_real, z_imag, stoich, dudx, maxiter=400, seed=1):
    """Fit the lean-model impedance to measured EIS data.

    Parameters
    ----------
    freq_hz, z_real, z_imag : array_like
        Frequency [Hz] and complex impedance components [Ohm]
        (z_imag negative for capacitive behavior).
    stoich : float
        Filling fraction at which the spectrum was measured.
    dudx : float
        OCV slope dU/dx [V] at `stoich`.
    maxiter : int
        dual_annealing iteration budget.

    Returns
    -------
    result : dict
        Fitted parameters by name, plus "rel_rms" (relative residual) and
        "Z_fit" (complex model impedance at the data frequencies).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    z_data = np.asarray(z_real, dtype=float) + 1j * np.asarray(z_imag, float)
    scale = float(np.abs(z_data).max())

    def unpack(p):
        r_ohm, l_rz, l_daw, l_dap, l_dac, r_hf = p
        return r_ohm, 10.0 ** l_rz, 10.0 ** l_daw, 10.0 ** l_dap, \
            10.0 ** l_dac, r_hf

    def objective(p):
        r_ohm, r_z, daw, dap, dac, r_hf = unpack(p)
        try:
            zm = eis_model(freq_hz, r_ohm, r_z, daw, dap, dac, r_hf,
                           stoich, dudx)
        except Exception:
            return 1e6
        if not np.all(np.isfinite(zm)):
            return 1e6
        r = np.abs(zm - z_data) / (np.abs(z_data) + 1e-3 * scale)
        return float(np.mean(r ** 2))

    bounds = [
        (0.0, scale),                                    # R_ohm
        (np.log10(scale) - 4.0, np.log10(scale) + 1.5),  # log10 R_z
        (-2.0, 5.0),                                     # log10 Da_w
        (-1.0, 5.0),                                     # log10 Da_p1
        (1.0, 9.0),                                      # log10 Da_c1
        (0.0, 1.0),                                      # R_hf
    ]

    def residuals(p):
        r_ohm, r_z, daw, dap, dac, r_hf = unpack(p)
        zm = eis_model(freq_hz, r_ohm, r_z, daw, dap, dac, r_hf,
                       stoich, dudx)
        zm = np.where(np.isfinite(zm), zm, 1e6)
        w = 1.0 / (np.abs(z_data) + 1e-3 * scale)
        return np.concatenate([(zm.real - z_data.real) * w,
                               (zm.imag - z_data.imag) * w])

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    best_x, best_f = None, np.inf
    for s in (seed, seed + 1, seed + 2):
        res = dual_annealing(objective, bounds, maxiter=maxiter, seed=s)
        # local least-squares polish from the annealing solution
        try:
            pol = least_squares(residuals, np.clip(res.x, lo, hi),
                                bounds=(lo, hi))
            x, fval = pol.x, objective(pol.x)
        except Exception:
            x, fval = res.x, res.fun
        if min(fval, res.fun) < best_f:
            best_x, best_f = (x if fval <= res.fun else res.x), \
                min(fval, res.fun)
    r_ohm, r_z, daw, dap, dac, r_hf = unpack(best_x)
    z_fit = eis_model(freq_hz, r_ohm, r_z, daw, dap, dac, r_hf, stoich, dudx)
    rel_rms = float(np.sqrt(np.mean(
        (np.abs(z_fit - z_data) / (np.abs(z_data) + 1e-12)) ** 2)))
    return {
        "R_ohm": r_ohm, "R_z": r_z, "Da_w": daw, "Da_p1": dap,
        "Da_c1": dac, "R_hf": r_hf, "rel_rms": rel_rms, "Z_fit": z_fit,
    }
