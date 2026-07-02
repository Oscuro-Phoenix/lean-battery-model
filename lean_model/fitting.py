"""Fit lean-model descriptors to measured discharge curves."""

from __future__ import annotations

import numpy as np
from scipy.optimize import dual_annealing

from lean_model.model import model_V

# (a, Da_w, beta, Da, Da_p1, R_s, frac)
FIT_BOUNDS = [
    (0.05, 0.55),   # a: start-of-discharge stoichiometry
    (1e-1, 5e3),    # Da_w (total wiring group)
    (0.0, 0.5),     # beta = Da_w_sigma / Da_w (sigma >> kappa branch)
    (1e-2, 5e3),    # Da (electrolyte group)
    (1e-1, 5e3),    # Da_p at 1C
    (0.0, 0.25),    # R_s [V per C]
    (0.70, 1.0),    # frac: usable fraction of measured capacity
]

PARAM_NAMES = ["a", "Da_w", "beta", "Da", "Da_p1", "R_s", "frac"]


def _objective(p, curves, ocv, w_tail=1.0):
    a, Da_w, beta, Da, Da_p1, R_s, frac = p
    if not (0 < a < 0.8 and Da_w > 0 and 0 <= beta <= 0.5 and Da > 0
            and Da_p1 > 0 and R_s >= 0 and 0 < frac <= 1.0):
        return 1e6
    tot, n = 0.0, 0
    for crate, (soc, V) in curves.items():
        try:
            Vp = model_V(soc, p, crate, ocv)
        except Exception:
            return 1e6
        if not np.all(np.isfinite(Vp)):
            return 1e6
        # gently up-weight the low-voltage tail (end of discharge)
        w = 1.0 + w_tail * (V.max() - V) / max(V.max() - V.min(), 1e-3)
        tot += np.sum(w * (Vp - V) ** 2)
        n += np.sum(w)
    return tot / max(n, 1)


def plain_rms(p, curves, ocv):
    """Unweighted voltage RMS residual [V] over all curves."""
    se, n = 0.0, 0
    for crate, (soc, V) in curves.items():
        r = model_V(soc, p, crate, ocv) - V
        se += np.sum(r * r)
        n += r.size
    return np.sqrt(se / n)


def fit_descriptors(curves, ocv, maxiter=600, seed=1, callback=None):
    """Fit the seven lean-model descriptors to measured discharge data.

    Parameters
    ----------
    curves : dict
        {crate: (soc_array, voltage_array)} with soc normalized to [0, 1].
    ocv : callable
        OCV(soc) interpolant over the same normalized capacity axis.
    maxiter : int
        dual_annealing iteration budget (larger = slower but more robust).
    callback : callable or None
        Optional dual_annealing callback(x, f, context) for progress.

    Returns
    -------
    result : dict
        Fitted parameters by name, plus "rms" [V] and the raw vector "x".
    """
    res = dual_annealing(_objective, FIT_BOUNDS, args=(curves, ocv),
                         maxiter=maxiter, seed=seed, callback=callback)
    out = dict(zip(PARAM_NAMES, res.x))
    out["rms"] = plain_rms(res.x, curves, ocv)
    out["x"] = res.x
    return out
