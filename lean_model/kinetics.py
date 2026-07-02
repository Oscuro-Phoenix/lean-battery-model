"""CIET / Marcus-Hush-Chidsey (MHC) exchange-current kinetics (NumPy only)."""

from __future__ import annotations

import numpy as np
import scipy.special as spl

V_T = 0.0257  # thermal voltage at 298.15 K [V]

# Reorganization energy in thermal-voltage units (lambda / V_T)
_LAMBDA_DIM = 0.112 / 0.0257


def _activity_correction(c_lyte):
    """Electrolyte activity correction  c~ = c*1.9*e^-1 / (1 + c*1.9*e^-1)."""
    fac = c_lyte * 1.9 * np.exp(-1)
    return fac / (1.0 + fac)


def ecd_mhc(c_sld, c_lyte=1.0, k0=5e-6, R_film=0.0):
    """Dimensionless exchange-current-density factor f for CIET/MHC kinetics.

    Parameters
    ----------
    c_sld : array_like
        Dimensionless solid filling fraction, in (0, 1).
    c_lyte : float
        Dimensionless electrolyte concentration (default 1).
    k0 : float
        Rate-constant prefactor [A m^-2] (only enters via the film term).
    R_film : float
        Film resistance [Ohm m^2].
    """
    c_sld = np.asarray(c_sld, dtype=float)
    c_lyte = _activity_correction(c_lyte)
    eta = np.log(c_lyte / c_sld)
    a = 1.0 + np.sqrt(_LAMBDA_DIM)
    erf_term = 1.0 - spl.erf(
        (_LAMBDA_DIM - np.sqrt(a + eta ** 2)) / (2.0 * np.sqrt(_LAMBDA_DIM))
    )
    f = (1.0 - c_sld) * c_sld * c_lyte / (c_sld + c_lyte) * erf_term / 2.0
    return f / (k0 * f * R_film / V_T + 1.0)


def ecd_mhc_df_dclyte(c_sld, c_lyte=1.0, k0=5e-6, R_film=0.0):
    """Log-derivative alpha of the electrolyte-concentration factor.

    Used for the electrolyte-transport correction in the V-Q model.
    """
    c_sld = np.asarray(c_sld, dtype=float)
    c_lyte_corr = _activity_correction(c_lyte)
    g = (
        (1.0 / (c_sld + c_lyte_corr)
         - c_lyte_corr / (c_sld + c_lyte_corr) ** 2)
        / (c_lyte_corr / (c_sld + c_lyte_corr))
    )
    return g
