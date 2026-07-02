"""Open-circuit voltage sources: built-in NMC532 polynomial or user table."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def nmc532_ocv(x):
    """NMC532 OCV polynomial from Colclasure et al. 2020 (stoichiometry x)."""
    x = np.asarray(x, dtype=float)
    return (
        5.314735633000300e+00
        - 3.640117692001490e+03 * x ** 14.0
        + 1.317657544484270e+04 * x ** 13.0
        - 1.455742062291360e+04 * x ** 12.0
        - 1.571094264365090e+03 * x ** 11.0
        + 1.265630978512400e+04 * x ** 10.0
        - 2.057808873526350e+03 * x ** 9.0
        - 1.074374333186190e+04 * x ** 8.0
        + 8.698112755348720e+03 * x ** 7.0
        - 8.297904604107030e+02 * x ** 6.0
        - 2.073765547574810e+03 * x ** 5.0
        + 1.190223421193310e+03 * x ** 4.0
        - 2.724851668445780e+02 * x ** 3.0
        + 2.723409218042130e+01 * x ** 2.0
        - 4.158276603609060e+00 * x
        - 5.573191762723310e-04
        * np.exp(6.560240842659690e+00 * x ** 4.148209275061330e+01)
    )


def ocv_from_table(soc, voltage):
    """Linear-interpolation OCV(soc) from tabulated (soc, V) data."""
    soc = np.asarray(soc, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    order = np.argsort(soc)
    return interp1d(soc[order], voltage[order], kind="linear",
                    bounds_error=False, fill_value="extrapolate")
