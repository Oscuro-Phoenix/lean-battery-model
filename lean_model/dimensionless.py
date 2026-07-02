"""Dimensional cell parameters -> lean-model Damkohler groups."""

from __future__ import annotations

F_CONST = 96485.0  # Faraday constant [C/mol]
RG = 8.314         # gas constant [J/mol/K]


def compute_da_numbers(L, ap, eps_p, eps_am, D_ref, sigma_s, kappa_l,
                       t_plus, j0, C_DL, clref, csmax, crate, T=298.15):
    """Return the lean-model dimensionless groups as a dict.

    All inputs SI. Definitions follow the manuscript:

        Da   = L^2 j0 ap (1 - t+) / (F eps_p D_ref c_l,ref)
        Da_p = t_p j0 ap / (eps_am F c_s,max)
        Da_w = L^2 j0 ap / (V_T sigma_eff)
        Da_c = j0 t_p / (V_T C_DL)

    with the series effective conductivity

        sigma_eff = [1/sigma_s + 1/kappa_l
                     + (2 V_T / (L^2 j0 ap)) (1 - t+) Da]^-1.
    """
    thermal_v = RG * T / F_CONST
    t_p = 3600.0 / crate
    react_rate = j0 * ap  # exchange current per volume [A/m^3]

    da = (L ** 2) * react_rate * (1.0 - t_plus) / (F_CONST * eps_p * D_ref * clref)
    da_p = t_p * react_rate / (eps_am * F_CONST * csmax)

    sigma_eff = 1.0 / (
        1.0 / sigma_s
        + 1.0 / kappa_l
        + (2.0 * thermal_v / ((L ** 2) * react_rate)) * (1.0 - t_plus) * da
    )

    da_w = (L ** 2) * react_rate / (thermal_v * sigma_eff)
    da_w_sigma = (L ** 2) * react_rate / (thermal_v * sigma_s)
    da_w_kappa = (L ** 2) * react_rate / (thermal_v * kappa_l)
    da_c = j0 * t_p / (thermal_v * C_DL)

    return {
        "t_p": t_p,
        "thermal_v": thermal_v,
        "sigma_eff": sigma_eff,
        "Da": da,
        "Da_p": da_p,
        "Da_w": da_w,
        "Da_w_sigma": da_w_sigma,
        "Da_w_kappa": da_w_kappa,
        "Da_c": da_c,
    }
