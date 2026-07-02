"""Lean porous-electrode model: analytical V-Q prediction and data fitting.

Self-contained implementation of the "lean model" of Pathak & Bazant,
*Scaling and Analytical Approximation of Porous Electrode Theory for
Reaction-limited Batteries* (J. Electrochem. Soc.).
"""

from lean_model.kinetics import (
    ecd_mhc, ecd_mhc_df_dclyte, ecd_mhc_learnable, V_T,
)
from lean_model.model import predict_vq, model_V
from lean_model.dimensionless import compute_da_numbers
from lean_model.ocv import (
    nmc532_ocv, nmc532_ocv_deriv, ocv_derivative, ocv_from_table,
)
from lean_model.fitting import fit_descriptors, plain_rms, FIT_BOUNDS
from lean_model.eis import eis_model, fit_eis

__all__ = [
    "ecd_mhc", "ecd_mhc_df_dclyte", "ecd_mhc_learnable", "V_T",
    "predict_vq", "model_V",
    "compute_da_numbers",
    "nmc532_ocv", "nmc532_ocv_deriv", "ocv_derivative", "ocv_from_table",
    "fit_descriptors", "plain_rms", "FIT_BOUNDS",
    "eis_model", "fit_eis",
]
