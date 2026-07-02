# Lean Battery Model — web app

Interactive web app around the analytical **lean porous-electrode model** of
Pathak & Bazant, *Scaling and Analytical Approximation of Porous Electrode
Theory for Reaction-limited Batteries* (J. Electrochem. Soc.).

Predict galvanostatic discharge curves from four dimensionless Damköhler
groups, or fit measured rate data to extract an electrode's dimensionless
fingerprint — no PDE solver required.

## Features

- **Predict discharge** — enter either the dimensionless groups
  (`Da_w`, `Da`, `Da_p`, series resistance, usable-capacity fraction) or
  dimensional cell parameters (thickness, particle size, conductivities,
  kinetics, …) and get V–Q curves at any C-rates. Built-in NMC532 OCV or
  upload your own. An advanced panel lets you swap the pure CIET/MHC
  exchange-current prefactor for a custom `f(c)` shape with a live preview.
- **Fit discharge data** — upload discharge CSVs at 2–3 C-rates plus an OCV
  reference; a global (dual-annealing) fit returns the lean descriptors
  (`Da_w`, `Da`, `Da_p` at 1C, `R_s`, start stoichiometry, usable-capacity
  fraction) and the voltage RMS. Optionally **learn the exchange-current
  shape** `f(c)`: a 3-parameter deformation (sharpness γ, asymmetry p₁,
  width p₂) of the CIET/MHC prefactor, fitted alongside the descriptors and
  compared against the pure form graphically. Includes a demo dataset
  (HP-NMC111 half-cell, digitized from Ren et al. 2019).
- **Fit EIS** — upload an impedance spectrum (frequency, Z′, Z″) and fit the
  *analytical* lean-model impedance directly (no equivalent circuit),
  returning `R_Ω`, the impedance scale, `Da_w`, `Da_p`, `Da_c`, and the
  conductivity ratio, with an interactive Nyquist overlay. A synthetic demo
  spectrum is included.
- **Damköhler calculator** — dimensional parameters → all four groups plus
  the effective conductivity and an electrolyte-limitation check
  (`Da` vs `Da_p`).

All charts are interactive (Plotly): hover for values, zoom into arcs and
end-of-discharge knees.

## Data format

Discharge/OCV: CSV with two columns, **normalized capacity (0–1)** and
**voltage (V)**. A header row is optional. If the capacity column exceeds
1.5 the app assumes raw capacity (e.g. mAh/g) and rescales by the maximum.
Name discharge files like `0.5C.csv`, `1C.csv`, `2C.csv` for automatic
C-rate detection.

EIS: CSV with three columns, **frequency (Hz)**, **Z′ (Ω)**, **Z″ (Ω)**
(a mostly-positive imaginary column is auto-interpreted as −Z″).

Sample files are in `sample_data/`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Host online (free)

**Streamlit Community Cloud** (easiest):

1. Push this folder to a GitHub repository (it is fully self-contained).
2. Go to https://share.streamlit.io, sign in with GitHub, click
   **New app**, select the repo and `app.py`, and deploy.
3. You get a public `*.streamlit.app` URL to share.

**Hugging Face Spaces**: create a new Space with the *Streamlit* SDK and
push these files; the Space builds from `requirements.txt` automatically.

**Docker / any server**:

```bash
docker run -p 8501:8501 -v "$PWD":/app -w /app python:3.12-slim \
    sh -c "pip install -r requirements.txt && streamlit run app.py --server.address 0.0.0.0"
```

## Project layout

```
app.py                     Streamlit UI (predict / fit V-Q / fit EIS /
                           calculator / about)
lean_model/
  kinetics.py              CIET/MHC exchange-current factor f(c), learnable
                           shape variant, electrolyte log-derivative
  model.py                 leading-order analytical V-Q solution (predict_vq)
  eis.py                   analytical impedance + EIS fitter
  dimensionless.py         dimensional parameters -> Da, Da_p, Da_w, Da_c
  ocv.py                   built-in NMC532 OCV (+derivative) and
                           tabulated-OCV interpolant
  fitting.py               dual-annealing fit of the 7 lean descriptors
sample_data/               demo half-cell data (Ren et al. 2019, HP-NMC111)
                           + synthetic EIS spectrum
requirements.txt
```

## The model in one line

Galvanostatic voltage:
`V = U(x) − V_T · Λ² Z(Λ, β) / (Da_p f(x)) − R_s·C`, with
`Λ² = f·Da_w + α·Da/Da_p` and `f(x)` the CIET/MHC exchange-current factor.

## Citation

S. Pathak and M. Z. Bazant, *Scaling and Analytical Approximation of Porous
Electrode Theory for Reaction-limited Batteries*, Journal of The
Electrochemical Society (2026).
