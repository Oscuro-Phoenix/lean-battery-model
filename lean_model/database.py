"""Public community database of characterized cells.

Every cell (reference literature electrodes + user-submitted fits) is stored
as one row of dimensionless descriptors: (Da_w, Da, Da_p, Da_c, ...). This
powers the "Community Database" page, which plots every submitted cell in
Damkohler space alongside the lean-model validity/electrolyte-limitation
regions from the paper.

Persistence
-----------
Submissions are always appended to a local CSV (``community_data/cells.csv``
next to this package). On Streamlit Community Cloud that file resets on
reboot/redeploy, so for durable, genuinely public storage this module can
optionally commit the updated CSV back to a GitHub repo via the Contents
API -- configure it with Streamlit secrets:

    [github]
    token = "ghp_..."             # repo-scoped personal access token
    repo  = "owner/name"          # e.g. "Oscuro-Phoenix/lean-battery-model"
    path  = "community_data/cells.csv"
    branch = "main"

If no such secrets are configured, the app still works with local-only
persistence (submissions survive within the running instance / local dev,
but not across a cloud redeploy) and shows a note to that effect.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import uuid

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with streamlit
    requests = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CSV = os.path.join(ROOT, "community_data", "cells.csv")

COLUMNS = [
    "id", "timestamp", "contributor", "cell_name", "chemistry", "method",
    "source_note", "crate_ref", "a", "Da_w", "Da", "Da_p", "Da_c", "R_s",
    "frac", "rms_mV", "is_reference",
]

# Literature reference electrodes from the manuscript's design-envelope
# figure, computed with this package's own compute_da_numbers so they are
# on an identical dimensionless footing to user submissions.
SEED_ELECTRODES = [
    dict(cell_name="NMC811", chemistry="cathode", crate_ref=1.0,
        Da_w=2.36953, Da=0.538396, Da_p=1.15955, Da_c=1.96119e6,
        source_note="Chen et al., J. Electrochem. Soc. 167, 080534 (2020)"),
    dict(cell_name="Graphite (LG M50)", chemistry="anode", crate_ref=1.0,
        Da_w=0.204169, Da=0.111642, Da_p=0.195709, Da_c=237845.0,
        source_note="Chen et al., J. Electrochem. Soc. 167, 080534 (2020)"),
    dict(cell_name="Graphite (LFP cell)", chemistry="anode", crate_ref=1.0,
        Da_w=0.0596722, Da=0.014365, Da_p=0.366338, Da_c=350314.0,
        source_note="Prada et al., J. Electrochem. Soc. 160, A616 (2013)"),
    dict(cell_name="LFP", chemistry="cathode", crate_ref=1.0,
        Da_w=2.89433, Da=0.465878, Da_p=4.90812, Da_c=35031.4,
        source_note="Prada et al., J. Electrochem. Soc. 160, A616 (2013)"),
    dict(cell_name="LTO", chemistry="anode", crate_ref=1.0,
        Da_w=208.219, Da=20.2082, Da_p=683.052, Da_c=2.43698e7,
        source_note="Kashkooli et al., Electrochim. Acta 196, 33 (2016)"),
    dict(cell_name="alpha-MnO2 pseudocapacitor", chemistry="pseudocapacitor",
        crate_ref=10.0, Da_w=14.0699, Da=2.90201, Da_p=2.23641, Da_c=1996.24,
        source_note="Guillemet et al., Electrochim. Acta 67, 41 (2012)"),
]


def _seed_dataframe() -> pd.DataFrame:
    rows = []
    for e in SEED_ELECTRODES:
        row = {c: None for c in COLUMNS}
        row.update(e)
        row["id"] = "seed-" + e["cell_name"].lower().replace(" ", "-")
        row["timestamp"] = None
        row["contributor"] = "Pathak & Bazant (manuscript)"
        row["method"] = "literature"
        row["a"] = None
        row["R_s"] = None
        row["frac"] = None
        row["rms_mV"] = None
        row["is_reference"] = True
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _coerce_is_reference(df: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips turn the bool column into strings ('True'/'False');
    normalize back to real booleans so ``~df['is_reference']`` behaves."""
    if "is_reference" in df.columns:
        df["is_reference"] = (
            df["is_reference"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False}).fillna(False))
    return df


def _load_local() -> pd.DataFrame:
    if not os.path.exists(LOCAL_CSV):
        return _empty_dataframe()
    try:
        df = pd.read_csv(LOCAL_CSV)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        return _coerce_is_reference(df[COLUMNS])
    except Exception:
        return _empty_dataframe()


def _save_local(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)


def _github_config():
    """Return (token, repo, path, branch) from st.secrets, or None if unset."""
    try:
        import streamlit as st
        gh = st.secrets.get("github")
    except Exception:
        gh = None
    if not gh or not gh.get("token") or not gh.get("repo"):
        return None
    return (gh["token"], gh["repo"], gh.get("path", "community_data/cells.csv"),
            gh.get("branch", "main"))


def github_configured() -> bool:
    return _github_config() is not None


def _github_fetch() -> pd.DataFrame | None:
    """Fetch the current community CSV from GitHub, or None on any failure."""
    cfg = _github_config()
    if cfg is None or requests is None:
        return None
    token, repo, path, branch = cfg
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(api, headers=headers, params={"ref": branch}, timeout=10)
        if r.status_code != 200:
            return None
        raw = base64.b64decode(r.json()["content"])
        df = pd.read_csv(io.BytesIO(raw))
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        return _coerce_is_reference(df[COLUMNS])
    except Exception:
        return None


def _github_sync(df: pd.DataFrame) -> tuple[bool, str]:
    """Commit the community CSV to GitHub via the Contents API, if configured."""
    cfg = _github_config()
    if cfg is None or requests is None:
        return False, "GitHub persistence not configured."
    token, repo, path, branch = cfg
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    sha = None
    try:
        r = requests.get(api, headers=headers, params={"ref": branch}, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception as exc:
        return False, f"GitHub read failed: {exc}"
    payload = {
        "message": "Add community cell submission",
        "content": content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(api, headers=headers, data=json.dumps(payload), timeout=10)
        if r.status_code in (200, 201):
            return True, "Synced to GitHub."
        return False, f"GitHub write failed ({r.status_code}): {r.text[:200]}"
    except Exception as exc:
        return False, f"GitHub write failed: {exc}"


def _load_community() -> pd.DataFrame:
    """Community submissions: GitHub is the source of truth when configured
    (so the database survives cloud reboots/redeploys); the local CSV is a
    cache and the fallback when GitHub is unreachable or unconfigured."""
    if github_configured():
        df = _github_fetch()
        if df is not None:
            _save_local(df)
            return df
    return _load_local()


def load_database(include_reference: bool = True) -> pd.DataFrame:
    """Full community database: seed literature electrodes + submissions."""
    community = _load_community()
    if not include_reference:
        return community
    if community.empty:
        return _seed_dataframe()
    return pd.concat([_seed_dataframe(), community], ignore_index=True)


def append_entry(entry: dict) -> tuple[bool, str]:
    """Append one submission to the database (local CSV, and GitHub if
    configured -- see module docstring for the secrets needed).

    ``entry`` should contain at least: contributor, cell_name, chemistry,
    method, source_note, crate_ref, Da_w, Da, Da_p; other COLUMNS are
    optional and default to None.
    """
    row = {c: entry.get(c) for c in COLUMNS}
    row["id"] = uuid.uuid4().hex[:12]
    row["timestamp"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row["is_reference"] = False

    df = _load_community()
    new_row = pd.DataFrame([row], columns=COLUMNS)
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    _save_local(df)

    if github_configured():
        ok, msg = _github_sync(df)
        return ok, msg
    return True, ("Saved locally. Configure GitHub secrets for durable, "
                  "publicly-shared persistence across app restarts.")
