"""Public community database of characterized cells.

Every cell (reference literature electrodes + user-submitted fits) is stored
as one row of dimensionless descriptors: (Da_w, Da, Da_p, Da_c, ...). This
powers the "Community Database" page, which plots every submitted cell in
Damkohler space alongside the lean-model validity/electrolyte-limitation
regions from the paper.

Persistence
-----------
Submissions are always appended to a local CSV (``community_data/cells.csv``
next to this package) as a fallback/cache. For a genuinely durable, public
database that survives Streamlit Community Cloud redeploys, point this
module at any SQLAlchemy-compatible database (a free Postgres instance from
Supabase, Neon, or Render works well) via Streamlit's native SQL connection
secrets:

    [connections.cells_db]
    url = "postgresql://user:password@host:5432/dbname"

or, equivalently, discrete fields (dialect/host/port/database/username/
password) -- see https://docs.streamlit.io/develop/tutorials/databases.
With that configured, ``st.connection("cells_db", type="sql")`` picks it up
automatically; the community table is created on first use. Without it, the
app still works with local-only persistence and shows a note to that effect.
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid

import pandas as pd

TABLE = "community_cells"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CSV = os.path.join(ROOT, "community_data", "cells.csv")

COLUMNS = [
    "id", "timestamp", "contributor", "cell_name", "chemistry", "method",
    "source_note", "crate_ref", "a", "Da_w", "Da", "Da_p", "Da_c", "R_s",
    "frac", "rms_mV", "is_reference",
]
DB_COLUMNS = [c for c in COLUMNS if c != "is_reference"]

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
        row["is_reference"] = True
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _coerce_is_reference(df: pd.DataFrame) -> pd.DataFrame:
    """CSV/SQL round-trips can turn the bool column into strings; normalize
    back to real booleans so ``~df['is_reference']`` behaves."""
    if "is_reference" in df.columns:
        df["is_reference"] = (
            df["is_reference"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False))
    return df


# ── local CSV fallback ───────────────────────────────────────────────────

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


def _append_local(row: dict) -> None:
    df = _load_local()
    new_row = pd.DataFrame([row], columns=COLUMNS)
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)


# ── SQL database backend (optional, durable) ────────────────────────────

def _secrets_files_exist() -> bool:
    """Check for a secrets.toml on disk without touching st.secrets, which
    would otherwise print a visible in-app warning when none exists."""
    home = os.path.expanduser(
        "~/.streamlit/secrets.toml" if os.name != "nt"
        else os.path.join(os.environ.get("USERPROFILE", ""), ".streamlit",
                          "secrets.toml"))
    project = os.path.join(ROOT, ".streamlit", "secrets.toml")
    return os.path.exists(home) or os.path.exists(project)


def _get_conn():
    """Return a cached st.connection to the community DB, or None if the
    ``[connections.cells_db]`` secret isn't configured / unreachable."""
    if not _secrets_files_exist():
        return None
    try:
        import streamlit as st
        secrets = st.secrets
        if "connections" not in secrets or "cells_db" not in secrets["connections"]:
            return None
        return st.connection("cells_db", type="sql")
    except Exception:
        return None


def db_configured() -> bool:
    return _get_conn() is not None


def _ensure_table(conn) -> None:
    from sqlalchemy import text
    cols_sql = ",\n            ".join(
        f"{c} DOUBLE PRECISION" if c in
        ("crate_ref", "a", "Da_w", "Da", "Da_p", "Da_c", "R_s", "frac", "rms_mV")
        else f"{c} TEXT" for c in DB_COLUMNS)
    with conn.session as s:
        s.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
            {cols_sql},
            PRIMARY KEY (id)
            )
        """))
        s.commit()


def _load_from_db(conn) -> pd.DataFrame:
    _ensure_table(conn)
    df = conn.query(f"SELECT * FROM {TABLE} ORDER BY timestamp", ttl=15)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["is_reference"] = False
    return _coerce_is_reference(df[COLUMNS])


def _insert_into_db(conn, row: dict) -> None:
    from sqlalchemy import text
    _ensure_table(conn)
    cols = DB_COLUMNS
    placeholders = ", ".join(f":{c}" for c in cols)
    collist = ", ".join(cols)
    with conn.session as s:
        s.execute(text(f"INSERT INTO {TABLE} ({collist}) VALUES ({placeholders})"),
                  {c: _to_python(row[c]) for c in cols})
        s.commit()
    try:
        import streamlit as st
        st.cache_data.clear()
    except Exception:
        pass


# ── public API ───────────────────────────────────────────────────────────

def load_database(include_reference: bool = True) -> pd.DataFrame:
    """Full community database: seed literature electrodes + submissions."""
    conn = _get_conn()
    if conn is not None:
        try:
            community = _load_from_db(conn)
        except Exception:
            community = _load_local()
    else:
        community = _load_local()
    if not include_reference:
        return community
    if community.empty:
        return _seed_dataframe()
    return pd.concat([_seed_dataframe(), community], ignore_index=True)


def _to_python(v):
    """Coerce numpy scalars (e.g. from scipy fits) to plain Python types.

    psycopg2 has no adapter for numpy.float64/int64 and silently falls back
    to embedding their repr() (``np.float64(...)``) as raw SQL text, which
    Postgres then tries to parse as a call into a schema named ``np``.
    """
    if v is None:
        return None
    if hasattr(v, "item") and not isinstance(v, str):
        try:
            return v.item()
        except Exception:
            return v
    return v


def append_entry(entry: dict) -> tuple[bool, str]:
    """Add one submission to the shared database (or the local fallback).

    ``entry`` should contain at least: contributor, cell_name, chemistry,
    method, source_note, crate_ref, Da_w, Da, Da_p; other COLUMNS are
    optional and default to None.
    """
    row = {c: _to_python(entry.get(c)) for c in COLUMNS}
    row["id"] = uuid.uuid4().hex[:12]
    row["timestamp"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row["is_reference"] = False

    conn = _get_conn()
    if conn is not None:
        try:
            _insert_into_db(conn, row)
            return True, "Saved to the shared database — visible to everyone."
        except Exception as exc:
            _append_local(row)
            return False, f"Database write failed ({exc}); saved locally only."

    _append_local(row)
    return True, ("Saved locally only. Configure a database connection "
                  "(see README) for durable, publicly-shared storage.")
