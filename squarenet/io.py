from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

logger = logging.getLogger(__name__)

_PARQUET_ENGINE: Optional[str] = None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def dump_cif(struct: Structure, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    CifWriter(struct, symprec=0.01).write_file(path)


def _get_parquet_engine() -> str:
    """Return an available parquet engine (pyarrow preferred). Raises with a clear message if none."""
    global _PARQUET_ENGINE
    if _PARQUET_ENGINE is not None:
        return _PARQUET_ENGINE

    try:
        import pyarrow  # noqa: F401
        _PARQUET_ENGINE = "pyarrow"
        return _PARQUET_ENGINE
    except Exception:
        pass

    try:
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
        return _PARQUET_ENGINE
    except Exception as e:
        raise ImportError(
            "Parquet write failed because no parquet engine is installed.\n"
            "Install one of:\n"
            "  - pip install pyarrow\n"
            "  - pip install fastparquet\n"
        ) from e


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: convert common problematic object values to strings/JSON so parquet can serialize.

    This is intentionally conservative: only touches object dtype columns that look non-scalar.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    scalar_types = (str, int, float, bool, type(None), pd.Timestamp)

    for col in out.columns:
        if out[col].dtype != "object":
            continue
        s = out[col].dropna()
        if s.empty:
            continue

        sample = s.iloc[:25].tolist()

        def is_non_scalar(v: Any) -> bool:
            if isinstance(v, scalar_types):
                return False
            # common containers + numpy-ish
            if isinstance(v, (dict, list, tuple, set)):
                return True
            if hasattr(v, "tolist"):  # numpy arrays, etc.
                return True
            # other custom objects (e.g., pymatgen types)
            return True

        if any(is_non_scalar(v) for v in sample):
            def coerce(v: Any) -> Any:
                if v is None or isinstance(v, scalar_types):
                    return v
                if isinstance(v, (dict, list, tuple, set)):
                    return json.dumps(v, default=str)
                if hasattr(v, "tolist"):
                    try:
                        return json.dumps(v.tolist(), default=str)
                    except Exception:
                        return str(v)
                return str(v)

            out[col] = out[col].map(coerce)

    return out


def update_processed_ids_log(
    out_dir: str,
    processed_ids: List[str],
    *,
    filename: str = "processed_ids.txt",
    append: bool = True,
    header: Optional[str] = None,
) -> str:
    """Update a plain-text log of processed material IDs."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    ids = [str(x).strip() for x in processed_ids if str(x).strip()]
    mode = "a" if append else "w"

    with open(path, mode, encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        if append:
            for mid in ids:
                f.write(mid + "\n")
            f.write("\n")
        else:
            for mid in sorted(set(ids)):
                f.write(mid + "\n")
        f.flush()
        os.fsync(f.fileno())

    return path


# ---------- OVERWRITE writers (single file) ----------

def write_tables_v2(
    out_dir: str,
    *,
    material_features: pd.DataFrame,
    axis_species_features: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
    write_csv: bool = True,
    write_parquet: bool = True,
    parquet_safe: bool = True,
    strict_parquet: bool = True,
):
    """Overwrite-mode writer (compatible with your existing call sites)."""
    ensure_dir(out_dir)

    if meta is not None:
        _atomic_write_json(os.path.join(out_dir, "run_meta.json"), meta)

    def _write_one(df: Optional[pd.DataFrame], name: str):
        if df is None:
            return

        if write_csv:
            _atomic_write_csv(df, os.path.join(out_dir, f"{name}.csv"))

        if write_parquet:
            engine = _get_parquet_engine()
            out_path = os.path.join(out_dir, f"{name}.parquet")
            try:
                wdf = _parquet_safe(df) if parquet_safe else df
                tmp = f"{out_path}.tmp-{uuid.uuid4().hex[:8]}"
                wdf.to_parquet(tmp, index=False, engine=engine)
                os.replace(tmp, out_path)
            except Exception as e:
                msg = f"FAILED to write parquet for {name} -> {out_path}: {type(e).__name__}: {e}"
                if strict_parquet:
                    raise RuntimeError(msg) from e
                logger.exception(msg)

    _write_one(material_features, "materials")
    _write_one(axis_species_features, "axis_species")


# ---------- APPEND writers (for flush/batching) ----------

def _append_csv(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        return
    ensure_dir(os.path.dirname(path) or ".")
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, header=write_header)
        f.flush()
        os.fsync(f.fileno())


def _append_parquet(df: pd.DataFrame, path: str, *, parquet_safe: bool = True) -> None:
    """Append parquet robustly.

    Preferred: dataset-dir append:
      path = out_dir/{name}.parquet  (directory)
      writes part files: part-*.parquet

    If `path` already exists as a *file*, falls back to merge+rewrite (slower).
    """
    if df is None or df.empty:
        return

    engine = _get_parquet_engine()
    wdf = _parquet_safe(df) if parquet_safe else df

    # Dataset directory mode
    if (not os.path.exists(path)) or os.path.isdir(path):
        ensure_dir(path)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        part = f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        out_path = os.path.join(path, part)
        wdf.to_parquet(out_path, index=False, engine=engine)
        return

    # File exists: compat merge+rewrite
    prev = pd.read_parquet(path)
    merged = pd.concat([prev, wdf], ignore_index=True)
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    merged.to_parquet(tmp, index=False, engine=engine)
    os.replace(tmp, path)


def append_tables_v2(
    out_dir: str,
    *,
    material_features: pd.DataFrame,
    axis_species_features: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
    write_csv: bool = True,
    write_parquet: bool = True,
    parquet_safe: bool = True,
    strict_parquet: bool = True,
):
    """Append-mode writer intended for per-flush batching."""
    ensure_dir(out_dir)

    # Meta is safe to overwrite; write atomically so partial files aren't possible.
    if meta is not None:
        try:
            _atomic_write_json(os.path.join(out_dir, "run_meta.json"), meta)
        except Exception:
            logger.exception("Failed writing run_meta.json")

    try:
        if write_csv:
            _append_csv(material_features, os.path.join(out_dir, "materials.csv"))
            _append_csv(axis_species_features, os.path.join(out_dir, "axis_species.csv"))

        if write_parquet:
            _append_parquet(material_features, os.path.join(out_dir, "materials.parquet"), parquet_safe=parquet_safe)
            _append_parquet(axis_species_features, os.path.join(out_dir, "axis_species.parquet"), parquet_safe=parquet_safe)

    except Exception as e:
        msg = f"FAILED append_tables_v2: {type(e).__name__}: {e}"
        if strict_parquet:
            raise RuntimeError(msg) from e
        logger.exception(msg)
