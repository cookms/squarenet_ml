
import json
import ast
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd


# ---------------- existing helpers (keep if you already have them) ----------------
def _dict_to_json_str(d: Any) -> str:
    """Return a stable JSON string for dict/list-like cells."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "{}"
    d2 = _maybe_literal(d)
    if isinstance(d2, (dict, list, tuple, set, np.ndarray)):
        return json.dumps(d2, sort_keys=True, default=_json_default)
    # if it's already a scalar/string, keep it
    return json.dumps(str(d2))


def _counts_major(d: Dict[str, int]) -> Tuple[Optional[str], float, float, int]:
    """
    Return (major_species, major_fraction, total_sites, n_species).
    If d empty/invalid -> (None, nan, nan, 0)
    """
    if not d:
        return None, float("nan"), float("nan"), 0
    try:
        total = float(sum(d.values()))
        if total <= 0:
            return None, float("nan"), float("nan"), int(len(d))
        major = max(d.items(), key=lambda kv: kv[1])[0]
        frac = float(d[major]) / (total + 1e-12)
        return str(major), float(frac), float(total), int(len(d))
    except Exception:
        return None, float("nan"), float("nan"), 0


def _row_num(row: pd.Series, col: str) -> float:
    """Numeric getter from a row (Series), tolerant to missing/strings."""
    try:
        return float(pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0])
    except Exception:
        return float("nan")


def _row_str(row: pd.Series, col: str) -> Optional[str]:
    """String getter from a row, tolerant to missing."""
    if col not in row.index:
        return None
    v = row.get(col, None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)

def _safe_dict(x) -> Dict[str, int]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return {}
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x

    def _to_bool(v):
        s = str(v).strip().lower()
        if s in ("true", "1", "t", "yes", "y"):
            return True
        if s in ("false", "0", "f", "no", "n"):
            return False
        return bool(v)

    return x.map(_to_bool).astype(bool)


# ---------------- NEW helpers for v2 ----------------

def _json_default(obj):
    """Best-effort conversion for numpy-ish objects during json.dumps."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _maybe_literal(x):
    """
    If x is a string that looks like a python literal dict/list, try ast.literal_eval.
    Otherwise return x unchanged.
    """
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s or s.lower() == "nan":
        return None
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return ast.literal_eval(s)
        except Exception:
            return x
    return x


def _reasons_to_str(x) -> str:
    """Normalize a 'passes2_fail_reasons' cell into a compact pipe-separated string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    x2 = _maybe_literal(x)
    if isinstance(x2, (list, tuple, set)):
        vals = [str(v) for v in x2 if v is not None and str(v).strip() and str(v).strip().lower() != "nan"]
        return "|".join(vals)
    if isinstance(x2, dict):
        # if someone stores structured reasons, stringify keys/values compactly
        try:
            return "|".join([f"{k}:{x2[k]}" for k in sorted(x2.keys(), key=lambda z: str(z))])
        except Exception:
            return str(x2)
    return str(x2)


def _make_layers_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a CSV-safe copy of df:
      - dict/list/tuple cells are JSON-serialized
      - strings that look like python dict/list literals are parsed then JSON-serialized
    """
    out = df.copy()

    def _cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return v
        vv = _maybe_literal(v)
        if isinstance(vv, (dict, list, tuple, set, np.ndarray)):
            return json.dumps(vv, sort_keys=True, default=_json_default)
        return vv

    # Only touch object columns to avoid messing up numeric dtypes
    obj_cols = [c for c in out.columns if out[c].dtype == object]
    for c in obj_cols:
        out[c] = out[c].map(_cell)
    return out


def _ensure_pass_columns(df: pd.DataFrame, pass_col: str) -> pd.DataFrame:
    """
    Ensure pass columns exist and are boolean.
    Also enforces: passes2 implies passes (if both exist).
    """
    out = df.copy()

    # parse/ensure common pass cols
    for c in ("passes", "passes2", pass_col):
        if c in out.columns:
            out[c] = _to_bool_series(out[c])
        else:
            out[c] = False

    if "passes" in out.columns and "passes2" in out.columns:
        out["passes2"] = out["passes2"] & out["passes"]

    # if pass_col is not passes2/passes, still keep it as-is (already parsed above)
    return out


def _pick_adj_dist_col(df: pd.DataFrame) -> str:
    """Choose a min-adjacent-distance column for ranking (larger is better)."""
    for c in ("min_adj_dist_any_atom", "min_adj_dist_any", "min_adj_dist_any_plane"):
        if c in df.columns:
            return c
    # fallback: create a synthetic column name that will be treated as missing
    return "min_adj_dist_any_atom"


def rank_layers_for_dominance(
    df: pd.DataFrame,
    *,
    pass_col: str = "passes2",
    score_col: str = "mean_score",
) -> pd.DataFrame:
    """
    Return df sorted by the dominance ranking rule and annotated with:
      - dominance_rank (1 = best)
      - is_dominant_layer (True for rank==1)
    Does NOT drop any original columns.
    """
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    d = df.copy()
    d = _ensure_pass_columns(d, pass_col=pass_col)

    # Ranking columns (NaNs are treated as worst)
    adj_col = _pick_adj_dist_col(d)

    d["_rank_pass"] = d[pass_col].astype(int)

    # higher better
    d["_rank_score"] = pd.to_numeric(d.get(score_col, np.nan), errors="coerce").fillna(-np.inf)

    # lower better
    d["_rank_tol"] = pd.to_numeric(d.get("tol_ratio_any", np.nan), errors="coerce").fillna(np.inf)
    d["_rank_nn"] = pd.to_numeric(d.get("nn_intra_min", np.nan), errors="coerce").fillna(np.inf)

    # higher better
    d["_rank_adj"] = pd.to_numeric(d.get(adj_col, np.nan), errors="coerce").fillna(-np.inf)
    d["_rank_nsites"] = pd.to_numeric(d.get("n_sites", np.nan), errors="coerce").fillna(-np.inf)

    # Stable sort
    d = d.sort_values(
        ["_rank_pass", "_rank_score", "_rank_tol", "_rank_nn", "_rank_adj", "_rank_nsites"],
        ascending=[False, False, True, True, False, False],
        kind="mergesort",
    )

    d["dominance_rank"] = np.arange(1, len(d) + 1, dtype=int)
    d["is_dominant_layer"] = d["dominance_rank"] == 1

    # Remove internal helper columns
    d = d.drop(columns=[c for c in d.columns if c.startswith("_rank_")], errors="ignore")
    return d


def select_dominant_layer(
    df: pd.DataFrame,
    *,
    pass_col: str = "passes2",
    score_col: str = "mean_score",
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    Select the dominant layer row using the ranked sort rule.
    Returns:
      (dominant_row_or_None, ranked_df_with_dominance_rank_and_is_dominant_layer)
    """
    ranked = rank_layers_for_dominance(df, pass_col=pass_col, score_col=score_col)
    if ranked is None or len(ranked) == 0:
        return None, ranked
    return ranked.iloc[0], ranked


# ---------------- NEW per-material summarizer v2 ----------------

def summarize_square_net_one_material_v2(
    layer_df: pd.DataFrame,
    material_id: str,
    formula: str,
    *,
    pass_col: str = "passes2",
    score_col: str = "mean_score",
    abridged_summary: bool = False,
    top_k: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarize ONE material's plane-by-plane results (v2).

    Returns:
      material_summary_df: 1 row per material (dominant-layer-focused, compact)
      axis_species_summary_df: rows per (material_id, axis, species) (best-layer-focused, compact)
      layers_csv_df: raw per-layer rows + material_id/formula + dominance_rank/is_dominant_layer, JSON-safe for CSV
    """
    # Always return a layers_csv_df
    if layer_df is None:
        empty_layers = pd.DataFrame(columns=["material_id", "formula"])
        material_summary_df = pd.DataFrame([{
            "material_id": material_id,
            "formula": formula,
            "n_layers_total": 0,
            "n_axes": 0,
            "n_species": 0,
            "n_pass": 0,
            "pass_fraction_total": 0.0,
            "has_any_pass": 0,
            "dominant_has_pass": 0,
            "dominant_axis": None,
            "dominant_species": None,
            "dominant_plane_id": None,
            "dominant_plane_center_frac": np.nan,
        }])
        axis_species_summary_df = pd.DataFrame(columns=[
            "material_id", "formula", "axis", "species",
            "n_layers", "n_pass", "pass_fraction",
            "best_layer_has_pass",
            "best_layer_plane_id", "best_layer_plane_center_frac",
            f"best_layer_{score_col}", "best_layer_tol_ratio_any", "best_layer_nn_intra_min", "best_layer_min_adj_dist_any_atom",
            "is_dominant_axsp",
        ])
        return material_summary_df, axis_species_summary_df, empty_layers

    # Preserve raw rows for CSV output (do not drop columns)
    raw = layer_df.copy()
    raw["material_id"] = material_id
    raw["formula"] = formula

    # Working df for ranking/summaries (still keep all cols; just parse booleans)
    df = raw.copy()
    df = _ensure_pass_columns(df, pass_col=pass_col)

    # Dominance ranking across all layers
    dom_row, ranked_all = select_dominant_layer(df, pass_col=pass_col, score_col=score_col)

    # Attach rank annotations back to layers csv output by index
    layers_csv_df = raw.join(ranked_all[["dominance_rank", "is_dominant_layer"]], how="left")
    layers_csv_df = _make_layers_csv_safe(layers_csv_df)

    # Handle empty (but non-None) layer_df
    if len(df) == 0 or dom_row is None:
        material_summary_df = pd.DataFrame([{
            "material_id": material_id,
            "formula": formula,
            "n_layers_total": 0,
            "n_axes": 0,
            "n_species": 0,
            "n_pass": 0,
            "pass_fraction_total": 0.0,
            "has_any_pass": 0,
            "dominant_has_pass": 0,
            "dominant_axis": None,
            "dominant_species": None,
            "dominant_plane_id": None,
            "dominant_plane_center_frac": np.nan,
        }])
        axis_species_summary_df = pd.DataFrame(columns=[
            "material_id", "formula", "axis", "species",
            "n_layers", "n_pass", "pass_fraction",
            "best_layer_has_pass",
            "best_layer_plane_id", "best_layer_plane_center_frac",
            f"best_layer_{score_col}", "best_layer_tol_ratio_any", "best_layer_nn_intra_min", "best_layer_min_adj_dist_any_atom",
            "is_dominant_axsp",
        ])
        return material_summary_df, axis_species_summary_df, layers_csv_df

    # Count fields
    n_layers_total = int(len(df))
    n_axes = int(df["axis"].nunique()) if "axis" in df.columns else 0
    n_species = int(df["species"].nunique()) if "species" in df.columns else 0
    n_pass = int(df[pass_col].sum()) if pass_col in df.columns else 0
    pass_fraction_total = float(n_pass / (n_layers_total + 1e-12))
    has_any_pass = int(n_pass > 0)

    # Dominant identifiers
    dominant_axis = dom_row.get("axis", None)
    dominant_species = dom_row.get("species", None)
    dominant_plane_id = dom_row.get("plane_id", None)
    dominant_plane_center_frac = dom_row.get("plane_center_frac", np.nan)
    dominant_has_pass = int(bool(dom_row.get(pass_col, False)))  # will be 0 if no passing layers

    # Dominant geometry
    adj_col = _pick_adj_dist_col(df)
    dominant_geom = {
        f"dominant_{score_col}": float(pd.to_numeric(pd.Series([dom_row.get(score_col, np.nan)]), errors="coerce").iloc[0])
        if score_col in df.columns else np.nan,
        "dominant_tol_ratio_any": float(pd.to_numeric(pd.Series([dom_row.get("tol_ratio_any", np.nan)]), errors="coerce").iloc[0])
        if "tol_ratio_any" in df.columns else np.nan,
        "dominant_nn_intra_min": float(pd.to_numeric(pd.Series([dom_row.get("nn_intra_min", np.nan)]), errors="coerce").iloc[0])
        if "nn_intra_min" in df.columns else np.nan,
        f"dominant_{adj_col}": float(pd.to_numeric(pd.Series([dom_row.get(adj_col, np.nan)]), errors="coerce").iloc[0])
        if adj_col in df.columns else np.nan,
    }

    # Optional dominant extras (always create columns; fill NaN if absent)
    optional_cols = [
        "uv_ang_deg_mean",
        "uv_ang_err_mean",
        "uv_len_err_mean",
        "cnn_in_plane_bond_angle_deg_mean",
        "cnn_out_of_plane_tilt_angle_deg_mean",
    ]
    dominant_optional = {}
    for c in optional_cols:
        key = f"dominant_{c}"
        if c in df.columns:
            dominant_optional[key] = float(pd.to_numeric(pd.Series([dom_row.get(c, np.nan)]), errors="coerce").iloc[0])
        else:
            dominant_optional[key] = np.nan

    # Optional dominant fail reasons
    if "passes2_fail_reasons" in df.columns:
        dominant_fail_reasons = _reasons_to_str(dom_row.get("passes2_fail_reasons", ""))
    else:
        dominant_fail_reasons = ""
    
    # --- Dominant co-plane chemistry ---
    dom_coplane_counts = _safe_dict(dom_row.get("coplane_species_counts", {}))
    dom_coplane_other_counts = _safe_dict(dom_row.get("coplane_other_species_counts", {}))
    
    dom_coplane_major, dom_coplane_major_frac, dom_coplane_total, dom_coplane_n_species = _counts_major(dom_coplane_counts)
    _, _, dom_coplane_other_total, dom_coplane_other_n_species = _counts_major(dom_coplane_other_counts)
    dom_coplane_other_fraction = float(dom_coplane_other_total / (dom_coplane_total + 1e-12)) if np.isfinite(dom_coplane_total) else float("nan")
    
    # --- Dominant CrystalNN out-of-plane NN summary ---
    dom_cnn_oop_nn_species = _row_str(dom_row, "cnn_out_of_plane_nn_species")
    dom_cnn_oop_nn_dist = _row_num(dom_row, "cnn_out_of_plane_nn_dist")
    
    dom_cnn_oop_bonded_counts = _safe_dict(dom_row.get("cnn_out_of_plane_bonded_species_counts", {}))
    dom_cnn_oop_major, dom_cnn_oop_major_frac, dom_cnn_oop_total, dom_cnn_oop_n_species = _counts_major(dom_cnn_oop_bonded_counts)
    
    # --- Other useful dominant chemical info (only if present) ---
    # (these are safe even if columns don’t exist; we’ll store NaN/None)
    dom_oxi_mean = _row_num(dom_row, "square_species_oxi_state_mean")
    dom_oxi_std  = _row_num(dom_row, "square_species_oxi_state_std")
    dom_has_oop_same_sp_bond = int(bool(dom_row.get("has_out_of_plane_same_species_bond", False)))
    
    # Adjacent-plane composition (atom-based + plane-based) if you have them in layers_df
    dom_adj_atom_major_species = _row_str(dom_row, "closest_by_atom_plane_major_species")
    dom_adj_atom_major_frac = _row_num(dom_row, "closest_by_atom_plane_major_fraction")
    dom_adj_atom_counts = _safe_dict(dom_row.get("closest_by_atom_plane_species_counts", {}))
    
    dom_adj_plane_major_species = _row_str(dom_row, "closest_by_plane_plane_major_species")
    dom_adj_plane_major_frac = _row_num(dom_row, "closest_by_plane_plane_major_fraction")
    dom_adj_plane_counts = _safe_dict(dom_row.get("closest_by_plane_plane_species_counts", {}))


    # Material summary (compact)
    material_out: Dict[str, Any] = {
        "material_id": material_id,
        "formula": formula,
        "n_layers_total": n_layers_total,
        "n_axes": n_axes,
        "n_species": n_species,
        "n_pass": n_pass,
        "pass_fraction_total": pass_fraction_total,
        "has_any_pass": has_any_pass,
        "dominant_has_pass": dominant_has_pass,
        "dominant_axis": dominant_axis,
        "dominant_species": dominant_species,
        "dominant_plane_id": dominant_plane_id,
        "dominant_plane_center_frac": float(pd.to_numeric(pd.Series([dominant_plane_center_frac]), errors="coerce").iloc[0]),
        "dominant_passes2_fail_reasons": dominant_fail_reasons,
    }
    material_out.update(dominant_geom)
    material_out.update(dominant_optional)

    material_out.update({
    # --- co-plane chemistry (dominant layer) ---
    "dominant_coplane_species_counts_json": _dict_to_json_str(dom_coplane_counts),
    "dominant_coplane_n_species": int(dom_coplane_n_species),
    "dominant_coplane_major_species": dom_coplane_major,
    "dominant_coplane_major_fraction": float(dom_coplane_major_frac) if np.isfinite(dom_coplane_major_frac) else float("nan"),
    "dominant_coplane_other_species_counts_json": _dict_to_json_str(dom_coplane_other_counts),
    "dominant_coplane_other_n_species": int(dom_coplane_other_n_species),
    "dominant_coplane_other_fraction": float(dom_coplane_other_fraction) if np.isfinite(dom_coplane_other_fraction) else float("nan"),

    # --- CrystalNN out-of-plane NN chemistry (dominant layer) ---
    "dominant_cnn_out_of_plane_nn_species": dom_cnn_oop_nn_species,
    "dominant_cnn_out_of_plane_nn_dist": float(dom_cnn_oop_nn_dist) if np.isfinite(dom_cnn_oop_nn_dist) else float("nan"),
    "dominant_cnn_out_of_plane_bonded_species_counts_json": _dict_to_json_str(dom_cnn_oop_bonded_counts),
    "dominant_cnn_out_of_plane_bonded_major_species": dom_cnn_oop_major,
    "dominant_cnn_out_of_plane_bonded_major_fraction": float(dom_cnn_oop_major_frac) if np.isfinite(dom_cnn_oop_major_frac) else float("nan"),
    "dominant_cnn_out_of_plane_bonded_n_species": int(dom_cnn_oop_n_species),

    # --- other chemical context (dominant layer) ---
    "dominant_square_species_oxi_state_mean": float(dom_oxi_mean) if np.isfinite(dom_oxi_mean) else float("nan"),
    "dominant_square_species_oxi_state_std": float(dom_oxi_std) if np.isfinite(dom_oxi_std) else float("nan"),
    "dominant_has_out_of_plane_same_species_bond": int(dom_has_oop_same_sp_bond),

    # Adjacent plane composition context (if available in your layer rows)
    "dominant_adj_atom_plane_major_species": dom_adj_atom_major_species,
    "dominant_adj_atom_plane_major_fraction": float(dom_adj_atom_major_frac) if np.isfinite(dom_adj_atom_major_frac) else float("nan"),
    "dominant_adj_atom_plane_species_counts_json": _dict_to_json_str(dom_adj_atom_counts),

    "dominant_adj_plane_plane_major_species": dom_adj_plane_major_species,
    "dominant_adj_plane_plane_major_fraction": float(dom_adj_plane_major_frac) if np.isfinite(dom_adj_plane_major_frac) else float("nan"),
    "dominant_adj_plane_plane_species_counts_json": _dict_to_json_str(dom_adj_plane_counts),
    })

    # Optional Top-K snapshots (compact)
    if top_k is not None and int(top_k) > 0:
        k = int(top_k)
        top = ranked_all.head(k)
        for i in range(k):
            prefix = f"top{i+1}_"
            if i >= len(top):
                material_out[prefix + "has_pass"] = 0
                material_out[prefix + "axis"] = None
                material_out[prefix + "species"] = None
                material_out[prefix + "plane_id"] = None
                material_out[prefix + score_col] = np.nan
                material_out[prefix + "tol_ratio_any"] = np.nan
                material_out[prefix + "nn_intra_min"] = np.nan
                material_out[prefix + adj_col] = np.nan
                continue

            r = top.iloc[i]
            material_out[prefix + "has_pass"] = int(bool(r.get(pass_col, False)))
            material_out[prefix + "axis"] = r.get("axis", None)
            material_out[prefix + "species"] = r.get("species", None)
            material_out[prefix + "plane_id"] = r.get("plane_id", None)
            material_out[prefix + score_col] = float(pd.to_numeric(pd.Series([r.get(score_col, np.nan)]), errors="coerce").iloc[0]) if score_col in top.columns else np.nan
            material_out[prefix + "tol_ratio_any"] = float(pd.to_numeric(pd.Series([r.get("tol_ratio_any", np.nan)]), errors="coerce").iloc[0]) if "tol_ratio_any" in top.columns else np.nan
            material_out[prefix + "nn_intra_min"] = float(pd.to_numeric(pd.Series([r.get("nn_intra_min", np.nan)]), errors="coerce").iloc[0]) if "nn_intra_min" in top.columns else np.nan
            material_out[prefix + adj_col] = float(pd.to_numeric(pd.Series([r.get(adj_col, np.nan)]), errors="coerce").iloc[0]) if adj_col in top.columns else np.nan

    material_summary_df = pd.DataFrame([material_out])

    # Axis/species summary (compact, best-layer focused)
    rows: list[Dict[str, Any]] = []
    if "axis" in df.columns and "species" in df.columns:
        for (ax, sp), g in df.groupby(["axis", "species"], dropna=False, sort=False):
            best_row, _ranked = select_dominant_layer(g, pass_col=pass_col, score_col=score_col)
            if best_row is None:
                continue

            n_layers = int(len(g))
            n_pass_g = int(g[pass_col].sum()) if pass_col in g.columns else 0
            pass_frac_g = float(n_pass_g / (n_layers + 1e-12))

            best_plane_id = best_row.get("plane_id", None)
            best_plane_center_frac = best_row.get("plane_center_frac", np.nan)
            best_has_pass = int(bool(best_row.get(pass_col, False)))

            best_score = float(pd.to_numeric(pd.Series([best_row.get(score_col, np.nan)]), errors="coerce").iloc[0]) if score_col in g.columns else np.nan
            best_tol = float(pd.to_numeric(pd.Series([best_row.get("tol_ratio_any", np.nan)]), errors="coerce").iloc[0]) if "tol_ratio_any" in g.columns else np.nan
            best_nn = float(pd.to_numeric(pd.Series([best_row.get("nn_intra_min", np.nan)]), errors="coerce").iloc[0]) if "nn_intra_min" in g.columns else np.nan
            best_adj = float(pd.to_numeric(pd.Series([best_row.get(adj_col, np.nan)]), errors="coerce").iloc[0]) if adj_col in g.columns else np.nan

            if "passes2_fail_reasons" in g.columns:
                best_fail_reasons = _reasons_to_str(best_row.get("passes2_fail_reasons", ""))
            else:
                best_fail_reasons = ""

            row: Dict[str, Any] = {
                "material_id": material_id,
                "formula": formula,
                "axis": ax,
                "species": sp,
                "n_layers": n_layers,
                "n_pass": n_pass_g,
                "pass_fraction": pass_frac_g,
                "best_layer_has_pass": best_has_pass,
                "best_layer_plane_id": best_plane_id,
                "best_layer_plane_center_frac": float(pd.to_numeric(pd.Series([best_plane_center_frac]), errors="coerce").iloc[0]),
                f"best_layer_{score_col}": best_score,
                "best_layer_tol_ratio_any": best_tol,
                "best_layer_nn_intra_min": best_nn,
                f"best_layer_{adj_col}": best_adj,
                "best_layer_passes2_fail_reasons": best_fail_reasons,
                "is_dominant_axsp": bool((ax == dominant_axis) and (sp == dominant_species)),
            }

            best_coplane_counts = _safe_dict(best_row.get("coplane_species_counts", {}))
            best_coplane_other_counts = _safe_dict(best_row.get("coplane_other_species_counts", {}))
            best_coplane_major, best_coplane_major_frac, best_coplane_total, best_coplane_n_species = _counts_major(best_coplane_counts)
            _, _, best_coplane_other_total, best_coplane_other_n_species = _counts_major(best_coplane_other_counts)
            best_coplane_other_fraction = float(best_coplane_other_total / (best_coplane_total + 1e-12)) if np.isfinite(best_coplane_total) else float("nan")
            
            best_cnn_oop_nn_species = _row_str(best_row, "cnn_out_of_plane_nn_species")
            best_cnn_oop_nn_dist = _row_num(best_row, "cnn_out_of_plane_nn_dist")
            best_cnn_oop_bonded_counts = _safe_dict(best_row.get("cnn_out_of_plane_bonded_species_counts", {}))
            best_cnn_oop_major, best_cnn_oop_major_frac, _, best_cnn_oop_n_species = _counts_major(best_cnn_oop_bonded_counts)
            
            best_oxi_mean = _row_num(best_row, "square_species_oxi_state_mean")
            best_oxi_std = _row_num(best_row, "square_species_oxi_state_std")
            best_has_oop_same_sp_bond = int(bool(best_row.get("has_out_of_plane_same_species_bond", False)))
            
            row.update({
                # co-plane chemistry
                "best_layer_coplane_species_counts_json": _dict_to_json_str(best_coplane_counts),
                "best_layer_coplane_n_species": int(best_coplane_n_species),
                "best_layer_coplane_major_species": best_coplane_major,
                "best_layer_coplane_major_fraction": float(best_coplane_major_frac) if np.isfinite(best_coplane_major_frac) else float("nan"),
                "best_layer_coplane_other_species_counts_json": _dict_to_json_str(best_coplane_other_counts),
                "best_layer_coplane_other_n_species": int(best_coplane_other_n_species),
                "best_layer_coplane_other_fraction": float(best_coplane_other_fraction) if np.isfinite(best_coplane_other_fraction) else float("nan"),
            
                # out-of-plane bonding chemistry (CrystalNN)
                "best_layer_cnn_out_of_plane_nn_species": best_cnn_oop_nn_species,
                "best_layer_cnn_out_of_plane_nn_dist": float(best_cnn_oop_nn_dist) if np.isfinite(best_cnn_oop_nn_dist) else float("nan"),
                "best_layer_cnn_out_of_plane_bonded_species_counts_json": _dict_to_json_str(best_cnn_oop_bonded_counts),
                "best_layer_cnn_out_of_plane_bonded_major_species": best_cnn_oop_major,
                "best_layer_cnn_out_of_plane_bonded_major_fraction": float(best_cnn_oop_major_frac) if np.isfinite(best_cnn_oop_major_frac) else float("nan"),
                "best_layer_cnn_out_of_plane_bonded_n_species": int(best_cnn_oop_n_species),
            
                # other chemistry
                "best_layer_square_species_oxi_state_mean": float(best_oxi_mean) if np.isfinite(best_oxi_mean) else float("nan"),
                "best_layer_square_species_oxi_state_std": float(best_oxi_std) if np.isfinite(best_oxi_std) else float("nan"),
                "best_layer_has_out_of_plane_same_species_bond": int(best_has_oop_same_sp_bond),
            })
            
            # Optional best-layer extras (always present as columns; NaN if absent)
            for c in optional_cols:
                k2 = f"best_layer_{c}"
                if c in g.columns:
                    row[k2] = float(pd.to_numeric(pd.Series([best_row.get(c, np.nan)]), errors="coerce").iloc[0])
                else:
                    row[k2] = np.nan

            

            rows.append(row)

    axis_species_summary_df = pd.DataFrame(rows)

    # Abridged mode: keep only a minimal column set (layers_csv_df always full)
    if abridged_summary:
        mat_keep = [
            "material_id", "formula",
            "n_layers_total", "n_axes", "n_species", "n_pass", "pass_fraction_total", "has_any_pass",
            "dominant_has_pass",
            "dominant_axis", "dominant_species", "dominant_plane_id", "dominant_plane_center_frac",
            f"dominant_{score_col}", "dominant_tol_ratio_any", "dominant_nn_intra_min", f"dominant_{adj_col}",
            "dominant_passes2_fail_reasons",
        ]
        material_summary_df = material_summary_df[[c for c in mat_keep if c in material_summary_df.columns]]

        axsp_keep = [
            "material_id", "formula", "axis", "species",
            "n_layers", "n_pass", "pass_fraction",
            "best_layer_has_pass",
            "best_layer_plane_id", "best_layer_plane_center_frac",
            f"best_layer_{score_col}", "best_layer_tol_ratio_any", "best_layer_nn_intra_min", f"best_layer_{adj_col}",
            "best_layer_passes2_fail_reasons",
            "is_dominant_axsp",
        ]

        mat_keep += [
            "dominant_coplane_species_counts_json",
            "dominant_coplane_major_species",
            "dominant_coplane_major_fraction",
            "dominant_coplane_other_fraction",
            "dominant_cnn_out_of_plane_nn_species",
            "dominant_cnn_out_of_plane_nn_dist",
            "dominant_square_species_oxi_state_mean",
            "dominant_has_out_of_plane_same_species_bond",
        ]
        
        axsp_keep += [
            "best_layer_coplane_major_species",
            "best_layer_coplane_major_fraction",
            "best_layer_coplane_other_fraction",
            "best_layer_cnn_out_of_plane_nn_species",
            "best_layer_cnn_out_of_plane_nn_dist",
            "best_layer_square_species_oxi_state_mean",
            "best_layer_has_out_of_plane_same_species_bond",
        ]
        axis_species_summary_df = axis_species_summary_df[[c for c in axsp_keep if c in axis_species_summary_df.columns]]

    return material_summary_df, axis_species_summary_df, layers_csv_df
