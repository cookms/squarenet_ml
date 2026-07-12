from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Tuple, List, Optional, Set, Dict, Any

import pandas as pd


from .config import PipelineConfig
from .mp_query import search_candidates, fetch_structure, load_material_ids_txt
from .io import ensure_dir, dump_cif, update_processed_ids_log, append_tables_v2


def _read_existing_table(out_dir: str, name: str) -> Optional[pd.DataFrame]:
    """Read an existing output table (parquet preferred, else csv).

    Supports:
      - single parquet file: out_dir/{name}.parquet
      - parquet dataset directory: out_dir/{name}.parquet/part-*.parquet
    """
    p = os.path.join(out_dir, f"{name}.parquet")
    c = os.path.join(out_dir, f"{name}.csv")

    if os.path.exists(p):
        try:
            return pd.read_parquet(p)  # works for file OR directory dataset (pyarrow strongly recommended)
        except Exception:
            pass

    if os.path.exists(c):
        try:
            return pd.read_csv(c)
        except Exception:
            pass

    return None


def _load_processed_ids_log(out_dir: str, filename: str) -> Set[str]:
    """Fast resume: load processed IDs from the log file (if present)."""
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return set()

    out: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.add(s.split()[0])
    except Exception:
        return set()
    return out


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Atomic JSON write (best-effort)."""
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def run_pipeline(cfg: "PipelineConfig") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run MP query -> preprocess -> detect -> summarize.

    Batch-writes outputs so partial progress is preserved if the run stops or errors.
    Returns (materials_out, axis_species_out) read back from disk at the end.
    """
    out_dir = cfg.output.out_dir
    ensure_dir(out_dir)

    # How often to flush to disk
    flush_every = int(getattr(cfg.output, "flush_every", 50))
    flush_every = max(flush_every, 1)

    # --- Candidate selection ---
    material_ids = None
    if cfg.mp.material_ids:
        material_ids = [str(x).strip() for x in cfg.mp.material_ids if str(x).strip()]
    elif cfg.mp.material_ids_path:
        material_ids = load_material_ids_txt(cfg.mp.material_ids_path)

    # --- Resume / skip-existing ---
    processed_ids: Set[str] = set()
    materials_prev = None
    axis_species_prev = None

    if cfg.output.resume:
        # Prefer processed log (fast)
        processed_ids = _load_processed_ids_log(out_dir, cfg.output.processed_log_name)

        # If log wasn't present/usable, fall back to reading materials table
        if not processed_ids:
            materials_prev = _read_existing_table(out_dir, "materials")
            axis_species_prev = _read_existing_table(out_dir, "axis_species")
            if materials_prev is not None and "material_id" in materials_prev.columns:
                processed_ids = set(materials_prev["material_id"].astype(str))

    if material_ids and cfg.output.resume and cfg.output.skip_existing and processed_ids:
        material_ids = [mid for mid in material_ids if mid not in processed_ids]

    # --- MP query ---
    if material_ids:
        summary_docs = search_candidates(
            api_key=cfg.mp.api_key,
            material_ids=material_ids,
            include_deprecated=cfg.mp.include_deprecated,
            limit=max(len(material_ids), 1),
        )
    else:
        summary_docs = search_candidates(
            api_key=cfg.mp.api_key,
            elements_all=cfg.mp.elements_all,
            elements_any=cfg.mp.elements_any,
            exclude_elements=cfg.mp.exclude_elements,
            spacegroups=cfg.mp.spacegroups,
            crystal_systems=cfg.mp.crystal_systems,
            band_gap_min=cfg.mp.band_gap_min,
            band_gap_max=cfg.mp.band_gap_max,
            is_metal=cfg.mp.is_metal,
            include_deprecated=cfg.mp.include_deprecated,
            theoretical=cfg.mp.theoretical,
            energy_above_hull_max=cfg.mp.energy_above_hull_max,
            limit=cfg.mp.limit,
        )

        if cfg.output.resume and cfg.output.skip_existing and processed_ids:
            summary_docs = [d for d in summary_docs if str(d.get("material_id")) not in processed_ids]

    meta = {
        "mp_query": cfg.mp.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "detect": cfg.detect.__dict__,
        "user_meta": cfg.meta,
    }

    # Write meta early so it's present even if we crash later
    try:
        _atomic_write_json(os.path.join(out_dir, "meta.json"), meta)
    except Exception:
        pass

    cif_dir = os.path.join(out_dir, "cifs")
    if cfg.output.export_cifs:
        ensure_dir(cif_dir)

    # --- Batch buffers ---
    materials_buf: List[pd.DataFrame] = []
    axis_species_buf: List[pd.DataFrame] = []
    mids_buf: List[str] = []

    def _flush(reason: str) -> None:
        """Flush current buffers to disk + update processed ids log."""
        nonlocal materials_buf, axis_species_buf, mids_buf, processed_ids

        if not mids_buf:
            return

        materials_batch = pd.concat(materials_buf, ignore_index=True) if materials_buf else pd.DataFrame()
        axis_batch = pd.concat(axis_species_buf, ignore_index=True) if axis_species_buf else pd.DataFrame()

        # Defensive within-batch de-dupe
        if not materials_batch.empty and "material_id" in materials_batch.columns:
            materials_batch = materials_batch.drop_duplicates(subset=["material_id"], keep="last")
        if not axis_batch.empty and set(["material_id", "axis", "species"]).issubset(axis_batch.columns):
            axis_batch = axis_batch.drop_duplicates(subset=["material_id", "axis", "species"], keep="last")

        # >>> CHANGE: write via io.append_tables_v2 so every flush appends CSV+Parquet
        # strict_parquet=True means you will SEE the actual error (missing pyarrow / bad columns)
        append_tables_v2(
            out_dir,
            material_features=materials_batch,
            axis_species_features=axis_batch,
            meta=meta,
            write_csv=cfg.output.write_csv,
            write_parquet=cfg.output.write_parquet,
            parquet_safe=True,
            strict_parquet=True,
        )

        # processed_ids log (append per flush)
        try:
            stamp = datetime.now().isoformat(timespec="seconds")
        except Exception:
            stamp = "run"
        header = f"# processed_ids flush @ {stamp}  n={len(mids_buf)}  reason={reason}"
        update_processed_ids_log(
            out_dir,
            mids_buf,
            filename=cfg.output.processed_log_name,
            append=True,
            header=header,
        )

        processed_ids.update(mids_buf)
        materials_buf.clear()
        axis_species_buf.clear()
        mids_buf.clear()

    # --- Main loop (with finally flush) ---
    try:
        for i, doc in enumerate(summary_docs, start=1):
            mid = str(doc["material_id"])

            if cfg.output.resume and cfg.output.skip_existing and mid in processed_ids:
                continue

            name = doc.get("formula_pretty")

            try:
                s_raw = fetch_structure(mid, api_key=cfg.mp.api_key)
            except Exception as e:
                print(f"[{i}/{len(summary_docs)}] {mid} structure fetch/preprocess failed: {e}")
                continue

            try:
                layer_results = find_square_net_planes(s_raw)
                layers_df = pd.DataFrame(layer_results)
            except Exception as e:
                print(f"[{i}/{len(summary_docs)}] {mid} layer detection failed: {e}")
                continue

            try:
                material_row, axis_species_rows, layers_row = summarize_square_net_one_material_v2(
                    layers_df,
                    material_id=mid,
                    formula=name,
                    pass_col="passes2",
                    score_col="mean_score",
                    abridged_summary=False,
                    top_k=None,
                )
            except Exception as e:
                print(f"[{i}/{len(summary_docs)}] {mid} summarization failed: {e}")
                continue

            # Add selected MP summary fields to outputs
            for k in ["sg_number", "sg_symbol", "crystal_system", "energy_above_hull", "band_gap"]:
                if k in doc and k not in material_row.columns:
                    material_row[k] = doc.get(k)

            axis_species_rows = axis_species_rows.copy()
            if len(axis_species_rows) > 0:
                if "formula_pretty" not in axis_species_rows.columns:
                    axis_species_rows.insert(1, "formula_pretty", name)
                for k in ["sg_number", "sg_symbol", "crystal_system", "energy_above_hull", "band_gap"]:
                    if k in doc and k not in axis_species_rows.columns:
                        axis_species_rows[k] = doc.get(k)

            # Buffer
            materials_buf.append(material_row)
            axis_species_buf.append(axis_species_rows)
            mids_buf.append(mid)

            # CIF export (unchanged)
            if cfg.output.export_cifs:
                try:
                    has_sn = bool(material_row.iloc[0].get("has_any_pass", False))
                    if (not cfg.output.export_positive_only) or has_sn:
                        dump_cif(s_conv, os.path.join(cif_dir, f"{mid}_conv.cif"))
                        dump_cif(s_final, os.path.join(cif_dir, f"{mid}_screening_supercell.cif"))
                except Exception as e:
                    print(f"[{i}/{len(summary_docs)}] {mid} CIF export failed: {e}")

            # Flush on batch boundary
            if len(mids_buf) >= flush_every:
                _flush(reason=f"batch_size={flush_every}")

            if i % 10 == 0:
                print(f"Processed {i}/{len(summary_docs)} materials...")

    finally:
        # Ensure we persist whatever we've already computed, even if interrupted/crashed
        try:
            _flush(reason="finalize/finally")
        except Exception as e:
            print(f"[flush] final flush failed: {e}")

    materials_out = _read_existing_table(out_dir, "materials") or pd.DataFrame()
    axis_species_out = _read_existing_table(out_dir, "axis_species") or pd.DataFrame()
    return materials_out, axis_species_out
