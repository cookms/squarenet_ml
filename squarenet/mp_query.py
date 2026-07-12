from __future__ import annotations

from pathlib import Path
import math
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from mp_api.client import MPRester
    _HAS_MP = True
except Exception:
    _HAS_MP = False


@contextmanager
def get_mpr(api_key: Optional[str] = None):
    if not _HAS_MP:
        raise ImportError("mp-api is not installed. Install with: pip install mp-api")
    key = api_key or os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
    if not key:
        raise ValueError("Materials Project API key not provided. Set MP_API_KEY or pass api_key=...")
    with MPRester(key) as mpr:
        yield mpr


def load_material_ids_txt(path: str) -> List[str]:
    """Load material IDs from a text file.

    - One ID per line
    - Blank lines ignored
    - Lines starting with '#' ignored
    - If a line has extra tokens/commas, the first token is taken

    Returns a de-duplicated list preserving order.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"material_ids_path not found: {p}")
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # allow comma-separated or whitespace-separated
        s = s.split()[0]
        s = s.split(",")[0]
        ids.append(s)

    seen = set()
    out: List[str] = []
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out

def search_candidates(
    *,
    api_key: Optional[str] = None,
    material_ids: Optional[List[str]] = None,
    elements_all: Optional[List[str]] = None,
    elements_any: Optional[List[str]] = None,
    exclude_elements: Optional[List[str]] = None,
    spacegroups: Optional[List[int]] = None,
    crystal_systems: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    is_metal: Optional[bool] = None,
    include_deprecated: bool = False,
    theoretical: Optional[bool] = None,
    energy_above_hull_max: Optional[float] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Return MP summary docs as plain dicts (stable schema)."""

    band_gap_tuple = None
    if band_gap_min is not None or band_gap_max is not None:
        lo = float(band_gap_min) if band_gap_min is not None else 0.0
        hi = float(band_gap_max) if band_gap_max is not None else 10.0
        band_gap_tuple = (lo, hi)

    eah_tuple = None
    if energy_above_hull_max is not None:
        eah_tuple = (0.0, float(energy_above_hull_max))

    theor = theoretical if theoretical is not None else None

    fields = [
        "material_id","formula_pretty","symmetry","band_gap","is_stable","theoretical",
        "energy_above_hull","elements","deprecated"
    ]

    # If explicit IDs are provided, bypass the broad summary search.
    # We still fetch summary docs so we can attach metadata columns to outputs.
    if material_ids:
        # MP API can be sensitive to very large queries; chunk if needed.
        ids = [str(x) for x in material_ids]
        docs_all = []
        with get_mpr(api_key) as mpr:
            chunk = 200
            for j in range(0, len(ids), chunk):
                sub = ids[j:j+chunk]
                try:
                    docs = mpr.materials.summary.search(
                        material_ids=sub,
                        deprecated=include_deprecated,
                        fields=fields,
                        all_fields=False,
                    )
                except TypeError:
                    # alternate kw name used in some mp-api versions
                    docs = mpr.materials.summary.search(
                        material_id=sub,
                        deprecated=include_deprecated,
                        fields=fields,
                        all_fields=False,
                    )
                docs_all.extend(list(docs))

        # Convert to plain dicts (same schema as normal search)
        results: List[Dict[str, Any]] = []
        for d in docs_all:
            #sg_num = d.symmetry.number
            #sg_sym = d.symmetry.symbol
            results.append({
                "material_id": str(getattr(d, "material_id", None) or getattr(d, "material_id", "")),
                "formula_pretty": getattr(d, "formula_pretty", None),
                "symmetry": getattr(d, "symmetry", None),
                "sg_number": d.symmetry.number or None,
                "sg_symbol": d.symmetry.symbol or None,
                "crystal_system": d.symmetry.crystal_system or None,
                "band_gap": getattr(d, "band_gap", None),
                "is_stable": getattr(d, "is_stable", None),
                "theoretical": getattr(d, "theoretical", None),
                "energy_above_hull": getattr(d, "energy_above_hull", None),
                "elements": list(getattr(d, "elements", []) or []),
                "deprecated": getattr(d, "deprecated", None),
            })
        # Keep only up to limit, but default to all provided ids.
        if limit is not None and limit > 0:
            return results[: min(limit, len(results))]
        return results

    def _run_one(cs: Optional[str]):
        with get_mpr(api_key) as mpr:
            return mpr.materials.summary.search(
                band_gap=band_gap_tuple,
                elements=elements_all or None,
                exclude_elements=exclude_elements or None,
                spacegroup_number=spacegroups or None,
                crystal_system=cs if cs else None,
                is_metal=is_metal,
                deprecated=include_deprecated,
                theoretical=theor,
                energy_above_hull=eah_tuple,
                fields=fields,
                chunk_size=min(1000, max(100, limit)),
                num_chunks=max(1, math.ceil(limit/1000)),
                all_fields=False,
            )

    docs_all = []
    if crystal_systems:
        seen = set()
        for cs in crystal_systems:
            docs = _run_one(cs)
            for d in docs:
                mid = str(d.material_id)
                if mid in seen:
                    continue
                seen.add(mid)
                docs_all.append(d)
                if len(docs_all) >= limit:
                    break
            if len(docs_all) >= limit:
                break
    else:
        docs_all = list(_run_one(None))

    results: List[Dict[str, Any]] = []
    for d in docs_all:
        if elements_any and not any(e in d.elements for e in elements_any):
            continue   
        results.append({
            "material_id": str(d.material_id),
            "formula_pretty": getattr(d, "formula_pretty", None),
            "sg_symbol": d.symmetry.symbol if getattr(d, "symmetry", None) else None,
            "sg_number": d.symmetry.number if getattr(d, "symmetry", None) else None,
            "crystal_system": d.symmetry.crystal_system if getattr(d, "symmetry", None) else None,
            "band_gap": getattr(d, "band_gap", None),
            "is_stable": getattr(d, "is_stable", None),
            "theoretical": getattr(d, "theoretical", None),
            "energy_above_hull": getattr(d, "energy_above_hull", None),
            "elements": list(getattr(d, "elements", []) or []),
            "deprecated": getattr(d, "deprecated", None),
        })
        if len(results) >= limit:
            break
    return results

'''
def fetch_structure(material_id: str, api_key: Optional[str] = None) -> Structure:
    with get_mpr(api_key) as mpr:
        try:
            s = mpr.materials.get_structure_by_material_id(material_id)
        except Exception:
            s = mpr.get_structure_by_material_id(material_id)
    if s is None:
        raise ValueError(f"Could not retrieve structure for {material_id}.")
    return s
'''
def fetch_structure(
    material_id: str,
    api_key: Optional[str] = None,
    *,
    conventional: bool = True,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> Structure:
    """
    Fetch a structure from the Materials Project.

    Args:
        material_id: e.g. "mp-149"
        api_key: MP API key (or None if handled by your get_mpr wrapper)
        conventional: if True, return the standard conventional cell
        symprec/angle_tolerance: only used when we have to convert locally

    Returns:
        pymatgen Structure
    """
    with get_mpr(api_key) as mpr:
        s = None
        got_conventional_from_api = False

        # Prefer the top-level convenience method when conventional is requested:
        # it officially supports `conventional_unit_cell`.
        if conventional:
            try:
                s = mpr.get_structure_by_material_id(
                    material_id,
                    conventional_unit_cell=True,
                )
                got_conventional_from_api = True
            except TypeError:
                # Client exists but doesn't accept this kwarg
                pass
            except Exception:
                # Fall back to your original retrieval logic
                pass

        # Original retrieval logic (works broadly)
        if s is None:
            try:
                s = mpr.materials.get_structure_by_material_id(material_id)
            except Exception:
                s = mpr.get_structure_by_material_id(material_id)

    if s is None:
        raise ValueError(f"Could not retrieve structure for {material_id}.")

    # If we couldn't request a conventional cell directly, convert locally
    if conventional and not got_conventional_from_api:
        s = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance) \
            .get_conventional_standard_structure()

    return s
