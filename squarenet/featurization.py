"""Feature engineering used by Experiment A.

The notebook intentionally imports these transformations instead of carrying their
implementation inline. This keeps the portfolio narrative focused on the data,
feature hypotheses, diagnostics, and experimental conclusions.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

try:
    from pymatgen.core import Composition, Element, Structure
except ImportError:  # Lightweight fallback for environments without pymatgen.
    Composition = None
    Element = None
    Structure = None


ELEMENT_SYMBOLS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y",
    "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
    "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir",
    "Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm",
    "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc",
    "Lv","Ts","Og",
]
ATOMIC_NUMBER = {symbol: i + 1 for i, symbol in enumerate(ELEMENT_SYMBOLS)}
METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}
HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po", "Lv"}
LANTHANIDES = set(ELEMENT_SYMBOLS[56:71])
ACTINIDES = set(ELEMENT_SYMBOLS[88:103])
NONMETALS = {"H","He","C","N","O","F","Ne","P","S","Cl","Ar","Se","Br","Kr","I","Xe","Rn","Og"}
STRUCTURE_FEATURE_COLUMNS = [
    "struct__density_g_cm3", "struct__volume_a3", "struct__volume_per_atom_a3",
    "struct__n_sites", "struct__lattice_a_a", "struct__lattice_b_a", "struct__lattice_c_a",
    "struct__lattice_b_over_a", "struct__lattice_c_over_a", "struct__lattice_c_over_b",
    "struct__alpha_deg", "struct__beta_deg", "struct__gamma_deg",
    "struct__effective_sphere_packing_fraction", "struct__radius_site_coverage",
    "struct__atomic_radius__mean", "struct__atomic_radius__std",
    "struct__atomic_radius__min", "struct__atomic_radius__max", "struct__atomic_radius__range",
]


def _fallback_parse_formula(formula: str) -> dict[str, float]:
    """Parse simple chemical formulas when pymatgen is unavailable."""
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula.replace(" ", ""))
    amounts: dict[str, float] = {}
    for symbol, amount_text in matches:
        if symbol not in ATOMIC_NUMBER:
            continue
        amount = float(amount_text) if amount_text else 1.0
        amounts[symbol] = amounts.get(symbol, 0.0) + amount
    if not amounts:
        raise ValueError(f"Could not parse formula: {formula!r}")
    return amounts


def weighted_stats(values: np.ndarray, fractions: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.isfinite(values) & np.isfinite(fractions)
    values = values[finite]
    fractions = fractions[finite]
    if len(values) == 0 or fractions.sum() <= 0:
        return {f"{prefix}__{name}": np.nan for name in ["mean", "std", "min", "max", "range"]}
    fractions = fractions / fractions.sum()
    mean = float(np.dot(values, fractions))
    variance = float(np.dot((values - mean) ** 2, fractions))
    return {
        f"{prefix}__mean": mean,
        f"{prefix}__std": math.sqrt(max(variance, 0.0)),
        f"{prefix}__min": float(values.min()),
        f"{prefix}__max": float(values.max()),
        f"{prefix}__range": float(values.max() - values.min()),
    }


MP_PROPERTY_COLUMNS = [
    "mp__band_gap_ev",
    "mp__energy_above_hull_ev_atom",
    "mp__formation_energy_per_atom_ev",
    "mp__is_metal",
    "mp__is_stable",
    "mp__theoretical",
]


def fetch_mp_properties(
    material_ids: Iterable[Any],
    api_key: str | None = None,
    batch_size: int = 500,
) -> dict[str, dict[str, Any]]:
    """Fetch selected Materials Project properties keyed by material ID."""
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError("Materials Project retrieval requires mp-api.") from exc

    ids = list(dict.fromkeys(
        text for value in material_ids
        if (text := clean_text_scalar(value)) is not None
    ))

    if not ids:
        return {}

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    fields = [
        "material_id",
        "band_gap",
        "energy_above_hull",
        "formation_energy_per_atom",
        "is_metal",
        "is_stable",
        "theoretical",
    ]

    properties = {}

    with MPRester(api_key=api_key) as mpr:
        for start in range(0, len(ids), batch_size):
            documents = mpr.materials.summary.search(
                material_ids=ids[start:start + batch_size],
                fields=fields,
            )

            for doc in documents:
                properties[str(doc.material_id)] = {
                    "mp__band_gap_ev": doc.band_gap,
                    "mp__energy_above_hull_ev_atom": doc.energy_above_hull,
                    "mp__formation_energy_per_atom_ev": doc.formation_energy_per_atom,
                    "mp__is_metal": doc.is_metal,
                    "mp__is_stable": doc.is_stable,
                    "mp__theoretical": doc.theoretical,
                }

    return properties


def add_mp_property_features(
    data: pd.DataFrame,
    material_id_column: str = "material_id",
    api_key: str | None = None,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Fetch Materials Project properties and append them to a dataframe."""
    if material_id_column not in data.columns:
        raise KeyError(
            f"Material-ID column {material_id_column!r} is not present."
        )

    properties = fetch_mp_properties(
        data[material_id_column],
        api_key=api_key,
        batch_size=batch_size,
    )

    result = data.copy()

    feature_rows = []
    for value in result[material_id_column]:
        material_id = clean_text_scalar(value)

        if material_id is None or material_id not in properties:
            feature_rows.append({
                column: np.nan
                for column in MP_PROPERTY_COLUMNS
            })
        else:
            feature_rows.append(properties[material_id])

    features = pd.DataFrame(feature_rows, index=result.index)

    return pd.concat([result, features], axis=1)


def composition_record(formula: Any) -> dict[str, Any]:
    """Turn one chemical formula into compact composition statistics."""
    if pd.isna(formula):
        return {}
    formula = str(formula)
    if Composition is not None:
        composition = Composition(formula)
        amounts = {el.symbol: float(amount) for el, amount in composition.items()}
        reduced_formula = composition.reduced_formula
    else:
        amounts = _fallback_parse_formula(formula)
        reduced_formula = formula.replace(" ", "")

    chemical_system = "-".join(sorted(amounts))
    elements = list(amounts)
    raw_amounts = np.array([amounts[e] for e in elements], dtype=float)
    total = float(raw_amounts.sum())
    fractions = raw_amounts / total

    record: dict[str, Any] = {
        "reduced_formula_derived": reduced_formula,
        "chemical_system_derived": chemical_system,
        "comp__n_elements": float(len(elements)),
        "comp__total_atoms_formula": total,
        "comp__stoich_entropy": float(scipy_entropy(fractions)) if len(fractions) else np.nan,
        "comp__max_element_fraction": float(fractions.max()),
        "comp__min_element_fraction": float(fractions.min()),
        "comp__fraction_l2": float(np.sqrt(np.sum(fractions**2))),
        "comp__fraction_metal": float(sum(f for e, f in zip(elements, fractions) if e not in NONMETALS and e not in METALLOIDS)),
        "comp__fraction_metalloid": float(sum(f for e, f in zip(elements, fractions) if e in METALLOIDS)),
        "comp__fraction_halogen": float(sum(f for e, f in zip(elements, fractions) if e in HALOGENS)),
        "comp__fraction_chalcogen": float(sum(f for e, f in zip(elements, fractions) if e in CHALCOGENS)),
        "comp__fraction_lanthanide": float(sum(f for e, f in zip(elements, fractions) if e in LANTHANIDES)),
        "comp__fraction_actinide": float(sum(f for e, f in zip(elements, fractions) if e in ACTINIDES)),
    }
    record.update(weighted_stats(np.array([ATOMIC_NUMBER.get(e, np.nan) for e in elements]), fractions, "comp__atomic_number"))

    if Element is not None:
        property_getters = {
            "atomic_mass": lambda el: float(el.atomic_mass),
            "electronegativity": lambda el: float(el.X) if el.X is not None else np.nan,
            "row": lambda el: float(el.row) if el.row is not None else np.nan,
            "group": lambda el: float(el.group) if el.group is not None else np.nan,
            "atomic_radius": lambda el: float(el.atomic_radius_calculated or el.atomic_radius)
            if (el.atomic_radius_calculated or el.atomic_radius) is not None else np.nan,
        }
        pymatgen_elements = [Element(e) for e in elements]
        for property_name, getter in property_getters.items():
            values = np.array([getter(el) for el in pymatgen_elements], dtype=float)
            record.update(weighted_stats(values, fractions, f"comp__{property_name}"))
    return record


def add_composition_features(data: pd.DataFrame, formula_column: str = "formula") -> pd.DataFrame:
    """Append formula-derived composition descriptors to a material table."""
    features = pd.DataFrame([composition_record(v) for v in data[formula_column]], index=data.index)
    return pd.concat([data, features], axis=1)


def _element_radius(element: Any) -> float:
    """Return a consistent elemental radius in angstrom, or NaN if unavailable."""
    radius = element.atomic_radius_calculated or element.atomic_radius
    return float(radius) if radius is not None else np.nan


def _coerce_structure(value: Any) -> Any:
    """Accept a pymatgen Structure or an MP-style structure dictionary."""
    if Structure is None:
        raise ImportError("Structural features require pymatgen.")
    if isinstance(value, Structure):
        return value
    if isinstance(value, Mapping):
        return Structure.from_dict(dict(value))
    raise TypeError(f"Expected a pymatgen Structure or structure mapping, got {type(value).__name__}.")


def structure_record(structure: Any) -> dict[str, float]:
    """Calculate cell, lattice, radius, and sphere-packing descriptors.

    The effective packing fraction treats each occupied site as a hard sphere
    with the pymatgen calculated atomic radius (falling back to atomic_radius).
    It is therefore a geometric descriptor, not a crystallographic refinement
    of ionic or bonding radii.
    """
    if is_missing_scalar(structure):
        return {column: np.nan for column in STRUCTURE_FEATURE_COLUMNS}
    structure = _coerce_structure(structure)
    lattice = structure.lattice
    a, b, c = (float(x) for x in lattice.abc)
    volume = float(structure.volume)
    n_sites = float(structure.num_sites)

    radius_values: list[float] = []
    radius_weights: list[float] = []
    occupied_sphere_volume = 0.0
    for site in structure:
        for species, occupancy in site.species.items():
            # Species may be an Element, Specie, or DummySpecies. Only real
            # elements have the radius data needed by this descriptor.
            symbol = getattr(species, "symbol", None)
            try:
                radius = _element_radius(Element(symbol)) if symbol is not None else np.nan
            except (TypeError, ValueError):
                radius = np.nan
            if np.isfinite(radius):
                weight = float(occupancy)
                radius_values.append(radius)
                radius_weights.append(weight)
                occupied_sphere_volume += weight * (4.0 * math.pi / 3.0) * radius**3

    if radius_values:
        radius_features = weighted_stats(
            np.asarray(radius_values, dtype=float),
            np.asarray(radius_weights, dtype=float),
            "struct__atomic_radius",
        )
        radius_coverage = float(sum(radius_weights) / n_sites) if n_sites > 0 else np.nan
    else:
        radius_features = weighted_stats(np.array([]), np.array([]), "struct__atomic_radius")
        radius_coverage = 0.0 if n_sites > 0 else np.nan

    record = {
        "struct__density_g_cm3": float(structure.density),
        "struct__volume_a3": volume,
        "struct__volume_per_atom_a3": volume / n_sites if n_sites > 0 else np.nan,
        "struct__n_sites": n_sites,
        "struct__lattice_a_a": a,
        "struct__lattice_b_a": b,
        "struct__lattice_c_a": c,
        "struct__lattice_b_over_a": b / a if a > 0 else np.nan,
        "struct__lattice_c_over_a": c / a if a > 0 else np.nan,
        "struct__lattice_c_over_b": c / b if b > 0 else np.nan,
        "struct__alpha_deg": float(lattice.alpha),
        "struct__beta_deg": float(lattice.beta),
        "struct__gamma_deg": float(lattice.gamma),
        "struct__effective_sphere_packing_fraction": (
            occupied_sphere_volume / volume if volume > 0 else np.nan
        ),
        "struct__radius_site_coverage": radius_coverage,
    }
    record.update(radius_features)
    return record


def add_structure_features(
    data: pd.DataFrame,
    structure_column: str = "structure",
) -> pd.DataFrame:
    """Append descriptors calculated from pymatgen/Materials Project structures."""
    if structure_column not in data.columns:
        raise KeyError(f"Structure column {structure_column!r} is not present.")
    features = pd.DataFrame(
        [structure_record(value) for value in data[structure_column]],
        index=data.index,
    )
    return pd.concat([data, features], axis=1)


def fetch_mp_structures(
    material_ids: Iterable[Any],
    api_key: str | None = None,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Fetch final structures from Materials Project, keyed by material ID.

    ``api_key`` may be omitted when ``MP_API_KEY`` is configured. Fetching is
    intentionally separate from featurization so callers can cache structures
    and avoid spending API requests on every feature-engineering run. Material
    IDs are requested in batches because the MP API limits the length of an ID
    filter.
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError("Materials Project retrieval requires mp-api.") from exc

    ids = list(dict.fromkeys(
        text for value in material_ids if (text := clean_text_scalar(value)) is not None
    ))
    if not ids:
        return {}
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    structures: dict[str, Any] = {}
    with MPRester(api_key=api_key) as mpr:
        for start in range(0, len(ids), batch_size):
            documents = mpr.materials.summary.search(
                material_ids=ids[start:start + batch_size],
                fields=["material_id", "structure"],
            )
            structures.update(
                {str(doc.material_id): doc.structure for doc in documents}
            )
    return structures


def add_mp_structure_features(
    data: pd.DataFrame,
    material_id_column: str = "material_id",
    api_key: str | None = None,
    batch_size: int = 500,
    keep_structure: bool = False,
) -> pd.DataFrame:
    """Fetch MP structures and append all structure-derived descriptors.

    Rows whose IDs are missing or not returned by Materials Project receive
    NaN structural features. Duplicate material IDs are fetched only once.
    """
    if material_id_column not in data.columns:
        raise KeyError(f"Material-ID column {material_id_column!r} is not present.")
    structures = fetch_mp_structures(
        data[material_id_column],
        api_key=api_key,
        batch_size=batch_size,
    )
    result = data.copy()
    result["structure"] = [
        structures.get(text) if (text := clean_text_scalar(value)) is not None else None
        for value in result[material_id_column]
    ]
    result = add_structure_features(result, structure_column="structure")
    return result if keep_structure else result.drop(columns="structure")


def is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def clean_text_scalar(value: Any) -> str | None:
    if is_missing_scalar(value):
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "<na>", "nat"} else text


def normalize_cv_groups(data: pd.DataFrame, group_column: str) -> pd.Series:
    """Create non-null plain-string group labels suitable for sklearn splitters."""
    groups = []
    for row_number, value in enumerate(data[group_column].tolist()):
        text = clean_text_scalar(value)
        groups.append(f"formula::{text}" if text is not None else f"row::{row_number:08d}")
    return pd.Series(groups, index=data.index, dtype=object)


def parse_count_mapping(value: Any) -> dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, Mapping):
        mapping = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            return {}
        try:
            mapping = json.loads(text)
        except Exception:
            return {}
    cleaned: dict[str, float] = {}
    for key, amount in mapping.items():
        number = pd.to_numeric(pd.Series([amount]), errors="coerce").iloc[0]
        if pd.notna(number) and float(number) >= 0:
            cleaned[str(key)] = float(number)
    return cleaned


def count_mapping_summary(value: Any, prefix: str) -> dict[str, float]:
    mapping = parse_count_mapping(value)
    if not mapping:
        return {
            f"{prefix}__total": 0.0,
            f"{prefix}__richness": 0.0,
            f"{prefix}__entropy": 0.0,
            f"{prefix}__concentration": 0.0,
            f"{prefix}__max_fraction": np.nan,
        }
    counts = np.array(list(mapping.values()), dtype=float)
    total = float(counts.sum())
    fractions = counts / total if total > 0 else np.zeros_like(counts)
    return {
        f"{prefix}__total": total,
        f"{prefix}__richness": float(len(mapping)),
        f"{prefix}__entropy": float(scipy_entropy(fractions)) if total > 0 else 0.0,
        f"{prefix}__concentration": float(np.sum(fractions**2)) if total > 0 else 0.0,
        f"{prefix}__max_fraction": float(fractions.max()) if total > 0 else np.nan,
    }


def expand_json_count_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Replace high-cardinality count JSON fields with compact diversity summaries."""
    result = data.copy()
    json_columns = [c for c in result.columns if str(c).lower().endswith("counts_json")]
    for column in json_columns:
        prefix = "chemjson__" + re.sub(r"_counts_json$", "", column)
        expanded = pd.DataFrame([count_mapping_summary(v, prefix) for v in result[column]], index=result.index)
        result = pd.concat([result, expanded], axis=1)
    return result, json_columns
