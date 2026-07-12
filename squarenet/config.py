from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Union, Dict, Any


@dataclass(frozen=True)
class MPQueryConfig:
    api_key: Optional[str] = None
    # Optional: provide explicit material IDs to analyze (bypasses search)
    material_ids: Optional[List[str]] = None
    # Optional: path to a .txt file with one material_id per line (comments starting with # allowed)
    material_ids_path: Optional[str] = None
    elements_all: Optional[List[str]] = None
    elements_any: Optional[List[str]] = None
    exclude_elements: Optional[List[str]] = None
    spacegroups: Optional[List[int]] = None
    crystal_systems: Optional[List[str]] = None
    band_gap_min: Optional[float] = None
    band_gap_max: Optional[float] = None
    is_metal: Optional[bool] = None
    include_deprecated: bool = False
    theoretical: Optional[bool] = None
    energy_above_hull_max: Optional[float] = None
    limit: int = 500


@dataclass(frozen=True)
class PreprocessConfig:
    # Which structure to pass into detection:
    #   raw: fetched structure, unchanged
    #   conventional: conventional standard cell, no supercell
    #   processed: conventional structure plus configured supercell/sym_supercell
    structure_source: str = "raw"
    to_conventional: bool = True
    symprec: float = 1e-3
    angle_tolerance: float = 5.0

    # either supercell OR sym_supercell can be used. sym_supercell recenters around 0 (Cartesian)
    supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None
    sym_supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = (3, 3, 3)


@dataclass(frozen=True)
class DetectConfig:
    # Primary square-net scan controls. These map directly to
    # find_square_net_planes(...).
    axes: Sequence[str] = ("a", "b", "c")
    plane_tol: float = 0.01
    species: Optional[Sequence[str]] = None
    k_nn: int = 9
    len_tol: float = 0.05
    ang_tol_deg: float = 7.0
    min_pass_fraction: float = 0.6
    score_threshold: float = 0.5
    return_all: bool = True
    preserve_visualization_data: bool = False
    adjacent_by: str = "atom"

    # Secondary pass criteria (None means ignore the bound).
    nn_intra_min_min: Optional[float] = None
    nn_intra_min_max: Optional[float] = 4.0
    tol_ratio_any_min: Optional[float] = None
    tol_ratio_any_max: Optional[float] = None
    min_adj_dist_any_atom_min: Optional[float] = 2.0
    min_adj_dist_any_atom_max: Optional[float] = None
    min_adj_dist_any_plane_min: Optional[float] = None
    min_adj_dist_any_plane_max: Optional[float] = None
    closest_by_plane_sep_ang_min: Optional[float] = None
    closest_by_plane_sep_ang_max: Optional[float] = None
    adj_same_species_by: str = "atom"
    forbid_coplane_mixed_species: Optional[bool] = True
    isolate_same_species_adjacent: Optional[bool] = True
    isolate_same_species_adjacent_dist_min: Optional[float] = 2.0

    # CrystalNN bond filter and feature controls.
    enforce_no_out_of_plane_same_species_bonds: bool = True
    bond_in_plane_tol: Optional[float] = None
    crystalnn_weight_cutoff: float = 0.0
    crystalnn_kwargs: Optional[Dict[str, Any]] = None
    compute_crystalnn_features: bool = True
    guess_oxi_states_for_crystalnn: bool = True
    bva_kwargs: Optional[Dict[str, Any]] = None
    bva_fallback_to_composition_guess: bool = True

    # Backward-compatible aliases from the older detector configuration.
    # Prefer species, plane_tol, ang_tol_deg, and min_pass_fraction in new code.
    candidate_species: Optional[Sequence[str]] = None
    plane_tol_A: Optional[float] = None
    angle_tol_deg: Optional[float] = None
    pass_tol: Optional[float] = None

    # Legacy options retained for old config files; not used by
    # find_square_net_planes.
    int_tol: float = 0.25
    n_candidates: int = 6
    origin_trials: int = 10
    min_atoms_per_plane: int = 6


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "squarenet_out"
    # Name of a simple progress log file stored in out_dir.
    processed_log_name: str = "processed_ids.txt"
    # If True, append run stamps to processed_log_name each run.
    processed_log_append: bool = True
    # If True, load existing outputs in out_dir and append new results (deduplicated)
    resume: bool = True
    # If resume is True, skip material_ids already present in existing materials table
    skip_existing: bool = True
    export_cifs: bool = False
    export_positive_only: bool = True
    write_csv: bool = True
    write_parquet: bool = True
    abridged_summary: bool = False
    flush_every: int = 100


@dataclass(frozen=True)
class PipelineConfig:
    mp: MPQueryConfig = field(default_factory=MPQueryConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # attach arbitrary metadata to stamp into outputs
    meta: Dict[str, Any] = field(default_factory=dict)
