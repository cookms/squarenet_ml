import os
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
    to_conventional: bool = True
    symprec: float = 1e-3
    angle_tolerance: float = 5.0

    # either supercell OR sym_supercell can be used. sym_supercell recenters around 0 (Cartesian)
    supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None
    sym_supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = (3, 3, 3)


@dataclass(frozen=True)
class DetectConfig:
    candidate_species: Optional[Sequence[str]] = None
    axes: Sequence[str] = ("c", "a", "b")

    # plane binning thickness along the axis-normal direction, in Angstrom
    plane_tol_A: float = 0.05

    # square-lattice tolerances
    len_tol: float = 0.10
    angle_tol_deg: float = 10.0
    int_tol: float = 0.25
    pass_tol: float = 0.40
    n_candidates: int = 6
    origin_trials: int = 10

    # Minimum atoms per plane to attempt lattice detection
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
