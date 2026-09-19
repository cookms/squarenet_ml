"""Configuration for the square-net feature-ablation experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    requested_splits: int = 5
    requested_repeats: int = 3
    demo_splits: int = 3
    demo_repeats: int = 3
    bootstrap_draws: int = 4000
    near_hull_threshold_ev_atom: float = 0.05
    metal_gap_threshold_ev: float = 1e-6
    ridge_alpha: float = 10.0
    logistic_c: float = 1.0
    onehot_min_frequency: int = 5
    demo_onehot_min_frequency: int = 2
    min_materials_for_claim: int = 100
    min_groups_for_claim: int = 50
    grouping_col: str = "reduced_formula_derived"
    data_path_override: str | None = None


CONFIG = ExperimentConfig()
REPO_ROOT = Path.cwd()
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiment_A"
FIGURE_DIR = OUTPUT_DIR / "figures"
