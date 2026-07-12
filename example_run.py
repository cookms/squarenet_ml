"""Example Materials Project pipeline run.

Set the Materials Project API key in your environment before running:

    PowerShell: $env:MP_API_KEY = "your-key"
    bash/zsh:   export MP_API_KEY="your-key"

Do not paste real API keys into this file.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from squarenet.config import DetectConfig, MPQueryConfig, OutputConfig, PipelineConfig, PreprocessConfig


def resolve_api_key() -> Optional[str]:
    """Return the Materials Project API key from supported environment variables."""
    return os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")


def build_config(
    *,
    api_key: Optional[str] = None,
    material_ids_path: str = None, #"mpids.txt",
    out_dir: str = "squarenet_out",
    limit: int = 20,
) -> PipelineConfig:
    """Build the example pipeline configuration without starting a run."""
    return PipelineConfig(
        mp=MPQueryConfig(
            api_key=api_key,
            material_ids_path=material_ids_path,
            limit=limit,
        ),
        preprocess=PreprocessConfig(
            structure_source="processed",
            to_conventional=True,
            symprec=1e-3,
            angle_tolerance=5.0,
            supercell=None,
            sym_supercell=(3, 3, 3),
        ),
        detect=DetectConfig(
            species=None,
            axes=("c", "a", "b"),
            plane_tol=0.01,
            k_nn=9,
            len_tol=0.10,
            ang_tol_deg=5.0,
            min_pass_fraction=0.55,
            score_threshold=0.5,
            return_all=True,
            adjacent_by="atom",
            nn_intra_min_max=4.0,
            min_adj_dist_any_atom_min=2.0,
            forbid_coplane_mixed_species=True,
            isolate_same_species_adjacent=True,
            isolate_same_species_adjacent_dist_min=2.0,
            enforce_no_out_of_plane_same_species_bonds=False,
            compute_crystalnn_features=False,
            crystalnn_weight_cutoff=0.0,
        ),
        output=OutputConfig(
            out_dir=out_dir,
            write_csv=True,
            write_parquet=False,
            resume=True,
            skip_existing=True,
            processed_log_name="processed_ids.txt",
            processed_log_append=True,
        ),
        meta={"note": "example Materials Project square-net screen"},
    )


def main() -> Tuple[object, object]:
    """Run the example pipeline."""
    from squarenet.pipeline import run_pipeline

    cfg = build_config(api_key=resolve_api_key())
    materials_out, axis_species_out = run_pipeline(cfg)
    print(f"Wrote {len(materials_out)} material rows to {cfg.output.out_dir}")
    print(f"Wrote {len(axis_species_out)} axis/species rows to {cfg.output.out_dir}")
    return materials_out, axis_species_out


if __name__ == "__main__":
    main()
