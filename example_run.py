"""Example Materials Project pipeline run.

Set the Materials Project API key in your environment before running:

    PowerShell: $env:MP_API_KEY = "your-key"
    bash/zsh:   export MP_API_KEY="your-key"

Do not paste real API keys into this file.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from squarenet.config import DetectConfig, MPQueryConfig, OutputConfig, PipelineConfig


def resolve_api_key() -> Optional[str]:
    """Return the Materials Project API key from supported environment variables."""
    return os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")


def build_config(
    *,
    api_key: Optional[str] = None,
    material_ids_path: str = "mpids.txt",
    out_dir: str = "squarenet_out",
    limit: int = 10,
) -> PipelineConfig:
    """Build the example pipeline configuration without starting a run."""
    return PipelineConfig(
        mp=MPQueryConfig(
            api_key=api_key,
            material_ids_path=material_ids_path,
            limit=limit,
        ),
        detect=DetectConfig(
            candidate_species=None,
            axes=("c", "a", "b"),
            plane_tol_A=0.15,
            min_atoms_per_plane=5,
            len_tol=0.10,
            angle_tol_deg=5.0,
            pass_tol=0.55,
            origin_trials=8,
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
