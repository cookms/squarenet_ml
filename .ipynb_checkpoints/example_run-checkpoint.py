from squarenet.config import MPQueryConfig, PreprocessConfig, DetectConfig, OutputConfig, PipelineConfig
from squarenet.pipeline import run_pipeline

cfg = PipelineConfig(
    mp=MPQueryConfig(
        api_key="MP_API_KEY",
        material_ids_path="mpids.txt",
        limit=10
    ),
    detect=DetectConfig(
        candidate_species=None,
        axes=("c","a","b"),
        plane_tol_A=0.15,
        min_atoms_per_plane=5,
        len_tol=0.10,
        angle_tol_deg=5.0,
        pass_tol=0.55,
        origin_trials=8,
    ),
    output=OutputConfig(
        out_dir="squarenet_outnew_allmpid",
        write_csv=True,
        write_parquet=True,
        resume=True,
        skip_existing=True,
        processed_log_name="processed_ids.txt",
        processed_log_append=True,
    ),
    meta={"note": "v5 processed log fix"},
)

materials_out, axis_species_out = run_pipeline(cfg)
