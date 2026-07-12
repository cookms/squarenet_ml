"""
Square-net screening package.

Core outputs:
- layers_df: per (material_id, species, axis, layer_id)
- species_df: per (material_id, species)
- materials_df: per material_id

Design note:
This package keeps heavy dependencies (pymatgen, mp-api) imported inside modules.
"""
from .config import MPQueryConfig, PreprocessConfig, DetectConfig, PipelineConfig, OutputConfig

def run_pipeline(cfg: PipelineConfig):
    from .pipeline import run_pipeline as _run
    return _run(cfg)

def detect_square_net_layers(*args, **kwargs):
    from .detect import detect_square_net_layers as _d
    return _d(*args, **kwargs)

def summarize_square_net_axes(*args, **kwargs):
    from .summarize import summarize_square_net_axes as _s
    return _s(*args, **kwargs)

def build_species_table(*args, **kwargs):
    from .summarize import build_species_table as _b
    return _b(*args, **kwargs)

def build_material_table(*args, **kwargs):
    from .summarize import build_material_table as _b
    return _b(*args, **kwargs)


def positions_dataframe(*args, **kwargs):
    from .positions import positions_dataframe as _p
    return _p(*args, **kwargs)

def wyckoff_summary(*args, **kwargs):
    from .positions import wyckoff_summary as _w
    return _w(*args, **kwargs)
