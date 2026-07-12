# Squarenet ML

Utilities for detecting candidate square-net layers in inorganic crystal
structures and generating layer-level features for filtering or machine
learning workflows.

The core detector scans planes normal to the crystallographic `a`, `b`, and
`c` axes, groups atoms into candidate layers, measures in-plane square-lattice
geometry, and can add CrystalNN-based bonding and chemistry features.

## Features

- Detect square-net-like planes by species and crystallographic axis.
- Score local square geometry from in-plane nearest-neighbor vectors.
- Compare candidate layers with adjacent-plane distance and composition
  descriptors.
- Apply stricter `passes2` filters for geometry, composition, and bonding.
- Optionally compute CrystalNN features for bonded neighbors, coordination,
  oxidation state summaries, and bond-angle descriptors.
- Run a Materials Project screening pipeline that writes material and
  axis/species summary tables.

## Installation

Create a Python environment, then install the listed dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are `numpy`, `pandas`, `scipy`, `pymatgen`, `mp-api`, and
`scikit-learn`. `pyarrow` is optional but recommended when writing Parquet
outputs.

## Quick Start

Use the low-level detector when you already have a pymatgen `Structure`:

```python
from pymatgen.core import Structure
from squarenet.detect import find_square_net_planes

structure = Structure.from_file("structure.cif")

results = find_square_net_planes(
    structure,
    axes=("c", "a", "b"),
    plane_tol=0.01,
    score_threshold=0.5,
    min_pass_fraction=0.6,
    enforce_no_out_of_plane_same_species_bonds=True,
    compute_crystalnn_features=True,
)

strong_candidates = [result for result in results if result.passes2]
```

Each result represents one candidate `(axis, plane_id, species)` layer and
includes geometric scores, adjacent-plane descriptors, pass/fail flags, and
optional CrystalNN-derived features.

## Materials Project Pipeline

The repository also includes a pipeline for querying Materials Project,
fetching structures, detecting square-net layers, and writing summary tables.

Edit `example_run.py` with your Materials Project API key and desired material
IDs or search filters, then run:

```bash
python example_run.py
```

The pipeline is configured with:

- `MPQueryConfig` for Materials Project API settings and query filters.
- `PreprocessConfig` for structure conversion and supercell options.
- `DetectConfig` for square-net detection settings.
- `OutputConfig` for output location, CSV/Parquet writing, and resume behavior.

## Outputs

Pipeline runs write outputs to the configured `out_dir`:

- `materials.csv` / `materials.parquet`: one row per material.
- `axis_species.csv` / `axis_species.parquet`: one row per material, axis, and
  species summary.
- `meta.json`: run configuration metadata.
- `processed_ids.txt`: resume/progress log.
- Optional CIF exports when enabled in `OutputConfig`.

## Documentation

See `documentation.md` for detailed information about detector arguments,
result fields, scoring logic, `passes` versus `passes2`, and CrystalNN feature
definitions.
