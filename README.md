# Squarenet ML

**Squarenet ML** is a materials-informatics and machine-learning project for studying whether physically motivated descriptors of square-net-like structural motifs improve the prediction of inorganic materials properties.

The repository combines two related components:

1. **A crystal-structure analysis pipeline** that searches Materials Project structures for candidate square-net layers and converts their local geometry, chemistry, bonding environment, and neighboring-layer structure into quantitative descriptors.
2. **A machine-learning workflow** that treats those descriptors as materials-informatics features and tests whether they provide predictive information beyond inexpensive composition and global crystal-structure features.

The broader goal is not simply to classify structures as "square net" or "not square net." Instead, the detector is used as a **physics-informed feature generator**: each detection run converts local structural motifs into tabular descriptors that can be tested systematically against electronic and thermodynamic material properties.

---

## Motivation

Crystal structures contain considerably more information than composition or space group alone. Local coordination, dimensionality, layer separation, geometric distortion, chemical environment, and bonding topology can all influence electronic structure and thermodynamic stability.

Square-net motifs provide a useful test case because they can be described using interpretable structural quantities such as:

* in-plane nearest-neighbor distances,
* deviations from 90° bond geometry,
* uniformity of the in-plane network,
* layer spacing,
* composition of neighboring planes,
* local coordination and bonding,
* oxidation-state information,
* and the chemical identity of the atoms forming the candidate net.

Rather than reducing these structures to a single binary label, this project asks:

> **Can the intermediate structural information generated while detecting square-net motifs serve as useful machine-learning features for predicting material properties?**

This makes the detector both a scientific analysis tool and a feature-engineering pipeline.

---

## Research Questions

The project is organized around several related machine-learning questions.

### 1. Do square-net structural descriptors improve property prediction?

Starting from inexpensive baseline information such as composition and global crystal descriptors, progressively add information obtained from the detector:

```text
Composition + global structure
            ↓
Square-net detection labels
            ↓
Local layer geometry
            ↓
Layer chemistry and neighboring-plane environment
            ↓
Bonding / coordination descriptors
```

The objective is to determine whether these increasingly detailed structural representations provide measurable predictive value.

### 2. Which structural features contain the most useful information?

Because the detector produces physically interpretable quantities, model performance and feature importance can be connected back to structural questions:

* Does distortion of a candidate square net correlate with electronic behavior?
* Does layer separation contain information about stability or band gap?
* Does the chemical environment above and below a square-net layer matter?
* Are bonding and coordination descriptors more informative than a simple square-net classification?
* Is the presence of a square-net motif useful by itself, or is the continuous structural information generated during detection more important?

### 3. Can expensive structural analysis eventually be approximated from cheaper features?

A complementary experiment is to predict detector outcomes from composition and inexpensive global crystal descriptors.

If successful, this could provide a fast pre-screening model for identifying structures likely to contain interesting square-net motifs before running the full structural-analysis pipeline.

---

## Machine-Learning Targets

The current workflow focuses on Materials Project properties that provide both regression and related classification tasks.

### Regression

| Target                      | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `band_gap`                  | Calculated electronic band gap                   |
| `formation_energy_per_atom` | Formation energy per atom                        |
| `energy_above_hull`         | Energy relative to the thermodynamic convex hull |

### Classification

| Target                            | Description                                 |
| --------------------------------- | ------------------------------------------- |
| `is_metal`                        | Metal vs. nonmetal classification           |
| `is_stable` / hull classification | Stable or near-hull vs. above-hull material |

These paired tasks make it possible to investigate the same physical problem at different levels. For example, `band_gap` can be treated as a continuous regression target while `is_metal` provides a related binary classification problem.

Likewise, thermodynamic stability can be studied using continuous `energy_above_hull` values or a threshold-based stability classification.

---

## Feature Families

A central goal of this repository is to compare **nested feature sets** rather than train a single black-box model on every available column.

### Baseline — inexpensive material descriptors

Features available without running detailed square-net analysis, such as:

* elemental/compositional descriptors,
* number of species,
* crystallographic information,
* space group,
* crystal system,
* lattice or global structure descriptors.

This establishes how well conventional tabular materials-informatics features perform on their own.

### Detector labels

High-level outputs from square-net detection, for example:

* whether any candidate layer passes,
* number of detected layers,
* number or fraction of passing candidates,
* dominant candidate species,
* dominant crystallographic axis,
* detector pass/fail indicators.

These features test whether knowing that a material contains a square-net-like motif provides additional predictive information.

### Layer geometry

Continuous descriptors describing how closely candidate layers resemble an ideal square net, including quantities related to:

* in-plane nearest-neighbor distances,
* distance uniformity,
* angular deviations,
* local square-net scores,
* candidate-layer pass fractions,
* geometric tolerances and failure modes,
* plane spacing.

These retain substantially more information than a binary detector label.

### Layer chemistry and environment

Descriptors describing the chemical environment surrounding a candidate plane, including:

* square-net species,
* co-planar species,
* adjacent-plane composition,
* neighboring-plane separation,
* species counts and chemical context.

### Bonding and coordination

Optional CrystalNN-derived descriptors can add information about:

* bonded neighbors,
* coordination environments,
* oxidation-state summaries,
* out-of-plane bonding,
* bond-angle statistics,
* local chemical connectivity.

Together, these feature groups allow the project to test not only whether square nets matter, but **which aspects of the square-net environment carry useful information**.

---

## Experimental Design

A primary experiment compares models trained on progressively richer feature sets while keeping the data split, preprocessing, model family, and evaluation procedure fixed.

A typical comparison is:

```text
Experiment 1
Baseline
    composition + inexpensive global crystal descriptors

Experiment 2
Baseline + detector labels

Experiment 3
Baseline + layer geometry

Experiment 4
Baseline + layer chemistry / local environment

Experiment 5
Combined
    all available feature families
```

The important comparison is therefore not simply:

> Which model obtains the highest score?

but instead:

> **How much predictive information is added when physically motivated local structural descriptors are introduced?**

Paired comparisons using identical cross-validation folds make these differences easier to interpret than independent model scores.

---

## Workflow

The repository connects Materials Project data collection, structural analysis, feature engineering, exploratory analysis, and machine learning:

```text
Materials Project
       │
       ▼
Crystal structures + material properties
       │
       ▼
Structure preprocessing
       │
       ▼
Square-net candidate detection
       │
       ├── geometry
       ├── layer spacing
       ├── neighboring-plane chemistry
       ├── bonding / coordination
       └── detector diagnostics
       │
       ▼
Material-level feature tables
       │
       ▼
Data validation / exploratory analysis
       │
       ▼
Feature-set construction
       │
       ▼
Regression + classification experiments
       │
       ▼
Cross-validation / model comparison
       │
       ▼
Feature importance + physical interpretation
```

This separation is intentional: the detector generates reproducible structural descriptors, while the notebooks are used to investigate how those descriptors behave statistically and whether they improve predictive models.

---

## Square-Net Detection

The detector scans planes normal to the crystallographic `a`, `b`, and `c` axes, groups atoms into candidate layers, and evaluates the local in-plane geometry.

Candidate layers can then be characterized using:

* local square-lattice geometry,
* nearest-neighbor vectors,
* distance and angular deviations,
* adjacent-plane distances,
* adjacent-plane compositions,
* same-species out-of-plane bonding,
* CrystalNN coordination environments,
* oxidation-state and bonding descriptors.

The detector therefore produces considerably more information than a binary structural classification.

Each candidate `(axis, plane_id, species)` can be viewed as a local structural observation from which material-level descriptors are constructed.

---

## Materials Project Pipeline

The repository includes a configurable pipeline for:

1. querying Materials Project,
2. retrieving crystal structures,
3. preprocessing structures,
4. detecting candidate square-net layers,
5. computing structural and chemical descriptors,
6. aggregating detector results,
7. and writing tabular datasets for later analysis and machine learning.

Configuration is separated into:

* `MPQueryConfig` — Materials Project queries and API settings,
* `PreprocessConfig` — structure preprocessing,
* `DetectConfig` — detector and feature-generation settings,
* `OutputConfig` — output paths, formats, and resume behavior.

A YAML configuration can also be supplied to make detection runs reproducible.

```python
from squarenet import run_pipeline

materials_df, axis_species_df = run_pipeline("my_detector_config.yaml")
```

Run metadata and detector settings are stored alongside the generated datasets so that downstream ML experiments can be associated with the structural-analysis configuration used to create their features.

---

## Output Data

Detection runs produce several complementary data products.

### `materials.csv` / `materials.parquet`

One row per material.

This table contains material-level summaries suitable for exploratory analysis and machine-learning workflows.

Examples include:

* number of detected layers,
* number of tested axes/species,
* detector pass counts,
* total pass fraction,
* dominant candidate axis/species,
* aggregated structural descriptors,
* Materials Project metadata and property targets.

### `axis_species.csv` / `axis_species.parquet`

A more detailed table containing summaries for individual material/axis/species combinations.

This table preserves more of the local structural information generated during detection and is useful for understanding how the material-level representation was constructed.

### Run metadata

The pipeline records detector and preprocessing settings so that datasets can be traced back to the configuration used to generate them.

This is particularly important for machine-learning experiments because changes to detector thresholds effectively change the feature-generation process.

---

## Notebook Workflow

The notebooks serve as the experimental and explanatory layer of the project.

They are intended to document the progression from raw detector output to a validated machine-learning dataset:

```text
Data collection
      ↓
Detector demonstration
      ↓
Data validation
      ↓
Exploratory data analysis
      ↓
Feature construction
      ↓
Regression / classification experiments
      ↓
Model comparison
      ↓
Physical interpretation
```

The notebooks are deliberately separated from the core detector code. Reusable structural-analysis logic belongs in the `squarenet` package, while notebooks are used for:

* demonstrating the scientific workflow,
* inspecting individual structures,
* validating generated datasets,
* visualizing distributions,
* identifying missing or suspicious values,
* examining correlations,
* comparing feature families,
* training candidate models,
* and interpreting results.

---

## Model Evaluation Philosophy

This project is intended as an experiment in **feature value**, not simply model optimization.

For this reason, model comparisons should use:

* identical train/test or cross-validation splits,
* identical preprocessing where possible,
* multiple model families,
* repeated or grouped cross-validation when appropriate,
* uncertainty estimates across folds,
* paired fold-level comparisons between feature sets.

Useful regression metrics include:

* MAE,
* RMSE,
* \(R^2\).

Useful classification metrics include:

* ROC-AUC,
* PR-AUC,
* balanced accuracy,
* F1 score,
* confusion matrices,
* calibration where appropriate.

The most informative result is often the difference between two otherwise identical experiments:

```text
Δ score = score(structural features + baseline)
        - score(baseline)
```

rather than the absolute score of a single model.

---

## Interpretable Materials Informatics

One advantage of this approach over an entirely learned crystal representation is interpretability.

Features such as

```text
square-net distortion
layer separation
adjacent-plane chemistry
coordination number
bond geometry
candidate pass fraction
```

have direct structural meanings.

This allows machine-learning results to be used not only for prediction but also for hypothesis generation.

For example, if layer-separation or distortion descriptors consistently improve band-gap prediction, that result can motivate a more targeted physical investigation of those structural relationships.

The project therefore sits between conventional descriptor-based materials informatics and fully learned crystal representations such as graph neural networks.

---

## Visualization

The repository contains visualization utilities for inspecting candidate square-net layers and understanding detector decisions.

For example:

```python
from pymatgen.core import Structure
from squarenet.detect import find_square_net_planes
from squarenet.visualization import plot_detection_summary, save_figure

structure = Structure.from_file("structure.cif")

results = find_square_net_planes(
    structure,
    axes=("c", "a", "b"),
    preserve_visualization_data=True,
    compute_crystalnn_features=False,
)

result = results[0]

fig, axes = plot_detection_summary(
    structure,
    result,
    representative_site="worst",
)
```

Visual inspection is useful both for validating the detector and for understanding what the numerical descriptors represent physically.

---

## Repository Structure

```text
squarenet_ml/
│
├── configs/
│   └── detector / experiment configuration
│
├── notebooks/
│   └── data collection, validation, EDA, and ML experiments
│
├── outputs/
│   └── example detector outputs
│
├── reports/
│   └── generated figures and analysis outputs
│
├── squarenet/
│   ├── detector
│   ├── Materials Project pipeline
│   ├── preprocessing
│   ├── feature generation
│   └── visualization
│
├── tests/
│   └── detector and pipeline tests
│
├── detector_config.example.yaml
├── example_run.py
├── documentation.md
├── glossary.md
└── requirements.txt
```

---

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Major dependencies include:

* NumPy
* pandas
* SciPy
* scikit-learn
* pymatgen
* mp-api

`pyarrow` is recommended when working with Parquet datasets.

---

## Quick Start: Detecting Square Nets

For an existing `pymatgen.Structure`:

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

strong_candidates = [
    result for result in results
    if result.passes2
]
```

Each result contains geometric scores, pass/fail diagnostics, adjacent-plane descriptors, and optional bonding/chemistry information that can subsequently be aggregated into ML features.

---

## Tests

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests
```

---

## Documentation

See [`documentation.md`](documentation.md) for more detailed information about:

* detector parameters,
* scoring logic,
* result fields,
* `passes` vs. `passes2`,
* geometry thresholds,
* adjacent-plane analysis,
* CrystalNN descriptors.

See [`glossary.md`](glossary.md) for definitions of detector terminology and output quantities.

---

## Current Project Scope

The project is currently focused on **tabular, interpretable materials informatics** rather than attempting to replace crystal graph neural networks or other end-to-end structure-learning methods.

The primary objective is to establish whether domain-informed descriptors derived from square-net analysis contain measurable information about material properties.

Future comparisons could naturally include:

* composition-only models,
* generic crystal descriptors,
* square-net-specific descriptors,
* graph neural-network embeddings,
* or hybrid models combining learned and physics-informed representations.

Such comparisons would help determine when explicitly engineered structural knowledge is complementary to representations learned directly from crystal structures.

---

## Long-Term Goal

The broader goal is to turn structural motif detection into a reusable materials-informatics strategy:

```text
scientific hypothesis
       ↓
physically motivated detector
       ↓
interpretable structural descriptors
       ↓
large materials database
       ↓
machine-learning experiment
       ↓
quantify predictive value
       ↓
new physical hypotheses
```

Square-net materials are the first application, but the same workflow could be extended to other structural motifs where local geometry and chemistry may provide useful information that is not captured by composition alone.

