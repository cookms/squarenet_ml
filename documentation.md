# Square-net plane detection and feature extraction

This code identifies **candidate square-net layers** (2D square lattices) in an inorganic crystal structure by scanning planes normal to `a`, `b`, and/or `c`, measuring “squareness” using in-plane nearest-neighbor geometry, and optionally adding **CrystalNN-based bonding/chemistry features** for machine learning and filtering.

---

## Dependencies

- **Core**
  - `numpy`
  - `scipy.spatial.cKDTree`

- **Optional (only required if CrystalNN features/filter are enabled)**
  - `pymatgen.analysis.local_env.CrystalNN`
  - `pymatgen.analysis.bond_valence.BVAnalyzer` (for oxidation state guessing)

---

## High-level workflow

For each requested axis (`a`, `b`, `c`):

1. **Group sites into planes** perpendicular to the axis (periodic along that axis).
2. For each plane and each species present in that plane:
   - Compute **in-plane nearest-neighbor distances** (square size proxy).
   - Compute a **“squareness score”** for each site from local 2D neighbor geometry:
     - prefers **orthogonal directions** (~90°) and **equal lengths**,
     - requires approximate opposite vectors to exist (±u, ±v).
   - Aggregate scores → determine `passes` and compute summary statistics.
3. Compute **adjacent-plane** information (closest “prev/next” plane by atom distance and by plane spacing).
4. Compute `passes2`, a stricter pass flag that can apply additional numeric bounds, co-plane/adjacent composition constraints, and a **CrystalNN-based bond filter**.
5. If CrystalNN is enabled, compute **bond-based and chemical features** per candidate layer (for ML).

The function returns a list of `SquarePlaneResult` objects, sorted best-first.

---

## Public API

### `find_square_net_planes(structure, ...) -> List[SquarePlaneResult]`

#### Required input
- `structure`: a **pymatgen Structure-like** object with:
  - `structure.lattice.matrix` (3×3)
  - `structure.sites` each having `frac_coords` and `specie`

#### Returned value
- `List[SquarePlaneResult]`: one result per **(axis, plane_id, species)** candidate layer.

---

## Inputs (arguments)

### Plane scanning / squareness parameters

- `axes: Tuple[str, ...] = ("a","b","c")`  
  Which crystallographic axes to scan; planes are normal to each axis.

- `plane_tol: float = 0.01`  
  Fractional tolerance for grouping atoms into the same plane along a chosen axis.

- `species: Optional[Tuple[str, ...]] = None`  
  If provided, restrict analysis to these element symbols; otherwise analyze all species present.

- `k_nn: int = 9`  
  Number of nearest neighbors (in 2D tiled space) to consider when forming squareness vectors per site.

- `len_tol: float = 0.05`  
  Tolerance (relative error) used in the squareness length-match scoring.

- `ang_tol_deg: float = 7.0`  
  Tolerance (degrees) used in the squareness angle scoring around 90°.

- `min_pass_fraction: float = 0.6`  
  Fraction of sites in the plane+species group that must exceed `score_threshold` for `passes=True`.

- `score_threshold: float = 0.5`  
  Per-site squareness score cutoff.

- `return_all: bool = True`  
  If `False`, only candidates with `passes=True` are returned.

---

### Adjacent-plane mode

- `adjacent_by: str = "atom"`  
  Controls which “closest adjacent plane” is considered “primary” for some derived metrics:
  - `"atom"`: choose adjacent plane based on *minimum atom-to-atom distance*
  - `"plane"`: choose adjacent plane based on *plane spacing* (fractional distance between plane centers)

---

### Secondary pass criteria (`passes2`)

`passes2` begins as `passes` and then is refined by optional bounds and constraints.

**Numeric bounds (each enforced only if bounds are not `None`):**
- `nn_intra_min_min`, `nn_intra_min_max`
- `tol_ratio_any_min`, `tol_ratio_any_max`
- `min_adj_dist_any_atom_min`, `min_adj_dist_any_atom_max`
- `min_adj_dist_any_plane_min`, `min_adj_dist_any_plane_max`
- `closest_by_plane_sep_ang_min`, `closest_by_plane_sep_ang_max`

**Logical requirements:**
- `require_coplane_other_species: Optional[bool]`
  - `True`: require other species in same plane group
  - `False`: forbid other species in same plane group
  - `None`: ignore
- `require_adj_same_species: Optional[bool]`
  - `True`: require adjacent plane contains the same square-net species
  - `False`: forbid adjacent plane containing the same square-net species
  - `None`: ignore
- `adj_same_species_by: str = "atom"`  
  Which adjacent plane composition to use for that rule:
  - `"atom"`, `"plane"`, or `"primary"` (follows `adjacent_by`)

---

### CrystalNN bonding filter (`passes2`)

These control whether `passes2` can require the candidate square-net layer to have **no out-of-plane bonds to the same species** as the net:

- `enforce_no_out_of_plane_same_species_bonds: bool = True`  
  If `True`, `passes2` fails when any square-net-species site has a bonded neighbor of the **same species** that is classified as **out-of-plane**.

- `bond_in_plane_tol: Optional[float] = None`  
  Tolerance (fractional along the scanned axis) used to classify a bond as in-plane vs out-of-plane.
  - If `None`, uses `plane_tol`.

- `crystalnn_weight_cutoff: float = 0.0`  
  CrystalNN returns neighbor “weights”; neighbors below this are ignored.

- `crystalnn_kwargs: Optional[Dict] = None`  
  Passed directly to `CrystalNN(**crystalnn_kwargs)`.

---

### CrystalNN feature extraction toggle

- `compute_crystalnn_features: bool = True`  
  If `True`, CrystalNN-derived features are computed and stored in the result even if you’re not enforcing the bond filter.

---

### Oxidation state decoration (for CrystalNN)

CrystalNN can operate without oxidation states but may warn; this code attempts to decorate the structure:

- `guess_oxi_states_for_crystalnn: bool = True`  
  If `True`, tries to assign oxidation states before running CrystalNN.

- `bva_kwargs: Optional[Dict] = None`  
  Passed to `BVAnalyzer(**bva_kwargs)`. If `None`, defaults to `{"distance_scale_factor": 1.0}`.

- `bva_fallback_to_composition_guess: bool = True`  
  If BVAnalyzer fails, tries `structure.composition.oxi_state_guesses()` and decorates by element.

---

## Outputs

### `SquarePlaneResult` objects

Each result corresponds to a candidate layer defined by:
- `axis`: `"a"`, `"b"`, or `"c"` (normal direction)
- `plane_id`: index of the grouped plane along that axis
- `plane_center_frac`: median fractional coordinate along that axis for the plane group (wrapped into [0,1))
- `species`: element symbol for the candidate net species (e.g., `"Cu"`)
- `n_sites`: number of sites of that species in that plane group

---

## Interpretation of `passes` and `passes2`

### `passes`
Geometric “square-net-like” based on in-plane neighbor vectors:
- Compute per-site squareness scores.
- `passes=True` if `pass_fraction >= min_pass_fraction`, where:
  - `pass_fraction = fraction(scores >= score_threshold)`.

### `passes2`
Starts as `passes` and additionally enforces:
- optional numeric bounds,
- optional composition constraints,
- optional adjacent-plane same-species constraints,
- optional CrystalNN out-of-plane same-species bond rule:
  - If `enforce_no_out_of_plane_same_species_bonds=True`, then
    - `passes2=False` when a same-species bonded neighbor exists out of plane.

---

## What the algorithm measures

### (1) Squareness features (2D geometry)

Per candidate plane+species:
- `pass_fraction`, `mean_score`, `median_score`, `min_score`, `max_score`

Per-site squareness is based on:
- finding two non-collinear neighbor vectors `u` and `v`
- preferring:
  - `|u| ≈ |v|`
  - `angle(u,v) ≈ 90°`
  - existence of approximate opposite vectors `-u` and `-v`

### (2) Intralayer NN distance

- `nn_intra_min`: minimum in-plane nearest neighbor distance among sites of that species in that plane
- `nn_intra_mean`: mean of nearest-neighbor distances

### (3) Tolerance ratio

- `tol_ratio_any = nn_intra_min / min_adj`  
  where `min_adj` is the “primary” adjacent distance chosen by `adjacent_by` (atom vs plane mode).

### (4) Adjacent-plane descriptors

The code computes adjacency two ways:

**Atom-based adjacency (closest by atom-to-atom distance):**
- `min_adj_dist_any_atom`
- `closest_by_atom_side`: `"prev"` or `"next"`
- `closest_by_atom_plane_id`, `closest_by_atom_plane_center_frac`
- `closest_by_atom_plane_species_counts`
- `closest_by_atom_plane_major_species`, `closest_by_atom_plane_major_fraction`

**Plane-based adjacency (closest by plane spacing):**
- `min_adj_dist_any_plane` (atom distance to that plane)
- `closest_by_plane_side`, `closest_by_plane_plane_id`, `closest_by_plane_plane_center_frac`
- `closest_by_plane_plane_species_counts`
- `closest_by_plane_plane_major_species`, `closest_by_plane_plane_major_fraction`
- `closest_by_plane_sep_frac`: fractional plane separation
- `closest_by_plane_sep_ang`: separation converted to Å using reciprocal lattice scaling

### (5) Vector error statistics (from squareness step)

Aggregated across sites:
- `uv_len_err_mean`, `uv_ang_deg_mean`, `uv_ang_err_mean`
- min/max for `u` and `v` lengths and angle/length errors

---

## CrystalNN-derived features (bonding + chemistry)

These are aggregated over sites of the candidate net species in the plane.

### (6) Bond filter diagnostic
- `has_out_of_plane_same_species_bond`: whether any **same-species** bond exists out of plane for that candidate

### (7) Nearest bonded neighbors (in-plane/out-of-plane)
- `cnn_in_plane_nn_dist`, `cnn_in_plane_nn_species`
- `cnn_out_of_plane_nn_dist`, `cnn_out_of_plane_nn_species`

Interpretation:
- In-plane NN: shortest CrystalNN bond classified as in-plane
- Out-of-plane NN: shortest CrystalNN bond classified as out-of-plane

### Coordination number summaries
- `cnn_cn_mean`: mean total number of CrystalNN neighbors per site
- `cnn_cn_in_plane_mean`: mean in-plane neighbors per site
- `cnn_cn_out_of_plane_mean`: mean out-of-plane neighbors per site

### Bonded species counts
- `cnn_in_plane_bonded_species_counts`: total counts of bonded neighbor species classified in-plane
- `cnn_out_of_plane_bonded_species_counts`: total counts of bonded neighbor species classified out-of-plane

### Oxidation state summaries (for the net species sites)
- `square_species_oxi_state_mean`
- `square_species_oxi_state_std`

These come from the oxidation-state decorated `structure_nn` (BVAnalyzer or composition guess).

---

## CrystalNN bond-angle features

### (8) In-plane “best” bond angle near 90°
For each site:
- collect up to 4 shortest in-plane bond vectors
- choose the pair whose angle is closest to 90°
- aggregate over sites:

Outputs:
- `cnn_in_plane_bond_angle_deg_mean`
- `cnn_in_plane_bond_angle_deg_std`
- `cnn_in_plane_bond_angle_err90_mean` (mean absolute deviation from 90°)

### Out-of-plane tilt angle
For each site:
- take the shortest out-of-plane bond vector
- compute its tilt **relative to the plane**:
  - 0° = lies in-plane
  - 90° = perpendicular to plane
- aggregate over sites:

Outputs:
- `cnn_out_of_plane_tilt_angle_deg_mean`
- `cnn_out_of_plane_tilt_angle_deg_std`

### Optional out-of-plane pair angle
For each site:
- take the two shortest out-of-plane bond vectors
- compute the angle between them (often near 180° if there are symmetric bonds above and below)

Outputs:
- `cnn_out_of_plane_pair_angle_deg_mean`
- `cnn_out_of_plane_pair_angle_deg_std`

---

## Sorting of results

Results are sorted in descending order by:
1. `passes` (True before False)
2. `mean_score`
3. `pass_fraction`
4. `n_sites`

So “best” candidates appear first.

---

## Typical usage example

```python
# structure is a pymatgen Structure
results = find_square_net_planes(
    structure,
    axes=("c",),
    plane_tol=0.01,
    score_threshold=0.5,
    min_pass_fraction=0.6,
    enforce_no_out_of_plane_same_species_bonds=True,
    compute_crystalnn_features=True,
    guess_oxi_states_for_crystalnn=True,
    crystalnn_weight_cutoff=0.1,
)

# Keep only strong candidates
strong = [r for r in results if r.passes2]



