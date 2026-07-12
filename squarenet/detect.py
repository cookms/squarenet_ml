
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
from scipy.spatial import cKDTree
from functools import lru_cache


# --- helpers ---

def _unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def _group_planes_periodic(frac: np.ndarray, axis: int, tol: float) -> List[np.ndarray]:
    x = frac[:, axis] % 1.0
    order = np.argsort(x)
    xs = x[order]

    groups = []
    cur = [order[0]]
    for i in range(1, len(xs)):
        if (xs[i] - xs[i - 1]) <= tol:
            cur.append(order[i])
        else:
            groups.append(np.array(cur, dtype=int))
            cur = [order[i]]
    groups.append(np.array(cur, dtype=int))

    if len(groups) >= 2:
        first = groups[0]
        last = groups[-1]
        if (xs[0] + 1.0 - xs[-1]) <= tol:
            merged = np.concatenate([last, first])
            groups = [merged] + groups[1:-1]

    return groups

def _plane_basis_from_lattice(lattice_matrix: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    other = [0, 1, 2]
    other.remove(axis)
    t1 = lattice_matrix[other[0]]
    t2 = lattice_matrix[other[1]]

    e1 = _unit(t1)
    t2p = t2 - np.dot(t2, e1) * e1
    e2 = _unit(t2p)
    return e1, e2

def _tile_in_plane(cart: np.ndarray, lattice_matrix: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    other = [0, 1, 2]
    other.remove(axis)
    v1 = lattice_matrix[other[0]]
    v2 = lattice_matrix[other[1]]

    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shifts.append(i * v1 + j * v2)
    shifts = np.array(shifts)

    n = len(cart)
    tiled = (cart[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    origin = np.tile(np.arange(n, dtype=int), len(shifts))
    return tiled, origin

def _in_plane_image_offsets(axis: int) -> np.ndarray:
    """Return fractional image offsets in the same order as _tile_in_plane."""
    other = [0, 1, 2]
    other.remove(axis)
    offsets = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            off = np.zeros(3, dtype=int)
            off[other[0]] = i
            off[other[1]] = j
            offsets.append(off)
    return np.array(offsets, dtype=int)

def _square_score_site(
    v2d: np.ndarray,
    len_tol: float,
    ang_tol_deg: float,
    return_details: bool = False,
):
    nan_info = {"du": float("nan"), "dv": float("nan"), "len_err": float("nan"), "ang_deg": float("nan"), "ang_err": float("nan")}
    empty_details = {
        "scored_neighbor_positions": [],
        "u_position": None,
        "v_position": None,
        "opposite_u_position": None,
        "opposite_v_position": None,
        "has_opposite_u": False,
        "has_opposite_v": False,
    }

    if v2d.shape[0] < 4:
        return (0.0, nan_info, empty_details) if return_details else (0.0, nan_info)

    d = np.linalg.norm(v2d, axis=1)
    order = np.argsort(d)
    v = v2d[order[:8]]
    details = dict(empty_details)
    details["scored_neighbor_positions"] = [int(x) for x in order[:8].tolist()]

    u = v[0]
    u_pos = int(order[0])
    details["u_position"] = u_pos
    du = np.linalg.norm(u)
    if du < 1e-12:
        return (0.0, nan_info, details) if return_details else (0.0, nan_info)

    vvec = None
    v_pos = None
    for local_pos, cand in enumerate(v[1:], start=1):
        dv = np.linalg.norm(cand)
        if dv < 1e-12:
            continue
        cross = abs(u[0] * cand[1] - u[1] * cand[0])
        if cross / (du * dv + 1e-15) > np.sin(np.deg2rad(10.0)):
            vvec = cand
            v_pos = int(order[local_pos])
            break
    if vvec is None:
        return (0.0, nan_info, details) if return_details else (0.0, nan_info)
    details["v_position"] = v_pos

    dv = np.linalg.norm(vvec)

    cosang = np.dot(u, vvec) / (du * dv + 1e-15)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_deg = float(np.rad2deg(np.arccos(cosang)))
    ang_err = float(abs(ang_deg - 90.0))

    len_err = float(abs(du - dv) / max((du + dv) * 0.5, 1e-12))

    def opposite_position(vec: np.ndarray, vecs: np.ndarray, source_positions: np.ndarray, tol: float):
        target = -vec
        dd = np.linalg.norm(vecs - target[None, :], axis=1)
        if dd.size == 0:
            return False, None
        imin = int(np.argmin(dd))
        ok = float(dd[imin]) <= tol
        return ok, int(source_positions[imin]) if ok else None

    opp_tol = 0.15 * max(du, dv)
    has_opp_u, opp_u_pos = opposite_position(u, v, order[:8], opp_tol)
    has_opp_v, opp_v_pos = opposite_position(vvec, v, order[:8], opp_tol)
    details["has_opposite_u"] = bool(has_opp_u)
    details["has_opposite_v"] = bool(has_opp_v)
    details["opposite_u_position"] = opp_u_pos
    details["opposite_v_position"] = opp_v_pos
    opp_ok = has_opp_u and has_opp_v
    if not opp_ok:
        info = {"du": du, "dv": dv, "len_err": len_err, "ang_deg": ang_deg, "ang_err": ang_err}
        return (0.0, info, details) if return_details else (0.0, info)

    s_len = float(np.exp(-(len_err / max(len_tol, 1e-6)) ** 2))
    s_ang = float(np.exp(-(ang_err / max(ang_tol_deg, 1e-6)) ** 2))
    score = s_len * s_ang

    info = {"du": du, "dv": dv, "len_err": len_err, "ang_deg": ang_deg, "ang_err": ang_err}
    return (score, info, details) if return_details else (score, info)


def _tile_shifts_3d(lattice_matrix: np.ndarray) -> np.ndarray:
    a, b, c = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]
    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shifts.append(i * a + j * b + k * c)
    return np.array(shifts, dtype=float)

def _tile_points(cart: np.ndarray, shifts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tile arbitrary 3D points by provided shifts.
    Returns (tiled_points, origin_idx) where origin_idx maps tiled -> original index in cart.
    """
    n = len(cart)
    tiled = (cart[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    origin = np.tile(np.arange(n, dtype=int), len(shifts))
    return tiled, origin

def _counts(items: np.ndarray) -> Dict[str, int]:
    # items is dtype object (strings)
    out: Dict[str, int] = {}
    for x in items.tolist():
        out[x] = out.get(x, 0) + 1
    return out

def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num / den)


@dataclass
class LayerVisualizationData:
    """Optional detector diagnostics used by squarenet.visualization.

    The arrays here are not needed for tabular ML exports and are only
    populated when find_square_net_planes(..., preserve_visualization_data=True)
    is requested.
    """

    candidate_site_indices: np.ndarray
    candidate_plane_indices: np.ndarray
    adjacent_plane_indices_by_side: Dict[str, np.ndarray]
    adjacent_plane_centers_frac_by_side: Dict[str, float]
    plane_cartesian_basis: np.ndarray
    plane_normal: np.ndarray
    plane_center_cartesian: np.ndarray
    projected_coordinates: np.ndarray
    tiled_projected_coordinates: np.ndarray
    tiled_origin_indices: np.ndarray
    tiled_image_offsets: np.ndarray
    local_site_scores: np.ndarray
    local_site_pass_flags: np.ndarray
    local_site_details: List[Dict[str, Any]]
    selected_neighbor_edges: List[Dict[str, Any]]
    adjacent_atom_connections: Dict[str, Dict[str, Any]]
    neighbor_length_measurements: np.ndarray
    neighbor_angle_measurements: np.ndarray
    detector_thresholds: Dict[str, float]


@dataclass
class SquarePlaneResult:
    axis: str
    plane_id: int
    plane_center_frac: float
    species: str
    n_sites: int

    # label / filter info for ML
    passes: bool
    passes2: bool

    # (1) squareness features
    pass_fraction: float
    mean_score: float
    median_score: float
    min_score: float
    max_score: float

    # (2) intralayer NN distance (square size proxy)
    nn_intra_min: float
    nn_intra_mean: float

    # (3) tolerance ratio
    tol_ratio_any: float

    # (4) adjacent-layer distance and attribution
    min_adj_dist_any_atom: float = float("nan")
    closest_by_atom_side: Optional[str] = None
    closest_by_atom_plane_id: Optional[int] = None
    closest_by_atom_plane_center_frac: float = float("nan")
    closest_by_atom_plane_species_counts: Dict[str, int] = field(default_factory=dict)
    closest_by_atom_plane_major_species: Optional[str] = None
    closest_by_atom_plane_major_fraction: float = float("nan")
    
    min_adj_dist_any_plane: float = float("nan")
    closest_by_plane_side: Optional[str] = None
    closest_by_plane_plane_id: Optional[int] = None
    closest_by_plane_plane_center_frac: float = float("nan")
    closest_by_plane_plane_species_counts: Dict[str, int] = field(default_factory=dict)
    closest_by_plane_plane_major_species: Optional[str] = None
    closest_by_plane_plane_major_fraction: float = float("nan")
    closest_by_plane_sep_ang: float = float("nan")
    closest_by_plane_sep_frac: float = float("nan")

    # (5) score vector length and angle errors
    uv_len_err_mean: float = float("nan")
    uv_ang_deg_mean: float = float("nan")
    uv_ang_err_mean: float = float("nan")
    u_len_min: float = float("nan")
    v_len_min: float = float("nan")
    u_len_max: float = float("nan")
    v_len_max: float = float("nan")
    uv_len_err_min: float = float("nan")
    uv_len_err_max: float = float("nan")
    uv_ang_deg_min: float = float("nan")
    uv_ang_deg_max: float = float("nan")

    # same-plane mixing info (plane_tol grouping)
    coplane_species_counts: Dict[str, int] = field(default_factory=dict)
    has_coplane_other_species: bool = False
    coplane_other_species_counts: Dict[str, int] = field(default_factory=dict)

    # (6) bonding filter diagnostics
    has_out_of_plane_same_species_bond: bool = False

    # (7) CrystalNN features (aggregated over sites in this plane+species)
    cnn_in_plane_nn_dist: float = float("nan")
    cnn_in_plane_nn_species: Optional[str] = None
    cnn_out_of_plane_nn_dist: float = float("nan")
    cnn_out_of_plane_nn_species: Optional[str] = None

    cnn_cn_mean: float = float("nan")
    cnn_cn_in_plane_mean: float = float("nan")
    cnn_cn_out_of_plane_mean: float = float("nan")

    cnn_in_plane_bonded_species_counts: Dict[str, int] = field(default_factory=dict)
    cnn_out_of_plane_bonded_species_counts: Dict[str, int] = field(default_factory=dict)

    square_species_oxi_state_mean: float = float("nan")
    square_species_oxi_state_std: float = float("nan")

    # (8) CrystalNN bond-angle features
    # In-plane: per site pick the pair of shortest in-plane bonds whose angle is closest to 90°
    cnn_in_plane_bond_angle_deg_mean: float = float("nan")
    cnn_in_plane_bond_angle_deg_std: float = float("nan")
    cnn_in_plane_bond_angle_err90_mean: float = float("nan")

    # Out-of-plane: shortest out-of-plane bond tilt relative to the plane (0=in plane, 90=perpendicular)
    cnn_out_of_plane_tilt_angle_deg_mean: float = float("nan")
    cnn_out_of_plane_tilt_angle_deg_std: float = float("nan")

    # Optional: angle between the two shortest out-of-plane bonds (often ~180 for above/below)
    cnn_out_of_plane_pair_angle_deg_mean: float = float("nan")
    cnn_out_of_plane_pair_angle_deg_std: float = float("nan")

    # Debugging: why passes2 failed (empty list => passes2 True)
    passes2_fail_reasons: List[str] = field(default_factory=list)

    # Optional visualization diagnostics. Kept out of normal detector runs by
    # default because these arrays are useful for figures, not tabular exports.
    visualization_data: Optional[LayerVisualizationData] = None

    


def find_square_net_planes(
    structure,
    axes: Tuple[str, ...] = ("a", "b", "c"),
    plane_tol: float = 0.01,
    species: Optional[Tuple[str, ...]] = None,
    k_nn: int = 9,
    len_tol: float = 0.05,
    ang_tol_deg: float = 7.0,
    min_pass_fraction: float = 0.6,
    score_threshold: float = 0.5,
    return_all: bool = True,   # return failing cases too
    preserve_visualization_data: bool = False,
    adjacent_by: str = "atom",   # NEW: "atom" (current) or "plane"

    # --- passes2 criteria (optional bounds; None = ignore) ---
    nn_intra_min_min: Optional[float] = None,
    nn_intra_min_max: Optional[float] = 4.0,

    tol_ratio_any_min: Optional[float] = None,
    tol_ratio_any_max: Optional[float] = None,

    min_adj_dist_any_atom_min: Optional[float] = 2.0,
    min_adj_dist_any_atom_max: Optional[float] = None,

    min_adj_dist_any_plane_min: Optional[float] = None,
    min_adj_dist_any_plane_max: Optional[float] = None,

    closest_by_plane_sep_ang_min: Optional[float] = None,
    closest_by_plane_sep_ang_max: Optional[float] = None,

    adj_same_species_by: str = "atom",                      # "primary" | "atom" | "plane"

    # --- NEW: passes2 criteria (optional) ---
    forbid_coplane_mixed_species: Optional[bool] = True,
    # If True: passes2 fails when other species are present in the plane group.
    # If False: passes2 fails when plane group is pure (rarely useful; included for symmetry).
    # If None: ignore this criterion.

    isolate_same_species_adjacent: Optional[bool] = True,
    # If True: passes2 fails if the closest atom in the closest adjacent plane (by atom distance)
    #          is the same species AND greater than isolate_same_species_adjacent_dist_min (Å).
    # If None: ignore this criterion.

    isolate_same_species_adjacent_dist_min: Optional[float] = 2.0,
    # Distance cutoff (Å) used only when isolate_same_species_adjacent is True.

    # --- bonding-based filter for passes2 ---
    enforce_no_out_of_plane_same_species_bonds: bool = True,
    bond_in_plane_tol: Optional[float] = None,     # if None -> use plane_tol
    crystalnn_weight_cutoff: float = 0.0,          # ignore nn with weight < cutoff (0 keeps all)
    crystalnn_kwargs: Optional[Dict] = None,       # passed to CrystalNN(**kwargs)

    # --- NEW: always compute CrystalNN ML features even if not enforcing passes2 bond filter ---
    compute_crystalnn_features: bool = True,

    # --- assign oxidation states (for CrystalNN) via Bond Valence Analyzer ---
    guess_oxi_states_for_crystalnn: bool = True,
    bva_kwargs: Optional[Dict] = None,             # passed to BVAnalyzer(**bva_kwargs)
    bva_fallback_to_composition_guess: bool = True,
) -> List[SquarePlaneResult]:

    #####helper functions#######
    
    def _major_species_fraction(counts: Dict[str, int]) -> Tuple[Optional[str], float]:
        if not counts:
            return None, float("nan")
        total = sum(counts.values())
        if total <= 0:
            return None, float("nan")
        major = max(counts.items(), key=lambda kv: kv[1])[0]
        return major, float(counts[major]) / float(total)

    def _within_bounds(x: float, lo: Optional[float], hi: Optional[float]) -> bool:
        """Return True if bounds are not enforced, else require finite x within [lo, hi]."""
        if lo is None and hi is None:
            return True
        if not np.isfinite(x):
            return False
        if lo is not None and x < lo:
            return False
        if hi is not None and x > hi:
            return False
        return True
    
    axis_map = {"a": 0, "b": 1, "c": 2}
    axes = tuple(axes)

    lat = structure.lattice.matrix
    frac_all = np.array([s.frac_coords for s in structure.sites], dtype=float)
    sp_all = np.array([getattr(s.specie, "symbol", str(s.specie)) for s in structure.sites], dtype=object)

    # Precompute cartesian coords for base cell (used often; avoids repeating frac @ lat)
    cart_all = frac_all @ lat

    # --- CrystalNN usage toggles ---
    _use_bond_filter = bool(enforce_no_out_of_plane_same_species_bonds)
    _use_crystalnn = bool(_use_bond_filter or compute_crystalnn_features)

    # --- build a structure for CrystalNN that *has oxidation states* (avoid warnings) ---
    structure_nn = structure  # default: original structure

    if guess_oxi_states_for_crystalnn:
        def _site_has_oxi(site) -> bool:
            try:
                return getattr(site.specie, "oxi_state", None) is not None
            except Exception:
                return False

        already_decorated = all(_site_has_oxi(site) for site in structure.sites)

                # Quick heuristics: BVAnalyzer is most reliable for ionic, ordered structures.
        # Skip for disordered or very complex structures.
        if (not structure.is_ordered) or (len(structure.composition.elements) > 4) or (len(structure) > 30):
            print('Complicated structure -- Skipping BVA')
            structure_nn = structure
        else:
            # BVAnalyzer call as before...
            if not already_decorated:
                try:
                    from pymatgen.analysis.bond_valence import BVAnalyzer
                    bva = BVAnalyzer(**(bva_kwargs or {}))
    
                    # Works only for ordered structures; raises ValueError if it can't determine valences
                    structure_nn = bva.get_oxi_state_decorated_structure(structure)
    
                except Exception:
                    # Optional fallback: element-wise oxidation state guesses from composition.
                    # Less reliable (esp. mixed valence), but can still reduce CrystalNN warnings.
                    if bva_fallback_to_composition_guess:
                        try:
                            guesses = structure.composition.oxi_state_guesses()
                            if guesses:
                                structure_nn = structure.copy()
                                structure_nn.add_oxidation_state_by_element(guesses[0])
                        except Exception:
                            structure_nn = structure

    # --- CrystalNN neighbor finder (cached) ---
    if _use_crystalnn:
        try:
            from pymatgen.analysis.local_env import CrystalNN
        except Exception as e:
            raise ImportError(
                "pymatgen is required for CrystalNN features/filter. "
                "Install pymatgen or set compute_crystalnn_features=False and "
                "enforce_no_out_of_plane_same_species_bonds=False."
            ) from e

        _cnn_kwargs = crystalnn_kwargs or {}
        cnn = CrystalNN(**_cnn_kwargs)

        @lru_cache(maxsize=None)
        def _cnn_nn_info(site_index: int):
            """
            Cached CrystalNN neighbor info for a given site index.
            Returns list of dicts (typical keys: 'site', 'site_index', 'image', 'weight', ...).
            """
            try:
                return cnn.get_nn_info(structure_nn, int(site_index))
            except Exception:
                return []

        def _nn_site_index(nn_dict):
            j = nn_dict.get("site_index", None) if isinstance(nn_dict, dict) else None
            return int(j) if j is not None else None

        def _nn_image_vec(nn_dict):
            im = (0, 0, 0)
            if isinstance(nn_dict, dict):
                im = nn_dict.get("image", im)
            try:
                v = np.array(im, dtype=float)
                if v.shape != (3,):
                    v = np.zeros(3, dtype=float)
            except Exception:
                v = np.zeros(3, dtype=float)
            return v

    #########################################

    if species is None:
        species_list = tuple(sorted(set(sp_all.tolist())))
    else:
        species_list = tuple(species)

    shifts_3d = _tile_shifts_3d(lat)
    results: List[SquarePlaneResult] = []

    for ax in axes:
        aidx = axis_map[ax]
        plane_groups = _group_planes_periodic(frac_all, axis=aidx, tol=plane_tol)
        nplanes = len(plane_groups)

        e1, e2 = _plane_basis_from_lattice(lat, axis=aidx)

        # --- scale factor to convert Δ(fractional along axis) -> Å plane spacing ---
        inv_lat = np.linalg.inv(lat)              # since cart = frac @ lat
        g_axis = inv_lat[:, aidx]                 # a*, b*, or c* depending on axis
        g_norm = float(np.linalg.norm(g_axis))    # units: 1/Å
        frac_to_plane_ang = (1.0 / g_norm) if g_norm > 0 else float("nan")  # Å per fractional unit

        # Precompute plane centers and species counts for all planes along this axis
        plane_centers = []
        plane_species_counts: List[Dict[str, int]] = []
        for g in plane_groups:
            plane_centers.append(float(np.median((frac_all[g, aidx] % 1.0))))
            plane_species_counts.append(_counts(sp_all[g]))

        for plane_id, g in enumerate(plane_groups):
            plane_center = plane_centers[plane_id]

            # adjacent planes in the grouped list (wrap-around)
            prev_id = (plane_id - 1) % nplanes if nplanes >= 2 else None
            next_id = (plane_id + 1) % nplanes if nplanes >= 2 else None

            # Build KD trees for prev and next planes separately (so we can attribute which is closest)
            prev_tree = next_tree = None
            prev_origin = next_origin = None
            prev_atom_ids = next_atom_ids = None

            if prev_id is not None:
                prev_g = plane_groups[prev_id]
                cart_prev = cart_all[prev_g]
                tiled_prev, prev_origin = _tile_points(cart_prev, shifts_3d)
                prev_tree = cKDTree(tiled_prev)
                prev_atom_ids = prev_g  # map origin index -> global atom index

            if next_id is not None:
                next_g = plane_groups[next_id]
                cart_next = cart_all[next_g]
                tiled_next, next_origin = _tile_points(cart_next, shifts_3d)
                next_tree = cKDTree(tiled_next)
                next_atom_ids = next_g

            # co-plane composition (all species in this plane group)
            coplane_counts = plane_species_counts[plane_id]

            for sp in species_list:
                idx = g[sp_all[g] == sp]
                if idx.size == 0:
                    continue

                # Cartesian coords of target-species sites in the plane
                cart = cart_all[idx]

                # ---- (2) intralayer NN distances via in-plane tiling & 2D projection ----
                tiled_cart, origin_inplane = _tile_in_plane(cart, lat, axis=aidx)
                tiled_2d = np.column_stack([tiled_cart @ e1, tiled_cart @ e2])
                base_2d = np.column_stack([cart @ e1, cart @ e2])
                image_offsets_inplane = _in_plane_image_offsets(aidx)
                tiled_image_offsets = np.repeat(image_offsets_inplane, len(cart), axis=0)
                tiled_global_indices = idx[origin_inplane]

                tree = cKDTree(tiled_2d)

                if len(tiled_2d) >= 2:
                    d2, _ = tree.query(base_2d, k=2)  # [self ~0, nearest neighbor]
                    nn_dists = d2[:, 1].astype(float) if d2.ndim == 2 else np.array([d2[1]], dtype=float)
                    nn_intra_min = float(np.min(nn_dists)) if nn_dists.size else float("nan")
                    nn_intra_mean = float(np.mean(nn_dists)) if nn_dists.size else float("nan")
                else:
                    nn_intra_min = float("nan")
                    nn_intra_mean = float("nan")

                # ---- (1) squareness scores (existing logic) ----
                scores = []
                uv_ang_degs = []
                uv_len_errs = []
                uv_ang_errs = []
                u_lengths = []
                v_lengths = []
                local_site_details: List[Dict[str, Any]] = []
                selected_neighbor_edges: List[Dict[str, Any]] = []
                neighbor_length_measurements: List[float] = []
                neighbor_angle_measurements: List[float] = []
                
                for si in range(len(base_2d)):
                    p = base_2d[si]
                    d, nn = tree.query(p, k=k_nn)
                    nn = np.atleast_1d(nn)
                    d = np.atleast_1d(d)
                    valid_nn = (nn >= 0) & (nn < len(tiled_2d)) & np.isfinite(d)
                    nn = nn[valid_nn]
                
                    vecs_all = tiled_2d[nn] - p[None, :]
                    dist_all = np.linalg.norm(vecs_all, axis=1)
                
                    score_mask = dist_all > 1e-8
                    vecs_score = vecs_all[score_mask]
                    nn_score = nn[score_mask]
                    if preserve_visualization_data:
                        s, score_dict, score_details = _square_score_site(
                            vecs_score,
                            len_tol=len_tol,
                            ang_tol_deg=ang_tol_deg,
                            return_details=True,
                        )
                    else:
                        s, score_dict = _square_score_site(vecs_score, len_tol=len_tol, ang_tol_deg=ang_tol_deg)
                        score_details = None
                    scores.append(s)

                    uv_len_err_plane = score_dict['len_err']
                    uv_ang_deg_plane = score_dict['ang_deg']
                    uv_ang_err_plane = score_dict['ang_err']
                    u_len = score_dict['du']
                    v_len = score_dict['dv']

                    uv_ang_degs.append(uv_ang_deg_plane)
                    uv_len_errs.append(uv_len_err_plane)
                    uv_ang_errs.append(uv_ang_err_plane)
                    u_lengths.append(u_len)
                    v_lengths.append(v_len)

                    if preserve_visualization_data:
                        selected_positions = []
                        if score_details is not None:
                            for key in (
                                "u_position",
                                "v_position",
                                "opposite_u_position",
                                "opposite_v_position",
                            ):
                                pos = score_details.get(key)
                                if pos is not None and 0 <= int(pos) < len(nn_score):
                                    selected_positions.append(int(pos))
                        selected_positions_unique = []
                        seen_positions = set()
                        for pos in selected_positions:
                            if pos in seen_positions:
                                continue
                            seen_positions.add(pos)
                            selected_positions_unique.append(pos)
                        selected_positions = selected_positions_unique
                        selected_tiled = [int(nn_score[pos]) for pos in selected_positions]
                        selected_vectors = [vecs_score[pos].astype(float) for pos in selected_positions]
                        selected_distances = [float(np.linalg.norm(vv)) for vv in selected_vectors]

                        for dist_value in (u_len, v_len):
                            if np.isfinite(dist_value):
                                neighbor_length_measurements.append(float(dist_value))
                        if np.isfinite(uv_ang_deg_plane):
                            neighbor_angle_measurements.append(float(uv_ang_deg_plane))

                        site_global = int(idx[si])
                        local_detail = {
                            "site_index": site_global,
                            "site_local_index": int(si),
                            "site_projected": p.astype(float),
                            "neighbor_global_indices": tiled_global_indices[nn_score].astype(int),
                            "neighbor_image_offsets": tiled_image_offsets[nn_score].astype(int),
                            "neighbor_vectors": vecs_score.astype(float),
                            "neighbor_distances": np.linalg.norm(vecs_score, axis=1).astype(float),
                            "score": float(s),
                            "score_info": dict(score_dict),
                            "score_details": dict(score_details or {}),
                            "selected_neighbor_positions": np.array(selected_positions, dtype=int),
                            "selected_neighbor_global_indices": tiled_global_indices[selected_tiled].astype(int) if selected_tiled else np.array([], dtype=int),
                            "selected_neighbor_image_offsets": tiled_image_offsets[selected_tiled].astype(int) if selected_tiled else np.empty((0, 3), dtype=int),
                            "selected_neighbor_vectors": np.array(selected_vectors, dtype=float) if selected_vectors else np.empty((0, 2), dtype=float),
                            "selected_neighbor_distances": np.array(selected_distances, dtype=float),
                        }
                        local_site_details.append(local_detail)

                        for pos, tiled_idx in zip(selected_positions, selected_tiled):
                            selected_neighbor_edges.append({
                                "site_index": site_global,
                                "site_local_index": int(si),
                                "neighbor_index": int(tiled_global_indices[tiled_idx]),
                                "neighbor_image_offset": tiled_image_offsets[tiled_idx].astype(int),
                                "start": p.astype(float),
                                "end": (p + vecs_score[pos]).astype(float),
                                "vector": vecs_score[pos].astype(float),
                                "distance": float(np.linalg.norm(vecs_score[pos])),
                                "role_position": int(pos),
                                "site_score": float(s),
                            })

                scores = np.array(scores, dtype=float)

                u_lengths = np.array(u_lengths, dtype=float)
                v_lengths = np.array(v_lengths, dtype=float)
                u_len_min = np.min(u_lengths)
                v_len_min = np.min(v_lengths)
                u_len_max = np.max(u_lengths)
                v_len_max = np.max(v_lengths)
                
                uv_ang_degs = np.array(uv_ang_degs, dtype=float)
                uv_ang_errs = np.array(uv_ang_errs, dtype=float)
                uv_len_errs = np.array(uv_len_errs, dtype=float)
                uv_ang_degs_mean = float(np.nanmean(uv_ang_degs)) if uv_ang_degs.size else float("nan")
                uv_ang_errs_mean = float(np.nanmean(uv_ang_errs)) if uv_ang_errs.size else float("nan")
                uv_len_errs_mean = float(np.nanmean(uv_len_errs)) if uv_len_errs.size else float("nan")
                uv_len_err_min = np.min(uv_len_errs)
                uv_len_err_max = np.max(uv_len_errs)
                uv_ang_deg_min = np.min(uv_ang_degs)
                uv_ang_deg_max = np.max(uv_ang_degs)
                
                pass_fraction = float(np.mean(scores >= score_threshold)) if scores.size else 0.0
                mean_score = float(np.mean(scores)) if scores.size else 0.0
                median_score = float(np.median(scores)) if scores.size else 0.0
                min_score = float(np.min(scores)) if scores.size else 0.0
                max_score = float(np.max(scores)) if scores.size else 0.0

                passes = bool(pass_fraction >= min_pass_fraction)

                # ============================================================
                # CrystalNN-derived ML features (in-plane vs out-of-plane)
                # ============================================================

                has_out_of_plane_same_sp_bond = False

                cnn_in_plane_nn_dist = float("nan")
                cnn_in_plane_nn_species = None
                cnn_out_of_plane_nn_dist = float("nan")
                cnn_out_of_plane_nn_species = None

                cnn_cn_mean = float("nan")
                cnn_cn_in_plane_mean = float("nan")
                cnn_cn_out_of_plane_mean = float("nan")

                cnn_in_plane_bonded_species_counts: Dict[str, int] = {}
                cnn_out_of_plane_bonded_species_counts: Dict[str, int] = {}

                square_species_oxi_state_mean = float("nan")
                square_species_oxi_state_std = float("nan")

                cnn_in_plane_bond_angle_deg_mean = float("nan")
                cnn_in_plane_bond_angle_deg_std = float("nan")
                cnn_in_plane_bond_angle_err90_mean = float("nan")

                cnn_out_of_plane_tilt_angle_deg_mean = float("nan")
                cnn_out_of_plane_tilt_angle_deg_std = float("nan")

                cnn_out_of_plane_pair_angle_deg_mean = float("nan")
                cnn_out_of_plane_pair_angle_deg_std = float("nan")
                

                if _use_crystalnn:
                    tol_plane_for_bonds = plane_tol if (bond_in_plane_tol is None) else float(bond_in_plane_tol)

                    # Plane normal (cartesian) for out-of-plane tilt angles
                    plane_normal = _unit(np.cross(e1, e2))

                    def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
                        nu = float(np.linalg.norm(u))
                        nv = float(np.linalg.norm(v))
                        if nu < 1e-12 or nv < 1e-12:
                            return float("nan")
                        c = float(np.dot(u, v) / (nu * nv))
                        c = float(np.clip(c, -1.0, 1.0))
                        return float(np.degrees(np.arccos(c)))

                    # Accumulate per-site angle measurements (then aggregate over the plane)
                    in_plane_best_angles = []
                    in_plane_best_angle_errs = []
                    out_tilt_angles = []
                    out_pair_angles = []


                    # Oxidation state stats for square-net species sites (if available)
                    oxi_vals = []
                    for i_site in idx:
                        try:
                            ox = float(getattr(structure_nn.sites[int(i_site)].specie, "oxi_state"))
                            if np.isfinite(ox):
                                oxi_vals.append(ox)
                        except Exception:
                            pass
                    if len(oxi_vals) > 0:
                        square_species_oxi_state_mean = float(np.mean(oxi_vals))
                        square_species_oxi_state_std = float(np.std(oxi_vals))

                    cn_sum = 0.0
                    cn_in_sum = 0.0
                    cn_out_sum = 0.0

                    # Loop each site in the candidate layer and gather neighbor info
                    for i_site in idx:
                        i_site = int(i_site)
                        nn_info = _cnn_nn_info(i_site)

                        cn_i = 0
                        cn_i_in = 0
                        cn_i_out = 0

                        ri = cart_all[i_site]

                        # Collect bond vectors for angle calculations (keep a few shortest)
                        in_vecs = []   # list of (dist, vec)
                        out_vecs = []  # list of (dist, vec)


                        for nn in nn_info:
                            if not isinstance(nn, dict):
                                continue

                            w = float(nn.get("weight", 1.0))
                            if w < float(crystalnn_weight_cutoff):
                                continue

                            j_site = _nn_site_index(nn)
                            if j_site is None:
                                continue
                            j_site = int(j_site)

                            im = _nn_image_vec(nn)

                            # classify in-plane/out-of-plane by fractional displacement along plane axis
                            df_axis = (frac_all[j_site] + im)[aidx] - frac_all[i_site][aidx]
                            in_plane = (abs(float(df_axis)) <= tol_plane_for_bonds)

                            # bond distance (cart) using periodic image shift
                            rj = cart_all[j_site] + (im @ lat)
                            dist = float(np.linalg.norm(rj - ri))

                            spj = str(sp_all[j_site])

                            #vec for bond angle calcs
                            vec = (rj - ri)

                            if in_plane:
                                in_vecs.append((dist, vec))
                            else:
                                out_vecs.append((dist, vec))                            

                            cn_i += 1
                            if in_plane:
                                cn_i_in += 1
                                cnn_in_plane_bonded_species_counts[spj] = cnn_in_plane_bonded_species_counts.get(spj, 0) + 1

                                if (not np.isfinite(cnn_in_plane_nn_dist)) or (dist < cnn_in_plane_nn_dist):
                                    cnn_in_plane_nn_dist = dist
                                    cnn_in_plane_nn_species = spj
                            else:
                                cn_i_out += 1
                                cnn_out_of_plane_bonded_species_counts[spj] = cnn_out_of_plane_bonded_species_counts.get(spj, 0) + 1

                                if (not np.isfinite(cnn_out_of_plane_nn_dist)) or (dist < cnn_out_of_plane_nn_dist):
                                    cnn_out_of_plane_nn_dist = dist
                                    cnn_out_of_plane_nn_species = spj

                                # same-species out-of-plane bond diagnostic (used by passes2 filter)
                                if spj == sp:
                                    has_out_of_plane_same_sp_bond = True

                        # ---- In-plane bond angle: choose pair (among up to 4 shortest) closest to 90° ----
                        if len(in_vecs) >= 2:
                            in_vecs_sorted = sorted(in_vecs, key=lambda x: x[0])[:4]
                            vecs = [v for _, v in in_vecs_sorted]

                            best_ang = float("nan")
                            best_err = float("nan")

                            for ii in range(len(vecs)):
                                for jj in range(ii + 1, len(vecs)):
                                    ang = _angle_deg(vecs[ii], vecs[jj])
                                    if not np.isfinite(ang):
                                        continue
                                    err = abs(ang - 90.0)
                                    if (not np.isfinite(best_err)) or (err < best_err):
                                        best_err = err
                                        best_ang = ang

                            if np.isfinite(best_ang):
                                in_plane_best_angles.append(best_ang)
                                in_plane_best_angle_errs.append(best_err)

                        # ---- Out-of-plane bond tilt: shortest out-of-plane bond angle relative to plane ----
                        # 0° = lies in plane, 90° = perfectly perpendicular to plane
                        if len(out_vecs) >= 1:
                            out_vecs_sorted = sorted(out_vecs, key=lambda x: x[0])
                            v_short = out_vecs_sorted[0][1]
                            nv = float(np.linalg.norm(v_short))
                            if nv > 1e-12:
                                vhat = v_short / nv
                                d = abs(float(np.dot(vhat, plane_normal)))  # ignore above/below sign
                                d = float(np.clip(d, 0.0, 1.0))
                                ang_to_normal = float(np.degrees(np.arccos(d)))      # 0 = along normal
                                ang_to_plane = 90.0 - ang_to_normal                   # 0 = in plane
                                out_tilt_angles.append(float(ang_to_plane))

                        # ---- Optional: out-of-plane pair angle (two shortest out-of-plane bonds) ----
                        if len(out_vecs) >= 2:
                            out_vecs_sorted = sorted(out_vecs, key=lambda x: x[0])
                            v1 = out_vecs_sorted[0][1]
                            v2 = out_vecs_sorted[1][1]
                            ang12 = _angle_deg(v1, v2)
                            if np.isfinite(ang12):
                                out_pair_angles.append(float(ang12))
                        ############################################

                        
                        cn_sum += float(cn_i)
                        cn_in_sum += float(cn_i_in)
                        cn_out_sum += float(cn_i_out)

                    n_idx = float(len(idx)) if len(idx) > 0 else 0.0
                    if n_idx > 0:
                        cnn_cn_mean = cn_sum / n_idx
                        cnn_cn_in_plane_mean = cn_in_sum / n_idx
                        cnn_cn_out_of_plane_mean = cn_out_sum / n_idx

                    # Aggregate angle stats over sites in this plane/species
                    if len(in_plane_best_angles) > 0:
                        cnn_in_plane_bond_angle_deg_mean = float(np.mean(in_plane_best_angles))
                        cnn_in_plane_bond_angle_deg_std = float(np.std(in_plane_best_angles))
                    if len(in_plane_best_angle_errs) > 0:
                        cnn_in_plane_bond_angle_err90_mean = float(np.mean(in_plane_best_angle_errs))

                    if len(out_tilt_angles) > 0:
                        cnn_out_of_plane_tilt_angle_deg_mean = float(np.mean(out_tilt_angles))
                        cnn_out_of_plane_tilt_angle_deg_std = float(np.std(out_tilt_angles))

                    if len(out_pair_angles) > 0:
                        cnn_out_of_plane_pair_angle_deg_mean = float(np.mean(out_pair_angles))
                        cnn_out_of_plane_pair_angle_deg_std = float(np.std(out_pair_angles))


                # ---- (3) adjacent-layer distances (ANY species) + attribution ----
                min_prev = float("nan")
                prev_closest_species = None
                prev_closest_connection = None
                if prev_tree is not None and prev_origin is not None and prev_atom_ids is not None:
                    d_prev, j_prev = prev_tree.query(cart, k=1)
                    if np.size(d_prev):
                        imin = int(np.argmin(d_prev))
                        min_prev = float(d_prev[imin])
                        tile_j = int(j_prev[imin])
                        origin_j = int(prev_origin[tile_j])
                        global_atom = int(prev_atom_ids[origin_j])
                        prev_closest_species = str(sp_all[global_atom])
                        prev_closest_connection = {
                            "side": "prev",
                            "candidate_index": int(idx[imin]),
                            "adjacent_index": int(global_atom),
                            "start_cartesian": cart[imin].astype(float),
                            "end_cartesian": prev_tree.data[tile_j].astype(float),
                            "distance": float(min_prev),
                            "adjacent_species": prev_closest_species,
                        }

                min_next = float("nan")
                next_closest_species = None
                next_closest_connection = None
                if next_tree is not None and next_origin is not None and next_atom_ids is not None:
                    d_next, j_next = next_tree.query(cart, k=1)
                    if np.size(d_next):
                        imin = int(np.argmin(d_next))
                        min_next = float(d_next[imin])
                        tile_j = int(j_next[imin])
                        origin_j = int(next_origin[tile_j])
                        global_atom = int(next_atom_ids[origin_j])
                        next_closest_species = str(sp_all[global_atom])
                        next_closest_connection = {
                            "side": "next",
                            "candidate_index": int(idx[imin]),
                            "adjacent_index": int(global_atom),
                            "start_cartesian": cart[imin].astype(float),
                            "end_cartesian": next_tree.data[tile_j].astype(float),
                            "distance": float(min_next),
                            "adjacent_species": next_closest_species,
                        }

                # ---------- (A) compute plane-center separations (prev/next) ----------
                dfrac_prev_signed = dfrac_next_signed = float("nan")
                dfrac_prev = dfrac_next = float("nan")

                if prev_id is not None:
                    cprev = float(plane_centers[prev_id])
                    dfrac_prev_signed = ((cprev - plane_center + 0.5) % 1.0) - 0.5
                    dfrac_prev = abs(dfrac_prev_signed)

                if next_id is not None:
                    cnext = float(plane_centers[next_id])
                    dfrac_next_signed = ((cnext - plane_center + 0.5) % 1.0) - 0.5
                    dfrac_next = abs(dfrac_next_signed)

                # ---------- (B) choose closest plane by ATOM distance ----------
                closest_by_atom_side = None
                closest_by_atom_plane_id = None
                closest_by_atom_plane_center = float("nan")
                closest_by_atom_plane_species_counts: Dict[str, int] = {}
                closest_by_atom_atom_species = None
                min_adj_dist_any_atom = float("nan")

                if np.isfinite(min_prev) or np.isfinite(min_next):
                    use_prev_atom = (np.isfinite(min_prev) and (not np.isfinite(min_next) or (min_prev <= min_next)))
                    use_next_atom = (np.isfinite(min_next) and (not np.isfinite(min_prev) or (min_next < min_prev)))

                    if use_prev_atom:
                        closest_by_atom_side = "prev"
                        closest_by_atom_plane_id = prev_id
                        closest_by_atom_plane_center = float(plane_centers[prev_id]) if prev_id is not None else float("nan")
                        closest_by_atom_plane_species_counts = plane_species_counts[prev_id] if prev_id is not None else {}
                        closest_by_atom_atom_species = prev_closest_species
                        min_adj_dist_any_atom = float(min_prev)
                    elif use_next_atom:
                        closest_by_atom_side = "next"
                        closest_by_atom_plane_id = next_id
                        closest_by_atom_plane_center = float(plane_centers[next_id]) if next_id is not None else float("nan")
                        closest_by_atom_plane_species_counts = plane_species_counts[next_id] if next_id is not None else {}
                        closest_by_atom_atom_species = next_closest_species
                        min_adj_dist_any_atom = float(min_next)

                closest_by_atom_plane_major_species, closest_by_atom_plane_major_fraction = _major_species_fraction(
                    closest_by_atom_plane_species_counts
                )

                # ---------- (C) choose closest plane by PLANE spacing ----------
                closest_by_plane_side = None
                closest_by_plane_plane_id = None
                closest_by_plane_plane_center = float("nan")
                closest_by_plane_plane_species_counts: Dict[str, int] = {}
                closest_by_plane_atom_species = None

                closest_by_plane_sep_frac = float("nan")
                closest_by_plane_sep_frac_signed = float("nan")
                closest_by_plane_sep_ang = float("nan")
                closest_by_plane_sep_ang_signed = float("nan")

                min_adj_dist_any_plane = float("nan")  # atom-to-atom distance to the plane-chosen plane

                if np.isfinite(dfrac_prev) or np.isfinite(dfrac_next):
                    use_prev_plane = (np.isfinite(dfrac_prev) and (not np.isfinite(dfrac_next) or (dfrac_prev <= dfrac_next)))
                    use_next_plane = (np.isfinite(dfrac_next) and (not np.isfinite(dfrac_prev) or (dfrac_next < dfrac_prev)))

                    if use_prev_plane:
                        closest_by_plane_side = "prev"
                        closest_by_plane_plane_id = prev_id
                        closest_by_plane_plane_center = float(plane_centers[prev_id]) if prev_id is not None else float("nan")
                        closest_by_plane_plane_species_counts = plane_species_counts[prev_id] if prev_id is not None else {}
                        closest_by_plane_atom_species = prev_closest_species

                        closest_by_plane_sep_frac = float(dfrac_prev)
                        closest_by_plane_sep_frac_signed = float(dfrac_prev_signed)

                        if np.isfinite(min_prev):
                            min_adj_dist_any_plane = float(min_prev)

                    elif use_next_plane:
                        closest_by_plane_side = "next"
                        closest_by_plane_plane_id = next_id
                        closest_by_plane_plane_center = float(plane_centers[next_id]) if next_id is not None else float("nan")
                        closest_by_plane_plane_species_counts = plane_species_counts[next_id] if next_id is not None else {}
                        closest_by_plane_atom_species = next_closest_species

                        closest_by_plane_sep_frac = float(dfrac_next)
                        closest_by_plane_sep_frac_signed = float(dfrac_next_signed)

                        if np.isfinite(min_next):
                            min_adj_dist_any_plane = float(min_next)

                # convert plane spacing to Å if available
                if np.isfinite(closest_by_plane_sep_frac) and np.isfinite(frac_to_plane_ang):
                    closest_by_plane_sep_ang = float(closest_by_plane_sep_frac * frac_to_plane_ang)
                    closest_by_plane_sep_ang_signed = float(closest_by_plane_sep_frac_signed * frac_to_plane_ang)

                closest_by_plane_plane_major_species, closest_by_plane_plane_major_fraction = _major_species_fraction(
                    closest_by_plane_plane_species_counts
                )

                # ---------- (D) pick "primary" closest_adj_* fields based on adjacent_by ----------
                mode = str(adjacent_by).lower().strip()
                if mode not in ("atom", "plane"):
                    raise ValueError(f"adjacent_by must be 'atom' or 'plane', got {adjacent_by!r}")

                if mode == "atom":
                    closest_adj_side = closest_by_atom_side
                    closest_adj_plane_id = closest_by_atom_plane_id
                    closest_adj_plane_center = closest_by_atom_plane_center
                    closest_adj_plane_counts = closest_by_atom_plane_species_counts
                    closest_adj_atom_species = closest_by_atom_atom_species
                    min_adj = min_adj_dist_any_atom
                else:
                    closest_adj_side = closest_by_plane_side
                    closest_adj_plane_id = closest_by_plane_plane_id
                    closest_adj_plane_center = closest_by_plane_plane_center
                    closest_adj_plane_counts = closest_by_plane_plane_species_counts
                    closest_adj_atom_species = closest_by_plane_atom_species
                    min_adj = min_adj_dist_any_plane

                # keep your existing "primary" major species/fraction calculation
                closest_adj_plane_major_species, closest_adj_plane_major_fraction = _major_species_fraction(
                    closest_adj_plane_counts
                )
                
                if closest_adj_plane_counts:
                    total = sum(closest_adj_plane_counts.values())
                    if total > 0:
                        closest_adj_plane_major_species = max(
                            closest_adj_plane_counts.items(), key=lambda kv: kv[1]
                        )[0]
                        major_count = closest_adj_plane_counts[closest_adj_plane_major_species]
                        closest_adj_plane_major_fraction = float(major_count) / float(total)

                # ---- (4) tolerance ratio ----
                tol_ratio_any = _safe_ratio(nn_intra_min, min_adj)

                # ---- co-plane other species info ----
                other_counts = dict(coplane_counts)
                other_counts.pop(sp, None)
                has_other = len(other_counts) > 0
                is_pure_plane_for_sp = (not has_other)

                # ---- secondary pass flag + debug reasons (keeps old `passes` untouched) ----
                passes2_fail_reasons: List[str] = []
                
                if not passes:
                    passes2 = False
                    passes2_fail_reasons.append("primary_pass_failed")
                else:
                    passes2 = True
                
                    # ---- numeric tolerance criteria (only enforced if bounds are provided) ----
                    if not _within_bounds(nn_intra_min, nn_intra_min_min, nn_intra_min_max):
                        passes2 = False
                        passes2_fail_reasons.append("nn_intra_min_out_of_bounds")
                
                    if not _within_bounds(tol_ratio_any, tol_ratio_any_min, tol_ratio_any_max):
                        passes2 = False
                        passes2_fail_reasons.append("tol_ratio_any_out_of_bounds")
                
                    if not _within_bounds(min_adj_dist_any_atom, min_adj_dist_any_atom_min, min_adj_dist_any_atom_max):
                        passes2 = False
                        passes2_fail_reasons.append("min_adj_dist_any_atom_out_of_bounds")
                
                    if not _within_bounds(min_adj_dist_any_plane, min_adj_dist_any_plane_min, min_adj_dist_any_plane_max):
                        passes2 = False
                        passes2_fail_reasons.append("min_adj_dist_any_plane_out_of_bounds")
                
                    if not _within_bounds(closest_by_plane_sep_ang, closest_by_plane_sep_ang_min, closest_by_plane_sep_ang_max):
                        passes2 = False
                        passes2_fail_reasons.append("closest_by_plane_sep_ang_out_of_bounds")
                
                    # ---- (1) Co-plane purity criterion ----
                    # If forbid_coplane_mixed_species=True => FAIL when has_other is True
                    if forbid_coplane_mixed_species is not None:
                        if bool(forbid_coplane_mixed_species) and has_other:
                            passes2 = False
                            passes2_fail_reasons.append("coplane_mixed_species")
                
                        # (optional inversion support; rarely used)
                        if (not bool(forbid_coplane_mixed_species)) and (not has_other):
                            passes2 = False
                            passes2_fail_reasons.append("coplane_pure_but_required_mixed")
                
                    # ---- (2) Adjacent-plane isolation criterion (atom-based) ----
                    # Your intent (from earlier): FAIL if closest adjacent atom is SAME species and WITHIN a cutoff.
                    if bool(isolate_same_species_adjacent):
                        if isolate_same_species_adjacent_dist_min is None:
                            # can't evaluate; fail closed (or delete this if you prefer "ignore")
                            passes2 = False
                            passes2_fail_reasons.append("adjacent_isolation_missing_cutoff")
                        else:
                            same_species_too_close = (
                                (closest_by_atom_atom_species is not None) and
                                (str(closest_by_atom_atom_species) == str(sp)) and
                                np.isfinite(min_adj_dist_any_atom) and
                                (float(min_adj_dist_any_atom) <= float(isolate_same_species_adjacent_dist_min))
                            )
                            if same_species_too_close:
                                passes2 = False
                                passes2_fail_reasons.append("adjacent_same_species_too_close")
                
                    # ---- bond-based criterion ----
                    if _use_bond_filter and has_out_of_plane_same_sp_bond:
                        passes2 = False
                        passes2_fail_reasons.append("out_of_plane_same_species_bond")


                # Decide whether to include failing cases
                if (not return_all) and (not passes):
                    continue

                visualization_data = None
                if preserve_visualization_data:
                    adjacent_indices_by_side: Dict[str, np.ndarray] = {}
                    adjacent_centers_by_side: Dict[str, float] = {}
                    if prev_id is not None:
                        adjacent_indices_by_side["prev"] = np.array(plane_groups[prev_id], dtype=int)
                        adjacent_centers_by_side["prev"] = float(plane_centers[prev_id])
                    if next_id is not None:
                        adjacent_indices_by_side["next"] = np.array(plane_groups[next_id], dtype=int)
                        adjacent_centers_by_side["next"] = float(plane_centers[next_id])

                    plane_normal = _unit(np.cross(e1, e2))
                    plane_center_cartesian = (
                        np.mean(cart_all[g], axis=0).astype(float)
                        if len(g) > 0
                        else np.full(3, float("nan"), dtype=float)
                    )

                    visualization_data = LayerVisualizationData(
                        candidate_site_indices=np.array(idx, dtype=int),
                        candidate_plane_indices=np.array(g, dtype=int),
                        adjacent_plane_indices_by_side=adjacent_indices_by_side,
                        adjacent_plane_centers_frac_by_side=adjacent_centers_by_side,
                        plane_cartesian_basis=np.vstack([e1, e2]).astype(float),
                        plane_normal=plane_normal.astype(float),
                        plane_center_cartesian=plane_center_cartesian,
                        projected_coordinates=base_2d.astype(float),
                        tiled_projected_coordinates=tiled_2d.astype(float),
                        tiled_origin_indices=tiled_global_indices.astype(int),
                        tiled_image_offsets=tiled_image_offsets.astype(int),
                        local_site_scores=scores.astype(float),
                        local_site_pass_flags=(scores >= float(score_threshold)).astype(bool),
                        local_site_details=local_site_details,
                        selected_neighbor_edges=selected_neighbor_edges,
                        adjacent_atom_connections={
                            side: conn
                            for side, conn in {
                                "prev": prev_closest_connection,
                                "next": next_closest_connection,
                            }.items()
                            if conn is not None
                        },
                        neighbor_length_measurements=np.array(neighbor_length_measurements, dtype=float),
                        neighbor_angle_measurements=np.array(neighbor_angle_measurements, dtype=float),
                        detector_thresholds={
                            "plane_tol": float(plane_tol),
                            "k_nn": float(k_nn),
                            "len_tol": float(len_tol),
                            "ang_tol_deg": float(ang_tol_deg),
                            "score_threshold": float(score_threshold),
                            "min_pass_fraction": float(min_pass_fraction),
                        },
                    )

                results.append(
                    SquarePlaneResult(
                        axis=ax,
                        plane_id=int(plane_id),
                        plane_center_frac=float(plane_center),
                        species=sp,
                        n_sites=int(idx.size),

                        passes=passes,
                        passes2=passes2,
                        passes2_fail_reasons=passes2_fail_reasons,

                        pass_fraction=pass_fraction,
                        mean_score=mean_score,
                        median_score=median_score,
                        min_score=min_score,
                        max_score=max_score,

                        nn_intra_min=nn_intra_min,
                        nn_intra_mean=nn_intra_mean,

                        min_adj_dist_any_atom=min_adj_dist_any_atom,
                        closest_by_atom_side=closest_by_atom_side,
                        closest_by_atom_plane_id=closest_by_atom_plane_id,
                        closest_by_atom_plane_center_frac=closest_by_atom_plane_center,
                        closest_by_atom_plane_species_counts=closest_by_atom_plane_species_counts,
                        closest_by_atom_plane_major_species=closest_by_atom_plane_major_species,
                        closest_by_atom_plane_major_fraction=closest_by_atom_plane_major_fraction,

                        min_adj_dist_any_plane=min_adj_dist_any_plane,
                        closest_by_plane_side=closest_by_plane_side,
                        closest_by_plane_plane_id=closest_by_plane_plane_id,
                        closest_by_plane_plane_center_frac=closest_by_plane_plane_center,
                        closest_by_plane_plane_species_counts=closest_by_plane_plane_species_counts,
                        closest_by_plane_plane_major_species=closest_by_plane_plane_major_species,
                        closest_by_plane_plane_major_fraction=closest_by_plane_plane_major_fraction,
                        closest_by_plane_sep_ang=closest_by_plane_sep_ang,
                        closest_by_plane_sep_frac=closest_by_plane_sep_frac,

                        has_out_of_plane_same_species_bond=has_out_of_plane_same_sp_bond,

                        tol_ratio_any=tol_ratio_any,

                        coplane_species_counts=coplane_counts,
                        has_coplane_other_species=has_other,
                        coplane_other_species_counts=other_counts,

                        uv_len_err_mean=uv_len_errs_mean,
                        uv_ang_deg_mean=uv_ang_degs_mean,
                        uv_ang_err_mean=uv_ang_errs_mean,

                        u_len_min=u_len_min,
                        v_len_min=v_len_min,
                        u_len_max=u_len_max,
                        v_len_max=v_len_max,
                        uv_len_err_min=uv_len_err_min,
                        uv_len_err_max=uv_len_err_max,
                        uv_ang_deg_min=uv_ang_deg_min,
                        uv_ang_deg_max=uv_ang_deg_max,

                        # --- NEW: CrystalNN features ---
                        cnn_in_plane_nn_dist=cnn_in_plane_nn_dist,
                        cnn_in_plane_nn_species=cnn_in_plane_nn_species,
                        cnn_out_of_plane_nn_dist=cnn_out_of_plane_nn_dist,
                        cnn_out_of_plane_nn_species=cnn_out_of_plane_nn_species,

                        cnn_cn_mean=cnn_cn_mean,
                        cnn_cn_in_plane_mean=cnn_cn_in_plane_mean,
                        cnn_cn_out_of_plane_mean=cnn_cn_out_of_plane_mean,

                        cnn_in_plane_bonded_species_counts=cnn_in_plane_bonded_species_counts,
                        cnn_out_of_plane_bonded_species_counts=cnn_out_of_plane_bonded_species_counts,

                        square_species_oxi_state_mean=square_species_oxi_state_mean,
                        square_species_oxi_state_std=square_species_oxi_state_std,

                        cnn_in_plane_bond_angle_deg_mean=cnn_in_plane_bond_angle_deg_mean,
                        cnn_in_plane_bond_angle_deg_std=cnn_in_plane_bond_angle_deg_std,
                        cnn_in_plane_bond_angle_err90_mean=cnn_in_plane_bond_angle_err90_mean,

                        cnn_out_of_plane_tilt_angle_deg_mean=cnn_out_of_plane_tilt_angle_deg_mean,
                        cnn_out_of_plane_tilt_angle_deg_std=cnn_out_of_plane_tilt_angle_deg_std,

                        cnn_out_of_plane_pair_angle_deg_mean=cnn_out_of_plane_pair_angle_deg_mean,
                        cnn_out_of_plane_pair_angle_deg_std=cnn_out_of_plane_pair_angle_deg_std,

                        visualization_data=visualization_data,

                    )
                )

    # Sort best-first, but keep failing results included (passes first)
    results.sort(
        key=lambda r: (r.passes, r.mean_score, r.pass_fraction, r.n_sites),
        reverse=True
    )
    return results
