"""Static Matplotlib visualizations for square-net detector results.

The functions in this module explain detector decisions rather than attempting
full atomistic rendering. They prefer the optional ``LayerVisualizationData``
stored by ``find_square_net_planes(..., preserve_visualization_data=True)``.
When older or lightweight results lack those diagnostics, reduced plots are
produced where possible and a warning explains the fallback.

Coordinate conventions
----------------------
* Crystal sites from pymatgen structures are read in fractional coordinates and
  converted to Cartesian coordinates using ``cart = frac @ lattice.matrix``.
* Projected layer coordinates use the detector's in-plane basis when available.
  The plotted origin is recentered for readability, but axis units remain
  Angstrom.
* Periodic image offsets are kept as integer fractional-cell offsets and are
  shown with lower opacity or unfilled markers.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


__all__ = [
    "VisualizationError",
    "MissingVisualizationDiagnostics",
    "build_species_style_map",
    "save_figure",
    "plot_structure_overview",
    "plot_candidate_plane_3d",
    "plot_candidate_plane_3d_interactive",
    "plot_projected_layer",
    "select_representative_site",
    "plot_site_geometry",
    "plot_score_components",
    "plot_neighbor_length_distribution",
    "plot_neighbor_angle_distribution",
    "plot_adjacent_plane_environment",
    "plot_coplanar_composition",
    "plot_detection_summary",
    "plot_material_layer_summary",
    "plot_pass_fail_counts",
    "plot_score_distribution",
    "plot_candidates_per_material",
    "plot_score_vs_environment",
    "plot_missingness",
]


class VisualizationError(ValueError):
    """Raised when a visualization cannot be produced from the supplied data."""


class MissingVisualizationDiagnostics(VisualizationError):
    """Raised when a plot requires detector diagnostics that are not present."""


_AXIS_TO_INDEX = {"a": 0, "b": 1, "c": 2}
_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8")


def _axis_index(axis: str) -> int:
    try:
        return _AXIS_TO_INDEX[str(axis).lower()]
    except KeyError as exc:
        raise VisualizationError("axis must be one of 'a', 'b', or 'c'") from exc


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _plane_basis_from_lattice_like_detector(lattice_matrix: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the detector's axis-parallel in-plane basis for fallback plots."""
    aidx = _axis_index(axis)
    other = [0, 1, 2]
    other.remove(aidx)
    t1 = np.asarray(lattice_matrix[other[0]], dtype=float)
    t2 = np.asarray(lattice_matrix[other[1]], dtype=float)
    e1 = _unit(t1)
    t2p = t2 - float(np.dot(t2, e1)) * e1
    e2 = _unit(t2p)
    return e1, e2


def _structure_arrays(structure: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fractional coordinates, Cartesian coordinates, and species labels."""
    if structure is None or not hasattr(structure, "sites") or not hasattr(structure, "lattice"):
        raise VisualizationError("structure must be a pymatgen Structure-like object")
    frac = np.array([site.frac_coords for site in structure.sites], dtype=float)
    lat = np.asarray(structure.lattice.matrix, dtype=float)
    cart = frac @ lat
    species = np.array([getattr(site.specie, "symbol", str(site.specie)) for site in structure.sites], dtype=object)
    return frac, cart, species


def _formula(structure: Any) -> str:
    comp = getattr(structure, "composition", None)
    return str(getattr(comp, "reduced_formula", "")) if comp is not None else ""


def _viz(result: Any) -> Any:
    return getattr(result, "visualization_data", None)


def _thresholds(result: Any) -> Dict[str, float]:
    viz = _viz(result)
    data = getattr(viz, "detector_thresholds", None)
    return dict(data or {})


def _result_passes(result: Any) -> bool:
    if hasattr(result, "passes2"):
        return bool(getattr(result, "passes2"))
    return bool(getattr(result, "passes", False))


def _failure_reasons(result: Any) -> List[str]:
    reasons = getattr(result, "passes2_fail_reasons", None)
    if reasons is None:
        return []
    if isinstance(reasons, str):
        return [x for x in reasons.replace(",", "|").split("|") if x]
    return [str(x) for x in reasons]


def _validate_indices(structure: Any, indices: Optional[Iterable[int]], label: str) -> np.ndarray:
    if indices is None:
        return np.array([], dtype=int)
    arr = np.array(list(indices), dtype=int)
    n = len(structure.sites)
    bad = arr[(arr < 0) | (arr >= n)]
    if bad.size:
        raise VisualizationError(f"{label} contains atom indices outside the structure: {bad.tolist()}")
    return arr


def _circular_fraction_delta(values: np.ndarray, center: float) -> np.ndarray:
    return ((values - float(center) + 0.5) % 1.0) - 0.5


def _candidate_indices_from_result(
    structure: Any,
    result: Any,
    *,
    plane_tol: Optional[float] = None,
    warn: bool = True,
) -> np.ndarray:
    viz = _viz(result)
    if viz is not None and getattr(viz, "candidate_site_indices", None) is not None:
        return _validate_indices(structure, getattr(viz, "candidate_site_indices"), "candidate_site_indices")
    if hasattr(result, "candidate_site_indices"):
        return _validate_indices(structure, getattr(result, "candidate_site_indices"), "candidate_site_indices")

    axis = getattr(result, "axis", None)
    center = getattr(result, "plane_center_frac", None)
    species_name = getattr(result, "species", None)
    if axis is None or center is None or species_name is None:
        raise MissingVisualizationDiagnostics(
            "candidate atom indices are unavailable; rerun detection with preserve_visualization_data=True"
        )
    thresholds = _thresholds(result)
    tol = float(plane_tol if plane_tol is not None else thresholds.get("plane_tol", 0.01))
    frac, _, species = _structure_arrays(structure)
    aidx = _axis_index(str(axis))
    mask = (np.abs(_circular_fraction_delta(frac[:, aidx] % 1.0, float(center))) <= tol) & (
        species.astype(str) == str(species_name)
    )
    indices = np.nonzero(mask)[0].astype(int)
    if warn:
        warnings.warn(
            "Result lacks exact visualization diagnostics; reconstructed candidate indices from "
            "axis/species/plane_center for a reduced plot. Rerun detection with "
            "preserve_visualization_data=True for exact neighbor diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    if indices.size == 0:
        raise VisualizationError("candidate plane is empty for the supplied structure/result")
    return indices


def build_species_style_map(species: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Return stable colors and markers for a species set.

    Parameters
    ----------
    species
        Iterable of species symbols. The mapping is sorted by symbol so the same
        set receives deterministic assignments across figures.

    Returns
    -------
    dict
        ``{symbol: {"color": rgba, "marker": marker}}`` suitable for plotting.
    """
    names = sorted({str(s) for s in species})
    cmap = plt.get_cmap("tab20")
    styles: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(names):
        styles[name] = {
            "color": cmap((i % 20) / 19.0),
            "marker": _MARKERS[i % len(_MARKERS)],
        }
    return styles


def _style_map(species: Iterable[str], override: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    styles = build_species_style_map(species)
    if override:
        for key, value in override.items():
            styles.setdefault(str(key), {}).update(dict(value))
    return styles


def _make_ax(
    ax: Optional[Axes],
    *,
    projection: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
) -> Tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    if projection == "3d":
        return fig, fig.add_subplot(111, projection="3d")

    return fig, fig.add_subplot(111)


def _unit_cell_edges(lattice_matrix: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    corners_frac = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    corners = corners_frac @ np.asarray(lattice_matrix, dtype=float)
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return corners, edges


def _draw_unit_cell(ax: Axes, lattice_matrix: np.ndarray, *, color: str = "0.35", alpha: float = 0.65) -> None:
    corners, edges = _unit_cell_edges(lattice_matrix)
    for i, j in edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot(xs, ys, zs, color=color, linewidth=0.8, alpha=alpha)


def _replication_range(n: int) -> List[int]:
    n = max(int(n), 1)
    lo = -(n // 2)
    return list(range(lo, lo + n))


def _replication_offsets(lattice_matrix: np.ndarray, replicate: Sequence[int]) -> List[np.ndarray]:
    if len(replicate) != 3:
        raise VisualizationError("replicate must be a length-3 sequence")
    ranges = [_replication_range(int(v)) for v in replicate]
    lat = np.asarray(lattice_matrix, dtype=float)
    offsets = []
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                offsets.append(i * lat[0] + j * lat[1] + k * lat[2])
    return offsets


def _set_3d_equal(ax: Axes, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _apply_axis_view(ax: Axes, view: str, axis: Optional[str] = None) -> None:
    view = str(view).lower()
    chosen = str(axis).lower() if view == "auto" and axis is not None else view
    if chosen == "a":
        ax.view_init(elev=0, azim=0)
    elif chosen == "b":
        ax.view_init(elev=0, azim=90)
    elif chosen == "c":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=24, azim=35)


def _scatter_species_3d(
    ax: Axes,
    xyz: np.ndarray,
    species: np.ndarray,
    styles: Mapping[str, Mapping[str, Any]],
    *,
    indices: np.ndarray,
    candidate_set: set[int],
    adjacent_set: set[int],
    show_labels: bool = False,
) -> None:
    for sp in sorted({str(x) for x in species.tolist()}):
        mask = species.astype(str) == sp
        pts = xyz[mask]
        ids = indices[mask]
        if pts.size == 0:
            continue
        style = styles.get(sp, {})

        other = np.array([(int(i) not in candidate_set and int(i) not in adjacent_set) for i in ids], dtype=bool)
        if np.any(other):
            ax.scatter(
                pts[other, 0],
                pts[other, 1],
                pts[other, 2],
                s=28,
                c=[style.get("color", "0.5")],
                marker=style.get("marker", "o"),
                alpha=0.25,
                linewidths=0.3,
                edgecolors="0.5",
                label=f"{sp} other",
            )

        adj = np.array([(int(i) in adjacent_set and int(i) not in candidate_set) for i in ids], dtype=bool)
        if np.any(adj):
            ax.scatter(
                pts[adj, 0],
                pts[adj, 1],
                pts[adj, 2],
                s=58,
                c=[style.get("color", "0.5")],
                marker=style.get("marker", "o"),
                alpha=0.72,
                linewidths=1.0,
                edgecolors="0.2",
                label=f"{sp} adjacent",
            )

        cand = np.array([int(i) in candidate_set for i in ids], dtype=bool)
        if np.any(cand):
            ax.scatter(
                pts[cand, 0],
                pts[cand, 1],
                pts[cand, 2],
                s=95,
                c=[style.get("color", "0.5")],
                marker=style.get("marker", "o"),
                alpha=0.95,
                linewidths=1.8,
                edgecolors="black",
                label=f"{sp} candidate",
            )
            if show_labels:
                for idx, point in zip(ids[cand], pts[cand]):
                    ax.text(point[0], point[1], point[2], str(int(idx)), fontsize=8)


def _plot_plane_surface(ax: Axes, center: np.ndarray, e1: np.ndarray, e2: np.ndarray, scale: float, *, color: str = "0.7") -> None:
    if not np.all(np.isfinite(center)):
        return
    corners = np.array(
        [
            center - scale * e1 - scale * e2,
            center + scale * e1 - scale * e2,
            center + scale * e1 + scale * e2,
            center - scale * e1 + scale * e2,
        ]
    )
    poly = Poly3DCollection([corners], facecolors=color, edgecolors="0.4", linewidths=0.5, alpha=0.18)
    ax.add_collection3d(poly)


def plot_structure_overview(
    structure: Any,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    adjacent_indices: Optional[Iterable[int]] = None,
    axis: Optional[str] = None,
    plane_center: Optional[float] = None,
    replicate: Sequence[int] = (1, 1, 1),
    view: str = "auto",
    show_unit_cell: bool = True,
    show_labels: bool = False,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Show the full structure with candidate and adjacent atoms emphasized.

    Parameters
    ----------
    structure
        Pymatgen Structure-like object.
    candidate_indices, adjacent_indices
        Atom indices to emphasize. Invalid indices raise ``VisualizationError``.
    axis, plane_center
        Optional crystallographic axis and fractional plane center. When both
        are supplied a translucent plane and normal arrow are drawn.
    replicate
        Number of cells shown along ``a``, ``b``, and ``c``. ``(1,1,1)`` shows
        the central unit cell; ``(3,3,1)`` includes neighboring images.
    view
        ``"auto"``, ``"a"``, ``"b"``, ``"c"``, or ``"oblique"``.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and 3D axes. The function does not call ``plt.show``.
    """
    frac, cart, species = _structure_arrays(structure)
    lat = np.asarray(structure.lattice.matrix, dtype=float)
    cand = _validate_indices(structure, candidate_indices, "candidate_indices")
    adj = _validate_indices(structure, adjacent_indices, "adjacent_indices")

    fig, ax = _make_ax(ax, projection="3d", figsize=(6.5, 5.2))
    styles = _style_map(species, species_style_map)

    all_points: List[np.ndarray] = []
    base_indices = np.arange(len(structure.sites), dtype=int)
    candidate_set = set(cand.tolist())
    adjacent_set = set(adj.tolist())
    for shift in _replication_offsets(lat, replicate):
        pts = cart + shift
        all_points.append(pts)
        _scatter_species_3d(
            ax,
            pts,
            species,
            styles,
            indices=base_indices,
            candidate_set=candidate_set,
            adjacent_set=adjacent_set,
            show_labels=show_labels and np.allclose(shift, 0.0),
        )

    if show_unit_cell:
        _draw_unit_cell(ax, lat)

    if axis is not None:
        aidx = _axis_index(axis)
        e1, e2 = _plane_basis_from_lattice_like_detector(lat, axis)
        normal = _unit(np.cross(e1, e2))
        center_frac = np.full(3, 0.5, dtype=float)
        if plane_center is not None:
            center_frac[aidx] = float(plane_center) % 1.0
        center_cart = center_frac @ lat
        scale = max(float(np.linalg.norm(lat[i])) for i in range(3)) * 0.45
        _plot_plane_surface(ax, center_cart, e1, e2, scale)
        ax.quiver(
            center_cart[0],
            center_cart[1],
            center_cart[2],
            normal[0],
            normal[1],
            normal[2],
            length=scale * 0.55,
            color="black",
            linewidth=1.4,
            arrow_length_ratio=0.18,
        )
        ax.text(*(center_cart + normal * scale * 0.62), f"{axis}-normal", fontsize=8)

    points = np.vstack(all_points) if all_points else cart
    _set_3d_equal(ax, points)
    _apply_axis_view(ax, view, axis=axis)
    ax.set_xlabel("Cartesian x (A)")
    ax.set_ylabel("Cartesian y (A)")
    ax.set_zlabel("Cartesian z (A)")
    ax.set_title(f"{_formula(structure)} structure overview".strip())
    ax.legend(loc="best", fontsize=10)
    return fig, ax


def _projected_lattice_vectors(lattice_matrix: np.ndarray, axis: str, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aidx = _axis_index(axis)
    other = [0, 1, 2]
    other.remove(aidx)
    v1 = np.asarray(lattice_matrix[other[0]], dtype=float)
    v2 = np.asarray(lattice_matrix[other[1]], dtype=float)
    e1, e2 = np.asarray(basis[0], dtype=float), np.asarray(basis[1], dtype=float)
    return np.array([np.dot(v1, e1), np.dot(v1, e2)]), np.array([np.dot(v2, e1), np.dot(v2, e2)])


def _tile_projected_points(
    projected: np.ndarray,
    lattice_matrix: np.ndarray,
    axis: str,
    basis: np.ndarray,
    tile: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile projected points in the detector plane basis.

    This helper is intentionally small: it applies periodic image offsets to
    already projected coordinates and does not infer neighbors or bonds.
    """
    if len(tile) != 2:
        raise VisualizationError("tile must be a length-2 sequence")
    projected = np.asarray(projected, dtype=float)
    du, dv = _projected_lattice_vectors(lattice_matrix, axis, basis)
    aidx = _axis_index(axis)
    other = [0, 1, 2]
    other.remove(aidx)
    offsets_2d = []
    offsets_frac = []
    for i in _replication_range(int(tile[0])):
        for j in _replication_range(int(tile[1])):
            offsets_2d.append(i * du + j * dv)
            off = np.zeros(3, dtype=int)
            off[other[0]] = i
            off[other[1]] = j
            offsets_frac.append(off)
    offsets_2d = np.asarray(offsets_2d, dtype=float)
    offsets_frac = np.asarray(offsets_frac, dtype=int)
    n = len(projected)
    tiled = (projected[None, :, :] + offsets_2d[:, None, :]).reshape(-1, 2)
    origin_local = np.tile(np.arange(n, dtype=int), len(offsets_2d))
    tiled_offsets = np.repeat(offsets_frac, n, axis=0)
    return tiled, origin_local, tiled_offsets


def _projected_layer_data(
    structure: Any,
    result: Any,
    *,
    tile: Sequence[int] = (3, 3),
    plane_tol: Optional[float] = None,
) -> Dict[str, Any]:
    frac, cart, species = _structure_arrays(structure)
    lat = np.asarray(structure.lattice.matrix, dtype=float)
    axis = str(getattr(result, "axis", ""))
    _axis_index(axis)
    viz = _viz(result)

    if viz is not None and getattr(viz, "projected_coordinates", None) is not None:
        candidate_indices = _candidate_indices_from_result(structure, result, warn=False)
        projected = np.asarray(viz.projected_coordinates, dtype=float)
        if len(projected) != len(candidate_indices):
            raise VisualizationError("projected coordinate count does not match candidate_site_indices")
        basis = np.asarray(viz.plane_cartesian_basis, dtype=float)
        normal = np.asarray(viz.plane_normal, dtype=float)
        if tuple(tile) == (3, 3):
            tiled = np.asarray(viz.tiled_projected_coordinates, dtype=float)
            tiled_origin_indices = np.asarray(viz.tiled_origin_indices, dtype=int)
            tiled_offsets = np.asarray(viz.tiled_image_offsets, dtype=int)
        else:
            tiled, origin_local, tiled_offsets = _tile_projected_points(projected, lat, axis, basis, tile)
            tiled_origin_indices = candidate_indices[origin_local]
        exact = True
    else:
        warnings.warn(
            "Projected detector diagnostics are missing; using result axis/species/plane center for a reduced plot.",
            RuntimeWarning,
            stacklevel=2,
        )
        candidate_indices = _candidate_indices_from_result(structure, result, plane_tol=plane_tol, warn=False)
        e1, e2 = _plane_basis_from_lattice_like_detector(lat, axis)
        basis = np.vstack([e1, e2])
        normal = _unit(np.cross(e1, e2))
        projected = np.column_stack([cart[candidate_indices] @ e1, cart[candidate_indices] @ e2])
        tiled, origin_local, tiled_offsets = _tile_projected_points(projected, lat, axis, basis, tile)
        tiled_origin_indices = candidate_indices[origin_local]
        exact = False

    return {
        "frac": frac,
        "cart": cart,
        "species": species,
        "axis": axis,
        "candidate_indices": candidate_indices,
        "projected": projected,
        "basis": basis,
        "normal": normal,
        "tiled_projected": tiled,
        "tiled_origin_indices": tiled_origin_indices,
        "tiled_offsets": tiled_offsets,
        "exact": exact,
    }


def _draw_projected_cell(ax: Axes, lattice_matrix: np.ndarray, axis: str, basis: np.ndarray, origin_shift: np.ndarray) -> None:
    du, dv = _projected_lattice_vectors(lattice_matrix, axis, basis)
    verts = np.array([[0, 0], du, du + dv, dv], dtype=float) - origin_shift
    poly = Polygon(verts, closed=True, fill=False, edgecolor="0.35", linewidth=0.9, linestyle=":")
    ax.add_patch(poly)


def _edge_key(edge: Mapping[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    a = tuple(np.round(np.asarray(edge["start"], dtype=float), 6).tolist())
    b = tuple(np.round(np.asarray(edge["end"], dtype=float), 6).tolist())
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _dedupe_edges(edges: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    seen = set()
    out: List[Mapping[str, Any]] = []
    for edge in edges:
        try:
            key = _edge_key(edge)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def plot_projected_layer(
    structure: Any,
    result: Any,
    *,
    tile: Sequence[int] = (3, 3),
    show_neighbors: bool = True,
    annotate_distances: bool = False,
    annotate_indices: bool = False,
    show_ideal_square: bool = False,
    equal_aspect: bool = True,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot the candidate layer in the detector's 2D projected coordinates.

    Expected result fields
    ----------------------
    Best results include ``result.visualization_data`` with projected
    coordinates, tiled periodic images, and selected neighbor edges captured by
    the detector. Without those diagnostics, this function plots only the
    projected candidate atoms and warns.

    Notes on periodic boundaries
    ----------------------------
    Central-cell atoms are filled markers; periodic images are unfilled and
    lower opacity. Neighbor edges use the detector-provided image offsets when
    available, so wraparound connections are drawn in the projected image where
    the detector evaluated them.
    """
    data = _projected_layer_data(structure, result, tile=tile)
    fig, ax = _make_ax(ax, figsize=(6.2, 5.2))
    species = data["species"]
    candidate_indices = data["candidate_indices"]
    candidate_species = str(getattr(result, "species", species[candidate_indices[0]] if len(candidate_indices) else "candidate"))
    styles = _style_map(species, species_style_map)
    style = styles.get(candidate_species, {"color": "0.4", "marker": "o"})

    projected = data["projected"]
    origin_shift = projected.mean(axis=0) if len(projected) else np.zeros(2)
    tiled = data["tiled_projected"] - origin_shift
    offsets = data["tiled_offsets"]
    central_mask = np.all(offsets == 0, axis=1)

    if np.any(~central_mask):
        ax.scatter(
            tiled[~central_mask, 0],
            tiled[~central_mask, 1],
            s=42,
            facecolors="none",
            edgecolors=[style.get("color", "0.5")],
            marker=style.get("marker", "o"),
            alpha=0.35,
            linewidths=0.9,
            label="periodic images",
        )

    central = projected - origin_shift
    ax.scatter(
        central[:, 0],
        central[:, 1],
        s=82,
        c=[style.get("color", "0.5")],
        marker=style.get("marker", "o"),
        edgecolors="black",
        linewidths=1.4,
        label=f"{candidate_species} candidate sites",
        zorder=4,
    )

    if annotate_indices:
        for idx, xy in zip(candidate_indices, central):
            ax.annotate(str(int(idx)), xy, xytext=(4, 4), textcoords="offset points", fontsize=8)

    viz = _viz(result)
    if show_neighbors:
        edges = _dedupe_edges(getattr(viz, "selected_neighbor_edges", []) if viz is not None else [])
        if not edges:
            warnings.warn(
                "Selected detector neighbor edges are unavailable; projected atoms are shown without edge diagnostics.",
                RuntimeWarning,
                stacklevel=2,
            )
        for edge in edges:
            start = np.asarray(edge["start"], dtype=float) - origin_shift
            end = np.asarray(edge["end"], dtype=float) - origin_shift
            ax.plot([start[0], end[0]], [start[1], end[1]], color="0.1", linewidth=1.1, alpha=0.82)
            if annotate_distances:
                mid = 0.5 * (start + end)
                ax.annotate(f"{float(edge.get('distance', np.nan)):.2f} A", mid, fontsize=7, ha="center", va="bottom")

    if show_ideal_square:
        try:
            site_idx = select_representative_site(result, strategy="median")
            detail = _site_detail(result, site_idx)
            vecs = np.asarray(detail.get("selected_neighbor_vectors", []), dtype=float)
            if len(vecs) >= 2:
                u = vecs[0]
                v = vecs[1]
                p = np.asarray(detail["site_projected"], dtype=float) - origin_shift
                guide = np.array([p, p + u, p + u + v, p + v, p])
                ax.plot(guide[:, 0], guide[:, 1], linestyle="--", color="0.25", linewidth=1.0, label="selected square guide")
        except VisualizationError:
            if np.isfinite(getattr(result, "nn_intra_mean", float("nan"))) and len(central):
                side = float(getattr(result, "nn_intra_mean"))
                p = central[0]
                square = np.array([p, p + [side, 0], p + [side, side], p + [0, side], p])
                ax.plot(square[:, 0], square[:, 1], linestyle="--", color="0.25", linewidth=1.0, label="reference square")

    _draw_projected_cell(ax, np.asarray(structure.lattice.matrix, dtype=float), data["axis"], data["basis"], origin_shift)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected coordinate u (A)")
    ax.set_ylabel("Projected coordinate v (A)")
    status = "PASS" if _result_passes(result) else "FAIL"
    formula = _formula(structure)
    title_parts = [p for p in [formula, f"{candidate_species} {data['axis']}-plane", status] if p]
    ax.set_title(
        " | ".join(title_parts)
        + f"\nmean score={float(getattr(result, 'mean_score', np.nan)):.3g}, pass fraction={float(getattr(result, 'pass_fraction', np.nan)):.3g}",
        pad=25
    )
    ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=10,
    borderaxespad=0,
    )

    fig.subplots_adjust(right=0.76)
    return fig, ax


def _plane_context(structure: Any, result: Any) -> Dict[str, Any]:
    frac, cart, species = _structure_arrays(structure)
    lat = np.asarray(structure.lattice.matrix, dtype=float)
    axis = str(getattr(result, "axis", ""))
    viz = _viz(result)
    if viz is not None and getattr(viz, "plane_cartesian_basis", None) is not None:
        basis = np.asarray(viz.plane_cartesian_basis, dtype=float)
        normal = np.asarray(viz.plane_normal, dtype=float)
        center = np.asarray(viz.plane_center_cartesian, dtype=float)
        candidate = _candidate_indices_from_result(structure, result, warn=False)
        plane_indices = _validate_indices(structure, getattr(viz, "candidate_plane_indices", candidate), "candidate_plane_indices")
        adjacent = {
            side: _validate_indices(structure, ids, f"{side} adjacent indices")
            for side, ids in getattr(viz, "adjacent_plane_indices_by_side", {}).items()
        }
    else:
        warnings.warn(
            "Plane diagnostics are missing; reconstructing plane context from result axis/species/center.",
            RuntimeWarning,
            stacklevel=2,
        )
        candidate = _candidate_indices_from_result(structure, result, warn=False)
        plane_indices = candidate
        e1, e2 = _plane_basis_from_lattice_like_detector(lat, axis)
        basis = np.vstack([e1, e2])
        normal = _unit(np.cross(e1, e2))
        center = np.mean(cart[candidate], axis=0) if len(candidate) else np.full(3, np.nan)
        adjacent = {}
    return {
        "frac": frac,
        "cart": cart,
        "species": species,
        "axis": axis,
        "basis": basis,
        "normal": normal,
        "center": center,
        "candidate_indices": candidate,
        "plane_indices": plane_indices,
        "adjacent": adjacent,
    }


def plot_candidate_plane_3d(
    structure: Any,
    result: Any,
    *,
    plane_window_angstrom: Optional[float] = None,
    include_adjacent_planes: bool = True,
    include_periodic_images: bool = True,
    periodic_image_range: int = 1,
    periodic_images_candidate_only: bool = False,
    show_in_plane_neighbor_lines: bool = True,
    neighbor_cutoff_scale: float = 1.15,
    annotate_species: bool = False,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
    figsize: Tuple[float, float] = (8.2, 6.2),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot a candidate square-net plane in a fixed oblique 3D view.

    This static Matplotlib counterpart to
    :func:`plot_candidate_plane_3d_interactive` displays the candidate plane,
    optional neighboring planes, in-plane periodic images, and candidate-site
    nearest-neighbor connections. Periodic images are generated only along the
    two lattice vectors lying in the candidate plane, so layers are not copied
    along the plane-normal direction.

    Parameters
    ----------
    structure
        Pymatgen Structure-like object.
    result
        Candidate-layer result returned by ``find_square_net_planes``.
    plane_window_angstrom
        Include central-cell atoms whose perpendicular distance from the
        candidate plane is no greater than this value. ``None`` includes the
        full central-cell structure.
    include_adjacent_planes
        Highlight detector-recorded adjacent planes in the central cell.
    include_periodic_images
        Repeat candidate-plane atoms using in-plane lattice translations.
    periodic_image_range
        Number of cells included on either side along each in-plane lattice
        direction. ``1`` gives a 3 x 3 patch and ``2`` gives a 5 x 5 patch.
    periodic_images_candidate_only
        If ``True``, repeat only candidate-species atoms. If ``False``, repeat
        all atoms recorded in the candidate plane, including coplanar species.
    show_in_plane_neighbor_lines
        Draw nearest-neighbor connections from central candidate sites to the
        complete tiled candidate layer.
    neighbor_cutoff_scale
        Multiplicative tolerance applied to the median nearest-neighbor
        distance when constructing displayed connections.
    annotate_species
        Label central candidate and coplanar sites as, for example, ``Si12``.
    species_style_map
        Optional species-style overrides.
    view_elev, view_azim
        Fixed Matplotlib camera elevation and azimuth in degrees. These replace
        the previous axis-dependent camera selection.
    figsize
        Figure size used when ``ax`` is not supplied.
    ax
        Optional existing Matplotlib 3D axes.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and 3D axes. The function does not call ``plt.show``.
    """
    if plane_window_angstrom is not None and plane_window_angstrom < 0:
        raise VisualizationError(
            "plane_window_angstrom must be non-negative or None"
        )

    if int(periodic_image_range) < 0:
        raise VisualizationError(
            "periodic_image_range must be non-negative"
        )

    if float(neighbor_cutoff_scale) <= 1.0:
        raise VisualizationError(
            "neighbor_cutoff_scale must be greater than 1.0"
        )

    ctx = _plane_context(structure, result)
    fig, ax = _make_ax(
        ax,
        projection="3d",
        figsize=figsize,
    )

    cart = np.asarray(ctx["cart"], dtype=float)
    species = np.asarray(ctx["species"], dtype=object)
    center = np.asarray(ctx["center"], dtype=float)
    e1 = _unit(
        np.asarray(ctx["basis"][0], dtype=float)
    )
    e2 = _unit(
        np.asarray(ctx["basis"][1], dtype=float)
    )
    normal = _unit(
        np.asarray(ctx["normal"], dtype=float)
    )

    styles = _style_map(
        species,
        species_style_map,
    )

    candidate_indices = np.unique(
        np.asarray(
            ctx["candidate_indices"],
            dtype=int,
        )
    )

    plane_indices = np.unique(
        np.asarray(
            ctx.get(
                "plane_indices",
                candidate_indices,
            ),
            dtype=int,
        )
    )

    adjacent_by_side: Dict[str, np.ndarray] = {}

    if include_adjacent_planes:
        adjacent_by_side = {
            str(side): np.unique(
                np.asarray(indices, dtype=int)
            )
            for side, indices in ctx["adjacent"].items()
            if len(indices)
        }

    if adjacent_by_side:
        adjacent_indices = np.unique(
            np.concatenate(
                list(adjacent_by_side.values())
            )
        )
    else:
        adjacent_indices = np.array(
            [],
            dtype=int,
        )

    candidate_set = set(
        candidate_indices.tolist()
    )
    plane_set = set(
        plane_indices.tolist()
    )
    adjacent_set = set(
        adjacent_indices.tolist()
    )

    signed_distance = (
        cart - center
    ) @ normal

    if plane_window_angstrom is None:
        keep = np.ones(
            len(cart),
            dtype=bool,
        )
    else:
        keep = (
            np.abs(signed_distance)
            <= float(plane_window_angstrom)
        )

    forced_indices = (
        candidate_set
        | plane_set
        | adjacent_set
    )

    if forced_indices:
        keep[
            np.fromiter(
                forced_indices,
                dtype=int,
            )
        ] = True

    def atom_role(index: int) -> str:
        if index in candidate_set:
            return "candidate"

        if index in plane_set:
            return "coplanar"

        if index in adjacent_set:
            return "adjacent"

        return "environment"

    role_styles = {
        "candidate": {
            "size": 95,
            "alpha": 0.98,
            "line_width": 1.6,
        },
        "coplanar": {
            "size": 62,
            "alpha": 0.82,
            "line_width": 1.0,
        },
        "adjacent": {
            "size": 52,
            "alpha": 0.68,
            "line_width": 0.8,
        },
        "environment": {
            "size": 28,
            "alpha": 0.22,
            "line_width": 0.35,
        },
    }

    role_labels = {
        "candidate": "candidate",
        "coplanar": "same plane",
        "adjacent": "adjacent plane",
        "environment": "environment",
    }

    central_image = (
        0,
        0,
        0,
    )

    periodic_translations = [
        (
            central_image,
            np.zeros(3, dtype=float),
        )
    ]

    if include_periodic_images:
        periodic_translations = (
            _in_plane_periodic_translations(
                structure,
                axis=str(ctx["axis"]),
                image_range=int(
                    periodic_image_range
                ),
            )
        )

    display_records: List[
        Dict[str, Any]
    ] = []

    for index in np.flatnonzero(keep):
        index = int(index)
        role = atom_role(index)

        repeat_atom = (
            include_periodic_images
            and role
            in {
                "candidate",
                "coplanar",
            }
        )

        if (
            periodic_images_candidate_only
            and role != "candidate"
        ):
            repeat_atom = False

        if repeat_atom:
            translations = periodic_translations
        else:
            translations = [
                (
                    central_image,
                    np.zeros(
                        3,
                        dtype=float,
                    ),
                )
            ]

        for image, translation in translations:
            display_records.append(
                {
                    "site_index": index,
                    "species": str(
                        species[index]
                    ),
                    "role": role,
                    "image": image,
                    "position": (
                        cart[index]
                        + translation
                    ),
                    "is_central": (
                        image
                        == central_image
                    ),
                }
            )

    # Draw central-cell atoms and periodic images separately.
    for current_role in (
        "environment",
        "adjacent",
        "coplanar",
        "candidate",
    ):
        role_records = [
            record
            for record in display_records
            if (
                record["role"]
                == current_role
            )
        ]

        current_species = sorted(
            {
                record["species"]
                for record
                in role_records
            }
        )

        for symbol in current_species:
            species_records = [
                record
                for record
                in role_records
                if (
                    record["species"]
                    == symbol
                )
            ]

            style = styles.get(
                symbol,
                {},
            )

            color = style.get(
                "color",
                "0.5",
            )

            marker = style.get(
                "marker",
                "o",
            )

            base_style = role_styles[
                current_role
            ]

            for image_category in (
                "central",
                "periodic",
            ):
                is_periodic = (
                    image_category
                    == "periodic"
                )

                records = [
                    record
                    for record
                    in species_records
                    if (
                        (
                            not record[
                                "is_central"
                            ]
                        )
                        == is_periodic
                    )
                ]

                if not records:
                    continue

                xyz = np.asarray(
                    [
                        record["position"]
                        for record
                        in records
                    ],
                    dtype=float,
                )

                size = (
                    base_style["size"]
                    * (
                        0.78
                        if is_periodic
                        else 1.0
                    )
                )

                alpha = (
                    base_style["alpha"]
                    * (
                        0.55
                        if is_periodic
                        else 1.0
                    )
                )

                linewidth = (
                    base_style[
                        "line_width"
                    ]
                    * (
                        0.75
                        if is_periodic
                        else 1.0
                    )
                )

                suffix = (
                    " periodic"
                    if is_periodic
                    else ""
                )

                scatter_kwargs: Dict[
                    str,
                    Any,
                ] = {
                    "s": size,
                    "marker": marker,
                    "alpha": alpha,
                    "linewidths": linewidth,
                    "label": (
                        f"{symbol} "
                        f"{role_labels[current_role]}"
                        f"{suffix}"
                    ),
                    "depthshade": True,
                }

                if is_periodic:
                    scatter_kwargs.update(
                        facecolors="none",
                        edgecolors=[
                            color
                        ],
                    )
                else:
                    scatter_kwargs.update(
                        c=[color],
                        edgecolors=(
                            "black"
                            if (
                                current_role
                                == "candidate"
                            )
                            else "0.3"
                        ),
                    )

                ax.scatter(
                    xyz[:, 0],
                    xyz[:, 1],
                    xyz[:, 2],
                    **scatter_kwargs,
                )

                if (
                    annotate_species
                    and not is_periodic
                    and current_role
                    in {
                        "candidate",
                        "coplanar",
                    }
                ):
                    for (
                        record,
                        point,
                    ) in zip(
                        records,
                        xyz,
                    ):
                        ax.text(
                            point[0],
                            point[1],
                            point[2],
                            (
                                f"{record['species']}"
                                f"{record['site_index']}"
                            ),
                            fontsize=8,
                        )

    # Draw nearest-neighbor connections from central candidate
    # sites to the complete tiled candidate layer.
    candidate_symbol = str(
        getattr(
            result,
            "species",
            "",
        )
    )

    tiled_candidate_records = [
        record
        for record in display_records
        if (
            record["role"]
            == "candidate"
            and (
                not candidate_symbol
                or record["species"]
                == candidate_symbol
            )
        )
    ]

    nearest_distance: Optional[
        float
    ] = None

    if (
        show_in_plane_neighbor_lines
        and len(
            tiled_candidate_records
        )
        >= 2
    ):
        positions = np.asarray(
            [
                record["position"]
                for record
                in tiled_candidate_records
            ],
            dtype=float,
        )

        central_rows = [
            row
            for row, record
            in enumerate(
                tiled_candidate_records
            )
            if record["is_central"]
        ]

        nearest_distances: List[
            float
        ] = []

        for source in central_rows:
            distances = np.linalg.norm(
                positions
                - positions[source],
                axis=1,
            )

            positive = distances[
                distances > 1e-8
            ]

            if positive.size:
                nearest_distances.append(
                    float(
                        np.min(
                            positive
                        )
                    )
                )

        if nearest_distances:
            nearest_distance = float(
                np.median(
                    nearest_distances
                )
            )

            bond_cutoff = (
                float(
                    neighbor_cutoff_scale
                )
                * nearest_distance
            )

            seen_edges = set()

            for source in central_rows:
                distances = np.linalg.norm(
                    positions
                    - positions[source],
                    axis=1,
                )

                targets = np.where(
                    (
                        distances > 1e-8
                    )
                    & (
                        distances
                        <= bond_cutoff
                    )
                )[0]

                for target in targets:
                    target = int(target)

                    source_record = (
                        tiled_candidate_records[
                            source
                        ]
                    )

                    target_record = (
                        tiled_candidate_records[
                            target
                        ]
                    )

                    source_key = (
                        int(
                            source_record[
                                "site_index"
                            ]
                        ),
                        tuple(
                            source_record[
                                "image"
                            ]
                        ),
                    )

                    target_key = (
                        int(
                            target_record[
                                "site_index"
                            ]
                        ),
                        tuple(
                            target_record[
                                "image"
                            ]
                        ),
                    )

                    edge_key = tuple(
                        sorted(
                            (
                                source_key,
                                target_key,
                            ),
                            key=str,
                        )
                    )

                    if (
                        edge_key
                        in seen_edges
                    ):
                        continue

                    seen_edges.add(
                        edge_key
                    )

                    p0 = positions[
                        source
                    ]

                    p1 = positions[
                        target
                    ]

                    ax.plot(
                        [
                            p0[0],
                            p1[0],
                        ],
                        [
                            p0[1],
                            p1[1],
                        ],
                        [
                            p0[2],
                            p1[2],
                        ],
                        color="0.15",
                        linewidth=1.15,
                        alpha=0.62,
                    )

            # Empty artist for one legend entry.
            ax.plot(
                [],
                [],
                [],
                color="0.15",
                linewidth=1.15,
                alpha=0.62,
                label=(
                    "candidate nearest "
                    "neighbors "
                    f"(~{nearest_distance:.2f} A)"
                ),
            )

    # Size the translucent surface from all displayed
    # candidate-plane atoms.
    surface_positions = np.asarray(
        [
            record["position"]
            for record
            in display_records
            if record["role"]
            in {
                "candidate",
                "coplanar",
            }
        ],
        dtype=float,
    )

    lattice_matrix = np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )

    lattice_scale = max(
        float(
            np.linalg.norm(vector)
        )
        for vector
        in lattice_matrix
    )

    fallback_half_width = max(
        0.35 * lattice_scale,
        1.0,
    )

    if surface_positions.size:
        offsets = (
            surface_positions
            - center
        )

        u_coordinates = (
            offsets @ e1
        )

        v_coordinates = (
            offsets @ e2
        )
    else:
        u_coordinates = np.array(
            []
        )

        v_coordinates = np.array(
            []
        )

    def padded_limits(
        values: np.ndarray,
    ) -> Tuple[float, float]:
        if (
            values.size
            and np.ptp(values)
            > 1e-8
        ):
            span = float(
                np.ptp(values)
            )

            padding = max(
                0.08 * span,
                0.35,
            )

            return (
                float(
                    np.min(values)
                    - padding
                ),
                float(
                    np.max(values)
                    + padding
                ),
            )

        return (
            -fallback_half_width,
            fallback_half_width,
        )

    u_min, u_max = (
        padded_limits(
            u_coordinates
        )
    )

    v_min, v_max = (
        padded_limits(
            v_coordinates
        )
    )

    plane_corners = np.array(
        [
            (
                center
                + u_min * e1
                + v_min * e2
            ),
            (
                center
                + u_max * e1
                + v_min * e2
            ),
            (
                center
                + u_max * e1
                + v_max * e2
            ),
            (
                center
                + u_min * e1
                + v_max * e2
            ),
        ],
        dtype=float,
    )

    plane_surface = (
        Poly3DCollection(
            [plane_corners],
            facecolors="0.72",
            edgecolors="0.35",
            linewidths=0.6,
            alpha=0.14,
        )
    )

    ax.add_collection3d(
        plane_surface
    )

    plane_span = max(
        u_max - u_min,
        v_max - v_min,
    )

    normal_length = max(
        0.18 * plane_span,
        1.5,
    )

    normal_tip = (
        center
        + normal_length
        * normal
    )

    ax.quiver(
        center[0],
        center[1],
        center[2],
        normal[0],
        normal[1],
        normal[2],
        length=normal_length,
        color="black",
        linewidth=1.3,
        arrow_length_ratio=0.14,
    )

    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        marker="+",
        s=75,
        color="black",
        label="plane center",
    )

    if include_adjacent_planes:
        for (
            side,
            indices,
        ) in adjacent_by_side.items():
            separation = float(
                np.mean(
                    (
                        cart[indices]
                        - center
                    )
                    @ normal
                )
            )

            label_position = (
                center
                + separation
                * normal
            )

            ax.text(
                label_position[0],
                label_position[1],
                label_position[2],
                (
                    f"{side}: "
                    f"{separation:+.2f} A"
                ),
                fontsize=8,
                ha="center",
                va="bottom",
            )

    all_display_positions = np.asarray(
        [
            record["position"]
            for record
            in display_records
        ],
        dtype=float,
    )

    limit_points = [
        all_display_positions,
        plane_corners,
        normal_tip[None, :],
    ]

    _set_3d_equal(
        ax,
        np.vstack(
            [
                points
                for points
                in limit_points
                if points.size
            ]
        ),
    )

    # Fixed oblique camera, independent of result.axis.
    ax.view_init(
        elev=float(view_elev),
        azim=float(view_azim),
    )

    ax.set_xlabel(
        "Cartesian x (A)"
    )

    ax.set_ylabel(
        "Cartesian y (A)"
    )

    ax.set_zlabel(
        "Cartesian z (A)"
    )

    status = (
        "PASS"
        if _result_passes(result)
        else "FAIL"
    )

    if include_periodic_images:
        tile_size = (
            2
            * int(
                periodic_image_range
            )
            + 1
        )
    else:
        tile_size = 1

    title = (
        f"{_formula(structure)} | "
        f"{candidate_symbol} "
        f"{ctx['axis']}-plane | "
        f"{status}"
        f"\nin-plane patch="
        f"{tile_size}x{tile_size}, "
        f"elev={float(view_elev):.0f} deg, "
        f"azim={float(view_azim):.0f} deg"
    )

    ax.set_title(title)

    # Deduplicate labels and place the legend outside.
    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique: Dict[
        str,
        Any,
    ] = {}

    for (
        handle,
        label,
    ) in zip(
        handles,
        labels,
    ):
        if (
            label
            and label
            not in unique
        ):
            unique[label] = handle

    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            bbox_to_anchor=(
                1.02,
                1.0,
            ),
            borderaxespad=0.0,
            fontsize=10,
        )

        fig.subplots_adjust(
            right=0.76
        )

    return fig, ax

def _in_plane_periodic_translations(
    structure: Any,
    axis: str,
    image_range: int,
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    """Return lattice translations confined to a candidate plane.

    Parameters
    ----------
    structure
        Pymatgen Structure-like object.

    axis
        Crystallographic axis normal to the candidate plane: ``a``, ``b``,
        or ``c``.

    image_range
        Number of periodic cells to include in each in-plane direction.
        For example, ``1`` returns a 3 x 3 patch.

    Returns
    -------
    list
        Tuples containing the integer image vector and Cartesian translation.
    """
    if image_range < 0:
        raise VisualizationError(
            "periodic_image_range must be non-negative"
        )

    axis = str(axis).lower()

    if axis not in {"a", "b", "c"}:
        raise VisualizationError(
            f"Unsupported candidate-plane axis: {axis!r}"
        )

    lattice = np.asarray(structure.lattice.matrix, dtype=float)

    # Pymatgen lattice.matrix stores a, b, and c as its rows.
    normal_axis_index = {
        "a": 0,
        "b": 1,
        "c": 2,
    }[axis]

    in_plane_axis_indices = [
        index
        for index in range(3)
        if index != normal_axis_index
    ]

    first_axis, second_axis = in_plane_axis_indices
    translations = []

    for first_shift in range(-image_range, image_range + 1):
        for second_shift in range(-image_range, image_range + 1):
            image = [0, 0, 0]
            image[first_axis] = first_shift
            image[second_axis] = second_shift

            image_tuple = tuple(image)

            translation = (
                first_shift * lattice[first_axis]
                + second_shift * lattice[second_axis]
            )

            translations.append(
                (
                    image_tuple,
                    np.asarray(translation, dtype=float),
                )
            )

    return translations


def plot_candidate_plane_3d_interactive(
    structure: Any,
    result: Any,
    *,
    plane_window_angstrom: Optional[float] = None,
    include_adjacent_planes: bool = True,
    include_periodic_images: bool = True,
    periodic_image_range: int = 1,
    periodic_images_candidate_only: bool = False,
    show_in_plane_neighbor_lines: bool = True,
    neighbor_cutoff_scale: float = 1.15,
    annotate_species: bool = False,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    width: int = 1000,
    height: int = 700,
) -> Any:
    """Return an interactive Plotly view of a candidate square-net plane.

    Candidate-plane atoms can be repeated only along the two lattice vectors
    lying in the detector plane. With the default ``periodic_image_range=1``,
    the plot contains a 3 x 3 in-plane patch, making square connectivity across
    unit-cell boundaries visible while avoiding repeated out-of-plane layers.

    Parameters
    ----------
    structure
        Pymatgen Structure-like object.
    result
        Candidate-layer result returned by ``find_square_net_planes``.
    plane_window_angstrom
        Include central-cell atoms whose perpendicular distance from the
        candidate plane is no greater than this value. ``None`` includes the
        full central-cell structure.
    include_adjacent_planes
        Highlight detector-recorded adjacent planes in the central cell.
    include_periodic_images
        Repeat candidate-plane atoms using in-plane lattice translations.
    periodic_image_range
        Number of cells to include on either side along each in-plane lattice
        direction. ``1`` gives a 3 x 3 patch and ``2`` gives a 5 x 5 patch.
    periodic_images_candidate_only
        If ``True``, repeat only atoms of the candidate species. If ``False``,
        repeat all atoms recorded in the candidate plane, including coplanar
        atoms of other species.
    show_in_plane_neighbor_lines
        Draw nearest-neighbor connections from central candidate sites to the
        complete tiled candidate layer.
    neighbor_cutoff_scale
        Multiplicative tolerance applied to the median nearest-neighbor
        distance when constructing the displayed connection graph.
    annotate_species
        Label central candidate and coplanar atoms as, for example, ``Si12``.
    species_style_map
        Optional species-style overrides using the same format as the static
        Matplotlib visualization functions.
    width, height
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive three-dimensional Plotly figure for Jupyter notebooks.
    """
    try:
        import plotly.graph_objects as go
        from matplotlib.colors import to_hex
    except ImportError as exc:
        raise ImportError(
            "Interactive plotting requires Plotly. Install it with "
            "`python -m pip install plotly`."
        ) from exc

    if plane_window_angstrom is not None and plane_window_angstrom < 0:
        raise VisualizationError("plane_window_angstrom must be non-negative or None")
    if int(periodic_image_range) < 0:
        raise VisualizationError("periodic_image_range must be non-negative")
    if float(neighbor_cutoff_scale) <= 1.0:
        raise VisualizationError("neighbor_cutoff_scale must be greater than 1.0")

    ctx = _plane_context(structure, result)
    cart = np.asarray(ctx["cart"], dtype=float)
    species = np.asarray(ctx["species"], dtype=object)
    center = np.asarray(ctx["center"], dtype=float)
    e1 = _unit(np.asarray(ctx["basis"][0], dtype=float))
    e2 = _unit(np.asarray(ctx["basis"][1], dtype=float))
    normal = _unit(np.asarray(ctx["normal"], dtype=float))
    styles = _style_map(species, species_style_map)

    candidate_indices = np.unique(np.asarray(ctx["candidate_indices"], dtype=int))
    plane_indices = np.unique(
        np.asarray(ctx.get("plane_indices", candidate_indices), dtype=int)
    )

    adjacent_by_side: Dict[str, np.ndarray] = {}
    if include_adjacent_planes:
        adjacent_by_side = {
            str(side): np.unique(np.asarray(indices, dtype=int))
            for side, indices in ctx["adjacent"].items()
            if len(indices)
        }

    if adjacent_by_side:
        adjacent_indices = np.unique(np.concatenate(list(adjacent_by_side.values())))
    else:
        adjacent_indices = np.array([], dtype=int)

    candidate_set = set(candidate_indices.tolist())
    plane_set = set(plane_indices.tolist())
    adjacent_set = set(adjacent_indices.tolist())
    signed_distance = (cart - center) @ normal

    if plane_window_angstrom is None:
        keep = np.ones(len(cart), dtype=bool)
    else:
        keep = np.abs(signed_distance) <= float(plane_window_angstrom)

    forced_indices = candidate_set | plane_set | adjacent_set
    if forced_indices:
        keep[np.fromiter(forced_indices, dtype=int)] = True

    def atom_role(index: int) -> str:
        if index in candidate_set:
            return "candidate"
        if index in plane_set:
            return "coplanar"
        if index in adjacent_set:
            return "adjacent"
        return "environment"

    role_styles = {
        "candidate": {"size": 11, "symbol": "circle", "opacity": 1.00, "line_width": 2.0},
        "coplanar": {"size": 8, "symbol": "diamond", "opacity": 0.85, "line_width": 1.2},
        "adjacent": {"size": 8, "symbol": "square", "opacity": 0.75, "line_width": 1.0},
        "environment": {"size": 5, "symbol": "circle", "opacity": 0.25, "line_width": 0.4},
    }
    role_labels = {
        "candidate": "candidate",
        "coplanar": "same plane",
        "adjacent": "adjacent plane",
        "environment": "environment",
    }

    central_image = (0, 0, 0)
    periodic_translations = [(central_image, np.zeros(3, dtype=float))]
    if include_periodic_images:
        periodic_translations = _in_plane_periodic_translations(
            structure, axis=str(ctx["axis"]), image_range=int(periodic_image_range)
        )

    display_records: List[Dict[str, Any]] = []
    for index in np.flatnonzero(keep):
        index = int(index)
        role = atom_role(index)
        repeat_atom = include_periodic_images and role in {"candidate", "coplanar"}
        if periodic_images_candidate_only and role != "candidate":
            repeat_atom = False
        translations = periodic_translations if repeat_atom else [
            (central_image, np.zeros(3, dtype=float))
        ]
        for image, translation in translations:
            display_records.append(
                {
                    "site_index": index,
                    "species": str(species[index]),
                    "role": role,
                    "image": image,
                    "position": cart[index] + translation,
                    "plane_distance": float(signed_distance[index]),
                    "is_central": image == central_image,
                }
            )

    fig = go.Figure()
    for current_role in ("environment", "adjacent", "coplanar", "candidate"):
        role_records = [r for r in display_records if r["role"] == current_role]
        for symbol in sorted({r["species"] for r in role_records}):
            species_records = [r for r in role_records if r["species"] == symbol]
            for image_category in ("central", "periodic"):
                is_periodic = image_category == "periodic"
                records = [r for r in species_records if r["is_central"] != is_periodic]
                if not records:
                    continue

                xyz = np.asarray([r["position"] for r in records], dtype=float)
                style = role_styles[current_role]
                color_value = styles.get(symbol, {}).get("color", "0.5")
                try:
                    plotly_color = to_hex(color_value)
                except (TypeError, ValueError):
                    plotly_color = str(color_value)

                marker_size = style["size"] - (1 if is_periodic else 0)
                marker_opacity = style["opacity"] * (0.65 if is_periodic else 1.0)
                line_width = style["line_width"] * (0.7 if is_periodic else 1.0)
                mode = "markers"
                labels = None
                if annotate_species and current_role in {"candidate", "coplanar"} and not is_periodic:
                    labels = [f"{r['species']}{r['site_index']}" for r in records]
                    mode = "markers+text"

                customdata = np.empty((len(records), 8), dtype=object)
                for row, record in enumerate(records):
                    image = record["image"]
                    customdata[row] = [
                        record["site_index"], record["species"], role_labels[current_role],
                        record["plane_distance"], image[0], image[1], image[2],
                        "central cell" if record["is_central"] else "periodic image",
                    ]

                suffix = " (periodic)" if is_periodic else ""
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode=mode,
                        text=labels, textposition="top center", textfont=dict(size=10),
                        name=f"{symbol} — {role_labels[current_role]}{suffix}",
                        legendgroup=f"{symbol}-{current_role}-{image_category}",
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[1]}</b><br>"
                            "site index: %{customdata[0]}<br>"
                            "role: %{customdata[2]}<br>"
                            "cell type: %{customdata[7]}<br>"
                            "image: (%{customdata[4]}, %{customdata[5]}, %{customdata[6]})<br>"
                            "x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>z: %{z:.3f} Å<br>"
                            "distance from plane: %{customdata[3]:+.3f} Å<extra></extra>"
                        ),
                        marker=dict(
                            size=max(marker_size, 3), symbol=style["symbol"],
                            color=plotly_color, opacity=marker_opacity,
                            line=dict(color="black", width=line_width),
                        ),
                    )
                )

    candidate_symbol = str(getattr(result, "species", ""))
    tiled_candidate_records = [
        r for r in display_records
        if r["role"] == "candidate" and (not candidate_symbol or r["species"] == candidate_symbol)
    ]

    if show_in_plane_neighbor_lines and len(tiled_candidate_records) >= 2:
        positions = np.asarray([r["position"] for r in tiled_candidate_records], dtype=float)
        central_rows = [i for i, r in enumerate(tiled_candidate_records) if r["is_central"]]
        nearest_distances: List[float] = []
        for source in central_rows:
            distances = np.linalg.norm(positions - positions[source], axis=1)
            positive = distances[distances > 1e-8]
            if positive.size:
                nearest_distances.append(float(np.min(positive)))

        if nearest_distances:
            nearest_distance = float(np.median(nearest_distances))
            bond_cutoff = float(neighbor_cutoff_scale) * nearest_distance
            line_x: List[Optional[float]] = []
            line_y: List[Optional[float]] = []
            line_z: List[Optional[float]] = []
            seen_edges = set()

            for source in central_rows:
                distances = np.linalg.norm(positions - positions[source], axis=1)
                targets = np.where((distances > 1e-8) & (distances <= bond_cutoff))[0]
                for target in targets:
                    source_record = tiled_candidate_records[source]
                    target_record = tiled_candidate_records[int(target)]
                    source_key = (int(source_record["site_index"]), tuple(source_record["image"]))
                    target_key = (int(target_record["site_index"]), tuple(target_record["image"]))
                    edge_key = tuple(sorted((source_key, target_key), key=str))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    p0, p1 = positions[source], positions[int(target)]
                    line_x.extend([float(p0[0]), float(p1[0]), None])
                    line_y.extend([float(p0[1]), float(p1[1]), None])
                    line_z.extend([float(p0[2]), float(p1[2]), None])

            if line_x:
                fig.add_trace(
                    go.Scatter3d(
                        x=line_x, y=line_y, z=line_z, mode="lines",
                        line=dict(color="rgba(40,40,40,0.55)", width=4),
                        name=f"candidate nearest neighbors (~{nearest_distance:.2f} Å)",
                        hoverinfo="skip",
                    )
                )

    surface_positions = np.asarray(
        [r["position"] for r in display_records if r["role"] in {"candidate", "coplanar"}],
        dtype=float,
    )
    if surface_positions.size:
        offsets = surface_positions - center
        u_coordinates = offsets @ e1
        v_coordinates = offsets @ e2
    else:
        u_coordinates = np.array([])
        v_coordinates = np.array([])

    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    lattice_scale = max(float(np.linalg.norm(vector)) for vector in lattice_matrix)
    fallback_half_width = max(0.35 * lattice_scale, 1.0)

    def padded_limits(values: np.ndarray) -> Tuple[float, float]:
        if values.size and np.ptp(values) > 1e-8:
            span = float(np.ptp(values))
            padding = max(0.08 * span, 0.35)
            return float(np.min(values) - padding), float(np.max(values) + padding)
        return -fallback_half_width, fallback_half_width

    u_min, u_max = padded_limits(u_coordinates)
    v_min, v_max = padded_limits(v_coordinates)
    plane_corners = np.array([
        center + u_min * e1 + v_min * e2,
        center + u_max * e1 + v_min * e2,
        center + u_max * e1 + v_max * e2,
        center + u_min * e1 + v_max * e2,
    ])
    fig.add_trace(
        go.Mesh3d(
            x=plane_corners[:, 0], y=plane_corners[:, 1], z=plane_corners[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3], color="gray", opacity=0.14,
            flatshading=True, name="candidate plane", hoverinfo="skip",
        )
    )

    plane_span = max(u_max - u_min, v_max - v_min)
    normal_length = max(0.18 * plane_span, 1.5)
    normal_tip = center + normal_length * normal
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], normal_tip[0]], y=[center[1], normal_tip[1]],
            z=[center[2], normal_tip[2]], mode="lines",
            line=dict(color="black", width=6), name=f"{ctx['axis']}-plane normal",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[normal_tip[0]], y=[normal_tip[1]], z=[normal_tip[2]],
            u=[normal[0]], v=[normal[1]], w=[normal[2]], anchor="tip",
            sizemode="absolute", sizeref=max(0.18 * normal_length, 0.25),
            colorscale=[[0.0, "black"], [1.0, "black"]], showscale=False,
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
            marker=dict(size=6, symbol="cross", color="black"), name="plane center",
            hovertemplate=(
                "plane center<br>x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>"
                "z: %{z:.3f} Å<extra></extra>"
            ),
        )
    )

    annotations = []
    for side, indices in adjacent_by_side.items():
        separation = float(np.mean((cart[indices] - center) @ normal))
        label_position = center + separation * normal
        annotations.append(
            dict(
                x=float(label_position[0]), y=float(label_position[1]),
                z=float(label_position[2]), text=f"{side}: {separation:+.2f} Å",
                showarrow=False, font=dict(size=11),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(80,80,80,0.4)", borderwidth=1,
            )
        )

    camera_direction = _unit(1.25 * e1 + 1.00 * e2 + 0.85 * normal)
    camera_eye = 2.15 * camera_direction
    status = "PASS" if _result_passes(result) else "FAIL"
    formula = _formula(structure)
    mean_score = float(getattr(result, "mean_score", np.nan))
    pass_fraction = float(getattr(result, "pass_fraction", np.nan))
    tile_size = 2 * int(periodic_image_range) + 1 if include_periodic_images else 1
    title = (
        f"{formula} | {candidate_symbol} {ctx['axis']}-plane | {status}"
        f"<br><sup>mean score={mean_score:.3g}, pass fraction={pass_fraction:.3g}; "
        f"in-plane patch={tile_size}×{tile_size}</sup>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.02), width=int(width), height=int(height),
        margin=dict(l=0, r=250, b=0, t=80),
        legend=dict(
            x=1.02, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(80,80,80,0.35)", borderwidth=1,
            groupclick="togglegroup",
        ),
        scene=dict(
            xaxis_title="Cartesian x (Å)", yaxis_title="Cartesian y (Å)",
            zaxis_title="Cartesian z (Å)", aspectmode="data",
            annotations=annotations,
            camera=dict(eye=dict(
                x=float(camera_eye[0]), y=float(camera_eye[1]), z=float(camera_eye[2])
            )),
        ),
        hoverlabel=dict(namelength=-1),
    )
    return fig

def select_representative_site(result: Any, strategy: str = "median") -> int:
    """Select a candidate site index by local score.

    Parameters
    ----------
    result
        Detector result with ``visualization_data.local_site_scores`` and
        ``candidate_site_indices``.
    strategy
        ``"median"``, ``"best"``, or ``"worst"``.

    Returns
    -------
    int
        Global atom index for the selected candidate site.
    """
    viz = _viz(result)
    if viz is None or getattr(viz, "local_site_scores", None) is None:
        raise MissingVisualizationDiagnostics("local site diagnostics are unavailable")
    scores = np.asarray(viz.local_site_scores, dtype=float)
    indices = np.asarray(viz.candidate_site_indices, dtype=int)
    if scores.size == 0 or indices.size == 0:
        raise VisualizationError("result has no candidate sites")
    if scores.size != indices.size:
        raise VisualizationError("local_site_scores length does not match candidate_site_indices")

    strategy = str(strategy).lower()
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        chosen = 0
    elif strategy == "best":
        chosen = int(finite[np.argmax(scores[finite])])
    elif strategy == "worst":
        chosen = int(finite[np.argmin(scores[finite])])
    elif strategy == "median":
        med = float(np.nanmedian(scores[finite]))
        chosen = int(finite[np.argmin(np.abs(scores[finite] - med))])
    else:
        raise VisualizationError("strategy must be one of 'median', 'best', or 'worst'")
    return int(indices[chosen])


def _site_detail(result: Any, site_index: int) -> Dict[str, Any]:
    viz = _viz(result)
    if viz is None or getattr(viz, "local_site_details", None) is None:
        raise MissingVisualizationDiagnostics("local site diagnostics are unavailable")
    details = list(viz.local_site_details)
    for detail in details:
        if int(detail.get("site_index", -1)) == int(site_index):
            return detail
    # Be forgiving for callers who pass a local candidate index.
    if 0 <= int(site_index) < len(details):
        return details[int(site_index)]
    raise VisualizationError(f"site_index {site_index!r} is not present in this candidate layer")


def plot_site_geometry(
    result: Any,
    site_index: int,
    *,
    annotate_lengths: bool = True,
    annotate_angles: bool = True,
    show_ideal_guides: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Explain the local square score for one candidate atom.

    The plot is in projected detector coordinates centered on the selected atom.
    Gray arrows show nearby vectors considered by the detector; dark arrows show
    the selected u/v/opposite vectors when available.
    """
    detail = _site_detail(result, site_index)
    fig, ax = _make_ax(ax, figsize=(5.0, 4.6))

    vectors = np.asarray(detail.get("neighbor_vectors", []), dtype=float)
    distances = np.asarray(detail.get("neighbor_distances", []), dtype=float)
    selected = np.asarray(detail.get("selected_neighbor_vectors", []), dtype=float)
    selected_dist = np.asarray(detail.get("selected_neighbor_distances", []), dtype=float)
    threshold = _thresholds(result).get("score_threshold", 0.5)
    score = float(detail.get("score", np.nan))
    passes = bool(np.isfinite(score) and score >= threshold)

    for vec in vectors:
        ax.arrow(0, 0, vec[0], vec[1], length_includes_head=True, head_width=0.04, head_length=0.08, color="0.65", alpha=0.45)
    for i, vec in enumerate(selected):
        ax.arrow(0, 0, vec[0], vec[1], length_includes_head=True, head_width=0.07, head_length=0.11, color="0.08", linewidth=1.4)
        ax.scatter([vec[0]], [vec[1]], s=52, facecolors="white", edgecolors="0.08", linewidths=1.2, zorder=4)
        if annotate_lengths:
            dist = selected_dist[i] if i < len(selected_dist) else float(np.linalg.norm(vec))
            mid = 0.52 * vec
            ax.annotate(f"{dist:.2f} A", mid, fontsize=8, ha="center", va="bottom")

    ax.scatter([0], [0], s=90, c="0.12", marker="o", edgecolors="black", linewidths=1.3, label="central site")

    info = dict(detail.get("score_info", {}))
    if annotate_angles and np.isfinite(info.get("ang_deg", np.nan)):
        ax.annotate(f"angle={float(info['ang_deg']):.1f} deg", (0.03, 0.95), xycoords="axes fraction", fontsize=9, va="top")

    if show_ideal_guides and len(selected) >= 2:
        u = selected[0]
        length = float(np.linalg.norm(u))
        if length > 1e-12:
            ux = u / length
            uy = np.array([-ux[1], ux[0]])
            ax.plot([0, ux[0] * length], [0, ux[1] * length], linestyle="--", color="0.25", linewidth=1.0, label="ideal guide")
            ax.plot([0, uy[0] * length], [0, uy[1] * length], linestyle="--", color="0.25", linewidth=1.0)
            arc = Arc((0, 0), width=0.45 * length, height=0.45 * length, theta1=0, theta2=90, color="0.35", linestyle=":")
            ax.add_patch(arc)

    if vectors.size:
        lim = max(1.0, float(np.nanmax(np.linalg.norm(vectors, axis=1))) * 1.25)
    elif selected.size:
        lim = max(1.0, float(np.nanmax(np.linalg.norm(selected, axis=1))) * 1.25)
    else:
        lim = 1.0
        ax.text(0.5, 0.5, "No local neighbor diagnostics", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.85", linewidth=0.7)
    ax.axvline(0, color="0.85", linewidth=0.7)
    ax.set_xlabel("Delta u (A)")
    ax.set_ylabel("Delta v (A)")
    status = "passes local cutoff" if passes else "fails local cutoff"
    ax.set_title(f"Site {int(detail.get('site_index', site_index))}: score={score:.3g} ({status})")
    return fig, ax


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def plot_score_components(
    result: Any,
    *,
    settings: Optional[Mapping[str, Any]] = None,
    include_diagnostics: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot the criteria that actually determine ``passes`` and ``passes2``.

    Quantitative criteria are represented as normalized margins:

        - lower-bound criterion: measured / lower_bound
        - upper-bound criterion: upper_bound / measured

    A normalized margin >= 1 satisfies the criterion.

    Boolean criteria are shown as pass/fail statuses and are not assigned an
    artificial numerical margin.

    Parameters
    ----------
    result
        A SquarePlaneResult-like object.
    settings
        Detector settings used to produce the result. This is needed for
        passes2 thresholds that are not stored directly on the result.
    include_diagnostics
        Add non-decision quantities such as mean score and mean geometric
        errors in a text box.
    ax
        Optional Matplotlib axis.
    """
    fig, ax = _make_ax(ax, figsize=(8.0, 5.2))

    settings = dict(settings or {})
    stored_thresholds = _thresholds(result)

    # Prefer explicitly supplied settings, then fall back to thresholds stored
    # with the visualization data.
    def threshold(name: str, default: Any = None) -> Any:
        if name in settings:
            return settings[name]
        return stored_thresholds.get(name, default)

    quantitative_rows: List[Dict[str, Any]] = []
    boolean_rows: List[Dict[str, Any]] = []

    def add_lower_bound(
        label: str,
        measured: Any,
        lower: Any,
        *,
        units: str = "",
    ) -> None:
        value = _finite_float(measured)
        bound = _finite_float(lower)

        if bound is None:
            return

        if value is None:
            quantitative_rows.append(
                {
                    "label": label,
                    "margin": 0.0,
                    "passed": False,
                    "text": f"missing; required ≥ {bound:g}{units}",
                }
            )
            return

        if bound <= 0:
            return

        margin = value / bound
        quantitative_rows.append(
            {
                "label": label,
                "margin": margin,
                "passed": value >= bound,
                "text": f"{value:.3g}{units} ≥ {bound:.3g}{units}",
            }
        )

    def add_upper_bound(
        label: str,
        measured: Any,
        upper: Any,
        *,
        units: str = "",
    ) -> None:
        value = _finite_float(measured)
        bound = _finite_float(upper)

        if bound is None:
            return

        if value is None:
            quantitative_rows.append(
                {
                    "label": label,
                    "margin": 0.0,
                    "passed": False,
                    "text": f"missing; required ≤ {bound:g}{units}",
                }
            )
            return

        if value <= 1e-12:
            margin = np.inf
        else:
            margin = bound / value

        quantitative_rows.append(
            {
                "label": label,
                "margin": margin,
                "passed": value <= bound,
                "text": f"{value:.3g}{units} ≤ {bound:.3g}{units}",
            }
        )

    def add_boolean(label: str, passed: bool, text: str) -> None:
        boolean_rows.append(
            {
                "label": label,
                "passed": bool(passed),
                "text": text,
            }
        )

    # ------------------------------------------------------------------
    # Actual primary-pass criterion
    # ------------------------------------------------------------------
    add_lower_bound(
        "sites meeting local-score threshold",
        getattr(result, "pass_fraction", np.nan),
        threshold("min_pass_fraction", 0.6),
    )

    # ------------------------------------------------------------------
    # Actual passes2 numeric criteria
    # ------------------------------------------------------------------
    add_lower_bound(
        "minimum adjacent-atom clearance",
        getattr(result, "min_adj_dist_any_atom", np.nan),
        threshold("min_adj_dist_any_atom_min", 2.0),
        units=" Å",
    )

    add_upper_bound(
        "shortest in-plane same-species spacing",
        getattr(result, "nn_intra_min", np.nan),
        threshold("nn_intra_min_max", 4.0),
        units=" Å",
    )

    # Add optional bounds only when enabled.
    add_lower_bound(
        "in-plane / adjacent-distance ratio",
        getattr(result, "tol_ratio_any", np.nan),
        threshold("tol_ratio_any_min", None),
    )
    add_upper_bound(
        "in-plane / adjacent-distance ratio",
        getattr(result, "tol_ratio_any", np.nan),
        threshold("tol_ratio_any_max", None),
    )

    add_lower_bound(
        "plane-selected adjacent distance",
        getattr(result, "min_adj_dist_any_plane", np.nan),
        threshold("min_adj_dist_any_plane_min", None),
        units=" Å",
    )
    add_upper_bound(
        "plane-selected adjacent distance",
        getattr(result, "min_adj_dist_any_plane", np.nan),
        threshold("min_adj_dist_any_plane_max", None),
        units=" Å",
    )

    add_lower_bound(
        "nearest plane-center separation",
        getattr(result, "closest_by_plane_sep_ang", np.nan),
        threshold("closest_by_plane_sep_ang_min", None),
        units=" Å",
    )
    add_upper_bound(
        "nearest plane-center separation",
        getattr(result, "closest_by_plane_sep_ang", np.nan),
        threshold("closest_by_plane_sep_ang_max", None),
        units=" Å",
    )

    # ------------------------------------------------------------------
    # Actual passes2 Boolean criteria
    # ------------------------------------------------------------------
    primary_passed = bool(getattr(result, "passes", False))
    add_boolean(
        "primary square-geometry screen",
        primary_passed,
        "passes=True" if primary_passed else "passes=False",
    )

    forbid_mixed = threshold("forbid_coplane_mixed_species", True)
    if forbid_mixed is not None:
        mixed = bool(getattr(result, "has_coplane_other_species", False))

        if bool(forbid_mixed):
            add_boolean(
                "candidate plane contains only target species",
                not mixed,
                "pure" if not mixed else "mixed",
            )
        else:
            add_boolean(
                "candidate plane contains multiple species",
                mixed,
                "mixed" if mixed else "pure",
            )

    isolate_adjacent = threshold("isolate_same_species_adjacent", True)
    isolation_cutoff = threshold(
        "isolate_same_species_adjacent_dist_min",
        2.0,
    )

    if isolate_adjacent:
        adjacent_species = getattr(
            result,
            "closest_by_atom_atom_species",
            None,
        )

        # Some result classes may not retain the closest atom species directly.
        # In that case, use the recorded failure reason when available.
        reasons = set(getattr(result, "passes2_fail_reasons", []) or [])
        failed_isolation = "adjacent_same_species_too_close" in reasons

        if adjacent_species is not None:
            distance = _finite_float(
                getattr(result, "min_adj_dist_any_atom", np.nan)
            )
            failed_isolation = (
                str(adjacent_species) == str(getattr(result, "species", ""))
                and distance is not None
                and isolation_cutoff is not None
                and distance <= float(isolation_cutoff)
            )

        add_boolean(
            "no same-species adjacent atom inside cutoff",
            not failed_isolation,
            (
                f"clear of {float(isolation_cutoff):g} Å cutoff"
                if not failed_isolation
                else f"same species within {float(isolation_cutoff):g} Å"
            ),
        )

    enforce_bond_filter = threshold(
        "enforce_no_out_of_plane_same_species_bonds",
        True,
    )
    if enforce_bond_filter:
        oop_bond = bool(
            getattr(
                result,
                "has_out_of_plane_same_species_bond",
                False,
            )
        )
        add_boolean(
            "no out-of-plane same-species CrystalNN bond",
            not oop_bond,
            "none found" if not oop_bond else "bond found",
        )

    if not quantitative_rows and not boolean_rows:
        ax.text(
            0.5,
            0.5,
            "No detector decision diagnostics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    # ------------------------------------------------------------------
    # Draw quantitative margins
    # ------------------------------------------------------------------
    labels: List[str] = []
    display_values: List[float] = []
    passed_flags: List[bool] = []
    annotations: List[str] = []
    row_types: List[str] = []

    clip_margin = 2.5

    for row in quantitative_rows:
        raw_margin = float(row["margin"])
        display_margin = (
            clip_margin
            if not np.isfinite(raw_margin)
            else min(raw_margin, clip_margin)
        )

        labels.append(row["label"])
        display_values.append(display_margin)
        passed_flags.append(bool(row["passed"]))

        margin_label = (
            f">{clip_margin:g}"
            if not np.isfinite(raw_margin) or raw_margin > clip_margin
            else f"{raw_margin:.2f}"
        )
        annotations.append(
            f"{'PASS' if row['passed'] else 'FAIL'}  "
            f"{row['text']}  [margin {margin_label}]"
        )
        row_types.append("quantitative")

    # Boolean rows are given a fixed visual position, but the axis annotation
    # explicitly identifies them as Boolean rather than quantitative margins.
    for row in boolean_rows:
        labels.append(row["label"])
        display_values.append(1.15 if row["passed"] else 0.15)
        passed_flags.append(bool(row["passed"]))
        annotations.append(
            f"{'PASS' if row['passed'] else 'FAIL'}  {row['text']}"
        )
        row_types.append("boolean")

    y = np.arange(len(labels), dtype=float)

    bars = ax.barh(
        y,
        display_values,
        color=["0.62" if ok else "0.88" for ok in passed_flags],
        edgecolor="0.15",
        linewidth=0.9,
    )

    for bar, ok, row_type in zip(bars, passed_flags, row_types):
        if not ok:
            bar.set_hatch("//")
        if row_type == "boolean":
            bar.set_alpha(0.55)

    for yi, value, ok, text, row_type in zip(
        y,
        display_values,
        passed_flags,
        annotations,
        row_types,
    ):
        marker = "o" if ok else "x"
        ax.scatter(
            value,
            yi,
            marker=marker,
            color="0.05",
            zorder=4,
        )

        ax.text(
            1.02,
            yi,
            text,
            va="center",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

        if row_type == "boolean":
            ax.text(
                0.02,
                yi,
                "Boolean",
                va="center",
                fontsize=7,
                transform=ax.get_yaxis_transform(),
            )

    ax.axvline(
        1.0,
        color="0.1",
        linestyle="--",
        linewidth=1.0,
        label="quantitative pass boundary",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, clip_margin)
    ax.set_xlabel("Normalized margin for quantitative criteria")
    ax.set_title(
        f"Detector decision gates: "
        f"passes={bool(getattr(result, 'passes', False))}, "
        f"passes2={bool(getattr(result, 'passes2', False))}"
    )
    ax.invert_yaxis()

    # ------------------------------------------------------------------
    # Supporting diagnostics that do not independently determine the label
    # ------------------------------------------------------------------
    if include_diagnostics:
        diagnostic_lines = []

        diagnostics = [
            ("mean local score", getattr(result, "mean_score", np.nan), ""),
            ("median local score", getattr(result, "median_score", np.nan), ""),
            ("mean length error", getattr(result, "uv_len_err_mean", np.nan), ""),
            ("mean angle error", getattr(result, "uv_ang_err_mean", np.nan), "°"),
            ("mean in-plane NN", getattr(result, "nn_intra_mean", np.nan), " Å"),
        ]

        for name, raw_value, units in diagnostics:
            value = _finite_float(raw_value)
            if value is not None:
                diagnostic_lines.append(f"{name}: {value:.3g}{units}")

        failure_reasons = list(
            getattr(result, "passes2_fail_reasons", []) or []
        )
        if failure_reasons:
            diagnostic_lines.append(
                "passes2 reasons: " + ", ".join(failure_reasons)
            )

        if diagnostic_lines:
            ax.text(
                0.0,
                -0.16,
                "Supporting diagnostics — not independent decision gates\n"
                + "\n".join(diagnostic_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

    return fig, ax
    

def _measurement_values(result: Any, attr: str, fallback_attrs: Sequence[str], label: str) -> np.ndarray:
    viz = _viz(result)
    values = np.array(getattr(viz, attr, []), dtype=float) if viz is not None else np.array([], dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        return values
    fallback = []
    for name in fallback_attrs:
        x = _finite_float(getattr(result, name, np.nan))
        if x is not None:
            fallback.append(x)
    if fallback:
        warnings.warn(f"Using aggregate {label} values because per-site measurements are unavailable.", RuntimeWarning, stacklevel=2)
    return np.array(fallback, dtype=float)


def plot_neighbor_length_distribution(
    result: Any,
    *,
    show_tolerance: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Show selected in-plane neighbor length measurements in Angstrom."""
    fig, ax = _make_ax(ax, figsize=(5.8, 3.8))
    values = _measurement_values(result, "neighbor_length_measurements", ("u_len_min", "u_len_max", "nn_intra_mean"), "length")
    if values.size == 0:
        ax.text(0.5, 0.5, "No neighbor length diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax
    if values.size <= 20:
        ax.scatter(values, np.zeros_like(values), marker="o", facecolors="white", edgecolors="0.1", zorder=3)
        ax.set_yticks([])
    else:
        ax.hist(values, bins=min(20, max(5, int(math.sqrt(values.size)))), color="0.75", edgecolor="0.25")
    ref = float(np.nanmedian(values))
    ax.axvline(ref, color="0.1", linestyle="-", linewidth=1.0, label=f"reference {ref:.2f} A")
    if show_tolerance:
        lt = _thresholds(result).get("len_tol", None)
        if lt is not None and np.isfinite(ref):
            lo, hi = ref * (1.0 - float(lt)), ref * (1.0 + float(lt))
            ax.axvspan(lo, hi, color="0.7", alpha=0.22, label=f"+/-{float(lt):.1%}")
    ax.set_xlabel("Selected neighbor length (A)")
    ax.set_title("In-plane neighbor length distribution")
    ax.legend(loc="best", fontsize=10)
    return fig, ax


def plot_neighbor_angle_distribution(
    result: Any,
    *,
    show_tolerance: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Show selected u-v angle measurements in degrees."""
    fig, ax = _make_ax(ax, figsize=(5.8, 3.8))
    values = _measurement_values(result, "neighbor_angle_measurements", ("uv_ang_deg_min", "uv_ang_deg_max", "uv_ang_deg_mean"), "angle")
    if values.size == 0:
        ax.text(0.5, 0.5, "No neighbor angle diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax
    if values.size <= 20:
        ax.scatter(values, np.zeros_like(values), marker="o", facecolors="white", edgecolors="0.1", zorder=3)
        ax.set_yticks([])
    else:
        ax.hist(values, bins=min(20, max(5, int(math.sqrt(values.size)))), color="0.75", edgecolor="0.25")
    ax.axvline(90.0, color="0.1", linestyle="-", linewidth=1.0, label="ideal 90 deg")
    if show_tolerance:
        at = _thresholds(result).get("ang_tol_deg", None)
        if at is not None:
            ax.axvspan(90.0 - float(at), 90.0 + float(at), color="0.7", alpha=0.22, label=f"+/-{float(at):.1f} deg")
    ax.set_xlabel("Selected neighbor angle (deg)")
    ax.set_title("In-plane neighbor angle distribution")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def plot_adjacent_plane_environment(
    structure: Any,
    result: Any,
    *,
    mode: str = "both",
    projection: str = "side",
    annotate_separations: bool = True,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Visualize candidate and neighboring planes in side projection.

    The x-axis is projected coordinate ``u``; the y-axis is signed distance from
    the candidate plane along the detector normal. The plot labels the adjacent
    plane selected by atom distance and by plane spacing when those fields are
    present in the result.
    """
    if str(projection).lower() != "side":
        raise VisualizationError("only projection='side' is currently supported")
    if str(mode).lower() not in {"atom", "plane", "both"}:
        raise VisualizationError("mode must be 'atom', 'plane', or 'both'")
    ctx = _plane_context(structure, result)
    fig, ax = _make_ax(ax, figsize=(6.2, 4.2))
    cart = ctx["cart"]
    species = ctx["species"]
    styles = _style_map(species, species_style_map)
    e1 = ctx["basis"][0]
    normal = ctx["normal"]
    center = ctx["center"]

    def project(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = cart[ids]
        return (pts - center) @ e1, (pts - center) @ normal

    candidate = ctx["candidate_indices"]
    x, y = project(candidate)
    cand_sp = str(getattr(result, "species", species[candidate[0]] if len(candidate) else "candidate"))
    cand_style = styles.get(cand_sp, {"color": "0.4", "marker": "o"})
    ax.scatter(x, y, s=78, c=[cand_style.get("color", "0.5")], edgecolors="black", linewidths=1.3, label="candidate plane", zorder=4)
    ax.axhline(0.0, color="0.1", linewidth=1.0)

    for side, ids in ctx["adjacent"].items():
        if len(ids) == 0:
            continue
        side_x, side_y = project(ids)
        maj = _major_species(species[ids])
        style = styles.get(maj, {"color": "0.6", "marker": "s"})
        ax.scatter(side_x, side_y, s=46, c=[style.get("color", "0.6")], marker=style.get("marker", "s"), alpha=0.68, edgecolors="0.25", label=f"{side} plane ({maj})")
        sep = float(np.nanmedian(side_y))
        ax.axhline(sep, color="0.35", linestyle=":", linewidth=0.8)
        if annotate_separations:
            ax.annotate(f"{side}: {sep:+.2f} A", (0.02, sep), xycoords=("axes fraction", "data"), fontsize=8, va="bottom")

    viz = _viz(result)
    connections = getattr(viz, "adjacent_atom_connections", {}) if viz is not None else {}
    for side, conn in connections.items():
        if mode != "both" and side != getattr(result, f"closest_by_{mode}_side", None):
            continue
        start = np.asarray(conn.get("start_cartesian"), dtype=float)
        end = np.asarray(conn.get("end_cartesian"), dtype=float)
        sx, sy = float((start - center) @ e1), float((start - center) @ normal)
        ex, ey = float((end - center) @ e1), float((end - center) @ normal)
        ax.plot([sx, ex], [sy, ey], color="0.05", linewidth=1.1, linestyle="-", label=f"{side} closest atom")

    atom_side = getattr(result, "closest_by_atom_side", None)
    plane_side = getattr(result, "closest_by_plane_side", None)
    notes = []
    if mode in {"atom", "both"} and atom_side:
        notes.append(f"atom-distance neighbor: {atom_side}")
    if mode in {"plane", "both"} and plane_side:
        notes.append(f"plane-spacing neighbor: {plane_side}")
    if notes:
        ax.text(0.99, 0.98, "\n".join(notes), ha="right", va="top", transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Projected coordinate u (A)")
    ax.set_ylabel("Plane-normal distance (A)")
    ax.set_title("Adjacent-plane environment")
    ax.legend(loc="best", fontsize=10)
    return fig, ax


def _major_species(values: Sequence[Any]) -> str:
    names, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    if len(names) == 0:
        return ""
    return str(names[int(np.argmax(counts))])


def plot_coplanar_composition(result: Any, *, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """Summarize species sharing the candidate plane."""
    fig, ax = _make_ax(ax, figsize=(5.4, 1.8))
    counts = dict(getattr(result, "coplane_species_counts", {}) or {})
    if not counts:
        ax.text(0.5, 0.5, "No coplanar composition data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax
    total = sum(float(v) for v in counts.values())
    left = 0.0
    styles = build_species_style_map(counts.keys())
    candidate = str(getattr(result, "species", ""))
    for sp, count in sorted(counts.items()):
        width = float(count) / total if total else 0.0
        hatch = "" if sp == candidate else "//"
        ax.barh([0], [width], left=[left], color=styles[sp]["color"], edgecolor="0.15", hatch=hatch, label=f"{sp} ({count})")
        if width > 0.08:
            ax.text(left + width / 2, 0, sp, ha="center", va="center", fontsize=8)
        left += width
    major = max(counts.items(), key=lambda kv: kv[1])[0]
    frac = float(counts[major]) / total if total else float("nan")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of coplanar sites")
    ax.set_title(f"Coplanar composition: major {major} ({frac:.1%})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=max(1, min(4, len(counts))), fontsize=8)
    return fig, ax


def _summary_text(structure: Any, result: Any) -> str:
    status = "PASS" if _result_passes(result) else "FAIL"
    reasons = _failure_reasons(result)
    reason_text = ", ".join(reasons) if reasons else "none"
    material = _formula(structure) or str(getattr(result, "material_id", "material"))
    return (
        f"{material} | species={getattr(result, 'species', '?')} | axis={getattr(result, 'axis', '?')} "
        f"| plane={getattr(result, 'plane_id', '?')} | n={getattr(result, 'n_sites', '?')} | {status}\n"
        f"mean score={float(getattr(result, 'mean_score', np.nan)):.3g}, "
        f"pass fraction={float(getattr(result, 'pass_fraction', np.nan)):.3g} | failure reasons: {reason_text}"
    )

def plot_detection_summary(
    structure: Any,
    result: Any,
    *,
    config: Optional[Any] = None,
    representative_site: Any = "worst",
    figsize: Tuple[float, float] = (16.0, 12.0),
) -> Tuple[Figure, Dict[str, Axes]]:
    """Create a four-panel diagnostic figure for one candidate layer."""

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    fig.get_layout_engine().set(
        w_pad=0.08,
        h_pad=0.12,
        wspace=0.10,
        hspace=0.18,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.0),
        height_ratios=(1.0, 1.08),
    )
    axes: Dict[str, Axes] = {
        "plane_3d": fig.add_subplot(gs[0, 0], projection="3d"),
        "projected_layer": fig.add_subplot(gs[0, 1]),
        "site_geometry": fig.add_subplot(gs[1, 0]),
        "score_components": fig.add_subplot(gs[1, 1]),
    }

    plot_candidate_plane_3d(structure, result, ax=axes["plane_3d"])
    plot_projected_layer(structure, result, show_neighbors=True, annotate_distances=False, ax=axes["projected_layer"])

    try:
        if isinstance(representative_site, str):
            site = select_representative_site(result, representative_site)
        else:
            site = int(representative_site)
        plot_site_geometry(result, site, ax=axes["site_geometry"])
    except VisualizationError as exc:
        axes["site_geometry"].text(0.5, 0.5, str(exc), ha="center", va="center", transform=axes["site_geometry"].transAxes)
        axes["site_geometry"].set_axis_off()

    plot_score_components(result, ax=axes["score_components"])
    fig.suptitle(
        _summary_text(structure, result),
        fontsize=13,
        y=1.025,
    )    
    return fig, axes


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise VisualizationError(f"DataFrame is missing required columns: {missing}")


def _as_dataframe(table: Any) -> Any:
    try:
        import pandas as pd
    except Exception as exc:
        raise VisualizationError("pandas is required for table visualizations") from exc
    if isinstance(table, pd.DataFrame):
        return table
    return pd.DataFrame(table)


def plot_material_layer_summary(
    layer_table: Any,
    material_id: str,
    *,
    score_column: str = "mean_score",
    pass_column: str = "passes_final",
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Show all candidate layers for one material across axes and species."""
    df = _as_dataframe(layer_table)
    if "passes_final" not in df.columns:
        pass_column = "passes2" if "passes2" in df.columns else "passes"
    _require_columns(df, ["material_id", "axis", "species", "plane_center_frac", score_column, pass_column])
    d = df[df["material_id"].astype(str) == str(material_id)].copy()
    fig, ax = _make_ax(ax, figsize=(7.2, 4.0))
    if d.empty:
        ax.text(0.5, 0.5, f"No layers for {material_id}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax
    d["_row"] = d["axis"].astype(str) + " / " + d["species"].astype(str)
    rows = list(dict.fromkeys(d["_row"].tolist()))
    ymap = {r: i for i, r in enumerate(rows)}
    y = d["_row"].map(ymap).to_numpy(dtype=float)
    scores = d[score_column].astype(float).to_numpy()
    sizes = 40 + 18 * np.sqrt(np.maximum(d.get("n_sites", 1).astype(float).to_numpy(), 1.0))
    passed = d[pass_column].astype(bool).to_numpy()
    sc = ax.scatter(
        d["plane_center_frac"].astype(float),
        y,
        c=scores,
        s=sizes,
        cmap="viridis",
        marker="o",
        edgecolors=["black" if p else "0.35" for p in passed],
        linewidths=[1.4 if p else 0.8 for p in passed],
    )
    for xi, yi, p in zip(d["plane_center_frac"].astype(float), y, passed):
        if not p:
            ax.scatter([xi], [yi], marker="x", color="0.05", s=42, linewidths=1.2)
    if "is_dominant_layer" in d.columns:
        dom = d["is_dominant_layer"].astype(bool).to_numpy()
        ax.scatter(d.loc[dom, "plane_center_frac"].astype(float), y[dom], marker="*", s=sizes[dom] + 60, facecolors="none", edgecolors="black", linewidths=1.2, label="dominant")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Plane center along axis (fractional)")
    ax.set_ylabel("Axis / species")
    ax.set_title(f"Candidate layers in {material_id}")
    fig.colorbar(sc, ax=ax, label=score_column)
    return fig, ax


def plot_pass_fail_counts(materials_df: Any, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """Plot material-level pass/fail counts."""
    df = _as_dataframe(materials_df)
    col = "has_any_pass" if "has_any_pass" in df.columns else "dominant_has_pass"
    _require_columns(df, [col])
    passed = df[col].astype(bool)
    counts = [int((~passed).sum()), int(passed.sum())]
    fig, ax = _make_ax(ax, figsize=(4.4, 3.4))
    bars = ax.bar(["fail", "pass"], counts, color=["0.82", "0.55"], edgecolor="0.2")
    bars[0].set_hatch("//")
    ax.set_ylabel("Materials")
    ax.set_title("Material pass/fail counts")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    return fig, ax


def plot_score_distribution(layers_df: Any, group_by_pass: bool = True, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """Plot the distribution of candidate-layer mean scores."""
    df = _as_dataframe(layers_df)
    score_col = "mean_score" if "mean_score" in df.columns else "dominant_mean_score"
    _require_columns(df, [score_col])
    fig, ax = _make_ax(ax, figsize=(5.4, 3.6))
    if group_by_pass:
        pass_col = "passes2" if "passes2" in df.columns else ("passes" if "passes" in df.columns else None)
        if pass_col is not None:
            for value, label, hatch in [(False, "fail", "//"), (True, "pass", "")]:
                vals = df.loc[df[pass_col].astype(bool) == value, score_col].astype(float).dropna()
                if len(vals):
                    _, _, patches = ax.hist(vals, bins=12, alpha=0.55, edgecolor="0.2", label=label)
                    for patch in patches:
                        patch.set_hatch(hatch)
        else:
            ax.hist(df[score_col].astype(float).dropna(), bins=12, color="0.7", edgecolor="0.2")
    else:
        ax.hist(df[score_col].astype(float).dropna(), bins=12, color="0.7", edgecolor="0.2")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Candidate layers")
    ax.set_title("Score distribution")
    if group_by_pass:
        ax.legend(loc="best", fontsize=10)
    return fig, ax


def plot_candidates_per_material(materials_df: Any, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """Plot the distribution of candidate-layer counts per material."""
    df = _as_dataframe(materials_df)
    _require_columns(df, ["n_layers_total"])
    values = df["n_layers_total"].astype(float).dropna()
    fig, ax = _make_ax(ax, figsize=(5.2, 3.5))
    ax.hist(values, bins=min(20, max(1, int(values.max()) if len(values) else 1)), color="0.7", edgecolor="0.2")
    ax.set_xlabel("Candidate layers per material")
    ax.set_ylabel("Materials")
    ax.set_title("Candidate count distribution")
    return fig, ax


def plot_score_vs_environment(
    layers_df: Any,
    x: str = "nn_intra_min",
    y: str = "min_adj_dist_any_atom",
    color_by: str = "passes_final",
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Scatter detector score/environment quantities for layer-level tables."""
    df = _as_dataframe(layers_df)
    if color_by == "passes_final" and color_by not in df.columns:
        color_by = "passes2" if "passes2" in df.columns else "passes"
    _require_columns(df, [x, y, color_by])
    fig, ax = _make_ax(ax, figsize=(5.4, 4.1))
    passed = df[color_by].astype(bool)
    for value, label, marker, hatch_color in [(False, "fail", "x", "0.1"), (True, "pass", "o", "0.1")]:
        d = df[passed == value]
        if marker == "x":
            ax.scatter(d[x].astype(float), d[y].astype(float), marker=marker, c=hatch_color, label=label, alpha=0.8)
        else:
            ax.scatter(d[x].astype(float), d[y].astype(float), marker=marker, facecolors="none", edgecolors=hatch_color, label=label, alpha=0.8)
    ax.set_xlabel(f"{x} (A)" if "dist" in x or "nn" in x else x)
    ax.set_ylabel(f"{y} (A)" if "dist" in y or "nn" in y else y)
    ax.set_title("Score/environment relationship")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def plot_missingness(df: Any, columns: Optional[Sequence[str]] = None, ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """Plot missing-value fraction for selected table columns."""
    table = _as_dataframe(df)
    cols = list(columns) if columns is not None else list(table.columns)
    _require_columns(table, cols)
    missing = table[cols].isna().mean().sort_values(ascending=True)
    fig, ax = _make_ax(ax, figsize=(6.0, max(2.4, 0.25 * len(missing) + 1.2)))
    bars = ax.barh(missing.index.astype(str), missing.to_numpy(), color="0.72", edgecolor="0.2")
    for bar, val in zip(bars, missing.to_numpy()):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.0, max(0.1, float(missing.max()) + 0.12)))
    ax.set_xlabel("Missing fraction")
    ax.set_title("Table missingness")
    return fig, ax


def save_figure(
    fig: Figure,
    path: Any,
    *,
    dpi: int = 200,
    transparent: bool = False,
    close: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save a Matplotlib figure as PNG, SVG, PDF, or another supported format.

    Parent directories are created automatically. Existing files are not
    overwritten unless ``overwrite=True`` is passed.
    """
    out = Path(path).expanduser().resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass overwrite=True to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=transparent)
    if close:
        plt.close(fig)
    return out
