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


def _make_ax(ax: Optional[Axes], *, projection: Optional[str] = None, figsize: Tuple[float, float] = (6.0, 4.5)) -> Tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure(figsize=figsize)
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
    ax.legend(loc="best", fontsize=8)
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
        + f"\nmean score={float(getattr(result, 'mean_score', np.nan)):.3g}, pass fraction={float(getattr(result, 'pass_fraction', np.nan)):.3g}"
    )
    ax.legend(loc="best", fontsize=8)
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
    annotate_species: bool = False,
    species_style_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Isolate the candidate plane and nearby atomic environment in 3D.

    The plot shows candidate atoms, optional previous/next adjacent planes, a
    translucent candidate-plane surface, and the plane normal. Distances are in
    Cartesian Angstrom coordinates.
    """
    ctx = _plane_context(structure, result)
    fig, ax = _make_ax(ax, projection="3d", figsize=(6.2, 5.0))
    cart = ctx["cart"]
    species = ctx["species"]
    styles = _style_map(species, species_style_map)

    candidate_set = set(ctx["candidate_indices"].tolist())
    adjacent_indices = np.concatenate(list(ctx["adjacent"].values())) if include_adjacent_planes and ctx["adjacent"] else np.array([], dtype=int)
    adjacent_set = set(adjacent_indices.tolist())

    signed = (cart - ctx["center"]) @ ctx["normal"]
    if plane_window_angstrom is None:
        keep = np.ones(len(cart), dtype=bool)
    else:
        keep = np.abs(signed) <= float(plane_window_angstrom)
        keep[list(candidate_set | adjacent_set)] = True

    _scatter_species_3d(
        ax,
        cart[keep],
        species[keep],
        styles,
        indices=np.arange(len(cart), dtype=int)[keep],
        candidate_set=candidate_set,
        adjacent_set=adjacent_set,
        show_labels=annotate_species,
    )

    scale = max(float(np.linalg.norm(np.asarray(structure.lattice.matrix)[i])) for i in range(3)) * 0.42
    _plot_plane_surface(ax, ctx["center"], ctx["basis"][0], ctx["basis"][1], scale)
    c = ctx["center"]
    n = ctx["normal"]
    ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2], length=scale * 0.55, color="black", linewidth=1.3)
    ax.scatter([c[0]], [c[1]], [c[2]], marker="+", s=70, color="black", label="plane center")

    if include_adjacent_planes:
        for side, ids in ctx["adjacent"].items():
            if len(ids) == 0:
                continue
            sep = float(np.mean((cart[ids] - c) @ n))
            plane_center = c + sep * n
            ax.text(plane_center[0], plane_center[1], plane_center[2], f"{side} {sep:+.2f} A", fontsize=8)

    _set_3d_equal(ax, cart[keep])
    _apply_axis_view(ax, "auto", axis=ctx["axis"])
    ax.set_xlabel("Cartesian x (A)")
    ax.set_ylabel("Cartesian y (A)")
    ax.set_zlabel("Cartesian z (A)")
    ax.set_title(f"{_formula(structure)} {getattr(result, 'species', '')} candidate plane".strip())
    ax.legend(loc="best", fontsize=8)
    return fig, ax


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
    include_thresholds: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Plot normalized pass margins for the detector decision.

    Normalized pass margin is defined so values greater than or equal to 1 meet
    that criterion. For error criteria, the margin is ``threshold / measured``;
    for score/fraction criteria, it is ``measured / threshold``.
    """
    fig, ax = _make_ax(ax, figsize=(6.2, 4.4))
    thresholds = _thresholds(result)
    rows: List[Tuple[str, float, bool, str]] = []

    pf = _finite_float(getattr(result, "pass_fraction", np.nan))
    mpf = thresholds.get("min_pass_fraction", None)
    if pf is not None and mpf:
        margin = pf / float(mpf)
        rows.append(("site pass fraction", margin, margin >= 1.0, f"{pf:.3g} / {float(mpf):.3g}"))

    ms = _finite_float(getattr(result, "mean_score", np.nan))
    st = thresholds.get("score_threshold", None)
    if ms is not None and st:
        margin = ms / float(st)
        rows.append(("mean local score", margin, margin >= 1.0, f"{ms:.3g} / {float(st):.3g}"))

    le = _finite_float(getattr(result, "uv_len_err_mean", np.nan))
    lt = thresholds.get("len_tol", None)
    if le is not None and lt:
        margin = float("inf") if le <= 1e-12 else float(lt) / le
        rows.append(("length error", margin, margin >= 1.0, f"{le:.3g} <= {float(lt):.3g}"))

    ae = _finite_float(getattr(result, "uv_ang_err_mean", np.nan))
    at = thresholds.get("ang_tol_deg", None)
    if ae is not None and at:
        margin = float("inf") if ae <= 1e-12 else float(at) / ae
        rows.append(("angle error", margin, margin >= 1.0, f"{ae:.3g} <= {float(at):.3g} deg"))

    oop = bool(getattr(result, "has_out_of_plane_same_species_bond", False))
    if hasattr(result, "has_out_of_plane_same_species_bond"):
        rows.append(("no same-species out-of-plane bond", 0.0 if oop else 1.2, not oop, "yes" if not oop else "no"))

    mixed = bool(getattr(result, "has_coplane_other_species", False))
    if hasattr(result, "has_coplane_other_species"):
        rows.append(("coplanar composition rule", 0.0 if mixed else 1.2, not mixed, "pure" if not mixed else "mixed"))

    if not rows:
        ax.text(0.5, 0.5, "No score component diagnostics available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    labels = [r[0] for r in rows]
    margins = np.array([min(r[1], 2.5) if np.isfinite(r[1]) else 2.5 for r in rows], dtype=float)
    passed = [r[2] for r in rows]
    y = np.arange(len(rows), dtype=float)
    bars = ax.barh(y, margins, color=["0.62" if ok else "0.88" for ok in passed], edgecolor="0.15", linewidth=0.9)
    for bar, ok in zip(bars, passed):
        if not ok:
            bar.set_hatch("//")
    for yi, margin, ok, label in zip(y, margins, passed, [r[3] for r in rows]):
        marker = "o" if ok else "x"
        ax.scatter([min(margin, 2.5)], [yi], marker=marker, color="0.05", zorder=4)
        ax.text(2.56, yi, ("PASS" if ok else "FAIL") + f"  {label}", va="center", fontsize=8)

    if include_thresholds:
        ax.axvline(1.0, color="0.1", linestyle="--", linewidth=1.0, label="pass boundary")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 3.35)
    ax.set_xlabel("Normalized pass margin (>=1 passes)")
    ax.set_title("Detector score components")
    ax.invert_yaxis()
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
    ax.legend(loc="best", fontsize=8)
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
    ax.legend(loc="best", fontsize=8)
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
) -> Tuple[Figure, Dict[str, Axes]]:
    """Create a four-panel diagnostic figure for one candidate layer.

    Panels show the 3D plane context, projected detector layer, representative
    site geometry, and normalized score components. This is intended as the main
    figure for positive and negative examples in notebooks.
    """
    fig = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
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
    fig.suptitle(_summary_text(structure, result), fontsize=11)
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
        ax.legend(loc="best", fontsize=8)
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
