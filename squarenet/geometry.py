from __future__ import annotations

import numpy as np
from typing import Tuple, List


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def plane_frame_from_axis(lattice_matrix: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an orthonormal in-plane frame (e1,e2) and unit normal n_hat for an axis-parallel plane."""
    a_vec, b_vec, c_vec = lattice_matrix
    if axis == "c":
        u_vec, v_vec = a_vec, b_vec
    elif axis == "b":
        u_vec, v_vec = a_vec, c_vec
    elif axis == "a":
        u_vec, v_vec = b_vec, c_vec
    else:
        raise ValueError("axis must be one of 'a','b','c'")

    n_hat = unit(np.cross(u_vec, v_vec))
    e1 = unit(u_vec)
    e2 = unit(np.cross(n_hat, e1))
    return e1, e2, n_hat


def project_plane_to_2d(points_cart: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    origin = points_cart.mean(axis=0)
    X = points_cart - origin
    pts2d = np.c_[X @ e1, X @ e2]
    return pts2d, origin


def group_points_into_planes(points_cart: np.ndarray, n_hat: np.ndarray, plane_tol_A: float):
    """Group points into planes by binning distances along a provided normal direction."""
    d = points_cart @ n_hat
    bins = np.round(d / plane_tol_A).astype(int)

    planes: List[np.ndarray] = []
    centers: List[float] = []
    bin_ids: List[int] = []

    for b in np.sort(np.unique(bins)):
        mask = (bins == b)
        pl = points_cart[mask]
        planes.append(pl)
        centers.append(float(np.mean(d[mask])))
        bin_ids.append(int(b))

    return planes, centers, bin_ids
