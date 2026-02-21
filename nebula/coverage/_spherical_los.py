"""
Spherical line-of-sight utilities in Cartesian coordinates.

All functions are Numba-jitted and assume all inputs use a consistent coordinate
system and unit scale. The blocking sphere center is user-defined and defaults
to the origin.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_POINT_EPS = 1e-12
_T_EPS = 1e-12


@njit(cache=True, inline="always")
def _los_clear_components_sphere(
    ox: float,
    oy: float,
    oz: float,
    tx: float,
    ty: float,
    tz: float,
    radius_sq: float,
) -> bool:
    onorm2 = ox * ox + oy * oy + oz * oz
    tnorm2 = tx * tx + ty * ty + tz * tz

    # Inside sphere means blocked by definition.
    if onorm2 < radius_sq - _POINT_EPS or tnorm2 < radius_sq - _POINT_EPS:
        return False

    dx = tx - ox
    dy = ty - oy
    dz = tz - oz

    a = dx * dx + dy * dy + dz * dz
    if a <= 1e-30:
        return onorm2 >= radius_sq - _POINT_EPS

    b = 2.0 * (ox * dx + oy * dy + oz * dz)
    c = onorm2 - radius_sq

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return True

    sqrt_disc = math.sqrt(disc)
    inv_2a = 0.5 / a
    t1 = (-b - sqrt_disc) * inv_2a
    t2 = (-b + sqrt_disc) * inv_2a

    # Intersection strictly between endpoints blocks LoS.
    if (_T_EPS < t1 < 1.0 - _T_EPS) or (_T_EPS < t2 < 1.0 - _T_EPS):
        return False
    return True


@njit(cache=True, parallel=True)
def _shift_points(points: np.ndarray, cx: float, cy: float, cz: float) -> np.ndarray:
    n = points.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        out[i, 0] = points[i, 0] - cx
        out[i, 1] = points[i, 1] - cy
        out[i, 2] = points[i, 2] - cz
    return out


@njit(cache=True, parallel=True)
def _los_sphere_many_to_many_origin(
    observers_pos: np.ndarray, targets_pos: np.ndarray, radius_sq: float
) -> np.ndarray:
    n = observers_pos.shape[0]
    m = targets_pos.shape[0]
    out = np.empty((n, m), dtype=np.bool_)
    for i in prange(n):
        ox = observers_pos[i, 0]
        oy = observers_pos[i, 1]
        oz = observers_pos[i, 2]
        for j in range(m):
            out[i, j] = _los_clear_components_sphere(
                ox, oy, oz, targets_pos[j, 0], targets_pos[j, 1], targets_pos[j, 2], radius_sq
            )
    return out


@njit(cache=True, parallel=True)
def _los_sphere_many_to_many_offset(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
    cx: float,
    cy: float,
    cz: float,
) -> np.ndarray:
    n = observers_pos.shape[0]
    targets_shifted = _shift_points(targets_pos, cx, cy, cz)
    m = targets_shifted.shape[0]
    out = np.empty((n, m), dtype=np.bool_)
    for i in prange(n):
        ox = observers_pos[i, 0] - cx
        oy = observers_pos[i, 1] - cy
        oz = observers_pos[i, 2] - cz
        for j in range(m):
            out[i, j] = _los_clear_components_sphere(
                ox,
                oy,
                oz,
                targets_shifted[j, 0],
                targets_shifted[j, 1],
                targets_shifted[j, 2],
                radius_sq,
            )
    return out


@njit(cache=True, parallel=True)
def _los_sphere_one_to_many_origin(
    observer_pos: np.ndarray, targets_pos: np.ndarray, radius_sq: float
) -> np.ndarray:
    m = targets_pos.shape[0]
    out = np.empty(m, dtype=np.bool_)
    ox = observer_pos[0]
    oy = observer_pos[1]
    oz = observer_pos[2]
    for j in prange(m):
        out[j] = _los_clear_components_sphere(
            ox, oy, oz, targets_pos[j, 0], targets_pos[j, 1], targets_pos[j, 2], radius_sq
        )
    return out


@njit(cache=True, parallel=True)
def _los_sphere_one_to_many_offset(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
    cx: float,
    cy: float,
    cz: float,
) -> np.ndarray:
    targets_shifted = _shift_points(targets_pos, cx, cy, cz)
    m = targets_shifted.shape[0]
    out = np.empty(m, dtype=np.bool_)
    ox = observer_pos[0] - cx
    oy = observer_pos[1] - cy
    oz = observer_pos[2] - cz
    for j in prange(m):
        out[j] = _los_clear_components_sphere(
            ox, oy, oz, targets_shifted[j, 0], targets_shifted[j, 1], targets_shifted[j, 2], radius_sq
        )
    return out


@njit(cache=True, inline="always")
def los_clear_sphere(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float = 0.0,
    sphere_center_y: float = 0.0,
    sphere_center_z: float = 0.0,
) -> bool:
    """
    Check whether observer->target line-of-sight is clear of a blocking sphere.
    """
    if observer_pos.shape[0] != 3 or target_pos.shape[0] != 3:
        raise ValueError("observer_pos and target_pos must have shape (3,)")
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be > 0")

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_clear_components_sphere(
            observer_pos[0], observer_pos[1], observer_pos[2], target_pos[0], target_pos[1], target_pos[2], radius_sq
        )
    return _los_clear_components_sphere(
        observer_pos[0] - sphere_center_x,
        observer_pos[1] - sphere_center_y,
        observer_pos[2] - sphere_center_z,
        target_pos[0] - sphere_center_x,
        target_pos[1] - sphere_center_y,
        target_pos[2] - sphere_center_z,
        radius_sq,
    )


@njit(cache=True)
def los_clear_sphere_many_to_many(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float = 0.0,
    sphere_center_y: float = 0.0,
    sphere_center_z: float = 0.0,
) -> np.ndarray:
    """
    Compute LoS-clear matrix for many observers to many targets.
    """
    if observers_pos.ndim != 2 or observers_pos.shape[1] != 3:
        raise ValueError("observers_pos must have shape (N, 3)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be > 0")

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_sphere_many_to_many_origin(observers_pos, targets_pos, radius_sq)
    return _los_sphere_many_to_many_offset(
        observers_pos, targets_pos, radius_sq, sphere_center_x, sphere_center_y, sphere_center_z
    )


@njit(cache=True)
def los_clear_sphere_one_to_many(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float = 0.0,
    sphere_center_y: float = 0.0,
    sphere_center_z: float = 0.0,
) -> np.ndarray:
    """
    Compute LoS-clear values from one observer to many targets.
    """
    if observer_pos.shape[0] != 3:
        raise ValueError("observer_pos must have shape (3,)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be > 0")

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_sphere_one_to_many_origin(observer_pos, targets_pos, radius_sq)
    return _los_sphere_one_to_many_offset(
        observer_pos, targets_pos, radius_sq, sphere_center_x, sphere_center_y, sphere_center_z
    )


# Backward-compatible names.
los_clear_sphere_ecef = los_clear_sphere
los_clear_sphere_ecef_many_to_many = los_clear_sphere_many_to_many
los_clear_sphere_ecef_one_to_many = los_clear_sphere_one_to_many


__all__ = [
    "los_clear_sphere",
    "los_clear_sphere_many_to_many",
    "los_clear_sphere_one_to_many",
    "los_clear_sphere_ecef",
    "los_clear_sphere_ecef_many_to_many",
    "los_clear_sphere_ecef_one_to_many",
]
