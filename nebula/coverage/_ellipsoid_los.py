"""
WGS84 ellipsoid line-of-sight utilities in ECEF.

All functions in this module are Numba-jitted and operate on ECEF meters.
`True` means line-of-sight is clear; `False` means blocked by the WGS84 ellipsoid.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from nebula.transform.constants import WGS84_A, WGS84_B

_INV_A2 = 1.0 / (WGS84_A * WGS84_A)
_INV_B2 = 1.0 / (WGS84_B * WGS84_B)
_INV_A = 1.0 / WGS84_A
_INV_B = 1.0 / WGS84_B
_POINT_EPS = 1e-12
_T_EPS = 1e-12


@njit(cache=True, inline="always")
def _ellipsoid_level(x: float, y: float, z: float) -> float:
    return x * x * _INV_A2 + y * y * _INV_A2 + z * z * _INV_B2


@njit(cache=True, inline="always")
def _los_clear_components(
    ox: float,
    oy: float,
    oz: float,
    tx: float,
    ty: float,
    tz: float,
) -> bool:
    # Inside ellipsoid means blocked by definition for this LoS test.
    o_level = _ellipsoid_level(ox, oy, oz)
    t_level = _ellipsoid_level(tx, ty, tz)
    if o_level < 1.0 - _POINT_EPS or t_level < 1.0 - _POINT_EPS:
        return False

    dx = tx - ox
    dy = ty - oy
    dz = tz - oz

    sx = ox * _INV_A
    sy = oy * _INV_A
    sz = oz * _INV_B
    dsx = dx * _INV_A
    dsy = dy * _INV_A
    dsz = dz * _INV_B

    a = dsx * dsx + dsy * dsy + dsz * dsz
    if a <= 1e-30:
        # Degenerate segment: clear when not inside.
        return o_level >= 1.0 - _POINT_EPS

    b = 2.0 * (sx * dsx + sy * dsy + sz * dsz)
    c = sx * sx + sy * sy + sz * sz - 1.0

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


@njit(cache=True, inline="always")
def los_clear_wgs84_ecef(observer_ecef_m: np.ndarray, target_ecef_m: np.ndarray) -> bool:
    """
    Check whether observer->target line-of-sight is clear of the WGS84 ellipsoid.

    Parameters
    ----------
    observer_ecef_m : np.ndarray
        Observer ECEF position with shape (3,), meters.
    target_ecef_m : np.ndarray
        Target ECEF position with shape (3,), meters.

    Returns
    -------
    bool
        `True` if clear, `False` if blocked by the Earth ellipsoid.
    """
    if observer_ecef_m.shape[0] != 3 or target_ecef_m.shape[0] != 3:
        raise ValueError("observer_ecef_m and target_ecef_m must have shape (3,)")
    return _los_clear_components(
        observer_ecef_m[0],
        observer_ecef_m[1],
        observer_ecef_m[2],
        target_ecef_m[0],
        target_ecef_m[1],
        target_ecef_m[2],
    )


@njit(cache=True, parallel=True)
def los_clear_wgs84_ecef_many_to_many(
    observers_ecef_m: np.ndarray, targets_ecef_m: np.ndarray
) -> np.ndarray:
    """
    Compute LoS-clear matrix for many observers to many targets.

    Parameters
    ----------
    observers_ecef_m : np.ndarray
        Observer ECEF positions, shape (N, 3), meters.
    targets_ecef_m : np.ndarray
        Target ECEF positions, shape (M, 3), meters.

    Returns
    -------
    np.ndarray
        Boolean matrix with shape (N, M). Entry [i, j] is `True` when observer i
        has clear line-of-sight to target j.
    """
    if observers_ecef_m.ndim != 2 or observers_ecef_m.shape[1] != 3:
        raise ValueError("observers_ecef_m must have shape (N, 3)")
    if targets_ecef_m.ndim != 2 or targets_ecef_m.shape[1] != 3:
        raise ValueError("targets_ecef_m must have shape (M, 3)")

    n = observers_ecef_m.shape[0]
    m = targets_ecef_m.shape[0]
    out = np.empty((n, m), dtype=np.bool_)

    for i in prange(n):
        ox = observers_ecef_m[i, 0]
        oy = observers_ecef_m[i, 1]
        oz = observers_ecef_m[i, 2]
        for j in range(m):
            out[i, j] = _los_clear_components(
                ox,
                oy,
                oz,
                targets_ecef_m[j, 0],
                targets_ecef_m[j, 1],
                targets_ecef_m[j, 2],
            )
    return out


@njit(cache=True, parallel=True)
def los_clear_wgs84_ecef_one_to_many(
    observer_ecef_m: np.ndarray, targets_ecef_m: np.ndarray
) -> np.ndarray:
    """
    Compute LoS-clear values from one observer to many targets.

    Parameters
    ----------
    observer_ecef_m : np.ndarray
        Observer ECEF position, shape (3,), meters.
    targets_ecef_m : np.ndarray
        Target ECEF positions, shape (M, 3), meters.

    Returns
    -------
    np.ndarray
        Boolean array with shape (M,). Entry [j] is `True` when observer has clear
        line-of-sight to target j.
    """
    if observer_ecef_m.shape[0] != 3:
        raise ValueError("observer_ecef_m must have shape (3,)")
    if targets_ecef_m.ndim != 2 or targets_ecef_m.shape[1] != 3:
        raise ValueError("targets_ecef_m must have shape (M, 3)")

    m = targets_ecef_m.shape[0]
    out = np.empty(m, dtype=np.bool_)

    ox = observer_ecef_m[0]
    oy = observer_ecef_m[1]
    oz = observer_ecef_m[2]
    for j in prange(m):
        out[j] = _los_clear_components(
            ox,
            oy,
            oz,
            targets_ecef_m[j, 0],
            targets_ecef_m[j, 1],
            targets_ecef_m[j, 2],
        )
    return out


__all__ = [
    "los_clear_wgs84_ecef",
    "los_clear_wgs84_ecef_many_to_many",
    "los_clear_wgs84_ecef_one_to_many",
]
