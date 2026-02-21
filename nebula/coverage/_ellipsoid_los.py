"""
General ellipsoid and WGS84 line-of-sight utilities.

All functions are Numba-jitted and assume a consistent coordinate system and
unit scale across positions, ellipsoid axes, and center values.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from nebula.transform.constants import WGS84_A, WGS84_B

_POINT_EPS = 1e-12
_T_EPS = 1e-12

_WGS84_INV_A2 = 1.0 / (WGS84_A * WGS84_A)
_WGS84_INV_B2 = 1.0 / (WGS84_B * WGS84_B)


@njit(cache=True, inline="always")
def _los_clear_components_ellipsoid_axis_aligned(
    ox: float,
    oy: float,
    oz: float,
    tx: float,
    ty: float,
    tz: float,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
) -> bool:
    o_level = ox * ox * inv_a2 + oy * oy * inv_b2 + oz * oz * inv_c2
    t_level = tx * tx * inv_a2 + ty * ty * inv_b2 + tz * tz * inv_c2

    # Inside ellipsoid means blocked by definition.
    if o_level < 1.0 - _POINT_EPS or t_level < 1.0 - _POINT_EPS:
        return False

    dx = tx - ox
    dy = ty - oy
    dz = tz - oz

    a = dx * dx * inv_a2 + dy * dy * inv_b2 + dz * dz * inv_c2
    if a <= 1e-30:
        return o_level >= 1.0 - _POINT_EPS

    b = 2.0 * (ox * dx * inv_a2 + oy * dy * inv_b2 + oz * dz * inv_c2)
    c = o_level - 1.0

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
def _to_body_frame(
    px: float,
    py: float,
    pz: float,
    center_x: float,
    center_y: float,
    center_z: float,
    orientation_ellipsoid_to_frame: np.ndarray,
):
    """
    Convert frame coordinates to ellipsoid body coordinates.

    `orientation_ellipsoid_to_frame` maps body axes into frame axes, so this
    applies its transpose to move frame->body.
    """
    dx = px - center_x
    dy = py - center_y
    dz = pz - center_z

    bx = (
        orientation_ellipsoid_to_frame[0, 0] * dx
        + orientation_ellipsoid_to_frame[1, 0] * dy
        + orientation_ellipsoid_to_frame[2, 0] * dz
    )
    by = (
        orientation_ellipsoid_to_frame[0, 1] * dx
        + orientation_ellipsoid_to_frame[1, 1] * dy
        + orientation_ellipsoid_to_frame[2, 1] * dz
    )
    bz = (
        orientation_ellipsoid_to_frame[0, 2] * dx
        + orientation_ellipsoid_to_frame[1, 2] * dy
        + orientation_ellipsoid_to_frame[2, 2] * dz
    )
    return bx, by, bz


@njit(cache=True, inline="always")
def _is_identity_orientation(orientation_ellipsoid_to_frame: np.ndarray) -> bool:
    return (
        orientation_ellipsoid_to_frame[0, 0] == 1.0
        and orientation_ellipsoid_to_frame[0, 1] == 0.0
        and orientation_ellipsoid_to_frame[0, 2] == 0.0
        and orientation_ellipsoid_to_frame[1, 0] == 0.0
        and orientation_ellipsoid_to_frame[1, 1] == 1.0
        and orientation_ellipsoid_to_frame[1, 2] == 0.0
        and orientation_ellipsoid_to_frame[2, 0] == 0.0
        and orientation_ellipsoid_to_frame[2, 1] == 0.0
        and orientation_ellipsoid_to_frame[2, 2] == 1.0
    )


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
def _transform_points_to_body(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
    orientation_ellipsoid_to_frame: np.ndarray,
) -> np.ndarray:
    n = points.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        bx, by, bz = _to_body_frame(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            center_x,
            center_y,
            center_z,
            orientation_ellipsoid_to_frame,
        )
        out[i, 0] = bx
        out[i, 1] = by
        out[i, 2] = bz
    return out


@njit(cache=True, parallel=True)
def _los_many_to_many_body(
    observers_body: np.ndarray,
    targets_body: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
) -> np.ndarray:
    n = observers_body.shape[0]
    m = targets_body.shape[0]
    out = np.empty((n, m), dtype=np.bool_)
    for i in prange(n):
        ox = observers_body[i, 0]
        oy = observers_body[i, 1]
        oz = observers_body[i, 2]
        for j in range(m):
            out[i, j] = _los_clear_components_ellipsoid_axis_aligned(
                ox, oy, oz, targets_body[j, 0], targets_body[j, 1], targets_body[j, 2], inv_a2, inv_b2, inv_c2
            )
    return out


@njit(cache=True, parallel=True)
def _los_one_to_many_body(
    ox: float,
    oy: float,
    oz: float,
    targets_body: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
) -> np.ndarray:
    m = targets_body.shape[0]
    out = np.empty(m, dtype=np.bool_)
    for j in prange(m):
        out[j] = _los_clear_components_ellipsoid_axis_aligned(
            ox, oy, oz, targets_body[j, 0], targets_body[j, 1], targets_body[j, 2], inv_a2, inv_b2, inv_c2
        )
    return out


@njit(cache=True, parallel=True)
def _los_many_to_many_offset(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    targets_shifted = _shift_points(targets_pos, center_x, center_y, center_z)
    n = observers_pos.shape[0]
    m = targets_shifted.shape[0]
    out = np.empty((n, m), dtype=np.bool_)
    for i in prange(n):
        ox = observers_pos[i, 0] - center_x
        oy = observers_pos[i, 1] - center_y
        oz = observers_pos[i, 2] - center_z
        for j in range(m):
            out[i, j] = _los_clear_components_ellipsoid_axis_aligned(
                ox,
                oy,
                oz,
                targets_shifted[j, 0],
                targets_shifted[j, 1],
                targets_shifted[j, 2],
                inv_a2,
                inv_b2,
                inv_c2,
            )
    return out


@njit(cache=True)
def _los_one_to_many_offset(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    targets_shifted = _shift_points(targets_pos, center_x, center_y, center_z)
    ox = observer_pos[0] - center_x
    oy = observer_pos[1] - center_y
    oz = observer_pos[2] - center_z
    return _los_one_to_many_body(ox, oy, oz, targets_shifted, inv_a2, inv_b2, inv_c2)


@njit(cache=True, inline="always")
def _validate_axes(semi_axis_a: float, semi_axis_b: float, semi_axis_c: float) -> None:
    if semi_axis_a <= 0.0 or semi_axis_b <= 0.0 or semi_axis_c <= 0.0:
        raise ValueError("semi-axis values must be > 0")


@njit(cache=True, inline="always")
def los_clear_ellipsoid(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> bool:
    """
    Axis-aligned ellipsoid line-of-sight check.
    """
    if observer_pos.shape[0] != 3 or target_pos.shape[0] != 3:
        raise ValueError("observer_pos and target_pos must have shape (3,)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_clear_components_ellipsoid_axis_aligned(
            observer_pos[0], observer_pos[1], observer_pos[2], target_pos[0], target_pos[1], target_pos[2], inv_a2, inv_b2, inv_c2
        )
    return _los_clear_components_ellipsoid_axis_aligned(
        observer_pos[0] - center_x,
        observer_pos[1] - center_y,
        observer_pos[2] - center_z,
        target_pos[0] - center_x,
        target_pos[1] - center_y,
        target_pos[2] - center_z,
        inv_a2,
        inv_b2,
        inv_c2,
    )


@njit(cache=True)
def los_clear_ellipsoid_many_to_many(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> np.ndarray:
    """
    Axis-aligned ellipsoid LoS for many observers to many targets.
    """
    if observers_pos.ndim != 2 or observers_pos.shape[1] != 3:
        raise ValueError("observers_pos must have shape (N, 3)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_many_to_many_body(observers_pos, targets_pos, inv_a2, inv_b2, inv_c2)
    return _los_many_to_many_offset(
        observers_pos, targets_pos, inv_a2, inv_b2, inv_c2, center_x, center_y, center_z
    )


@njit(cache=True)
def los_clear_ellipsoid_one_to_many(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> np.ndarray:
    """
    Axis-aligned ellipsoid LoS for one observer to many targets.
    """
    if observer_pos.shape[0] != 3:
        raise ValueError("observer_pos must have shape (3,)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_one_to_many_body(
            observer_pos[0], observer_pos[1], observer_pos[2], targets_pos, inv_a2, inv_b2, inv_c2
        )
    return _los_one_to_many_offset(
        observer_pos, targets_pos, inv_a2, inv_b2, inv_c2, center_x, center_y, center_z
    )


@njit(cache=True, inline="always")
def los_clear_ellipsoid_oriented(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> bool:
    """
    Oriented ellipsoid line-of-sight check.

    `orientation_ellipsoid_to_frame` is a (3,3) rotation mapping ellipsoid body
    axes into frame axes.
    """
    if (
        orientation_ellipsoid_to_frame.ndim != 2
        or orientation_ellipsoid_to_frame.shape[0] != 3
        or orientation_ellipsoid_to_frame.shape[1] != 3
    ):
        raise ValueError("orientation_ellipsoid_to_frame must have shape (3, 3)")

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return los_clear_ellipsoid(
            observer_pos, target_pos, semi_axis_a, semi_axis_b, semi_axis_c, center_x, center_y, center_z
        )

    if observer_pos.shape[0] != 3 or target_pos.shape[0] != 3:
        raise ValueError("observer_pos and target_pos must have shape (3,)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    obx, oby, obz = _to_body_frame(
        observer_pos[0],
        observer_pos[1],
        observer_pos[2],
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
    )
    tbx, tby, tbz = _to_body_frame(
        target_pos[0],
        target_pos[1],
        target_pos[2],
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
    )
    return _los_clear_components_ellipsoid_axis_aligned(
        obx, oby, obz, tbx, tby, tbz, inv_a2, inv_b2, inv_c2
    )


@njit(cache=True, parallel=True)
def los_clear_ellipsoid_many_to_many_oriented(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> np.ndarray:
    """
    Oriented ellipsoid LoS for many observers to many targets.
    """
    if (
        orientation_ellipsoid_to_frame.ndim != 2
        or orientation_ellipsoid_to_frame.shape[0] != 3
        or orientation_ellipsoid_to_frame.shape[1] != 3
    ):
        raise ValueError("orientation_ellipsoid_to_frame must have shape (3, 3)")

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return los_clear_ellipsoid_many_to_many(
            observers_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c, center_x, center_y, center_z
        )

    if observers_pos.ndim != 2 or observers_pos.shape[1] != 3:
        raise ValueError("observers_pos must have shape (N, 3)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    targets_body = _transform_points_to_body(
        targets_pos, center_x, center_y, center_z, orientation_ellipsoid_to_frame
    )
    n = observers_pos.shape[0]
    m = targets_body.shape[0]
    out = np.empty((n, m), dtype=np.bool_)

    for i in prange(n):
        obx, oby, obz = _to_body_frame(
            observers_pos[i, 0],
            observers_pos[i, 1],
            observers_pos[i, 2],
            center_x,
            center_y,
            center_z,
            orientation_ellipsoid_to_frame,
        )
        for j in range(m):
            out[i, j] = _los_clear_components_ellipsoid_axis_aligned(
                obx, oby, obz, targets_body[j, 0], targets_body[j, 1], targets_body[j, 2], inv_a2, inv_b2, inv_c2
            )
    return out


@njit(cache=True)
def los_clear_ellipsoid_one_to_many_oriented(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> np.ndarray:
    """
    Oriented ellipsoid LoS for one observer to many targets.
    """
    if (
        orientation_ellipsoid_to_frame.ndim != 2
        or orientation_ellipsoid_to_frame.shape[0] != 3
        or orientation_ellipsoid_to_frame.shape[1] != 3
    ):
        raise ValueError("orientation_ellipsoid_to_frame must have shape (3, 3)")

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return los_clear_ellipsoid_one_to_many(
            observer_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c, center_x, center_y, center_z
        )

    if observer_pos.shape[0] != 3:
        raise ValueError("observer_pos must have shape (3,)")
    if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
        raise ValueError("targets_pos must have shape (M, 3)")
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)

    inv_a2 = 1.0 / (semi_axis_a * semi_axis_a)
    inv_b2 = 1.0 / (semi_axis_b * semi_axis_b)
    inv_c2 = 1.0 / (semi_axis_c * semi_axis_c)

    targets_body = _transform_points_to_body(
        targets_pos, center_x, center_y, center_z, orientation_ellipsoid_to_frame
    )
    obx, oby, obz = _to_body_frame(
        observer_pos[0],
        observer_pos[1],
        observer_pos[2],
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
    )
    return _los_one_to_many_body(obx, oby, obz, targets_body, inv_a2, inv_b2, inv_c2)


@njit(cache=True, inline="always")
def los_clear_wgs84_ecef(observer_ecef_m: np.ndarray, target_ecef_m: np.ndarray) -> bool:
    """
    Optimized WGS84-ECEF LoS helper (axis-aligned, origin-centered).
    """
    if observer_ecef_m.shape[0] != 3 or target_ecef_m.shape[0] != 3:
        raise ValueError("observer_ecef_m and target_ecef_m must have shape (3,)")
    return _los_clear_components_ellipsoid_axis_aligned(
        observer_ecef_m[0],
        observer_ecef_m[1],
        observer_ecef_m[2],
        target_ecef_m[0],
        target_ecef_m[1],
        target_ecef_m[2],
        _WGS84_INV_A2,
        _WGS84_INV_A2,
        _WGS84_INV_B2,
    )


@njit(cache=True)
def los_clear_wgs84_ecef_many_to_many(
    observers_ecef_m: np.ndarray, targets_ecef_m: np.ndarray
) -> np.ndarray:
    """
    Optimized WGS84-ECEF LoS helper for many observers to many targets.
    """
    if observers_ecef_m.ndim != 2 or observers_ecef_m.shape[1] != 3:
        raise ValueError("observers_ecef_m must have shape (N, 3)")
    if targets_ecef_m.ndim != 2 or targets_ecef_m.shape[1] != 3:
        raise ValueError("targets_ecef_m must have shape (M, 3)")
    return _los_many_to_many_body(
        observers_ecef_m,
        targets_ecef_m,
        _WGS84_INV_A2,
        _WGS84_INV_A2,
        _WGS84_INV_B2,
    )


@njit(cache=True)
def los_clear_wgs84_ecef_one_to_many(
    observer_ecef_m: np.ndarray, targets_ecef_m: np.ndarray
) -> np.ndarray:
    """
    Optimized WGS84-ECEF LoS helper for one observer to many targets.
    """
    if observer_ecef_m.shape[0] != 3:
        raise ValueError("observer_ecef_m must have shape (3,)")
    if targets_ecef_m.ndim != 2 or targets_ecef_m.shape[1] != 3:
        raise ValueError("targets_ecef_m must have shape (M, 3)")
    return _los_one_to_many_body(
        observer_ecef_m[0],
        observer_ecef_m[1],
        observer_ecef_m[2],
        targets_ecef_m,
        _WGS84_INV_A2,
        _WGS84_INV_A2,
        _WGS84_INV_B2,
    )


__all__ = [
    "los_clear_ellipsoid",
    "los_clear_ellipsoid_many_to_many",
    "los_clear_ellipsoid_one_to_many",
    "los_clear_ellipsoid_oriented",
    "los_clear_ellipsoid_many_to_many_oriented",
    "los_clear_ellipsoid_one_to_many_oriented",
    "los_clear_wgs84_ecef",
    "los_clear_wgs84_ecef_many_to_many",
    "los_clear_wgs84_ecef_one_to_many",
]
