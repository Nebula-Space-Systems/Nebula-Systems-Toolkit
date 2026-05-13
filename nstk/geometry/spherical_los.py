"""
Unified spherical line-of-sight utilities in Cartesian coordinates.

The public ``los_clear_sphere`` interface accepts either one point with shape
``(3,)`` or a stack of points with shape ``(N, 3)`` for each endpoint input.
It returns a scalar, a 1D boolean array, or a 2D boolean matrix based on the
input combination, and the same forms work inside ``@numba.njit`` callers.
Pairwise row-by-row comparisons are available via
``los_clear_sphere_pairwise``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from numba.extending import overload as numba_overload

from nstk.transforms._api_utils import (
    is_numba_absent,
    is_numba_array1d,
    is_numba_array2d,
    is_numba_scalar,
)

_POINT_EPS = 1e-12
_T_EPS = 1e-12


def _as_point_input(value: np.ndarray, name: str) -> np.ndarray:
    """Convert a Python value to a point or point-stack array."""

    arr = np.asarray(value)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(f"{name} must have shape (3,) or (N, 3)")


@njit(cache=True, inline="always")
def _validate_xyz_point(point: np.ndarray) -> None:
    if point.ndim != 1 or point.shape[0] != 3:
        raise ValueError("point inputs must have shape (3,)")


@njit(cache=True, inline="always")
def _validate_nx3_points(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("batched point inputs must have shape (N, 3)")


@njit(cache=True, inline="always")
def _validate_sphere_radius(sphere_radius: float) -> None:
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be > 0")


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

    # Inside the blocking sphere means the segment is occluded by definition.
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

    # An intersection strictly between the endpoints blocks line of sight.
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
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
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
                ox,
                oy,
                oz,
                targets_pos[j, 0],
                targets_pos[j, 1],
                targets_pos[j, 2],
                radius_sq,
            )
    return out


@njit(cache=True, parallel=True)
def _los_sphere_pairwise_origin(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_nx3_points(targets_pos)
    n = observers_pos.shape[0]
    if targets_pos.shape[0] != n:
        raise ValueError("observer_pos and target_pos must have the same number of rows")
    out = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        out[i] = _los_clear_components_sphere(
            observers_pos[i, 0],
            observers_pos[i, 1],
            observers_pos[i, 2],
            targets_pos[i, 0],
            targets_pos[i, 1],
            targets_pos[i, 2],
            radius_sq,
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


@njit(cache=True)
def _los_sphere_pairwise_offset(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
    cx: float,
    cy: float,
    cz: float,
) -> np.ndarray:
    observers_shifted = _shift_points(observers_pos, cx, cy, cz)
    targets_shifted = _shift_points(targets_pos, cx, cy, cz)
    return _los_sphere_pairwise_origin(observers_shifted, targets_shifted, radius_sq)


@njit(cache=True, parallel=True)
def _los_sphere_one_to_many_origin(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    radius_sq: float,
) -> np.ndarray:
    m = targets_pos.shape[0]
    out = np.empty(m, dtype=np.bool_)
    ox = observer_pos[0]
    oy = observer_pos[1]
    oz = observer_pos[2]
    for j in prange(m):
        out[j] = _los_clear_components_sphere(
            ox,
            oy,
            oz,
            targets_pos[j, 0],
            targets_pos[j, 1],
            targets_pos[j, 2],
            radius_sq,
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
            ox,
            oy,
            oz,
            targets_shifted[j, 0],
            targets_shifted[j, 1],
            targets_shifted[j, 2],
            radius_sq,
        )
    return out


@njit(cache=True)
def _los_clear_sphere_scalar(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float,
    sphere_center_y: float,
    sphere_center_z: float,
) -> bool:
    _validate_xyz_point(observer_pos)
    _validate_xyz_point(target_pos)
    _validate_sphere_radius(sphere_radius)

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_clear_components_sphere(
            observer_pos[0],
            observer_pos[1],
            observer_pos[2],
            target_pos[0],
            target_pos[1],
            target_pos[2],
            radius_sq,
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
def _los_clear_sphere_point_to_points(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float,
    sphere_center_y: float,
    sphere_center_z: float,
) -> np.ndarray:
    _validate_xyz_point(observer_pos)
    _validate_nx3_points(targets_pos)
    _validate_sphere_radius(sphere_radius)

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_sphere_one_to_many_origin(observer_pos, targets_pos, radius_sq)
    return _los_sphere_one_to_many_offset(
        observer_pos,
        targets_pos,
        radius_sq,
        sphere_center_x,
        sphere_center_y,
        sphere_center_z,
    )


@njit(cache=True)
def _los_clear_sphere_points_to_point(
    observers_pos: np.ndarray,
    target_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float,
    sphere_center_y: float,
    sphere_center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_xyz_point(target_pos)
    return _los_clear_sphere_point_to_points(
        target_pos,
        observers_pos,
        sphere_radius,
        sphere_center_x,
        sphere_center_y,
        sphere_center_z,
    )


@njit(cache=True)
def _los_clear_sphere_points_to_points(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float,
    sphere_center_y: float,
    sphere_center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_nx3_points(targets_pos)
    _validate_sphere_radius(sphere_radius)

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_sphere_many_to_many_origin(observers_pos, targets_pos, radius_sq)
    return _los_sphere_many_to_many_offset(
        observers_pos,
        targets_pos,
        radius_sq,
        sphere_center_x,
        sphere_center_y,
        sphere_center_z,
    )


@njit(cache=True)
def _los_clear_sphere_points_pairwise(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float,
    sphere_center_y: float,
    sphere_center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_nx3_points(targets_pos)
    _validate_sphere_radius(sphere_radius)

    radius_sq = sphere_radius * sphere_radius
    if sphere_center_x == 0.0 and sphere_center_y == 0.0 and sphere_center_z == 0.0:
        return _los_sphere_pairwise_origin(observers_pos, targets_pos, radius_sq)
    return _los_sphere_pairwise_offset(
        observers_pos,
        targets_pos,
        radius_sq,
        sphere_center_x,
        sphere_center_y,
        sphere_center_z,
    )


def los_clear_sphere(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float = 0.0,
    sphere_center_y: float = 0.0,
    sphere_center_z: float = 0.0,
) -> bool | np.ndarray:
    """Check whether line segments are clear of a blocking sphere.

    This unified interface accepts either a single point ``(3,)`` or a stack of
    points ``(N, 3)`` for each endpoint argument.

    Accepted input forms
    --------------------
    - ``los_clear_sphere(observer_pos, target_pos, radius)`` returns ``bool``
    - ``los_clear_sphere(observer_pos, targets_pos, radius)`` returns ``(M,)``
    - ``los_clear_sphere(observers_pos, target_pos, radius)`` returns ``(N,)``
    - ``los_clear_sphere(observers_pos, targets_pos, radius)`` returns ``(N, M)``

    Parameters
    ----------
    observer_pos, target_pos : np.ndarray
        Endpoint coordinates in a common Cartesian frame. Each input must have
        shape ``(3,)`` or ``(N, 3)``.
    sphere_radius : float
        Blocking sphere radius in the same distance units as the inputs.
    sphere_center_x, sphere_center_y, sphere_center_z : float, optional
        Sphere center coordinates. The default sphere is centered at the origin.

    Returns
    -------
    bool or np.ndarray
        Visibility flags with shape determined by the input combination. When
        both inputs are batched, the result is the full observer-target matrix,
        not a pairwise elementwise comparison.

    Notes
    -----
    The same input forms work inside ``@numba.njit`` callers through a Numba
    overload, so compiled code stays fully in nopython mode.
    """

    observer_arr = _as_point_input(observer_pos, "observer_pos")
    target_arr = _as_point_input(target_pos, "target_pos")

    if observer_arr.ndim == 1 and target_arr.ndim == 1:
        return _los_clear_sphere_scalar(
            observer_arr,
            target_arr,
            sphere_radius,
            sphere_center_x,
            sphere_center_y,
            sphere_center_z,
        )
    if observer_arr.ndim == 1 and target_arr.ndim == 2:
        return _los_clear_sphere_point_to_points(
            observer_arr,
            target_arr,
            sphere_radius,
            sphere_center_x,
            sphere_center_y,
            sphere_center_z,
        )
    if observer_arr.ndim == 2 and target_arr.ndim == 1:
        return _los_clear_sphere_points_to_point(
            observer_arr,
            target_arr,
            sphere_radius,
            sphere_center_x,
            sphere_center_y,
            sphere_center_z,
        )
    return _los_clear_sphere_points_to_points(
        observer_arr,
        target_arr,
        sphere_radius,
        sphere_center_x,
        sphere_center_y,
        sphere_center_z,
    )


def los_clear_sphere_pairwise(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    sphere_radius: float,
    sphere_center_x: float = 0.0,
    sphere_center_y: float = 0.0,
    sphere_center_z: float = 0.0,
) -> bool | np.ndarray:
    """Check pairwise line-of-sight for equal-length endpoint stacks.

    Accepted input forms
    --------------------
    - ``los_clear_sphere_pairwise(observer_pos, target_pos, radius)`` returns ``bool``
    - ``los_clear_sphere_pairwise(observers_pos, targets_pos, radius)`` returns ``(N,)``
    """

    observer_arr = _as_point_input(observer_pos, "observer_pos")
    target_arr = _as_point_input(target_pos, "target_pos")

    if observer_arr.ndim == 1 and target_arr.ndim == 1:
        return _los_clear_sphere_scalar(
            observer_arr,
            target_arr,
            sphere_radius,
            sphere_center_x,
            sphere_center_y,
            sphere_center_z,
        )
    if observer_arr.ndim == 2 and target_arr.ndim == 2:
        return _los_clear_sphere_points_pairwise(
            observer_arr,
            target_arr,
            sphere_radius,
            sphere_center_x,
            sphere_center_y,
            sphere_center_z,
        )
    raise ValueError(
        "pairwise LOS requires observer_pos and target_pos to both be shape (3,) "
        "or both be shape (N, 3)"
    )


@numba_overload(los_clear_sphere)
def _ol_los_clear_sphere(
    observer_pos,
    target_pos,
    sphere_radius,
    sphere_center_x=0.0,
    sphere_center_y=0.0,
    sphere_center_z=0.0,
):
    centers_ok = (
        (is_numba_scalar(sphere_center_x) or is_numba_absent(sphere_center_x))
        and (is_numba_scalar(sphere_center_y) or is_numba_absent(sphere_center_y))
        and (is_numba_scalar(sphere_center_z) or is_numba_absent(sphere_center_z))
    )
    if not is_numba_scalar(sphere_radius) or not centers_ok:
        return None

    if is_numba_array1d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_scalar(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    if is_numba_array1d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_point_to_points(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_points_to_point(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_points_to_points(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    return None


@numba_overload(los_clear_sphere_pairwise)
def _ol_los_clear_sphere_pairwise(
    observer_pos,
    target_pos,
    sphere_radius,
    sphere_center_x=0.0,
    sphere_center_y=0.0,
    sphere_center_z=0.0,
):
    centers_ok = (
        (is_numba_scalar(sphere_center_x) or is_numba_absent(sphere_center_x))
        and (is_numba_scalar(sphere_center_y) or is_numba_absent(sphere_center_y))
        and (is_numba_scalar(sphere_center_z) or is_numba_absent(sphere_center_z))
    )
    if not is_numba_scalar(sphere_radius) or not centers_ok:
        return None

    if is_numba_array1d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_scalar(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            sphere_radius,
            sphere_center_x=0.0,
            sphere_center_y=0.0,
            sphere_center_z=0.0,
        ):
            return _los_clear_sphere_points_pairwise(
                observer_pos,
                target_pos,
                sphere_radius,
                sphere_center_x,
                sphere_center_y,
                sphere_center_z,
            )

        return impl

    return None


__all__ = [
    "los_clear_sphere",
    "los_clear_sphere_pairwise",
]
