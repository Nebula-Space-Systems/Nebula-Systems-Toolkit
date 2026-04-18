"""
Unified ellipsoid line-of-sight utilities.

The public ``los_clear_ellipsoid`` and ``los_clear_ellipsoid_oriented``
interfaces accept either one point with shape ``(3,)`` or a stack of points
with shape ``(N, 3)`` for each endpoint input. They return a scalar, a 1D
boolean array, or a 2D boolean matrix based on the input combination, and the
same forms work inside ``@numba.njit`` callers.
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


def _as_orientation_matrix(value: np.ndarray) -> np.ndarray:
    """Convert a Python value to a ``(3, 3)`` orientation matrix."""

    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[0] != 3 or arr.shape[1] != 3:
        raise ValueError("orientation_ellipsoid_to_frame must have shape (3, 3)")
    return arr


@njit(cache=True, inline="always")
def _validate_xyz_point(point: np.ndarray) -> None:
    if point.ndim != 1 or point.shape[0] != 3:
        raise ValueError("point inputs must have shape (3,)")


@njit(cache=True, inline="always")
def _validate_nx3_points(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("batched point inputs must have shape (N, 3)")


@njit(cache=True, inline="always")
def _validate_orientation_matrix(orientation_ellipsoid_to_frame: np.ndarray) -> None:
    if (
        orientation_ellipsoid_to_frame.ndim != 2
        or orientation_ellipsoid_to_frame.shape[0] != 3
        or orientation_ellipsoid_to_frame.shape[1] != 3
    ):
        raise ValueError("orientation_ellipsoid_to_frame must have shape (3, 3)")


@njit(cache=True, inline="always")
def _validate_axes(semi_axis_a: float, semi_axis_b: float, semi_axis_c: float) -> None:
    if semi_axis_a <= 0.0 or semi_axis_b <= 0.0 or semi_axis_c <= 0.0:
        raise ValueError("semi-axis values must be > 0")


@njit(cache=True, inline="always")
def _inverse_axis_squares(
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
):
    _validate_axes(semi_axis_a, semi_axis_b, semi_axis_c)
    return (
        1.0 / (semi_axis_a * semi_axis_a),
        1.0 / (semi_axis_b * semi_axis_b),
        1.0 / (semi_axis_c * semi_axis_c),
    )


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

    # Inside the blocking ellipsoid means the segment is occluded by definition.
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

    # An intersection strictly between the endpoints blocks line of sight.
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

    ``orientation_ellipsoid_to_frame`` maps body axes into frame axes, so this
    applies its transpose to move frame coordinates into the body frame.
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
                ox,
                oy,
                oz,
                targets_body[j, 0],
                targets_body[j, 1],
                targets_body[j, 2],
                inv_a2,
                inv_b2,
                inv_c2,
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
            ox,
            oy,
            oz,
            targets_body[j, 0],
            targets_body[j, 1],
            targets_body[j, 2],
            inv_a2,
            inv_b2,
            inv_c2,
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


@njit(cache=True)
def _los_clear_ellipsoid_scalar(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> bool:
    _validate_xyz_point(observer_pos)
    _validate_xyz_point(target_pos)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_clear_components_ellipsoid_axis_aligned(
            observer_pos[0],
            observer_pos[1],
            observer_pos[2],
            target_pos[0],
            target_pos[1],
            target_pos[2],
            inv_a2,
            inv_b2,
            inv_c2,
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
def _los_clear_ellipsoid_point_to_points(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_xyz_point(observer_pos)
    _validate_nx3_points(targets_pos)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_one_to_many_body(
            observer_pos[0],
            observer_pos[1],
            observer_pos[2],
            targets_pos,
            inv_a2,
            inv_b2,
            inv_c2,
        )
    return _los_one_to_many_offset(
        observer_pos,
        targets_pos,
        inv_a2,
        inv_b2,
        inv_c2,
        center_x,
        center_y,
        center_z,
    )


@njit(cache=True)
def _los_clear_ellipsoid_points_to_point(
    observers_pos: np.ndarray,
    target_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_xyz_point(target_pos)
    return _los_clear_ellipsoid_point_to_points(
        target_pos,
        observers_pos,
        inv_a2,
        inv_b2,
        inv_c2,
        center_x,
        center_y,
        center_z,
    )


@njit(cache=True)
def _los_clear_ellipsoid_points_to_points(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_nx3_points(targets_pos)

    if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
        return _los_many_to_many_body(observers_pos, targets_pos, inv_a2, inv_b2, inv_c2)
    return _los_many_to_many_offset(
        observers_pos,
        targets_pos,
        inv_a2,
        inv_b2,
        inv_c2,
        center_x,
        center_y,
        center_z,
    )


def los_clear_ellipsoid(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    semi_axis_a: float,
    semi_axis_b: float,
    semi_axis_c: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> bool | np.ndarray:
    """Check whether line segments are clear of an axis-aligned ellipsoid.

    This unified interface accepts either a single point ``(3,)`` or a stack of
    points ``(N, 3)`` for each endpoint argument.

    Accepted input forms
    --------------------
    - ``los_clear_ellipsoid(observer_pos, target_pos, a, b, c)`` returns ``bool``
    - ``los_clear_ellipsoid(observer_pos, targets_pos, a, b, c)`` returns ``(M,)``
    - ``los_clear_ellipsoid(observers_pos, target_pos, a, b, c)`` returns ``(N,)``
    - ``los_clear_ellipsoid(observers_pos, targets_pos, a, b, c)`` returns ``(N, M)``

    Parameters
    ----------
    observer_pos, target_pos : np.ndarray
        Endpoint coordinates in a common Cartesian frame. Each input must have
        shape ``(3,)`` or ``(N, 3)``.
    semi_axis_a, semi_axis_b, semi_axis_c : float
        Ellipsoid semi-axis lengths in the same distance units as the inputs.
        This lets callers use Earth models other than WGS84 when needed.
    center_x, center_y, center_z : float, optional
        Ellipsoid center coordinates. The default ellipsoid is centered at the
        origin.

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

    Examples
    --------
    Use WGS84 Earth axes with ECEF points by passing the WGS84 equatorial and
    polar semi-axes explicitly:

    >>> from nstk.transforms.constants import WGS84_A, WGS84_B
    >>> clear = los_clear_ellipsoid(
    ...     observer_ecef_m,
    ...     target_ecef_m,
    ...     WGS84_A,
    ...     WGS84_A,
    ...     WGS84_B,
    ... )
    """

    observer_arr = _as_point_input(observer_pos, "observer_pos")
    target_arr = _as_point_input(target_pos, "target_pos")
    inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
        semi_axis_a,
        semi_axis_b,
        semi_axis_c,
    )

    if observer_arr.ndim == 1 and target_arr.ndim == 1:
        return _los_clear_ellipsoid_scalar(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
        )
    if observer_arr.ndim == 1 and target_arr.ndim == 2:
        return _los_clear_ellipsoid_point_to_points(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
        )
    if observer_arr.ndim == 2 and target_arr.ndim == 1:
        return _los_clear_ellipsoid_points_to_point(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
        )
    return _los_clear_ellipsoid_points_to_points(
        observer_arr,
        target_arr,
        inv_a2,
        inv_b2,
        inv_c2,
        center_x,
        center_y,
        center_z,
    )


@numba_overload(los_clear_ellipsoid)
def _ol_los_clear_ellipsoid(
    observer_pos,
    target_pos,
    semi_axis_a,
    semi_axis_b,
    semi_axis_c,
    center_x=0.0,
    center_y=0.0,
    center_z=0.0,
):
    centers_ok = (
        (is_numba_scalar(center_x) or is_numba_absent(center_x))
        and (is_numba_scalar(center_y) or is_numba_absent(center_y))
        and (is_numba_scalar(center_z) or is_numba_absent(center_z))
    )
    axes_ok = (
        is_numba_scalar(semi_axis_a)
        and is_numba_scalar(semi_axis_b)
        and is_numba_scalar(semi_axis_c)
    )
    if not axes_ok or not centers_ok:
        return None

    if is_numba_array1d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_scalar(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array1d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_point_to_points(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_points_to_point(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_points_to_points(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                center_x,
                center_y,
                center_z,
            )

        return impl

    return None


@njit(cache=True)
def _los_clear_ellipsoid_oriented_scalar(
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
) -> bool:
    _validate_xyz_point(observer_pos)
    _validate_xyz_point(target_pos)
    _validate_orientation_matrix(orientation_ellipsoid_to_frame)

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return _los_clear_ellipsoid_scalar(
            observer_pos,
            target_pos,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
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
        obx,
        oby,
        obz,
        tbx,
        tby,
        tbz,
        inv_a2,
        inv_b2,
        inv_c2,
    )


@njit(cache=True)
def _los_clear_ellipsoid_oriented_point_to_points(
    observer_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_xyz_point(observer_pos)
    _validate_nx3_points(targets_pos)
    _validate_orientation_matrix(orientation_ellipsoid_to_frame)

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return _los_clear_ellipsoid_point_to_points(
            observer_pos,
            targets_pos,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
        )

    targets_body = _transform_points_to_body(
        targets_pos,
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
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


@njit(cache=True)
def _los_clear_ellipsoid_oriented_points_to_point(
    observers_pos: np.ndarray,
    target_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_xyz_point(target_pos)
    return _los_clear_ellipsoid_oriented_point_to_points(
        target_pos,
        observers_pos,
        inv_a2,
        inv_b2,
        inv_c2,
        orientation_ellipsoid_to_frame,
        center_x,
        center_y,
        center_z,
    )


@njit(cache=True)
def _los_clear_ellipsoid_oriented_points_to_points(
    observers_pos: np.ndarray,
    targets_pos: np.ndarray,
    inv_a2: float,
    inv_b2: float,
    inv_c2: float,
    orientation_ellipsoid_to_frame: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
) -> np.ndarray:
    _validate_nx3_points(observers_pos)
    _validate_nx3_points(targets_pos)
    _validate_orientation_matrix(orientation_ellipsoid_to_frame)

    if _is_identity_orientation(orientation_ellipsoid_to_frame):
        return _los_clear_ellipsoid_points_to_points(
            observers_pos,
            targets_pos,
            inv_a2,
            inv_b2,
            inv_c2,
            center_x,
            center_y,
            center_z,
        )

    observers_body = _transform_points_to_body(
        observers_pos,
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
    )
    targets_body = _transform_points_to_body(
        targets_pos,
        center_x,
        center_y,
        center_z,
        orientation_ellipsoid_to_frame,
    )
    return _los_many_to_many_body(observers_body, targets_body, inv_a2, inv_b2, inv_c2)


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
) -> bool | np.ndarray:
    """Check whether line segments are clear of an oriented ellipsoid.

    This unified interface accepts either a single point ``(3,)`` or a stack of
    points ``(N, 3)`` for each endpoint argument.

    Accepted input forms
    --------------------
    - ``los_clear_ellipsoid_oriented(observer_pos, target_pos, ...)`` returns ``bool``
    - ``los_clear_ellipsoid_oriented(observer_pos, targets_pos, ...)`` returns ``(M,)``
    - ``los_clear_ellipsoid_oriented(observers_pos, target_pos, ...)`` returns ``(N,)``
    - ``los_clear_ellipsoid_oriented(observers_pos, targets_pos, ...)`` returns ``(N, M)``

    Parameters
    ----------
    observer_pos, target_pos : np.ndarray
        Endpoint coordinates in a common Cartesian frame. Each input must have
        shape ``(3,)`` or ``(N, 3)``.
    semi_axis_a, semi_axis_b, semi_axis_c : float
        Ellipsoid semi-axis lengths in the same distance units as the inputs.
    orientation_ellipsoid_to_frame : np.ndarray
        A ``(3, 3)`` rotation matrix whose columns are the ellipsoid body axes
        expressed in the input coordinate frame.
    center_x, center_y, center_z : float, optional
        Ellipsoid center coordinates. The default ellipsoid is centered at the
        origin.

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
    orientation_arr = _as_orientation_matrix(orientation_ellipsoid_to_frame)
    inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
        semi_axis_a,
        semi_axis_b,
        semi_axis_c,
    )

    if observer_arr.ndim == 1 and target_arr.ndim == 1:
        return _los_clear_ellipsoid_oriented_scalar(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            orientation_arr,
            center_x,
            center_y,
            center_z,
        )
    if observer_arr.ndim == 1 and target_arr.ndim == 2:
        return _los_clear_ellipsoid_oriented_point_to_points(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            orientation_arr,
            center_x,
            center_y,
            center_z,
        )
    if observer_arr.ndim == 2 and target_arr.ndim == 1:
        return _los_clear_ellipsoid_oriented_points_to_point(
            observer_arr,
            target_arr,
            inv_a2,
            inv_b2,
            inv_c2,
            orientation_arr,
            center_x,
            center_y,
            center_z,
        )
    return _los_clear_ellipsoid_oriented_points_to_points(
        observer_arr,
        target_arr,
        inv_a2,
        inv_b2,
        inv_c2,
        orientation_arr,
        center_x,
        center_y,
        center_z,
    )


@numba_overload(los_clear_ellipsoid_oriented)
def _ol_los_clear_ellipsoid_oriented(
    observer_pos,
    target_pos,
    semi_axis_a,
    semi_axis_b,
    semi_axis_c,
    orientation_ellipsoid_to_frame,
    center_x=0.0,
    center_y=0.0,
    center_z=0.0,
):
    centers_ok = (
        (is_numba_scalar(center_x) or is_numba_absent(center_x))
        and (is_numba_scalar(center_y) or is_numba_absent(center_y))
        and (is_numba_scalar(center_z) or is_numba_absent(center_z))
    )
    axes_ok = (
        is_numba_scalar(semi_axis_a)
        and is_numba_scalar(semi_axis_b)
        and is_numba_scalar(semi_axis_c)
    )
    if not axes_ok or not centers_ok or not is_numba_array2d(orientation_ellipsoid_to_frame):
        return None

    if is_numba_array1d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            orientation_ellipsoid_to_frame,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_oriented_scalar(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                orientation_ellipsoid_to_frame,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array1d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            orientation_ellipsoid_to_frame,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_oriented_point_to_points(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                orientation_ellipsoid_to_frame,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array1d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            orientation_ellipsoid_to_frame,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_oriented_points_to_point(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                orientation_ellipsoid_to_frame,
                center_x,
                center_y,
                center_z,
            )

        return impl

    if is_numba_array2d(observer_pos) and is_numba_array2d(target_pos):

        def impl(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            orientation_ellipsoid_to_frame,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        ):
            inv_a2, inv_b2, inv_c2 = _inverse_axis_squares(
                semi_axis_a,
                semi_axis_b,
                semi_axis_c,
            )
            return _los_clear_ellipsoid_oriented_points_to_points(
                observer_pos,
                target_pos,
                inv_a2,
                inv_b2,
                inv_c2,
                orientation_ellipsoid_to_frame,
                center_x,
                center_y,
                center_z,
            )

        return impl

    return None


__all__ = [
    "los_clear_ellipsoid",
    "los_clear_ellipsoid_oriented",
]
