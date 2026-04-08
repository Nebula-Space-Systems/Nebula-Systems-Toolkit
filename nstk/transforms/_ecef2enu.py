"""ECEF to local ENU transforms."""

from __future__ import annotations

import math
from typing import overload as typing_overload

import numpy as np
from numba import njit, prange
from numba.extending import overload as numba_overload

from ._api_utils import (
    as_1d_array,
    as_nx3_array,
    is_numba_absent,
    is_numba_array1d,
    is_numba_array2d,
    is_numba_scalar,
    require_not_none,
    validate_matching_lengths,
)
from ._geodetic2ecef import geodetic2ecef
from ._ecef2geodetic import ecef2geodetic


@njit(cache=True, inline="always")
def enu_basis_from_ecef_xyz(x_m: float, y_m: float, z_m: float) -> np.ndarray:
    """Build the local ENU basis matrix at an ECEF position.

    Parameters
    ----------
    x_m, y_m, z_m : float
        ECEF position coordinates in meters.

    Returns
    -------
    np.ndarray
        A ``(3, 3)`` rotation matrix whose columns are the local east,
        north, and up unit vectors expressed in the ECEF basis.

    Notes
    -----
    Multiplying an ECEF delta vector by this basis maps it into ENU
    coordinates at the corresponding geodetic location.
    """

    lat, lon, _ = ecef2geodetic(x_m, y_m, z_m)

    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)

    R = np.empty((3, 3), dtype=np.float64)

    R[0, 0] = -slon
    R[1, 0] = clon
    R[2, 0] = 0.0

    R[0, 1] = -slat * clon
    R[1, 1] = -slat * slon
    R[2, 1] = clat

    R[0, 2] = clat * clon
    R[1, 2] = clat * slon
    R[2, 2] = slat

    return R


@njit(cache=True, inline="always")
def ecef2enu_delta(
    dx_m: float,
    dy_m: float,
    dz_m: float,
    lat0_rad: float,
    lon0_rad: float,
):
    """Rotate an ECEF delta vector into a local ENU frame.

    Parameters
    ----------
    dx_m, dy_m, dz_m : float
        Delta vector components in the ECEF basis, in meters.
    lat0_rad, lon0_rad : float
        Geodetic latitude and longitude of the ENU frame origin, in radians.

    Returns
    -------
    tuple[float, float, float]
        ``(e_m, n_m, u_m)`` east, north, and up components in meters.

    Notes
    -----
    This function transforms only a delta vector. It does not subtract the
    observer position for you. Use :func:`ecef2enu` when you start from
    absolute target coordinates.
    """

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    e = -slon * dx_m + clon * dy_m
    n = -slat * clon * dx_m - slat * slon * dy_m + clat * dz_m
    u = clat * clon * dx_m + clat * slon * dy_m + slat * dz_m
    return e, n, u


@njit(cache=True, inline="always")
def _ecef2enu_scalar(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar ECEF to ENU kernel."""

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    dx = x_m - x0
    dy = y_m - y0
    dz = z_m - z0
    return ecef2enu_delta(dx, dy, dz, lat0_rad, lon0_rad)


@njit(cache=True, parallel=True)
def _ecef2enu_vector_xyz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector ECEF to ENU kernel for split ``x/y/z`` arrays."""

    n = x_m.shape[0]
    if y_m.shape[0] != n or z_m.shape[0] != n:
        raise ValueError("x_m, y_m, z_m must have the same length")

    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        dx = x_m[i] - x0
        dy = y_m[i] - y0
        dz = z_m[i] - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u


@njit(cache=True, parallel=True)
def _ecef2enu_vector_ecef(
    r_ecef_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector ECEF to ENU kernel for an ``(N, 3)`` input array."""

    if r_ecef_m.ndim != 2 or r_ecef_m.shape[1] != 3:
        raise ValueError("r_ecef_m must have shape (N, 3)")

    n = r_ecef_m.shape[0]
    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        dx = r_ecef_m[i, 0] - x0
        dy = r_ecef_m[i, 1] - y0
        dz = r_ecef_m[i, 2] - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u


@typing_overload
def ecef2enu(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def ecef2enu(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def ecef2enu(
    x_m: np.ndarray,
    y_m: None = None,
    z_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def ecef2enu(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray | None = None,
    z_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert absolute ECEF coordinates to local ENU coordinates.

    The ENU frame is defined by the geodetic reference location
    ``(lat0_rad, lon0_rad, h0_m)``. The same reference point is applied to
    every row when array input is used.

    Accepted input forms
    --------------------
    - ``ecef2enu(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)``
    - ``ecef2enu(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``ecef2enu(r_ecef_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    x_m, y_m, z_m : float or np.ndarray
        Absolute ECEF coordinates in meters. If ``y_m`` and ``z_m`` are
        omitted, ``x_m`` must be an array of shape ``(N, 3)``.
    lat0_rad, lon0_rad : float
        Geodetic latitude and longitude of the ENU origin in radians.
    h0_m : float
        Height of the ENU origin above the WGS84 ellipsoid in meters.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(e_m, n_m, u_m)`` local east, north, and up coordinates in meters.
        Scalar input returns three scalars. Array input returns three
        same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For array inputs, the observer origin remains scalar and is broadcast
      across all rows.

    Examples
    --------
    >>> e_m, n_m, u_m = ecef2enu(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)
    >>> e_m, n_m, u_m = ecef2enu(r_ecef_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if y_m is None and z_m is None:
        r_ecef_m = as_nx3_array(x_m, "r_ecef_m")
        return _ecef2enu_vector_ecef(r_ecef_m, lat0_rad, lon0_rad, h0_m)

    if y_m is None or z_m is None:
        raise TypeError("Provide either `x_m, y_m, z_m` or one `(N, 3)` array")

    if np.isscalar(x_m) and np.isscalar(y_m) and np.isscalar(z_m):
        return _ecef2enu_scalar(
            float(x_m),
            float(y_m),
            float(z_m),
            float(lat0_rad),
            float(lon0_rad),
            float(h0_m),
        )

    x_arr = as_1d_array(x_m, "x_m")
    y_arr = as_1d_array(y_m, "y_m")
    z_arr = as_1d_array(z_m, "z_m")
    validate_matching_lengths(("x_m", x_arr), ("y_m", y_arr), ("z_m", z_arr))
    return _ecef2enu_vector_xyz(x_arr, y_arr, z_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(ecef2enu)
def _ol_ecef2enu(
    x_m,
    y_m=None,
    z_m=None,
    lat0_rad=None,
    lon0_rad=None,
    h0_m=None,
):
    if (
        is_numba_scalar(x_m)
        and is_numba_scalar(y_m)
        and is_numba_scalar(z_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(x_m, y_m=None, z_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _ecef2enu_scalar(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array1d(x_m)
        and is_numba_array1d(y_m)
        and is_numba_array1d(z_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(x_m, y_m=None, z_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _ecef2enu_vector_xyz(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array2d(x_m)
        and is_numba_absent(y_m)
        and is_numba_absent(z_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(x_m, y_m=None, z_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _ecef2enu_vector_ecef(x_m, lat0_rad, lon0_rad, h0_m)

        return impl

    return None

__all__ = [
    "enu_basis_from_ecef_xyz",
    "ecef2enu_delta",
    "ecef2enu",
]
