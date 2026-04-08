"""WGS84 geodetic to local ENU transforms."""

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


@njit(cache=True, inline="always")
def _uvw2enu(dx_m: float, dy_m: float, dz_m: float, lat0_rad: float, lon0_rad: float):
    """Convert an ECEF delta vector into local ENU coordinates."""

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    e = -slon * dx_m + clon * dy_m
    n = -slat * clon * dx_m - slat * slon * dy_m + clat * dz_m
    u = clat * clon * dx_m + clat * slon * dy_m + slat * dz_m
    return e, n, u


@njit(cache=True, inline="always")
def enu_basis_from_latlon(lat0: float, lon0: float) -> np.ndarray:
    """Build the local ENU basis matrix from geodetic latitude and longitude.

    Parameters
    ----------
    lat0, lon0 : float
        Geodetic latitude and longitude in radians.

    Returns
    -------
    np.ndarray
        A ``(3, 3)`` rotation matrix whose columns are the local east,
        north, and up unit vectors expressed in the ECEF basis.

    Notes
    -----
    This basis depends only on latitude and longitude, not height.
    """

    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    R = np.empty((3, 3), dtype=np.float64)

    R[0, 0] = -sin_lon
    R[1, 0] = cos_lon
    R[2, 0] = 0.0

    R[0, 1] = -sin_lat * cos_lon
    R[1, 1] = -sin_lat * sin_lon
    R[2, 1] = cos_lat

    R[0, 2] = cos_lat * cos_lon
    R[1, 2] = cos_lat * sin_lon
    R[2, 2] = sin_lat

    return R


@njit(cache=True, inline="always")
def _geodetic2enu_scalar(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar geodetic to ENU kernel."""

    x, y, z = geodetic2ecef(lat_rad, lon_rad, h_m)
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    return _uvw2enu(dx, dy, dz, lat0_rad, lon0_rad)


@njit(cache=True, parallel=True)
def _geodetic2enu_vector_llh(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector geodetic to ENU kernel for split ``lat/lon/h`` arrays."""

    n = lat_rad.shape[0]
    if lon_rad.shape[0] != n or h_m.shape[0] != n:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        x, y, z = geodetic2ecef(lat_rad[i], lon_rad[i], h_m[i])
        dx = x - x0
        dy = y - y0
        dz = z - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u


@njit(cache=True, parallel=True)
def _geodetic2enu_vector_lla(
    lla_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector geodetic to ENU kernel for an ``(N, 3)`` input array."""

    if lla_rad_m.ndim != 2 or lla_rad_m.shape[1] != 3:
        raise ValueError("lla_rad_m must have shape (N, 3)")

    n = lla_rad_m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        x, y, z = geodetic2ecef(lla_rad_m[i, 0], lla_rad_m[i, 1], lla_rad_m[i, 2])
        dx = x - x0
        dy = y - y0
        dz = z - z0

        out[i, 0] = -slon * dx + clon * dy
        out[i, 1] = -slat * clon * dx - slat * slon * dy + clat * dz
        out[i, 2] = clat * clon * dx + clat * slon * dy + slat * dz

    return out


@typing_overload
def geodetic2enu(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def geodetic2enu(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def geodetic2enu(
    lat_rad: np.ndarray,
    lon_rad: None = None,
    h_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def geodetic2enu(
    lat_rad: float | np.ndarray,
    lon_rad: float | np.ndarray | None = None,
    h_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to local ENU coordinates.

    The ENU frame is defined by the geodetic reference location
    ``(lat0_rad, lon0_rad, h0_m)``. The same reference point is applied to
    every row when array input is used.

    Accepted input forms
    --------------------
    - ``geodetic2enu(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)``
    - ``geodetic2enu(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``geodetic2enu(lla_rad_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    lat_rad, lon_rad : float or np.ndarray
        Geodetic latitude and longitude in radians. If ``lon_rad`` and
        ``h_m`` are omitted, ``lat_rad`` must be an array of shape ``(N, 3)``
        with columns ``[lat_rad, lon_rad, h_m]``.
    h_m : float or np.ndarray, optional
        Height above the WGS84 ellipsoid in meters.
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
    >>> e_m, n_m, u_m = geodetic2enu(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)
    >>> e_m, n_m, u_m = geodetic2enu(lla_rad_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if lon_rad is None and h_m is None:
        lla_rad_m = as_nx3_array(lat_rad, "lla_rad_m")
        out = _geodetic2enu_vector_lla(lla_rad_m, lat0_rad, lon0_rad, h0_m)
        return out[:, 0], out[:, 1], out[:, 2]

    if lon_rad is None or h_m is None:
        raise TypeError("Provide either `lat_rad, lon_rad, h_m` or one `(N, 3)` array")

    if np.isscalar(lat_rad) and np.isscalar(lon_rad) and np.isscalar(h_m):
        return _geodetic2enu_scalar(
            float(lat_rad),
            float(lon_rad),
            float(h_m),
            float(lat0_rad),
            float(lon0_rad),
            float(h0_m),
        )

    lat_arr = as_1d_array(lat_rad, "lat_rad")
    lon_arr = as_1d_array(lon_rad, "lon_rad")
    h_arr = as_1d_array(h_m, "h_m")
    validate_matching_lengths(("lat_rad", lat_arr), ("lon_rad", lon_arr), ("h_m", h_arr))
    return _geodetic2enu_vector_llh(lat_arr, lon_arr, h_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(geodetic2enu)
def _ol_geodetic2enu(
    lat_rad,
    lon_rad=None,
    h_m=None,
    lat0_rad=None,
    lon0_rad=None,
    h0_m=None,
):
    if (
        is_numba_scalar(lat_rad)
        and is_numba_scalar(lon_rad)
        and is_numba_scalar(h_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(lat_rad, lon_rad=None, h_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _geodetic2enu_scalar(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array1d(lat_rad)
        and is_numba_array1d(lon_rad)
        and is_numba_array1d(h_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(lat_rad, lon_rad=None, h_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _geodetic2enu_vector_llh(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array2d(lat_rad)
        and is_numba_absent(lon_rad)
        and is_numba_absent(h_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(lat_rad, lon_rad=None, h_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            out = _geodetic2enu_vector_lla(lat_rad, lat0_rad, lon0_rad, h0_m)
            return out[:, 0], out[:, 1], out[:, 2]

        return impl

    return None

__all__ = [
    "_uvw2enu",
    "enu_basis_from_latlon",
    "geodetic2enu",
]
