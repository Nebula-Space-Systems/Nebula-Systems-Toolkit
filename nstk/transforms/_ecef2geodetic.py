"""ECEF to WGS84 geodetic transforms."""

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
    validate_matching_lengths,
)
from .constants import RAD2DEG, WGS84_A, WGS84_B, WGS84_E2, WGS84_EP2


# Treat p = hypot(x,y) smaller than this as "numerically at the pole".
# This is intentionally larger than 1e-12 to handle round-off from geodetic->ECEF at ±90°.
_POLE_P_EPS_M = 1e-8


@njit(cache=True, inline="always")
def _wrap_lon_pi(lon_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)`` in a Numba-friendly way."""

    twopi = 2.0 * math.pi
    t = lon_rad + math.pi
    k = math.floor(t / twopi)
    t = t - k * twopi
    return t - math.pi


@njit(cache=True, inline="always")
def _ecef2geodetic_scalar(x_m: float, y_m: float, z_m: float):
    """Scalar ECEF to geodetic kernel in radians."""

    lon = _wrap_lon_pi(math.atan2(y_m, x_m))
    p = math.hypot(x_m, y_m)

    if p == 0.0 and z_m == 0.0:
        return 0.0, 0.0, -WGS84_A

    if p < _POLE_P_EPS_M:
        lat = 0.5 * math.pi if z_m >= 0.0 else -0.5 * math.pi
        h = abs(z_m) - WGS84_B
        return lat, 0.0, h

    theta = math.atan2(z_m * WGS84_A, p * WGS84_B)
    st = math.sin(theta)
    ct = math.cos(theta)

    st2 = st * st
    ct2 = ct * ct
    st3 = st2 * st
    ct3 = ct2 * ct

    lat = math.atan2(
        z_m + WGS84_EP2 * WGS84_B * st3,
        p - WGS84_E2 * WGS84_A * ct3,
    )

    sphi = math.sin(lat)
    cphi = math.cos(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
    N = WGS84_A / denom

    if abs(cphi) > 1e-12:
        h = p / cphi - N
    else:
        h = z_m / sphi - N * (1.0 - WGS84_E2)

    return lat, lon, h


@njit(cache=True, inline="always")
def _ecef2geodetic_scalar_deg(x_m: float, y_m: float, z_m: float):
    """Scalar ECEF to geodetic kernel with degree angles."""

    lat_rad, lon_rad, h_m = _ecef2geodetic_scalar(x_m, y_m, z_m)
    return lat_rad * RAD2DEG, lon_rad * RAD2DEG, h_m


@njit(cache=True, parallel=True)
def _ecef2geodetic_vector_xyz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
):
    """Vector ECEF to geodetic kernel for split ``x/y/z`` arrays in radians."""

    n = x_m.shape[0]
    if y_m.shape[0] != n or z_m.shape[0] != n:
        raise ValueError("x_m, y_m, z_m must have the same length")

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = _ecef2geodetic_scalar(x_m[i], y_m[i], z_m[i])
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def _ecef2geodetic_vector_xyz_deg(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
):
    """Vector ECEF to geodetic kernel for split ``x/y/z`` arrays in degrees."""

    lat_rad, lon_rad, h_m = _ecef2geodetic_vector_xyz(x_m, y_m, z_m)
    n = lat_rad.shape[0]
    lat_deg = np.empty(n, dtype=np.float64)
    lon_deg = np.empty(n, dtype=np.float64)

    for i in prange(n):
        lat_deg[i] = lat_rad[i] * RAD2DEG
        lon_deg[i] = lon_rad[i] * RAD2DEG

    return lat_deg, lon_deg, h_m


@njit(cache=True, parallel=True)
def _ecef2geodetic_vector_ecef(r_ecef_m: np.ndarray):
    """Vector ECEF to geodetic kernel for ``(N, 3)`` positions in radians."""

    if r_ecef_m.ndim != 2 or r_ecef_m.shape[1] != 3:
        raise ValueError("r_ecef_m must have shape (N, 3)")

    n = r_ecef_m.shape[0]
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = _ecef2geodetic_scalar(r_ecef_m[i, 0], r_ecef_m[i, 1], r_ecef_m[i, 2])
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def _ecef2geodetic_vector_ecef_deg(r_ecef_m: np.ndarray):
    """Vector ECEF to geodetic kernel for ``(N, 3)`` positions in degrees."""

    lat_rad, lon_rad, h_m = _ecef2geodetic_vector_ecef(r_ecef_m)
    n = lat_rad.shape[0]
    lat_deg = np.empty(n, dtype=np.float64)
    lon_deg = np.empty(n, dtype=np.float64)

    for i in prange(n):
        lat_deg[i] = lat_rad[i] * RAD2DEG
        lon_deg[i] = lon_rad[i] * RAD2DEG

    return lat_deg, lon_deg, h_m


@typing_overload
def ecef2geodetic(
    x_m: float,
    y_m: float,
    z_m: float,
    *,
    degrees: bool = False,
) -> tuple[float, float, float]:
    ...


@typing_overload
def ecef2geodetic(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    *,
    degrees: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def ecef2geodetic(
    x_m: np.ndarray,
    y_m: None = None,
    z_m: None = None,
    *,
    degrees: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def ecef2geodetic(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray | None = None,
    z_m: float | np.ndarray | None = None,
    *,
    degrees: bool = False,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates.

    This is the unified public interface for ECEF-to-geodetic conversion.
    Use the same function name for single points, split component arrays, or
    stacked ``(N, 3)`` position arrays.

    Accepted input forms
    --------------------
    - ``ecef2geodetic(x_m, y_m, z_m)`` for one point
    - ``ecef2geodetic(x_m, y_m, z_m)`` for matching 1D arrays
    - ``ecef2geodetic(r_ecef_m)`` for an ``(N, 3)`` array

    Parameters
    ----------
    x_m, y_m, z_m : float or np.ndarray
        ECEF coordinates in meters. If ``y_m`` and ``z_m`` are omitted,
        ``x_m`` must be an array of shape ``(N, 3)`` with columns
        ``[x_m, y_m, z_m]``.
    degrees : bool, default=False
        If ``True``, latitude and longitude are returned in degrees.
        Otherwise they are returned in radians.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(lat, lon, h)`` where:

        - ``lat`` is geodetic latitude
        - ``lon`` is longitude wrapped to ``[-pi, pi)`` or ``[-180, 180)``
        - ``h`` is ellipsoidal height above the WGS84 reference ellipsoid in meters

        Scalar input returns three scalars. Array input returns three
        same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For best throughput on large batches, prefer contiguous ``float64`` arrays.

    Examples
    --------
    >>> lat_rad, lon_rad, h_m = ecef2geodetic(x_m, y_m, z_m)
    >>> lat_rad, lon_rad, h_m = ecef2geodetic(r_ecef_m)
    >>> lat_deg, lon_deg, h_m = ecef2geodetic(r_ecef_m, degrees=True)
    """

    if y_m is None and z_m is None:
        r_ecef_m = as_nx3_array(x_m, "r_ecef_m")
        if degrees:
            return _ecef2geodetic_vector_ecef_deg(r_ecef_m)
        return _ecef2geodetic_vector_ecef(r_ecef_m)

    if y_m is None or z_m is None:
        raise TypeError("Provide either `x_m, y_m, z_m` or one `(N, 3)` array")

    if np.isscalar(x_m) and np.isscalar(y_m) and np.isscalar(z_m):
        if degrees:
            return _ecef2geodetic_scalar_deg(float(x_m), float(y_m), float(z_m))
        return _ecef2geodetic_scalar(float(x_m), float(y_m), float(z_m))

    x_arr = as_1d_array(x_m, "x_m")
    y_arr = as_1d_array(y_m, "y_m")
    z_arr = as_1d_array(z_m, "z_m")
    validate_matching_lengths(("x_m", x_arr), ("y_m", y_arr), ("z_m", z_arr))

    if degrees:
        return _ecef2geodetic_vector_xyz_deg(x_arr, y_arr, z_arr)
    return _ecef2geodetic_vector_xyz(x_arr, y_arr, z_arr)


@numba_overload(ecef2geodetic)
def _ol_ecef2geodetic(x_m, y_m=None, z_m=None, degrees=False):
    if is_numba_scalar(x_m) and is_numba_scalar(y_m) and is_numba_scalar(z_m):

        def impl(x_m, y_m=None, z_m=None, degrees=False):
            if degrees:
                return _ecef2geodetic_scalar_deg(x_m, y_m, z_m)
            return _ecef2geodetic_scalar(x_m, y_m, z_m)

        return impl

    if is_numba_array1d(x_m) and is_numba_array1d(y_m) and is_numba_array1d(z_m):

        def impl(x_m, y_m=None, z_m=None, degrees=False):
            if degrees:
                return _ecef2geodetic_vector_xyz_deg(x_m, y_m, z_m)
            return _ecef2geodetic_vector_xyz(x_m, y_m, z_m)

        return impl

    if is_numba_array2d(x_m) and is_numba_absent(y_m) and is_numba_absent(z_m):

        def impl(x_m, y_m=None, z_m=None, degrees=False):
            if degrees:
                return _ecef2geodetic_vector_ecef_deg(x_m)
            return _ecef2geodetic_vector_ecef(x_m)

        return impl

    return None

__all__ = [
    "ecef2geodetic",
    "_wrap_lon_pi",
]
