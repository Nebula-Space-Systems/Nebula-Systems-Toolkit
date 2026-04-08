"""WGS84 geodetic to ECEF transforms."""

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
from .constants import WGS84_A, WGS84_B2_OVER_A2, WGS84_E2


@njit(cache=True, inline="always")
def _geodetic2ecef_scalar(lat_rad: float, lon_rad: float, h_m: float):
    """Scalar geodetic to ECEF kernel."""

    sphi = math.sin(lat_rad)
    cphi = math.cos(lat_rad)
    slam = math.sin(lon_rad)
    clam = math.cos(lon_rad)

    denom = math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
    N = WGS84_A / denom

    Nh = N + h_m
    x = Nh * cphi * clam
    y = Nh * cphi * slam
    z = (N * WGS84_B2_OVER_A2 + h_m) * sphi
    return x, y, z


@njit(cache=True, parallel=True)
def _geodetic2ecef_vector_llh(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
):
    """Vector geodetic to ECEF kernel for split ``lat/lon/h`` arrays."""

    n = lat_rad.shape[0]
    if lon_rad.shape[0] != n or h_m.shape[0] != n:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    for i in prange(n):
        xi, yi, zi = _geodetic2ecef_scalar(lat_rad[i], lon_rad[i], h_m[i])
        x[i] = xi
        y[i] = yi
        z[i] = zi

    return x, y, z


@njit(cache=True, parallel=True)
def _geodetic2ecef_vector_lla(lla_rad_m: np.ndarray):
    """Vector geodetic to ECEF kernel for an ``(N, 3)`` input array."""

    if lla_rad_m.ndim != 2 or lla_rad_m.shape[1] != 3:
        raise ValueError("lla_rad_m must have shape (N, 3)")

    n = lla_rad_m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)

    for i in prange(n):
        xi, yi, zi = _geodetic2ecef_scalar(
            lla_rad_m[i, 0],
            lla_rad_m[i, 1],
            lla_rad_m[i, 2],
        )
        out[i, 0] = xi
        out[i, 1] = yi
        out[i, 2] = zi

    return out


@typing_overload
def geodetic2ecef(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def geodetic2ecef(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def geodetic2ecef(
    lat_rad: np.ndarray,
    lon_rad: None = None,
    h_m: None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def geodetic2ecef(
    lat_rad: float | np.ndarray,
    lon_rad: float | np.ndarray | None = None,
    h_m: float | np.ndarray | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    This is the unified public interface for geodetic-to-ECEF conversion.
    Use the same function name for single points, split component arrays, or
    stacked ``(N, 3)`` geodetic arrays.

    Accepted input forms
    --------------------
    - ``geodetic2ecef(lat_rad, lon_rad, h_m)`` for one point
    - ``geodetic2ecef(lat_rad, lon_rad, h_m)`` for matching 1D arrays
    - ``geodetic2ecef(lla_rad_m)`` for an ``(N, 3)`` array

    Parameters
    ----------
    lat_rad, lon_rad : float or np.ndarray
        Geodetic latitude and longitude in radians. If ``lon_rad`` and
        ``h_m`` are omitted, ``lat_rad`` must be an array of shape ``(N, 3)``
        with columns ``[lat_rad, lon_rad, h_m]``.
    h_m : float or np.ndarray, optional
        Ellipsoidal height above the WGS84 reference ellipsoid in meters.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, z)`` ECEF coordinates in meters. Scalar input returns three
        scalars. Array input returns three same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For best throughput on large batches, prefer contiguous ``float64`` arrays.

    Examples
    --------
    >>> x_m, y_m, z_m = geodetic2ecef(lat_rad, lon_rad, h_m)
    >>> x_m, y_m, z_m = geodetic2ecef(lat_vec, lon_vec, h_vec)
    >>> x_m, y_m, z_m = geodetic2ecef(lla_rad_m)
    """

    if lon_rad is None and h_m is None:
        lla_rad_m = as_nx3_array(lat_rad, "lla_rad_m")
        out = _geodetic2ecef_vector_lla(lla_rad_m)
        return out[:, 0], out[:, 1], out[:, 2]

    if lon_rad is None or h_m is None:
        raise TypeError("Provide either `lat_rad, lon_rad, h_m` or one `(N, 3)` array")

    if np.isscalar(lat_rad) and np.isscalar(lon_rad) and np.isscalar(h_m):
        return _geodetic2ecef_scalar(float(lat_rad), float(lon_rad), float(h_m))

    lat_arr = as_1d_array(lat_rad, "lat_rad")
    lon_arr = as_1d_array(lon_rad, "lon_rad")
    h_arr = as_1d_array(h_m, "h_m")
    validate_matching_lengths(("lat_rad", lat_arr), ("lon_rad", lon_arr), ("h_m", h_arr))
    return _geodetic2ecef_vector_llh(lat_arr, lon_arr, h_arr)


@numba_overload(geodetic2ecef)
def _ol_geodetic2ecef(lat_rad, lon_rad=None, h_m=None):
    if is_numba_scalar(lat_rad) and is_numba_scalar(lon_rad) and is_numba_scalar(h_m):

        def impl(lat_rad, lon_rad=None, h_m=None):
            return _geodetic2ecef_scalar(lat_rad, lon_rad, h_m)

        return impl

    if is_numba_array1d(lat_rad) and is_numba_array1d(lon_rad) and is_numba_array1d(h_m):

        def impl(lat_rad, lon_rad=None, h_m=None):
            return _geodetic2ecef_vector_llh(lat_rad, lon_rad, h_m)

        return impl

    if is_numba_array2d(lat_rad) and is_numba_absent(lon_rad) and is_numba_absent(h_m):

        def impl(lat_rad, lon_rad=None, h_m=None):
            out = _geodetic2ecef_vector_lla(lat_rad)
            return out[:, 0], out[:, 1], out[:, 2]

        return impl

    return None

__all__ = [
    "geodetic2ecef",
]
