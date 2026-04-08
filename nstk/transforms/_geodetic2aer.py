"""WGS84 geodetic to AER transforms."""

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
from ._geodetic2enu import geodetic2enu


@njit(cache=True, inline="always")
def enu2aer(e_m: float, n_m: float, u_m: float):
    """Convert local ENU coordinates to azimuth, elevation, and slant range.

    Parameters
    ----------
    e_m, n_m, u_m : float
        East, north, and up coordinates in meters.

    Returns
    -------
    tuple[float, float, float]
        ``(az_rad, el_rad, srange_m)`` where azimuth is measured clockwise
        from north on ``[0, 2*pi)``, elevation is measured from the local
        horizon, and slant range is in meters.

    Notes
    -----
    The zero-range case returns ``(0.0, 0.0, 0.0)`` by convention.
    """

    if abs(e_m) < 1e-12:
        e_m = 0.0
    if abs(n_m) < 1e-12:
        n_m = 0.0
    if abs(u_m) < 1e-12:
        u_m = 0.0

    r_h = math.hypot(e_m, n_m)
    srange = math.hypot(r_h, u_m)

    if srange == 0.0:
        return 0.0, 0.0, 0.0

    el = math.atan2(u_m, r_h)
    az = math.atan2(e_m, n_m)
    if az < 0.0:
        az += 2.0 * math.pi

    return az, el, srange


@njit(cache=True, inline="always")
def _geodetic2aer_scalar(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar geodetic to AER kernel."""

    e, n, u = geodetic2enu(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)
    return enu2aer(e, n, u)


@njit(cache=True, parallel=True)
def _geodetic2aer_vector_llh(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector geodetic to AER kernel for split ``lat/lon/h`` arrays."""

    npts = lat_rad.shape[0]
    if lon_rad.shape[0] != npts or h_m.shape[0] != npts:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    az = np.empty(npts, dtype=np.float64)
    el = np.empty(npts, dtype=np.float64)
    sr = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        a, e, r = _geodetic2aer_scalar(
            lat_rad[i],
            lon_rad[i],
            h_m[i],
            lat0_rad,
            lon0_rad,
            h0_m,
        )
        az[i] = a
        el[i] = e
        sr[i] = r

    return az, el, sr


@typing_overload
def geodetic2aer(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def geodetic2aer(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def geodetic2aer(
    lat_rad: np.ndarray,
    lon_rad: None = None,
    h_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def geodetic2aer(
    lat_rad: float | np.ndarray,
    lon_rad: float | np.ndarray | None = None,
    h_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute azimuth, elevation, and slant range from an observer to geodetic targets.

    The observer location is given in geodetic coordinates
    ``(lat0_rad, lon0_rad, h0_m)``. The same observer is applied to every row
    when array input is used.

    Accepted input forms
    --------------------
    - ``geodetic2aer(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)``
    - ``geodetic2aer(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``geodetic2aer(lla_rad_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    lat_rad, lon_rad : float or np.ndarray
        Target geodetic latitude and longitude in radians. If ``lon_rad`` and
        ``h_m`` are omitted, ``lat_rad`` must be an array of shape ``(N, 3)``
        with columns ``[lat_rad, lon_rad, h_m]``.
    h_m : float or np.ndarray, optional
        Target height above the WGS84 ellipsoid in meters.
    lat0_rad, lon0_rad : float
        Observer geodetic latitude and longitude in radians.
    h0_m : float
        Observer height above the WGS84 ellipsoid in meters.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(az_rad, el_rad, srange_m)`` where azimuth is measured clockwise
        from north on ``[0, 2*pi)``, elevation is measured from the local
        horizon, and slant range is in meters. Scalar input returns three
        scalars. Array input returns three same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For array inputs, the observer remains scalar and is broadcast across
      all rows.

    Examples
    --------
    >>> az_rad, el_rad, srange_m = geodetic2aer(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)
    >>> az_rad, el_rad, srange_m = geodetic2aer(lla_rad_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if lon_rad is None and h_m is None:
        lla_rad_m = as_nx3_array(lat_rad, "lla_rad_m")
        return _geodetic2aer_vector_llh(
            lla_rad_m[:, 0],
            lla_rad_m[:, 1],
            lla_rad_m[:, 2],
            lat0_rad,
            lon0_rad,
            h0_m,
        )

    if lon_rad is None or h_m is None:
        raise TypeError("Provide either `lat_rad, lon_rad, h_m` or one `(N, 3)` array")

    if np.isscalar(lat_rad) and np.isscalar(lon_rad) and np.isscalar(h_m):
        return _geodetic2aer_scalar(
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
    return _geodetic2aer_vector_llh(lat_arr, lon_arr, h_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(geodetic2aer)
def _ol_geodetic2aer(
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
            return _geodetic2aer_scalar(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)

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
            return _geodetic2aer_vector_llh(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)

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
            return _geodetic2aer_vector_llh(
                lat_rad[:, 0],
                lat_rad[:, 1],
                lat_rad[:, 2],
                lat0_rad,
                lon0_rad,
                h0_m,
            )

        return impl

    return None

__all__ = [
    "enu2aer",
    "geodetic2aer",
]
