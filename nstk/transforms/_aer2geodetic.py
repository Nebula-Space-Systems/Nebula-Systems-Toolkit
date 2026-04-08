"""AER to WGS84 geodetic transforms."""

from __future__ import annotations

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
from ._aer2ecef import aer2ecef
from ._ecef2geodetic import ecef2geodetic


@njit(cache=True, inline="always")
def _aer2geodetic_scalar(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar AER to geodetic kernel."""

    x, y, z = aer2ecef(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)
    return ecef2geodetic(x, y, z)


@njit(cache=True, parallel=True)
def _aer2geodetic_vector_aer(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector AER to geodetic kernel for split ``az/el/range`` arrays."""

    npts = az_rad.shape[0]
    if el_rad.shape[0] != npts or srange_m.shape[0] != npts:
        raise ValueError("az_rad, el_rad, srange_m must have the same length")

    lat = np.empty(npts, dtype=np.float64)
    lon = np.empty(npts, dtype=np.float64)
    h = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        la, lo, hi = _aer2geodetic_scalar(
            az_rad[i],
            el_rad[i],
            srange_m[i],
            lat0_rad,
            lon0_rad,
            h0_m,
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def _aer2geodetic_vector_aer3(
    aer_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector AER to geodetic kernel for an ``(N, 3)`` input array."""

    if aer_rad_m.ndim != 2 or aer_rad_m.shape[1] != 3:
        raise ValueError("aer_rad_m must have shape (N, 3)")

    npts = aer_rad_m.shape[0]
    lat = np.empty(npts, dtype=np.float64)
    lon = np.empty(npts, dtype=np.float64)
    h = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        la, lo, hi = _aer2geodetic_scalar(
            aer_rad_m[i, 0],
            aer_rad_m[i, 1],
            aer_rad_m[i, 2],
            lat0_rad,
            lon0_rad,
            h0_m,
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@typing_overload
def aer2geodetic(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def aer2geodetic(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def aer2geodetic(
    az_rad: np.ndarray,
    el_rad: None = None,
    srange_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def aer2geodetic(
    az_rad: float | np.ndarray,
    el_rad: float | np.ndarray | None = None,
    srange_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert azimuth, elevation, and slant range to WGS84 geodetic coordinates.

    The observer location is given in geodetic coordinates
    ``(lat0_rad, lon0_rad, h0_m)``. The same observer is applied to every row
    when array input is used.

    Accepted input forms
    --------------------
    - ``aer2geodetic(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)``
    - ``aer2geodetic(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``aer2geodetic(aer_rad_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    az_rad, el_rad : float or np.ndarray
        Azimuth and elevation in radians. Azimuth is measured clockwise from
        north and elevation is measured from the local horizon. If ``el_rad``
        and ``srange_m`` are omitted, ``az_rad`` must be an array of shape
        ``(N, 3)`` with columns ``[az_rad, el_rad, srange_m]``.
    srange_m : float or np.ndarray, optional
        Slant range in meters.
    lat0_rad, lon0_rad : float
        Observer geodetic latitude and longitude in radians.
    h0_m : float
        Observer height above the WGS84 ellipsoid in meters.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(lat, lon, h)`` where latitude and longitude are in radians and
        height is in meters. Scalar input returns three scalars. Array input
        returns three same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For array inputs, the observer remains scalar and is broadcast across
      all rows.

    Examples
    --------
    >>> lat_rad, lon_rad, h_m = aer2geodetic(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)
    >>> lat_rad, lon_rad, h_m = aer2geodetic(aer_rad_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if el_rad is None and srange_m is None:
        aer_rad_m = as_nx3_array(az_rad, "aer_rad_m")
        return _aer2geodetic_vector_aer3(aer_rad_m, lat0_rad, lon0_rad, h0_m)

    if el_rad is None or srange_m is None:
        raise TypeError("Provide either `az_rad, el_rad, srange_m` or one `(N, 3)` array")

    if np.isscalar(az_rad) and np.isscalar(el_rad) and np.isscalar(srange_m):
        return _aer2geodetic_scalar(
            float(az_rad),
            float(el_rad),
            float(srange_m),
            float(lat0_rad),
            float(lon0_rad),
            float(h0_m),
        )

    az_arr = as_1d_array(az_rad, "az_rad")
    el_arr = as_1d_array(el_rad, "el_rad")
    sr_arr = as_1d_array(srange_m, "srange_m")
    validate_matching_lengths(("az_rad", az_arr), ("el_rad", el_arr), ("srange_m", sr_arr))
    return _aer2geodetic_vector_aer(az_arr, el_arr, sr_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(aer2geodetic)
def _ol_aer2geodetic(
    az_rad,
    el_rad=None,
    srange_m=None,
    lat0_rad=None,
    lon0_rad=None,
    h0_m=None,
):
    if (
        is_numba_scalar(az_rad)
        and is_numba_scalar(el_rad)
        and is_numba_scalar(srange_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(az_rad, el_rad=None, srange_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _aer2geodetic_scalar(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array1d(az_rad)
        and is_numba_array1d(el_rad)
        and is_numba_array1d(srange_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(az_rad, el_rad=None, srange_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _aer2geodetic_vector_aer(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array2d(az_rad)
        and is_numba_absent(el_rad)
        and is_numba_absent(srange_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(az_rad, el_rad=None, srange_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _aer2geodetic_vector_aer3(az_rad, lat0_rad, lon0_rad, h0_m)

        return impl

    return None

__all__ = [
    "aer2geodetic",
]
