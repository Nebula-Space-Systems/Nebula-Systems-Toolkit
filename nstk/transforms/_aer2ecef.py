"""AER to ECEF transforms."""

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
from ._enu2ecef import enu2ecef_delta


@njit(cache=True, inline="always")
def aer2enu(az_rad: float, el_rad: float, srange_m: float):
    """Convert azimuth, elevation, and slant range to local ENU coordinates.

    Parameters
    ----------
    az_rad, el_rad : float
        Azimuth and elevation in radians. Azimuth is measured clockwise from
        north and elevation is measured from the local horizon.
    srange_m : float
        Slant range in meters.

    Returns
    -------
    tuple[float, float, float]
        ``(e_m, n_m, u_m)`` east, north, and up coordinates in meters.

    Notes
    -----
    The zero-range case returns ``(0.0, 0.0, 0.0)`` by convention.
    """

    if srange_m == 0.0:
        return 0.0, 0.0, 0.0

    sel = math.sin(el_rad)
    cel = math.cos(el_rad)
    saz = math.sin(az_rad)
    caz = math.cos(az_rad)

    r = srange_m * cel
    e = r * saz
    n = r * caz
    u = srange_m * sel
    return e, n, u


@njit(cache=True, inline="always")
def _aer2ecef_scalar(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar AER to ECEF kernel."""

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    e, n, u = aer2enu(az_rad, el_rad, srange_m)
    dx, dy, dz = enu2ecef_delta(e, n, u, lat0_rad, lon0_rad)
    return x0 + dx, y0 + dy, z0 + dz


@njit(cache=True, parallel=True)
def _aer2ecef_vector_aer(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector AER to ECEF kernel for split ``az/el/range`` arrays."""

    npts = az_rad.shape[0]
    if el_rad.shape[0] != npts or srange_m.shape[0] != npts:
        raise ValueError("az_rad, el_rad, srange_m must have the same length")

    x = np.empty(npts, dtype=np.float64)
    y = np.empty(npts, dtype=np.float64)
    z = np.empty(npts, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(npts):
        az = az_rad[i]
        el = el_rad[i]
        sr = srange_m[i]

        if sr == 0.0:
            x[i] = x0
            y[i] = y0
            z[i] = z0
            continue

        sel = math.sin(el)
        cel = math.cos(el)
        saz = math.sin(az)
        caz = math.cos(az)

        r = sr * cel
        e = r * saz
        n = r * caz
        u = sr * sel

        dx = -slon * e - slat * clon * n + clat * clon * u
        dy = clon * e - slat * slon * n + clat * slon * u
        dz = clat * n + slat * u

        x[i] = x0 + dx
        y[i] = y0 + dy
        z[i] = z0 + dz

    return x, y, z


@njit(cache=True, parallel=True)
def _aer2ecef_vector_aer3(
    aer_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector AER to ECEF kernel for an ``(N, 3)`` input array."""

    if aer_rad_m.ndim != 2 or aer_rad_m.shape[1] != 3:
        raise ValueError("aer_rad_m must have shape (N, 3)")

    npts = aer_rad_m.shape[0]
    out = np.empty((npts, 3), dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(npts):
        az = aer_rad_m[i, 0]
        el = aer_rad_m[i, 1]
        sr = aer_rad_m[i, 2]

        if sr == 0.0:
            out[i, 0] = x0
            out[i, 1] = y0
            out[i, 2] = z0
            continue

        sel = math.sin(el)
        cel = math.cos(el)
        saz = math.sin(az)
        caz = math.cos(az)

        r = sr * cel
        e = r * saz
        n = r * caz
        u = sr * sel

        dx = -slon * e - slat * clon * n + clat * clon * u
        dy = clon * e - slat * slon * n + clat * slon * u
        dz = clat * n + slat * u

        out[i, 0] = x0 + dx
        out[i, 1] = y0 + dy
        out[i, 2] = z0 + dz

    return out


@typing_overload
def aer2ecef(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def aer2ecef(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def aer2ecef(
    az_rad: np.ndarray,
    el_rad: None = None,
    srange_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def aer2ecef(
    az_rad: float | np.ndarray,
    el_rad: float | np.ndarray | None = None,
    srange_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert azimuth, elevation, and slant range to absolute ECEF coordinates.

    The observer location is given in geodetic coordinates
    ``(lat0_rad, lon0_rad, h0_m)``. The same observer is applied to every row
    when array input is used.

    Accepted input forms
    --------------------
    - ``aer2ecef(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)``
    - ``aer2ecef(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``aer2ecef(aer_rad_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

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
        ``(x_m, y_m, z_m)`` ECEF coordinates in meters. Scalar input returns
        three scalars. Array input returns three same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For array inputs, the observer remains scalar and is broadcast across
      all rows.

    Examples
    --------
    >>> x_m, y_m, z_m = aer2ecef(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)
    >>> x_m, y_m, z_m = aer2ecef(aer_rad_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if el_rad is None and srange_m is None:
        aer_rad_m = as_nx3_array(az_rad, "aer_rad_m")
        out = _aer2ecef_vector_aer3(aer_rad_m, lat0_rad, lon0_rad, h0_m)
        return out[:, 0], out[:, 1], out[:, 2]

    if el_rad is None or srange_m is None:
        raise TypeError("Provide either `az_rad, el_rad, srange_m` or one `(N, 3)` array")

    if np.isscalar(az_rad) and np.isscalar(el_rad) and np.isscalar(srange_m):
        return _aer2ecef_scalar(
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
    return _aer2ecef_vector_aer(az_arr, el_arr, sr_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(aer2ecef)
def _ol_aer2ecef(
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
            return _aer2ecef_scalar(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)

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
            return _aer2ecef_vector_aer(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)

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
            out = _aer2ecef_vector_aer3(az_rad, lat0_rad, lon0_rad, h0_m)
            return out[:, 0], out[:, 1], out[:, 2]

        return impl

    return None

__all__ = [
    "aer2enu",
    "aer2ecef",
]
