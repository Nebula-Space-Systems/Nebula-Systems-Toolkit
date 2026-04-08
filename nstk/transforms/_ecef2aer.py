"""ECEF to AER transforms."""

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
from ._ecef2enu import ecef2enu
from ._geodetic2aer import enu2aer


@njit(cache=True, inline="always")
def _ecef2aer_scalar(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar ECEF to AER kernel."""

    e, n, u = ecef2enu(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)
    return enu2aer(e, n, u)


@njit(cache=True, parallel=True)
def _ecef2aer_vector_xyz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector ECEF to AER kernel for split ``x/y/z`` arrays."""

    npts = x_m.shape[0]
    if y_m.shape[0] != npts or z_m.shape[0] != npts:
        raise ValueError("x_m, y_m, z_m must have the same length")

    az = np.empty(npts, dtype=np.float64)
    el = np.empty(npts, dtype=np.float64)
    sr = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        a, e, r = _ecef2aer_scalar(x_m[i], y_m[i], z_m[i], lat0_rad, lon0_rad, h0_m)
        az[i] = a
        el[i] = e
        sr[i] = r

    return az, el, sr


@typing_overload
def ecef2aer(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def ecef2aer(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def ecef2aer(
    x_m: np.ndarray,
    y_m: None = None,
    z_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def ecef2aer(
    x_m: float | np.ndarray,
    y_m: float | np.ndarray | None = None,
    z_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute azimuth, elevation, and slant range from an observer to ECEF targets.

    The observer location is given in geodetic coordinates
    ``(lat0_rad, lon0_rad, h0_m)``. The same observer is applied to every row
    when array input is used.

    Accepted input forms
    --------------------
    - ``ecef2aer(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)``
    - ``ecef2aer(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``ecef2aer(r_ecef_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    x_m, y_m, z_m : float or np.ndarray
        Absolute ECEF target coordinates in meters. If ``y_m`` and ``z_m``
        are omitted, ``x_m`` must be an array of shape ``(N, 3)``.
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
    >>> az_rad, el_rad, srange_m = ecef2aer(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)
    >>> az_rad, el_rad, srange_m = ecef2aer(r_ecef_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if y_m is None and z_m is None:
        r_ecef_m = as_nx3_array(x_m, "r_ecef_m")
        return _ecef2aer_vector_xyz(
            r_ecef_m[:, 0],
            r_ecef_m[:, 1],
            r_ecef_m[:, 2],
            lat0_rad,
            lon0_rad,
            h0_m,
        )

    if y_m is None or z_m is None:
        raise TypeError("Provide either `x_m, y_m, z_m` or one `(N, 3)` array")

    if np.isscalar(x_m) and np.isscalar(y_m) and np.isscalar(z_m):
        return _ecef2aer_scalar(
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
    return _ecef2aer_vector_xyz(x_arr, y_arr, z_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(ecef2aer)
def _ol_ecef2aer(
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
            return _ecef2aer_scalar(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)

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
            return _ecef2aer_vector_xyz(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)

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
            return _ecef2aer_vector_xyz(
                x_m[:, 0],
                x_m[:, 1],
                x_m[:, 2],
                lat0_rad,
                lon0_rad,
                h0_m,
            )

        return impl

    return None

__all__ = [
    "ecef2aer",
]
