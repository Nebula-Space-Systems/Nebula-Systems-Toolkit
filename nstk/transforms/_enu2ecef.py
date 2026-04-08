"""Local ENU to ECEF transforms."""

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
def enu2ecef_delta(
    e_m: float,
    n_m: float,
    u_m: float,
    lat0_rad: float,
    lon0_rad: float,
):
    """Rotate a local ENU delta vector into the ECEF basis.

    Parameters
    ----------
    e_m, n_m, u_m : float
        East, north, and up components in meters.
    lat0_rad, lon0_rad : float
        Geodetic latitude and longitude of the ENU frame origin, in radians.

    Returns
    -------
    tuple[float, float, float]
        ``(dx_m, dy_m, dz_m)`` delta vector components in the ECEF basis,
        in meters.

    Notes
    -----
    This function rotates only a delta vector. It does not add the observer
    position. Use :func:`enu2ecef` when you want absolute ECEF coordinates.
    """

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    dx = -slon * e_m - slat * clon * n_m + clat * clon * u_m
    dy = clon * e_m - slat * slon * n_m + clat * slon * u_m
    dz = clat * n_m + slat * u_m
    return dx, dy, dz


@njit(cache=True, inline="always")
def _enu2ecef_scalar(
    e_m: float,
    n_m: float,
    u_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Scalar ENU to ECEF kernel."""

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    dx, dy, dz = enu2ecef_delta(e_m, n_m, u_m, lat0_rad, lon0_rad)
    return x0 + dx, y0 + dy, z0 + dz


@njit(cache=True, parallel=True)
def _enu2ecef_vector_enu(
    e_m: np.ndarray,
    n_m: np.ndarray,
    u_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector ENU to ECEF kernel for split ``e/n/u`` arrays."""

    n = e_m.shape[0]
    if n_m.shape[0] != n or u_m.shape[0] != n:
        raise ValueError("e_m, n_m, u_m must have the same length")

    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        ei = e_m[i]
        ni = n_m[i]
        ui = u_m[i]

        dx = -slon * ei - slat * clon * ni + clat * clon * ui
        dy = clon * ei - slat * slon * ni + clat * slon * ui
        dz = clat * ni + slat * ui

        x[i] = x0 + dx
        y[i] = y0 + dy
        z[i] = z0 + dz

    return x, y, z


@njit(cache=True, parallel=True)
def _enu2ecef_vector_enu3(
    enu_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """Vector ENU to ECEF kernel for an ``(N, 3)`` input array."""

    if enu_m.ndim != 2 or enu_m.shape[1] != 3:
        raise ValueError("enu_m must have shape (N, 3)")

    n = enu_m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        ei = enu_m[i, 0]
        ni = enu_m[i, 1]
        ui = enu_m[i, 2]

        dx = -slon * ei - slat * clon * ni + clat * clon * ui
        dy = clon * ei - slat * slon * ni + clat * slon * ui
        dz = clat * ni + slat * ui

        out[i, 0] = x0 + dx
        out[i, 1] = y0 + dy
        out[i, 2] = z0 + dz

    return out


@typing_overload
def enu2ecef(
    e_m: float,
    n_m: float,
    u_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[float, float, float]:
    ...


@typing_overload
def enu2ecef(
    e_m: np.ndarray,
    n_m: np.ndarray,
    u_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@typing_overload
def enu2ecef(
    e_m: np.ndarray,
    n_m: None = None,
    u_m: None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def enu2ecef(
    e_m: float | np.ndarray,
    n_m: float | np.ndarray | None = None,
    u_m: float | np.ndarray | None = None,
    lat0_rad: float | None = None,
    lon0_rad: float | None = None,
    h0_m: float | None = None,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local ENU coordinates to absolute ECEF coordinates.

    The ENU frame is defined by the geodetic reference location
    ``(lat0_rad, lon0_rad, h0_m)``. The same reference point is applied to
    every row when array input is used.

    Accepted input forms
    --------------------
    - ``enu2ecef(e_m, n_m, u_m, lat0_rad, lon0_rad, h0_m)``
    - ``enu2ecef(e_m, n_m, u_m, lat0_rad, lon0_rad, h0_m)`` for matching 1D arrays
    - ``enu2ecef(enu_m, lat0_rad=..., lon0_rad=..., h0_m=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    e_m, n_m, u_m : float or np.ndarray
        Local east, north, and up coordinates in meters. If ``n_m`` and
        ``u_m`` are omitted, ``e_m`` must be an array of shape ``(N, 3)``
        with columns ``[e_m, n_m, u_m]``.
    lat0_rad, lon0_rad : float
        Geodetic latitude and longitude of the ENU origin in radians.
    h0_m : float
        Height of the ENU origin above the WGS84 ellipsoid in meters.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x_m, y_m, z_m)`` ECEF coordinates in meters. Scalar input returns
        three scalars. Array input returns three same-length 1D arrays.

    Notes
    -----
    - The same scalar, split-array, and ``(N, 3)`` forms work inside
      ``@numba.njit`` callers.
    - For array inputs, the observer origin remains scalar and is broadcast
      across all rows.

    Examples
    --------
    >>> x_m, y_m, z_m = enu2ecef(e_m, n_m, u_m, lat0_rad, lon0_rad, h0_m)
    >>> x_m, y_m, z_m = enu2ecef(enu_m, lat0_rad=lat0_rad, lon0_rad=lon0_rad, h0_m=h0_m)
    """

    lat0_rad = require_not_none(lat0_rad, "lat0_rad")
    lon0_rad = require_not_none(lon0_rad, "lon0_rad")
    h0_m = require_not_none(h0_m, "h0_m")

    if n_m is None and u_m is None:
        enu_m = as_nx3_array(e_m, "enu_m")
        out = _enu2ecef_vector_enu3(enu_m, lat0_rad, lon0_rad, h0_m)
        return out[:, 0], out[:, 1], out[:, 2]

    if n_m is None or u_m is None:
        raise TypeError("Provide either `e_m, n_m, u_m` or one `(N, 3)` array")

    if np.isscalar(e_m) and np.isscalar(n_m) and np.isscalar(u_m):
        return _enu2ecef_scalar(
            float(e_m),
            float(n_m),
            float(u_m),
            float(lat0_rad),
            float(lon0_rad),
            float(h0_m),
        )

    e_arr = as_1d_array(e_m, "e_m")
    n_arr = as_1d_array(n_m, "n_m")
    u_arr = as_1d_array(u_m, "u_m")
    validate_matching_lengths(("e_m", e_arr), ("n_m", n_arr), ("u_m", u_arr))
    return _enu2ecef_vector_enu(e_arr, n_arr, u_arr, lat0_rad, lon0_rad, h0_m)


@numba_overload(enu2ecef)
def _ol_enu2ecef(
    e_m,
    n_m=None,
    u_m=None,
    lat0_rad=None,
    lon0_rad=None,
    h0_m=None,
):
    if (
        is_numba_scalar(e_m)
        and is_numba_scalar(n_m)
        and is_numba_scalar(u_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(e_m, n_m=None, u_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _enu2ecef_scalar(e_m, n_m, u_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array1d(e_m)
        and is_numba_array1d(n_m)
        and is_numba_array1d(u_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(e_m, n_m=None, u_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            return _enu2ecef_vector_enu(e_m, n_m, u_m, lat0_rad, lon0_rad, h0_m)

        return impl

    if (
        is_numba_array2d(e_m)
        and is_numba_absent(n_m)
        and is_numba_absent(u_m)
        and is_numba_scalar(lat0_rad)
        and is_numba_scalar(lon0_rad)
        and is_numba_scalar(h0_m)
    ):

        def impl(e_m, n_m=None, u_m=None, lat0_rad=None, lon0_rad=None, h0_m=None):
            out = _enu2ecef_vector_enu3(e_m, lat0_rad, lon0_rad, h0_m)
            return out[:, 0], out[:, 1], out[:, 2]

        return impl

    return None

__all__ = [
    "enu2ecef_delta",
    "enu2ecef",
]
