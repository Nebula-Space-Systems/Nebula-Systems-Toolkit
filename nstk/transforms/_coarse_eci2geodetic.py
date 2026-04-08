"""Coarse, fast ECI(native)-to-geodetic utilities."""

from __future__ import annotations

from typing import overload as typing_overload

import numpy as np
from numba import njit, prange
from numba.extending import overload as numba_overload

from ._api_utils import (
    as_nx3_array,
    is_numba_absent,
    is_numba_array2d,
    is_numba_scalar,
    require_not_none,
)
from ._coarse_eci2itrf import _coarse_eci2ecef_pos_iau76_shortnut
from ._ecef2geodetic import ecef2geodetic


@njit(cache=False, fastmath=True)
def _coarse_eci2geodetic_scalar(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    degrees: bool = False,
):
    """Approximate scalar ECI(native) to WGS84 geodetic kernel."""

    x_ecef, y_ecef, z_ecef = _coarse_eci2ecef_pos_iau76_shortnut(
        x_eci_m,
        y_eci_m,
        z_eci_m,
        jd_ut1,
        jd_tt,
        xp_rad,
        yp_rad,
    )
    return ecef2geodetic(x_ecef, y_ecef, z_ecef, degrees=degrees)


@njit(cache=False, fastmath=True, parallel=True)
def _coarse_eci2geodetic_vector(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    degrees: bool = False,
):
    """Approximate vector ECI(native) to WGS84 geodetic kernel."""

    if r_eci_m.ndim != 2 or r_eci_m.shape[1] != 3:
        raise ValueError("r_eci_m must have shape (N, 3)")

    n = r_eci_m.shape[0]
    if jd_ut1.shape[0] != n or jd_tt.shape[0] != n:
        raise ValueError("jd_ut1 and jd_tt must have shape (N,)")

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = _coarse_eci2geodetic_scalar(
            r_eci_m[i, 0],
            r_eci_m[i, 1],
            r_eci_m[i, 2],
            jd_ut1[i],
            jd_tt[i],
            xp_rad,
            yp_rad,
            degrees,
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@typing_overload
def coarse_eci2geodetic(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    degrees: bool = False,
) -> tuple[float, float, float]:
    ...


@typing_overload
def coarse_eci2geodetic(
    x_eci_m: np.ndarray,
    y_eci_m: None = None,
    z_eci_m: None = None,
    jd_ut1: np.ndarray | None = None,
    jd_tt: np.ndarray | None = None,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    degrees: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def coarse_eci2geodetic(
    x_eci_m: float | np.ndarray,
    y_eci_m: float | None = None,
    z_eci_m: float | None = None,
    jd_ut1: float | np.ndarray | None = None,
    jd_tt: float | np.ndarray | None = None,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    degrees: bool = False,
) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate native ECI coordinates as WGS84 geodetic coordinates.

    This is a fast, coarse transform intended for workflows where a lightweight
    Earth-orientation model is sufficient. It uses the package's coarse
    ECI-to-ECEF rotation path and then converts the result to WGS84 geodetic
    latitude, longitude, and height.

    Accepted input forms
    --------------------
    - ``coarse_eci2geodetic(x_eci_m, y_eci_m, z_eci_m, jd_ut1, jd_tt)``
    - ``coarse_eci2geodetic(r_eci_m, jd_ut1=..., jd_tt=...)`` for an ``(N, 3)`` array

    Parameters
    ----------
    x_eci_m, y_eci_m, z_eci_m : float or np.ndarray
        Native ECI position coordinates in meters. If ``y_eci_m`` and
        ``z_eci_m`` are omitted, ``x_eci_m`` must be an array of shape
        ``(N, 3)``.
    jd_ut1, jd_tt : float or np.ndarray
        UT1 and TT Julian dates corresponding to each sample. Scalar input
        expects scalar dates. Array input expects 1D arrays of length ``N``.
    xp_rad, yp_rad : float, default=0.0
        Optional polar-motion corrections in radians.
    degrees : bool, default=False
        If ``True``, latitude and longitude are returned in degrees.
        Otherwise they are returned in radians.

    Returns
    -------
    tuple[float, float, float] or tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(lat, lon, h)`` geodetic coordinates, with height in meters.
        Scalar input returns three scalars. Array input returns three
        same-length 1D arrays.

    Notes
    -----
    - This interface is available inside ``@numba.njit`` for both scalar and
      ``(N, 3)`` array inputs.
    - Use the precise timed frame transforms when you need higher-fidelity
      Earth-orientation modeling.

    Examples
    --------
    >>> lat_rad, lon_rad, h_m = coarse_eci2geodetic(x_eci_m, y_eci_m, z_eci_m, jd_ut1, jd_tt)
    >>> lat_deg, lon_deg, h_m = coarse_eci2geodetic(r_eci_m, jd_ut1=jd_ut1, jd_tt=jd_tt, degrees=True)
    """

    jd_ut1 = require_not_none(jd_ut1, "jd_ut1")
    jd_tt = require_not_none(jd_tt, "jd_tt")

    if y_eci_m is None and z_eci_m is None:
        r_eci_m = as_nx3_array(x_eci_m, "r_eci_m")
        jd_ut1_arr = np.asarray(jd_ut1)
        jd_tt_arr = np.asarray(jd_tt)
        return _coarse_eci2geodetic_vector(
            r_eci_m,
            jd_ut1_arr,
            jd_tt_arr,
            xp_rad,
            yp_rad,
            degrees,
        )

    if y_eci_m is None or z_eci_m is None:
        raise TypeError("Provide either `x_eci_m, y_eci_m, z_eci_m` or one `(N, 3)` array")

    return _coarse_eci2geodetic_scalar(
        float(x_eci_m),
        float(y_eci_m),
        float(z_eci_m),
        float(jd_ut1),
        float(jd_tt),
        float(xp_rad),
        float(yp_rad),
        bool(degrees),
    )


@numba_overload(coarse_eci2geodetic)
def _ol_coarse_eci2geodetic(
    x_eci_m,
    y_eci_m=None,
    z_eci_m=None,
    jd_ut1=None,
    jd_tt=None,
    xp_rad=0.0,
    yp_rad=0.0,
    degrees=False,
):
    if (
        is_numba_scalar(x_eci_m)
        and is_numba_scalar(y_eci_m)
        and is_numba_scalar(z_eci_m)
        and is_numba_scalar(jd_ut1)
        and is_numba_scalar(jd_tt)
        and (is_numba_scalar(xp_rad) or is_numba_absent(xp_rad))
        and (is_numba_scalar(yp_rad) or is_numba_absent(yp_rad))
    ):

        def impl(
            x_eci_m,
            y_eci_m=None,
            z_eci_m=None,
            jd_ut1=None,
            jd_tt=None,
            xp_rad=0.0,
            yp_rad=0.0,
            degrees=False,
        ):
            return _coarse_eci2geodetic_scalar(
                x_eci_m,
                y_eci_m,
                z_eci_m,
                jd_ut1,
                jd_tt,
                xp_rad,
                yp_rad,
                degrees,
            )

        return impl

    if (
        is_numba_array2d(x_eci_m)
        and is_numba_absent(y_eci_m)
        and is_numba_absent(z_eci_m)
    ):

        def impl(
            x_eci_m,
            y_eci_m=None,
            z_eci_m=None,
            jd_ut1=None,
            jd_tt=None,
            xp_rad=0.0,
            yp_rad=0.0,
            degrees=False,
        ):
            return _coarse_eci2geodetic_vector(
                x_eci_m,
                jd_ut1,
                jd_tt,
                xp_rad,
                yp_rad,
                degrees,
            )

        return impl

    return None

__all__ = [
    "coarse_eci2geodetic",
]
