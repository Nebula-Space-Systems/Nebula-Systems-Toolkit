"""
Coarse, fast ECI(native)-to-geodetic utilities.

These helpers reuse the approximate IAU-76 + truncated IAU-1980 + GAST
ECI(native)->ITRF chain and then convert ITRF/ECEF to WGS84 geodetic.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._coarse_eci2itrf import _coarse_eci2itrf_pos_iau76_shortnut
from ._ecef2geodetic import ecef2geodetic, ecef2geodetic_deg


@njit(cache=False, fastmath=True)
def _coarse_eci2geodetic_iau76_shortnut(
    x_eci: float,
    y_eci: float,
    z_eci: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Approximate ECI(native)->WGS84 geodetic (radians).
    """
    x_ecef, y_ecef, z_ecef = _coarse_eci2itrf_pos_iau76_shortnut(
        x_eci, y_eci, z_eci, jd_ut1, jd_tt, xp_rad, yp_rad
    )
    return ecef2geodetic(x_ecef, y_ecef, z_ecef)


@njit(cache=False, fastmath=True)
def _coarse_eci2geodetic_deg_iau76_shortnut(
    x_eci: float,
    y_eci: float,
    z_eci: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Approximate ECI(native)->WGS84 geodetic (degrees).
    """
    x_ecef, y_ecef, z_ecef = _coarse_eci2itrf_pos_iau76_shortnut(
        x_eci, y_eci, z_eci, jd_ut1, jd_tt, xp_rad, yp_rad
    )
    return ecef2geodetic_deg(x_ecef, y_ecef, z_ecef)


@njit(cache=False, fastmath=True, parallel=True)
def _coarse_eci2geodetic_vec_iau76_shortnut(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Vectorized approximate ECI(native)->WGS84 geodetic (radians).
    """
    if r_eci_m.ndim != 2 or r_eci_m.shape[1] != 3:
        raise ValueError("r_eci_m must have shape (N, 3)")
    n = r_eci_m.shape[0]
    if jd_ut1.shape[0] != n or jd_tt.shape[0] != n:
        raise ValueError("jd_ut1 and jd_tt must have shape (N,)")

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = _coarse_eci2geodetic_iau76_shortnut(
            r_eci_m[i, 0],
            r_eci_m[i, 1],
            r_eci_m[i, 2],
            jd_ut1[i],
            jd_tt[i],
            xp_rad,
            yp_rad,
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=False, fastmath=True, parallel=True)
def _coarse_eci2geodetic_vec_deg_iau76_shortnut(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    wrap_lon: bool = True,
):
    """
    Vectorized approximate ECI(native)->WGS84 geodetic (degrees).
    """
    lat_rad, lon_rad, h = _coarse_eci2geodetic_vec_iau76_shortnut(
        r_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad
    )
    n = lat_rad.shape[0]

    lat_deg = np.empty(n, dtype=np.float64)
    lon_deg = np.empty(n, dtype=np.float64)
    rad2deg = 180.0 / math.pi

    for i in prange(n):
        lat_deg[i] = lat_rad[i] * rad2deg
        lon_deg[i] = lon_rad[i] * rad2deg

    if wrap_lon:
        for i in range(n):
            x = lon_deg[i] + 180.0
            x = x - 360.0 * math.floor(x / 360.0)
            lon_deg[i] = x - 180.0

    return lat_deg, lon_deg, h


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci2geodetic(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform one ECI position to WGS84 geodetic coordinates.

    Parameters
    ----------
    x_eci_m, y_eci_m, z_eci_m : float
        ECI position components [m] in the native inertial basis used by this model.
    jd_ut1 : float
        UT1 Julian date.
    jd_tt : float
        TT Julian date.
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (lat_rad, lon_rad, h_m) : tuple[float, float, float]
        Geodetic latitude [rad], longitude [rad], and ellipsoidal height [m].
    """
    return _coarse_eci2geodetic_iau76_shortnut(
        x_eci_m, y_eci_m, z_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad
    )


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci2geodetic_deg(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform one ECI position to WGS84 geodetic coordinates with degree angles.

    Parameters
    ----------
    x_eci_m, y_eci_m, z_eci_m : float
        ECI position components [m].
    jd_ut1 : float
        UT1 Julian date.
    jd_tt : float
        TT Julian date.
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (lat_deg, lon_deg, h_m) : tuple[float, float, float]
        Geodetic latitude [deg], longitude [deg], and ellipsoidal height [m].
    """
    return _coarse_eci2geodetic_deg_iau76_shortnut(
        x_eci_m, y_eci_m, z_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad
    )


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci2geodetic_vec(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform a batch of ECI positions to WGS84 geodetic coordinates.

    Parameters
    ----------
    r_eci_m : np.ndarray
        ECI positions with shape (N, 3), meters.
    jd_ut1 : np.ndarray
        UT1 Julian dates with shape (N,).
    jd_tt : np.ndarray
        TT Julian dates with shape (N,).
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (lat_rad, lon_rad, h_m) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of geodetic latitude [rad], longitude [rad], and height [m], each shape (N,).
    """
    return _coarse_eci2geodetic_vec_iau76_shortnut(
        r_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad
    )


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci2geodetic_vec_deg(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    wrap_lon: bool = True,
):
    """
    Transform a batch of ECI positions to WGS84 geodetic coordinates with degree angles.

    Parameters
    ----------
    r_eci_m : np.ndarray
        ECI positions with shape (N, 3), meters.
    jd_ut1 : np.ndarray
        UT1 Julian dates with shape (N,).
    jd_tt : np.ndarray
        TT Julian dates with shape (N,).
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.
    wrap_lon : bool, optional
        If True, wraps longitude to [-180, 180). Defaults to True.

    Returns
    -------
    (lat_deg, lon_deg, h_m) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of geodetic latitude [deg], longitude [deg], and height [m], each shape (N,).
    """
    return _coarse_eci2geodetic_vec_deg_iau76_shortnut(
        r_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad, wrap_lon
    )


__all__ = [
    "coarse_eci2geodetic",
    "coarse_eci2geodetic_deg",
    "coarse_eci2geodetic_vec",
    "coarse_eci2geodetic_vec_deg",
]
