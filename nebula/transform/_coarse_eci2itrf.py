"""
Coarse, fast ECI(native)-to-ITRF rotation utilities shared by propagation code.

This module centralizes the approximate transform chain used by fast propagation:
IAU-76 precession + truncated IAU-1980 nutation + GAST rotation, with optional
epoch polar motion.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

J2000_JD = 2451545.0
EARTH_OMEGA = 7.2921150e-5  # rad/s
DAS2R = math.pi / (180.0 * 3600.0)


@njit(cache=False, fastmath=True, inline="always")
def _coarse_mod2pi_rad(x: float) -> float:
    twopi = 2.0 * math.pi
    return x - twopi * math.floor(x / twopi)


@njit(cache=False, fastmath=True)
def _coarse_mean_obliquity_iau1980_rad(T: float) -> float:
    t2 = T * T
    t3 = t2 * T
    eps0_as = 84381.448 - 46.8150 * T - 0.00059 * t2 + 0.001813 * t3
    return eps0_as * DAS2R


@njit(cache=False, fastmath=True)
def _coarse_nutation_short_rad(jd_tt: float):
    """
    10-term truncated IAU-1980 nutation used by fast-orbit transforms.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    t2 = T * T
    t3 = t2 * T

    D = math.radians(297.85036 + 445267.111480 * T - 0.0019142 * t2 + t3 / 189474.0)
    M = math.radians(357.52772 + 35999.050340 * T - 0.0001603 * t2 - t3 / 300000.0)
    Mp = math.radians(134.96298 + 477198.867398 * T + 0.0086972 * t2 + t3 / 56250.0)
    F = math.radians(93.27191 + 483202.017538 * T - 0.0036825 * t2 + t3 / 327270.0)
    om = math.radians(125.04452 - 1934.136261 * T + 0.0020708 * t2 + t3 / 450000.0)

    D = _coarse_mod2pi_rad(D)
    M = _coarse_mod2pi_rad(M)
    Mp = _coarse_mod2pi_rad(Mp)
    F = _coarse_mod2pi_rad(F)
    om = _coarse_mod2pi_rad(om)

    arg1 = om
    arg2 = 2.0 * Mp - 2.0 * F + 2.0 * om
    arg3 = 2.0 * Mp + 2.0 * om
    arg4 = 2.0 * om
    arg5 = M
    arg6 = M + 2.0 * Mp - 2.0 * F + 2.0 * om
    arg7 = 2.0 * Mp + om
    arg8 = M + om
    arg9 = D
    arg10 = M - 2.0 * Mp + 2.0 * F

    dpsi_1e4as = (
        (-171996.0 - 174.2 * T) * math.sin(arg1)
        + (-13187.0 - 1.6 * T) * math.sin(arg2)
        + (-2274.0 - 0.2 * T) * math.sin(arg3)
        + (2062.0 + 0.2 * T) * math.sin(arg4)
        + (1426.0 - 3.4 * T) * math.sin(arg5)
        + (712.0 + 0.1 * T) * math.sin(arg6)
        + (-517.0 + 1.2 * T) * math.sin(arg7)
        + (-386.0 - 0.4 * T) * math.sin(arg8)
        + (-301.0 + 0.0 * T) * math.sin(arg9)
        + (217.0 - 0.5 * T) * math.sin(arg10)
    )
    deps_1e4as = (
        (92025.0 + 8.9 * T) * math.cos(arg1)
        + (5736.0 - 3.1 * T) * math.cos(arg2)
        + (977.0 - 0.5 * T) * math.cos(arg3)
        + (-895.0 + 0.5 * T) * math.cos(arg4)
        + (54.0 - 0.1 * T) * math.cos(arg5)
        + (-7.0 + 0.0 * T) * math.cos(arg6)
        + (224.0 - 0.6 * T) * math.cos(arg7)
        + (200.0 + 0.0 * T) * math.cos(arg8)
        + (0.0 + 0.0 * T) * math.cos(arg9)
        + (-5.0 + 0.0 * T) * math.cos(arg10)
    )

    conv = DAS2R * 1.0e-4
    return dpsi_1e4as * conv, deps_1e4as * conv, om, T


@njit(cache=False, fastmath=True)
def _coarse_gmst_vallado_rad(jd_ut1: float) -> float:
    d = jd_ut1 - J2000_JD
    T = d / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    gmst_deg = gmst_deg - 360.0 * math.floor(gmst_deg / 360.0)
    return gmst_deg * (math.pi / 180.0)


@njit(cache=False, fastmath=True, inline="always")
def _coarse_rot1_cs(x: float, y: float, z: float, c: float, s: float):
    return x, c * y + s * z, -s * y + c * z


@njit(cache=False, fastmath=True, inline="always")
def _coarse_rot2_cs(x: float, y: float, z: float, c: float, s: float):
    return c * x - s * z, y, s * x + c * z


@njit(cache=False, fastmath=True, inline="always")
def _coarse_rot3_cs(x: float, y: float, z: float, c: float, s: float):
    return c * x + s * y, -s * x + c * y, z


@njit(cache=False, fastmath=True)
def _coarse_tod_to_native_pos_iau76_shortnut(
    x_tod: float, y_tod: float, z_tod: float, jd_tt: float
):
    """
    Convert true-of-date coordinates to the native inertial frame basis.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    t2 = T * T
    t3 = t2 * T

    zeta = (2306.2181 * T + 0.30188 * t2 + 0.017998 * t3) * DAS2R
    theta = (2004.3109 * T - 0.42665 * t2 - 0.041833 * t3) * DAS2R
    zang = (2306.2181 * T + 1.09468 * t2 + 0.018203 * t3) * DAS2R

    dpsi, deps, _om, _ = _coarse_nutation_short_rad(jd_tt)
    eps0 = _coarse_mean_obliquity_iau1980_rad(T)
    eps = eps0 + deps

    c = math.cos(eps)
    s = math.sin(eps)
    x, y, z = _coarse_rot1_cs(x_tod, y_tod, z_tod, c, s)

    c = math.cos(dpsi)
    s = math.sin(dpsi)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    c = math.cos(-eps0)
    s = math.sin(-eps0)
    x, y, z = _coarse_rot1_cs(x, y, z, c, s)

    c = math.cos(zang)
    s = math.sin(zang)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    c = math.cos(-theta)
    s = math.sin(-theta)
    x, y, z = _coarse_rot2_cs(x, y, z, c, s)

    c = math.cos(zeta)
    s = math.sin(zeta)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    return x, y, z


@njit(cache=False, fastmath=True)
def _coarse_j2_axis_native_iau76_shortnut(jd_tt: float):
    """
    Earth's figure/rotation axis expressed in native inertial coordinates.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    t2 = T * T
    t3 = t2 * T

    zeta = (2306.2181 * T + 0.30188 * t2 + 0.017998 * t3) * DAS2R
    theta = (2004.3109 * T - 0.42665 * t2 - 0.041833 * t3) * DAS2R
    zang = (2306.2181 * T + 1.09468 * t2 + 0.018203 * t3) * DAS2R

    dpsi, deps, _om, _ = _coarse_nutation_short_rad(jd_tt)
    eps0 = _coarse_mean_obliquity_iau1980_rad(T)
    eps = eps0 + deps

    x, y, z0 = 0.0, 0.0, 1.0

    c = math.cos(eps)
    s = math.sin(eps)
    x, y, z0 = _coarse_rot1_cs(x, y, z0, c, s)

    c = math.cos(dpsi)
    s = math.sin(dpsi)
    x, y, z0 = _coarse_rot3_cs(x, y, z0, c, s)

    c = math.cos(-eps0)
    s = math.sin(-eps0)
    x, y, z0 = _coarse_rot1_cs(x, y, z0, c, s)

    c = math.cos(zang)
    s = math.sin(zang)
    x, y, z0 = _coarse_rot3_cs(x, y, z0, c, s)

    c = math.cos(-theta)
    s = math.sin(-theta)
    x, y, z0 = _coarse_rot2_cs(x, y, z0, c, s)

    c = math.cos(zeta)
    s = math.sin(zeta)
    x, y, z0 = _coarse_rot3_cs(x, y, z0, c, s)

    n = math.sqrt(x * x + y * y + z0 * z0)
    return x / n, y / n, z0 / n


@njit(cache=False, fastmath=True)
def _coarse_eci_to_itrf_pos_iau76_shortnut(
    x_eci: float,
    y_eci: float,
    z_eci: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Approximate ECI(native)->ITRF transform for positions.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    t2 = T * T
    t3 = t2 * T

    zeta = (2306.2181 * T + 0.30188 * t2 + 0.017998 * t3) * DAS2R
    theta = (2004.3109 * T - 0.42665 * t2 - 0.041833 * t3) * DAS2R
    zang = (2306.2181 * T + 1.09468 * t2 + 0.018203 * t3) * DAS2R

    c = math.cos(zeta)
    s = -math.sin(zeta)
    x, y, z = _coarse_rot3_cs(x_eci, y_eci, z_eci, c, s)

    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = _coarse_rot2_cs(x, y, z, c, s)

    c = math.cos(zang)
    s = -math.sin(zang)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    dpsi, deps, om, _ = _coarse_nutation_short_rad(jd_tt)
    eps0 = _coarse_mean_obliquity_iau1980_rad(T)
    eps = eps0 + deps

    c = math.cos(eps0)
    s = math.sin(eps0)
    x, y, z = _coarse_rot1_cs(x, y, z, c, s)

    c = math.cos(dpsi)
    s = -math.sin(dpsi)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    c = math.cos(eps)
    s = -math.sin(eps)
    x, y, z = _coarse_rot1_cs(x, y, z, c, s)

    gmst = _coarse_gmst_vallado_rad(jd_ut1)
    eqeq = dpsi * math.cos(eps) + (0.00264 * math.sin(om) + 0.000063 * math.sin(2.0 * om)) * DAS2R
    gast = _coarse_mod2pi_rad(gmst + eqeq)

    c = math.cos(gast)
    s = math.sin(gast)
    x, y, z = _coarse_rot3_cs(x, y, z, c, s)

    if xp_rad != 0.0 or yp_rad != 0.0:
        c = math.cos(-yp_rad)
        s = math.sin(-yp_rad)
        x, y, z = _coarse_rot1_cs(x, y, z, c, s)

        c = math.cos(-xp_rad)
        s = math.sin(-xp_rad)
        x, y, z = _coarse_rot2_cs(x, y, z, c, s)

    return x, y, z


@njit(cache=False, fastmath=True)
def _coarse_eci_to_itrf_pv_iau76_shortnut(
    rx: float,
    ry: float,
    rz: float,
    vx: float,
    vy: float,
    vz: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float,
    yp_rad: float,
):
    """
    Approximate ECI(native)->ITRF transform for position and velocity.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    t2 = T * T
    t3 = t2 * T

    zeta = (2306.2181 * T + 0.30188 * t2 + 0.017998 * t3) * DAS2R
    theta = (2004.3109 * T - 0.42665 * t2 - 0.041833 * t3) * DAS2R
    zang = (2306.2181 * T + 1.09468 * t2 + 0.018203 * t3) * DAS2R

    c = math.cos(zeta)
    s = -math.sin(zeta)
    rx, ry, rz = _coarse_rot3_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot3_cs(vx, vy, vz, c, s)

    c = math.cos(theta)
    s = math.sin(theta)
    rx, ry, rz = _coarse_rot2_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot2_cs(vx, vy, vz, c, s)

    c = math.cos(zang)
    s = -math.sin(zang)
    rx, ry, rz = _coarse_rot3_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot3_cs(vx, vy, vz, c, s)

    dpsi, deps, om, _ = _coarse_nutation_short_rad(jd_tt)
    eps0 = _coarse_mean_obliquity_iau1980_rad(T)
    eps = eps0 + deps

    c = math.cos(eps0)
    s = math.sin(eps0)
    rx, ry, rz = _coarse_rot1_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot1_cs(vx, vy, vz, c, s)

    c = math.cos(dpsi)
    s = -math.sin(dpsi)
    rx, ry, rz = _coarse_rot3_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot3_cs(vx, vy, vz, c, s)

    c = math.cos(eps)
    s = -math.sin(eps)
    rx, ry, rz = _coarse_rot1_cs(rx, ry, rz, c, s)
    vx, vy, vz = _coarse_rot1_cs(vx, vy, vz, c, s)

    gmst = _coarse_gmst_vallado_rad(jd_ut1)
    eqeq = dpsi * math.cos(eps) + (0.00264 * math.sin(om) + 0.000063 * math.sin(2.0 * om)) * DAS2R
    gast = _coarse_mod2pi_rad(gmst + eqeq)

    c = math.cos(gast)
    s = math.sin(gast)
    x, y, z = _coarse_rot3_cs(rx, ry, rz, c, s)
    vx_e, vy_e, vz_e = _coarse_rot3_cs(vx, vy, vz, c, s)

    vx_e = vx_e + EARTH_OMEGA * y
    vy_e = vy_e - EARTH_OMEGA * x

    if xp_rad != 0.0 or yp_rad != 0.0:
        c = math.cos(-yp_rad)
        s = math.sin(-yp_rad)
        x, y, z = _coarse_rot1_cs(x, y, z, c, s)
        vx_e, vy_e, vz_e = _coarse_rot1_cs(vx_e, vy_e, vz_e, c, s)

        c = math.cos(-xp_rad)
        s = math.sin(-xp_rad)
        x, y, z = _coarse_rot2_cs(x, y, z, c, s)
        vx_e, vy_e, vz_e = _coarse_rot2_cs(vx_e, vy_e, vz_e, c, s)

    return x, y, z, vx_e, vy_e, vz_e


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci_to_itrf_pos_iau76_shortnut(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform one ECI position to ITRF using the coarse fast model.

    Parameters
    ----------
    x_eci_m, y_eci_m, z_eci_m : float
        ECI position components [m] in the module's native inertial basis.
    jd_ut1 : float
        UT1 Julian date.
    jd_tt : float
        TT Julian date.
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (x_itrf_m, y_itrf_m, z_itrf_m) : tuple[float, float, float]
        ITRF/ECEF position [m].
    """
    return _coarse_eci_to_itrf_pos_iau76_shortnut(
        x_eci_m, y_eci_m, z_eci_m, jd_ut1, jd_tt, xp_rad, yp_rad
    )


@njit(cache=False, fastmath=True, inline="always")
def coarse_eci_to_itrf_pos_vel_iau76_shortnut(
    x_eci_m: float,
    y_eci_m: float,
    z_eci_m: float,
    vx_eci_mps: float,
    vy_eci_mps: float,
    vz_eci_mps: float,
    jd_ut1: float,
    jd_tt: float,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform one ECI position/velocity state to ITRF using the coarse fast model.

    Parameters
    ----------
    x_eci_m, y_eci_m, z_eci_m : float
        ECI position components [m].
    vx_eci_mps, vy_eci_mps, vz_eci_mps : float
        ECI velocity components [m/s].
    jd_ut1 : float
        UT1 Julian date.
    jd_tt : float
        TT Julian date.
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (x_itrf_m, y_itrf_m, z_itrf_m, vx_itrf_mps, vy_itrf_mps, vz_itrf_mps)
        ITRF position [m] and velocity [m/s].
    """
    return _coarse_eci_to_itrf_pv_iau76_shortnut(
        x_eci_m,
        y_eci_m,
        z_eci_m,
        vx_eci_mps,
        vy_eci_mps,
        vz_eci_mps,
        jd_ut1,
        jd_tt,
        xp_rad,
        yp_rad,
    )


@njit(cache=False, fastmath=True, parallel=True)
def coarse_eci_to_itrf_pos_vec_iau76_shortnut(
    r_eci_m: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform a batch of ECI positions to ITRF using the coarse fast model.

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
    np.ndarray
        ITRF positions with shape (N, 3), meters.
    """
    if r_eci_m.ndim != 2 or r_eci_m.shape[1] != 3:
        raise ValueError("r_eci_m must have shape (N, 3)")

    n = r_eci_m.shape[0]
    if jd_ut1.shape[0] != n or jd_tt.shape[0] != n:
        raise ValueError("jd_ut1 and jd_tt must have shape (N,)")

    r_itrf_m = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        x, y, z = _coarse_eci_to_itrf_pos_iau76_shortnut(
            r_eci_m[i, 0],
            r_eci_m[i, 1],
            r_eci_m[i, 2],
            jd_ut1[i],
            jd_tt[i],
            xp_rad,
            yp_rad,
        )
        r_itrf_m[i, 0] = x
        r_itrf_m[i, 1] = y
        r_itrf_m[i, 2] = z
    return r_itrf_m


@njit(cache=False, fastmath=True, parallel=True)
def coarse_eci_to_itrf_pos_vel_vec_iau76_shortnut(
    r_eci_m: np.ndarray,
    v_eci_mps: np.ndarray,
    jd_ut1: np.ndarray,
    jd_tt: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Transform a batch of ECI position/velocity states to ITRF using the coarse fast model.

    Parameters
    ----------
    r_eci_m : np.ndarray
        ECI positions with shape (N, 3), meters.
    v_eci_mps : np.ndarray
        ECI velocities with shape (N, 3), meters/second.
    jd_ut1 : np.ndarray
        UT1 Julian dates with shape (N,).
    jd_tt : np.ndarray
        TT Julian dates with shape (N,).
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad]. Defaults to zero.

    Returns
    -------
    (r_itrf_m, v_itrf_mps) : tuple[np.ndarray, np.ndarray]
        ITRF positions and velocities with shape (N, 3).
    """
    if r_eci_m.ndim != 2 or r_eci_m.shape[1] != 3:
        raise ValueError("r_eci_m must have shape (N, 3)")
    if v_eci_mps.ndim != 2 or v_eci_mps.shape[1] != 3:
        raise ValueError("v_eci_mps must have shape (N, 3)")

    n = r_eci_m.shape[0]
    if v_eci_mps.shape[0] != n:
        raise ValueError("r_eci_m and v_eci_mps must have the same length")
    if jd_ut1.shape[0] != n or jd_tt.shape[0] != n:
        raise ValueError("jd_ut1 and jd_tt must have shape (N,)")

    r_itrf_m = np.empty((n, 3), dtype=np.float64)
    v_itrf_mps = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        x, y, z, vx, vy, vz = _coarse_eci_to_itrf_pv_iau76_shortnut(
            r_eci_m[i, 0],
            r_eci_m[i, 1],
            r_eci_m[i, 2],
            v_eci_mps[i, 0],
            v_eci_mps[i, 1],
            v_eci_mps[i, 2],
            jd_ut1[i],
            jd_tt[i],
            xp_rad,
            yp_rad,
        )
        r_itrf_m[i, 0] = x
        r_itrf_m[i, 1] = y
        r_itrf_m[i, 2] = z
        v_itrf_mps[i, 0] = vx
        v_itrf_mps[i, 1] = vy
        v_itrf_mps[i, 2] = vz
    return r_itrf_m, v_itrf_mps


__all__ = [
    "coarse_eci_to_itrf_pos_iau76_shortnut",
    "coarse_eci_to_itrf_pos_vel_iau76_shortnut",
    "coarse_eci_to_itrf_pos_vec_iau76_shortnut",
    "coarse_eci_to_itrf_pos_vel_vec_iau76_shortnut",
]
