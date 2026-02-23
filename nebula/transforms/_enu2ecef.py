# _enu2ecef.py
import math
import numpy as np
from numba import njit, prange

# Import scalar kernel for reference point ECEF.
from ._geodetic2ecef import geodetic2ecef


@njit(cache=True, inline="always")
def enu2ecef_delta(
    e_m: float, n_m: float, u_m: float, lat0_rad: float, lon0_rad: float
):
    """
    Convert a local ENU vector into an ECEF delta vector at a reference geodetic point.

    This is a pure rotation (no translation). Use enu2ecef() to obtain an absolute
    ECEF position by adding the reference ECEF origin.

    Parameters
    ----------
    e_m, n_m, u_m : float
        ENU vector components in meters.
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians defining the ENU frame.

    Returns
    -------
    (dx_m, dy_m, dz_m) : tuple[float, float, float]
        ECEF delta vector components in meters.
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
def enu2ecef(
    e_m: float,
    n_m: float,
    u_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert local ENU coordinates to absolute ECEF coordinates.

    Model
    -----
    1) Compute reference ECEF (x0,y0,z0) from (lat0,lon0,h0).
    2) Rotate ENU vector into ECEF delta (dx,dy,dz) using the ENU frame at (lat0,lon0).
    3) Add translation: (x,y,z) = (x0,y0,z0) + (dx,dy,dz).

    Conventions
    -----------
    - lat/lon are radians.
    - h0 and ENU are meters.
    - ECEF outputs are meters.

    Parameters
    ----------
    e_m, n_m, u_m : float
        ENU coordinates in meters.
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians.
    h0_m : float
        Reference height above WGS84 ellipsoid in meters.

    Returns
    -------
    (x_m, y_m, z_m) : tuple[float, float, float]
        Absolute ECEF coordinates in meters.
    """
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    dx, dy, dz = enu2ecef_delta(e_m, n_m, u_m, lat0_rad, lon0_rad)
    return x0 + dx, y0 + dy, z0 + dz


@njit(cache=True, parallel=True)
def enu2ecef_vec_enu(
    e_m: np.ndarray,
    n_m: np.ndarray,
    u_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ENU->ECEF conversion in parallel.

    Parameters
    ----------
    e_m, n_m, u_m : np.ndarray
        1D arrays (N,) of ENU coordinates in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    x_m, y_m, z_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of absolute ECEF coordinates in meters.
    """
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
def enu2ecef_vec_enu3(
    enu_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ENU->ECEF conversion for an (N,3) ENU array in parallel.

    Parameters
    ----------
    enu_m : np.ndarray
        2D array (N,3) with columns [e_m, n_m, u_m] in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    r_ecef_m : np.ndarray
        2D array (N,3) of absolute ECEF coordinates in meters.
    """
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
