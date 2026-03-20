# _enu2geodetic.py
import math
import numpy as np
from numba import njit, prange

# Import scalar kernels (Numba-friendly).
from ._geodetic2ecef import geodetic2ecef

from ._ecef2geodetic import ecef2geodetic


@njit(cache=True, inline="always")
def _enu2uvw(e_m: float, n_m: float, u_m: float, lat0_rad: float, lon0_rad: float):
    """
    Convert an ENU vector into an ECEF delta vector at a reference geodetic point.

    ENU axes are defined at the reference (lat0, lon0):
      - +E: tangent to parallel, increasing longitude
      - +N: tangent to meridian, increasing latitude
      - +U: outward normal

    Parameters
    ----------
    e_m, n_m, u_m : float
        ENU vector components in meters.
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians.

    Returns
    -------
    (dx_m, dy_m, dz_m) : tuple[float, float, float]
        ECEF delta vector in meters.
    """
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    # Standard ENU->ECEF rotation applied to vector
    dx = -slon * e_m - slat * clon * n_m + clat * clon * u_m
    dy = clon * e_m - slat * slon * n_m + clat * slon * u_m
    dz = clat * n_m + slat * u_m
    return dx, dy, dz


@njit(cache=True, inline="always")
def enu2geodetic(
    e_m: float,
    n_m: float,
    u_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert a local ENU coordinate (about a reference point) to geodetic coordinates.

    This computes:
      1) reference ECEF from (lat0, lon0, h0)
      2) rotate ENU vector into ECEF delta using the reference tangent frame
      3) target ECEF = reference ECEF + delta
      4) convert target ECEF to geodetic (lat, lon, h)

    Conventions
    -----------
    - lat/lon are radians.
    - heights and ENU inputs are meters.
    - Longitude returned is wrapped per ecef2geodetic() (typically to [-π, π)).

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
    (lat_rad, lon_rad, h_m) : tuple[float, float, float]
        Target geodetic latitude [rad], longitude [rad], height [m].
    """
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    dx, dy, dz = _enu2uvw(e_m, n_m, u_m, lat0_rad, lon0_rad)

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return ecef2geodetic(x, y, z)


@njit(cache=True, parallel=True)
def enu2geodetic_vec_enu(
    e_m: np.ndarray,
    n_m: np.ndarray,
    u_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ENU->geodetic conversion in parallel (prange).

    Parameters
    ----------
    e_m, n_m, u_m : np.ndarray
        1D arrays (N,) of ENU coordinates in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    lat_rad, lon_rad, h_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of geodetic latitude [rad], longitude [rad], height [m].
    """
    n = e_m.shape[0]
    if n_m.shape[0] != n or u_m.shape[0] != n:
        raise ValueError("e_m, n_m, u_m must have the same length")

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    # Precompute reference ECEF and trig once
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

        la, lo, hi = ecef2geodetic(x0 + dx, y0 + dy, z0 + dz)
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def enu2geodetic_vec_enu3(
    enu_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ENU->geodetic conversion for an (N,3) ENU array in parallel.

    Parameters
    ----------
    enu_m : np.ndarray
        2D array (N,3) with columns [e_m, n_m, u_m] in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    lat_rad, lon_rad, h_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of geodetic latitude [rad], longitude [rad], height [m].
    """
    if enu_m.ndim != 2 or enu_m.shape[1] != 3:
        raise ValueError("enu_m must have shape (N, 3)")

    n = enu_m.shape[0]
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

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

        la, lo, hi = ecef2geodetic(x0 + dx, y0 + dy, z0 + dz)
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h
