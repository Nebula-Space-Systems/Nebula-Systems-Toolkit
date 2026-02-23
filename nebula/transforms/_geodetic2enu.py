# _geodetic2enu.py
import math
import numpy as np
from numba import njit, prange

# Import scalar geodetic->ECEF kernel (Numba-friendly).
from ._geodetic2ecef import geodetic2ecef


@njit(cache=True, inline="always")
def _uvw2enu(dx_m: float, dy_m: float, dz_m: float, lat0_rad: float, lon0_rad: float):
    """
    Convert an ECEF delta vector (dx,dy,dz) into local ENU at a reference geodetic point.

    ENU axes are defined at the reference (lat0, lon0):
      - +E: tangent to parallel, increasing longitude
      - +N: tangent to meridian, increasing latitude
      - +U: outward normal (approximately radial for WGS84)

    Parameters
    ----------
    dx_m, dy_m, dz_m : float
        ECEF delta vector components in meters (target_ecef - ref_ecef).
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians.

    Returns
    -------
    (e_m, n_m, u_m) : tuple[float, float, float]
        Local ENU vector in meters.
    """
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    # Standard ECEF->ENU rotation applied to delta vector
    e = -slon * dx_m + clon * dy_m
    n = -slat * clon * dx_m - slat * slon * dy_m + clat * dz_m
    u = clat * clon * dx_m + clat * slon * dy_m + slat * dz_m
    return e, n, u


@njit(cache=True, inline="always")
def enu_basis_from_latlon(lat0: float, lon0: float) -> np.ndarray:
    """
    Compute the ENU basis vectors at a given latitude and longitude in radians.

    Args:
        lat0 (float): Latitude (rad)
        lon0 (float): Longitude (rad)

    Returns:
        np.ndarray: 3x3 matrix whose columns are [E, N, U] basis vectors in ECEF.
    """
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    R = np.empty((3, 3), dtype=np.float64)

    # East basis vector
    R[0, 0] = -sin_lon
    R[1, 0] = cos_lon
    R[2, 0] = 0.0

    # North basis vector
    R[0, 1] = -sin_lat * cos_lon
    R[1, 1] = -sin_lat * sin_lon
    R[2, 1] = cos_lat

    # Up basis vector
    R[0, 2] = cos_lat * cos_lon
    R[1, 2] = cos_lat * sin_lon
    R[2, 2] = sin_lat

    return R


@njit(cache=True, inline="always")
def geodetic2enu(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert target geodetic coordinates to local ENU coordinates about a reference point.

    This computes:
      1) target ECEF and reference ECEF from (lat,lon,h)
      2) delta ECEF = target - reference
      3) rotate delta ECEF into the reference local tangent frame (ENU)

    Conventions
    -----------
    - All lat/lon inputs are radians.
    - All heights and ENU outputs are meters.
    - ENU frame is defined at the *reference* geodetic point (lat0, lon0, h0).

    Parameters
    ----------
    lat_rad, lon_rad : float
        Target geodetic latitude/longitude in radians.
    h_m : float
        Target height above WGS84 ellipsoid in meters.
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians.
    h0_m : float
        Reference height above WGS84 ellipsoid in meters.

    Returns
    -------
    (e_m, n_m, u_m) : tuple[float, float, float]
        Target position expressed in the reference local ENU frame [m].

    Notes
    -----
    - This is purely geometric and uses the WGS84 ellipsoid through geodetic2ecef().
    - For best numerical conditioning, avoid using a reference exactly at the poles.
    """
    x, y, z = geodetic2ecef(lat_rad, lon_rad, h_m)
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    return _uvw2enu(dx, dy, dz, lat0_rad, lon0_rad)


@njit(cache=True, parallel=True)
def geodetic2enu_vec_llh(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized geodetic->ENU conversion in parallel (prange).

    Parameters
    ----------
    lat_rad, lon_rad, h_m : np.ndarray
        1D arrays (N,) of target geodetic latitude [rad], longitude [rad], height [m].
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    e_m, n_m, u_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of ENU coordinates in meters.
    """
    n = lat_rad.shape[0]
    if lon_rad.shape[0] != n or h_m.shape[0] != n:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    # Precompute reference ECEF once (big speedup)
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    # Precompute trig for rotation once
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        x, y, z = geodetic2ecef(lat_rad[i], lon_rad[i], h_m[i])
        dx = x - x0
        dy = y - y0
        dz = z - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u


@njit(cache=True, parallel=True)
def geodetic2enu_vec_lla(
    lla_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized geodetic->ENU conversion for an (N,3) array in parallel.

    Parameters
    ----------
    lla_rad_m : np.ndarray
        2D array (N,3) with columns [lat_rad, lon_rad, h_m].
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    enu_m : np.ndarray
        2D array (N,3) with columns [e_m, n_m, u_m] in meters.
    """
    if lla_rad_m.ndim != 2 or lla_rad_m.shape[1] != 3:
        raise ValueError("lla_rad_m must have shape (N, 3)")

    n = lla_rad_m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)

    # Precompute reference ECEF once
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    # Precompute trig once
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        x, y, z = geodetic2ecef(lla_rad_m[i, 0], lla_rad_m[i, 1], lla_rad_m[i, 2])
        dx = x - x0
        dy = y - y0
        dz = z - z0

        out[i, 0] = -slon * dx + clon * dy
        out[i, 1] = -slat * clon * dx - slat * slon * dy + clat * dz
        out[i, 2] = clat * clon * dx + clat * slon * dy + slat * dz

    return out
