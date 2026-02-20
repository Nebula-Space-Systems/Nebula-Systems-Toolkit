# _ecef2enu.py
import math
import numpy as np
from numba import njit, prange

# Import scalar kernel for reference point ECEF.
from ._geodetic2ecef import geodetic2ecef
from ._ecef2geodetic import ecef2geodetic


@njit(cache=True, inline="always")
def enu_basis_from_ecef_xyz(x_m: float, y_m: float, z_m: float) -> np.ndarray:
    """
    Compute ENU basis vectors (in ECEF coordinates) at the geodetic location
    corresponding to an ECEF position.

    This computes geodetic (lat, lon) from (x,y,z) on WGS84, then returns a 3x3
    matrix whose columns are the unit basis vectors [E, N, U] expressed in ECEF.

    Parameters
    ----------
    x_m, y_m, z_m : float
        ECEF coordinates in meters.

    Returns
    -------
    R : np.ndarray
        3x3 matrix (float64) with columns:
          R[:,0] = East  unit vector in ECEF
          R[:,1] = North unit vector in ECEF
          R[:,2] = Up    unit vector in ECEF

    Notes
    -----
    - Uses geodetic latitude (ellipsoid normal), not geocentric latitude.
    - At the poles, East/North are not uniquely defined; the formulas still produce
      a consistent choice given lon.
    """
    lat, lon, _ = ecef2geodetic(x_m, y_m, z_m)

    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)

    R = np.empty((3, 3), dtype=np.float64)

    # East
    R[0, 0] = -slon
    R[1, 0] = clon
    R[2, 0] = 0.0

    # North
    R[0, 1] = -slat * clon
    R[1, 1] = -slat * slon
    R[2, 1] = clat

    # Up
    R[0, 2] = clat * clon
    R[1, 2] = clat * slon
    R[2, 2] = slat

    return R


@njit(cache=True, inline="always")
def ecef2enu_delta(
    dx_m: float, dy_m: float, dz_m: float, lat0_rad: float, lon0_rad: float
):
    """
    Convert an ECEF delta vector into an ENU vector at a reference geodetic point.

    This is a pure rotation (no translation). Use ecef2enu() when you have an
    absolute ECEF target coordinate and want ENU relative to a reference origin.

    Parameters
    ----------
    dx_m, dy_m, dz_m : float
        ECEF delta vector components in meters (target_ecef - ref_ecef).
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians defining the ENU frame.

    Returns
    -------
    (e_m, n_m, u_m) : tuple[float, float, float]
        ENU vector components in meters.
    """
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    e = -slon * dx_m + clon * dy_m
    n = -slat * clon * dx_m - slat * slon * dy_m + clat * dz_m
    u = clat * clon * dx_m + clat * slon * dy_m + slat * dz_m
    return e, n, u


@njit(cache=True, inline="always")
def ecef2enu(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert absolute ECEF coordinates to local ENU coordinates about a reference point.

    Model
    -----
    1) Compute reference ECEF (x0,y0,z0) from (lat0,lon0,h0).
    2) Form delta ECEF: (dx,dy,dz) = (x,y,z) - (x0,y0,z0).
    3) Rotate delta ECEF into ENU using the tangent frame at (lat0,lon0).

    Conventions
    -----------
    - lat/lon are radians.
    - ECEF inputs are meters.
    - ENU outputs are meters.

    Parameters
    ----------
    x_m, y_m, z_m : float
        Target ECEF coordinates in meters.
    lat0_rad, lon0_rad : float
        Reference geodetic latitude/longitude in radians.
    h0_m : float
        Reference height above WGS84 ellipsoid in meters.

    Returns
    -------
    (e_m, n_m, u_m) : tuple[float, float, float]
        Target position expressed in the reference local ENU frame [m].
    """
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)
    dx = x_m - x0
    dy = y_m - y0
    dz = z_m - z0
    return ecef2enu_delta(dx, dy, dz, lat0_rad, lon0_rad)


@njit(cache=True, parallel=True)
def ecef2enu_vec_xyz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ECEF->ENU conversion in parallel.

    Parameters
    ----------
    x_m, y_m, z_m : np.ndarray
        1D arrays (N,) of target ECEF coordinates in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    e_m, n_m, u_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of ENU coordinates in meters.
    """
    n = x_m.shape[0]
    if y_m.shape[0] != n or z_m.shape[0] != n:
        raise ValueError("x_m, y_m, z_m must have the same length")

    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        dx = x_m[i] - x0
        dy = y_m[i] - y0
        dz = z_m[i] - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u


@njit(cache=True, parallel=True)
def ecef2enu_vec_ecef(
    r_ecef_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ECEF->ENU conversion for an (N,3) ECEF array in parallel.

    Parameters
    ----------
    r_ecef_m : np.ndarray
        2D array (N,3) with columns [x_m, y_m, z_m] in meters.
    lat0_rad, lon0_rad, h0_m : float
        Reference geodetic point defining the ENU frame.

    Returns
    -------
    e_m, n_m, u_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of ENU coordinates in meters.
    """
    if r_ecef_m.ndim != 2 or r_ecef_m.shape[1] != 3:
        raise ValueError("r_ecef_m must have shape (N, 3)")

    n = r_ecef_m.shape[0]
    e = np.empty(n, dtype=np.float64)
    n_out = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(n):
        dx = r_ecef_m[i, 0] - x0
        dy = r_ecef_m[i, 1] - y0
        dz = r_ecef_m[i, 2] - z0

        e[i] = -slon * dx + clon * dy
        n_out[i] = -slat * clon * dx - slat * slon * dy + clat * dz
        u[i] = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n_out, u
