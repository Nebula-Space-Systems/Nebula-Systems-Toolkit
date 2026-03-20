# _geodetic2ecef.py
import math
import numpy as np
from numba import njit, prange

from .constants import WGS84_A, WGS84_E2, WGS84_B2_OVER_A2


@njit(cache=True, inline="always")
def geodetic2ecef(lat_rad: float, lon_rad: float, h_m: float):
    """
    Convert WGS84 geodetic latitude/longitude/height to ECEF Cartesian coordinates.

    Model
    -----
    WGS84 ellipsoid parameters:
      - semi-major axis a
      - first eccentricity squared e²

    For geodetic latitude φ, longitude λ, and height h:
      N(φ) = a / sqrt(1 - e² sin²φ)                (prime-vertical radius)

      x = (N + h) cosφ cosλ
      y = (N + h) cosφ sinλ
      z = (N (b²/a²) + h) sinφ   where b²/a² = 1 - e²

    Conventions
    -----------
    - Inputs: lat/lon in radians, height in meters.
    - Outputs: x/y/z in meters.

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude φ in radians.
    lon_rad : float
        Geodetic longitude λ in radians.
    h_m : float
        Height above WGS84 ellipsoid in meters.

    Returns
    -------
    (x_m, y_m, z_m) : tuple[float, float, float]
        ECEF coordinates in meters.

    Notes
    -----
    - High-throughput @njit implementation: no allocations, uses math.* intrinsics.
    - No longitude wrapping is needed for correctness (sin/cos are periodic).
    """
    sphi = math.sin(lat_rad)
    cphi = math.cos(lat_rad)
    slam = math.sin(lon_rad)
    clam = math.cos(lon_rad)

    denom = math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
    N = WGS84_A / denom

    Nh = N + h_m
    x = Nh * cphi * clam
    y = Nh * cphi * slam
    z = (N * WGS84_B2_OVER_A2 + h_m) * sphi
    return x, y, z


@njit(cache=True, parallel=True)
def geodetic2ecef_vec_llh(lat_rad: np.ndarray, lon_rad: np.ndarray, h_m: np.ndarray):
    """
    Convert arrays of WGS84 geodetic coordinates to ECEF in parallel.

    Parameters
    ----------
    lat_rad : np.ndarray
        1D array (N,) of geodetic latitude [rad].
    lon_rad : np.ndarray
        1D array (N,) of geodetic longitude [rad].
    h_m : np.ndarray
        1D array (N,) of height [m].

    Returns
    -------
    x_m, y_m, z_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        ECEF coordinates [m], each shape (N,).

    Notes
    -----
    - Uses prange for parallel loop over points.
    - Best performance when inputs are float64 and contiguous.
    """
    n = lat_rad.shape[0]
    if lon_rad.shape[0] != n or h_m.shape[0] != n:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    for i in prange(n):
        xi, yi, zi = geodetic2ecef(lat_rad[i], lon_rad[i], h_m[i])
        x[i] = xi
        y[i] = yi
        z[i] = zi

    return x, y, z


@njit(cache=True, parallel=True)
def geodetic2ecef_vec_lla(lla_rad_m: np.ndarray):
    """
    Convert an (N,3) array of [lat_rad, lon_rad, h_m] to an (N,3) ECEF array in parallel.

    Parameters
    ----------
    lla_rad_m : np.ndarray
        2D array (N,3):
          - col 0: latitude [rad]
          - col 1: longitude [rad]
          - col 2: height [m]

    Returns
    -------
    r_ecef_m : np.ndarray
        2D array (N,3) of ECEF coordinates [m].

    Notes
    -----
    - Uses prange for parallel loop over points.
    - Best performance when input is float64 and contiguous.
    """
    if lla_rad_m.ndim != 2 or lla_rad_m.shape[1] != 3:
        raise ValueError("lla_rad_m must have shape (N, 3)")

    n = lla_rad_m.shape[0]
    out = np.empty((n, 3), dtype=np.float64)

    for i in prange(n):
        xi, yi, zi = geodetic2ecef(lla_rad_m[i, 0], lla_rad_m[i, 1], lla_rad_m[i, 2])
        out[i, 0] = xi
        out[i, 1] = yi
        out[i, 2] = zi

    return out
