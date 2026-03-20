# _ecef2geodetic.py
import math
import numpy as np
from numba import njit, prange

from .constants import WGS84_A, WGS84_E2, WGS84_B, WGS84_EP2


# Treat p = hypot(x,y) smaller than this as "numerically at the pole".
# This is intentionally larger than 1e-12 to handle round-off from geodetic->ECEF at ±90°.
_POLE_P_EPS_M = 1e-8


@njit(cache=True, inline="always")
def _wrap_lon_pi(lon_rad: float) -> float:
    """
    Wrap an angle to the interval [-π, π).

    This implementation is Numba-friendly (avoids math.fmod / Python % on floats).

    Parameters
    ----------
    lon_rad : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in radians in [-π, π).
    """
    twopi = 2.0 * math.pi
    t = lon_rad + math.pi
    k = math.floor(t / twopi)
    t = t - k * twopi
    return t - math.pi


@njit(cache=True, inline="always")
def ecef2geodetic(x_m: float, y_m: float, z_m: float):
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates to WGS84 geodetic
    latitude/longitude/height using a Bowring-style closed-form solution.

    Inputs are interpreted as meters in a single Cartesian frame (typically ECEF):
        r = [x_m, y_m, z_m]

    Outputs:
      - latitude  (geodetic) in radians, in [-π/2, +π/2]
      - longitude in radians, wrapped to [-π, π)
      - height above the WGS84 ellipsoid in meters

    Model notes
    ----------
    - Uses WGS84 ellipsoid parameters (a, b, e², e'²).
    - Bowring-style method is robust and accurate for points inside/outside the ellipsoid.
    - Designed for Numba @njit throughput: no allocations, math.* intrinsics only.

    Edge cases
    ----------
    - Origin (0,0,0): geodetic lat/lon undefined. Returns (0, 0, -a).
    - Poles (x≈0,y≈0): longitude is undefined; this implementation sets lon=0,
      lat=±π/2, and h=|z|-b.

    Parameters
    ----------
    x_m, y_m, z_m : float
        ECEF coordinates in meters.

    Returns
    -------
    (lat_rad, lon_rad, h_m) : tuple[float, float, float]
        Geodetic latitude [rad], wrapped longitude [rad], height above WGS84 [m].
    """
    # Longitude (wrap to [-pi, pi) explicitly).
    lon = _wrap_lon_pi(math.atan2(y_m, x_m))

    # Distance from Z-axis
    p = math.hypot(x_m, y_m)

    # Origin: undefined; choose a consistent sentinel
    if p == 0.0 and z_m == 0.0:
        return 0.0, 0.0, -WGS84_A

    # Poles / near-poles (lon is undefined; choose lon=0 by convention)
    if p < _POLE_P_EPS_M:
        lat = 0.5 * math.pi if z_m >= 0.0 else -0.5 * math.pi
        h = abs(z_m) - WGS84_B
        return lat, 0.0, h

    # Bowring-style auxiliary angle
    # theta = atan2(z*a, p*b)
    theta = math.atan2(z_m * WGS84_A, p * WGS84_B)
    st = math.sin(theta)
    ct = math.cos(theta)

    st2 = st * st
    ct2 = ct * ct
    st3 = st2 * st
    ct3 = ct2 * ct

    # latitude
    # lat = atan2(z + e'^2 * b * sin^3(theta), p - e^2 * a * cos^3(theta))
    lat = math.atan2(
        z_m + WGS84_EP2 * WGS84_B * st3,
        p - WGS84_E2 * WGS84_A * ct3,
    )

    # radius of curvature in the prime vertical
    sphi = math.sin(lat)
    cphi = math.cos(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * sphi * sphi)
    N = WGS84_A / denom

    # height (use a pole-stable form when cos(lat) is tiny)
    if abs(cphi) > 1e-12:
        h = p / cphi - N
    else:
        h = z_m / sphi - N * (1.0 - WGS84_E2)

    return lat, lon, h


@njit(cache=True, inline="always")
def ecef2geodetic_deg(x_m: float, y_m: float, z_m: float):
    """
    Scalar ECEF -> geodetic conversion with angular outputs in degrees.

    Returns
    -------
    (lat_deg, lon_deg, h_m) : tuple[float, float, float]
        Geodetic latitude [deg], wrapped longitude [deg] in [-180, 180), height [m].
    """
    lat_rad, lon_rad, h_m = ecef2geodetic(x_m, y_m, z_m)
    return lat_rad * (180.0 / math.pi), lon_rad * (180.0 / math.pi), h_m


@njit(cache=True, parallel=True)
def ecef2geodetic_vec_xyz(x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray):
    """
    Convert arrays of ECEF coordinates to geodetic coordinates in parallel.

    Parameters
    ----------
    x_m, y_m, z_m : np.ndarray
        1D arrays of identical length N containing ECEF coordinates [m].

    Returns
    -------
    lat_rad : np.ndarray
        Geodetic latitude in radians, shape (N,).
    lon_rad : np.ndarray
        Wrapped longitude in radians in [-π, π), shape (N,).
    h_m : np.ndarray
        Height above WGS84 ellipsoid in meters, shape (N,).
    """
    n = x_m.shape[0]
    if y_m.shape[0] != n or z_m.shape[0] != n:
        raise ValueError("x_m, y_m, z_m must have the same length")

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = ecef2geodetic(x_m[i], y_m[i], z_m[i])
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def ecef2geodetic_vec_ecef(r_ecef_m: np.ndarray):
    """
    Convert an (N,3) ECEF array to geodetic coordinates in parallel.

    Parameters
    ----------
    r_ecef_m : np.ndarray
        2D array of shape (N, 3) containing ECEF positions [m].

    Returns
    -------
    lat_rad : np.ndarray
        Geodetic latitude in radians, shape (N,).
    lon_rad : np.ndarray
        Wrapped longitude in radians in [-π, π), shape (N,).
    h_m : np.ndarray
        Height above WGS84 ellipsoid in meters, shape (N,).
    """
    if r_ecef_m.ndim != 2 or r_ecef_m.shape[1] != 3:
        raise ValueError("r_ecef_m must have shape (N, 3)")

    n = r_ecef_m.shape[0]
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)

    for i in prange(n):
        la, lo, hi = ecef2geodetic(r_ecef_m[i, 0], r_ecef_m[i, 1], r_ecef_m[i, 2])
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def ecef2geodetic_vec_ecef_deg(r_ecef_m: np.ndarray, wrap_lon: bool = True):
    """
    Vectorized ECEF -> geodetic conversion with angular outputs in degrees.

    Parameters
    ----------
    r_ecef_m : np.ndarray
        2D array of shape (N, 3) containing ECEF positions [m].
    wrap_lon : bool
        Keep longitude wrapped to [-180, 180). Included for API compatibility.
        Values from `ecef2geodetic` are already wrapped by construction.

    Returns
    -------
    lat_deg : np.ndarray
        Geodetic latitude in degrees, shape (N,).
    lon_deg : np.ndarray
        Wrapped longitude in degrees in [-180, 180), shape (N,).
    h_m : np.ndarray
        Height above WGS84 ellipsoid in meters, shape (N,).
    """
    lat_rad, lon_rad, h_m = ecef2geodetic_vec_ecef(r_ecef_m)
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

    return lat_deg, lon_deg, h_m
