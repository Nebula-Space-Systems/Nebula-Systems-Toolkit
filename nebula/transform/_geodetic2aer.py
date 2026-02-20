# _geodetic2aer.py
import math
import numpy as np
from numba import njit, prange

from ._geodetic2enu import geodetic2enu


@njit(cache=True, inline="always")
def enu2aer(e_m: float, n_m: float, u_m: float):
    """
    Convert local ENU coordinates to AER (azimuth, elevation, slant range).

    Definitions
    ----------
    Given a local tangent frame (East, North, Up) with coordinates (e, n, u) [m]:

      slant_range  ρ = sqrt(e^2 + n^2 + u^2)                      [m]
      elevation    el = atan2(u, sqrt(e^2 + n^2))                 [rad]
      azimuth      az = atan2(e, n) wrapped to [0, 2π)            [rad]

    Conventions
    -----------
    - azimuth is clockwise from North toward East:
        az=0 at +North, az=π/2 at +East
    - elevation is positive above the local horizon
    - all angles are returned in radians

    Parameters
    ----------
    e_m, n_m, u_m : float
        ENU coordinates in meters.

    Returns
    -------
    (az_rad, el_rad, srange_m) : tuple[float, float, float]
        Azimuth [rad] in [0, 2π), elevation [rad] in [-π/2, π/2],
        and slant range [m].

    Notes
    -----
    - For e=n=u=0, returns az=0, el=0, srange=0 by convention.
    """
    # stabilize tiny values (optional; helps atan2 stability near zero)
    if abs(e_m) < 1e-12:
        e_m = 0.0
    if abs(n_m) < 1e-12:
        n_m = 0.0
    if abs(u_m) < 1e-12:
        u_m = 0.0

    r_h = math.hypot(e_m, n_m)
    srange = math.hypot(r_h, u_m)

    if srange == 0.0:
        return 0.0, 0.0, 0.0

    el = math.atan2(u_m, r_h)

    az = math.atan2(e_m, n_m)
    if az < 0.0:
        az += 2.0 * math.pi

    return az, el, srange


@njit(cache=True, inline="always")
def geodetic2aer(
    lat_rad: float,
    lon_rad: float,
    h_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Compute AER (azimuth, elevation, slant range) from an observer to a target,
    where both are specified in WGS84 geodetic coordinates.

    Model
    -----
    1) Convert target geodetic -> ENU about observer using geodetic2enu()
    2) Convert ENU -> AER using enu2aer()

    Conventions
    -----------
    - All lat/lon inputs are radians.
    - Heights are meters above WGS84 ellipsoid.
    - Outputs: azimuth in [0,2π) rad, elevation in radians, range in meters.

    Parameters
    ----------
    lat_rad, lon_rad, h_m : float
        Target geodetic coordinates (lat [rad], lon [rad], height [m]).
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coordinates (lat [rad], lon [rad], height [m]).

    Returns
    -------
    (az_rad, el_rad, srange_m) : tuple[float, float, float]
        Azimuth [rad], elevation [rad], slant range [m].

    Notes
    -----
    - The result is purely geometric in the local tangent frame at the observer.
    - Refraction / Earth curvature effects beyond WGS84 geometry are not included.
    """
    e, n, u = geodetic2enu(lat_rad, lon_rad, h_m, lat0_rad, lon0_rad, h0_m)
    return enu2aer(e, n, u)


@njit(cache=True, parallel=True)
def geodetic2aer_vec_llh(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    h_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized geodetic->AER conversion in parallel (prange).

    Parameters
    ----------
    lat_rad, lon_rad, h_m : np.ndarray
        1D arrays (N,) of target geodetic coords [rad, rad, m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coords [rad, rad, m].

    Returns
    -------
    az_rad, el_rad, srange_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of azimuth [rad], elevation [rad], slant range [m].
    """
    npts = lat_rad.shape[0]
    if lon_rad.shape[0] != npts or h_m.shape[0] != npts:
        raise ValueError("lat_rad, lon_rad, h_m must have the same length")

    az = np.empty(npts, dtype=np.float64)
    el = np.empty(npts, dtype=np.float64)
    sr = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        a, e, r = geodetic2aer(lat_rad[i], lon_rad[i], h_m[i], lat0_rad, lon0_rad, h0_m)
        az[i] = a
        el[i] = e
        sr[i] = r

    return az, el, sr
