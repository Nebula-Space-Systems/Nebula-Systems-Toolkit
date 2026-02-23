# _ecef2aer.py
import math
import numpy as np
from numba import njit, prange

from ._ecef2enu import ecef2enu

from ._geodetic2aer import enu2aer


@njit(cache=True, inline="always")
def ecef2aer(
    x_m: float,
    y_m: float,
    z_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Compute AER (azimuth, elevation, slant range) from an observer to a target,
    where the target is specified in ECEF and the observer is specified in geodetic.

    Model
    -----
    1) Convert target ECEF -> ENU about observer using ecef2enu()
    2) Convert ENU -> AER using enu2aer()

    Conventions
    -----------
    - ECEF inputs are meters.
    - Observer lat/lon are radians, height is meters above WGS84 ellipsoid.
    - Outputs: azimuth in [0,2π) rad, elevation in radians, range in meters.

    Parameters
    ----------
    x_m, y_m, z_m : float
        Target ECEF coordinates [m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic latitude [rad], longitude [rad], height [m].

    Returns
    -------
    (az_rad, el_rad, srange_m) : tuple[float, float, float]
        Azimuth [rad], elevation [rad], slant range [m].

    Notes
    -----
    - This is geometric AER in the observer’s local tangent frame.
    - No atmospheric refraction correction is applied.
    """
    e, n, u = ecef2enu(x_m, y_m, z_m, lat0_rad, lon0_rad, h0_m)
    return enu2aer(e, n, u)


@njit(cache=True, parallel=True)
def ecef2aer_vec_xyz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized ECEF->AER conversion in parallel (prange).

    Parameters
    ----------
    x_m, y_m, z_m : np.ndarray
        1D arrays (N,) of target ECEF coordinates [m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic latitude [rad], longitude [rad], height [m].

    Returns
    -------
    az_rad, el_rad, srange_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of azimuth [rad], elevation [rad], slant range [m].
    """
    npts = x_m.shape[0]
    if y_m.shape[0] != npts or z_m.shape[0] != npts:
        raise ValueError("x_m, y_m, z_m must have the same length")

    az = np.empty(npts, dtype=np.float64)
    el = np.empty(npts, dtype=np.float64)
    sr = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        a, e, r = ecef2aer(x_m[i], y_m[i], z_m[i], lat0_rad, lon0_rad, h0_m)
        az[i] = a
        el[i] = e
        sr[i] = r

    return az, el, sr
