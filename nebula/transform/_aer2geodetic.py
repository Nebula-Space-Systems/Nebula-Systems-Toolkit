# _aer2geodetic.py
import numpy as np
from numba import njit, prange

from ._aer2ecef import aer2ecef

from ._ecef2geodetic import ecef2geodetic


@njit(cache=True, inline="always")
def aer2geodetic(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert AER (azimuth, elevation, slant range) from an observer into target geodetic
    coordinates (latitude, longitude, height) on WGS84.

    Model
    -----
    1) Convert AER -> target ECEF using aer2ecef()
    2) Convert target ECEF -> target geodetic using ecef2geodetic()

    Conventions
    -----------
    - azimuth [rad] is clockwise from North toward East
    - elevation [rad] is above the local horizon
    - slant range [m]
    - observer lat/lon [rad], height [m]
    - output lat/lon [rad], height [m] (lon wrapped per ecef2geodetic())

    Parameters
    ----------
    az_rad, el_rad : float
        Azimuth and elevation in radians.
    srange_m : float
        Slant range in meters.
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coordinates (radians, radians, meters).

    Returns
    -------
    (lat_rad, lon_rad, h_m) : tuple[float, float, float]
        Target geodetic latitude [rad], longitude [rad], height [m].
    """
    x, y, z = aer2ecef(az_rad, el_rad, srange_m, lat0_rad, lon0_rad, h0_m)
    return ecef2geodetic(x, y, z)


@njit(cache=True, parallel=True)
def aer2geodetic_vec_aer(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized AER->geodetic conversion in parallel (prange).

    Parameters
    ----------
    az_rad, el_rad, srange_m : np.ndarray
        1D arrays (N,) of azimuth [rad], elevation [rad], slant range [m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coords.

    Returns
    -------
    lat_rad, lon_rad, h_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of latitude [rad], longitude [rad], height [m].
    """
    npts = az_rad.shape[0]
    if el_rad.shape[0] != npts or srange_m.shape[0] != npts:
        raise ValueError("az_rad, el_rad, srange_m must have the same length")

    lat = np.empty(npts, dtype=np.float64)
    lon = np.empty(npts, dtype=np.float64)
    h = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        la, lo, hi = aer2geodetic(
            az_rad[i], el_rad[i], srange_m[i], lat0_rad, lon0_rad, h0_m
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h


@njit(cache=True, parallel=True)
def aer2geodetic_vec_aer3(
    aer_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized AER->geodetic conversion for an (N,3) array in parallel.

    Parameters
    ----------
    aer_rad_m : np.ndarray
        2D array (N,3) with columns [az_rad, el_rad, srange_m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coords.

    Returns
    -------
    lat_rad, lon_rad, h_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of latitude [rad], longitude [rad], height [m].
    """
    if aer_rad_m.ndim != 2 or aer_rad_m.shape[1] != 3:
        raise ValueError("aer_rad_m must have shape (N, 3)")

    npts = aer_rad_m.shape[0]
    lat = np.empty(npts, dtype=np.float64)
    lon = np.empty(npts, dtype=np.float64)
    h = np.empty(npts, dtype=np.float64)

    for i in prange(npts):
        la, lo, hi = aer2geodetic(
            aer_rad_m[i, 0], aer_rad_m[i, 1], aer_rad_m[i, 2], lat0_rad, lon0_rad, h0_m
        )
        lat[i] = la
        lon[i] = lo
        h[i] = hi

    return lat, lon, h
