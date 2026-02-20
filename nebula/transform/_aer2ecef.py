# _aer2ecef.py
import math
import numpy as np
from numba import njit, prange

from ._geodetic2ecef import geodetic2ecef

from ._enu2ecef import enu2ecef_delta


@njit(cache=True, inline="always")
def aer2enu(az_rad: float, el_rad: float, srange_m: float):
    """
    Convert AER (azimuth, elevation, slant range) to local ENU coordinates.

    Definitions (local tangent frame)
    -------------------------------
    - azimuth az is measured clockwise from North toward East:
        az=0   => +North
        az=π/2 => +East
    - elevation el is measured above the local horizon:
        el=0   => horizontal
        el=π/2 => straight up
    - slant range is Euclidean distance to the target.

    With r = srange * cos(el):
        e = r * sin(az)
        n = r * cos(az)
        u = srange * sin(el)

    Parameters
    ----------
    az_rad : float
        Azimuth in radians (any real; periodic).
    el_rad : float
        Elevation in radians.
    srange_m : float
        Slant range in meters (should be >= 0).

    Returns
    -------
    (e_m, n_m, u_m) : tuple[float, float, float]
        ENU coordinates in meters.

    Notes
    -----
    - If srange_m == 0, returns (0,0,0).
    """
    if srange_m == 0.0:
        return 0.0, 0.0, 0.0

    sel = math.sin(el_rad)
    cel = math.cos(el_rad)
    saz = math.sin(az_rad)
    caz = math.cos(az_rad)

    r = srange_m * cel
    e = r * saz
    n = r * caz
    u = srange_m * sel
    return e, n, u


@njit(cache=True, inline="always")
def aer2ecef(
    az_rad: float,
    el_rad: float,
    srange_m: float,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Convert AER (azimuth, elevation, slant range) from an observer into target ECEF.

    Model
    -----
    1) Convert observer geodetic (lat0, lon0, h0) -> ECEF origin (x0,y0,z0).
    2) Convert (az, el, srange) -> local ENU vector (e,n,u).
    3) Rotate ENU vector -> ECEF delta (dx,dy,dz) at the observer tangent frame.
    4) Translate: target_ecef = origin_ecef + delta

    Conventions
    -----------
    - lat/lon in radians.
    - heights and ranges in meters.
    - ECEF outputs in meters.

    Parameters
    ----------
    az_rad : float
        Azimuth [rad], clockwise from North toward East.
    el_rad : float
        Elevation [rad] above horizon.
    srange_m : float
        Slant range [m] (>= 0).
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic latitude [rad], longitude [rad], height [m].

    Returns
    -------
    (x_m, y_m, z_m) : tuple[float, float, float]
        Target ECEF coordinates [m].

    Notes
    -----
    - No refraction correction is applied.
    - If srange_m == 0, returns the observer ECEF position.
    """
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    e, n, u = aer2enu(az_rad, el_rad, srange_m)
    dx, dy, dz = enu2ecef_delta(e, n, u, lat0_rad, lon0_rad)

    return x0 + dx, y0 + dy, z0 + dz


@njit(cache=True, parallel=True)
def aer2ecef_vec_aer(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    srange_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized AER->ECEF conversion in parallel (prange).

    Parameters
    ----------
    az_rad, el_rad, srange_m : np.ndarray
        1D arrays (N,) of azimuth [rad], elevation [rad], slant range [m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coordinates defining the local tangent frame.

    Returns
    -------
    x_m, y_m, z_m : tuple[np.ndarray, np.ndarray, np.ndarray]
        1D arrays (N,) of target ECEF coordinates [m].
    """
    npts = az_rad.shape[0]
    if el_rad.shape[0] != npts or srange_m.shape[0] != npts:
        raise ValueError("az_rad, el_rad, srange_m must have the same length")

    x = np.empty(npts, dtype=np.float64)
    y = np.empty(npts, dtype=np.float64)
    z = np.empty(npts, dtype=np.float64)

    # Precompute observer origin once
    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    # Precompute trig for ENU->ECEF delta once
    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(npts):
        az = az_rad[i]
        el = el_rad[i]
        sr = srange_m[i]

        if sr == 0.0:
            x[i] = x0
            y[i] = y0
            z[i] = z0
            continue

        sel = math.sin(el)
        cel = math.cos(el)
        saz = math.sin(az)
        caz = math.cos(az)

        r = sr * cel
        e = r * saz
        n = r * caz
        u = sr * sel

        # enu2ecef_delta(e,n,u,lat0,lon0) expanded
        dx = -slon * e - slat * clon * n + clat * clon * u
        dy = clon * e - slat * slon * n + clat * slon * u
        dz = clat * n + slat * u

        x[i] = x0 + dx
        y[i] = y0 + dy
        z[i] = z0 + dz

    return x, y, z


@njit(cache=True, parallel=True)
def aer2ecef_vec_aer3(
    aer_rad_m: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    h0_m: float,
):
    """
    Vectorized AER->ECEF conversion for an (N,3) array in parallel.

    Parameters
    ----------
    aer_rad_m : np.ndarray
        2D array (N,3) with columns [az_rad, el_rad, srange_m].
    lat0_rad, lon0_rad, h0_m : float
        Observer geodetic coordinates.

    Returns
    -------
    r_ecef_m : np.ndarray
        2D array (N,3) of target ECEF coordinates [m].
    """
    if aer_rad_m.ndim != 2 or aer_rad_m.shape[1] != 3:
        raise ValueError("aer_rad_m must have shape (N, 3)")

    npts = aer_rad_m.shape[0]
    out = np.empty((npts, 3), dtype=np.float64)

    x0, y0, z0 = geodetic2ecef(lat0_rad, lon0_rad, h0_m)

    slat = math.sin(lat0_rad)
    clat = math.cos(lat0_rad)
    slon = math.sin(lon0_rad)
    clon = math.cos(lon0_rad)

    for i in prange(npts):
        az = aer_rad_m[i, 0]
        el = aer_rad_m[i, 1]
        sr = aer_rad_m[i, 2]

        if sr == 0.0:
            out[i, 0] = x0
            out[i, 1] = y0
            out[i, 2] = z0
            continue

        sel = math.sin(el)
        cel = math.cos(el)
        saz = math.sin(az)
        caz = math.cos(az)

        r = sr * cel
        e = r * saz
        n = r * caz
        u = sr * sel

        dx = -slon * e - slat * clon * n + clat * clon * u
        dy = clon * e - slat * slon * n + clat * slon * u
        dz = clat * n + slat * u

        out[i, 0] = x0 + dx
        out[i, 1] = y0 + dy
        out[i, 2] = z0 + dz

    return out
