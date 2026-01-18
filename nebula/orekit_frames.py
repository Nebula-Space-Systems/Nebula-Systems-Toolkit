"""
High-throughput frame transforms for Orekit (via orekit-jpype) operating on NumPy arrays.

This module provides vectorized position and position+velocity (PV) transforms between common
Orekit frames, with special care to avoid per-state Java calls whenever possible.

Key features
------------
- Supports transforming **large batches** of states (N x 3 arrays) efficiently.
- Accepts times as:
  - org.orekit.time.AbsoluteDate (scalar)
  - astropy.time.Time (scalar) (optional dependency)
  - Iterable of AbsoluteDate / scalar astropy Time (length N)
  - astropy.time.Time array (shape (N,)) (optional dependency)
- Automatically optimizes the multi-time case by grouping repeated epochs:
  - If your per-row dates reuse a small set of AbsoluteDate objects (common K times),
    transforms are applied K times to slices instead of N times.

Performance model
-----------------
- **Single time for all rows**: one Orekit transform -> pure NumPy (fast).
- **Multiple times**:
  - If times repeat: group by unique times -> K transforms + vectorized NumPy (fast).
  - If times are all unique: must evaluate Orekit transform per row (slow; unavoidable).

Notes on frames
---------------
- GCRF, ITRF, TEME, and EME2000 are provided by Orekit FramesFactory.
- Astropy uses close analogs (GCRS, ITRS), but they are not identical definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Union

import jdk4py
import numpy as np
import orekit_jpype

# -----------------------------------------------------------------------------
# Orekit JVM + data setup
# -----------------------------------------------------------------------------
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
orekit_jpype.initVM()

from orekit_jpype.pyhelpers import setup_orekit_curdir

setup_orekit_curdir(
    filename=os.path.join(os.path.dirname(__file__), "..", "data", "orekit-data")
)

from org.orekit.frames import FramesFactory  # type: ignore
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
from org.orekit.utils import IERSConventions  # type: ignore

# -----------------------------------------------------------------------------
# Optional Astropy support
# -----------------------------------------------------------------------------
try:
    from astropy.time import Time as AstropyTime  # type: ignore
except Exception:  # pragma: no cover
    AstropyTime = None  # type: ignore

DateLike = Union[AbsoluteDate, "AstropyTime"]
DatesLike = Union[DateLike, Iterable[DateLike]]

# -----------------------------------------------------------------------------
# Global caches (avoid repeated JVM calls / object creation)
# -----------------------------------------------------------------------------
_UTC = None
_GCRF = None
_ITRF_CACHE = {}  # key: (iers, simple_eop) -> Frame

_J2000 = None
_J2000_GCRF_PARTS = None
_GCRF_J2000_PARTS = None

_TEME = None

# -----------------------------------------------------------------------------
# WGS84 constants for geodetic <-> ITRF (pure NumPy)
# -----------------------------------------------------------------------------
_WGS84_A = 6378137.0  # semi-major [m]
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)  # semi-minor [m]
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # first eccentricity^2
_WGS84_EP2 = (_WGS84_A * _WGS84_A - _WGS84_B * _WGS84_B) / (_WGS84_B * _WGS84_B)


# =============================================================================
# Frame getters (cached)
# =============================================================================
def _get_utc():
    """Return Orekit UTC time scale (cached)."""
    global _UTC
    if _UTC is None:
        _UTC = TimeScalesFactory.getUTC()
    return _UTC


def _get_gcrf():
    """Return Orekit GCRF frame (cached)."""
    global _GCRF
    if _GCRF is None:
        _GCRF = FramesFactory.getGCRF()
    return _GCRF


def _get_itrf(iers: IERSConventions, simple_eop: bool):
    """
    Return an Orekit ITRF frame for the requested IERS convention and EOP handling.

    Parameters
    ----------
    iers : IERSConventions
        IERS convention (e.g., IERS_2010).
    simple_eop : bool
        Orekit's "simple EOP" flag. If True, uses simplified EOP interpolation/handling.

    Returns
    -------
    org.orekit.frames.Frame
        The ITRF frame object.
    """
    key = (iers, bool(simple_eop))
    fr = _ITRF_CACHE.get(key)
    if fr is None:
        fr = FramesFactory.getITRF(iers, simple_eop)
        _ITRF_CACHE[key] = fr
    return fr


def _get_j2000():
    """
    Return Orekit's EME2000 frame (often referred to as 'J2000') (cached).

    In Orekit, the commonly-used J2000-like inertial frame is FramesFactory.getEME2000().
    """
    global _J2000
    if _J2000 is None:
        _J2000 = FramesFactory.getEME2000()
    return _J2000


def _get_teme():
    """
    Return Orekit TEME frame (cached).

    TEME (True Equator, Mean Equinox) is commonly used with SGP4/TLE state outputs.
    """
    global _TEME
    if _TEME is None:
        _TEME = FramesFactory.getTEME()
    return _TEME


# =============================================================================
# Public API: J2000 (EME2000) <-> GCRF (constant inertial bias)
# =============================================================================
def _get_j2000_epoch() -> AbsoluteDate:
    """Return the Orekit J2000 epoch constant."""
    return AbsoluteDate.J2000_EPOCH


def _get_j2000_gcrf_parts():
    """
    Cache constant kinematic transform parts between EME2000 (J2000) and GCRF.

    In Orekit these are inertial frames with an extremely small, effectively constant rotation bias.
    We cache the transform at J2000 epoch and apply it via NumPy for speed.
    """
    global _J2000_GCRF_PARTS, _GCRF_J2000_PARTS
    if _J2000_GCRF_PARTS is None or _GCRF_J2000_PARTS is None:
        date0 = _get_j2000_epoch()
        j2000 = _get_j2000()
        gcrf = _get_gcrf()
        _J2000_GCRF_PARTS = _get_kinematic_parts(j2000, gcrf, date0)
        _GCRF_J2000_PARTS = _get_kinematic_parts(gcrf, j2000, date0)
    return _J2000_GCRF_PARTS, _GCRF_J2000_PARTS


def j2000_to_gcrf_positions(r_j2000_m: np.ndarray) -> np.ndarray:
    """
    Convert positions from EME2000 (J2000) to GCRF.

    This transform is treated as time-independent here (constant inertial bias).
    It is implemented as a cached rotation+translation and applied in pure NumPy.

    Parameters
    ----------
    r_j2000_m : ndarray, shape (N, 3)
        Positions in EME2000 meters.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in GCRF meters.
    """
    r = _normalize_states_array(r_j2000_m, "r_j2000_m")
    parts, _ = _get_j2000_gcrf_parts()
    return _apply_transform_positions(parts, r)


def gcrf_to_j2000_positions(r_gcrf_m: np.ndarray) -> np.ndarray:
    """
    Convert positions from GCRF to EME2000 (J2000).

    This transform is treated as time-independent here (constant inertial bias).

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in EME2000 meters.
    """
    r = _normalize_states_array(r_gcrf_m, "r_gcrf_m")
    _, parts = _get_j2000_gcrf_parts()
    return _apply_transform_positions(parts, r)


def j2000_to_gcrf_pv(
    r_j2000_m: np.ndarray, v_j2000_mps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from EME2000 (J2000) to GCRF.

    This transform is treated as time-independent here (constant inertial bias).

    Parameters
    ----------
    r_j2000_m : ndarray, shape (N, 3)
        Positions in EME2000 meters.
    v_j2000_mps : ndarray, shape (N, 3)
        Velocities in EME2000 m/s.

    Returns
    -------
    (r_gcrf_m, v_gcrf_mps) : tuple of ndarrays
        Transformed positions and velocities in GCRF.
    """
    r = _normalize_states_array(r_j2000_m, "r_j2000_m")
    v = _normalize_states_array(v_j2000_mps, "v_j2000_mps")
    if r.shape[0] != v.shape[0]:
        raise ValueError(
            f"r_j2000_m and v_j2000_mps must share N; got {r.shape[0]} and {v.shape[0]}"
        )
    parts, _ = _get_j2000_gcrf_parts()
    return _apply_transform_pv(parts, r, v)


def gcrf_to_j2000_pv(
    r_gcrf_m: np.ndarray, v_gcrf_mps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from GCRF to EME2000 (J2000).

    This transform is treated as time-independent here (constant inertial bias).

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    v_gcrf_mps : ndarray, shape (N, 3)
        Velocities in GCRF m/s.

    Returns
    -------
    (r_j2000_m, v_j2000_mps) : tuple of ndarrays
        Transformed positions and velocities in EME2000.
    """
    r = _normalize_states_array(r_gcrf_m, "r_gcrf_m")
    v = _normalize_states_array(v_gcrf_mps, "v_gcrf_mps")
    if r.shape[0] != v.shape[0]:
        raise ValueError(
            f"r_gcrf_m and v_gcrf_mps must share N; got {r.shape[0]} and {v.shape[0]}"
        )
    _, parts = _get_j2000_gcrf_parts()
    return _apply_transform_pv(parts, r, v)


# =============================================================================
# Public API: TEME <-> GCRF / ITRF
# =============================================================================
def teme_to_gcrf_positions(r_teme_m: np.ndarray, dates: DatesLike) -> np.ndarray:  # type: ignore
    """
    Convert positions from TEME to GCRF.

    Parameters
    ----------
    r_teme_m : ndarray, shape (N, 3)
        Positions in TEME meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        - Scalar AbsoluteDate / scalar astropy Time: applies one transform to all rows (fast).
        - Iterable length N: per-row epochs (grouped when repeated).
        - Astropy Time array shape (N,): grouped by unique JD pairs.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in GCRF meters.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_positions(r_teme_m, dates, teme, gcrf)


def gcrf_to_teme_positions(r_gcrf_m: np.ndarray, dates: DatesLike) -> np.ndarray:  # type: ignore
    """
    Convert positions from GCRF to TEME.

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in TEME meters.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_positions(r_gcrf_m, dates, gcrf, teme)


def teme_to_gcrf_pv(
    r_teme_m: np.ndarray, v_teme_mps: np.ndarray, dates: DatesLike  # type: ignore
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from TEME to GCRF.

    Parameters
    ----------
    r_teme_m : ndarray, shape (N, 3)
        Positions in TEME meters.
    v_teme_mps : ndarray, shape (N, 3)
        Velocities in TEME m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).

    Returns
    -------
    (r_gcrf_m, v_gcrf_mps) : tuple of ndarrays
        Transformed positions and velocities in GCRF.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_pv(r_teme_m, v_teme_mps, dates, teme, gcrf)


def gcrf_to_teme_pv(
    r_gcrf_m: np.ndarray, v_gcrf_mps: np.ndarray, dates: DatesLike  # type: ignore
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from GCRF to TEME.

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    v_gcrf_mps : ndarray, shape (N, 3)
        Velocities in GCRF m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).

    Returns
    -------
    (r_teme_m, v_teme_mps) : tuple of ndarrays
        Transformed positions and velocities in TEME.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_pv(r_gcrf_m, v_gcrf_mps, dates, gcrf, teme)


def teme_to_itrf_positions(
    r_teme_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert positions from TEME to ITRF.

    Parameters
    ----------
    r_teme_m : ndarray, shape (N, 3)
        Positions in TEME meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).
    iers : IERSConventions
        IERS convention for ITRF realization (default: IERS_2010).
    simple_eop : bool
        Orekit "simple EOP" flag (default: True).

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in ITRF meters.
    """
    teme = _get_teme()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_positions(r_teme_m, dates, teme, itrf)


def itrf_to_teme_positions(
    r_itrf_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert positions from ITRF to TEME.

    Parameters
    ----------
    r_itrf_m : ndarray, shape (N, 3)
        Positions in ITRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).
    iers : IERSConventions
        IERS convention for ITRF realization (default: IERS_2010).
    simple_eop : bool
        Orekit "simple EOP" flag (default: True).

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in TEME meters.
    """
    teme = _get_teme()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_positions(r_itrf_m, dates, itrf, teme)


def teme_to_itrf_pv(
    r_teme_m: np.ndarray,
    v_teme_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from TEME to ITRF.

    Parameters
    ----------
    r_teme_m : ndarray, shape (N, 3)
        Positions in TEME meters.
    v_teme_mps : ndarray, shape (N, 3)
        Velocities in TEME m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).
    iers : IERSConventions
        IERS convention for ITRF realization (default: IERS_2010).
    simple_eop : bool
        Orekit "simple EOP" flag (default: True).

    Returns
    -------
    (r_itrf_m, v_itrf_mps) : tuple of ndarrays
        Transformed positions and velocities in ITRF.
    """
    teme = _get_teme()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_pv(r_teme_m, v_teme_mps, dates, teme, itrf)


def itrf_to_teme_pv(
    r_itrf_m: np.ndarray,
    v_itrf_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from ITRF to TEME.

    Parameters
    ----------
    r_itrf_m : ndarray, shape (N, 3)
        Positions in ITRF meters.
    v_itrf_mps : ndarray, shape (N, 3)
        Velocities in ITRF m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_positions`).
    iers : IERSConventions
        IERS convention for ITRF realization (default: IERS_2010).
    simple_eop : bool
        Orekit "simple EOP" flag (default: True).

    Returns
    -------
    (r_teme_m, v_teme_mps) : tuple of ndarrays
        Transformed positions and velocities in TEME.
    """
    teme = _get_teme()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_pv(r_itrf_m, v_itrf_mps, dates, itrf, teme)


# =============================================================================
# Public API: Geodetic (WGS84) <-> ITRF (ECEF) + pipelines via GCRF
# =============================================================================
def geodetic_to_itrf_positions(
    lla: np.ndarray,
    *,
    degrees: bool = False,
) -> np.ndarray:
    """
    Convert WGS84 geodetic coordinates to ITRF/ECEF Cartesian positions.

    This is a pure-NumPy implementation intended to be fast for large batches.

    Parameters
    ----------
    lla : ndarray
        Geodetic coordinates. Accepted shapes:
        - (N, 3): [lat, lon, alt]
        - (3,):  [lat, lon, alt]
        - (N, 2): [lat, lon] with alt assumed 0
        - (2,):  [lat, lon] with alt assumed 0

        Latitude/longitude are radians unless `degrees=True`. Altitude is meters.
    degrees : bool
        If True, interpret lat/lon in degrees.

    Returns
    -------
    ndarray, shape (N, 3)
        ITRF/ECEF positions [x, y, z] in meters.
    """
    lla_r = _normalize_lla_array(lla, "lla").copy()
    if degrees:
        lla_r[:, 0:2] = np.deg2rad(lla_r[:, 0:2])

    lat = lla_r[:, 0]
    lon = lla_r[:, 1]
    h = lla_r[:, 2]

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Prime vertical radius of curvature
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + h) * sin_lat

    return np.stack([x, y, z], axis=1)


def itrf_to_geodetic_positions(
    r_itrf_m: np.ndarray,
    *,
    degrees: bool = False,
    wrap_lon: bool = True,
) -> np.ndarray:
    """
    Convert ITRF/ECEF Cartesian positions to WGS84 geodetic coordinates.

    This is a pure-NumPy implementation intended to be fast for large batches.

    Parameters
    ----------
    r_itrf_m : ndarray
        ITRF/ECEF positions. Accepted shapes:
        - (N, 3)
        - (3,)
        Units: meters.
    degrees : bool
        If True, return lat/lon in degrees. Otherwise radians.
    wrap_lon : bool
        If True, wrap longitude to (-pi, pi] (or equivalently (-180, 180] in degrees).

    Returns
    -------
    ndarray, shape (N, 3)
        Geodetic coordinates [lat, lon, alt], where lat/lon are radians (or degrees) and alt is meters.
    """
    r = _normalize_states_array(np.atleast_2d(r_itrf_m), "r_itrf_m")
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    p = np.hypot(x, y)
    lon = np.arctan2(y, x)

    # Bowring’s method
    theta = np.arctan2(z * _WGS84_A, p * _WGS84_B)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    lat = np.arctan2(
        z + _WGS84_EP2 * _WGS84_B * sin_t * sin_t * sin_t,
        p - _WGS84_E2 * _WGS84_A * cos_t * cos_t * cos_t,
    )

    pole = p < 1e-12
    if np.any(pole):
        lat[pole] = np.sign(z[pole]) * (0.5 * np.pi)
        lon[pole] = 0.0

    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    cos_lat = np.cos(lat)
    h = np.empty_like(lat)
    npole = ~pole
    if np.any(npole):
        h[npole] = p[npole] / cos_lat[npole] - N[npole]
    if np.any(pole):
        h[pole] = np.abs(z[pole]) - _WGS84_B

    if wrap_lon:
        lon = _wrap_lon_rad(lon)

    if degrees:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

    return np.stack([lat, lon, h], axis=1)


def geodetic_to_gcrf_positions(
    lla: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    degrees: bool = False,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert WGS84 geodetic coordinates to GCRF positions.

    Pipeline:
        geodetic -> ITRF(ECEF) via NumPy -> GCRF via Orekit transform

    Parameters
    ----------
    lla : ndarray
        Geodetic coordinates (see `geodetic_to_itrf_positions`).
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification used for ITRF->GCRF transform.
    degrees : bool
        If True, interpret lat/lon in degrees.
    iers : IERSConventions
        IERS convention used for the ITRF frame.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in GCRF meters.
    """
    r_itrf = geodetic_to_itrf_positions(lla, degrees=degrees)
    return itrf_to_gcrf_positions(r_itrf, dates, iers=iers, simple_eop=simple_eop)


def gcrf_to_geodetic_positions(
    r_gcrf_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    degrees: bool = False,
    wrap_lon: bool = True,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert GCRF positions to WGS84 geodetic coordinates.

    Pipeline:
        GCRF -> ITRF via Orekit transform -> geodetic via NumPy

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification used for GCRF->ITRF transform.
    degrees : bool
        If True, return lat/lon in degrees.
    wrap_lon : bool
        If True, wrap longitude to (-pi, pi] (or (-180, 180] in degrees).
    iers : IERSConventions
        IERS convention used for the ITRF frame.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    ndarray, shape (N, 3)
        Geodetic coordinates [lat, lon, alt].
    """
    r_itrf = gcrf_to_itrf_positions(r_gcrf_m, dates, iers=iers, simple_eop=simple_eop)
    return itrf_to_geodetic_positions(r_itrf, degrees=degrees, wrap_lon=wrap_lon)


# =============================================================================
# Public API: GCRF <-> ITRF (positions + PV)
# =============================================================================
def gcrf_to_itrf_positions(
    r_gcrf_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert positions from GCRF to ITRF.

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification:
        - Scalar AbsoluteDate / scalar astropy Time: one transform applied to all rows.
        - Iterable length N: per-row epochs (fast if repeated objects).
        - Astropy Time array shape (N,): grouped by unique times.
    iers : IERSConventions
        IERS convention used for ITRF.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in ITRF meters.
    """
    gcrf = _get_gcrf()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_positions(r_gcrf_m, dates, gcrf, itrf)


def itrf_to_gcrf_positions(
    r_itrf_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> np.ndarray:
    """
    Convert positions from ITRF to GCRF.

    Parameters
    ----------
    r_itrf_m : ndarray, shape (N, 3)
        Positions in ITRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `gcrf_to_itrf_positions`).
    iers : IERSConventions
        IERS convention used for ITRF.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in GCRF meters.
    """
    gcrf = _get_gcrf()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_positions(r_itrf_m, dates, itrf, gcrf)


def gcrf_to_itrf_pv(
    r_gcrf_m: np.ndarray,
    v_gcrf_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from GCRF to ITRF.

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    v_gcrf_mps : ndarray, shape (N, 3)
        Velocities in GCRF m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `gcrf_to_itrf_positions`).
    iers : IERSConventions
        IERS convention used for ITRF.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    (r_itrf_m, v_itrf_mps) : tuple of ndarrays
        Transformed positions and velocities in ITRF.
    """
    gcrf = _get_gcrf()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_pv(r_gcrf_m, v_gcrf_mps, dates, gcrf, itrf)


def itrf_to_gcrf_pv(
    r_itrf_m: np.ndarray,
    v_itrf_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    *,
    iers: IERSConventions = IERSConventions.IERS_2010,
    simple_eop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position+velocity (PV) from ITRF to GCRF.

    Parameters
    ----------
    r_itrf_m : ndarray, shape (N, 3)
        Positions in ITRF meters.
    v_itrf_mps : ndarray, shape (N, 3)
        Velocities in ITRF m/s.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `gcrf_to_itrf_positions`).
    iers : IERSConventions
        IERS convention used for ITRF.
    simple_eop : bool
        Orekit "simple EOP" flag.

    Returns
    -------
    (r_gcrf_m, v_gcrf_mps) : tuple of ndarrays
        Transformed positions and velocities in GCRF.
    """
    gcrf = _get_gcrf()
    itrf = _get_itrf(iers, simple_eop)
    return _transform_pv(r_itrf_m, v_itrf_mps, dates, itrf, gcrf)


# =============================================================================
# Internal utilities
# =============================================================================
def _is_astropy_time(x: Any) -> bool:
    """Return True if x is an astropy.time.Time instance (and astropy is installed)."""
    return AstropyTime is not None and isinstance(x, AstropyTime)


def _is_astropy_scalar_time(t: Any) -> bool:
    """Return True if t is a scalar astropy time (shape == ())."""
    return _is_astropy_time(t) and getattr(t, "shape", None) == ()


def _is_astropy_array_time(t: Any) -> bool:
    """Return True if t is a non-scalar astropy time (shape != ())."""
    return _is_astropy_time(t) and getattr(t, "shape", None) != ()


def _astropy_scalar_to_datetime_utc(t: "AstropyTime") -> datetime:  # type: ignore
    """Convert scalar astropy Time to timezone-aware UTC datetime."""
    dt = t.utc.to_datetime(timezone=timezone.utc)
    if isinstance(dt, np.ndarray):
        dt = dt.item()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _absolutedate_from_datetime_utc(dt: datetime) -> AbsoluteDate:
    """Convert a timezone-aware (or assumed UTC) datetime to Orekit AbsoluteDate (UTC scale)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    sec = float(dt.second) + float(dt.microsecond) * 1e-6
    return AbsoluteDate(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec, _get_utc())


def _to_absolutedate(date: DateLike) -> AbsoluteDate:  # type: ignore
    """
    Convert supported time representations to Orekit AbsoluteDate.

    Accepted:
    - AbsoluteDate
    - scalar astropy Time (if astropy installed)

    Raises
    ------
    TypeError
        If an unsupported type is passed.
    """
    if isinstance(date, AbsoluteDate):
        return date
    if _is_astropy_scalar_time(date):
        return _absolutedate_from_datetime_utc(_astropy_scalar_to_datetime_utc(date))  # type: ignore[arg-type]
    raise TypeError(
        "date must be an org.orekit.time.AbsoluteDate or a scalar astropy.time.Time (if astropy is installed)"
    )


def _normalize_states_array(x: np.ndarray, name: str) -> np.ndarray:
    """Ensure x is a contiguous float64 array of shape (N, 3)."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {a.shape}")
    return a


def _normalize_lla_array(x: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize geodetic inputs to shape (N, 3): [lat, lon, alt].

    Accepted input shapes:
    - (3,) -> (1, 3)
    - (2,) -> (1, 3) with alt=0
    - (N, 3)
    - (N, 2) with alt=0
    """
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] == 3:
            return a.reshape(1, 3)
        if a.shape[0] == 2:
            out = np.empty((1, 3), dtype=np.float64)
            out[0, 0:2] = a
            out[0, 2] = 0.0
            return out
        raise ValueError(f"{name} must be (3,), (2,), (N,3), or (N,2); got {a.shape}")
    if a.ndim == 2:
        if a.shape[1] == 3:
            return a
        if a.shape[1] == 2:
            out = np.empty((a.shape[0], 3), dtype=np.float64)
            out[:, 0:2] = a
            out[:, 2] = 0.0
            return out
        raise ValueError(f"{name} must have second dim 2 or 3; got {a.shape}")
    raise ValueError(f"{name} must be 1D or 2D; got {a.shape}")


def _wrap_lon_rad(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude to (-pi, pi]."""
    return (lon + np.pi) % (2.0 * np.pi) - np.pi


def _vec3_to_np(v) -> np.ndarray:
    """Convert an Orekit/Hipparchus Vector3D-like object to numpy array (3,)."""
    return np.array([v.getX(), v.getY(), v.getZ()], dtype=np.float64)


@dataclass(frozen=True)
class _KinematicParts:
    """
    Minimal kinematic representation of an Orekit frame transform for vectorized application.

    Attributes
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix mapping vectors in old frame to new frame: v_new = R * v_old.
    t : ndarray, shape (3,)
        Translation of origin (in new frame coordinates).
    v : ndarray, shape (3,)
        Translation rate (velocity of origin, in new frame coordinates).
    omega : ndarray, shape (3,)
        Rotation rate vector (in new frame coordinates), used for PV transforms.
    """

    R: np.ndarray
    t: np.ndarray
    v: np.ndarray
    omega: np.ndarray


def _get_kinematic_parts(from_frame, to_frame, date: AbsoluteDate) -> _KinematicParts:
    """
    Extract rotation/translation/velocity/rotation-rate from Orekit Transform into NumPy arrays.
    """
    tr = from_frame.getTransformTo(to_frame, date)

    rot = tr.getRotation()
    R = np.array(rot.getMatrix(), dtype=np.float64)

    t = _vec3_to_np(tr.getTranslation())
    v = _vec3_to_np(tr.getVelocity())
    omega = _vec3_to_np(tr.getRotationRate())  # expressed in NEW frame

    return _KinematicParts(R=R, t=t, v=v, omega=omega)


def _apply_transform_positions(parts: _KinematicParts, r_old: np.ndarray) -> np.ndarray:
    """
    Apply a cached kinematic transform to positions.

    r_new = R * r_old + t
    """
    return r_old @ parts.R.T + parts.t


def _apply_transform_pv(
    parts: _KinematicParts,
    r_old: np.ndarray,
    v_old: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a cached kinematic transform to position+velocity (PV).

    Formulation matches Orekit Transform conventions:
      r_new = R * r_old + t
      v_new = R * v_old + v_trans - omega x r_new
    """
    r_new = r_old @ parts.R.T + parts.t
    v_rot = v_old @ parts.R.T
    v_new = v_rot + parts.v - np.cross(parts.omega, r_new)
    return r_new, v_new


def _group_from_inverse(
    inverse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert group labels into index ranges suitable for fast slice-wise application.

    Parameters
    ----------
    inverse : ndarray, shape (N,)
        Integer group ids in [0..K-1] for each row.

    Returns
    -------
    (order, starts, ends)
        - order: indices that sort rows by group id (stable).
        - starts: start offsets into `order` for each group.
        - ends: end offsets into `order` for each group.
    """
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [inverse.size]))
    return order, starts, ends


def _transform_positions_grouped_by_dates(
    r_old: np.ndarray,
    dates_list: list[AbsoluteDate],
    from_frame,
    to_frame,
) -> np.ndarray:
    """
    Transform positions with per-row AbsoluteDate epochs, grouping repeated date *objects*.

    This grouping is by Python object identity (id(date)), which is extremely fast and
    works well when the caller constructs dates as references to a small K-element list:
        dates_per_row = [dates_k[i] for i in date_idx]
    """
    N = r_old.shape[0]
    ids = np.fromiter((id(d) for d in dates_list), dtype=np.int64, count=N)
    uniq_ids, inverse = np.unique(ids, return_inverse=True)
    K = uniq_ids.size

    if K > 0.8 * N:
        out = np.empty_like(r_old)
        for i, d in enumerate(dates_list):
            parts = _get_kinematic_parts(from_frame, to_frame, d)
            out[i] = _apply_transform_positions(parts, r_old[i : i + 1])[0]
        return out

    id_to_date: dict[int, AbsoluteDate] = {}
    for d in dates_list:
        did = id(d)
        if did not in id_to_date:
            id_to_date[did] = d
            if len(id_to_date) == K:
                break

    order, starts, ends = _group_from_inverse(inverse)
    out = np.empty_like(r_old)

    for g in range(K):
        did = int(uniq_ids[g])
        d = id_to_date[did]
        parts = _get_kinematic_parts(from_frame, to_frame, d)
        idx = order[starts[g] : ends[g]]
        out[idx] = _apply_transform_positions(parts, r_old[idx])
    return out


def _transform_pv_grouped_by_dates(
    r_old: np.ndarray,
    v_old: np.ndarray,
    dates_list: list[AbsoluteDate],
    from_frame,
    to_frame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform PV with per-row AbsoluteDate epochs, grouping repeated date *objects*.
    """
    N = r_old.shape[0]
    ids = np.fromiter((id(d) for d in dates_list), dtype=np.int64, count=N)
    uniq_ids, inverse = np.unique(ids, return_inverse=True)
    K = uniq_ids.size

    if K > 0.8 * N:
        r_out = np.empty_like(r_old)
        v_out = np.empty_like(v_old)
        for i, d in enumerate(dates_list):
            parts = _get_kinematic_parts(from_frame, to_frame, d)
            ri, vi = _apply_transform_pv(parts, r_old[i : i + 1], v_old[i : i + 1])
            r_out[i] = ri[0]
            v_out[i] = vi[0]
        return r_out, v_out

    id_to_date: dict[int, AbsoluteDate] = {}
    for d in dates_list:
        did = id(d)
        if did not in id_to_date:
            id_to_date[did] = d
            if len(id_to_date) == K:
                break

    order, starts, ends = _group_from_inverse(inverse)
    r_out = np.empty_like(r_old)
    v_out = np.empty_like(v_old)

    for g in range(K):
        did = int(uniq_ids[g])
        d = id_to_date[did]
        parts = _get_kinematic_parts(from_frame, to_frame, d)
        idx = order[starts[g] : ends[g]]
        ri, vi = _apply_transform_pv(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def _transform_positions_grouped_by_astropy_time(
    r_old: np.ndarray,
    t: "AstropyTime",  # type: ignore
    from_frame,
    to_frame,
) -> np.ndarray:
    """
    Transform positions with astropy Time array epochs by grouping unique times.

    Grouping key is (jd1, jd2) to preserve astropy's high precision representation.
    """
    jd1 = np.ascontiguousarray(t.utc.jd1, dtype=np.float64)
    jd2 = np.ascontiguousarray(t.utc.jd2, dtype=np.float64)
    if jd1.shape != jd2.shape:
        raise ValueError("Astropy Time jd1/jd2 shape mismatch")

    key = np.stack([jd1, jd2], axis=1)
    uniq_key, inverse = np.unique(key, axis=0, return_inverse=True)
    K = uniq_key.shape[0]

    if K > 0.8 * r_old.shape[0]:
        dates_list: list[AbsoluteDate] = []
        for ti in t.utc:
            dates_list.append(_to_absolutedate(ti))
        return _transform_positions_grouped_by_dates(
            r_old, dates_list, from_frame, to_frame
        )

    uniq_dates: list[AbsoluteDate] = []
    for k in range(K):
        ti = AstropyTime(uniq_key[k, 0], uniq_key[k, 1], format="jd", scale="utc")  # type: ignore[misc]
        uniq_dates.append(_to_absolutedate(ti))

    order, starts, ends = _group_from_inverse(inverse)
    out = np.empty_like(r_old)

    for g in range(K):
        parts = _get_kinematic_parts(from_frame, to_frame, uniq_dates[g])
        idx = order[starts[g] : ends[g]]
        out[idx] = _apply_transform_positions(parts, r_old[idx])
    return out


def _transform_pv_grouped_by_astropy_time(
    r_old: np.ndarray,
    v_old: np.ndarray,
    t: "AstropyTime",  # type: ignore
    from_frame,
    to_frame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform PV with astropy Time array epochs by grouping unique times.
    """
    jd1 = np.ascontiguousarray(t.utc.jd1, dtype=np.float64)
    jd2 = np.ascontiguousarray(t.utc.jd2, dtype=np.float64)
    key = np.stack([jd1, jd2], axis=1)
    uniq_key, inverse = np.unique(key, axis=0, return_inverse=True)
    K = uniq_key.shape[0]

    if K > 0.8 * r_old.shape[0]:
        dates_list: list[AbsoluteDate] = []
        for ti in t.utc:
            dates_list.append(_to_absolutedate(ti))
        return _transform_pv_grouped_by_dates(
            r_old, v_old, dates_list, from_frame, to_frame
        )

    uniq_dates: list[AbsoluteDate] = []
    for k in range(K):
        ti = AstropyTime(uniq_key[k, 0], uniq_key[k, 1], format="jd", scale="utc")  # type: ignore[misc]
        uniq_dates.append(_to_absolutedate(ti))

    order, starts, ends = _group_from_inverse(inverse)
    r_out = np.empty_like(r_old)
    v_out = np.empty_like(v_old)

    for g in range(K):
        parts = _get_kinematic_parts(from_frame, to_frame, uniq_dates[g])
        idx = order[starts[g] : ends[g]]
        ri, vi = _apply_transform_pv(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def _transform_positions(
    r_old_m: np.ndarray,
    dates: DatesLike,  # type: ignore
    from_frame,
    to_frame,
) -> np.ndarray:
    """
    Internal vectorized dispatcher for position transforms (N x 3 arrays).

    This implements the "fast path" selection based on time input:
    - scalar time -> one transform applied to all rows
    - astropy Time array -> group unique times, then transform groups
    - iterable length N -> group repeated AbsoluteDate objects where possible
    """
    r_old = _normalize_states_array(r_old_m, "r_old_m")
    N = r_old.shape[0]

    if isinstance(dates, AbsoluteDate) or _is_astropy_scalar_time(dates):
        date0 = _to_absolutedate(dates)  # type: ignore[arg-type]
        parts = _get_kinematic_parts(from_frame, to_frame, date0)
        return _apply_transform_positions(parts, r_old)

    if _is_astropy_array_time(dates):
        if getattr(dates, "shape", None) != (N,):
            raise ValueError(
                f"Astropy Time shape {dates.shape} must be (N,) where N={N}"
            )
        return _transform_positions_grouped_by_astropy_time(r_old, dates, from_frame, to_frame)  # type: ignore[arg-type]

    dates_list_raw = list(dates)  # type: ignore[arg-type]
    if len(dates_list_raw) != N:
        raise ValueError(f"dates length {len(dates_list_raw)} must match N={N}")

    dates_list: list[AbsoluteDate] = []
    append = dates_list.append
    for d in dates_list_raw:
        if isinstance(d, AbsoluteDate):
            append(d)
        elif _is_astropy_scalar_time(d):
            append(_to_absolutedate(d))  # type: ignore[arg-type]
        else:
            raise TypeError(
                "dates iterable must contain AbsoluteDate or scalar astropy Time elements"
            )

    return _transform_positions_grouped_by_dates(
        r_old, dates_list, from_frame, to_frame
    )


def _transform_pv(
    r_old_m: np.ndarray,
    v_old_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    from_frame,
    to_frame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal vectorized dispatcher for PV transforms (N x 3 arrays).

    Same time handling strategy as `_transform_positions`.
    """
    r_old = _normalize_states_array(r_old_m, "r_old_m")
    v_old = _normalize_states_array(v_old_mps, "v_old_mps")
    if v_old.shape[0] != r_old.shape[0]:
        raise ValueError(
            f"r_old_m and v_old_mps must have same N; got {r_old.shape[0]} and {v_old.shape[0]}"
        )
    N = r_old.shape[0]

    if isinstance(dates, AbsoluteDate) or _is_astropy_scalar_time(dates):
        date0 = _to_absolutedate(dates)  # type: ignore[arg-type]
        parts = _get_kinematic_parts(from_frame, to_frame, date0)
        return _apply_transform_pv(parts, r_old, v_old)

    if _is_astropy_array_time(dates):
        if getattr(dates, "shape", None) != (N,):
            raise ValueError(
                f"Astropy Time shape {dates.shape} must be (N,) where N={N}"
            )
        return _transform_pv_grouped_by_astropy_time(r_old, v_old, dates, from_frame, to_frame)  # type: ignore[arg-type]

    dates_list_raw = list(dates)  # type: ignore[arg-type]
    if len(dates_list_raw) != N:
        raise ValueError(f"dates length {len(dates_list_raw)} must match N={N}")

    dates_list: list[AbsoluteDate] = []
    append = dates_list.append
    for d in dates_list_raw:
        if isinstance(d, AbsoluteDate):
            append(d)
        elif _is_astropy_scalar_time(d):
            append(_to_absolutedate(d))  # type: ignore[arg-type]
        else:
            raise TypeError(
                "dates iterable must contain AbsoluteDate or scalar astropy Time elements"
            )

    return _transform_pv_grouped_by_dates(
        r_old, v_old, dates_list, from_frame, to_frame
    )


# -----------------------------
# Tests / benchmarks
# -----------------------------
def _test1():
    import numpy as np
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
    from org.orekit.utils import IERSConventions, PVCoordinates  # type: ignore

    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
    date1 = date0.shiftedBy(123.456)
    date2 = date0.shiftedBy(9876.0)

    # Optional astropy tests
    has_astropy = False
    try:
        from astropy.time import Time as AstropyTime  # type: ignore

        has_astropy = True
    except Exception:
        has_astropy = False

    def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    gcrf = FramesFactory.getGCRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    N = 1000
    rng = np.random.default_rng(0)
    r_gcrf = rng.normal(size=(N, 3)).astype(np.float64)
    r_gcrf *= 7_000_000.0 / np.linalg.norm(r_gcrf, axis=1, keepdims=True)

    # Single date
    r_itrf = gcrf_to_itrf_positions(r_gcrf, date0)
    r_back = itrf_to_gcrf_positions(r_itrf, date0)
    err = max_abs_err(r_gcrf, r_back)
    print(f"[Test 1] Position roundtrip (single date) max abs err: {err:.6e} m")
    assert err < 1e-6

    # Validate vs Orekit direct
    t = gcrf.getTransformTo(itrf, date0)
    idx = rng.integers(0, N, size=10)
    for i in idx:
        v = Vector3D(r_gcrf[i, 0], r_gcrf[i, 1], r_gcrf[i, 2])
        w = t.transformPosition(v)
        ref = np.array([w.getX(), w.getY(), w.getZ()], dtype=np.float64)
        e = float(np.max(np.abs(ref - r_itrf[i])))
        assert e < 1e-8
    print("[Test 1b] Position vs Orekit transformPosition (sample) OK")

    # PV single date
    v_gcrf = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0
    r_itrf2, v_itrf2 = gcrf_to_itrf_pv(r_gcrf, v_gcrf, date0)
    r_back2, v_back2 = itrf_to_gcrf_pv(r_itrf2, v_itrf2, date0)
    err_r = max_abs_err(r_gcrf, r_back2)
    err_v = max_abs_err(v_gcrf, v_back2)
    print(
        f"[Test 2] PV roundtrip (single date) max abs err r: {err_r:.6e} m, v: {err_v:.6e} m/s"
    )
    assert err_r < 1e-6
    assert err_v < 1e-9

    t = gcrf.getTransformTo(itrf, date0)
    for i in idx:
        pv = PVCoordinates(
            Vector3D(r_gcrf[i, 0], r_gcrf[i, 1], r_gcrf[i, 2]),
            Vector3D(v_gcrf[i, 0], v_gcrf[i, 1], v_gcrf[i, 2]),
        )
        pv2 = t.transformPVCoordinates(pv)
        ref_r = np.array(
            [
                pv2.getPosition().getX(),
                pv2.getPosition().getY(),
                pv2.getPosition().getZ(),
            ],
            dtype=np.float64,
        )
        ref_v = np.array(
            [
                pv2.getVelocity().getX(),
                pv2.getVelocity().getY(),
                pv2.getVelocity().getZ(),
            ],
            dtype=np.float64,
        )
        assert np.max(np.abs(ref_r - r_itrf2[i])) < 1e-8
        assert np.max(np.abs(ref_v - v_itrf2[i])) < 1e-8
    print("[Test 2b] PV vs Orekit transformPVCoordinates (sample) OK")

    # Per-row dates with repetition (identity-groupable)
    dates = [date0, date1, date2] * (N // 3) + [date0] * (N % 3)
    dates = dates[:N]
    r_itrf3 = gcrf_to_itrf_positions(r_gcrf, dates)
    r_back3 = itrf_to_gcrf_positions(r_itrf3, dates)
    err3 = max_abs_err(r_gcrf, r_back3)
    print(f"[Test 3] Position roundtrip (per-row dates) max abs err: {err3:.6e} m")
    assert err3 < 1e-6

    r_itrf4, v_itrf4 = gcrf_to_itrf_pv(r_gcrf, v_gcrf, dates)
    r_back4, v_back4 = itrf_to_gcrf_pv(r_itrf4, v_itrf4, dates)
    err4r = max_abs_err(r_gcrf, r_back4)
    err4v = max_abs_err(v_gcrf, v_back4)
    print(
        f"[Test 4] PV roundtrip (per-row dates) max abs err r: {err4r:.6e} m, v: {err4v:.6e} m/s"
    )
    assert err4r < 1e-6
    assert err4v < 1e-9

    if has_astropy:
        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        r_itrf_a = gcrf_to_itrf_positions(r_gcrf, t0)
        r_back_a = itrf_to_gcrf_positions(r_itrf_a, t0)
        erra = max_abs_err(r_gcrf, r_back_a)
        print(
            f"[Test 5] Astropy scalar time position roundtrip max abs err: {erra:.6e} m"
        )
        assert erra < 1e-6

        t_arr = AstropyTime(
            ["2026-01-16T12:00:00", "2026-01-16T12:02:03.456", "2026-01-16T14:44:36.0"],
            scale="utc",
        )
        # Repeat to length N (array-valued astropy Time)
        t_arrN = AstropyTime(np.resize(t_arr.utc.isot, N), scale="utc")

        r_itrf_b = gcrf_to_itrf_positions(r_gcrf, t_arrN)
        r_back_b = itrf_to_gcrf_positions(r_itrf_b, t_arrN)
        errb = max_abs_err(r_gcrf, r_back_b)
        print(
            f"[Test 5b] Astropy per-row time position roundtrip max abs err: {errb:.6e} m"
        )
        assert errb < 1e-6

        r_itrf_c, v_itrf_c = gcrf_to_itrf_pv(r_gcrf, v_gcrf, t0)
        r_back_c, v_back_c = itrf_to_gcrf_pv(r_itrf_c, v_itrf_c, t0)
        errc_r = max_abs_err(r_gcrf, r_back_c)
        errc_v = max_abs_err(v_gcrf, v_back_c)
        print(
            f"[Test 5c] Astropy scalar time PV roundtrip max abs err r: {errc_r:.6e} m, v: {errc_v:.6e} m/s"
        )
        assert errc_r < 1e-6
        assert errc_v < 1e-9

    print("All tests passed.")


def _speed_test():
    """
    Fair speed comparison between:
      - Orekit (this library): gcrf_to_itrf_positions / gcrf_to_itrf_pv
      - Astropy: GCRS -> ITRS positions, and (optionally) velocities via CartesianDifferential

    Fairness rules applied:
      1) Exactly ONE frame transform call per benchmark iteration for both libraries.
      2) Same N, same input arrays.
      3) For multi-time cases, both libraries are grouped by the same K unique times and transform each group once.
      4) Output extraction is included for both (materialize Nx3 numpy arrays).
      5) Astropy single-date benchmark is fixed (previous version called transform_to() 3x).

    Notes:
      - Orekit uses GCRF<->ITRF (IERS, EOP handling depends on Orekit config).
      - Astropy uses GCRS<->ITRS. These are not strictly identical frames, but this is the closest practical match.
      - If you want Astropy to use IERS-A tables, ensure astropy-iers-data is installed and IERS is configured.
    """
    import os
    import timeit

    import numpy as np
    from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore

    # Optional astropy
    try:
        import astropy.units as u  # type: ignore
        from astropy.coordinates import (  # type: ignore
            GCRS,
            ITRS,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy.time import Time as AstropyTime  # type: ignore

        HAS_ASTROPY = True
    except Exception:
        HAS_ASTROPY = False

    # Report thread env that can affect NumPy matmul performance
    thread_env = {
        k: os.environ.get(k)
        for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    }
    print(f"Thread env: {thread_env}")

    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

    rng = np.random.default_rng(0)

    def make_states(N: int):
        r = rng.normal(size=(N, 3)).astype(np.float64)
        r *= 7_000_000.0 / np.linalg.norm(r, axis=1, keepdims=True)
        v = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0
        return r, v

    def bench_min(fn, repeat=5, number=1) -> float:
        return min(timeit.repeat(fn, repeat=repeat, number=number))

    REPEATS = 3
    NUMBER = 1

    # ---------------------------------------------------------------------
    # Helpers: Astropy transforms (single-date + grouped multi-date)
    # ---------------------------------------------------------------------
    def _astropy_pos_single(r_gcrf: np.ndarray, t_ast):
        rep = CartesianRepresentation(
            r_gcrf[:, 0] * u.m, r_gcrf[:, 1] * u.m, r_gcrf[:, 2] * u.m
        )
        g = GCRS(rep, obstime=t_ast)
        itrs = g.transform_to(ITRS(obstime=t_ast))  # ONE transform_to call
        c = itrs.cartesian
        return np.stack(
            [c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], axis=1
        )

    def _astropy_pv_single(r_gcrf: np.ndarray, v_gcrf: np.ndarray, t_ast):
        rep = CartesianRepresentation(
            r_gcrf[:, 0] * u.m,
            r_gcrf[:, 1] * u.m,
            r_gcrf[:, 2] * u.m,
            differentials=CartesianDifferential(
                v_gcrf[:, 0] * (u.m / u.s),
                v_gcrf[:, 1] * (u.m / u.s),
                v_gcrf[:, 2] * (u.m / u.s),
            ),
        )
        g = GCRS(rep, obstime=t_ast)
        itrs = g.transform_to(ITRS(obstime=t_ast))  # ONE transform_to call
        c = itrs.cartesian
        r = np.stack([c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], axis=1)
        # velocity in astropy is in the differential
        d = c.differentials["s"]
        v = np.stack(
            [
                d.d_x.to_value(u.m / u.s),
                d.d_y.to_value(u.m / u.s),
                d.d_z.to_value(u.m / u.s),
            ],
            axis=1,
        )
        return r, v

    def _astropy_pos_grouped(r_gcrf: np.ndarray, times_k, date_idx: np.ndarray):
        # group indices once
        K = len(times_k)
        groups = [np.where(date_idx == i)[0] for i in range(K)]
        out = np.empty_like(r_gcrf)
        for i in range(K):
            g = groups[i]
            rep = CartesianRepresentation(
                r_gcrf[g, 0] * u.m, r_gcrf[g, 1] * u.m, r_gcrf[g, 2] * u.m
            )
            gcrs = GCRS(rep, obstime=times_k[i])
            itrs = gcrs.transform_to(
                ITRS(obstime=times_k[i])
            )  # ONE transform_to call per group
            c = itrs.cartesian
            out[g, 0] = c.x.to_value(u.m)
            out[g, 1] = c.y.to_value(u.m)
            out[g, 2] = c.z.to_value(u.m)
        return out

    def _astropy_pv_grouped(
        r_gcrf: np.ndarray, v_gcrf: np.ndarray, times_k, date_idx: np.ndarray
    ):
        K = len(times_k)
        groups = [np.where(date_idx == i)[0] for i in range(K)]
        r_out = np.empty_like(r_gcrf)
        v_out = np.empty_like(v_gcrf)
        for i in range(K):
            g = groups[i]
            rep = CartesianRepresentation(
                r_gcrf[g, 0] * u.m,
                r_gcrf[g, 1] * u.m,
                r_gcrf[g, 2] * u.m,
                differentials=CartesianDifferential(
                    v_gcrf[g, 0] * (u.m / u.s),
                    v_gcrf[g, 1] * (u.m / u.s),
                    v_gcrf[g, 2] * (u.m / u.s),
                ),
            )
            gcrs = GCRS(rep, obstime=times_k[i])
            itrs = gcrs.transform_to(
                ITRS(obstime=times_k[i])
            )  # ONE transform_to call per group
            c = itrs.cartesian
            r_out[g, 0] = c.x.to_value(u.m)
            r_out[g, 1] = c.y.to_value(u.m)
            r_out[g, 2] = c.z.to_value(u.m)
            d = c.differentials["s"]
            v_out[g, 0] = d.d_x.to_value(u.m / u.s)
            v_out[g, 1] = d.d_y.to_value(u.m / u.s)
            v_out[g, 2] = d.d_z.to_value(u.m / u.s)
        return r_out, v_out

    # ---------------------------------------------------------------------
    # Bench configurations
    # ---------------------------------------------------------------------
    Ns = [10_000, 50_000, 200_000]
    K_fixed = 16
    dt_seconds = 60.0
    orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K_fixed)]

    if HAS_ASTROPY:
        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        ast_times_k = t0 + (np.arange(K_fixed) * dt_seconds) * u.s  # (K,)

    # ---------------------------------------------------------------------
    # A) N sweep
    # ---------------------------------------------------------------------
    print("\n=== N sweep (fair, single-date + K-times) ===")
    for N in Ns:
        r_gcrf, v_gcrf = make_states(N)
        date_idx = np.arange(N) % K_fixed
        orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

        # Orekit single-date
        t_ok_pos_single = bench_min(
            lambda: gcrf_to_itrf_positions(r_gcrf, date0), REPEATS, NUMBER
        )
        t_ok_pv_single = bench_min(
            lambda: gcrf_to_itrf_pv(r_gcrf, v_gcrf, date0), REPEATS, NUMBER
        )

        # Orekit K-times grouped (AbsoluteDate list with reused objects)
        t_ok_pos_k = bench_min(
            lambda: gcrf_to_itrf_positions(r_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )
        t_ok_pv_k = bench_min(
            lambda: gcrf_to_itrf_pv(r_gcrf, v_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )

        print(
            f"N={N:>8} | Orekit POS single {t_ok_pos_single:.4f}s | PV single {t_ok_pv_single:.4f}s"
            f" | POS K {t_ok_pos_k:.4f}s | PV K {t_ok_pv_k:.4f}s"
        )

        if HAS_ASTROPY:
            # Astropy single-date
            t_ast_pos_single = bench_min(
                lambda: _astropy_pos_single(r_gcrf, ast_times_k[0]), REPEATS, NUMBER
            )
            t_ast_pv_single = bench_min(
                lambda: _astropy_pv_single(r_gcrf, v_gcrf, ast_times_k[0]),
                REPEATS,
                NUMBER,
            )

            # Astropy K-times grouped
            t_ast_pos_k = bench_min(
                lambda: _astropy_pos_grouped(r_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            t_ast_pv_k = bench_min(
                lambda: _astropy_pv_grouped(r_gcrf, v_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )

            print(
                f"          | Astropy POS single {t_ast_pos_single:.4f}s | PV single {t_ast_pv_single:.4f}s"
                f" | POS K {t_ast_pos_k:.4f}s | PV K {t_ast_pv_k:.4f}s"
            )

            print(
                f"          | Ratios Astropy/Orekit: POS single {t_ast_pos_single / t_ok_pos_single:.2f}x,"
                f" PV single {t_ast_pv_single / t_ok_pv_single:.2f}x,"
                f" POS K {t_ast_pos_k / t_ok_pos_k:.2f}x,"
                f" PV K {t_ast_pv_k / t_ok_pv_k:.2f}x"
            )
        else:
            print("          | Astropy not available; skipped Astropy comparisons.")

    # ---------------------------------------------------------------------
    # B) K sweep at fixed N
    # ---------------------------------------------------------------------
    print("\n=== K sweep (fixed N=200k, fair grouped multi-time) ===")
    N = 200_000
    r_gcrf, v_gcrf = make_states(N)

    for K in [1, 4, 16, 64, 128]:
        orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K)]
        date_idx = np.arange(N) % K
        orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

        t_ok_pos_k = bench_min(
            lambda: gcrf_to_itrf_positions(r_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )
        t_ok_pv_k = bench_min(
            lambda: gcrf_to_itrf_pv(r_gcrf, v_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )

        line = f"K={K:>4} | Orekit POS {t_ok_pos_k:.4f}s | PV {t_ok_pv_k:.4f}s"

        if HAS_ASTROPY:
            t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
            ast_times_k = t0 + (np.arange(K) * dt_seconds) * u.s
            t_ast_pos_k = bench_min(
                lambda: _astropy_pos_grouped(r_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            t_ast_pv_k = bench_min(
                lambda: _astropy_pv_grouped(r_gcrf, v_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            line += f" | Astropy POS {t_ast_pos_k:.4f}s | PV {t_ast_pv_k:.4f}s | POS ratio {t_ast_pos_k / t_ok_pos_k:.2f}x | PV ratio {t_ast_pv_k / t_ok_pv_k:.2f}x"

        print(line)

    # ---------------------------------------------------------------------
    # C) Worst-case unique times (small N)
    # ---------------------------------------------------------------------
    print("\n=== Worst-case unique times (small N) ===")
    Nw = 1_000
    r_w, v_w = make_states(Nw)

    unique_dates = [date0.shiftedBy(float(i)) for i in range(Nw)]
    t_ok_pos_unique = bench_min(
        lambda: gcrf_to_itrf_positions(r_w, unique_dates), repeat=3, number=1
    )
    t_ok_pv_unique = bench_min(
        lambda: gcrf_to_itrf_pv(r_w, v_w, unique_dates), repeat=3, number=1
    )
    print(
        f"N={Nw} unique | Orekit POS {t_ok_pos_unique:.4f}s | PV {t_ok_pv_unique:.4f}s"
    )

    if HAS_ASTROPY:
        # Astropy unique-time: group size is 1 per time, so this is very expensive;
        # keep Nw small and do 1 repeat.
        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        times_unique = t0 + (np.arange(Nw) * 1.0) * u.s

        # Implement as a loop of single-point transforms (fair but slow)
        def astropy_unique_pos():
            out = np.empty_like(r_w)
            for i in range(Nw):
                rep = CartesianRepresentation(
                    r_w[i, 0] * u.m, r_w[i, 1] * u.m, r_w[i, 2] * u.m
                )
                g = GCRS(rep, obstime=times_unique[i])
                itrs = g.transform_to(ITRS(obstime=times_unique[i]))
                c = itrs.cartesian
                out[i, 0] = c.x.to_value(u.m)
                out[i, 1] = c.y.to_value(u.m)
                out[i, 2] = c.z.to_value(u.m)
            return out

        t_ast_pos_unique = bench_min(astropy_unique_pos, repeat=1, number=1)
        print(
            f"N={Nw} unique | Astropy POS {t_ast_pos_unique:.4f}s | POS ratio {t_ast_pos_unique / t_ok_pos_unique:.2f}x"
        )

    # ---------------------------------------------------------------------
    # D) Input overhead for Orekit: AbsoluteDate list vs astropy Time array
    # ---------------------------------------------------------------------
    if HAS_ASTROPY:
        print(
            "\n=== Input overhead (Orekit): AbsoluteDate list vs astropy Time array ==="
        )
        N = 200_000
        K = 16
        r_gcrf, _ = make_states(N)

        orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K)]
        date_idx = np.arange(N) % K
        orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        ast_times_k = t0 + (np.arange(K) * dt_seconds) * u.s
        ast_times_per_row = ast_times_k[date_idx]  # array-valued Time

        t_abs = bench_min(
            lambda: gcrf_to_itrf_positions(r_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )
        t_ast = bench_min(
            lambda: gcrf_to_itrf_positions(r_gcrf, ast_times_per_row), REPEATS, NUMBER
        )

        print(f"Orekit POS K-times with AbsoluteDate list: {t_abs:.4f}s")
        print(f"Orekit POS K-times with astropy Time array: {t_ast:.4f}s")
        print(f"Overhead factor: {t_ast / t_abs:.2f}x")


# =============================
# TEST CODE: add this test function (and call it from __main__)
# =============================
def _test_geodetic():
    import timeit

    import numpy as np
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid  # type: ignore
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
    from org.orekit.utils import Constants, IERSConventions  # type: ignore

    # Helpers
    def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    def ang_err_rad(a: np.ndarray, b: np.ndarray) -> float:
        # minimal angular difference
        d = a - b
        d = np.arctan2(np.sin(d), np.cos(d))
        return float(np.max(np.abs(d)))

    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    rng = np.random.default_rng(123)
    N = 50_000

    # Random geodetic (avoid exactly +/-90 deg to keep conditioning clean)
    lat = rng.uniform(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, size=N)
    lon = rng.uniform(-np.pi, np.pi, size=N)
    alt = rng.uniform(-1000.0, 1_000_000.0, size=N)  # -1 km to 1000 km
    lla = np.stack([lat, lon, alt], axis=1)

    # 1) Roundtrip geodetic <-> ITRF (NumPy)
    r_itrf = geodetic_to_itrf_positions(lla)
    lla_back = itrf_to_geodetic_positions(r_itrf)

    e_lat = ang_err_rad(lla[:, 0], lla_back[:, 0])
    e_lon = ang_err_rad(lla[:, 1], lla_back[:, 1])
    e_alt = max_abs_err(lla[:, 2], lla_back[:, 2])

    print(
        f"[Geo Test 1] lla -> itrf -> lla max err lat: {e_lat:.3e} rad, lon: {e_lon:.3e} rad, alt: {e_alt:.3e} m"
    )
    assert e_lat < 1e-8
    assert e_lon < 1e-8
    assert e_alt < 1e-2  # ~0.1m typical

    # 2) Cross-check vs Orekit OneAxisEllipsoid for a small sample (positions)
    M = 200
    idx = rng.integers(0, N, size=M)

    max_pos = 0.0
    max_lat = 0.0
    max_lon = 0.0
    max_alt = 0.0

    for i in idx:
        gp = GeodeticPoint(float(lla[i, 0]), float(lla[i, 1]), float(lla[i, 2]))
        v = earth.transform(gp)  # Vector3D in body frame (ITRF)
        ref = np.array([v.getX(), v.getY(), v.getZ()], dtype=np.float64)
        max_pos = max(max_pos, float(np.max(np.abs(ref - r_itrf[i]))))

        # inverse: Orekit cartesian -> geodetic (needs frame+date)
        gp2 = earth.transform(Vector3D(ref[0], ref[1], ref[2]), itrf, date0)
        ref_lla = np.array(
            [gp2.getLatitude(), gp2.getLongitude(), gp2.getAltitude()], dtype=np.float64
        )

        max_lat = max(
            max_lat,
            abs(
                float(
                    np.arctan2(
                        np.sin(ref_lla[0] - lla_back[i, 0]),
                        np.cos(ref_lla[0] - lla_back[i, 0]),
                    )
                )
            ),
        )
        max_lon = max(
            max_lon,
            abs(
                float(
                    np.arctan2(
                        np.sin(ref_lla[1] - lla_back[i, 1]),
                        np.cos(ref_lla[1] - lla_back[i, 1]),
                    )
                )
            ),
        )
        max_alt = max(max_alt, abs(float(ref_lla[2] - lla_back[i, 2])))

    print(f"[Geo Test 2] vs Orekit (sample) max |pos| component err: {max_pos:.3e} m")
    print(
        f"[Geo Test 2] vs Orekit (sample) max err lat: {max_lat:.3e} rad, lon: {max_lon:.3e} rad, alt: {max_alt:.3e} m"
    )
    assert max_pos < 1e-4  # conservative
    assert max_lat < 1e-8
    assert max_lon < 1e-8
    assert max_alt < 1e-2

    # 3) Roundtrip through GCRF with time
    r_gcrf = itrf_to_gcrf_positions(r_itrf, date0)
    lla_from_gcrf = gcrf_to_geodetic_positions(r_gcrf, date0)

    e_lat2 = ang_err_rad(lla[:, 0], lla_from_gcrf[:, 0])
    e_lon2 = ang_err_rad(lla[:, 1], lla_from_gcrf[:, 1])
    e_alt2 = max_abs_err(lla[:, 2], lla_from_gcrf[:, 2])
    print(
        f"[Geo Test 3] lla -> itrf -> gcrf -> itrf -> lla max err lat: {e_lat2:.3e} rad, lon: {e_lon2:.3e} rad, alt: {e_alt2:.3e} m"
    )
    assert e_lat2 < 1e-9
    assert e_lon2 < 1e-9
    assert (
        e_alt2 < 1e-2
    )  # frame pipeline + EOP may introduce tiny mm–cm-level differences

    # 4) Degrees interface sanity
    lla_deg = np.stack([np.rad2deg(lat), np.rad2deg(lon), alt], axis=1)
    r_itrf_deg = geodetic_to_itrf_positions(lla_deg, degrees=True)
    assert max_abs_err(r_itrf_deg, r_itrf) < 1e-8

    lla_deg_back = itrf_to_geodetic_positions(r_itrf, degrees=True)
    # compare degrees for lat/lon, meters for alt
    e_latd = max_abs_err(lla_deg[:, 0], lla_deg_back[:, 0])
    e_lond = max_abs_err(
        ((lla_deg[:, 1] - lla_deg_back[:, 1] + 180.0) % 360.0) - 180.0, 0.0
    )
    e_altd = max_abs_err(lla_deg[:, 2], lla_deg_back[:, 2])
    print(
        f"[Geo Test 4] degrees interface max err lat: {e_latd:.3e} deg, lon (wrapped): {e_lond:.3e} deg, alt: {e_altd:.3e} m"
    )

    # 5) Speed sanity (vectorized numpy vs Orekit point-loop for geodetic->ITRF)
    # Vectorized (full N)
    t_vec = min(
        timeit.repeat(lambda: geodetic_to_itrf_positions(lla), repeat=5, number=1)
    )
    print(f"[Geo Speed] vectorized geodetic->ITRF: {t_vec:.4f}s for N={N}")

    # Orekit point-loop (small M to avoid huge runtimes)
    M2 = 10_000
    lla2 = lla[:M2]

    def orekit_loop():
        out = np.empty((M2, 3), dtype=np.float64)
        for i in range(M2):
            gp = GeodeticPoint(float(lla2[i, 0]), float(lla2[i, 1]), float(lla2[i, 2]))
            v = earth.transform(gp)
            out[i, 0] = v.getX()
            out[i, 1] = v.getY()
            out[i, 2] = v.getZ()
        return out

    t_loop = min(timeit.repeat(orekit_loop, repeat=3, number=1))
    print(f"[Geo Speed] Orekit loop geodetic->ITRF: {t_loop:.4f}s for N={M2}")

    print("Geodetic tests passed.")


# =============================
# TEST CODE: add this test function and call it from __main__
# =============================
def _test_j2000_gcrf():
    import numpy as np
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
    from org.orekit.utils import PVCoordinates  # type: ignore

    def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
    date1 = date0.shiftedBy(86400.0)

    j2000 = FramesFactory.getEME2000()
    gcrf = FramesFactory.getGCRF()

    N = 10000
    rng = np.random.default_rng(0)

    r = rng.normal(size=(N, 3)).astype(np.float64)
    r *= 7_000_000.0 / np.linalg.norm(r, axis=1, keepdims=True)
    v = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

    # Roundtrip POS
    r_g = j2000_to_gcrf_positions(r)
    r_back = gcrf_to_j2000_positions(r_g)
    e = max_abs_err(r, r_back)
    print(f"[J2000/GCRF Test 1] POS roundtrip max abs err: {e:.3e} m")
    assert e < 1e-8

    # Roundtrip PV
    r_g2, v_g2 = j2000_to_gcrf_pv(r, v)
    r_back2, v_back2 = gcrf_to_j2000_pv(r_g2, v_g2)
    er = max_abs_err(r, r_back2)
    ev = max_abs_err(v, v_back2)
    print(
        f"[J2000/GCRF Test 2] PV roundtrip max abs err r: {er:.3e} m, v: {ev:.3e} m/s"
    )
    assert er < 1e-8
    assert ev < 1e-10

    # Validate vs Orekit Transform at two different dates (should match; frames are inertial)
    idx = rng.integers(0, N, size=20)

    for d in [date0, date1]:
        t = j2000.getTransformTo(gcrf, d)

        # position
        for i in idx:
            vv = Vector3D(r[i, 0], r[i, 1], r[i, 2])
            ww = t.transformPosition(vv)
            ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
            assert np.max(np.abs(ref - r_g[i])) < 1e-8

        # PV
        for i in idx:
            pv = PVCoordinates(
                Vector3D(r[i, 0], r[i, 1], r[i, 2]),
                Vector3D(v[i, 0], v[i, 1], v[i, 2]),
            )
            pv2 = t.transformPVCoordinates(pv)
            ref_r = np.array(
                [
                    pv2.getPosition().getX(),
                    pv2.getPosition().getY(),
                    pv2.getPosition().getZ(),
                ],
                dtype=np.float64,
            )
            ref_v = np.array(
                [
                    pv2.getVelocity().getX(),
                    pv2.getVelocity().getY(),
                    pv2.getVelocity().getZ(),
                ],
                dtype=np.float64,
            )
            assert np.max(np.abs(ref_r - r_g2[i])) < 1e-8
            assert np.max(np.abs(ref_v - v_g2[i])) < 1e-8

    print(
        "[J2000/GCRF Test 3] Matches Orekit transformPosition/transformPVCoordinates at multiple dates"
    )
    print("J2000/GCRF tests passed.")


# =============================
# TESTS: TEME transforms
# =============================
def _test_teme():
    import numpy as np
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
    from org.orekit.utils import IERSConventions, PVCoordinates  # type: ignore

    def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
    date1 = date0.shiftedBy(123.456)
    date2 = date0.shiftedBy(9876.0)

    teme = FramesFactory.getTEME()
    gcrf = FramesFactory.getGCRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    N = 5000
    rng = np.random.default_rng(42)

    r_teme = rng.normal(size=(N, 3)).astype(np.float64)
    r_teme *= 7_000_000.0 / np.linalg.norm(r_teme, axis=1, keepdims=True)
    v_teme = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

    # -------------------------
    # 1) Roundtrip: TEME <-> GCRF (single date)
    # -------------------------
    r_g = teme_to_gcrf_positions(r_teme, date0)
    r_back = gcrf_to_teme_positions(r_g, date0)
    e = max_abs_err(r_teme, r_back)
    print(f"[TEME Test 1] POS TEME->GCRF->TEME (single date) max abs err: {e:.3e} m")
    assert e < 1e-6

    r_g2, v_g2 = teme_to_gcrf_pv(r_teme, v_teme, date0)
    r_back2, v_back2 = gcrf_to_teme_pv(r_g2, v_g2, date0)
    er = max_abs_err(r_teme, r_back2)
    ev = max_abs_err(v_teme, v_back2)
    print(
        f"[TEME Test 2] PV TEME->GCRF->TEME (single date) max abs err r: {er:.3e} m, v: {ev:.3e} m/s"
    )
    assert er < 1e-6
    assert ev < 1e-6

    # -------------------------
    # 2) Roundtrip: TEME <-> ITRF (single date)
    # -------------------------
    r_i = teme_to_itrf_positions(r_teme, date0)
    r_back3 = itrf_to_teme_positions(r_i, date0)
    e3 = max_abs_err(r_teme, r_back3)
    print(f"[TEME Test 3] POS TEME->ITRF->TEME (single date) max abs err: {e3:.3e} m")
    assert e3 < 1e-6

    r_i2, v_i2 = teme_to_itrf_pv(r_teme, v_teme, date0)
    r_back4, v_back4 = itrf_to_teme_pv(r_i2, v_i2, date0)
    er4 = max_abs_err(r_teme, r_back4)
    ev4 = max_abs_err(v_teme, v_back4)
    print(
        f"[TEME Test 4] PV TEME->ITRF->TEME (single date) max abs err r: {er4:.3e} m, v: {ev4:.3e} m/s"
    )
    assert er4 < 1e-6
    assert ev4 < 1e-6

    # -------------------------
    # 3) Grouped multi-date path (repeated dates)
    # -------------------------
    dates = [date0, date1, date2] * (N // 3) + [date0] * (N % 3)
    dates = dates[:N]

    r_gk = teme_to_gcrf_positions(r_teme, dates)
    r_backk = gcrf_to_teme_positions(r_gk, dates)
    ek = max_abs_err(r_teme, r_backk)
    print(f"[TEME Test 5] POS TEME<->GCRF (multi-date grouped) max abs err: {ek:.3e} m")
    assert ek < 1e-6

    r_ik, v_ik = teme_to_itrf_pv(r_teme, v_teme, dates)
    r_backk2, v_backk2 = itrf_to_teme_pv(r_ik, v_ik, dates)
    erk = max_abs_err(r_teme, r_backk2)
    evk = max_abs_err(v_teme, v_backk2)
    print(
        f"[TEME Test 6] PV TEME<->ITRF (multi-date grouped) max abs err r: {erk:.3e} m, v: {evk:.3e} m/s"
    )
    assert erk < 1e-6
    assert evk < 1e-6

    # -------------------------
    # 4) Spot-check vs Orekit direct transforms (sample)
    # -------------------------
    idx = rng.integers(0, N, size=20)

    # TEME -> GCRF spot-checks at date0
    t_tg = teme.getTransformTo(gcrf, date0)
    for i in idx:
        vv = Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2])
        ww = t_tg.transformPosition(vv)
        ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
        assert np.max(np.abs(ref - r_g[i])) < 1e-8

        pv = PVCoordinates(
            Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2]),
            Vector3D(v_teme[i, 0], v_teme[i, 1], v_teme[i, 2]),
        )
        pv2 = t_tg.transformPVCoordinates(pv)
        ref_r = np.array(
            [
                pv2.getPosition().getX(),
                pv2.getPosition().getY(),
                pv2.getPosition().getZ(),
            ],
            dtype=np.float64,
        )
        ref_v = np.array(
            [
                pv2.getVelocity().getX(),
                pv2.getVelocity().getY(),
                pv2.getVelocity().getZ(),
            ],
            dtype=np.float64,
        )
        assert np.max(np.abs(ref_r - r_g2[i])) < 1e-8
        assert np.max(np.abs(ref_v - v_g2[i])) < 1e-8

    # TEME -> ITRF spot-checks at date0
    t_ti = teme.getTransformTo(itrf, date0)
    for i in idx:
        vv = Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2])
        ww = t_ti.transformPosition(vv)
        ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
        assert np.max(np.abs(ref - r_i[i])) < 1e-8

        pv = PVCoordinates(
            Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2]),
            Vector3D(v_teme[i, 0], v_teme[i, 1], v_teme[i, 2]),
        )
        pv2 = t_ti.transformPVCoordinates(pv)
        ref_r = np.array(
            [
                pv2.getPosition().getX(),
                pv2.getPosition().getY(),
                pv2.getPosition().getZ(),
            ],
            dtype=np.float64,
        )
        ref_v = np.array(
            [
                pv2.getVelocity().getX(),
                pv2.getVelocity().getY(),
                pv2.getVelocity().getZ(),
            ],
            dtype=np.float64,
        )
        assert np.max(np.abs(ref_r - r_i2[i])) < 1e-8
        assert np.max(np.abs(ref_v - v_i2[i])) < 1e-8

    print(
        "[TEME Test 7] Matches Orekit transformPosition/transformPVCoordinates (sample) OK"
    )
    print("TEME tests passed.")


def _test_teme_vs_astropy():
    """
    Cross-check Orekit TEME transforms against Astropy's TEME transforms.

    This is an *inter-library* comparison and is sensitive to EOP/IERS configuration.
    Your TEME implementation is already validated against Orekit itself (TEME Test 7);
    this test is mainly to catch gross mistakes, not to guarantee mm-level agreement.

    Strategy:
      - Force Astropy to use IERS-B (offline, stable) and disable downloads.
      - Use Orekit ITRF with simple_eop=False for closer parity with Astropy.
      - Assert only loose bounds (meters / cm/s). Print guidance if larger.
    """
    import numpy as np

    try:
        import astropy.units as u  # type: ignore
        from astropy.coordinates import (  # type: ignore
            GCRS,
            ITRS,
            TEME,
            CartesianDifferential,
            CartesianRepresentation,
        )
        from astropy.time import Time as AstropyTime  # type: ignore
        from astropy.utils import iers  # type: ignore

        HAS_ASTROPY = True
    except Exception:
        HAS_ASTROPY = False

    if not HAS_ASTROPY:
        print("Astropy not available; skipping TEME vs Astropy cross-check.")
        return

    # -------------------------
    # Pin Astropy IERS behavior (offline, stable)
    # -------------------------
    try:
        iers.conf.auto_download = False
        iers.conf.iers_degraded_accuracy = "warn"

        iers_b = None
        try:
            iers_b = iers.IERS_B.open(iers.IERS_B_FILE)
        except Exception:
            iers_b = None

        # Use the global Earth orientation table if available (Astropy versions vary)
        # If this fails, Astropy will still run but may use a different EOP source/behavior.
        if iers_b is not None:
            try:
                # Astropy >= 5-ish
                iers.earth_orientation_table.set(iers_b)
            except Exception:
                pass
    except Exception:
        pass

    rng = np.random.default_rng(123)

    N = 50_000
    K = 16
    dt = 60.0

    t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
    times_k = t0 + (np.arange(K) * dt) * u.s
    date_idx = np.arange(N) % K
    times_per_row = times_k[date_idx]

    # Random TEME states
    r_teme = rng.normal(size=(N, 3)).astype(np.float64)
    r_teme *= 7_000_000.0 / np.linalg.norm(r_teme, axis=1, keepdims=True)
    v_teme = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

    # -------------------------
    # Orekit results
    # Use simple_eop=False here (closer parity to typical Astropy behavior)
    # -------------------------
    r_itrf_ok = teme_to_itrf_positions(r_teme, times_per_row, simple_eop=False)
    r_gcrf_ok = teme_to_gcrf_positions(r_teme, times_per_row)

    r_itrf_ok_pv, v_itrf_ok_pv = teme_to_itrf_pv(
        r_teme, v_teme, times_per_row, simple_eop=False
    )
    r_gcrf_ok_pv, v_gcrf_ok_pv = teme_to_gcrf_pv(r_teme, v_teme, times_per_row)

    # -------------------------
    # Astropy results (grouped by K)
    # -------------------------
    r_itrf_ast = np.empty_like(r_teme)
    r_gcrs_ast = np.empty_like(r_teme)
    v_itrf_ast = np.empty_like(v_teme)
    v_gcrs_ast = np.empty_like(v_teme)

    groups = [np.where(date_idx == i)[0] for i in range(K)]
    for i in range(K):
        idxs = groups[i]
        ti = times_k[i]

        rep = CartesianRepresentation(
            r_teme[idxs, 0] * u.m,
            r_teme[idxs, 1] * u.m,
            r_teme[idxs, 2] * u.m,
            differentials=CartesianDifferential(
                v_teme[idxs, 0] * (u.m / u.s),
                v_teme[idxs, 1] * (u.m / u.s),
                v_teme[idxs, 2] * (u.m / u.s),
            ),
        )

        teme = TEME(rep, obstime=ti)

        itrs = teme.transform_to(ITRS(obstime=ti))
        c = itrs.cartesian
        r_itrf_ast[idxs, 0] = c.x.to_value(u.m)
        r_itrf_ast[idxs, 1] = c.y.to_value(u.m)
        r_itrf_ast[idxs, 2] = c.z.to_value(u.m)
        d = c.differentials["s"]
        v_itrf_ast[idxs, 0] = d.d_x.to_value(u.m / u.s)
        v_itrf_ast[idxs, 1] = d.d_y.to_value(u.m / u.s)
        v_itrf_ast[idxs, 2] = d.d_z.to_value(u.m / u.s)

        gcrs = teme.transform_to(GCRS(obstime=ti))
        c = gcrs.cartesian
        r_gcrs_ast[idxs, 0] = c.x.to_value(u.m)
        r_gcrs_ast[idxs, 1] = c.y.to_value(u.m)
        r_gcrs_ast[idxs, 2] = c.z.to_value(u.m)
        d = c.differentials["s"]
        v_gcrs_ast[idxs, 0] = d.d_x.to_value(u.m / u.s)
        v_gcrs_ast[idxs, 1] = d.d_y.to_value(u.m / u.s)
        v_gcrs_ast[idxs, 2] = d.d_z.to_value(u.m / u.s)

    def max_abs(a: np.ndarray) -> float:
        return float(np.max(np.abs(a)))

    pos_itrf_err = max_abs(r_itrf_ok - r_itrf_ast)
    pos_gcrf_err = max_abs(r_gcrf_ok - r_gcrs_ast)

    pv_itrf_r_err = max_abs(r_itrf_ok_pv - r_itrf_ast)
    pv_itrf_v_err = max_abs(v_itrf_ok_pv - v_itrf_ast)

    pv_gcrf_r_err = max_abs(r_gcrf_ok_pv - r_gcrs_ast)
    pv_gcrf_v_err = max_abs(v_gcrf_ok_pv - v_gcrs_ast)

    print(f"[TEME vs Astropy] POS TEME->ITRF(ITRS) max abs err: {pos_itrf_err:.6e} m")
    print(f"[TEME vs Astropy] POS TEME->GCRF(GCRS) max abs err: {pos_gcrf_err:.6e} m")
    print(
        f"[TEME vs Astropy] PV  TEME->ITRF(ITRS) max abs err: r {pv_itrf_r_err:.6e} m, v {pv_itrf_v_err:.6e} m/s"
    )
    print(
        f"[TEME vs Astropy] PV  TEME->GCRF(GCRS) max abs err: r {pv_gcrf_r_err:.6e} m, v {pv_gcrf_v_err:.6e} m/s"
    )

    # Loose bounds for inter-library checks.
    # If these fail, it *still* may be EOP/IERS differences; inspect the printed errors.
    POS_TOL_M = 20.0
    VEL_TOL_MPS = 0.10

    assert pos_itrf_err < POS_TOL_M, (
        f"TEME->ITRF mismatch too large ({pos_itrf_err:.3f} m). "
        "Likely EOP/IERS mismatch; verify both libraries use comparable EOP sources."
    )
    assert pos_gcrf_err < POS_TOL_M, (
        f"TEME->GCRF mismatch too large ({pos_gcrf_err:.3f} m). "
        "Likely frame/EOP differences (GCRF vs GCRS)."
    )
    assert pv_itrf_v_err < VEL_TOL_MPS, (
        f"TEME->ITRF velocity mismatch too large ({pv_itrf_v_err:.3f} m/s). "
        "Likely EOP/IERS mismatch."
    )
    assert pv_gcrf_v_err < VEL_TOL_MPS, (
        f"TEME->GCRF velocity mismatch too large ({pv_gcrf_v_err:.3f} m/s). "
        "Likely frame/EOP differences (GCRF vs GCRS)."
    )

    print("TEME vs Astropy cross-check passed (loose inter-library tolerances).")


if __name__ == "__main__":
    _test1()
    _test_geodetic()
    _test_j2000_gcrf()
    _test_teme()
    _test_teme_vs_astropy()
    _speed_test()
