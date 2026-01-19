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
from nebula import ensure_setup

ensure_setup()
# from nebula import NEBULA_ROOT_DIR

# os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
# orekit_jpype.initVM()

# from orekit_jpype.pyhelpers import setup_orekit_curdir

# setup_orekit_curdir(filename=os.path.join(NEBULA_ROOT_DIR, "..", "data", "orekit-data"))

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


def j2000_to_gcrf_pos(r_j2000_m: np.ndarray) -> np.ndarray:
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
    return _apply_transform_pos(parts, r)


def gcrf_to_j2000_pos(r_gcrf_m: np.ndarray) -> np.ndarray:
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
    return _apply_transform_pos(parts, r)


def j2000_to_gcrf_pos_vel(
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
    return _apply_transform_pos_vel(parts, r, v)


def gcrf_to_j2000_pos_vel(
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
    return _apply_transform_pos_vel(parts, r, v)


# =============================================================================
# Public API: TEME <-> GCRF / ITRF
# =============================================================================
def teme_to_gcrf_pos(r_teme_m: np.ndarray, dates: DatesLike) -> np.ndarray:  # type: ignore
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
    return _transform_pos(r_teme_m, dates, teme, gcrf)


def gcrf_to_teme_pos(r_gcrf_m: np.ndarray, dates: DatesLike) -> np.ndarray:  # type: ignore
    """
    Convert positions from GCRF to TEME.

    Parameters
    ----------
    r_gcrf_m : ndarray, shape (N, 3)
        Positions in GCRF meters.
    dates : AbsoluteDate | astropy.time.Time | iterable
        Epoch specification (see `teme_to_gcrf_pos`).

    Returns
    -------
    ndarray, shape (N, 3)
        Positions in TEME meters.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_pos(r_gcrf_m, dates, gcrf, teme)


def teme_to_gcrf_pos_vel(
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
        Epoch specification (see `teme_to_gcrf_pos`).

    Returns
    -------
    (r_gcrf_m, v_gcrf_mps) : tuple of ndarrays
        Transformed positions and velocities in GCRF.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_pos_vel(r_teme_m, v_teme_mps, dates, teme, gcrf)


def gcrf_to_teme_pos_vel(
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
        Epoch specification (see `teme_to_gcrf_pos`).

    Returns
    -------
    (r_teme_m, v_teme_mps) : tuple of ndarrays
        Transformed positions and velocities in TEME.
    """
    teme = _get_teme()
    gcrf = _get_gcrf()
    return _transform_pos_vel(r_gcrf_m, v_gcrf_mps, dates, gcrf, teme)


def teme_to_itrf_pos(
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
        Epoch specification (see `teme_to_gcrf_pos`).
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
    return _transform_pos(r_teme_m, dates, teme, itrf)


def itrf_to_teme_pos(
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
        Epoch specification (see `teme_to_gcrf_pos`).
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
    return _transform_pos(r_itrf_m, dates, itrf, teme)


def teme_to_itrf_pos_vel(
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
        Epoch specification (see `teme_to_gcrf_pos`).
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
    return _transform_pos_vel(r_teme_m, v_teme_mps, dates, teme, itrf)


def itrf_to_teme_pos_vel(
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
        Epoch specification (see `teme_to_gcrf_pos`).
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
    return _transform_pos_vel(r_itrf_m, v_itrf_mps, dates, itrf, teme)


# =============================================================================
# Public API: Geodetic (WGS84) <-> ITRF (ECEF) + pipelines via GCRF
# =============================================================================
def geodetic_to_itrf_pos(
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


def itrf_to_geodetic_pos(
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


def geodetic_to_gcrf_pos(
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
        Geodetic coordinates (see `geodetic_to_itrf_pos`).
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
    r_itrf = geodetic_to_itrf_pos(lla, degrees=degrees)
    return itrf_to_gcrf_pos(r_itrf, dates, iers=iers, simple_eop=simple_eop)


def gcrf_to_geodetic_pos(
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
    r_itrf = gcrf_to_itrf_pos(r_gcrf_m, dates, iers=iers, simple_eop=simple_eop)
    return itrf_to_geodetic_pos(r_itrf, degrees=degrees, wrap_lon=wrap_lon)


# =============================================================================
# Public API: GCRF <-> ITRF (positions + PV)
# =============================================================================
def gcrf_to_itrf_pos(
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
    return _transform_pos(r_gcrf_m, dates, gcrf, itrf)


def itrf_to_gcrf_pos(
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
        Epoch specification (see `gcrf_to_itrf_pos`).
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
    return _transform_pos(r_itrf_m, dates, itrf, gcrf)


def gcrf_to_itrf_pos_vel(
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
        Epoch specification (see `gcrf_to_itrf_pos`).
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
    return _transform_pos_vel(r_gcrf_m, v_gcrf_mps, dates, gcrf, itrf)


def itrf_to_gcrf_pos_vel(
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
        Epoch specification (see `gcrf_to_itrf_pos`).
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
    return _transform_pos_vel(r_itrf_m, v_itrf_mps, dates, itrf, gcrf)


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


def _apply_transform_pos(parts: _KinematicParts, r_old: np.ndarray) -> np.ndarray:
    """
    Apply a cached kinematic transform to positions.

    r_new = R * r_old + t
    """
    return r_old @ parts.R.T + parts.t


def _apply_transform_pos_vel(
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


def _transform_pos_grouped_by_dates(
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
            out[i] = _apply_transform_pos(parts, r_old[i : i + 1])[0]
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
        out[idx] = _apply_transform_pos(parts, r_old[idx])
    return out


def _transform_pos_vel_grouped_by_dates(
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
            ri, vi = _apply_transform_pos_vel(parts, r_old[i : i + 1], v_old[i : i + 1])
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
        ri, vi = _apply_transform_pos_vel(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def _transform_pos_grouped_by_astropy_time(
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
        return _transform_pos_grouped_by_dates(r_old, dates_list, from_frame, to_frame)

    uniq_dates: list[AbsoluteDate] = []
    for k in range(K):
        ti = AstropyTime(uniq_key[k, 0], uniq_key[k, 1], format="jd", scale="utc")  # type: ignore[misc]
        uniq_dates.append(_to_absolutedate(ti))

    order, starts, ends = _group_from_inverse(inverse)
    out = np.empty_like(r_old)

    for g in range(K):
        parts = _get_kinematic_parts(from_frame, to_frame, uniq_dates[g])
        idx = order[starts[g] : ends[g]]
        out[idx] = _apply_transform_pos(parts, r_old[idx])
    return out


def _transform_pos_vel_grouped_by_astropy_time(
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
        return _transform_pos_vel_grouped_by_dates(
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
        ri, vi = _apply_transform_pos_vel(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def _transform_pos(
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
        return _apply_transform_pos(parts, r_old)

    if _is_astropy_array_time(dates):
        if getattr(dates, "shape", None) != (N,):
            raise ValueError(
                f"Astropy Time shape {dates.shape} must be (N,) where N={N}"
            )
        return _transform_pos_grouped_by_astropy_time(r_old, dates, from_frame, to_frame)  # type: ignore[arg-type]

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

    return _transform_pos_grouped_by_dates(r_old, dates_list, from_frame, to_frame)


def _transform_pos_vel(
    r_old_m: np.ndarray,
    v_old_mps: np.ndarray,
    dates: DatesLike,  # type: ignore
    from_frame,
    to_frame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal vectorized dispatcher for PV transforms (N x 3 arrays).

    Same time handling strategy as `_transform_pos`.
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
        return _apply_transform_pos_vel(parts, r_old, v_old)

    if _is_astropy_array_time(dates):
        if getattr(dates, "shape", None) != (N,):
            raise ValueError(
                f"Astropy Time shape {dates.shape} must be (N,) where N={N}"
            )
        return _transform_pos_vel_grouped_by_astropy_time(r_old, v_old, dates, from_frame, to_frame)  # type: ignore[arg-type]

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

    return _transform_pos_vel_grouped_by_dates(
        r_old, v_old, dates_list, from_frame, to_frame
    )
