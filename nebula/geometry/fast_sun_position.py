# sun_position.py
"""
High-accuracy analytic solar position in an apparent-of-date ECI frame
and ECEF using a Vallado-style algorithm.

This module implements a closed-form solar ephemeris (after Vallado,
"Fundamentals of Astrodynamics and Applications") and returns the Sun's
geocentric position in:

- An Earth-centered inertial frame aligned with the **true equator and true
  equinox of date** (apparent-of-date, similar to the TETE frame in Astropy).
- An Earth-fixed frame obtained using the same approximate rotation chain as
  ``nebula.propagation._fast_orbit_backend`` (IAU-76 precession, truncated IAU-1980
  nutation, GAST, optional epoch polar motion), giving an ITRF-like ECEF
  realization.

No external ephemeris files are required.

Time scales
-----------
- The solar position model itself is evaluated in TT (Terrestrial Time).
- The Earth rotation terms are evaluated in UT1.

You must supply:
- jd_tt  : Julian Date in TT.
- jd_ut1 : Julian Date in UT1 (often approximated by UTC if high-precision
           UT1 is not available).

Accuracy
--------
- Directional (angular) accuracy of the solar position is ~0.01° or better
  over several centuries relative to high-precision ephemerides when
  interpreted in the same frame (apparent-of-date / TETE).
- ECEF position accuracy will also depend on how accurately jd_tt and jd_ut1
  are provided (Δ(UT1–UTC), Δ(TT–UTC), etc.); if UT1 is approximated by UTC,
  errors are typically negligible for coverage / lighting / RF work.

All functions are numba-njit compatible.
"""

import numpy as np
from numba import njit
from nebula.transform._coarse_eci2itrf import (
    _coarse_eci_to_itrf_pos_iau76_shortnut,
    _coarse_gmst_vallado_rad,
    _coarse_tod_to_native_pos_iau76_shortnut,
)

# ---- constants ----

J2000_JD = 2451545.0  # Julian Date of J2000.0 epoch

# IAU 2012 astronomical unit (exact)
AU_METERS = 149597870700.0

DEG2RAD = np.pi / 180.0


@njit(fastmath=True)
def _wrap_deg(angle_deg: float) -> float:
    """
    Wrap an angle in degrees to [0, 360).
    """
    w = angle_deg % 360.0
    if w < 0.0:
        w += 360.0
    return w


@njit(fastmath=True)
def julian_date(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """
    Compute the Julian Date (JD) from a Gregorian calendar date and time.

    This uses the standard astronomical algorithm and is valid for
    Gregorian dates (year >= 1582).

    Parameters
    ----------
    year, month, day : int
        Calendar date in UTC-like scale.
    hour, minute : int
        Time of day.
    second : float
        Seconds (may include fractional part).

    Returns
    -------
    jd : float
        Julian Date corresponding to the given calendar date and time.
    """
    y = year
    m = month
    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)

    day_frac = (hour + (minute + second / 60.0) / 60.0) / 24.0

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + B - 1524.5 + day_frac
    return jd


@njit(fastmath=True)
def sun_position_eci(jd_tt: float):
    """
    Geocentric position of the Sun in an apparent-of-date ECI frame, from
    Julian Date TT.

    The returned coordinates are expressed in an Earth-centered inertial frame
    aligned with the **true equator and true equinox of date** (i.e. an
    apparent-of-date frame, similar to Astropy's TETE frame):

      - The ecliptic longitude used is the apparent longitude (including
        small nutation and aberration corrections).
      - The rotation from ecliptic to equatorial uses the true obliquity of
        the ecliptic.

    This follows Vallado-style low-precision solar ephemeris formulas and
    provides high accuracy for typical engineering applications (coverage,
    lighting, RF, etc.) when used in a consistent frame.

    Parameters
    ----------
    jd_tt : float
        Julian Date in Terrestrial Time (TT).

    Returns
    -------
    x_eci, y_eci, z_eci : float
        Sun's geocentric position in the apparent-of-date ECI frame [meters].
    """
    T = (jd_tt - J2000_JD) / 36525.0  # Julian centuries of TT from J2000.0

    # Mean longitude of the Sun (deg)
    L = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    L = _wrap_deg(L)

    # Mean anomaly of the Sun (deg)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T - 0.00000048 * T * T * T
    M = _wrap_deg(M)

    # Eccentricity of Earth's orbit
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T

    # Convert to radians
    M_rad = M * DEG2RAD

    # Sun's equation of the center (deg)
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * np.sin(M_rad)
        + (0.019993 - 0.000101 * T) * np.sin(2.0 * M_rad)
        + 0.000289 * np.sin(3.0 * M_rad)
    )
    C_rad = C * DEG2RAD

    # True longitude of the Sun (deg → rad)
    true_long_rad = (L * DEG2RAD) + C_rad

    # Apparent longitude of the Sun (deg → rad)
    Omega_deg = 125.04 - 1934.136 * T
    Omega_rad = Omega_deg * DEG2RAD
    lambda_app_rad = (
        true_long_rad - 0.00569 * DEG2RAD - 0.00478 * DEG2RAD * np.sin(Omega_rad)
    )

    # Mean obliquity of the ecliptic (deg)
    eps0_deg = 23.439291 - 0.0130042 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T
    eps0_rad = eps0_deg * DEG2RAD

    # True obliquity with nutation term (rad)
    eps_rad = eps0_rad + 0.00256 * DEG2RAD * np.cos(Omega_rad)

    # True anomaly (rad)
    v_rad = M_rad + C_rad

    # Sun–Earth distance in AU (Vallado form)
    R_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * np.cos(v_rad))

    # Geocentric equatorial coordinates of the Sun in AU
    cos_lambda = np.cos(lambda_app_rad)
    sin_lambda = np.sin(lambda_app_rad)
    cos_eps = np.cos(eps_rad)
    sin_eps = np.sin(eps_rad)

    x_au = R_au * cos_lambda
    y_au = R_au * cos_eps * sin_lambda
    z_au = R_au * sin_eps * sin_lambda

    # Convert AU to meters
    x_eci = x_au * AU_METERS
    y_eci = y_au * AU_METERS
    z_eci = z_au * AU_METERS

    return x_eci, y_eci, z_eci


@njit(fastmath=True)
def gmst_angle(jd_ut1: float) -> float:
    """
    Backward-compatible GMST helper using the fast_orbit Vallado expression.
    """
    return _coarse_gmst_vallado_rad(jd_ut1)


@njit(fastmath=True)
def sun_position_ecef(
    jd_ut1: float, jd_tt: float, xp_rad: float = 0.0, yp_rad: float = 0.0
):
    """
    Geocentric Sun position in an ITRF-like ECEF frame.

    Uses the same approximate inertial->ECEF rotation chain as fast_orbit:
    IAU-76 precession + 10-term IAU-1980 nutation + GAST, with optional
    epoch polar motion.
    """
    # Sun model returns true-equator/true-equinox-of-date coordinates (TOD-like).
    x_tod, y_tod, z_tod = sun_position_eci(jd_tt)

    # Convert TOD-like Sun vector into fast_orbit native inertial frame.
    x_eci, y_eci, z_eci = _coarse_tod_to_native_pos_iau76_shortnut(
        x_tod, y_tod, z_tod, jd_tt
    )

    # Apply the same native->ECEF chain used in fast_orbit.
    return _coarse_eci_to_itrf_pos_iau76_shortnut(
        x_eci, y_eci, z_eci, jd_ut1, jd_tt, xp_rad, yp_rad
    )


