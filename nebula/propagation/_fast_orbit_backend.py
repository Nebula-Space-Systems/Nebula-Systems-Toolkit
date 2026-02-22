"""
FastOrbit: high-speed, numpy/numba orbital propagation.

Frame semantics
---------------
- "native" is an internal inertial basis used by this implementation's Kepler/J2 math.
  It is J2000/EME2000-like, but it is not an Orekit Frame object and is not guaranteed
  to be exactly equal to Orekit GCRF/EME2000 at all times.
- "itrf" is derived from native using the built-in IAU-76 precession, 10-term nutation,
  GAST/GMST rotation, and optional epoch polar motion in this file
  (no full Orekit IERS/EOP frame chain).

For strict Orekit frame fidelity/traceability, use `nebula.propagation.orbit.Orbit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union
import math

import numpy as np
from numba import njit, prange
from nebula.transform._coarse_eci2itrf import (
    _coarse_eci_to_itrf_pv_iau76_shortnut,
    _coarse_j2_axis_native_iau76_shortnut,
)
from nebula.transform._ecef2geodetic import (
    ecef2geodetic_deg as _ecef2geodetic_deg,
    ecef2geodetic_vec_ecef_deg as _ecef2geodetic_vec_ecef_deg,
)

try:
    from astropy.time import Time as AstropyTime  # type: ignore
    from astropy.coordinates.builtin_frames.utils import (  # type: ignore
        get_polar_motion as _astropy_get_polar_motion,
    )
except Exception:  # pragma: no cover
    AstropyTime = None  # type: ignore
    _astropy_get_polar_motion = None  # type: ignore

FrameKind = Literal["native", "itrf"]
AngleType = Literal["true", "mean", "eccentric"]
J2Mode = Literal["secular", "osculating"]

# ---------------------------------------------------------------------------
# Constants (WGS84 / Earth)
# ---------------------------------------------------------------------------
WGS84_A = 6378137.0

EARTH_MU = 3.986004418e14  # [m^3/s^2]
EARTH_J2 = 1.08262668e-3
EARTH_OMEGA = 7.2921150e-5  # [rad/s]
J2000_JD = 2451545.0

# Cache padding to reduce reallocations
_CACHE_MARGIN = 64
_HERMITE_PAR_THRESHOLD = 4096


# ---------------------------------------------------------------------------
# Core math (numba)
# ---------------------------------------------------------------------------
@njit(cache=False, fastmath=True, inline="always")
def _wrap_pi(x: float) -> float:
    twopi = 2.0 * math.pi
    y = x + math.pi
    y = y - twopi * math.floor(y / twopi)
    return y - math.pi


@njit(cache=False, fastmath=True)
def _kepler_E_from_M(M: float, e: float) -> float:
    M = _wrap_pi(M)
    if e < 1e-12:
        return M

    if e < 0.8:
        E = M + e * math.sin(M)
    else:
        E = math.pi

    for _ in range(12):
        sE = math.sin(E)
        cE = math.cos(E)
        f = E - e * sE - M
        fp = 1.0 - e * cE
        d = -f / fp
        E = E + d
        if abs(d) < 1e-13:
            break
    return E


@njit(cache=False, fastmath=True)
def _mean_from_true_anomaly(f: float, e: float) -> float:
    if e < 1e-12:
        return _wrap_pi(f)

    sf2 = math.sin(0.5 * f)
    cf2 = math.cos(0.5 * f)

    # Robust quadrant handling:
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * sf2, math.sqrt(1.0 + e) * cf2)

    M = E - e * math.sin(E)
    return _wrap_pi(M)


@njit(cache=False, fastmath=True)
def _mean_from_ecc_anomaly(E: float, e: float) -> float:
    return _wrap_pi(E - e * math.sin(E))


@njit(cache=False, fastmath=True)
def _oe_to_pv_eci(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    M: float,
    mu: float,
):
    E = _kepler_E_from_M(M, e)
    sE = math.sin(E)
    cE = math.cos(E)

    one_me2 = 1.0 - e * e
    sqrt_1me2 = math.sqrt(max(0.0, one_me2))

    r = a * (1.0 - e * cE)
    x_pf = a * (cE - e)
    y_pf = a * (sqrt_1me2 * sE)

    fac = math.sqrt(mu * a) / r
    vx_pf = -fac * sE
    vy_pf = fac * (sqrt_1me2 * cE)

    cO = math.cos(raan)
    sO = math.sin(raan)
    ci = math.cos(inc)
    si = math.sin(inc)
    cw = math.cos(argp)
    sw = math.sin(argp)

    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    # r13 = sO * si
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    # r23 = -cO * si
    r31 = sw * si
    r32 = cw * si
    # r33 = ci

    rx = r11 * x_pf + r12 * y_pf
    ry = r21 * x_pf + r22 * y_pf
    rz = r31 * x_pf + r32 * y_pf

    vx = r11 * vx_pf + r12 * vy_pf
    vy = r21 * vx_pf + r22 * vy_pf
    vz = r31 * vx_pf + r32 * vy_pf

    return rx, ry, rz, vx, vy, vz


@njit(cache=False, fastmath=True, inline="always")
def _accel_two_body_j2(
    rx: float,
    ry: float,
    rz: float,
    mu: float,
    J2: float,
    Re_m: float,
    kx: float,
    ky: float,
    kz: float,
):
    # a = -mu*r/r^3 + a_J2, with J2 evaluated about axis k (unit) expressed in the
    # same inertial frame as (rx,ry,rz).
    r2 = rx * rx + ry * ry + rz * rz
    r = math.sqrt(r2)
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r2 * inv_r

    ax = -mu * rx * inv_r3
    ay = -mu * ry * inv_r3
    az = -mu * rz * inv_r3

    if J2 != 0.0:
        kr = rx * kx + ry * ky + rz * kz  # k·r
        kr2 = kr * kr
        inv_r5 = inv_r3 * inv_r2
        fac = 1.5 * J2 * mu * (Re_m * Re_m) * inv_r5
        s = 5.0 * kr2 * inv_r2 - 1.0
        ax += fac * (rx * s - 2.0 * kr * kx)
        ay += fac * (ry * s - 2.0 * kr * ky)
        az += fac * (rz * s - 2.0 * kr * kz)

    return ax, ay, az


@njit(cache=False, fastmath=True, inline="always")
def _rk4_step_two_body_j2(
    rx: float,
    ry: float,
    rz: float,
    vx: float,
    vy: float,
    vz: float,
    dt: float,
    mu: float,
    J2: float,
    Re_m: float,
    kx: float,
    ky: float,
    kz: float,
):
    # RK4 on state [r, v]
    ax1, ay1, az1 = _accel_two_body_j2(rx, ry, rz, mu, J2, Re_m, kx, ky, kz)

    k1rx, k1ry, k1rz = vx, vy, vz
    k1vx, k1vy, k1vz = ax1, ay1, az1

    r2x = rx + 0.5 * dt * k1rx
    r2y = ry + 0.5 * dt * k1ry
    r2z = rz + 0.5 * dt * k1rz
    v2x = vx + 0.5 * dt * k1vx
    v2y = vy + 0.5 * dt * k1vy
    v2z = vz + 0.5 * dt * k1vz
    ax2, ay2, az2 = _accel_two_body_j2(r2x, r2y, r2z, mu, J2, Re_m, kx, ky, kz)
    k2rx, k2ry, k2rz = v2x, v2y, v2z
    k2vx, k2vy, k2vz = ax2, ay2, az2

    r3x = rx + 0.5 * dt * k2rx
    r3y = ry + 0.5 * dt * k2ry
    r3z = rz + 0.5 * dt * k2rz
    v3x = vx + 0.5 * dt * k2vx
    v3y = vy + 0.5 * dt * k2vy
    v3z = vz + 0.5 * dt * k2vz
    ax3, ay3, az3 = _accel_two_body_j2(r3x, r3y, r3z, mu, J2, Re_m, kx, ky, kz)
    k3rx, k3ry, k3rz = v3x, v3y, v3z
    k3vx, k3vy, k3vz = ax3, ay3, az3

    r4x = rx + dt * k3rx
    r4y = ry + dt * k3ry
    r4z = rz + dt * k3rz
    v4x = vx + dt * k3vx
    v4y = vy + dt * k3vy
    v4z = vz + dt * k3vz
    ax4, ay4, az4 = _accel_two_body_j2(r4x, r4y, r4z, mu, J2, Re_m, kx, ky, kz)
    k4rx, k4ry, k4rz = v4x, v4y, v4z
    k4vx, k4vy, k4vz = ax4, ay4, az4

    rxn = rx + (dt / 6.0) * (k1rx + 2.0 * k2rx + 2.0 * k3rx + k4rx)
    ryn = ry + (dt / 6.0) * (k1ry + 2.0 * k2ry + 2.0 * k3ry + k4ry)
    rzn = rz + (dt / 6.0) * (k1rz + 2.0 * k2rz + 2.0 * k3rz + k4rz)

    vxn = vx + (dt / 6.0) * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
    vyn = vy + (dt / 6.0) * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
    vzn = vz + (dt / 6.0) * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz)

    return rxn, ryn, rzn, vxn, vyn, vzn


@njit(cache=False, fastmath=True)
def _propagate_chain_uniform_j2(
    n_steps: int,
    dt_step: float,
    rx0: float,
    ry0: float,
    rz0: float,
    vx0: float,
    vy0: float,
    vz0: float,
    t0_s: float,
    epoch_ut1_jd: float,
    epoch_tt_jd: float,
    mu: float,
    J2: float,
    Re_m: float,
    kx: float,
    ky: float,
    kz: float,
    xp_rad: float,
    yp_rad: float,
    substeps: int,
    fill_reverse: bool,
):
    rN = np.empty((n_steps, 3), np.float64)
    vN = np.empty((n_steps, 3), np.float64)
    rI = np.empty((n_steps, 3), np.float64)
    vI = np.empty((n_steps, 3), np.float64)

    rx, ry, rz = rx0, ry0, rz0
    vx, vy, vz = vx0, vy0, vz0
    t = t0_s

    h = dt_step / float(substeps)

    for i in range(n_steps):
        for _ in range(substeps):
            rx, ry, rz, vx, vy, vz = _rk4_step_two_body_j2(
                rx, ry, rz, vx, vy, vz, h, mu, J2, Re_m, kx, ky, kz
            )

        t = t + dt_step

        jd_ut1 = epoch_ut1_jd + t / 86400.0
        jd_tt = epoch_tt_jd + t / 86400.0
        x, y, z, vx_e, vy_e, vz_e = _coarse_eci_to_itrf_pv_iau76_shortnut(
            rx, ry, rz, vx, vy, vz, jd_ut1, jd_tt, xp_rad, yp_rad
        )

        out_i = (n_steps - 1 - i) if fill_reverse else i

        rN[out_i, 0] = rx
        rN[out_i, 1] = ry
        rN[out_i, 2] = rz
        vN[out_i, 0] = vx
        vN[out_i, 1] = vy
        vN[out_i, 2] = vz

        rI[out_i, 0] = x
        rI[out_i, 1] = y
        rI[out_i, 2] = z
        vI[out_i, 0] = vx_e
        vI[out_i, 1] = vy_e
        vI[out_i, 2] = vz_e

    return rN, vN, rI, vI


@njit(cache=False, fastmath=True)
def _propagate_batch(
    dt_s: np.ndarray,
    epoch_ut1_jd: float,
    epoch_tt_jd: float,
    a: float,
    e: float,
    inc: float,
    raan0: float,
    argp0: float,
    M0: float,
    mu: float,
    raan_dot: float,
    argp_dot: float,
    M_dot: float,
    xp_rad: float,
    yp_rad: float,
):
    n = dt_s.shape[0]
    rN = np.empty((n, 3), np.float64)
    vN = np.empty((n, 3), np.float64)
    rI = np.empty((n, 3), np.float64)
    vI = np.empty((n, 3), np.float64)

    for i in range(n):
        t = dt_s[i]

        raan = raan0 + raan_dot * t
        argp = argp0 + argp_dot * t
        M = M0 + M_dot * t

        rx, ry, rz, vx, vy, vz = _oe_to_pv_eci(a, e, inc, raan, argp, M, mu)

        rN[i, 0] = rx
        rN[i, 1] = ry
        rN[i, 2] = rz
        vN[i, 0] = vx
        vN[i, 1] = vy
        vN[i, 2] = vz

        jd_ut1 = epoch_ut1_jd + t / 86400.0
        jd_tt = epoch_tt_jd + t / 86400.0

        x, y, z, vx_e, vy_e, vz_e = _coarse_eci_to_itrf_pv_iau76_shortnut(
            rx, ry, rz, vx, vy, vz, jd_ut1, jd_tt, xp_rad, yp_rad
        )

        rI[i, 0] = x
        rI[i, 1] = y
        rI[i, 2] = z
        vI[i, 0] = vx_e
        vI[i, 1] = vy_e
        vI[i, 2] = vz_e

    return rN, vN, rI, vI


EPS64 = 2.220446049250313e-16  # np.finfo(np.float64).eps


@njit(cache=False, fastmath=True, inline="always")
def _k_from_t_uniform(tq: float, inv_h: float) -> int:
    q = tq * inv_h
    k = int(math.floor(q))
    kf = float(k)

    # snap to k if q is extremely close to k
    tol = 16.0 * EPS64 * max(1.0, abs(q))
    if abs(q - kf) <= tol:
        return k
    # snap to k+1 if q is extremely close to the next integer (important for negatives too)
    if abs(q - (kf + 1.0)) <= tol:
        return k + 1
    return k


@njit(cache=False, fastmath=True, parallel=True)
def _hermite_pv_uniform_twosided_nb_par(t_query, k_min, dt, r_samples, v_samples):
    nS = r_samples.shape[0]
    nQ = t_query.shape[0]
    r_out = np.empty((nQ, 3), np.float64)
    v_out = np.empty((nQ, 3), np.float64)

    h = dt
    inv_h = 1.0 / h

    for j in prange(nQ):
        tq = t_query[j]
        k = _k_from_t_uniform(tq, inv_h)

        if k < k_min:
            k = k_min
        if k > k_min + nS - 2:
            k = k_min + nS - 2

        i = k - k_min
        t_k = k * h
        u = (tq - t_k) * inv_h

        u2 = u * u
        u3 = u2 * u

        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        dh00 = 6.0 * u2 - 6.0 * u
        dh10 = 3.0 * u2 - 4.0 * u + 1.0
        dh01 = -6.0 * u2 + 6.0 * u
        dh11 = 3.0 * u2 - 2.0 * u

        for c in range(3):
            r0 = r_samples[i, c]
            r1 = r_samples[i + 1, c]
            v0 = v_samples[i, c]
            v1 = v_samples[i + 1, c]
            r_out[j, c] = h00 * r0 + h10 * (h * v0) + h01 * r1 + h11 * (h * v1)
            v_out[j, c] = (dh00 * r0 + dh01 * r1) * inv_h + dh10 * v0 + dh11 * v1

    return r_out, v_out


@njit(cache=False, fastmath=True)
def _hermite_pv_uniform_twosided_nb(
    t_query: np.ndarray,
    k_min: int,
    dt: float,
    r_samples: np.ndarray,
    v_samples: np.ndarray,
):
    nS = r_samples.shape[0]
    nQ = t_query.shape[0]

    r_out = np.empty((nQ, 3), np.float64)
    v_out = np.empty((nQ, 3), np.float64)

    h = dt
    inv_h = 1.0 / h

    for j in range(nQ):
        tq = t_query[j]

        # correct for negatives and faster than tq/h in tight loops
        k = _k_from_t_uniform(tq, inv_h)

        if k < k_min:
            k = k_min
        if k > k_min + nS - 2:
            k = k_min + nS - 2

        i = k - k_min
        t_k = k * h
        u = (tq - t_k) * inv_h

        u2 = u * u
        u3 = u2 * u

        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        dh00 = 6.0 * u2 - 6.0 * u
        dh10 = 3.0 * u2 - 4.0 * u + 1.0
        dh01 = -6.0 * u2 + 6.0 * u
        dh11 = 3.0 * u2 - 2.0 * u

        for c in range(3):
            r0 = r_samples[i, c]
            r1 = r_samples[i + 1, c]
            v0 = v_samples[i, c]
            v1 = v_samples[i + 1, c]

            r_out[j, c] = h00 * r0 + h10 * (h * v0) + h01 * r1 + h11 * (h * v1)
            v_out[j, c] = (dh00 * r0 + dh01 * r1) * inv_h + dh10 * v0 + dh11 * v1

    return r_out, v_out


@njit(cache=False, fastmath=True)
def _fill_uniform_kepler_into(
    k_start: int,
    n_steps: int,
    dt: float,
    write_start: int,
    epoch_ut1_jd: float,
    epoch_tt_jd: float,
    a: float,
    e: float,
    inc: float,
    raan0: float,
    argp0: float,
    M0: float,
    mu: float,
    raan_dot: float,
    argp_dot: float,
    M_dot: float,
    xp_rad: float,
    yp_rad: float,
    rN: np.ndarray,
    vN: np.ndarray,
    rI: np.ndarray,
    vI: np.ndarray,
):
    inv_day = 1.0 / 86400.0

    # keep a small seconds accumulator (or compute from (k_start + j))
    t = k_start * dt

    raan = raan0 + raan_dot * t
    argp = argp0 + argp_dot * t
    M = M0 + M_dot * t

    d_raan = raan_dot * dt
    d_argp = argp_dot * dt
    d_M = M_dot * dt

    for j in range(n_steps):
        rx, ry, rz, vx, vy, vz = _oe_to_pv_eci(a, e, inc, raan, argp, M, mu)

        idx = write_start + j
        rN[idx, 0], rN[idx, 1], rN[idx, 2] = rx, ry, rz
        vN[idx, 0], vN[idx, 1], vN[idx, 2] = vx, vy, vz

        # recompute JD from small t each iteration (no jd += d_jd drift)
        jd_ut1 = epoch_ut1_jd + t * inv_day
        jd_tt = epoch_tt_jd + t * inv_day

        x, y, z, vx_e, vy_e, vz_e = _coarse_eci_to_itrf_pv_iau76_shortnut(
            rx, ry, rz, vx, vy, vz, jd_ut1, jd_tt, xp_rad, yp_rad
        )
        rI[idx, 0], rI[idx, 1], rI[idx, 2] = x, y, z
        vI[idx, 0], vI[idx, 1], vI[idx, 2] = vx_e, vy_e, vz_e

        # advance angles and time
        raan += d_raan
        argp += d_argp
        M += d_M
        t += dt


# ---------------------------------------------------------------------------
# J2 secular rates (simple)
# ---------------------------------------------------------------------------
def j2_secular_rates(
    a_m: float,
    e: float,
    inc_rad: float,
    *,
    mu: float = EARTH_MU,
    Re_m: float = WGS84_A,
    J2: float = EARTH_J2,
) -> Tuple[float, float, float]:
    """
    Returns (raan_dot, argp_dot, M_dot) [rad/s] using simple secular J2 rates.
    """
    a = float(a_m)
    e = float(e)
    inc = float(inc_rad)

    p = a * (1.0 - e * e)
    n = math.sqrt(mu / (a * a * a))
    ci = math.cos(inc)

    k = J2 * (Re_m * Re_m) / (p * p)

    raan_dot = -1.5 * n * k * ci
    argp_dot = 0.75 * n * k * (5.0 * ci * ci - 1.0)
    M_dot = n + 0.75 * n * k * math.sqrt(max(0.0, 1.0 - e * e)) * (3.0 * ci * ci - 1.0)
    return raan_dot, argp_dot, M_dot


# ---------------------------------------------------------------------------
# FastOrbit (single-satellite, cached, Hermite interpolation)
# ---------------------------------------------------------------------------
TimeLike = Union[float, np.ndarray, "AstropyTime"]


@dataclass
class FastOrbit:
    """
    Pure numpy/numba fast propagator + cached Orbit.

    Time inputs
    -----------
    - t as float seconds from epoch (fast)
    - or np.ndarray of float seconds from epoch (fast)
    - or astropy.time.Time (scalar or vector)

    Frames
    ------
    - frame="native": internal inertial basis (J2000/EME2000-like) used by this model.
      This is not a formal Orekit frame object (for strict Orekit frame semantics, use
      `nebula.propagation.orbit.Orbit`).
    - frame="itrf": Earth-fixed state derived from native via this file's built-in
      IAU-76 + 10-term nutation + GAST/GMST transform path, with optional epoch
      polar motion (xp, yp). This remains an approximation to Orekit's full EOP chain.

    Cache
    -----
    Uniform samples at dt_save_s; cubic Hermite interpolation for arbitrary query times.
    """

    epoch: "AstropyTime"  # type: ignore[name-defined]
    a_m: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    M0_rad: float
    mu: float = EARTH_MU
    dt_save_s: float = 60.0
    enable_j2: bool = False
    j2_mode: J2Mode = "secular"
    j2_substeps: int = 1
    J2: float = EARTH_J2
    Re_m: float = WGS84_A
    use_polar_motion: bool = True

    def __post_init__(self) -> None:
        if AstropyTime is None:
            raise RuntimeError("astropy is required (astropy.time.Time).")
        if not isinstance(self.epoch, AstropyTime):
            raise TypeError("epoch must be an astropy.time.Time (scalar).")
        if getattr(self.epoch, "shape", None) not in ((), None):
            raise TypeError("epoch must be a scalar astropy.time.Time.")
        if self.dt_save_s <= 0.0:
            raise ValueError("dt_save_s must be > 0")
        if self.j2_substeps < 1:
            raise ValueError("j2_substeps must be >= 1")

        self._dt = float(self.dt_save_s)

        self._epoch_utc = self.epoch.utc
        self._epoch_tt = self.epoch.tt

        # UT1 may require IERS; fall back gracefully
        try:
            self._epoch_ut1 = self.epoch.ut1
            self._epoch_ut1_jd = float(self._epoch_ut1.jd)
        except Exception:
            self._epoch_ut1 = self._epoch_utc
            self._epoch_ut1_jd = float(self._epoch_utc.jd)

        self._epoch_tt_jd = float(self._epoch_tt.jd)

        # Polar motion at epoch (xp, yp in radians), optional.
        # Keeping this constant over the cache improves ITRF agreement materially
        # at negligible cost and keeps all propagation/rotation loops njit'd.
        xp_rad = 0.0
        yp_rad = 0.0
        if bool(self.use_polar_motion) and _astropy_get_polar_motion is not None:
            try:
                xp, yp = _astropy_get_polar_motion(self._epoch_utc)  # type: ignore[misc]
                xp_rad = float(xp)
                yp_rad = float(yp)
            except Exception:
                xp_rad = 0.0
                yp_rad = 0.0
        self._xp_rad = float(xp_rad)
        self._yp_rad = float(yp_rad)

        a = float(self.a_m)
        n = math.sqrt(self.mu / (a * a * a))

        # For Orekit "degree=2, order=0" behavior, use:
        #   enable_j2=True, j2_mode="osculating"
        self._use_j2_cart = bool(self.enable_j2 and self.j2_mode == "osculating")
        if self.enable_j2:
            kx, ky, kz = _coarse_j2_axis_native_iau76_shortnut(self._epoch_tt_jd)
        else:
            kx, ky, kz = 0.0, 0.0, 1.0
        self._j2_kx = float(kx)
        self._j2_ky = float(ky)
        self._j2_kz = float(kz)

        if self.enable_j2 and (not self._use_j2_cart):
            # legacy: mean-elements secular drift only
            raan_dot, argp_dot, M_dot = j2_secular_rates(
                self.a_m, self.e, self.i_rad, mu=self.mu, Re_m=self.Re_m, J2=self.J2
            )
        else:
            # two-body (and also the base rates for osculating mode; not used for stepping)
            raan_dot = 0.0
            argp_dot = 0.0
            M_dot = n

        self._raan_dot = float(raan_dot)
        self._argp_dot = float(argp_dot)
        self._M_dot = float(M_dot)

        # Cache indexing:
        # - contiguous samples for k in [k_min, k_max]
        # - sample for k_min is stored at index _start
        self._k_min = 0
        self._k_max = 0
        self._n = 1

        self._cap = 256
        self._start = (self._cap - self._n) // 2

        self._r_native = np.empty((self._cap, 3), np.float64)
        self._v_native = np.empty((self._cap, 3), np.float64)
        self._r_itrf = np.empty((self._cap, 3), np.float64)
        self._v_itrf = np.empty((self._cap, 3), np.float64)

        # Seed sample at k=0 (no temporary arrays)
        rx, ry, rz, vx, vy, vz = _oe_to_pv_eci(
            float(self.a_m),
            float(self.e),
            float(self.i_rad),
            float(self.raan_rad),
            float(self.argp_rad),
            float(self.M0_rad),
            float(self.mu),
        )
        x, y, z, vx_e, vy_e, vz_e = _coarse_eci_to_itrf_pv_iau76_shortnut(
            rx,
            ry,
            rz,
            vx,
            vy,
            vz,
            float(self._epoch_ut1_jd),
            float(self._epoch_tt_jd),
            float(self._xp_rad),
            float(self._yp_rad),
        )
        self._r_native[self._start, 0] = rx
        self._r_native[self._start, 1] = ry
        self._r_native[self._start, 2] = rz
        self._v_native[self._start, 0] = vx
        self._v_native[self._start, 1] = vy
        self._v_native[self._start, 2] = vz
        self._r_itrf[self._start, 0] = x
        self._r_itrf[self._start, 1] = y
        self._r_itrf[self._start, 2] = z
        self._v_itrf[self._start, 0] = vx_e
        self._v_itrf[self._start, 1] = vy_e
        self._v_itrf[self._start, 2] = vz_e

    @property
    def dt(self) -> float:
        return self._dt

    def coverage(self) -> Tuple[float, float]:
        return float(self._k_min) * self._dt, float(self._k_max) * self._dt

    def precompute(self, t_min_s: float, t_max_s: float) -> None:
        t_min_s = float(t_min_s)
        t_max_s = float(t_max_s)
        if t_max_s < t_min_s:
            t_min_s, t_max_s = t_max_s, t_min_s
        self._ensure_covered(np.array([t_min_s, t_max_s], np.float64))

    def pv(self, t: TimeLike, frame: FrameKind = "native"):
        dt_s, is_scalar = self._to_dt_seconds(t)
        if dt_s.size == 0:
            return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)

        self._ensure_covered(dt_s)

        # If only one sample, allow exact-knot query.
        if (self._k_max - self._k_min) < 1:
            if (
                dt_s.size == 1
                and abs(float(dt_s[0]) - float(self._k_min) * self._dt) <= 1e-12
            ):
                if frame == "native":
                    r = self._r_native[self._start].copy()
                    v = self._v_native[self._start].copy()
                else:
                    r = self._r_itrf[self._start].copy()
                    v = self._v_itrf[self._start].copy()
                return r, v

        rS, vS = self._samples(frame)
        if dt_s.size >= _HERMITE_PAR_THRESHOLD:
            rQ, vQ = _hermite_pv_uniform_twosided_nb_par(
                dt_s, self._k_min, self._dt, rS, vS
            )
        else:
            rQ, vQ = _hermite_pv_uniform_twosided_nb(
                dt_s, self._k_min, self._dt, rS, vS
            )

        if is_scalar:
            return rQ[0], vQ[0]
        return rQ, vQ

    def pos(self, t: TimeLike, frame: FrameKind = "native") -> np.ndarray:
        r, _ = self.pv(t, frame=frame)
        return r

    def vel(self, t: TimeLike, frame: FrameKind = "native") -> np.ndarray:
        _, v = self.pv(t, frame=frame)
        return v

    def pv_itrf(self, t: TimeLike):
        return self.pv(t, frame="itrf")

    def pos_itrf(self, t: TimeLike) -> np.ndarray:
        return self.pos(t, frame="itrf")

    def vel_itrf(self, t: TimeLike) -> np.ndarray:
        return self.vel(t, frame="itrf")

    def lla(self, t: TimeLike):
        r_itrf = self.pos_itrf(t)
        if r_itrf.ndim == 1:
            la, lo, alt = _ecef2geodetic_deg(
                float(r_itrf[0]), float(r_itrf[1]), float(r_itrf[2])
            )
            lo = (lo + 180.0) % 360.0 - 180.0
            return la, lo, alt
        lat, lon, alt = _ecef2geodetic_vec_ecef_deg(r_itrf, wrap_lon=True)
        return lat, lon, alt

    # -----------------------------------------------------------------------
    # Cache internals
    # -----------------------------------------------------------------------
    def _samples(self, frame: FrameKind):
        s = self._start
        n = self._n
        if frame == "native":
            return self._r_native[s : s + n], self._v_native[s : s + n]
        return self._r_itrf[s : s + n], self._v_itrf[s : s + n]

    def _ensure_covered(self, dt_s: np.ndarray) -> None:
        if np.any(~np.isfinite(dt_s)):
            raise ValueError("Non-finite query times are not supported")

        lo = float(dt_s.min())
        hi = float(dt_s.max())

        k_need_lo = int(math.floor(lo / self._dt))
        k_need_hi = int(math.ceil(hi / self._dt))

        if k_need_lo < self._k_min:
            self._extend_backward_to(k_need_lo)
        if k_need_hi > self._k_max:
            self._extend_forward_to(k_need_hi)

    def _ensure_capacity_for_range(self, new_k_min: int, new_k_max: int) -> None:
        """
        Ensure internal arrays have space to extend current cache to [new_k_min, new_k_max]
        without changing the current (k_min, k_max) invariants.

        Invariants before/after:
          - current cached samples cover [self._k_min, self._k_max]
          - they are stored contiguously at indices [self._start, self._start + self._n)
        """
        if new_k_min > self._k_min:
            new_k_min = self._k_min
        if new_k_max < self._k_max:
            new_k_max = self._k_max

        L = self._k_min - new_k_min  # needed slots before current start
        R = new_k_max - self._k_max  # needed slots after current end

        # Check if current arrays already have space.
        if self._start >= L and (self._start + self._n + R) <= self._cap:
            return

        need = self._n + L + R + 2 * _CACHE_MARGIN

        cap = self._cap
        while cap < need:
            cap *= 2

        rN = np.empty((cap, 3), np.float64)
        vN = np.empty((cap, 3), np.float64)
        rI = np.empty((cap, 3), np.float64)
        vI = np.empty((cap, 3), np.float64)

        # Place current block at new_start so we have L space before and R space after.
        new_start = _CACHE_MARGIN + L

        s = self._start
        n = self._n
        rN[new_start : new_start + n] = self._r_native[s : s + n]
        vN[new_start : new_start + n] = self._v_native[s : s + n]
        rI[new_start : new_start + n] = self._r_itrf[s : s + n]
        vI[new_start : new_start + n] = self._v_itrf[s : s + n]

        self._r_native, self._v_native, self._r_itrf, self._v_itrf = rN, vN, rI, vI
        self._cap = cap
        self._start = new_start

    def _extend_forward_to(self, k_target: int) -> None:
        if k_target <= self._k_max:
            return

        self._ensure_capacity_for_range(self._k_min, k_target)

        n_add = k_target - self._k_max

        if self._use_j2_cart:
            last_i = self._start + self._n - 1
            rx0, ry0, rz0 = (
                float(self._r_native[last_i, 0]),
                float(self._r_native[last_i, 1]),
                float(self._r_native[last_i, 2]),
            )
            vx0, vy0, vz0 = (
                float(self._v_native[last_i, 0]),
                float(self._v_native[last_i, 1]),
                float(self._v_native[last_i, 2]),
            )
            t0_s = float(self._k_max) * self._dt

            rN, vN, rI, vI = _propagate_chain_uniform_j2(
                int(n_add),
                float(self._dt),
                rx0,
                ry0,
                rz0,
                vx0,
                vy0,
                vz0,
                float(t0_s),
                float(self._epoch_ut1_jd),
                float(self._epoch_tt_jd),
                float(self.mu),
                float(self.J2),
                float(self.Re_m),
                float(self._j2_kx),
                float(self._j2_ky),
                float(self._j2_kz),
                float(self._xp_rad),
                float(self._yp_rad),
                int(self.j2_substeps),
                False,
            )

            write_start = self._start + self._n
            self._r_native[write_start : write_start + n_add] = rN
            self._v_native[write_start : write_start + n_add] = vN
            self._r_itrf[write_start : write_start + n_add] = rI
            self._v_itrf[write_start : write_start + n_add] = vI

            self._k_max = k_target
            self._n += n_add
            return

        # Kepler / secular mode: fill directly into existing arrays (no temp alloc)
        write_start = self._start + self._n
        _fill_uniform_kepler_into(
            int(self._k_max + 1),
            int(n_add),
            float(self._dt),
            int(write_start),
            float(self._epoch_ut1_jd),
            float(self._epoch_tt_jd),
            float(self.a_m),
            float(self.e),
            float(self.i_rad),
            float(self.raan_rad),
            float(self.argp_rad),
            float(self.M0_rad),
            float(self.mu),
            float(self._raan_dot),
            float(self._argp_dot),
            float(self._M_dot),
            float(self._xp_rad),
            float(self._yp_rad),
            self._r_native,
            self._v_native,
            self._r_itrf,
            self._v_itrf,
        )

        self._k_max = k_target
        self._n += n_add

    def _extend_backward_to(self, k_target: int) -> None:
        if k_target >= self._k_min:
            return

        self._ensure_capacity_for_range(k_target, self._k_max)

        n_add = self._k_min - k_target

        if self._use_j2_cart:
            first_i = self._start
            rx0, ry0, rz0 = (
                float(self._r_native[first_i, 0]),
                float(self._r_native[first_i, 1]),
                float(self._r_native[first_i, 2]),
            )
            vx0, vy0, vz0 = (
                float(self._v_native[first_i, 0]),
                float(self._v_native[first_i, 1]),
                float(self._v_native[first_i, 2]),
            )
            t0_s = float(self._k_min) * self._dt

            rN, vN, rI, vI = _propagate_chain_uniform_j2(
                int(n_add),
                float(-self._dt),
                rx0,
                ry0,
                rz0,
                vx0,
                vy0,
                vz0,
                float(t0_s),
                float(self._epoch_ut1_jd),
                float(self._epoch_tt_jd),
                float(self.mu),
                float(self.J2),
                float(self.Re_m),
                float(self._j2_kx),
                float(self._j2_ky),
                float(self._j2_kz),
                float(self._xp_rad),
                float(self._yp_rad),
                int(self.j2_substeps),
                True,  # fill in ascending k order
            )

            write_start = self._start - n_add
            self._r_native[write_start : self._start] = rN
            self._v_native[write_start : self._start] = vN
            self._r_itrf[write_start : self._start] = rI
            self._v_itrf[write_start : self._start] = vI

            self._start = write_start
            self._k_min = k_target
            self._n += n_add
            return

        # Kepler / secular mode: fill directly into existing arrays (no temp alloc)
        write_start = self._start - n_add
        _fill_uniform_kepler_into(
            int(k_target),
            int(n_add),
            float(self._dt),
            int(write_start),
            float(self._epoch_ut1_jd),
            float(self._epoch_tt_jd),
            float(self.a_m),
            float(self.e),
            float(self.i_rad),
            float(self.raan_rad),
            float(self.argp_rad),
            float(self.M0_rad),
            float(self.mu),
            float(self._raan_dot),
            float(self._argp_dot),
            float(self._M_dot),
            float(self._xp_rad),
            float(self._yp_rad),
            self._r_native,
            self._v_native,
            self._r_itrf,
            self._v_itrf,
        )

        self._start = write_start
        self._k_min = k_target
        self._n += n_add

    # -----------------------------------------------------------------------
    # Time normalization
    # -----------------------------------------------------------------------
    def _to_dt_seconds(self, t: TimeLike) -> Tuple[np.ndarray, bool]:
        # float seconds
        if isinstance(t, (float, int)):
            return np.array([float(t)], np.float64), True

        # ndarray of seconds
        if isinstance(t, np.ndarray):
            if t.dtype.kind not in ("f", "i"):
                raise TypeError(
                    "numpy array time input must be float/int seconds from epoch"
                )
            arr = np.ascontiguousarray(t.astype(np.float64))
            is_scalar = arr.ndim == 0
            return np.atleast_1d(arr), is_scalar

        # astropy Time (scalar or vector)
        if AstropyTime is not None and isinstance(t, AstropyTime):
            is_scalar = getattr(t, "shape", None) == ()
            dt = (t.utc - self._epoch_utc).to_value("s")  # type: ignore
            dt_arr = np.atleast_1d(np.asarray(dt, dtype=np.float64))
            return dt_arr, is_scalar

        raise TypeError(
            "t must be float seconds, np.ndarray of seconds, or astropy.time.Time"
        )

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------
    @classmethod
    def from_kepler(
        cls,
        *,
        epoch: "AstropyTime",  # type: ignore[name-defined]
        a_m: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type: AngleType = "true",
        degrees: bool = False,
        mu: float = EARTH_MU,
        dt_save_s: float = 60.0,
        enable_j2: bool = False,
        j2_mode: J2Mode = "secular",
        j2_substeps: int = 1,
        J2: float = EARTH_J2,
        Re_m: float = WGS84_A,
        use_polar_motion: bool = True,
    ) -> "FastOrbit":
        """
        Build a FastOrbit from Keplerian elements.

        Notes
        -----
        - Propagation occurs in this class's internal "native" inertial basis.
        - That native basis is J2000/EME2000-like, but not a strict Orekit frame object.
        - use_polar_motion=True applies epoch xp/yp polar motion in the native->ITRF
          rotation chain, improving Orekit agreement at low runtime cost.
        - If you need explicit Orekit inertial frame selection (GCRF/EME2000/ICRF/TEME/etc.),
          use `nebula.propagation.orbit.Orbit`.
        """
        if AstropyTime is None:
            raise RuntimeError("astropy is required (astropy.time.Time).")
        if not isinstance(epoch, AstropyTime):
            raise TypeError("epoch must be an astropy.time.Time")

        if degrees:
            i = math.radians(float(i))
            raan = math.radians(float(raan))
            argp = math.radians(float(argp))
            anomaly = math.radians(float(anomaly))

        e = float(e)
        if anomaly_type == "mean":
            M0 = _wrap_pi(float(anomaly))
        elif anomaly_type == "eccentric":
            M0 = _mean_from_ecc_anomaly(float(anomaly), e)
        elif anomaly_type == "true":
            M0 = _mean_from_true_anomaly(float(anomaly), e)
        else:
            raise ValueError("anomaly_type must be 'true', 'mean', or 'eccentric'")

        return cls(
            epoch=epoch,
            a_m=float(a_m),
            e=e,
            i_rad=float(i),
            raan_rad=float(raan),
            argp_rad=float(argp),
            M0_rad=float(M0),
            mu=float(mu),
            dt_save_s=float(dt_save_s),
            enable_j2=bool(enable_j2),
            j2_mode=j2_mode,
            j2_substeps=int(j2_substeps),
            J2=float(J2),
            Re_m=float(Re_m),
            use_polar_motion=bool(use_polar_motion),
        )


# ---------------------------------------------------------------------------
# Optional: very fast multi-satellite batch propagation (no cache)
# ---------------------------------------------------------------------------
@njit(cache=False, fastmath=True)
def propagate_constellation_pv(
    dt_s: np.ndarray,
    epoch_ut1_jd: float,
    epoch_tt_jd: float,
    a: np.ndarray,
    e: np.ndarray,
    inc: np.ndarray,
    raan0: np.ndarray,
    argp0: np.ndarray,
    M0: np.ndarray,
    mu: float,
    raan_dot: np.ndarray,
    argp_dot: np.ndarray,
    M_dot: np.ndarray,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Propagate many satellites at the same time grid.

    Inputs:
      - dt_s: (Nt,)
      - elements/rates: (Ns,)
    Outputs:
      - r_eci, v_eci, r_ecef, v_ecef: (Ns, Nt, 3)
    """
    Nt = dt_s.shape[0]
    Ns = a.shape[0]

    rN = np.empty((Ns, Nt, 3), np.float64)
    vN = np.empty((Ns, Nt, 3), np.float64)
    rI = np.empty((Ns, Nt, 3), np.float64)
    vI = np.empty((Ns, Nt, 3), np.float64)

    for s in range(Ns):
        for i in range(Nt):
            t = dt_s[i]
            raan = raan0[s] + raan_dot[s] * t
            argp = argp0[s] + argp_dot[s] * t
            M = M0[s] + M_dot[s] * t

            rx, ry, rz, vx, vy, vz = _oe_to_pv_eci(
                a[s], e[s], inc[s], raan, argp, M, mu
            )

            rN[s, i, 0] = rx
            rN[s, i, 1] = ry
            rN[s, i, 2] = rz
            vN[s, i, 0] = vx
            vN[s, i, 1] = vy
            vN[s, i, 2] = vz

            jd_ut1 = epoch_ut1_jd + t / 86400.0
            jd_tt = epoch_tt_jd + t / 86400.0
            x, y, z, vx_e, vy_e, vz_e = _coarse_eci_to_itrf_pv_iau76_shortnut(
                rx, ry, rz, vx, vy, vz, jd_ut1, jd_tt, xp_rad, yp_rad
            )

            rI[s, i, 0] = x
            rI[s, i, 1] = y
            rI[s, i, 2] = z
            vI[s, i, 0] = vx_e
            vI[s, i, 1] = vy_e
            vI[s, i, 2] = vz_e

    return rN, vN, rI, vI
