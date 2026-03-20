# _analytical_core.py

import numpy as np
import math
from numba import njit, prange
from nstk.transforms._ecef2geodetic import ecef2geodetic
from nstk.transforms.constants import (
    WGS84_A,
    WGS84_E2,
)


TWOPI = 2.0 * np.pi
PI = np.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0


@njit(cache=False, inline="always")
def _wrap_pi(x):
    two_pi = TWOPI
    y = x + np.pi
    y = y - math.floor(y / two_pi) * two_pi
    return y - np.pi


@njit(cache=False, fastmath=True, inline="always")
def _reff_gauss(lat_rad: float) -> float:
    """Higher-fidelity effective radius via Gaussian radius sqrt(M*N)."""
    s = np.sin(lat_rad)
    den = 1.0 - WGS84_E2 * s * s
    # Meridional radius M = a(1-e^2)/(1-e^2 sin^2)^(3/2)
    M = WGS84_A * (1.0 - WGS84_E2) / (den * np.sqrt(den))
    # Prime-vertical radius N = a/sqrt(1-e^2 sin^2)
    N = WGS84_A / np.sqrt(den)
    return np.sqrt(M * N)


@njit(cache=False, fastmath=True, inline="always")
def _psi_half_angle(reff: float, h_m: float, elev_min_rad: float) -> float:
    """Spherical footprint half-angle ψ(h,e_min,R)."""
    if h_m < 0.0:
        h_m = 0.0
    se = np.sin(elev_min_rad)
    ce = np.cos(elev_min_rad)
    alpha = reff / (reff + h_m)  # R / (R + h)
    a2c2 = (alpha * alpha) * (ce * ce)
    root = 1.0 - a2c2
    if root < 0.0:
        root = 0.0
    cospsi = alpha * (ce * ce) + se * np.sqrt(root)
    if cospsi > 1.0:
        cospsi = 1.0
    elif cospsi < -1.0:
        cospsi = -1.0
    return np.arccos(cospsi)


@njit(cache=False, inline="always")
def _fill_range_stamp_set(stamp: np.ndarray, j: int, i0: int, i1: int, epoch):
    # contiguous slice write (Numba emits a tight loop)
    stamp[j, i0 : i1 + 1] = epoch


@njit(cache=False, inline="always")
def _fill_range_stamp_clear(stamp: np.ndarray, j: int, i0: int, i1: int):
    stamp[j, i0 : i1 + 1] = 0


@njit(cache=False, inline="always")
def _collect_arc_intervals_0_2pi(lon0_2pi: float, halfw: float):
    """
    Returns up to 2 intervals [a0,b0], [a1,b1] in [0, 2π], inclusive.
    Encoded as (n, a0, b0, a1, b1) where n in {1,2}.
    """
    two_pi = TWOPI
    # raw endpoints can leave [0,2π)
    s = lon0_2pi - halfw
    e = lon0_2pi + halfw

    # clamp-ish epsilon to avoid exact 2π
    eps = 1e-15

    if s < 0.0:
        # [0, e] and [s+2π, 2π)
        a0 = 0.0
        b0 = e
        if b0 > two_pi:
            b0 = two_pi
        a1 = s + two_pi
        b1 = two_pi - eps
        if a1 < 0.0:
            a1 = 0.0
        return 2, a0, b0, a1, b1

    if e > two_pi:
        # [s, 2π) and [0, e-2π]
        a0 = s
        b0 = two_pi - eps
        a1 = 0.0
        b1 = e - two_pi
        if b1 < 0.0:
            b1 = 0.0
        return 2, a0, b0, a1, b1

    # single interval
    a0 = s
    b0 = e
    if b0 >= two_pi:
        b0 = two_pi - eps
    return 1, a0, b0, 0.0, 0.0


@njit(cache=False, inline="always")
def _clamp_i0_i1(i0: int, i1: int, nx: int):
    if i0 < 0:
        i0 = 0
    if i1 > nx - 1:
        i1 = nx - 1
    return i0, i1, i1 >= i0


@njit(cache=False, inline="always")
def _interval_to_i0_i1(
    a: float,
    b: float,
    t0: float,
    t1: float,
    base: float,
    dlon: float,
    nx: int,
):

    s = a if a > t0 else t0
    e = b if b < t1 else t1
    if s > e:
        return 0, -1, False

    i0 = int(math.ceil((s - base) / dlon))
    i1 = int(math.floor((e - base) / dlon))
    return _clamp_i0_i1(i0, i1, nx)


@njit(cache=False, parallel=True)
def coverage_stamp_kernel_analytical(
    ox: float,  # observer ECEF x meters
    oy: float,  # observer ECEF y meters
    oz: float,  # observer ECEF z meters
    # precomputed latitude rows (geocentric + sin/cos)
    lat_gc_rows: np.ndarray,  # (ny,)
    sin_lat_rows: np.ndarray,  # (ny,)
    cos_lat_rows: np.ndarray,  # (ny,)
    # longitude target domain in a continuous [0,2π)+k*2π system:
    lon_start_2pi: float,  # in [0,2π)
    lon_span: float,  # > 0
    dlon: float,  # radians, uniform
    base: float,  # lon_start_2pi (+half_dlon if using cell-centers)
    nx: int,
    # elevation constraints
    min_el_rad: float,
    max_el_rad: float,
    # stamp buffer + epoch
    stamp: np.ndarray,  # (ny, nx) uint16/uint8/etc preallocated
    epoch,  # np.uint8 (or other unsigned) current epoch
):
    ny = lat_gc_rows.size
    obs_lat_rad, _, h_m = ecef2geodetic(ox, oy, oz)

    # cap center from observer position vector (geocentric)
    lon0 = math.atan2(oy, ox)
    phi0_gc = math.atan2(oz, math.sqrt(ox * ox + oy * oy))

    # wrap lon into [-pi, pi)
    lon_c = _wrap_pi(lon0)

    sphi = math.sin(phi0_gc)
    cphi = math.cos(phi0_gc)

    # effective Earth radius at geodetic latitude
    Reff = _reff_gauss(obs_lat_rad)

    # outer cap: elevation >= min_el
    psi_outer = _psi_half_angle(Reff, h_m, min_el_rad)
    cps_outer = math.cos(psi_outer)

    # inner cap (to subtract): elevation >= max_el (only if max_el < 90°)
    half_pi = 0.5 * PI
    use_max_el = max_el_rad < (half_pi - 1e-12)

    psi_inner = 0.0
    cps_inner = 1.0
    if use_max_el:
        if max_el_rad <= min_el_rad + 1e-15:
            return stamp
        psi_inner = _psi_half_angle(Reff, h_m, max_el_rad)
        cps_inner = math.cos(psi_inner)

    # target lon domain [t0, t1] in continuous coordinates
    t0 = lon_start_2pi
    t1 = lon_start_2pi + lon_span
    two_pi = TWOPI
    need_shift = t1 > (two_pi + 1e-15)

    DENOM_TOL = 1e-8

    # center lon in [0,2π)
    lon0_2pi = lon_c + PI
    if lon0_2pi < 0.0:
        lon0_2pi += two_pi
    if lon0_2pi >= two_pi:
        lon0_2pi -= two_pi

    for j in prange(ny):
        slat = sin_lat_rows[j]
        clat = cos_lat_rows[j]
        lat_row = lat_gc_rows[j]

        denom = clat * cphi

        # ---- OUTER cap (Option A: no eps "full-row" shortcuts; always solve halfw) ----
        if abs(lat_row - phi0_gc) > psi_outer:
            continue

        num = cps_outer - slat * sphi

        if abs(denom) < DENOM_TOL:
            # degeneracy: boundary independent of lon; either whole row passes or none
            if slat * sphi >= cps_outer:
                _fill_range_stamp_set(stamp, j, 0, nx - 1, epoch)
            else:
                continue
        else:
            arg = num / denom
            if arg > 1.0:
                arg = 1.0
            elif arg < -1.0:
                arg = -1.0
            halfw = math.acos(arg)
            n_int, a0, b0, a1, b1 = _collect_arc_intervals_0_2pi(lon0_2pi, halfw)

            # interval 0
            i0, i1, ok = _interval_to_i0_i1(a0, b0, t0, t1, base, dlon, nx)
            if ok:
                _fill_range_stamp_set(stamp, j, i0, i1, epoch)
            if need_shift:
                i0, i1, ok = _interval_to_i0_i1(
                    a0 + two_pi, b0 + two_pi, t0, t1, base, dlon, nx
                )
                if ok:
                    _fill_range_stamp_set(stamp, j, i0, i1, epoch)

            # interval 1
            if n_int == 2:
                i0, i1, ok = _interval_to_i0_i1(a1, b1, t0, t1, base, dlon, nx)
                if ok:
                    _fill_range_stamp_set(stamp, j, i0, i1, epoch)
                if need_shift:
                    i0, i1, ok = _interval_to_i0_i1(
                        a1 + two_pi, b1 + two_pi, t0, t1, base, dlon, nx
                    )
                    if ok:
                        _fill_range_stamp_set(stamp, j, i0, i1, epoch)

        # ---- INNER cap subtraction (Option A: no eps shortcuts; always solve halfw) ----
        if use_max_el:
            if abs(lat_row - phi0_gc) > psi_inner:
                continue

            num_i = cps_inner - slat * sphi

            if abs(denom) < DENOM_TOL:
                if slat * sphi >= cps_inner:
                    _fill_range_stamp_clear(stamp, j, 0, nx - 1)
                continue

            arg_i = num_i / denom
            if arg_i > 1.0:
                arg_i = 1.0
            elif arg_i < -1.0:
                arg_i = -1.0

            halfw_i = math.acos(arg_i)
            n2, ia0, ib0, ia1, ib1 = _collect_arc_intervals_0_2pi(lon0_2pi, halfw_i)

            # clear interval 0
            i0, i1, ok = _interval_to_i0_i1(ia0, ib0, t0, t1, base, dlon, nx)
            if ok:
                _fill_range_stamp_clear(stamp, j, i0, i1)
            if need_shift:
                i0, i1, ok = _interval_to_i0_i1(
                    ia0 + two_pi, ib0 + two_pi, t0, t1, base, dlon, nx
                )
                if ok:
                    _fill_range_stamp_clear(stamp, j, i0, i1)

            # clear interval 1
            if n2 == 2:
                i0, i1, ok = _interval_to_i0_i1(ia1, ib1, t0, t1, base, dlon, nx)
                if ok:
                    _fill_range_stamp_clear(stamp, j, i0, i1)
                if need_shift:
                    i0, i1, ok = _interval_to_i0_i1(
                        ia1 + two_pi, ib1 + two_pi, t0, t1, base, dlon, nx
                    )
                    if ok:
                        _fill_range_stamp_clear(stamp, j, i0, i1)

    return stamp


# ------------------------------------------------------------
# Analytical row-intervals (scaled spherical cap) -> index intervals
#   Matches coverage_stamp_kernel_analytical semantics, but returns
#   up to two [i0,i1] inclusive intervals for OUTER and INNER caps.
# ------------------------------------------------------------


@njit(cache=True, inline="always")
def _merge_add_interval_i(
    n: int,
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    L: int,
    R: int,
):
    """Merge [L,R] into up to 2 stored inclusive index intervals (sorted, merged if adjacent)."""
    if L > R:
        return n, a0, b0, a1, b1

    if n == 0:
        return 1, L, R, 0, -1

    if n == 1:
        # insert/merge with [a0,b0]
        if R < a0 - 1:
            return 2, L, R, a0, b0
        if L > b0 + 1:
            return 2, a0, b0, L, R
        # overlap/adjacent -> merge
        if L < a0:
            a0 = L
        if R > b0:
            b0 = R
        return 1, a0, b0, 0, -1

    # n == 2; assume [a0,b0], [a1,b1] with b0 < a1
    # try merge with first
    if not (R < a0 - 1 or L > b0 + 1):
        if L < a0:
            a0 = L
        if R > b0:
            b0 = R
    # try merge with second
    elif not (R < a1 - 1 or L > b1 + 1):
        if L < a1:
            a1 = L
        if R > b1:
            b1 = R
    else:
        # would create a 3rd disjoint interval; should not occur for arc∩domain,
        # but collapse to a single hull interval as a safe fallback.
        if L < a0:
            a0 = L
        if R > b1:
            b1 = R
        return 1, a0, b1, 0, -1

    # if merging caused overlap/adjacency between the two, collapse to 1
    if b0 >= a1 - 1:
        if b1 > b0:
            b0 = b1
        return 1, a0, b0, 0, -1

    return 2, a0, b0, a1, b1


@njit(cache=True, inline="always")
def _arc_to_index_intervals(
    lon0_2pi: float,
    halfw: float,
    t0: float,
    t1: float,
    base: float,
    dlon: float,
    nx: int,
    need_shift: bool,
):
    """
    Convert a circular arc centered at lon0_2pi with half-width halfw into
    up to 2 inclusive index intervals within the continuous target domain [t0,t1].
    """
    n = 0
    a0i = 0
    b0i = -1
    a1i = 0
    b1i = -1

    n_int, a0, b0, a1, b1 = _collect_arc_intervals_0_2pi(lon0_2pi, halfw)

    # interval 0
    i0, i1, ok = _interval_to_i0_i1(a0, b0, t0, t1, base, dlon, nx)
    if ok:
        n, a0i, b0i, a1i, b1i = _merge_add_interval_i(n, a0i, b0i, a1i, b1i, i0, i1)
    if need_shift:
        i0, i1, ok = _interval_to_i0_i1(a0 + TWOPI, b0 + TWOPI, t0, t1, base, dlon, nx)
        if ok:
            n, a0i, b0i, a1i, b1i = _merge_add_interval_i(n, a0i, b0i, a1i, b1i, i0, i1)

    # interval 1 (if wrap)
    if n_int == 2:
        i0, i1, ok = _interval_to_i0_i1(a1, b1, t0, t1, base, dlon, nx)
        if ok:
            n, a0i, b0i, a1i, b1i = _merge_add_interval_i(n, a0i, b0i, a1i, b1i, i0, i1)
        if need_shift:
            i0, i1, ok = _interval_to_i0_i1(
                a1 + TWOPI, b1 + TWOPI, t0, t1, base, dlon, nx
            )
            if ok:
                n, a0i, b0i, a1i, b1i = _merge_add_interval_i(
                    n, a0i, b0i, a1i, b1i, i0, i1
                )

    return n, a0i, b0i, a1i, b1i


@njit(cache=True, fastmath=True, inline="always")
def analytical_cap_params(
    ox: float,
    oy: float,
    oz: float,
    min_el_rad: float,
    max_el_rad: float,
):
    """
    Precompute per-observer analytical-cap parameters used by row-interval kernels.
    Returns:
      lon0_2pi, phi0_gc, sphi, cphi, psi_outer, cps_outer, use_max, psi_inner, cps_inner
    """
    obs_lat_rad, _, h_m = ecef2geodetic(ox, oy, oz)

    # cap center from observer position vector (geocentric)
    lon0 = math.atan2(oy, ox)
    phi0_gc = math.atan2(oz, math.sqrt(ox * ox + oy * oy))
    sphi = math.sin(phi0_gc)
    cphi = math.cos(phi0_gc)

    # wrap lon into [-pi, pi)
    lon_c = _wrap_pi(lon0)

    # center lon in [0,2π)
    lon0_2pi = lon_c + PI
    if lon0_2pi < 0.0:
        lon0_2pi += TWOPI
    if lon0_2pi >= TWOPI:
        lon0_2pi -= TWOPI

    # effective Earth radius at geodetic latitude
    Reff = _reff_gauss(obs_lat_rad)

    # outer cap: elevation >= min_el
    psi_outer = _psi_half_angle(Reff, h_m, min_el_rad)
    cps_outer = math.cos(psi_outer)

    # inner cap (exclude): elevation >= max_el, only if max_el < 90°
    half_pi = 0.5 * PI
    use_max = (max_el_rad < (half_pi - 1e-12)) and (max_el_rad > (min_el_rad + 1e-15))

    psi_inner = 0.0
    cps_inner = 1.0
    if use_max:
        psi_inner = _psi_half_angle(Reff, h_m, max_el_rad)
        cps_inner = math.cos(psi_inner)

    return (
        lon0_2pi,
        phi0_gc,
        sphi,
        cphi,
        psi_outer,
        cps_outer,
        use_max,
        psi_inner,
        cps_inner,
    )


@njit(cache=True, fastmath=True, inline="always")
def coverage_row_intervals_analytical_1row(
    lat_row: float,
    slat: float,
    clat: float,
    lon0_2pi: float,
    phi0_gc: float,
    sphi: float,
    cphi: float,
    psi_outer: float,
    cps_outer: float,
    use_max_el: bool,
    psi_inner: float,
    cps_inner: float,
    t0: float,
    t1: float,
    base: float,
    dlon: float,
    nx: int,
    need_shift: bool,
):
    """
    Return (outer_n, oa0,ob0, oa1,ob1, inner_n, ia0,ib0, ia1,ib1) as inclusive indices.
    outer_n/inner_n in {0,1,2}. Empty uses b=-1.
    """
    DENOM_TOL = 1e-8

    # ---- OUTER ----
    if abs(lat_row - phi0_gc) > psi_outer:
        return 0, 0, -1, 0, -1, 0, 0, -1, 0, -1

    denom = clat * cphi
    if abs(denom) < DENOM_TOL:
        # boundary independent of lon
        if slat * sphi >= cps_outer:
            outer_n, oa0, ob0, oa1, ob1 = 1, 0, nx - 1, 0, -1
        else:
            return 0, 0, -1, 0, -1, 0, 0, -1, 0, -1
    else:
        arg = (cps_outer - slat * sphi) / denom
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        halfw = math.acos(arg)
        outer_n, oa0, ob0, oa1, ob1 = _arc_to_index_intervals(
            lon0_2pi, halfw, t0, t1, base, dlon, nx, need_shift
        )
        if outer_n == 0:
            return 0, 0, -1, 0, -1, 0, 0, -1, 0, -1

    # ---- INNER (exclude) ----
    inner_n, ia0, ib0, ia1, ib1 = 0, 0, -1, 0, -1
    if use_max_el:
        if abs(lat_row - phi0_gc) <= psi_inner:
            if abs(denom) < DENOM_TOL:
                if slat * sphi >= cps_inner:
                    inner_n, ia0, ib0, ia1, ib1 = 1, 0, nx - 1, 0, -1
            else:
                arg_i = (cps_inner - slat * sphi) / denom
                if arg_i > 1.0:
                    arg_i = 1.0
                elif arg_i < -1.0:
                    arg_i = -1.0
                halfw_i = math.acos(arg_i)
                inner_n, ia0, ib0, ia1, ib1 = _arc_to_index_intervals(
                    lon0_2pi, halfw_i, t0, t1, base, dlon, nx, need_shift
                )

    return outer_n, oa0, ob0, oa1, ob1, inner_n, ia0, ib0, ia1, ib1
