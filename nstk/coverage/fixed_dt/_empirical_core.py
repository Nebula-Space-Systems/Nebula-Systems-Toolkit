# _empirical_core.py

import numpy as np
from numba import njit, prange


# ------------------------------------------------------------
# Empirical elevation predicate (no sqrt, just dot products)
#   elev >= el  <=>  (v·up) >= sin(el)*|v|
#   Use squared form with sign check:
#       (v·up) > 0  and  (v·up)^2 >= sin(el)^2 * (v·v)
# ------------------------------------------------------------
@njit(cache=True, inline="always")
def _passes_elev_sin2(
    ox: float,
    oy: float,
    oz: float,
    gx: float,
    gy: float,
    gz: float,
    ux: float,
    uy: float,
    uz: float,
    sin2_el: float,
) -> bool:
    dx = ox - gx
    dy = oy - gy
    dz = oz - gz
    du = dx * ux + dy * uy + dz * uz
    if du < 0.0:
        return False
    v2 = dx * dx + dy * dy + dz * dz
    return (du * du) > (sin2_el * v2)


# ------------------------------------------------------------
# per-cell predicate computed from compact 1D tables
#   instead of loading (ny,nx,3) arrays.
# ------------------------------------------------------------
@njit(cache=True, inline="always")
def _passes_elev_sin2_cell(
    ox: float,
    oy: float,
    oz: float,
    Ncos_row_m: np.ndarray,  # (ny,)
    Nz_row_m: np.ndarray,  # (ny,)
    cos_lat_row: np.ndarray,  # (ny,)
    sin_lat_row: np.ndarray,  # (ny,)
    cos_lon_col: np.ndarray,  # (nx,)
    sin_lon_col: np.ndarray,  # (nx,)
    j: int,
    i: int,
    sin2_el: float,
) -> bool:
    clon = cos_lon_col[i]
    slon = sin_lon_col[i]

    # ground ECEF at height=0
    gx = Ncos_row_m[j] * clon
    gy = Ncos_row_m[j] * slon
    gz = Nz_row_m[j]

    # Up unit vector in ECEF
    ux = cos_lat_row[j] * clon
    uy = cos_lat_row[j] * slon
    uz = sin_lat_row[j]

    return _passes_elev_sin2(ox, oy, oz, gx, gy, gz, ux, uy, uz, sin2_el)


# ------------------------------------------------------------
# Stamp set/clear primitives
# ------------------------------------------------------------
@njit(cache=True, inline="always")
def _stamp_run_minus_inners(
    stamp: np.ndarray,
    j: int,
    L: int,
    R: int,
    epoch,
    inner_n: int,
    ia0: int,
    ib0: int,
    ia1: int,
    ib1: int,
):
    """
    Stamp [L,R] as visible, excluding up to two inner intervals [ia0,ib0] and [ia1,ib1].
    All indices are inclusive. inner intervals must be sorted by start.
    """
    if L > R:
        return

    start = L

    if inner_n >= 1:
        a = ia0
        b = ib0
        if not (b < start or a > R):
            # fully covered
            if a <= start and b >= R:
                return
            # left visible chunk
            if a > start:
                stamp[j, start:a] = epoch  # [start, a-1]
            start = b + 1
            if start > R:
                return

    if inner_n == 2:
        a = ia1
        b = ib1
        if not (b < start or a > R):
            if a <= start and b >= R:
                return
            if a > start:
                stamp[j, start:a] = epoch
            start = b + 1
            if start > R:
                return

    stamp[j, start : R + 1] = epoch


# @njit(cache=True, inline="always")
# def _center_index_from_lon0(
#     lon0_2pi: float, t0: float, t1: float, base: float, dlon: float, nx: int
# ) -> int:
#     """
#     Map lon0_2pi (in [0,2π)) into the continuous [t0,t1] domain, then to the nearest grid-center index.
#     Works for dateline-crossing extents where t1 can exceed 2π.
#     """
#     lonc = lon0_2pi
#     # Shift lonc into the same continuous band as [t0,t1]
#     if lonc < t0:
#         lonc += 2.0 * np.pi
#     if lonc > t1:
#         lonc -= 2.0 * np.pi

#     # nearest center index
#     i = int(np.floor((lonc - base) / dlon + 0.5))
#     if i < 0:
#         i = 0
#     elif i > nx - 1:
#         i = nx - 1
#     return i


@njit(cache=True, inline="always")
def _center_index_from_lon0(
    lon0_2pi: float, t0: float, t1: float, base: float, dlon: float, nx: int
) -> int:
    two_pi = 2.0 * np.pi

    # Consider three equivalent representations and choose the one closest to [t0, t1]
    lonc0 = lon0_2pi
    lonc1 = lon0_2pi + two_pi
    lonc2 = lon0_2pi - two_pi

    # distance to interval [t0, t1]
    def dist_to_interval(x: float) -> float:
        if x < t0:
            return t0 - x
        if x > t1:
            return x - t1
        return 0.0

    d0 = dist_to_interval(lonc0)
    d1 = dist_to_interval(lonc1)
    d2 = dist_to_interval(lonc2)

    lonc = lonc0
    best = d0
    if d1 < best:
        lonc = lonc1
        best = d1
    if d2 < best:
        lonc = lonc2
        best = d2

    # If still outside, clamp to nearest boundary so we seed at the nearest edge
    if lonc < t0:
        lonc = t0
    elif lonc > t1:
        lonc = t1

    # nearest center index
    i = int(np.floor((lonc - base) / dlon + 0.5))
    if i < 0:
        i = 0
    elif i > nx - 1:
        i = nx - 1
    return i


@njit(cache=True, inline="always")
def _wrap_i_1(i: int, nx: int) -> int:
    if i >= nx:
        return i - nx
    if i < 0:
        return i + nx
    return i


@njit(cache=True, inline="always")
def _find_true_near_index(
    j: int,
    i_center: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
) -> int:
    """
    Find a True seed at/near i_center using a small neighborhood search.
    Returns -1 if none found.
    """
    if _passes_elev_sin2_cell(
        ox,
        oy,
        oz,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        j,
        i_center,
        sin2_el,
    ):
        return i_center

    K = 2
    for k in range(1, K + 1):
        il = i_center - k
        if il >= 0:
            if _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                il,
                sin2_el,
            ):
                return il
        ir = i_center + k
        if ir < nx:
            if _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                ir,
                sin2_el,
            ):
                return ir

    return -1


@njit(cache=True)
def _first_true_in_range(
    j: int,
    lo: int,
    hi: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
) -> int:
    if lo > hi:
        return -1

    # if hi is False, there is no True in [lo,hi] (assumes monotonic True block at the right end)
    if (
        _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            hi,
            sin2_el,
        )
        == False
    ):
        return -1

    if _passes_elev_sin2_cell(
        ox,
        oy,
        oz,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        j,
        lo,
        sin2_el,
    ):
        return lo

    l = lo
    r = hi
    while l < r:
        m = (l + r) // 2
        if _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            m,
            sin2_el,
        ):
            r = m
        else:
            l = m + 1
    return l


@njit(cache=True)
def _last_true_in_range(
    j: int,
    lo: int,
    hi: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
) -> int:
    if lo > hi:
        return -1

    # if lo is False, there is no True in [lo,hi] (assumes monotonic True block at the left end)
    if (
        _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            lo,
            sin2_el,
        )
        == False
    ):
        return -1

    if _passes_elev_sin2_cell(
        ox,
        oy,
        oz,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        j,
        hi,
        sin2_el,
    ):
        return hi

    l = lo
    r = hi
    while l < r:
        m = (l + r + 1) // 2
        if _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            m,
            sin2_el,
        ):
            l = m
        else:
            r = m - 1
    return l


@njit(cache=True)
def _true_span_around_seed(
    j: int,
    seed: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
):
    # seed must satisfy predicate == True

    # ---- find left boundary (first True) ----
    if seed == 0:
        left = 0
    else:
        step = 1
        last_true = seed
        while True:
            cand = seed - step
            if cand < 0:
                if _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    0,
                    sin2_el,
                ):
                    left = 0
                else:
                    left = _first_true_in_range(
                        j,
                        0,
                        last_true,
                        Ncos_row_m,
                        Nz_row_m,
                        cos_lat_row,
                        sin_lat_row,
                        cos_lon_col,
                        sin_lon_col,
                        ox,
                        oy,
                        oz,
                        sin2_el,
                    )
                break

            if not _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                cand,
                sin2_el,
            ):
                left = _first_true_in_range(
                    j,
                    cand + 1,
                    last_true,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    ox,
                    oy,
                    oz,
                    sin2_el,
                )
                break

            last_true = cand
            step <<= 1

    # ---- find right boundary (last True) ----
    if seed == nx - 1:
        right = nx - 1
    else:
        step = 1
        last_true = seed
        while True:
            cand = seed + step
            if cand >= nx:
                if _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    nx - 1,
                    sin2_el,
                ):
                    right = nx - 1
                else:
                    right = _last_true_in_range(
                        j,
                        last_true,
                        nx - 1,
                        Ncos_row_m,
                        Nz_row_m,
                        cos_lat_row,
                        sin_lat_row,
                        cos_lon_col,
                        sin_lon_col,
                        ox,
                        oy,
                        oz,
                        sin2_el,
                    )
                break

            if not _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                cand,
                sin2_el,
            ):
                right = _last_true_in_range(
                    j,
                    last_true,
                    cand - 1,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    ox,
                    oy,
                    oz,
                    sin2_el,
                )
                break

            last_true = cand
            step <<= 1

    return left, right


@njit(cache=True)
def _last_true_offset_right(
    j: int,
    seed: int,
    max_off: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
) -> int:
    if max_off <= 0:
        return 0

    last_true = 0
    step = 1

    while True:
        s = step if step <= max_off else max_off
        ii = _wrap_i_1(seed + s, nx)

        if not _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            ii,
            sin2_el,
        ):
            lo = last_true
            hi = s - 1
            if hi < lo:
                return last_true
            while lo < hi:
                mid = (lo + hi + 1) // 2
                iim = _wrap_i_1(seed + mid, nx)
                if _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    iim,
                    sin2_el,
                ):
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        if s == max_off:
            return max_off

        last_true = s
        step <<= 1


@njit(cache=True)
def _last_true_offset_left(
    j: int,
    seed: int,
    max_off: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
) -> int:
    if max_off <= 0:
        return 0

    last_true = 0
    step = 1

    while True:
        s = step if step <= max_off else max_off
        ii = _wrap_i_1(seed - s, nx)

        if not _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            ii,
            sin2_el,
        ):
            lo = last_true
            hi = s - 1
            if hi < lo:
                return last_true
            while lo < hi:
                mid = (lo + hi + 1) // 2
                iim = _wrap_i_1(seed - mid, nx)
                if _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    iim,
                    sin2_el,
                ):
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        if s == max_off:
            return max_off

        last_true = s
        step <<= 1


@njit(cache=True)
def _last_false_offset_right(
    j: int,
    seed_false: int,
    max_off: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
) -> int:
    if max_off <= 0:
        return 0

    last_false = 0
    step = 1

    while True:
        s = step if step <= max_off else max_off
        ii = _wrap_i_1(seed_false + s, nx)

        if _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            ii,
            sin2_el,
        ):
            lo = last_false
            hi = s - 1
            if hi < lo:
                return last_false
            while lo < hi:
                mid = (lo + hi + 1) // 2
                iim = _wrap_i_1(seed_false + mid, nx)
                if not _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    iim,
                    sin2_el,
                ):
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        if s == max_off:
            return max_off

        last_false = s
        step <<= 1


@njit(cache=True)
def _last_false_offset_left(
    j: int,
    seed_false: int,
    max_off: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
) -> int:
    if max_off <= 0:
        return 0

    last_false = 0
    step = 1

    while True:
        s = step if step <= max_off else max_off
        ii = _wrap_i_1(seed_false - s, nx)

        if _passes_elev_sin2_cell(
            ox,
            oy,
            oz,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            j,
            ii,
            sin2_el,
        ):
            lo = last_false
            hi = s - 1
            if hi < lo:
                return last_false
            while lo < hi:
                mid = (lo + hi + 1) // 2
                iim = _wrap_i_1(seed_false - mid, nx)
                if not _passes_elev_sin2_cell(
                    ox,
                    oy,
                    oz,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    j,
                    iim,
                    sin2_el,
                ):
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        if s == max_off:
            return max_off

        last_false = s
        step <<= 1


@njit(cache=True)
def _true_arc_global(
    j: int,
    seed: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
    ox: float,
    oy: float,
    oz: float,
    sin2_el: float,
    nx: int,
):
    """
    Returns (n, a0,b0,a1,b1) with n in {1,2} representing True indices on a circular row.
    This handles BOTH cases:
      - True arc <= 180°
      - True arc >  180° (find the False gap near the antipode and take complement)
    """
    half = nx // 2

    r_off = _last_true_offset_right(
        j,
        seed,
        half,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        ox,
        oy,
        oz,
        sin2_el,
        nx,
    )
    l_off = _last_true_offset_left(
        j,
        seed,
        half,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        ox,
        oy,
        oz,
        sin2_el,
        nx,
    )

    # If True stays True out to half-circle both ways, this is either:
    #   (A) full row True, or
    #   (B) True arc > 180° with a narrow False gap near the antipode.
    if r_off == half and l_off == half:
        anti = _wrap_i_1(seed + half, nx)

        false_seed = -1
        K = 16
        for k in range(K + 1):
            ii = _wrap_i_1(anti + k, nx)
            if not _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                ii,
                sin2_el,
            ):
                false_seed = ii
                break
            ii = _wrap_i_1(anti - k, nx)
            if not _passes_elev_sin2_cell(
                ox,
                oy,
                oz,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                j,
                ii,
                sin2_el,
            ):
                false_seed = ii
                break

        if false_seed < 0:
            return 1, 0, nx - 1, 0, 0

        rf_off = _last_false_offset_right(
            j,
            false_seed,
            half,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_el,
            nx,
        )
        lf_off = _last_false_offset_left(
            j,
            false_seed,
            half,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_el,
            nx,
        )

        f_left = _wrap_i_1(false_seed - lf_off, nx)
        f_right = _wrap_i_1(false_seed + rf_off, nx)

        if f_left <= f_right:
            # False = [f_left, f_right]
            if f_left == 0 and f_right == nx - 1:
                return 1, 0, -1, 0, 0
            if f_left == 0:
                return 1, f_right + 1, nx - 1, 0, 0
            if f_right == nx - 1:
                return 1, 0, f_left - 1, 0, 0
            return 2, 0, f_left - 1, f_right + 1, nx - 1
        else:
            # False wraps: [0, f_right] U [f_left, nx-1]
            # True is the middle
            return 1, f_right + 1, f_left - 1, 0, 0

    # Normal case: True arc <= 180° from seed
    left = _wrap_i_1(seed - l_off, nx)
    right = _wrap_i_1(seed + r_off, nx)

    if left <= right:
        return 1, left, right, 0, 0
    else:
        return 2, 0, right, left, nx - 1


# ------------------------------------------------------------
# Main coverage stamping with refining kernel
# ------------------------------------------------------------
@njit(cache=True, parallel=True)
def coverage_stamp_kernel_empirical(
    ox: float,  # observer ECEF x meters
    oy: float,  # observer ECEF y meters
    oz: float,  # observer ECEF z meters
    lat_gc_rows: np.ndarray,  # (ny,)
    lon_start_2pi: float,
    lon_span: float,
    dlon: float,
    base: float,
    nx: int,
    min_el_rad: float,
    max_el_rad: float,
    stamp: np.ndarray,
    epoch,
    Ncos_row_m: np.ndarray,  # (ny,)
    Nz_row_m: np.ndarray,  # (ny,)
    cos_lat_row: np.ndarray,  # (ny,)
    sin_lat_row: np.ndarray,  # (ny,)
    cos_lon_col: np.ndarray,  # (nx,)
    sin_lon_col: np.ndarray,  # (nx,)
):
    """
    Compute and stamp visibility regions for observer location across longitude/latitude grid.
    This function determines which grid points are visible from an observer location
    in ECEF coordinates, subject to elevation angle constraints. It marks visible regions
    in a stamp array by accounting for both minimum elevation (outer boundary) and
    maximum elevation (inner exclusion) angles.
    Parameters
    ----------
    ox : float
        Observer ECEF x-coordinate in meters.
    oy : float
        Observer ECEF y-coordinate in meters.
    oz : float
        Observer ECEF z-coordinate in meters.
    lat_gc_rows : np.ndarray
        Geocentric latitudes of grid rows, shape (ny,).
    lon_start_2pi : float
        Starting longitude of domain in [0, 2π) radians.
    lon_span : float
        Longitude span in radians.
    dlon : float
        Longitude step size in radians.
    base : float
        Base offset for longitude indexing.
    nx : int
        Number of longitude columns in grid.
    min_el_rad : float
        Minimum elevation angle threshold in radians.
    max_el_rad : float
        Maximum elevation angle threshold in radians.
    stamp : np.ndarray
        Output array to mark visible grid points. Modified in-place.
    epoch : float
        Time epoch for visibility computation.
    Ncos_row_m : np.ndarray
        Precomputed cosine factors per row, shape (ny,).
    Nz_row_m : np.ndarray
        Precomputed z-component factors per row, shape (ny,).
    cos_lat_row : np.ndarray
        Cosine of geocentric latitude per row, shape (ny,).
    sin_lat_row : np.ndarray
        Sine of geocentric latitude per row, shape (ny,).
    cos_lon_col : np.ndarray
        Cosine of longitude per column, shape (nx,).
    sin_lon_col : np.ndarray
        Sine of longitude per column, shape (nx,).
    Returns
    -------
    np.ndarray
        The modified stamp array with visible regions marked.
    Notes
    -----
    - If max_el_rad <= min_el_rad, no regions are marked as visible (early exit).
    - Handles both global (lon_span ≈ 2π) and regional longitude domains.
    - Computes outer visible spans (elevation >= min_el_rad) and inner exclusion
      spans (elevation >= max_el_rad) per row, then marks the set difference.
    - Uses fast nearest-neighbor seeding for span detection to optimize performance.
    """
    ny = np.size(lat_gc_rows)

    # empirical thresholds (once)
    sin_min = np.sin(min_el_rad)
    sin2_min = sin_min * sin_min
    sin_max = np.sin(max_el_rad)
    sin2_max = sin_max * sin_max

    # target lon domain [t0, t1] in continuous coordinates
    t0 = lon_start_2pi
    t1 = lon_start_2pi + lon_span
    two_pi = 2.0 * np.pi

    # Early out: if max <= min, visible region is empty.
    if max_el_rad <= min_el_rad + 1e-15:
        return stamp

    # only need inner exclusion if max_el < 90deg
    half_pi = 0.5 * np.pi
    use_max_el = max_el_rad < (half_pi - 1e-12)

    # center lon in [0,2π)
    lon0 = np.atan2(oy, ox)  # [-pi, pi]
    lon0_2pi = lon0 + np.pi
    if lon0_2pi < 0.0:
        lon0_2pi += two_pi
    if lon0_2pi >= two_pi:
        lon0_2pi -= two_pi

    # index closest to observer lon within this lon-domain (or nearest edge)
    i_center = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

    # If the lon domain is effectively global, the seam can split an interval.
    # Otherwise, "end0 & end1 true" implies full-row, not wrap.
    is_global = lon_span >= (two_pi - 1e-12)

    for j in prange(ny):
        # -----------------------------
        # OUTER span(s): elev >= min_el
        # -----------------------------
        seed_min = _find_true_near_index(
            j,
            i_center,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_min,
            nx,
        )
        if seed_min < 0:
            continue

        outer_n = 1
        oa0 = 0
        ob0 = -1
        oa1 = 0
        ob1 = -1

        if is_global:
            outer_n, oa0, ob0, oa1, ob1 = _true_arc_global(
                j,
                seed_min,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_min,
                nx,
            )
        else:
            oa0, ob0 = _true_span_around_seed(
                j,
                seed_min,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_min,
                nx,
            )
            outer_n = 1
            oa1 = 0
            ob1 = -1

        # -----------------------------
        # INNER span(s): elev >= max_el   (to exclude)
        # -----------------------------
        inner_n = 0
        ia0 = 0
        ib0 = -1
        ia1 = 0
        ib1 = -1

        if use_max_el:
            seed_max = _find_true_near_index(
                j,
                i_center,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_max,
                nx,
            )
            if seed_max >= 0:
                inner_n = 1
                if is_global:
                    inner_n, ia0, ib0, ia1, ib1 = _true_arc_global(
                        j,
                        seed_max,
                        Ncos_row_m,
                        Nz_row_m,
                        cos_lat_row,
                        sin_lat_row,
                        cos_lon_col,
                        sin_lon_col,
                        ox,
                        oy,
                        oz,
                        sin2_max,
                        nx,
                    )
                else:
                    ia0, ib0 = _true_span_around_seed(
                        j,
                        seed_max,
                        Ncos_row_m,
                        Nz_row_m,
                        cos_lat_row,
                        sin_lat_row,
                        cos_lon_col,
                        sin_lon_col,
                        ox,
                        oy,
                        oz,
                        sin2_max,
                        nx,
                    )
                    inner_n = 1
                    ia1 = 0
                    ib1 = -1

        # -----------------------------
        # Stamp ONLY visible: (outer) minus (inner)
        # -----------------------------
        if outer_n == 1:
            _stamp_run_minus_inners(
                stamp, j, oa0, ob0, epoch, inner_n, ia0, ib0, ia1, ib1
            )
        else:
            _stamp_run_minus_inners(
                stamp, j, oa0, ob0, epoch, inner_n, ia0, ib0, ia1, ib1
            )
            _stamp_run_minus_inners(
                stamp, j, oa1, ob1, epoch, inner_n, ia0, ib0, ia1, ib1
            )

    return stamp


@njit(cache=True, parallel=True)
def coverage_row_intervals_empirical(
    ox: float,
    oy: float,
    oz: float,
    lat_gc_rows: np.ndarray,  # (ny,) only used for ny sizing / signature consistency
    lon_start_2pi: float,
    lon_span: float,
    dlon: float,
    base: float,
    nx: int,
    min_el_rad: float,
    max_el_rad: float,
    # refined geometry tables
    Ncos_row_m: np.ndarray,  # (ny,)
    Nz_row_m: np.ndarray,  # (ny,)
    cos_lat_row: np.ndarray,  # (ny,)
    sin_lat_row: np.ndarray,  # (ny,)
    cos_lon_col: np.ndarray,  # (nx,)
    sin_lon_col: np.ndarray,  # (nx,)
    # outputs (preallocated by caller)
    outer_n: np.ndarray,  # (ny,) uint8
    oa0: np.ndarray,  # (ny,) int32
    ob0: np.ndarray,  # (ny,) int32
    oa1: np.ndarray,  # (ny,) int32
    ob1: np.ndarray,  # (ny,) int32
    inner_n: np.ndarray,  # (ny,) uint8
    ia0: np.ndarray,  # (ny,) int32
    ib0: np.ndarray,  # (ny,) int32
    ia1: np.ndarray,  # (ny,) int32
    ib1: np.ndarray,  # (ny,) int32
) -> None:
    ny = lat_gc_rows.size

    # empirical thresholds (once)
    sin_min = np.sin(min_el_rad)
    sin2_min = sin_min * sin_min
    sin_max = np.sin(max_el_rad)
    sin2_max = sin_max * sin_max

    # target lon domain [t0, t1] in continuous coordinates
    t0 = lon_start_2pi
    t1 = lon_start_2pi + lon_span
    two_pi = 2.0 * np.pi

    # Early out: if max <= min, visible region is empty (consistent with your stamp kernel)
    if max_el_rad <= min_el_rad + 1e-15:
        # zero all outputs
        for j in prange(ny):
            outer_n[j] = 0
            inner_n[j] = 0
            oa0[j] = 0
            ob0[j] = -1
            oa1[j] = 0
            ob1[j] = -1
            ia0[j] = 0
            ib0[j] = -1
            ia1[j] = 0
            ib1[j] = -1
        return

    # only need inner exclusion if max_el < 90deg
    half_pi = 0.5 * np.pi
    use_max_el = max_el_rad < (half_pi - 1e-12)

    # center lon in [0,2π)
    lon0 = np.atan2(oy, ox)  # [-pi, pi]
    lon0_2pi = lon0 + np.pi
    if lon0_2pi < 0.0:
        lon0_2pi += two_pi
    if lon0_2pi >= two_pi:
        lon0_2pi -= two_pi

    # index closest to observer lon within this lon-domain (or nearest edge)
    i_center = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

    # If the lon domain is effectively global, the seam can split an interval.
    # Otherwise, "end0 & end1 true" implies full-row, not wrap.
    is_global = lon_span >= (two_pi - 1e-12)

    # init outputs to empty
    for j in prange(ny):
        outer_n[j] = 0
        inner_n[j] = 0
        oa0[j] = 0
        ob0[j] = -1
        oa1[j] = 0
        ob1[j] = -1
        ia0[j] = 0
        ib0[j] = -1
        ia1[j] = 0
        ib1[j] = -1

    for j in prange(ny):
        # -----------------------------
        # OUTER span(s): elev >= min_el
        # -----------------------------
        seed_min = _find_true_near_index(
            j,
            i_center,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_min,
            nx,
        )
        if seed_min < 0:
            continue

        if is_global:
            n, a0, b0, a1_, b1_ = _true_arc_global(
                j,
                seed_min,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_min,
                nx,
            )
            outer_n[j] = np.uint8(n)
            oa0[j] = a0
            ob0[j] = b0
            oa1[j] = a1_
            ob1[j] = b1_
        else:
            a0, b0 = _true_span_around_seed(
                j,
                seed_min,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_min,
                nx,
            )
            outer_n[j] = np.uint8(1)
            oa0[j] = a0
            ob0[j] = b0

        # -----------------------------
        # INNER span(s): elev >= max_el (to exclude)
        # -----------------------------
        if not use_max_el:
            continue

        seed_max = _find_true_near_index(
            j,
            i_center,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_max,
            nx,
        )
        if seed_max < 0:
            continue

        if is_global:
            n2, a0, b0, a1_, b1_ = _true_arc_global(
                j,
                seed_max,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_max,
                nx,
            )
            inner_n[j] = np.uint8(n2)
            ia0[j] = a0
            ib0[j] = b0
            ia1[j] = a1_
            ib1[j] = b1_
        else:
            a0, b0 = _true_span_around_seed(
                j,
                seed_max,
                Ncos_row_m,
                Nz_row_m,
                cos_lat_row,
                sin_lat_row,
                cos_lon_col,
                sin_lon_col,
                ox,
                oy,
                oz,
                sin2_max,
                nx,
            )
            inner_n[j] = np.uint8(1)
            ia0[j] = a0
            ib0[j] = b0


# ----------------------------
# Return intervals for ONE row j (outer + inner)
# ----------------------------
@njit(cache=True, inline="always")
def coverage_row_intervals_empirical_1row(
    ox: float,
    oy: float,
    oz: float,
    j: int,
    i_center: int,
    is_global: bool,
    use_max_el: bool,
    sin2_min: float,
    sin2_max: float,
    nx: int,
    Ncos_row_m: np.ndarray,
    Nz_row_m: np.ndarray,
    cos_lat_row: np.ndarray,
    sin_lat_row: np.ndarray,
    cos_lon_col: np.ndarray,
    sin_lon_col: np.ndarray,
):
    # OUTER
    seed_min = _find_true_near_index(
        j,
        i_center,
        Ncos_row_m,
        Nz_row_m,
        cos_lat_row,
        sin_lat_row,
        cos_lon_col,
        sin_lon_col,
        ox,
        oy,
        oz,
        sin2_min,
        nx,
    )
    if seed_min < 0:
        # outer_n, oa0,ob0,oa1,ob1, inner_n, ia0,ib0,ia1,ib1
        return 0, 0, -1, 0, -1, 0, 0, -1, 0, -1

    if is_global:
        outer_n, oa0, ob0, oa1, ob1 = _true_arc_global(
            j,
            seed_min,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_min,
            nx,
        )
    else:
        oa0, ob0 = _true_span_around_seed(
            j,
            seed_min,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_min,
            nx,
        )
        outer_n, oa1, ob1 = 1, 0, -1

    # INNER (optional)
    inner_n, ia0, ib0, ia1, ib1 = 0, 0, -1, 0, -1
    if use_max_el:
        seed_max = _find_true_near_index(
            j,
            i_center,
            Ncos_row_m,
            Nz_row_m,
            cos_lat_row,
            sin_lat_row,
            cos_lon_col,
            sin_lon_col,
            ox,
            oy,
            oz,
            sin2_max,
            nx,
        )
        if seed_max >= 0:
            if is_global:
                inner_n, ia0, ib0, ia1, ib1 = _true_arc_global(
                    j,
                    seed_max,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    ox,
                    oy,
                    oz,
                    sin2_max,
                    nx,
                )
            else:
                ia0, ib0 = _true_span_around_seed(
                    j,
                    seed_max,
                    Ncos_row_m,
                    Nz_row_m,
                    cos_lat_row,
                    sin_lat_row,
                    cos_lon_col,
                    sin_lon_col,
                    ox,
                    oy,
                    oz,
                    sin2_max,
                    nx,
                )
                inner_n, ia1, ib1 = 1, 0, -1

    return outer_n, oa0, ob0, oa1, ob1, inner_n, ia0, ib0, ia1, ib1


# ----------------------------
# Convert (outer interval) minus (up to two inner intervals) into up to 3 segments (inclusive)
# This matches your _stamp_run_minus_inners semantics, but without touching cells.
# ----------------------------
@njit(cache=True, inline="always")
def _segments_minus_inners(
    L: int,
    R: int,
    inner_n: int,
    ia0: int,
    ib0: int,
    ia1: int,
    ib1: int,
):
    n = 0
    a0 = 0
    b0 = -1
    a1 = 0
    b1 = -1
    a2 = 0
    b2 = -1

    if L > R:
        return n, a0, b0, a1, b1, a2, b2

    start = L

    if inner_n >= 1:
        a = ia0
        b = ib0
        if not (b < start or a > R):
            if a <= start and b >= R:
                return 0, a0, b0, a1, b1, a2, b2
            if a > start:
                a0 = start
                b0 = a - 1
                n = 1
            start = b + 1
            if start > R:
                return n, a0, b0, a1, b1, a2, b2

    if inner_n == 2:
        a = ia1
        b = ib1
        if not (b < start or a > R):
            if a <= start and b >= R:
                return 0, a0, b0, a1, b1, a2, b2
            if a > start:
                if n == 0:
                    a0 = start
                    b0 = a - 1
                    n = 1
                else:
                    a1 = start
                    b1 = a - 1
                    n = 2
            start = b + 1
            if start > R:
                return n, a0, b0, a1, b1, a2, b2

    # final tail
    if start <= R:
        if n == 0:
            a0 = start
            b0 = R
            n = 1
        elif n == 1:
            a1 = start
            b1 = R
            n = 2
        else:
            a2 = start
            b2 = R
            n = 3

    return n, a0, b0, a1, b1, a2, b2


@njit(cache=True, inline="always")
def _diff_add_seg(diff_row: np.ndarray, a: int, b: int) -> None:
    if a <= b:
        diff_row[a] += 1
        diff_row[b + 1] -= 1  # diff_row is length nx+1, so b+1 is valid even if b==nx-1
