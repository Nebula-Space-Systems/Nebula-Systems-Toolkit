# _revisit_time.py
from __future__ import annotations

from typing import List, Literal

import numpy as np
from numba import njit, prange

from nstk.coverage.fixed_dt.config import CoverageConfig
from nstk.coverage.fixed_dt._empirical_core import (
    coverage_row_intervals_empirical_1row,
    _segments_minus_inners,
    _diff_add_seg,
    _center_index_from_lon0,
)


# Compute option enum (internal, for speed inside njit)
_OPT_AVERAGE = 0
_OPT_MAXIMUM = 1
_OPT_MINIMUM = 2
_OPT_STD_DEV = 3


@njit(cache=True, parallel=True)
def _revisit_time_empirical_kernel(
    time: np.ndarray,  # (nt,)
    obs_stack: np.ndarray,  # (nt, n_obs, 3)
    N_req: int,  # minimum concurrent assets required to be "in access"
    include_end_gaps: bool,  # STK "End Gaps": Include vs Ignore
    option: int,  # _OPT_*
    out: np.ndarray,  # (ny, nx) float64
    diff_grid: np.ndarray,  # (ny, nx+1) int32 (scratch)
    prev_access: np.ndarray,  # (ny, nx) uint8 (in/out)
    in_gap: np.ndarray,  # (ny, nx) uint8 (in/out)
    gap_started_at_t0: np.ndarray,  # (ny, nx) uint8 (in/out)
    gap_len: np.ndarray,  # (ny, nx) float64 (in/out)
    has_access: np.ndarray,  # (ny, nx) uint8 (in/out)
    gap_count: np.ndarray,  # (ny, nx) int32 (in/out)
    gap_sum: np.ndarray,  # (ny, nx) float64 (in/out)
    gap_sumsq: np.ndarray,  # (ny, nx) float64 (in/out)
    gap_min: np.ndarray,  # (ny, nx) float64 (in/out)
    gap_max: np.ndarray,  # (ny, nx) float64 (in/out)
    lat_gc_rows: np.ndarray,  # (ny,)
    lon_start_2pi: float,
    lon_span: float,
    dlon: float,
    base: float,
    nx: int,
    min_el_rad: float,
    max_el_rad: float,
    Ncos_row_m: np.ndarray,  # (ny,)
    Nz_row_m: np.ndarray,  # (ny,)
    cos_lat_row: np.ndarray,  # (ny,)
    sin_lat_row: np.ndarray,  # (ny,)
    cos_lon_col: np.ndarray,  # (nx,)
    sin_lon_col: np.ndarray,  # (nx,)
) -> None:
    nt = time.size
    ny = lat_gc_rows.size
    n_obs = obs_stack.shape[1]

    t_start = time[0]
    t_end = time[nt - 1]
    T = t_end - t_start
    if T <= 0.0:
        for j in prange(ny):
            for i in range(nx):
                out[j, i] = np.nan
        return

    two_pi = 2.0 * np.pi
    half_pi = 0.5 * np.pi
    dom0 = lon_start_2pi
    dom1 = lon_start_2pi + lon_span

    is_global = lon_span >= (two_pi - 1e-12)
    use_max_el = max_el_rad < (half_pi - 1e-12)

    sin_min = np.sin(min_el_rad)
    sin2_min = sin_min * sin_min
    sin_max = np.sin(max_el_rad)
    sin2_max = sin_max * sin_max

    # reused per timestep
    i_center_k = np.empty(n_obs, dtype=np.int32)

    # Sweep segments [time[ti], time[ti+1]) like access_duration
    for ti in range(nt - 1):
        dt = time[ti + 1] - time[ti]
        if dt <= 0.0:
            continue

        # precompute i_center per observer for this segment (serial)
        for k in range(n_obs):
            oy = obs_stack[ti, k, 1]
            ox = obs_stack[ti, k, 0]
            lon0 = np.atan2(oy, ox)  # [-pi, pi]
            lon0_2pi = lon0 + np.pi
            if lon0_2pi < 0.0:
                lon0_2pi += two_pi
            if lon0_2pi >= two_pi:
                lon0_2pi -= two_pi

            i_center_k[k] = _center_index_from_lon0(
                lon0_2pi, dom0, dom1, base, dlon, nx
            )

        # parallel over rows
        for j in prange(ny):
            diff_row = diff_grid[j]

            # zero diff row (nx+1)
            for ii in range(nx + 1):
                diff_row[ii] = 0

            # build diff_row from all observers' visible intervals on this row
            for k in range(n_obs):
                ox = obs_stack[ti, k, 0]
                oy = obs_stack[ti, k, 1]
                oz = obs_stack[ti, k, 2]
                ic = i_center_k[k]

                outer_n, oa0, ob0, oa1, ob1, inner_n, ia0, ib0, ia1, ib1 = (
                    coverage_row_intervals_empirical_1row(
                        ox,
                        oy,
                        oz,
                        j,
                        ic,
                        is_global,
                        use_max_el,
                        sin2_min,
                        sin2_max,
                        nx,
                        Ncos_row_m,
                        Nz_row_m,
                        cos_lat_row,
                        sin_lat_row,
                        cos_lon_col,
                        sin_lon_col,
                    )
                )

                if outer_n == 0:
                    continue

                # outer segment 0
                nseg, a0, b0, a1s, b1s, a2s, b2s = _segments_minus_inners(
                    oa0, ob0, inner_n, ia0, ib0, ia1, ib1
                )
                if nseg >= 1:
                    _diff_add_seg(diff_row, a0, b0)
                if nseg >= 2:
                    _diff_add_seg(diff_row, a1s, b1s)
                if nseg == 3:
                    _diff_add_seg(diff_row, a2s, b2s)

                # outer segment 1 (if present)
                if outer_n == 2:
                    nseg, a0, b0, a1s, b1s, a2s, b2s = _segments_minus_inners(
                        oa1, ob1, inner_n, ia0, ib0, ia1, ib1
                    )
                    if nseg >= 1:
                        _diff_add_seg(diff_row, a0, b0)
                    if nseg >= 2:
                        _diff_add_seg(diff_row, a1s, b1s)
                    if nseg == 3:
                        _diff_add_seg(diff_row, a2s, b2s)

            # prefix-sum counts, update gap tracking and stats
            run = 0
            for i in range(nx):
                run += diff_row[i]
                access_now = run >= N_req

                if access_now:
                    has_access[j, i] = np.uint8(1)

                    # If we were in a gap, that gap ends at the start of this segment.
                    if in_gap[j, i] != 0:
                        # close gap
                        g = gap_len[j, i]
                        if g > 0.0:
                            # Apply End Gaps rule for the initial gap
                            if include_end_gaps or (gap_started_at_t0[j, i] == 0):
                                gap_count[j, i] += 1
                                gap_sum[j, i] += g
                                gap_sumsq[j, i] += g * g
                                if g < gap_min[j, i]:
                                    gap_min[j, i] = g
                                if g > gap_max[j, i]:
                                    gap_max[j, i] = g

                        # reset gap state
                        in_gap[j, i] = np.uint8(0)
                        gap_started_at_t0[j, i] = np.uint8(0)
                        gap_len[j, i] = 0.0

                    prev_access[j, i] = np.uint8(1)

                else:
                    # In a gap during this segment; accumulate dt.
                    if prev_access[j, i] != 0:
                        # access -> gap transition at the start of this segment
                        in_gap[j, i] = np.uint8(1)
                        gap_len[j, i] = dt
                        gap_started_at_t0[j, i] = np.uint8(0)
                    else:
                        if in_gap[j, i] == 0:
                            # gap at the beginning (only possible at ti==0)
                            in_gap[j, i] = np.uint8(1)
                            gap_len[j, i] = dt
                            gap_started_at_t0[j, i] = np.uint8(1)
                        else:
                            gap_len[j, i] += dt

                    prev_access[j, i] = np.uint8(0)

    # finalize trailing gaps and compute requested statistic
    for j in prange(ny):
        for i in range(nx):
            if has_access[j, i] == 0:
                # STK: if no accesses exist, revisit time is duration of coverage interval
                out[j, i] = T
                continue

            # close trailing gap if we ended in a gap
            if in_gap[j, i] != 0:
                g = gap_len[j, i]
                if g > 0.0 and include_end_gaps:
                    gap_count[j, i] += 1
                    gap_sum[j, i] += g
                    gap_sumsq[j, i] += g * g
                    if g < gap_min[j, i]:
                        gap_min[j, i] = g
                    if g > gap_max[j, i]:
                        gap_max[j, i] = g

            n = gap_count[j, i]
            if n <= 0:
                # no included gaps (e.g., continuous coverage, or only end gaps and end_gaps="Ignore")
                out[j, i] = 0.0
                continue

            if option == _OPT_AVERAGE:
                out[j, i] = gap_sum[j, i] / n
            elif option == _OPT_MAXIMUM:
                out[j, i] = gap_max[j, i]
            elif option == _OPT_MINIMUM:
                out[j, i] = gap_min[j, i]
            else:
                # Std Deviation (population)
                mu = gap_sum[j, i] / n
                v = (gap_sumsq[j, i] / n) - (mu * mu)
                if v < 0.0:
                    v = 0.0
                out[j, i] = np.sqrt(v)


def calculate_revisit_time(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: List[np.ndarray],
    N: int = 1,
    *,
    option: Literal["average", "maximum", "minimum", "std_deviation"] = "average",
    end_gaps: Literal["include", "ignore"] = "include",
) -> np.ndarray:
    """
    Revisit Time (STK-style): statistics of the durations of coverage gaps.

    STK definition (summary):
      - Revisit Time measures the intervals during which coverage is not provided ("gaps").
      - If no accesses exist, revisit time is reported as the duration of the coverage interval.
      - Min # Assets applies: access is defined as having at least N simultaneous assets.
      - End Gaps can be Included or Ignored in the gap-duration computations.

    This implementation matches your empirical interval-to-diff-row approach and uses the
    same segment interpretation as calculate_access_duration (segment ti uses observer
    positions at time[ti] and spans dt = time[ti+1]-time[ti]).

    Parameters
    ----------
    config : CoverageConfig
        Coverage grid/configuration.
    time : np.ndarray
        1D strictly-increasing time array (seconds), length >= 2.
    observer_positions : List[np.ndarray]
        List of observer ECEF position histories; each element has shape (len(time), 3).
    N : int
        Minimum number of concurrent visible assets required to be "in access".
    option : {"average","maximum","minimum","std_deviation"}
        Which STK static compute option to return (subset implemented here).
    end_gaps : {"include","ignore"}
        Whether to include gaps that touch the ends of the analysis interval.

    Returns
    -------
    np.ndarray
        Revisit Time grid, shape (nlats, nlons), float64 (seconds).
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")

    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing (all dt > 0)")

    if not observer_positions:
        raise ValueError("observer_positions must be a non-empty list")

    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer")

    obs_arrays: List[np.ndarray] = []
    for obs in observer_positions:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape != (time.size, 3):
            raise ValueError(
                "Each observer position array must have shape (len(time), 3)"
            )
        obs_arrays.append(arr)

    obs_stack = np.ascontiguousarray(np.stack(obs_arrays, axis=1))  # (nt, n_obs, 3)
    n_obs = int(obs_stack.shape[1])

    ny = int(config.nlats)
    nx = int(config.nlons)

    # If N > n_obs, access is impossible everywhere => "no accesses exist" => duration of interval.
    T = float(time[-1] - time[0])
    if N > n_obs:
        return np.full((ny, nx), T, dtype=np.float64)

    include_end_gaps = end_gaps == "include"

    if option == "average":
        opt = _OPT_AVERAGE
    elif option == "maximum":
        opt = _OPT_MAXIMUM
    elif option == "minimum":
        opt = _OPT_MINIMUM
    elif option == "std_deviation":
        opt = _OPT_STD_DEV
    else:
        raise ValueError(
            "option must be one of: average, maximum, minimum, std_deviation"
        )

    out = np.zeros((ny, nx), dtype=np.float64)

    # Scratch: diff grid per row (nx+1), signed
    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)

    # State + stats (all per-cell)
    prev_access = np.zeros((ny, nx), dtype=np.uint8)
    in_gap = np.zeros((ny, nx), dtype=np.uint8)
    gap_started_at_t0 = np.zeros((ny, nx), dtype=np.uint8)
    gap_len = np.zeros((ny, nx), dtype=np.float64)

    has_access = np.zeros((ny, nx), dtype=np.uint8)

    gap_count = np.zeros((ny, nx), dtype=np.int32)
    gap_sum = np.zeros((ny, nx), dtype=np.float64)
    gap_sumsq = np.zeros((ny, nx), dtype=np.float64)
    gap_min = np.full((ny, nx), 1.0e300, dtype=np.float64)
    gap_max = np.zeros((ny, nx), dtype=np.float64)

    _revisit_time_empirical_kernel(
        time,
        obs_stack,
        int(N),
        bool(include_end_gaps),
        int(opt),
        out,
        diff_grid,
        prev_access,
        in_gap,
        gap_started_at_t0,
        gap_len,
        has_access,
        gap_count,
        gap_sum,
        gap_sumsq,
        gap_min,
        gap_max,
        config.lat_row_gc_rad,
        float(config.lon_start_2pi_rad),
        float(config.lon_span_rad),
        float(config.dlon_rad),
        float(config.lon_base_rad),
        int(config.nlons),
        float(config.min_el_eff_rad),
        float(config.max_el_eff_rad),
        config.Ncos_row_m,
        config.Nz_row_m,
        config.cos_lat_row_geod,
        config.sin_lat_row_geod,
        config.cos_lon_col,
        config.sin_lon_col,
    )

    return out
