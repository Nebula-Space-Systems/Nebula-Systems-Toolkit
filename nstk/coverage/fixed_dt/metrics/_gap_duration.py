# _gap_durations.py
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from numba import njit, prange

from nstk.coverage.fixed_dt.config import CoverageConfig
from nstk.coverage.fixed_dt._empirical_core import (
    coverage_row_intervals_empirical_1row,
    _segments_minus_inners,
    _diff_add_seg,
    _center_index_from_lon0,
)


@njit(cache=True, parallel=True)
def _gap_durations_empirical_kernel(
    time: np.ndarray,  # (nt,)
    obs_stack: np.ndarray,  # (nt, n_obs, 3)
    min_assets: int,
    include_end_gaps: int,  # 0/1 (numba-friendly)
    # outputs (in/out)
    prev_access: np.ndarray,  # (ny,nx) uint8
    gap_start: np.ndarray,  # (ny,nx) float64 (valid when in gap, else -1)
    any_access: np.ndarray,  # (ny,nx) uint8
    gap_count: np.ndarray,  # (ny,nx) int32
    gap_sum: np.ndarray,  # (ny,nx) float64
    gap_sumsq: np.ndarray,  # (ny,nx) float64
    gap_min: np.ndarray,  # (ny,nx) float64
    gap_max: np.ndarray,  # (ny,nx) float64
    # scratch
    diff_grid: np.ndarray,  # (ny,nx+1) int32
    # geometry + config scalars
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

    two_pi = 2.0 * np.pi
    half_pi = 0.5 * np.pi
    t0 = lon_start_2pi
    t1 = lon_start_2pi + lon_span

    is_global = lon_span >= (two_pi - 1e-12)
    use_max_el = max_el_rad < (half_pi - 1e-12)

    sin_min = np.sin(min_el_rad)
    sin2_min = sin_min * sin_min
    sin_max = np.sin(max_el_rad)
    sin2_max = sin_max * sin_max

    # reused per sample
    i_center_k = np.empty(n_obs, dtype=np.int32)

    # -------------------------------
    # initialize at sample ti=0
    # -------------------------------
    ti0 = 0
    for k in range(n_obs):
        oy = obs_stack[ti0, k, 1]
        ox = obs_stack[ti0, k, 0]
        lon0 = np.atan2(oy, ox)  # [-pi,pi]
        lon0_2pi = lon0 + np.pi
        if lon0_2pi < 0.0:
            lon0_2pi += two_pi
        if lon0_2pi >= two_pi:
            lon0_2pi -= two_pi
        i_center_k[k] = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

    for j in prange(ny):
        diff_row = diff_grid[j]
        for ii in range(nx + 1):
            diff_row[ii] = 0

        for k in range(n_obs):
            ox = obs_stack[ti0, k, 0]
            oy = obs_stack[ti0, k, 1]
            oz = obs_stack[ti0, k, 2]
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

            nseg, a0, b0, a1s, b1s, a2s, b2s = _segments_minus_inners(
                oa0, ob0, inner_n, ia0, ib0, ia1, ib1
            )
            if nseg >= 1:
                _diff_add_seg(diff_row, a0, b0)
            if nseg >= 2:
                _diff_add_seg(diff_row, a1s, b1s)
            if nseg == 3:
                _diff_add_seg(diff_row, a2s, b2s)

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

        run = 0
        prev_row = prev_access[j]
        gs_row = gap_start[j]
        any_row = any_access[j]

        t_init = time[0]
        for i in range(nx):
            run += diff_row[i]
            cur = 1 if run >= min_assets else 0
            prev_row[i] = np.uint8(cur)

            if cur == 1:
                any_row[i] = np.uint8(1)
                gs_row[i] = -1.0
            else:
                if include_end_gaps == 1:
                    gs_row[i] = t_init
                else:
                    gs_row[i] = -1.0

    # -------------------------------
    # process transitions at time[ti] for ti=1..nt-2
    # state at sample ti applies to [time[ti], time[ti+1])
    # transition boundary is time[ti]
    # -------------------------------
    for ti in range(1, nt - 1):
        t_boundary = time[ti]

        # centers for this sample
        for k in range(n_obs):
            oy = obs_stack[ti, k, 1]
            ox = obs_stack[ti, k, 0]
            lon0 = np.atan2(oy, ox)
            lon0_2pi = lon0 + np.pi
            if lon0_2pi < 0.0:
                lon0_2pi += two_pi
            if lon0_2pi >= two_pi:
                lon0_2pi -= two_pi
            i_center_k[k] = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

        for j in prange(ny):
            diff_row = diff_grid[j]
            for ii in range(nx + 1):
                diff_row[ii] = 0

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

                nseg, a0, b0, a1s, b1s, a2s, b2s = _segments_minus_inners(
                    oa0, ob0, inner_n, ia0, ib0, ia1, ib1
                )
                if nseg >= 1:
                    _diff_add_seg(diff_row, a0, b0)
                if nseg >= 2:
                    _diff_add_seg(diff_row, a1s, b1s)
                if nseg == 3:
                    _diff_add_seg(diff_row, a2s, b2s)

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

            run = 0
            prev_row = prev_access[j]
            gs_row = gap_start[j]
            any_row = any_access[j]

            c_row = gap_count[j]
            s_row = gap_sum[j]
            ss_row = gap_sumsq[j]
            mn_row = gap_min[j]
            mx_row = gap_max[j]

            for i in range(nx):
                run += diff_row[i]
                cur = 1 if run >= min_assets else 0
                prev = int(prev_row[i])

                if cur == 1:
                    any_row[i] = np.uint8(1)

                # transition 1->0: start gap (only if end gaps are counted or we have a prior access)
                if prev == 1 and cur == 0:
                    if include_end_gaps == 1:
                        gs_row[i] = t_boundary
                    else:
                        # interior gaps: start on loss of access
                        gs_row[i] = t_boundary

                # transition 0->1: close gap if one is open
                elif prev == 0 and cur == 1:
                    t0g = gs_row[i]
                    if t0g >= 0.0:
                        d = t_boundary - t0g
                        if d < 0.0:
                            d = 0.0
                        c_row[i] += 1
                        s_row[i] += d
                        ss_row[i] += d * d
                        if d < mn_row[i]:
                            mn_row[i] = d
                        if d > mx_row[i]:
                            mx_row[i] = d
                    gs_row[i] = -1.0

                prev_row[i] = np.uint8(cur)

    # -------------------------------
    # trailing end gap
    # -------------------------------
    if include_end_gaps == 1:
        t_end = time[nt - 1]
        for j in prange(ny):
            prev_row = prev_access[j]
            gs_row = gap_start[j]

            c_row = gap_count[j]
            s_row = gap_sum[j]
            ss_row = gap_sumsq[j]
            mn_row = gap_min[j]
            mx_row = gap_max[j]

            for i in range(nx):
                if int(prev_row[i]) == 0:
                    t0g = gs_row[i]
                    if t0g >= 0.0:
                        d = t_end - t0g
                        if d < 0.0:
                            d = 0.0
                        c_row[i] += 1
                        s_row[i] += d
                        ss_row[i] += d * d
                        if d < mn_row[i]:
                            mn_row[i] = d
                        if d > mx_row[i]:
                            mx_row[i] = d


GapStat = Literal["mean", "min", "max", "std", "count", "sum"]


def calculate_gap_duration(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: List[np.ndarray],
    *,
    min_assets: int = 1,
    stat: GapStat = "mean",
    include_end_gaps: bool = True,
    no_access_value: float = np.nan,
    nan_if_never_access: bool = False,
) -> np.ndarray:
    """
    Gap Duration metric (STK-style concept): per grid cell, compute statistics of the
    no-access gaps between access intervals, for a required concurrency threshold
    (min_assets).

    Access state is evaluated at each provided sample index ti and treated as valid
    over [time[ti], time[ti+1]) (same convention as your access_duration kernel).

    Parameters
    ----------
    config : CoverageConfig
        Coverage grid + precomputed geometry tables.
    time : np.ndarray
        1D float seconds array, strictly increasing, length >= 2.
    observer_positions : list[np.ndarray]
        Each array is (nt, 3) ECEF meters, all share same nt.
    min_assets : int
        Required number of simultaneous visible assets to be considered "in access".
    stat : {"mean","min","max","std","count","sum"}
        Statistic over gap durations per cell.
    include_end_gaps : bool
        If True, include the initial gap (from start time until first access) and
        trailing gap (from last access until end time). If False, include only gaps
        strictly between access intervals.
    no_access_value : float
        Fill value when the statistic is undefined (e.g., no gaps counted).
    nan_if_never_access : bool
        If True, cells that never achieve access (ever) are forced to no_access_value
        (typically np.nan), even if include_end_gaps would otherwise yield a single
        full-interval gap.

    Returns
    -------
    np.ndarray
        (nlats, nlons) array. dtype depends on stat:
          - "count" returns int32
          - others return float64
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing (all dt > 0)")

    if not observer_positions or len(observer_positions) == 0:
        raise ValueError("observer_positions must be a non-empty list")

    obs_arrays: List[np.ndarray] = []
    nt = -1
    for obs in observer_positions:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Each observer position array must have shape (nt, 3)")
        if nt < 0:
            nt = int(arr.shape[0])
            if nt < 1:
                raise ValueError("Observer position arrays must have nt >= 1")
        elif arr.shape[0] != nt:
            raise ValueError("All observer position arrays must share the same nt")
        obs_arrays.append(arr)

    if nt != time.size:
        raise ValueError("Each observer position array must have shape (len(time), 3)")

    n_obs = len(obs_arrays)
    if min_assets <= 0:
        raise ValueError("min_assets must be a positive integer")
    # allow min_assets > n_obs (means no access ever), but keep semantics consistent
    min_assets_i = int(min_assets)

    obs_stack = np.ascontiguousarray(np.stack(obs_arrays, axis=1))  # (nt,n_obs,3)

    ny = int(config.nlats)
    nx = int(config.nlons)

    # per-cell state + accumulators
    prev_access = np.zeros((ny, nx), dtype=np.uint8)
    gap_start = np.full((ny, nx), -1.0, dtype=np.float64)
    any_access = np.zeros((ny, nx), dtype=np.uint8)

    gap_count = np.zeros((ny, nx), dtype=np.int32)
    gap_sum = np.zeros((ny, nx), dtype=np.float64)
    gap_sumsq = np.zeros((ny, nx), dtype=np.float64)
    gap_min = np.full((ny, nx), 1.0e300, dtype=np.float64)
    gap_max = np.zeros((ny, nx), dtype=np.float64)

    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)

    _gap_durations_empirical_kernel(
        time,
        obs_stack,
        min_assets_i,
        1 if include_end_gaps else 0,
        prev_access,
        gap_start,
        any_access,
        gap_count,
        gap_sum,
        gap_sumsq,
        gap_min,
        gap_max,
        diff_grid,
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

    # post-process to requested stat
    if stat == "count":
        out_i = gap_count.copy()
        if nan_if_never_access:
            # count has no NaN; use 0 as "no access" if user asks for that masking
            out_i[any_access == 0] = 0
        # also zero-out undefined "no gaps" vs 0 gaps is already 0
        return out_i

    # float outputs
    out = np.full((ny, nx), no_access_value, dtype=np.float64)
    mask = gap_count > 0

    if stat == "sum":
        out[mask] = gap_sum[mask]  # (mask not strictly required here)
    elif stat == "mean":
        out[mask] = gap_sum[mask] / gap_count[mask]
    elif stat == "min":
        out[mask] = gap_min[mask]
    elif stat == "max":
        out[mask] = gap_max[mask]
    elif stat == "std":
        mean = np.empty((ny, nx), dtype=np.float64)
        mean[mask] = gap_sum[mask] / gap_count[mask]
        var = np.empty((ny, nx), dtype=np.float64)
        var[mask] = (gap_sumsq[mask] / gap_count[mask]) - mean[mask] * mean[mask]
        var[var < 0.0] = 0.0
        out[mask] = np.sqrt(var[mask])
    else:
        raise ValueError(f"Unsupported stat: {stat!r}")

    if nan_if_never_access:
        out[any_access == 0] = no_access_value

    return out
