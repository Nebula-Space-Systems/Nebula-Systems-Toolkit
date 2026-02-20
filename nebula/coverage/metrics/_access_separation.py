# _access_separation.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
from numba import njit, prange

from nebula.coverage.config import CoverageConfig
from nebula.coverage._empirical_core import (
    coverage_row_intervals_empirical_1row,
    _segments_minus_inners,
    _diff_add_seg,
    _center_index_from_lon0,
)


@njit(cache=True, parallel=True)
def _access_separation_empirical_kernel(
    time: np.ndarray,  # (nt,)
    obs_stack: np.ndarray,  # (nt, n_obs, 3)
    min_assets: int,
    min_sep_s: float,
    max_sep_s: float,
    satisfied: np.ndarray,  # (ny, nx) uint8 (out)
    in_access: np.ndarray,  # (ny, nx) uint8 (in/out)
    last_end: np.ndarray,  # (ny, nx) float64 (in/out) last access end time
    row_remaining: np.ndarray,  # (ny,) int32 (in/out) number of cells not yet satisfied
    diff_grid: np.ndarray,  # (ny, nx+1) int32 scratch
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

    # reused per timestep
    i_center_k = np.empty(n_obs, dtype=np.int32)

    # Access intervals are modeled on [time[ti], time[ti+1]) using obs at ti
    for ti in range(nt - 1):
        t_i = time[ti]

        # precompute i_center per observer for this timestep (serial)
        for k in range(n_obs):
            oy = obs_stack[ti, k, 1]
            ox = obs_stack[ti, k, 0]
            lon0 = np.atan2(oy, ox)  # [-pi, pi]
            lon0_2pi = lon0 + np.pi
            if lon0_2pi < 0.0:
                lon0_2pi += two_pi
            if lon0_2pi >= two_pi:
                lon0_2pi -= two_pi
            i_center_k[k] = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

        for j in prange(ny):
            if row_remaining[j] == 0:
                continue

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

            # prefix-sum counts; detect access interval boundaries for (count >= min_assets)
            run = 0
            sat_row = satisfied[j]
            acc_row = in_access[j]
            end_row = last_end[j]
            rem = row_remaining[j]

            for i in range(nx):
                run += diff_row[i]
                if sat_row[i] != 0:
                    continue

                curr = run >= min_assets
                prev = acc_row[i] != 0

                # end of previous access interval at boundary t_i
                if prev and (not curr):
                    end_row[i] = t_i
                    acc_row[i] = 0
                    continue

                # start of new access interval at boundary t_i
                if (not prev) and curr:
                    acc_row[i] = 1
                    le = end_row[i]
                    if not np.isnan(le):
                        gap = t_i - le

                        # Apply max always; apply min only if nonzero
                        if gap <= max_sep_s and (min_sep_s <= 0.0 or gap >= min_sep_s):
                            sat_row[i] = 1
                            rem -= 1
                    continue

                # steady-state
                acc_row[i] = 1 if curr else 0

            row_remaining[j] = rem


def calculate_access_separation(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: List[np.ndarray],
    min_assets: int = 1,
    min_separation_s: float = 0.0,
    max_separation_s: float = np.inf,
    *,
    no_access_value: Optional[float] = np.nan,
) -> np.ndarray:
    """
    Access Separation (STK-style binary FOM over a sampled coverage timeline).

    Returns:
      - 1 where there exists at least one qualifying pair of consecutive access intervals
      - 0 where access occurs but no qualifying pair exists
      - no_access_value (default NaN) where access NEVER occurs at all

    Qualification is based on separation (gap) between end of leading interval and
    start of trailing interval:
        gap <= max_separation_s
        and if min_separation_s > 0: gap >= min_separation_s

    Notes:
    - Sampling model matches access_duration: state at sample ti is assumed valid over
      [time[ti], time[ti+1]).
    - Output:
        * float64 if no_access_value is not None (so NaN is representable)
        * uint8 if no_access_value is None
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing (all dt > 0)")

    if not observer_positions:
        raise ValueError("observer_positions must be a non-empty list")

    min_assets = int(min_assets)
    if min_assets <= 0:
        raise ValueError("min_assets must be >= 1")

    min_separation_s = float(min_separation_s)
    max_separation_s = float(max_separation_s)

    if min_separation_s < 0.0:
        raise ValueError("min_separation_s must be >= 0")

    # Numba robustness: avoid inf inside kernel if desired
    if not np.isfinite(max_separation_s):
        max_separation_s = 1.0e300

    if max_separation_s <= 0.0:
        raise ValueError("max_separation_s must be > 0 (or np.inf)")
    if max_separation_s < min_separation_s:
        raise ValueError("max_separation_s must be >= min_separation_s")

    # Validate and stack (nt, n_obs, 3)
    nt = int(time.size)
    obs_arrays: List[np.ndarray] = []
    for obs in observer_positions:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape != (nt, 3):
            raise ValueError(
                "Each observer position array must have shape (len(time), 3)"
            )
        obs_arrays.append(arr)

    obs_stack = np.ascontiguousarray(np.stack(obs_arrays, axis=1))  # (nt, n_obs, 3)
    n_obs = int(obs_stack.shape[1])

    ny = int(config.nlats)
    nx = int(config.nlons)

    # If requirement exceeds number of observers, it can never be satisfied.
    if min_assets > n_obs:
        if no_access_value is None:
            return np.zeros((ny, nx), dtype=np.uint8)
        out = np.full((ny, nx), no_access_value, dtype=np.float64)
        return out

    satisfied = np.zeros((ny, nx), dtype=np.uint8)
    in_access = np.zeros((ny, nx), dtype=np.uint8)
    last_end = np.full((ny, nx), np.nan, dtype=np.float64)

    row_remaining = np.full(ny, nx, dtype=np.int32)
    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)

    _access_separation_empirical_kernel(
        time,
        obs_stack,
        min_assets,
        min_separation_s,
        max_separation_s,
        satisfied,
        in_access,
        last_end,
        row_remaining,
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

    if no_access_value is None:
        return satisfied

    # A cell has "ever had access" if:
    # - it ended at least one interval (last_end finite), OR
    # - it is still in access at the end (in_access==1), OR
    # - it is satisfied (implies access happened)
    ever_access = (in_access != 0) | np.isfinite(last_end) | (satisfied != 0)

    out = satisfied.astype(np.float64)
    out[~ever_access] = no_access_value
    return out
