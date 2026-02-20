# _mtta.py

from __future__ import annotations

from typing import List

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
def _mtta_empirical_kernel(
    time: np.ndarray,  # (nt,)
    obs_stack: np.ndarray,  # (nt,n_obs,3)
    N_req: int,  # minimum concurrent assets required
    wrap: bool,  # treat the analysis interval as periodic (recommended for "steady-state" MTTA)
    no_access_value: float,  # value to assign if the point never achieves access
    out: np.ndarray,  # (ny,nx) accumulated integral, then overwritten with MTTA
    diff_grid: np.ndarray,  # (ny,nx+1) int32
    state: np.ndarray,  # (ny,nx) uint8, 1 if in-access for previous segment
    gap_start: np.ndarray,  # (ny,nx) float64, valid when state==0
    first_start: np.ndarray,  # (ny,nx) float64, first access start time or <0 if never
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

    t0 = time[0]
    t_end = time[nt - 1]
    T = t_end - t0
    if T <= 0.0:
        # caller validates strictly increasing time; keep a guard anyway
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

    # Main sweep over segments [time[ti], time[ti+1])
    for ti in range(nt - 1):
        # still validate dt inside kernel (robustness)
        dt = time[ti + 1] - time[ti]
        if dt <= 0.0:
            continue

        t = time[ti]

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

            i_center_k[k] = _center_index_from_lon0(
                lon0_2pi, dom0, dom1, base, dlon, nx
            )

        # parallel over rows
        for j in prange(ny):
            diff_row = diff_grid[j]

            # zero diff row (nx+1)
            for ii in range(nx + 1):
                diff_row[ii] = 0

            # build diff_row from all observers' intervals on this row
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

            # prefix-sum counts and update per-cell access/gap bookkeeping at boundary t
            run = 0
            for i in range(nx):
                run += diff_row[i]  # count at (j,i) for this segment
                access_now = run >= N_req
                prev_access = state[j, i] != 0

                if (not prev_access) and access_now:
                    # gap ended at time t (access starts)
                    gs = gap_start[j, i]
                    g = t - gs
                    if g > 0.0:
                        out[j, i] += 0.5 * g * g

                    if first_start[j, i] < 0.0:
                        first_start[j, i] = t

                    state[j, i] = np.uint8(1)

                elif prev_access and (not access_now):
                    # access ended at time t (gap starts)
                    gap_start[j, i] = t
                    state[j, i] = np.uint8(0)

                # else: no transition at this boundary

    # Finalize trailing portion + divide by T
    for j in prange(ny):
        for i in range(nx):
            fs = first_start[j, i]
            if fs < 0.0:
                # never in access at any time
                out[j, i] = no_access_value
                continue

            # If we're in a gap at the end, add the tail contribution.
            if state[j, i] == 0:
                tail = t_end - gap_start[j, i]
                if tail > 0.0:
                    if wrap:
                        # In wrap mode, the "next access" after t_end is the first access of the next period:
                        # integral over tail of (g_lead + (t_end - t)) dt = g_lead*tail + 0.5*tail^2
                        g_lead = fs - t0
                        if g_lead < 0.0:
                            g_lead = 0.0
                        out[j, i] += g_lead * tail + 0.5 * tail * tail
                    else:
                        # Non-wrap mode: clip the final gap to the analysis end bound
                        out[j, i] += 0.5 * tail * tail

            out[j, i] /= T


def calculate_mtta(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: List[np.ndarray],
    N: int = 1,
    *,
    wrap: bool = False,
    no_access_value: float = np.nan,
) -> np.ndarray:
    """
    Mean Time To Access (MTTA) per grid point.

    Definition:
      MTTA is the time-average of "time until the next access begins", with value 0 while
      currently in access. This is equivalent to the common "Response Time (Average)" figure
      of merit used in aerospace coverage analysis tools.

    Parameters
    ----------
    config : CoverageConfig
        Coverage grid/configuration.
    time : np.ndarray
        1D strictly-increasing time array (seconds).
    observer_positions : List[np.ndarray]
        List of observer ECEF position histories; each element has shape (len(time), 3).
    N : int
        Minimum number of concurrent visible assets required to consider the point "in access".
    wrap : bool
        If True, treat the analysis interval as periodic when evaluating the final trailing gap.
        This is generally preferred for "steady-state" MTTA to reduce boundary effects.
        If False, the trailing gap is clipped to the analysis end time.
    no_access_value : float
        Value assigned if a grid point never achieves access during the analysis interval.

    Returns
    -------
    np.ndarray
        MTTA grid, shape (nlats, nlons), float64 (seconds).
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")

    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing (all dt > 0)")

    if not observer_positions or len(observer_positions) == 0:
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

    obs_stack = np.stack(obs_arrays, axis=1)
    n_obs = obs_stack.shape[1]

    # If N > n_obs, access is impossible everywhere; return constant no_access_value.
    if N > n_obs:
        return np.full(
            (config.nlats, config.nlons), float(no_access_value), dtype=np.float64
        )

    ny = int(config.nlats)
    nx = int(config.nlons)

    out = np.zeros((ny, nx), dtype=np.float64)  # accum integral, then MTTA
    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)  # SIGNED

    # state for previous segment, and gap metadata
    state = np.zeros((ny, nx), dtype=np.uint8)  # start in "gap"
    gap_start = np.full((ny, nx), float(time[0]), dtype=np.float64)
    first_start = np.full((ny, nx), -1.0, dtype=np.float64)

    _mtta_empirical_kernel(
        time,
        obs_stack,
        int(N),
        bool(wrap),
        float(no_access_value),
        out,
        diff_grid,
        state,
        gap_start,
        first_start,
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
