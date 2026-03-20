# _max_asset.py
from __future__ import annotations

from typing import List

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
def _max_asset_empirical_kernel(
    obs_stack: np.ndarray,  # (nt, n_obs, 3)
    max_counts: np.ndarray,  # (ny, nx) int32 (in/out)
    row_remaining: np.ndarray,  # (ny,) int32 (in/out) number of cells not yet at n_obs in each row
    diff_grid: np.ndarray,  # (ny, nx+1) int32 (scratch)
    lat_gc_rows: np.ndarray,  # (ny,) (only for ny sizing / signature parity)
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
    nt = obs_stack.shape[0]
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

    # Evaluate at every provided sample; max() does not depend on dt.
    for ti in range(nt):
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

        # parallel over rows
        for j in prange(ny):
            # If this entire row has already reached n_obs everywhere, it can never improve.
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

            # prefix-sum counts; update per-cell maximum count
            run = 0
            row_max = max_counts[j]
            rem = row_remaining[j]

            for i in range(nx):
                run += diff_row[i]  # instantaneous count at (j,i)

                # Only update if we found a larger instantaneous count than previously seen.
                if run > row_max[i]:
                    row_max[i] = run

                    # If it just reached n_obs, reduce remaining count; enables future early-exit.
                    if run == n_obs:
                        rem -= 1

            row_remaining[j] = rem


def calculate_max_asset(
    config: CoverageConfig,
    observer_positions: List[np.ndarray],
) -> np.ndarray:
    """
    max_asset metric: for each grid cell, return the maximum instantaneous number of
    visible assets across all provided observer ECEF samples.

    Notes / semantics
    -----------------
    - This metric does NOT use time deltas. It evaluates visibility at each provided
      sample index (ti = 0..nt-1) and takes the maximum count over those samples.
    - If a grid cell ever sees all assets simultaneously (count == n_obs) at any
      sample, its result becomes n_obs.

    Parameters
    ----------
    config : CoverageConfig
        Coverage grid + precomputed geometry tables.
    observer_positions : list[np.ndarray]
        List of observers. Each element must be shape (nt, 3) in ECEF meters.
        All observers must share the same nt.

    Returns
    -------
    np.ndarray
        Shape (config.nlats, config.nlons), dtype int32. Values in [0, n_obs].
    """
    if not observer_positions or len(observer_positions) == 0:
        raise ValueError("observer_positions must be a non-empty list")

    # Validate and stack (nt, n_obs, 3)
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

    obs_stack = np.ascontiguousarray(np.stack(obs_arrays, axis=1))  # (nt, n_obs, 3)
    ny = int(config.nlats)
    nx = int(config.nlons)

    # Initialize to minimum possible; kernel increases via max.
    max_counts = np.zeros((ny, nx), dtype=np.int32)

    # Track how many cells per row have not yet hit n_obs; enables early exit per row.
    row_remaining = np.full(ny, nx, dtype=np.int32)

    # Scratch: diff grid per row (nx+1), signed
    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)

    _max_asset_empirical_kernel(
        obs_stack,
        max_counts,
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

    return max_counts
