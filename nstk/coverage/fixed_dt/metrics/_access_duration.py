from __future__ import annotations

from collections.abc import Iterable as _Iterable
from typing import List, Union, Dict

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
def _access_duration_empirical_kernel(
    time: np.ndarray,  # (nt,)
    obs_stack: np.ndarray,  # (nt,n_obs,3)
    thresh_count_lut: np.ndarray,  # (n_obs+1,)
    durations: np.ndarray,  # (nN,ny,nx)
    diff_grid: np.ndarray,  # (ny,nx+1) int32
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

    for ti in range(nt - 1):
        dt = time[ti + 1] - time[ti]
        if dt <= 0.0:
            continue

        # precompute i_center per observer for this timestep (serial)
        for k in range(n_obs):
            oy = obs_stack[ti, k, 1]
            ox = obs_stack[ti, k, 0]
            lon0 = np.atan2(oy, ox)  # [-pi,pi]
            lon0_2pi = lon0 + np.pi
            if lon0_2pi < 0.0:
                lon0_2pi += two_pi
            if lon0_2pi >= two_pi:
                lon0_2pi -= two_pi

            i_center_k[k] = _center_index_from_lon0(lon0_2pi, t0, t1, base, dlon, nx)

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

            # prefix-sum counts and update durations
            run = 0
            for i in range(nx):
                run += diff_row[i]  # count at (j,i)
                c = run  # 0..n_obs
                mmax = thresh_count_lut[c]
                for m in range(mmax):
                    durations[m, j, i] += dt


def calculate_access_duration(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: List[np.ndarray],
    N: Union[List[int], int] = 1,
) -> Dict[int, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing (all dt > 0)")

    if not observer_positions or len(observer_positions) == 0:
        raise ValueError("observer_positions must be a non-empty list")

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

    if isinstance(N, _Iterable) and not isinstance(N, (bytes, str)):
        N_list = [int(x) for x in N]
    else:
        N_list = [int(N)]

    if any(n <= 0 for n in N_list):
        raise ValueError("All elements in N must be positive integers")

    N_list = sorted(set(N_list))

    thresh_count_lut = np.zeros(n_obs + 1, dtype=np.int32)
    m = 0
    for c in range(n_obs + 1):
        while m < len(N_list) and N_list[m] <= c:
            m += 1
        thresh_count_lut[c] = m

    ny = config.nlats
    nx = config.nlons
    durations = np.zeros((len(N_list), ny, nx), dtype=np.float64)

    # IMPORTANT: diff_grid must be (ny, nx+1) and SIGNED
    diff_grid = np.zeros((ny, nx + 1), dtype=np.int32)

    _access_duration_empirical_kernel(
        time,
        obs_stack,
        thresh_count_lut,
        durations,
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

    out: Dict[int, np.ndarray] = {}
    for idx, n in enumerate(N_list):
        out[n] = durations[idx]
    return out
