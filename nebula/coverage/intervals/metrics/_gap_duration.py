from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit, prange

from nebula.coverage.intervals._exact_intervals import (
    AccessIntervalStore,
    _ROOT_EPS,
    _apply_events_at_time,
    _initialize_target_state,
    _next_event_time,
    _resolve_window,
)

GapStat = Literal["mean", "min", "max", "std", "count", "sum"]


@njit(cache=True, parallel=True)
def _gap_stats_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    n_req: int,
    t_start: float,
    t_stop: float,
    include_end_gaps: bool,
    gap_count: np.ndarray,
    gap_sum: np.ndarray,
    gap_sumsq: np.ndarray,
    gap_min: np.ndarray,
    gap_max: np.ndarray,
    has_access: np.ndarray,
) -> None:
    for target_idx in prange(n_targets):
        idx = np.empty(n_obs, dtype=np.int64)
        end_idx = np.empty(n_obs, dtype=np.int64)
        active = np.zeros(n_obs, dtype=np.uint8)

        count = _initialize_target_state(
            target_idx,
            n_obs,
            n_targets,
            pair_offsets,
            start_times,
            stop_times,
            t_start,
            idx,
            end_idx,
            active,
        )

        access = count >= n_req
        seen_access = 1 if access else 0
        gap_open = 1 if (include_end_gaps and (not access)) else 0
        gap_start = t_start

        g_count = 0
        g_sum = 0.0
        g_sumsq = 0.0
        g_min = 1.0e300
        g_max = 0.0

        cur = t_start
        while cur < t_stop - _ROOT_EPS:
            nxt = _next_event_time(
                n_obs, idx, end_idx, active, start_times, stop_times, t_stop
            )
            if nxt < cur:
                nxt = cur
            if nxt > t_stop:
                nxt = t_stop

            if nxt >= t_stop - _ROOT_EPS:
                break

            prev_access = access
            count = _apply_events_at_time(
                nxt, n_obs, idx, end_idx, active, start_times, stop_times, count
            )
            access = count >= n_req

            if access:
                seen_access = 1

            if prev_access and (not access):
                gap_open = 1
                gap_start = nxt
            elif (not prev_access) and access:
                if gap_open != 0:
                    g = nxt - gap_start
                    if g > 0.0:
                        g_count += 1
                        g_sum += g
                        g_sumsq += g * g
                        if g < g_min:
                            g_min = g
                        if g > g_max:
                            g_max = g
                gap_open = 0

            cur = nxt

        if include_end_gaps and gap_open != 0:
            g = t_stop - gap_start
            if g > 0.0:
                g_count += 1
                g_sum += g
                g_sumsq += g * g
                if g < g_min:
                    g_min = g
                if g > g_max:
                    g_max = g

        gap_count[target_idx] = g_count
        gap_sum[target_idx] = g_sum
        gap_sumsq[target_idx] = g_sumsq
        gap_min[target_idx] = g_min
        gap_max[target_idx] = g_max
        has_access[target_idx] = np.uint8(seen_access)


def calculate_gap_duration(
    store: AccessIntervalStore,
    *,
    min_assets: int = 1,
    stat: GapStat = "mean",
    include_end_gaps: bool = True,
    no_access_value: float = np.nan,
    nan_if_never_access: bool = False,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based gap duration statistics per target.
    """
    n_req = int(min_assets)
    if n_req <= 0:
        raise ValueError("min_assets must be a positive integer")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    n_targets = int(store.n_targets)

    gap_count = np.zeros(n_targets, dtype=np.int32)
    gap_sum = np.zeros(n_targets, dtype=np.float64)
    gap_sumsq = np.zeros(n_targets, dtype=np.float64)
    gap_min = np.full(n_targets, 1.0e300, dtype=np.float64)
    gap_max = np.zeros(n_targets, dtype=np.float64)
    has_access = np.zeros(n_targets, dtype=np.uint8)

    _gap_stats_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        n_targets,
        n_req,
        t0,
        t1,
        bool(include_end_gaps),
        gap_count,
        gap_sum,
        gap_sumsq,
        gap_min,
        gap_max,
        has_access,
    )

    if stat == "count":
        out_i = gap_count.copy()
        if nan_if_never_access:
            out_i[has_access == 0] = 0
        return store.reshape_target_values(out_i) if reshape else out_i

    out = np.full(n_targets, float(no_access_value), dtype=np.float64)
    mask = gap_count > 0

    if stat == "sum":
        out[mask] = gap_sum[mask]
    elif stat == "mean":
        out[mask] = gap_sum[mask] / gap_count[mask]
    elif stat == "min":
        out[mask] = gap_min[mask]
    elif stat == "max":
        out[mask] = gap_max[mask]
    elif stat == "std":
        mean = np.zeros(n_targets, dtype=np.float64)
        mean[mask] = gap_sum[mask] / gap_count[mask]
        var = np.zeros(n_targets, dtype=np.float64)
        var[mask] = (gap_sumsq[mask] / gap_count[mask]) - mean[mask] * mean[mask]
        var[var < 0.0] = 0.0
        out[mask] = np.sqrt(var[mask])
    else:
        raise ValueError(f"Unsupported stat: {stat!r}")

    if nan_if_never_access:
        out[has_access == 0] = float(no_access_value)

    return store.reshape_target_values(out) if reshape else out

