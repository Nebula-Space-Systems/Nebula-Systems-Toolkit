from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from nstk.coverage.intervals._exact_intervals import (
    AccessIntervalStore,
    _ROOT_EPS,
    _apply_events_at_time,
    _initialize_target_state,
    _next_event_time,
    _resolve_window,
)


@njit(cache=True, parallel=True)
def _access_separation_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    n_req: int,
    min_sep_s: float,
    max_sep_s: float,
    t_start: float,
    t_stop: float,
    satisfied: np.ndarray,
    ever_access: np.ndarray,
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
        sat = 0
        last_end = np.nan

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

            if prev_access and (not access):
                last_end = nxt
                seen_access = 1
            elif (not prev_access) and access:
                seen_access = 1
                if not np.isnan(last_end):
                    gap = nxt - last_end
                    if gap <= max_sep_s and (min_sep_s <= 0.0 or gap >= min_sep_s):
                        sat = 1
                        break

            cur = nxt

        if access:
            seen_access = 1

        satisfied[target_idx] = np.uint8(sat)
        ever_access[target_idx] = np.uint8(seen_access)


def calculate_access_separation(
    store: AccessIntervalStore,
    min_assets: int = 1,
    min_separation_s: float = 0.0,
    max_separation_s: float = np.inf,
    *,
    no_access_value: Optional[float] = np.nan,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based Access Separation metric (STK-style binary FOM).
    """
    n_req = int(min_assets)
    if n_req <= 0:
        raise ValueError("min_assets must be >= 1")

    min_sep_s = float(min_separation_s)
    max_sep_s = float(max_separation_s)
    if min_sep_s < 0.0:
        raise ValueError("min_separation_s must be >= 0")
    if not np.isfinite(max_sep_s):
        max_sep_s = 1.0e300
    if max_sep_s <= 0.0:
        raise ValueError("max_separation_s must be > 0 (or np.inf)")
    if max_sep_s < min_sep_s:
        raise ValueError("max_separation_s must be >= min_separation_s")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    n_targets = int(store.n_targets)

    if n_req > int(store.n_observers):
        if no_access_value is None:
            out_u8 = np.zeros(n_targets, dtype=np.uint8)
            return store.reshape_target_values(out_u8) if reshape else out_u8
        out_na = np.full(n_targets, float(no_access_value), dtype=np.float64)
        return store.reshape_target_values(out_na) if reshape else out_na

    satisfied = np.zeros(n_targets, dtype=np.uint8)
    ever_access = np.zeros(n_targets, dtype=np.uint8)
    _access_separation_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        n_targets,
        n_req,
        min_sep_s,
        max_sep_s,
        t0,
        t1,
        satisfied,
        ever_access,
    )

    if no_access_value is None:
        return store.reshape_target_values(satisfied) if reshape else satisfied

    out = satisfied.astype(np.float64)
    out[ever_access == 0] = float(no_access_value)
    return store.reshape_target_values(out) if reshape else out

