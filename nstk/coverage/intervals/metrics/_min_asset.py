from __future__ import annotations

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
def _min_asset_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    t_start: float,
    t_stop: float,
    out: np.ndarray,
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

        cur = t_start
        min_count = count

        while cur < t_stop - _ROOT_EPS:
            nxt = _next_event_time(
                n_obs, idx, end_idx, active, start_times, stop_times, t_stop
            )
            if nxt < cur:
                nxt = cur
            if nxt > t_stop:
                nxt = t_stop

            if nxt > cur and count < min_count:
                min_count = count

            if nxt >= t_stop - _ROOT_EPS:
                break

            count = _apply_events_at_time(
                nxt, n_obs, idx, end_idx, active, start_times, stop_times, count
            )
            if count < min_count:
                min_count = count
            cur = nxt

        out[target_idx] = min_count


def calculate_min_asset(
    store: AccessIntervalStore,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based minimum concurrent visible assets per target.
    """
    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.int32)
    _min_asset_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        t0,
        t1,
        out,
    )
    return store.reshape_target_values(out) if reshape else out

