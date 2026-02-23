from __future__ import annotations

import numpy as np

from nebula.coverage.intervals._exact_intervals import (
    AccessIntervalStore,
    _max_asset_by_target_kernel,
    _resolve_window,
)


def max_asset_by_target(
    store: AccessIntervalStore,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Maximum concurrent observers in access per target over the query window.
    """
    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.int32)
    _max_asset_by_target_kernel(
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


def calculate_max_asset(
    store: AccessIntervalStore,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based max asset metric.
    """
    return max_asset_by_target(
        store,
        t_start=t_start,
        t_stop=t_stop,
        reshape=reshape,
    )
