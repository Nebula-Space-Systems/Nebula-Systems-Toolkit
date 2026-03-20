from __future__ import annotations

import numpy as np

from nstk.coverage.intervals._exact_intervals import (
    AccessIntervalStore,
    _mtta_by_target_kernel,
    _resolve_window,
)


def mtta_by_target(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    wrap: bool = False,
    no_access_value: float = np.nan,
    reshape: bool = True,
) -> np.ndarray:
    """
    Mean Time To Access (MTTA) per target from precomputed exact intervals.
    """
    n_req = int(N)
    if n_req <= 0:
        raise ValueError("N must be >= 1")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.float64)
    if n_req > store.n_observers:
        out.fill(float(no_access_value))
        return store.reshape_target_values(out) if reshape else out

    _mtta_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        n_req,
        t0,
        t1,
        bool(wrap),
        float(no_access_value),
        out,
    )
    return store.reshape_target_values(out) if reshape else out


def calculate_mtta(
    store: AccessIntervalStore,
    N: int = 1,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    wrap: bool = False,
    no_access_value: float = np.nan,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based Mean Time To Access (MTTA) metric.
    """
    return mtta_by_target(
        store,
        N=int(N),
        t_start=t_start,
        t_stop=t_stop,
        wrap=bool(wrap),
        no_access_value=float(no_access_value),
        reshape=reshape,
    )
