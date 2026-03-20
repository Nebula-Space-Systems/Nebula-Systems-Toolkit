from __future__ import annotations

from collections.abc import Iterable as _Iterable
from typing import Dict, Union

import numpy as np

from nstk.coverage.intervals._exact_intervals import (
    AccessIntervalStore,
    _duration_by_target_kernel,
    _resolve_window,
)


def _duration_scale(
    t_start: float,
    t_stop: float,
    *,
    normalize_to_day: bool,
) -> float:
    scale = 1.0
    if not normalize_to_day:
        return scale

    window_s = float(t_stop - t_start)
    if window_s <= 0.0:
        raise ValueError("Normalization requires a positive query window")
    scale *= 86400.0 / window_s
    return scale


def access_duration_by_target(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = False,
    normalize_to_day: bool = False,
) -> np.ndarray:
    """
    Access duration in seconds per target requiring at least `N` concurrent observers.

    Parameters
    ----------
    normalize_to_day : bool
        If True, return average seconds/day over the selected query window.
        If False, return total seconds within the selected query window.
    """
    n_req = int(N)
    if n_req <= 0:
        raise ValueError("N must be >= 1")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.float64)
    if n_req > store.n_observers:
        return store.reshape_target_values(out) if reshape else out

    _duration_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        n_req,
        t0,
        t1,
        out,
    )

    scale = _duration_scale(t0, t1, normalize_to_day=normalize_to_day)
    if scale != 1.0:
        out *= scale

    return store.reshape_target_values(out) if reshape else out


def calculate_access_duration(
    store: AccessIntervalStore,
    N: Union[list[int], int] = 1,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = False,
    normalize_to_day: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Interval-based access duration metric.

    Parameters
    ----------
    store : AccessIntervalStore
        Precomputed access interval store.
    N : int | list[int]
        Required minimum simultaneous observers.
    t_start, t_stop : float | None
        Optional query window inside the store time bounds.
    reshape : bool
        If True, reshape to `store.target_shape` when available.
    normalize_to_day : bool
        If True, return average seconds/day over the selected query window.
        If False, return total seconds within the selected query window.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from threshold `N` to duration array.
    """
    if isinstance(N, _Iterable) and not isinstance(N, (bytes, str)):
        n_list = [int(x) for x in N]
    else:
        n_list = [int(N)]

    if any(n <= 0 for n in n_list):
        raise ValueError("All elements in N must be positive integers")

    n_list = sorted(set(n_list))
    out: Dict[int, np.ndarray] = {}
    for n_req in n_list:
        out[n_req] = access_duration_by_target(
            store,
            N=n_req,
            t_start=t_start,
            t_stop=t_stop,
            reshape=reshape,
            normalize_to_day=normalize_to_day,
        )
    return out
