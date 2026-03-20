from __future__ import annotations

from typing import Literal

import numpy as np

from nstk.coverage.intervals._exact_intervals import AccessIntervalStore, _resolve_window
from nstk.coverage.intervals.metrics._gap_duration import _gap_stats_by_target_kernel


def calculate_revisit_time(
    store: AccessIntervalStore,
    N: int = 1,
    *,
    option: Literal["average", "maximum", "minimum", "std_deviation"] = "average",
    end_gaps: Literal["include", "ignore"] = "include",
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Interval-based Revisit Time (STK-style) per target.
    """
    n_req = int(N)
    if n_req <= 0:
        raise ValueError("N must be a positive integer")

    if option not in ("average", "maximum", "minimum", "std_deviation"):
        raise ValueError(
            "option must be one of: average, maximum, minimum, std_deviation"
        )
    if end_gaps not in ("include", "ignore"):
        raise ValueError("end_gaps must be either 'include' or 'ignore'")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    n_targets = int(store.n_targets)
    T = float(t1 - t0)

    if n_req > int(store.n_observers):
        out_full = np.full(n_targets, T, dtype=np.float64)
        return store.reshape_target_values(out_full) if reshape else out_full

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
        bool(end_gaps == "include"),
        gap_count,
        gap_sum,
        gap_sumsq,
        gap_min,
        gap_max,
        has_access,
    )

    out = np.zeros(n_targets, dtype=np.float64)
    no_access_mask = has_access == 0
    out[no_access_mask] = T

    gap_mask = (~no_access_mask) & (gap_count > 0)
    if option == "average":
        out[gap_mask] = gap_sum[gap_mask] / gap_count[gap_mask]
    elif option == "maximum":
        out[gap_mask] = gap_max[gap_mask]
    elif option == "minimum":
        out[gap_mask] = gap_min[gap_mask]
    else:
        mean = np.zeros(n_targets, dtype=np.float64)
        mean[gap_mask] = gap_sum[gap_mask] / gap_count[gap_mask]
        var = np.zeros(n_targets, dtype=np.float64)
        var[gap_mask] = (gap_sumsq[gap_mask] / gap_count[gap_mask]) - mean[
            gap_mask
        ] * mean[gap_mask]
        var[var < 0.0] = 0.0
        out[gap_mask] = np.sqrt(var[gap_mask])

    # For cells with access but no included gaps, revisit time is defined as zero.
    return store.reshape_target_values(out) if reshape else out

