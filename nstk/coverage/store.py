from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TimeGateStore:
    scope: str
    item_offsets: np.ndarray
    start_times: np.ndarray
    stop_times: np.ndarray

    def item_intervals(self, item_index: int) -> tuple[np.ndarray, np.ndarray]:
        i0 = int(self.item_offsets[item_index])
        i1 = int(self.item_offsets[item_index + 1])
        return self.start_times[i0:i1], self.stop_times[i0:i1]

    @property
    def n_items(self) -> int:
        return int(self.item_offsets.size - 1)


@dataclass(frozen=True)
class PairChannelStore:
    name: str
    scope: str
    sample_times: np.ndarray
    values: np.ndarray
    n_observers: int | None = None
    n_targets: int | None = None


@dataclass(frozen=True)
class IntervalStore:
    time_start: float
    time_stop: float
    n_observers: int
    n_targets: int
    pair_offsets: np.ndarray
    start_times: np.ndarray
    stop_times: np.ndarray
    interpolation: str = "linear"
    root_tolerance_s: float = 1e-3

    def pair_index(self, observer_index: int, target_index: int) -> int:
        return int(observer_index) * int(self.n_targets) + int(target_index)

    def pair_intervals(self, observer_index: int, target_index: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.pair_index(observer_index, target_index)
        i0 = int(self.pair_offsets[p])
        i1 = int(self.pair_offsets[p + 1])
        return self.start_times[i0:i1], self.stop_times[i0:i1]

    @property
    def n_pairs(self) -> int:
        return int(self.n_observers * self.n_targets)


def from_access_interval_store(store: Any) -> IntervalStore:
    return IntervalStore(
        time_start=float(store.time_start),
        time_stop=float(store.time_stop),
        n_observers=int(store.n_observers),
        n_targets=int(store.n_targets),
        pair_offsets=np.ascontiguousarray(np.asarray(store.pair_offsets, dtype=np.int64)),
        start_times=np.ascontiguousarray(np.asarray(store.start_times, dtype=np.float64)),
        stop_times=np.ascontiguousarray(np.asarray(store.stop_times, dtype=np.float64)),
        interpolation=str(getattr(store, "interpolation", "linear")),
        root_tolerance_s=float(getattr(store, "root_tolerance_s", 1e-3)),
    )


def to_access_interval_store(store: IntervalStore) -> Any:
    from nstk.coverage.intervals._exact_intervals import AccessIntervalStore

    return AccessIntervalStore(
        time_start=float(store.time_start),
        time_stop=float(store.time_stop),
        n_observers=int(store.n_observers),
        n_targets=int(store.n_targets),
        pair_offsets=np.asarray(store.pair_offsets, dtype=np.int64),
        start_times=np.asarray(store.start_times, dtype=np.float64),
        stop_times=np.asarray(store.stop_times, dtype=np.float64),
        min_elevation_rad=0.0,
        max_elevation_rad=0.5 * np.pi,
        interpolation=str(store.interpolation),
        root_tolerance_s=float(store.root_tolerance_s),
    )


def _merge_interval_list(starts: list[float], stops: list[float], *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    order = np.argsort(np.asarray(starts, dtype=np.float64))
    sorted_starts = np.asarray(starts, dtype=np.float64)[order]
    sorted_stops = np.asarray(stops, dtype=np.float64)[order]

    out_starts = [float(sorted_starts[0])]
    out_stops = [float(sorted_stops[0])]
    for start, stop in zip(sorted_starts[1:], sorted_stops[1:]):
        if float(start) <= out_stops[-1] + eps:
            out_stops[-1] = max(out_stops[-1], float(stop))
        else:
            out_starts.append(float(start))
            out_stops.append(float(stop))
    return np.asarray(out_starts, dtype=np.float64), np.asarray(out_stops, dtype=np.float64)


def _intersect_lists(
    starts_a: np.ndarray,
    stops_a: np.ndarray,
    starts_b: np.ndarray,
    stops_b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    i = 0
    j = 0
    out_starts: list[float] = []
    out_stops: list[float] = []
    while i < starts_a.size and j < starts_b.size:
        start = max(float(starts_a[i]), float(starts_b[j]))
        stop = min(float(stops_a[i]), float(stops_b[j]))
        if stop > start + eps:
            out_starts.append(start)
            out_stops.append(stop)
        if float(stops_a[i]) <= float(stops_b[j]):
            i += 1
        else:
            j += 1
    return (
        np.asarray(out_starts, dtype=np.float64),
        np.asarray(out_stops, dtype=np.float64),
    )


def _clip_lists(
    starts: np.ndarray,
    stops: np.ndarray,
    t0: float,
    t1: float,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    out_starts: list[float] = []
    out_stops: list[float] = []
    for start, stop in zip(starts, stops):
        s = max(float(start), t0)
        e = min(float(stop), t1)
        if e > s + eps:
            out_starts.append(s)
            out_stops.append(e)
    return np.asarray(out_starts, dtype=np.float64), np.asarray(out_stops, dtype=np.float64)


def _build_store_from_pair_lists(
    time_start: float,
    time_stop: float,
    n_observers: int,
    n_targets: int,
    pair_data: list[tuple[np.ndarray, np.ndarray]],
    *,
    interpolation: str = "linear",
    root_tolerance_s: float = 1e-3,
) -> IntervalStore:
    counts = np.asarray([starts.size for starts, _ in pair_data], dtype=np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    start_times = np.empty(int(offsets[-1]), dtype=np.float64)
    stop_times = np.empty(int(offsets[-1]), dtype=np.float64)
    cursor = 0
    for starts, stops in pair_data:
        n = int(starts.size)
        start_times[cursor : cursor + n] = starts
        stop_times[cursor : cursor + n] = stops
        cursor += n
    return IntervalStore(
        time_start=float(time_start),
        time_stop=float(time_stop),
        n_observers=int(n_observers),
        n_targets=int(n_targets),
        pair_offsets=offsets,
        start_times=start_times,
        stop_times=stop_times,
        interpolation=interpolation,
        root_tolerance_s=float(root_tolerance_s),
    )


def subset_interval_store(
    store: IntervalStore,
    *,
    observer_indices: Iterable[int] | None = None,
    target_indices: Iterable[int] | None = None,
    time_window: tuple[float, float] | None = None,
) -> IntervalStore:
    obs_idx = (
        np.arange(store.n_observers, dtype=np.int64)
        if observer_indices is None
        else np.asarray(list(observer_indices), dtype=np.int64)
    )
    tgt_idx = (
        np.arange(store.n_targets, dtype=np.int64)
        if target_indices is None
        else np.asarray(list(target_indices), dtype=np.int64)
    )
    if time_window is None:
        t0 = float(store.time_start)
        t1 = float(store.time_stop)
    else:
        t0 = float(time_window[0])
        t1 = float(time_window[1])
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in obs_idx:
        for tgt in tgt_idx:
            starts, stops = store.pair_intervals(int(obs), int(tgt))
            clipped = _clip_lists(starts, stops, t0, t1)
            pair_data.append(clipped)
    return _build_store_from_pair_lists(
        t0,
        t1,
        int(obs_idx.size),
        int(tgt_idx.size),
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def intersect_pair_gate_store(store: IntervalStore, gate_store: IntervalStore) -> IntervalStore:
    if store.n_observers != gate_store.n_observers or store.n_targets != gate_store.n_targets:
        raise ValueError("Pair gate store shape does not match interval store")
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in range(store.n_observers):
        for tgt in range(store.n_targets):
            starts, stops = store.pair_intervals(obs, tgt)
            gate_starts, gate_stops = gate_store.pair_intervals(obs, tgt)
            pair_data.append(_intersect_lists(starts, stops, gate_starts, gate_stops))
    return _build_store_from_pair_lists(
        max(store.time_start, gate_store.time_start),
        min(store.time_stop, gate_store.time_stop),
        store.n_observers,
        store.n_targets,
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def intersect_target_gates(store: IntervalStore, gates: TimeGateStore) -> IntervalStore:
    if gates.scope != "target" or gates.n_items != store.n_targets:
        raise ValueError("Target gate store shape does not match interval store")
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in range(store.n_observers):
        for tgt in range(store.n_targets):
            starts, stops = store.pair_intervals(obs, tgt)
            gate_starts, gate_stops = gates.item_intervals(tgt)
            pair_data.append(_intersect_lists(starts, stops, gate_starts, gate_stops))
    return _build_store_from_pair_lists(
        store.time_start,
        store.time_stop,
        store.n_observers,
        store.n_targets,
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def intersect_observer_gates(store: IntervalStore, gates: TimeGateStore) -> IntervalStore:
    if gates.scope != "observer" or gates.n_items != store.n_observers:
        raise ValueError("Observer gate store shape does not match interval store")
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in range(store.n_observers):
        gate_starts, gate_stops = gates.item_intervals(obs)
        for tgt in range(store.n_targets):
            starts, stops = store.pair_intervals(obs, tgt)
            pair_data.append(_intersect_lists(starts, stops, gate_starts, gate_stops))
    return _build_store_from_pair_lists(
        store.time_start,
        store.time_stop,
        store.n_observers,
        store.n_targets,
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def intersect_global_gates(store: IntervalStore, gates: TimeGateStore) -> IntervalStore:
    if gates.scope != "global" or gates.n_items != 1:
        raise ValueError("Global gate store must have scope='global' and exactly one item")
    gate_starts, gate_stops = gates.item_intervals(0)
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in range(store.n_observers):
        for tgt in range(store.n_targets):
            starts, stops = store.pair_intervals(obs, tgt)
            pair_data.append(_intersect_lists(starts, stops, gate_starts, gate_stops))
    return _build_store_from_pair_lists(
        store.time_start,
        store.time_stop,
        store.n_observers,
        store.n_targets,
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def filter_min_duration(store: IntervalStore, min_duration_s: float) -> IntervalStore:
    threshold = float(min_duration_s)
    pair_data: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in range(store.n_observers):
        for tgt in range(store.n_targets):
            starts, stops = store.pair_intervals(obs, tgt)
            keep = (stops - starts) >= threshold
            pair_data.append((starts[keep], stops[keep]))
    return _build_store_from_pair_lists(
        store.time_start,
        store.time_stop,
        store.n_observers,
        store.n_targets,
        pair_data,
        interpolation=store.interpolation,
        root_tolerance_s=store.root_tolerance_s,
    )


def _scores_to_intervals(time: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = scores >= 0.0
    starts: list[float] = []
    stops: list[float] = []
    current_start: float | None = None

    for idx in range(time.size - 1):
        t0 = float(time[idx])
        t1 = float(time[idx + 1])
        s0 = float(scores[idx])
        s1 = float(scores[idx + 1])
        a0 = bool(active[idx])
        a1 = bool(active[idx + 1])
        crossing = None
        if (s0 < 0.0 and s1 > 0.0) or (s0 > 0.0 and s1 < 0.0):
            crossing = t0 + (t1 - t0) * (-s0) / (s1 - s0)

        if current_start is None:
            if a0:
                current_start = t0
            elif crossing is not None and a1:
                current_start = float(crossing)

        if current_start is not None:
            if (not a1) and crossing is not None:
                starts.append(current_start)
                stops.append(float(crossing))
                current_start = None
            elif (not a1) and crossing is None and idx == time.size - 2:
                starts.append(current_start)
                stops.append(t0)
                current_start = None

        if current_start is None and crossing is not None and (not a0) and a1:
            current_start = float(crossing)

    if current_start is not None:
        starts.append(current_start)
        stops.append(float(time[-1]))

    return _merge_interval_list(starts, stops)


def build_pair_gate_store_from_scores(
    time: np.ndarray,
    scores: np.ndarray,
    *,
    n_observers: int,
    n_targets: int,
    interpolation: str = "sampled",
) -> IntervalStore:
    times = np.asarray(time, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.shape != (times.size, n_observers * n_targets):
        raise ValueError("scores must have shape (nt, n_observers * n_targets)")
    pair_data = [_scores_to_intervals(times, score_arr[:, pair_idx]) for pair_idx in range(score_arr.shape[1])]
    return _build_store_from_pair_lists(
        float(times[0]),
        float(times[-1]),
        int(n_observers),
        int(n_targets),
        pair_data,
        interpolation=interpolation,
        root_tolerance_s=0.0,
    )


def build_time_gate_store_from_scores(
    time: np.ndarray,
    scores: np.ndarray,
    *,
    scope: str,
) -> TimeGateStore:
    times = np.asarray(time, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.ndim != 2 or score_arr.shape[0] != times.size:
        raise ValueError("scores must have shape (nt, n_items)")
    per_item = [_scores_to_intervals(times, score_arr[:, item_idx]) for item_idx in range(score_arr.shape[1])]
    counts = np.asarray([starts.size for starts, _ in per_item], dtype=np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    starts = np.empty(int(offsets[-1]), dtype=np.float64)
    stops = np.empty(int(offsets[-1]), dtype=np.float64)
    cursor = 0
    for item_starts, item_stops in per_item:
        n = int(item_starts.size)
        starts[cursor : cursor + n] = item_starts
        stops[cursor : cursor + n] = item_stops
        cursor += n
    return TimeGateStore(
        scope=scope,
        item_offsets=offsets,
        start_times=starts,
        stop_times=stops,
    )


def interval_count_by_pair(store: IntervalStore) -> np.ndarray:
    return np.diff(store.pair_offsets).reshape(store.n_observers, store.n_targets)


__all__ = [
    "TimeGateStore",
    "PairChannelStore",
    "IntervalStore",
    "from_access_interval_store",
    "to_access_interval_store",
    "subset_interval_store",
    "intersect_pair_gate_store",
    "intersect_target_gates",
    "intersect_observer_gates",
    "intersect_global_gates",
    "filter_min_duration",
    "build_pair_gate_store_from_scores",
    "build_time_gate_store_from_scores",
    "interval_count_by_pair",
]
