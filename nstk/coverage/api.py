from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - exercised when astropy is installed
    import astropy.units as u
    from astropy.time import Time
except Exception:  # pragma: no cover - fallback when astropy extras are absent
    u = None
    Time = None  # type: ignore[assignment]

from nstk.coverage.intervals._exact_intervals import build_access_interval_store
from nstk.coverage.intervals.metrics import (
    access_duration_by_target,
    calculate_access_separation,
    calculate_gap_duration,
    calculate_max_asset,
    calculate_min_asset,
    calculate_mtta,
    calculate_revisit_time,
)
from nstk.geometry.fast_sun_position import sun_position_ecef

from .constraints import (
    AzimuthConstraint,
    ConstraintSet,
    ElevationConstraint,
    MinAccessDurationConstraint,
    RangeConstraint,
    TargetLocalTimeConstraint,
    TargetSunElevationConstraint,
)
from .diagnostics import CoverageDiagnostics
from .domains import GlobalEarthDomain, TargetDomain, coerce_domain
from .metrics import CompiledMetric
from .observers import Observer, resolve_observers
from .results import CoverageArray, CoverageField, CoverageStack, TargetTimeline, _wrap_array
from .store import (
    IntervalStore,
    PairChannelStore,
    build_pair_gate_store_from_scores,
    build_time_gate_store_from_scores,
    filter_min_duration,
    from_access_interval_store,
    intersect_pair_gate_store,
    intersect_target_gates,
    subset_interval_store,
    to_access_interval_store,
)
from .targets import CoverageTargets, LatitudeLongitudeSampler, TargetSampler
from .timeline import CoverageTimeline


TimeUnit = str
_COVERAGE_ANALYSIS_FRAME = "itrf"


def _coerce_targets(
    targets: CoverageTargets | TargetDomain | Any | None,
    *,
    domain: TargetDomain | Any | None = None,
    sampler: TargetSampler | None = None,
) -> CoverageTargets:
    if isinstance(targets, CoverageTargets):
        return targets

    materialize_domain = domain if targets is None else targets
    if materialize_domain is None:
        materialize_domain = GlobalEarthDomain()
    domain_obj = (
        materialize_domain
        if isinstance(materialize_domain, TargetDomain)
        else coerce_domain(materialize_domain)
    )
    sampler_obj = sampler if sampler is not None else LatitudeLongitudeSampler()
    return CoverageTargets.from_domain(domain_obj, sampler=sampler_obj)


def _time_scale(unit: TimeUnit) -> tuple[float, str]:
    key = str(unit).lower()
    if key in {"s", "sec", "second", "seconds"}:
        return 1.0, "seconds"
    if key in {"m", "min", "minute", "minutes"}:
        return 1.0 / 60.0, "minutes"
    if key in {"h", "hr", "hour", "hours"}:
        return 1.0 / 3600.0, "hours"
    if key in {"d", "day", "days"}:
        return 1.0 / 86400.0, "days"
    raise ValueError("unit must be one of: seconds, minutes, hours, days")


def _convert_duration(
    values_s: np.ndarray,
    *,
    window_s: float,
    unit: TimeUnit,
    normalize: str | bool | None,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(values_s, dtype=np.float64)
    norm = normalize
    if norm is True:
        norm = "day"
    if norm is False:
        norm = None

    if norm in {"fraction", "coverage_fraction"}:
        if window_s <= 0.0:
            raise ValueError("normalize='fraction' requires a positive window")
        return arr / window_s, "fraction"

    scale, unit_label = _time_scale(unit)
    out = arr * scale
    if norm is None:
        return out, unit_label
    if str(norm).lower() == "day":
        if window_s <= 0.0:
            raise ValueError("normalize='day' requires a positive window")
        return out * (86400.0 / window_s), f"{unit_label}/day"
    raise ValueError("normalize must be one of: None, 'day', 'fraction'")


def _bounded_score(
    values: np.ndarray,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> np.ndarray:
    out = np.full(np.asarray(values).shape, 1.0, dtype=np.float64)
    if min_value is not None:
        out = np.minimum(out, np.asarray(values, dtype=np.float64) - float(min_value))
    if max_value is not None:
        out = np.minimum(out, float(max_value) - np.asarray(values, dtype=np.float64))
    return out


def _wrapped_interval_score(
    values: np.ndarray,
    *,
    start: float,
    stop: float,
    period: float,
) -> np.ndarray:
    wrapped = np.mod(np.asarray(values, dtype=np.float64), float(period))
    start_v = float(start) % float(period)
    stop_v = float(stop) % float(period)
    if start_v <= stop_v:
        mask = (wrapped >= start_v) & (wrapped <= stop_v)
    else:
        mask = (wrapped >= start_v) | (wrapped <= stop_v)
    return np.where(mask, 1.0, -1.0)


def _timeline_absolute_time(timeline: CoverageTimeline) -> Any:
    if Time is None:
        raise RuntimeError("Absolute-time coverage helpers require astropy")
    if timeline.absolute_time is not None and isinstance(timeline.absolute_time, Time):
        return timeline.absolute_time
    if timeline.epoch is not None and isinstance(timeline.epoch, Time):
        return timeline.epoch + np.asarray(timeline.seconds, dtype=np.float64) * u.s
    raise ValueError(
        "This operation requires an absolute analysis timeline or a relative timeline with an astropy epoch"
    )


def _pair_indices(
    observer_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    n_targets: int,
) -> np.ndarray:
    return np.asarray(
        [
            int(obs) * int(n_targets) + int(tgt)
            for obs in observer_indices
            for tgt in target_indices
        ],
        dtype=np.int64,
    )


def _slice_pair_channel(
    channel: PairChannelStore,
    *,
    observer_indices: np.ndarray,
    target_indices: np.ndarray,
    n_targets_full: int,
) -> PairChannelStore:
    if channel.scope != "pair":
        raise ValueError(f"Channel {channel.name!r} is not a pair-scoped channel")
    pair_idx = _pair_indices(observer_indices, target_indices, n_targets=n_targets_full)
    return PairChannelStore(
        name=channel.name,
        scope=channel.scope,
        sample_times=channel.sample_times,
        values=np.asarray(channel.values[:, pair_idx], dtype=np.float64),
        n_observers=int(observer_indices.size),
        n_targets=int(target_indices.size),
    )


def _slice_target_channel(
    channel: PairChannelStore,
    *,
    target_indices: np.ndarray,
) -> PairChannelStore:
    if channel.scope != "target":
        raise ValueError(f"Channel {channel.name!r} is not a target-scoped channel")
    return PairChannelStore(
        name=channel.name,
        scope=channel.scope,
        sample_times=channel.sample_times,
        values=np.asarray(channel.values[:, target_indices], dtype=np.float64),
        n_targets=int(target_indices.size),
    )


def _compute_pair_channel(
    name: str,
    *,
    timeline: CoverageTimeline,
    observer_positions: Sequence[np.ndarray],
    targets: CoverageTargets,
) -> PairChannelStore:
    target_pos = np.asarray(targets.positions_ecef_m, dtype=np.float64)
    up = np.asarray(targets.up_vectors_ecef, dtype=np.float64)
    lat_rad = np.deg2rad(np.asarray(targets.lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(targets.lon_deg, dtype=np.float64))

    east = np.column_stack(
        (-np.sin(lon_rad), np.cos(lon_rad), np.zeros(targets.n_targets, dtype=np.float64))
    )
    north = np.column_stack(
        (
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad),
        )
    )

    nt = int(timeline.seconds.size)
    n_obs = len(observer_positions)
    values = np.empty((nt, n_obs * targets.n_targets), dtype=np.float64)
    target_pos_3d = target_pos[None, :, :]
    up_3d = up[None, :, :]
    east_3d = east[None, :, :]
    north_3d = north[None, :, :]

    for obs_idx, obs_pos in enumerate(observer_positions):
        rel = np.asarray(obs_pos, dtype=np.float64)[:, None, :] - target_pos_3d
        rng = np.linalg.norm(rel, axis=2)
        pair_slice = slice(obs_idx * targets.n_targets, (obs_idx + 1) * targets.n_targets)

        if name == "range":
            values[:, pair_slice] = rng
            continue

        if name == "elevation":
            safe_rng = np.maximum(rng, 1e-9)
            los_u = rel / safe_rng[:, :, None]
            dot = np.sum(los_u * up_3d, axis=2)
            dot = np.clip(dot, -1.0, 1.0)
            values[:, pair_slice] = np.rad2deg(np.arcsin(dot))
            continue

        if name == "azimuth":
            east_comp = np.sum(rel * east_3d, axis=2)
            north_comp = np.sum(rel * north_3d, axis=2)
            az_deg = (np.rad2deg(np.arctan2(east_comp, north_comp)) + 360.0) % 360.0
            values[:, pair_slice] = az_deg
            continue

        raise KeyError(f"Unsupported pair channel: {name!r}")

    return PairChannelStore(
        name=name,
        scope="pair",
        sample_times=np.asarray(timeline.seconds, dtype=np.float64),
        values=values,
        n_observers=n_obs,
        n_targets=targets.n_targets,
    )


def _compute_target_channel(
    name: str,
    *,
    timeline: CoverageTimeline,
    targets: CoverageTargets,
) -> PairChannelStore:
    nt = int(timeline.seconds.size)

    if name == "target_local_time":
        absolute_time = _timeline_absolute_time(timeline)
        unix = np.asarray(absolute_time.utc.unix, dtype=np.float64)
        utc_hours = np.mod(unix / 3600.0, 24.0)
        values = (utc_hours[:, None] + np.asarray(targets.lon_deg, dtype=np.float64)[None, :] / 15.0) % 24.0
        return PairChannelStore(
            name=name,
            scope="target",
            sample_times=np.asarray(timeline.seconds, dtype=np.float64),
            values=np.asarray(values, dtype=np.float64),
            n_targets=targets.n_targets,
        )

    if name == "target_sun_elevation":
        absolute_time = _timeline_absolute_time(timeline)
        jd_tt = np.asarray(absolute_time.tt.jd, dtype=np.float64)
        try:
            jd_ut1 = np.asarray(absolute_time.ut1.jd, dtype=np.float64)
        except Exception:
            jd_ut1 = np.asarray(absolute_time.utc.jd, dtype=np.float64)

        sun_pos = np.empty((nt, 3), dtype=np.float64)
        for idx in range(nt):
            x, y, z = sun_position_ecef(float(jd_ut1[idx]), float(jd_tt[idx]))
            sun_pos[idx, 0] = x
            sun_pos[idx, 1] = y
            sun_pos[idx, 2] = z

        rel = sun_pos[:, None, :] - np.asarray(targets.positions_ecef_m, dtype=np.float64)[None, :, :]
        rng = np.linalg.norm(rel, axis=2)
        safe_rng = np.maximum(rng, 1e-9)
        los_u = rel / safe_rng[:, :, None]
        dot = np.sum(
            los_u * np.asarray(targets.up_vectors_ecef, dtype=np.float64)[None, :, :],
            axis=2,
        )
        dot = np.clip(dot, -1.0, 1.0)
        values = np.rad2deg(np.arcsin(dot))
        return PairChannelStore(
            name=name,
            scope="target",
            sample_times=np.asarray(timeline.seconds, dtype=np.float64),
            values=np.asarray(values, dtype=np.float64),
            n_targets=targets.n_targets,
        )

    raise KeyError(f"Unsupported target channel: {name!r}")


@dataclass(frozen=True)
class CoverageTargetSelection:
    coverage: Any
    target_index: int
    label: str | None = None

    def timeline(self) -> TargetTimeline:
        return self.coverage.target_timeline(index=self.target_index, label=self.label)


class _ObserverSelection:
    def __init__(self, coverage: "IntervalCoverage | IntervalCoverageView") -> None:
        self._coverage = coverage

    def _resolve(self, selection: Iterable[int | str] | int | str) -> np.ndarray:
        if isinstance(selection, (int, np.integer, str)):
            items: list[int | str] = [selection]
        else:
            items = list(selection)

        resolved: list[int] = []
        names = [obs.name for obs in self._coverage.observer_items]
        for item in items:
            if isinstance(item, (int, np.integer)):
                resolved.append(int(item))
                continue
            if item in names:
                resolved.append(int(names.index(item)))
                continue
            raise KeyError(f"Unknown observer selection: {item!r}")
        return np.asarray(sorted(set(resolved)), dtype=np.int64)

    def only(self, selection: Iterable[int | str] | int | str) -> "IntervalCoverageView":
        return self._coverage._with_observer_selection(self._resolve(selection))

    def include(self, selection: Iterable[int | str] | int | str) -> "IntervalCoverageView":
        return self.only(selection)

    def exclude(self, selection: Iterable[int | str] | int | str) -> "IntervalCoverageView":
        remove = set(self._resolve(selection).tolist())
        keep = [
            idx
            for idx in range(len(self._coverage.observer_items))
            if idx not in remove
        ]
        return self._coverage._with_observer_selection(np.asarray(keep, dtype=np.int64))

    def by_tag(self, tag: str) -> "IntervalCoverageView":
        keep = [
            idx
            for idx, observer in enumerate(self._coverage.observer_items)
            if tag in observer.tags
        ]
        return self._coverage._with_observer_selection(np.asarray(keep, dtype=np.int64))

    def active(self, mask: Sequence[bool]) -> "IntervalCoverageView":
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (len(self._coverage.observer_items),):
            raise ValueError("Observer activity mask must match the current view length")
        return self._coverage._with_observer_selection(np.flatnonzero(arr))


class _TargetSelection:
    def __init__(self, coverage: "IntervalCoverage | IntervalCoverageView") -> None:
        self._coverage = coverage

    def _resolve(self, selection: Iterable[int] | int | np.ndarray) -> np.ndarray:
        if isinstance(selection, (int, np.integer)):
            return np.asarray([int(selection)], dtype=np.int64)
        arr = np.asarray(selection, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("Target selection must be 1D")
        return arr

    def only(self, selection: Iterable[int] | int | np.ndarray) -> "IntervalCoverageView":
        return self._coverage._with_target_selection(self._resolve(selection))

    def select(
        self,
        selection: Iterable[int] | int | np.ndarray | None = None,
        *,
        domain: TargetDomain | Any | None = None,
    ) -> "IntervalCoverageView":
        if domain is not None:
            domain_obj = domain if isinstance(domain, TargetDomain) else coerce_domain(domain)
            idx = self._coverage.target_set.select_domain(domain_obj)
            return self._coverage._with_target_selection(idx)
        if selection is None:
            raise ValueError("Provide either selection=... or domain=...")
        return self.only(selection)

    def exclude(self, selection: Iterable[int] | int | np.ndarray) -> "IntervalCoverageView":
        remove = set(self._resolve(selection).tolist())
        keep = [
            idx
            for idx in range(self._coverage.target_set.n_targets)
            if idx not in remove
        ]
        return self._coverage._with_target_selection(np.asarray(keep, dtype=np.int64))


class _CoverageAnalysisMixin:
    observers: _ObserverSelection
    targets: _TargetSelection
    diagnostics: CoverageDiagnostics

    @property
    def store(self) -> IntervalStore:
        raise NotImplementedError

    @property
    def raw_store(self) -> IntervalStore:
        raise NotImplementedError

    @property
    def observer_items(self) -> list[Observer]:
        raise NotImplementedError

    @property
    def target_set(self) -> CoverageTargets:
        raise NotImplementedError

    @property
    def timeline(self) -> CoverageTimeline:
        raise NotImplementedError

    @property
    def channels(self) -> dict[str, PairChannelStore]:
        raise NotImplementedError

    def _ensure_channels(self, names: set[str]) -> dict[str, PairChannelStore]:
        raise NotImplementedError

    def _channel_subset(self, name: str) -> PairChannelStore:
        raise NotImplementedError

    def _selected_window(self) -> tuple[float, float]:
        return (float(self.store.time_start), float(self.store.time_stop))

    def _with_observer_selection(self, observer_indices: np.ndarray) -> "IntervalCoverageView":
        raise NotImplementedError

    def _with_target_selection(self, target_indices: np.ndarray) -> "IntervalCoverageView":
        raise NotImplementedError

    def _wrap_metric_result(
        self,
        values: np.ndarray,
        *,
        metric_name: str,
        label: str | None,
        unit: str | None,
        dims: tuple[str, ...] = ("target",),
        coords: dict[str, np.ndarray] | None = None,
        fill_value: float | int | None = None,
    ) -> CoverageArray | CoverageField | CoverageStack:
        coord_map = dict(coords or {})
        if "target" in dims and "target" not in coord_map:
            coord_map["target"] = np.arange(self.target_set.n_targets, dtype=np.int64)
        window_start_s, window_stop_s = self._selected_window()
        return _wrap_array(
            values=np.asarray(values),
            dims=dims,
            coords=coord_map,
            targets=self.target_set if "target" in dims else None,
            unit=unit,
            label=label,
            attrs={"timeline_label": self.timeline.label},
            metric_name=metric_name,
            window_start_s=window_start_s,
            window_stop_s=window_stop_s,
            fill_value=fill_value,
        )

    def window(
        self,
        start: float | None = None,
        stop: float | None = None,
    ) -> "IntervalCoverageView":
        t0 = float(self.store.time_start if start is None else start)
        t1 = float(self.store.time_stop if stop is None else stop)
        if t1 <= t0:
            raise ValueError("window requires stop > start")
        base_start = (
            float(self._coverage.raw_store.time_start)
            if isinstance(self, IntervalCoverageView)
            else float(self.raw_store.time_start)
        )
        base_stop = (
            float(self._coverage.raw_store.time_stop)
            if isinstance(self, IntervalCoverageView)
            else float(self.raw_store.time_stop)
        )
        if t0 < base_start:
            raise ValueError("window start must be inside the computed coverage span")
        if t1 > base_stop:
            raise ValueError("window stop must be inside the computed coverage span")
        return IntervalCoverageView(
            coverage=self if isinstance(self, IntervalCoverage) else self._coverage,
            observer_indices=self._observer_indices if isinstance(self, IntervalCoverageView) else None,
            target_indices=self._target_indices if isinstance(self, IntervalCoverageView) else None,
            time_window=(t0, t1),
            constraints=self._extra_constraints if isinstance(self, IntervalCoverageView) else None,
        )

    def with_constraints(
        self,
        constraints: ConstraintSet | Sequence[Any] | Any,
    ) -> "IntervalCoverageView":
        extra = ConstraintSet.from_any(constraints)
        if isinstance(self, IntervalCoverageView):
            merged = ConstraintSet(items=self._extra_constraints.items + extra.items)
            return IntervalCoverageView(
                coverage=self._coverage,
                observer_indices=self._observer_indices,
                target_indices=self._target_indices,
                time_window=self._time_window,
                constraints=merged,
            )
        return IntervalCoverageView(coverage=self, constraints=extra)

    def target_timeline(
        self,
        *,
        index: int | None = None,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        label: str | None = None,
    ) -> TargetTimeline:
        if index is None:
            if lat_deg is None or lon_deg is None:
                raise ValueError("Provide either index=... or lat_deg=... and lon_deg=...")
            index = self.target_set.nearest_target_index(lat_deg=float(lat_deg), lon_deg=float(lon_deg))
        idx = int(index)
        starts_by_observer: list[np.ndarray] = []
        stops_by_observer: list[np.ndarray] = []
        for obs_idx in range(self.store.n_observers):
            starts, stops = self.store.pair_intervals(obs_idx, idx)
            starts_by_observer.append(starts.copy())
            stops_by_observer.append(stops.copy())
        target_label = label
        if target_label is None and self.target_set.labels is not None:
            target_label = self.target_set.labels[idx]
        if target_label is None:
            target_label = f"Target {idx}"
        return TargetTimeline(
            targets=self.target_set,
            target_index=idx,
            target_label=target_label,
            observer_names=[obs.name or f"observer_{i}" for i, obs in enumerate(self.observer_items)],
            starts_by_observer=starts_by_observer,
            stops_by_observer=stops_by_observer,
            time_start_s=float(self.store.time_start),
            time_stop_s=float(self.store.time_stop),
        )

    def target(
        self,
        *,
        index: int | None = None,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        label: str | None = None,
    ) -> CoverageTargetSelection:
        if index is None:
            if lat_deg is None or lon_deg is None:
                raise ValueError("Provide either index=... or lat_deg=... and lon_deg=...")
            index = self.target_set.nearest_target_index(lat_deg=float(lat_deg), lon_deg=float(lon_deg))
        return CoverageTargetSelection(coverage=self, target_index=int(index), label=label)

    def analyze(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(self, *args, **kwargs)

    def channel(self, name: str) -> PairChannelStore:
        self._ensure_channels({name})
        return self._channel_subset(name)

    def evaluate(self, metric: CompiledMetric) -> CoverageArray | CoverageField | CoverageStack:
        values = metric.kernel(
            np.asarray(self.store.pair_offsets, dtype=np.int64),
            np.asarray(self.store.start_times, dtype=np.float64),
            np.asarray(self.store.stop_times, dtype=np.float64),
            int(self.store.n_observers),
            int(self.store.n_targets),
            float(self.store.time_start),
            float(self.store.time_stop),
        )
        dims = tuple(metric.dims)
        coords = dict(metric.coords or {})
        if "target" in dims and "target" not in coords:
            coords["target"] = np.arange(self.target_set.n_targets, dtype=np.int64)
        return self._wrap_metric_result(
            np.asarray(values),
            metric_name=metric.name,
            label=metric.label or metric.name,
            unit=metric.unit,
            dims=dims,
            coords=coords,
        )

    def access_duration(
        self,
        *,
        min_assets: int | Sequence[int] = 1,
        normalize: str | bool | None = None,
        unit: TimeUnit = "seconds",
    ) -> CoverageField | CoverageStack:
        store = to_access_interval_store(self.store)
        if isinstance(min_assets, Sequence) and not isinstance(min_assets, (str, bytes)):
            n_values = np.asarray([int(v) for v in min_assets], dtype=np.int64)
            out = np.empty((n_values.size, self.target_set.n_targets), dtype=np.float64)
            for row_idx, n_req in enumerate(n_values):
                raw = access_duration_by_target(
                    store,
                    N=int(n_req),
                    normalize_to_day=bool(normalize == "day" or normalize is True),
                    reshape=False,
                )
                if str(normalize).lower() in {"fraction", "coverage_fraction"}:
                    raw, unit_label = _convert_duration(
                        raw,
                        window_s=float(self.store.time_stop - self.store.time_start),
                        unit=unit,
                        normalize=normalize,
                    )
                else:
                    scale, unit_label = _time_scale(unit)
                    raw = raw * scale
                    if normalize == "day" or normalize is True:
                        unit_label = f"{unit_label}/day"
                out[row_idx] = raw
            return self._wrap_metric_result(
                out,
                metric_name="access_duration",
                label="Access Duration",
                unit=unit_label,
                dims=("min_assets", "target"),
                coords={"min_assets": n_values},
            )  # type: ignore[return-value]

        raw = access_duration_by_target(
            store,
            N=int(min_assets),
            normalize_to_day=bool(normalize == "day" or normalize is True),
            reshape=False,
        )
        values, unit_label = _convert_duration(
            raw,
            window_s=float(self.store.time_stop - self.store.time_start),
            unit=unit,
            normalize=normalize if normalize not in {True, "day"} else ("day" if normalize else None),
        )
        return self._wrap_metric_result(
            values,
            metric_name="access_duration",
            label=f"Access Duration (min_assets={int(min_assets)})",
            unit=unit_label,
            fill_value=0.0,
        )  # type: ignore[return-value]

    def max_asset(self) -> CoverageField:
        values = calculate_max_asset(to_access_interval_store(self.store), reshape=False).astype(np.float64)
        return self._wrap_metric_result(
            values,
            metric_name="max_asset",
            label="Maximum Concurrent Observers",
            unit="count",
            fill_value=0.0,
        )  # type: ignore[return-value]

    def min_asset(self) -> CoverageField:
        values = calculate_min_asset(to_access_interval_store(self.store), reshape=False).astype(np.float64)
        return self._wrap_metric_result(
            values,
            metric_name="min_asset",
            label="Minimum Concurrent Observers",
            unit="count",
            fill_value=0.0,
        )  # type: ignore[return-value]

    def mtta(
        self,
        *,
        min_assets: int = 1,
        wrap: bool = False,
        fill_value: float = np.nan,
        unit: TimeUnit = "seconds",
    ) -> CoverageField:
        raw = calculate_mtta(
            to_access_interval_store(self.store),
            N=int(min_assets),
            wrap=bool(wrap),
            no_access_value=float(fill_value),
            reshape=False,
        )
        scale, unit_label = _time_scale(unit)
        return self._wrap_metric_result(
            raw * scale,
            metric_name="mtta",
            label=f"Mean Time To Access (min_assets={int(min_assets)})",
            unit=unit_label,
            fill_value=fill_value,
        )  # type: ignore[return-value]

    def gap_duration(
        self,
        *,
        min_assets: int = 1,
        statistic: str = "mean",
        include_end_gaps: bool = True,
        fill_value: float = np.nan,
        unit: TimeUnit = "seconds",
    ) -> CoverageField:
        raw = calculate_gap_duration(
            to_access_interval_store(self.store),
            min_assets=int(min_assets),
            stat=str(statistic),
            include_end_gaps=bool(include_end_gaps),
            no_access_value=float(fill_value),
            nan_if_never_access=True,
            reshape=False,
        )
        scale, unit_label = _time_scale(unit)
        return self._wrap_metric_result(
            np.asarray(raw, dtype=np.float64) * scale,
            metric_name="gap_duration",
            label=f"Gap Duration ({statistic}, min_assets={int(min_assets)})",
            unit=unit_label,
            fill_value=fill_value,
        )  # type: ignore[return-value]

    def revisit_time(
        self,
        *,
        min_assets: int = 1,
        statistic: str = "average",
        include_end_gaps: bool = True,
        unit: TimeUnit = "seconds",
    ) -> CoverageField:
        option_map = {
            "average": "average",
            "mean": "average",
            "maximum": "maximum",
            "max": "maximum",
            "minimum": "minimum",
            "min": "minimum",
            "std": "std_deviation",
            "std_deviation": "std_deviation",
        }
        raw = calculate_revisit_time(
            to_access_interval_store(self.store),
            N=int(min_assets),
            option=option_map[str(statistic).lower()],
            end_gaps="include" if include_end_gaps else "ignore",
            reshape=False,
        )
        scale, unit_label = _time_scale(unit)
        return self._wrap_metric_result(
            np.asarray(raw, dtype=np.float64) * scale,
            metric_name="revisit_time",
            label=f"Revisit Time ({statistic}, min_assets={int(min_assets)})",
            unit=unit_label,
            fill_value=float(self.store.time_stop - self.store.time_start) * scale,
        )  # type: ignore[return-value]

    def access_separation(
        self,
        *,
        min_assets: int = 1,
        min_separation_s: float = 0.0,
        max_separation_s: float = np.inf,
        fill_value: float | None = np.nan,
    ) -> CoverageField:
        values = calculate_access_separation(
            to_access_interval_store(self.store),
            min_assets=int(min_assets),
            min_separation_s=float(min_separation_s),
            max_separation_s=float(max_separation_s),
            no_access_value=fill_value,
            reshape=False,
        )
        return self._wrap_metric_result(
            np.asarray(values),
            metric_name="access_separation",
            label="Access Separation",
            unit="binary",
            fill_value=fill_value,
        )  # type: ignore[return-value]


class IntervalCoverage(_CoverageAnalysisMixin):
    def __init__(
        self,
        *,
        timeline: CoverageTimeline,
        target_set: CoverageTargets,
        observer_items: list[Observer],
        observer_positions: list[np.ndarray],
        observer_velocities: list[np.ndarray | None],
        raw_store: IntervalStore,
        store: IntervalStore,
        constraints: ConstraintSet,
        channels: dict[str, PairChannelStore],
        interpolation: str,
        root_tolerance_s: float,
        max_root_iterations: int,
    ) -> None:
        self._timeline = timeline
        self._target_set = target_set
        self._observer_items = list(observer_items)
        self._observer_positions = [np.asarray(arr, dtype=np.float64) for arr in observer_positions]
        self._observer_velocities = observer_velocities
        self._raw_store = raw_store
        self._store = store
        self._constraints = constraints
        self._channels = dict(channels)
        self.frame = _COVERAGE_ANALYSIS_FRAME
        self.interpolation = str(interpolation)
        self.root_tolerance_s = float(root_tolerance_s)
        self.max_root_iterations = int(max_root_iterations)
        self.observers = _ObserverSelection(self)
        self.targets = _TargetSelection(self)
        self.diagnostics = CoverageDiagnostics(self)

    @classmethod
    def compute(
        cls,
        *,
        timeline: CoverageTimeline | Any,
        observers: Sequence[Any] | Any | None = None,
        orbits: Sequence[Any] | Any | None = None,
        targets: CoverageTargets | TargetDomain | Any | None = None,
        domain: TargetDomain | Any | None = None,
        sampler: TargetSampler | None = None,
        constraints: ConstraintSet | Sequence[Any] | Any | None = None,
        interpolation: str = "cubic",
        root_tolerance_s: float = 1e-3,
        max_root_iterations: int = 64,
        channels: Iterable[str] | None = None,
    ) -> "IntervalCoverage":
        """Compute coverage against Earth-fixed targets.

        Coverage targets are represented in ITRF/ECEF, so orbit-backed observers are
        always sampled in ITRF internally.
        """
        timeline_obj = CoverageTimeline.from_any(timeline)
        target_set = _coerce_targets(targets, domain=domain, sampler=sampler)
        observer_input = observers if observers is not None else orbits
        if observer_input is None:
            raise ValueError("Provide observers=... or orbits=...")
        observer_positions, observer_velocities, observer_items = resolve_observers(
            observer_input,
            timeline_obj,
            frame=_COVERAGE_ANALYSIS_FRAME,
        )

        constraint_set = ConstraintSet.from_any(constraints)
        elevation = constraint_set.elevation()
        min_el = 0.0 if elevation is None or elevation.min_deg is None else float(elevation.min_deg)
        max_el = 90.0 if elevation is None or elevation.max_deg is None else float(elevation.max_deg)

        access_store = build_access_interval_store(
            time=np.asarray(timeline_obj.seconds, dtype=np.float64),
            observer_positions=observer_positions,
            target_positions=np.asarray(target_set.positions_ecef_m, dtype=np.float64),
            target_up_vectors=np.asarray(target_set.up_vectors_ecef, dtype=np.float64),
            min_elevation=min_el,
            max_elevation=max_el,
            degrees=True,
            interpolation=interpolation,
            root_tolerance_s=root_tolerance_s,
            max_root_iterations=max_root_iterations,
            target_shape=None,
            target_lon_deg=np.asarray(target_set.lon_deg, dtype=np.float64),
            target_lat_deg=np.asarray(target_set.lat_deg, dtype=np.float64),
        )
        raw_store = from_access_interval_store(access_store)

        coverage = cls(
            timeline=timeline_obj,
            target_set=target_set,
            observer_items=observer_items,
            observer_positions=observer_positions,
            observer_velocities=observer_velocities,
            raw_store=raw_store,
            store=raw_store,
            constraints=constraint_set,
            channels={},
            interpolation=interpolation,
            root_tolerance_s=root_tolerance_s,
            max_root_iterations=max_root_iterations,
        )

        requested_channels = set(channels or ()) | constraint_set.requested_channels()
        if requested_channels:
            coverage._ensure_channels(requested_channels)

        coverage._store = coverage._apply_constraints(
            coverage._raw_store,
            constraint_set,
            observer_indices=np.arange(len(observer_items), dtype=np.int64),
            target_indices=np.arange(target_set.n_targets, dtype=np.int64),
            skip_elevation=True,
        )
        return coverage

    @classmethod
    def from_orbits(
        cls,
        *,
        orbits: Sequence[Any] | Any,
        timeline: CoverageTimeline | Any,
        targets: CoverageTargets | TargetDomain | Any | None = None,
        domain: TargetDomain | Any | None = None,
        sampler: TargetSampler | None = None,
        constraints: ConstraintSet | Sequence[Any] | Any | None = None,
        interpolation: str = "cubic",
        root_tolerance_s: float = 1e-3,
        max_root_iterations: int = 64,
        channels: Iterable[str] | None = None,
    ) -> "IntervalCoverage":
        """Compute coverage for orbit-backed observers sampled in ITRF internally."""
        return cls.compute(
            timeline=timeline,
            orbits=orbits,
            targets=targets,
            domain=domain,
            sampler=sampler,
            constraints=constraints,
            interpolation=interpolation,
            root_tolerance_s=root_tolerance_s,
            max_root_iterations=max_root_iterations,
            channels=channels,
        )

    @property
    def timeline(self) -> CoverageTimeline:
        return self._timeline

    @property
    def target_set(self) -> CoverageTargets:
        return self._target_set

    @property
    def observer_items(self) -> list[Observer]:
        return list(self._observer_items)

    @property
    def raw_store(self) -> IntervalStore:
        return self._raw_store

    @property
    def store(self) -> IntervalStore:
        return self._store

    @property
    def channels(self) -> dict[str, PairChannelStore]:
        return self._channels

    @property
    def constraints(self) -> ConstraintSet:
        return self._constraints

    @property
    def sampled_positions(self) -> list[np.ndarray]:
        return [arr.copy() for arr in self._observer_positions]

    def channel(self, name: str) -> PairChannelStore:
        return self._ensure_channels({name})[name]

    def _ensure_channels(self, names: set[str]) -> dict[str, PairChannelStore]:
        for name in sorted(names):
            if name in self._channels:
                continue
            if name in {"elevation", "azimuth", "range"}:
                self._channels[name] = _compute_pair_channel(
                    name,
                    timeline=self._timeline,
                    observer_positions=self._observer_positions,
                    targets=self._target_set,
                )
                continue
            if name in {"target_sun_elevation", "target_local_time"}:
                self._channels[name] = _compute_target_channel(
                    name,
                    timeline=self._timeline,
                    targets=self._target_set,
                )
                continue
            raise KeyError(f"Unsupported channel: {name!r}")
        return self._channels

    def _channel_subset(self, name: str) -> PairChannelStore:
        return self.channel(name)

    def _apply_constraints(
        self,
        store: IntervalStore,
        constraints: ConstraintSet,
        *,
        observer_indices: np.ndarray,
        target_indices: np.ndarray,
        skip_elevation: bool = False,
    ) -> IntervalStore:
        out = store

        for item in constraints.pair_constraints():
            if isinstance(item, ElevationConstraint):
                if skip_elevation:
                    continue
                channel = _slice_pair_channel(
                    self._ensure_channels({"elevation"})["elevation"],
                    observer_indices=observer_indices,
                    target_indices=target_indices,
                    n_targets_full=self._target_set.n_targets,
                )
                score = _bounded_score(
                    channel.values,
                    min_value=item.min_deg,
                    max_value=item.max_deg,
                )
            elif isinstance(item, RangeConstraint):
                channel = _slice_pair_channel(
                    self._ensure_channels({"range"})["range"],
                    observer_indices=observer_indices,
                    target_indices=target_indices,
                    n_targets_full=self._target_set.n_targets,
                )
                score = _bounded_score(
                    channel.values,
                    min_value=item.min_m,
                    max_value=item.max_m,
                )
            elif isinstance(item, AzimuthConstraint):
                channel = _slice_pair_channel(
                    self._ensure_channels({"azimuth"})["azimuth"],
                    observer_indices=observer_indices,
                    target_indices=target_indices,
                    n_targets_full=self._target_set.n_targets,
                )
                if item.min_deg is not None and item.max_deg is not None:
                    score = _wrapped_interval_score(
                        channel.values,
                        start=float(item.min_deg),
                        stop=float(item.max_deg),
                        period=360.0,
                    )
                else:
                    score = _bounded_score(
                        channel.values,
                        min_value=item.min_deg,
                        max_value=item.max_deg,
                    )
            else:  # pragma: no cover - defensive fallback
                continue

            gate = build_pair_gate_store_from_scores(
                np.asarray(self._timeline.seconds, dtype=np.float64),
                np.asarray(score, dtype=np.float64),
                n_observers=int(observer_indices.size),
                n_targets=int(target_indices.size),
            )
            out = intersect_pair_gate_store(out, gate)

        for item in constraints.target_gates():
            if isinstance(item, TargetSunElevationConstraint):
                channel = _slice_target_channel(
                    self._ensure_channels({"target_sun_elevation"})["target_sun_elevation"],
                    target_indices=target_indices,
                )
                score = _bounded_score(
                    channel.values,
                    min_value=item.min_deg,
                    max_value=item.max_deg,
                )
            elif isinstance(item, TargetLocalTimeConstraint):
                channel = _slice_target_channel(
                    self._ensure_channels({"target_local_time"})["target_local_time"],
                    target_indices=target_indices,
                )
                score = _wrapped_interval_score(
                    channel.values,
                    start=float(item.start_hour),
                    stop=float(item.stop_hour),
                    period=24.0,
                )
            else:  # pragma: no cover - defensive fallback
                continue

            gate = build_time_gate_store_from_scores(
                np.asarray(self._timeline.seconds, dtype=np.float64),
                np.asarray(score, dtype=np.float64),
                scope="target",
            )
            out = intersect_target_gates(out, gate)

        for item in constraints.post_interval():
            if isinstance(item, MinAccessDurationConstraint):
                out = filter_min_duration(out, float(item.min_seconds))
        return out

    def _with_observer_selection(self, observer_indices: np.ndarray) -> "IntervalCoverageView":
        return IntervalCoverageView(coverage=self, observer_indices=observer_indices)

    def _with_target_selection(self, target_indices: np.ndarray) -> "IntervalCoverageView":
        return IntervalCoverageView(coverage=self, target_indices=target_indices)


class IntervalCoverageView(_CoverageAnalysisMixin):
    def __init__(
        self,
        *,
        coverage: IntervalCoverage,
        observer_indices: np.ndarray | None = None,
        target_indices: np.ndarray | None = None,
        time_window: tuple[float, float] | None = None,
        constraints: ConstraintSet | Sequence[Any] | Any | None = None,
    ) -> None:
        self._coverage = coverage
        self._observer_indices = (
            np.arange(coverage.store.n_observers, dtype=np.int64)
            if observer_indices is None
            else np.asarray(observer_indices, dtype=np.int64)
        )
        self._target_indices = (
            np.arange(coverage.store.n_targets, dtype=np.int64)
            if target_indices is None
            else np.asarray(target_indices, dtype=np.int64)
        )
        self._time_window = (
            (float(coverage.store.time_start), float(coverage.store.time_stop))
            if time_window is None
            else (float(time_window[0]), float(time_window[1]))
        )
        self._extra_constraints = ConstraintSet.from_any(constraints)
        self._store_cache: IntervalStore | None = None
        self._raw_store_cache: IntervalStore | None = None
        self._target_set_cache: CoverageTargets | None = None
        self.observers = _ObserverSelection(self)
        self.targets = _TargetSelection(self)
        self.diagnostics = CoverageDiagnostics(self)

    @property
    def timeline(self) -> CoverageTimeline:
        return self._coverage.timeline

    @property
    def target_set(self) -> CoverageTargets:
        if self._target_set_cache is None:
            self._target_set_cache = self._coverage.target_set.subset(self._target_indices)
        return self._target_set_cache

    @property
    def observer_items(self) -> list[Observer]:
        return [self._coverage.observer_items[int(idx)] for idx in self._observer_indices]

    @property
    def raw_store(self) -> IntervalStore:
        if self._raw_store_cache is None:
            self._raw_store_cache = subset_interval_store(
                self._coverage.raw_store,
                observer_indices=self._observer_indices,
                target_indices=self._target_indices,
                time_window=self._time_window,
            )
        return self._raw_store_cache

    @property
    def store(self) -> IntervalStore:
        if self._store_cache is None:
            base = subset_interval_store(
                self._coverage.store,
                observer_indices=self._observer_indices,
                target_indices=self._target_indices,
                time_window=self._time_window,
            )
            self._store_cache = self._coverage._apply_constraints(
                base,
                self._extra_constraints,
                observer_indices=self._observer_indices,
                target_indices=self._target_indices,
                skip_elevation=False,
            )
        return self._store_cache

    @property
    def channels(self) -> dict[str, PairChannelStore]:
        return {
            name: self._channel_subset(name)
            for name in self._coverage.channels
        }

    def _channel_subset(self, name: str) -> PairChannelStore:
        channel = self._coverage.channel(name)
        if channel.scope == "pair":
            return _slice_pair_channel(
                channel,
                observer_indices=self._observer_indices,
                target_indices=self._target_indices,
                n_targets_full=self._coverage.target_set.n_targets,
            )
        if channel.scope == "target":
            return _slice_target_channel(channel, target_indices=self._target_indices)
        raise KeyError(f"Unsupported channel scope: {channel.scope!r}")

    def _ensure_channels(self, names: set[str]) -> dict[str, PairChannelStore]:
        self._coverage._ensure_channels(names)
        return {name: self._channel_subset(name) for name in names}

    def _with_observer_selection(self, observer_indices: np.ndarray) -> "IntervalCoverageView":
        return IntervalCoverageView(
            coverage=self._coverage,
            observer_indices=self._observer_indices[observer_indices],
            target_indices=self._target_indices,
            time_window=self._time_window,
            constraints=self._extra_constraints,
        )

    def _with_target_selection(self, target_indices: np.ndarray) -> "IntervalCoverageView":
        return IntervalCoverageView(
            coverage=self._coverage,
            observer_indices=self._observer_indices,
            target_indices=self._target_indices[target_indices],
            time_window=self._time_window,
            constraints=self._extra_constraints,
        )


__all__ = [
    "CoverageTargetSelection",
    "IntervalCoverage",
    "IntervalCoverageView",
]
