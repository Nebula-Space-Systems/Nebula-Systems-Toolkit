from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .domains import TargetDomain, coerce_domain
from .targets import CoverageTargets


def _weighted_reduce(values: np.ndarray, op: str, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return float("nan")
    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64).ravel()[mask]
        if not np.any(weights_arr > 0.0):
            weights_arr = None

    key = op.lower()
    if key in {"mean", "average"}:
        if weights_arr is None:
            return float(np.mean(arr))
        return float(np.average(arr, weights=weights_arr))
    if key == "min":
        return float(np.min(arr))
    if key == "max":
        return float(np.max(arr))
    if key == "std":
        if weights_arr is None:
            return float(np.std(arr))
        mean = float(np.average(arr, weights=weights_arr))
        var = float(np.average((arr - mean) ** 2, weights=weights_arr))
        return float(np.sqrt(var))
    if key.startswith("p"):
        q = float(key[1:])
        return float(np.percentile(arr, q))
    raise ValueError(f"Unsupported reduction op: {op!r}")


@dataclass(frozen=True)
class CoverageResult:
    targets: CoverageTargets | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_numpy(self, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def to_records(self) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class CoverageArray(CoverageResult):
    values: np.ndarray = field(default_factory=lambda: np.empty(0))
    dims: tuple[str, ...] = field(default_factory=tuple)
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    unit: str | None = None
    label: str | None = None

    def to_numpy(self, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.values)

    def to_records(self) -> list[dict[str, Any]]:
        return [{"index": int(i), "value": float(v)} for i, v in enumerate(np.asarray(self.values).ravel())]

    def sel(self, **coords: Any) -> "CoverageArray | CoverageField | CoverageStack":
        values = np.asarray(self.values)
        dims = list(self.dims)
        new_coords = {key: np.asarray(val) for key, val in self.coords.items()}
        for dim_name, wanted in coords.items():
            if dim_name not in dims:
                raise KeyError(f"{dim_name!r} is not a dimension on this array")
            axis = dims.index(dim_name)
            coord_values = np.asarray(new_coords[dim_name])
            matches = np.where(coord_values == wanted)[0]
            if matches.size != 1:
                raise KeyError(f"Could not select a unique value for {dim_name}={wanted!r}")
            values = np.take(values, int(matches[0]), axis=axis)
            dims.pop(axis)
            new_coords.pop(dim_name)
        return _wrap_array(
            values=values,
            dims=tuple(dims),
            coords=new_coords,
            targets=self.targets,
            unit=self.unit,
            label=self.label,
            attrs=self.attrs,
            metric_name=getattr(self, "metric_name", "coverage_metric"),
            window_start_s=getattr(self, "window_start_s", 0.0),
            window_stop_s=getattr(self, "window_stop_s", 0.0),
            fill_value=getattr(self, "fill_value", None),
        )

    def reduce(
        self,
        dim: str,
        op: str,
        *,
        weights: np.ndarray | None = None,
    ) -> "CoverageArray | CoverageField | CoverageStack":
        if dim not in self.dims:
            raise KeyError(f"{dim!r} is not a dimension on this array")
        axis = self.dims.index(dim)
        values = np.asarray(self.values, dtype=np.float64)
        if op.lower() in {"mean", "average"} and weights is not None:
            result = np.average(values, axis=axis, weights=np.asarray(weights, dtype=np.float64))
        elif op.lower() == "min":
            result = np.min(values, axis=axis)
        elif op.lower() == "max":
            result = np.max(values, axis=axis)
        elif op.lower() == "std":
            result = np.std(values, axis=axis)
        elif op.lower().startswith("p"):
            result = np.percentile(values, float(op.lower()[1:]), axis=axis)
        else:
            raise ValueError(f"Unsupported reduction op: {op!r}")
        dims = tuple(name for name in self.dims if name != dim)
        coords = {key: value for key, value in self.coords.items() if key != dim}
        return _wrap_array(
            values=result,
            dims=dims,
            coords=coords,
            targets=self.targets if "target" in dims else None,
            unit=self.unit,
            label=self.label,
            attrs=self.attrs,
            metric_name=getattr(self, "metric_name", "coverage_metric"),
            window_start_s=getattr(self, "window_start_s", 0.0),
            window_stop_s=getattr(self, "window_stop_s", 0.0),
            fill_value=getattr(self, "fill_value", None),
        )

    def to_xarray(self) -> Any:
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - xarray is optional
            raise RuntimeError("xarray is not installed") from exc
        return xr.DataArray(self.values, dims=self.dims, coords=self.coords, attrs=dict(self.attrs))

    def plot_histogram(self, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_coverage_histogram

        return plot_coverage_histogram(self, **kwargs)

    def plot_ecdf(self, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_coverage_ecdf

        return plot_coverage_ecdf(self, **kwargs)


@dataclass(frozen=True)
class CoverageField(CoverageArray):
    metric_name: str = "coverage_metric"
    window_start_s: float = 0.0
    window_stop_s: float = 0.0
    fill_value: float | int | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError("CoverageField.values must be 1D")
        if self.targets is None:
            raise ValueError("CoverageField requires CoverageTargets")
        if values.shape != (self.targets.n_targets,):
            raise ValueError("CoverageField.values must match the number of targets")
        if not self.dims:
            object.__setattr__(self, "dims", ("target",))
        if not self.coords:
            object.__setattr__(self, "coords", {"target": np.arange(self.targets.n_targets, dtype=np.int64)})

    def plot_map(self, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
        from .plotting import plot_coverage_map

        return plot_coverage_map(self, **kwargs)

    def plot(self, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
        return self.plot_map(**kwargs)

    def to_records(self) -> list[dict[str, Any]]:
        assert self.targets is not None
        labels = self.targets.labels or [None] * self.targets.n_targets
        return [
            {
                "target_index": idx,
                "lat_deg": float(self.targets.lat_deg[idx]),
                "lon_deg": float(self.targets.lon_deg[idx]),
                "value": float(self.values[idx]) if np.issubdtype(np.asarray(self.values).dtype, np.floating) else self.values[idx],
                "area_weight": float(self.targets.area_weights[idx]),
                "label": labels[idx],
            }
            for idx in range(self.targets.n_targets)
        ]

    def at_index(self, target_index: int) -> float | int:
        return self.values[int(target_index)].item()

    def at(self, *, lat_deg: float, lon_deg: float) -> float | int:
        assert self.targets is not None
        idx = self.targets.nearest_target_index(lat_deg=lat_deg, lon_deg=lon_deg)
        return self.at_index(idx)

    def summary(self) -> dict[str, float]:
        vals = np.asarray(self.values, dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        return {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        }

    def reduce_targets(self, op: str, *, weights: str | np.ndarray | None = "area") -> float:
        assert self.targets is not None
        if isinstance(weights, str) and weights == "area":
            weights_arr = self.targets.area_weights
        elif weights is None:
            weights_arr = None
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
        return _weighted_reduce(np.asarray(self.values, dtype=np.float64), op, weights_arr)

    def reduce_region(
        self,
        domain: TargetDomain | Any,
        *,
        op: str = "mean",
        weights: str | np.ndarray | None = "area",
    ) -> float:
        assert self.targets is not None
        region = coerce_domain(domain) if not isinstance(domain, TargetDomain) else domain
        idx = self.targets.select_domain(region)
        if idx.size == 0:
            return float("nan")
        values = np.asarray(self.values, dtype=np.float64)[idx]
        if isinstance(weights, str) and weights == "area":
            weights_arr = self.targets.area_weights[idx]
        elif weights is None:
            weights_arr = None
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)[idx]
        return _weighted_reduce(values, op, weights_arr)

    def threshold(self, threshold: float) -> "CoverageField":
        values = (np.asarray(self.values, dtype=np.float64) >= float(threshold)).astype(np.float64)
        return CoverageField(
            targets=self.targets,
            values=values,
            metric_name=f"{self.metric_name}_threshold",
            window_start_s=self.window_start_s,
            window_stop_s=self.window_stop_s,
            unit="fraction",
            label=f"{self.metric_name} >= {threshold}",
            fill_value=0.0,
            attrs=dict(self.attrs),
        )

    def covered_fraction(self, *, weights: str | np.ndarray | None = "area") -> float:
        assert self.targets is not None
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.isfinite(values)
        active = values[mask] > 0.0
        if isinstance(weights, str) and weights == "area":
            w = self.targets.area_weights[mask]
        elif weights is None:
            w = np.ones(int(mask.sum()), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)[mask]
        return float(np.sum(w[active]) / np.sum(w))


@dataclass(frozen=True)
class CoverageStack(CoverageArray):
    metric_name: str = "coverage_metric"

    def __post_init__(self) -> None:
        if "target" not in self.dims:
            raise ValueError("CoverageStack must include a 'target' dimension")

    def plot_small_multiples(self, *, dim: str, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_coverage_small_multiples

        return plot_coverage_small_multiples(self, dim=dim, **kwargs)


@dataclass(frozen=True)
class TargetTimeline(CoverageResult):
    target_index: int = 0
    target_label: str | None = None
    observer_names: list[str] | None = None
    starts_by_observer: list[np.ndarray] = field(default_factory=list)
    stops_by_observer: list[np.ndarray] = field(default_factory=list)
    time_start_s: float = 0.0
    time_stop_s: float = 0.0

    def to_numpy(self, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.to_records(), dtype=object)

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        names = self.observer_names or [str(i) for i in range(len(self.starts_by_observer))]
        for obs_idx, (starts, stops) in enumerate(zip(self.starts_by_observer, self.stops_by_observer)):
            for start, stop in zip(starts, stops):
                records.append(
                    {
                        "observer_index": obs_idx,
                        "observer_name": names[obs_idx],
                        "start_s": float(start),
                        "stop_s": float(stop),
                        "duration_s": float(stop - start),
                    }
                )
        return records

    def concurrency_profile(self) -> tuple[np.ndarray, np.ndarray]:
        events: dict[float, int] = {}
        active0 = 0
        for starts, stops in zip(self.starts_by_observer, self.stops_by_observer):
            for start, stop in zip(starts, stops):
                if float(start) <= self.time_start_s < float(stop):
                    active0 += 1
                if self.time_start_s < float(start) < self.time_stop_s:
                    events[float(start)] = events.get(float(start), 0) + 1
                if self.time_start_s < float(stop) < self.time_stop_s:
                    events[float(stop)] = events.get(float(stop), 0) - 1
        times = [float(self.time_start_s)]
        counts = [int(active0)]
        current = int(active0)
        for event_time in sorted(events):
            times.extend([event_time, event_time])
            counts.extend([current, current + events[event_time]])
            current += events[event_time]
        times.append(float(self.time_stop_s))
        counts.append(current)
        return np.asarray(times, dtype=np.float64), np.asarray(counts, dtype=np.int32)

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_target_timeline

        return plot_target_timeline(self, **kwargs)


def _wrap_array(
    *,
    values: np.ndarray,
    dims: tuple[str, ...],
    coords: dict[str, np.ndarray],
    targets: CoverageTargets | None,
    unit: str | None,
    label: str | None,
    attrs: dict[str, Any],
    metric_name: str = "coverage_metric",
    window_start_s: float = 0.0,
    window_stop_s: float = 0.0,
    fill_value: float | int | None = None,
) -> CoverageArray | CoverageField | CoverageStack:
    array = np.asarray(values)
    if dims == ("target",):
        return CoverageField(
            targets=targets,
            values=array.astype(np.float64 if np.issubdtype(array.dtype, np.floating) else array.dtype),
            dims=dims,
            coords=coords,
            unit=unit,
            label=label,
            attrs=dict(attrs),
            metric_name=metric_name,
            window_start_s=float(window_start_s),
            window_stop_s=float(window_stop_s),
            fill_value=fill_value,
        )
    if "target" in dims:
        return CoverageStack(
            targets=targets,
            values=array,
            dims=dims,
            coords=coords,
            unit=unit,
            label=label,
            attrs=dict(attrs),
            metric_name=metric_name,
        )
    return CoverageArray(
        targets=targets,
        values=array,
        dims=dims,
        coords=coords,
        unit=unit,
        label=label,
        attrs=dict(attrs),
    )


__all__ = [
    "CoverageResult",
    "CoverageArray",
    "CoverageField",
    "CoverageStack",
    "TargetTimeline",
]
