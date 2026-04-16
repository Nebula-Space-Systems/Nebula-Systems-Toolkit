from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - exercised when astropy is installed
    import astropy.units as u
    from astropy.time import Time
except Exception:  # pragma: no cover - fallback when astropy extras are absent
    u = None
    Time = None  # type: ignore[assignment]

from .timeline import CoverageTimeline


def _coerce_time_axis(value: Any) -> np.ndarray:
    if Time is not None and isinstance(value, Time):
        return np.asarray(value.utc.unix, dtype=np.float64)
    if u is not None and isinstance(value, u.Quantity):
        return np.asarray(value.to_value(u.s), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


class ObserverSource:
    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        raise NotImplementedError

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        return None


@dataclass(frozen=True)
class OrbitObserverSource(ObserverSource):
    orbit: Any
    frame: str = "itrf"
    use_velocity: bool = True
    precompute: bool = False

    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        requested = frame or self.frame
        if self.precompute and hasattr(self.orbit, "precompute"):
            self.orbit.precompute(float(timeline.start), float(timeline.stop))
        getter = getattr(self.orbit, "get_position", None)
        if getter is None:
            getter = getattr(self.orbit, "get_p")
        if timeline.absolute_time is not None:
            return np.asarray(
                getter(timeline.absolute_time, frame=requested),
                dtype=np.float64,
            )
        return np.asarray(
            getter(timeline.seconds, frame=requested),
            dtype=np.float64,
        )

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        if not self.use_velocity:
            return None
        getter = getattr(self.orbit, "get_velocity", None)
        if getter is None:
            getter = getattr(self.orbit, "get_v", None)
        if getter is None:
            return None
        requested = frame or self.frame
        if timeline.absolute_time is not None:
            return np.asarray(
                getter(timeline.absolute_time, frame=requested),
                dtype=np.float64,
            )
        return np.asarray(
            getter(timeline.seconds, frame=requested),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SampledObserverSource(ObserverSource):
    positions: np.ndarray
    velocities: np.ndarray | None = None

    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        arr = np.asarray(self.positions, dtype=np.float64)
        if arr.shape != (timeline.seconds.size, 3):
            raise ValueError(
                "Sampled observer positions must match the analysis timeline shape (len(time), 3)"
            )
        return arr

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        if self.velocities is None:
            return None
        arr = np.asarray(self.velocities, dtype=np.float64)
        if arr.shape != (timeline.seconds.size, 3):
            raise ValueError(
                "Sampled observer velocities must match the analysis timeline shape (len(time), 3)"
            )
        return arr


@dataclass(frozen=True)
class TabulatedObserverSource(ObserverSource):
    time: Any
    positions: np.ndarray
    velocities: np.ndarray | None = None

    def _interpolate(self, values: np.ndarray, timeline: CoverageTimeline) -> np.ndarray:
        source_axis = _coerce_time_axis(self.time)
        target_axis = timeline.absolute_seconds()
        if target_axis is None:
            target_axis = timeline.seconds
            if Time is not None and isinstance(self.time, Time):
                raise ValueError(
                    "Absolute tabulated observer times require an absolute analysis timeline"
                )
        out = np.empty((target_axis.size, values.shape[1]), dtype=np.float64)
        for col in range(values.shape[1]):
            out[:, col] = np.interp(target_axis, source_axis, values[:, col])
        return out

    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        arr = np.asarray(self.positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Tabulated observer positions must have shape (nt, 3)")
        return self._interpolate(arr, timeline)

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        if self.velocities is None:
            return None
        arr = np.asarray(self.velocities, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Tabulated observer velocities must have shape (nt, 3)")
        return self._interpolate(arr, timeline)


@dataclass(frozen=True)
class CallableObserverSource(ObserverSource):
    sampler: Callable[..., np.ndarray]
    velocity_sampler: Callable[..., np.ndarray] | None = None

    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        out = self.sampler(timeline, frame=frame)
        arr = np.asarray(out, dtype=np.float64)
        if arr.shape != (timeline.seconds.size, 3):
            raise ValueError("Callable observer sampler must return shape (len(time), 3)")
        return arr

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        if self.velocity_sampler is None:
            return None
        out = self.velocity_sampler(timeline, frame=frame)
        arr = np.asarray(out, dtype=np.float64)
        if arr.shape != (timeline.seconds.size, 3):
            raise ValueError("Callable velocity sampler must return shape (len(time), 3)")
        return arr


@dataclass(frozen=True)
class Observer:
    source: ObserverSource
    name: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_orbit(
        cls,
        orbit: Any,
        *,
        frame: str = "itrf",
        name: str | None = None,
        tags: Sequence[str] = (),
        use_velocity: bool = True,
        precompute: bool = False,
    ) -> "Observer":
        return cls(
            source=OrbitObserverSource(
                orbit=orbit,
                frame=frame,
                use_velocity=use_velocity,
                precompute=precompute,
            ),
            name=name,
            tags=tuple(tags),
        )

    @classmethod
    def from_samples(
        cls,
        positions: np.ndarray,
        *,
        velocities: np.ndarray | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
    ) -> "Observer":
        return cls(
            source=SampledObserverSource(positions=np.asarray(positions), velocities=velocities),
            name=name,
            tags=tuple(tags),
        )

    @classmethod
    def from_tabulated(
        cls,
        time: Any,
        positions: np.ndarray,
        *,
        velocities: np.ndarray | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
    ) -> "Observer":
        return cls(
            source=TabulatedObserverSource(time=time, positions=np.asarray(positions), velocities=velocities),
            name=name,
            tags=tuple(tags),
        )

    @classmethod
    def from_callable(
        cls,
        sampler: Callable[..., np.ndarray],
        *,
        velocity_sampler: Callable[..., np.ndarray] | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
    ) -> "Observer":
        return cls(
            source=CallableObserverSource(sampler=sampler, velocity_sampler=velocity_sampler),
            name=name,
            tags=tuple(tags),
        )


def coerce_observer(value: Any) -> Observer:
    if isinstance(value, Observer):
        return value
    if hasattr(value, "get_position") or hasattr(value, "get_p"):
        return Observer.from_orbit(value, name=getattr(value, "name", None))
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return Observer.from_samples(arr)
    raise TypeError(f"Unsupported observer input: {type(value)!r}")


def resolve_observers(
    observers: Sequence[Any] | Any,
    timeline: CoverageTimeline,
    *,
    frame: str = "itrf",
) -> tuple[list[np.ndarray], list[np.ndarray | None], list[Observer]]:
    if isinstance(observers, np.ndarray) and observers.ndim == 2 and observers.shape[1] == 3:
        seq = [observers]
    elif isinstance(observers, Sequence) and not isinstance(observers, (bytes, str)):
        seq = list(observers)
    else:
        seq = [observers]
    resolved = [coerce_observer(item) for item in seq]
    positions = [obs.source.sample_positions(timeline, frame=frame) for obs in resolved]
    velocities = [obs.source.sample_velocities(timeline, frame=frame) for obs in resolved]
    return positions, velocities, resolved


__all__ = [
    "ObserverSource",
    "Observer",
    "coerce_observer",
    "resolve_observers",
]
