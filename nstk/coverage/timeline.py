from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when astropy is installed
    import astropy.units as u
    from astropy.time import Time
except Exception:  # pragma: no cover - fallback when astropy extras are absent
    u = None
    Time = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CoverageTimeline:
    """Normalized analysis time axis for coverage workflows."""

    seconds: np.ndarray
    epoch: Any | None = None
    absolute_time: Any | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        secs = np.asarray(self.seconds, dtype=np.float64)
        if secs.ndim != 1 or secs.size < 2:
            raise ValueError("CoverageTimeline.seconds must be a 1D array with length >= 2")
        if not np.all(np.isfinite(secs)):
            raise ValueError("CoverageTimeline.seconds must be finite")
        if np.any(np.diff(secs) <= 0.0):
            raise ValueError("CoverageTimeline.seconds must be strictly increasing")
        object.__setattr__(self, "seconds", np.ascontiguousarray(secs))

    @classmethod
    def absolute(cls, time: Any, *, label: str | None = None) -> "CoverageTimeline":
        if Time is None or not isinstance(time, Time):
            raise TypeError("CoverageTimeline.absolute requires astropy.time.Time")
        if getattr(time, "shape", None) == ():
            raise ValueError("CoverageTimeline.absolute requires a vector of times")
        unix = np.asarray(time.utc.unix, dtype=np.float64)
        seconds = unix - float(unix[0])
        return cls(
            seconds=seconds,
            epoch=time[0],
            absolute_time=time,
            label=label,
        )

    @classmethod
    def relative(
        cls,
        seconds: Any,
        *,
        epoch: Any | None = None,
        label: str | None = None,
    ) -> "CoverageTimeline":
        if u is not None and isinstance(seconds, u.Quantity):
            values = np.asarray(seconds.to_value(u.s), dtype=np.float64)
        else:
            values = np.asarray(seconds, dtype=np.float64)
        return cls(seconds=values, epoch=epoch, absolute_time=None, label=label)

    @classmethod
    def linspace(
        cls,
        start: Any,
        stop: Any,
        step: Any,
        *,
        label: str | None = None,
    ) -> "CoverageTimeline":
        if Time is not None and isinstance(start, Time):
            if not isinstance(stop, Time):
                raise TypeError("stop must also be astropy.time.Time when start is Time")
            if u is not None and isinstance(step, u.Quantity):
                step_s = float(step.to_value(u.s))
            else:
                step_s = float(step)
            duration_s = float((stop - start).to_value(u.s))
            count = int(np.floor(duration_s / step_s + 1e-12)) + 1
            values = start + np.arange(count, dtype=np.float64) * step_s * u.s
            return cls.absolute(values, label=label)

        if u is not None and isinstance(step, u.Quantity):
            step_s = float(step.to_value(u.s))
        else:
            step_s = float(step)
        start_s = float(start)
        stop_s = float(stop)
        count = int(np.floor((stop_s - start_s) / step_s + 1e-12)) + 1
        return cls.relative(
            start_s + np.arange(count, dtype=np.float64) * step_s,
            label=label,
        )

    @classmethod
    def from_any(
        cls,
        value: Any,
        *,
        epoch: Any | None = None,
        label: str | None = None,
    ) -> "CoverageTimeline":
        if isinstance(value, CoverageTimeline):
            return value
        if Time is not None and isinstance(value, Time):
            return cls.absolute(value, label=label)
        return cls.relative(value, epoch=epoch, label=label)

    @property
    def start(self) -> float:
        return float(self.seconds[0])

    @property
    def stop(self) -> float:
        return float(self.seconds[-1])

    @property
    def duration(self) -> float:
        return float(self.seconds[-1] - self.seconds[0])

    def absolute_seconds(self) -> np.ndarray | None:
        if Time is not None and isinstance(self.absolute_time, Time):
            return np.asarray(self.absolute_time.utc.unix, dtype=np.float64)
        if Time is not None and isinstance(self.epoch, Time) and u is not None:
            return np.asarray(
                (self.epoch + np.asarray(self.seconds, dtype=np.float64) * u.s).utc.unix,
                dtype=np.float64,
            )
        return None

    def axis_for_interpolation(self) -> np.ndarray:
        absolute = self.absolute_seconds()
        if absolute is not None:
            return np.ascontiguousarray(absolute)
        return self.seconds


__all__ = ["CoverageTimeline"]
