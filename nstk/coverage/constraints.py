from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ElevationConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class AzimuthConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class RangeConstraint:
    min_m: float | None = None
    max_m: float | None = None


@dataclass(frozen=True)
class TargetSunElevationConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class TargetLocalTimeConstraint:
    start_hour: float
    stop_hour: float
    solar: bool = True


@dataclass(frozen=True)
class MinAccessDurationConstraint:
    min_seconds: float


Constraint = (
    ElevationConstraint
    | AzimuthConstraint
    | RangeConstraint
    | TargetSunElevationConstraint
    | TargetLocalTimeConstraint
    | MinAccessDurationConstraint
)


@dataclass(frozen=True)
class ConstraintSet:
    items: tuple[Constraint, ...] = ()

    @classmethod
    def from_any(cls, value: Any | None) -> "ConstraintSet":
        if value is None:
            return cls()
        if isinstance(value, ConstraintSet):
            return value
        if isinstance(value, (ElevationConstraint, AzimuthConstraint, RangeConstraint, TargetSunElevationConstraint, TargetLocalTimeConstraint, MinAccessDurationConstraint)):
            return cls(items=(value,))
        if isinstance(value, Iterable):
            return cls(items=tuple(value))
        raise TypeError(f"Unsupported constraints input: {type(value)!r}")

    def elevation(self) -> ElevationConstraint | None:
        for item in self.items:
            if isinstance(item, ElevationConstraint):
                return item
        return None

    def pair_constraints(self) -> tuple[Constraint, ...]:
        return tuple(
            item
            for item in self.items
            if isinstance(item, (ElevationConstraint, AzimuthConstraint, RangeConstraint))
        )

    def target_gates(self) -> tuple[Constraint, ...]:
        return tuple(
            item
            for item in self.items
            if isinstance(item, (TargetSunElevationConstraint, TargetLocalTimeConstraint))
        )

    def post_interval(self) -> tuple[Constraint, ...]:
        return tuple(
            item
            for item in self.items
            if isinstance(item, MinAccessDurationConstraint)
        )

    def requested_channels(self) -> set[str]:
        channels: set[str] = set()
        for item in self.items:
            if isinstance(item, ElevationConstraint):
                channels.add("elevation")
            elif isinstance(item, AzimuthConstraint):
                channels.add("azimuth")
            elif isinstance(item, RangeConstraint):
                channels.add("range")
            elif isinstance(item, TargetSunElevationConstraint):
                channels.add("target_sun_elevation")
            elif isinstance(item, TargetLocalTimeConstraint):
                channels.add("target_local_time")
        return channels


__all__ = [
    "ConstraintSet",
    "ElevationConstraint",
    "AzimuthConstraint",
    "RangeConstraint",
    "TargetSunElevationConstraint",
    "TargetLocalTimeConstraint",
    "MinAccessDurationConstraint",
]
