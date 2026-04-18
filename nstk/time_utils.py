"""Shared Orekit/Astropy time conversion and normalization utilities.

This module centralizes time handling used across propagation and transform
APIs so logic stays consistent and avoids duplicated conversion code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import astropy.units as u
import numpy as np
from astropy.time import Time


BindJavaFn = Callable[[], None]
AstropyToOrekitFn = Callable[[Time], Any]


def _maybe_bind(bind_java: BindJavaFn | None) -> None:
    if bind_java is not None:
        bind_java()


def _initialize_default_runtime() -> None:
    """Best-effort Orekit runtime bootstrap when no explicit binder is provided."""

    try:
        from nstk._orekit_runtime import ensure_orekit_runtime  # local import on purpose

        ensure_orekit_runtime()
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize Orekit runtime automatically. "
            "If you need eager setup or a custom Orekit data directory, call "
            "nstk.initialize(...) or nstk.set_orekit_data_path(...) before using "
            "Nebula Space Toolkit time helpers."
        ) from exc


def _resolve_absolute_date_class(
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
) -> Any:
    _maybe_bind(bind_java)
    if absolute_date_cls is not None:
        return absolute_date_cls
    try:
        from org.orekit.time import AbsoluteDate as _AbsoluteDate  # type: ignore
    except Exception:
        _initialize_default_runtime()
        from org.orekit.time import AbsoluteDate as _AbsoluteDate  # type: ignore

    return _AbsoluteDate


def _resolve_time_scales_factory(
    *,
    bind_java: BindJavaFn | None = None,
    time_scales_factory: Any | None = None,
) -> Any:
    _maybe_bind(bind_java)
    if time_scales_factory is not None:
        return time_scales_factory
    try:
        from org.orekit.time import TimeScalesFactory as _TimeScalesFactory  # type: ignore
    except Exception:
        _initialize_default_runtime()
        from org.orekit.time import TimeScalesFactory as _TimeScalesFactory  # type: ignore

    return _TimeScalesFactory


def astropy_time_to_orekit_date(
    time: Time,
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
    time_scales_factory: Any | None = None,
) -> Any:
    """Convert scalar ``astropy.time.Time`` (UTC) to Orekit ``AbsoluteDate``."""

    if not isinstance(time, Time):
        raise TypeError("time must be astropy.time.Time")
    if getattr(time, "shape", None) not in ((), None):
        raise TypeError("time must be scalar astropy.time.Time")

    absolute_date_cls = _resolve_absolute_date_class(
        bind_java=bind_java, absolute_date_cls=absolute_date_cls
    )
    time_scales_factory = _resolve_time_scales_factory(
        bind_java=bind_java, time_scales_factory=time_scales_factory
    )

    utc = time_scales_factory.getUTC()
    c = time.utc.ymdhms
    return absolute_date_cls(
        int(c.year),
        int(c.month),
        int(c.day),
        int(c.hour),
        int(c.minute),
        float(c.second),
        utc,
    )


def orekit_date_to_astropy_time(
    date: Any,
    *,
    bind_java: BindJavaFn | None = None,
    time_scales_factory: Any | None = None,
) -> Time:
    """Convert Orekit ``AbsoluteDate`` to scalar UTC ``astropy.time.Time``."""

    time_scales_factory = _resolve_time_scales_factory(
        bind_java=bind_java, time_scales_factory=time_scales_factory
    )
    utc = time_scales_factory.getUTC()
    components = date.getComponents(utc)
    d = components.getDate()
    t = components.getTime()
    return Time(
        {
            "year": int(d.getYear()),
            "month": int(d.getMonth()),
            "day": int(d.getDay()),
            "hour": int(t.getHour()),
            "minute": int(t.getMinute()),
            "second": float(t.getSecond()),
        },
        format="ymdhms",
        scale="utc",
    )


def safe_orekit_date_to_astropy_time(
    date: Any,
    fallback: Time,
    *,
    bind_java: BindJavaFn | None = None,
    time_scales_factory: Any | None = None,
) -> Time:
    """Convert an Orekit date to Astropy, falling back when conversion fails.

    This is primarily useful for Orekit infinity dates, which Astropy cannot
    represent directly.
    """

    try:
        return orekit_date_to_astropy_time(
            date,
            bind_java=bind_java,
            time_scales_factory=time_scales_factory,
        )
    except Exception:
        return fallback


def make_times_astropy(epoch: Time, delta_times_sec: np.ndarray) -> Time:
    """Build a UTC Astropy time vector from an epoch and second offsets.

    Parameters
    ----------
    epoch
        Reference epoch as a scalar ``astropy.time.Time``.
    delta_times_sec
        Scalar or array-like offsets in seconds from ``epoch``.

    Returns
    -------
    astropy.time.Time
        Vectorized sample epochs with shape ``(N,)`` in UTC.
    """

    return Time(
        epoch.utc.unix + np.asarray(delta_times_sec, dtype=np.float64),
        format="unix",
        scale="utc",
    )


def is_orekit_absolute_date(
    obj: Any,
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
) -> bool:
    """Return ``True`` when ``obj`` is an Orekit ``AbsoluteDate`` instance."""

    absolute_date_cls = _resolve_absolute_date_class(
        bind_java=bind_java, absolute_date_cls=absolute_date_cls
    )
    try:
        return bool(absolute_date_cls.class_.isInstance(obj))
    except Exception:
        return False


def _resolve_astropy_to_orekit(
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
    astropy_to_orekit: AstropyToOrekitFn | None = None,
    time_scales_factory: Any | None = None,
) -> AstropyToOrekitFn:
    if astropy_to_orekit is not None:
        return astropy_to_orekit

    def _default_converter(t: Time) -> Any:
        return astropy_time_to_orekit_date(
            t,
            bind_java=bind_java,
            absolute_date_cls=absolute_date_cls,
            time_scales_factory=time_scales_factory,
        )

    return _default_converter


def normalize_time_to_epoch_seconds(
    time_like: Any,
    epoch: Time,
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
    astropy_to_orekit: AstropyToOrekitFn | None = None,
    time_scales_factory: Any | None = None,
) -> tuple[np.ndarray, bool]:
    """Normalize time-like input to seconds-from-epoch.

    Accepted inputs:
    - astropy Time scalar/vector
    - Orekit AbsoluteDate scalar/vector
    - scalar float/int seconds
    - numpy/list of float/int seconds
    - astropy Quantity with time units (converted to seconds)

    Returns
    -------
    tuple[np.ndarray, bool]
        ``(dt_seconds, is_scalar)`` where ``dt_seconds`` is ``float64`` seconds
        from ``epoch``.
    """

    absolute_date_cls = _resolve_absolute_date_class(
        bind_java=bind_java, absolute_date_cls=absolute_date_cls
    )
    to_orekit = _resolve_astropy_to_orekit(
        bind_java=bind_java,
        absolute_date_cls=absolute_date_cls,
        astropy_to_orekit=astropy_to_orekit,
        time_scales_factory=time_scales_factory,
    )

    if is_orekit_absolute_date(time_like, absolute_date_cls=absolute_date_cls):
        epoch_date = to_orekit(epoch)
        return np.asarray([float(time_like.durationFrom(epoch_date))], dtype=np.float64), True

    if isinstance(time_like, Time):
        is_scalar = getattr(time_like, "shape", None) == ()
        dt = np.asarray(time_like.utc.unix, dtype=np.float64) - float(epoch.utc.unix)
        out = np.atleast_1d(np.asarray(dt, dtype=np.float64))
        if not np.all(np.isfinite(out)):
            raise ValueError("time contains non-finite values")
        return out, is_scalar

    if isinstance(time_like, u.Quantity):
        secs = np.asarray(time_like.to_value(u.s), dtype=np.float64)
        if secs.ndim == 0:
            out = np.asarray([float(secs)], dtype=np.float64)
            return out, True
        if secs.ndim > 1:
            raise ValueError("time quantity input must be scalar or 1D")
        if not np.all(np.isfinite(secs)):
            raise ValueError("time contains non-finite values")
        return secs, False

    if isinstance(time_like, (float, int, np.floating, np.integer)) and not isinstance(
        time_like, bool
    ):
        return np.asarray([float(time_like)], dtype=np.float64), True

    if isinstance(time_like, (list, tuple, np.ndarray)):
        arr = np.asarray(time_like)
        if arr.ndim == 0 and is_orekit_absolute_date(arr.item(), absolute_date_cls=absolute_date_cls):
            epoch_date = to_orekit(epoch)
            dt = float(arr.item().durationFrom(epoch_date))
            return np.asarray([dt], dtype=np.float64), True

        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=np.float64), True
        if arr.ndim > 1:
            raise ValueError("time input must be scalar or 1D")

        if arr.size > 0 and is_orekit_absolute_date(arr[0], absolute_date_cls=absolute_date_cls):
            epoch_date = to_orekit(epoch)
            out = np.empty(arr.size, dtype=np.float64)
            for idx, val in enumerate(arr):
                if not is_orekit_absolute_date(val, absolute_date_cls=absolute_date_cls):
                    raise TypeError(
                        "absolute-date arrays must contain only Orekit AbsoluteDate entries"
                    )
                out[idx] = float(val.durationFrom(epoch_date))
            if not np.all(np.isfinite(out)):
                raise ValueError("time contains non-finite values")
            return out, False

        out = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(out)):
            raise ValueError("time contains non-finite values")
        return out, False

    raise TypeError(
        "time must be astropy Time, Orekit AbsoluteDate, seconds scalar/array, or Quantity with time units"
    )


def _is_scalar_astropy_time(obj: Any) -> bool:
    return isinstance(obj, Time) and getattr(obj, "shape", None) == ()


@dataclass(frozen=True)
class TimedTransformTimeSpec:
    """Normalized time specification for timed frame transforms."""

    mode: str  # "offsets" or "dates"
    shape: tuple[int, ...]
    epoch: Any | None
    offsets: np.ndarray | None
    dates: np.ndarray | None


def normalize_timed_transform_time_input(
    time: Any,
    *,
    bind_java: BindJavaFn | None = None,
    absolute_date_cls: Any | None = None,
    astropy_to_orekit: AstropyToOrekitFn | None = None,
    time_scales_factory: Any | None = None,
) -> TimedTransformTimeSpec:
    """Normalize timed-rotation API ``time`` input to offsets or date arrays."""

    absolute_date_cls = _resolve_absolute_date_class(
        bind_java=bind_java, absolute_date_cls=absolute_date_cls
    )
    to_orekit = _resolve_astropy_to_orekit(
        bind_java=bind_java,
        absolute_date_cls=absolute_date_cls,
        astropy_to_orekit=astropy_to_orekit,
        time_scales_factory=time_scales_factory,
    )

    if is_orekit_absolute_date(time, absolute_date_cls=absolute_date_cls):
        return TimedTransformTimeSpec(
            mode="dates",
            shape=(),
            epoch=None,
            offsets=None,
            dates=np.asarray([time], dtype=object),
        )

    if isinstance(time, Time):
        if getattr(time, "shape", None) == ():
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=(),
                epoch=to_orekit(time),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )

        unix = np.asarray(time.utc.unix, dtype=np.float64)
        if unix.ndim == 0:
            epoch_unix = float(unix)
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=(),
                epoch=to_orekit(Time(epoch_unix, format="unix", scale="utc")),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )

        flat = np.ascontiguousarray(unix.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=tuple(unix.shape),
                epoch=to_orekit(Time(0.0, format="unix", scale="utc")),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )

        epoch_unix = float(flat[0])
        return TimedTransformTimeSpec(
            mode="offsets",
            shape=tuple(unix.shape),
            epoch=to_orekit(Time(epoch_unix, format="unix", scale="utc")),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    if isinstance(time, u.Quantity):
        secs = np.asarray(time.to_value(u.s), dtype=np.float64)
        if secs.ndim == 0:
            unix_s = float(secs)
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=(),
                epoch=to_orekit(Time(unix_s, format="unix", scale="utc")),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )
        flat = np.ascontiguousarray(secs.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=tuple(secs.shape),
                epoch=to_orekit(Time(0.0, format="unix", scale="utc")),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )
        epoch_unix = float(flat[0])
        return TimedTransformTimeSpec(
            mode="offsets",
            shape=tuple(secs.shape),
            epoch=to_orekit(Time(epoch_unix, format="unix", scale="utc")),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    if isinstance(time, (float, int, np.floating, np.integer)) and not isinstance(time, bool):
        unix_s = float(time)
        return TimedTransformTimeSpec(
            mode="offsets",
            shape=(),
            epoch=to_orekit(Time(unix_s, format="unix", scale="utc")),
            offsets=np.asarray([0.0], dtype=np.float64),
            dates=None,
        )

    if isinstance(time, (list, tuple, np.ndarray)):
        arr_obj = np.asarray(time, dtype=object)
        if arr_obj.ndim == 0:
            scalar_obj = arr_obj.item()
            if is_orekit_absolute_date(scalar_obj, absolute_date_cls=absolute_date_cls):
                return TimedTransformTimeSpec(
                    mode="dates",
                    shape=(),
                    epoch=None,
                    offsets=None,
                    dates=np.asarray([scalar_obj], dtype=object),
                )
            if _is_scalar_astropy_time(scalar_obj):
                return normalize_timed_transform_time_input(
                    scalar_obj,
                    absolute_date_cls=absolute_date_cls,
                    astropy_to_orekit=to_orekit,
                    time_scales_factory=time_scales_factory,
                )
            return normalize_timed_transform_time_input(
                float(scalar_obj),
                absolute_date_cls=absolute_date_cls,
                astropy_to_orekit=to_orekit,
                time_scales_factory=time_scales_factory,
            )

        if arr_obj.size > 0:
            first = arr_obj.flat[0]
            if is_orekit_absolute_date(first, absolute_date_cls=absolute_date_cls):
                for item in arr_obj.flat:
                    if not is_orekit_absolute_date(item, absolute_date_cls=absolute_date_cls):
                        raise TypeError(
                            "time iterable with AbsoluteDate values must contain only AbsoluteDate entries"
                        )
                return TimedTransformTimeSpec(
                    mode="dates",
                    shape=tuple(arr_obj.shape),
                    epoch=None,
                    offsets=None,
                    dates=np.ascontiguousarray(arr_obj.reshape(-1), dtype=object),
                )
            if _is_scalar_astropy_time(first):
                unix_vals = np.empty(arr_obj.size, dtype=np.float64)
                for idx, item in enumerate(arr_obj.flat):
                    if not _is_scalar_astropy_time(item):
                        raise TypeError(
                            "time iterable with astropy scalar Time values must be homogeneous"
                        )
                    unix_vals[idx] = float(item.utc.unix)
                epoch_unix = float(unix_vals[0])
                return TimedTransformTimeSpec(
                    mode="offsets",
                    shape=tuple(arr_obj.shape),
                    epoch=to_orekit(Time(epoch_unix, format="unix", scale="utc")),
                    offsets=np.ascontiguousarray(unix_vals - epoch_unix, dtype=np.float64),
                    dates=None,
                )

        nums = np.asarray(time, dtype=np.float64)
        if nums.ndim == 0:
            return normalize_timed_transform_time_input(
                float(nums),
                absolute_date_cls=absolute_date_cls,
                astropy_to_orekit=to_orekit,
                time_scales_factory=time_scales_factory,
            )
        if not np.all(np.isfinite(nums)):
            raise ValueError("time contains non-finite values")
        flat = np.ascontiguousarray(nums.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return TimedTransformTimeSpec(
                mode="offsets",
                shape=tuple(nums.shape),
                epoch=to_orekit(Time(0.0, format="unix", scale="utc")),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )
        epoch_unix = float(flat[0])
        return TimedTransformTimeSpec(
            mode="offsets",
            shape=tuple(nums.shape),
            epoch=to_orekit(Time(epoch_unix, format="unix", scale="utc")),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    raise TypeError(
        "time must be astropy Time, Orekit AbsoluteDate, unix-seconds numeric values, "
        "or an iterable/array of those types"
    )


__all__ = [
    "TimedTransformTimeSpec",
    "astropy_time_to_orekit_date",
    "is_orekit_absolute_date",
    "make_times_astropy",
    "normalize_timed_transform_time_input",
    "normalize_time_to_epoch_seconds",
    "orekit_date_to_astropy_time",
    "safe_orekit_date_to_astropy_time",
]
