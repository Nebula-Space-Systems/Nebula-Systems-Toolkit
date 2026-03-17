"""High-performance timed frame transforms backed by a Java Orekit bridge.

This module keeps Python-side work focused on input normalization and unit
handling while all timed transform loops execute in Java.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import astropy.units as u
import jpype
import numpy as np
from astropy.time import Time

from nebula.propagation.orbit import astropy_time_to_orekit_date

from ._timed_rotations_java_bridge import (
    get_timed_rotation_bridge_class,
    initialize_timed_rotations_runtime,
)


# Lazy-initialized Java bindings
_RUNTIME_BOUND = False
_JavaTimedRotationBridge = None
FramesFactory = None
IERSConventions = None
AbsoluteDate = None


def initialize_timed_rotations(*, data_path: str | None = None) -> None:
    """Initialize JVM/runtime for Java-backed timed frame transforms."""

    initialize_timed_rotations_runtime(data_path=data_path)


def _bind_java() -> None:
    global _RUNTIME_BOUND
    global _JavaTimedRotationBridge, FramesFactory, IERSConventions, AbsoluteDate

    if _RUNTIME_BOUND:
        return

    initialize_timed_rotations_runtime()

    from org.orekit.frames import FramesFactory as _FramesFactory  # type: ignore
    from org.orekit.time import AbsoluteDate as _AbsoluteDate  # type: ignore
    from org.orekit.utils import IERSConventions as _IERSConventions  # type: ignore

    FramesFactory = _FramesFactory
    AbsoluteDate = _AbsoluteDate
    IERSConventions = _IERSConventions
    _JavaTimedRotationBridge = get_timed_rotation_bridge_class()
    _RUNTIME_BOUND = True


def _normalize_frame_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _coerce_iers(iers_convention):
    _bind_java()
    if iers_convention is None:
        return IERSConventions.IERS_2010
    return iers_convention


def _resolve_named_frame(name: str, *, iers, simple_eop: bool):
    _bind_java()
    key = _normalize_frame_name(name)

    aliases = {
        "j2000": "eme2000",
        "gcrs": "gcrf",
        "itrs": "itrf",
        "ecef": "itrf",
        "eci": "gcrf",
        "veis": "veis1950",
        "veis50": "veis1950",
        "meanofdate": "mod",
        "trueofdate": "tod",
    }
    key = aliases.get(key, key)

    if key == "gcrf":
        return FramesFactory.getGCRF()
    if key == "icrf":
        return FramesFactory.getICRF()
    if key == "eme2000":
        return FramesFactory.getEME2000()
    if key == "teme":
        return FramesFactory.getTEME()
    if key == "mod":
        return FramesFactory.getMOD(iers)
    if key == "tod":
        return FramesFactory.getTOD(iers, bool(simple_eop))
    if key == "cirf":
        return FramesFactory.getCIRF(iers, bool(simple_eop))
    if key == "ecliptic":
        return FramesFactory.getEcliptic(iers)
    if key in ("veis1950",):
        return FramesFactory.getVeis1950()
    if key == "itrf":
        return FramesFactory.getITRF(iers, bool(simple_eop))

    raise ValueError(
        f"Unsupported frame string: '{name}'. "
        "Use Orekit Frame objects for custom frames."
    )


def _resolve_frame(frame_like: Any, *, iers, simple_eop: bool):
    if isinstance(frame_like, str):
        return _resolve_named_frame(frame_like, iers=iers, simple_eop=bool(simple_eop))
    return frame_like


def _is_absolute_date(obj: Any) -> bool:
    _bind_java()
    try:
        return bool(AbsoluteDate.class_.isInstance(obj))
    except Exception:
        return False


def _is_scalar_astropy_time(obj: Any) -> bool:
    return isinstance(obj, Time) and getattr(obj, "shape", None) == ()


@dataclass(frozen=True)
class _TimeSpec:
    mode: str  # "offsets" or "dates"
    shape: tuple[int, ...]
    epoch: Any | None
    offsets: np.ndarray | None
    dates: np.ndarray | None


def _normalize_time_input(time: Any) -> _TimeSpec:
    _bind_java()

    if _is_absolute_date(time):
        return _TimeSpec(
            mode="dates",
            shape=(),
            epoch=None,
            offsets=None,
            dates=np.asarray([time], dtype=object),
        )

    if isinstance(time, Time):
        if getattr(time, "shape", None) == ():
            return _TimeSpec(
                mode="offsets",
                shape=(),
                epoch=astropy_time_to_orekit_date(time),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )

        unix = np.asarray(time.utc.unix, dtype=np.float64)
        if unix.ndim == 0:
            epoch_unix = float(unix)
            return _TimeSpec(
                mode="offsets",
                shape=(),
                epoch=astropy_time_to_orekit_date(
                    Time(epoch_unix, format="unix", scale="utc")
                ),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )

        flat = np.ascontiguousarray(unix.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return _TimeSpec(
                mode="offsets",
                shape=tuple(unix.shape),
                epoch=astropy_time_to_orekit_date(
                    Time(0.0, format="unix", scale="utc")
                ),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )

        epoch_unix = float(flat[0])
        return _TimeSpec(
            mode="offsets",
            shape=tuple(unix.shape),
            epoch=astropy_time_to_orekit_date(
                Time(epoch_unix, format="unix", scale="utc")
            ),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    if isinstance(time, u.Quantity):
        secs = np.asarray(time.to_value(u.s), dtype=np.float64)
        if secs.ndim == 0:
            unix_s = float(secs)
            return _TimeSpec(
                mode="offsets",
                shape=(),
                epoch=astropy_time_to_orekit_date(
                    Time(unix_s, format="unix", scale="utc")
                ),
                offsets=np.asarray([0.0], dtype=np.float64),
                dates=None,
            )
        flat = np.ascontiguousarray(secs.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return _TimeSpec(
                mode="offsets",
                shape=tuple(secs.shape),
                epoch=astropy_time_to_orekit_date(
                    Time(0.0, format="unix", scale="utc")
                ),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )
        epoch_unix = float(flat[0])
        return _TimeSpec(
            mode="offsets",
            shape=tuple(secs.shape),
            epoch=astropy_time_to_orekit_date(
                Time(epoch_unix, format="unix", scale="utc")
            ),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    if isinstance(time, (float, int, np.floating, np.integer)) and not isinstance(
        time, bool
    ):
        unix_s = float(time)
        return _TimeSpec(
            mode="offsets",
            shape=(),
            epoch=astropy_time_to_orekit_date(Time(unix_s, format="unix", scale="utc")),
            offsets=np.asarray([0.0], dtype=np.float64),
            dates=None,
        )

    if isinstance(time, (list, tuple, np.ndarray)):
        arr_obj = np.asarray(time, dtype=object)
        if arr_obj.ndim == 0:
            scalar_obj = arr_obj.item()
            if _is_absolute_date(scalar_obj):
                return _TimeSpec(
                    mode="dates",
                    shape=(),
                    epoch=None,
                    offsets=None,
                    dates=np.asarray([scalar_obj], dtype=object),
                )
            if _is_scalar_astropy_time(scalar_obj):
                return _normalize_time_input(scalar_obj)
            return _normalize_time_input(float(scalar_obj))

        if arr_obj.size > 0:
            first = arr_obj.flat[0]
            if _is_absolute_date(first):
                for item in arr_obj.flat:
                    if not _is_absolute_date(item):
                        raise TypeError(
                            "time iterable with AbsoluteDate values must contain only AbsoluteDate entries"
                        )
                return _TimeSpec(
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
                return _TimeSpec(
                    mode="offsets",
                    shape=tuple(arr_obj.shape),
                    epoch=astropy_time_to_orekit_date(
                        Time(epoch_unix, format="unix", scale="utc")
                    ),
                    offsets=np.ascontiguousarray(
                        unix_vals - epoch_unix, dtype=np.float64
                    ),
                    dates=None,
                )

        nums = np.asarray(time, dtype=np.float64)
        if nums.ndim == 0:
            return _normalize_time_input(float(nums))
        if not np.all(np.isfinite(nums)):
            raise ValueError("time contains non-finite values")
        flat = np.ascontiguousarray(nums.reshape(-1), dtype=np.float64)
        if flat.size == 0:
            return _TimeSpec(
                mode="offsets",
                shape=tuple(nums.shape),
                epoch=astropy_time_to_orekit_date(
                    Time(0.0, format="unix", scale="utc")
                ),
                offsets=np.asarray([], dtype=np.float64),
                dates=None,
            )
        epoch_unix = float(flat[0])
        return _TimeSpec(
            mode="offsets",
            shape=tuple(nums.shape),
            epoch=astropy_time_to_orekit_date(
                Time(epoch_unix, format="unix", scale="utc")
            ),
            offsets=np.ascontiguousarray(flat - epoch_unix, dtype=np.float64),
            dates=None,
        )

    raise TypeError(
        "time must be astropy Time, Orekit AbsoluteDate, unix-seconds numeric values, "
        "or an iterable/array of those types"
    )


def _normalize_vector_component(
    x: Union[np.ndarray, u.Quantity],
    *,
    unit: u.Unit,
    name: str,
) -> tuple[np.ndarray, bool]:
    is_quantity = isinstance(x, u.Quantity)
    arr = np.asarray(x.to_value(unit) if is_quantity else x, dtype=np.float64)

    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must have shape (..., 3); got {arr.shape}")
    elif arr.ndim >= 2:
        if arr.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (..., 3); got {arr.shape}")
    else:
        raise ValueError(f"{name} must have shape (..., 3); got {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")

    return np.ascontiguousarray(arr, dtype=np.float64), is_quantity


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    out: tuple[int, ...] = ()
    for shp in shapes:
        out = np.broadcast_shapes(out, shp)
    return out


def _reshape_output(flat: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if target_shape == ():
        return flat.reshape(3)
    return flat.reshape(target_shape + (3,))


def transform(
    from_frame: Any,
    to_frame: Any,
    time: Any,
    position: Union[np.ndarray, u.Quantity],
    velocity: Optional[Union[np.ndarray, u.Quantity]] = None,
    acceleration: Optional[Union[np.ndarray, u.Quantity]] = None,
    *,
    iers_convention: Any = None,
    simple_eop: bool = True,
) -> Tuple[
    Union[np.ndarray, u.Quantity],
    Optional[Union[np.ndarray, u.Quantity]],
    Optional[Union[np.ndarray, u.Quantity]],
]:
    """Transform Cartesian state vectors between arbitrary Orekit frames.

    Parameters
    ----------
    from_frame, to_frame : Frame | str
        Source and destination frames. Strings map common Orekit frames
        (for example ``"gcrf"``, ``"itrf"``, ``"teme"``, ``"mod"``, ``"tod"``).
        For any custom frame, pass an Orekit ``Frame`` object directly.
    time : Any
        Scalar or array-like time input. Supported forms:
        - ``astropy.time.Time`` (scalar or array)
        - Orekit ``AbsoluteDate`` (scalar or array/iterable)
        - unix seconds as scalar/array numeric values
        - time ``Quantity`` in seconds
    position : ndarray | Quantity
        Cartesian positions with shape ``(..., 3)`` in meters.
    velocity : ndarray | Quantity, optional
        Cartesian velocities with shape ``(..., 3)`` in m/s.
    acceleration : ndarray | Quantity, optional
        Cartesian accelerations with shape ``(..., 3)`` in m/s^2.
        If provided without velocity, zero input velocity is assumed internally
        to evaluate the acceleration transform.
    iers_convention : optional
        Orekit IERS convention object used when resolving Earth-fixed frames.
        Defaults to ``IERS_2010``.
    simple_eop : bool, default True
        ``simpleEOP`` flag used when resolving Earth-fixed frames.

    Returns
    -------
    tuple
        ``(position, velocity, acceleration)`` transformed to ``to_frame``.
        Missing optional components are returned as ``None``.
        Output shape follows broadcasted input shape with trailing ``(3,)``.
    """

    _bind_java()

    iers = _coerce_iers(iers_convention)
    from_fr = _resolve_frame(from_frame, iers=iers, simple_eop=bool(simple_eop))
    to_fr = _resolve_frame(to_frame, iers=iers, simple_eop=bool(simple_eop))

    t_spec = _normalize_time_input(time)
    p_arr, p_is_q = _normalize_vector_component(position, unit=u.m, name="position")
    v_arr = None
    a_arr = None
    v_is_q = False
    a_is_q = False

    if velocity is not None:
        v_arr, v_is_q = _normalize_vector_component(
            velocity, unit=(u.m / u.s), name="velocity"
        )
    if acceleration is not None:
        a_arr, a_is_q = _normalize_vector_component(
            acceleration, unit=(u.m / (u.s**2)), name="acceleration"
        )

    target_shape = _broadcast_shape(
        t_spec.shape,
        p_arr.shape[:-1],
        v_arr.shape[:-1] if v_arr is not None else (),
        a_arr.shape[:-1] if a_arr is not None else (),
    )

    p_b = np.broadcast_to(p_arr, target_shape + (3,)).reshape(-1, 3)
    v_b = (
        np.broadcast_to(v_arr, target_shape + (3,)).reshape(-1, 3)
        if v_arr is not None
        else None
    )
    a_b = (
        np.broadcast_to(a_arr, target_shape + (3,)).reshape(-1, 3)
        if a_arr is not None
        else None
    )

    if t_spec.mode == "offsets":
        t_arr = (
            np.asarray(t_spec.offsets, dtype=np.float64).reshape(t_spec.shape)
            if t_spec.shape != ()
            else np.asarray(float(t_spec.offsets[0]), dtype=np.float64)
        )
        dt_b = (
            np.broadcast_to(t_arr, target_shape)
            .reshape(-1)
            .astype(np.float64, copy=False)
        )
        dt_b = np.ascontiguousarray(dt_b, dtype=np.float64)
    else:
        d_arr = (
            np.asarray(t_spec.dates, dtype=object).reshape(t_spec.shape)
            if t_spec.shape != ()
            else np.asarray(t_spec.dates[0], dtype=object)
        )
        dates_b = np.broadcast_to(d_arr, target_shape).reshape(-1)
        dates_b = np.ascontiguousarray(dates_b, dtype=object)

    p_flat = np.ascontiguousarray(p_b.reshape(-1), dtype=np.float64)
    v_flat = (
        np.ascontiguousarray(v_b.reshape(-1), dtype=np.float64)
        if v_b is not None
        else None
    )
    a_flat = (
        np.ascontiguousarray(a_b.reshape(-1), dtype=np.float64)
        if a_b is not None
        else None
    )

    vel_requested = v_flat is not None
    acc_requested = a_flat is not None

    v_for_java = v_flat
    if acc_requested and not vel_requested:
        v_for_java = np.zeros_like(p_flat)

    if t_spec.mode == "offsets":
        result = _JavaTimedRotationBridge.transformAtOffsets(
            from_fr,
            to_fr,
            t_spec.epoch,
            dt_b,
            p_flat,
            v_for_java,
            a_flat,
        )
    else:
        dates_java = jpype.JArray(AbsoluteDate)(list(dates_b.tolist()))
        result = _JavaTimedRotationBridge.transformAtDates(
            from_fr,
            to_fr,
            dates_java,
            p_flat,
            v_for_java,
            a_flat,
        )

    p_out_np = _reshape_output(np.asarray(result.p, dtype=np.float64), target_shape)
    v_out_np = (
        _reshape_output(np.asarray(result.v, dtype=np.float64), target_shape)
        if vel_requested
        else None
    )
    a_out_np = (
        _reshape_output(np.asarray(result.a, dtype=np.float64), target_shape)
        if acc_requested
        else None
    )

    p_out = p_out_np * u.m if p_is_q else p_out_np
    v_out = (v_out_np * (u.m / u.s)) if (v_out_np is not None and v_is_q) else v_out_np
    a_out = (
        a_out_np * (u.m / (u.s**2)) if (a_out_np is not None and a_is_q) else a_out_np
    )

    return p_out, v_out, a_out


__all__ = [
    "initialize_timed_rotations",
    "transform",
]
