"""High-performance timed frame transforms backed by a Java Orekit bridge.

This module keeps Python-side work focused on input normalization and unit
handling while all timed transform loops execute in Java.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import astropy.units as u
import jpype
import numpy as np

from nstk._orekit_frames import (
    _coerce_iers,
    _resolve_frame,
)
from nstk.time_utils import (
    TimedTransformTimeSpec,
    astropy_time_to_orekit_date,
    normalize_timed_transform_time_input,
)

from ._timed_rotations_java_bridge import (
    get_timed_rotation_bridge_class,
)


# Lazy-initialized Java bindings
_RUNTIME_BOUND = False
_JavaTimedRotationBridge = None
AbsoluteDate = None
TimeScalesFactory = None


def _bind_timed_rotations_java() -> None:
    """Bind timed-rotation Java bridge classes and supporting Orekit types lazily."""

    global _RUNTIME_BOUND
    global _JavaTimedRotationBridge, AbsoluteDate, TimeScalesFactory

    if _RUNTIME_BOUND:
        return

    from nstk._orekit_runtime import ensure_orekit_runtime
    import nstk._orekit_frames as _orekit_frames

    ensure_orekit_runtime()
    _orekit_frames._bind_java()

    from org.orekit.time import AbsoluteDate as _AbsoluteDate  # type: ignore
    from org.orekit.time import TimeScalesFactory as _TimeScalesFactory  # type: ignore

    AbsoluteDate = _AbsoluteDate
    TimeScalesFactory = _TimeScalesFactory
    _JavaTimedRotationBridge = get_timed_rotation_bridge_class()
    _RUNTIME_BOUND = True


def _normalize_time_input(time: Any) -> TimedTransformTimeSpec:
    _bind_timed_rotations_java()
    return normalize_timed_transform_time_input(
        time,
        absolute_date_cls=AbsoluteDate,
        astropy_to_orekit=lambda t: astropy_time_to_orekit_date(
            t,
            absolute_date_cls=AbsoluteDate,
            time_scales_factory=TimeScalesFactory,
        ),
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
        (for example ``"gcrf"``, ``"itrf"``, ``"itrf2014"``, ``"tod"``,
        ``"tod2010"``, ``"itrfcio"``, or ``"TOD_CONVENTIONS_2010_SIMPLE_EOP"``).
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
        Orekit IERS convention object used when resolving Earth-fixed frames
        that do not already encode a specific convention/version in the frame
        string. Defaults to the most recent convention supported by Orekit.
    simple_eop : bool, default True
        ``simpleEOP`` flag used when resolving Earth-fixed frames.

    Returns
    -------
    tuple
        ``(position, velocity, acceleration)`` transformed to ``to_frame``.
        Missing optional components are returned as ``None``.
        Output shape follows broadcasted input shape with trailing ``(3,)``.
    """

    _bind_timed_rotations_java()

    iers = _coerce_iers(iers_convention, when_none="latest")
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
    "transform",
]
