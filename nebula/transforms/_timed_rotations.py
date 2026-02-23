"""
Timed frame rotations for position / position+velocity arrays.

This module provides a single high-level interface that accepts:
- times (scalar or vector),
- positions (N,3) or (3,),
- from/to frame names as strings,
- optional velocities.

It then applies Orekit frame transforms with grouped-time fast paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Union

import numpy as np
from nebula.transforms._coarse_eci2itrf import (
    coarse_eci2itrf_pos as _coarse_eci2itrf_pos_kernel,
    coarse_eci2itrf_pos_vec as _coarse_eci2itrf_pos_vec_kernel,
    coarse_eci2itrf_pos_vel as _coarse_eci2itrf_pos_vel_kernel,
    coarse_eci2itrf_pos_vel_vec as _coarse_eci2itrf_pos_vel_vec_kernel,
)

try:
    from astropy.time import Time as AstropyTime  # type: ignore
except Exception:  # pragma: no cover
    AstropyTime = None  # type: ignore

from nebula.propagation.orbit import initialize_orekit

# Lazily bound Orekit classes/singletons.
_FramesFactory = None
_AbsoluteDateType = None
_TimeScalesFactory = None
_IERSConventions = None
_UTC = None

_ITRF_CACHE: dict[tuple[object, bool], object] = {}


def _ensure_orekit_bindings() -> None:
    global _FramesFactory, _AbsoluteDateType, _TimeScalesFactory, _IERSConventions, _UTC
    if _FramesFactory is not None:
        return
    initialize_orekit()
    from org.orekit.frames import FramesFactory as _FF  # type: ignore
    from org.orekit.time import AbsoluteDate as _AD  # type: ignore
    from org.orekit.time import TimeScalesFactory as _TSF  # type: ignore
    from org.orekit.utils import IERSConventions as _IC  # type: ignore

    _FramesFactory = _FF
    _AbsoluteDateType = _AD
    _TimeScalesFactory = _TSF
    _IERSConventions = _IC
    _UTC = _TimeScalesFactory.getUTC()


def _normalize_frame_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _resolve_iers(iers: Optional[Union[object, str]]) -> object:
    _ensure_orekit_bindings()
    if iers is None:
        return _IERSConventions.IERS_2010
    if isinstance(iers, str):
        key = _normalize_frame_name(iers)
        if key in ("iers2010", "2010"):
            return _IERSConventions.IERS_2010
        if key in ("iers2003", "2003"):
            return _IERSConventions.IERS_2003
        if key in ("iers1996", "1996"):
            return _IERSConventions.IERS_1996
        raise ValueError("iers must be one of: 'IERS_2010', 'IERS_2003', 'IERS_1996'")
    return iers


def _get_itrf(iers: object, simple_eop: bool):
    key = (iers, bool(simple_eop))
    fr = _ITRF_CACHE.get(key)
    if fr is None:
        fr = _FramesFactory.getITRF(iers, bool(simple_eop))
        _ITRF_CACHE[key] = fr
    return fr


def _resolve_frame(name: str, *, iers: object, simple_eop: bool):
    _ensure_orekit_bindings()
    key = _normalize_frame_name(name)
    key = {
        "j2000": "eme2000",
        "gcrs": "gcrf",
        "itrs": "itrf",
        "ecef": "itrf",
        "eci": "gcrf",
        "veis": "veis1950",
        "veis50": "veis1950",
        "meanofdate": "mod",
        "trueofdate": "tod",
    }.get(key, key)

    if key == "gcrf":
        return _FramesFactory.getGCRF()
    if key == "icrf":
        return _FramesFactory.getICRF()
    if key == "eme2000":
        return _FramesFactory.getEME2000()
    if key == "teme":
        return _FramesFactory.getTEME()
    if key == "mod":
        return _FramesFactory.getMOD(iers)
    if key == "tod":
        return _FramesFactory.getTOD(iers, bool(simple_eop))
    if key == "cirf":
        return _FramesFactory.getCIRF(iers, bool(simple_eop))
    if key == "ecliptic":
        return _FramesFactory.getEcliptic(iers)
    if key == "veis1950":
        return _FramesFactory.getVeis1950()
    if key == "itrf":
        return _get_itrf(iers, simple_eop)
    raise ValueError(
        f"Unsupported frame '{name}'. Supported: gcrf, icrf, eme2000/j2000, "
        "teme, mod, tod, cirf, ecliptic, veis1950, itrf/ecef."
    )


def _astropy_scalar_to_datetime_utc(t: "AstropyTime") -> datetime:  # type: ignore
    dt = t.utc.to_datetime(timezone=timezone.utc)
    if isinstance(dt, np.ndarray):
        dt = dt.item()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _absolutedate_from_datetime_utc(dt: datetime):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    sec = float(dt.second) + float(dt.microsecond) * 1e-6
    return _AbsoluteDateType(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec, _UTC)


def _is_astropy_scalar_time(x: Any) -> bool:
    return (
        AstropyTime is not None
        and isinstance(x, AstropyTime)
        and getattr(x, "shape", None) == ()
    )


def _is_astropy_array_time(x: Any) -> bool:
    return (
        AstropyTime is not None
        and isinstance(x, AstropyTime)
        and getattr(x, "shape", None) != ()
    )


def _is_absolute_date(x: Any) -> bool:
    return isinstance(x, _AbsoluteDateType)


def _to_absolutedate_scalar(t: Any):
    if _is_absolute_date(t):
        return t
    if _is_astropy_scalar_time(t):
        return _absolutedate_from_datetime_utc(_astropy_scalar_to_datetime_utc(t))
    raise TypeError(
        "times must be org.orekit.time.AbsoluteDate, astropy.time.Time scalar/array, "
        "or an iterable of AbsoluteDate / scalar astropy times."
    )


def _normalize_states_array(x: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] != 3:
            raise ValueError(f"{name} must be (3,) or (N,3); got {a.shape}")
        return a.reshape(1, 3), True
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be (3,) or (N,3); got {a.shape}")
    return a, False


@dataclass(frozen=True)
class _KinematicParts:
    R: np.ndarray
    t: np.ndarray
    v: np.ndarray
    omega: np.ndarray


def _vec3_to_np(v) -> np.ndarray:
    return np.array([v.getX(), v.getY(), v.getZ()], dtype=np.float64)


def _get_kinematic_parts(from_frame, to_frame, date) -> _KinematicParts:
    tr = from_frame.getTransformTo(to_frame, date)
    rot = tr.getRotation()
    R = np.array(rot.getMatrix(), dtype=np.float64)
    t = _vec3_to_np(tr.getTranslation())
    v = _vec3_to_np(tr.getVelocity())
    omega = _vec3_to_np(tr.getRotationRate())
    return _KinematicParts(R=R, t=t, v=v, omega=omega)


def _apply_transform_pos(parts: _KinematicParts, r_old: np.ndarray) -> np.ndarray:
    return r_old @ parts.R.T + parts.t


def _apply_transform_pos_vel(
    parts: _KinematicParts, r_old: np.ndarray, v_old: np.ndarray
):
    r_new = r_old @ parts.R.T + parts.t
    v_rot = v_old @ parts.R.T
    v_new = v_rot + parts.v - np.cross(parts.omega, r_new)
    return r_new, v_new


def _group_from_inverse(inverse: np.ndarray):
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [inverse.size]))
    return order, starts, ends


def _transform_pos_grouped_by_dates(
    r_old: np.ndarray, dates_list: list, from_frame, to_frame
):
    N = r_old.shape[0]
    ids = np.fromiter((id(d) for d in dates_list), dtype=np.int64, count=N)
    uniq_ids, inverse = np.unique(ids, return_inverse=True)
    K = uniq_ids.size

    if K > 0.8 * N:
        out = np.empty_like(r_old)
        for i, d in enumerate(dates_list):
            parts = _get_kinematic_parts(from_frame, to_frame, d)
            out[i] = _apply_transform_pos(parts, r_old[i : i + 1])[0]
        return out

    id_to_date: dict[int, Any] = {}
    for d in dates_list:
        did = id(d)
        if did not in id_to_date:
            id_to_date[did] = d
            if len(id_to_date) == K:
                break

    order, starts, ends = _group_from_inverse(inverse)
    out = np.empty_like(r_old)
    for g in range(K):
        d = id_to_date[int(uniq_ids[g])]
        parts = _get_kinematic_parts(from_frame, to_frame, d)
        idx = order[starts[g] : ends[g]]
        out[idx] = _apply_transform_pos(parts, r_old[idx])
    return out


def _transform_pos_vel_grouped_by_dates(
    r_old: np.ndarray, v_old: np.ndarray, dates_list: list, from_frame, to_frame
):
    N = r_old.shape[0]
    ids = np.fromiter((id(d) for d in dates_list), dtype=np.int64, count=N)
    uniq_ids, inverse = np.unique(ids, return_inverse=True)
    K = uniq_ids.size

    if K > 0.8 * N:
        r_out = np.empty_like(r_old)
        v_out = np.empty_like(v_old)
        for i, d in enumerate(dates_list):
            parts = _get_kinematic_parts(from_frame, to_frame, d)
            ri, vi = _apply_transform_pos_vel(parts, r_old[i : i + 1], v_old[i : i + 1])
            r_out[i] = ri[0]
            v_out[i] = vi[0]
        return r_out, v_out

    id_to_date: dict[int, Any] = {}
    for d in dates_list:
        did = id(d)
        if did not in id_to_date:
            id_to_date[did] = d
            if len(id_to_date) == K:
                break

    order, starts, ends = _group_from_inverse(inverse)
    r_out = np.empty_like(r_old)
    v_out = np.empty_like(v_old)
    for g in range(K):
        d = id_to_date[int(uniq_ids[g])]
        parts = _get_kinematic_parts(from_frame, to_frame, d)
        idx = order[starts[g] : ends[g]]
        ri, vi = _apply_transform_pos_vel(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def _transform_pos_grouped_by_astropy_time(r_old: np.ndarray, t: "AstropyTime", from_frame, to_frame):  # type: ignore
    jd1 = np.ascontiguousarray(t.utc.jd1, dtype=np.float64)
    jd2 = np.ascontiguousarray(t.utc.jd2, dtype=np.float64)
    key = np.stack([jd1, jd2], axis=1)
    uniq_key, inverse = np.unique(key, axis=0, return_inverse=True)
    K = uniq_key.shape[0]

    if K > 0.8 * r_old.shape[0]:
        dates_list = [_to_absolutedate_scalar(ti) for ti in t.utc]
        return _transform_pos_grouped_by_dates(r_old, dates_list, from_frame, to_frame)

    uniq_dates = []
    for k in range(K):
        ti = AstropyTime(uniq_key[k, 0], uniq_key[k, 1], format="jd", scale="utc")  # type: ignore[misc]
        uniq_dates.append(_to_absolutedate_scalar(ti))

    order, starts, ends = _group_from_inverse(inverse)
    out = np.empty_like(r_old)
    for g in range(K):
        parts = _get_kinematic_parts(from_frame, to_frame, uniq_dates[g])
        idx = order[starts[g] : ends[g]]
        out[idx] = _apply_transform_pos(parts, r_old[idx])
    return out


def _transform_pos_vel_grouped_by_astropy_time(
    r_old: np.ndarray, v_old: np.ndarray, t: "AstropyTime", from_frame, to_frame  # type: ignore
):
    jd1 = np.ascontiguousarray(t.utc.jd1, dtype=np.float64)
    jd2 = np.ascontiguousarray(t.utc.jd2, dtype=np.float64)
    key = np.stack([jd1, jd2], axis=1)
    uniq_key, inverse = np.unique(key, axis=0, return_inverse=True)
    K = uniq_key.shape[0]

    if K > 0.8 * r_old.shape[0]:
        dates_list = [_to_absolutedate_scalar(ti) for ti in t.utc]
        return _transform_pos_vel_grouped_by_dates(
            r_old, v_old, dates_list, from_frame, to_frame
        )

    uniq_dates = []
    for k in range(K):
        ti = AstropyTime(uniq_key[k, 0], uniq_key[k, 1], format="jd", scale="utc")  # type: ignore[misc]
        uniq_dates.append(_to_absolutedate_scalar(ti))

    order, starts, ends = _group_from_inverse(inverse)
    r_out = np.empty_like(r_old)
    v_out = np.empty_like(v_old)
    for g in range(K):
        parts = _get_kinematic_parts(from_frame, to_frame, uniq_dates[g])
        idx = order[starts[g] : ends[g]]
        ri, vi = _apply_transform_pos_vel(parts, r_old[idx], v_old[idx])
        r_out[idx] = ri
        v_out[idx] = vi
    return r_out, v_out


def transform_timed(
    times: Any,
    positions_m: np.ndarray,
    from_frame: str,
    to_frame: str,
    velocities_mps: Optional[np.ndarray] = None,
    *,
    iers: Optional[Union[object, str]] = None,
    simple_eop: bool = True,
):
    """
    Transform positions (and optional velocities) between named Orekit frames.

    Parameters
    ----------
    times
        Scalar AbsoluteDate / scalar astropy Time, astropy Time array, or iterable of
        AbsoluteDate / scalar astropy Time values.
    positions_m : np.ndarray
        Input positions in `from_frame`, shape (3,) or (N,3), meters.
    from_frame, to_frame : str
        Frame names (e.g., "gcrf", "itrf", "teme", "j2000"/"eme2000").
    velocities_mps : np.ndarray | None
        Optional input velocities in `from_frame`, shape (3,) or (N,3), m/s.
        If provided, the function returns `(positions, velocities)`.
    iers : object | str | None
        IERS convention object or one of "IERS_2010", "IERS_2003", "IERS_1996".
    simple_eop : bool
        Orekit simple EOP flag when resolving Earth-fixed frames.
    """
    _ensure_orekit_bindings()

    r_in, r_scalar = _normalize_states_array(positions_m, "positions_m")
    has_vel = velocities_mps is not None
    if has_vel:
        v_in, v_scalar = _normalize_states_array(velocities_mps, "velocities_mps")
        if r_in.shape[0] != v_in.shape[0]:
            raise ValueError(
                f"positions_m and velocities_mps must share N; got {r_in.shape[0]} and {v_in.shape[0]}"
            )
        if r_scalar != v_scalar:
            if r_in.shape[0] == 1 and v_in.shape[0] > 1:
                r_in = np.repeat(r_in, v_in.shape[0], axis=0)
            elif v_in.shape[0] == 1 and r_in.shape[0] > 1:
                v_in = np.repeat(v_in, r_in.shape[0], axis=0)

    N = r_in.shape[0]

    iers_obj = _resolve_iers(iers)
    from_fr = _resolve_frame(from_frame, iers=iers_obj, simple_eop=bool(simple_eop))
    to_fr = _resolve_frame(to_frame, iers=iers_obj, simple_eop=bool(simple_eop))

    if from_fr == to_fr:
        if _is_absolute_date(times) or _is_astropy_scalar_time(times):
            n_time = 1
        elif _is_astropy_array_time(times):
            n_time = int(np.prod(times.shape))
        else:
            n_time = len(list(times))

        if n_time not in (1, N):
            if N == 1 and n_time > 1:
                r_in = np.repeat(r_in, n_time, axis=0)
                if has_vel:
                    v_in = np.repeat(v_in, n_time, axis=0)
            else:
                raise ValueError(f"times length {n_time} must match N={N}")

        if has_vel:
            r_out = r_in.copy()
            v_out = v_in.copy()
            if r_scalar and r_out.shape[0] == 1:
                return r_out[0], v_out[0]
            return r_out, v_out
        r_out = r_in.copy()
        if r_scalar and r_out.shape[0] == 1:
            return r_out[0]
        return r_out

    # Scalar time fast path.
    if _is_absolute_date(times) or _is_astropy_scalar_time(times):
        d0 = _to_absolutedate_scalar(times)
        parts = _get_kinematic_parts(from_fr, to_fr, d0)
        if has_vel:
            r_out, v_out = _apply_transform_pos_vel(parts, r_in, v_in)
            if r_scalar:
                return r_out[0], v_out[0]
            return r_out, v_out
        r_out = _apply_transform_pos(parts, r_in)
        if r_scalar:
            return r_out[0]
        return r_out

    # Astropy vector time path.
    if _is_astropy_array_time(times):
        if getattr(times, "shape", None) != (N,):
            if N == 1:
                r_in = np.repeat(r_in, int(np.prod(times.shape)), axis=0)
                if has_vel:
                    v_in = np.repeat(v_in, int(np.prod(times.shape)), axis=0)
                N = r_in.shape[0]
                times = times.reshape((N,))
            else:
                raise ValueError(
                    f"Astropy Time shape {times.shape} must be (N,) where N={N}"
                )
        if has_vel:
            r_out, v_out = _transform_pos_vel_grouped_by_astropy_time(r_in, v_in, times, from_fr, to_fr)  # type: ignore[arg-type]
            return r_out, v_out
        return _transform_pos_grouped_by_astropy_time(r_in, times, from_fr, to_fr)  # type: ignore[arg-type]

    # Iterable time path.
    times_list_raw = list(times)
    if len(times_list_raw) != N:
        if N == 1 and len(times_list_raw) > 1:
            r_in = np.repeat(r_in, len(times_list_raw), axis=0)
            if has_vel:
                v_in = np.repeat(v_in, len(times_list_raw), axis=0)
            N = r_in.shape[0]
        else:
            raise ValueError(f"times length {len(times_list_raw)} must match N={N}")

    dates_list = []
    for t in times_list_raw:
        dates_list.append(_to_absolutedate_scalar(t))

    if has_vel:
        r_out, v_out = _transform_pos_vel_grouped_by_dates(
            r_in, v_in, dates_list, from_fr, to_fr
        )
        return r_out, v_out
    return _transform_pos_grouped_by_dates(r_in, dates_list, from_fr, to_fr)


def transform_positions_timed(
    times: Any,
    positions_m: np.ndarray,
    from_frame: str,
    to_frame: str,
    *,
    iers: Optional[Union[object, str]] = None,
    simple_eop: bool = True,
) -> np.ndarray:
    return transform_timed(
        times,
        positions_m,
        from_frame,
        to_frame,
        velocities_mps=None,
        iers=iers,
        simple_eop=simple_eop,
    )


def transform_pos_vel_timed(
    times: Any,
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    from_frame: str,
    to_frame: str,
    *,
    iers: Optional[Union[object, str]] = None,
    simple_eop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return transform_timed(
        times,
        positions_m,
        from_frame,
        to_frame,
        velocities_mps=velocities_mps,
        iers=iers,
        simple_eop=simple_eop,
    )


def _normalize_component_input(x: Any, name: str) -> tuple[np.ndarray, bool]:
    if isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, bool):
        return np.asarray([float(x)], dtype=np.float64), True

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64), True
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D array; got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64), False


def _normalize_astropy_time_for_coarse(times: Any) -> tuple[np.ndarray, bool]:
    if AstropyTime is None:
        raise RuntimeError("astropy is required for timed coarse ECI2ITRF wrappers")
    if not isinstance(times, AstropyTime):
        raise TypeError("times must be an astropy.time.Time scalar or 1D array")

    is_scalar = getattr(times, "shape", None) == ()
    if is_scalar:
        return np.asarray([float(times.ut1.jd)], dtype=np.float64), True

    if getattr(times, "ndim", 1) != 1:
        raise ValueError(
            f"times must be scalar or 1D astropy.time.Time; got shape {times.shape}"
        )

    jd_ut1 = np.asarray(times.ut1.jd, dtype=np.float64)
    return np.ascontiguousarray(jd_ut1), False


def _broadcast_length_for_components(
    jd_ut1: np.ndarray, arrays: list[np.ndarray]
) -> int:
    n_time = int(jd_ut1.shape[0])
    lengths = [n_time] + [int(a.shape[0]) for a in arrays]
    n = max(lengths)
    for ln in lengths:
        if ln not in (1, n):
            raise ValueError(
                f"Input lengths must be broadcast-compatible (1 or N={n}); got {lengths}"
            )
    return n


def _repeat_if_needed(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, n)
    raise ValueError("Unexpected broadcast failure")


def coarse_eci2itrf(
    times: Any,
    x_eci_m: Any,
    y_eci_m: Any,
    z_eci_m: Any,
    *,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Coarse ECI->ITRF conversion using astropy time(s) directly.

    Parameters
    ----------
    times : astropy.time.Time
        Scalar or 1D time array. UT1/TT Julian dates are derived internally.
    x_eci_m, y_eci_m, z_eci_m : float | np.ndarray
        ECI components [m], each scalar or 1D array.
    xp_rad, yp_rad : float, optional
        Polar motion coordinates [rad].

    Returns
    -------
    (x_itrf_m, y_itrf_m, z_itrf_m)
        Scalars for scalar inputs, otherwise 1D arrays.
    """
    jd_ut1, t_scalar = _normalize_astropy_time_for_coarse(times)
    jd_tt = np.asarray(
        times.tt.jd if not t_scalar else [float(times.tt.jd)], dtype=np.float64
    )

    x_arr, x_scalar = _normalize_component_input(x_eci_m, "x_eci_m")
    y_arr, y_scalar = _normalize_component_input(y_eci_m, "y_eci_m")
    z_arr, z_scalar = _normalize_component_input(z_eci_m, "z_eci_m")

    n = _broadcast_length_for_components(jd_ut1, [x_arr, y_arr, z_arr])
    x_arr = _repeat_if_needed(x_arr, n)
    y_arr = _repeat_if_needed(y_arr, n)
    z_arr = _repeat_if_needed(z_arr, n)
    jd_ut1 = _repeat_if_needed(jd_ut1, n)
    jd_tt = _repeat_if_needed(jd_tt, n)

    if n == 1:
        x, y, z = _coarse_eci2itrf_pos_kernel(
            float(x_arr[0]),
            float(y_arr[0]),
            float(z_arr[0]),
            float(jd_ut1[0]),
            float(jd_tt[0]),
            float(xp_rad),
            float(yp_rad),
        )
        if t_scalar and x_scalar and y_scalar and z_scalar:
            return float(x), float(y), float(z)
        return (
            np.asarray([x], dtype=np.float64),
            np.asarray([y], dtype=np.float64),
            np.asarray([z], dtype=np.float64),
        )

    r_eci = np.column_stack((x_arr, y_arr, z_arr)).astype(np.float64)
    r_itrf = _coarse_eci2itrf_pos_vec_kernel(
        r_eci,
        jd_ut1.astype(np.float64),
        jd_tt.astype(np.float64),
        float(xp_rad),
        float(yp_rad),
    )
    return r_itrf[:, 0], r_itrf[:, 1], r_itrf[:, 2]


def coarse_eci2itrf_pos_vel(
    times: Any,
    x_eci_m: Any,
    y_eci_m: Any,
    z_eci_m: Any,
    vx_eci_mps: Any,
    vy_eci_mps: Any,
    vz_eci_mps: Any,
    *,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
):
    """
    Coarse ECI->ITRF position/velocity conversion using astropy time(s).

    Inputs may be scalars or 1D arrays and are broadcast against time length.
    """
    jd_ut1, t_scalar = _normalize_astropy_time_for_coarse(times)
    jd_tt = np.asarray(
        times.tt.jd if not t_scalar else [float(times.tt.jd)], dtype=np.float64
    )

    x_arr, x_scalar = _normalize_component_input(x_eci_m, "x_eci_m")
    y_arr, y_scalar = _normalize_component_input(y_eci_m, "y_eci_m")
    z_arr, z_scalar = _normalize_component_input(z_eci_m, "z_eci_m")
    vx_arr, vx_scalar = _normalize_component_input(vx_eci_mps, "vx_eci_mps")
    vy_arr, vy_scalar = _normalize_component_input(vy_eci_mps, "vy_eci_mps")
    vz_arr, vz_scalar = _normalize_component_input(vz_eci_mps, "vz_eci_mps")

    n = _broadcast_length_for_components(
        jd_ut1,
        [x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr],
    )
    x_arr = _repeat_if_needed(x_arr, n)
    y_arr = _repeat_if_needed(y_arr, n)
    z_arr = _repeat_if_needed(z_arr, n)
    vx_arr = _repeat_if_needed(vx_arr, n)
    vy_arr = _repeat_if_needed(vy_arr, n)
    vz_arr = _repeat_if_needed(vz_arr, n)
    jd_ut1 = _repeat_if_needed(jd_ut1, n)
    jd_tt = _repeat_if_needed(jd_tt, n)

    if n == 1:
        x, y, z, vx, vy, vz = _coarse_eci2itrf_pos_vel_kernel(
            float(x_arr[0]),
            float(y_arr[0]),
            float(z_arr[0]),
            float(vx_arr[0]),
            float(vy_arr[0]),
            float(vz_arr[0]),
            float(jd_ut1[0]),
            float(jd_tt[0]),
            float(xp_rad),
            float(yp_rad),
        )
        if (
            t_scalar
            and x_scalar
            and y_scalar
            and z_scalar
            and vx_scalar
            and vy_scalar
            and vz_scalar
        ):
            return float(x), float(y), float(z), float(vx), float(vy), float(vz)
        return (
            np.asarray([x], dtype=np.float64),
            np.asarray([y], dtype=np.float64),
            np.asarray([z], dtype=np.float64),
            np.asarray([vx], dtype=np.float64),
            np.asarray([vy], dtype=np.float64),
            np.asarray([vz], dtype=np.float64),
        )

    r_eci = np.column_stack((x_arr, y_arr, z_arr)).astype(np.float64)
    v_eci = np.column_stack((vx_arr, vy_arr, vz_arr)).astype(np.float64)
    r_itrf, v_itrf = _coarse_eci2itrf_pos_vel_vec_kernel(
        r_eci,
        v_eci,
        jd_ut1.astype(np.float64),
        jd_tt.astype(np.float64),
        float(xp_rad),
        float(yp_rad),
    )
    return (
        r_itrf[:, 0],
        r_itrf[:, 1],
        r_itrf[:, 2],
        v_itrf[:, 0],
        v_itrf[:, 1],
        v_itrf[:, 2],
    )


__all__ = [
    "transform_timed",
    "transform_positions_timed",
    "transform_pos_vel_timed",
    "coarse_eci2itrf",
    "coarse_eci2itrf_pos_vel",
]
