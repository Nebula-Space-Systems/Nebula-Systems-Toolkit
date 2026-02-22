from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numba import njit, prange

from nebula.coverage.config import CoverageConfig


_ROOT_EPS = 1e-12
_MERGE_EPS = 1e-10
_EVENT_EPS = 1e-10
_HALF_PI = 0.5 * np.pi


@dataclass
class AccessIntervalStore:
    """
    Compact interval storage for observer-target access windows.

    Intervals are stored in a CSR-like layout:
    - `pair_offsets[p] : pair_offsets[p+1]` indexes intervals for pair `p`
    - `p = observer_index * n_targets + target_index`
    """

    time_start: float
    time_stop: float
    n_observers: int
    n_targets: int
    pair_offsets: np.ndarray
    start_times: np.ndarray
    stop_times: np.ndarray
    min_elevation_rad: float
    max_elevation_rad: float
    interpolation: str
    root_tolerance_s: float
    root_bracket_substeps: int
    target_shape: tuple[int, int] | None = None

    def pair_index(self, observer_index: int, target_index: int) -> int:
        if observer_index < 0 or observer_index >= self.n_observers:
            raise IndexError("observer_index out of range")
        if target_index < 0 or target_index >= self.n_targets:
            raise IndexError("target_index out of range")
        return observer_index * self.n_targets + target_index

    def pair_intervals(
        self, observer_index: int, target_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.pair_index(observer_index, target_index)
        i0 = int(self.pair_offsets[p])
        i1 = int(self.pair_offsets[p + 1])
        return self.start_times[i0:i1], self.stop_times[i0:i1]

    def reshape_target_values(self, values: np.ndarray) -> np.ndarray:
        if self.target_shape is None:
            return values
        return values.reshape(self.target_shape)


def build_surface_targets_from_config(
    config: CoverageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build flattened surface target geometry from `CoverageConfig`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `target_positions` and `target_up_vectors`, both shape `(nlats*nlons, 3)`.
    """
    ny = int(config.nlats)
    nx = int(config.nlons)
    n_targets = ny * nx

    target_positions = np.empty((n_targets, 3), dtype=np.float64)
    target_up = np.empty((n_targets, 3), dtype=np.float64)

    idx = 0
    for j in range(ny):
        ncos = float(config.Ncos_row_m[j])
        nz = float(config.Nz_row_m[j])
        clat = float(config.cos_lat_row_geod[j])
        slat = float(config.sin_lat_row_geod[j])
        for i in range(nx):
            clon = float(config.cos_lon_col[i])
            slon = float(config.sin_lon_col[i])

            target_positions[idx, 0] = ncos * clon
            target_positions[idx, 1] = ncos * slon
            target_positions[idx, 2] = nz

            target_up[idx, 0] = clat * clon
            target_up[idx, 1] = clat * slon
            target_up[idx, 2] = slat
            idx += 1

    return target_positions, target_up


def build_access_interval_store_from_config(
    config: CoverageConfig,
    time: np.ndarray,
    observer_positions: Iterable[np.ndarray],
    *,
    min_elevation_deg: float | None = None,
    max_elevation_deg: float | None = None,
    interpolation: str = "cubic",
    root_tolerance_s: float = 1e-3,
    max_root_iterations: int = 64,
    root_bracket_substeps: int = 32,
) -> AccessIntervalStore:
    """
    Convenience wrapper for gridded surface coverage from `CoverageConfig`.
    """
    min_el = (
        float(config.min_elevation_deg)
        if min_elevation_deg is None
        else float(min_elevation_deg)
    )
    max_el = (
        float(config.max_elevation_deg)
        if max_elevation_deg is None
        else float(max_elevation_deg)
    )

    target_positions, target_up = build_surface_targets_from_config(config)
    return build_access_interval_store(
        time=time,
        observer_positions=observer_positions,
        target_positions=target_positions,
        target_up_vectors=target_up,
        min_elevation=min_el,
        max_elevation=max_el,
        degrees=True,
        interpolation=interpolation,
        root_tolerance_s=root_tolerance_s,
        max_root_iterations=max_root_iterations,
        root_bracket_substeps=root_bracket_substeps,
        target_shape=(int(config.nlats), int(config.nlons)),
    )


def build_access_interval_store(
    time: np.ndarray,
    observer_positions: Iterable[np.ndarray],
    target_positions: np.ndarray,
    target_up_vectors: np.ndarray,
    *,
    min_elevation: float = 0.0,
    max_elevation: float = 90.0,
    degrees: bool = True,
    interpolation: str = "cubic",
    root_tolerance_s: float = 1e-3,
    max_root_iterations: int = 64,
    root_bracket_substeps: int = 32,
    target_shape: tuple[int, int] | None = None,
) -> AccessIntervalStore:
    """
    Build exact access start/stop intervals for all observer-target pairs.

    Notes
    -----
    - Interpolation supports piecewise `linear` and `cubic` (Hermite) observer motion.
    - Candidate transition brackets are found analytically; transition times are then
      root-refined via bisection to `root_tolerance_s` (seconds). For cubic, brackets
      are detected by sign scans with `root_bracket_substeps` subdivisions per segment.
    """
    times = _validate_time_array(time)
    obs_stack = _stack_observers(times, observer_positions)
    targets, target_up = _validate_targets(target_positions, target_up_vectors)

    interp = str(interpolation).lower()
    if interp not in ("linear", "cubic"):
        raise ValueError("interpolation must be 'linear' or 'cubic'")

    root_tol_s = float(root_tolerance_s)
    if root_tol_s < 0.0:
        raise ValueError("root_tolerance_s must be >= 0")

    root_max_iter = int(max_root_iterations)
    if root_max_iter < 1:
        raise ValueError("max_root_iterations must be >= 1")

    bracket_steps = int(root_bracket_substeps)
    if bracket_steps < 4:
        raise ValueError("root_bracket_substeps must be >= 4")

    n_obs = int(obs_stack.shape[1])
    obs_vel_stack = np.empty((0, 0, 0), dtype=np.float64)
    interp_code = np.int64(0)
    if interp == "cubic":
        obs_vel_stack = _estimate_observer_velocities(times, obs_stack)
        interp_code = np.int64(1)

    n_targets = int(targets.shape[0])
    n_pairs = n_obs * n_targets

    min_el = float(min_elevation)
    max_el = float(max_elevation)
    if degrees:
        min_el = float(np.deg2rad(min_el))
        max_el = float(np.deg2rad(max_el))

    if max_el <= min_el + 1e-15:
        pair_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
        return AccessIntervalStore(
            time_start=float(times[0]),
            time_stop=float(times[-1]),
            n_observers=n_obs,
            n_targets=n_targets,
            pair_offsets=pair_offsets,
            start_times=np.empty(0, dtype=np.float64),
            stop_times=np.empty(0, dtype=np.float64),
            min_elevation_rad=min_el,
            max_elevation_rad=max_el,
            interpolation=interp,
            root_tolerance_s=root_tol_s,
            root_bracket_substeps=bracket_steps,
            target_shape=target_shape,
        )

    use_max = bool(max_el < (_HALF_PI - 1e-12))
    sin_min = float(np.sin(min_el))
    sin2_min = sin_min * sin_min
    sin_max = float(np.sin(max_el))
    sin2_max = sin_max * sin_max

    pair_counts = np.zeros(n_pairs, dtype=np.int64)
    _count_pair_intervals_kernel(
        times,
        obs_stack,
        targets,
        target_up,
        obs_vel_stack,
        min_el,
        max_el,
        sin2_min,
        sin2_max,
        use_max,
        interp_code,
        root_tol_s,
        root_max_iter,
        bracket_steps,
        pair_counts,
    )

    pair_offsets = np.empty(n_pairs + 1, dtype=np.int64)
    pair_offsets[0] = 0
    np.cumsum(pair_counts, out=pair_offsets[1:])

    n_intervals = int(pair_offsets[-1])
    start_times = np.empty(n_intervals, dtype=np.float64)
    stop_times = np.empty(n_intervals, dtype=np.float64)

    _fill_pair_intervals_kernel(
        times,
        obs_stack,
        targets,
        target_up,
        obs_vel_stack,
        min_el,
        max_el,
        sin2_min,
        sin2_max,
        use_max,
        interp_code,
        root_tol_s,
        root_max_iter,
        bracket_steps,
        pair_offsets,
        start_times,
        stop_times,
    )

    return AccessIntervalStore(
        time_start=float(times[0]),
        time_stop=float(times[-1]),
        n_observers=n_obs,
        n_targets=n_targets,
        pair_offsets=pair_offsets,
        start_times=start_times,
        stop_times=stop_times,
        min_elevation_rad=min_el,
        max_elevation_rad=max_el,
        interpolation=interp,
        root_tolerance_s=root_tol_s,
        root_bracket_substeps=bracket_steps,
        target_shape=target_shape,
    )


def access_duration_by_target(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Total time in access per target requiring at least `N` concurrent observers.
    """
    n_req = int(N)
    if n_req <= 0:
        raise ValueError("N must be >= 1")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.float64)
    if n_req > store.n_observers:
        return store.reshape_target_values(out) if reshape else out

    _duration_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        n_req,
        t0,
        t1,
        out,
    )
    return store.reshape_target_values(out) if reshape else out


def max_asset_by_target(
    store: AccessIntervalStore,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    reshape: bool = True,
) -> np.ndarray:
    """
    Maximum concurrent observers in access per target over the query window.
    """
    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.int32)
    _max_asset_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        t0,
        t1,
        out,
    )
    return store.reshape_target_values(out) if reshape else out


def mtta_by_target(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    wrap: bool = False,
    no_access_value: float = np.nan,
    reshape: bool = True,
) -> np.ndarray:
    """
    Mean Time To Access (MTTA) per target from precomputed exact intervals.
    """
    n_req = int(N)
    if n_req <= 0:
        raise ValueError("N must be >= 1")

    t0, t1 = _resolve_window(store, t_start, t_stop)
    out = np.zeros(store.n_targets, dtype=np.float64)
    if n_req > store.n_observers:
        out.fill(float(no_access_value))
        return store.reshape_target_values(out) if reshape else out

    _mtta_by_target_kernel(
        store.pair_offsets,
        store.start_times,
        store.stop_times,
        int(store.n_observers),
        int(store.n_targets),
        n_req,
        t0,
        t1,
        bool(wrap),
        float(no_access_value),
        out,
    )
    return store.reshape_target_values(out) if reshape else out


def _resolve_window(
    store: AccessIntervalStore, t_start: float | None, t_stop: float | None
) -> tuple[float, float]:
    t0 = store.time_start if t_start is None else float(t_start)
    t1 = store.time_stop if t_stop is None else float(t_stop)
    if t1 <= t0:
        raise ValueError("Require t_stop > t_start")
    if t0 < store.time_start - 1e-12 or t1 > store.time_stop + 1e-12:
        raise ValueError("Query window must be inside store time bounds")
    return t0, t1


def _validate_time_array(time: np.ndarray) -> np.ndarray:
    times = np.asarray(time, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time must be a 1D array with length >= 2")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing")
    return np.ascontiguousarray(times)


def _stack_observers(times: np.ndarray, observer_positions: Iterable[np.ndarray]) -> np.ndarray:
    obs_list = []
    for obs in observer_positions:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape != (times.size, 3):
            raise ValueError("Each observer array must have shape (len(time), 3)")
        obs_list.append(arr)
    if len(obs_list) == 0:
        raise ValueError("observer_positions must be non-empty")
    return np.ascontiguousarray(np.stack(obs_list, axis=1))


def _validate_targets(
    target_positions: np.ndarray, target_up_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(target_positions, dtype=np.float64)
    up = np.asarray(target_up_vectors, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_positions must have shape (n_targets, 3)")
    if up.shape != targets.shape:
        raise ValueError("target_up_vectors must have shape (n_targets, 3)")
    if targets.shape[0] == 0:
        raise ValueError("target_positions must be non-empty")

    up_norm = np.linalg.norm(up, axis=1)
    if np.any(up_norm <= 0.0):
        raise ValueError("target_up_vectors rows must be non-zero")
    up = up / up_norm[:, None]
    return np.ascontiguousarray(targets), np.ascontiguousarray(up)


def _estimate_observer_velocities(time: np.ndarray, obs_stack: np.ndarray) -> np.ndarray:
    vel = np.empty_like(obs_stack)
    _estimate_observer_velocities_kernel(time, obs_stack, vel)
    return vel


@njit(cache=True, parallel=True)
def _estimate_observer_velocities_kernel(
    time: np.ndarray, obs_stack: np.ndarray, vel: np.ndarray
) -> None:
    nt = time.size
    n_obs = obs_stack.shape[1]
    if nt < 2:
        return

    for k in prange(n_obs):
        dt0 = time[1] - time[0]
        inv_dt0 = 1.0 / dt0
        for c in range(3):
            vel[0, k, c] = (obs_stack[1, k, c] - obs_stack[0, k, c]) * inv_dt0

        for i in range(1, nt - 1):
            dt = time[i + 1] - time[i - 1]
            inv_dt = 1.0 / dt
            for c in range(3):
                vel[i, k, c] = (
                    obs_stack[i + 1, k, c] - obs_stack[i - 1, k, c]
                ) * inv_dt

        dtn = time[nt - 1] - time[nt - 2]
        inv_dtn = 1.0 / dtn
        for c in range(3):
            vel[nt - 1, k, c] = (
                obs_stack[nt - 1, k, c] - obs_stack[nt - 2, k, c]
            ) * inv_dtn


@njit(cache=True, inline="always")
def _add_breakpoint(points: np.ndarray, n: int, x: float) -> int:
    if x <= _ROOT_EPS or x >= (1.0 - _ROOT_EPS):
        return n
    for i in range(n):
        if abs(points[i] - x) <= _ROOT_EPS:
            return n
    points[n] = x
    return n + 1


@njit(cache=True, inline="always")
def _add_quadratic_roots(points: np.ndarray, n: int, q2: float, q1: float, q0: float) -> int:
    if abs(q2) <= _ROOT_EPS:
        if abs(q1) <= _ROOT_EPS:
            return n
        return _add_breakpoint(points, n, -q0 / q1)

    disc = q1 * q1 - 4.0 * q2 * q0
    if disc < 0.0:
        return n
    if disc < _ROOT_EPS:
        return _add_breakpoint(points, n, -0.5 * q1 / q2)

    root = np.sqrt(disc)
    n = _add_breakpoint(points, n, (-q1 - root) / (2.0 * q2))
    n = _add_breakpoint(points, n, (-q1 + root) / (2.0 * q2))
    return n


@njit(cache=True, inline="always")
def _collect_breakpoints(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    points: np.ndarray,
) -> int:
    n = 0
    points[n] = 0.0
    n += 1
    points[n] = 1.0
    n += 1

    q2 = b * b - sin2_min * e
    q1 = 2.0 * a * b - sin2_min * d
    q0 = a * a - sin2_min * c
    n = _add_quadratic_roots(points, n, q2, q1, q0)
    if abs(b) > _ROOT_EPS:
        n = _add_breakpoint(points, n, -a / b)

    if use_max:
        q2 = b * b - sin2_max * e
        q1 = 2.0 * a * b - sin2_max * d
        q0 = a * a - sin2_max * c
        n = _add_quadratic_roots(points, n, q2, q1, q0)
        if abs(b) > _ROOT_EPS:
            n = _add_breakpoint(points, n, -a / b)

    for i in range(1, n):
        key = points[i]
        j = i - 1
        while j >= 0 and points[j] > key:
            points[j + 1] = points[j]
            j -= 1
        points[j + 1] = key

    m = 1
    last = points[0]
    for i in range(1, n):
        if points[i] - last > _ROOT_EPS:
            points[m] = points[i]
            last = points[i]
            m += 1
    return m


@njit(cache=True, inline="always")
def _passes_threshold_at_s(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    s: float,
    sin2_el: float,
) -> bool:
    du = a + b * s
    if du < 0.0:
        return False
    v2 = c + d * s + e * s * s
    return (du * du - sin2_el * v2) >= 0.0


@njit(cache=True, inline="always")
def _band_access_state_at_s(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    s: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
) -> bool:
    if not _passes_threshold_at_s(a, b, c, d, e, s, sin2_min):
        return False
    if use_max and _passes_threshold_at_s(a, b, c, d, e, s, sin2_max):
        return False
    return True


@njit(cache=True, inline="always")
def _elevation_minus_threshold(
    a: float, b: float, c: float, d: float, e: float, s: float, threshold_rad: float
) -> float:
    du = a + b * s
    v2 = c + d * s + e * s * s
    h2 = v2 - du * du
    if h2 < 0.0:
        h2 = 0.0
    elev = np.arctan2(du, np.sqrt(h2))
    return elev - threshold_rad


@njit(cache=True, inline="always")
def _bisect_elevation_root(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    threshold_rad: float,
    lo: float,
    hi: float,
    s_tol: float,
    max_iter: int,
) -> float:
    flo = _elevation_minus_threshold(a, b, c, d, e, lo, threshold_rad)
    fhi = _elevation_minus_threshold(a, b, c, d, e, hi, threshold_rad)

    if abs(flo) <= _ROOT_EPS:
        return lo
    if abs(fhi) <= _ROOT_EPS:
        return hi
    if flo * fhi > 0.0:
        return 0.5 * (lo + hi)

    l = lo
    r = hi
    fl = flo
    for _ in range(max_iter):
        if (r - l) <= s_tol:
            break
        m = 0.5 * (l + r)
        fm = _elevation_minus_threshold(a, b, c, d, e, m, threshold_rad)
        if abs(fm) <= _ROOT_EPS:
            return m
        if fl * fm <= 0.0:
            r = m
        else:
            l = m
            fl = fm
    return 0.5 * (l + r)


@njit(cache=True, inline="always")
def _refine_transition_s(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    min_el_rad: float,
    max_el_rad: float,
    use_max: bool,
    root_tol_s: float,
    root_max_iter: int,
    dt: float,
    left_mid: float,
    right_mid: float,
    pmin_left: bool,
    pmin_right: bool,
    pmax_left: bool,
    pmax_right: bool,
    boundary_s: float,
) -> float:
    if root_tol_s <= 0.0:
        return boundary_s
    if right_mid <= left_mid + _ROOT_EPS:
        return boundary_s

    s_tol = root_tol_s / dt
    if s_tol <= 0.0:
        return boundary_s

    if pmin_left != pmin_right:
        return _bisect_elevation_root(
            a, b, c, d, e, min_el_rad, left_mid, right_mid, s_tol, root_max_iter
        )
    if use_max and (pmax_left != pmax_right):
        return _bisect_elevation_root(
            a, b, c, d, e, max_el_rad, left_mid, right_mid, s_tol, root_max_iter
        )
    return boundary_s


@njit(cache=True, inline="always")
def _segment_access_intervals_s(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    dt: float,
    root_tol_s: float,
    root_max_iter: int,
    bp: np.ndarray,
    pass_min: np.ndarray,
    pass_max: np.ndarray,
    access: np.ndarray,
    seg_start_s: np.ndarray,
    seg_stop_s: np.ndarray,
) -> int:
    n_bp = _collect_breakpoints(a, b, c, d, e, sin2_min, sin2_max, use_max, bp)
    n_int = n_bp - 1
    if n_int <= 0:
        return 0

    for i in range(n_int):
        sa = bp[i]
        sb = bp[i + 1]
        sm = 0.5 * (sa + sb)
        pmin = _passes_threshold_at_s(a, b, c, d, e, sm, sin2_min)
        pass_min[i] = np.uint8(1 if pmin else 0)
        if use_max:
            pmax = _passes_threshold_at_s(a, b, c, d, e, sm, sin2_max)
            pass_max[i] = np.uint8(1 if pmax else 0)
        else:
            pass_max[i] = np.uint8(0)
        acc = pmin and (not (use_max and pass_max[i] != 0))
        access[i] = np.uint8(1 if acc else 0)

    n_seg = 0
    for i in range(n_int):
        if access[i] == 0:
            continue

        sa = bp[i]
        sb = bp[i + 1]

        if i > 0 and access[i - 1] == 0:
            left_mid = 0.5 * (bp[i - 1] + bp[i])
            right_mid = 0.5 * (bp[i] + bp[i + 1])
            sa = _refine_transition_s(
                a,
                b,
                c,
                d,
                e,
                min_el_rad,
                max_el_rad,
                use_max,
                root_tol_s,
                root_max_iter,
                dt,
                left_mid,
                right_mid,
                pass_min[i - 1] != 0,
                pass_min[i] != 0,
                pass_max[i - 1] != 0,
                pass_max[i] != 0,
                bp[i],
            )

        if i < (n_int - 1) and access[i + 1] == 0:
            left_mid = 0.5 * (bp[i] + bp[i + 1])
            right_mid = 0.5 * (bp[i + 1] + bp[i + 2])
            sb = _refine_transition_s(
                a,
                b,
                c,
                d,
                e,
                min_el_rad,
                max_el_rad,
                use_max,
                root_tol_s,
                root_max_iter,
                dt,
                left_mid,
                right_mid,
                pass_min[i] != 0,
                pass_min[i + 1] != 0,
                pass_max[i] != 0,
                pass_max[i + 1] != 0,
                bp[i + 1],
            )

        if sa < 0.0:
            sa = 0.0
        if sb > 1.0:
            sb = 1.0

        if sb > sa + _ROOT_EPS:
            seg_start_s[n_seg] = sa
            seg_stop_s[n_seg] = sb
            n_seg += 1

    return n_seg


@njit(cache=True, inline="always")
def _cubic_rel_component(
    r0: float, r1: float, v0: float, v1: float, dt: float, g: float
) -> tuple[float, float, float, float]:
    m0 = v0 * dt
    m1 = v1 * dt
    c0 = r0 - g
    c1 = m0
    c2 = -3.0 * r0 + 3.0 * r1 - 2.0 * m0 - m1
    c3 = 2.0 * r0 - 2.0 * r1 + m0 + m1
    return c0, c1, c2, c3


@njit(cache=True, inline="always")
def _poly3(c0: float, c1: float, c2: float, c3: float, s: float) -> float:
    return ((c3 * s + c2) * s + c1) * s + c0


@njit(cache=True, inline="always")
def _du_v2_at_s_cubic(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    s: float,
) -> tuple[float, float]:
    dx = _poly3(cx0, cx1, cx2, cx3, s)
    dy = _poly3(cy0, cy1, cy2, cy3, s)
    dz = _poly3(cz0, cz1, cz2, cz3, s)
    du = dx * ux + dy * uy + dz * uz
    v2 = dx * dx + dy * dy + dz * dz
    return du, v2


@njit(cache=True, inline="always")
def _passes_threshold_from_du_v2(du: float, v2: float, sin2_el: float) -> bool:
    if du < 0.0:
        return False
    return (du * du - sin2_el * v2) >= 0.0


@njit(cache=True, inline="always")
def _elevation_minus_threshold_cubic(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    s: float,
    threshold_rad: float,
) -> float:
    dx = _poly3(cx0, cx1, cx2, cx3, s)
    dy = _poly3(cy0, cy1, cy2, cy3, s)
    dz = _poly3(cz0, cz1, cz2, cz3, s)
    du = dx * ux + dy * uy + dz * uz
    v2 = dx * dx + dy * dy + dz * dz
    h2 = v2 - du * du
    if h2 < 0.0:
        h2 = 0.0
    elev = np.arctan2(du, np.sqrt(h2))
    return elev - threshold_rad


@njit(cache=True, inline="always")
def _passes_threshold_at_s_cubic(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    s: float,
    threshold_rad: float,
) -> bool:
    sin2 = np.sin(threshold_rad)
    sin2 = sin2 * sin2
    du, v2 = _du_v2_at_s_cubic(
        cx0, cx1, cx2, cx3, cy0, cy1, cy2, cy3, cz0, cz1, cz2, cz3, ux, uy, uz, s
    )
    return _passes_threshold_from_du_v2(du, v2, sin2)


@njit(cache=True, inline="always")
def _bisect_elevation_root_cubic(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    threshold_rad: float,
    lo: float,
    hi: float,
    s_tol: float,
    max_iter: int,
) -> float:
    flo = _elevation_minus_threshold_cubic(
        cx0,
        cx1,
        cx2,
        cx3,
        cy0,
        cy1,
        cy2,
        cy3,
        cz0,
        cz1,
        cz2,
        cz3,
        ux,
        uy,
        uz,
        lo,
        threshold_rad,
    )
    fhi = _elevation_minus_threshold_cubic(
        cx0,
        cx1,
        cx2,
        cx3,
        cy0,
        cy1,
        cy2,
        cy3,
        cz0,
        cz1,
        cz2,
        cz3,
        ux,
        uy,
        uz,
        hi,
        threshold_rad,
    )

    if abs(flo) <= _ROOT_EPS:
        return lo
    if abs(fhi) <= _ROOT_EPS:
        return hi
    if flo * fhi > 0.0:
        return 0.5 * (lo + hi)

    l = lo
    r = hi
    fl = flo
    for _ in range(max_iter):
        if (r - l) <= s_tol:
            break
        m = 0.5 * (l + r)
        fm = _elevation_minus_threshold_cubic(
            cx0,
            cx1,
            cx2,
            cx3,
            cy0,
            cy1,
            cy2,
            cy3,
            cz0,
            cz1,
            cz2,
            cz3,
            ux,
            uy,
            uz,
            m,
            threshold_rad,
        )
        if abs(fm) <= _ROOT_EPS:
            return m
        if fl * fm <= 0.0:
            r = m
        else:
            l = m
            fl = fm
    return 0.5 * (l + r)


@njit(cache=True, inline="always")
def _add_root_unique(roots: np.ndarray, n: int, x: float, tol: float) -> int:
    if x <= _ROOT_EPS or x >= (1.0 - _ROOT_EPS):
        return n
    for i in range(n):
        if abs(roots[i] - x) <= tol:
            return n
    roots[n] = x
    return n + 1


@njit(cache=True, inline="always")
def _collect_roots_cubic_threshold(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    threshold_rad: float,
    n_scan: int,
    s_tol: float,
    max_iter: int,
    roots: np.ndarray,
) -> int:
    n = 0
    ds = 1.0 / float(n_scan)
    s_prev = 0.0
    f_prev = _elevation_minus_threshold_cubic(
        cx0,
        cx1,
        cx2,
        cx3,
        cy0,
        cy1,
        cy2,
        cy3,
        cz0,
        cz1,
        cz2,
        cz3,
        ux,
        uy,
        uz,
        s_prev,
        threshold_rad,
    )

    if abs(f_prev) <= _ROOT_EPS:
        n = _add_root_unique(roots, n, s_prev, max(s_tol, _ROOT_EPS))

    for j in range(1, n_scan + 1):
        s_cur = 1.0 if j == n_scan else (j * ds)
        f_cur = _elevation_minus_threshold_cubic(
            cx0,
            cx1,
            cx2,
            cx3,
            cy0,
            cy1,
            cy2,
            cy3,
            cz0,
            cz1,
            cz2,
            cz3,
            ux,
            uy,
            uz,
            s_cur,
            threshold_rad,
        )

        if abs(f_cur) <= _ROOT_EPS:
            n = _add_root_unique(roots, n, s_cur, max(s_tol, _ROOT_EPS))
        elif f_prev * f_cur < 0.0:
            r = _bisect_elevation_root_cubic(
                cx0,
                cx1,
                cx2,
                cx3,
                cy0,
                cy1,
                cy2,
                cy3,
                cz0,
                cz1,
                cz2,
                cz3,
                ux,
                uy,
                uz,
                threshold_rad,
                s_prev,
                s_cur,
                s_tol,
                max_iter,
            )
            n = _add_root_unique(roots, n, r, max(s_tol, _ROOT_EPS))

        s_prev = s_cur
        f_prev = f_cur

    return n


@njit(cache=True, inline="always")
def _segment_access_intervals_cubic_s(
    cx0: float,
    cx1: float,
    cx2: float,
    cx3: float,
    cy0: float,
    cy1: float,
    cy2: float,
    cy3: float,
    cz0: float,
    cz1: float,
    cz2: float,
    cz3: float,
    ux: float,
    uy: float,
    uz: float,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    dt: float,
    root_tol_s: float,
    root_max_iter: int,
    root_bracket_substeps: int,
    node_band: np.ndarray,
    node_min: np.ndarray,
    node_max: np.ndarray,
    seg_start_s: np.ndarray,
    seg_stop_s: np.ndarray,
) -> int:
    s_tol = (root_tol_s / dt) if root_tol_s > 0.0 else 0.0
    if s_tol <= _ROOT_EPS:
        s_tol = _ROOT_EPS
    n_scan = root_bracket_substeps
    ds = 1.0 / float(n_scan)

    # Sample band state once on a fixed scan grid.
    for j in range(n_scan + 1):
        s = 1.0 if j == n_scan else (j * ds)
        du, v2 = _du_v2_at_s_cubic(
            cx0, cx1, cx2, cx3, cy0, cy1, cy2, cy3, cz0, cz1, cz2, cz3, ux, uy, uz, s
        )
        pmin = _passes_threshold_from_du_v2(du, v2, sin2_min)
        node_min[j] = np.uint8(1 if pmin else 0)
        if use_max:
            pmax = _passes_threshold_from_du_v2(du, v2, sin2_max)
            node_max[j] = np.uint8(1 if pmax else 0)
        else:
            node_max[j] = np.uint8(0)
        node_band[j] = np.uint8(1 if (pmin and (not (node_max[j] != 0))) else 0)

    n_seg = 0
    is_open = node_band[0] != 0
    open_s = 0.0

    for j in range(n_scan):
        b0 = node_band[j] != 0
        b1 = node_band[j + 1] != 0
        if b0 == b1:
            continue

        s0 = j * ds
        s1 = 1.0 if (j + 1) == n_scan else ((j + 1) * ds)

        # Determine which threshold boundary changed, then refine that root.
        pmin0 = node_min[j] != 0
        pmin1 = node_min[j + 1] != 0
        pmax0 = node_max[j] != 0
        pmax1 = node_max[j + 1] != 0

        threshold = min_el_rad
        if pmin0 != pmin1:
            threshold = min_el_rad
        elif use_max and (pmax0 != pmax1):
            threshold = max_el_rad

        sr = _bisect_elevation_root_cubic(
            cx0,
            cx1,
            cx2,
            cx3,
            cy0,
            cy1,
            cy2,
            cy3,
            cz0,
            cz1,
            cz2,
            cz3,
            ux,
            uy,
            uz,
            threshold,
            s0,
            s1,
            s_tol,
            root_max_iter,
        )
        if sr < 0.0:
            sr = 0.0
        if sr > 1.0:
            sr = 1.0

        if (not b0) and b1:
            is_open = True
            open_s = sr
        elif b0 and (not b1):
            if is_open and sr > open_s + _ROOT_EPS:
                seg_start_s[n_seg] = open_s
                seg_stop_s[n_seg] = sr
                n_seg += 1
            is_open = False

    if is_open:
        if 1.0 > open_s + _ROOT_EPS:
            seg_start_s[n_seg] = open_s
            seg_stop_s[n_seg] = 1.0
            n_seg += 1

    return n_seg


@njit(cache=True, parallel=True)
def _count_pair_intervals_kernel(
    time: np.ndarray,
    obs_stack: np.ndarray,
    target_positions: np.ndarray,
    target_up: np.ndarray,
    obs_vel_stack: np.ndarray,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    interp_code: int,
    root_tol_s: float,
    root_max_iter: int,
    root_bracket_substeps: int,
    pair_counts: np.ndarray,
) -> None:
    nt = time.size
    n_obs = obs_stack.shape[1]
    n_targets = target_positions.shape[0]
    n_pairs = n_obs * n_targets

    for p in prange(n_pairs):
        obs_idx = p // n_targets
        tgt_idx = p - obs_idx * n_targets

        gx = target_positions[tgt_idx, 0]
        gy = target_positions[tgt_idx, 1]
        gz = target_positions[tgt_idx, 2]
        ux_up = target_up[tgt_idx, 0]
        uy_up = target_up[tgt_idx, 1]
        uz_up = target_up[tgt_idx, 2]

        count = 0
        is_open = False
        open_stop = 0.0

        bp_lin = np.empty(10, dtype=np.float64)
        pass_min_lin = np.empty(9, dtype=np.uint8)
        pass_max_lin = np.empty(9, dtype=np.uint8)
        access_lin = np.empty(9, dtype=np.uint8)
        seg_start_lin = np.empty(9, dtype=np.float64)
        seg_stop_lin = np.empty(9, dtype=np.float64)

        node_band = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        node_min = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        node_max = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        seg_start_cub = np.empty(2 * root_bracket_substeps + 3, dtype=np.float64)
        seg_stop_cub = np.empty(2 * root_bracket_substeps + 3, dtype=np.float64)

        for ti in range(nt - 1):
            t0 = time[ti]
            t1 = time[ti + 1]
            dt = t1 - t0
            if dt <= 0.0:
                continue

            o0x = obs_stack[ti, obs_idx, 0]
            o0y = obs_stack[ti, obs_idx, 1]
            o0z = obs_stack[ti, obs_idx, 2]

            o1x = obs_stack[ti + 1, obs_idx, 0]
            o1y = obs_stack[ti + 1, obs_idx, 1]
            o1z = obs_stack[ti + 1, obs_idx, 2]

            vx = o1x - o0x
            vy = o1y - o0y
            vz = o1z - o0z

            d0x = o0x - gx
            d0y = o0y - gy
            d0z = o0z - gz

            a = d0x * ux_up + d0y * uy_up + d0z * uz_up
            b = vx * ux_up + vy * uy_up + vz * uz_up
            c = d0x * d0x + d0y * d0y + d0z * d0z
            d = 2.0 * (d0x * vx + d0y * vy + d0z * vz)
            e = vx * vx + vy * vy + vz * vz

            n_seg = 0
            if interp_code == 0:
                n_seg = _segment_access_intervals_s(
                    a,
                    b,
                    c,
                    d,
                    e,
                    min_el_rad,
                    max_el_rad,
                    sin2_min,
                    sin2_max,
                    use_max,
                    dt,
                    root_tol_s,
                    root_max_iter,
                    bp_lin,
                    pass_min_lin,
                    pass_max_lin,
                    access_lin,
                    seg_start_lin,
                    seg_stop_lin,
                )
            else:
                v0x = obs_vel_stack[ti, obs_idx, 0]
                v0y = obs_vel_stack[ti, obs_idx, 1]
                v0z = obs_vel_stack[ti, obs_idx, 2]
                v1x = obs_vel_stack[ti + 1, obs_idx, 0]
                v1y = obs_vel_stack[ti + 1, obs_idx, 1]
                v1z = obs_vel_stack[ti + 1, obs_idx, 2]

                cx0, cx1, cx2, cx3 = _cubic_rel_component(o0x, o1x, v0x, v1x, dt, gx)
                cy0, cy1, cy2, cy3 = _cubic_rel_component(o0y, o1y, v0y, v1y, dt, gy)
                cz0, cz1, cz2, cz3 = _cubic_rel_component(o0z, o1z, v0z, v1z, dt, gz)

                n_seg = _segment_access_intervals_cubic_s(
                    cx0,
                    cx1,
                    cx2,
                    cx3,
                    cy0,
                    cy1,
                    cy2,
                    cy3,
                    cz0,
                    cz1,
                    cz2,
                    cz3,
                    ux_up,
                    uy_up,
                    uz_up,
                    min_el_rad,
                    max_el_rad,
                    sin2_min,
                    sin2_max,
                    use_max,
                    dt,
                    root_tol_s,
                    root_max_iter,
                    root_bracket_substeps,
                    node_band,
                    node_min,
                    node_max,
                    seg_start_cub,
                    seg_stop_cub,
                )

            for i in range(n_seg):
                if interp_code == 0:
                    ts = t0 + seg_start_lin[i] * dt
                    te = t0 + seg_stop_lin[i] * dt
                else:
                    ts = t0 + seg_start_cub[i] * dt
                    te = t0 + seg_stop_cub[i] * dt
                if not is_open:
                    is_open = True
                    open_stop = te
                else:
                    if ts <= open_stop + _MERGE_EPS:
                        if te > open_stop:
                            open_stop = te
                    else:
                        count += 1
                        open_stop = te

        if is_open:
            count += 1
        pair_counts[p] = count


@njit(cache=True, parallel=True)
def _fill_pair_intervals_kernel(
    time: np.ndarray,
    obs_stack: np.ndarray,
    target_positions: np.ndarray,
    target_up: np.ndarray,
    obs_vel_stack: np.ndarray,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    interp_code: int,
    root_tol_s: float,
    root_max_iter: int,
    root_bracket_substeps: int,
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
) -> None:
    nt = time.size
    n_obs = obs_stack.shape[1]
    n_targets = target_positions.shape[0]
    n_pairs = n_obs * n_targets

    for p in prange(n_pairs):
        obs_idx = p // n_targets
        tgt_idx = p - obs_idx * n_targets

        gx = target_positions[tgt_idx, 0]
        gy = target_positions[tgt_idx, 1]
        gz = target_positions[tgt_idx, 2]
        ux_up = target_up[tgt_idx, 0]
        uy_up = target_up[tgt_idx, 1]
        uz_up = target_up[tgt_idx, 2]

        write_idx = int(pair_offsets[p])
        is_open = False
        open_start = 0.0
        open_stop = 0.0

        bp_lin = np.empty(10, dtype=np.float64)
        pass_min_lin = np.empty(9, dtype=np.uint8)
        pass_max_lin = np.empty(9, dtype=np.uint8)
        access_lin = np.empty(9, dtype=np.uint8)
        seg_start_lin = np.empty(9, dtype=np.float64)
        seg_stop_lin = np.empty(9, dtype=np.float64)

        node_band = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        node_min = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        node_max = np.empty(root_bracket_substeps + 1, dtype=np.uint8)
        seg_start_cub = np.empty(2 * root_bracket_substeps + 3, dtype=np.float64)
        seg_stop_cub = np.empty(2 * root_bracket_substeps + 3, dtype=np.float64)

        for ti in range(nt - 1):
            t0 = time[ti]
            t1 = time[ti + 1]
            dt = t1 - t0
            if dt <= 0.0:
                continue

            o0x = obs_stack[ti, obs_idx, 0]
            o0y = obs_stack[ti, obs_idx, 1]
            o0z = obs_stack[ti, obs_idx, 2]

            o1x = obs_stack[ti + 1, obs_idx, 0]
            o1y = obs_stack[ti + 1, obs_idx, 1]
            o1z = obs_stack[ti + 1, obs_idx, 2]

            vx = o1x - o0x
            vy = o1y - o0y
            vz = o1z - o0z

            d0x = o0x - gx
            d0y = o0y - gy
            d0z = o0z - gz

            a = d0x * ux_up + d0y * uy_up + d0z * uz_up
            b = vx * ux_up + vy * uy_up + vz * uz_up
            c = d0x * d0x + d0y * d0y + d0z * d0z
            d = 2.0 * (d0x * vx + d0y * vy + d0z * vz)
            e = vx * vx + vy * vy + vz * vz

            n_seg = 0
            if interp_code == 0:
                n_seg = _segment_access_intervals_s(
                    a,
                    b,
                    c,
                    d,
                    e,
                    min_el_rad,
                    max_el_rad,
                    sin2_min,
                    sin2_max,
                    use_max,
                    dt,
                    root_tol_s,
                    root_max_iter,
                    bp_lin,
                    pass_min_lin,
                    pass_max_lin,
                    access_lin,
                    seg_start_lin,
                    seg_stop_lin,
                )
            else:
                v0x = obs_vel_stack[ti, obs_idx, 0]
                v0y = obs_vel_stack[ti, obs_idx, 1]
                v0z = obs_vel_stack[ti, obs_idx, 2]
                v1x = obs_vel_stack[ti + 1, obs_idx, 0]
                v1y = obs_vel_stack[ti + 1, obs_idx, 1]
                v1z = obs_vel_stack[ti + 1, obs_idx, 2]

                cx0, cx1, cx2, cx3 = _cubic_rel_component(o0x, o1x, v0x, v1x, dt, gx)
                cy0, cy1, cy2, cy3 = _cubic_rel_component(o0y, o1y, v0y, v1y, dt, gy)
                cz0, cz1, cz2, cz3 = _cubic_rel_component(o0z, o1z, v0z, v1z, dt, gz)

                n_seg = _segment_access_intervals_cubic_s(
                    cx0,
                    cx1,
                    cx2,
                    cx3,
                    cy0,
                    cy1,
                    cy2,
                    cy3,
                    cz0,
                    cz1,
                    cz2,
                    cz3,
                    ux_up,
                    uy_up,
                    uz_up,
                    min_el_rad,
                    max_el_rad,
                    sin2_min,
                    sin2_max,
                    use_max,
                    dt,
                    root_tol_s,
                    root_max_iter,
                    root_bracket_substeps,
                    node_band,
                    node_min,
                    node_max,
                    seg_start_cub,
                    seg_stop_cub,
                )

            for i in range(n_seg):
                if interp_code == 0:
                    ts = t0 + seg_start_lin[i] * dt
                    te = t0 + seg_stop_lin[i] * dt
                else:
                    ts = t0 + seg_start_cub[i] * dt
                    te = t0 + seg_stop_cub[i] * dt
                if not is_open:
                    is_open = True
                    open_start = ts
                    open_stop = te
                else:
                    if ts <= open_stop + _MERGE_EPS:
                        if te > open_stop:
                            open_stop = te
                    else:
                        start_times[write_idx] = open_start
                        stop_times[write_idx] = open_stop
                        write_idx += 1
                        open_start = ts
                        open_stop = te

        if is_open:
            start_times[write_idx] = open_start
            stop_times[write_idx] = open_stop


@njit(cache=True, inline="always")
def _initialize_target_state(
    target_idx: int,
    n_obs: int,
    n_targets: int,
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    t_start: float,
    idx: np.ndarray,
    end_idx: np.ndarray,
    active: np.ndarray,
) -> int:
    count = 0
    for k in range(n_obs):
        p = k * n_targets + target_idx
        i0 = pair_offsets[p]
        i1 = pair_offsets[p + 1]
        i = i0
        while i < i1 and stop_times[i] <= t_start:
            i += 1
        idx[k] = i
        end_idx[k] = i1
        if i < i1 and start_times[i] <= t_start and stop_times[i] > t_start:
            active[k] = 1
            count += 1
        else:
            active[k] = 0
    return count


@njit(cache=True, inline="always")
def _next_event_time(
    n_obs: int,
    idx: np.ndarray,
    end_idx: np.ndarray,
    active: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    t_limit: float,
) -> float:
    t_next = t_limit
    for k in range(n_obs):
        i = idx[k]
        if i >= end_idx[k]:
            continue
        ev = stop_times[i] if active[k] != 0 else start_times[i]
        if ev < t_next:
            t_next = ev
    return t_next


@njit(cache=True, inline="always")
def _apply_events_at_time(
    t: float,
    n_obs: int,
    idx: np.ndarray,
    end_idx: np.ndarray,
    active: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    count: int,
) -> int:
    eps = _EVENT_EPS * (1.0 + abs(t))

    for k in range(n_obs):
        i = idx[k]
        if active[k] != 0 and i < end_idx[k]:
            if abs(stop_times[i] - t) <= eps:
                active[k] = 0
                idx[k] = i + 1
                count -= 1

    for k in range(n_obs):
        i = idx[k]
        if active[k] == 0 and i < end_idx[k]:
            if abs(start_times[i] - t) <= eps:
                active[k] = 1
                count += 1

    return count


@njit(cache=True, parallel=True)
def _duration_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    n_req: int,
    t_start: float,
    t_stop: float,
    out: np.ndarray,
) -> None:
    for target_idx in prange(n_targets):
        idx = np.empty(n_obs, dtype=np.int64)
        end_idx = np.empty(n_obs, dtype=np.int64)
        active = np.zeros(n_obs, dtype=np.uint8)

        count = _initialize_target_state(
            target_idx,
            n_obs,
            n_targets,
            pair_offsets,
            start_times,
            stop_times,
            t_start,
            idx,
            end_idx,
            active,
        )

        cur = t_start
        dur = 0.0

        while cur < t_stop - _ROOT_EPS:
            nxt = _next_event_time(
                n_obs, idx, end_idx, active, start_times, stop_times, t_stop
            )
            if nxt < cur:
                nxt = cur
            if nxt > t_stop:
                nxt = t_stop

            if nxt > cur and count >= n_req:
                dur += nxt - cur

            if nxt >= t_stop - _ROOT_EPS:
                break

            count = _apply_events_at_time(
                nxt, n_obs, idx, end_idx, active, start_times, stop_times, count
            )
            cur = nxt

        out[target_idx] = dur


@njit(cache=True, parallel=True)
def _max_asset_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    t_start: float,
    t_stop: float,
    out: np.ndarray,
) -> None:
    for target_idx in prange(n_targets):
        idx = np.empty(n_obs, dtype=np.int64)
        end_idx = np.empty(n_obs, dtype=np.int64)
        active = np.zeros(n_obs, dtype=np.uint8)

        count = _initialize_target_state(
            target_idx,
            n_obs,
            n_targets,
            pair_offsets,
            start_times,
            stop_times,
            t_start,
            idx,
            end_idx,
            active,
        )

        cur = t_start
        max_count = count

        while cur < t_stop - _ROOT_EPS:
            nxt = _next_event_time(
                n_obs, idx, end_idx, active, start_times, stop_times, t_stop
            )
            if nxt < cur:
                nxt = cur
            if nxt > t_stop:
                nxt = t_stop

            if nxt > cur and count > max_count:
                max_count = count

            if nxt >= t_stop - _ROOT_EPS:
                break

            count = _apply_events_at_time(
                nxt, n_obs, idx, end_idx, active, start_times, stop_times, count
            )
            if count > max_count:
                max_count = count
            cur = nxt

        out[target_idx] = max_count


@njit(cache=True, parallel=True)
def _mtta_by_target_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_obs: int,
    n_targets: int,
    n_req: int,
    t_start: float,
    t_stop: float,
    wrap: bool,
    no_access_value: float,
    out: np.ndarray,
) -> None:
    total_window = t_stop - t_start
    if total_window <= 0.0:
        for i in prange(n_targets):
            out[i] = np.nan
        return

    for target_idx in prange(n_targets):
        idx = np.empty(n_obs, dtype=np.int64)
        end_idx = np.empty(n_obs, dtype=np.int64)
        active = np.zeros(n_obs, dtype=np.uint8)

        count = _initialize_target_state(
            target_idx,
            n_obs,
            n_targets,
            pair_offsets,
            start_times,
            stop_times,
            t_start,
            idx,
            end_idx,
            active,
        )

        access = count >= n_req
        gap_start = t_start
        first_start = -1.0
        if access:
            first_start = t_start

        integral = 0.0
        cur = t_start

        while cur < t_stop - _ROOT_EPS:
            nxt = _next_event_time(
                n_obs, idx, end_idx, active, start_times, stop_times, t_stop
            )
            if nxt < cur:
                nxt = cur
            if nxt > t_stop:
                nxt = t_stop

            if nxt >= t_stop - _ROOT_EPS:
                break

            prev_access = access
            count = _apply_events_at_time(
                nxt, n_obs, idx, end_idx, active, start_times, stop_times, count
            )
            access = count >= n_req

            if (not prev_access) and access:
                gap = nxt - gap_start
                if gap > 0.0:
                    integral += 0.5 * gap * gap
                if first_start < 0.0:
                    first_start = nxt
            elif prev_access and (not access):
                gap_start = nxt

            cur = nxt

        if first_start < 0.0:
            out[target_idx] = no_access_value
            continue

        if not access:
            tail = t_stop - gap_start
            if tail > 0.0:
                if wrap:
                    lead = first_start - t_start
                    if lead < 0.0:
                        lead = 0.0
                    integral += lead * tail + 0.5 * tail * tail
                else:
                    integral += 0.5 * tail * tail

        out[target_idx] = integral / total_window
