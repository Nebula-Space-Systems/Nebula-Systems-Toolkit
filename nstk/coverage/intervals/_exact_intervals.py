from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numba import njit, prange

from nstk.coverage.intervals.config import ExactCoverageConfig
from nstk.transforms.constants import WGS84_A, WGS84_E2


_ROOT_EPS = 1e-12
_MERGE_EPS = 1e-10
_EVENT_EPS = 1e-10
_HALF_PI = 0.5 * np.pi
_CUBIC_SCAN_MIN = 8
_CUBIC_SCAN_MAX = 64
_CUBIC_SCAN_TARGET_ANGLE_RAD = np.deg2rad(0.35)
_CUBIC_SCAN_CURVATURE_GAIN = 0.35


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
    target_shape: tuple[int, int] | None = None
    target_lon_deg: np.ndarray | None = None
    target_lat_deg: np.ndarray | None = None
    target_row_offsets: np.ndarray | None = None
    target_lat_rows_deg: np.ndarray | None = None

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

    def has_surface_target_grid(self) -> bool:
        return (
            self.target_lon_deg is not None
            and self.target_lat_deg is not None
            and self.target_row_offsets is not None
            and self.target_lat_rows_deg is not None
        )

    def require_surface_target_grid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.has_surface_target_grid():
            raise ValueError(
                "This access interval store does not include surface target grid metadata. "
                "Build it with compute_access_intervals(...) or supply target grid metadata "
                "to build_access_interval_store(...)."
            )
        return (
            self.target_lon_deg,  # type: ignore[return-value]
            self.target_lat_deg,  # type: ignore[return-value]
            self.target_row_offsets,  # type: ignore[return-value]
            self.target_lat_rows_deg,  # type: ignore[return-value]
        )

    def nearest_target_index(self, lat_deg: float, lon_deg: float) -> int:
        lon_arr, lat_arr, _, _ = self.require_surface_target_grid()

        lat0 = np.deg2rad(float(lat_deg))
        lon0 = np.deg2rad(float(lon_deg))
        lat = np.deg2rad(lat_arr)
        lon = np.deg2rad(lon_arr)

        dlat = lat - lat0
        dlon = (lon - lon0 + np.pi) % (2.0 * np.pi) - np.pi
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(
            dlon / 2.0
        ) ** 2
        return int(np.argmin(a))


def build_surface_targets_from_config(
    config: ExactCoverageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build flattened surface target geometry from `ExactCoverageConfig`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `target_positions` and `target_up_vectors`, shape `(config.n_targets, 3)`.
    """
    n_targets = int(config.n_targets)
    target_positions = np.empty((n_targets, 3), dtype=np.float64)
    target_up = np.empty((n_targets, 3), dtype=np.float64)

    # NOTE:
    # In this Python 3.14 environment, the equivalent vectorized array math inside
    # a function scope was producing corrupted latitude-dependent geometry for some
    # rows, even though the same formulas evaluated correctly at module scope.
    # Build each target with scalar math here so config-backed coverage remains
    # numerically trustworthy across runtimes.
    for idx, (lat_deg, lon_deg) in enumerate(
        zip(config.lat_deg_flat, config.lon_deg_flat)
    ):
        lat_rad = float(np.deg2rad(float(lat_deg)))
        lon_rad = float(np.deg2rad(float(lon_deg)))
        sin_lat = float(np.sin(lat_rad))
        cos_lat = float(np.cos(lat_rad))
        sin_lon = float(np.sin(lon_rad))
        cos_lon = float(np.cos(lon_rad))

        prime_vertical_radius = float(
            WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        )
        xy_radius = prime_vertical_radius * cos_lat
        z_ecef = (1.0 - WGS84_E2) * prime_vertical_radius * sin_lat

        target_positions[idx, 0] = xy_radius * cos_lon
        target_positions[idx, 1] = xy_radius * sin_lon
        target_positions[idx, 2] = z_ecef

        target_up[idx, 0] = cos_lat * cos_lon
        target_up[idx, 1] = cos_lat * sin_lon
        target_up[idx, 2] = sin_lat

    return target_positions, target_up


def compute_access_intervals(
    config: ExactCoverageConfig,
    time: np.ndarray,
    observer_positions: Iterable[np.ndarray],
    *,
    min_elevation_deg: float | None = None,
    max_elevation_deg: float | None = None,
    interpolation: str = "cubic",
    root_tolerance_s: float = 1e-3,
    max_root_iterations: int = 64,
) -> AccessIntervalStore:
    """
    Convenience wrapper for gridded surface coverage from `ExactCoverageConfig`.
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
        target_shape=config.target_shape,
        target_lon_deg=config.lon_deg_flat,
        target_lat_deg=config.lat_deg_flat,
        target_row_offsets=config.row_offsets,
        target_lat_rows_deg=config.lat_deg_rows,
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
    target_shape: tuple[int, int] | None = None,
    target_lon_deg: np.ndarray | None = None,
    target_lat_deg: np.ndarray | None = None,
    target_row_offsets: np.ndarray | None = None,
    target_lat_rows_deg: np.ndarray | None = None,
) -> AccessIntervalStore:
    """
    Build exact access start/stop intervals for all observer-target pairs.

    Notes
    -----
    - Interpolation supports piecewise `linear` and `cubic` (Hermite) observer motion.
    - Candidate transition brackets are found analytically; transition times are then
      root-refined via bisection to `root_tolerance_s` (seconds). For cubic, bracket
      scan density is selected internally and adaptively per segment.
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

    n_obs = int(obs_stack.shape[1])
    obs_vel_stack = np.empty((0, 0, 0), dtype=np.float64)
    cubic_coeff_stack = np.empty((0, 0, 0, 0), dtype=np.float64)
    interp_code = np.int64(0)
    if interp == "cubic":
        obs_vel_stack = _estimate_observer_velocities(times, obs_stack)
        cubic_coeff_stack = _build_cubic_coefficients(times, obs_stack, obs_vel_stack)
        interp_code = np.int64(1)

    n_targets = int(targets.shape[0])
    n_pairs = n_obs * n_targets

    lon_meta = _validate_optional_target_vector(
        target_lon_deg,
        n_targets,
        name="target_lon_deg",
    )
    lat_meta = _validate_optional_target_vector(
        target_lat_deg,
        n_targets,
        name="target_lat_deg",
    )
    row_offsets_meta, lat_rows_meta = _validate_optional_surface_grid_rows(
        target_row_offsets,
        target_lat_rows_deg,
        n_targets,
    )

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
            target_shape=target_shape,
            target_lon_deg=lon_meta,
            target_lat_deg=lat_meta,
            target_row_offsets=row_offsets_meta,
            target_lat_rows_deg=lat_rows_meta,
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
        cubic_coeff_stack,
        min_el,
        max_el,
        sin2_min,
        sin2_max,
        use_max,
        interp_code,
        root_tol_s,
        root_max_iter,
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
        cubic_coeff_stack,
        min_el,
        max_el,
        sin2_min,
        sin2_max,
        use_max,
        interp_code,
        root_tol_s,
        root_max_iter,
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
        target_shape=target_shape,
        target_lon_deg=lon_meta,
        target_lat_deg=lat_meta,
        target_row_offsets=row_offsets_meta,
        target_lat_rows_deg=lat_rows_meta,
    )


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


def _stack_observers(
    times: np.ndarray, observer_positions: Iterable[np.ndarray]
) -> np.ndarray:
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


def _validate_optional_target_vector(
    values: np.ndarray | None,
    n_targets: int,
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_targets,):
        raise ValueError(f"{name} must have shape ({n_targets},)")
    return np.ascontiguousarray(arr)


def _validate_optional_surface_grid_rows(
    row_offsets: np.ndarray | None,
    lat_rows_deg: np.ndarray | None,
    n_targets: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if row_offsets is None and lat_rows_deg is None:
        return None, None
    if row_offsets is None or lat_rows_deg is None:
        raise ValueError(
            "target_row_offsets and target_lat_rows_deg must either both be provided or both be omitted"
        )

    offsets = np.asarray(row_offsets, dtype=np.int64)
    lat_rows = np.asarray(lat_rows_deg, dtype=np.float64)
    if offsets.ndim != 1 or lat_rows.ndim != 1:
        raise ValueError("target_row_offsets and target_lat_rows_deg must be 1D arrays")
    if offsets.size != lat_rows.size + 1:
        raise ValueError("target_row_offsets must have length len(target_lat_rows_deg) + 1")
    if offsets[0] != 0 or offsets[-1] != n_targets:
        raise ValueError(
            "target_row_offsets must start at 0 and end at the number of targets"
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError("target_row_offsets must be non-decreasing")
    return np.ascontiguousarray(offsets), np.ascontiguousarray(lat_rows)


def _estimate_observer_velocities(
    time: np.ndarray, obs_stack: np.ndarray
) -> np.ndarray:
    vel = np.empty_like(obs_stack)
    _estimate_observer_velocities_kernel(time, obs_stack, vel)
    return vel


def _build_cubic_coefficients(
    time: np.ndarray, obs_stack: np.ndarray, obs_vel_stack: np.ndarray
) -> np.ndarray:
    coeff = np.empty((time.size - 1, obs_stack.shape[1], 3, 4), dtype=np.float64)
    _build_cubic_coefficients_kernel(time, obs_stack, obs_vel_stack, coeff)
    return coeff


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


@njit(cache=True, parallel=True)
def _build_cubic_coefficients_kernel(
    time: np.ndarray,
    obs_stack: np.ndarray,
    obs_vel_stack: np.ndarray,
    coeff: np.ndarray,
) -> None:
    nt = time.size
    n_obs = obs_stack.shape[1]
    for ti in prange(nt - 1):
        dt = time[ti + 1] - time[ti]
        for k in range(n_obs):
            for c in range(3):
                r0 = obs_stack[ti, k, c]
                r1 = obs_stack[ti + 1, k, c]
                m0 = obs_vel_stack[ti, k, c] * dt
                m1 = obs_vel_stack[ti + 1, k, c] * dt
                coeff[ti, k, c, 0] = r0
                coeff[ti, k, c, 1] = m0
                coeff[ti, k, c, 2] = -3.0 * r0 + 3.0 * r1 - 2.0 * m0 - m1
                coeff[ti, k, c, 3] = 2.0 * r0 - 2.0 * r1 + m0 + m1


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
def _add_quadratic_roots(
    points: np.ndarray, n: int, q2: float, q1: float, q0: float
) -> int:
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
def _norm3(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)


@njit(cache=True, inline="always")
def _choose_cubic_scan_steps(
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
    dt: float,
) -> int:
    """
    Adaptive cubic bracket scan density from local line-of-sight dynamics.

    We scale scan count by estimated LOS angular rate and apply a mild
    curvature boost from endpoint acceleration. This keeps brackets dense
    when dynamics are fast while avoiding fixed oversampling elsewhere.
    """
    if dt <= 0.0:
        return _CUBIC_SCAN_MIN

    # Relative vectors d(s) at segment endpoints.
    d0x = cx0
    d0y = cy0
    d0z = cz0
    d1x = cx0 + cx1 + cx2 + cx3
    d1y = cy0 + cy1 + cy2 + cy3
    d1z = cz0 + cz1 + cz2 + cz3

    inv_dt = 1.0 / dt
    inv_dt2 = inv_dt * inv_dt

    # Velocity dr/dt at s=0 and s=1 from cubic coefficients.
    v0x = cx1 * inv_dt
    v0y = cy1 * inv_dt
    v0z = cz1 * inv_dt
    v1x = (cx1 + 2.0 * cx2 + 3.0 * cx3) * inv_dt
    v1y = (cy1 + 2.0 * cy2 + 3.0 * cy3) * inv_dt
    v1z = (cz1 + 2.0 * cz2 + 3.0 * cz3) * inv_dt

    # Angular LOS rate |d x v| / |d|^2.
    d0_norm2 = d0x * d0x + d0y * d0y + d0z * d0z
    d1_norm2 = d1x * d1x + d1y * d1y + d1z * d1z
    eps = 1e-18

    c0x = d0y * v0z - d0z * v0y
    c0y = d0z * v0x - d0x * v0z
    c0z = d0x * v0y - d0y * v0x
    c1x = d1y * v1z - d1z * v1y
    c1y = d1z * v1x - d1x * v1z
    c1z = d1x * v1y - d1y * v1x
    w0 = _norm3(c0x, c0y, c0z) / (d0_norm2 + eps)
    w1 = _norm3(c1x, c1y, c1z) / (d1_norm2 + eps)
    w_max = w0 if w0 >= w1 else w1

    # Mild curvature boost from endpoint acceleration.
    a0x = (2.0 * cx2) * inv_dt2
    a0y = (2.0 * cy2) * inv_dt2
    a0z = (2.0 * cz2) * inv_dt2
    a1x = (2.0 * cx2 + 6.0 * cx3) * inv_dt2
    a1y = (2.0 * cy2 + 6.0 * cy3) * inv_dt2
    a1z = (2.0 * cz2 + 6.0 * cz3) * inv_dt2

    v0n = _norm3(v0x, v0y, v0z)
    v1n = _norm3(v1x, v1y, v1z)
    an0 = _norm3(a0x, a0y, a0z)
    an1 = _norm3(a1x, a1y, a1z)
    v_ref = v0n if v0n >= v1n else v1n
    a_ref = an0 if an0 >= an1 else an1

    curvature = a_ref * dt / (v_ref + eps)
    if curvature > 4.0:
        curvature = 4.0
    boost = 1.0 + _CUBIC_SCAN_CURVATURE_GAIN * curvature

    target = _CUBIC_SCAN_TARGET_ANGLE_RAD
    if target <= 0.0:
        target = 1e-6
    n_scan = int(np.ceil((dt * w_max * boost) / target))
    if n_scan < _CUBIC_SCAN_MIN:
        n_scan = _CUBIC_SCAN_MIN
    if n_scan > _CUBIC_SCAN_MAX:
        n_scan = _CUBIC_SCAN_MAX
    return n_scan


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
    n_scan: int,
    node_band: np.ndarray,
    node_min: np.ndarray,
    node_max: np.ndarray,
    seg_start_s: np.ndarray,
    seg_stop_s: np.ndarray,
) -> int:
    s_tol = (root_tol_s / dt) if root_tol_s > 0.0 else 0.0
    if s_tol <= _ROOT_EPS:
        s_tol = _ROOT_EPS
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
    cubic_coeff_stack: np.ndarray,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    interp_code: int,
    root_tol_s: float,
    root_max_iter: int,
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

        node_band = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        node_min = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        node_max = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        seg_start_cub = np.empty(2 * _CUBIC_SCAN_MAX + 3, dtype=np.float64)
        seg_stop_cub = np.empty(2 * _CUBIC_SCAN_MAX + 3, dtype=np.float64)

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

            n_seg = 0
            if interp_code == 0:
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
                cx0 = cubic_coeff_stack[ti, obs_idx, 0, 0] - gx
                cx1 = cubic_coeff_stack[ti, obs_idx, 0, 1]
                cx2 = cubic_coeff_stack[ti, obs_idx, 0, 2]
                cx3 = cubic_coeff_stack[ti, obs_idx, 0, 3]
                cy0 = cubic_coeff_stack[ti, obs_idx, 1, 0] - gy
                cy1 = cubic_coeff_stack[ti, obs_idx, 1, 1]
                cy2 = cubic_coeff_stack[ti, obs_idx, 1, 2]
                cy3 = cubic_coeff_stack[ti, obs_idx, 1, 3]
                cz0 = cubic_coeff_stack[ti, obs_idx, 2, 0] - gz
                cz1 = cubic_coeff_stack[ti, obs_idx, 2, 1]
                cz2 = cubic_coeff_stack[ti, obs_idx, 2, 2]
                cz3 = cubic_coeff_stack[ti, obs_idx, 2, 3]
                n_scan = _choose_cubic_scan_steps(
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
                    dt,
                )

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
                    n_scan,
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
    cubic_coeff_stack: np.ndarray,
    min_el_rad: float,
    max_el_rad: float,
    sin2_min: float,
    sin2_max: float,
    use_max: bool,
    interp_code: int,
    root_tol_s: float,
    root_max_iter: int,
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

        node_band = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        node_min = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        node_max = np.empty(_CUBIC_SCAN_MAX + 1, dtype=np.uint8)
        seg_start_cub = np.empty(2 * _CUBIC_SCAN_MAX + 3, dtype=np.float64)
        seg_stop_cub = np.empty(2 * _CUBIC_SCAN_MAX + 3, dtype=np.float64)

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

            n_seg = 0
            if interp_code == 0:
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
                cx0 = cubic_coeff_stack[ti, obs_idx, 0, 0] - gx
                cx1 = cubic_coeff_stack[ti, obs_idx, 0, 1]
                cx2 = cubic_coeff_stack[ti, obs_idx, 0, 2]
                cx3 = cubic_coeff_stack[ti, obs_idx, 0, 3]
                cy0 = cubic_coeff_stack[ti, obs_idx, 1, 0] - gy
                cy1 = cubic_coeff_stack[ti, obs_idx, 1, 1]
                cy2 = cubic_coeff_stack[ti, obs_idx, 1, 2]
                cy3 = cubic_coeff_stack[ti, obs_idx, 1, 3]
                cz0 = cubic_coeff_stack[ti, obs_idx, 2, 0] - gz
                cz1 = cubic_coeff_stack[ti, obs_idx, 2, 1]
                cz2 = cubic_coeff_stack[ti, obs_idx, 2, 2]
                cz3 = cubic_coeff_stack[ti, obs_idx, 2, 3]
                n_scan = _choose_cubic_scan_steps(
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
                    dt,
                )

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
                    n_scan,
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
