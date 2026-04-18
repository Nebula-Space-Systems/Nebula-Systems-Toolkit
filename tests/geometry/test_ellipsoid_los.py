from __future__ import annotations

import numpy as np
from numba import njit

from nstk.geometry.ellipsoid_los import (
    los_clear_ellipsoid,
    los_clear_ellipsoid_oriented,
)
from nstk.transforms.constants import WGS84_A, WGS84_B


def _random_outside_earth(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(WGS84_A + 1.0e5, WGS84_A + 4.0e7, size=n)
    return (u * r[:, None]).astype(np.float64)


def test_los_scalar_surface_to_zenith_clear() -> None:
    observer = np.array([WGS84_A, 0.0, 0.0], dtype=np.float64)
    target = np.array([WGS84_A + 5.0e5, 0.0, 0.0], dtype=np.float64)
    assert bool(los_clear_ellipsoid(observer, target, WGS84_A, WGS84_A, WGS84_B))


def test_los_scalar_opposite_sides_blocked() -> None:
    observer = np.array([WGS84_A + 7.0e5, 0.0, 0.0], dtype=np.float64)
    target = np.array([-WGS84_A - 7.0e5, 0.0, 0.0], dtype=np.float64)
    assert not bool(los_clear_ellipsoid(observer, target, WGS84_A, WGS84_A, WGS84_B))


def test_los_scalar_tangent_is_blocked() -> None:
    observer = np.array([-1.0e7, WGS84_A, 0.0], dtype=np.float64)
    target = np.array([1.0e7, WGS84_A, 0.0], dtype=np.float64)
    assert not bool(los_clear_ellipsoid(observer, target, WGS84_A, WGS84_A, WGS84_B))


def test_los_scalar_inside_endpoint_is_blocked() -> None:
    observer = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([WGS84_A + 4.0e5, 0.0, 0.0], dtype=np.float64)
    assert not bool(los_clear_ellipsoid(observer, target, WGS84_A, WGS84_A, WGS84_B))


def test_los_many_to_many_matches_scalar() -> None:
    observers = _random_outside_earth(16, seed=1)
    targets = _random_outside_earth(21, seed=2)

    got = los_clear_ellipsoid(observers, targets, WGS84_A, WGS84_A, WGS84_B)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_ellipsoid(observers[i], targets[j], WGS84_A, WGS84_A, WGS84_B)

    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, ref)


def test_los_one_to_many_matches_many_to_many_row() -> None:
    observers = _random_outside_earth(5, seed=7)
    targets = _random_outside_earth(64, seed=8)

    row = los_clear_ellipsoid(observers[0], targets, WGS84_A, WGS84_A, WGS84_B)
    mat = los_clear_ellipsoid(observers[:1], targets, WGS84_A, WGS84_A, WGS84_B)

    assert row.dtype == np.bool_
    np.testing.assert_array_equal(row, mat[0])


def test_general_ellipsoid_scalar_center_offset_changes_result() -> None:
    # Sphere represented as an ellipsoid with equal axes.
    a = b = c = 2.0
    observer = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([10.0, 0.0, 0.0], dtype=np.float64)

    assert bool(los_clear_ellipsoid(observer, target, a, b, c, 0.0, 5.0, 0.0))
    assert not bool(los_clear_ellipsoid(observer, target, a, b, c, 5.0, 0.0, 0.0))


def test_general_ellipsoid_oriented_changes_result() -> None:
    # Axis-aligned ellipsoid is long in x, short in y: line at y=1.2 is clear.
    # After +90 deg rotation about z, long axis aligns with y: same line is blocked.
    a, b, c = 4.0, 1.0, 1.0
    observer = np.array([-5.0, 1.2, 0.0], dtype=np.float64)
    target = np.array([5.0, 1.2, 0.0], dtype=np.float64)

    r_identity = np.eye(3, dtype=np.float64)
    r_z90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert bool(los_clear_ellipsoid_oriented(observer, target, a, b, c, r_identity))
    assert not bool(los_clear_ellipsoid_oriented(observer, target, a, b, c, r_z90))


def test_general_ellipsoid_many_to_many_matches_scalar() -> None:
    rng = np.random.default_rng(123)
    a, b, c = 3.0, 2.0, 1.0
    observers = rng.uniform(-10.0, 10.0, size=(11, 3)).astype(np.float64)
    targets = rng.uniform(-10.0, 10.0, size=(17, 3)).astype(np.float64)

    got = los_clear_ellipsoid(observers, targets, a, b, c, 1.0, -2.0, 0.5)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_ellipsoid(
                observers[i], targets[j], a, b, c, 1.0, -2.0, 0.5
            )
    np.testing.assert_array_equal(got, ref)


def test_general_ellipsoid_one_to_many_matches_many_to_many_row() -> None:
    rng = np.random.default_rng(456)
    a, b, c = 5.0, 3.0, 2.0
    observer = rng.uniform(-12.0, 12.0, size=3).astype(np.float64)
    targets = rng.uniform(-12.0, 12.0, size=(29, 3)).astype(np.float64)

    row = los_clear_ellipsoid(observer, targets, a, b, c, -3.0, 0.2, 2.0)
    mat = los_clear_ellipsoid(observer.reshape(1, 3), targets, a, b, c, -3.0, 0.2, 2.0)
    np.testing.assert_array_equal(row, mat[0])


def test_general_ellipsoid_oriented_vectorized_matches_scalar() -> None:
    rng = np.random.default_rng(789)
    a, b, c = 4.0, 2.0, 1.5
    observers = rng.uniform(-15.0, 15.0, size=(9, 3)).astype(np.float64)
    targets = rng.uniform(-15.0, 15.0, size=(13, 3)).astype(np.float64)
    r_z45 = np.array(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    got = los_clear_ellipsoid_oriented(observers, targets, a, b, c, r_z45, 1.0, 2.0, -1.0)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_ellipsoid_oriented(
                observers[i], targets[j], a, b, c, r_z45, 1.0, 2.0, -1.0
            )
    np.testing.assert_array_equal(got, ref)

    row = los_clear_ellipsoid_oriented(observers[0], targets, a, b, c, r_z45, 1.0, 2.0, -1.0)
    np.testing.assert_array_equal(row, got[0])


def test_general_ellipsoid_many_to_one_matches_many_to_many_column() -> None:
    rng = np.random.default_rng(987)
    a, b, c = 6.0, 4.0, 2.5
    observers = rng.uniform(-15.0, 15.0, size=(12, 3)).astype(np.float64)
    targets = rng.uniform(-15.0, 15.0, size=(7, 3)).astype(np.float64)

    col = los_clear_ellipsoid(observers, targets[0], a, b, c, 0.5, -1.5, 2.0)
    mat = los_clear_ellipsoid(observers, targets[:1], a, b, c, 0.5, -1.5, 2.0)
    np.testing.assert_array_equal(col, mat[:, 0])


def test_general_ellipsoid_oriented_many_to_one_matches_many_to_many_column() -> None:
    rng = np.random.default_rng(654)
    a, b, c = 4.5, 3.5, 1.5
    observers = rng.uniform(-18.0, 18.0, size=(10, 3)).astype(np.float64)
    targets = rng.uniform(-18.0, 18.0, size=(6, 3)).astype(np.float64)
    r_z30 = np.array(
        [
            [np.sqrt(3.0) / 2.0, -0.5, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    col = los_clear_ellipsoid_oriented(observers, targets[0], a, b, c, r_z30, -2.0, 1.0, 0.5)
    mat = los_clear_ellipsoid_oriented(
        observers,
        targets[:1],
        a,
        b,
        c,
        r_z30,
        -2.0,
        1.0,
        0.5,
    )
    np.testing.assert_array_equal(col, mat[:, 0])


def test_ellipsoid_unified_interfaces_work_inside_njit() -> None:
    observer = np.array([9.0, -1.0, 2.0], dtype=np.float64)
    target = np.array([-7.0, 3.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(2468)
    observers = rng.uniform(-20.0, 20.0, size=(8, 3)).astype(np.float64)
    targets = rng.uniform(-20.0, 20.0, size=(9, 3)).astype(np.float64)
    r_z45 = np.array(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    @njit(cache=True)
    def los_clear_ellipsoid_scalar_jit(
        observer_pos,
        target_pos,
        semi_axis_a,
        semi_axis_b,
        semi_axis_c,
        center_x,
        center_y,
        center_z,
    ):
        return los_clear_ellipsoid(
            observer_pos,
            target_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            center_x,
            center_y,
            center_z,
        )

    @njit(cache=True)
    def los_clear_ellipsoid_row_jit(observer_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c):
        return los_clear_ellipsoid(observer_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c)

    @njit(cache=True)
    def los_clear_ellipsoid_col_jit(observers_pos, target_pos, semi_axis_a, semi_axis_b, semi_axis_c):
        return los_clear_ellipsoid(observers_pos, target_pos, semi_axis_a, semi_axis_b, semi_axis_c)

    @njit(cache=True)
    def los_clear_ellipsoid_mat_jit(observers_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c):
        return los_clear_ellipsoid(observers_pos, targets_pos, semi_axis_a, semi_axis_b, semi_axis_c)

    @njit(cache=True)
    def los_clear_ellipsoid_oriented_mat_jit(
        observers_pos,
        targets_pos,
        semi_axis_a,
        semi_axis_b,
        semi_axis_c,
        orientation_ellipsoid_to_frame,
    ):
        return los_clear_ellipsoid_oriented(
            observers_pos,
            targets_pos,
            semi_axis_a,
            semi_axis_b,
            semi_axis_c,
            orientation_ellipsoid_to_frame,
        )

    expected_scalar = los_clear_ellipsoid(observer, target, 4.0, 3.0, 2.0, 1.0, -2.0, 0.5)
    got_scalar = los_clear_ellipsoid_scalar_jit(observer, target, 4.0, 3.0, 2.0, 1.0, -2.0, 0.5)
    assert bool(got_scalar) == bool(expected_scalar)

    expected_row = los_clear_ellipsoid(observer, targets, 4.0, 3.0, 2.0)
    got_row = los_clear_ellipsoid_row_jit(observer, targets, 4.0, 3.0, 2.0)
    np.testing.assert_array_equal(got_row, expected_row)

    expected_col = los_clear_ellipsoid(observers, target, 4.0, 3.0, 2.0)
    got_col = los_clear_ellipsoid_col_jit(observers, target, 4.0, 3.0, 2.0)
    np.testing.assert_array_equal(got_col, expected_col)

    expected_mat = los_clear_ellipsoid(observers, targets, 4.0, 3.0, 2.0)
    got_mat = los_clear_ellipsoid_mat_jit(observers, targets, 4.0, 3.0, 2.0)
    np.testing.assert_array_equal(got_mat, expected_mat)

    expected_oriented = los_clear_ellipsoid_oriented(observers, targets, 4.0, 3.0, 2.0, r_z45)
    got_oriented = los_clear_ellipsoid_oriented_mat_jit(observers, targets, 4.0, 3.0, 2.0, r_z45)
    np.testing.assert_array_equal(got_oriented, expected_oriented)
