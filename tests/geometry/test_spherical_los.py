from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from nstk.geometry.spherical_los import los_clear_sphere, los_clear_sphere_pairwise


def _random_outside_sphere(n: int, radius: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(radius + 1.0e5, radius + 4.0e7, size=n)
    return (u * r[:, None]).astype(np.float64)


def test_spherical_los_scalar_surface_to_zenith_clear() -> None:
    radius = 6_371_000.0
    observer = np.array([radius, 0.0, 0.0], dtype=np.float64)
    target = np.array([radius + 4.0e5, 0.0, 0.0], dtype=np.float64)
    assert bool(los_clear_sphere(observer, target, radius))


def test_spherical_los_scalar_opposite_sides_blocked() -> None:
    radius = 6_371_000.0
    observer = np.array([radius + 5.0e5, 0.0, 0.0], dtype=np.float64)
    target = np.array([-radius - 5.0e5, 0.0, 0.0], dtype=np.float64)
    assert not bool(los_clear_sphere(observer, target, radius))


def test_spherical_los_scalar_tangent_is_blocked() -> None:
    radius = 6_371_000.0
    observer = np.array([-1.0e7, radius, 0.0], dtype=np.float64)
    target = np.array([1.0e7, radius, 0.0], dtype=np.float64)
    assert not bool(los_clear_sphere(observer, target, radius))


def test_spherical_los_radius_sensitivity() -> None:
    observer = np.array([7_000_000.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 7_000_000.0, 0.0], dtype=np.float64)

    assert bool(los_clear_sphere(observer, target, 1_000_000.0))
    assert not bool(los_clear_sphere(observer, target, 6_371_000.0))


def test_spherical_los_center_offset_changes_result() -> None:
    radius = 2.0
    observer = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([10.0, 0.0, 0.0], dtype=np.float64)

    # Sphere centered away from the path: clear.
    assert bool(los_clear_sphere(observer, target, radius, 0.0, 5.0, 0.0))
    # Sphere centered on the path: blocked.
    assert not bool(los_clear_sphere(observer, target, radius, 5.0, 0.0, 0.0))


def test_spherical_los_many_to_many_matches_scalar() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(13, radius, seed=1)
    targets = _random_outside_sphere(19, radius, seed=2)

    got = los_clear_sphere(observers, targets, radius)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_sphere(observers[i], targets[j], radius)

    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, ref)


def test_spherical_los_pairwise_matches_many_to_many_diagonal() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(13, radius, seed=101)
    targets = _random_outside_sphere(13, radius, seed=102)

    pair = los_clear_sphere_pairwise(observers, targets, radius)
    mat = los_clear_sphere(observers, targets, radius)
    np.testing.assert_array_equal(pair, np.diag(mat))


def test_spherical_los_pairwise_rejects_mismatched_row_counts() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(7, radius, seed=110)
    targets = _random_outside_sphere(9, radius, seed=111)
    with pytest.raises(ValueError, match="same number of rows"):
        los_clear_sphere_pairwise(observers, targets, radius)


def test_spherical_los_one_to_many_matches_many_to_many_row() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(7, radius, seed=9)
    targets = _random_outside_sphere(41, radius, seed=10)

    row = los_clear_sphere(observers[0], targets, radius)
    mat = los_clear_sphere(observers[:1], targets, radius)

    assert row.dtype == np.bool_
    np.testing.assert_array_equal(row, mat[0])


def test_spherical_los_many_to_one_matches_many_to_many_column() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(11, radius, seed=21)
    targets = _random_outside_sphere(5, radius, seed=22)

    col = los_clear_sphere(observers, targets[0], radius)
    mat = los_clear_sphere(observers, targets[:1], radius)

    assert col.dtype == np.bool_
    np.testing.assert_array_equal(col, mat[:, 0])


def test_spherical_los_unified_interface_works_inside_njit() -> None:
    radius = 6_371_000.0
    observer = np.array([radius + 4.0e5, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, radius + 5.0e5, 0.0], dtype=np.float64)
    observers = _random_outside_sphere(8, radius, seed=30)
    targets = _random_outside_sphere(9, radius, seed=31)

    @njit(cache=True)
    def los_clear_sphere_scalar_jit(observer_pos, target_pos, sphere_radius, cx, cy, cz):
        return los_clear_sphere(observer_pos, target_pos, sphere_radius, cx, cy, cz)

    @njit(cache=True)
    def los_clear_sphere_row_jit(observer_pos, targets_pos, sphere_radius):
        return los_clear_sphere(observer_pos, targets_pos, sphere_radius)

    @njit(cache=True)
    def los_clear_sphere_col_jit(observers_pos, target_pos, sphere_radius):
        return los_clear_sphere(observers_pos, target_pos, sphere_radius)

    @njit(cache=True)
    def los_clear_sphere_mat_jit(observers_pos, targets_pos, sphere_radius):
        return los_clear_sphere(observers_pos, targets_pos, sphere_radius)

    @njit(cache=True)
    def los_clear_sphere_pair_jit(observers_pos, targets_pos, sphere_radius):
        return los_clear_sphere_pairwise(observers_pos, targets_pos, sphere_radius)

    expected_scalar = los_clear_sphere(observer, target, 2.0, 1.0, -2.0, 0.5)
    got_scalar = los_clear_sphere_scalar_jit(observer, target, 2.0, 1.0, -2.0, 0.5)
    assert bool(got_scalar) == bool(expected_scalar)

    expected_row = los_clear_sphere(observer, targets, radius)
    got_row = los_clear_sphere_row_jit(observer, targets, radius)
    np.testing.assert_array_equal(got_row, expected_row)

    expected_col = los_clear_sphere(observers, target, radius)
    got_col = los_clear_sphere_col_jit(observers, target, radius)
    np.testing.assert_array_equal(got_col, expected_col)

    expected_mat = los_clear_sphere(observers, targets, radius)
    got_mat = los_clear_sphere_mat_jit(observers, targets, radius)
    np.testing.assert_array_equal(got_mat, expected_mat)

    expected_pair = los_clear_sphere_pairwise(observers, targets[:8], radius)
    got_pair = los_clear_sphere_pair_jit(observers, targets[:8], radius)
    np.testing.assert_array_equal(got_pair, expected_pair)
