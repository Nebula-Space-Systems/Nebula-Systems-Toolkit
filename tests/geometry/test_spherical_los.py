from __future__ import annotations

import numpy as np

from nebula.geometry.spherical_los import (
    los_clear_sphere,
    los_clear_sphere_many_to_many,
    los_clear_sphere_one_to_many,
)


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

    got = los_clear_sphere_many_to_many(observers, targets, radius)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_sphere(observers[i], targets[j], radius)

    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, ref)


def test_spherical_los_one_to_many_matches_many_to_many_row() -> None:
    radius = 6_371_000.0
    observers = _random_outside_sphere(7, radius, seed=9)
    targets = _random_outside_sphere(41, radius, seed=10)

    row = los_clear_sphere_one_to_many(observers[0], targets, radius)
    mat = los_clear_sphere_many_to_many(observers[:1], targets, radius)

    assert row.dtype == np.bool_
    np.testing.assert_array_equal(row, mat[0])
