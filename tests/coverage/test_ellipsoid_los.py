from __future__ import annotations

import numpy as np

from nebula.coverage.ellipsoid_los import (
    los_clear_ellipsoid,
    los_clear_ellipsoid_many_to_many,
    los_clear_ellipsoid_one_to_many,
    los_clear_ellipsoid_oriented,
    los_clear_ellipsoid_many_to_many_oriented,
    los_clear_ellipsoid_one_to_many_oriented,
    los_clear_wgs84_ecef,
    los_clear_wgs84_ecef_many_to_many,
    los_clear_wgs84_ecef_one_to_many,
)
from nebula.transform.constants import WGS84_A


def _random_outside_earth(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(WGS84_A + 1.0e5, WGS84_A + 4.0e7, size=n)
    return (u * r[:, None]).astype(np.float64)


def test_los_scalar_surface_to_zenith_clear() -> None:
    observer = np.array([WGS84_A, 0.0, 0.0], dtype=np.float64)
    target = np.array([WGS84_A + 5.0e5, 0.0, 0.0], dtype=np.float64)
    assert bool(los_clear_wgs84_ecef(observer, target))


def test_los_scalar_opposite_sides_blocked() -> None:
    observer = np.array([WGS84_A + 7.0e5, 0.0, 0.0], dtype=np.float64)
    target = np.array([-WGS84_A - 7.0e5, 0.0, 0.0], dtype=np.float64)
    assert not bool(los_clear_wgs84_ecef(observer, target))


def test_los_scalar_tangent_is_blocked() -> None:
    observer = np.array([-1.0e7, WGS84_A, 0.0], dtype=np.float64)
    target = np.array([1.0e7, WGS84_A, 0.0], dtype=np.float64)
    assert not bool(los_clear_wgs84_ecef(observer, target))


def test_los_scalar_inside_endpoint_is_blocked() -> None:
    observer = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([WGS84_A + 4.0e5, 0.0, 0.0], dtype=np.float64)
    assert not bool(los_clear_wgs84_ecef(observer, target))


def test_los_many_to_many_matches_scalar() -> None:
    observers = _random_outside_earth(16, seed=1)
    targets = _random_outside_earth(21, seed=2)

    got = los_clear_wgs84_ecef_many_to_many(observers, targets)
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_wgs84_ecef(observers[i], targets[j])

    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, ref)


def test_los_one_to_many_matches_many_to_many_row() -> None:
    observers = _random_outside_earth(5, seed=7)
    targets = _random_outside_earth(64, seed=8)

    row = los_clear_wgs84_ecef_one_to_many(observers[0], targets)
    mat = los_clear_wgs84_ecef_many_to_many(observers[:1], targets)

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

    got = los_clear_ellipsoid_many_to_many(observers, targets, a, b, c, 1.0, -2.0, 0.5)
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

    row = los_clear_ellipsoid_one_to_many(observer, targets, a, b, c, -3.0, 0.2, 2.0)
    mat = los_clear_ellipsoid_many_to_many(
        observer.reshape(1, 3), targets, a, b, c, -3.0, 0.2, 2.0
    )
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

    got = los_clear_ellipsoid_many_to_many_oriented(
        observers, targets, a, b, c, r_z45, 1.0, 2.0, -1.0
    )
    ref = np.empty((observers.shape[0], targets.shape[0]), dtype=np.bool_)
    for i in range(observers.shape[0]):
        for j in range(targets.shape[0]):
            ref[i, j] = los_clear_ellipsoid_oriented(
                observers[i], targets[j], a, b, c, r_z45, 1.0, 2.0, -1.0
            )
    np.testing.assert_array_equal(got, ref)

    row = los_clear_ellipsoid_one_to_many_oriented(
        observers[0], targets, a, b, c, r_z45, 1.0, 2.0, -1.0
    )
    np.testing.assert_array_equal(row, got[0])
