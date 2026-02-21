from __future__ import annotations

import numpy as np

from nebula.coverage._ellipsoid_los import (
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
