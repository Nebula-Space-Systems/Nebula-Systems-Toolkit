from __future__ import annotations

import numpy as np

from nebula.localization.measurements.gdop import (
    dop_covariance_matrix,
    gdop_pdop_tdop,
    gdop_pdop_tdop_many_observers,
)


def _numpy_dop(observer_pos: np.ndarray, assets: np.ndarray):
    h = np.empty((assets.shape[0], 4), dtype=np.float64)
    for i in range(assets.shape[0]):
        rho = assets[i] - observer_pos
        r = np.linalg.norm(rho)
        if r == 0.0:
            return np.inf, np.inf, np.inf
        u = rho / r
        h[i, 0] = -u[0]
        h[i, 1] = -u[1]
        h[i, 2] = -u[2]
        h[i, 3] = 1.0

    a = h.T @ h
    q = np.linalg.inv(a)
    pdop = np.sqrt(q[0, 0] + q[1, 1] + q[2, 2])
    tdop = np.sqrt(q[3, 3])
    gdop = np.sqrt(pdop * pdop + tdop * tdop)
    return gdop, pdop, tdop


def test_gdop_tetrahedron_known_solution() -> None:
    obs = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r = 20_200_000.0
    s = 1.0 / np.sqrt(3.0)
    assets = r * np.array(
        [
            [s, s, s],
            [s, -s, -s],
            [-s, s, -s],
            [-s, -s, s],
        ],
        dtype=np.float64,
    )

    gd, pd, td = gdop_pdop_tdop(obs, assets)

    assert np.isclose(pd, 1.5, rtol=0.0, atol=1e-12)
    assert np.isclose(td, 0.5, rtol=0.0, atol=1e-12)
    assert np.isclose(gd, np.sqrt(2.5), rtol=0.0, atol=1e-12)


def test_gdop_matches_numpy_reference_random_geometry() -> None:
    rng = np.random.default_rng(123)
    obs = np.array([6_800_000.0, 1_100_000.0, -900_000.0], dtype=np.float64)

    dirs = rng.standard_normal((12, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.uniform(20_000_000.0, 27_000_000.0, size=12)
    assets = obs + dirs * radii[:, None]

    gd_n, pd_n, td_n = gdop_pdop_tdop(obs, assets.astype(np.float64))
    gd_r, pd_r, td_r = _numpy_dop(obs, assets.astype(np.float64))

    assert np.isclose(gd_n, gd_r, rtol=1e-10, atol=1e-12)
    assert np.isclose(pd_n, pd_r, rtol=1e-10, atol=1e-12)
    assert np.isclose(td_n, td_r, rtol=1e-10, atol=1e-12)


def test_gdop_invariant_to_common_translation_and_scale() -> None:
    rng = np.random.default_rng(7)
    obs = np.array([1000.0, -2000.0, 3000.0], dtype=np.float64)
    dirs = rng.standard_normal((8, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    assets = obs + dirs * 21_000_000.0

    gd0, pd0, td0 = gdop_pdop_tdop(obs, assets)

    shift = np.array([5_000_000.0, -4_000_000.0, 9_000_000.0], dtype=np.float64)
    gd1, pd1, td1 = gdop_pdop_tdop(obs + shift, assets + shift)

    scale = 3.75
    gd2, pd2, td2 = gdop_pdop_tdop(obs * scale, assets * scale)

    assert np.isclose(gd0, gd1, atol=1e-12, rtol=0.0)
    assert np.isclose(pd0, pd1, atol=1e-12, rtol=0.0)
    assert np.isclose(td0, td1, atol=1e-12, rtol=0.0)

    assert np.isclose(gd0, gd2, atol=1e-12, rtol=0.0)
    assert np.isclose(pd0, pd2, atol=1e-12, rtol=0.0)
    assert np.isclose(td0, td2, atol=1e-12, rtol=0.0)


def test_gdop_invalid_geometry_returns_inf() -> None:
    obs = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    assets_too_few = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    gd, pd, td = gdop_pdop_tdop(obs, assets_too_few)
    assert np.isinf(gd) and np.isinf(pd) and np.isinf(td)

    assets_singular = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    gd, pd, td = gdop_pdop_tdop(obs, assets_singular)
    assert np.isinf(gd) and np.isinf(pd) and np.isinf(td)

    q = dop_covariance_matrix(obs, assets_singular)
    assert np.isinf(q).all()


def test_gdop_many_observers_matches_scalar_calls() -> None:
    rng = np.random.default_rng(99)
    observers = rng.uniform(-2.0e6, 2.0e6, size=(10, 3)).astype(np.float64)
    dirs = rng.standard_normal((14, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    assets = (dirs * 24_000_000.0).astype(np.float64)

    out = gdop_pdop_tdop_many_observers(observers, assets)
    assert out.shape == (10, 3)

    ref = np.empty_like(out)
    for i in range(observers.shape[0]):
        ref[i] = np.array(gdop_pdop_tdop(observers[i], assets), dtype=np.float64)
    np.testing.assert_allclose(out, ref, atol=0.0, rtol=0.0)
