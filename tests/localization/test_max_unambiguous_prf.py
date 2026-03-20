from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from nstk.localization.max_unambiguous_prf import max_unambiguous_prf
from nstk.localization.max_unambiguous_prf import max_unambiguous_prf_batched
from nstk.localization.max_unambiguous_prf import max_unambiguous_prf_batched_const_sigma


# -----------------------------
# Test helpers (pure python)
# -----------------------------
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_C0 = 299792458.0


def geodetic2ecef_wgs84(lat_rad: float, lon_rad: float, h_m: float) -> np.ndarray:
    """ECEF (m) from geodetic lat/lon (rad) and height (m) on WGS84."""
    a = _WGS84_A
    f = _WGS84_F
    e2 = f * (2.0 - f)

    slat = math.sin(lat_rad)
    clat = math.cos(lat_rad)
    slon = math.sin(lon_rad)
    clon = math.cos(lon_rad)

    N = a / math.sqrt(1.0 - e2 * slat * slat)
    x = (N + h_m) * clat * clon
    y = (N + h_m) * clat * slon
    z = (N * (1.0 - e2) + h_m) * slat
    return np.array([x, y, z], dtype=np.float64)


def ellipsoid_surface_from_unit_dir(u: np.ndarray) -> np.ndarray:
    """Project direction u to WGS84 ellipsoid surface by radial intersection."""
    a = _WGS84_A
    b = _WGS84_B
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    s = (ux * ux + uy * uy) * inva2 + (uz * uz) * invb2
    t = 1.0 / math.sqrt(s)
    return np.array([t * ux, t * uy, t * uz], dtype=np.float64)


def random_unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float64)


def ps_vectors(obs_ecef: np.ndarray) -> np.ndarray:
    """p_s = A p where A=diag(1/a^2,1/a^2,1/b^2)."""
    a = _WGS84_A
    b = _WGS84_B
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)
    ps = np.empty_like(obs_ecef, dtype=np.float64)
    ps[:, 0] = obs_ecef[:, 0] * inva2
    ps[:, 1] = obs_ecef[:, 1] * inva2
    ps[:, 2] = obs_ecef[:, 2] * invb2
    return ps


def visible_to_all(x_ecef: np.ndarray, obs_ecef: np.ndarray) -> bool:
    """Exact point LOS condition used by solver: p_s^T x >= 1 for all observers."""
    ps = ps_vectors(obs_ecef)
    vals = ps @ x_ecef
    return bool(np.all(vals >= 1.0 - 1e-14))


def delay_spread_W_seconds(x_ecef: np.ndarray, obs_ecef: np.ndarray) -> float:
    """W(x) = max tau - min tau in seconds."""
    d = np.linalg.norm(obs_ecef - x_ecef[None, :], axis=1)
    tau = d / _C0
    return float(np.max(tau) - np.min(tau))


def margin_hard_seconds(sigmas_s: np.ndarray, k_sigma: float) -> float:
    s = np.sort(sigmas_s.astype(np.float64))
    s1 = float(s[-1])
    s2 = float(s[-2]) if s.size >= 2 else 0.0
    return float(k_sigma * (s1 + s2))


def margin_rss_seconds(sigmas_1sigma_s: np.ndarray, k_sigma: float) -> float:
    s = np.sort(sigmas_1sigma_s.astype(np.float64))
    s1 = float(s[-1])
    s2 = float(s[-2]) if s.size >= 2 else 0.0
    return float(k_sigma * math.sqrt(s1 * s1 + s2 * s2))


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def compile_numba_once() -> None:
    obs = np.ascontiguousarray(
        np.array(
            [
                geodetic2ecef_wgs84(0.0, 0.0, 550e3),
                geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, 550e3),
                geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, 550e3),
            ],
            dtype=np.float64,
        )
    )
    sig = np.ascontiguousarray(np.array([50e-9, 50e-9, 50e-9], dtype=np.float64))
    try:
        max_unambiguous_prf(obs, sig, 1000.0, 1.0, 10000, 1)
    except Exception:
        pass


@pytest.fixture(scope="module")
def baseline_geometry() -> Tuple[np.ndarray, np.ndarray]:
    h = 550e3
    obs = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    sig = np.array([200e-9, 200e-9, 200e-9], dtype=np.float64)
    return np.ascontiguousarray(obs), np.ascontiguousarray(sig)


# -----------------------------
# Tests
# -----------------------------
def test_returns_are_self_consistent(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_hz = 10.0

    prf_lo, prf_hi, prf_gap, pri_u_us, pri_l_us, pri_gap_us = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=1.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    assert np.isfinite([prf_lo, prf_hi, prf_gap, pri_u_us, pri_l_us, pri_gap_us]).all()
    assert prf_lo > 0.0
    assert prf_hi >= prf_lo
    assert prf_gap >= 0.0
    assert prf_gap <= tol_hz + 1e-12
    assert pri_u_us >= pri_l_us > 0.0
    assert abs((pri_u_us - pri_l_us) - pri_gap_us) < 1e-9

    # strict-pad means safe PRF is not above the reciprocal of PRI upper bound
    assert prf_lo <= (1.0 / (pri_u_us * 1e-6))


def test_permutation_invariance(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_hz = 10.0

    out1 = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=1.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )
    perm = np.array([2, 0, 1], dtype=np.int64)
    out2 = max_unambiguous_prf(
        np.ascontiguousarray(obs[perm]),
        np.ascontiguousarray(sig[perm]),
        tol_prf_hz=tol_hz,
        k_sigma=1.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    # Pri bounds should be effectively identical.
    assert abs(out1[3] - out2[3]) < 1e-6  # pri_upper_us
    assert abs(out1[4] - out2[4]) < 1e-6  # pri_lower_us
    assert abs(out1[0] - out2[0]) < 1e-9  # prf_lower_hz


def test_noise_shifts_pri_by_expected_constant_hard_margin(
    compile_numba_once, baseline_geometry
) -> None:
    obs, sig = baseline_geometry
    k_sigma = 3.0
    tol_hz = 12.0

    out1 = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=k_sigma,
        max_nodes=1_000_000,
        max_iters=200_000,
    )
    sig0 = np.zeros_like(sig)
    out0 = max_unambiguous_prf(
        obs,
        sig0,
        tol_prf_hz=tol_hz,
        k_sigma=k_sigma,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    pri_u1, pri_l1, pri_gap1 = out1[3], out1[4], out1[5]
    pri_u0, pri_l0, pri_gap0 = out0[3], out0[4], out0[5]

    expected_delta_us = 2.0 * (margin_hard_seconds(sig, k_sigma) - 0.0) * 1e6
    slack_us = pri_gap1 + pri_gap0 + 1e-3

    assert abs((pri_u1 - pri_u0) - expected_delta_us) <= slack_us
    assert abs((pri_l1 - pri_l0) - expected_delta_us) <= slack_us
    assert out1[0] <= out0[0]  # safe PRF should drop with added timing uncertainty


def test_margin_mode_ordering(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    k_sigma = 3.0
    tol_hz = 12.0

    out_hard = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=k_sigma,
        max_nodes=1_000_000,
        max_iters=200_000,
        margin_mode=0,
    )
    out_rss = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=k_sigma,
        max_nodes=1_000_000,
        max_iters=200_000,
        margin_mode=1,
    )

    # Hard-bound margin >= RSS margin for same sigmas.
    assert margin_hard_seconds(sig, k_sigma) >= margin_rss_seconds(sig, k_sigma)
    assert out_hard[3] >= out_rss[3]  # pri_upper_us
    assert out_hard[0] <= out_rss[0]  # prf_lower_hz


def test_k_sigma_monotonicity(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_hz = 12.0

    out1 = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=1.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )
    out3 = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=tol_hz,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    assert out3[3] >= out1[3]  # pri_upper_us
    assert out3[0] <= out1[0]  # prf_lower_hz


def test_safe_for_random_visible_points(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    k_sigma = 3.0

    prf_lo, _prf_hi, _prf_gap, _pri_u_us, _pri_l_us, _pri_gap_us = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=12.0,
        k_sigma=k_sigma,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    # Convert returned safe PRF to explicit safe PRI and verify no-wrap on sampled points.
    pri_safe_s = 1.0 / prf_lo
    half_pri_safe = 0.5 * pri_safe_s
    margin = margin_hard_seconds(sig, k_sigma)

    u = random_unit_vectors(6000, seed=1)
    xs = np.stack([ellipsoid_surface_from_unit_dir(ui) for ui in u], axis=0)
    vis_mask = np.array([visible_to_all(x, obs) for x in xs], dtype=bool)
    assert int(vis_mask.sum()) > 50

    # Allow tiny numerical slack.
    slack_s = 5e-12
    for x in xs[vis_mask][:500]:
        W = delay_spread_W_seconds(x, obs)
        assert (W + margin) <= (half_pri_safe + slack_s)


def test_empty_overlap_raises(compile_numba_once) -> None:
    obs = np.ascontiguousarray(
        np.array(
            [
                geodetic2ecef_wgs84(0.0, 0.0, 0.0),
                geodetic2ecef_wgs84(0.0, math.pi, 0.0),
            ],
            dtype=np.float64,
        )
    )
    sig = np.ascontiguousarray(np.array([0.0, 0.0], dtype=np.float64))

    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, tol_prf_hz=1000.0, max_nodes=200_000, max_iters=50_000)


def test_invalid_inputs_raise(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry

    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, tol_prf_hz=0.0)
    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, k_sigma=-1.0)
    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, margin_mode=7)
    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, strict_pri_pad_s=-1e-12)
    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, max_nodes=-1)
    with pytest.raises(ValueError):
        max_unambiguous_prf(obs, sig, auto_expand_max_nodes=True, max_nodes_growth=1.0)


def test_auto_expand_max_nodes_recovers(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry

    out = max_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=12.0,
        k_sigma=1.0,
        max_nodes=100_000,
        max_iters=200_000,
        auto_expand_max_nodes=True,
        max_nodes_growth=2.0,
        max_nodes_hard_cap=2_000_000,
    )
    assert np.isfinite(out).all()


def test_disable_auto_expand_can_raise_max_nodes(
    compile_numba_once, baseline_geometry
) -> None:
    obs, sig = baseline_geometry

    with pytest.raises(ValueError, match="MAX_NODES|max_nodes"):
        max_unambiguous_prf(
            obs,
            sig,
            tol_prf_hz=12.0,
            k_sigma=1.0,
            max_nodes=100_000,
            max_iters=200_000,
            auto_expand_max_nodes=False,
        )


def test_batched_matches_scalar_const_sigma(compile_numba_once) -> None:
    h = 550e3
    obs0 = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs1 = np.array(
        [
            geodetic2ecef_wgs84(0.1 * math.pi / 180.0, 0.2 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.1 * math.pi / 180.0, 1.2 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.1 * math.pi / 180.0, -0.8 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs2 = np.array(
        [
            geodetic2ecef_wgs84(-0.1 * math.pi / 180.0, 0.3 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(-0.1 * math.pi / 180.0, 1.3 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(-0.1 * math.pi / 180.0, -0.7 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs_series = np.ascontiguousarray(np.stack([obs0, obs1, obs2], axis=0))
    sig = np.ascontiguousarray(np.array([200e-9, 120e-9, 90e-9], dtype=np.float64))

    out_b = max_unambiguous_prf_batched_const_sigma(
        obs_series,
        sig,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    for t in range(obs_series.shape[0]):
        out_s = max_unambiguous_prf(
            obs_series[t],
            sig,
            tol_prf_hz=12.0,
            k_sigma=3.0,
            max_nodes=1_000_000,
            max_iters=200_000,
        )
        for i in range(6):
            assert abs(out_b[i][t] - out_s[i]) < 1e-12


def test_batched_matches_scalar_time_varying_sigma(compile_numba_once) -> None:
    h = 550e3
    obs0 = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs1 = np.array(
        [
            geodetic2ecef_wgs84(0.05 * math.pi / 180.0, 0.15 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.05 * math.pi / 180.0, 1.15 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.05 * math.pi / 180.0, -0.85 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs_series = np.ascontiguousarray(np.stack([obs0, obs1], axis=0))
    sig_series = np.ascontiguousarray(
        np.array(
            [
                [200e-9, 120e-9, 90e-9],
                [210e-9, 110e-9, 95e-9],
            ],
            dtype=np.float64,
        )
    )

    out_b = max_unambiguous_prf_batched(
        obs_series,
        sig_series,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
    )

    for t in range(obs_series.shape[0]):
        out_s = max_unambiguous_prf(
            obs_series[t],
            sig_series[t],
            tol_prf_hz=12.0,
            k_sigma=3.0,
            max_nodes=1_000_000,
            max_iters=200_000,
        )
        for i in range(6):
            assert abs(out_b[i][t] - out_s[i]) < 1e-12


def test_batched_temporal_warm_start_matches_disabled_const_sigma(
    compile_numba_once,
) -> None:
    h = 550e3
    obs0 = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs1 = np.array(
        [
            geodetic2ecef_wgs84(0.02 * math.pi / 180.0, 0.05 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.02 * math.pi / 180.0, 1.05 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.02 * math.pi / 180.0, -0.95 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs2 = np.array(
        [
            geodetic2ecef_wgs84(0.04 * math.pi / 180.0, 0.10 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.04 * math.pi / 180.0, 1.10 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.04 * math.pi / 180.0, -0.90 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs_series = np.ascontiguousarray(np.stack([obs0, obs1, obs2], axis=0))
    sig = np.ascontiguousarray(np.array([200e-9, 120e-9, 90e-9], dtype=np.float64))

    out_warm = max_unambiguous_prf_batched_const_sigma(
        obs_series,
        sig,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
        temporal_warm_start=True,
    )
    out_cold = max_unambiguous_prf_batched_const_sigma(
        obs_series,
        sig,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
        temporal_warm_start=False,
    )
    for i in range(6):
        assert np.max(np.abs(out_warm[i] - out_cold[i])) < 1e-12


def test_batched_temporal_warm_start_matches_disabled_varying_sigma(
    compile_numba_once,
) -> None:
    h = 550e3
    obs0 = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs1 = np.array(
        [
            geodetic2ecef_wgs84(0.03 * math.pi / 180.0, 0.07 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.03 * math.pi / 180.0, 1.07 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.03 * math.pi / 180.0, -0.93 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    obs_series = np.ascontiguousarray(np.stack([obs0, obs1], axis=0))
    sig_series = np.ascontiguousarray(
        np.array(
            [
                [200e-9, 120e-9, 90e-9],
                [210e-9, 110e-9, 95e-9],
            ],
            dtype=np.float64,
        )
    )

    out_warm = max_unambiguous_prf_batched(
        obs_series,
        sig_series,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
        temporal_warm_start=True,
    )
    out_cold = max_unambiguous_prf_batched(
        obs_series,
        sig_series,
        tol_prf_hz=12.0,
        k_sigma=3.0,
        max_nodes=1_000_000,
        max_iters=200_000,
        temporal_warm_start=False,
    )
    for i in range(6):
        assert np.max(np.abs(out_warm[i] - out_cold[i])) < 1e-12
