from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from nebula.localization.max_unambiguous_prf import max_unambiguous_prf


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
    # For numeric tolerance, allow tiny negative slack.
    vals = ps @ x_ecef
    return bool(np.all(vals >= 1.0 - 1e-14))


def delay_spread_W_seconds(x_ecef: np.ndarray, obs_ecef: np.ndarray) -> float:
    """W(x) = max tau - min tau in seconds."""
    d = np.linalg.norm(obs_ecef - x_ecef[None, :], axis=1)
    tau = d / _C0
    return float(np.max(tau) - np.min(tau))


def worst_case_margin_seconds(sigmas_1sigma_s: np.ndarray, k_sigma: float) -> float:
    """Independent 1σ per observer => worst-case baseline σ = sqrt(s1^2+s2^2)."""
    s = np.sort(sigmas_1sigma_s.astype(np.float64))
    s1 = float(s[-1])
    s2 = float(s[-2]) if s.size >= 2 else 0.0
    sigma_tdoa = math.sqrt(s1 * s1 + s2 * s2)
    return float(k_sigma * sigma_tdoa)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def compile_numba_once() -> None:
    # Trigger compilation cheaply (max_iters=1 means it will likely raise quickly).
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
    # 3 LEO observers near each other to ensure common-visible region exists.
    h = 550e3
    obs = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    sig = np.array([200e-9, 200e-9, 200e-9], dtype=np.float64)  # 1σ seconds
    return np.ascontiguousarray(obs), np.ascontiguousarray(sig)


# -----------------------------
# Tests
# -----------------------------
def test_returns_are_self_consistent(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry

    tol_us = 20.0
    k_sigma = 3.0
    prf_hz, pri_u_us, pri_l_us, gap_us = max_unambiguous_prf(
        obs, sig, tol_us, k_sigma, 1_000_000, 200_000
    )

    assert np.isfinite([prf_hz, pri_u_us, pri_l_us, gap_us]).all()
    assert prf_hz > 0.0
    assert pri_u_us >= pri_l_us
    assert gap_us <= tol_us + 1e-6  # allow tiny numeric slack


def test_permutation_invariance(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_us = 20.0
    k_sigma = 3.0

    out1 = max_unambiguous_prf(obs, sig, tol_us, k_sigma, 1_000_000, 200_000)

    perm = np.array([2, 0, 1], dtype=np.int64)
    obs2 = np.ascontiguousarray(obs[perm])
    sig2 = np.ascontiguousarray(sig[perm])
    out2 = max_unambiguous_prf(obs2, sig2, tol_us, k_sigma, 1_000_000, 200_000)

    # Should be identical to within tiny numerical differences.
    assert abs(out1[1] - out2[1]) < 1e-6  # pri_upper_us
    assert abs(out1[2] - out2[2]) < 1e-6  # pri_lower_us
    assert abs(out1[0] - out2[0]) < 1e-9  # prf_safe_hz


def test_noise_shifts_pri_by_expected_constant(
    compile_numba_once, baseline_geometry
) -> None:
    obs, sig = baseline_geometry
    tol_us = 50.0
    k_sigma = 3.0

    # With noise
    prf1, pri_u1, pri_l1, gap1 = max_unambiguous_prf(
        obs, sig, tol_us, k_sigma, 1_000_000, 200_000
    )

    # Zero noise
    sig0 = np.zeros_like(sig)
    prf0, pri_u0, pri_l0, gap0 = max_unambiguous_prf(
        obs, sig0, tol_us, k_sigma, 1_000_000, 200_000
    )

    # Expected delta PRI = 2*(margin1 - margin0)
    m1 = worst_case_margin_seconds(sig, k_sigma)
    m0 = worst_case_margin_seconds(sig0, k_sigma)
    expected_delta_us = 2.0 * (m1 - m0) * 1e6

    # Upper bounds each have up to tol_us looseness; allow combined slack.
    slack_us = tol_us + tol_us + 1e-3
    assert abs((pri_u1 - pri_u0) - expected_delta_us) <= slack_us
    assert abs((pri_l1 - pri_l0) - expected_delta_us) <= slack_us

    # PRF should decrease when adding noise (since PRI increases).
    assert prf1 <= prf0


def test_k_sigma_monotonicity(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_us = 50.0

    prf1, pri_u1, *_ = max_unambiguous_prf(obs, sig, tol_us, 1.0, 1_000_000, 200_000)
    prf3, pri_u3, *_ = max_unambiguous_prf(obs, sig, tol_us, 3.0, 1_000_000, 200_000)

    assert pri_u3 >= pri_u1
    assert prf3 <= prf1


def test_safe_for_random_visible_points(compile_numba_once, baseline_geometry) -> None:
    obs, sig = baseline_geometry
    tol_us = 20.0
    k_sigma = 3.0

    prf_hz, pri_u_us, pri_l_us, gap_us = max_unambiguous_prf(
        obs, sig, tol_us, k_sigma, 1_000_000, 200_000
    )
    pri_u_s = pri_u_us * 1e-6
    margin = worst_case_margin_seconds(sig, k_sigma)

    # Random surface points; check the no-wrap inequality on those that are visible to all.
    u = random_unit_vectors(6000, seed=1)
    xs = np.stack([ellipsoid_surface_from_unit_dir(ui) for ui in u], axis=0)

    vis_mask = np.array([visible_to_all(x, obs) for x in xs], dtype=bool)
    # Ensure we actually sampled some common-visible points
    assert int(vis_mask.sum()) > 50

    # For visible points, the solver's PRI_upper should satisfy:
    #   W(x) + margin <= PRI_upper/2   (allow tiny numeric slack)
    half_pri = 0.5 * pri_u_s
    slack = 5e-12  # seconds
    for x in xs[vis_mask][:500]:  # cap work
        W = delay_spread_W_seconds(x, obs)
        assert (W + margin) <= (half_pri + slack)


def test_empty_overlap_raises(compile_numba_once) -> None:
    # Two antipodal observers on the WGS84 surface: no surface point is visible to both.
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
        # Loose tolerances and small caps to keep this test quick.
        max_unambiguous_prf(obs, sig, 1000.0, 1.0, 200_000, 50_000)
