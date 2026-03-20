from __future__ import annotations

import math

import numpy as np

from nstk.localization.network_unambiguous_prf import is_pri_network_unambiguous
from nstk.localization.network_unambiguous_prf import network_unambiguous_prf


def geodetic2ecef_wgs84(lat_rad: float, lon_rad: float, h_m: float) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257223563
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


def _baseline_geometry():
    h = 550e3
    obs = np.array(
        [
            geodetic2ecef_wgs84(0.0, 0.0, h),
            geodetic2ecef_wgs84(0.0, 1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, -1.0 * math.pi / 180.0, h),
            geodetic2ecef_wgs84(0.0, 2.0 * math.pi / 180.0, h),
        ],
        dtype=np.float64,
    )
    sig = np.array([200e-9, 200e-9, 200e-9, 200e-9], dtype=np.float64)
    return np.ascontiguousarray(obs), np.ascontiguousarray(sig)


def test_pri_predicate_baseline_wrap_only():
    obs, sig = _baseline_geometry()

    safe_large = is_pri_network_unambiguous(
        obs,
        sig,
        1e-2,
        k_sigma=3.0,
        max_nodes=150_000,
        max_pair_pops=300_000,
    )
    safe_small = is_pri_network_unambiguous(
        obs,
        sig,
        1e-8,
        k_sigma=3.0,
        max_nodes=150_000,
        max_pair_pops=300_000,
    )

    assert safe_large is True
    assert safe_small is False


def test_network_critical_prf_interval_self_consistent():
    # Geometry where wrap-induced ambiguity is not found down to PRI floor.
    # This exercises the public bisection API without relying on exact N=2 paths.
    a = 6378137.0
    h = 550e3
    r = a + h
    obs = np.array([[r, 0.0, 0.0], [0.0, r, 0.0], [0.0, 0.0, r]], dtype=np.float64)
    sig = np.array([100e-9, 100e-9, 100e-9], dtype=np.float64)

    out = network_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=20.0,
        k_sigma=1.0,
        pri_low_s=1e-8,
        pri_high_s=1e-3,
        max_bisect_iters=16,
        max_nodes=120_000,
        max_pair_pops=300_000,
        max_heap_pairs=600_000,
    )
    prf_lo, prf_hi, prf_gap, pri_u_us, pri_l_us, pri_gap_us = out

    assert np.isinf(prf_lo)
    assert np.isinf(prf_hi)
    assert prf_gap == 0.0
    assert pri_u_us == 0.0
    assert pri_l_us == 0.0
    assert pri_gap_us == 0.0


def test_intrinsic_ambiguity_allowed_wrap_only_solver():
    a = 6378137.0
    h = 550e3
    r = a + h
    obs = np.array(
        [
            [r, 0.0, 0.0],
            [r, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    sig = np.array([100e-9, 100e-9], dtype=np.float64)

    safe_large = is_pri_network_unambiguous(
        obs,
        sig,
        1e-2,
        k_sigma=1.0,
        max_nodes=100_000,
        max_pair_pops=200_000,
        max_heap_pairs=400_000,
    )
    assert safe_large is True


def test_inconclusive_raises_instead_of_silent_false():
    obs, sig = _baseline_geometry()
    with np.testing.assert_raises_regex(
        ValueError,
        "wrap predicate inconclusive: increase internal caps \\(REASON=[0-9]+\\)",
    ):
        _ = is_pri_network_unambiguous(
            obs,
            sig,
            1e-6,
            k_sigma=3.0,
            max_nodes=2_000,
            max_pair_pops=4_000,
            max_heap_pairs=8_000,
        )


def test_n2_inconclusive_is_node_cap_not_pop_cap():
    a = 6378137.0
    h = 550e3
    r = a + h
    deg = math.pi / 180.0
    lons = np.array([0.0, 2.0]) * deg
    obs = np.stack(
        [r * np.cos(lons), r * np.sin(lons), np.zeros_like(lons)], axis=1
    ).astype(np.float64)
    sig = np.array([100e-9, 100e-9], dtype=np.float64)

    with np.testing.assert_raises_regex(
        ValueError,
        "wrap predicate inconclusive: increase internal caps \\(REASON=1\\)",
    ):
        _ = is_pri_network_unambiguous(
            obs,
            sig,
            1e-3,
            k_sigma=1.0,
            max_nodes=32,
            max_pair_pops=1,  # should be ignored for N=2 path
            max_heap_pairs=64,
        )


def test_n2_result_not_affected_by_pop_cap():
    a = 6378137.0
    h = 550e3
    r = a + h
    obs = np.array([[r, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
    sig = np.array([100e-9, 100e-9], dtype=np.float64)

    out_lo = network_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=1e9,
        k_sigma=1.0,
        pri_low_s=1e-8,
        pri_high_s=1e-3,
        max_bisect_iters=8,
        max_nodes=100_000,
        max_pair_pops=1,
        max_heap_pairs=4_000,
    )
    out_hi = network_unambiguous_prf(
        obs,
        sig,
        tol_prf_hz=1e9,
        k_sigma=1.0,
        pri_low_s=1e-8,
        pri_high_s=1e-3,
        max_bisect_iters=8,
        max_nodes=100_000,
        max_pair_pops=1_000_000,
        max_heap_pairs=4_000,
    )

    assert np.allclose(np.array(out_lo), np.array(out_hi), rtol=0.0, atol=0.0)
