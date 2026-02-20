from __future__ import annotations

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from nebula.propagation.fast_orbit import (
    EARTH_MU,
    FastOrbit,
    WGS84_A,
    _ecef2geodetic_deg,
    _propagate_batch,
    propagate_constellation_pv,
)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


@pytest.fixture
def fast_two_body() -> FastOrbit:
    ep = FastOrbit.from_kepler(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=98.0,
        raan=10.0,
        argp=20.0,
        anomaly=30.0,
        anomaly_type="true",
        degrees=True,
        dt_save_s=60.0,
        enable_j2=False,
    )
    ep.precompute(0.0, 86400.0)
    return ep


def test_fast_geodetic_equator_sanity() -> None:
    la, lo, h = _ecef2geodetic_deg(WGS84_A, 0.0, 0.0)
    assert abs(la) < 1e-8
    assert abs(lo) < 1e-8
    assert abs(h) < 1e-3


def test_fast_precompute_cache_invariants(fast_two_body: FastOrbit) -> None:
    cov0, cov1 = fast_two_body.coverage()
    assert cov0 <= 0.0 + 1e-12
    assert cov1 >= 86400.0 - 1e-12
    assert fast_two_body._n == (fast_two_body._k_max - fast_two_body._k_min + 1)
    assert fast_two_body._cap >= fast_two_body._n


def test_fast_scalar_vector_time_inputs(fast_two_body: FastOrbit) -> None:
    r0, v0 = fast_two_body.pv(0.0, frame="native")
    assert r0.shape == (3,)
    assert v0.shape == (3,)
    assert np.isfinite(r0).all() and np.isfinite(v0).all()

    r0b = fast_two_body.pos(0.0, frame="native")
    v0b = fast_two_body.vel(0.0, frame="native")
    np.testing.assert_allclose(r0b, r0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v0b, v0, atol=0.0, rtol=0.0)

    t_np0 = np.asarray(120.0, dtype=np.float64)
    r_np0, v_np0 = fast_two_body.pv(t_np0)
    if r_np0.shape == (1, 3):
        r_s, v_s = fast_two_body.pv(120.0, frame="native")
        np.testing.assert_allclose(r_np0[0], r_s, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(v_np0[0], v_s, atol=1e-12, rtol=0.0)
    else:
        assert r_np0.shape == (3,)
        assert v_np0.shape == (3,)

    t_np1 = np.array([120.0], dtype=np.float64)
    r_np1, v_np1 = fast_two_body.pv(t_np1)
    assert r_np1.shape == (1, 3)
    assert v_np1.shape == (1, 3)

    t_np = np.array([0.0, 30.0, 60.0, 90.0, 120.0], dtype=np.float64)
    r_np, v_np = fast_two_body.pv(t_np)
    assert r_np.shape == (t_np.size, 3)
    assert v_np.shape == (t_np.size, 3)
    assert np.isfinite(r_np).all() and np.isfinite(v_np).all()

    t_ast = fast_two_body.epoch + np.array([0.0, 30.0, 60.0, 90.0, 120.0]) * u.s
    r_ast, v_ast = fast_two_body.pv(t_ast)
    assert r_ast.shape == (t_np.size, 3)
    assert v_ast.shape == (t_np.size, 3)
    np.testing.assert_allclose(r_ast, r_np, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(v_ast, v_np, atol=1e-9, rtol=0.0)


def test_fast_hermite_exact_at_knots(fast_two_body: FastOrbit) -> None:
    ks = np.array([0, 1, 2, 10, 100, 1000, 1440], dtype=np.int64)
    ks = ks[ks * fast_two_body.dt <= 86400.0 + 1e-12]
    for k in ks:
        t_k = float(k) * fast_two_body.dt
        r_k, v_k = fast_two_body.pv(t_k, frame="native")
        rN, vN, _, _ = _propagate_batch(
            np.array([t_k], np.float64),
            fast_two_body._epoch_ut1_jd,
            fast_two_body._epoch_tt_jd,
            float(fast_two_body.a_m),
            float(fast_two_body.e),
            float(fast_two_body.i_rad),
            float(fast_two_body.raan_rad),
            float(fast_two_body.argp_rad),
            float(fast_two_body.M0_rad),
            float(fast_two_body.mu),
            float(fast_two_body._raan_dot),
            float(fast_two_body._argp_dot),
            float(fast_two_body._M_dot),
            float(fast_two_body._xp_rad),
            float(fast_two_body._yp_rad),
        )
        np.testing.assert_allclose(r_k, rN[0], atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(v_k, vN[0], atol=1e-8, rtol=0.0)


def test_fast_cache_extension_stability(fast_two_body: FastOrbit) -> None:
    r0_before, v0_before = fast_two_body.pv(0.0, frame="native")

    r_neg, v_neg = fast_two_body.pv(-3600.0, frame="native")
    assert np.isfinite(r_neg).all() and np.isfinite(v_neg).all()

    r_far, v_far = fast_two_body.pv(200000.0, frame="native")
    assert np.isfinite(r_far).all() and np.isfinite(v_far).all()

    assert fast_two_body._n == (fast_two_body._k_max - fast_two_body._k_min + 1)
    assert fast_two_body._cap >= fast_two_body._n

    r0_after, v0_after = fast_two_body.pv(0.0, frame="native")
    np.testing.assert_allclose(r0_after, r0_before, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v0_after, v0_before, atol=0.0, rtol=0.0)


def test_fast_itrf_radius_matches_native_at_knots(fast_two_body: FastOrbit) -> None:
    ks = np.array([0, 2, 10, 100, 1000, 1440], dtype=np.int64)
    ks = ks[ks * fast_two_body.dt <= fast_two_body.coverage()[1] + 1e-12]
    for k in ks:
        t_k = float(k) * fast_two_body.dt
        r_n, _ = fast_two_body.pv(t_k, frame="native")
        r_i, _ = fast_two_body.pv(t_k, frame="itrf")
        np.testing.assert_allclose(_norm(r_n), _norm(r_i), atol=1e-8, rtol=0.0)


def test_fast_lla_bounds_and_finite(fast_two_body: FastOrbit) -> None:
    t_ast = fast_two_body.epoch + np.array([0.0, 30.0, 60.0, 90.0, 120.0]) * u.s
    lat, lon, alt = fast_two_body.lla(t_ast)
    assert np.isfinite(lat).all()
    assert np.isfinite(lon).all()
    assert np.isfinite(alt).all()
    assert np.all(lat >= -90.0) and np.all(lat <= 90.0)
    assert np.all(lon >= -180.0) and np.all(lon <= 180.0)


def test_fast_j2_changes_trajectory() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    ep_noj2 = FastOrbit.from_kepler(
        epoch=epoch,
        a_m=7000e3,
        e=0.001,
        i=98.0,
        raan=10.0,
        argp=20.0,
        anomaly=30.0,
        anomaly_type="true",
        degrees=True,
        dt_save_s=60.0,
        enable_j2=False,
    )
    ep_j2 = FastOrbit.from_kepler(
        epoch=epoch,
        a_m=7000e3,
        e=0.001,
        i=98.0,
        raan=10.0,
        argp=20.0,
        anomaly=30.0,
        anomaly_type="true",
        degrees=True,
        dt_save_s=60.0,
        enable_j2=True,
    )
    ep_noj2.precompute(0.0, 86400.0)
    ep_j2.precompute(0.0, 86400.0)

    r_noj2, _ = ep_noj2.pv(86400.0, frame="native")
    r_yesj2, _ = ep_j2.pv(86400.0, frame="native")
    assert _norm(r_noj2 - r_yesj2) > 1e-3


def test_fast_propagate_constellation_shapes_and_finite() -> None:
    dt_grid = np.linspace(0.0, 600.0, 11, dtype=np.float64)
    a = np.array([7000e3, 7100e3], np.float64)
    e = np.array([0.001, 0.01], np.float64)
    inc = np.array([math.radians(98.0), math.radians(55.0)], np.float64)
    raan0 = np.array([math.radians(10.0), math.radians(20.0)], np.float64)
    argp0 = np.array([math.radians(20.0), math.radians(30.0)], np.float64)
    M0 = np.array([math.radians(30.0), math.radians(40.0)], np.float64)

    rdot = np.zeros(2, np.float64)
    wdot = np.zeros(2, np.float64)
    Mdot = np.sqrt(EARTH_MU / (a * a * a)).copy()

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    try:
        epoch_ut1_jd = float(epoch.ut1.jd)
    except Exception:
        epoch_ut1_jd = float(epoch.utc.jd)
    epoch_tt_jd = float(epoch.tt.jd)

    rC, vC, rCe, vCe = propagate_constellation_pv(
        dt_grid,
        epoch_ut1_jd,
        epoch_tt_jd,
        a,
        e,
        inc,
        raan0,
        argp0,
        M0,
        EARTH_MU,
        rdot,
        wdot,
        Mdot,
    )
    assert rC.shape == (2, dt_grid.size, 3)
    assert vC.shape == (2, dt_grid.size, 3)
    assert rCe.shape == (2, dt_grid.size, 3)
    assert vCe.shape == (2, dt_grid.size, 3)
    assert np.isfinite(rC).all() and np.isfinite(vC).all()
    assert np.isfinite(rCe).all() and np.isfinite(vCe).all()
