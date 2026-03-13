from __future__ import annotations

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time

from nebula.propagation.orbit import (
    Orbit,
    _absdate_to_astropy_utc,
    _astropy_to_absdate_utc,
    _dt_seconds_from_epoch,
    _hermite_pv_uniform_twosided,
)
from tests.helpers.orekit_ephemeris import (
    direct_pv_from_propagator,
    make_time_grid,
    stats,
)


def _maybe_count_step_handlers(propagator) -> int | None:
    try:
        mux = propagator.getMultiplexer()
        try:
            hs = mux.getHandlers()
            return int(hs.size())
        except Exception:
            hs = mux.getStepHandlers()
            return int(hs.size())
    except Exception:
        try:
            hs = propagator.getStepHandlers()
            return int(hs.size())
        except Exception:
            return None


def _interp_native_to_itrf(ephem_obj: Orbit, t: Time) -> tuple[np.ndarray, np.ndarray]:
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.utils import PVCoordinates  # type: ignore

    rN, vN = ephem_obj.pv(t, frame="native")
    abs_t = _astropy_to_absdate_utc(t)
    tr = ephem_obj._frame_native.getTransformTo(ephem_obj._itrf, abs_t)  # type: ignore[attr-defined]
    pvN = PVCoordinates(Vector3D(*rN.tolist()), Vector3D(*vN.tolist()))
    pvI = tr.transformPVCoordinates(pvN)
    rI = np.asarray(pvI.getPosition().toArray(), dtype=np.float64)
    vI = np.asarray(pvI.getVelocity().toArray(), dtype=np.float64)
    return rI, vI


def test_time_roundtrip_nonleap_migrated() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    ad = _astropy_to_absdate_utc(t0)
    t1 = _absdate_to_astropy_utc(ad)
    dt = float((t1.utc - t0.utc).to_value("s"))
    assert abs(dt) < 1e-6


def test_time_roundtrip_leapsecond_if_available_migrated() -> None:
    try:
        t0 = Time("2016-12-31T23:59:60.0", scale="utc")
    except Exception:
        pytest.skip("leap-second parsing unavailable in this environment")
    ad = _astropy_to_absdate_utc(t0)
    t1 = _absdate_to_astropy_utc(ad)
    dt = float((t1.utc - t0.utc).to_value("s"))
    assert abs(dt) < 1e-6


def test_hermite_exact_at_knots_migrated() -> None:
    dt = 10.0
    k_min = -2
    ks = np.arange(k_min, k_min + 6, dtype=np.int64)

    a = np.array([1.0, -2.0, 0.5])
    b = np.array([0.1, 0.2, -0.1])
    c = np.array([1e-3, -2e-3, 3e-3])
    d = np.array([5e-6, 1e-6, -3e-6])

    def r_of(t):
        return a + b * t + c * t * t + d * t * t * t

    def v_of(t):
        return b + 2 * c * t + 3 * d * t * t

    ts = ks.astype(np.float64) * dt
    r_samples = np.stack([r_of(t) for t in ts], axis=0)
    v_samples = np.stack([v_of(t) for t in ts], axis=0)

    r_q, v_q = _hermite_pv_uniform_twosided(ts, k_min, dt, r_samples, v_samples)
    np.testing.assert_allclose(r_q, r_samples, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(v_q, v_samples, atol=1e-12, rtol=0.0)


def test_pv_at_epoch_works_without_expanding_cache_migrated() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(45.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(20.0),
        anomaly=np.deg2rad(30.0),
        dt_save_s=60.0,
        gravity_model="newtonian",
    )
    cov0 = e.coverage()
    r0, v0 = e.pv(e.epoch, frame="native")
    cov1 = e.coverage()
    np.testing.assert_allclose(cov0, cov1, atol=0.0, rtol=0.0)
    assert r0.shape == (3,)
    assert v0.shape == (3,)


def test_ephemeris_scalar_vector_consistency_migrated() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(45.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(20.0),
        anomaly=np.deg2rad(30.0),
        dt_save_s=60.0,
        gravity_model="newtonian",
    )
    t_vec = make_time_grid(e.epoch, t_min_s=-3600.0, t_max_s=3600.0, n=41)
    rV = e.pos(t_vec, frame="native")
    vV = e.vel(t_vec, frame="native")
    for idx in (0, 1, 10, 20, 40):
        rS = e.pos(t_vec[idx], frame="native")
        vS = e.vel(t_vec[idx], frame="native")
        np.testing.assert_allclose(rS, rV[idx], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(vS, vV[idx], atol=0.0, rtol=0.0)


def test_ephemeris_cache_expands_both_directions_migrated() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=8000e3,
        e=0.05,
        i=np.deg2rad(10.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
        dt_save_s=120.0,
        gravity_model="newtonian",
    )
    np.testing.assert_allclose(e.coverage(), (0.0, 0.0), atol=0.0, rtol=0.0)
    t_query = make_time_grid(e.epoch, t_min_s=-3600.0, t_max_s=7200.0, n=5)
    _ = e.pos(t_query, frame="native")
    cov1 = e.coverage()
    assert cov1[0] <= -3600.0
    assert cov1[1] >= 7200.0


@pytest.mark.slow
def test_ephemeris_matches_direct_propagation_newtonian_migrated() -> None:
    dt_save = 60.0
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(63.4),
        raan=np.deg2rad(40.0),
        argp=np.deg2rad(270.0),
        anomaly=np.deg2rad(5.0),
        dt_save_s=dt_save,
        gravity_model="newtonian",
        interpolation_mode="quintic",
        position_tolerance_m=0.1,
        initial_step_s=30.0,
    )
    t_grid = make_time_grid(e.epoch, t_min_s=0.5 * dt_save, t_max_s=3 * 3600.0, n=301)
    r_i, v_i = e.pv(t_grid, frame="native")
    r_d = np.empty_like(r_i)
    v_d = np.empty_like(v_i)
    for k in range(len(t_grid)):
        r_d[k], v_d[k] = direct_pv_from_propagator(e, t_grid[k], frame="native")
    s_pos = stats(np.linalg.norm(r_i - r_d, axis=1))
    s_vel = stats(np.linalg.norm(v_i - v_d, axis=1))
    assert s_pos["p99"] < 3.0
    assert s_pos["max"] < 8.0
    assert s_vel["p99"] < 0.02
    assert s_vel["max"] < 0.03


@pytest.mark.slow
def test_dt_save_monotonic_accuracy_migrated() -> None:
    epoch_in = Time("2026-01-01T00:00:00", scale="utc")
    kepler = dict(
        epoch=epoch_in,
        a_m=26600e3,
        e=0.3,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(270.0),
        anomaly=np.deg2rad(0.0),
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
        initial_step_s=60.0,
    )
    e60 = Orbit.from_kepler_precise(dt_save_s=60.0, **kepler)
    e10 = Orbit.from_kepler_precise(dt_save_s=10.0, **kepler)
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 6 * 3600.0, 400, dtype=np.float64)
    jitter = rng.uniform(0.1, 0.9, size=base.shape)
    t60 = e60.epoch + (base + jitter * 60.0) * u.s
    t10 = e10.epoch + (base + jitter * 10.0) * u.s

    def compute_err(eph: Orbit, t: Time) -> np.ndarray:
        r_i = eph.pos(t, frame="native")
        r_d = np.empty_like(r_i)
        for k in range(len(t)):
            r_d[k], _ = direct_pv_from_propagator(eph, t[k], frame="native")
        return np.linalg.norm(r_i - r_d, axis=1)

    s60 = stats(compute_err(e60, t60))
    s10 = stats(compute_err(e10, t10))
    assert s10["mean"] <= 1.05 * s60["mean"]
    assert s10["p90"] <= 1.05 * s60["p90"]


def test_transform_consistency_at_cached_knots_migrated() -> None:
    dt_save = 120.0
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(98.0),
        raan=np.deg2rad(15.0),
        argp=np.deg2rad(45.0),
        anomaly=np.deg2rad(0.0),
        dt_save_s=dt_save,
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
    )
    _ = e.pos(
        make_time_grid(e.epoch, t_min_s=0.0, t_max_s=10 * dt_save, n=11), frame="native"
    )
    ks = np.arange(0, 11, dtype=np.int64)
    t_knots = e.epoch + (ks.astype(np.float64) * dt_save) * u.s
    rN_i, vN_i = e.pv(t_knots, frame="native")
    rI_i, vI_i = e.pv(t_knots, frame="itrf")
    if getattr(e, "_java_engine", False):
        # Java engine manages its own internal cache; verify knot consistency
        # against scalar queries instead of Python-side cache arrays.
        for idx, tk in enumerate(t_knots):
            r_s, v_s = e.pv(tk, frame="native")
            np.testing.assert_allclose(rN_i[idx], r_s, atol=1e-7, rtol=0.0)
            np.testing.assert_allclose(vN_i[idx], v_s, atol=1e-7, rtol=0.0)
    else:
        k0 = e._k_min
        np.testing.assert_allclose(rN_i, e._r_native[ks - k0], atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(vN_i, e._v_native[ks - k0], atol=1e-7, rtol=0.0)
    if getattr(e, "_cache_itrf_samples", True):
        if getattr(e, "_java_engine", False):
            for idx, tk in enumerate(t_knots):
                r_s, v_s = e.pv(tk, frame="itrf")
                np.testing.assert_allclose(rI_i[idx], r_s, atol=1e-7, rtol=0.0)
                np.testing.assert_allclose(vI_i[idx], v_s, atol=1e-7, rtol=0.0)
        else:
            k0 = e._k_min
            np.testing.assert_allclose(rI_i, e._r_itrf[ks - k0], atol=1e-7, rtol=0.0)
            np.testing.assert_allclose(vI_i, e._v_itrf[ks - k0], atol=1e-7, rtol=0.0)
    else:
        for idx, t in enumerate(t_knots):
            r_ref, v_ref = _interp_native_to_itrf(e, t)
            np.testing.assert_allclose(rI_i[idx], r_ref, atol=1e-7, rtol=0.0)
            np.testing.assert_allclose(vI_i[idx], v_ref, atol=1e-7, rtol=0.0)


def test_from_pv_reproduces_initial_state_migrated() -> None:
    ref = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=8000e3,
        e=0.02,
        i=np.deg2rad(20.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(40.0),
        anomaly=np.deg2rad(5.0),
        dt_save_s=60.0,
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
    )
    r0N, v0N = direct_pv_from_propagator(ref, ref.epoch, frame="native")
    e2 = Orbit.from_pv(
        r0N,
        v0N,
        ref.epoch,
        frame="gcrf",
        propagate_inertial_frame="gcrf",
        dt_save_s=60.0,
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
    )
    r_chk, v_chk = e2.pv(e2.epoch, frame="native")
    np.testing.assert_allclose(r_chk, r0N, atol=1e-4, rtol=0.0)
    np.testing.assert_allclose(v_chk, v0N, atol=1e-7, rtol=0.0)


def test_lla_ranges_and_shapes_migrated() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(45.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
        dt_save_s=60.0,
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
    )
    t_vec = make_time_grid(e.epoch, t_min_s=0.0, t_max_s=3600.0, n=101)
    lat, lon, alt = e.lla(t_vec)
    assert lat.shape == (101,)
    assert lon.shape == (101,)
    assert alt.shape == (101,)
    assert np.isfinite(lat).all() and np.isfinite(lon).all() and np.isfinite(alt).all()
    assert np.all(lat >= -90.0) and np.all(lat <= 90.0)
    assert np.all(lon >= -180.0) and np.all(lon <= 180.0)
    latS, lonS, altS = e.lla(e.epoch + 123.0 * u.s)
    assert np.isscalar(latS) or getattr(latS, "shape", ()) == ()
    assert np.isscalar(lonS) or getattr(lonS, "shape", ()) == ()
    assert np.isscalar(altS) or getattr(altS, "shape", ()) == ()


def test_nan_inf_inputs_rejected_migrated() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(45.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
        dt_save_s=60.0,
        gravity_model="newtonian",
    )
    with pytest.raises(ValueError):
        e._ensure_covered(np.array([np.nan], dtype=np.float64))  # type: ignore[attr-defined]


@pytest.mark.slow
def test_no_step_handler_growth_with_many_extensions_migrated() -> None:
    dt_save = 60.0
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(98.0),
        raan=np.deg2rad(15.0),
        argp=np.deg2rad(45.0),
        anomaly=np.deg2rad(0.0),
        dt_save_s=dt_save,
        gravity_model="harmonic",
        gravity_degree=20,
        gravity_order=20,
        position_tolerance_m=0.1,
    )
    c0 = _maybe_count_step_handlers(e.propagator)
    if c0 is None:
        pytest.skip("step-handler introspection unavailable")
    for k in range(1, 60):
        _ = e.pos(e.epoch + (k * dt_save) * u.s, frame="native")
        _ = e.pos(e.epoch - (k * dt_save) * u.s, frame="native")
    c1 = _maybe_count_step_handlers(e.propagator)
    if c1 is None:
        pytest.skip("step-handler introspection unavailable")
    assert c1 <= c0 + 2


@pytest.mark.slow
def test_ephemeris_matches_direct_propagation_backward_migrated() -> None:
    dt_save = 60.0
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(63.4),
        raan=np.deg2rad(40.0),
        argp=np.deg2rad(270.0),
        anomaly=np.deg2rad(5.0),
        dt_save_s=dt_save,
        gravity_model="newtonian",
        interpolation_mode="quintic",
        position_tolerance_m=0.1,
        initial_step_s=30.0,
    )
    t_grid = make_time_grid(e.epoch, t_min_s=-3 * 3600.0, t_max_s=-0.5 * dt_save, n=301)
    r_i, v_i = e.pv(t_grid, frame="native")
    r_d = np.empty_like(r_i)
    v_d = np.empty_like(v_i)
    for k in range(len(t_grid)):
        r_d[k], v_d[k] = direct_pv_from_propagator(e, t_grid[k], frame="native")
    s_pos = stats(np.linalg.norm(r_i - r_d, axis=1))
    s_vel = stats(np.linalg.norm(v_i - v_d, axis=1))
    assert s_pos["p99"] < 3.0
    assert s_pos["max"] < 8.0
    assert s_vel["p99"] < 0.02
    assert s_vel["max"] < 0.03


@pytest.mark.slow
def test_force_model_matrix_matches_direct_propagation_migrated() -> None:
    epoch_in = Time("2026-01-01T00:00:00", scale="utc")
    configs = [
        dict(
            name="harmonic_only",
            kwargs=dict(
                gravity_model="harmonic",
                gravity_degree=20,
                gravity_order=20,
                enable_third_body=False,
                enable_drag=False,
                enable_srp=False,
                enable_relativity=False,
            ),
        ),
        dict(
            name="harmonic_plus_thirdbody",
            kwargs=dict(
                gravity_model="harmonic",
                gravity_degree=20,
                gravity_order=20,
                enable_third_body=True,
                third_bodies=("sun", "moon"),
                enable_drag=False,
                enable_srp=False,
            ),
        ),
        dict(
            name="harmonic_plus_srp",
            kwargs=dict(
                gravity_model="harmonic",
                gravity_degree=20,
                gravity_order=20,
                enable_srp=True,
                enable_third_body=False,
                enable_drag=False,
            ),
        ),
        dict(
            name="harmonic_plus_drag",
            kwargs=dict(
                gravity_model="harmonic",
                gravity_degree=20,
                gravity_order=20,
                enable_drag=True,
                solar_activity_strength="average",
                enable_third_body=False,
                enable_srp=False,
            ),
        ),
    ]
    base_kwargs = dict(
        epoch=epoch_in,
        a_m=7200e3,
        e=0.02,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(80.0),
        anomaly=np.deg2rad(5.0),
        dt_save_s=15.0,
        position_tolerance_m=0.1,
        initial_step_s=30.0,
        min_step_s=0.001,
        max_step_s=300.0,
    )
    rng = np.random.default_rng(123)
    t_offsets = np.sort(rng.uniform(5.0, 2 * 3600.0 - 5.0, size=61))
    for cfg in configs:
        try:
            e = Orbit.from_kepler_precise(**base_kwargs, **cfg["kwargs"])
        except Exception:
            pytest.skip("force-model configuration unavailable with current data setup")
        t_grid = e.epoch + t_offsets * u.s
        r_i, v_i = e.pv(t_grid, frame="native")
        r_d = np.empty_like(r_i)
        v_d = np.empty_like(v_i)
        for k in range(len(t_grid)):
            r_d[k], v_d[k] = direct_pv_from_propagator(e, t_grid[k], frame="native")
        s_pos = stats(np.linalg.norm(r_i - r_d, axis=1))
        s_vel = stats(np.linalg.norm(v_i - v_d, axis=1))
        assert s_pos["p99"] < 10.0
        assert s_pos["max"] < 25.0
        assert s_vel["p99"] < 0.08
        assert s_vel["max"] < 0.12


@pytest.mark.slow
def test_frame_consistency_transform_mode_matches_native_transform_migrated() -> None:
    epoch_in = Time("2026-01-01T00:00:00", scale="utc")

    def build(dt_save: float) -> Orbit:
        return Orbit.from_kepler_precise(
            epoch=epoch_in,
            a_m=7000e3,
            e=0.01,
            i=np.deg2rad(98.0),
            raan=np.deg2rad(15.0),
            argp=np.deg2rad(45.0),
            anomaly=np.deg2rad(0.0),
            dt_save_s=dt_save,
            itrf_query_mode="transform",
            gravity_model="harmonic",
            gravity_degree=20,
            gravity_order=20,
            position_tolerance_m=0.1,
        )

    e60 = build(60.0)
    e10 = build(10.0)

    def mismatch_stats(e: Orbit, dt_save: float) -> tuple[float, float]:
        ts = (
            e.epoch
            + (np.arange(1, 181, dtype=np.float64) * dt_save + 0.37 * dt_save) * u.s
        )
        rI_a, vI_a = e.pv(ts, frame="itrf")
        rI_b = np.empty_like(rI_a)
        vI_b = np.empty_like(vI_a)
        for k in range(len(ts)):
            rI_b[k], vI_b[k] = _interp_native_to_itrf(e, ts[k])
        dr = np.linalg.norm(rI_a - rI_b, axis=1)
        dv = np.linalg.norm(vI_a - vI_b, axis=1)
        return float(np.mean(dr)), float(np.mean(dv))

    r60, v60 = mismatch_stats(e60, 60.0)
    r10, v10 = mismatch_stats(e10, 10.0)
    assert r60 < 1e-6
    assert v60 < 1e-9
    assert r10 < 1e-6
    assert v10 < 1e-9


def test_leap_second_crossing_monotonic_dt_and_pv_migrated() -> None:
    try:
        _ = Time("2016-12-31T23:59:60.0", scale="utc")
    except Exception:
        pytest.skip("leap-second parsing unavailable in this environment")

    epoch_in = Time("2016-12-31T23:59:30.0", scale="utc")
    ts = epoch_in + np.arange(0, 90, dtype=np.float64) * u.s
    dt_s, is_scalar = _dt_seconds_from_epoch(ts, epoch_in)
    assert not is_scalar
    np.testing.assert_allclose(
        dt_s, np.arange(0, 90, dtype=np.float64), atol=1e-9, rtol=0.0
    )
    assert np.all(np.diff(dt_s) > 0)

    e = Orbit.from_kepler_precise(
        epoch=epoch_in,
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(63.4),
        raan=np.deg2rad(40.0),
        argp=np.deg2rad(270.0),
        anomaly=np.deg2rad(5.0),
        dt_save_s=10.0,
        gravity_model="newtonian",
        position_tolerance_m=0.1,
        initial_step_s=10.0,
    )
    for i in [0, 10, 20, 29, 30, 31, 40, 60, 80]:
        r_i, v_i = e.pv(ts[i], frame="native")
        r_d, v_d = direct_pv_from_propagator(e, ts[i], frame="native")
        np.testing.assert_allclose(r_i, r_d, atol=5.0, rtol=0.0)
        np.testing.assert_allclose(v_i, v_d, atol=0.05, rtol=0.0)
