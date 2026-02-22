from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.time import Time

from nebula.propagation import Orbit
from nebula.propagation._fast_orbit_backend import FastOrbit


def _kepler_case() -> dict[str, float | Time]:
    return dict(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )


def test_orbit_precision_constructor_mode_and_state() -> None:
    e = Orbit.from_kepler_precise(
        **_kepler_case(),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    assert e.mode == "precision"
    assert e.is_precision
    assert not e.is_efficiency

    r0, v0 = e.pv(0.0, frame="native")
    r1, v1 = e.pv(e.epoch, frame="native")
    np.testing.assert_allclose(r0, r1, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(v0, v1, atol=1e-9, rtol=0.0)


def test_orbit_precision_mode_accepts_seconds_inputs() -> None:
    e = Orbit.from_kepler_precise(
        **_kepler_case(),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    r_s, v_s = e.pv(123.0, frame="native")
    r_t, v_t = e.pv(e.epoch + 123.0 * u.s, frame="native")
    np.testing.assert_allclose(r_s, r_t, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(v_s, v_t, atol=1e-9, rtol=0.0)

    dt_arr = np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float64)
    r_a, v_a = e.pv(dt_arr, frame="itrf")
    r_b, v_b = e.pv(e.epoch + dt_arr * u.s, frame="itrf")
    np.testing.assert_allclose(r_a, r_b, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(v_a, v_b, atol=1e-9, rtol=0.0)

    scalar_np = np.float64(45.0)
    r_np, v_np = e.pv(scalar_np, frame="native")
    r_ref, v_ref = e.pv(45.0, frame="native")
    np.testing.assert_allclose(r_np, r_ref, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(v_np, v_ref, atol=1e-12, rtol=0.0)


def test_orbit_fast_mode_matches_fastorbit_backend_behavior() -> None:
    kwargs = dict(
        **_kepler_case(),
        dt_save_s=20.0,
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=3,
    )
    e = Orbit.from_kepler_fast(**kwargs)
    f = FastOrbit.from_kepler(**kwargs)

    assert e.mode == "efficiency"
    assert e.is_efficiency
    assert not e.is_precision

    e.precompute(-600.0, 1200.0)
    f.precompute(-600.0, 1200.0)
    c0 = e.coverage()
    c1 = f.coverage()
    assert c0[0] <= -600.0 and c0[1] >= 1200.0
    np.testing.assert_allclose(c0, c1, atol=0.0, rtol=0.0)

    ts = np.array([-120.0, 0.0, 45.0, 300.0], dtype=np.float64)
    r_en, v_en = e.pv(ts, frame="native")
    r_fn, v_fn = f.pv(ts, frame="native")
    np.testing.assert_allclose(r_en, r_fn, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v_en, v_fn, atol=0.0, rtol=0.0)

    r_ei, v_ei = e.pv(ts, frame="itrf")
    r_fi, v_fi = f.pv(ts, frame="itrf")
    np.testing.assert_allclose(r_ei, r_fi, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v_ei, v_fi, atol=0.0, rtol=0.0)

    t_ast = e.epoch + ts * u.s
    r_e_ast, v_e_ast = e.pv(t_ast, frame="native")
    r_f_ast, v_f_ast = f.pv(t_ast, frame="native")
    np.testing.assert_allclose(r_e_ast, r_f_ast, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v_e_ast, v_f_ast, atol=0.0, rtol=0.0)

    lat_e, lon_e, alt_e = e.lla(ts)
    lat_f, lon_f, alt_f = f.lla(ts)
    np.testing.assert_allclose(lat_e, lat_f, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lon_e, lon_f, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(alt_e, alt_f, atol=0.0, rtol=0.0)


def test_orbit_from_pv_is_consistent() -> None:
    ref = Orbit.from_kepler_precise(
        **_kepler_case(),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )
    r0, v0 = ref.pv(ref.epoch, frame="native")

    a = Orbit.from_pv(
        r0,
        v0,
        ref.epoch,
        frame="gcrf",
        propagate_inertial_frame="gcrf",
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    ra, va = a.pv(a.epoch, frame="native")
    np.testing.assert_allclose(r0, ra, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(v0, va, atol=1e-9, rtol=0.0)


def test_orbit_from_spacecraft_state_is_consistent() -> None:
    ref = Orbit.from_kepler_precise(
        **_kepler_case(),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )
    state = ref._state0  # type: ignore[attr-defined]

    a = Orbit.from_spacecraft_state(
        state,
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    ra, va = a.pv(a.epoch, frame="native")
    rr, vr = ref.pv(ref.epoch, frame="native")
    np.testing.assert_allclose(ra, rr, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(va, vr, atol=1e-9, rtol=0.0)
