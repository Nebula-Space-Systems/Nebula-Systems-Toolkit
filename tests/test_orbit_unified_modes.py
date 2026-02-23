from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
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


def _angle_delta_rad(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return np.arctan2(np.sin(aa - bb), np.cos(aa - bb))


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


def test_orbit_keplerian_epoch_matches_input_elements() -> None:
    case = _kepler_case()
    e = Orbit.from_kepler_precise(
        **case,
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    kep = e.keplerian(e.epoch, frame="native", anomaly_type="true", angle_unit="rad")

    assert isinstance(kep["a_m"], float)
    assert isinstance(kep["e"], float)
    assert isinstance(kep["i"], float)
    assert isinstance(kep["anomaly"], float)
    assert kep["anomaly_type"] == "true"
    assert kep["angle_unit"] == "rad"

    np.testing.assert_allclose(float(kep["a_m"]), float(case["a_m"]), atol=1e-3, rtol=0.0)
    np.testing.assert_allclose(float(kep["e"]), float(case["e"]), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(float(kep["i"]), float(case["i"]), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        _angle_delta_rad(float(kep["raan"]), float(case["raan"])),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _angle_delta_rad(float(kep["argp"]), float(case["argp"])),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _angle_delta_rad(float(kep["anomaly"]), float(case["anomaly"])),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )


def test_orbit_keplerian_anomaly_type_consistency() -> None:
    e = Orbit.from_kepler_precise(
        **_kepler_case(),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )
    ts = np.array([0.0, 60.0, 120.0, 300.0], dtype=np.float64)

    k_true = e.keplerian(ts, frame="native", anomaly_type="true", angle_unit="rad")
    k_ecc = e.keplerian(ts, frame="native", anomaly_type="eccentric", angle_unit="rad")
    k_mean = e.keplerian(ts, frame="native", anomaly_type="mean", angle_unit="rad")

    e_mag = np.asarray(k_true["e"], dtype=np.float64)
    nu = np.asarray(k_true["anomaly"], dtype=np.float64)
    ecc = np.asarray(k_ecc["anomaly"], dtype=np.float64)
    mean = np.asarray(k_mean["anomaly"], dtype=np.float64)

    cos_e = (e_mag + np.cos(nu)) / (1.0 + e_mag * np.cos(nu))
    sin_e = np.sqrt(np.maximum(0.0, 1.0 - e_mag * e_mag)) * np.sin(nu) / (
        1.0 + e_mag * np.cos(nu)
    )
    ecc_ref = np.mod(np.arctan2(sin_e, cos_e), 2.0 * np.pi)
    mean_ref = np.mod(ecc_ref - e_mag * np.sin(ecc_ref), 2.0 * np.pi)

    np.testing.assert_allclose(_angle_delta_rad(ecc, ecc_ref), 0.0, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(_angle_delta_rad(mean, mean_ref), 0.0, atol=1e-10, rtol=0.0)


def test_orbit_keplerian_efficiency_frame_validation() -> None:
    e = Orbit.from_kepler_fast(**_kepler_case(), dt_save_s=20.0)
    with pytest.raises(ValueError, match="Efficiency mode supports"):
        e.keplerian(0.0, frame="gcrf")


def test_orbit_keplerian_precision_supports_named_inertial_frames() -> None:
    e = Orbit.from_kepler_precise(
        **_kepler_case(),
        inertial_frame="teme",
        gravity_model="newtonian",
        dt_save_s=60.0,
    )
    ts = np.array([0.0, 180.0], dtype=np.float64)
    kep = e.keplerian(ts, frame="gcrf", anomaly_type="true", angle_unit="deg")

    assert kep["frame"] == "gcrf"
    assert kep["angle_unit"] == "deg"
    assert np.asarray(kep["a_m"]).shape == (2,)
    assert np.asarray(kep["e"]).shape == (2,)
    assert np.isfinite(np.asarray(kep["a_m"])).all()
    assert np.isfinite(np.asarray(kep["e"])).all()

    kep_scalar = e.keplerian(e.epoch, frame="gcrf", anomaly_type="true", angle_unit="rad")
    assert isinstance(kep_scalar["a_m"], float)
    assert np.isfinite(float(kep_scalar["a_m"]))


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
