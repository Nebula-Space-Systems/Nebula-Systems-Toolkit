from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

import nstk.propagation.orbit as orbit_module
from nstk.propagation import (
    NumericalPropagatorFactory,
    Orbit,
    TwoBodyPropagatorFactory,
    build_nadir_sun_constrained_attitude_provider,
    build_j2_j3_j4_propagator,
    build_numerical_propagator,
    build_two_body_propagator,
)
from nstk.time_utils import astropy_time_to_orekit_date


def _make_two_body_orbit(*args, **kwargs) -> Orbit:
    return Orbit(build_two_body_propagator(*args, **kwargs))


def _make_numerical_orbit(*args, **kwargs) -> Orbit:
    return Orbit(build_numerical_propagator(*args, **kwargs))


def test_orbit_cartesian_accessors_return_vectorized_numpy() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_two_body_orbit(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )

    ts = Time(epoch.unix + np.array([0.0, 30.0, 60.0], dtype=np.float64), format="unix", scale="utc")

    r = orbit.get_position(ts)
    v = orbit.get_velocity(ts)
    a = orbit.get_acceleration(ts)
    rp, vp = orbit.get_pv(ts)
    rpva, vpva, apva = orbit.get_pva(ts)

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert a.shape == (3, 3)
    assert rp.shape == (3, 3)
    assert vp.shape == (3, 3)
    assert rpva.shape == (3, 3)
    assert vpva.shape == (3, 3)
    assert apva.shape == (3, 3)

    rs = orbit.get_position(epoch)
    vs = orbit.get_velocity(epoch)
    assert rs.shape == (1, 3)
    assert vs.shape == (1, 3)

    np.testing.assert_allclose(rp, r)
    np.testing.assert_allclose(vp, v)
    np.testing.assert_allclose(rpva, r)
    np.testing.assert_allclose(vpva, v)
    np.testing.assert_allclose(apva, a)


def test_orbit_geodetic_and_attitude_accessors() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_two_body_orbit(
        epoch=epoch,
        a=7200e3,
        e=0.002,
        i=np.deg2rad(40.0),
        raan=np.deg2rad(5.0),
        argp=np.deg2rad(30.0),
        anomaly=np.deg2rad(20.0),
    )

    ts = Time(epoch.unix + np.array([0.0, 20.0, 40.0, 60.0], dtype=np.float64), format="unix", scale="utc")

    geodetic_deg = orbit.get_geodetic(ts)
    geodetic_rad = orbit.get_geodetic(ts, degrees=False)
    att = orbit.get_attitude_quat(ts, quaternion_convention="scalar_last")

    assert geodetic_deg.shape == (4, 3)
    assert geodetic_rad.shape == (4, 3)
    assert att.shape == (4, 4)

    assert isinstance(att, np.ndarray)
    assert isinstance(geodetic_deg, np.ndarray)
    assert np.all(np.isfinite(att))
    assert np.all(np.isfinite(geodetic_deg))

    np.testing.assert_allclose(np.deg2rad(geodetic_deg[:, :2]), geodetic_rad[:, :2])
    np.testing.assert_allclose(geodetic_deg[:, 2], geodetic_rad[:, 2])

    state0 = orbit.propagator.propagate(orbit.propagator.getInitialState().getDate())
    rot0 = state0.getAttitude().getRotation()
    att0 = orbit.get_attitude_quat(0.0, quaternion_convention="scalar_last")
    expected_stk_order = np.array(
        [
            float(rot0.getQ1()),
            float(rot0.getQ2()),
            float(rot0.getQ3()),
            float(rot0.getQ0()),
        ],
        dtype=np.float64,
    )
    assert att0.shape == (1, 4)
    assert np.allclose(att0[0], expected_stk_order)


def test_orbit_attitude_rate_and_acceleration_accessors() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_two_body_orbit(
        epoch=epoch,
        a=7200e3,
        e=0.002,
        i=np.deg2rad(40.0),
        raan=np.deg2rad(5.0),
        argp=np.deg2rad(30.0),
        anomaly=np.deg2rad(20.0),
    )

    dt_s = np.array([0.0, 20.0, 40.0, 60.0], dtype=np.float64)
    ts = Time(epoch.unix + dt_s, format="unix", scale="utc")

    rate = orbit.get_attitude_spin(ts)
    accel = orbit.get_attitude_acceleration(ts)

    assert rate.shape == (4, 3)
    assert accel.shape == (4, 3)
    assert np.all(np.isfinite(rate))
    assert np.all(np.isfinite(accel))

    rate_scalar = orbit.get_attitude_spin(0.0)
    accel_scalar = orbit.get_attitude_acceleration(0.0)
    assert rate_scalar.shape == (1, 3)
    assert accel_scalar.shape == (1, 3)

    date0 = orbit.propagator.getInitialState().getDate()
    expected_rate = np.empty((dt_s.size, 3), dtype=np.float64)
    expected_accel = np.empty((dt_s.size, 3), dtype=np.float64)

    for i, dt in enumerate(dt_s):
        state = orbit.propagator.propagate(date0.shiftedBy(float(dt)))
        spin = state.getAttitude().getSpin()
        rotation_accel = state.getAttitude().getRotationAcceleration()
        expected_rate[i] = [float(spin.getX()), float(spin.getY()), float(spin.getZ())]
        expected_accel[i] = [
            float(rotation_accel.getX()),
            float(rotation_accel.getY()),
            float(rotation_accel.getZ()),
        ]

    assert np.allclose(rate, expected_rate)
    assert np.allclose(accel, expected_accel)
    assert np.allclose(rate_scalar[0], expected_rate[0])
    assert np.allclose(accel_scalar[0], expected_accel[0])


def test_orbit_constructor_choices_two_body_and_numerical() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    two_body = _make_two_body_orbit(
        epoch,
        7000e3,
        0.001,
        np.deg2rad(53.0),
        np.deg2rad(20.0),
        np.deg2rad(15.0),
        np.deg2rad(10.0),
    )
    numerical = _make_numerical_orbit(
        epoch,
        7000e3,
        0.001,
        np.deg2rad(53.0),
        np.deg2rad(20.0),
        np.deg2rad(15.0),
        np.deg2rad(10.0),
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )

    assert str(two_body.propagator.__class__.__name__) == "org.orekit.propagation.analytical.KeplerianPropagator"
    assert str(numerical.propagator.__class__.__name__) == "org.orekit.propagation.numerical.NumericalPropagator"


def test_state_based_factory_classes_cover_old_constructor_behavior() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    seed = _make_two_body_orbit(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )
    state = seed.propagator.getInitialState()

    two_body = TwoBodyPropagatorFactory()(state)
    numerical = NumericalPropagatorFactory(
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )(state)

    assert str(two_body.__class__.__name__) == "org.orekit.propagation.analytical.KeplerianPropagator"
    assert str(numerical.__class__.__name__) == "org.orekit.propagation.numerical.NumericalPropagator"


def test_nadir_sun_constrained_attitude_provider_matches_expected_axes() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.attitudes import PredefinedTarget  # type: ignore
    from org.orekit.bodies import CelestialBodyFactory  # type: ignore

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    provider = build_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    propagator = build_two_body_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        attitude_provider=provider,
    )

    frame = orbit_module.FramesFactory.getGCRF()
    earth = orbit_module.WGS84_ELLIPSOID
    sun = CelestialBodyFactory.getSun()
    state = propagator.propagate(propagator.getInitialState().getDate().shiftedBy(120.0))
    pv = state.getPVCoordinates(frame)

    nadir_dir = PredefinedTarget.NADIR.getTargetDirection(sun, earth, pv, frame).normalize()
    sun_dir = PredefinedTarget.SUN.getTargetDirection(sun, earth, pv, frame).normalize()
    sun_tangent_dir = sun_dir.subtract(
        nadir_dir.scalarMultiply(sun_dir.dotProduct(nadir_dir))
    ).normalize()

    rotation = state.getAttitude().getRotation()
    body_x_in_ref = rotation.applyInverseTo(Vector3D.PLUS_I).normalize()
    body_y_in_ref = rotation.applyInverseTo(Vector3D.PLUS_J).normalize()
    body_z_in_ref = rotation.applyInverseTo(Vector3D.PLUS_K).normalize()

    assert body_z_in_ref.dotProduct(nadir_dir) == pytest.approx(1.0, abs=1.0e-12)
    assert body_x_in_ref.dotProduct(sun_tangent_dir) == pytest.approx(1.0, abs=1.0e-12)
    assert body_x_in_ref.dotProduct(body_z_in_ref) == pytest.approx(0.0, abs=1.0e-12)
    assert (
        body_x_in_ref.crossProduct(body_y_in_ref).normalize().dotProduct(body_z_in_ref)
        == pytest.approx(1.0, abs=1.0e-12)
    )


def test_nadir_sun_constrained_attitude_provider_can_be_used_with_numerical_builder() -> None:
    provider = build_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    propagator = build_numerical_propagator(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        attitude_provider=provider,
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )

    state = propagator.propagate(propagator.getInitialState().getDate().shiftedBy(60.0))
    assert state.getAttitude() is not None


def test_orbit_default_attitude_quaternions_are_available() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    t_query = np.array([0.0, 60.0, 120.0], dtype=np.float64)

    orb_default = _make_two_body_orbit(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )
    q_default = orb_default.get_attitude_quat(t_query, quaternion_convention="scalar_last")
    assert q_default.shape == (3, 4)
    assert np.all(np.isfinite(q_default))


def test_orbit_accepts_orekit_absolutedate_time_inputs() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_two_body_orbit(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )

    t0 = astropy_time_to_orekit_date(epoch)
    t1 = t0.shiftedBy(60.0)
    t2 = t0.shiftedBy(120.0)

    r_from_dates = orbit.get_position([t0, t1, t2], frame="gcrf")
    r_from_seconds = orbit.get_position(
        np.array([0.0, 60.0, 120.0], dtype=np.float64),
        frame="gcrf",
    )

    assert r_from_dates.shape == (3, 3)
    assert np.all(np.isfinite(r_from_dates))
    assert np.allclose(r_from_dates, r_from_seconds, rtol=0.0, atol=1.0e-8)

    r_scalar = orbit.get_position(t1, frame="gcrf")
    assert r_scalar.shape == (1, 3)
    assert np.all(np.isfinite(r_scalar))
    assert np.allclose(r_scalar[0], r_from_seconds[1], rtol=0.0, atol=1.0e-8)


def test_orbit_no_longer_exposes_propagator_constructor_classmethods() -> None:
    assert not hasattr(Orbit, "from_spacecraft_state")
    assert not hasattr(Orbit, "from_kepler_two_body")
    assert not hasattr(Orbit, "from_kepler_numerical")


def test_j2_j3_j4_builder_returns_eckstein_hechler_propagator() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    propagator = build_j2_j3_j4_propagator(
        epoch=epoch,
        a=7050e3,
        e=0.001,
        i=np.deg2rad(50.0),
        raan=np.deg2rad(15.0),
        argp=np.deg2rad(12.0),
        anomaly=np.deg2rad(8.0),
    )

    assert (
        str(propagator.__class__.__name__)
        == "org.orekit.propagation.analytical.EcksteinHechlerPropagator"
    )
