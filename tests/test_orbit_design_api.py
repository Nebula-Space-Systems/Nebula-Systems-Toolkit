from __future__ import annotations

import numpy as np
from astropy.time import Time

from nstk.propagation.orbit import Orbit
from nstk.time_utils import astropy_time_to_orekit_date


def test_orbit_cartesian_accessors_return_vectorized_numpy() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = Orbit.from_kepler_two_body(
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
    orbit = Orbit.from_kepler_two_body(
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
    orbit = Orbit.from_kepler_two_body(
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

    two_body = Orbit.from_kepler_two_body(
        epoch,
        7000e3,
        0.001,
        np.deg2rad(53.0),
        np.deg2rad(20.0),
        np.deg2rad(15.0),
        np.deg2rad(10.0),
    )
    numerical = Orbit.from_kepler_numerical(
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


def test_orbit_default_attitude_quaternions_are_available() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    t_query = np.array([0.0, 60.0, 120.0], dtype=np.float64)

    orb_default = Orbit.from_kepler_two_body(
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
    orbit = Orbit.from_kepler_two_body(
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
