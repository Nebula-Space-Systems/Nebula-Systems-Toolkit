from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time

from nstk.propagation.orbit import Orbit
from nstk.time_utils import astropy_time_to_orekit_date


def test_orbit_quantity_accessors_shape_and_units() -> None:
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

    r = orbit.get_p(ts)
    v = orbit.get_v(ts)
    a = orbit.get_a(ts)
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

    assert r.unit == u.m
    assert v.unit == (u.m / u.s)
    assert a.unit == (u.m / (u.s**2))

    rs = orbit.get_p(epoch)
    vs = orbit.get_v(epoch)
    assert rs.shape == (3,)
    assert vs.shape == (3,)

    r_np = orbit.get_p(ts, as_quantity=False)
    v_np = orbit.get_v(ts, as_quantity=False)
    a_np = orbit.get_a(ts, as_quantity=False)
    rp_np, vp_np = orbit.get_pv(ts, as_quantity=False)
    rpva_np, vpva_np, apva_np = orbit.get_pva(ts, as_quantity=False)

    assert isinstance(r_np, np.ndarray)
    assert isinstance(v_np, np.ndarray)
    assert isinstance(a_np, np.ndarray)
    assert isinstance(rp_np, np.ndarray)
    assert isinstance(vp_np, np.ndarray)
    assert isinstance(rpva_np, np.ndarray)
    assert isinstance(vpva_np, np.ndarray)
    assert isinstance(apva_np, np.ndarray)
    assert r_np.shape == (3, 3)
    assert v_np.shape == (3, 3)
    assert a_np.shape == (3, 3)
    assert np.allclose(r_np, r.to_value(u.m))
    assert np.allclose(v_np, v.to_value(u.m / u.s))
    assert np.allclose(a_np, a.to_value(u.m / (u.s**2)))
    assert np.allclose(rp_np, rp.to_value(u.m))
    assert np.allclose(vp_np, vp.to_value(u.m / u.s))
    assert np.allclose(rpva_np, rpva.to_value(u.m))
    assert np.allclose(vpva_np, vpva.to_value(u.m / u.s))
    assert np.allclose(apva_np, apva.to_value(u.m / (u.s**2)))


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

    lat, lon, alt = orbit.get_geodetic(ts)
    att = orbit.get_attitude(ts)

    assert lat.shape == (4,)
    assert lon.shape == (4,)
    assert alt.shape == (4,)
    assert att.shape == (4, 4)

    assert lat.unit == u.deg
    assert lon.unit == u.deg
    assert alt.unit == u.m
    assert isinstance(att, np.ndarray)
    assert np.all(np.isfinite(att))

    lat_np, lon_np, alt_np = orbit.get_geodetic(ts, as_quantity=False)
    att_np = orbit.get_attitude(ts)

    assert isinstance(lat_np, np.ndarray)
    assert isinstance(lon_np, np.ndarray)
    assert isinstance(alt_np, np.ndarray)
    assert isinstance(att_np, np.ndarray)
    assert lat_np.shape == (4,)
    assert lon_np.shape == (4,)
    assert alt_np.shape == (4,)
    assert att_np.shape == (4, 4)
    assert np.allclose(lat_np, lat.to_value(u.deg))
    assert np.allclose(lon_np, lon.to_value(u.deg))
    assert np.allclose(alt_np, alt.to_value(u.m))
    assert np.allclose(att_np, att)

    state0 = orbit.propagator.propagate(orbit.propagator.getInitialState().getDate())
    rot0 = state0.getAttitude().getRotation()
    att0 = orbit.get_attitude(0.0)
    expected_stk_order = np.array(
        [
            float(rot0.getQ1()),
            float(rot0.getQ2()),
            float(rot0.getQ3()),
            float(rot0.getQ0()),
        ],
        dtype=np.float64,
    )
    assert np.allclose(att0, expected_stk_order)


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

    rate = orbit.get_attitude_rate(ts)
    accel = orbit.get_attitude_acceleration(ts)

    assert rate.shape == (4, 3)
    assert accel.shape == (4, 3)
    assert rate.unit == (u.rad / u.s)
    assert accel.unit == (u.rad / (u.s**2))
    assert np.all(np.isfinite(rate.to_value(u.rad / u.s)))
    assert np.all(np.isfinite(accel.to_value(u.rad / (u.s**2))))

    rate_scalar = orbit.get_attitude_rate(0.0, as_quantity=False)
    accel_scalar = orbit.get_attitude_acceleration(0.0, as_quantity=False)
    assert rate_scalar.shape == (3,)
    assert accel_scalar.shape == (3,)

    rate_np = orbit.get_attitude_rate(ts, as_quantity=False)
    accel_np = orbit.get_attitude_acceleration(ts, as_quantity=False)

    assert isinstance(rate_np, np.ndarray)
    assert isinstance(accel_np, np.ndarray)
    assert rate_np.shape == (4, 3)
    assert accel_np.shape == (4, 3)
    assert np.allclose(rate_np, rate.to_value(u.rad / u.s))
    assert np.allclose(accel_np, accel.to_value(u.rad / (u.s**2)))

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

    assert np.allclose(rate_np, expected_rate)
    assert np.allclose(accel_np, expected_accel)
    assert np.allclose(rate_scalar, expected_rate[0])
    assert np.allclose(accel_scalar, expected_accel[0])


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


def test_orbit_attitude_override_constructor_and_setter() -> None:
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
    q_default = orb_default.get_attitude(t_query)
    assert q_default.shape == (3, 4)
    assert np.all(np.isfinite(q_default))

    orb_vvlh = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        attitude="vvlh",
    )
    q_vvlh = orb_vvlh.get_attitude(t_query)
    assert np.allclose(q_default, q_vvlh)

    orb_tnw = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        attitude="tnw",
    )
    q_tnw = orb_tnw.get_attitude(t_query)
    assert q_tnw.shape == (3, 4)
    assert np.all(np.isfinite(q_tnw))
    assert not np.allclose(q_default, q_tnw)

    from org.orekit.attitudes import LofOffset
    from org.orekit.frames import LOFType

    orb_tnw.set_attitude_law(LofOffset(orb_tnw.get_native_frame(), LOFType.QSW))
    q_qsw = orb_tnw.get_attitude(t_query)
    assert q_qsw.shape == (3, 4)
    assert np.all(np.isfinite(q_qsw))
    assert not np.allclose(q_tnw, q_qsw)

    orb_tnw.set_attitude_law({"type": "nadir"})
    q_nadir = orb_tnw.get_attitude(t_query)
    assert q_nadir.shape == (3, 4)
    assert np.all(np.isfinite(q_nadir))

    orb_tnw.set_attitude_law()
    q_reset = orb_tnw.get_attitude(t_query)
    assert np.allclose(q_default, q_reset)

    orb_qsw = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        attitude=LOFType.QSW,
    )
    q_qsw_ctor = orb_qsw.get_attitude(t_query)
    assert q_qsw_ctor.shape == (3, 4)
    assert np.all(np.isfinite(q_qsw_ctor))
    assert not np.allclose(q_default, q_qsw_ctor)

    orb_qsw.set_attitude_law({"type": "lof", "lof": LOFType.VVLH})
    q_from_mapping = orb_qsw.get_attitude(t_query)
    assert np.allclose(q_default, q_from_mapping)


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

    r_from_dates = orbit.get_p([t0, t1, t2], frame="gcrf", as_quantity=False)
    r_from_seconds = orbit.get_p(
        np.array([0.0, 60.0, 120.0], dtype=np.float64),
        frame="gcrf",
        as_quantity=False,
    )

    assert r_from_dates.shape == (3, 3)
    assert np.all(np.isfinite(r_from_dates))
    assert np.allclose(r_from_dates, r_from_seconds, rtol=0.0, atol=1.0e-8)

    r_scalar = orbit.get_p(t1, frame="gcrf", as_quantity=False)
    assert r_scalar.shape == (3,)
    assert np.all(np.isfinite(r_scalar))
    assert np.allclose(r_scalar, r_from_seconds[1], rtol=0.0, atol=1.0e-8)
