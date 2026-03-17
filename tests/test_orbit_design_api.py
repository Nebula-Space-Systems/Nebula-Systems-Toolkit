from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time

from nebula.propagation.orbit import Orbit, astropy_time_to_orekit_date


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

    assert np.all(np.isfinite(att))


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
    q_default = orb_default.get_attitude_np(t_query)
    assert q_default.shape == (3, 4)
    assert np.all(np.isfinite(q_default))

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
    q_tnw = orb_tnw.get_attitude_np(t_query)
    assert q_tnw.shape == (3, 4)
    assert np.all(np.isfinite(q_tnw))
    assert not np.allclose(q_default, q_tnw)

    from org.orekit.attitudes import LofOffset
    from org.orekit.frames import LOFType

    orb_tnw.set_attitude_law(LofOffset(orb_tnw.get_native_frame(), LOFType.QSW))
    q_qsw = orb_tnw.get_attitude_np(t_query)
    assert q_qsw.shape == (3, 4)
    assert np.all(np.isfinite(q_qsw))
    assert not np.allclose(q_tnw, q_qsw)

    orb_tnw.set_attitude_law({"type": "nadir"})
    q_nadir = orb_tnw.get_attitude_np(t_query)
    assert q_nadir.shape == (3, 4)
    assert np.all(np.isfinite(q_nadir))


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

    r_from_dates = orbit.get_p_np([t0, t1, t2], frame="gcrf")
    r_from_seconds = orbit.get_p_np(np.array([0.0, 60.0, 120.0], dtype=np.float64), frame="gcrf")

    assert r_from_dates.shape == (3, 3)
    assert np.all(np.isfinite(r_from_dates))
    assert np.allclose(r_from_dates, r_from_seconds, rtol=0.0, atol=1.0e-8)

    r_scalar = orbit.get_p_np(t1, frame="gcrf")
    assert r_scalar.shape == (3,)
    assert np.all(np.isfinite(r_scalar))
    assert np.allclose(r_scalar, r_from_seconds[1], rtol=0.0, atol=1.0e-8)
