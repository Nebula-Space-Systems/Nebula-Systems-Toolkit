from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time

from nebula.propagation.orbit import (
    Orbit,
    FramesFactory,
    _astropy_to_absdate_utc,
)


def _ephemeris_generators_size(propagator) -> int:
    cls = propagator.getClass()
    while cls is not None:
        try:
            field = cls.getDeclaredField("ephemerisGenerators")
            field.setAccessible(True)
            return int(field.get(propagator).size())
        except Exception:
            cls = cls.getSuperclass()
    raise RuntimeError("Could not introspect ephemerisGenerators list")


def _native_to_itrf(ephem: Orbit, t: Time) -> tuple[np.ndarray, np.ndarray]:
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.utils import PVCoordinates  # type: ignore

    r_n, v_n = ephem.pv(t, frame="native")
    abs_t = _astropy_to_absdate_utc(t)
    tr = ephem._frame_native.getTransformTo(ephem._itrf, abs_t)  # type: ignore[attr-defined]
    pv_n = PVCoordinates(Vector3D(*r_n.tolist()), Vector3D(*v_n.tolist()))
    pv_i = tr.transformPVCoordinates(pv_n)
    r_i = np.asarray(pv_i.getPosition().toArray(), dtype=np.float64)
    v_i = np.asarray(pv_i.getVelocity().toArray(), dtype=np.float64)
    return r_i, v_i


def test_numerical_propagator_uses_cartesian_orbit_type() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(40.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
        gravity_model="newtonian",
    )
    assert str(e.propagator.getOrbitType()) == "CARTESIAN"


def test_ephemeris_generator_list_does_not_grow() -> None:
    e = Orbit.from_kepler_precise(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(10.0),
        argp=np.deg2rad(20.0),
        anomaly=0.0,
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    before = _ephemeris_generators_size(e.propagator)
    assert before == 0

    for k in range(1, 6):
        _ = e.pos(e.epoch + k * 60.0 * u.s, frame="native")
        _ = e.pos(e.epoch - k * 60.0 * u.s, frame="native")

    after = _ephemeris_generators_size(e.propagator)
    assert after <= 1


def test_from_pv_respects_propagate_inertial_frame_for_inertial_input() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    ref = Orbit.from_kepler_precise(
        epoch=epoch,
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(45.0),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(10.0),
        anomaly=np.deg2rad(15.0),
        inertial_frame="teme",
        gravity_model="newtonian",
    )

    r_teme, v_teme = ref.pv(epoch, frame="native")
    e2 = Orbit.from_pv(
        r_teme,
        v_teme,
        epoch,
        frame="teme",
        propagate_inertial_frame="gcrf",
        gravity_model="newtonian",
    )

    # Must now propagate in requested inertial frame, not the input frame.
    assert e2._frame_native.getName() == "GCRF"  # type: ignore[attr-defined]

    # Cross-check epoch state against direct TEME->GCRF transform.
    from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
    from org.orekit.utils import PVCoordinates  # type: ignore

    abs_t = _astropy_to_absdate_utc(epoch)
    teme = FramesFactory.getTEME()
    gcrf = FramesFactory.getGCRF()
    tr = teme.getTransformTo(gcrf, abs_t)
    pv_t = PVCoordinates(Vector3D(*r_teme.tolist()), Vector3D(*v_teme.tolist()))
    pv_g = tr.transformPVCoordinates(pv_t)

    r_expected = np.asarray(pv_g.getPosition().toArray(), dtype=np.float64)
    v_expected = np.asarray(pv_g.getVelocity().toArray(), dtype=np.float64)
    r_chk, v_chk = e2.pv(epoch, frame="native")

    np.testing.assert_allclose(r_chk, r_expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(v_chk, v_expected, atol=1e-9, rtol=0.0)


def test_itrf_transform_mode_is_frame_consistent() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    common = dict(
        epoch=epoch,
        a_m=7000e3,
        e=0.01,
        i=np.deg2rad(63.4),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(40.0),
        anomaly=np.deg2rad(5.0),
        gravity_model="newtonian",
        dt_save_s=60.0,
    )
    e_cached = Orbit.from_kepler_precise(itrf_query_mode="cached", **common)
    e_xform = Orbit.from_kepler_precise(itrf_query_mode="transform", **common)

    # Midpoint-like queries to stress interpolation behavior.
    ts = epoch + (np.arange(1, 31, dtype=np.float64) * 60.0 + 17.0) * u.s

    err_cached = []
    err_xform = []
    for t in ts:
        r_c, v_c = e_cached.pv(t, frame="itrf")
        r_c_ref, v_c_ref = _native_to_itrf(e_cached, t)
        err_cached.append(
            float(np.linalg.norm(r_c - r_c_ref) + np.linalg.norm(v_c - v_c_ref))
        )

        r_x, v_x = e_xform.pv(t, frame="itrf")
        r_x_ref, v_x_ref = _native_to_itrf(e_xform, t)
        err_xform.append(
            float(np.linalg.norm(r_x - r_x_ref) + np.linalg.norm(v_x - v_x_ref))
        )

    err_cached = np.asarray(err_cached)
    err_xform = np.asarray(err_xform)

    assert float(np.mean(err_xform)) < float(np.mean(err_cached))
    assert float(np.max(err_xform)) < 1e-6
