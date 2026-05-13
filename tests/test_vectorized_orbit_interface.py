from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

import nstk._orekit_frames as orekit_frames
import nstk.propagation.orbit as orbit_module
import nstk.propagation._propagator_utils as propagator_utils
import nstk.time_utils as time_utils
from nstk.propagation import Orbit, build_two_body_propagator


def _wrap_0_2pi(x: float) -> float:
    wrapped = float(x) % (2.0 * np.pi)
    if abs(wrapped - 2.0 * np.pi) < 1.0e-12:
        return 0.0
    return wrapped


def _make_orbit(*, should_cache: bool = True, inertial_frame: str = "gcrf") -> Orbit:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    return Orbit(
        build_two_body_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame=inertial_frame,
        ),
        should_cache=should_cache,
    )


def _make_orbit_with_additional_data() -> Orbit:
    orbit_module._bind_orbit_java()

    import jpype
    from org.orekit.orbits import KeplerianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.utils import Constants

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    frame = orbit_module.FramesFactory.getGCRF()
    orbit0 = KeplerianOrbit(
        7000e3,
        0.001,
        np.deg2rad(53.0),
        np.deg2rad(15.0),
        np.deg2rad(20.0),
        np.deg2rad(10.0),
        propagator_utils._coerce_position_angle_type("mean"),
        frame,
        orbit_module.astropy_time_to_orekit_date(epoch),
        Constants.WGS84_EARTH_MU,
    )

    doubles = jpype.JArray(jpype.JDouble)
    state0 = SpacecraftState(orbit0, 750.0)
    state0 = state0.addAdditionalData("clock_bias", doubles([1.5]))
    state0 = state0.addAdditionalData("control_mode", doubles([2.0, 3.0]))
    state0 = state0.addAdditionalStateDerivative("clock_bias_dot", doubles([0.01]))

    propagator = KeplerianPropagator(orbit0)
    propagator.resetInitialState(state0)
    return Orbit(propagator)


def test_orbit_raw_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orbit = _make_orbit()
    wrapper = Orbit(orbit.propagator, should_cache=False)

    real_make_times_astropy = time_utils.make_times_astropy
    calls = {"count": 0}

    def _tracking_make_times_astropy(epoch: Time, delta_times_sec: np.ndarray) -> Time:
        calls["count"] += 1
        return real_make_times_astropy(epoch, delta_times_sec)

    monkeypatch.setattr(orbit_module, "make_times_astropy", _tracking_make_times_astropy)

    sampled = wrapper.sample(0.0, position=True, velocity=True, mass=True)

    assert sampled.n == 1
    assert sampled.input_was_scalar is True
    assert sampled.position_m.shape == (1, 3)
    assert sampled.velocity_mps.shape == (1, 3)
    assert sampled.mass_kg.shape == (1,)
    assert calls["count"] == 0
    times_astropy = sampled.times_astropy
    assert times_astropy is not None and times_astropy.shape == (1,)
    assert calls["count"] == 1
    assert sampled.times_astropy is times_astropy
    assert calls["count"] == 1
    assert set(sampled.available_fields) == {"position_m", "velocity_mps", "mass_kg"}

    p_raw, v_raw, a_raw = orbit.get_pva([0.0, 60.0])
    assert p_raw.shape == (2, 3)
    assert v_raw.shape == (2, 3)
    assert a_raw.shape == (2, 3)

    geo_deg = orbit.get_geodetic([0.0, 60.0])
    geo_rad = orbit.get_geodetic([0.0, 60.0], degrees=False)
    assert geo_deg.shape == (2, 3)
    assert geo_rad.shape == (2, 3)
    np.testing.assert_allclose(np.deg2rad(geo_deg[:, :2]), geo_rad[:, :2], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(geo_deg[:, 2], geo_rad[:, 2], atol=0.0, rtol=0.0)

    states = orbit.get_java_states([0.0, 60.0])
    assert isinstance(states, list)
    assert len(states) == 2

    direct = wrapper.get_position([0.0, 60.0], frame=orbit.native_frame)
    from_states = np.asarray(
        [
            state.getPVCoordinates(orbit.native_frame).getPosition().toArray()
            for state in states
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(direct, from_states, atol=1e-8, rtol=0.0)


def test_vectorized_sample_matches_direct_orekit_for_attitude_and_elements() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import RotationConvention, RotationOrder
    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    orbit = _make_orbit(inertial_frame="eme2000")
    gcrf = orbit_module.FramesFactory.getGCRF()
    dt_s = np.array([0.0, 120.0, 240.0], dtype=np.float64)

    sampled = orbit.sample(
        dt_s,
        attitude_reference_frame=gcrf,
        attitude_quat=True,
        attitude_matrix=True,
        attitude_euler=True,
        attitude_spin=True,
        attitude_acceleration=True,
        attitude_euler_sequence="zyx",
        quaternion_convention="scalar_last",
        elements_frame=gcrf,
        keplerian=True,
        equinoctial=True,
    )

    base = orbit.propagator.getInitialState().getDate()

    expected_quat = np.empty((dt_s.size, 4), dtype=np.float64)
    expected_matrix = np.empty((dt_s.size, 3, 3), dtype=np.float64)
    expected_euler = np.empty((dt_s.size, 3), dtype=np.float64)
    expected_spin = np.empty((dt_s.size, 3), dtype=np.float64)
    expected_accel = np.empty((dt_s.size, 3), dtype=np.float64)
    expected_kepler = np.empty((dt_s.size, 6), dtype=np.float64)
    expected_equinoctial = np.empty((dt_s.size, 6), dtype=np.float64)

    for idx, dt in enumerate(dt_s):
        state = orbit.propagator.propagate(base.shiftedBy(float(dt)))
        attitude = state.getAttitude().withReferenceFrame(gcrf)
        rotation = attitude.getRotation()
        expected_quat[idx] = [
            float(rotation.getQ1()),
            float(rotation.getQ2()),
            float(rotation.getQ3()),
            float(rotation.getQ0()),
        ]
        expected_matrix[idx] = np.asarray(rotation.getMatrix(), dtype=np.float64)
        expected_euler[idx] = np.asarray(
            rotation.getAngles(RotationOrder.ZYX, RotationConvention.FRAME_TRANSFORM),
            dtype=np.float64,
        )
        expected_spin[idx] = np.asarray(attitude.getSpin().toArray(), dtype=np.float64)
        expected_accel[idx] = np.asarray(
            attitude.getRotationAcceleration().toArray(),
            dtype=np.float64,
        )

        orbit_in_frame = CartesianOrbit(
            state.getPVCoordinates(gcrf),
            gcrf,
            state.getDate(),
            float(state.getOrbit().getMu()),
        )
        kep = KeplerianOrbit(orbit_in_frame)
        equi = EquinoctialOrbit(orbit_in_frame)
        expected_kepler[idx] = [
            float(kep.getA()),
            float(kep.getE()),
            float(kep.getI()),
            _wrap_0_2pi(float(kep.getRightAscensionOfAscendingNode())),
            float(kep.getPerigeeArgument()),
            _wrap_0_2pi(float(kep.getMeanAnomaly())),
        ]
        expected_equinoctial[idx] = [
            float(equi.getA()),
            float(equi.getEquinoctialEx()),
            float(equi.getEquinoctialEy()),
            float(equi.getHx()),
            float(equi.getHy()),
            float(equi.getLM()),
        ]

    np.testing.assert_allclose(sampled.attitude_quat_ref_to_body, expected_quat, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sampled.attitude_matrix_ref_to_body, expected_matrix, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sampled.attitude_euler_ref_to_body, expected_euler, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sampled.attitude_spin_body_rad_s, expected_spin, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sampled.attitude_accel_body_rad_s2, expected_accel, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        orbit.get_keplerian_classical(dt_s, frame=gcrf),
        expected_kepler,
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        orbit.get_equinoctial(dt_s, frame=gcrf),
        expected_equinoctial,
        atol=1e-8,
        rtol=0.0,
    )


def test_vectorized_additional_states_and_strict_derivative_handling() -> None:
    orbit = _make_orbit_with_additional_data()

    assert set(orbit.list_additional_states()) == {"clock_bias", "control_mode"}
    assert orbit.list_additional_state_derivatives() == ["clock_bias_dot"]

    sampled = orbit.sample([0.0, 60.0], additional_states=("clock_bias", "control_mode"))

    assert set(sampled.additional) == {"clock_bias", "control_mode"}
    assert sampled.additional["clock_bias"].shape == (2,)
    assert sampled.additional["control_mode"].shape == (2, 2)
    np.testing.assert_allclose(sampled.additional["clock_bias"], [1.5, 1.5], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        sampled.additional["control_mode"],
        np.asarray([[2.0, 3.0], [2.0, 3.0]], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )

    with pytest.raises(Exception, match="clock_bias_dot"):
        orbit.sample([0.0], additional_state_derivatives=("clock_bias_dot",))

    loose = orbit.sample(
        [0.0],
        additional_state_derivatives=("clock_bias_dot",),
        strict=False,
    )
    assert loose.additional_derivatives == {}


def test_vectorized_cache_behavior_is_optional() -> None:
    cached = _make_orbit(should_cache=True)
    uncached = _make_orbit(should_cache=False)

    assert cached.coverage() == (0.0, 0.0)
    assert uncached.coverage() == (0.0, 0.0)

    # Analytical orbit-only requests use the direct orbit fast path and skip ephemeris caching.
    cached.sample([0.0, 60.0], position=True)
    uncached.sample([0.0, 60.0], position=True)

    assert cached.coverage() == (0.0, 0.0)
    assert uncached.coverage() == (0.0, 0.0)

    # Requests that need full states still engage the optional cache.
    cached.sample([0.0, 60.0], attitude_quat=True)
    assert cached.coverage() == (0.0, 60.0)
    cached.clear_cache()
    uncached.precompute(0.0, 60.0)

    assert cached.coverage() == (0.0, 0.0)
    assert uncached.coverage() == (0.0, 0.0)


def test_orbit_resolve_named_frame_accepts_versioned_itrf_aliases() -> None:
    orbit_module._bind_orbit_java()

    from org.orekit.frames import FramesFactory, ITRFVersion
    from org.orekit.utils import IERSConventions

    iers = IERSConventions.IERS_2010
    expected = FramesFactory.getITRF(ITRFVersion.ITRF_2014, iers, True)

    for name in ("itrf2014", "ITRF_2014", "itrs2014", "ecef2014"):
        resolved = orbit_module._resolve_named_frame(name, iers=iers, simple_eop=True)
        assert resolved == expected


@pytest.mark.parametrize(
    ("name", "expected_factory"),
    [
        ("mod2003", lambda ff, ic: ff.getMOD(ic.IERS_2003)),
        ("tod2010", lambda ff, ic: ff.getTOD(ic.IERS_2010, True)),
        ("cirf1996", lambda ff, ic: ff.getCIRF(ic.IERS_1996, True)),
        ("gtod2010", lambda ff, ic: ff.getGTOD(ic.IERS_2010, True)),
        ("tirf2003", lambda ff, ic: ff.getTIRF(ic.IERS_2003, True)),
        ("ecliptic2010", lambda ff, ic: ff.getEcliptic(ic.IERS_2010)),
        ("itrfcio2010", lambda ff, ic: ff.getITRF(ic.IERS_2010, True)),
        ("itrfequinox2003", lambda ff, ic: ff.getITRFEquinox(ic.IERS_2003, True)),
    ],
)
def test_orbit_resolve_named_frame_accepts_all_iers_versioned_families(
    name: str,
    expected_factory,
) -> None:
    orbit_module._bind_orbit_java()

    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions

    resolved = orbit_module._resolve_named_frame(
        name,
        iers=IERSConventions.IERS_1996,
        simple_eop=True,
    )
    expected = expected_factory(FramesFactory, IERSConventions)
    assert resolved == expected


def test_orbit_resolve_named_frame_uses_latest_version_when_unspecified() -> None:
    orbit_module._bind_orbit_java()

    from org.orekit.frames import FramesFactory, Predefined
    from org.orekit.utils import IERSConventions

    latest_iers = orekit_frames._latest_iers_convention()
    latest_itrf = orekit_frames._latest_itrf_version()

    expected_latest = {
        "itrf": FramesFactory.getITRF(latest_itrf, IERSConventions.IERS_1996, True),
        "itrs": FramesFactory.getITRF(latest_itrf, IERSConventions.IERS_1996, True),
        "ecef": FramesFactory.getITRF(latest_itrf, IERSConventions.IERS_1996, True),
        "mod": FramesFactory.getMOD(latest_iers),
        "tod": FramesFactory.getTOD(latest_iers, True),
        "cirf": FramesFactory.getCIRF(latest_iers, True),
        "gtod": FramesFactory.getGTOD(latest_iers, True),
        "tirf": FramesFactory.getTIRF(latest_iers, True),
        "ecliptic": FramesFactory.getEcliptic(latest_iers),
        "itrfcio": FramesFactory.getITRF(latest_iers, True),
        "itrfequinox": FramesFactory.getITRFEquinox(latest_iers, True),
    }

    for name, expected in expected_latest.items():
        resolved = orbit_module._resolve_named_frame(
            name,
            iers=IERSConventions.IERS_1996,
            simple_eop=True,
        )
        assert resolved == expected

    predefined_name = "ITRF_EQUINOX_CONV_2003_ACCURATE_EOP"
    assert orbit_module._resolve_named_frame(
        predefined_name,
        iers=IERSConventions.IERS_1996,
        simple_eop=True,
    ) == FramesFactory.getFrame(getattr(Predefined, predefined_name))


@pytest.mark.parametrize("frame_name", ["itrf2014", "tod2010", "gtod2010", "itrfequinox2003"])
def test_orbit_public_frame_resolution_accepts_versioned_names(frame_name: str) -> None:
    orbit = _make_orbit()
    orbit_module._bind_orbit_java()

    from org.orekit.frames import FramesFactory, ITRFVersion
    from org.orekit.utils import IERSConventions

    t = np.array([0.0, 60.0], dtype=np.float64)
    expected_by_name = {
        "tod2010": FramesFactory.getTOD(IERSConventions.IERS_2010, True),
        "gtod2010": FramesFactory.getGTOD(IERSConventions.IERS_2010, True),
        "itrfequinox2003": FramesFactory.getITRFEquinox(IERSConventions.IERS_2003, True),
        "itrf2014": FramesFactory.getITRF(ITRFVersion.ITRF_2014, IERSConventions.IERS_2010, True),
    }

    from_string = orbit.get_position(t, frame=frame_name)
    from_frame = orbit.get_position(t, frame=expected_by_name[frame_name])

    np.testing.assert_allclose(from_string, from_frame, atol=1e-8, rtol=0.0)


def test_orbit_resolve_named_frame_rejects_unknown_itrf_version() -> None:
    orbit_module._bind_orbit_java()

    with pytest.raises(ValueError, match="Supported ITRF versions"):
        orbit_module._resolve_named_frame("itrf2099", iers=orbit_module._iers_default(), simple_eop=True)
