from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.time import Time

import nstk.propagation as propagation
from nstk.propagation import (
    J2J3J4PropagatorFactory,
    NumericalPropagatorFactory,
    Orbit,
    TwoBodyPropagatorFactory,
    build_j2_j3_j4_propagator,
    build_j2_j3_j4_walker_constellation,
    build_numerical_propagator,
    build_numerical_walker_constellation,
    build_two_body_propagator,
    build_two_body_walker_constellation,
    build_walker_constellation,
    build_walker_initial_states,
    build_walker_propagators,
)


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_0_2pi(x: float) -> float:
    wrapped = float(x) % (2.0 * math.pi)
    if abs(wrapped - 2.0 * math.pi) < 1.0e-12:
        return 0.0
    return wrapped


def _ang_diff(a: float, b: float) -> float:
    return abs(_wrap_pm_pi(float(a) - float(b)))


def _state_raan_anomaly(state, *, anomaly_type: str = "mean") -> tuple[float, float]:
    from org.orekit.orbits import KeplerianOrbit  # type: ignore

    kep = KeplerianOrbit(state.getOrbit())
    key = anomaly_type.strip().lower()
    if key == "mean":
        anomaly = float(kep.getMeanAnomaly())
    elif key == "true":
        anomaly = float(kep.getTrueAnomaly())
    elif key == "eccentric":
        anomaly = float(kep.getEccentricAnomaly())
    else:
        raise ValueError("unsupported anomaly_type in test helper")
    return float(kep.getRightAscensionOfAscendingNode()), anomaly


def _initial_raan_anomaly(orbit: Orbit, *, anomaly_type: str = "mean") -> tuple[float, float]:
    return _state_raan_anomaly(orbit.propagator.getInitialState(), anomaly_type=anomaly_type)


def _force_model_names(orbit: Orbit) -> tuple[str, ...]:
    models = orbit.propagator.getAllForceModels()
    return tuple(str(models.get(i).getClass().getSimpleName()) for i in range(models.size()))


def _attitude_quaternion(state) -> tuple[float, float, float, float]:
    rot = state.getAttitude().getRotation()
    return (
        float(rot.getQ0()),
        float(rot.getQ1()),
        float(rot.getQ2()),
        float(rot.getQ3()),
    )


def _make_two_body_orbit(*args, **kwargs) -> Orbit:
    return Orbit(build_two_body_propagator(*args, **kwargs))


def _make_numerical_orbit(*args, **kwargs) -> Orbit:
    return Orbit(build_numerical_propagator(*args, **kwargs))


def test_two_body_walker_constellation_defaults_to_full_raan_span_and_phasing_one() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    seed = _make_two_body_orbit(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )

    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
        total_satellites=6,
        num_planes=3,
        include_seed=True,
    )
    assert len(walker) == 6

    seed_raan, seed_mean = _initial_raan_anomaly(seed, anomaly_type="mean")
    p = 3
    s = 2
    t = 6

    for idx, orb in enumerate(walker):
        plane = idx // s
        slot = idx % s
        d_raan = 2.0 * math.pi * (plane / p)
        d_mean = 2.0 * math.pi * (slot / s + (plane / t))
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _initial_raan_anomaly(orb, anomaly_type="mean")
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9


def test_walker_three_plane_raan_canonicalized_to_0_120_240_degrees() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=0.0,
        argp=np.deg2rad(15.0),
        anomaly=0.0,
        anomaly_type="mean",
        total_satellites=6,
        num_planes=3,
        phasing=1,
        include_seed=True,
    )

    sats_per_plane = 2
    plane_slot_zero_indices = [0, sats_per_plane, 2 * sats_per_plane]
    plane_raans = [
        _wrap_0_2pi(_initial_raan_anomaly(walker[idx], anomaly_type="mean")[0])
        for idx in plane_slot_zero_indices
    ]
    expected_raans = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    for actual, expected in zip(plane_raans, expected_raans, strict=True):
        assert _ang_diff(actual, expected) < 1e-9


def test_walker_three_plane_mean_anomaly_pattern_is_consistent_modulo_2pi() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    total_satellites = 6
    num_planes = 3
    phasing = 1
    sats_per_plane = total_satellites // num_planes

    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=0.0,
        argp=np.deg2rad(15.0),
        anomaly=0.0,
        anomaly_type="mean",
        total_satellites=total_satellites,
        num_planes=num_planes,
        phasing=phasing,
        include_seed=True,
    )

    mean_matrix = np.zeros((num_planes, sats_per_plane), dtype=np.float64)
    for plane_idx in range(num_planes):
        for slot_idx in range(sats_per_plane):
            orbit_idx = plane_idx * sats_per_plane + slot_idx
            _, mean_anomaly = _initial_raan_anomaly(walker[orbit_idx], anomaly_type="mean")
            mean_matrix[plane_idx, slot_idx] = _wrap_0_2pi(mean_anomaly)

    for plane_idx in range(num_planes):
        intra_plane_slot_step = _wrap_pm_pi(mean_matrix[plane_idx, 1] - mean_matrix[plane_idx, 0])
        assert abs(abs(intra_plane_slot_step) - math.pi) < 1e-9

    expected_inter_plane_phase = 2.0 * math.pi * phasing / total_satellites
    for slot_idx in range(sats_per_plane):
        step_01 = _wrap_pm_pi(mean_matrix[1, slot_idx] - mean_matrix[0, slot_idx])
        step_12 = _wrap_pm_pi(mean_matrix[2, slot_idx] - mean_matrix[1, slot_idx])
        assert abs(step_01 - expected_inter_plane_phase) < 1e-9
        assert abs(step_12 - expected_inter_plane_phase) < 1e-9

    expected_means_deg = np.array([[0.0, 180.0], [60.0, 240.0], [120.0, 300.0]])
    np.testing.assert_allclose(np.rad2deg(mean_matrix), expected_means_deg, atol=1e-8)


def test_walker_three_plane_output_order_is_plane_major_slot_major() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=0.0,
        argp=np.deg2rad(15.0),
        anomaly=0.0,
        anomaly_type="mean",
        total_satellites=6,
        num_planes=3,
        phasing=1,
        include_seed=True,
    )

    ordered_pairs_deg = []
    for orbit in walker:
        raan, mean_anomaly = _initial_raan_anomaly(orbit, anomaly_type="mean")
        ordered_pairs_deg.append(
            (
                np.rad2deg(_wrap_0_2pi(raan)),
                np.rad2deg(_wrap_0_2pi(mean_anomaly)),
            )
        )

    expected_pairs_deg = [
        (0.0, 0.0),
        (0.0, 180.0),
        (120.0, 60.0),
        (120.0, 240.0),
        (240.0, 120.0),
        (240.0, 300.0),
    ]
    np.testing.assert_allclose(np.asarray(ordered_pairs_deg), np.asarray(expected_pairs_deg), atol=1e-8)


def test_walker_keplerian_api_reports_raan_and_anomaly_in_positive_ranges() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=0.0,
        argp=np.deg2rad(15.0),
        anomaly=0.0,
        anomaly_type="mean",
        total_satellites=6,
        num_planes=3,
        phasing=1,
        include_seed=True,
    )

    for orbit in walker:
        kep_deg = orbit.get_keplerian_classical(0.0, degrees=True)[0]
        assert 0.0 <= float(kep_deg[3]) < 360.0
        assert 0.0 <= float(kep_deg[5]) < 360.0

        kep_rad = orbit.get_keplerian_classical(0.0, degrees=False)[0]
        assert 0.0 <= float(kep_rad[3]) < (2.0 * math.pi)
        assert 0.0 <= float(kep_rad[5]) < (2.0 * math.pi)


def test_walker_custom_raan_span_offsets_and_true_anomaly() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    seed = _make_two_body_orbit(
        epoch=epoch,
        a=7050e3,
        e=0.02,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="true",
    )

    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7050e3,
        e=0.02,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="true",
        total_satellites=6,
        num_planes=3,
        phasing=2,
        raan_span=math.pi,
        initial_raan_offset=np.deg2rad(15.0),
        initial_anomaly_offset=np.deg2rad(7.5),
        include_seed=False,
    )
    assert len(walker) == 5

    seed_raan, seed_true = _initial_raan_anomaly(seed, anomaly_type="true")
    base_raan = _wrap_pm_pi(seed_raan + np.deg2rad(15.0))
    base_true = _wrap_pm_pi(seed_true + np.deg2rad(7.5))
    p = 3
    s = 2
    t = 6

    expected = []
    for plane in range(p):
        for slot in range(s):
            if plane == 0 and slot == 0:
                continue
            d_raan = math.pi * (plane / p)
            d_true = 2.0 * math.pi * (slot / s + (2.0 * plane / t))
            expected.append(
                (
                    _wrap_pm_pi(base_raan + d_raan),
                    _wrap_pm_pi(base_true + d_true),
                )
            )

    for orb, (exp_raan, exp_true) in zip(walker, expected, strict=True):
        raan, true_anomaly = _initial_raan_anomaly(orb, anomaly_type="true")
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(true_anomaly, exp_true) < 1e-9


def test_walker_include_seed_false_and_validation() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    walker = build_two_body_walker_constellation(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
        total_satellites=6,
        num_planes=3,
        phasing=0,
        include_seed=False,
    )
    assert len(walker) == 5

    try:
        _ = build_two_body_walker_constellation(
            epoch=epoch,
            a=7000e3,
            e=0.001,
            i=np.deg2rad(53.0),
            raan=np.deg2rad(20.0),
            argp=np.deg2rad(15.0),
            anomaly=np.deg2rad(10.0),
            anomaly_type="mean",
            total_satellites=5,
            num_planes=3,
        )
        assert False, "expected ValueError for non-divisible walker geometry"
    except ValueError:
        pass

    with pytest.raises(ValueError, match="<= 2\\*pi"):
        build_two_body_walker_constellation(
            epoch=epoch,
            a=7000e3,
            e=0.001,
            i=np.deg2rad(53.0),
            raan=np.deg2rad(20.0),
            argp=np.deg2rad(15.0),
            anomaly=np.deg2rad(10.0),
            anomaly_type="mean",
            total_satellites=6,
            num_planes=3,
            raan_span=2.1 * math.pi,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'pattern'"):
        build_two_body_walker_constellation(
            epoch=epoch,
            a=7000e3,
            e=0.001,
            i=np.deg2rad(53.0),
            raan=np.deg2rad(20.0),
            argp=np.deg2rad(15.0),
            anomaly=np.deg2rad(10.0),
            anomaly_type="mean",
            total_satellites=6,
            num_planes=3,
            pattern="star",
        )

    with pytest.raises(TypeError, match="integer"):
        build_two_body_walker_constellation(
            epoch=epoch,
            a=7000e3,
            e=0.001,
            i=np.deg2rad(53.0),
            raan=np.deg2rad(20.0),
            argp=np.deg2rad(15.0),
            anomaly=np.deg2rad(10.0),
            anomaly_type="mean",
            total_satellites=True,
            num_planes=3,
        )

    with pytest.raises(TypeError, match="constructor"):
        build_two_body_walker_constellation(
            epoch=epoch,
            a=7000e3,
            e=0.001,
            i=np.deg2rad(53.0),
            raan=np.deg2rad(20.0),
            argp=np.deg2rad(15.0),
            anomaly=np.deg2rad(10.0),
            anomaly_type="mean",
            total_satellites=6,
            num_planes=3,
            constructor="numerical",
        )


def test_walker_clones_seed_numerical_configuration() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    seed = _make_numerical_orbit(
        epoch=epoch,
        a=7100e3,
        e=0.002,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(25.0),
        anomaly=np.deg2rad(5.0),
        anomaly_type="mean",
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=True,
        enable_relativity=True,
        enable_srp=True,
    )

    walker = build_numerical_walker_constellation(
        epoch=epoch,
        a=7100e3,
        e=0.002,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(25.0),
        anomaly=np.deg2rad(5.0),
        anomaly_type="mean",
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=True,
        enable_relativity=True,
        enable_srp=True,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        include_seed=True,
    )
    assert len(walker) == 4

    seed_force_models = _force_model_names(seed)
    assert "SolarRadiationPressure" in seed_force_models
    assert "Relativity" in seed_force_models
    assert all(
        str(o.propagator.__class__.__name__) == "org.orekit.propagation.numerical.NumericalPropagator"
        for o in walker
    )
    assert all(_force_model_names(o) == seed_force_models for o in walker)

    np.testing.assert_allclose(
        walker[0].get_position(0.0, frame="native"),
        seed.get_position(0.0, frame="native"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        walker[0].get_velocity(0.0, frame="native"),
        seed.get_velocity(0.0, frame="native"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        walker[0].get_attitude_quat(0.0, quaternion_convention="scalar_last"),
        seed.get_attitude_quat(0.0, quaternion_convention="scalar_last"),
        atol=1e-12,
    )


def test_build_walker_initial_states_from_orbit_seed() -> None:
    seed = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7050e3,
        e=0.0015,
        i=np.deg2rad(54.0),
        raan=np.deg2rad(12.0),
        argp=np.deg2rad(18.0),
        anomaly=np.deg2rad(6.0),
        anomaly_type="mean",
        mass=275.0,
    )

    walker_states = build_walker_initial_states(
        seed,
        total_satellites=6,
        num_planes=3,
        phasing=1,
        raan_span=math.pi,
        include_seed=True,
    )
    assert len(walker_states) == 6

    seed_state = seed.propagator.getInitialState()
    seed_raan, seed_mean = _state_raan_anomaly(seed_state, anomaly_type="mean")
    seed_quat = _attitude_quaternion(seed_state)
    p = 3
    s = 2
    t = 6

    for idx, state in enumerate(walker_states):
        plane = idx // s
        slot = idx % s
        d_raan = math.pi * (plane / p)
        d_mean = 2.0 * math.pi * (slot / s + plane / t)
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _state_raan_anomaly(state, anomaly_type="mean")
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9
        assert float(state.getMass()) == pytest.approx(275.0)
        assert state.getDate().durationFrom(seed_state.getDate()) == pytest.approx(0.0)

    assert _attitude_quaternion(walker_states[0]) == pytest.approx(seed_quat)


def test_build_walker_initial_states_from_spacecraft_state_seed() -> None:
    seed_orbit = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7050e3,
        e=0.0015,
        i=np.deg2rad(54.0),
        raan=np.deg2rad(12.0),
        argp=np.deg2rad(18.0),
        anomaly=np.deg2rad(6.0),
        anomaly_type="mean",
        mass=275.0,
    )
    seed_state = seed_orbit.propagator.getInitialState()

    walker_states = build_walker_initial_states(
        seed_state,
        total_satellites=6,
        num_planes=3,
        phasing=1,
        raan_span=math.pi,
        include_seed=True,
    )
    assert len(walker_states) == 6

    seed_raan, seed_mean = _state_raan_anomaly(seed_state, anomaly_type="mean")
    seed_quat = _attitude_quaternion(seed_state)
    p = 3
    s = 2
    t = 6

    for idx, state in enumerate(walker_states):
        plane = idx // s
        slot = idx % s
        d_raan = math.pi * (plane / p)
        d_mean = 2.0 * math.pi * (slot / s + plane / t)
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _state_raan_anomaly(state, anomaly_type="mean")
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9
        assert float(state.getMass()) == pytest.approx(275.0)
        assert state.getDate().durationFrom(seed_state.getDate()) == pytest.approx(0.0)

    assert _attitude_quaternion(walker_states[0]) == pytest.approx(seed_quat)


def test_build_walker_constellation_with_spacecraft_state_seed_and_propagator_factory() -> None:
    seed_orbit = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7200e3,
        e=0.002,
        i=np.deg2rad(56.0),
        raan=np.deg2rad(22.0),
        argp=np.deg2rad(11.0),
        anomaly=np.deg2rad(8.0),
        anomaly_type="mean",
        mass=310.0,
    )
    seed_state = seed_orbit.propagator.getInitialState()

    built_states = []

    def propagator_factory(state):
        built_states.append(state)
        return TwoBodyPropagatorFactory()(state)

    walker = build_walker_constellation(
        seed_state,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        propagator_factory=propagator_factory,
    )
    assert len(walker) == 4
    assert len(built_states) == 4

    seed_raan, seed_mean = _state_raan_anomaly(seed_state, anomaly_type="mean")
    for idx, orb in enumerate(walker):
        plane = idx // 2
        slot = idx % 2
        d_raan = 2.0 * math.pi * (plane / 2)
        d_mean = 2.0 * math.pi * (slot / 2 + plane / 4)
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _initial_raan_anomaly(orb, anomaly_type="mean")
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9


def test_build_walker_propagators_returns_raw_propagators() -> None:
    seed = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7150e3,
        e=0.001,
        i=np.deg2rad(54.0),
        raan=np.deg2rad(18.0),
        argp=np.deg2rad(11.0),
        anomaly=np.deg2rad(7.0),
        anomaly_type="mean",
    )

    propagators = build_walker_propagators(
        seed,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        propagator_factory=J2J3J4PropagatorFactory(),
    )

    assert len(propagators) == 4
    assert all(
        str(propagator.__class__.__name__)
        == "org.orekit.propagation.analytical.EcksteinHechlerPropagator"
        for propagator in propagators
    )


def test_j2_j3_j4_walker_constellation_builds_fast_analytical_orbits() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    walker = build_j2_j3_j4_walker_constellation(
        epoch=epoch,
        a=7050e3,
        e=0.001,
        i=np.deg2rad(50.0),
        raan=np.deg2rad(12.0),
        argp=np.deg2rad(9.0),
        anomaly=np.deg2rad(6.0),
        total_satellites=4,
        num_planes=2,
        phasing=1,
    )

    assert len(walker) == 4
    assert all(
        str(orbit.propagator.__class__.__name__)
        == "org.orekit.propagation.analytical.EcksteinHechlerPropagator"
        for orbit in walker
    )


def test_walker_requires_callable_propagator_factory() -> None:
    seed_orbit = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7200e3,
        e=0.002,
        i=np.deg2rad(56.0),
        raan=np.deg2rad(22.0),
        argp=np.deg2rad(11.0),
        anomaly=np.deg2rad(8.0),
        anomaly_type="mean",
        mass=310.0,
    )
    seed_state = seed_orbit.propagator.getInitialState()

    with pytest.raises(TypeError, match="callable"):
        build_walker_constellation(
            seed_state,
            total_satellites=4,
            num_planes=2,
            propagator_factory=object(),
        )


def test_build_walker_constellation_requires_propagator_factory() -> None:
    seed = _make_two_body_orbit(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )
    with pytest.raises(ValueError, match="propagator_factory"):
        build_walker_constellation(
            seed,
            total_satellites=6,
            num_planes=3,
        )


def test_spacecraft_state_from_kepler_is_not_public_anymore() -> None:
    assert "spacecraft_state_from_kepler" not in dir(propagation)
    with pytest.raises(AttributeError):
        _ = propagation.spacecraft_state_from_kepler
