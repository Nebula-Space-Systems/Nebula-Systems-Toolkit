from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.time import Time

import nstk.propagation as propagation
from nstk.propagation import (
    Orbit,
    build_walker_constellation,
    build_walker_initial_states,
)


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


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


def test_orbit_build_walker_constellation_defaults_to_full_raan_span_and_phasing_one() -> None:
    seed = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )

    walker = seed.build_walker_constellation(
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


def test_walker_custom_raan_span_offsets_and_true_anomaly() -> None:
    seed = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7050e3,
        e=0.02,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="true",
    )

    walker = seed.build_walker_constellation(
        total_satellites=6,
        num_planes=3,
        phasing=2,
        raan_span=math.pi,
        initial_raan_offset=np.deg2rad(15.0),
        initial_anomaly_offset=np.deg2rad(7.5),
        anomaly_type="true",
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
    seed = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )

    walker = seed.build_walker_constellation(
        total_satellites=6,
        num_planes=3,
        phasing=0,
        include_seed=False,
    )
    assert len(walker) == 5

    try:
        _ = seed.build_walker_constellation(total_satellites=5, num_planes=3)
        assert False, "expected ValueError for non-divisible walker geometry"
    except ValueError:
        pass

    with pytest.raises(ValueError, match="<= 2\\*pi"):
        seed.build_walker_constellation(
            total_satellites=6,
            num_planes=3,
            raan_span=2.1 * math.pi,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'pattern'"):
        seed.build_walker_constellation(
            total_satellites=6,
            num_planes=3,
            pattern="star",
        )

    with pytest.raises(TypeError, match="integer"):
        seed.build_walker_constellation(
            total_satellites=True,
            num_planes=3,
        )

    with pytest.raises(TypeError, match="constructor"):
        seed.build_walker_constellation(
            total_satellites=6,
            num_planes=3,
            constructor="numerical",
        )


def test_walker_clones_seed_numerical_configuration() -> None:
    seed = Orbit.from_kepler_numerical(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
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
    seed.set_attitude_law("tnw")

    walker = seed.build_walker_constellation(
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
        walker[0].get_p(0.0, frame="native", as_quantity=False),
        seed.get_p(0.0, frame="native", as_quantity=False),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        walker[0].get_v(0.0, frame="native", as_quantity=False),
        seed.get_v(0.0, frame="native", as_quantity=False),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        walker[0].get_attitude(0.0),
        seed.get_attitude(0.0),
        atol=1e-12,
    )


def test_orbit_build_walker_initial_states_convenience_method() -> None:
    seed = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7050e3,
        e=0.0015,
        i=np.deg2rad(54.0),
        raan=np.deg2rad(12.0),
        argp=np.deg2rad(18.0),
        anomaly=np.deg2rad(6.0),
        anomaly_type="mean",
        mass=275.0,
        attitude="tnw",
    )

    walker_states = seed.build_walker_initial_states(
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
    seed_orbit = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7050e3,
        e=0.0015,
        i=np.deg2rad(54.0),
        raan=np.deg2rad(12.0),
        argp=np.deg2rad(18.0),
        anomaly=np.deg2rad(6.0),
        anomaly_type="mean",
        mass=275.0,
        attitude="tnw",
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


def test_build_walker_constellation_with_spacecraft_state_seed_and_orbit_factory() -> None:
    seed_orbit = Orbit.from_kepler_two_body(
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

    def orbit_factory(state) -> Orbit:
        built_states.append(state)
        return Orbit.from_spacecraft_state(state)

    walker = build_walker_constellation(
        seed_state,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        orbit_factory=orbit_factory,
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


def test_walker_requires_callable_orbit_factory() -> None:
    seed_orbit = Orbit.from_kepler_two_body(
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
            orbit_factory=object(),
        )


def test_walker_requires_orbit_factory_for_generic_orbit_wrappers() -> None:
    seed = Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )
    generic_seed = Orbit(seed.propagator, iers=seed.iers, simple_eop=seed.simple_eop)

    with pytest.raises(ValueError, match="orbit_factory"):
        build_walker_constellation(
            generic_seed,
            total_satellites=6,
            num_planes=3,
        )


def test_spacecraft_state_from_kepler_is_not_public_anymore() -> None:
    assert "spacecraft_state_from_kepler" not in dir(propagation)
    with pytest.raises(AttributeError):
        _ = propagation.spacecraft_state_from_kepler
