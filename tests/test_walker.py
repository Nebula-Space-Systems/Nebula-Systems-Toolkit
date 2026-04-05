from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.time import Time

from nstk.propagation import (
    Orbit,
    build_walker_constellation,
    build_walker_initial_states,
    spacecraft_state_from_kepler,
)


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _ang_diff(a: float, b: float) -> float:
    return abs(_wrap_pm_pi(float(a) - float(b)))


def _state_raan_mean_anomaly(state) -> tuple[float, float]:
    from org.orekit.orbits import KeplerianOrbit  # type: ignore

    kep = KeplerianOrbit(state.getOrbit())
    return float(kep.getRightAscensionOfAscendingNode()), float(kep.getMeanAnomaly())


def _initial_raan_mean_anomaly(orbit: Orbit) -> tuple[float, float]:
    return _state_raan_mean_anomaly(orbit.propagator.getInitialState())


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


def test_walker_two_body_delta_phasing() -> None:
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

    walker = build_walker_constellation(
        seed,
        total_satellites=6,
        num_planes=3,
        phasing=1,
        pattern="delta",
        include_seed=True,
    )
    assert len(walker) == 6

    seed_raan, seed_mean = _initial_raan_mean_anomaly(seed)
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

        raan, mean = _initial_raan_mean_anomaly(orb)
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9


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

    walker = build_walker_constellation(
        seed,
        total_satellites=6,
        num_planes=3,
        phasing=0,
        include_seed=False,
    )
    assert len(walker) == 5

    try:
        _ = build_walker_constellation(seed, total_satellites=5, num_planes=3)
        assert False, "expected ValueError for non-divisible walker geometry"
    except ValueError:
        pass

    with pytest.raises(TypeError, match="constructor"):
        build_walker_constellation(
            seed,
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

    walker = build_walker_constellation(
        seed,
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


def test_build_walker_initial_states_from_spacecraft_state_seed() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    seed_state = spacecraft_state_from_kepler(
        epoch=epoch,
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

    walker_states = build_walker_initial_states(
        seed_state,
        total_satellites=6,
        num_planes=3,
        phasing=1,
        include_seed=True,
    )
    assert len(walker_states) == 6

    seed_raan, seed_mean = _state_raan_mean_anomaly(seed_state)
    seed_quat = _attitude_quaternion(seed_state)
    p = 3
    s = 2
    t = 6

    for idx, state in enumerate(walker_states):
        plane = idx // s
        slot = idx % s
        d_raan = 2.0 * math.pi * (plane / p)
        d_mean = 2.0 * math.pi * (slot / s + plane / t)
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _state_raan_mean_anomaly(state)
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9
        assert float(state.getMass()) == pytest.approx(275.0)
        assert state.getDate().durationFrom(seed_state.getDate()) == pytest.approx(0.0)

    assert _attitude_quaternion(walker_states[0]) == pytest.approx(seed_quat)


def test_build_walker_constellation_with_spacecraft_state_seed_and_orbit_factory() -> None:
    seed_state = spacecraft_state_from_kepler(
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

    seed_raan, seed_mean = _state_raan_mean_anomaly(seed_state)
    for idx, orb in enumerate(walker):
        plane = idx // 2
        slot = idx % 2
        d_raan = 2.0 * math.pi * (plane / 2)
        d_mean = 2.0 * math.pi * (slot / 2 + plane / 4)
        exp_raan = _wrap_pm_pi(seed_raan + d_raan)
        exp_mean = _wrap_pm_pi(seed_mean + d_mean)

        raan, mean = _initial_raan_mean_anomaly(orb)
        assert _ang_diff(raan, exp_raan) < 1e-9
        assert _ang_diff(mean, exp_mean) < 1e-9


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
