from __future__ import annotations

import math

import numpy as np
from astropy.time import Time

from nebula.propagation import Orbit, build_walker_constellation


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _ang_diff(a: float, b: float) -> float:
    return abs(_wrap_pm_pi(float(a) - float(b)))


def _initial_raan_mean_anomaly(orbit: Orbit) -> tuple[float, float]:
    from org.orekit.orbits import KeplerianOrbit  # type: ignore

    state0 = orbit.propagator.getInitialState()
    kep = KeplerianOrbit(state0.getOrbit())
    return float(kep.getRightAscensionOfAscendingNode()), float(kep.getMeanAnomaly())


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


def test_walker_numerical_auto_constructor() -> None:
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
        enable_third_body=False,
        enable_srp=False,
    )

    walker = build_walker_constellation(
        seed,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        include_seed=True,
    )
    assert len(walker) == 4
    assert all(
        str(o.propagator.__class__.__name__) == "org.orekit.propagation.numerical.NumericalPropagator"
        for o in walker
    )
