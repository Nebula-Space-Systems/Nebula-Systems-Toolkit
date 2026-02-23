from __future__ import annotations

import math

import numpy as np
from astropy.time import Time

from nebula.propagation import Orbit, build_walker_constellation


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _ang_diff(a: float, b: float) -> float:
    return abs(_wrap_pm_pi(float(a) - float(b)))


def test_walker_fast_delta_phasing() -> None:
    seed = Orbit.from_kepler_fast(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=3,
        dt_save_s=30.0,
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
    assert all(o.is_efficiency for o in walker)

    impl0 = seed._fast_impl  # type: ignore[attr-defined]
    seed_raan = float(impl0.raan_rad)
    seed_mean = float(impl0.M0_rad)
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

        impl = orb._fast_impl  # type: ignore[attr-defined]
        assert _ang_diff(float(impl.raan_rad), exp_raan) < 1e-12
        assert _ang_diff(float(impl.M0_rad), exp_mean) < 1e-12


def test_walker_include_seed_false_and_validation() -> None:
    seed = Orbit.from_kepler_fast(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
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


def test_walker_precision_mode_and_epoch_consistency() -> None:
    if hasattr(Orbit, "from_kepler_precision"):
        ctor = Orbit.from_kepler_precision  # type: ignore[attr-defined]
    elif hasattr(Orbit, "from_kepler_precise"):
        ctor = Orbit.from_kepler_precise  # type: ignore[attr-defined]
    else:
        ctor = Orbit.from_kepler

    seed = ctor(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7100e3,
        e=0.002,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(25.0),
        anomaly=np.deg2rad(5.0),
        anomaly_type="mean",
        gravity_model="newtonian",
        dt_save_s=60.0,
    )

    walker = build_walker_constellation(
        seed,
        total_satellites=4,
        num_planes=2,
        phasing=1,
        include_seed=True,
    )
    assert len(walker) == 4
    assert all(o.is_precision for o in walker)

    r_seed, v_seed = seed.pv(seed.epoch, frame="native")
    r0, v0 = walker[0].pv(walker[0].epoch, frame="native")
    np.testing.assert_allclose(r0, r_seed, atol=1e-3, rtol=0.0)
    np.testing.assert_allclose(v0, v_seed, atol=1e-6, rtol=0.0)
