from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tests.helpers.propagation import (
    build_fast,
    build_orekit_j2_only,
    build_orekit_newtonian,
    pv_error_stats,
    seconds_grid,
)


MU_EARTH_M3_S2 = 3.986004418e14


@dataclass(frozen=True)
class RegimeCase:
    name: str
    a_m: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    anomaly_rad: float

    def as_kwargs(self) -> dict[str, float]:
        return {
            "a_m": self.a_m,
            "e": self.e,
            "i_rad": self.i_rad,
            "raan_rad": self.raan_rad,
            "argp_rad": self.argp_rad,
            "anomaly_rad": self.anomaly_rad,
        }


ORBITAL_REGIMES: tuple[RegimeCase, ...] = (
    RegimeCase(
        name="leo_midinc",
        a_m=6_778_000.0,
        e=5e-4,
        i_rad=float(np.deg2rad(51.6)),
        raan_rad=float(np.deg2rad(20.0)),
        argp_rad=float(np.deg2rad(15.0)),
        anomaly_rad=float(np.deg2rad(10.0)),
    ),
    RegimeCase(
        name="leo_sso",
        a_m=7_078_000.0,
        e=1e-3,
        i_rad=float(np.deg2rad(98.6)),
        raan_rad=float(np.deg2rad(120.0)),
        argp_rad=float(np.deg2rad(87.0)),
        anomaly_rad=float(np.deg2rad(250.0)),
    ),
    RegimeCase(
        name="meo_gnss",
        a_m=26_560_000.0,
        e=0.01,
        i_rad=float(np.deg2rad(55.0)),
        raan_rad=float(np.deg2rad(80.0)),
        argp_rad=float(np.deg2rad(270.0)),
        anomaly_rad=float(np.deg2rad(40.0)),
    ),
    RegimeCase(
        name="geo_near_circular",
        a_m=42_164_000.0,
        e=2e-4,
        i_rad=float(np.deg2rad(0.1)),
        raan_rad=float(np.deg2rad(200.0)),
        argp_rad=float(np.deg2rad(15.0)),
        anomaly_rad=float(np.deg2rad(180.0)),
    ),
    RegimeCase(
        name="gto_high_e",
        a_m=24_361_000.0,
        e=0.73,
        i_rad=float(np.deg2rad(27.0)),
        raan_rad=float(np.deg2rad(45.0)),
        argp_rad=float(np.deg2rad(180.0)),
        anomaly_rad=float(np.deg2rad(0.0)),
    ),
    RegimeCase(
        name="molniya",
        a_m=26_600_000.0,
        e=0.74,
        i_rad=float(np.deg2rad(63.4)),
        raan_rad=float(np.deg2rad(250.0)),
        argp_rad=float(np.deg2rad(270.0)),
        anomaly_rad=float(np.deg2rad(30.0)),
    ),
    RegimeCase(
        name="retrograde_leo",
        a_m=6_978_000.0,
        e=0.01,
        i_rad=float(np.deg2rad(140.0)),
        raan_rad=float(np.deg2rad(300.0)),
        argp_rad=float(np.deg2rad(20.0)),
        anomaly_rad=float(np.deg2rad(120.0)),
    ),
)


def _orbital_period_s(a_m: float) -> float:
    return float(2.0 * np.pi * np.sqrt((a_m**3) / MU_EARTH_M3_S2))


def _regime_duration_s(a_m: float) -> float:
    # Cover at least a meaningful arc in every regime while keeping runtime bounded.
    return float(min(12.0 * 3600.0, max(2.0 * 3600.0, 1.25 * _orbital_period_s(a_m))))


def _invariant_relative_spreads(r_m: np.ndarray, v_mps: np.ndarray) -> tuple[float, float]:
    radius = np.linalg.norm(r_m, axis=1)
    speed2 = np.sum(v_mps * v_mps, axis=1)

    specific_energy = 0.5 * speed2 - MU_EARTH_M3_S2 / radius
    h_norm = np.linalg.norm(np.cross(r_m, v_mps), axis=1)

    rel_energy = float(
        (np.max(specific_energy) - np.min(specific_energy))
        / max(1.0, abs(float(np.mean(specific_energy))))
    )
    rel_h = float((np.max(h_norm) - np.min(h_norm)) / max(1.0, abs(float(np.mean(h_norm)))))
    return rel_energy, rel_h


@pytest.mark.slow
@pytest.mark.parametrize("regime", ORBITAL_REGIMES, ids=lambda case: case.name)
def test_two_body_invariants_hold_for_orbit_and_fastorbit(epoch_utc, regime: RegimeCase) -> None:
    duration_s = _regime_duration_s(regime.a_m)
    samples = seconds_grid(epoch_utc, duration_s=duration_s, n_samples=121)

    ore = build_orekit_newtonian(epoch=epoch_utc, dt_save_s=20.0, **regime.as_kwargs())
    fast = build_fast(
        epoch=epoch_utc,
        dt_save_s=20.0,
        enable_j2=False,
        **regime.as_kwargs(),
    )

    for ephemeris in (ore, fast):
        r_native, v_native = ephemeris.pv(samples, frame="native")
        assert np.isfinite(r_native).all()
        assert np.isfinite(v_native).all()

        rel_energy, rel_h = _invariant_relative_spreads(r_native, v_native)
        assert rel_energy < 1e-5
        assert rel_h < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("regime", ORBITAL_REGIMES, ids=lambda case: case.name)
def test_orbit_and_fastorbit_two_body_agree_across_regimes(epoch_utc, regime: RegimeCase) -> None:
    ore = build_orekit_newtonian(epoch=epoch_utc, dt_save_s=10.0, **regime.as_kwargs())
    fast = build_fast(
        epoch=epoch_utc,
        dt_save_s=10.0,
        enable_j2=False,
        **regime.as_kwargs(),
    )

    r0_ore, v0_ore = ore.pv(epoch_utc, frame="native")
    r0_fast, v0_fast = fast.pv(epoch_utc, frame="native")
    assert float(np.linalg.norm(r0_fast - r0_ore)) < 1e-3
    assert float(np.linalg.norm(v0_fast - v0_ore)) < 1e-6

    duration_s = _regime_duration_s(regime.a_m)
    samples = seconds_grid(epoch_utc, duration_s=duration_s, n_samples=161)

    r_ore, v_ore = ore.pv(samples, frame="native")
    r_fast, v_fast = fast.pv(samples, frame="native")
    stats = pv_error_stats(r_ore, v_ore, r_fast, v_fast)

    assert stats.pos_max_m < 0.05
    assert stats.vel_max_mps < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("regime", ORBITAL_REGIMES, ids=lambda case: case.name)
def test_fast_j2_osculating_matches_orekit_j2_only_across_regimes(
    epoch_utc, regime: RegimeCase
) -> None:
    ore_j2 = build_orekit_j2_only(epoch=epoch_utc, dt_save_s=20.0, **regime.as_kwargs())
    fast_osc = build_fast(
        epoch=epoch_utc,
        dt_save_s=20.0,
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=4,
        **regime.as_kwargs(),
    )

    duration_s = _regime_duration_s(regime.a_m)
    samples = seconds_grid(epoch_utc, duration_s=duration_s, n_samples=121)

    r_ore, v_ore = ore_j2.pv(samples, frame="native")
    r_osc, v_osc = fast_osc.pv(samples, frame="native")
    stats = pv_error_stats(r_ore, v_ore, r_osc, v_osc)

    assert stats.pos_max_m < 35.0
    assert stats.vel_max_mps < 0.03
