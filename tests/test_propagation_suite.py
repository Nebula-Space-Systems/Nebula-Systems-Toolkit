from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.propagation import (
    build_fast,
    build_orekit_j2_only,
    build_orekit_newtonian,
    pv_error_stats,
    seconds_grid,
)


@pytest.mark.slow
def test_ephemeris_dt_resolution_consistency(epoch_utc, leo_case) -> None:
    # Explicit seconds (not day-based float offsets) for astropy Time.
    samples = seconds_grid(epoch_utc, duration_s=24.0 * 3600.0, n_samples=721)

    e_fast = build_orekit_newtonian(epoch=epoch_utc, dt_save_s=10.0, **leo_case)
    e_slow = build_orekit_newtonian(epoch=epoch_utc, dt_save_s=60.0, **leo_case)

    r_fast, v_fast = e_fast.pv(samples, frame="itrf")
    r_slow, v_slow = e_slow.pv(samples, frame="itrf")
    stats = pv_error_stats(r_fast, v_fast, r_slow, v_slow)

    assert stats.pos_max_m < 5.0
    assert stats.vel_max_mps < 0.15


@pytest.mark.slow
def test_fast_ephemeris_matches_orekit_newtonian(epoch_utc, leo_case) -> None:
    samples = seconds_grid(epoch_utc, duration_s=24.0 * 3600.0, n_samples=721)

    ore = build_orekit_newtonian(epoch=epoch_utc, dt_save_s=10.0, **leo_case)
    fast = build_fast(
        epoch=epoch_utc,
        dt_save_s=10.0,
        enable_j2=False,
        **leo_case,
    )

    r_ore, v_ore = ore.pv(samples, frame="itrf")
    r_fast, v_fast = fast.pv(samples, frame="itrf")
    stats = pv_error_stats(r_ore, v_ore, r_fast, v_fast)

    assert stats.pos_max_m < 120.0
    assert stats.vel_max_mps < 0.15


@pytest.mark.slow
def test_fast_j2_osculating_vs_orekit_j2_only(epoch_utc, leo_case) -> None:
    samples = seconds_grid(epoch_utc, duration_s=24.0 * 3600.0, n_samples=181)

    ore_j2 = build_orekit_j2_only(epoch=epoch_utc, dt_save_s=20.0, **leo_case)
    fast_osc = build_fast(
        epoch=epoch_utc,
        dt_save_s=20.0,
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=4,
        **leo_case,
    )

    r_ore, v_ore = ore_j2.pv(samples, frame="native")
    r_osc, v_osc = fast_osc.pv(samples, frame="native")
    stats = pv_error_stats(r_ore, v_ore, r_osc, v_osc)

    assert stats.pos_max_m < 5.0
    assert stats.vel_max_mps < 0.01


@pytest.mark.slow
def test_fast_j2_secular_is_bounded_and_worse_than_osculating(epoch_utc, leo_case) -> None:
    samples = seconds_grid(epoch_utc, duration_s=24.0 * 3600.0, n_samples=181)

    ore_j2 = build_orekit_j2_only(epoch=epoch_utc, dt_save_s=20.0, **leo_case)
    fast_sec = build_fast(
        epoch=epoch_utc,
        dt_save_s=20.0,
        enable_j2=True,
        j2_mode="secular",
        j2_substeps=4,
        **leo_case,
    )
    fast_osc = build_fast(
        epoch=epoch_utc,
        dt_save_s=20.0,
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=4,
        **leo_case,
    )

    r_ore, v_ore = ore_j2.pv(samples, frame="native")
    r_sec, v_sec = fast_sec.pv(samples, frame="native")
    r_osc, v_osc = fast_osc.pv(samples, frame="native")

    sec = pv_error_stats(r_ore, v_ore, r_sec, v_sec)
    osc = pv_error_stats(r_ore, v_ore, r_osc, v_osc)

    assert sec.pos_max_m < 500e3
    assert sec.vel_max_mps < 1000.0
    assert osc.pos_mean_m < 0.05 * sec.pos_mean_m
    assert osc.vel_mean_mps < 0.05 * sec.vel_mean_mps
