from __future__ import annotations

import numpy as np

from nstk.coverage.intervals._exact_intervals import AccessIntervalStore
from nstk.coverage.intervals.metrics import (
    calculate_access_duration,
    calculate_access_separation,
    calculate_gap_duration,
    calculate_max_asset,
    calculate_min_asset,
    calculate_mtta,
    calculate_revisit_time,
)


def _build_store() -> AccessIntervalStore:
    # n_obs=2, n_targets=3, window=[0,10]
    #
    # target 0:
    #   obs0: [1,4], [7,9]
    #   obs1: [3,5], [8,10]
    # target 1:
    #   obs0: none
    #   obs1: [2,6]
    # target 2:
    #   obs0/obs1: none
    pair_offsets = np.array([0, 2, 2, 2, 4, 5, 5], dtype=np.int64)
    start_times = np.array([1.0, 7.0, 3.0, 8.0, 2.0], dtype=np.float64)
    stop_times = np.array([4.0, 9.0, 5.0, 10.0, 6.0], dtype=np.float64)
    return AccessIntervalStore(
        time_start=0.0,
        time_stop=10.0,
        n_observers=2,
        n_targets=3,
        pair_offsets=pair_offsets,
        start_times=start_times,
        stop_times=stop_times,
        min_elevation_rad=0.0,
        max_elevation_rad=0.5 * np.pi,
        interpolation="linear",
        root_tolerance_s=1e-3,
        target_shape=(1, 3),
    )


def test_interval_metrics_duration_and_asset_counts() -> None:
    store = _build_store()

    dur = calculate_access_duration(store, N=[1, 2], reshape=False)
    np.testing.assert_allclose(dur[1], np.array([7.0, 4.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(dur[2], np.array([2.0, 0.0, 0.0]), atol=1e-12)

    mx = calculate_max_asset(store, reshape=False)
    mn = calculate_min_asset(store, reshape=False)
    np.testing.assert_array_equal(mx, np.array([2, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(mn, np.array([0, 0, 0], dtype=np.int32))


def test_interval_metrics_duration_normalization() -> None:
    store = _build_store()

    # Window is 10 s, so per-day factor = 86400 / 10 = 8640.
    dur_seconds_per_day = calculate_access_duration(
        store,
        N=1,
        reshape=False,
        normalize_to_day=True,
    )[1]
    np.testing.assert_allclose(
        dur_seconds_per_day,
        np.array([7.0, 4.0, 0.0]) * 86400.0 / 10.0,
        atol=1e-12,
    )

    dur_seconds_window = calculate_access_duration(
        store,
        N=2,
        reshape=False,
        normalize_to_day=False,
    )[2]
    np.testing.assert_allclose(
        dur_seconds_window, np.array([2.0, 0.0, 0.0]), atol=1e-12
    )


def test_interval_metrics_mtta() -> None:
    store = _build_store()

    mtta_no_wrap = calculate_mtta(store, N=1, wrap=False, reshape=False)
    np.testing.assert_allclose(mtta_no_wrap[:2], np.array([0.25, 1.0]), atol=1e-12)
    assert np.isnan(mtta_no_wrap[2])

    mtta_wrap = calculate_mtta(store, N=1, wrap=True, reshape=False)
    np.testing.assert_allclose(mtta_wrap[:2], np.array([0.25, 1.8]), atol=1e-12)
    assert np.isnan(mtta_wrap[2])


def test_interval_metrics_gap_duration_stats() -> None:
    store = _build_store()

    g_mean = calculate_gap_duration(
        store, min_assets=1, stat="mean", include_end_gaps=True, reshape=False
    )
    np.testing.assert_allclose(g_mean, np.array([1.5, 3.0, 10.0]), atol=1e-12)

    g_count = calculate_gap_duration(
        store, min_assets=1, stat="count", include_end_gaps=True, reshape=False
    )
    np.testing.assert_array_equal(g_count, np.array([2, 2, 1], dtype=np.int32))

    g_std = calculate_gap_duration(
        store, min_assets=1, stat="std", include_end_gaps=True, reshape=False
    )
    np.testing.assert_allclose(g_std, np.array([0.5, 1.0, 0.0]), atol=1e-12)

    g_ignore = calculate_gap_duration(
        store, min_assets=1, stat="mean", include_end_gaps=False, reshape=False
    )
    np.testing.assert_allclose(g_ignore[0], 2.0, atol=1e-12)
    assert np.isnan(g_ignore[1])
    assert np.isnan(g_ignore[2])

    g_nan_no_access = calculate_gap_duration(
        store,
        min_assets=1,
        stat="mean",
        include_end_gaps=True,
        nan_if_never_access=True,
        reshape=False,
    )
    np.testing.assert_allclose(g_nan_no_access[:2], np.array([1.5, 3.0]), atol=1e-12)
    assert np.isnan(g_nan_no_access[2])


def test_interval_metrics_revisit_time() -> None:
    store = _build_store()

    rv_inc = calculate_revisit_time(
        store, N=1, option="average", end_gaps="include", reshape=False
    )
    np.testing.assert_allclose(rv_inc, np.array([1.5, 3.0, 10.0]), atol=1e-12)

    rv_ign = calculate_revisit_time(
        store, N=1, option="average", end_gaps="ignore", reshape=False
    )
    np.testing.assert_allclose(rv_ign, np.array([2.0, 0.0, 10.0]), atol=1e-12)


def test_interval_metrics_access_separation() -> None:
    store = _build_store()

    sep = calculate_access_separation(
        store,
        min_assets=1,
        min_separation_s=1.0,
        max_separation_s=3.0,
        no_access_value=np.nan,
        reshape=False,
    )
    np.testing.assert_allclose(sep[:2], np.array([1.0, 0.0]), atol=1e-12)
    assert np.isnan(sep[2])

    sep_u8 = calculate_access_separation(
        store,
        min_assets=1,
        min_separation_s=1.0,
        max_separation_s=3.0,
        no_access_value=None,
        reshape=False,
    )
    np.testing.assert_array_equal(sep_u8, np.array([1, 0, 0], dtype=np.uint8))

    sep_n2 = calculate_access_separation(
        store,
        min_assets=2,
        min_separation_s=1.0,
        max_separation_s=3.0,
        no_access_value=None,
        reshape=False,
    )
    np.testing.assert_array_equal(sep_n2, np.array([0, 0, 0], dtype=np.uint8))


def test_interval_metrics_reshape_behavior() -> None:
    store = _build_store()
    mx = calculate_max_asset(store, reshape=True)
    assert mx.shape == (1, 3)
    np.testing.assert_array_equal(mx[0], np.array([2, 1, 0], dtype=np.int32))
