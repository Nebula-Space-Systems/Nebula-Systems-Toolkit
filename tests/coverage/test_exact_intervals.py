import numpy as np

from nebula.coverage import (
    ExactCoverageConfig,
    build_access_interval_store,
    build_access_interval_store_from_config,
    access_duration_by_target,
    max_asset_by_target,
    mtta_by_target,
)


def _observer_from_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros((z.size, 3), dtype=np.float64)
    out[:, 2] = z
    return out


def test_exact_interval_crossing_and_merge_across_segments() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    observer = _observer_from_z(np.array([-1.0, 1.0, 2.0], dtype=np.float64))

    store = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="linear",
        degrees=True,
    )

    starts, stops = store.pair_intervals(0, 0)
    assert starts.shape == (1,)
    assert stops.shape == (1,)
    np.testing.assert_allclose(starts[0], 0.5, atol=1e-10)
    np.testing.assert_allclose(stops[0], 2.0, atol=1e-10)


def test_duration_and_max_asset_from_stored_intervals() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    obs1 = _observer_from_z(np.array([-1.0, 1.0, 1.0], dtype=np.float64))
    obs2 = _observer_from_z(np.array([-1.0, -1.0, 1.0], dtype=np.float64))

    store = build_access_interval_store(
        time=time,
        observer_positions=[obs1, obs2],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="linear",
        degrees=True,
    )

    dur_n1 = access_duration_by_target(store, N=1, reshape=False)
    dur_n2 = access_duration_by_target(store, N=2, reshape=False)
    mx = max_asset_by_target(store, reshape=False)

    np.testing.assert_allclose(dur_n1[0], 1.5, atol=1e-10)
    np.testing.assert_allclose(dur_n2[0], 0.5, atol=1e-10)
    assert int(mx[0]) == 2


def test_mtta_from_stored_intervals_wrap_and_no_wrap() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    observer = _observer_from_z(np.array([-1.0, 1.0, -1.0], dtype=np.float64))

    store = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="linear",
        degrees=True,
    )

    mtta_no_wrap = mtta_by_target(store, N=1, wrap=False, reshape=False)
    mtta_wrap = mtta_by_target(store, N=1, wrap=True, reshape=False)

    np.testing.assert_allclose(mtta_no_wrap[0], 0.125, atol=1e-10)
    np.testing.assert_allclose(mtta_wrap[0], 0.25, atol=1e-10)


def test_config_wrapper_sets_target_shape_for_grid_queries() -> None:
    cfg = ExactCoverageConfig(
        nlats=2, nlons_equator=3, scale_longitude_by_latitude=False
    )
    time = np.array([0.0, 1.0], dtype=np.float64)
    observer = np.array([[0.0, 0.0, 8_000_000.0], [0.0, 0.0, 8_000_000.0]])

    store = build_access_interval_store_from_config(
        config=cfg,
        time=time,
        observer_positions=[observer],
        interpolation="linear",
    )

    assert store.n_targets == 6
    assert store.target_shape == (2, 3)

    mx_grid = max_asset_by_target(store, reshape=True)
    mx_flat = max_asset_by_target(store, reshape=False)
    assert mx_grid.shape == (2, 3)
    assert mx_flat.shape == (6,)


def test_root_tolerance_controls_transition_time_refinement() -> None:
    time = np.array([0.0, 1.0], dtype=np.float64)
    observer = _observer_from_z(np.array([-1.0, 1.0], dtype=np.float64))

    store_tight = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="linear",
        degrees=True,
        root_tolerance_s=1e-6,
    )
    store_loose = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="linear",
        degrees=True,
        root_tolerance_s=1e-1,
    )

    start_tight = store_tight.start_times[0]
    start_loose = store_loose.start_times[0]

    assert abs(start_tight - 0.5) <= 1e-4
    assert abs(start_loose - 0.5) <= 1e-1


def test_cubic_interpolation_mode_runs_and_stores_metadata() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    observer = _observer_from_z(np.array([-1.0, 1.0, -1.0], dtype=np.float64))

    store = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_up_vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        min_elevation=0.0,
        max_elevation=90.0,
        interpolation="cubic",
        degrees=True,
        root_tolerance_s=1e-3,
    )

    assert store.interpolation == "cubic"
    assert store.root_tolerance_s == 1e-3
    assert store.start_times.size >= 1
    assert np.all(store.stop_times > store.start_times)


def test_exact_config_latitude_scaling_reduces_targets() -> None:
    cfg_uniform = ExactCoverageConfig(
        nlats=61, nlons_equator=121, scale_longitude_by_latitude=False
    )
    cfg_scaled = ExactCoverageConfig(
        nlats=61, nlons_equator=121, scale_longitude_by_latitude=True
    )

    assert cfg_uniform.n_targets == 61 * 121
    assert cfg_uniform.target_shape == (61, 121)

    assert cfg_scaled.n_targets < cfg_uniform.n_targets
    assert cfg_scaled.target_shape is None
    assert int(cfg_scaled.row_sizes[0]) < int(cfg_scaled.row_sizes[30])


def test_exact_config_poles_degenerate_to_single_point() -> None:
    cfg = ExactCoverageConfig(
        nlats=181,
        nlons_equator=361,
        scale_longitude_by_latitude=True,
        include_lat_endpoints=True,
    )

    # First/last latitude rows are poles, so longitude is degenerate.
    assert int(cfg.row_sizes[0]) == 1
    assert int(cfg.row_sizes[-1]) == 1
