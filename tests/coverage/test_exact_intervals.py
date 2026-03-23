import numpy as np

from nstk.coverage import (
    ExactCoverageConfig,
    build_access_interval_store,
    compute_access_intervals,
    access_duration_by_target,
    max_asset_by_target,
    mtta_by_target,
)
from nstk.coverage.intervals._exact_intervals import build_surface_targets_from_config
from nstk.transforms.constants import WGS84_A, WGS84_E2


def _observer_from_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros((z.size, 3), dtype=np.float64)
    out[:, 2] = z
    return out


def _target_from_latlon(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(float(lat_deg))
    lon_rad = np.deg2rad(float(lon_deg))
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    prime_vertical_radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    position = np.array(
        [
            prime_vertical_radius * cos_lat * cos_lon,
            prime_vertical_radius * cos_lat * sin_lon,
            (1.0 - WGS84_E2) * prime_vertical_radius * sin_lat,
        ],
        dtype=np.float64,
    )
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return position, up


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

    store = compute_access_intervals(
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


def test_build_surface_targets_from_config_matches_geodetic_geometry() -> None:
    cfg = ExactCoverageConfig(
        nlats=181,
        nlons_equator=361,
        scale_longitude_by_latitude=True,
        include_lat_endpoints=True,
        include_lon_endpoints=False,
    )

    target_positions, target_up = build_surface_targets_from_config(cfg)

    for lat_deg, lon_deg in [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (80.0, 0.0)]:
        idx = int(
            np.argmin(
                (cfg.lat_deg_flat - lat_deg) ** 2
                + (((cfg.lon_deg_flat - lon_deg + 180.0) % 360.0) - 180.0) ** 2
            )
        )
        expected_position, expected_up = _target_from_latlon(
            cfg.lat_deg_flat[idx],
            cfg.lon_deg_flat[idx],
        )
        np.testing.assert_allclose(target_positions[idx], expected_position, atol=1e-9)
        np.testing.assert_allclose(target_up[idx], expected_up, atol=1e-12)


def test_compute_access_intervals_matches_manual_target_build_for_selected_points() -> None:
    cfg = ExactCoverageConfig(
        nlats=181,
        nlons_equator=361,
        scale_longitude_by_latitude=True,
        include_lat_endpoints=True,
        include_lon_endpoints=False,
    )
    time = np.linspace(0.0, 1800.0, 7, dtype=np.float64)
    observer = np.tile(np.array([[8_000_000.0, 0.0, 0.0]], dtype=np.float64), (time.size, 1))

    config_store = compute_access_intervals(
        config=cfg,
        time=time,
        observer_positions=[observer],
        interpolation="linear",
    )

    sample_points = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (80.0, 0.0)]
    sample_indices = [config_store.nearest_target_index(lat_deg=lat, lon_deg=lon) for lat, lon in sample_points]

    manual_positions = np.empty((len(sample_indices), 3), dtype=np.float64)
    manual_up = np.empty((len(sample_indices), 3), dtype=np.float64)
    for out_idx, sample_idx in enumerate(sample_indices):
        position, up = _target_from_latlon(
            cfg.lat_deg_flat[sample_idx],
            cfg.lon_deg_flat[sample_idx],
        )
        manual_positions[out_idx] = position
        manual_up[out_idx] = up

    manual_store = build_access_interval_store(
        time=time,
        observer_positions=[observer],
        target_positions=manual_positions,
        target_up_vectors=manual_up,
        min_elevation=cfg.min_elevation_deg,
        max_elevation=cfg.max_elevation_deg,
        degrees=True,
        interpolation="linear",
    )

    config_duration = access_duration_by_target(config_store, reshape=False)[sample_indices]
    manual_duration = access_duration_by_target(manual_store, reshape=False)
    np.testing.assert_allclose(config_duration, manual_duration, atol=1e-10)


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
