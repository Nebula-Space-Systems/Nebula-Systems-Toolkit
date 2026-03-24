from __future__ import annotations

import numpy as np
from shapely.geometry import box

from nstk.coverage import (
    BBoxDomain,
    CoverageTargets,
    EqualAreaSampler,
    LatitudeAdaptiveSampler,
    LatitudeLongitudeSampler,
    PolygonDomain,
)


def test_equal_area_sampler_materializes_notebook_composite_domain() -> None:
    composite_domain = (
        BBoxDomain(west_deg=-125.0, east_deg=-65.0, south_deg=24.0, north_deg=50.0)
        & PolygonDomain(geometry=box(-105.0, 28.0, -75.0, 48.0), name="Box AOI")
    )

    targets = CoverageTargets.from_domain(
        composite_domain,
        sampler=EqualAreaSampler(target_count=256),
    )

    assert targets.n_targets == 256
    assert np.all((targets.lat_deg >= 28.0) & (targets.lat_deg <= 48.0))
    assert np.all((targets.lon_deg >= -105.0) & (targets.lon_deg <= -75.0))
    assert np.count_nonzero((targets.lat_deg < 38.0) & (targets.lon_deg < -90.0)) > 0
    assert np.count_nonzero((targets.lat_deg < 38.0) & (targets.lon_deg >= -90.0)) > 0
    assert np.count_nonzero((targets.lat_deg >= 38.0) & (targets.lon_deg < -90.0)) > 0
    assert np.count_nonzero((targets.lat_deg >= 38.0) & (targets.lon_deg >= -90.0)) > 0
    np.testing.assert_allclose(targets.area_weights.sum(), 1.0, atol=1e-12)


def test_equal_area_sampler_spreads_targets_across_small_composite_domain() -> None:
    composite_domain = (
        BBoxDomain(west_deg=-10.0, east_deg=10.0, south_deg=-5.0, north_deg=5.0)
        & PolygonDomain(geometry=box(-8.0, -4.0, 8.0, 4.0), name="Inner Box")
    )

    targets = CoverageTargets.from_domain(
        composite_domain,
        sampler=EqualAreaSampler(target_count=32),
    )

    assert targets.n_targets == 32
    assert float(targets.lat_deg.min()) < -3.0
    assert float(targets.lat_deg.max()) > 3.0
    assert float(targets.lon_deg.min()) < -6.0
    assert float(targets.lon_deg.max()) > 6.0
    assert np.count_nonzero((targets.lat_deg < 0.0) & (targets.lon_deg < 0.0)) > 0
    assert np.count_nonzero((targets.lat_deg < 0.0) & (targets.lon_deg >= 0.0)) > 0
    assert np.count_nonzero((targets.lat_deg >= 0.0) & (targets.lon_deg < 0.0)) > 0
    assert np.count_nonzero((targets.lat_deg >= 0.0) & (targets.lon_deg >= 0.0)) > 0


def test_target_domain_add_alias_materializes_union_region() -> None:
    west_box = BBoxDomain(west_deg=-20.0, east_deg=-10.0, south_deg=-5.0, north_deg=5.0)
    east_box = BBoxDomain(west_deg=10.0, east_deg=20.0, south_deg=-5.0, north_deg=5.0)

    targets = CoverageTargets.from_domain(
        west_box + east_box,
        sampler=EqualAreaSampler(target_count=64),
    )

    assert targets.n_targets == 64
    assert np.all(
        west_box.contains_latlon(targets.lat_deg, targets.lon_deg)
        | east_box.contains_latlon(targets.lat_deg, targets.lon_deg)
    )
    assert np.count_nonzero(west_box.contains_latlon(targets.lat_deg, targets.lon_deg)) > 0
    assert np.count_nonzero(east_box.contains_latlon(targets.lat_deg, targets.lon_deg)) > 0


def test_coverage_targets_union_accepts_existing_target_sets() -> None:
    west_targets = CoverageTargets.region_bbox(
        west_deg=-20.0,
        east_deg=-10.0,
        south_deg=-5.0,
        north_deg=5.0,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=5),
    )
    east_targets = CoverageTargets.region_bbox(
        west_deg=10.0,
        east_deg=20.0,
        south_deg=-5.0,
        north_deg=5.0,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=5),
    )
    west_box = BBoxDomain(west_deg=-20.0, east_deg=-10.0, south_deg=-5.0, north_deg=5.0)
    east_box = BBoxDomain(west_deg=10.0, east_deg=20.0, south_deg=-5.0, north_deg=5.0)

    targets = CoverageTargets.union(
        west_targets,
        east_targets,
        sampler=EqualAreaSampler(target_count=48),
        name="Two Boxes",
    )

    assert targets.n_targets == 48
    assert targets.attrs["domain"] == "Two Boxes"
    assert np.all(
        west_box.contains_latlon(targets.lat_deg, targets.lon_deg)
        | east_box.contains_latlon(targets.lat_deg, targets.lon_deg)
    )
    assert np.count_nonzero(west_box.contains_latlon(targets.lat_deg, targets.lon_deg)) > 0
    assert np.count_nonzero(east_box.contains_latlon(targets.lat_deg, targets.lon_deg)) > 0


def test_coverage_targets_extent_uses_boundary_geometry_and_padding() -> None:
    targets = CoverageTargets.region_bbox(
        west_deg=-20.0,
        east_deg=20.0,
        south_deg=-10.0,
        north_deg=10.0,
        sampler=EqualAreaSampler(target_count=64),
    )

    assert targets.extent() == (-20.0, 20.0, -10.0, 10.0)
    assert targets.extent(pad_deg=2.0) == (-22.0, 22.0, -12.0, 12.0)
    assert targets.extent(pad_lon_deg=5.0, pad_lat_deg=1.0) == (-25.0, 25.0, -11.0, 11.0)


def test_coverage_targets_extent_falls_back_to_sampled_points_and_clips_global_bounds() -> None:
    manual_targets = CoverageTargets(
        positions_ecef_m=np.zeros((2, 3), dtype=np.float64),
        up_vectors_ecef=np.zeros((2, 3), dtype=np.float64),
        lat_deg=np.asarray([-89.0, 5.0], dtype=np.float64),
        lon_deg=np.asarray([-170.0, 170.0], dtype=np.float64),
        area_weights=np.ones(2, dtype=np.float64),
        boundary_geometry=None,
    )

    assert manual_targets.extent() == (-170.0, 170.0, -89.0, 5.0)
    assert manual_targets.extent(pad_deg=20.0) == (-180.0, 180.0, -90.0, 25.0)


def test_special_domains_use_separate_outline_geometry_when_needed() -> None:
    global_targets = CoverageTargets.global_earth(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
    )
    land_targets = CoverageTargets.land(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
        resolution="110m",
    )
    ocean_targets = CoverageTargets.ocean(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
        resolution="110m",
    )

    assert global_targets.outline_geometry is None
    assert global_targets.boundary_geometry is not None
    assert land_targets.boundary_geometry is not None
    assert ocean_targets.boundary_geometry is not None
    assert getattr(land_targets.outline_geometry, "geom_type", "") in {"LineString", "MultiLineString"}
    assert getattr(ocean_targets.outline_geometry, "geom_type", "") in {"LineString", "MultiLineString"}


def test_latitude_adaptive_sampler_scales_longitude_density_by_latitude() -> None:
    targets = CoverageTargets.global_earth(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
    )

    assert targets.surface_grid is None
    assert np.all(targets.area_weights > 0.0)

    unique_lat, counts = np.unique(targets.lat_deg, return_counts=True)
    row_counts = {float(lat): int(count) for lat, count in zip(unique_lat, counts)}
    assert row_counts[-90.0] == 1
    assert row_counts[0.0] == 12
    assert row_counts[60.0] < row_counts[30.0] < row_counts[0.0]
