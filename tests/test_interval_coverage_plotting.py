from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.figure import Figure

cartopy_geoaxes = pytest.importorskip("cartopy.mpl.geoaxes")

from nstk.coverage import (
    BBoxDomain,
    CoverageArray,
    CoverageField,
    CoverageTargets,
    CoverageTimeline,
    IntervalCoverage,
    LatitudeAdaptiveSampler,
    LatitudeLongitudeSampler,
    Observer,
)
from nstk.plotting import (
    LIGHT_DETAILED,
    plot_coverage_ecdf,
    plot_coverage_histogram,
    plot_coverage_map,
    plot_coverage_small_multiples,
    plot_target_timeline,
)
from nstk.plotting.map import ExtentConfig


def _coverage() -> IntervalCoverage:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 1200.0, 13))
    targets = CoverageTargets.from_domain(
        BBoxDomain(west_deg=-20.0, east_deg=20.0, south_deg=-10.0, north_deg=10.0),
        sampler=LatitudeLongitudeSampler(nlats=7, nlons=9),
    )
    obs0 = np.tile(np.array([[7_000_000.0, 0.0, 0.0]], dtype=np.float64), (timeline.seconds.size, 1))
    obs1 = np.tile(np.array([[0.0, 7_000_000.0, 0.0]], dtype=np.float64), (timeline.seconds.size, 1))
    return IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_samples(obs0, name="A"),
            Observer.from_samples(obs1, name="B"),
        ],
        targets=targets,
        interpolation="linear",
    )


def _dark_outline_fraction(
    fig: Figure,
    ax: cartopy_geoaxes.GeoAxes,
    *,
    lon_deg: float,
    lat_deg: float,
    radius_px: int = 8,
) -> float:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    x_disp, y_disp = transform.transform((float(lon_deg), float(lat_deg)))
    x_px, y_px = map(int, np.round((x_disp, y_disp)))
    y_img = rgba.shape[0] - 1 - y_px
    x0 = max(0, x_px - int(radius_px))
    x1 = min(rgba.shape[1], x_px + int(radius_px) + 1)
    y0 = max(0, y_img - int(radius_px))
    y1 = min(rgba.shape[0], y_img + int(radius_px) + 1)
    patch = rgba[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return 0.0
    luminance = np.mean(patch, axis=2)
    return float(np.mean(luminance < 70.0))


def _patch_saturation(
    fig: Figure,
    ax: cartopy_geoaxes.GeoAxes,
    *,
    lon_deg: float,
    lat_deg: float,
    radius_px: int = 4,
) -> float:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    x_disp, y_disp = transform.transform((float(lon_deg), float(lat_deg)))
    x_px, y_px = map(int, np.round((x_disp, y_disp)))
    y_img = rgba.shape[0] - 1 - y_px
    x0 = max(0, x_px - int(radius_px))
    x1 = min(rgba.shape[1], x_px + int(radius_px) + 1)
    y0 = max(0, y_img - int(radius_px))
    y1 = min(rgba.shape[0], y_img + int(radius_px) + 1)
    patch = rgba[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(np.max(patch, axis=2) - np.min(patch, axis=2)))


def _assert_global_coastline_overlay(fig: Figure, ax: cartopy_geoaxes.GeoAxes) -> None:
    assert _dark_outline_fraction(fig, ax, lon_deg=12.0, lat_deg=44.0) > 0.05
    assert _dark_outline_fraction(fig, ax, lon_deg=-74.0, lat_deg=40.0) > 0.05
    assert _dark_outline_fraction(fig, ax, lon_deg=140.0, lat_deg=36.0) > 0.05
    assert _dark_outline_fraction(fig, ax, lon_deg=-72.0, lat_deg=-45.0) > 0.05
    assert _dark_outline_fraction(fig, ax, lon_deg=-40.0, lat_deg=55.0) < 0.02
    assert _dark_outline_fraction(fig, ax, lon_deg=160.0, lat_deg=0.0) < 0.02
    assert _dark_outline_fraction(fig, ax, lon_deg=150.0, lat_deg=-80.0) < 0.02


def test_generic_coverage_plotting_helpers_return_figures_and_axes() -> None:
    coverage = _coverage()
    field = coverage.access_duration(unit="minutes")
    stack = coverage.access_duration(min_assets=[1, 2], unit="minutes")
    timeline = coverage.target(index=0).timeline()

    fig_map, ax_map, artist, cbar = plot_coverage_map(field, map_cfg=LIGHT_DETAILED, title="Coverage")
    assert isinstance(fig_map, Figure)
    assert isinstance(ax_map, cartopy_geoaxes.GeoAxes)
    assert ax_map.get_title() == "Coverage"
    assert artist is not None
    assert cbar is not None
    plt.close(fig_map)

    fig_hist, ax_hist = plot_coverage_histogram(field, bins=10, title="Histogram")
    assert isinstance(fig_hist, Figure)
    assert ax_hist.get_title() == "Histogram"
    plt.close(fig_hist)

    fig_ecdf, ax_ecdf = plot_coverage_ecdf(field, title="ECDF")
    assert isinstance(fig_ecdf, Figure)
    assert ax_ecdf.get_title() == "ECDF"
    plt.close(fig_ecdf)

    fig_stack, axes_stack = plot_coverage_small_multiples(stack, dim="min_assets", map_cfg=LIGHT_DETAILED)
    assert isinstance(fig_stack, Figure)
    assert np.atleast_1d(axes_stack).size >= 2
    plt.close(fig_stack)

    fig_timeline, axes_timeline = plot_target_timeline(timeline, title="Target Timeline")
    assert isinstance(fig_timeline, Figure)
    assert len(fig_timeline.axes) == 2
    plt.close(fig_timeline)


def test_result_object_plot_methods_delegate_to_generic_plotting() -> None:
    coverage = _coverage()
    field = coverage.max_asset()
    stack = coverage.access_duration(min_assets=[1, 2], unit="minutes")
    timeline = coverage.target(index=0).timeline()

    fig_field, _, _, _ = field.plot(map_cfg=LIGHT_DETAILED)
    assert isinstance(fig_field, Figure)
    plt.close(fig_field)

    fig_stack, _ = stack.plot_small_multiples(dim="min_assets", map_cfg=LIGHT_DETAILED)
    assert isinstance(fig_stack, Figure)
    plt.close(fig_stack)

    fig_timeline, _ = timeline.plot()
    assert isinstance(fig_timeline, Figure)
    plt.close(fig_timeline)


def test_plot_coverage_histogram_handles_near_identical_float_maxima() -> None:
    arr = CoverageArray(
        values=np.array([96.0] * 9 + [96.00000000000001], dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(10, dtype=np.int64)},
        label="Synthetic Access",
    )

    fig, ax = plot_coverage_histogram(arr, bins=30, title="Histogram")
    assert isinstance(fig, Figure)
    assert ax.get_title() == "Histogram"
    assert int(round(sum(float(patch.get_height()) for patch in ax.patches))) == 10
    plt.close(fig)


def test_plot_coverage_map_renders_latitude_adaptive_targets_as_filled_surface_by_default() -> None:
    targets = CoverageTargets.global_earth(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
    )
    field = CoverageField(
        targets=targets,
        values=np.cos(np.deg2rad(targets.lat_deg)),
        metric_name="demo_metric",
        unit="unitless",
        label="Adaptive Surface",
    )

    fig_surface, _, artist_surface, _ = plot_coverage_map(field, map_cfg=LIGHT_DETAILED)
    assert isinstance(artist_surface, QuadMesh)
    plt.close(fig_surface)

    fig_points, _, artist_points, _ = plot_coverage_map(
        field,
        map_cfg=LIGHT_DETAILED,
        render="points",
    )
    assert isinstance(artist_points, PathCollection)
    plt.close(fig_points)


def test_plot_coverage_map_handles_ocean_boundary_geometry() -> None:
    targets = CoverageTargets.ocean(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
        resolution="110m",
    )
    field = CoverageField(
        targets=targets,
        values=np.cos(np.deg2rad(targets.lat_deg)),
        metric_name="ocean_metric",
        unit="unitless",
        label="Ocean Surface",
    )

    fig, ax, artist, _ = plot_coverage_map(field, map_cfg=LIGHT_DETAILED)
    assert isinstance(fig, Figure)
    assert artist is not None
    _assert_global_coastline_overlay(fig, ax)
    plt.close(fig)


def test_plot_coverage_map_handles_land_boundary_geometry() -> None:
    targets = CoverageTargets.land(
        sampler=LatitudeAdaptiveSampler(nlats=7, nlons_equator=12),
        resolution="110m",
    )
    field = CoverageField(
        targets=targets,
        values=np.cos(np.deg2rad(targets.lat_deg)),
        metric_name="land_metric",
        unit="unitless",
        label="Land Surface",
    )

    fig, ax, artist, _ = plot_coverage_map(field, map_cfg=LIGHT_DETAILED)
    assert isinstance(fig, Figure)
    assert artist is not None
    _assert_global_coastline_overlay(fig, ax)
    plt.close(fig)


def test_plot_coverage_map_clips_structured_grid_to_bbox_bounds() -> None:
    targets = CoverageTargets.region_bbox(
        west_deg=-20.0,
        east_deg=20.0,
        south_deg=-10.0,
        north_deg=10.0,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=5),
    )
    field = CoverageField(
        targets=targets,
        values=np.ones(targets.n_targets, dtype=np.float64),
        metric_name="bbox_metric",
        unit="unitless",
        label="BBox Surface",
    )
    map_cfg = replace(
        LIGHT_DETAILED,
        extent=ExtentConfig(global_map=False, extent=(-25.0, 25.0, -15.0, 15.0)),
    )

    fig, ax, _, _ = plot_coverage_map(
        field,
        map_cfg=map_cfg,
        alpha=1.0,
        vmin=0.0,
        vmax=1.0,
        boundary=False,
    )
    assert _patch_saturation(fig, ax, lon_deg=19.9, lat_deg=0.0) > 100.0
    assert _patch_saturation(fig, ax, lon_deg=20.3, lat_deg=0.0) < 10.0
    assert _patch_saturation(fig, ax, lon_deg=0.0, lat_deg=9.9) > 100.0
    assert _patch_saturation(fig, ax, lon_deg=0.0, lat_deg=10.3) < 10.0
    assert _patch_saturation(fig, ax, lon_deg=0.0, lat_deg=-9.9) > 100.0
    assert _patch_saturation(fig, ax, lon_deg=0.0, lat_deg=-10.3) < 10.0
    plt.close(fig)
