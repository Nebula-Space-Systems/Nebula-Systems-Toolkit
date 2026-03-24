from __future__ import annotations

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, QuadMesh
from shapely.geometry import box, mapping

from nstk.coverage import CoverageField, CoverageTargets, LatitudeLongitudeSampler
from nstk.plotting import (
    GeoMap,
    LIGHT_DETAILED,
    MapLayer,
    MapView,
    available_map_styles,
    compile_map_config,
    get_map_style,
    register_map_style,
)


cartopy_geoaxes = pytest.importorskip("cartopy.mpl.geoaxes")


def _demo_field() -> CoverageField:
    targets = CoverageTargets.region_bbox(
        west_deg=-20.0,
        east_deg=20.0,
        south_deg=-10.0,
        north_deg=10.0,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=7),
    )
    values = np.cos(np.deg2rad(targets.lat_deg)) * np.cos(np.deg2rad(targets.lon_deg))
    return CoverageField(
        targets=targets,
        values=values,
        metric_name="demo_metric",
        unit="unitless",
        label="Demo Field",
    )


def test_geomap_dispatch_and_auto_extent_union() -> None:
    m = GeoMap(theme="light_detailed", extent="auto", pad_deg=0.5)

    field_layer = m.add(_demo_field(), cmap="viridis")
    assert isinstance(field_layer, MapLayer)

    table = {
        "lon": np.array([35.0, 40.0], dtype=np.float64),
        "lat": np.array([5.0, 8.0], dtype=np.float64),
        "value": np.array([1.0, 2.0], dtype=np.float64),
    }
    point_layer = m.add(
        table,
        kind="points",
        x="lon",
        y="lat",
        values="value",
        colorbar=True,
        colorbar_label="score",
    )

    assert isinstance(point_layer, MapLayer)
    assert isinstance(point_layer.artist, PathCollection)
    assert point_layer.colorbar is not None

    point_layer.set_alpha(0.45)
    assert np.isclose(float(point_layer.artist.get_alpha()), 0.45)
    point_layer.set_visible(False)
    assert point_layer.artist.get_visible() is False
    point_layer.set_visible(True)

    west, east, south, north = m.ax.get_extent(crs=ccrs.PlateCarree())
    assert west <= -20.0
    assert east >= 40.0
    assert south <= -10.0
    assert north >= 10.0

    plt.close(m.fig)


def test_geomap_fit_uses_layer_bounds() -> None:
    m = GeoMap(theme="light_detailed", extent="global")
    layer_a = m.add_geometry(box(-15.0, -5.0, -5.0, 5.0), auto_extent=False)
    layer_b = m.add_geometry(box(20.0, 10.0, 30.0, 25.0), auto_extent=False)

    m.fit(layers=[layer_a, layer_b], pad_deg=1.0)
    west, east, south, north = m.ax.get_extent(crs=ccrs.PlateCarree())
    assert west <= -15.0
    assert east >= 30.0
    assert south <= -5.0
    assert north >= 25.0
    assert (east - west) < 80.0

    plt.close(m.fig)


def test_geomap_supports_raster_contours_vectors_and_annotations() -> None:
    lon = np.linspace(-12.0, 12.0, 7)
    lat = np.linspace(-8.0, 8.0, 5)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    data = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))

    m = GeoMap(theme="light_detailed", extent="auto")
    raster_layer = m.add_raster(data, lon_deg=lon, lat_deg=lat, colorbar=True, colorbar_label="value")
    contour_layer = m.add_contours(data, lon_deg=lon, lat_deg=lat, levels=4, colors="black")
    filled_layer = m.add_filled_contours(data, lon_deg=lon, lat_deg=lat, levels=4, alpha=0.2)
    quiver_layer = m.add_quiver(lon_grid, lat_grid, np.ones_like(data), np.zeros_like(data), color="black")
    text_layer = m.add_text(0.0, 0.0, "Center")
    labels_layer = m.add_labels(np.array([-10.0, 10.0]), np.array([0.0, 0.0]), ["West", "East"])

    assert isinstance(raster_layer.artist, QuadMesh)
    assert raster_layer.colorbar is not None
    assert contour_layer.artist is not None
    assert filled_layer.artist is not None
    assert quiver_layer.artist is not None
    assert text_layer.artist is not None
    assert len(labels_layer.artists) == 2

    plt.close(m.fig)


def test_geomap_country_geojson_and_feature_helpers() -> None:
    m = GeoMap(theme="light_detailed", extent="auto")

    geojson_layer = m.add(mapping(box(120.0, 20.0, 150.0, 50.0)), facecolor="none", edgecolor="blue")
    country_layer = m.add_country("Japan", facecolor="none", edgecolor="red")
    feature_layer = m.add_feature(cfeature.BORDERS, edgecolor="#666666", linewidth=0.2, alpha=0.6)

    assert isinstance(geojson_layer, MapLayer)
    assert isinstance(country_layer, MapLayer)
    assert isinstance(feature_layer, MapLayer)
    assert "japan" in country_layer.metadata["country_match"].display_name.lower()

    west, east, south, north = m.ax.get_extent(crs=ccrs.PlateCarree())
    assert west <= 120.0
    assert east >= 145.0
    assert south <= 20.0
    assert north >= 45.0

    plt.close(m.fig)


def test_map_style_presets_preserve_existing_looks_and_support_custom_registration() -> None:
    compiled = compile_map_config(style="light_detailed")
    assert compiled.land.facecolor == LIGHT_DETAILED.land.facecolor
    assert compiled.ocean.facecolor == LIGHT_DETAILED.ocean.facecolor
    assert compiled.coastlines.color == LIGHT_DETAILED.coastlines.color
    assert compiled.gridlines.alpha == LIGHT_DETAILED.gridlines.alpha

    paper = (
        get_map_style("light_detailed")
        .with_land(facecolor="#d8d0c4")
        .with_grid(alpha=0.12, draw_labels=False)
        .with_borders(enabled=False)
    )
    register_map_style("paper_test", paper, overwrite=True)
    assert "paper_test" in available_map_styles()

    m = GeoMap(style="paper_test", extent="auto")
    assert m.config.land.facecolor == "#d8d0c4"
    assert np.isclose(m.config.gridlines.alpha, 0.12)
    assert m.config.gridlines.draw_labels is False
    assert m.config.borders.enabled is False

    plt.close(m.fig)


def test_geomap_accepts_reusable_view_objects() -> None:
    conus = (
        MapView()
        .with_projection(name="Mercator")
        .with_extent((-125.0, -66.0, 24.0, 50.0), global_map=False)
    )
    m = GeoMap(style="light_detailed", view=conus)

    assert m.config.projection.name == "Mercator"
    assert m.config.extent.global_map is False
    assert m.config.extent.extent == (-125.0, -66.0, 24.0, 50.0)

    plt.close(m.fig)
