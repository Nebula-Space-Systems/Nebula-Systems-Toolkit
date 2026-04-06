from __future__ import annotations

import re
import warnings

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import nstk
import nstk.plotting as plotting
import nstk.propagation as propagation
import nstk.propagation.orbit as orbit_module
from nstk.plotting import MapView, get_map_style, plot_orbits
from nstk.plotting import map as map_module
from nstk.plotting.orbits import (
    _DEFAULT_COLORS,
    _make_info_text,
    _points_visible_from_view,
    _set_3d_globe_interaction,
    _surface_points_visible_from_view,
    _update_3d_scene,
    _view_direction_from_angles,
    plot_orbits as plot_orbits_impl,
)
from nstk.propagation.orbit import Orbit


def _make_orbit(*, epoch: Time, raan_deg: float, anomaly_deg: float) -> Orbit:
    return Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(raan_deg),
        argp=np.deg2rad(20.0),
        anomaly=np.deg2rad(anomaly_deg),
    )


def _collection_by_gid(ax, gid: str):
    for collection in ax.collections:
        if collection.get_gid() == gid:
            return collection
    raise AssertionError(f"missing collection with gid={gid!r}")


def _text_by_substring(fig: Figure, needle: str):
    for text in fig.texts:
        if needle in text.get_text():
            return text
    raise AssertionError(f"missing text containing {needle!r}")


def _visible_x_tick_bboxes(fig: Figure, ax: GeoAxes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [
        tick.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for tick in ax.get_xticklabels()
        if tick.get_visible() and tick.get_text().strip()
    ]


def test_plot_orbits_reexport_matches_orbits_module() -> None:
    assert nstk.plotting is plotting
    assert propagation.orbit is orbit_module
    assert plotting.plot_orbits is plot_orbits_impl


def test_plot_orbits_uses_view_specific_default_figsizes() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)

    fig_3d, _ = plot_orbits(orbit, view="3d", show=False)
    fig_2d, _ = plot_orbits(orbit, view="2d", duration=35.0 * u.min, show=False)

    assert np.allclose(fig_3d.get_size_inches(), (10.5, 7.0))
    assert np.allclose(fig_2d.get_size_inches(), (10.5, 6.2))

    plt.close(fig_3d)
    plt.close(fig_2d)


def test_plot_orbits_single_orbit_defaults_to_no_info_text() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)

    fig_3d, _ = plot_orbits(orbit, view="3d", show=False)
    fig_2d, _ = plot_orbits(orbit, view="2d", duration=35.0 * u.min, show=False)

    text_blob_3d = "\n".join(text.get_text() for text in fig_3d.texts)
    text_blob_2d = "\n".join(text.get_text() for text in fig_2d.texts)

    assert "Initial Keplerian Elements" not in text_blob_3d
    assert "Time Context" not in text_blob_3d
    assert "Marker shows the evaluated spacecraft state" not in text_blob_3d
    assert "Initial Keplerian Elements" not in text_blob_2d
    assert "Time Context" not in text_blob_2d
    assert "Marker shows the evaluated spacecraft ground-track position" not in text_blob_2d

    plt.close(fig_3d)
    plt.close(fig_2d)


def test_plot_orbits_multi_orbit_3d_defaults() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_a = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)
    orbit_b = _make_orbit(epoch=epoch, raan_deg=45.0, anomaly_deg=120.0)

    fig, ax = plot_orbits(
        [orbit_a, orbit_b],
        labels=["Alpha", "Bravo"],
        view="3d",
        samples=96,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert ax.name == "3d"

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Alpha", "Bravo"]
    assert to_rgba(legend.legend_handles[0].get_color()) == pytest.approx(to_rgba(map_module.NSTK_HIGHLIGHT_ORANGE))
    assert to_rgba(legend.legend_handles[1].get_color()) == pytest.approx(to_rgba("#55C1FF"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    axes_bbox = ax.get_position()
    assert legend_bbox.y1 <= axes_bbox.y0
    assert len(ax.collections) >= 5
    assert all("Keplerian period" not in text.get_text() for text in fig.texts)

    plt.close(fig)


def test_plot_orbits_multi_orbit_3d_distinguishes_repeated_colors() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_count = len(_DEFAULT_COLORS) + 2
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=float(idx * 20), anomaly_deg=float(idx * 25))
        for idx in range(orbit_count)
    ]

    fig, ax = plot_orbits(orbits, view="3d", samples=72, show=False)

    legend = ax.get_legend()
    assert legend is not None
    handles = legend.legend_handles
    assert to_rgba(handles[0].get_color()) == pytest.approx(to_rgba(handles[len(_DEFAULT_COLORS)].get_color()))
    assert handles[0].get_linestyle() != handles[len(_DEFAULT_COLORS)].get_linestyle()
    assert handles[0].get_marker() != handles[len(_DEFAULT_COLORS)].get_marker()

    plt.close(fig)


def test_make_info_text_aligns_annotation_value_columns() -> None:
    info_text = _make_info_text(
        {
            "a": "7,000.0 km",
            "e": "0.001000",
            "i": "  53.000 deg",
            "raan": "  18.000 deg",
            "argp": "  20.000 deg",
            "nu": "  35.066 deg",
        },
        [
            ("Evaluation", "2026-01-01 01:37:08.539 UTC"),
            ("Start offset", "12.00 min"),
            ("Trail window", "97.14 min"),
            ("Marker", "filled=near side, hollow=far side"),
        ],
        left_rows=[
            ("Epoch", "2026-01-01 00:00:00.000 UTC"),
            ("Frame", "gcrf"),
        ],
    )

    lines = info_text.splitlines()
    time_context_idx = lines.index("Time Context")
    left_lines = lines[1 : time_context_idx - 1]
    left_value_columns = {
        len(match.group(1)) + len(match.group(2))
        for line in left_lines
        for match in [re.match(r"^(.*?)(\s{2,})\S", line)]
        if match is not None
    }
    right_lines = lines[time_context_idx + 1 :]
    right_value_columns = {
        len(match.group(1)) + len(match.group(2))
        for line in right_lines
        for match in [re.match(r"^(.*?)(\s{2,})\S", line)]
        if match is not None
    }

    assert left_value_columns == {7}
    assert right_value_columns == {14}


def test_points_visible_from_view_occludes_backside_points() -> None:
    view_dir = _view_direction_from_angles(0.0, 0.0)
    points_km = np.asarray(
        [
            [7000.0, 0.0, 0.0],
            [-7000.0, 0.0, 0.0],
            [0.0, 7000.0, 0.0],
            [-1000.0, 7000.0, 0.0],
        ],
        dtype=np.float64,
    )

    visible = _points_visible_from_view(points_km, view_dir, 6378.137)
    assert visible.tolist() == [True, False, True, True]


def test_surface_points_visible_from_view_hides_backside_earth_lines() -> None:
    view_dir = _view_direction_from_angles(0.0, 0.0)
    points_km = np.asarray(
        [
            [6378.137, 0.0, 0.0],
            [-6378.137, 0.0, 0.0],
            [0.0, 6378.137, 0.0],
        ],
        dtype=np.float64,
    )

    visible = _surface_points_visible_from_view(points_km, view_dir)
    assert visible.tolist() == [True, False, True]


def test_orbit_plot_supports_explicit_time_window_in_3d() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=20.0, anomaly_deg=10.0)

    fig, ax = orbit.plot(
        start_time=epoch + 12.0 * u.min,
        duration=42.0 * u.min,
        view="3d",
        samples=72,
        title="Single Orbit Plot",
        show_info=True,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert ax.name == "3d"
    assert ax.get_title() == "Single Orbit Plot"
    assert len(ax.collections) >= 4
    assert ax.get_legend() is None

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" in text_blob
    assert "Epoch" in text_blob
    assert "Evaluation" in text_blob
    assert "Frame  gcrf" in text_blob
    assert "RAAN" in text_blob
    assert "Marker shows the evaluated spacecraft state" in text_blob

    info_box = _text_by_substring(fig, "Initial Keplerian Elements")
    axes_bbox = ax.get_position()
    assert "Time Context" in info_box.get_text()
    assert info_box.get_position()[1] < axes_bbox.y0
    assert np.isclose(info_box.get_position()[0], 0.5 * (axes_bbox.x0 + axes_bbox.x1))

    plt.close(fig)


def test_plot_orbits_single_orbit_3d_reports_keplerian_reference_frame() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
        inertial_frame="eme2000",
    )

    fig, _ = plot_orbits(orbit, view="3d", show_info=True, show=False)

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Frame  eme2000" in text_blob
    assert "Frame  itrf" not in text_blob

    plt.close(fig)


def test_orbit_plot_supports_custom_style_controls_in_3d() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=20.0, anomaly_deg=10.0)

    fig, ax = orbit.plot(
        view="3d",
        opacity=0.4,
        line_width=3.7,
        marker_size=9.0,
        samples=72,
        show=False,
    )

    glow = _collection_by_gid(ax, "nstk-orbit-trail-glow-0")
    core = _collection_by_gid(ax, "nstk-orbit-trail-core-0")
    assert isinstance(glow, Line3DCollection)
    assert isinstance(core, Line3DCollection)
    assert np.isclose(float(np.ravel(glow.get_linewidths())[0]), 7.4)
    assert np.isclose(float(glow.get_alpha()), 0.072)
    assert np.isclose(float(np.ravel(core.get_linewidths())[0]), 3.7)
    assert np.isclose(float(core.get_alpha()), 0.384)

    halo = _collection_by_gid(ax, "nstk-orbit-halo-0")
    marker = _collection_by_gid(ax, "nstk-orbit-marker-0")
    assert np.isclose(float(halo.get_sizes()[0]), (9.0 * 2.05) ** 2)
    assert np.isclose(float(halo.get_alpha()), 0.052)
    assert np.isclose(float(marker.get_sizes()[0]), 81.0)
    assert np.isclose(float(marker.get_alpha()), 0.4) or np.isclose(float(marker.get_alpha()), 0.312)

    plt.close(fig)


def test_plot_orbits_3d_updates_marker_style_when_camera_changes() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )

    fig, ax = orbit.plot(view="3d", show=False)

    marker = _collection_by_gid(ax, "nstk-orbit-marker-0")
    state = getattr(ax, "_nstk_3d_scene_state")
    initial_alpha = float(marker.get_alpha())
    initial_linewidth = float(np.ravel(marker.get_linewidths())[0])

    ax.view_init(elev=20.0, azim=180.0)
    assert _update_3d_scene(state) is True

    updated_alpha = float(marker.get_alpha())
    updated_linewidth = float(np.ravel(marker.get_linewidths())[0])
    assert updated_alpha > initial_alpha
    assert updated_linewidth < initial_linewidth

    plt.close(fig)


def test_plot_orbits_3d_does_not_emit_runtime_warnings() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=20.0, anomaly_deg=10.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = orbit.plot(view="3d", show=False)
        fig.canvas.draw()

    runtime_warnings = [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []

    plt.close(fig)


def test_plot_orbits_single_orbit_2d_ground_track() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=10.0, anomaly_deg=25.0)

    fig, ax = plot_orbits(
        orbit,
        view="2d",
        duration=35.0 * u.min,
        samples=120,
        title="Ground Track View",
        show_info=True,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert isinstance(ax, GeoAxes)
    assert not hasattr(ax, "zaxis")
    assert ax.get_title() == "Ground Track View"
    assert ax.get_legend() is None
    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 4

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" in text_blob
    assert "WGS84 geodetic" in text_blob
    assert "Evaluation" in text_blob
    assert "Marker shows the evaluated spacecraft ground-track position" in text_blob

    info_box = _text_by_substring(fig, "Initial Keplerian Elements")
    axes_bbox = ax.get_position()
    assert "Time Context" in info_box.get_text()
    assert info_box.get_position()[1] < axes_bbox.y0
    assert np.isclose(info_box.get_position()[0], 0.5 * (axes_bbox.x0 + axes_bbox.x1))

    plt.close(fig)


def test_orbit_plot_supports_custom_style_controls_in_2d() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=10.0, anomaly_deg=25.0)

    fig, ax = orbit.plot(
        view="2d",
        opacity=0.4,
        line_width=3.7,
        marker_size=9.0,
        samples=120,
        show=False,
    )

    assert all(np.isclose(line.get_linewidth(), 3.7) for line in ax.lines)
    assert all(np.isclose(line.get_alpha(), 0.36) for line in ax.lines)

    halo = ax.collections[-2]
    marker = ax.collections[-1]
    assert np.isclose(float(halo.get_sizes()[0]), (9.0 * 2.05) ** 2)
    assert np.isclose(float(halo.get_alpha()), 0.06)
    assert np.isclose(float(marker.get_sizes()[0]), 81.0)
    assert np.isclose(float(marker.get_alpha()), 0.4)

    plt.close(fig)


def test_plot_orbits_multi_orbit_2d_legend() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_a = _make_orbit(epoch=epoch, raan_deg=5.0, anomaly_deg=0.0)
    orbit_b = _make_orbit(epoch=epoch, raan_deg=95.0, anomaly_deg=140.0)

    fig, ax = plot_orbits(
        [orbit_a, orbit_b],
        labels=["Alpha", "Bravo"],
        view="2d",
        samples=90,
        show=False,
    )

    assert isinstance(ax, GeoAxes)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Alpha", "Bravo"]
    assert to_rgba(legend.legend_handles[0].get_color()) == pytest.approx(to_rgba(map_module.NSTK_HIGHLIGHT_ORANGE))
    assert to_rgba(legend.legend_handles[1].get_color()) == pytest.approx(to_rgba("#55C1FF"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    axes_bbox = ax.get_position()
    x_tick_bboxes = _visible_x_tick_bboxes(fig, ax)
    assert legend_bbox.y1 <= axes_bbox.y0
    assert x_tick_bboxes
    assert legend_bbox.y1 <= min(bbox.y0 for bbox in x_tick_bboxes)

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" not in text_blob
    assert "Keplerian period" not in text_blob

    plt.close(fig)


def test_plot_orbits_multi_orbit_2d_distinguishes_repeated_colors() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_count = len(_DEFAULT_COLORS) + 2
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=float(idx * 18), anomaly_deg=float(idx * 22))
        for idx in range(orbit_count)
    ]

    fig, ax = plot_orbits(
        orbits,
        view="2d",
        duration=35.0 * u.min,
        samples=80,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    handles = legend.legend_handles
    wrapped_idx = len(_DEFAULT_COLORS)
    assert to_rgba(handles[0].get_color()) == pytest.approx(to_rgba(handles[wrapped_idx].get_color()))
    assert handles[0].get_linestyle() != handles[wrapped_idx].get_linestyle()
    assert handles[0].get_marker() != handles[wrapped_idx].get_marker()
    wrapped_color = to_rgba(handles[wrapped_idx].get_color())
    matching_lines = [line for line in ax.lines if np.allclose(to_rgba(line.get_color()), wrapped_color)]
    assert any(line.is_dashed() for line in matching_lines)

    plt.close(fig)


def test_plot_orbits_2d_accepts_custom_map_style() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)
    paper = (
        get_map_style("light_detailed")
        .with_borders(enabled=False)
        .with_grid(alpha=0.12, draw_labels=False)
    )

    fig, ax = plot_orbits(
        orbit,
        view="2d",
        map_style=paper,
        samples=96,
        show=False,
    )

    assert isinstance(ax, GeoAxes)
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_orbits_3d_accepts_custom_map_style() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)
    paper = get_map_style("light_detailed")

    fig, ax = plot_orbits(
        orbit,
        view="3d",
        map_style=paper,
        show_info=True,
        samples=96,
        show=False,
    )

    coast = _collection_by_gid(ax, "nstk-earth-coast")
    country = _collection_by_gid(ax, "nstk-earth-country")
    info_text = next(text for text in fig.texts if "Marker shows the evaluated spacecraft state" in text.get_text())

    assert np.allclose(fig.get_facecolor(), to_rgba(paper.theme.figure_face))
    assert np.allclose(ax.get_facecolor(), to_rgba(paper.theme.axes_face))
    assert np.allclose(np.atleast_2d(coast.get_colors())[0], to_rgba(paper.coastlines.color, paper.coastlines.alpha))
    assert np.isclose(float(np.ravel(coast.get_linewidths())[0]), paper.coastlines.linewidth)
    assert np.allclose(np.atleast_2d(country.get_colors())[0], to_rgba(paper.borders.color, paper.borders.alpha))
    assert np.allclose(to_rgba(info_text.get_color()), to_rgba(paper.theme.grid_label_color, 0.82))

    plt.close(fig)


def test_plot_orbits_3d_uses_idle_and_drag_globe_surfaces() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)

    fig, ax = plot_orbits(
        orbit,
        view="3d",
        samples=96,
        show=False,
    )

    globe_surface = _collection_by_gid(ax, "nstk-earth-globe")
    globe_drag_surface = _collection_by_gid(ax, "nstk-earth-globe-interactive")
    scene_state = ax._nstk_3d_scene_state

    assert globe_surface.get_facecolors().shape[0] >= 28_000
    assert globe_drag_surface.get_facecolors().shape[0] <= 15_000
    assert globe_surface.get_visible() is True
    assert globe_drag_surface.get_visible() is False

    assert _set_3d_globe_interaction(scene_state, True) is True
    assert globe_surface.get_visible() is False
    assert globe_drag_surface.get_visible() is True

    assert _set_3d_globe_interaction(scene_state, False) is True
    assert globe_surface.get_visible() is True
    assert globe_drag_surface.get_visible() is False

    plt.close(fig)


def test_plot_orbits_2d_accepts_custom_map_view() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=18.0, anomaly_deg=35.0)
    atlantic = (
        MapView()
        .with_projection(name="Mercator")
        .with_extent((-100.0, 30.0, -10.0, 65.0), global_map=False)
    )

    fig, ax = plot_orbits(
        orbit,
        view="2d",
        map_view=atlantic,
        samples=96,
        show=False,
    )

    west, east, south, north = ax.get_extent(crs=ccrs.PlateCarree())
    assert isinstance(ax, GeoAxes)
    assert west <= -100.0 + 1e-6
    assert east >= 30.0 - 1e-6
    assert south <= -10.0 + 1e-6
    assert north >= 65.0 - 1e-6
    plt.close(fig)


def test_plot_orbits_multi_orbit_2d_legend_stays_below_map_for_many_entries() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=45.0 * idx, anomaly_deg=40.0 * idx)
        for idx in range(8)
    ]

    fig, ax = plot_orbits(
        orbits,
        labels=[f"Sat {idx + 1}" for idx in range(len(orbits))],
        view="2d",
        samples=90,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    axes_bbox = ax.get_position()
    x_tick_bboxes = _visible_x_tick_bboxes(fig, ax)
    assert legend_bbox.y1 <= axes_bbox.y0
    assert legend_bbox.y0 >= 0.0
    assert x_tick_bboxes
    assert legend_bbox.y1 <= min(bbox.y0 for bbox in x_tick_bboxes)

    plt.close(fig)


def test_plot_orbits_multi_orbit_3d_legend_stays_below_plot_for_many_entries() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=45.0 * idx, anomaly_deg=40.0 * idx)
        for idx in range(8)
    ]

    fig, ax = plot_orbits(
        orbits,
        labels=[f"Sat {idx + 1}" for idx in range(len(orbits))],
        view="3d",
        samples=90,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    axes_bbox = ax.get_position()
    assert legend_bbox.y1 <= axes_bbox.y0
    assert legend_bbox.y0 >= 0.0

    plt.close(fig)


def test_plot_orbits_multi_orbit_2d_hides_legend_above_24_entries() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=14.0 * idx, anomaly_deg=11.0 * idx)
        for idx in range(25)
    ]

    fig, ax = plot_orbits(
        orbits,
        labels=[f"Sat {idx + 1}" for idx in range(len(orbits))],
        view="2d",
        samples=72,
        show=False,
    )

    assert ax.get_legend() is None

    plt.close(fig)


def test_plot_orbits_multi_orbit_3d_hides_legend_above_24_entries() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=14.0 * idx, anomaly_deg=11.0 * idx)
        for idx in range(25)
    ]

    fig, ax = plot_orbits(
        orbits,
        labels=[f"Sat {idx + 1}" for idx in range(len(orbits))],
        view="3d",
        samples=72,
        show=False,
    )

    assert ax.get_legend() is None

    plt.close(fig)


def test_plot_orbits_multi_orbit_3d_keeps_legend_at_24_entries() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbits = [
        _make_orbit(epoch=epoch, raan_deg=15.0 * idx, anomaly_deg=9.0 * idx)
        for idx in range(24)
    ]

    fig, ax = plot_orbits(
        orbits,
        labels=[f"Sat {idx + 1}" for idx in range(len(orbits))],
        view="3d",
        samples=72,
        show=False,
    )

    assert ax.get_legend() is not None

    plt.close(fig)


def test_plot_orbits_rejects_invalid_view() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)

    try:
        plot_orbits(orbit, view="sideways", show=False)
        assert False, "expected ValueError for invalid view"
    except ValueError as exc:
        assert "view" in str(exc)


def test_plot_orbits_rejects_removed_frame_keyword() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)

    try:
        plot_orbits(orbit, view="3d", frame="gcrf", show=False)
        assert False, "expected TypeError for removed frame keyword"
    except TypeError as exc:
        assert "frame" in str(exc)


def test_plot_orbits_rejects_mismatched_axis_type() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    try:
        plot_orbits(orbit, view="2d", ax=ax_3d, show=False)
        assert False, "expected TypeError for 2d plot on 3d axes"
    except TypeError as exc:
        assert "view='2d'" in str(exc)
    finally:
        plt.close(fig_3d)

    fig_2d = plt.figure()
    ax_2d = fig_2d.add_subplot(111, projection=ccrs.PlateCarree())
    try:
        plot_orbits(orbit, view="3d", ax=ax_2d, show=False)
        assert False, "expected TypeError for 3d plot on 2d axes"
    except TypeError as exc:
        assert "view='3d'" in str(exc)
    finally:
        plt.close(fig_2d)
