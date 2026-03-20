"""2D and 3D Matplotlib orbit-visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import astropy.units as u
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from nebula.time_utils import normalize_time_to_epoch_seconds

from ._cartopy_data import (
    configure_cartopy_data_dir,
    get_natural_earth_shapefile,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D


configure_cartopy_data_dir()

_EARTH_RADIUS_M = 6378137.0
_EARTH_RADIUS_KM = _EARTH_RADIUS_M / 1000.0
_DEFAULT_COLORS = (
    "#55C1FF",
    "#FF7A59",
    "#9D7CFF",
    "#3DDC97",
    "#FFD166",
    "#FF5DA2",
    "#A1E44D",
    "#C792EA",
)


def _format_time_label(time_like: Any) -> str:
    return str(time_like.utc.isot).replace("T", " ")


def _format_angle_deg(angle_rad: float) -> str:
    angle_deg = (np.degrees(float(angle_rad)) + 360.0) % 360.0
    return f"{angle_deg:8.3f} deg"


def _annotation_box_style() -> dict[str, Any]:
    return {
        "boxstyle": "round,pad=0.55",
        "facecolor": "#081722",
        "edgecolor": "#2E6178",
        "linewidth": 1.0,
        "alpha": 0.95,
    }


def _extract_keplerian_summary(orbit: Any) -> dict[str, str]:
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore

    state0 = orbit.propagator.getInitialState()
    date0 = state0.getDate()
    frame0 = state0.getFrame()
    pv0 = state0.getPVCoordinates(frame0)
    mu = float(state0.getOrbit().getMu())

    if frame0.isPseudoInertial():
        inertial = frame0
        pv_inertial = pv0
    else:
        inertial = FramesFactory.getGCRF()
        tr = frame0.getTransformTo(inertial, date0)
        pv_inertial = tr.transformPVCoordinates(pv0)

    kep = KeplerianOrbit(CartesianOrbit(pv_inertial, inertial, date0, mu))
    return {
        "a": f"{float(kep.getA()) / 1000.0:,.1f} km",
        "e": f"{float(kep.getE()):.6f}",
        "i": _format_angle_deg(kep.getI()),
        "raan": _format_angle_deg(kep.getRightAscensionOfAscendingNode()),
        "argp": _format_angle_deg(kep.getPerigeeArgument()),
        "nu": _format_angle_deg(kep.getTrueAnomaly()),
    }


def _coerce_orbit_list(orbits: Any) -> list[Any]:
    if hasattr(orbits, "get_p_np"):
        out = [orbits]
    elif isinstance(orbits, Iterable):
        out = list(orbits)
    else:
        raise TypeError("orbits must be an Orbit or an iterable of Orbit objects")

    if not out:
        raise ValueError("orbits must contain at least one Orbit")
    for orbit in out:
        if not hasattr(orbit, "get_p_np"):
            raise TypeError("all items in orbits must be Orbit-like objects")
    return out


def _coerce_labels(labels: Sequence[str] | None, count: int) -> list[str] | None:
    if labels is None:
        return None
    if len(labels) != count:
        raise ValueError("labels must match the number of orbits")
    return [str(label) for label in labels]


def _coerce_colors(colors: Sequence[str] | None, count: int) -> list[str]:
    if colors is not None and len(colors) != count:
        raise ValueError("colors must match the number of orbits")
    if colors is None:
        return [_DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)] for idx in range(count)]
    return [str(color) for color in colors]


def _coerce_view(view: str) -> str:
    key = str(view).strip().lower()
    if key not in {"2d", "3d"}:
        raise ValueError("view must be either '2d' or '3d'")
    return key


def _coerce_duration_seconds(duration: Any, orbit: Any) -> float:
    if duration is None:
        from org.orekit.orbits import KeplerianOrbit  # type: ignore

        state0 = orbit.propagator.getInitialState()
        return float(KeplerianOrbit(state0.getOrbit()).getKeplerianPeriod())

    seconds = float(duration.to_value(u.s)) if isinstance(duration, u.Quantity) else float(duration)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("duration must be finite and > 0 seconds")
    return seconds


def _coerce_start_seconds(start_time: Any, orbit: Any) -> float:
    if start_time is None:
        return 0.0

    dt_s, is_scalar = normalize_time_to_epoch_seconds(start_time, orbit.epoch)
    if not is_scalar or dt_s.size != 1:
        raise ValueError("start_time must be a scalar time-like value")
    start_s = float(dt_s[0])
    if not np.isfinite(start_s):
        raise ValueError("start_time must be finite")
    return start_s


@lru_cache(maxsize=None)
def _natural_earth_geometries(
    resolution: str,
    category: str,
    name: str,
) -> tuple[Any, ...]:
    path = get_natural_earth_shapefile(
        resolution=resolution,
        category=category,
        name=name,
    )
    return tuple(Reader(str(path)).geometries())


def _collect_geometry_segments(geometry: Any, out: list[np.ndarray]) -> None:
    if geometry is None or geometry.is_empty:
        return

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"LineString", "LinearRing"}:
        coords = np.asarray(geometry.coords, dtype=np.float64)
        if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] >= 2:
            out.append(coords[:, :2])
        return

    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            _collect_geometry_segments(child, out)


@lru_cache(maxsize=None)
def _coastline_lonlat_segments(resolution: str = "110m") -> tuple[np.ndarray, ...]:
    segments: list[np.ndarray] = []
    for geom in _natural_earth_geometries(resolution, "physical", "coastline"):
        _collect_geometry_segments(geom, segments)
    return tuple(segments)


@lru_cache(maxsize=None)
def _country_outline_lonlat_segments(resolution: str = "110m") -> tuple[np.ndarray, ...]:
    segments: list[np.ndarray] = []
    for geom in _natural_earth_geometries(resolution, "cultural", "admin_0_boundary_lines_land"):
        _collect_geometry_segments(geom, segments)
    return tuple(segments)


def _split_dateline_segments(lon_deg: np.ndarray, lat_deg: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    lon_wrapped = ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    if lon_wrapped.size < 2:
        return [(lon_wrapped, lat_arr)]

    split_idx = np.where(np.abs(np.diff(lon_wrapped)) > 180.0)[0] + 1
    lon_parts = np.split(lon_wrapped, split_idx)
    lat_parts = np.split(lat_arr, split_idx)
    return [
        (lon_part, lat_part)
        for lon_part, lat_part in zip(lon_parts, lat_parts, strict=False)
        if lon_part.size >= 2
    ]


def _make_earth_facecolors(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    nx = np.cos(uu) * np.sin(vv)
    ny = np.sin(uu) * np.sin(vv)
    nz = np.cos(vv)

    light = np.array([0.43, -0.22, 0.88], dtype=np.float64)
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)

    # Smooth ocean palette that avoids texture banding in mplot3d.
    equator = 1.0 - np.abs(nz)
    swirl = 0.5 + 0.5 * np.sin(2.8 * uu + 0.85 * np.cos(2.0 * vv))
    variation = np.clip(0.74 + 0.26 * swirl, 0.66, 1.02)

    deep = np.array([0.03, 0.15, 0.29], dtype=np.float64)
    shallow = np.array([0.10, 0.35, 0.56], dtype=np.float64)
    base = deep + (shallow - deep) * (0.22 + 0.78 * equator)[..., None]
    base *= variation[..., None]

    brightness = 0.34 + 0.66 * diffuse
    specular = np.power(diffuse, 18.0) * 0.08

    facecolors = np.empty((*uu.shape, 4), dtype=np.float64)
    facecolors[..., :3] = np.clip(base * brightness[..., None] + specular[..., None], 0.0, 1.0)
    facecolors[..., 3] = 0.98
    return facecolors


def _lonlat_segments_to_xyz(
    segments: tuple[np.ndarray, ...],
    *,
    radius_scale: float,
    max_points_per_segment: int,
) -> list[np.ndarray]:
    radius = _EARTH_RADIUS_KM * float(radius_scale)
    out: list[np.ndarray] = []
    for segment in segments:
        for lon_seg, lat_seg in _split_dateline_segments(segment[:, 0], segment[:, 1]):
            if lon_seg.size < 2:
                continue
            step = max(1, int(np.ceil(lon_seg.size / max(2, int(max_points_per_segment)))))
            lon_use = lon_seg[::step]
            lat_use = lat_seg[::step]
            if lon_use.size < 2:
                continue

            lon_rad = np.radians(lon_use)
            lat_rad = np.radians(lat_use)
            cos_lat = np.cos(lat_rad)
            xyz = np.column_stack(
                (
                    radius * cos_lat * np.cos(lon_rad),
                    radius * cos_lat * np.sin(lon_rad),
                    radius * np.sin(lat_rad),
                )
            )
            if xyz.shape[0] >= 2:
                out.append(np.asarray(xyz, dtype=np.float64))
    return out


@lru_cache(maxsize=None)
def _earth_outline_xyz_segments(
    resolution: str,
    kind: str,
    radius_scale: float,
    max_points_per_segment: int,
) -> tuple[np.ndarray, ...]:
    if kind == "coastline":
        lonlat_segments = _coastline_lonlat_segments(resolution)
    elif kind == "country":
        lonlat_segments = _country_outline_lonlat_segments(resolution)
    else:
        raise ValueError("kind must be 'coastline' or 'country'")
    return tuple(
        _lonlat_segments_to_xyz(
            lonlat_segments,
            radius_scale=radius_scale,
            max_points_per_segment=max_points_per_segment,
        )
    )


def _draw_earth_globe(ax: Any, resolution: int = 112) -> None:
    u_grid = np.linspace(0.0, 2.0 * np.pi, int(resolution), dtype=np.float64)
    v_grid = np.linspace(0.0, np.pi, int(max(24, resolution // 2)), dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing="xy")

    x = _EARTH_RADIUS_KM * np.cos(uu) * np.sin(vv)
    y = _EARTH_RADIUS_KM * np.sin(uu) * np.sin(vv)
    z = _EARTH_RADIUS_KM * np.cos(vv)

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=_make_earth_facecolors(uu, vv),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=0,
    )

    coast_segments = _earth_outline_xyz_segments(
        "110m",
        "coastline",
        radius_scale=1.0002,
        max_points_per_segment=260,
    )
    if coast_segments:
        ax.add_collection3d(
            Line3DCollection(
                coast_segments,
                colors=(0.94, 0.985, 1.0, 0.55),
                linewidths=0.55,
                zorder=2,
            ),
            autolim=False,
        )

    country_segments = _earth_outline_xyz_segments(
        "110m",
        "country",
        radius_scale=1.0002,
        max_points_per_segment=200,
    )
    if country_segments:
        ax.add_collection3d(
            Line3DCollection(
                country_segments,
                colors=(0.76, 0.93, 1.0, 0.3),
                linewidths=0.32,
                zorder=2,
            ),
            autolim=False,
        )


def _plot_3d_orbit_trail(ax: Any, points_km: np.ndarray, color: str, linewidth: float) -> None:
    glow, = ax.plot(
        points_km[:, 0],
        points_km[:, 1],
        points_km[:, 2],
        color=color,
        linewidth=linewidth * 2.0,
        alpha=0.18,
        solid_capstyle="round",
        clip_on=False,
        zorder=14,
    )
    core, = ax.plot(
        points_km[:, 0],
        points_km[:, 1],
        points_km[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=0.96,
        solid_capstyle="round",
        clip_on=False,
        zorder=15,
    )
    for line in (glow, core):
        if hasattr(line, "set_sort_zpos"):
            line.set_sort_zpos(1.0e9)


def _style_3d_axes(ax: Any, span_km: float) -> None:
    bg = "#07111D"
    axis_line = (0.75, 0.9, 1.0, 0.12)
    grid_line = (0.75, 0.9, 1.0, 0.08)

    if hasattr(ax, "computed_zorder"):
        # Keep orbit traces/markers consistently visible over the globe.
        ax.computed_zorder = False

    ax.set_facecolor(bg)
    ax.figure.set_facecolor(bg)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-span_km, span_km)
    ax.set_ylim(-span_km, span_km)
    ax.set_zlim(-span_km, span_km)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(axis_line)
        axis.line.set_linewidth(0.8)
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = grid_line
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(colors="#CFEAF7")


def _style_2d_axes(ax: GeoAxes) -> None:
    ax.figure.set_facecolor("#07111D")
    ax.set_facecolor("#08131E")
    ax.set_global()
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_edgecolor("#2E6178")
        ax.outline_patch.set_linewidth(0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E6178")
        spine.set_linewidth(0.8)

    ax.add_geometries(
        _natural_earth_geometries("110m", "physical", "ocean"),
        crs=ccrs.PlateCarree(),
        facecolor="#08131E",
        edgecolor="none",
        zorder=0,
    )
    ax.add_geometries(
        _natural_earth_geometries("110m", "physical", "land"),
        crs=ccrs.PlateCarree(),
        facecolor="#122536",
        edgecolor="none",
        zorder=1,
    )
    ax.add_geometries(
        _natural_earth_geometries("110m", "physical", "coastline"),
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#D9F4FF",
        linewidth=0.45,
        zorder=2,
    )
    ax.add_geometries(
        _natural_earth_geometries("110m", "cultural", "admin_0_countries"),
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#A9E4FF",
        linewidth=0.28,
        alpha=0.55,
        zorder=2,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        color="#D2EDF9",
        alpha=0.14,
        linestyle="--",
        zorder=2,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))


def _style_legend(legend: Any) -> None:
    legend.get_frame().set_facecolor("#091724")
    legend.get_frame().set_edgecolor("#30596E")
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color("#E8F6FF")


def _make_info_text(
    summary: dict[str, str],
    right_lines: list[str],
    *,
    left_lines: Sequence[str] | None = None,
) -> tuple[str, str]:
    left_text = "\n".join(
        [
            "Initial Keplerian Elements",
            f"a     {summary['a']}",
            f"e     {summary['e']}",
            f"i     {summary['i']}",
            f"RAAN  {summary['raan']}",
            f"argp  {summary['argp']}",
            f"nu    {summary['nu']}",
            *([] if left_lines is None else list(left_lines)),
        ]
    )
    return left_text, "\n".join(right_lines)


def _add_single_orbit_3d_annotations(
    fig: Any,
    ax: Any,
    orbit: Any,
    marker_xyz: np.ndarray,
    start_s: float,
    duration_s: float,
    orbit_color: str,
    frame: Any,
) -> None:
    eval_time = orbit.epoch + (start_s + duration_s) * u.s
    ax.plot(
        [0.0, marker_xyz[0]],
        [0.0, marker_xyz[1]],
        [0.0, marker_xyz[2]],
        color=orbit_color,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        alpha=0.45,
        zorder=2,
    )

    left_text, right_text = _make_info_text(
        _extract_keplerian_summary(orbit),
        [
            "Time Context",
            f"Evaluation   {_format_time_label(eval_time)} UTC",
            f"Start offset {start_s / 60.0:,.2f} min",
            f"Trail window {duration_s / 60.0:,.2f} min",
        ],
        left_lines=[
            f"Epoch  {_format_time_label(orbit.epoch)} UTC",
            f"Frame  {frame if isinstance(frame, str) else 'custom frame'}",
        ],
    )
    box_style = _annotation_box_style()
    fig.text(
        0.03,
        0.92,
        left_text,
        ha="left",
        va="top",
        color="#EAF8FF",
        fontsize=10.5,
        family="monospace",
        bbox=box_style,
    )
    fig.text(
        0.70,
        0.92,
        right_text,
        ha="left",
        va="top",
        color="#EAF8FF",
        fontsize=10.0,
        family="monospace",
        bbox=box_style,
    )


def _add_single_orbit_2d_annotations(
    fig: Any,
    orbit: Any,
    start_s: float,
    duration_s: float,
) -> None:
    eval_time = orbit.epoch + (start_s + duration_s) * u.s
    left_text, right_text = _make_info_text(
        _extract_keplerian_summary(orbit),
        [
            "Time Context",
            f"Evaluation   {_format_time_label(eval_time)} UTC",
            f"Start offset {start_s / 60.0:,.2f} min",
            f"Trail window {duration_s / 60.0:,.2f} min",
        ],
        left_lines=[
            f"Epoch   {_format_time_label(orbit.epoch)} UTC",
            "Coords  WGS84 geodetic",
        ],
    )
    box_style = _annotation_box_style()
    fig.text(
        0.03,
        0.92,
        left_text,
        ha="left",
        va="top",
        color="#EAF8FF",
        fontsize=10.0,
        family="monospace",
        bbox=box_style,
    )
    fig.text(
        0.70,
        0.92,
        right_text,
        ha="left",
        va="top",
        color="#EAF8FF",
        fontsize=9.8,
        family="monospace",
        bbox=box_style,
    )


def _sample_plot_windows(
    orbit_list: list[Any],
    *,
    start_time: Any,
    duration: Any,
    frame: Any,
    labels: list[str] | None,
    colors: list[str],
    samples: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for idx, orbit in enumerate(orbit_list):
        start_s = _coerce_start_seconds(start_time, orbit)
        duration_s = _coerce_duration_seconds(duration, orbit)
        query_s = start_s + np.linspace(0.0, duration_s, int(samples), dtype=np.float64)
        label = labels[idx] if labels is not None else (f"Orbit {idx + 1}" if len(orbit_list) > 1 else "Orbit")
        color = colors[idx]

        points_km = np.asarray(orbit.get_p_np(query_s, frame=frame), dtype=np.float64) / 1000.0
        if points_km.ndim != 2 or points_km.shape[1] != 3:
            raise ValueError("orbit.get_p_np returned an unexpected shape")
        lat_deg, lon_deg, alt_m = orbit.get_geodetic_np(query_s)

        windows.append(
            {
                "orbit": orbit,
                "label": label,
                "color": color,
                "start_s": start_s,
                "duration_s": duration_s,
                "query_s": query_s,
                "points_km": points_km,
                "marker_xyz": points_km[-1],
                "lat_deg": np.asarray(lat_deg, dtype=np.float64),
                "lon_deg": np.asarray(lon_deg, dtype=np.float64),
                "alt_m": np.asarray(alt_m, dtype=np.float64),
                "marker_latlon": (
                    float(np.asarray(lat_deg, dtype=np.float64)[-1]),
                    float(np.asarray(lon_deg, dtype=np.float64)[-1]),
                ),
            }
        )
    return windows


def _render_3d(
    windows: list[dict[str, Any]],
    *,
    ax: Any | None,
    figsize: tuple[float, float],
    title: str | None,
    elev: float,
    azim: float,
    frame: Any,
    start_time: Any,
    duration: Any,
    show: bool,
) -> tuple["Figure", "Axes3D"]:
    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor="#07111D")
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        if not hasattr(ax, "zaxis"):
            raise TypeError("view='3d' requires a Matplotlib 3D axis")

    _draw_earth_globe(ax)

    trail_width = 2.5 if len(windows) <= 4 else 2.0
    handles: list[Line2D] = []
    all_points = [window["points_km"] for window in windows]
    for window in windows:
        color = window["color"]
        marker_xyz = window["marker_xyz"]
        _plot_3d_orbit_trail(ax, window["points_km"], color, trail_width)
        halo = ax.scatter(
            marker_xyz[0],
            marker_xyz[1],
            marker_xyz[2],
            s=260,
            c=[color],
            alpha=0.13,
            linewidths=0.0,
            depthshade=False,
            clip_on=False,
            zorder=18,
        )
        marker = ax.scatter(
            marker_xyz[0],
            marker_xyz[1],
            marker_xyz[2],
            s=62,
            c=[color],
            edgecolors="white",
            linewidths=0.9,
            depthshade=False,
            clip_on=False,
            zorder=19,
        )
        for artist in (halo, marker):
            if hasattr(artist, "set_sort_zpos"):
                artist.set_sort_zpos(1.0e9)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=trail_width,
                marker="o",
                markersize=6.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=window["label"],
            )
        )

    span_km = float(max(_EARTH_RADIUS_KM * 1.18, np.max(np.abs(np.concatenate(all_points, axis=0))) * 1.08))
    _style_3d_axes(ax, span_km)
    ax.view_init(elev=float(elev), azim=float(azim))
    if title is not None:
        ax.set_title(
            title,
            color="#E8F6FF",
            fontsize=16,
            pad=16.0,
        )

    if len(windows) == 1:
        window = windows[0]
        _add_single_orbit_3d_annotations(
            fig,
            ax,
            window["orbit"],
            window["marker_xyz"],
            window["start_s"],
            window["duration_s"],
            window["color"],
            frame,
        )
        info_line = "Marker shows the evaluated spacecraft state at the end of the plotted trail"
    else:
        info_line = (
            "Per-orbit default window: epoch through one Keplerian period"
            if start_time is None and duration is None
            else "Markers show the end of each plotted trail"
        )

    fig.text(0.5, 0.03, info_line, color="#9DC7DA", fontsize=10, ha="center", va="center")

    if len(windows) > 1:
        legend = ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10, borderpad=0.7, handlelength=2.3)
        _style_legend(legend)

    if show:
        plt.show()
    return fig, ax


def _render_2d(
    windows: list[dict[str, Any]],
    *,
    ax: Any | None,
    figsize: tuple[float, float],
    title: str | None,
    start_time: Any,
    duration: Any,
    show: bool,
) -> tuple["Figure", GeoAxes]:
    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor="#07111D")
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    else:
        fig = ax.figure
        if hasattr(ax, "zaxis") or not isinstance(ax, GeoAxes):
            raise TypeError("view='2d' requires a Cartopy GeoAxes or no axis")

    _style_2d_axes(ax)
    handles: list[Line2D] = []
    for window in windows:
        color = window["color"]
        for lon_seg, lat_seg in _split_dateline_segments(window["lon_deg"], window["lat_deg"]):
            ax.plot(
                lon_seg,
                lat_seg,
                transform=ccrs.PlateCarree(),
                color=color,
                linewidth=2.2 if len(windows) <= 4 else 1.8,
                alpha=0.9,
                zorder=3,
            )

        marker_lat, marker_lon = window["marker_latlon"]
        ax.scatter(
            [marker_lon],
            [marker_lat],
            transform=ccrs.PlateCarree(),
            s=180,
            c=[color],
            alpha=0.15,
            linewidths=0.0,
            zorder=4,
        )
        ax.scatter(
            [marker_lon],
            [marker_lat],
            transform=ccrs.PlateCarree(),
            s=42,
            c=[color],
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.2,
                marker="o",
                markersize=6.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=window["label"],
            )
        )

    if title is not None:
        ax.set_title(
            title,
            color="#E8F6FF",
            fontsize=16,
            pad=14.0,
        )

    if len(windows) == 1:
        window = windows[0]
        _add_single_orbit_2d_annotations(fig, window["orbit"], window["start_s"], window["duration_s"])
        info_line = "Marker shows the evaluated spacecraft ground-track position at the end of the plotted trail"
    else:
        info_line = (
            "Per-orbit default window: epoch through one Keplerian period"
            if start_time is None and duration is None
            else "Markers show the end of each plotted ground track"
        )
        legend = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=10, borderpad=0.7, handlelength=2.2)
        _style_legend(legend)

    fig.text(0.5, 0.03, info_line, color="#9DC7DA", fontsize=10, ha="center", va="center")

    if show:
        plt.show()
    return fig, ax


def plot_orbits(
    orbits: Any,
    *,
    start_time: Any = None,
    duration: Any = None,
    frame: Any = "gcrf",
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    samples: int = 360,
    ax: Any | None = None,
    figsize: tuple[float, float] = (9.5, 8.8),
    title: str | None = None,
    view: str = "3d",
    elev: float = 24.0,
    azim: float = 42.0,
    show: bool = True,
) -> tuple["Figure", Any]:
    """Plot one or more orbits as either a 3D Earth view or a 2D ground track."""

    orbit_list = _coerce_orbit_list(orbits)
    label_list = _coerce_labels(labels, len(orbit_list))
    color_list = _coerce_colors(colors, len(orbit_list))
    sample_count = int(samples)
    if sample_count < 2:
        raise ValueError("samples must be >= 2")

    windows = _sample_plot_windows(
        orbit_list,
        start_time=start_time,
        duration=duration,
        frame=frame,
        labels=label_list,
        colors=color_list,
        samples=sample_count,
    )

    if _coerce_view(view) == "3d":
        return _render_3d(
            windows,
            ax=ax,
            figsize=figsize,
            title=title,
            elev=elev,
            azim=azim,
            frame=frame,
            start_time=start_time,
            duration=duration,
            show=show,
        )
    return _render_2d(
        windows,
        ax=ax,
        figsize=figsize,
        title=title,
        start_time=start_time,
        duration=duration,
        show=show,
    )


__all__ = ["plot_orbits"]
