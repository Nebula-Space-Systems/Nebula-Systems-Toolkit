"""2D and 3D Matplotlib orbit-visualization helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, overload

import astropy.units as u
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from astropy.time import Time
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from nstk.time_utils import normalize_time_to_epoch_seconds

from ._cartopy_data import (
    configure_cartopy_data_dir,
    get_natural_earth_shapefile,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from nstk.propagation.orbit import Orbit
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from org.orekit.time import AbsoluteDate  # type: ignore


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
_MAX_LEGEND_ITEMS = 24
_DEFAULT_2D_MARKER_SIZE = math.sqrt(42.0)
_DEFAULT_3D_MARKER_SIZE = math.sqrt(62.0)
_DEFAULT_2D_LEGEND_MARKER_SIZE = 6.0
_DEFAULT_3D_LEGEND_MARKER_SIZE = 6.5
_MARKER_HALO_SCALE = 2.05
_PLOT_3D_FRAME = "itrf"
_VISIBLE_3D_OCCLUSION_RADIUS_KM = _EARTH_RADIUS_KM * 1.0005
_HIDDEN_3D_MARKER_ALPHA = 0.78
_HIDDEN_3D_CONNECTOR_ALPHA = 0.24
_VIEW_KEY_ROUND_DIGITS = 3


@dataclass
class _Orbit3DArtistSet:
    glow: Line3DCollection
    core: Line3DCollection
    halo: Any
    marker: Any
    connector: Line3DCollection | None
    points_km: np.ndarray
    marker_xyz: np.ndarray
    color: str
    opacity: float


@dataclass
class _Orbit3DSceneState:
    ax: Any
    coast_artist: Line3DCollection | None
    country_artist: Line3DCollection | None
    coast_segments: tuple[np.ndarray, ...]
    country_segments: tuple[np.ndarray, ...]
    orbit_artists: list[_Orbit3DArtistSet]
    redraw_pending: bool = False
    last_view_key: tuple[float, float] | None = None


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


def _format_frame_label(frame: Any) -> str:
    get_name = getattr(frame, "getName", None)
    if callable(get_name):
        name = str(get_name()).strip()
        if name:
            return name.lower()
    return str(frame).strip().lower()


def _extract_keplerian_summary(orbit: Any) -> dict[str, str]:
    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore

    state0 = orbit.propagator.getInitialState()
    date0 = state0.getDate()
    frame0 = state0.getFrame()
    pv0 = state0.getPVCoordinates(frame0)
    mu = float(state0.getOrbit().getMu())

    if frame0.isPseudoInertial():
        elements_frame = frame0
        pv_elements = pv0
    else:
        elements_frame = FramesFactory.getGCRF()
        tr = frame0.getTransformTo(elements_frame, date0)
        pv_elements = tr.transformPVCoordinates(pv0)

    kep = KeplerianOrbit(CartesianOrbit(pv_elements, elements_frame, date0, mu))
    return {
        "a": f"{float(kep.getA()) / 1000.0:,.1f} km",
        "e": f"{float(kep.getE()):.6f}",
        "i": _format_angle_deg(kep.getI()),
        "raan": _format_angle_deg(kep.getRightAscensionOfAscendingNode()),
        "argp": _format_angle_deg(kep.getPerigeeArgument()),
        "nu": _format_angle_deg(kep.getTrueAnomaly()),
        "frame": _format_frame_label(elements_frame),
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


def _coerce_opacity(opacity: float) -> float:
    alpha = float(opacity)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("opacity must be finite and between 0 and 1")
    return alpha


def _coerce_positive_style_value(name: str, value: float | None) -> float | None:
    if value is None:
        return None

    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return coerced


def _default_trail_width(view: str, orbit_count: int) -> float:
    if view == "3d":
        return 2.5 if orbit_count <= 4 else 2.0
    return 2.2 if orbit_count <= 4 else 1.8


def _default_marker_diameter(view: str) -> float:
    return _DEFAULT_3D_MARKER_SIZE if view == "3d" else _DEFAULT_2D_MARKER_SIZE


def _default_legend_marker_size(view: str) -> float:
    return _DEFAULT_3D_LEGEND_MARKER_SIZE if view == "3d" else _DEFAULT_2D_LEGEND_MARKER_SIZE


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
    facecolors[..., 3] = 1.0
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


def _make_3d_line_collection(
    *,
    gid: str,
    color: str | tuple[float, float, float] | tuple[float, float, float, float],
    linewidth: float,
    alpha: float,
    zorder: float,
    linestyle: str | tuple[Any, ...] | None = None,
) -> Line3DCollection:
    kwargs: dict[str, Any] = {
        "colors": [color],
        "linewidths": linewidth,
        "alpha": alpha,
        "zorder": zorder,
    }
    if linestyle is not None:
        kwargs["linestyles"] = linestyle
    artist = Line3DCollection([], **kwargs)
    artist.set_gid(gid)
    if hasattr(artist, "set_sort_zpos"):
        artist.set_sort_zpos(1.0e9)
    return artist


def _make_view_key(ax: Any) -> tuple[float, float]:
    return (
        round(float(ax.elev), _VIEW_KEY_ROUND_DIGITS),
        round(float(ax.azim), _VIEW_KEY_ROUND_DIGITS),
    )


def _update_surface_outline_artist(
    artist: Line3DCollection | None,
    segments: Sequence[np.ndarray],
    view_dir: np.ndarray,
) -> None:
    if artist is None:
        return
    artist.set_segments(_visible_surface_3d_segments(segments, view_dir))


def _set_marker_artist_style(marker: Any, color: str, opacity: float, *, visible: bool) -> None:
    if visible:
        marker.set_alpha(opacity)
        marker.set_facecolors([to_rgba(color, 1.0)])
        marker.set_edgecolors([to_rgba("white", 0.98)])
        marker.set_linewidths([0.9])
        return

    marker.set_alpha(_HIDDEN_3D_MARKER_ALPHA * opacity)
    marker.set_facecolors([(0.0, 0.0, 0.0, 0.0)])
    marker.set_edgecolors([to_rgba("#EAF8FF", 0.98)])
    marker.set_linewidths([1.35])


def _update_3d_orbit_artists(artist_set: _Orbit3DArtistSet, view_dir: np.ndarray) -> bool:
    visible = _points_visible_from_view(
        artist_set.points_km,
        view_dir,
        _VISIBLE_3D_OCCLUSION_RADIUS_KM,
    )
    segments = _split_visible_3d_segments(artist_set.points_km, visible)
    artist_set.glow.set_segments(segments)
    artist_set.core.set_segments(segments)

    marker_visible = bool(visible[-1])
    artist_set.halo.set_alpha((0.13 if marker_visible else 0.0) * artist_set.opacity)
    _set_marker_artist_style(
        artist_set.marker,
        artist_set.color,
        artist_set.opacity,
        visible=marker_visible,
    )

    if artist_set.connector is not None:
        artist_set.connector.set_segments([np.vstack((np.zeros(3), artist_set.marker_xyz))])
        artist_set.connector.set_alpha(
            (0.45 if marker_visible else _HIDDEN_3D_CONNECTOR_ALPHA) * artist_set.opacity
        )
    return marker_visible


def _update_3d_scene(state: _Orbit3DSceneState) -> bool:
    view_key = _make_view_key(state.ax)
    if state.last_view_key == view_key:
        return False

    state.last_view_key = view_key
    view_dir = _view_direction_from_angles(state.ax.elev, state.ax.azim)
    _update_surface_outline_artist(state.coast_artist, state.coast_segments, view_dir)
    _update_surface_outline_artist(state.country_artist, state.country_segments, view_dir)
    for artist_set in state.orbit_artists:
        _update_3d_orbit_artists(artist_set, view_dir)
    return True


def _install_3d_scene_sync(fig: Any, state: _Orbit3DSceneState) -> None:
    canvas = fig.canvas

    def _sync(event: Any | None = None) -> None:
        if event is not None and getattr(event, "inaxes", state.ax) not in (None, state.ax):
            return
        if _update_3d_scene(state) and not state.redraw_pending:
            state.redraw_pending = True
            canvas.draw_idle()

    def _on_draw(event: Any) -> None:
        if event.canvas is not canvas:
            return
        state.redraw_pending = False
        _sync()

    canvas.mpl_connect("draw_event", _on_draw)
    canvas.mpl_connect("motion_notify_event", _sync)
    canvas.mpl_connect("button_release_event", _sync)
    canvas.mpl_connect("scroll_event", _sync)
    _sync()


def _draw_earth_globe(
    ax: Any,
    view_dir: np.ndarray,
    resolution: int = 112,
) -> tuple[Line3DCollection | None, Line3DCollection | None]:
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
    coast_artist: Line3DCollection | None = None
    if coast_segments:
        coast_artist = _make_3d_line_collection(
            gid="nstk-earth-coast",
            color=(0.94, 0.985, 1.0),
            linewidth=0.55,
            alpha=0.55,
            zorder=2,
        )
        ax.add_collection3d(coast_artist, autolim=False)
        _update_surface_outline_artist(coast_artist, coast_segments, view_dir)

    country_segments = _earth_outline_xyz_segments(
        "110m",
        "country",
        radius_scale=1.0002,
        max_points_per_segment=200,
    )
    country_artist: Line3DCollection | None = None
    if country_segments:
        country_artist = _make_3d_line_collection(
            gid="nstk-earth-country",
            color=(0.76, 0.93, 1.0),
            linewidth=0.32,
            alpha=0.3,
            zorder=2,
        )
        ax.add_collection3d(country_artist, autolim=False)
        _update_surface_outline_artist(country_artist, country_segments, view_dir)
    return coast_artist, country_artist


def _create_3d_orbit_artists(
    ax: Any,
    *,
    index: int,
    points_km: np.ndarray,
    marker_xyz: np.ndarray,
    color: str,
    linewidth: float,
    opacity: float,
    marker_area: float,
    halo_size: float,
    show_connector: bool,
) -> _Orbit3DArtistSet:
    glow = _make_3d_line_collection(
        gid=f"nstk-orbit-trail-glow-{index}",
        color=color,
        linewidth=linewidth * 2.0,
        alpha=0.18 * opacity,
        zorder=14,
    )
    core = _make_3d_line_collection(
        gid=f"nstk-orbit-trail-core-{index}",
        color=color,
        linewidth=linewidth,
        alpha=0.96 * opacity,
        zorder=15,
    )
    ax.add_collection3d(glow, autolim=False)
    ax.add_collection3d(core, autolim=False)

    connector = None
    if show_connector:
        connector = _make_3d_line_collection(
            gid=f"nstk-orbit-connector-{index}",
            color=color,
            linewidth=1.2,
            alpha=0.45 * opacity,
            zorder=16,
            linestyle=(0, (4, 3)),
        )
        ax.add_collection3d(connector, autolim=False)

    halo = ax.scatter(
        marker_xyz[0],
        marker_xyz[1],
        marker_xyz[2],
        s=halo_size,
        c=[color],
        alpha=0.13 * opacity,
        linewidths=0.0,
        depthshade=False,
        clip_on=False,
        zorder=18,
    )
    halo.set_gid(f"nstk-orbit-halo-{index}")
    marker = ax.scatter(
        marker_xyz[0],
        marker_xyz[1],
        marker_xyz[2],
        s=marker_area,
        c=[color],
        alpha=opacity,
        edgecolors="white",
        linewidths=0.9,
        depthshade=False,
        clip_on=False,
        zorder=19,
    )
    marker.set_gid(f"nstk-orbit-marker-{index}")
    for artist in (halo, marker):
        if hasattr(artist, "set_sort_zpos"):
            artist.set_sort_zpos(1.0e9)

    return _Orbit3DArtistSet(
        glow=glow,
        core=core,
        halo=halo,
        marker=marker,
        connector=connector,
        points_km=np.asarray(points_km, dtype=np.float64),
        marker_xyz=np.asarray(marker_xyz, dtype=np.float64),
        color=color,
        opacity=float(opacity),
    )


def _view_direction_from_angles(elev: float, azim: float) -> np.ndarray:
    elev_rad = np.radians(float(elev))
    azim_rad = np.radians(float(azim))
    return np.asarray(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=np.float64,
    )


def _points_visible_from_view(
    points_km: np.ndarray,
    view_dir: np.ndarray,
    occlusion_radius_km: float,
) -> np.ndarray:
    points = np.asarray(points_km, dtype=np.float64)
    view = np.asarray(view_dir, dtype=np.float64).reshape(3)
    depth = np.sum(points * view[None, :], axis=1, dtype=np.float64)
    radial_sq = np.maximum(np.sum(points * points, axis=1, dtype=np.float64) - depth * depth, 0.0)
    occlusion_sq = float(occlusion_radius_km) ** 2
    outside_disk = radial_sq >= occlusion_sq
    surface_depth = np.sqrt(np.maximum(occlusion_sq - radial_sq, 0.0))
    return outside_disk | (depth >= surface_depth)


def _surface_points_visible_from_view(points_km: np.ndarray, view_dir: np.ndarray) -> np.ndarray:
    points = np.asarray(points_km, dtype=np.float64)
    view = np.asarray(view_dir, dtype=np.float64).reshape(3)
    depth = np.sum(points * view[None, :], axis=1, dtype=np.float64)
    return depth >= 0.0


def _visible_surface_3d_segments(
    segments: Sequence[np.ndarray],
    view_dir: np.ndarray,
) -> list[np.ndarray]:
    visible_segments: list[np.ndarray] = []
    for segment in segments:
        visible = _surface_points_visible_from_view(segment, view_dir)
        visible_segments.extend(_split_visible_3d_segments(segment, visible))
    return visible_segments


def _split_visible_3d_segments(points_km: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    if points_km.shape[0] < 2:
        return []

    segments: list[np.ndarray] = []
    start_idx: int | None = None
    for idx, is_visible in enumerate(np.asarray(visible, dtype=bool)):
        if is_visible and start_idx is None:
            start_idx = idx
        elif not is_visible and start_idx is not None:
            if idx - start_idx >= 2:
                segments.append(np.asarray(points_km[start_idx:idx], dtype=np.float64))
            start_idx = None

    if start_idx is not None and points_km.shape[0] - start_idx >= 2:
        segments.append(np.asarray(points_km[start_idx:], dtype=np.float64))
    return segments


def _style_3d_axes(ax: Any, span_km: float) -> None:
    bg = "#07111D"
    axis_line = (0.75, 0.9, 1.0, 0.08)
    grid_line = (0.75, 0.9, 1.0, 0.05)

    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False

    ax.set_facecolor(bg)
    ax.figure.set_facecolor(bg)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-span_km, span_km)
    ax.set_ylim(-span_km, span_km)
    ax.set_zlim(-span_km, span_km)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(axis_line)
        axis.line.set_linewidth(0.55)
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = grid_line
        axis._axinfo["grid"]["linewidth"] = 0.45
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


def _add_bottom_legend(fig: Any, ax: Any, handles: Sequence[Line2D]) -> None:
    if len(handles) > _MAX_LEGEND_ITEMS:
        return

    legend_columns = min(len(handles), 6, max(4, math.ceil(len(handles) / 4)))
    legend_rows = max(1, math.ceil(len(handles) / legend_columns))
    legend_bottom = 0.055
    legend_band_height = 0.055 + 0.04 * (legend_rows - 1)
    legend_top = legend_bottom + legend_band_height

    ax_position = ax.get_position()
    required_axes_bottom = legend_top + 0.015
    if ax_position.y0 < required_axes_bottom:
        shift = required_axes_bottom - ax_position.y0
        ax.set_position(
            [
                ax_position.x0,
                ax_position.y0 + shift,
                ax_position.width,
                ax_position.height - shift,
            ]
        )

    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_bottom),
        bbox_transform=fig.transFigure,
        ncol=legend_columns,
        frameon=True,
        fontsize=10,
        borderpad=0.7,
        handlelength=2.2,
        columnspacing=1.4,
        labelspacing=0.7,
    )
    legend.set_in_layout(False)
    _style_legend(legend)


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
    orbit: Any,
    start_s: float,
    duration_s: float,
) -> None:
    eval_time = orbit.epoch + (start_s + duration_s) * u.s
    summary = _extract_keplerian_summary(orbit)

    left_text, right_text = _make_info_text(
        summary,
        [
            "Time Context",
            f"Evaluation   {_format_time_label(eval_time)} UTC",
            f"Start offset {start_s / 60.0:,.2f} min",
            f"Trail window {duration_s / 60.0:,.2f} min",
            "Marker       filled=near side, hollow=far side",
        ],
        left_lines=[
            f"Epoch  {_format_time_label(orbit.epoch)} UTC",
            f"Frame  {summary['frame']}",
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
    view: str,
    start_time: Any,
    duration: Any,
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
        window = {
            "orbit": orbit,
            "label": label,
            "color": color,
            "start_s": start_s,
            "duration_s": duration_s,
            "query_s": query_s,
        }

        if view == "3d":
            points_km = np.asarray(orbit.get_p_np(query_s, frame=_PLOT_3D_FRAME), dtype=np.float64) / 1000.0
            if points_km.ndim != 2 or points_km.shape[1] != 3:
                raise ValueError("orbit.get_p_np returned an unexpected shape")
            window["points_km"] = points_km
            window["marker_xyz"] = points_km[-1]
        else:
            lat_deg, lon_deg, alt_m = orbit.get_geodetic_np(query_s)
            lat_arr = np.asarray(lat_deg, dtype=np.float64)
            lon_arr = np.asarray(lon_deg, dtype=np.float64)
            window["lat_deg"] = lat_arr
            window["lon_deg"] = lon_arr
            window["alt_m"] = np.asarray(alt_m, dtype=np.float64)
            window["marker_latlon"] = (
                float(lat_arr[-1]),
                float(lon_arr[-1]),
            )

        windows.append(window)
    return windows


def _render_3d(
    windows: list[dict[str, Any]],
    *,
    ax: Any | None,
    figsize: tuple[float, float],
    title: str | None,
    elev: float,
    azim: float,
    start_time: Any,
    duration: Any,
    opacity: float,
    line_width: float | None,
    marker_size: float | None,
    show: bool,
) -> tuple["Figure", "Axes3D"]:
    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor="#07111D")
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        if not hasattr(ax, "zaxis"):
            raise TypeError("view='3d' requires a Matplotlib 3D axis")

    view_dir = _view_direction_from_angles(elev, azim)
    coast_artist, country_artist = _draw_earth_globe(ax, view_dir)

    trail_width = _default_trail_width("3d", len(windows)) if line_width is None else line_width
    marker_diameter = _default_marker_diameter("3d") if marker_size is None else marker_size
    halo_size = (marker_diameter * _MARKER_HALO_SCALE) ** 2
    marker_area = marker_diameter**2
    legend_marker_size = _default_legend_marker_size("3d") if marker_size is None else marker_diameter
    handles: list[Line2D] = []
    all_points = [window["points_km"] for window in windows]
    orbit_artists: list[_Orbit3DArtistSet] = []
    for idx, window in enumerate(windows):
        color = window["color"]
        orbit_artists.append(
            _create_3d_orbit_artists(
                ax,
                index=idx,
                points_km=window["points_km"],
                marker_xyz=window["marker_xyz"],
                color=color,
                linewidth=trail_width,
                opacity=opacity,
                marker_area=marker_area,
                halo_size=halo_size,
                show_connector=(len(windows) == 1),
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=trail_width,
                marker="o",
                markersize=legend_marker_size,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=window["label"],
            )
        )

    span_km = float(max(_EARTH_RADIUS_KM * 1.18, np.max(np.abs(np.concatenate(all_points, axis=0))) * 1.08))
    _style_3d_axes(ax, span_km)
    ax.view_init(elev=float(elev), azim=float(azim))
    scene_state = _Orbit3DSceneState(
        ax=ax,
        coast_artist=coast_artist,
        country_artist=country_artist,
        coast_segments=_earth_outline_xyz_segments(
            "110m",
            "coastline",
            radius_scale=1.0002,
            max_points_per_segment=260,
        ),
        country_segments=_earth_outline_xyz_segments(
            "110m",
            "country",
            radius_scale=1.0002,
            max_points_per_segment=200,
        ),
        orbit_artists=orbit_artists,
    )
    _update_3d_scene(scene_state)
    ax._nstk_3d_scene_state = scene_state
    _install_3d_scene_sync(fig, scene_state)
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
            window["orbit"],
            window["start_s"],
            window["duration_s"],
        )
        info_line = (
            "Marker shows the evaluated spacecraft state at the end of the plotted trail; "
            "a hollow ring means the state is on the far side of the Earth"
        )
    else:
        info_line = (
            "Per-orbit default window: epoch through one Keplerian period"
            if start_time is None and duration is None
            else "Markers show the end of each plotted trail; hollow rings mark far-side states"
        )

    fig.text(0.5, 0.03, info_line, color="#9DC7DA", fontsize=10, ha="center", va="center")

    if len(windows) > 1:
        _add_bottom_legend(fig, ax, handles)

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
    opacity: float,
    line_width: float | None,
    marker_size: float | None,
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
    trail_width = _default_trail_width("2d", len(windows)) if line_width is None else line_width
    marker_diameter = _default_marker_diameter("2d") if marker_size is None else marker_size
    halo_size = (marker_diameter * _MARKER_HALO_SCALE) ** 2
    marker_area = marker_diameter**2
    legend_marker_size = _default_legend_marker_size("2d") if marker_size is None else marker_diameter
    handles: list[Line2D] = []
    for window in windows:
        color = window["color"]
        for lon_seg, lat_seg in _split_dateline_segments(window["lon_deg"], window["lat_deg"]):
            ax.plot(
                lon_seg,
                lat_seg,
                transform=ccrs.PlateCarree(),
                color=color,
                linewidth=trail_width,
                alpha=0.9 * opacity,
                zorder=3,
            )

        marker_lat, marker_lon = window["marker_latlon"]
        ax.scatter(
            [marker_lon],
            [marker_lat],
            transform=ccrs.PlateCarree(),
            s=halo_size,
            c=[color],
            alpha=0.15 * opacity,
            linewidths=0.0,
            zorder=4,
        )
        ax.scatter(
            [marker_lon],
            [marker_lat],
            transform=ccrs.PlateCarree(),
            s=marker_area,
            c=[color],
            alpha=opacity,
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=trail_width,
                marker="o",
                markersize=legend_marker_size,
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
        _add_bottom_legend(fig, ax, handles)

    fig.text(0.5, 0.03, info_line, color="#9DC7DA", fontsize=10, ha="center", va="center")

    if show:
        plt.show()
    return fig, ax


@overload
def plot_orbits(
    orbits: Orbit | Iterable[Orbit],
    *,
    start_time: Time | AbsoluteDate | float | int | u.Quantity | None = None,
    duration: float | int | u.Quantity | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    samples: int = 360,
    ax: GeoAxes | None = None,
    figsize: tuple[float, float] = (9.5, 8.8),
    title: str | None = None,
    view: Literal["2d"],
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show: bool = True,
) -> tuple[Figure, GeoAxes]: ...


@overload
def plot_orbits(
    orbits: Orbit | Iterable[Orbit],
    *,
    start_time: Time | AbsoluteDate | float | int | u.Quantity | None = None,
    duration: float | int | u.Quantity | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    samples: int = 360,
    ax: Axes3D | None = None,
    figsize: tuple[float, float] = (9.5, 8.8),
    title: str | None = None,
    view: Literal["3d"] = "3d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show: bool = True,
) -> tuple[Figure, Axes3D]: ...


@overload
def plot_orbits(
    orbits: Orbit | Iterable[Orbit],
    *,
    start_time: Time | AbsoluteDate | float | int | u.Quantity | None = None,
    duration: float | int | u.Quantity | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    samples: int = 360,
    ax: GeoAxes | Axes3D | None = None,
    figsize: tuple[float, float] = (9.5, 8.8),
    title: str | None = None,
    view: Literal["2d", "3d"] = "3d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show: bool = True,
) -> tuple[Figure, GeoAxes | Axes3D]: ...


def plot_orbits(
    orbits: Orbit | Iterable[Orbit],
    *,
    start_time: Time | AbsoluteDate | float | int | u.Quantity | None = None,
    duration: float | int | u.Quantity | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    samples: int = 360,
    ax: GeoAxes | Axes3D | None = None,
    figsize: tuple[float, float] = (9.5, 8.8),
    title: str | None = None,
    view: Literal["2d", "3d"] = "3d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show: bool = True,
) -> tuple[Figure, GeoAxes | Axes3D]:
    """Plot one or more orbits as either a 3D Earth view or a 2D ground track.

    Parameters
    ----------
    orbits
        Single orbit or iterable of orbit objects to plot.
    start_time
        Optional scalar evaluation start time. Numeric values are seconds from
        the orbit epoch.
    duration
        Optional trail duration. Defaults to one Keplerian period per orbit.
    labels, colors
        Optional per-orbit legend labels and colors.
    samples
        Number of sample points per plotted trail. Must be at least 2.
    ax
        Existing Matplotlib axis to draw into. Omit to create a new figure.
    figsize
        Figure size used when ``ax`` is not provided.
    title
        Optional plot title.
    view
        ``"3d"`` for an Earth globe view rendered in ITRF or ``"2d"`` for a
        ground-track map.
    elev, azim
        3D camera angles in degrees.
    opacity
        Global alpha multiplier applied to orbit trails and endpoint markers in
        both views. Must be between 0 and 1.
    line_width
        Optional orbit trail line width in points. Defaults to the built-in
        view-aware width.
    marker_size
        Optional endpoint marker diameter in points. The outer halo scales with
        this value automatically.
    show
        If ``True``, call ``matplotlib.pyplot.show()`` before returning.
    """

    orbit_list = _coerce_orbit_list(orbits)
    label_list = _coerce_labels(labels, len(orbit_list))
    color_list = _coerce_colors(colors, len(orbit_list))
    view_key = _coerce_view(view)
    sample_count = int(samples)
    alpha = _coerce_opacity(opacity)
    trail_width = _coerce_positive_style_value("line_width", line_width)
    marker_diameter = _coerce_positive_style_value("marker_size", marker_size)
    if sample_count < 2:
        raise ValueError("samples must be >= 2")

    windows = _sample_plot_windows(
        orbit_list,
        view=view_key,
        start_time=start_time,
        duration=duration,
        labels=label_list,
        colors=color_list,
        samples=sample_count,
    )

    if view_key == "3d":
        return _render_3d(
            windows,
            ax=ax,
            figsize=figsize,
            title=title,
            elev=elev,
            azim=azim,
            start_time=start_time,
            duration=duration,
            opacity=alpha,
            line_width=trail_width,
            marker_size=marker_diameter,
            show=show,
        )
    return _render_2d(
        windows,
        ax=ax,
        figsize=figsize,
        title=title,
        start_time=start_time,
        duration=duration,
        opacity=alpha,
        line_width=trail_width,
        marker_size=marker_diameter,
        show=show,
    )


__all__ = ["plot_orbits"]
