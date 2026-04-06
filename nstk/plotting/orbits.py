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

from .geo import GeoMap
from .map import MapStyle, NSTK_DEFAULT_COLOR_CYCLE, NSTK_HIGHLIGHT_ORANGE, get_map_style
from ._cartopy_data import (
    configure_cartopy_data_dir,
    get_natural_earth_shapefile,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from nstk.plotting.map import MapView
    from nstk.propagation.orbit import Orbit
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from org.orekit.time import AbsoluteDate  # type: ignore


configure_cartopy_data_dir()

_EARTH_RADIUS_M = 6378137.0
_EARTH_RADIUS_KM = _EARTH_RADIUS_M / 1000.0
_DEFAULT_COLORS = (
    NSTK_HIGHLIGHT_ORANGE,
    "#55C1FF",
    "#9D7CFF",
    "#3DDC97",
    "#FFD166",
    "#FF5DA2",
    "#A1E44D",
    "#C792EA",
)
_REPEATED_COLOR_STYLE_VARIANTS: tuple[tuple[str | tuple[Any, ...], str], ...] = (
    ("-", "o"),
    ((0, (7.0, 2.2)), "s"),
    ((0, (4.2, 1.8, 1.2, 1.8)), "D"),
    ((0, (1.2, 1.8)), "^"),
    ((0, (8.0, 2.2, 1.2, 2.0, 1.2, 2.0)), "v"),
    ((0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)), "P"),
    ((0, (9.0, 2.2)), "X"),
    ((0, (2.4, 1.4)), "h"),
)
_MAX_LEGEND_ITEMS = 24
_DEFAULT_2D_MARKER_SIZE = math.sqrt(42.0)
_DEFAULT_3D_MARKER_SIZE = math.sqrt(62.0)
_DEFAULT_2D_LEGEND_MARKER_SIZE = 6.0
_DEFAULT_3D_LEGEND_MARKER_SIZE = 6.5
_DEFAULT_2D_FIGSIZE = (10.5, 6.2)
_DEFAULT_3D_FIGSIZE = (10.5, 7.0)
_MARKER_HALO_SCALE = 2.05
_PLOT_3D_FRAME = "itrf"
_VISIBLE_3D_OCCLUSION_RADIUS_KM = _EARTH_RADIUS_KM * 1.0005
_HIDDEN_3D_MARKER_ALPHA = 0.78
_HIDDEN_3D_CONNECTOR_ALPHA = 0.24
_VIEW_KEY_ROUND_DIGITS = 3
_EARTH_GLOBE_IDLE_LON_SAMPLES = 240
_EARTH_GLOBE_IDLE_LAT_SAMPLES = 121
_EARTH_GLOBE_DRAG_LON_SAMPLES = 160
_EARTH_GLOBE_DRAG_LAT_SAMPLES = 81
_ANNOTATION_FONT_FAMILY = "DejaVu Sans Mono"
_ANNOTATION_LINE_SPACING = 1.18
_ANNOTATION_BAND_BOTTOM = 0.07
_ANNOTATION_AXES_GAP = 0.018


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
    marker_edge_color: tuple[float, float, float, float]


@dataclass(frozen=True)
class _OrbitTraceStyle:
    color: str
    line_style: str | tuple[Any, ...]
    marker: str


@dataclass
class _Orbit3DSceneState:
    ax: Any
    globe_artist: Any
    globe_drag_artist: Any | None
    coast_artist: Line3DCollection | None
    country_artist: Line3DCollection | None
    coast_segments: tuple[np.ndarray, ...]
    country_segments: tuple[np.ndarray, ...]
    orbit_artists: list[_Orbit3DArtistSet]
    redraw_pending: bool = False
    last_view_key: tuple[float, float] | None = None
    interacting: bool = False


@dataclass(frozen=True)
class _OrbitPlotStyle:
    map_style: MapStyle
    figure_face: tuple[float, float, float, float]
    axes_face: tuple[float, float, float, float]
    panel_face: tuple[float, float, float, float]
    panel_edge: tuple[float, float, float, float]
    primary_text: tuple[float, float, float, float]
    secondary_text: tuple[float, float, float, float]
    axis_line: tuple[float, float, float, float]
    grid_line: tuple[float, float, float, float]
    coast_color: tuple[float, float, float, float]
    coast_linewidth: float
    country_color: tuple[float, float, float, float]
    country_linewidth: float
    ocean_rgb: np.ndarray
    land_rgb: np.ndarray


def _mix_rgba(base: Any, other: Any, amount: float) -> tuple[float, float, float, float]:
    base_rgba = np.asarray(to_rgba(base), dtype=np.float64)
    other_rgba = np.asarray(to_rgba(other), dtype=np.float64)
    weight = float(np.clip(amount, 0.0, 1.0))
    mixed = (1.0 - weight) * base_rgba + weight * other_rgba
    return tuple(float(value) for value in mixed)


def _with_alpha(color: Any, alpha: float) -> tuple[float, float, float, float]:
    rgba = np.asarray(to_rgba(color), dtype=np.float64)
    rgba[3] = float(np.clip(alpha, 0.0, 1.0))
    return tuple(float(value) for value in rgba)


def _relative_luminance(color: Any) -> float:
    rgb = np.asarray(to_rgba(color)[:3], dtype=np.float64)
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def _resolve_orbit_plot_style(map_style: str | MapStyle | None) -> _OrbitPlotStyle:
    style = get_map_style("dark" if map_style is None else map_style)
    dark_background = _relative_luminance(style.theme.figure_face) < 0.45
    contrast = "white" if dark_background else "black"
    panel_blend = 0.08 if dark_background else 0.06
    panel_face = _mix_rgba(style.theme.axes_face, contrast, panel_blend)
    panel_edge_base = style.outline.color or style.theme.outline_edge
    grid_color = style.gridlines.color or style.theme.grid_color
    coast_color = style.coastlines.color or panel_edge_base
    country_color = style.borders.color or panel_edge_base
    grid_alpha = style.gridlines.alpha if style.gridlines.enabled else 0.0
    axis_line_alpha = 0.22 if dark_background else 0.18
    grid_line_alpha = 0.0 if grid_alpha <= 0.0 else min(0.32, 0.12 + 0.45 * float(grid_alpha))

    ocean_face = style.ocean.facecolor if style.ocean.enabled else style.theme.axes_face
    land_face = style.land.facecolor if style.land.enabled else ocean_face

    return _OrbitPlotStyle(
        map_style=style,
        figure_face=tuple(float(value) for value in to_rgba(style.theme.figure_face)),
        axes_face=tuple(float(value) for value in to_rgba(style.theme.axes_face)),
        panel_face=panel_face,
        panel_edge=_with_alpha(panel_edge_base, 0.9),
        primary_text=_with_alpha(style.theme.grid_label_color, 1.0),
        secondary_text=_with_alpha(style.theme.grid_label_color, 0.82),
        axis_line=_with_alpha(panel_edge_base, axis_line_alpha),
        grid_line=_with_alpha(grid_color, grid_line_alpha),
        coast_color=_with_alpha(
            coast_color,
            float(style.coastlines.alpha if style.coastlines.enabled else 0.0),
        ),
        coast_linewidth=float(style.coastlines.linewidth),
        country_color=_with_alpha(
            country_color,
            float(style.borders.alpha if style.borders.enabled else 0.0),
        ),
        country_linewidth=float(style.borders.linewidth),
        ocean_rgb=np.asarray(to_rgba(ocean_face)[:3], dtype=np.float64),
        land_rgb=np.asarray(to_rgba(land_face)[:3], dtype=np.float64),
    )


def _format_time_label(time_like: Any) -> str:
    return str(time_like.utc.isot).replace("T", " ")


def _format_angle_deg(angle_rad: float) -> str:
    angle_deg = (np.degrees(float(angle_rad)) + 360.0) % 360.0
    return f"{angle_deg:.3f} deg"


def _annotation_box_style(plot_style: _OrbitPlotStyle) -> dict[str, Any]:
    return {
        "boxstyle": "round,pad=0.55",
        "facecolor": plot_style.panel_face,
        "edgecolor": plot_style.panel_edge,
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


def _format_labeled_rows(rows: Sequence[tuple[str, str]]) -> list[str]:
    normalized_rows = [(str(label).strip(), str(value).strip()) for label, value in rows]
    if not normalized_rows:
        return []

    label_width = max(len(label) for label, _ in normalized_rows)
    return [f"{label:<{label_width}}  {value}" for label, value in normalized_rows]


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
    if hasattr(orbits, "get_p"):
        out = [orbits]
    elif isinstance(orbits, Iterable):
        out = list(orbits)
    else:
        raise TypeError("orbits must be an Orbit or an iterable of Orbit objects")

    if not out:
        raise ValueError("orbits must contain at least one Orbit")
    for orbit in out:
        if not hasattr(orbit, "get_p"):
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


def _assign_trace_styles(colors: Sequence[str]) -> list[_OrbitTraceStyle]:
    seen_by_color: dict[tuple[float, float, float, float], int] = {}
    styled: list[_OrbitTraceStyle] = []
    for color in colors:
        color_key = tuple(float(channel) for channel in to_rgba(color))
        occurrence = seen_by_color.get(color_key, 0)
        seen_by_color[color_key] = occurrence + 1
        line_style, marker = _REPEATED_COLOR_STYLE_VARIANTS[
            occurrence % len(_REPEATED_COLOR_STYLE_VARIANTS)
        ]
        styled.append(
            _OrbitTraceStyle(
                color=str(color),
                line_style=line_style,
                marker=marker,
            )
        )
    return styled


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


def _resolve_plot_figsize(view: str, figsize: tuple[float, float] | None) -> tuple[float, float]:
    if figsize is None:
        return _DEFAULT_3D_FIGSIZE if view == "3d" else _DEFAULT_2D_FIGSIZE

    width, height = float(figsize[0]), float(figsize[1])
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("figsize must contain positive finite width and height")
    return width, height


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


@lru_cache(maxsize=None)
def _land_geometry_union(resolution: str = "110m") -> Any:
    from shapely.ops import unary_union

    geometries = [
        geom
        for geom in _natural_earth_geometries(resolution, "physical", "land")
        if geom is not None and not geom.is_empty
    ]
    if not geometries:
        return None
    return unary_union(geometries)


def _earth_land_mask(uu: np.ndarray, vv: np.ndarray, *, resolution: str = "110m") -> np.ndarray:
    try:
        from shapely import contains_xy
    except Exception:
        return np.zeros(uu.shape, dtype=bool)

    land_geometry = _land_geometry_union(resolution)
    if land_geometry is None or getattr(land_geometry, "is_empty", False):
        return np.zeros(uu.shape, dtype=bool)

    lon_deg = ((np.degrees(uu) + 180.0) % 360.0) - 180.0
    lat_deg = 90.0 - np.degrees(vv)
    return np.asarray(contains_xy(land_geometry, lon_deg, lat_deg), dtype=bool)


@lru_cache(maxsize=None)
def _earth_globe_surface_mesh(
    lon_samples: int,
    lat_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_grid = np.linspace(0.0, 2.0 * np.pi, int(lon_samples), dtype=np.float64)
    v_grid = np.linspace(0.0, np.pi, int(lat_samples), dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing="xy")

    x = _EARTH_RADIUS_KM * np.cos(uu) * np.sin(vv)
    y = _EARTH_RADIUS_KM * np.sin(uu) * np.sin(vv)
    z = _EARTH_RADIUS_KM * np.cos(vv)
    return uu, vv, x, y, z


@lru_cache(maxsize=None)
def _earth_land_mask_for_surface(
    lon_samples: int,
    lat_samples: int,
    *,
    resolution: str = "110m",
) -> np.ndarray:
    uu, vv, _, _, _ = _earth_globe_surface_mesh(lon_samples, lat_samples)
    return _earth_land_mask(uu, vv, resolution=resolution)


def _make_earth_facecolors(
    uu: np.ndarray,
    vv: np.ndarray,
    *,
    land_mask: np.ndarray,
    plot_style: _OrbitPlotStyle,
) -> np.ndarray:
    nx = np.cos(uu) * np.sin(vv)
    ny = np.sin(uu) * np.sin(vv)
    nz = np.cos(vv)

    light = np.array([0.43, -0.22, 0.88], dtype=np.float64)
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)

    swirl = 0.5 + 0.5 * np.sin(2.6 * uu + 0.9 * np.cos(1.7 * vv))
    variation = np.clip(0.93 + 0.10 * swirl, 0.88, 1.04)
    ocean = np.asarray(plot_style.ocean_rgb, dtype=np.float64)
    land = np.asarray(plot_style.land_rgb, dtype=np.float64)
    mask = np.asarray(land_mask, dtype=bool)
    base = np.where(mask[..., None], land, ocean)
    base = np.clip(base * variation[..., None], 0.0, 1.0)

    brightness = 0.42 + 0.58 * diffuse
    ocean_specular = np.power(diffuse, 18.0) * 0.06

    facecolors = np.empty((*uu.shape, 4), dtype=np.float64)
    facecolors[..., :3] = np.clip(
        base * brightness[..., None] + (~mask)[..., None] * ocean_specular[..., None],
        0.0,
        1.0,
    )
    facecolors[..., 3] = 1.0
    return facecolors


def _add_earth_surface(
    ax: Any,
    *,
    plot_style: _OrbitPlotStyle,
    lon_samples: int,
    lat_samples: int,
    visible: bool,
    gid: str,
) -> Any:
    uu, vv, x, y, z = _earth_globe_surface_mesh(lon_samples, lat_samples)
    land_mask = _earth_land_mask_for_surface(lon_samples, lat_samples, resolution="110m")
    surface = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=_make_earth_facecolors(uu, vv, land_mask=land_mask, plot_style=plot_style),
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=0,
    )
    surface.set_gid(gid)
    surface.set_visible(visible)
    return surface


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


def _set_marker_artist_style(
    marker: Any,
    color: str,
    opacity: float,
    *,
    visible: bool,
    marker_edge_color: tuple[float, float, float, float],
) -> None:
    if visible:
        marker.set_alpha(opacity)
        marker.set_facecolors([to_rgba(color, 1.0)])
        marker.set_edgecolors([marker_edge_color])
        marker.set_linewidths([0.9])
        return

    marker.set_alpha(_HIDDEN_3D_MARKER_ALPHA * opacity)
    marker.set_facecolors([(0.0, 0.0, 0.0, 0.0)])
    marker.set_edgecolors([marker_edge_color])
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
        marker_edge_color=artist_set.marker_edge_color,
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


def _set_3d_globe_interaction(state: _Orbit3DSceneState, interacting: bool) -> bool:
    if state.interacting == interacting:
        return False
    state.interacting = interacting
    state.globe_artist.set_visible(not interacting)
    if state.globe_drag_artist is not None:
        state.globe_drag_artist.set_visible(interacting)
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

    def _request_draw() -> None:
        if not state.redraw_pending:
            state.redraw_pending = True
            canvas.draw_idle()

    def _on_press(event: Any) -> None:
        if getattr(event, "inaxes", None) is not state.ax:
            return
        if _set_3d_globe_interaction(state, True):
            _request_draw()

    def _on_release(event: Any) -> None:
        toggled = _set_3d_globe_interaction(state, False)
        if toggled:
            _request_draw()
        _sync(None)

    canvas.mpl_connect("draw_event", _on_draw)
    canvas.mpl_connect("button_press_event", _on_press)
    canvas.mpl_connect("motion_notify_event", _sync)
    canvas.mpl_connect("button_release_event", _on_release)
    canvas.mpl_connect("scroll_event", _sync)
    _sync()


def _draw_earth_globe(
    ax: Any,
    view_dir: np.ndarray,
    plot_style: _OrbitPlotStyle,
    idle_lon_samples: int = _EARTH_GLOBE_IDLE_LON_SAMPLES,
    idle_lat_samples: int = _EARTH_GLOBE_IDLE_LAT_SAMPLES,
    drag_lon_samples: int = _EARTH_GLOBE_DRAG_LON_SAMPLES,
    drag_lat_samples: int = _EARTH_GLOBE_DRAG_LAT_SAMPLES,
) -> tuple[Any, Any | None, Line3DCollection | None, Line3DCollection | None]:
    globe_artist = _add_earth_surface(
        ax,
        plot_style=plot_style,
        lon_samples=idle_lon_samples,
        lat_samples=idle_lat_samples,
        visible=True,
        gid="nstk-earth-globe",
    )
    globe_drag_artist: Any | None = None
    if (drag_lon_samples, drag_lat_samples) != (idle_lon_samples, idle_lat_samples):
        globe_drag_artist = _add_earth_surface(
            ax,
            plot_style=plot_style,
            lon_samples=drag_lon_samples,
            lat_samples=drag_lat_samples,
            visible=False,
            gid="nstk-earth-globe-interactive",
        )

    coast_segments = _earth_outline_xyz_segments(
        "110m",
        "coastline",
        radius_scale=1.0002,
        max_points_per_segment=260,
    )
    coast_artist: Line3DCollection | None = None
    if coast_segments and plot_style.map_style.coastlines.enabled:
        coast_artist = _make_3d_line_collection(
            gid="nstk-earth-coast",
            color=plot_style.coast_color[:3],
            linewidth=max(0.2, plot_style.coast_linewidth),
            alpha=plot_style.coast_color[3],
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
    if country_segments and plot_style.map_style.borders.enabled:
        country_artist = _make_3d_line_collection(
            gid="nstk-earth-country",
            color=plot_style.country_color[:3],
            linewidth=max(0.2, plot_style.country_linewidth),
            alpha=plot_style.country_color[3],
            zorder=2,
        )
        ax.add_collection3d(country_artist, autolim=False)
        _update_surface_outline_artist(country_artist, country_segments, view_dir)
    return globe_artist, globe_drag_artist, coast_artist, country_artist


def _create_3d_orbit_artists(
    ax: Any,
    *,
    index: int,
    points_km: np.ndarray,
    marker_xyz: np.ndarray,
    color: str,
    line_style: str | tuple[Any, ...],
    marker: str,
    linewidth: float,
    opacity: float,
    marker_area: float,
    halo_size: float,
    show_connector: bool,
    marker_edge_color: tuple[float, float, float, float],
) -> _Orbit3DArtistSet:
    glow = _make_3d_line_collection(
        gid=f"nstk-orbit-trail-glow-{index}",
        color=color,
        linewidth=linewidth * 2.0,
        alpha=0.18 * opacity,
        zorder=14,
        linestyle=line_style,
    )
    core = _make_3d_line_collection(
        gid=f"nstk-orbit-trail-core-{index}",
        color=color,
        linewidth=linewidth,
        alpha=0.96 * opacity,
        zorder=15,
        linestyle=line_style,
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
        marker=marker,
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
        edgecolors=[marker_edge_color],
        linewidths=0.9,
        marker=marker,
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
        marker_edge_color=marker_edge_color,
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


def _style_3d_axes(ax: Any, span_km: float, plot_style: _OrbitPlotStyle) -> None:
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False

    ax.set_facecolor(plot_style.axes_face)
    ax.figure.set_facecolor(plot_style.figure_face)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-span_km, span_km)
    ax.set_ylim(-span_km, span_km)
    ax.set_zlim(-span_km, span_km)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(plot_style.axis_line)
        axis.line.set_linewidth(0.55)
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = plot_style.grid_line
        axis._axinfo["grid"]["linewidth"] = 0.45
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(colors=plot_style.primary_text)


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


def _style_legend(legend: Any, plot_style: _OrbitPlotStyle) -> None:
    legend.get_frame().set_facecolor(plot_style.panel_face)
    legend.get_frame().set_edgecolor(plot_style.panel_edge)
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(plot_style.primary_text)


def _add_bottom_legend(fig: Any, ax: Any, handles: Sequence[Line2D], plot_style: _OrbitPlotStyle) -> None:
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
    _style_legend(legend, plot_style)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    ax_position = ax.get_position()
    tight_bbox = ax.get_tightbbox(renderer=renderer).transformed(fig.transFigure.inverted())
    label_overhang = max(0.0, ax_position.y0 - tight_bbox.y0)
    _reserve_bottom_band(ax, band_top=legend_bbox.y1 + label_overhang)


def _reserve_bottom_band(ax: Any, *, band_top: float) -> None:
    ax_position = ax.get_position()
    required_axes_bottom = band_top + _ANNOTATION_AXES_GAP
    if ax_position.y0 >= required_axes_bottom:
        return

    shift = required_axes_bottom - ax_position.y0
    ax.set_position(
        [
            ax_position.x0,
            ax_position.y0 + shift,
            ax_position.width,
            ax_position.height - shift,
        ]
    )


def _add_bottom_annotation_box(
    fig: Any,
    ax: Any,
    *,
    info_text: str,
    plot_style: _OrbitPlotStyle,
    fontsize: float,
) -> None:
    box_style = _annotation_box_style(plot_style)
    info_artist = fig.text(
        0.5,
        _ANNOTATION_BAND_BOTTOM,
        info_text,
        ha="center",
        va="bottom",
        multialignment="left",
        color=plot_style.primary_text,
        fontsize=fontsize,
        fontfamily=_ANNOTATION_FONT_FAMILY,
        linespacing=_ANNOTATION_LINE_SPACING,
        bbox=box_style,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    info_bbox = info_artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    ax_position = ax.get_position()
    tight_bbox = ax.get_tightbbox(renderer=renderer).transformed(fig.transFigure.inverted())
    label_overhang = max(0.0, ax_position.y0 - tight_bbox.y0)
    band_top = _ANNOTATION_BAND_BOTTOM + info_bbox.height
    _reserve_bottom_band(ax, band_top=band_top + label_overhang)
    ax_position = ax.get_position()
    info_artist.set_position((0.5 * (ax_position.x0 + ax_position.x1), _ANNOTATION_BAND_BOTTOM))


def _make_info_text(
    summary: dict[str, str],
    right_rows: Sequence[tuple[str, str]],
    *,
    left_rows: Sequence[tuple[str, str]] | None = None,
) -> str:
    left_body = _format_labeled_rows(
        [
            ("a", summary["a"]),
            ("e", summary["e"]),
            ("i", summary["i"]),
            ("RAAN", summary["raan"]),
            ("argp", summary["argp"]),
            ("nu", summary["nu"]),
            *(tuple(row) for row in ([] if left_rows is None else list(left_rows))),
        ]
    )
    right_body = _format_labeled_rows(right_rows)
    return "\n".join(
        [
            "Initial Keplerian Elements",
            *left_body,
            "",
            "Time Context",
            *right_body,
        ]
    )


def _add_single_orbit_3d_annotations(
    fig: Any,
    ax: Any,
    orbit: Any,
    start_s: float,
    duration_s: float,
    plot_style: _OrbitPlotStyle,
) -> None:
    eval_time = orbit.epoch + (start_s + duration_s) * u.s
    summary = _extract_keplerian_summary(orbit)

    info_text = _make_info_text(
        summary,
        [
            ("Evaluation", f"{_format_time_label(eval_time)} UTC"),
            ("Start offset", f"{start_s / 60.0:,.2f} min"),
            ("Trail window", f"{duration_s / 60.0:,.2f} min"),
            ("Marker", "filled=near side, hollow=far side"),
        ],
        left_rows=[
            ("Epoch", f"{_format_time_label(orbit.epoch)} UTC"),
            ("Frame", summary["frame"]),
        ],
    )
    _add_bottom_annotation_box(
        fig,
        ax,
        info_text=info_text,
        plot_style=plot_style,
        fontsize=10.15,
    )


def _add_single_orbit_2d_annotations(
    fig: Any,
    ax: Any,
    orbit: Any,
    start_s: float,
    duration_s: float,
    plot_style: _OrbitPlotStyle,
) -> None:
    eval_time = orbit.epoch + (start_s + duration_s) * u.s
    info_text = _make_info_text(
        _extract_keplerian_summary(orbit),
        [
            ("Evaluation", f"{_format_time_label(eval_time)} UTC"),
            ("Start offset", f"{start_s / 60.0:,.2f} min"),
            ("Trail window", f"{duration_s / 60.0:,.2f} min"),
        ],
        left_rows=[
            ("Epoch", f"{_format_time_label(orbit.epoch)} UTC"),
            ("Coords", "WGS84 geodetic"),
        ],
    )
    _add_bottom_annotation_box(
        fig,
        ax,
        info_text=info_text,
        plot_style=plot_style,
        fontsize=9.95,
    )


def _sample_plot_windows(
    orbit_list: list[Any],
    *,
    view: str,
    start_time: Any,
    duration: Any,
    labels: list[str] | None,
    styles: list[_OrbitTraceStyle],
    samples: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for idx, orbit in enumerate(orbit_list):
        start_s = _coerce_start_seconds(start_time, orbit)
        duration_s = _coerce_duration_seconds(duration, orbit)
        query_s = start_s + np.linspace(0.0, duration_s, int(samples), dtype=np.float64)
        label = labels[idx] if labels is not None else (f"Orbit {idx + 1}" if len(orbit_list) > 1 else "Orbit")
        style = styles[idx]
        window = {
            "orbit": orbit,
            "label": label,
            "color": style.color,
            "line_style": style.line_style,
            "marker": style.marker,
            "start_s": start_s,
            "duration_s": duration_s,
            "query_s": query_s,
        }

        if view == "3d":
            points_km = (
                np.asarray(orbit.get_p(query_s, frame=_PLOT_3D_FRAME, as_quantity=False), dtype=np.float64)
                / 1000.0
            )
            if points_km.ndim != 2 or points_km.shape[1] != 3:
                raise ValueError("orbit.get_p returned an unexpected shape")
            window["points_km"] = points_km
            window["marker_xyz"] = points_km[-1]
        else:
            lat_deg, lon_deg, alt_m = orbit.get_geodetic(query_s, as_quantity=False)
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
    map_style: MapStyle,
    elev: float,
    azim: float,
    start_time: Any,
    duration: Any,
    opacity: float,
    line_width: float | None,
    marker_size: float | None,
    show_info: bool,
    show: bool,
) -> tuple["Figure", "Axes3D"]:
    plot_style = _resolve_orbit_plot_style(map_style)
    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor=plot_style.figure_face)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_prop_cycle(color=NSTK_DEFAULT_COLOR_CYCLE)
    else:
        fig = ax.figure
        if not hasattr(ax, "zaxis"):
            raise TypeError("view='3d' requires a Matplotlib 3D axis")

    view_dir = _view_direction_from_angles(elev, azim)
    globe_artist, globe_drag_artist, coast_artist, country_artist = _draw_earth_globe(ax, view_dir, plot_style)

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
        line_style = window["line_style"]
        marker = window["marker"]
        orbit_artists.append(
            _create_3d_orbit_artists(
                ax,
                index=idx,
                points_km=window["points_km"],
                marker_xyz=window["marker_xyz"],
                color=color,
                line_style=line_style,
                marker=marker,
                linewidth=trail_width,
                opacity=opacity,
                marker_area=marker_area,
                halo_size=halo_size,
                show_connector=(len(windows) == 1),
                marker_edge_color=plot_style.primary_text,
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=trail_width,
                linestyle=line_style,
                marker=marker,
                markersize=legend_marker_size,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=window["label"],
            )
        )

    span_km = float(max(_EARTH_RADIUS_KM * 1.18, np.max(np.abs(np.concatenate(all_points, axis=0))) * 1.08))
    _style_3d_axes(ax, span_km, plot_style)
    ax.view_init(elev=float(elev), azim=float(azim))
    scene_state = _Orbit3DSceneState(
        ax=ax,
        globe_artist=globe_artist,
        globe_drag_artist=globe_drag_artist,
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
            color=plot_style.primary_text,
            fontsize=16,
            pad=16.0,
        )

    if show_info and len(windows) == 1:
        window = windows[0]
        _add_single_orbit_3d_annotations(
            fig,
            ax,
            window["orbit"],
            window["start_s"],
            window["duration_s"],
            plot_style,
        )
        info_line = (
            "Marker shows the evaluated spacecraft state at the end of the plotted trail; "
            "a hollow ring means the state is on the far side of the Earth"
        )
    elif show_info:
        info_line = (
            "Per-orbit default window: epoch through one Keplerian period"
            if start_time is None and duration is None
            else "Markers show the end of each plotted trail; hollow rings mark far-side states"
        )
    else:
        info_line = None

    if info_line is not None:
        fig.text(0.5, 0.03, info_line, color=plot_style.secondary_text, fontsize=10, ha="center", va="center")

    if len(windows) > 1:
        _add_bottom_legend(fig, ax, handles, plot_style)

    if show:
        plt.show()
    return fig, ax


def _render_2d(
    windows: list[dict[str, Any]],
    *,
    ax: Any | None,
    figsize: tuple[float, float],
    title: str | None,
    map_style: MapStyle,
    map_view: "MapView | None",
    start_time: Any,
    duration: Any,
    opacity: float,
    line_width: float | None,
    marker_size: float | None,
    show_info: bool,
    show: bool,
) -> tuple["Figure", GeoAxes]:
    plot_style = _resolve_orbit_plot_style(map_style)
    effective_extent = None if map_view is not None else "global"
    if ax is None:
        geo_map = GeoMap(
            style=plot_style.map_style,
            view=map_view,
            extent=effective_extent,
            figsize=figsize,
        )
        fig = geo_map.fig
        ax = geo_map.ax
    else:
        fig = ax.figure
        if hasattr(ax, "zaxis") or not isinstance(ax, GeoAxes):
            raise TypeError("view='2d' requires a Cartopy GeoAxes or no axis")
        geo_map = GeoMap(
            style=plot_style.map_style,
            view=map_view,
            extent=effective_extent,
            ax=ax,
        )
        ax = geo_map.ax

    trail_width = _default_trail_width("2d", len(windows)) if line_width is None else line_width
    marker_diameter = _default_marker_diameter("2d") if marker_size is None else marker_size
    legend_marker_size = _default_legend_marker_size("2d") if marker_size is None else marker_diameter
    handles: list[Line2D] = []
    for window in windows:
        color = window["color"]
        line_style = window["line_style"]
        marker = window["marker"]
        geo_map.add_ground_track(
            window["lon_deg"],
            window["lat_deg"],
            color=color,
            opacity=opacity,
            line_width=trail_width,
            line_style=line_style,
            marker_latlon=window["marker_latlon"],
            marker_size=marker_diameter,
            marker=marker,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=trail_width,
                linestyle=line_style,
                marker=marker,
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
            color=plot_style.primary_text,
            fontsize=16,
            pad=14.0,
        )

    if show_info and len(windows) == 1:
        window = windows[0]
        _add_single_orbit_2d_annotations(
            fig,
            ax,
            window["orbit"],
            window["start_s"],
            window["duration_s"],
            plot_style,
        )
        info_line = "Marker shows the evaluated spacecraft ground-track position at the end of the plotted trail"
    elif show_info:
        info_line = (
            "Per-orbit default window: epoch through one Keplerian period"
            if start_time is None and duration is None
            else "Markers show the end of each plotted ground track"
        )
    else:
        info_line = None

    if len(windows) > 1:
        _add_bottom_legend(fig, ax, handles, plot_style)

    if info_line is not None:
        fig.text(0.5, 0.03, info_line, color=plot_style.secondary_text, fontsize=10, ha="center", va="center")

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
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    map_style: str | "MapStyle" | None = None,
    map_view: "MapView | None" = None,
    view: Literal["2d", "3d"] = "2d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show_info: bool = False,
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
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    map_style: str | "MapStyle" | None = None,
    map_view: "MapView | None" = None,
    view: Literal["2d", "3d"] = "2d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show_info: bool = False,
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
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    map_style: str | "MapStyle" | None = None,
    map_view: "MapView | None" = None,
    view: Literal["2d", "3d"] = "2d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show_info: bool = False,
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
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    map_style: str | "MapStyle" | None = None,
    map_view: "MapView | None" = None,
    view: Literal["2d", "3d"] = "2d",
    elev: float = 24.0,
    azim: float = 42.0,
    opacity: float = 1.0,
    line_width: float | None = None,
    marker_size: float | None = None,
    show_info: bool = False,
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
        Optional per-orbit legend labels and colors. When a color is reused,
        NSTK automatically varies the trail line style and endpoint marker so
        repeated colors remain distinguishable.
    samples
        Number of sample points per plotted trail. Must be at least 2.
    ax
        Existing Matplotlib axis to draw into. Omit to create a new figure.
    figsize
        Optional figure size. When omitted, NSTK uses a view-specific default:
        a shorter global-map frame for ``view="2d"`` and a slightly taller
        globe frame for ``view="3d"``.
    title
        Optional plot title.
    map_style
        Optional shared NSTK style preset or custom `MapStyle`. In 2D it drives
        the basemap directly; in 3D it drives the globe, plot chrome, legends,
        and annotation colors.
    map_view
        Optional reusable 2D `MapView` controlling projection, extent, and
        layout. Ignored for the 3D globe view.
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
    show_info
        If ``True``, draw the single-orbit parameter box or the multi-orbit
        informational footer text. Defaults to ``False`` so a plain call only
        renders the orbit plot itself.
    show
        If ``True``, call ``matplotlib.pyplot.show()`` before returning.
    """

    orbit_list = _coerce_orbit_list(orbits)
    label_list = _coerce_labels(labels, len(orbit_list))
    color_list = _coerce_colors(colors, len(orbit_list))
    style_list = _assign_trace_styles(color_list)
    view_key = _coerce_view(view)
    resolved_map_style = get_map_style("dark" if map_style is None else map_style)
    resolved_figsize = _resolve_plot_figsize(view_key, figsize)
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
        styles=style_list,
        samples=sample_count,
    )

    if view_key == "3d":
        return _render_3d(
            windows,
            ax=ax,
            figsize=resolved_figsize,
            title=title,
            map_style=resolved_map_style,
            elev=elev,
            azim=azim,
            start_time=start_time,
            duration=duration,
            opacity=alpha,
            line_width=trail_width,
            marker_size=marker_diameter,
            show_info=show_info,
            show=show,
        )
    return _render_2d(
        windows,
        ax=ax,
        figsize=resolved_figsize,
        title=title,
        map_style=resolved_map_style,
        map_view=map_view,
        start_time=start_time,
        duration=duration,
        opacity=alpha,
        line_width=trail_width,
        marker_size=marker_diameter,
        show_info=show_info,
        show=show,
    )


__all__ = ["plot_orbits"]
