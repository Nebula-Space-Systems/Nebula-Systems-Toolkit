from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import to_rgba

from .country_shapes import (
    country_geometries,
    fuzzy_find_country_record,
)
from .map import (
    CFeatureScale,
    ExtentConfig,
    MapConfig,
    MapStyle,
    MapView,
    ProjectionConfig,
    _configure_basemap_axes,
    _projection_from_config,
    add_geodesic_trace,
    add_polygon,
    compile_map_config,
    get_map_style,
    make_basemap,
)

_GROUND_TRACK_HALO_SCALE = 2.05
_X_COORD_NAMES = ("lon", "longitude", "x")
_Y_COORD_NAMES = ("lat", "latitude", "y")

_PROJECTION_ALIASES = {
    "platecarree": "PlateCarree",
    "mercator": "Mercator",
    "robinson": "Robinson",
    "mollweide": "Mollweide",
    "orthographic": "Orthographic",
}

_LAYER_KIND_ALIASES = {
    "field": "field",
    "coverage": "field",
    "geometry": "geometry",
    "geometries": "geometries",
    "geojson": "geojson",
    "country": "country",
    "countries": "countries",
    "feature": "feature",
    "artist": "artist",
    "points": "points",
    "point": "points",
    "scatter": "points",
    "trace": "trace",
    "line": "trace",
    "lines": "trace",
    "raster": "raster",
    "image": "raster",
    "contour": "contours",
    "contours": "contours",
    "filledcontours": "filled_contours",
    "filled_contours": "filled_contours",
    "contourf": "filled_contours",
    "quiver": "quiver",
    "streamplot": "streamplot",
    "stream": "streamplot",
    "text": "text",
    "labels": "labels",
    "geodataframe": "geodataframe",
}


def _normalize_key(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _normalize_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    key = _normalize_key(kind)
    return _LAYER_KIND_ALIASES.get(key, key)


def get_map_preset(name: str) -> MapStyle:
    return get_map_style(name)


def _coerce_projection(projection: str | ProjectionConfig | None, base: ProjectionConfig) -> ProjectionConfig:
    if projection is None:
        return base
    if isinstance(projection, ProjectionConfig):
        return projection
    key = _normalize_key(projection)
    if key not in _PROJECTION_ALIASES:
        raise ValueError(f"Unsupported projection {projection!r}")
    return replace(base, name=_PROJECTION_ALIASES[key])


def _coerce_crs(value: Any | None) -> Any:
    if value is None:
        return ccrs.PlateCarree()
    if isinstance(value, ProjectionConfig):
        return _projection_from_config(value)
    if hasattr(value, "transform_points"):
        return value
    if isinstance(value, int):
        if int(value) == 4326:
            return ccrs.PlateCarree()
        return ccrs.epsg(int(value))
    if isinstance(value, str):
        key = _normalize_key(value)
        if key in {"4326", "epsg4326", "crs84", "ogccrs84", "wgs84", "platecarree"}:
            return ccrs.PlateCarree()
        if key in _PROJECTION_ALIASES:
            return _projection_from_config(ProjectionConfig(name=_PROJECTION_ALIASES[key]))
        if key.startswith("epsg"):
            digits = "".join(ch for ch in key if ch.isdigit())
            if digits:
                code = int(digits)
                if code == 4326:
                    return ccrs.PlateCarree()
                return ccrs.epsg(code)
    raise TypeError(f"Unsupported CRS specification: {value!r}")


def _coerce_object_crs(value: Any | None) -> Any | None:
    if value is None:
        return None
    candidate = getattr(value, "crs", value)
    if candidate is None:
        return None
    to_epsg = getattr(candidate, "to_epsg", None)
    if callable(to_epsg):
        epsg = to_epsg()
        if epsg is not None:
            return _coerce_crs(int(epsg))
    try:
        return _coerce_crs(candidate)
    except Exception:
        return None


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    *,
    pad_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    west, east, south, north = [float(value) for value in bounds]
    pad = float(pad_deg)
    if pad < 0.0:
        raise ValueError("pad_deg must be >= 0")
    west = max(-180.0, west - pad)
    east = min(180.0, east + pad)
    south = max(-90.0, south - pad)
    north = min(90.0, north + pad)
    return (west, east, south, north)


def _merge_bounds(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if left is None:
        return right
    if right is None:
        return left
    return (
        min(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        max(left[3], right[3]),
    )


def _coerce_extent_bounds(value: Any, *, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
    if isinstance(value, ExtentConfig):
        if value.extent is not None:
            return _expand_bounds(tuple(float(v) for v in value.extent), pad_deg=pad_deg)
        return (-180.0, 180.0, -90.0, 90.0)
    if isinstance(value, (tuple, list)) and len(value) == 4:
        return _expand_bounds(tuple(float(v) for v in value), pad_deg=pad_deg)
    if hasattr(value, "extent") and callable(value.extent):
        try:
            bounds = value.extent(pad_deg=pad_deg)
        except TypeError:
            bounds = value.extent()
        return tuple(float(v) for v in bounds)
    if hasattr(value, "bounds"):
        minx, miny, maxx, maxy = value.bounds
        return _expand_bounds((float(minx), float(maxx), float(miny), float(maxy)), pad_deg=pad_deg)
    raise TypeError(f"Unsupported extent specification: {type(value)!r}")


def _is_global_bounds(bounds: tuple[float, float, float, float]) -> bool:
    west, east, south, north = bounds
    return (east - west) >= 359.0 and (north - south) >= 179.0


def _transform_bounds_to_platecarree(
    bounds: tuple[float, float, float, float],
    *,
    src_crs: Any | None = None,
) -> tuple[float, float, float, float] | None:
    crs = _coerce_crs(src_crs)
    if isinstance(crs, ccrs.PlateCarree):
        return bounds

    west, east, south, north = [float(v) for v in bounds]
    xs = np.asarray([west, west, east, east], dtype=np.float64)
    ys = np.asarray([south, north, south, north], dtype=np.float64)
    points = ccrs.PlateCarree().transform_points(crs, xs, ys)
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if not np.any(finite):
        return None
    lon = points[finite, 0]
    lat = points[finite, 1]
    return (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))


def _bounds_from_lonlat(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    src_crs: Any | None = None,
) -> tuple[float, float, float, float] | None:
    crs = _coerce_crs(src_crs)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    lat_arr = np.asarray(lat_deg, dtype=np.float64)

    if lon_arr.size == 0 or lat_arr.size == 0:
        return None

    if lon_arr.shape != lat_arr.shape:
        if lon_arr.ndim == 1 and lat_arr.ndim == 1:
            lon = lon_arr[np.isfinite(lon_arr)]
            lat = lat_arr[np.isfinite(lat_arr)]
            if lon.size == 0 or lat.size == 0:
                return None
            if isinstance(crs, ccrs.PlateCarree):
                return (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
            points = ccrs.PlateCarree().transform_points(crs, lon_mesh.ravel(), lat_mesh.ravel())
            finite_points = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
            if not np.any(finite_points):
                return None
            lon_pc = points[finite_points, 0]
            lat_pc = points[finite_points, 1]
            return (float(np.min(lon_pc)), float(np.max(lon_pc)), float(np.min(lat_pc)), float(np.max(lat_pc)))
        raise ValueError("lon_deg and lat_deg must share the same shape unless both are 1D axes")

    lon = lon_arr.ravel()
    lat = lat_arr.ravel()
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(finite):
        return None
    lon = lon[finite]
    lat = lat[finite]
    if isinstance(crs, ccrs.PlateCarree):
        return (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))

    points = ccrs.PlateCarree().transform_points(crs, lon, lat)
    finite_points = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if not np.any(finite_points):
        return None
    lon_pc = points[finite_points, 0]
    lat_pc = points[finite_points, 1]
    return (float(np.min(lon_pc)), float(np.max(lon_pc)), float(np.min(lat_pc)), float(np.max(lat_pc)))


def _bounds_from_targets(targets: Any) -> tuple[float, float, float, float] | None:
    if targets is None:
        return None
    boundary = getattr(targets, "boundary_geometry", None)
    if boundary is not None and not getattr(boundary, "is_empty", False):
        minx, miny, maxx, maxy = boundary.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))
    lon = getattr(targets, "lon_deg", None)
    lat = getattr(targets, "lat_deg", None)
    if lon is None or lat is None:
        return None
    return _bounds_from_lonlat(np.asarray(lon), np.asarray(lat))


def _default_extent_for_targets(targets: Any) -> str:
    bounds = _bounds_from_targets(targets)
    if bounds is None or _is_global_bounds(bounds):
        return "global"
    return "auto"


def _resolve_map_config(
    *,
    style: str | MapStyle | None = None,
    theme: str | MapConfig | MapStyle | None = "light_detailed",
    view: MapView | None = None,
    projection: str | ProjectionConfig | None = None,
    extent: Any = None,
    figsize: tuple[float, float] | None = None,
    grid: bool | None = None,
    coastlines: bool | None = None,
    borders: bool | None = None,
    frame: bool | None = None,
    cfeature_scale: CFeatureScale | None = None,
    map_cfg: MapConfig | None = None,
) -> tuple[MapConfig, bool]:
    if map_cfg is not None:
        cfg = map_cfg
    elif isinstance(theme, MapConfig):
        cfg = theme
        if view is not None:
            cfg = replace(
                cfg,
                projection=view.projection,
                extent=view.extent,
                figsize=view.figsize,
                cfeature_scale=view.cfeature_scale,
                mpl=view.mpl,
                title=view.title,
            )
    else:
        style_value = style if style is not None else theme
        if style_value is None:
            style_value = "light_detailed"
        cfg = compile_map_config(style=style_value, view=view)

    cfg = replace(cfg, projection=_coerce_projection(projection, cfg.projection))
    if figsize is not None:
        cfg = replace(cfg, figsize=tuple(float(v) for v in figsize))
    if cfeature_scale is not None:
        cfg = replace(cfg, cfeature_scale=cfeature_scale)
    if grid is not None:
        cfg = replace(cfg, gridlines=replace(cfg.gridlines, enabled=bool(grid)))
    if coastlines is not None:
        cfg = replace(cfg, coastlines=replace(cfg.coastlines, enabled=bool(coastlines)))
    if borders is not None:
        cfg = replace(cfg, borders=replace(cfg.borders, enabled=bool(borders)))
    if frame is not None:
        cfg = replace(cfg, outline=replace(cfg.outline, enabled=bool(frame)))

    auto_extent_pending = False
    if extent is None:
        return cfg, auto_extent_pending
    if isinstance(extent, str):
        key = _normalize_key(extent)
        if key == "auto":
            auto_extent_pending = True
            cfg = replace(cfg, extent=ExtentConfig(global_map=True))
            return cfg, auto_extent_pending
        if key == "global":
            cfg = replace(cfg, extent=ExtentConfig(global_map=True))
            return cfg, auto_extent_pending
    cfg = replace(cfg, extent=ExtentConfig(global_map=False, extent=_coerce_extent_bounds(extent)))
    return cfg, auto_extent_pending


def _axis_edges(
    axis: np.ndarray,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64)
    if arr.size == 1:
        width = 1.0
        if clip_min is not None and clip_max is not None:
            width = float(clip_max) - float(clip_min)
        edges = np.asarray([arr[0] - 0.5 * width, arr[0] + 0.5 * width], dtype=np.float64)
    else:
        edges = np.empty(arr.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
        edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
        edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    if clip_min is not None:
        edges = np.maximum(edges, float(clip_min))
    if clip_max is not None:
        edges = np.minimum(edges, float(clip_max))
    return edges


def _is_xarray_dataarray(obj: Any) -> bool:
    cls = type(obj)
    return cls.__name__ == "DataArray" and cls.__module__.startswith("xarray")


def _looks_like_geometry(obj: Any) -> bool:
    return hasattr(obj, "geom_type")


def _looks_like_geojson_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping) and "type" in obj


def _looks_like_geodataframe(obj: Any) -> bool:
    return hasattr(obj, "geometry") and not _looks_like_geometry(obj)


def _coerce_column(data: Any, key: str) -> np.ndarray:
    if isinstance(data, Mapping):
        value = data[key]
    else:
        value = data[key]
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def _coerce_lon_lat_pair(
    data: Any,
    maybe_lat: Any = None,
    *,
    x: str | None = None,
    y: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if x is not None or y is not None:
        if x is None or y is None:
            raise ValueError("Both x and y column names are required together")
        return _coerce_column(data, x), _coerce_column(data, y)
    if maybe_lat is None:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return np.asarray(data[0]), np.asarray(data[1])
        raise ValueError("Expected lon/lat arrays or x/y column names")
    return np.asarray(data), np.asarray(maybe_lat)


def _extract_dataarray_grid(
    data: Any,
    *,
    lon_deg: Any = None,
    lat_deg: Any = None,
) -> tuple[np.ndarray, Any, Any]:
    values = np.asarray(data.values)
    if lon_deg is not None and lat_deg is not None:
        return values, lon_deg, lat_deg

    coords = getattr(data, "coords", {})
    lon = lon_deg
    lat = lat_deg
    for name in _X_COORD_NAMES:
        if name in coords:
            coord = coords[name]
            lon = np.asarray(getattr(coord, "values", coord))
            break
    for name in _Y_COORD_NAMES:
        if name in coords:
            coord = coords[name]
            lat = np.asarray(getattr(coord, "values", coord))
            break
    if lon is None or lat is None:
        raise ValueError(
            "Could not infer lon/lat coordinates from DataArray. Pass lon_deg=... and lat_deg=..."
        )
    return values, lon, lat


def _coerce_geojson_payload(data: Any) -> Mapping[str, Any]:
    if _looks_like_geojson_mapping(data):
        return data
    path = Path(data)
    return json.loads(path.read_text())


def _geometries_from_geojson_payload(payload: Mapping[str, Any]) -> list[Any]:
    from shapely.geometry import shape

    geo_type = payload.get("type")
    if geo_type == "FeatureCollection":
        return [shape(feature["geometry"]) for feature in payload.get("features", [])]
    if geo_type == "Feature":
        return [shape(payload["geometry"])]
    return [shape(payload)]


def _render_field_artist(
    ax: Any,
    field: Any,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    alpha: float,
    point_size: float,
    render: str,
) -> Any:
    from shapely import contains_xy

    try:
        import matplotlib.tri as mtri
    except Exception:  # pragma: no cover
        mtri = None

    targets = field.targets
    values = np.asarray(field.values, dtype=np.float64)
    render_key = str(render).lower()
    if render_key not in {"auto", "surface", "points"}:
        raise ValueError("render must be one of: 'auto', 'surface', 'points'")

    adaptive_attrs = getattr(targets, "attrs", {}) if targets is not None else {}
    has_adaptive_rows = (
        isinstance(adaptive_attrs, dict)
        and adaptive_attrs.get("sampler") == "latitude_adaptive"
        and "lat_rows_deg" in adaptive_attrs
        and "row_offsets" in adaptive_attrs
    )

    if render_key == "points":
        return ax.scatter(
            targets.lon_deg,
            targets.lat_deg,
            c=values,
            s=float(point_size),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=4,
            alpha=float(alpha),
        )

    if targets.surface_grid is not None:
        lon = targets.surface_grid.lon_deg
        lat = targets.surface_grid.lat_deg
        grid_boundary = targets.surface_grid.boundary_geometry
        if grid_boundary is None:
            grid_boundary = targets.boundary_geometry
        lon_clip = None
        lat_clip = None
        if grid_boundary is not None and not getattr(grid_boundary, "is_empty", False):
            min_lon, min_lat, max_lon, max_lat = grid_boundary.bounds
            lon_clip = (float(min_lon), float(max_lon))
            lat_clip = (float(min_lat), float(max_lat))
        return ax.pcolormesh(
            _axis_edges(
                lon,
                clip_min=None if lon_clip is None else lon_clip[0],
                clip_max=None if lon_clip is None else lon_clip[1],
            ),
            _axis_edges(
                lat,
                clip_min=None if lat_clip is None else lat_clip[0],
                clip_max=None if lat_clip is None else lat_clip[1],
            ),
            targets.to_grid(values, fill_value=np.nan),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="auto",
            rasterized=True,
            zorder=4,
            vmin=vmin,
            vmax=vmax,
            alpha=float(alpha),
        )

    if has_adaptive_rows:
        lat_rows = np.asarray(adaptive_attrs["lat_rows_deg"], dtype=np.float64)
        row_offsets = np.asarray(adaptive_attrs["row_offsets"], dtype=np.int64)
        lon_min_cont = float(adaptive_attrs["lon_min_cont_deg"])
        lon_max_cont = float(adaptive_attrs["lon_max_cont_deg"])
        lon_span = lon_max_cont - lon_min_cont
        row_sizes = np.diff(row_offsets)
        n_lon_display = int(np.max(row_sizes))
        if n_lon_display <= 1:
            lon_axis_cont = np.asarray([0.5 * (lon_min_cont + lon_max_cont)], dtype=np.float64)
        else:
            lon_axis_cont = lon_min_cont + (np.arange(n_lon_display, dtype=np.float64) + 0.5) * (
                lon_span / float(n_lon_display)
            )
        grid = np.empty((lat_rows.size, lon_axis_cont.size), dtype=np.float64)
        is_periodic = abs(lon_span - 360.0) < 1e-9
        for row_idx, (start, stop) in enumerate(zip(row_offsets[:-1], row_offsets[1:])):
            lon_row = np.asarray(targets.lon_deg[int(start) : int(stop)], dtype=np.float64)
            val_row = np.asarray(values[int(start) : int(stop)], dtype=np.float64)
            lon_row_cont = lon_row.copy()
            lon_row_cont[lon_row_cont < lon_min_cont] += 360.0
            if val_row.size == 1:
                grid[row_idx] = val_row[0]
                continue
            order = np.argsort(lon_row_cont)
            x = lon_row_cont[order]
            y = val_row[order]
            if is_periodic:
                x = np.concatenate([x - 360.0, x, x + 360.0])
                y = np.concatenate([y, y, y])
                grid[row_idx] = np.interp(lon_axis_cont, x, y)
            else:
                grid[row_idx] = np.interp(lon_axis_cont, x, y)
                outside = (lon_axis_cont < x[0]) | (lon_axis_cont > x[-1])
                grid[row_idx, outside] = np.nan

        if targets.boundary_geometry is not None:
            lon_mask = ((lon_axis_cont + 180.0) % 360.0) - 180.0
            lon_grid, lat_grid = np.meshgrid(lon_mask, lat_rows)
            inside = np.asarray(contains_xy(targets.boundary_geometry, lon_grid, lat_grid), dtype=bool)
            grid = np.where(inside, grid, np.nan)

        lat_clip_min = None
        lat_clip_max = None
        if targets.boundary_geometry is not None and not getattr(targets.boundary_geometry, "is_empty", False):
            _, min_lat, _, max_lat = targets.boundary_geometry.bounds
            lat_clip_min = float(min_lat)
            lat_clip_max = float(max_lat)

        return ax.pcolormesh(
            _axis_edges(lon_axis_cont, clip_min=lon_min_cont, clip_max=lon_max_cont),
            _axis_edges(lat_rows, clip_min=lat_clip_min, clip_max=lat_clip_max),
            np.ma.masked_invalid(grid),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="auto",
            rasterized=True,
            zorder=4,
            vmin=vmin,
            vmax=vmax,
            alpha=float(alpha),
        )

    if render_key == "surface" and mtri is None:
        raise RuntimeError("render='surface' requires matplotlib.tri")
    if mtri is None:
        return ax.scatter(
            targets.lon_deg,
            targets.lat_deg,
            c=values,
            s=float(point_size),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=4,
            alpha=float(alpha),
        )

    lon = np.asarray(targets.lon_deg, dtype=np.float64)
    lat = np.asarray(targets.lat_deg, dtype=np.float64)
    triangulation = mtri.Triangulation(lon, lat)
    tri_lon = lon[triangulation.triangles]
    tri_lat = lat[triangulation.triangles]
    tri_mask = (np.ptp(tri_lon, axis=1) > 40.0) | (np.ptp(tri_lat, axis=1) > 20.0)
    if np.any(tri_mask):
        triangulation.set_mask(tri_mask)
    return ax.tripcolor(
        triangulation,
        values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        alpha=float(alpha),
        zorder=4,
    )


@dataclass
class MapLayer:
    """Mutable handle for a rendered layer on a `GeoMap` canvas."""

    kind: str
    artists: list[Any] = field(default_factory=list)
    name: str | None = None
    colorbar: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artist(self) -> Any | None:
        return self.artists[0] if self.artists else None

    def _child_artists(self, artist: Any) -> list[Any]:
        children: list[Any] = []
        collections = getattr(artist, "collections", None)
        if collections is not None:
            children.extend(list(collections))
        for attr in ("lines", "arrows"):
            child = getattr(artist, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                children.extend(list(child))
            else:
                children.append(child)
        return children

    def _apply(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for artist in self.artists:
            method = getattr(artist, method_name, None)
            if callable(method):
                method(*args, **kwargs)
                continue
            for child in self._child_artists(artist):
                method = getattr(child, method_name, None)
                if callable(method):
                    method(*args, **kwargs)

    def remove(self) -> None:
        for artist in reversed(self.artists):
            removed = False
            method = getattr(artist, "remove", None)
            if callable(method):
                try:
                    method()
                    removed = True
                except Exception:
                    removed = False
            if removed:
                continue
            for child in reversed(self._child_artists(artist)):
                method = getattr(child, "remove", None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                try:
                    self.colorbar.ax.remove()
                except Exception:
                    pass

    def set_visible(self, visible: bool) -> "MapLayer":
        self._apply("set_visible", bool(visible))
        if self.colorbar is not None:
            self.colorbar.ax.set_visible(bool(visible))
        return self

    def set_alpha(self, alpha: float) -> "MapLayer":
        self._apply("set_alpha", float(alpha))
        return self

    def set_zorder(self, zorder: float) -> "MapLayer":
        self._apply("set_zorder", float(zorder))
        return self

    def restyle(self, **style: Any) -> "MapLayer":
        for key, value in style.items():
            setter = f"set_{key}"
            applied = False
            for artist in self.artists:
                method = getattr(artist, setter, None)
                if callable(method):
                    method(value)
                    applied = True
                    continue
                for child in self._child_artists(artist):
                    method = getattr(child, setter, None)
                    if callable(method):
                        method(value)
                        applied = True
            if not applied:
                raise AttributeError(f"Layer artists do not support style attribute {key!r}")
        return self


class GeoMap:
    """High-level, layer-oriented map canvas built on Cartopy GeoAxes.

    `GeoMap` is the shared 2D map surface for NSTK plotting. It is designed to
    be easy to use directly while still exposing the lower-level controls that
    more advanced notebooks and applications need.

    Typical usage:

    ```python
    from nstk.plotting import GeoMap, get_map_style

    paper = get_map_style("light_detailed").with_grid(alpha=0.15, draw_labels=False)
    m = GeoMap(style=paper, extent="auto", pad_deg=2.0)
    m.add_field(field, cmap="viridis")
    m.add_trace(lon_deg, lat_deg, color="tomato", linewidth=2.0)
    m.add_points(site_lon, site_lat, color="black", size=22.0)
    m.show()
    ```
    """

    def __init__(
        self,
        *,
        style: str | MapStyle | None = None,
        theme: str | MapConfig | MapStyle | None = "light_detailed",
        view: MapView | None = None,
        projection: str | ProjectionConfig | None = None,
        extent: Any = None,
        pad_deg: float = 0.0,
        figsize: tuple[float, float] | None = None,
        grid: bool | None = None,
        coastlines: bool | None = None,
        borders: bool | None = None,
        frame: bool | None = None,
        cfeature_scale: CFeatureScale | None = None,
        title: str | None = None,
        ax: Any = None,
        map_cfg: MapConfig | None = None,
    ) -> None:
        cfg, auto_extent_pending = _resolve_map_config(
            style=style,
            theme=theme,
            view=view,
            projection=projection,
            extent=extent,
            figsize=figsize,
            grid=grid,
            coastlines=coastlines,
            borders=borders,
            frame=frame,
            cfeature_scale=cfeature_scale,
            map_cfg=map_cfg,
        )
        self.config = cfg
        self.pad_deg = float(pad_deg)
        self.layers: list[MapLayer] = []
        self.colorbars: list[Any] = []
        self.artist: Any | None = None
        self.colorbar: Any | None = None
        self.gridliner: Any | None = None
        self._auto_extent_pending = bool(auto_extent_pending)
        self._auto_extent_applied = False
        self._auto_extent_bounds: tuple[float, float, float, float] | None = None

        if ax is None:
            self.fig, self.ax, self.crs, self.gridliner = make_basemap(cfg)
        else:
            if hasattr(ax, "zaxis") or not hasattr(ax, "projection"):
                raise TypeError("GeoMap requires a Cartopy GeoAxes when ax is provided")
            self.ax = ax
            self.fig = ax.figure
            self.crs, self.gridliner = _configure_basemap_axes(self.fig, self.ax, cfg)

        if title is not None:
            self.set_title(title)

    @property
    def figure(self) -> Any:
        return self.fig

    @property
    def axes(self) -> Any:
        return self.ax

    def __iter__(self):
        yield self.fig
        yield self.ax
        yield self.artist
        yield self.colorbar

    def _register_layer(
        self,
        *,
        kind: str,
        artists: Sequence[Any] | Any,
        name: str | None = None,
        colorbar: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MapLayer:
        if isinstance(artists, Sequence) and not isinstance(artists, (str, bytes)):
            artist_list = list(artists)
        else:
            artist_list = [artists]
        layer = MapLayer(
            kind=kind,
            artists=artist_list,
            name=name,
            colorbar=colorbar,
            metadata=dict(metadata or {}),
        )
        self.layers.append(layer)
        self.artist = layer.artist
        self.colorbar = colorbar
        if colorbar is not None:
            self.colorbars.append(colorbar)
        return layer

    def set_title(self, title: str, **kwargs: Any) -> "GeoMap":
        self.ax.set_title(title, **kwargs)
        return self

    def show(self) -> "GeoMap":
        import matplotlib.pyplot as plt

        plt.show()
        return self

    def savefig(self, *args: Any, **kwargs: Any) -> Any:
        return self.fig.savefig(*args, **kwargs)

    def set_extent(self, extent: Any, *, pad_deg: float | None = None) -> "GeoMap":
        bounds = _coerce_extent_bounds(extent, pad_deg=0.0 if pad_deg is None else float(pad_deg))
        if _is_global_bounds(bounds):
            self.ax.set_global()
        else:
            self.ax.set_extent(bounds, crs=ccrs.PlateCarree())
        self._auto_extent_pending = False
        self._auto_extent_applied = True
        self._auto_extent_bounds = bounds
        return self

    def fit(
        self,
        *,
        pad_deg: float | None = None,
        layers: Sequence[MapLayer] | None = None,
    ) -> "GeoMap":
        bounds: tuple[float, float, float, float] | None = None
        selected = self.layers if layers is None else list(layers)
        for layer in selected:
            layer_bounds = layer.metadata.get("bounds")
            if layer_bounds is None:
                continue
            bounds = _merge_bounds(bounds, tuple(float(v) for v in layer_bounds))
        if bounds is not None:
            self.set_extent(bounds, pad_deg=self.pad_deg if pad_deg is None else float(pad_deg))
        return self

    def _update_auto_extent(self, bounds: tuple[float, float, float, float] | None) -> None:
        if not self._auto_extent_pending or bounds is None:
            return
        self._auto_extent_bounds = _merge_bounds(self._auto_extent_bounds, bounds)
        if self._auto_extent_bounds is None:
            return
        expanded = _expand_bounds(self._auto_extent_bounds, pad_deg=self.pad_deg)
        if _is_global_bounds(expanded):
            self.ax.set_global()
        else:
            self.ax.set_extent(expanded, crs=ccrs.PlateCarree())
        self._auto_extent_applied = True

    def add_colorbar(self, artist_or_layer: Any, *, label: str | None = None, **kwargs: Any) -> Any:
        if isinstance(artist_or_layer, MapLayer):
            artist = artist_or_layer.artist
            layer = artist_or_layer
        else:
            artist = artist_or_layer
            layer = None
        cbar = self.fig.colorbar(artist, ax=self.ax, pad=0.02, **kwargs)
        if label:
            cbar.set_label(label)
        if layer is not None:
            layer.colorbar = cbar
        self.colorbars.append(cbar)
        self.colorbar = cbar
        return cbar

    def add_artist(
        self,
        artist: Any,
        *,
        kind: str = "artist",
        name: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        auto_extent: bool = False,
    ) -> MapLayer:
        if getattr(artist, "axes", None) is None and hasattr(self.ax, "add_artist"):
            self.ax.add_artist(artist)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(kind=kind, artists=[artist], name=name, metadata={"bounds": bounds})

    def add_feature(
        self,
        feature: Any,
        *,
        facecolor: str | None = None,
        edgecolor: str | None = None,
        linewidth: float | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        name: str | None = None,
    ) -> MapLayer:
        feature_kwargs: dict[str, Any] = {}
        if facecolor is not None:
            feature_kwargs["facecolor"] = facecolor
        if edgecolor is not None:
            feature_kwargs["edgecolor"] = edgecolor
        if linewidth is not None:
            feature_kwargs["linewidth"] = float(linewidth)
        if alpha is not None:
            feature_kwargs["alpha"] = float(alpha)
        if zorder is not None:
            feature_kwargs["zorder"] = float(zorder)
        artist = self.ax.add_feature(feature, **feature_kwargs)
        return self._register_layer(kind="feature", artists=[artist], name=name)

    def _geometry_artists(
        self,
        geometries: Sequence[Any],
        *,
        src_crs: Any | None = None,
        facecolor: str | None = None,
        fill_alpha: float = 0.35,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
    ) -> list[Any]:
        crs = _coerce_crs(src_crs)
        artists: list[Any] = []
        for geometry in geometries:
            geom_type = getattr(geometry, "geom_type", None)
            if geom_type in {"Polygon", "MultiPolygon"}:
                artist = add_polygon(
                    self.ax,
                    geometry,
                    src_crs=crs,
                    facecolor=facecolor,
                    fill_alpha=fill_alpha,
                    edgecolor=edgecolor,
                    edge_alpha=edge_alpha,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                )
            else:
                artist = self.ax.add_geometries(
                    [geometry],
                    crs=crs,
                    facecolor="none" if facecolor is None else to_rgba(facecolor, float(fill_alpha)),
                    edgecolor="none" if edgecolor is None else to_rgba(edgecolor, float(edge_alpha)),
                    linewidth=float(linewidth),
                    linestyle=linestyle,
                    zorder=float(zorder),
                )
            artists.append(artist)
        return artists

    def add_geometry(
        self,
        geometry: Any,
        *,
        src_crs: Any | None = None,
        facecolor: str | None = None,
        fill_alpha: float = 0.35,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        bounds = None
        if hasattr(geometry, "bounds"):
            minx, miny, maxx, maxy = geometry.bounds
            bounds = _transform_bounds_to_platecarree(
                (float(minx), float(maxx), float(miny), float(maxy)),
                src_crs=src_crs,
            )
        if auto_extent:
            self._update_auto_extent(bounds)

        artists = self._geometry_artists(
            [geometry],
            src_crs=src_crs,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        return self._register_layer(
            kind="geometry",
            artists=artists,
            name=name,
            metadata={"bounds": bounds, "source_crs": _coerce_crs(src_crs)},
        )

    def add_geometries(
        self,
        geometries: Sequence[Any],
        *,
        src_crs: Any | None = None,
        facecolor: str | None = None,
        fill_alpha: float = 0.35,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        bounds = None
        for geometry in geometries:
            if hasattr(geometry, "bounds"):
                minx, miny, maxx, maxy = geometry.bounds
                geom_bounds = _transform_bounds_to_platecarree(
                    (float(minx), float(maxx), float(miny), float(maxy)),
                    src_crs=src_crs,
                )
                bounds = _merge_bounds(bounds, geom_bounds)
        if auto_extent:
            self._update_auto_extent(bounds)

        artists = self._geometry_artists(
            list(geometries),
            src_crs=src_crs,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        return self._register_layer(
            kind="geometries",
            artists=artists,
            name=name,
            metadata={"bounds": bounds, "source_crs": _coerce_crs(src_crs)},
        )

    def add_boundary(
        self,
        geometry: Any,
        *,
        src_crs: Any | None = None,
        color: str = "#111111",
        alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.5,
        name: str | None = None,
        auto_extent: bool = False,
    ) -> MapLayer:
        return self.add_geometry(
            geometry,
            src_crs=src_crs,
            facecolor=None,
            edgecolor=color,
            edge_alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            name=name,
            auto_extent=auto_extent,
        )

    def add_geojson(
        self,
        data: Any,
        *,
        src_crs: Any | None = None,
        facecolor: str | None = None,
        fill_alpha: float = 0.35,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        payload = _coerce_geojson_payload(data)
        geometries = _geometries_from_geojson_payload(payload)
        layer_name = name
        if layer_name is None and not _looks_like_geojson_mapping(data):
            layer_name = Path(data).stem
        return self.add_geometries(
            geometries,
            src_crs=src_crs,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            name=layer_name,
            auto_extent=auto_extent,
        )

    def add_geodataframe(
        self,
        data: Any,
        *,
        facecolor: str | None = None,
        fill_alpha: float = 0.35,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        crs = _coerce_object_crs(data)
        geometries = [geom for geom in getattr(data, "geometry") if geom is not None]
        return self.add_geometries(
            geometries,
            src_crs=crs,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            name=name,
            auto_extent=auto_extent,
        )

    def add_country(
        self,
        name_or_code: str,
        *,
        resolution: str = "110m",
        facecolor: str | None = None,
        fill_alpha: float = 0.18,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        record, matches = fuzzy_find_country_record(name_or_code, resolution=resolution)
        layer = self.add_geometry(
            record.geometry,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            name=name or str(name_or_code),
            auto_extent=auto_extent,
        )
        layer.metadata["country_match"] = matches[0]
        layer.metadata["country_matches"] = matches
        return layer

    def add_countries(
        self,
        names_or_codes: Sequence[str],
        *,
        resolution: str = "110m",
        facecolor: str | None = None,
        fill_alpha: float = 0.18,
        edgecolor: str | None = "#111111",
        edge_alpha: float = 0.95,
        linewidth: float = 0.8,
        linestyle: str = "-",
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        geometries = country_geometries(names_or_codes, resolution=resolution)
        layer = self.add_geometries(
            geometries,
            facecolor=facecolor,
            fill_alpha=fill_alpha,
            edgecolor=edgecolor,
            edge_alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            name=name or ", ".join(str(item) for item in names_or_codes),
            auto_extent=auto_extent,
        )
        layer.metadata["country_names"] = list(names_or_codes)
        return layer

    def add_points(
        self,
        lon_deg: Any,
        lat_deg: Any,
        *,
        src_crs: Any | None = None,
        values: Any | None = None,
        size: float = 16.0,
        color: Any = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 1.0,
        edgecolors: Any = None,
        linewidths: float = 0.0,
        zorder: float = 4.0,
        label: str | None = None,
        name: str | None = None,
        auto_extent: bool = True,
        colorbar: bool = False,
        colorbar_label: str | None = None,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)

        scatter_kwargs = {
            "s": float(size),
            "transform": crs,
            "zorder": float(zorder),
            "alpha": float(alpha),
            "linewidths": float(linewidths),
        }
        if edgecolors is not None:
            scatter_kwargs["edgecolors"] = edgecolors
        if label is not None:
            scatter_kwargs["label"] = label
        if values is not None:
            scatter_kwargs.update({"c": np.asarray(values), "cmap": cmap, "vmin": vmin, "vmax": vmax})
        else:
            scatter_kwargs["c"] = color if color is not None else "C0"
        artist = self.ax.scatter(lon, lat, **scatter_kwargs)
        layer = self._register_layer(
            kind="points",
            artists=[artist],
            name=name or label,
            metadata={"bounds": bounds, "source_crs": crs},
        )
        if colorbar and values is not None:
            self.add_colorbar(layer, label=colorbar_label)
        return layer

    def add_trace(
        self,
        lon_deg: Any,
        lat_deg: Any,
        *,
        src_crs: Any | None = None,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1.5,
        linestyle: str = "-",
        marker: str | None = None,
        markersize: float = 2.0,
        zorder: float = 4.0,
        label: str | None = None,
        name: str | None = None,
        glow: bool = False,
        auto_extent: bool = True,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        artists = add_geodesic_trace(
            self.ax,
            lat,
            lon,
            src_crs=crs,
            linewidth=float(linewidth),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=float(markersize),
            zorder=float(zorder),
            label=label,
            glow=glow,
            plot_kwargs={"alpha": float(alpha)},
        )
        return self._register_layer(
            kind="trace",
            artists=artists,
            name=name or label,
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_ground_track(
        self,
        lon_deg: Any,
        lat_deg: Any,
        *,
        src_crs: Any | None = None,
        color: str = "C0",
        opacity: float = 1.0,
        line_width: float = 1.5,
        marker_latlon: tuple[float, float] | None = None,
        marker_size: float = 6.0,
        label: str | None = None,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)

        trace_layer = self.add_trace(
            lon,
            lat,
            src_crs=crs,
            color=color,
            alpha=0.9 * float(opacity),
            linewidth=float(line_width),
            label=label,
            auto_extent=False,
        )
        artists = list(trace_layer.artists)
        halo = None
        marker = None
        if marker_latlon is not None:
            marker_lat, marker_lon = marker_latlon
            halo = self.ax.scatter(
                [marker_lon],
                [marker_lat],
                transform=crs,
                s=(float(marker_size) * _GROUND_TRACK_HALO_SCALE) ** 2,
                c=[color],
                alpha=0.15 * float(opacity),
                linewidths=0.0,
                zorder=4,
            )
            marker = self.ax.scatter(
                [marker_lon],
                [marker_lat],
                transform=crs,
                s=float(marker_size) ** 2,
                c=[color],
                alpha=float(opacity),
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
            artists.extend([halo, marker])
        self.layers.pop()
        return self._register_layer(
            kind="ground_track",
            artists=artists,
            name=name or label,
            metadata={
                "bounds": bounds,
                "source_crs": crs,
                "trace_artists": trace_layer.artists,
                "halo": halo,
                "marker": marker,
            },
        )

    def add_raster(
        self,
        data: Any,
        *,
        lon_deg: Any = None,
        lat_deg: Any = None,
        extent: tuple[float, float, float, float] | None = None,
        src_crs: Any | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 1.0,
        zorder: float = 4.0,
        name: str | None = None,
        auto_extent: bool = True,
        colorbar: bool = False,
        colorbar_label: str | None = None,
        interpolation: str = "nearest",
        origin: str = "upper",
        rasterized: bool = True,
    ) -> MapLayer:
        values = np.asarray(data)
        if _is_xarray_dataarray(data):
            values, lon_deg, lat_deg = _extract_dataarray_grid(data, lon_deg=lon_deg, lat_deg=lat_deg)
        crs = _coerce_crs(src_crs)

        if extent is not None:
            artist = self.ax.imshow(
                values,
                extent=extent,
                transform=crs,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=float(alpha),
                interpolation=interpolation,
                origin=origin,
                zorder=float(zorder),
                rasterized=bool(rasterized),
            )
            bounds = _transform_bounds_to_platecarree(tuple(float(v) for v in extent), src_crs=crs)
        else:
            if lon_deg is None or lat_deg is None:
                raise ValueError("add_raster requires lon_deg/lat_deg coordinates unless extent is provided")
            lon = np.asarray(lon_deg, dtype=np.float64)
            lat = np.asarray(lat_deg, dtype=np.float64)
            if lon.ndim == 1 and lat.ndim == 1:
                artist = self.ax.pcolormesh(
                    _axis_edges(lon),
                    _axis_edges(lat),
                    values,
                    transform=crs,
                    cmap=cmap,
                    shading="auto",
                    rasterized=bool(rasterized),
                    zorder=float(zorder),
                    vmin=vmin,
                    vmax=vmax,
                    alpha=float(alpha),
                )
            else:
                artist = self.ax.pcolormesh(
                    lon,
                    lat,
                    values,
                    transform=crs,
                    cmap=cmap,
                    shading="auto",
                    rasterized=bool(rasterized),
                    zorder=float(zorder),
                    vmin=vmin,
                    vmax=vmax,
                    alpha=float(alpha),
                )
            bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)

        if auto_extent:
            self._update_auto_extent(bounds)
        layer = self._register_layer(
            kind="raster",
            artists=[artist],
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )
        if colorbar:
            self.add_colorbar(layer, label=colorbar_label)
        return layer

    def add_contours(
        self,
        data: Any,
        *,
        lon_deg: Any = None,
        lat_deg: Any = None,
        src_crs: Any | None = None,
        levels: int | Sequence[float] = 10,
        colors: Any = None,
        cmap: str | None = None,
        linewidths: float | Sequence[float] = 1.0,
        alpha: float = 1.0,
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        values = np.asarray(data)
        if _is_xarray_dataarray(data):
            values, lon_deg, lat_deg = _extract_dataarray_grid(data, lon_deg=lon_deg, lat_deg=lat_deg)
        if lon_deg is None or lat_deg is None:
            raise ValueError("add_contours requires lon_deg and lat_deg coordinates")
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        artist = self.ax.contour(
            lon,
            lat,
            values,
            transform=crs,
            levels=levels,
            colors=colors,
            cmap=cmap,
            linewidths=linewidths,
            alpha=float(alpha),
            zorder=float(zorder),
        )
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(
            kind="contours",
            artists=[artist],
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_filled_contours(
        self,
        data: Any,
        *,
        lon_deg: Any = None,
        lat_deg: Any = None,
        src_crs: Any | None = None,
        levels: int | Sequence[float] = 10,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 1.0,
        zorder: float = 4.0,
        name: str | None = None,
        auto_extent: bool = True,
        colorbar: bool = False,
        colorbar_label: str | None = None,
    ) -> MapLayer:
        values = np.asarray(data)
        if _is_xarray_dataarray(data):
            values, lon_deg, lat_deg = _extract_dataarray_grid(data, lon_deg=lon_deg, lat_deg=lat_deg)
        if lon_deg is None or lat_deg is None:
            raise ValueError("add_filled_contours requires lon_deg and lat_deg coordinates")
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        artist = self.ax.contourf(
            lon,
            lat,
            values,
            transform=crs,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=float(alpha),
            zorder=float(zorder),
        )
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        layer = self._register_layer(
            kind="filled_contours",
            artists=[artist],
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )
        if colorbar:
            self.add_colorbar(layer, label=colorbar_label)
        return layer

    def add_quiver(
        self,
        lon_deg: Any,
        lat_deg: Any,
        u: Any,
        v: Any,
        *,
        src_crs: Any | None = None,
        color: Any = None,
        cmap: str | None = None,
        scale: float | None = None,
        alpha: float = 1.0,
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        quiver_kwargs = {
            "transform": crs,
            "alpha": float(alpha),
            "zorder": float(zorder),
        }
        if color is not None:
            quiver_kwargs["color"] = color
        if cmap is not None:
            quiver_kwargs["cmap"] = cmap
        if scale is not None:
            quiver_kwargs["scale"] = float(scale)
        artist = self.ax.quiver(lon, lat, np.asarray(u), np.asarray(v), **quiver_kwargs)
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(
            kind="quiver",
            artists=[artist],
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_streamplot(
        self,
        lon_deg: Any,
        lat_deg: Any,
        u: Any,
        v: Any,
        *,
        src_crs: Any | None = None,
        color: Any = None,
        cmap: str | None = None,
        density: float | tuple[float, float] = 1.0,
        linewidth: float | Any = 1.0,
        arrowsize: float = 1.0,
        alpha: float = 1.0,
        zorder: float = 5.0,
        name: str | None = None,
        auto_extent: bool = True,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        artist = self.ax.streamplot(
            lon,
            lat,
            np.asarray(u),
            np.asarray(v),
            transform=crs,
            color=color,
            cmap=cmap,
            density=density,
            linewidth=linewidth,
            arrowsize=float(arrowsize),
            zorder=float(zorder),
        )
        try:
            artist.lines.set_alpha(float(alpha))
        except Exception:
            pass
        try:
            artist.arrows.set_alpha(float(alpha))
        except Exception:
            pass
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(
            kind="streamplot",
            artists=[artist],
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_text(
        self,
        lon_deg: float,
        lat_deg: float,
        text: str,
        *,
        src_crs: Any | None = None,
        color: Any = None,
        fontsize: float | None = None,
        ha: str = "center",
        va: str = "center",
        zorder: float = 7.0,
        name: str | None = None,
        auto_extent: bool = False,
        **kwargs: Any,
    ) -> MapLayer:
        crs = _coerce_crs(src_crs)
        artist = self.ax.text(
            float(lon_deg),
            float(lat_deg),
            str(text),
            transform=crs,
            color=color,
            fontsize=fontsize,
            ha=ha,
            va=va,
            zorder=float(zorder),
            **kwargs,
        )
        bounds = _bounds_from_lonlat(np.asarray([lon_deg]), np.asarray([lat_deg]), src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(
            kind="text",
            artists=[artist],
            name=name or str(text),
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_labels(
        self,
        lon_deg: Any,
        lat_deg: Any,
        labels: Sequence[str],
        *,
        src_crs: Any | None = None,
        color: Any = None,
        fontsize: float | None = None,
        ha: str = "center",
        va: str = "center",
        zorder: float = 7.0,
        name: str | None = None,
        auto_extent: bool = False,
        **kwargs: Any,
    ) -> MapLayer:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        crs = _coerce_crs(src_crs)
        artists = [
            self.ax.text(
                float(lon_i),
                float(lat_i),
                str(label),
                transform=crs,
                color=color,
                fontsize=fontsize,
                ha=ha,
                va=va,
                zorder=float(zorder),
                **kwargs,
            )
            for lon_i, lat_i, label in zip(lon, lat, labels, strict=False)
        ]
        bounds = _bounds_from_lonlat(lon, lat, src_crs=crs)
        if auto_extent:
            self._update_auto_extent(bounds)
        return self._register_layer(
            kind="labels",
            artists=artists,
            name=name,
            metadata={"bounds": bounds, "source_crs": crs},
        )

    def add_field(
        self,
        field: Any,
        *,
        title: str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str | None = None,
        alpha: float = 0.82,
        outline: bool = True,
        outline_color: str = "#111111",
        outline_width: float = 0.8,
        outline_alpha: float = 0.95,
        point_size: float = 16.0,
        render: str = "auto",
        name: str | None = None,
    ) -> MapLayer:
        targets = getattr(field, "targets", None)
        bounds = _bounds_from_targets(targets)
        self._update_auto_extent(bounds)
        artist = _render_field_artist(
            self.ax,
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=float(alpha),
            point_size=float(point_size),
            render=render,
        )
        artists: list[Any] = [artist]

        if outline and targets is not None:
            outline_geom = getattr(targets, "outline_geometry", None)
            if outline_geom is None:
                outline_geom = getattr(targets, "boundary_geometry", None)
            if outline_geom is not None:
                artists.extend(
                    self._geometry_artists(
                        [outline_geom],
                        facecolor=None,
                        edgecolor=outline_color,
                        edge_alpha=outline_alpha,
                        linewidth=outline_width,
                        zorder=5.5,
                    )
                )

        plot_title = title or getattr(field, "label", None) or getattr(field, "metric_name", None)
        if plot_title:
            self.set_title(plot_title)

        cbar = None
        if colorbar:
            label = colorbar_label or getattr(field, "unit", None)
            cbar = self.fig.colorbar(artist, ax=self.ax, pad=0.02)
            if label:
                cbar.set_label(label)

        return self._register_layer(
            kind="field",
            artists=artists,
            name=name or plot_title,
            colorbar=cbar,
            metadata={"bounds": bounds, "targets": targets},
        )

    def add(self, data: Any, *args: Any, kind: str | None = None, **kwargs: Any) -> MapLayer:
        """Add a layer with type-based dispatch for common geo data objects."""
        normalized_kind = _normalize_kind(kind)

        if normalized_kind == "artist":
            return self.add_artist(data, **kwargs)
        if normalized_kind == "feature":
            return self.add_feature(data, **kwargs)
        if normalized_kind == "field":
            return self.add_field(data, **kwargs)
        if normalized_kind == "geometry":
            return self.add_geometry(data, **kwargs)
        if normalized_kind == "geometries":
            return self.add_geometries(data, **kwargs)
        if normalized_kind == "geojson":
            return self.add_geojson(data, **kwargs)
        if normalized_kind == "country":
            return self.add_country(str(data), **kwargs)
        if normalized_kind == "countries":
            return self.add_countries(data, **kwargs)
        if normalized_kind == "geodataframe":
            return self.add_geodataframe(data, **kwargs)
        if normalized_kind == "points":
            x = kwargs.pop("x", None)
            y = kwargs.pop("y", None)
            lon, lat = _coerce_lon_lat_pair(data, args[0] if args else None, x=x, y=y)
            values = kwargs.get("values")
            if isinstance(values, str) and (x is not None or y is not None):
                kwargs["values"] = _coerce_column(data, values)
            return self.add_points(lon, lat, **kwargs)
        if normalized_kind == "trace":
            x = kwargs.pop("x", None)
            y = kwargs.pop("y", None)
            lon, lat = _coerce_lon_lat_pair(data, args[0] if args else None, x=x, y=y)
            return self.add_trace(lon, lat, **kwargs)
        if normalized_kind == "raster":
            return self.add_raster(data, **kwargs)
        if normalized_kind == "contours":
            return self.add_contours(data, **kwargs)
        if normalized_kind == "filled_contours":
            return self.add_filled_contours(data, **kwargs)
        if normalized_kind == "quiver":
            if len(args) < 3:
                raise ValueError("add(..., kind='quiver') requires lat_deg, u, and v arguments")
            lon, lat = _coerce_lon_lat_pair(data, args[0])
            return self.add_quiver(lon, lat, args[1], args[2], **kwargs)
        if normalized_kind == "streamplot":
            if len(args) < 3:
                raise ValueError("add(..., kind='streamplot') requires lat_deg, u, and v arguments")
            lon, lat = _coerce_lon_lat_pair(data, args[0])
            return self.add_streamplot(lon, lat, args[1], args[2], **kwargs)
        if normalized_kind == "text":
            if len(args) < 2:
                raise ValueError("add(..., kind='text') requires lat and text arguments")
            return self.add_text(float(data), float(args[0]), str(args[1]), **kwargs)
        if normalized_kind == "labels":
            x = kwargs.pop("x", None)
            y = kwargs.pop("y", None)
            text = kwargs.pop("text", None)
            lon, lat = _coerce_lon_lat_pair(data, args[0] if args else None, x=x, y=y)
            labels = args[1] if len(args) > 1 else text
            if isinstance(labels, str) and (x is not None or y is not None):
                labels = _coerce_column(data, labels)
            if labels is None:
                raise ValueError("add(..., kind='labels') requires label text or a text column")
            return self.add_labels(lon, lat, labels, **kwargs)

        if isinstance(data, MapLayer):
            self.layers.append(data)
            self.artist = data.artist
            self.colorbar = data.colorbar
            return data
        if hasattr(data, "targets") and hasattr(data, "values"):
            return self.add_field(data, **kwargs)
        if _looks_like_geometry(data):
            return self.add_geometry(data, **kwargs)
        if _looks_like_geojson_mapping(data):
            return self.add_geojson(data, **kwargs)
        if isinstance(data, (str, Path)) and str(data).lower().endswith((".geojson", ".json")):
            return self.add_geojson(data, **kwargs)
        if _looks_like_geodataframe(data):
            return self.add_geodataframe(data, **kwargs)
        if _is_xarray_dataarray(data):
            return self.add_raster(data, **kwargs)

        if len(args) == 1:
            raise ValueError(
                "Ambiguous lon/lat array input. Pass kind='points' or kind='trace' to make the intent explicit."
            )
        raise TypeError(f"Do not know how to add object of type {type(data)!r} to GeoMap")


__all__ = ["GeoMap", "MapLayer", "get_map_preset"]
