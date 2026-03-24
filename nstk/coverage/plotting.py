from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _coerce_values(obj: Any) -> np.ndarray:
    if hasattr(obj, "values"):
        return np.asarray(obj.values)
    return np.asarray(obj)


def _histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    n_bins = max(1, int(bins))
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    upper = float(np.nextafter(vmax, np.inf))
    if not np.isfinite(upper) or upper <= vmin:
        upper = vmin + 1.0
    return np.linspace(vmin, upper, n_bins + 1, dtype=np.float64)


def plot_coverage_histogram(obj: Any, *, bins: int = 40, ax: Any = None, title: str | None = None) -> tuple[Any, Any]:
    values = _coerce_values(obj).astype(np.float64).ravel()
    values = values[np.isfinite(values)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.hist(values, bins=_histogram_edges(values, bins), color="#1971c2", alpha=0.85, edgecolor="white")
    ax.set_xlabel(getattr(obj, "label", None) or getattr(obj, "metric_name", "Value"))
    ax.set_ylabel("Count")
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.2)
    return fig, ax


def plot_coverage_ecdf(
    obj: Any,
    *,
    ax: Any = None,
    title: str | None = None,
    weights: np.ndarray | None = None,
) -> tuple[Any, Any]:
    values = _coerce_values(obj).astype(np.float64).ravel()
    mask = np.isfinite(values)
    values = values[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()[mask]
    else:
        weights = np.ones(values.size, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.step(values, cdf, where="post", color="#d9480f", linewidth=2.0)
    ax.set_xlabel(getattr(obj, "label", None) or getattr(obj, "metric_name", "Value"))
    ax.set_ylabel("ECDF")
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.2)
    return fig, ax


def plot_coverage_map(
    field: Any,
    *,
    map_cfg: Any = None,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    alpha: float = 0.82,
    boundary: bool = True,
    point_size: float = 16.0,
    render: str = "auto",
) -> tuple[Any, Any, Any, Any]:
    try:
        import cartopy.crs as ccrs
    except Exception as exc:  # pragma: no cover - exercised when cartopy is missing
        raise RuntimeError(
            "plot_coverage_map requires cartopy. Install the plotting extra."
        ) from exc

    try:
        import matplotlib.tri as mtri
    except Exception:  # pragma: no cover - matplotlib always provides this in supported environments
        mtri = None

    from matplotlib.colors import to_rgba

    from nstk.plotting.map import LIGHT_DETAILED, add_polygon, make_basemap
    from shapely import contains_xy

    targets = field.targets
    values = np.asarray(field.values, dtype=np.float64)
    cbar = None
    if ax is None:
        fig, ax, _, _ = make_basemap(LIGHT_DETAILED if map_cfg is None else map_cfg)
    else:
        fig = ax.figure

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
        artist = ax.scatter(
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
    elif targets.surface_grid is not None:
        lon = targets.surface_grid.lon_deg
        lat = targets.surface_grid.lat_deg
        lon_clip = None
        lat_clip = None
        grid_boundary = targets.surface_grid.boundary_geometry or targets.boundary_geometry
        if grid_boundary is not None and not getattr(grid_boundary, "is_empty", False):
            min_lon, min_lat, max_lon, max_lat = grid_boundary.bounds
            lon_clip = (float(min_lon), float(max_lon))
            lat_clip = (float(min_lat), float(max_lat))
        lon_edges = _axis_edges(
            lon,
            clip_min=None if lon_clip is None else lon_clip[0],
            clip_max=None if lon_clip is None else lon_clip[1],
        )
        lat_edges = _axis_edges(
            lat,
            clip_min=None if lat_clip is None else lat_clip[0],
            clip_max=None if lat_clip is None else lat_clip[1],
        )
        grid = targets.to_grid(values, fill_value=np.nan)
        artist = ax.pcolormesh(
            lon_edges,
            lat_edges,
            grid,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="auto",
            rasterized=True,
            zorder=4,
            vmin=vmin,
            vmax=vmax,
            alpha=float(alpha),
        )
    elif has_adaptive_rows:
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

        artist = ax.pcolormesh(
            _axis_edges(lon_axis_cont, clip_min=lon_min_cont, clip_max=lon_max_cont),
            _axis_edges(
                lat_rows,
                clip_min=lat_clip_min,
                clip_max=lat_clip_max,
            ),
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
    else:
        if render_key == "surface" and mtri is None:
            raise RuntimeError("render='surface' requires matplotlib.tri")
        if mtri is None:
            artist = ax.scatter(
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
        else:
            lon = np.asarray(targets.lon_deg, dtype=np.float64)
            lat = np.asarray(targets.lat_deg, dtype=np.float64)
            triangulation = mtri.Triangulation(lon, lat)
            tri_lon = lon[triangulation.triangles]
            tri_lat = lat[triangulation.triangles]
            tri_mask = (np.ptp(tri_lon, axis=1) > 40.0) | (np.ptp(tri_lat, axis=1) > 20.0)
            if np.any(tri_mask):
                triangulation.set_mask(tri_mask)
            artist = ax.tripcolor(
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

    outline_geom = getattr(targets, "outline_geometry", None)
    if outline_geom is None:
        outline_geom = targets.boundary_geometry

    if boundary and outline_geom is not None:
        geom_type = getattr(outline_geom, "geom_type", None)
        if geom_type in {"Polygon", "MultiPolygon"}:
            add_polygon(
                ax,
                outline_geom,
                facecolor=None,
                edgecolor="#111111",
                edge_alpha=0.95,
                linewidth=0.8,
                zorder=5.5,
            )
        else:
            ax.add_geometries(
                [outline_geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor=to_rgba("#111111", 0.95),
                linewidth=0.8,
                zorder=5.5,
            )

    plot_title = title or getattr(field, "label", None) or getattr(field, "metric_name", None)
    if plot_title:
        ax.set_title(plot_title)
    if colorbar:
        cbar = fig.colorbar(artist, ax=ax, pad=0.02)
        label = colorbar_label or getattr(field, "unit", None)
        if label:
            cbar.set_label(label)
    return fig, ax, artist, cbar


def plot_coverage_small_multiples(
    stack: Any,
    *,
    dim: str,
    map_cfg: Any = None,
    ncols: int = 2,
    cmap: str = "viridis",
    **map_kwargs: Any,
) -> tuple[Any, np.ndarray]:
    try:
        import cartopy.crs as ccrs
    except Exception as exc:  # pragma: no cover - exercised when cartopy is missing
        raise RuntimeError(
            "plot_coverage_small_multiples requires cartopy. Install the plotting extra."
        ) from exc

    if dim not in stack.dims:
        raise ValueError(f"{dim!r} is not a dimension on this CoverageStack")
    axis = stack.dims.index(dim)
    coord = np.asarray(stack.coords[dim])
    n_panels = int(coord.size)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 3.8 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax in axes_arr[n_panels:]:
        ax.set_visible(False)
    for panel_idx, value in enumerate(coord):
        field = stack.sel(**{dim: value})
        plot_coverage_map(
            field,
            map_cfg=map_cfg,
            ax=axes_arr[panel_idx],
            title=f"{field.metric_name}: {dim}={value}",
            cmap=cmap,
            **map_kwargs,
        )
    return fig, axes


def plot_target_timeline(
    timeline: Any,
    *,
    ax: Any = None,
    concurrency_ax: Any = None,
    show_concurrency: bool = True,
    time_unit: str = "hours",
    title: str | None = None,
) -> tuple[Any, Any]:
    scale = {"seconds": 1.0, "minutes": 1.0 / 60.0, "hours": 1.0 / 3600.0}[time_unit]
    if ax is None:
        if show_concurrency:
            fig, (ax, concurrency_ax) = plt.subplots(
                2,
                1,
                figsize=(11, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [2.4, 1.0]},
            )
        else:
            fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    labels = timeline.observer_names or [str(i) for i in range(len(timeline.starts_by_observer))]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(labels))))
    for idx, (starts, stops) in enumerate(zip(timeline.starts_by_observer, timeline.stops_by_observer)):
        spans = [(float(start) * scale, float(stop - start) * scale) for start, stop in zip(starts, stops)]
        if spans:
            ax.broken_barh(spans, (idx - 0.4, 0.8), facecolors=colors[idx])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel("Observer")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(title or timeline.target_label or "Target Timeline")

    if show_concurrency and concurrency_ax is not None:
        t, n = timeline.concurrency_profile()
        concurrency_ax.step(t * scale, n, where="post", color="#d9480f", linewidth=2.0)
        concurrency_ax.fill_between(t * scale, n, step="post", alpha=0.18, color="#f08c00")
        concurrency_ax.set_ylabel("Concurrent")
        concurrency_ax.set_xlabel(f"Time [{time_unit}]")
        concurrency_ax.grid(alpha=0.2)
        return fig, (ax, concurrency_ax)

    ax.set_xlabel(f"Time [{time_unit}]")
    return fig, ax


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


__all__ = [
    "plot_coverage_histogram",
    "plot_coverage_ecdf",
    "plot_coverage_map",
    "plot_coverage_small_multiples",
    "plot_target_timeline",
]
