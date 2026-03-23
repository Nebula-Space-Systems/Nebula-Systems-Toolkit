from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from nstk.coverage import (
    AccessIntervalStore,
    access_duration_by_target,
    calculate_gap_duration,
    calculate_max_asset,
    calculate_mtta,
    calculate_revisit_time,
)

from .map import LIGHT_DETAILED, make_basemap


TimeUnit = Literal["seconds", "minutes", "hours"]


def _time_unit_scale(unit: TimeUnit) -> tuple[float, str]:
    key = str(unit).lower()
    if key == "seconds":
        return 1.0, "seconds"
    if key == "minutes":
        return 1.0 / 60.0, "minutes"
    if key == "hours":
        return 1.0 / 3600.0, "hours"
    raise ValueError("unit must be 'seconds', 'minutes', or 'hours'")


def _latitude_edges_from_rows(lat_rows_deg: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat_rows_deg, dtype=np.float64)
    edges = np.empty(lat.size + 1, dtype=np.float64)
    if lat.size == 1:
        edges[0] = lat[0] - 0.5
        edges[1] = lat[0] + 0.5
    else:
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
        edges[0] = lat[0] - 0.5 * (lat[1] - lat[0])
        edges[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    return np.clip(edges, -90.0, 90.0)


def _rasterize_surface_field(
    store: AccessIntervalStore,
    values: np.ndarray,
    *,
    nlon_render: int = 720,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_deg, _, row_offsets, lat_rows_deg = store.require_surface_target_grid()
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != (store.n_targets,):
        raise ValueError(f"values must have shape ({store.n_targets},)")

    nlon = int(max(90, nlon_render))
    lon_edges = np.linspace(-180.0, 180.0, nlon + 1, dtype=np.float64)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lon_centers_360 = np.mod(lon_centers + 360.0, 360.0)

    grid = np.empty((lat_rows_deg.size, nlon), dtype=np.float64)
    for row_idx in range(lat_rows_deg.size):
        i0 = int(row_offsets[row_idx])
        i1 = int(row_offsets[row_idx + 1])

        row_vals = vals[i0:i1]
        if row_vals.size == 1:
            grid[row_idx, :] = row_vals[0]
            continue

        lon_row = np.mod(lon_deg[i0:i1] + 360.0, 360.0)
        order = np.argsort(lon_row)
        xp = lon_row[order]
        fp = row_vals[order]

        keep = np.empty(xp.size, dtype=bool)
        keep[0] = True
        keep[1:] = np.diff(xp) > 1e-12
        xp = xp[keep]
        fp = fp[keep]

        if xp.size == 1:
            grid[row_idx, :] = fp[0]
        else:
            grid[row_idx, :] = np.interp(lon_centers_360, xp, fp, period=360.0)

    lat_edges = _latitude_edges_from_rows(lat_rows_deg)
    return lon_edges, lat_edges, grid


def _make_title(
    title: str | None,
    default: str,
) -> str:
    return default if title is None else str(title)


def plot_interval_metric(
    store: AccessIntervalStore,
    values: np.ndarray,
    *,
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    colorbar_label: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.78,
    nlon_render: int = 720,
    colorbar: bool = True,
    colorbar_pad: float = 0.02,
) -> tuple[Any, Any, Any, Any]:
    """Plot a target-wise interval metric on a Cartopy map using nstk presets."""

    lon_edges, lat_edges, grid = _rasterize_surface_field(
        store,
        values,
        nlon_render=nlon_render,
    )

    cbar = None
    if ax is None:
        fig, ax, _, _ = make_basemap(map_cfg)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        rasterized=True,
        zorder=4,
        alpha=float(alpha),
        vmin=vmin,
        vmax=vmax,
    )
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, pad=colorbar_pad)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    return fig, ax, mesh, cbar


def plot_interval_access_duration(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    normalize_to_day: bool = True,
    unit: TimeUnit = "hours",
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute and plot interval access duration."""

    values = access_duration_by_target(
        store,
        N=int(N),
        t_start=t_start,
        t_stop=t_stop,
        normalize_to_day=bool(normalize_to_day),
        reshape=False,
    )
    scale, unit_label = _time_unit_scale(unit)
    values = values * scale
    suffix = f"{unit_label}/day" if normalize_to_day else unit_label
    return plot_interval_metric(
        store,
        values,
        map_cfg=map_cfg,
        ax=ax,
        title=_make_title(title, f"Access Duration (N>={int(N)})"),
        colorbar_label=suffix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def plot_interval_max_asset(
    store: AccessIntervalStore,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "cividis",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute and plot the maximum concurrent visible observers per target."""

    values = calculate_max_asset(
        store,
        t_start=t_start,
        t_stop=t_stop,
        reshape=False,
    ).astype(np.float64)
    return plot_interval_metric(
        store,
        values,
        map_cfg=map_cfg,
        ax=ax,
        title=_make_title(title, "Maximum Concurrent Observers"),
        colorbar_label="count",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def plot_interval_mtta(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    t_start: float | None = None,
    t_stop: float | None = None,
    wrap: bool = False,
    no_access_value: float = np.nan,
    unit: TimeUnit = "minutes",
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "magma_r",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute and plot interval MTTA."""

    values = calculate_mtta(
        store,
        N=int(N),
        t_start=t_start,
        t_stop=t_stop,
        wrap=bool(wrap),
        no_access_value=float(no_access_value),
        reshape=False,
    )
    scale, unit_label = _time_unit_scale(unit)
    return plot_interval_metric(
        store,
        values * scale,
        map_cfg=map_cfg,
        ax=ax,
        title=_make_title(title, f"Mean Time To Access (N>={int(N)})"),
        colorbar_label=unit_label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def plot_interval_gap_duration(
    store: AccessIntervalStore,
    *,
    min_assets: int = 1,
    stat: Literal["mean", "min", "max", "std", "count", "sum"] = "mean",
    include_end_gaps: bool = True,
    no_access_value: float = np.nan,
    nan_if_never_access: bool = False,
    t_start: float | None = None,
    t_stop: float | None = None,
    unit: TimeUnit = "minutes",
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "plasma_r",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute and plot interval gap duration statistics."""

    values = calculate_gap_duration(
        store,
        min_assets=int(min_assets),
        stat=stat,
        include_end_gaps=bool(include_end_gaps),
        no_access_value=float(no_access_value),
        nan_if_never_access=bool(nan_if_never_access),
        t_start=t_start,
        t_stop=t_stop,
        reshape=False,
    )
    label = "count"
    if stat != "count":
        scale, unit_label = _time_unit_scale(unit)
        values = values * scale
        label = unit_label
    return plot_interval_metric(
        store,
        values,
        map_cfg=map_cfg,
        ax=ax,
        title=_make_title(title, f"Gap Duration ({stat})"),
        colorbar_label=label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def plot_interval_revisit_time(
    store: AccessIntervalStore,
    *,
    N: int = 1,
    option: Literal["average", "maximum", "minimum", "std_deviation"] = "average",
    end_gaps: Literal["include", "ignore"] = "include",
    t_start: float | None = None,
    t_stop: float | None = None,
    unit: TimeUnit = "minutes",
    map_cfg: Any = LIGHT_DETAILED,
    ax: Any = None,
    title: str | None = None,
    cmap: str = "plasma_r",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute and plot interval revisit time."""

    values = calculate_revisit_time(
        store,
        N=int(N),
        option=option,
        end_gaps=end_gaps,
        t_start=t_start,
        t_stop=t_stop,
        reshape=False,
    )
    scale, unit_label = _time_unit_scale(unit)
    return plot_interval_metric(
        store,
        values * scale,
        map_cfg=map_cfg,
        ax=ax,
        title=_make_title(title, f"Revisit Time ({option})"),
        colorbar_label=unit_label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def _resolve_target_index(
    store: AccessIntervalStore,
    *,
    target_index: int | None,
    lat_deg: float | None,
    lon_deg: float | None,
) -> int:
    if target_index is not None:
        idx = int(target_index)
        if idx < 0 or idx >= store.n_targets:
            raise IndexError("target_index out of range")
        return idx
    if lat_deg is None or lon_deg is None:
        raise ValueError(
            "Provide either target_index or both lat_deg and lon_deg to select a target"
        )
    return store.nearest_target_index(lat_deg=float(lat_deg), lon_deg=float(lon_deg))


def _time_axis_scale(unit: TimeUnit) -> tuple[float, str]:
    scale, unit_label = _time_unit_scale(unit)
    return scale, unit_label


def _build_concurrency_profile(
    store: AccessIntervalStore,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    events: dict[float, int] = {}
    active0 = 0

    for observer_index in range(store.n_observers):
        starts, stops = store.pair_intervals(observer_index, target_index)
        for start, stop in zip(starts, stops):
            start = float(start)
            stop = float(stop)
            if start <= store.time_start + eps < stop:
                active0 += 1
            if start > store.time_start + eps and start < store.time_stop - eps:
                events[start] = events.get(start, 0) + 1
            if stop > store.time_start + eps and stop < store.time_stop - eps:
                events[stop] = events.get(stop, 0) - 1

    times = [float(store.time_start)]
    counts = [int(active0)]
    current = int(active0)
    for event_time in sorted(events):
        times.extend([event_time, event_time])
        counts.extend([current, current + events[event_time]])
        current += events[event_time]
    times.append(float(store.time_stop))
    counts.append(current)

    return np.asarray(times, dtype=np.float64), np.asarray(counts, dtype=np.int32)


def plot_interval_target_timeline(
    store: AccessIntervalStore,
    *,
    target_index: int | None = None,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    target_name: str | None = None,
    time_unit: TimeUnit = "hours",
    show_concurrency: bool = True,
    observer_labels: Optional[list[str]] = None,
    colors: Any = None,
    ax: Any = None,
    concurrency_ax: Any = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot exact per-observer access intervals for a selected target."""

    idx = _resolve_target_index(
        store,
        target_index=target_index,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    scale, unit_label = _time_axis_scale(time_unit)

    if ax is None:
        if show_concurrency:
            fig, (ax, concurrency_ax) = plt.subplots(
                2,
                1,
                figsize=(12, 6.5),
                sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1.2]},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure
        if show_concurrency and concurrency_ax is None:
            raise ValueError(
                "When show_concurrency=True and ax is provided, concurrency_ax must also be provided"
            )

    if observer_labels is not None and len(observer_labels) != store.n_observers:
        raise ValueError("observer_labels must have length store.n_observers")

    color_values = (
        plt.cm.tab10(np.linspace(0.0, 1.0, store.n_observers))
        if colors is None
        else colors
    )

    for observer_index in range(store.n_observers):
        starts, stops = store.pair_intervals(observer_index, idx)
        spans = [
            (float(start) * scale, float(stop - start) * scale)
            for start, stop in zip(starts, stops)
        ]
        if spans:
            ax.broken_barh(
                spans,
                (observer_index - 0.4, 0.8),
                facecolors=color_values[observer_index],
            )

    y_labels = (
        observer_labels
        if observer_labels is not None
        else [str(i) for i in range(store.n_observers)]
    )
    ax.set_yticks(range(store.n_observers))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Observer")
    ax.grid(axis="x", alpha=0.25)

    lon_arr = lat_arr = None
    if store.has_surface_target_grid():
        lon_arr, lat_arr, _, _ = store.require_surface_target_grid()

    location_text = ""
    if lat_arr is not None and lon_arr is not None:
        location_text = (
            f" (target at {lat_arr[idx]:.2f} deg, {lon_arr[idx]:.2f} deg)"
        )
    name_text = f" for {target_name}" if target_name else ""
    ax.set_title(
        _make_title(
            title,
            f"Exact Access Intervals{name_text}{location_text}",
        )
    )

    if show_concurrency and concurrency_ax is not None:
        step_t, step_n = _build_concurrency_profile(store, idx)
        concurrency_ax.step(
            step_t * scale,
            step_n,
            where="post",
            color="#d9480f",
            linewidth=2.0,
        )
        concurrency_ax.fill_between(
            step_t * scale,
            step_n,
            step="post",
            alpha=0.18,
            color="#f08c00",
        )
        concurrency_ax.set_ylabel("Concurrent\nobservers")
        concurrency_ax.set_xlabel(f"Time [{unit_label}]")
        concurrency_ax.set_ylim(-0.1, max(1.0, float(step_n.max()) + 0.5))
        concurrency_ax.grid(alpha=0.25)
        return fig, (ax, concurrency_ax)

    ax.set_xlabel(f"Time [{unit_label}]")
    return fig, ax


__all__ = [
    "plot_interval_metric",
    "plot_interval_access_duration",
    "plot_interval_max_asset",
    "plot_interval_mtta",
    "plot_interval_gap_duration",
    "plot_interval_revisit_time",
    "plot_interval_target_timeline",
]
