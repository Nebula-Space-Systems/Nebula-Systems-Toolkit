from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nstk.plotting.geo import GeoMap
    from nstk.plotting.map import CFeatureScale, MapConfig, MapStyle, MapView, ProjectionConfig


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
    style: str | "MapStyle" | None = None,
    theme: str | "MapConfig" | "MapStyle" | None = "light_detailed",
    view: "MapView" | None = None,
    projection: str | ProjectionConfig | None = None,
    extent: Any = None,
    pad_deg: float = 0.0,
    figsize: tuple[float, float] | None = None,
    grid: bool | None = None,
    coastlines: bool | None = None,
    borders: bool | None = None,
    frame: bool | None = None,
    cfeature_scale: CFeatureScale | None = None,
    map_cfg: MapConfig | None = None,
    ax: Any = None,
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
) -> "GeoMap":
    """Render a coverage field on a shared `GeoMap` canvas.

    The default behavior is intentionally easy to use:
    local target sets auto-zoom to their bounds, while global target sets stay
    global. Pass `extent="global"` or a `(west, east, south, north)` tuple to
    take direct control, pass a reusable `MapView` via `view=...`, or use
    `map_cfg` for expert-level overrides.
    """
    from nstk.plotting.geo import GeoMap

    effective_extent = extent
    if effective_extent is None and map_cfg is None and view is None:
        effective_extent = "auto"

    map_view = GeoMap(
        style=style,
        theme=theme,
        view=view,
        projection=projection,
        extent=effective_extent,
        pad_deg=pad_deg,
        figsize=figsize,
        grid=grid,
        coastlines=coastlines,
        borders=borders,
        frame=frame,
        cfeature_scale=cfeature_scale,
        ax=ax,
        map_cfg=map_cfg,
    )
    map_view.add_field(
        field,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        alpha=alpha,
        outline=outline,
        outline_color=outline_color,
        outline_width=outline_width,
        outline_alpha=outline_alpha,
        point_size=point_size,
        render=render,
    )
    return map_view
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
__all__ = [
    "plot_coverage_histogram",
    "plot_coverage_ecdf",
    "plot_coverage_map",
    "plot_target_timeline",
]
