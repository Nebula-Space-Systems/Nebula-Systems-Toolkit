import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nebula_old.plots.country_shapes import fuzzy_find_countries
from nebula_old.plots.map import (
    make_basemap,
    add_geodesic_trace,
    add_polygon,
    DARK_DETAILED,
    LIGHT_DETAILED,
    DARK_RASTER,
    LIGHT_RASTER,
    DARK,
    LIGHT,
    DARK_DETAILED_NO_GRID,
    LIGHT_DETAILED_NO_GRID,
    DARK_NO_GRID,
    LIGHT_NO_GRID,
    DARK_RASTER_NO_GRID,
    LIGHT_RASTER_NO_GRID,
    plt,
)

# -----------------------------------------------------------------------------
# Test traces (exercise: dateline split, NaN split, wrap/no-wrap, dense points,
#               degenerate cases, invalids, projection cutoffs)
# -----------------------------------------------------------------------------


def _trace_dateline_crossing():
    # Crosses the dateline near the Aleutians
    lat = np.array([52, 53, 54, 55, 56, 57, 58], dtype=float)
    lon = np.array([170, 175, 179, -179, -175, -170, -165], dtype=float)
    return lat, lon


def _trace_dateline_multi_crossing():
    # Multiple crossings to stress segmentation logic
    lat = np.array([10, 12, 14, 16, 18, 20, 22, 24], dtype=float)
    lon = np.array([160, 170, 179, -179, -170, -160, 179, -179], dtype=float)
    return lat, lon


def _trace_with_nans():
    # Two segments with NaN break; second segment also crosses dateline
    lat = np.array([35, 36, 37, np.nan, 40, 41, 42, 43], dtype=float)
    lon = np.array([-10, 0, 10, np.nan, 170, 179, -179, -170], dtype=float)
    return lat, lon


def _trace_with_infs():
    # Inf/NaN mix: should behave like NaN breaks when allow_nan=True (drops bad points)
    lat = np.array([0, 5, np.inf, 10, 15, np.nan, 20, 25], dtype=float)
    lon = np.array([0, 10, 20, np.inf, 40, np.nan, 60, 70], dtype=float)
    return lat, lon


def _trace_near_poles():
    # High-latitude track (projection stress)
    lat = np.array([70, 72, 74, 76, 78, 80, 82], dtype=float)
    lon = np.array([-60, -20, 20, 60, 100, 140, 179], dtype=float)
    return lat, lon


def _trace_crosses_prime_meridian():
    # Crosses lon=0 seam (some projections show artifacts if splitting logic is wrong)
    lat = np.array([10, 12, 14, 16, 18], dtype=float)
    lon = np.array([-5, -2, 0, 2, 5], dtype=float)
    return lat, lon


def _trace_dense_sinewave():
    # Dense global-ish oscillation, forces many points and wrap issues
    lon = np.linspace(-200, 200, 1600)  # intentionally exceeds [-180,180]
    lat = 20.0 * np.sin(np.deg2rad(lon * 2.0))
    return lat.astype(float), lon.astype(float)


def _trace_sparse_long_segments():
    # Very sparse points with big gaps (tests densify=True)
    lat = np.array([0, 0, 0, 0], dtype=float)
    lon = np.array([-170, -30, 110, 179], dtype=float)
    return lat, lon


def _trace_lon_wrap_center_180_positive():
    # Crosses 180 in the + direction: [179, 181] style
    lon = np.array([140, 160, 179, 181, 200, 220], dtype=float)
    lat = np.array([0, 5, 10, 15, 10, 5], dtype=float)
    return lat, lon


def _trace_lon_wrap_center_180_negative():
    # Crosses -180 in the - direction: [-179, -181] style
    lon = np.array([-140, -160, -179, -181, -200, -220], dtype=float)
    lat = np.array([5, 10, 15, 10, 5, 0], dtype=float)
    return lat, lon


def _trace_exact_dateline_values():
    # Uses exactly +180 and -180; stresses wrap normalization and seam splitting.
    lat = np.array([0, 5, 10, 15, 10, 5], dtype=float)
    lon = np.array([170, 179.999, 180.0, -180.0, -179.999, -170], dtype=float)
    return lat, lon


def _trace_single_point():
    lat = np.array([10.0], dtype=float)
    lon = np.array([20.0], dtype=float)
    return lat, lon


def _trace_two_points_identical():
    lat = np.array([10.0, 10.0], dtype=float)
    lon = np.array([20.0, 20.0], dtype=float)
    return lat, lon


def _trace_all_nan():
    lat = np.array([np.nan, np.nan], dtype=float)
    lon = np.array([np.nan, np.nan], dtype=float)
    return lat, lon


def _trace_out_of_range_lat():
    # Out-of-range latitudes (Cartopy may project to non-finite / clip)
    lat = np.array([85, 92, 88, -95, -80], dtype=float)
    lon = np.array([0, 10, 20, 30, 40], dtype=float)
    return lat, lon


def _trace_out_of_range_lon():
    # Very large longitudes (wrap logic should tame if enabled)
    lat = np.array([0, 10, 0, -10, 0], dtype=float)
    lon = np.array([0, 450, 720, -540, -1080], dtype=float)
    return lat, lon


def _trace_antimeridian_zigzag():
    # Zigzagging around the dateline by small amounts
    lat = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    lon = np.array([179.5, -179.6, 179.7, -179.8, 179.9, -179.9], dtype=float)
    return lat, lon


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def add_test_traces(ax, *, theme_default="C1"):
    """
    Add a comprehensive suite of traces to one axes to exercise add_geodesic_trace().

    Covers:
    - dateline split (single + multi)
    - NaN/Inf handling
    - wrap center choices
    - exact +/-180 values
    - dense vs sparse (densify)
    - degenerate traces (single point, identical points)
    - all NaN
    - out-of-range inputs (lat/lon)
    - zigzag around antimeridian
    - prime meridian crossing
    - per-call parameter variations + plot_kwargs passthrough
    """

    # Helper to keep legend labels from duplicating across many segments
    def _label_once(lbl: str):
        return lbl

    # 1) Simple dateline crossing
    lat, lon = _trace_dateline_crossing()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color=theme_default,
        linewidth=2.2,
        glow=True,
        glow_alpha=0.35,
        label=_label_once("dateline crossing"),
        plot_kwargs={"solid_capstyle": "round"},
    )

    # 2) Multiple crossings (more splits)
    lat, lon = _trace_dateline_multi_crossing()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C2",
        linewidth=1.6,
        label=_label_once("multi dateline"),
        plot_kwargs={"linestyle": "--"},
    )

    # 3) NaN split + dateline split
    lat, lon = _trace_with_nans()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C3",
        linewidth=2.0,
        glow=False,
        label=_label_once("NaN breaks"),
        plot_kwargs={"marker": "o", "markersize": 3},
    )

    # 4) Inf + NaN mix (should drop invalid projected points cleanly)
    lat, lon = _trace_with_infs()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C7",
        linewidth=1.8,
        label=_label_once("Inf/NaN mix"),
        plot_kwargs={"linestyle": ":"},
    )

    # 5) Polar-ish segment
    lat, lon = _trace_near_poles()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C4",
        linewidth=1.8,
        label=_label_once("high-latitude"),
        plot_kwargs={"alpha": 0.9},
    )

    # 6) Crosses prime meridian (should not split)
    lat, lon = _trace_crosses_prime_meridian()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C0",
        linewidth=1.4,
        label=_label_once("prime-meridian"),
        plot_kwargs={"alpha": 0.9},
    )

    # 7) Dense global oscillation (wrap + many points)
    lat, lon = _trace_dense_sinewave()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C5",
        linewidth=1.1,
        label=_label_once("dense sine"),
        plot_kwargs={"alpha": 0.75},
    )

    # 8) Sparse long segments WITHOUT densify (may look “chordy”)
    lat, lon = _trace_sparse_long_segments()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C6",
        linewidth=1.4,
        label=_label_once("sparse (no densify)"),
        plot_kwargs={"linestyle": "-."},
    )

    # 9) Sparse long segments WITH densify (should look smoother in non-PlateCarree projections)
    lat, lon = _trace_sparse_long_segments()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="C6",
        linewidth=2.2,
        label=_label_once("sparse (densify)"),
        densify=True,
        densify_step_deg=2.0,
        plot_kwargs={"alpha": 0.55},
    )

    # 10) Wrap center @180 positive crossing (Basemap-style “Pacific centered”)
    lat, lon = _trace_lon_wrap_center_180_positive()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="magenta",
        linewidth=2.0,
        label=_label_once("wrap center 180 (+)"),
        wrap_center_deg=180.0,
        split_on_wrap=True,
    )

    # 11) Wrap center @180 negative crossing
    lat, lon = _trace_lon_wrap_center_180_negative()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="deeppink",
        linewidth=2.0,
        label=_label_once("wrap center 180 (-)"),
        wrap_center_deg=180.0,
        split_on_wrap=True,
        plot_kwargs={"linestyle": "--"},
    )

    # 12) Exact +/-180 values
    lat, lon = _trace_exact_dateline_values()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="orange",
        linewidth=2.0,
        label=_label_once("exact +/-180"),
        wrap_center_deg=0.0,
        split_on_wrap=True,
    )

    # 13) Antimeridian zigzag (many small crossings)
    lat, lon = _trace_antimeridian_zigzag()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="limegreen",
        linewidth=1.8,
        label=_label_once("anti-meridian zigzag"),
        wrap_center_deg=0.0,
        split_on_wrap=True,
        plot_kwargs={"linestyle": "-"},
    )

    # 14) Out-of-range longitude values (wrap logic should normalize)
    lat, lon = _trace_out_of_range_lon()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="gold",
        linewidth=1.8,
        label=_label_once("lon out-of-range"),
        wrap_center_deg=0.0,
        split_on_wrap=True,
        plot_kwargs={"alpha": 0.8},
    )

    # 15) Out-of-range latitude values (some points may drop; should not crash)
    lat, lon = _trace_out_of_range_lat()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="sienna",
        linewidth=1.8,
        label=_label_once("lat out-of-range"),
        plot_kwargs={"alpha": 0.75},
    )

    # 16) Degenerate: single point (should draw nothing, no crash)
    lat, lon = _trace_single_point()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="black",
        linewidth=3.0,
        label=_label_once("single point"),
    )

    # 17) Degenerate: two identical points (may draw nothing or tiny segment; must not crash)
    lat, lon = _trace_two_points_identical()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="black",
        linewidth=3.0,
        label=_label_once("two identical"),
        plot_kwargs={"marker": "x", "markersize": 6},
    )

    # 18) All NaN (should do nothing, no crash)
    lat, lon = _trace_all_nan()
    add_geodesic_trace(
        ax,
        lat,
        lon,
        color="black",
        linewidth=3.0,
        label=_label_once("all NaN"),
    )

    # Legend
    try:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fontsize=8,
            frameon=True,
            fancybox=True,
            ncol=4,
        )
        fig.tight_layout()
    except Exception:
        pass


if __name__ == "__main__":
    # Build all basemaps and add the same test traces to each
    configs = [
        LIGHT,
        DARK,
        LIGHT_NO_GRID,
        DARK_NO_GRID,
        LIGHT_DETAILED,
        DARK_DETAILED,
        LIGHT_DETAILED_NO_GRID,
        DARK_DETAILED_NO_GRID,
        LIGHT_RASTER,
        DARK_RASTER,
        LIGHT_RASTER_NO_GRID,
        DARK_RASTER_NO_GRID,
    ]
    from dataclasses import replace

    for cfg in configs:
        # cfg = replace(cfg, cfeature_scale="110m")
        # cfg = replace(cfg, extent=replace(cfg.extent, extent=(-10, 30, 35, 60)))
        fig, ax, crs, gl = make_basemap(cfg)
        theme_default = getattr(getattr(cfg, "theme", None), "trace_default", "C1")
        # add_test_traces(ax, theme_default=theme_default)
        add_polygon(
            ax,
            fuzzy_find_countries("Lesotho", "10m")[0].record,
            facecolor="#DA1A1A",
            edgecolor=None,
            fill_alpha=1,
        )
        add_polygon(
            ax,
            np.array([[90, 200, 200, 90], [-10, -10, 40, 40]]).T,
            facecolor="#1A7FDA",
        )
        plt.show()

    plt.show()
