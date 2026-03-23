from __future__ import annotations

import cartopy.mpl.geoaxes as cartopy_geoaxes
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from nstk.coverage import ExactCoverageConfig, compute_access_intervals
from nstk.plotting import (
    LIGHT_DETAILED,
    plot_interval_access_duration,
    plot_interval_max_asset,
    plot_interval_target_timeline,
)


def _build_store():
    config = ExactCoverageConfig(
        nlats=9,
        nlons_equator=17,
        scale_longitude_by_latitude=True,
        min_elevation_deg=0.0,
        max_elevation_deg=90.0,
    )
    time = np.linspace(0.0, 1800.0, 7, dtype=np.float64)
    obs0 = np.tile(np.array([[8_000_000.0, 0.0, 0.0]], dtype=np.float64), (time.size, 1))
    obs1 = np.tile(np.array([[0.0, 8_000_000.0, 0.0]], dtype=np.float64), (time.size, 1))
    store = compute_access_intervals(
        config=config,
        time=time,
        observer_positions=[obs0, obs1],
        interpolation="linear",
    )
    return store


def test_interval_store_surface_grid_metadata_available() -> None:
    store = _build_store()

    assert store.has_surface_target_grid()

    lon_deg, lat_deg, row_offsets, lat_rows_deg = store.require_surface_target_grid()
    idx = store.nearest_target_index(lat_deg=0.0, lon_deg=0.0)

    assert lon_deg.shape == (store.n_targets,)
    assert lat_deg.shape == (store.n_targets,)
    assert row_offsets[0] == 0
    assert row_offsets[-1] == store.n_targets
    assert lat_rows_deg.ndim == 1
    assert 0 <= idx < store.n_targets


def test_plot_interval_metric_wrappers_return_geoaxes() -> None:
    store = _build_store()

    fig_a, ax_a, mesh_a, cbar_a = plot_interval_access_duration(
        store,
        N=1,
        normalize_to_day=True,
        map_cfg=LIGHT_DETAILED,
        title="Access Duration Demo",
    )
    assert isinstance(fig_a, Figure)
    assert isinstance(ax_a, cartopy_geoaxes.GeoAxes)
    assert ax_a.get_title() == "Access Duration Demo"
    assert mesh_a.get_array().size > 0
    assert cbar_a is not None
    assert cbar_a.ax.get_ylabel() == "hours/day"
    plt.close(fig_a)

    fig_b, ax_b, mesh_b, cbar_b = plot_interval_max_asset(
        store,
        map_cfg=LIGHT_DETAILED,
        title="Max Asset Demo",
    )
    assert isinstance(fig_b, Figure)
    assert isinstance(ax_b, cartopy_geoaxes.GeoAxes)
    assert ax_b.get_title() == "Max Asset Demo"
    assert mesh_b.get_array().size > 0
    assert cbar_b is not None
    assert cbar_b.ax.get_ylabel() == "count"
    plt.close(fig_b)


def test_plot_interval_target_timeline_supports_lat_lon_selection() -> None:
    store = _build_store()

    fig, axes = plot_interval_target_timeline(
        store,
        lat_deg=0.0,
        lon_deg=0.0,
        target_name="Equator",
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert "Equator" in axes[0].get_title()
    assert axes[1].get_xlabel() == "Time [hours]"

    plt.close(fig)
