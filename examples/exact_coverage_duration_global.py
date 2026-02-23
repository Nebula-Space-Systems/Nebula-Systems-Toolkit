from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import cartopy.crs as ccrs

# Allow running this example directly:
#   python examples/exact_coverage_duration_global.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nebula.coverage import (
    ExactCoverageConfig,
    compute_access_intervals,
    access_duration_by_target,
)
from nebula.propagation import Orbit
from nebula.plotting import make_basemap, LIGHT_DETAILED


def _build_demo_constellation(epoch: Time) -> list[Orbit]:
    """
    Create a small mixed constellation for exact-interval coverage demonstration.
    """
    sats: list[Orbit] = []

    planes = [
        # a_m, e, inc_deg, raan_deg, count
        (6878e3, 0.001, 53.0, 0.0, 3),
        (6878e3, 0.001, 53.0, 120.0, 3),
        (51000e3, 0.6, 45.0, 60.0, 2),
    ]

    for a_m, e, inc_deg, raan_deg, count in planes:
        for idx in range(count):
            ma_deg = idx * (360.0 / float(count))
            sats.append(
                Orbit.from_kepler_fast(
                    epoch=epoch,
                    a_m=a_m,
                    e=e,
                    i=np.deg2rad(inc_deg),
                    raan=np.deg2rad(raan_deg),
                    argp=0.0,
                    anomaly=np.deg2rad(ma_deg),
                    anomaly_type="mean",
                    dt_save_s=30.0,
                    enable_j2=False,
                    j2_mode="secular",
                )
            )

    return sats


def _latitude_edges_from_rows(lat_rows_deg: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat_rows_deg, dtype=np.float64)
    n = lat.size
    edges = np.empty(n + 1, dtype=np.float64)
    if n == 1:
        edges[0] = lat[0] - 0.5
        edges[1] = lat[0] + 0.5
    else:
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
        edges[0] = lat[0] - 0.5 * (lat[1] - lat[0])
        edges[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    return np.clip(edges, -90.0, 90.0)


def _rasterize_lat_row_field_to_regular_grid(
    config: ExactCoverageConfig,
    values_flat: np.ndarray,
    *,
    nlon_render: int = 720,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert latitude-row values on a nonuniform longitude grid to a regular
    lon/lat raster for cohesive heatmap rendering.
    """
    vals = np.asarray(values_flat, dtype=np.float64)
    if vals.shape != (config.n_targets,):
        raise ValueError("values_flat must have shape (config.n_targets,)")

    nlon = int(max(90, nlon_render))
    lon_edges = np.linspace(-180.0, 180.0, nlon + 1, dtype=np.float64)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lon_centers_360 = np.mod(lon_centers + 360.0, 360.0)

    grid = np.empty((int(config.nlats), nlon), dtype=np.float64)
    for row_idx in range(int(config.nlats)):
        i0 = int(config.row_offsets[row_idx])
        i1 = int(config.row_offsets[row_idx + 1])

        row_vals = vals[i0:i1]
        if row_vals.size == 1:
            grid[row_idx, :] = row_vals[0]
            continue

        lon_row = config.lon_deg_flat[i0:i1]
        lon_row_360 = np.mod(lon_row + 360.0, 360.0)
        order = np.argsort(lon_row_360)
        xp = lon_row_360[order]
        fp = row_vals[order]

        # Drop duplicate longitudes (can occur when endpoints wrap together).
        keep = np.empty(xp.size, dtype=bool)
        keep[0] = True
        keep[1:] = np.diff(xp) > 1e-12
        xp = xp[keep]
        fp = fp[keep]

        if xp.size == 1:
            grid[row_idx, :] = fp[0]
        else:
            grid[row_idx, :] = np.interp(lon_centers_360, xp, fp, period=360.0)

    lat_edges = _latitude_edges_from_rows(config.lat_deg_rows)
    return lon_edges, lat_edges, grid


def main() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    duration_s = 24.0 * 3600.0 * 10
    step_s = 60.0
    n_steps = int(duration_s / step_s) + 1
    times = epoch + np.arange(n_steps, dtype=np.float64) * step_s * u.s
    t_seconds = (times - times[0]).to_value(u.s).astype(np.float64)

    sats = _build_demo_constellation(epoch)
    obs_positions = [sat.pos_itrf(times) for sat in sats]

    config = ExactCoverageConfig(
        nlats=181,
        nlons_equator=361,
        scale_longitude_by_latitude=False,
        min_lon_points_per_row=1,
        min_elevation_deg=0.0,
        max_elevation_deg=50.0,
        include_lat_endpoints=True,
        include_lon_endpoints=False,
    )

    import time

    t0 = time.time()
    store = compute_access_intervals(
        config=config,
        time=t_seconds,
        observer_positions=obs_positions,
        interpolation="linear",
        root_tolerance_s=1e-3,
    )
    t1 = time.time()
    print(f"Built access interval store in {t1 - t0:.2f} seconds")

    duration_s_per_target = access_duration_by_target(store, N=2, normalize_to_day=True)
    duration_hr_per_target = duration_s_per_target / 3600.0

    lon_edges, lat_edges, duration_hr_grid = _rasterize_lat_row_field_to_regular_grid(
        config,
        duration_hr_per_target,
        nlon_render=max(720, int(config.nlons_equator) * 2),
    )

    fig, ax, _, _ = make_basemap(LIGHT_DETAILED)
    data_crs = ccrs.PlateCarree()
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        duration_hr_grid,
        transform=data_crs,
        cmap="viridis",
        shading="auto",
        rasterized=True,
        zorder=4,
        alpha=0.7,
    )
    ax.set_title(
        f"Exact-Interval Coverage Duration (N>=1)\n"
        f"{len(sats)} satellites, {duration_s / 3600.0:.0f} h window"
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Average access duration [hours/day]")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
