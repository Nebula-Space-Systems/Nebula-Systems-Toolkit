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
    CoverageConfig,
    build_access_interval_store_from_config,
    access_duration_by_target,
)
from nebula.propagation import FastOrbit
from nebula.plotting import make_basemap, LIGHT_DETAILED


def _build_demo_constellation(epoch: Time) -> list[FastOrbit]:
    """
    Create a small mixed constellation for exact-interval coverage demonstration.
    """
    sats: list[FastOrbit] = []

    planes = [
        # a_m, e, inc_deg, raan_deg, count
        (6878e3, 0.001, 53.0, 0.0, 3),
        (6878e3, 0.001, 53.0, 120.0, 3),
        (20200e3, 0.01, 55.0, 60.0, 2),
    ]

    for a_m, e, inc_deg, raan_deg, count in planes:
        for idx in range(count):
            ma_deg = idx * (360.0 / float(count))
            sats.append(
                FastOrbit.from_kepler(
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


def main() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    duration_s = 24.0 * 3600.0 * 7.0
    step_s = 60.0
    n_steps = int(duration_s / step_s) + 1
    times = epoch + np.arange(n_steps, dtype=np.float64) * step_s * u.s
    t_seconds = (times - times[0]).to_value(u.s).astype(np.float64)

    sats = _build_demo_constellation(epoch)
    obs_positions = [sat.pos_itrf(times) for sat in sats]

    config = CoverageConfig(
        nlats=181,
        nlons=361,
        min_elevation_deg=10.0,
        max_elevation_deg=90.0,
    )

    import time

    t0 = time.time()
    store = build_access_interval_store_from_config(
        config=config,
        time=t_seconds,
        observer_positions=obs_positions,
        interpolation="cubic",
        root_tolerance_s=1e-3,
        root_bracket_substeps=32,
    )
    t1 = time.time()
    print(f"Built access interval store in {t1 - t0:.2f} seconds")

    duration_grid_s = access_duration_by_target(store, N=1, reshape=True)
    duration_grid_hr = duration_grid_s / 3600.0

    fig, ax, _, _ = make_basemap(LIGHT_DETAILED)
    data_crs = ccrs.PlateCarree()
    mesh = ax.pcolormesh(
        config.lon_edges_deg,
        config.lat_edges_deg,
        duration_grid_hr,
        transform=data_crs,
        shading="auto",
        cmap="viridis",
        alpha=0.65,
        rasterized=True,
        zorder=2,
    )
    ax.set_title(
        f"Exact-Interval Coverage Duration (N>=1)\n"
        f"{len(sats)} satellites, {duration_s / 3600.0:.0f} h window"
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Total access duration [hours]")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
