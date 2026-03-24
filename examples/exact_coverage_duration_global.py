from __future__ import annotations

import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

# Allow running this example directly:
#   python examples/exact_coverage_duration_global.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nstk.coverage import (
    CoverageTargets,
    CoverageTimeline,
    IntervalCoverage,
    LatitudeAdaptiveSampler,
)
from nstk.plotting import LIGHT_DETAILED
from nstk.propagation import Orbit, build_walker_constellation


def _build_demo_constellation(epoch: Time) -> list[Orbit]:
    seed = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=6_878_000.0,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
        anomaly_type="mean",
    )
    return build_walker_constellation(
        seed,
        total_satellites=10,
        num_planes=5,
        phasing=1,
    )


def main() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    timeline = CoverageTimeline.absolute(
        epoch + np.arange(0.0, 12.0 * 3600.0 + 300.0, 300.0) * u.s,
        label="12 hour global coverage demo",
    )

    sats = _build_demo_constellation(epoch)
    targets = CoverageTargets.global_earth(
        sampler=LatitudeAdaptiveSampler(nlats=61, nlons_equator=121)
    )

    coverage = IntervalCoverage.from_orbits(
        orbits=sats,
        timeline=timeline,
        targets=targets,
    )

    access = coverage.access_duration(
        min_assets=1,
        normalize="day",
        unit="hours",
    )
    two_asset = coverage.access_duration(
        min_assets=2,
        normalize="day",
        unit="hours",
    )
    print(access.summary())
    print(two_asset.summary())

    fig, _, _, _ = access.plot(
        map_cfg=LIGHT_DETAILED,
        title="Average Access Duration [hours/day]",
    )
    fig.tight_layout()
    plt.show()

    fig, _, _, _ = two_asset.plot(
        map_cfg=LIGHT_DETAILED,
        title="Average 2-Asset Access Duration [hours/day]",
        cmap="magma",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
