from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.time import Time

# Allow running this example directly:
#   python examples/walker_constellation_examples.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nstk.propagation import Orbit, build_walker_constellation


np.set_printoptions(precision=6, suppress=True)


def _seed_two_body() -> Orbit:
    return Orbit.from_kepler_two_body(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )


def _summarize(label: str, sats: list[Orbit]) -> None:
    print(f"\n=== {label} ===")
    print(f"satellite count: {len(sats)}")
    if not sats:
        return

    prop_name = str(sats[0].propagator.__class__.__name__)
    print(f"propagator: {prop_name}")

    sample_indices = [0, min(1, len(sats) - 1), len(sats) - 1]
    seen: set[int] = set()
    for idx in sample_indices:
        if idx in seen:
            continue
        seen.add(idx)
        r_m, v_mps = sats[idx].get_pv(0.0, frame="native", as_quantity=False)
        print(
            f"sat[{idx}] | |r|={np.linalg.norm(r_m):.1f} m, |v|={np.linalg.norm(v_mps):.3f} m/s"
        )


def example_two_body_walker() -> None:
    seed = _seed_two_body()

    delta = build_walker_constellation(
        seed,
        total_satellites=24,
        num_planes=6,
        phasing=1,
        pattern="delta",
        include_seed=True,
    )
    _summarize("Two-Body Walker Delta (T=24, P=6, F=1)", delta)

    star = build_walker_constellation(
        seed,
        total_satellites=24,
        num_planes=6,
        phasing=2,
        pattern="star",
        include_seed=False,
    )
    _summarize("Two-Body Walker Star (T=24, P=6, F=2, include_seed=False)", star)


def example_numerical_walker() -> None:
    seed = Orbit.from_kepler_numerical(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a=7100e3,
        e=0.002,
        i=np.deg2rad(55.0),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(25.0),
        anomaly=np.deg2rad(5.0),
        anomaly_type="mean",
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )

    walker = build_walker_constellation(
        seed,
        total_satellites=12,
        num_planes=3,
        phasing=1,
        pattern="delta",
        include_seed=True,
    )
    _summarize("Numerical Walker Delta (T=12, P=3, F=1)", walker)


def main() -> None:
    print("Nebula Space Toolkit Walker constellation builder examples")
    print("Creates constellations from a seed orbit using T/P/F Walker geometry.")

    example_two_body_walker()
    example_numerical_walker()


if __name__ == "__main__":
    main()
