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

from nebula.propagation import Orbit, build_walker_constellation


np.set_printoptions(precision=6, suppress=True)


def _fast_seed() -> Orbit:
    return Orbit.from_kepler_fast(
        epoch=Time("2026-01-01T00:00:00", scale="utc"),
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=3,
        dt_save_s=30.0,
    )


def _summarize(label: str, sats: list[Orbit]) -> None:
    print(f"\n=== {label} ===")
    print(f"satellite count: {len(sats)}")
    if not sats:
        return

    mode = "efficiency" if sats[0].is_efficiency else "precision"
    print(f"propagation mode: {mode}")

    sample_indices = [0, min(1, len(sats) - 1), len(sats) - 1]
    seen: set[int] = set()
    for idx in sample_indices:
        if idx in seen:
            continue
        seen.add(idx)
        r_m, v_mps = sats[idx].pv(sats[idx].epoch, frame="native")
        print(
            f"sat[{idx}] | |r|={np.linalg.norm(r_m):.1f} m, |v|={np.linalg.norm(v_mps):.3f} m/s"
        )


def example_fast_walker() -> None:
    seed = _fast_seed()

    # Walker Delta: T/P/F = 24/6/1, includes seed-equivalent slot.
    delta = build_walker_constellation(
        seed,
        total_satellites=24,
        num_planes=6,
        phasing=1,
        pattern="delta",
        include_seed=True,
    )
    _summarize("Fast Walker Delta (T=24, P=6, F=1)", delta)

    # Walker Star: T/P/F = 24/6/2, excluding seed-equivalent slot.
    star = build_walker_constellation(
        seed,
        total_satellites=24,
        num_planes=6,
        phasing=2,
        pattern="star",
        include_seed=False,
    )
    _summarize("Fast Walker Star (T=24, P=6, F=2, include_seed=False)", star)


def example_precision_walker_optional() -> None:
    ctor = Orbit.from_kepler_precise  # type: ignore[attr-defined]

    try:
        seed = ctor(
            epoch=Time("2026-01-01T00:00:00", scale="utc"),
            a_m=7100e3,
            e=0.002,
            i=np.deg2rad(55.0),
            raan=np.deg2rad(30.0),
            argp=np.deg2rad(25.0),
            anomaly=np.deg2rad(5.0),
            anomaly_type="mean",
            gravity_model="newtonian",
            dt_save_s=60.0,
        )

        walker = build_walker_constellation(
            seed,
            total_satellites=12,
            num_planes=3,
            phasing=1,
            pattern="delta",
            include_seed=True,
        )
        _summarize("Precision Walker Delta (T=12, P=3, F=1)", walker)
    except Exception as exc:
        print("\n=== Precision example skipped ===")
        print(f"reason: {type(exc).__name__}: {exc}")


def main() -> None:
    print("Nebula Walker constellation builder examples")
    print("Creates constellations from a seed orbit using T/P/F Walker geometry.")

    example_fast_walker()
    example_precision_walker_optional()


if __name__ == "__main__":
    main()
