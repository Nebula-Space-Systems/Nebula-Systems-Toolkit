from __future__ import annotations

import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

# Allow running this example directly:
#   python examples/timed_rotations_usage_examples.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nebula.time_utils import astropy_time_to_orekit_date
from nebula.transforms import transform


np.set_printoptions(precision=6, suppress=True)


def example_position_only_any_frames() -> None:
    print("\n=== Example 1: Position-only timed transform (string frames) ===")

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    times = Time(
        epoch.unix + np.linspace(0.0, 300.0, 6, dtype=np.float64),
        format="unix",
        scale="utc",
    )

    r_teme = np.array(
        [
            [7000e3, 0.0, 0.0],
            [6995e3, 80e3, 10e3],
            [6980e3, 160e3, 20e3],
            [6955e3, 240e3, 30e3],
            [6920e3, 320e3, 40e3],
            [6875e3, 400e3, 50e3],
        ],
        dtype=np.float64,
    )

    r_mod, _, _ = transform("teme", "mod", times, r_teme)
    print("input shape:", r_teme.shape, "output shape:", r_mod.shape)
    print("first transformed sample [m]:", r_mod[0])


def example_pos_vel_acc_with_quantities() -> None:
    print("\n=== Example 2: Position/velocity/acceleration with quantities ===")

    t_scalar = Time("2026-01-01T00:00:00", scale="utc")

    r_gcrf = np.array(
        [
            [7000e3, 0.0, 0.0],
            [6999e3, 75e3, 15e3],
            [6996e3, 150e3, 30e3],
        ],
        dtype=np.float64,
    ) * u.m
    v_gcrf = np.array(
        [
            [0.0, 7500.0, 1000.0],
            [-60.0, 7495.0, 1005.0],
            [-120.0, 7480.0, 1010.0],
        ],
        dtype=np.float64,
    ) * (u.m / u.s)
    a_gcrf = np.array(
        [
            [-8.2, 0.01, 0.0],
            [-8.2, -0.02, 0.01],
            [-8.2, -0.05, 0.01],
        ],
        dtype=np.float64,
    ) * (u.m / (u.s**2))

    # Scalar time is broadcast to each vector row.
    r_itrf, v_itrf, a_itrf = transform(
        "gcrf",
        "itrf",
        t_scalar,
        r_gcrf,
        velocity=v_gcrf,
        acceleration=a_gcrf,
    )
    print("position unit:", r_itrf.unit, "shape:", r_itrf.shape)
    print("velocity unit:", v_itrf.unit, "shape:", v_itrf.shape)
    print("accel unit:", a_itrf.unit, "shape:", a_itrf.shape)


def example_absolutedate_inputs() -> None:
    print("\n=== Example 3: Orekit AbsoluteDate inputs ===")

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    t0 = astropy_time_to_orekit_date(epoch)
    dates = [t0.shiftedBy(0.0), t0.shiftedBy(60.0), t0.shiftedBy(120.0)]

    r_eci = np.array(
        [
            [7000e3, 0.0, 0.0],
            [6999e3, 70e3, 15e3],
            [6996e3, 140e3, 30e3],
        ],
        dtype=np.float64,
    )

    r_itrf, _, _ = transform("gcrf", "itrf", dates, r_eci)
    print("AbsoluteDate transform shape:", r_itrf.shape)
    print("last transformed sample [m]:", r_itrf[-1])


def main() -> None:
    print("Timed rotations usage examples (Java-backed Orekit transforms)")
    print("Supports arbitrary frame pairs, broadcastable inputs, and AbsoluteDate.")

    example_position_only_any_frames()
    example_pos_vel_acc_with_quantities()
    example_absolutedate_inputs()


if __name__ == "__main__":
    main()
