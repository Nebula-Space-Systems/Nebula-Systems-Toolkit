from __future__ import annotations

"""Comprehensive usage examples for ``nebula.propagation.orbit``.

Run directly:
    python examples/orbit_usage_examples.py

This script demonstrates common workflows with the Java-backed ``Orbit`` API:
1. Two-body quickstart and cartesian state queries.
2. Accepted time inputs (astropy Time, seconds arrays, Quantity).
3. Frame queries and geodetic outputs.
4. Ephemeris precompute/coverage workflow.
5. High-fidelity numerical propagation setup.
6. Attitude-law overrides (default, LOF, mapping, callable, provider object).
7. ``get_state`` convenience interface.
8. Constructing from an existing ``SpacecraftState``.
"""

import sys
from pathlib import Path
from time import perf_counter

import astropy.units as u
import numpy as np
from astropy.time import Time

# Allow running this example directly:
#   python examples/orbit_usage_examples.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nebula import initialize_orekit
from nebula.propagation import Orbit
from nebula.time_utils import astropy_time_to_orekit_date


np.set_printoptions(precision=6, suppress=True)


def _epoch() -> Time:
    return Time("2026-01-01T00:00:00", scale="utc")


def _build_two_body(epoch: Time) -> Orbit:
    return Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        anomaly_type="mean",
    )


def example_1_two_body_quickstart() -> Orbit:
    """Create an analytical orbit and query p/v/a at absolute times."""

    print("\n=== Example 1: Two-Body Quickstart ===")
    epoch = _epoch()
    orbit = _build_two_body(epoch)

    times = Time(
        epoch.unix + np.array([0.0, 60.0, 120.0, 180.0], dtype=np.float64),
        format="unix",
        scale="utc",
    )
    p, v, a = orbit.get_pva(times, frame="gcrf")

    print("epoch:", orbit.epoch.isot)
    print("native frame:", orbit.get_native_frame().getName())
    print("position shape:", p.shape, "velocity shape:", v.shape, "accel shape:", a.shape)
    print("first |r| [km]:", float(np.linalg.norm(p[0].to_value(u.m)) / 1e3))
    return orbit


def example_2_time_input_forms(orbit: Orbit) -> None:
    """Show equivalent time query styles accepted by Orbit methods."""

    print("\n=== Example 2: Accepted Time Input Forms ===")
    epoch = orbit.epoch

    t_astropy = Time(epoch.unix + np.array([0.0, 30.0, 60.0]), format="unix", scale="utc")
    t_seconds = np.array([0.0, 30.0, 60.0], dtype=np.float64)
    t_quantity = np.array([0.0, 30.0, 60.0], dtype=np.float64) * u.s
    t0 = astropy_time_to_orekit_date(epoch)
    t_absolutedate = [t0.shiftedBy(0.0), t0.shiftedBy(30.0), t0.shiftedBy(60.0)]

    r_astropy = orbit.get_p_np(t_astropy, frame="gcrf")
    r_seconds = orbit.get_p_np(t_seconds, frame="gcrf")
    r_quantity = orbit.get_p_np(t_quantity, frame="gcrf")
    r_absdate = orbit.get_p_np(t_absolutedate, frame="gcrf")

    print("allclose(astropy, seconds):", bool(np.allclose(r_astropy, r_seconds)))
    print("allclose(seconds, quantity):", bool(np.allclose(r_seconds, r_quantity)))
    print("allclose(seconds, AbsoluteDate):", bool(np.allclose(r_seconds, r_absdate)))
    print("scalar query shape:", orbit.get_p_np(0.0, frame="gcrf").shape)


def example_3_frames_and_geodetic(orbit: Orbit) -> None:
    """Query the same times in multiple frames and geodetic coordinates."""

    print("\n=== Example 3: Frames and Geodetic Queries ===")
    dt_s = np.arange(0.0, 301.0, 60.0, dtype=np.float64)

    r_gcrf = orbit.get_p_np(dt_s, frame="gcrf")
    r_itrf = orbit.get_p_np(dt_s, frame="itrf")
    lat, lon, alt = orbit.get_geodetic(dt_s)

    print("GCRF shape:", r_gcrf.shape, "ITRF shape:", r_itrf.shape)
    print(
        "first geodetic sample [deg, deg, m]:",
        float(lat[0].to_value(u.deg)),
        float(lon[0].to_value(u.deg)),
        float(alt[0].to_value(u.m)),
    )


def example_4_precompute_and_coverage(orbit: Orbit) -> None:
    """Demonstrate ephemeris precompute and cached coverage window."""

    print("\n=== Example 4: Ephemeris Coverage and Precompute ===")
    print("coverage before:", orbit.coverage())
    orbit.precompute(0.0, 4.0 * 3600.0)
    print("coverage after precompute:", orbit.coverage())

    dt_s = np.arange(0.0, 4.0 * 3600.0, 10.0, dtype=np.float64)
    t0 = perf_counter()
    _ = orbit.get_pv_np(dt_s, frame="gcrf")
    t1 = perf_counter()
    _ = orbit.get_pv_np(dt_s, frame="gcrf")
    t2 = perf_counter()

    print(f"first query  ({dt_s.size} samples): {t1 - t0:.3f} s")
    print(f"second query ({dt_s.size} samples): {t2 - t1:.3f} s")


def example_5_numerical_propagator() -> Orbit:
    """Build a numerical orbit with a common perturbation configuration."""

    print("\n=== Example 5: Numerical Orbit with Perturbations ===")
    epoch = _epoch()

    orbit = Orbit.from_kepler_numerical(
        epoch=epoch,
        a=7050e3,
        e=0.002,
        i=np.deg2rad(97.4),
        raan=np.deg2rad(5.0),
        argp=np.deg2rad(45.0),
        anomaly=np.deg2rad(0.0),
        anomaly_type="mean",
        gravity_degree=12,
        gravity_order=12,
        enable_third_body=True,
        third_bodies=("sun", "moon"),
        enable_srp=True,
        srp_area_m2=1.0,
        srp_cr=1.2,
        enable_drag=False,
    )

    dt_s = np.arange(0.0, 2.0 * 3600.0, 30.0, dtype=np.float64)
    r_num, v_num = orbit.get_pv_np(dt_s, frame="gcrf")
    print("numerical propagator:", orbit.propagator.__class__.__name__)
    print("queried shapes:", r_num.shape, v_num.shape)
    return orbit


def _attitude_callable(inertial_frame, iers, simple_eop):
    """Example callable attitude factory accepted by ``set_attitude_law``."""

    del iers, simple_eop
    from org.orekit.attitudes import LofOffset  # type: ignore
    from org.orekit.frames import LOFType  # type: ignore

    return LofOffset(inertial_frame, LOFType.QSW)


def example_6_attitude_law_overrides(orbit: Orbit) -> None:
    """Show different supported attitude-law input styles."""

    print("\n=== Example 6: Attitude Law Overrides ===")
    t = np.array([0.0, 60.0, 120.0], dtype=np.float64)

    q_default = orbit.get_attitude_np(t)
    orbit.set_attitude_law("tnw")
    q_tnw = orbit.get_attitude_np(t)
    orbit.set_attitude_law({"type": "nadir"})
    q_nadir = orbit.get_attitude_np(t)
    orbit.set_attitude_law(_attitude_callable)
    q_callable = orbit.get_attitude_np(t)

    # Provider object form (requires JVM initialized; it is by now).
    from org.orekit.attitudes import LofOffset  # type: ignore
    from org.orekit.frames import LOFType  # type: ignore

    orbit.set_attitude_law(LofOffset(orbit.get_native_frame(), LOFType.LVLH_CCSDS))
    q_provider = orbit.get_attitude_np(t)

    print("default vs tnw differs:", bool(not np.allclose(q_default, q_tnw)))
    print("tnw vs nadir differs:", bool(not np.allclose(q_tnw, q_nadir)))
    print("nadir vs callable differs:", bool(not np.allclose(q_nadir, q_callable)))
    print("provider quaternion[0]:", q_provider[0])


def example_7_get_state_convenience(orbit: Orbit) -> None:
    """Use ``get_state`` to request grouped fields in one call."""

    print("\n=== Example 7: get_state Convenience API ===")
    dt_s = np.array([0.0, 45.0, 90.0], dtype=np.float64)

    state_np = orbit.get_state(dt_s, frame="gcrf", fields="pv", as_quantity=False)
    state_q = orbit.get_state(120.0, frame="gcrf", fields="a", as_quantity=True)

    print("state_np keys:", list(state_np.keys()))
    print("state_np['p'] shape:", state_np["p"].shape, "state_np['v'] shape:", state_np["v"].shape)
    print("state_q keys:", list(state_q.keys()), "type(a):", type(state_q["a"]).__name__)


def example_8_from_spacecraft_state(seed_orbit: Orbit) -> None:
    """Construct a new ``Orbit`` from an existing Orekit ``SpacecraftState``."""

    print("\n=== Example 8: Construct from SpacecraftState ===")
    state0 = seed_orbit.propagator.getInitialState()
    orb2 = Orbit.from_spacecraft_state(state0, attitude="default")

    p0 = orb2.get_p_np(0.0, frame="native")
    print("new orbit native frame:", orb2.get_native_frame().getName())
    print("initial |r| [km]:", float(np.linalg.norm(p0) / 1e3))


def main() -> None:
    """Run all Orbit API examples."""

    initialize_orekit()

    print("Nebula Orbit API examples")
    print("All heavy propagation/frame work is executed in Java through Orekit.")

    two_body = example_1_two_body_quickstart()
    example_2_time_input_forms(two_body)
    example_3_frames_and_geodetic(two_body)
    example_4_precompute_and_coverage(two_body)

    numerical = example_5_numerical_propagator()
    example_6_attitude_law_overrides(numerical)
    example_7_get_state_convenience(two_body)
    example_8_from_spacecraft_state(two_body)


if __name__ == "__main__":
    main()
