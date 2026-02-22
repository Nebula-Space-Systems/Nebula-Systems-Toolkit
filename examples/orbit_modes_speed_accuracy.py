from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy.time import Time

# Allow running this example directly:
#   python examples/orbit_modes_speed_accuracy.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nebula.propagation import Orbit


def _print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _pv_error_stats(
    r_ref: np.ndarray, v_ref: np.ndarray, r_cmp: np.ndarray, v_cmp: np.ndarray
) -> dict[str, float]:
    dr = np.linalg.norm(r_cmp - r_ref, axis=1)
    dv = np.linalg.norm(v_cmp - v_ref, axis=1)
    return {
        "pos_max_m": float(np.max(dr)),
        "pos_mean_m": float(np.mean(dr)),
        "pos_p95_m": float(np.percentile(dr, 95.0)),
        "vel_max_mps": float(np.max(dv)),
        "vel_mean_mps": float(np.mean(dv)),
        "vel_p95_mps": float(np.percentile(dv, 95.0)),
    }


def _time_pv_query(
    orbit: Orbit, query_s: np.ndarray, frame: str, repeat: int = 3
) -> float:
    # Warmup (ensures caches/JIT are ready).
    _ = orbit.pv(query_s[: min(8, query_s.size)], frame=frame)

    times = np.empty(repeat, dtype=np.float64)
    for i in range(repeat):
        t0 = perf_counter()
        _ = orbit.pv(query_s, frame=frame)
        times[i] = perf_counter() - t0
    return float(np.mean(times))


def _compare_constructors(epoch: Time) -> None:
    _print_section("Constructor Coverage")

    base = dict(
        epoch=epoch,
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        dt_save_s=30.0,
    )

    precise = Orbit.from_kepler_precise(**base, gravity_model="newtonian")
    fast = Orbit.from_kepler_fast(**base, enable_j2=False)

    print(f"from_kepler_precise mode:  {precise.mode}")
    print(f"from_kepler_fast mode:       {fast.mode}")

    r0, v0 = precise.pv(0.0, frame="native")
    rf0, vf0 = fast.pv(0.0, frame="native")
    print(
        "precision vs fast at epoch: "
        f"dr={np.linalg.norm(rf0-r0):.3e} m, dv={np.linalg.norm(vf0-v0):.3e} m/s"
    )

    # from_pv consistency
    pv_a = Orbit.from_pv(
        r0,
        v0,
        epoch,
        frame="gcrf",
        propagate_inertial_frame="gcrf",
        gravity_model="newtonian",
        dt_save_s=30.0,
    )
    rpa, vpa = pv_a.pv(0.0, frame="native")
    print(
        "from_pv epoch consistency: "
        f"dr={np.linalg.norm(rpa-r0):.3e} m, dv={np.linalg.norm(vpa-v0):.3e} m/s"
    )

    # from_spacecraft_state consistency
    state0 = precise.propagator.getInitialState()
    st_a = Orbit.from_spacecraft_state(
        state0,
        gravity_model="newtonian",
        dt_save_s=30.0,
    )
    rsa, vsa = st_a.pv(0.0, frame="native")
    print(
        "from_spacecraft_state epoch consistency: "
        f"dr={np.linalg.norm(rsa-r0):.3e} m, dv={np.linalg.norm(vsa-v0):.3e} m/s"
    )


def _compare_speed_and_accuracy(
    *,
    title: str,
    precise: Orbit,
    fast: Orbit,
    query_s: np.ndarray,
) -> None:
    _print_section(title)

    t_min = float(query_s.min())
    t_max = float(query_s.max())
    precise.precompute(t_min, t_max)
    fast.precompute(t_min, t_max)

    t_prec_native = _time_pv_query(precise, query_s, frame="native", repeat=3)
    t_fast_native = _time_pv_query(fast, query_s, frame="native", repeat=3)
    t_prec_itrf = _time_pv_query(precise, query_s, frame="itrf", repeat=3)
    t_fast_itrf = _time_pv_query(fast, query_s, frame="itrf", repeat=3)

    print(f"Native query mean time (precision):  {t_prec_native:.6f} s")
    print(f"Native query mean time (fast):       {t_fast_native:.6f} s")
    print(f"Native speedup (precision / fast):   {t_prec_native / t_fast_native:.2f}x")
    print(f"ITRF query mean time (precision):    {t_prec_itrf:.6f} s")
    print(f"ITRF query mean time (fast):         {t_fast_itrf:.6f} s")
    print(f"ITRF speedup (precision / fast):     {t_prec_itrf / t_fast_itrf:.2f}x")

    r_pn, v_pn = precise.pv(query_s, frame="native")
    r_fn, v_fn = fast.pv(query_s, frame="native")
    n_stats = _pv_error_stats(r_pn, v_pn, r_fn, v_fn)

    r_pi, v_pi = precise.pv(query_s, frame="itrf")
    r_fi, v_fi = fast.pv(query_s, frame="itrf")
    i_stats = _pv_error_stats(r_pi, v_pi, r_fi, v_fi)

    print("Native error stats (fast vs precision):")
    print(
        f"  pos max/mean/p95 [m]:   {n_stats['pos_max_m']:.6f} / "
        f"{n_stats['pos_mean_m']:.6f} / {n_stats['pos_p95_m']:.6f}"
    )
    print(
        f"  vel max/mean/p95 [m/s]: {n_stats['vel_max_mps']:.6f} / "
        f"{n_stats['vel_mean_mps']:.6f} / {n_stats['vel_p95_mps']:.6f}"
    )
    print("ITRF error stats (fast vs precision):")
    print(
        f"  pos max/mean/p95 [m]:   {i_stats['pos_max_m']:.6f} / "
        f"{i_stats['pos_mean_m']:.6f} / {i_stats['pos_p95_m']:.6f}"
    )
    print(
        f"  vel max/mean/p95 [m/s]: {i_stats['vel_max_mps']:.6f} / "
        f"{i_stats['vel_mean_mps']:.6f} / {i_stats['vel_p95_mps']:.6f}"
    )


def main() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    _compare_constructors(epoch)

    duration_s = 24.0 * 3600.0 * 7
    query_step_s = 5.0
    query_s = np.arange(0.0, duration_s + 1e-9, query_step_s, dtype=np.float64)

    base = dict(
        epoch=epoch,
        a_m=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        dt_save_s=20.0,
    )

    # Two-body comparison.
    precise_two_body = Orbit.from_kepler_precise(**base, gravity_model="newtonian")
    fast_two_body = Orbit.from_kepler_fast(**base, enable_j2=False)
    _compare_speed_and_accuracy(
        title="Two-Body: Precision vs Fast",
        precise=precise_two_body,
        fast=fast_two_body,
        query_s=query_s,
    )

    # J2 comparison (Orekit degree-2/order-0 vs fast osculating J2).
    precise_j2 = Orbit.from_kepler_precise(
        **base,
        gravity_model="harmonic",
        gravity_degree=2,
        gravity_order=0,
    )
    fast_j2 = Orbit.from_kepler_fast(
        **base,
        enable_j2=True,
        j2_mode="osculating",
        j2_substeps=4,
    )
    _compare_speed_and_accuracy(
        title="J2: Precision vs Fast",
        precise=precise_j2,
        fast=fast_j2,
        query_s=query_s,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
