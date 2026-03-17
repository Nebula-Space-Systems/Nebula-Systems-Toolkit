from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time

# Allow running this example directly:
#   python examples/transforms_usage_examples.py
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from nebula.transforms import (
    aer2geodetic,
    coarse_eci2geodetic_vec_deg,
    coarse_eci2ecef_pos_vec,
    ecef2aer,
    ecef2enu,
    ecef2geodetic,
    ecef2geodetic_deg,
    ecef2geodetic_vec_ecef_deg,
    enu2ecef,
    enu2geodetic,
    geodetic2aer,
    geodetic2ecef,
    geodetic2ecef_vec_llh,
    geodetic2enu,
    transform,
)


np.set_printoptions(precision=6, suppress=True)


def example_geodetic_ecef_roundtrip() -> None:
    print("\n=== Example 1: Geodetic <-> ECEF (scalar and vector) ===")

    lat = np.deg2rad(37.7749)
    lon = np.deg2rad(-122.4194)
    h_m = 25.0

    x_m, y_m, z_m = geodetic2ecef(lat, lon, h_m)
    lat2, lon2, h2_m = ecef2geodetic(x_m, y_m, z_m)
    lat2_deg, lon2_deg, _ = ecef2geodetic_deg(x_m, y_m, z_m)

    print(
        f"Input geodetic   [deg,m]: lat={np.rad2deg(lat):.6f}, lon={np.rad2deg(lon):.6f}, h={h_m:.3f}"
    )
    print(f"ECEF position    [m]:     x={x_m:.3f}, y={y_m:.3f}, z={z_m:.3f}")
    print(
        f"Round-trip       [deg,m]: lat={lat2_deg:.6f}, lon={lon2_deg:.6f}, h={h2_m:.3f}"
    )

    lat_vec = np.deg2rad(np.array([0.0, 15.0, 45.0, -30.0]))
    lon_vec = np.deg2rad(np.array([0.0, 40.0, -120.0, 170.0]))
    h_vec = np.array([0.0, 100.0, 550000.0, 10.0])

    x_vec, y_vec, z_vec = geodetic2ecef_vec_llh(lat_vec, lon_vec, h_vec)
    r_ecef = np.column_stack((x_vec, y_vec, z_vec))
    lat_deg_vec, lon_deg_vec, h_back = ecef2geodetic_vec_ecef_deg(r_ecef)

    print("Vectorized geodetic->ECEF->geodetic (deg):")
    print(np.column_stack((lat_deg_vec, lon_deg_vec, h_back)))


def example_enu_and_aer_workflows() -> None:
    print("\n=== Example 2: ENU and AER workflows ===")

    obs_lat = np.deg2rad(34.0)
    obs_lon = np.deg2rad(-118.0)
    obs_h = 250.0

    tgt_lat = np.deg2rad(34.05)
    tgt_lon = np.deg2rad(-117.85)
    tgt_h = 700.0

    e_m, n_m, u_m = geodetic2enu(tgt_lat, tgt_lon, tgt_h, obs_lat, obs_lon, obs_h)
    x_tgt, y_tgt, z_tgt = enu2ecef(e_m, n_m, u_m, obs_lat, obs_lon, obs_h)
    e2_m, n2_m, u2_m = ecef2enu(x_tgt, y_tgt, z_tgt, obs_lat, obs_lon, obs_h)

    az_rad, el_rad, slant_m = geodetic2aer(
        tgt_lat, tgt_lon, tgt_h, obs_lat, obs_lon, obs_h
    )
    lat3, lon3, h3 = aer2geodetic(az_rad, el_rad, slant_m, obs_lat, obs_lon, obs_h)

    az2, el2, slant2 = ecef2aer(x_tgt, y_tgt, z_tgt, obs_lat, obs_lon, obs_h)
    lat_from_enu, lon_from_enu, h_from_enu = enu2geodetic(
        e_m, n_m, u_m, obs_lat, obs_lon, obs_h
    )

    print(f"ENU offset [m]: E={e_m:.3f}, N={n_m:.3f}, U={u_m:.3f}")
    print(f"ECEF->ENU backcheck [m]: E={e2_m:.3f}, N={n2_m:.3f}, U={u2_m:.3f}")
    print(
        f"AER from geodetic [deg,m]: az={np.rad2deg(az_rad):.3f}, el={np.rad2deg(el_rad):.3f}, srange={slant_m:.3f}"
    )
    print(
        f"AER from ECEF     [deg,m]: az={np.rad2deg(az2):.3f}, el={np.rad2deg(el2):.3f}, srange={slant2:.3f}"
    )
    print(
        "Recovered target from AER [deg,m]: "
        f"lat={np.rad2deg(lat3):.6f}, lon={np.rad2deg(lon3):.6f}, h={h3:.3f}"
    )
    print(
        "Recovered target from ENU [deg,m]: "
        f"lat={np.rad2deg(lat_from_enu):.6f}, lon={np.rad2deg(lon_from_enu):.6f}, h={h_from_enu:.3f}"
    )


def example_coarse_eci_transforms() -> None:
    print("\n=== Example 3: Coarse ECI -> ITRF / geodetic ===")

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    n = 8
    dt_s = np.arange(n, dtype=np.float64) * 120.0

    # Simple synthetic circular trajectory in an ECI-like frame.
    radius_m = 6778e3
    omega = 2.0 * np.pi / (95.0 * 60.0)
    theta = omega * dt_s

    r_eci = np.column_stack(
        (
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.zeros_like(theta),
        )
    ).astype(np.float64)

    t_arr = epoch + dt_s * u.s
    jd_ut1 = t_arr.ut1.jd.astype(np.float64)
    jd_tt = t_arr.tt.jd.astype(np.float64)

    r_itrf = coarse_eci2ecef_pos_vec(r_eci, jd_ut1, jd_tt)
    lat_deg, lon_deg, h_m = coarse_eci2geodetic_vec_deg(r_eci, jd_ut1, jd_tt)

    print(f"ITRF positions shape: {r_itrf.shape}")
    print("First 3 coarse geodetic samples [lat_deg, lon_deg, h_m]:")
    print(np.column_stack((lat_deg, lon_deg, h_m))[:3])


def example_timed_frame_transforms() -> None:
    print("\n=== Example 4: Timed frame transforms (Orekit-backed) ===")

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    times = epoch + np.array([0.0, 30.0, 60.0], dtype=np.float64) * u.s

    # GCRF sample state history.
    r_gcrf = np.array(
        [
            [7000e3, 0.0, 0.0],
            [6999e3, 75e3, 20e3],
            [6996e3, 150e3, 40e3],
        ],
        dtype=np.float64,
    )
    v_gcrf = np.array(
        [
            [0.0, 7500.0, 1000.0],
            [-60.0, 7495.0, 1005.0],
            [-120.0, 7480.0, 1010.0],
        ],
        dtype=np.float64,
    )

    r_itrf, _, _ = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r_gcrf,
    )
    r_itrf2, v_itrf2, _ = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r_gcrf,
        velocity=v_gcrf,
    )

    print(f"Position-only transform output shape: {r_itrf.shape}")
    print(f"Pos+vel transform output shapes: r={r_itrf2.shape}, v={v_itrf2.shape}")
    print("First transformed position [m]:", r_itrf[0])


def main() -> None:
    print("Nebula transform usage examples")
    print(
        "Note: most transform APIs use radians for lat/lon/az/el and meters for distances."
    )

    example_geodetic_ecef_roundtrip()
    example_enu_and_aer_workflows()
    example_coarse_eci_transforms()
    example_timed_frame_transforms()


if __name__ == "__main__":
    main()
