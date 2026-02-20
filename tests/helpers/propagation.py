from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time

from nebula.propagation.fast_orbit import FastOrbit
from nebula.propagation.orbit import Orbit


SANE_INTEGRATOR_DEFAULTS = dict(
    position_tolerance_m=0.1,
    max_step_s=120.0,
    initial_step_s=20.0,
)


@dataclass
class PvErrorStats:
    pos_max_m: float
    pos_mean_m: float
    pos_p95_m: float
    vel_max_mps: float
    vel_mean_mps: float
    vel_p95_mps: float


def seconds_grid(epoch: Time, duration_s: float, n_samples: int) -> Time:
    """Build astropy times using explicit second offsets."""
    offsets_s = np.linspace(0.0, float(duration_s), int(n_samples), dtype=np.float64)
    return epoch + offsets_s * u.s


def pv_error_stats(
    r_ref: np.ndarray,
    v_ref: np.ndarray,
    r_cmp: np.ndarray,
    v_cmp: np.ndarray,
) -> PvErrorStats:
    pos_err_m = np.linalg.norm(r_cmp - r_ref, axis=1)
    vel_err_mps = np.linalg.norm(v_cmp - v_ref, axis=1)
    return PvErrorStats(
        pos_max_m=float(np.max(pos_err_m)),
        pos_mean_m=float(np.mean(pos_err_m)),
        pos_p95_m=float(np.percentile(pos_err_m, 95)),
        vel_max_mps=float(np.max(vel_err_mps)),
        vel_mean_mps=float(np.mean(vel_err_mps)),
        vel_p95_mps=float(np.percentile(vel_err_mps, 95)),
    )


def build_orekit_newtonian(
    *,
    epoch: Time,
    a_m: float,
    e: float,
    i_rad: float,
    raan_rad: float,
    argp_rad: float,
    anomaly_rad: float,
    dt_save_s: float,
    interpolation_mode: str = "cubic",
) -> Orbit:
    return Orbit.from_kepler(
        epoch=epoch,
        a_m=float(a_m),
        e=float(e),
        i=float(i_rad),
        raan=float(raan_rad),
        argp=float(argp_rad),
        anomaly=float(anomaly_rad),
        dt_save_s=float(dt_save_s),
        gravity_model="newtonian",
        itrf_query_mode="cached",
        interpolation_mode=interpolation_mode,  # type: ignore[arg-type]
        **SANE_INTEGRATOR_DEFAULTS,
    )


def build_orekit_j2_only(
    *,
    epoch: Time,
    a_m: float,
    e: float,
    i_rad: float,
    raan_rad: float,
    argp_rad: float,
    anomaly_rad: float,
    dt_save_s: float,
) -> Orbit:
    return Orbit.from_kepler(
        epoch=epoch,
        a_m=float(a_m),
        e=float(e),
        i=float(i_rad),
        raan=float(raan_rad),
        argp=float(argp_rad),
        anomaly=float(anomaly_rad),
        dt_save_s=float(dt_save_s),
        gravity_model="harmonic",
        gravity_degree=2,
        gravity_order=0,
        enable_drag=False,
        enable_third_body=False,
        enable_solid_tides=False,
        enable_ocean_tides=False,
        enable_relativity=False,
        enable_de_sitter=False,
        enable_lense_thirring=False,
        enable_srp=False,
        enable_erp=False,
        itrf_query_mode="cached",
        interpolation_mode="cubic",
        **SANE_INTEGRATOR_DEFAULTS,
    )


def build_fast(
    *,
    epoch: Time,
    a_m: float,
    e: float,
    i_rad: float,
    raan_rad: float,
    argp_rad: float,
    anomaly_rad: float,
    dt_save_s: float,
    enable_j2: bool,
    j2_mode: str = "secular",
    j2_substeps: int = 4,
) -> FastOrbit:
    kwargs: dict[str, Any] = dict(
        epoch=epoch,
        a_m=float(a_m),
        e=float(e),
        i=float(i_rad),
        raan=float(raan_rad),
        argp=float(argp_rad),
        anomaly=float(anomaly_rad),
        dt_save_s=float(dt_save_s),
        enable_j2=bool(enable_j2),
    )
    if enable_j2:
        kwargs["j2_mode"] = j2_mode
        kwargs["j2_substeps"] = int(j2_substeps)
    return FastOrbit.from_kepler(**kwargs)  # type: ignore[arg-type]
