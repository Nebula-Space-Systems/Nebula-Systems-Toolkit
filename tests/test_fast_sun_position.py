import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::erfa.core.ErfaWarning"),
    pytest.mark.filterwarnings(
        "ignore:Tried to get polar motions for times after IERS data is valid.*:astropy.utils.exceptions.AstropyWarning"
    ),
]

astropy_time = pytest.importorskip("astropy.time")
astropy_coordinates = pytest.importorskip("astropy.coordinates")
u = pytest.importorskip("astropy.units")

from nebula.geometry.fast_sun_position import sun_position_ecef

Time = astropy_time.Time
get_body = astropy_coordinates.get_body
ITRS = astropy_coordinates.ITRS


def _compute_error_stats(
    start_iso: str = "2000-01-01T00:00:00",
    end_iso: str = "2050-01-01T00:00:00",
    n_samples: int = 400,
    use_ut1: bool = True,
):
    t_start = Time(start_iso, scale="utc")
    t_end = Time(end_iso, scale="utc")

    total_days = (t_end - t_start).to_value(u.day)
    days = np.linspace(0.0, total_days, n_samples)
    times = t_start + days * u.day

    dist_errors_m = np.empty(n_samples, dtype=float)
    ang_errors_rad = np.empty(n_samples, dtype=float)

    for i, t in enumerate(times):
        sun_gcrs = get_body("sun", t)
        sun_itrs = sun_gcrs.transform_to(ITRS(obstime=t))
        xs, ys, zs = sun_itrs.cartesian.xyz.to_value(u.m)

        jd_tt = t.tt.jd
        if use_ut1:
            try:
                jd_ut1 = t.ut1.jd
            except Exception:
                jd_ut1 = t.utc.jd
        else:
            jd_ut1 = t.utc.jd

        xv, yv, zv = sun_position_ecef(jd_ut1, jd_tt)

        dx = xv - xs
        dy = yv - ys
        dz = zv - zs
        dist_errors_m[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

        v1_norm = np.sqrt(xv * xv + yv * yv + zv * zv)
        v2_norm = np.sqrt(xs * xs + ys * ys + zs * zs)

        if v1_norm == 0.0 or v2_norm == 0.0:
            ang_errors_rad[i] = 0.0
        else:
            dot = (xv * xs + yv * ys + zv * zs) / (v1_norm * v2_norm)
            dot = max(-1.0, min(1.0, dot))
            ang_errors_rad[i] = np.arccos(dot)

    dist_errors_km = dist_errors_m / 1000.0
    ang_errors_arcsec = ang_errors_rad * (180.0 / np.pi) * 3600.0

    return {
        "max_dist_km": float(np.max(dist_errors_km)),
        "rms_dist_km": float(np.sqrt(np.mean(dist_errors_km**2))),
        "max_ang_arcsec": float(np.max(ang_errors_arcsec)),
        "rms_ang_arcsec": float(np.sqrt(np.mean(ang_errors_arcsec**2))),
    }


def test_fast_sun_position_vs_astropy_ecef():
    fast = os.environ.get("FAST", "0") == "1"
    n_samples = 150 if fast else 400

    stats = _compute_error_stats(n_samples=n_samples, use_ut1=True)

    # Keep thresholds above expected truncated-model error while still catching regressions.
    assert stats["max_ang_arcsec"] <= 40.0
    assert stats["rms_ang_arcsec"] <= 20.0
