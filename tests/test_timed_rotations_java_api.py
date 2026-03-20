from __future__ import annotations

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

from nstk.time_utils import astropy_time_to_orekit_date
from nstk.transforms import transform

Time = astropy_time.Time
GCRS = astropy_coordinates.GCRS
ITRS = astropy_coordinates.ITRS
CartesianRepresentation = astropy_coordinates.CartesianRepresentation


def _random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rmag = rng.uniform(6.6e6, 4.2e7, size=n)
    r = d * rmag[:, None]

    v = rng.standard_normal((n, 3))
    v *= 7800.0 / np.linalg.norm(v, axis=1, keepdims=True)

    a = rng.standard_normal((n, 3)) * 1.0e-3
    return r.astype(np.float64), v.astype(np.float64), a.astype(np.float64)


def _astropy_transform_pos_gcrf_to_itrf(r_m: np.ndarray, times: Time) -> np.ndarray:
    rep = CartesianRepresentation(r_m[:, 0] * u.m, r_m[:, 1] * u.m, r_m[:, 2] * u.m)
    c0 = GCRS(rep, obstime=times)
    c1 = c0.transform_to(ITRS(obstime=times))
    return c1.cartesian.xyz.to_value(u.m).T.astype(np.float64)


def test_transform_identity_with_broadcast_scalar_time() -> None:
    r, v, a = _random_states(8, seed=11)
    t = Time("2026-01-01T00:00:00", scale="utc")

    p_out, v_out, a_out = transform(
        from_frame="gcrf",
        to_frame="gcrf",
        time=t,
        position=r,
        velocity=v,
        acceleration=a,
    )

    assert p_out.shape == (8, 3)
    assert v_out is not None and v_out.shape == (8, 3)
    assert a_out is not None and a_out.shape == (8, 3)
    np.testing.assert_allclose(p_out, r, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(v_out, v, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(a_out, a, atol=1e-12, rtol=0.0)


def test_transform_pos_matches_astropy_and_pos_vel_roundtrip() -> None:
    n = 20
    r, v, _ = _random_states(n, seed=12)
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    times = Time(
        t0.unix + np.linspace(0.0, 2.0 * 3600.0, n, dtype=np.float64),
        format="unix",
        scale="utc",
    )

    p_new, v_new, a_new = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r,
        velocity=v,
    )
    p_ast = _astropy_transform_pos_gcrf_to_itrf(r, times)
    err = np.linalg.norm(p_new - p_ast, axis=1)

    assert a_new is None
    assert float(np.max(err)) < 120.0
    assert float(np.mean(err)) < 40.0

    p_back, v_back, a_back = transform(
        from_frame="itrf",
        to_frame="gcrf",
        time=times,
        position=p_new,
        velocity=v_new,
    )
    assert a_back is None
    np.testing.assert_allclose(p_back, r, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(v_back, v, atol=2e-9, rtol=0.0)


def test_transform_accepts_absolutedate_vectors() -> None:
    n = 7
    r, _, _ = _random_states(n, seed=13)
    epoch = Time("2026-02-01T12:00:00", scale="utc")
    base = astropy_time_to_orekit_date(epoch)
    dates = [base.shiftedBy(float(dt)) for dt in np.linspace(0.0, 180.0, n)]

    p_abs, _, _ = transform(
        from_frame="teme",
        to_frame="mod",
        time=dates,
        position=r,
    )

    t_ast = Time(epoch.unix + np.linspace(0.0, 180.0, n), format="unix", scale="utc")
    p_ast, _, _ = transform(
        from_frame="teme",
        to_frame="mod",
        time=t_ast,
        position=r,
    )

    np.testing.assert_allclose(p_abs, p_ast, atol=1e-8, rtol=0.0)


def test_transform_quantity_and_acceleration_without_velocity() -> None:
    n = 5
    r, _, a = _random_states(n, seed=14)
    t0 = Time("2026-03-01T00:00:00", scale="utc")
    times = Time(t0.unix + np.arange(n, dtype=np.float64), format="unix", scale="utc")

    p_q, v_none, a_q = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r * u.km,
        acceleration=a * (u.km / (u.s**2)),
    )

    assert v_none is None
    assert hasattr(p_q, "unit") and p_q.unit == u.m
    assert hasattr(a_q, "unit") and a_q.unit == (u.m / (u.s**2))
    assert p_q.shape == (n, 3)
    assert a_q.shape == (n, 3)
    assert np.all(np.isfinite(p_q.to_value(u.m)))
    assert np.all(np.isfinite(a_q.to_value(u.m / (u.s**2))))


def test_transform_scalar_position_and_scalar_time_shape() -> None:
    t = Time("2026-03-01T00:00:00", scale="utc")
    p = np.array([7000e3, 0.0, 0.0], dtype=np.float64)

    p_out, v_out, a_out = transform("gcrf", "itrf", t, p)

    assert p_out.shape == (3,)
    assert v_out is None
    assert a_out is None
