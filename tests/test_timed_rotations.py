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

from nebula.transforms._timed_rotations import (
    transform_pos_vel_timed,
    transform_positions_timed,
    transform_timed,
)

Time = astropy_time.Time
GCRS = astropy_coordinates.GCRS
ITRS = astropy_coordinates.ITRS
TEME = astropy_coordinates.TEME
CartesianRepresentation = astropy_coordinates.CartesianRepresentation


def _random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rmag = rng.uniform(6.6e6, 4.2e7, size=n)
    r = d * rmag[:, None]

    v = rng.standard_normal((n, 3))
    v *= 7800.0 / np.linalg.norm(v, axis=1, keepdims=True)
    return r.astype(np.float64), v.astype(np.float64)


def _astropy_transform_pos(
    r_m: np.ndarray, times: Time, from_frame: str, to_frame: str
) -> np.ndarray:
    rep = CartesianRepresentation(r_m[:, 0] * u.m, r_m[:, 1] * u.m, r_m[:, 2] * u.m)

    frm = from_frame.lower()
    to = to_frame.lower()

    if frm == "gcrf":
        c0 = GCRS(rep, obstime=times)
    elif frm == "itrf":
        c0 = ITRS(rep, obstime=times)
    elif frm == "teme":
        c0 = TEME(rep, obstime=times)
    else:
        raise ValueError(f"Unsupported astropy comparison frame: {from_frame}")

    if to == "gcrf":
        c1 = c0.transform_to(GCRS(obstime=times))
    elif to == "itrf":
        c1 = c0.transform_to(ITRS(obstime=times))
    elif to == "teme":
        c1 = c0.transform_to(TEME(obstime=times))
    else:
        raise ValueError(f"Unsupported astropy comparison frame: {to_frame}")

    return c1.cartesian.xyz.to_value(u.m).T.astype(np.float64)


def test_timed_rotations_gcrf_to_itrf_vs_astropy_position() -> None:
    n = 96
    r, _ = _random_states(n, seed=1)
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    times = t0 + np.linspace(0.0, 3.0, n) * u.day

    r_ore = transform_positions_timed(times, r, "gcrf", "itrf")
    r_ast = _astropy_transform_pos(r, times, "gcrf", "itrf")

    err = np.linalg.norm(r_ore - r_ast, axis=1)
    assert float(np.max(err)) < 120.0
    assert float(np.mean(err)) < 40.0


def test_timed_rotations_teme_to_itrf_vs_astropy_position() -> None:
    n = 96
    r, _ = _random_states(n, seed=2)
    t0 = Time("2026-03-01T00:00:00", scale="utc")
    times = t0 + np.linspace(0.0, 2.0, n) * u.day

    r_ore = transform_positions_timed(times, r, "teme", "itrf")
    r_ast = _astropy_transform_pos(r, times, "teme", "itrf")

    err = np.linalg.norm(r_ore - r_ast, axis=1)
    assert float(np.max(err)) < 150.0
    assert float(np.mean(err)) < 50.0


def test_timed_rotations_pos_vel_roundtrip_is_stable() -> None:
    n = 128
    r, v = _random_states(n, seed=3)
    t0 = Time("2027-01-01T00:00:00", scale="utc")
    times = t0 + np.linspace(0.0, 6.0, n) * u.hour

    r_i, v_i = transform_pos_vel_timed(times, r, v, "gcrf", "itrf")
    r_b, v_b = transform_pos_vel_timed(times, r_i, v_i, "itrf", "gcrf")

    np.testing.assert_allclose(r_b, r, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(v_b, v, atol=2e-9, rtol=0.0)


def test_timed_rotations_interface_shapes_and_optional_velocity() -> None:
    r, v = _random_states(1, seed=4)
    t_scalar = Time("2026-01-01T00:00:00", scale="utc")
    t_vec = t_scalar + np.arange(5, dtype=np.float64) * u.s

    r1 = transform_timed(t_scalar, r[0], "gcrf", "itrf")
    assert isinstance(r1, np.ndarray) and r1.shape == (3,)

    r2 = transform_timed(t_vec, r[0], "gcrf", "itrf")
    assert isinstance(r2, np.ndarray) and r2.shape == (5, 3)

    r3, v3 = transform_timed(t_scalar, r[0], "gcrf", "itrf", velocities_mps=v[0])
    assert r3.shape == (3,)
    assert v3.shape == (3,)
