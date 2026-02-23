from __future__ import annotations

import numpy as np
import pytest

import nebula.transforms as transforms

pytestmark = pytest.mark.filterwarnings(
    "ignore:Tried to get polar motions for times after IERS data is valid.*:astropy.utils.exceptions.AstropyWarning"
)

astropy_time = pytest.importorskip("astropy.time")
u = pytest.importorskip("astropy.units")

from nebula.transforms._coarse_eci2itrf import (
    coarse_eci2itrf_pos,
    coarse_eci2itrf_pos_vec,
    coarse_eci2itrf_pos_vel as coarse_eci2itrf_pos_vel_kernel,
    coarse_eci2itrf_pos_vel_vec,
)
from nebula.transforms._timed_rotations import (
    coarse_eci2itrf,
    coarse_eci2itrf_pos_vel,
)

Time = astropy_time.Time


def test_coarse_eci2itrf_scalar() -> None:
    t = Time("2026-01-01T00:00:00", scale="utc")

    x, y, z = coarse_eci2itrf(
        t,
        7000e3,
        10e3,
        -2e3,
    )

    xr, yr, zr = coarse_eci2itrf_pos(
        7000e3, 10e3, -2e3, float(t.ut1.jd), float(t.tt.jd)
    )
    np.testing.assert_allclose([x, y, z], [xr, yr, zr], atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_vector_matches_kernel() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    dt = np.arange(6, dtype=np.float64) * 20.0
    t = t0 + dt * u.s

    x = np.linspace(6800e3, 6820e3, 6)
    y = np.linspace(100e3, 120e3, 6)
    z = np.linspace(-10e3, 5e3, 6)

    xo, yo, zo = coarse_eci2itrf(t, x, y, z)

    r_eci = np.column_stack((x, y, z)).astype(np.float64)
    r_ref = coarse_eci2itrf_pos_vec(
        r_eci,
        np.asarray(t.ut1.jd, dtype=np.float64),
        np.asarray(t.tt.jd, dtype=np.float64),
    )

    np.testing.assert_allclose(xo, r_ref[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(yo, r_ref[:, 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zo, r_ref[:, 2], atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_broadcasting() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t = t0 + np.array([0.0, 30.0, 60.0], dtype=np.float64) * u.s

    xo, yo, zo = coarse_eci2itrf(t, 7000e3, 0.0, np.array([0.0, 5.0, 10.0]))
    assert xo.shape == (3,)
    assert yo.shape == (3,)
    assert zo.shape == (3,)


def test_coarse_eci2itrf_pos_vel_vector_matches_kernel() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t = t0 + np.arange(5, dtype=np.float64) * 15.0 * u.s

    x = np.linspace(6900e3, 6920e3, 5)
    y = np.linspace(-50e3, -40e3, 5)
    z = np.linspace(0.0, 1e3, 5)
    vx = np.linspace(0.0, -10.0, 5)
    vy = np.linspace(7600.0, 7590.0, 5)
    vz = np.linspace(100.0, 110.0, 5)

    out = coarse_eci2itrf_pos_vel(t, x, y, z, vx, vy, vz)
    xo, yo, zo, vxo, vyo, vzo = out

    r_eci = np.column_stack((x, y, z)).astype(np.float64)
    v_eci = np.column_stack((vx, vy, vz)).astype(np.float64)
    r_ref, v_ref = coarse_eci2itrf_pos_vel_vec(
        r_eci,
        v_eci,
        np.asarray(t.ut1.jd, dtype=np.float64),
        np.asarray(t.tt.jd, dtype=np.float64),
    )

    np.testing.assert_allclose(xo, r_ref[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(yo, r_ref[:, 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(zo, r_ref[:, 2], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vxo, v_ref[:, 0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vyo, v_ref[:, 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vzo, v_ref[:, 2], atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_rejects_non_broadcastable_lengths() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t = t0 + np.array([0.0, 10.0, 20.0], dtype=np.float64) * u.s

    with pytest.raises(ValueError):
        coarse_eci2itrf(
            t,
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        )


def test_coarse_eci2itrf_pos_vel_scalar() -> None:
    t = Time("2026-01-01T00:00:00", scale="utc")

    x, y, z, vx, vy, vz = coarse_eci2itrf_pos_vel(
        t,
        7000e3,
        1e3,
        -500.0,
        0.0,
        7500.0,
        100.0,
    )

    ref = coarse_eci2itrf_pos_vel_kernel(
        7000e3,
        1e3,
        -500.0,
        0.0,
        7500.0,
        100.0,
        float(t.ut1.jd),
        float(t.tt.jd),
    )
    np.testing.assert_allclose([x, y, z, vx, vy, vz], ref, atol=0.0, rtol=0.0)


def test_new_timed_coarse_wrappers_exported_from_namespace() -> None:
    assert callable(transforms.coarse_eci2itrf)
    assert callable(transforms.coarse_eci2itrf_pos_vel)
