from __future__ import annotations

import numpy as np

import nebula.transforms as transforms
from nebula.transforms._coarse_eci2itrf import (
    _coarse_itrf2eci_pos_iau76_shortnut,
    _coarse_itrf2eci_pv_iau76_shortnut,
    coarse_ecef2eci,
    coarse_ecef2eci_pos,
    coarse_ecef2eci_pos_vec,
    coarse_ecef2eci_pos_vel,
    coarse_ecef2eci_pos_vel_vec,
    coarse_ecef2eci_vec,
    _coarse_eci2ecef_pos_iau76_shortnut,
    _coarse_eci2ecef_pv_iau76_shortnut,
    coarse_eci2ecef_pos,
    coarse_eci2ecef_pos_vec,
    coarse_eci2ecef_pos_vel,
    coarse_eci2ecef_pos_vel_vec,
)


def _random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rmag = rng.uniform(6.6e6, 4.2e7, size=n)
    r_eci = u * rmag[:, None]

    v = rng.standard_normal((n, 3))
    v *= 7800.0 / np.linalg.norm(v, axis=1, keepdims=True)

    jd_ut1 = 2451545.0 + rng.uniform(-20000.0, 20000.0, size=n)
    jd_tt = jd_ut1 + 69.184 / 86400.0
    return (
        r_eci.astype(np.float64),
        v.astype(np.float64),
        jd_ut1.astype(np.float64),
        jd_tt.astype(np.float64),
    )


def test_coarse_eci2itrf_scalar_pos_matches_internal_kernel() -> None:
    r, _, jd_ut1, jd_tt = _random_states(32, seed=1)

    for i in range(r.shape[0]):
        ref = _coarse_eci2ecef_pos_iau76_shortnut(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        got = coarse_eci2ecef_pos(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_scalar_pos_vel_matches_internal_kernel() -> None:
    r, v, jd_ut1, jd_tt = _random_states(32, seed=2)

    for i in range(r.shape[0]):
        ref = _coarse_eci2ecef_pv_iau76_shortnut(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(v[i, 0]),
            float(v[i, 1]),
            float(v[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        got = coarse_eci2ecef_pos_vel(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(v[i, 0]),
            float(v[i, 1]),
            float(v[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_vector_pos_matches_scalar_public_api() -> None:
    r, _, jd_ut1, jd_tt = _random_states(256, seed=3)
    r_out = coarse_eci2ecef_pos_vec(r, jd_ut1, jd_tt)

    r_ref = np.empty_like(r)
    for i in range(r.shape[0]):
        x, y, z = coarse_eci2ecef_pos(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
        )
        r_ref[i, 0] = x
        r_ref[i, 1] = y
        r_ref[i, 2] = z

    np.testing.assert_allclose(r_out, r_ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_vector_pos_vel_matches_scalar_public_api() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=4)
    r_out, v_out = coarse_eci2ecef_pos_vel_vec(r, v, jd_ut1, jd_tt)

    r_ref = np.empty_like(r)
    v_ref = np.empty_like(v)
    for i in range(r.shape[0]):
        x, y, z, vx, vy, vz = coarse_eci2ecef_pos_vel(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(v[i, 0]),
            float(v[i, 1]),
            float(v[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
        )
        r_ref[i, 0] = x
        r_ref[i, 1] = y
        r_ref[i, 2] = z
        v_ref[i, 0] = vx
        v_ref[i, 1] = vy
        v_ref[i, 2] = vz

    np.testing.assert_allclose(r_out, r_ref, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(v_out, v_ref, atol=0.0, rtol=0.0)


def test_coarse_ecef2eci_scalar_pos_matches_internal_kernel() -> None:
    r, _, jd_ut1, jd_tt = _random_states(32, seed=5)

    for i in range(r.shape[0]):
        r_ecef = coarse_eci2ecef_pos(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            1.0e-6,
            -1.5e-6,
        )
        ref = _coarse_itrf2eci_pos_iau76_shortnut(
            float(r_ecef[0]),
            float(r_ecef[1]),
            float(r_ecef[2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            1.0e-6,
            -1.5e-6,
        )
        got = coarse_ecef2eci_pos(
            float(r_ecef[0]),
            float(r_ecef[1]),
            float(r_ecef[2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            1.0e-6,
            -1.5e-6,
        )
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


def test_coarse_ecef2eci_scalar_pos_vel_matches_internal_kernel() -> None:
    r, v, jd_ut1, jd_tt = _random_states(32, seed=6)

    for i in range(r.shape[0]):
        rv_ecef = coarse_eci2ecef_pos_vel(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(v[i, 0]),
            float(v[i, 1]),
            float(v[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            2.0e-6,
            1.0e-6,
        )
        ref = _coarse_itrf2eci_pv_iau76_shortnut(
            float(rv_ecef[0]),
            float(rv_ecef[1]),
            float(rv_ecef[2]),
            float(rv_ecef[3]),
            float(rv_ecef[4]),
            float(rv_ecef[5]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            2.0e-6,
            1.0e-6,
        )
        got = coarse_ecef2eci_pos_vel(
            float(rv_ecef[0]),
            float(rv_ecef[1]),
            float(rv_ecef[2]),
            float(rv_ecef[3]),
            float(rv_ecef[4]),
            float(rv_ecef[5]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            2.0e-6,
            1.0e-6,
        )
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


def test_coarse_ecef2eci_roundtrip_position() -> None:
    r, _, jd_ut1, jd_tt = _random_states(256, seed=7)

    r_ecef = coarse_eci2ecef_pos_vec(r, jd_ut1, jd_tt, xp_rad=1.0e-6, yp_rad=-2.0e-6)
    r_back = coarse_ecef2eci_pos_vec(
        r_ecef, jd_ut1, jd_tt, xp_rad=1.0e-6, yp_rad=-2.0e-6
    )

    np.testing.assert_allclose(r_back, r, atol=2.0e-6, rtol=0.0)


def test_coarse_ecef2eci_roundtrip_position_velocity() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=8)

    r_ecef, v_ecef = coarse_eci2ecef_pos_vel_vec(
        r, v, jd_ut1, jd_tt, xp_rad=-1.0e-6, yp_rad=1.5e-6
    )
    r_back, v_back = coarse_ecef2eci_pos_vel_vec(
        r_ecef, v_ecef, jd_ut1, jd_tt, xp_rad=-1.0e-6, yp_rad=1.5e-6
    )

    np.testing.assert_allclose(r_back, r, atol=3.0e-6, rtol=0.0)
    np.testing.assert_allclose(v_back, v, atol=3.0e-9, rtol=0.0)


def test_coarse_ecef2eci_aliases_match_primary_functions() -> None:
    r, v, jd_ut1, jd_tt = _random_states(128, seed=9)

    r_ecef = coarse_eci2ecef_pos_vec(r, jd_ut1, jd_tt)
    r_back_a = coarse_ecef2eci_pos_vec(r_ecef, jd_ut1, jd_tt)
    r_back_b = coarse_ecef2eci_vec(r_ecef, jd_ut1, jd_tt)
    np.testing.assert_allclose(r_back_a, r_back_b, atol=0.0, rtol=0.0)

    x0, y0, z0 = coarse_ecef2eci_pos(
        float(r_ecef[0, 0]),
        float(r_ecef[0, 1]),
        float(r_ecef[0, 2]),
        float(jd_ut1[0]),
        float(jd_tt[0]),
    )
    x1, y1, z1 = coarse_ecef2eci(
        float(r_ecef[0, 0]),
        float(r_ecef[0, 1]),
        float(r_ecef[0, 2]),
        float(jd_ut1[0]),
        float(jd_tt[0]),
    )
    np.testing.assert_allclose([x0, y0, z0], [x1, y1, z1], atol=0.0, rtol=0.0)

    r_ecef2, v_ecef2 = coarse_eci2ecef_pos_vel_vec(r, v, jd_ut1, jd_tt)
    r_back, v_back = coarse_ecef2eci_pos_vel_vec(r_ecef2, v_ecef2, jd_ut1, jd_tt)
    np.testing.assert_allclose(r_back, r, atol=3.0e-6, rtol=0.0)
    np.testing.assert_allclose(v_back, v, atol=3.0e-9, rtol=0.0)


def test_public_names_exported_from_transform_namespace() -> None:
    assert callable(transforms.coarse_eci2ecef_pos)
    assert callable(transforms.coarse_eci2ecef_pos_vel)
    assert callable(transforms.coarse_eci2ecef_pos_vec)
    assert callable(transforms.coarse_eci2ecef_pos_vel_vec)
    assert callable(transforms.coarse_ecef2eci)
    assert callable(transforms.coarse_ecef2eci_pos)
    assert callable(transforms.coarse_ecef2eci_pos_vel)
    assert callable(transforms.coarse_ecef2eci_vec)
    assert callable(transforms.coarse_ecef2eci_pos_vec)
    assert callable(transforms.coarse_ecef2eci_pos_vel_vec)
