from __future__ import annotations

import numpy as np

import nebula.transform as transform
from nebula.transform._coarse_eci2itrf import (
    _coarse_eci_to_itrf_pos_iau76_shortnut,
    _coarse_eci_to_itrf_pv_iau76_shortnut,
    coarse_eci_to_itrf_pos_iau76_shortnut,
    coarse_eci_to_itrf_pos_vec_iau76_shortnut,
    coarse_eci_to_itrf_pos_vel_iau76_shortnut,
    coarse_eci_to_itrf_pos_vel_vec_iau76_shortnut,
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


def test_coarse_eci_to_itrf_scalar_pos_matches_internal_kernel() -> None:
    r, _, jd_ut1, jd_tt = _random_states(32, seed=1)

    for i in range(r.shape[0]):
        ref = _coarse_eci_to_itrf_pos_iau76_shortnut(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        got = coarse_eci_to_itrf_pos_iau76_shortnut(
            float(r[i, 0]),
            float(r[i, 1]),
            float(r[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


def test_coarse_eci_to_itrf_scalar_pos_vel_matches_internal_kernel() -> None:
    r, v, jd_ut1, jd_tt = _random_states(32, seed=2)

    for i in range(r.shape[0]):
        ref = _coarse_eci_to_itrf_pv_iau76_shortnut(
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
        got = coarse_eci_to_itrf_pos_vel_iau76_shortnut(
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


def test_coarse_eci_to_itrf_vector_pos_matches_scalar_public_api() -> None:
    r, _, jd_ut1, jd_tt = _random_states(256, seed=3)
    r_out = coarse_eci_to_itrf_pos_vec_iau76_shortnut(r, jd_ut1, jd_tt)

    r_ref = np.empty_like(r)
    for i in range(r.shape[0]):
        x, y, z = coarse_eci_to_itrf_pos_iau76_shortnut(
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


def test_coarse_eci_to_itrf_vector_pos_vel_matches_scalar_public_api() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=4)
    r_out, v_out = coarse_eci_to_itrf_pos_vel_vec_iau76_shortnut(r, v, jd_ut1, jd_tt)

    r_ref = np.empty_like(r)
    v_ref = np.empty_like(v)
    for i in range(r.shape[0]):
        x, y, z, vx, vy, vz = coarse_eci_to_itrf_pos_vel_iau76_shortnut(
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


def test_public_aliases_exported_from_transform_namespace() -> None:
    assert callable(transform.coarse_eci_to_itrf_pos_iau76_shortnut)
    assert callable(transform.coarse_eci_to_itrf_pos_vel_iau76_shortnut)
    assert callable(transform.coarse_eci_to_itrf_pos_vec_iau76_shortnut)
    assert callable(transform.coarse_eci_to_itrf_pos_vel_vec_iau76_shortnut)
