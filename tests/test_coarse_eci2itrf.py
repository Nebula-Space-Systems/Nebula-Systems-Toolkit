from __future__ import annotations

import numpy as np
from numba import njit

import nstk.transforms as transforms
from nstk.transforms._coarse_eci2itrf import (
    _coarse_eci2ecef_pos_iau76_shortnut,
    _coarse_eci2ecef_pv_iau76_shortnut,
    _coarse_itrf2eci_pos_iau76_shortnut,
    _coarse_itrf2eci_pv_iau76_shortnut,
    coarse_ecef2eci_pos,
    coarse_ecef2eci_pos_vel,
    coarse_eci2ecef_pos,
    coarse_eci2ecef_pos_vel,
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


def test_coarse_eci2itrf_split_array_pos_matches_scalar_public_api() -> None:
    r, _, jd_ut1, jd_tt = _random_states(256, seed=3)
    x_out, y_out, z_out = coarse_eci2ecef_pos(r[:, 0], r[:, 1], r[:, 2], jd_ut1, jd_tt)

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

    np.testing.assert_allclose(np.column_stack((x_out, y_out, z_out)), r_ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_matrix_pos_matches_scalar_public_api() -> None:
    r, _, jd_ut1, jd_tt = _random_states(256, seed=4)
    x_out, y_out, z_out = coarse_eci2ecef_pos(r, jd_ut1=jd_ut1, jd_tt=jd_tt)

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

    np.testing.assert_allclose(np.column_stack((x_out, y_out, z_out)), r_ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_split_array_pos_vel_matches_scalar_public_api() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=5)
    got = coarse_eci2ecef_pos_vel(
        r[:, 0],
        r[:, 1],
        r[:, 2],
        v[:, 0],
        v[:, 1],
        v[:, 2],
        jd_ut1,
        jd_tt,
    )

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

    np.testing.assert_allclose(np.column_stack(got[:3]), r_ref, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.column_stack(got[3:]), v_ref, atol=0.0, rtol=0.0)


def test_coarse_eci2itrf_matrix_pos_vel_matches_scalar_public_api() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=6)
    got = coarse_eci2ecef_pos_vel(r, v, jd_ut1=jd_ut1, jd_tt=jd_tt)

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

    np.testing.assert_allclose(np.column_stack(got[:3]), r_ref, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.column_stack(got[3:]), v_ref, atol=0.0, rtol=0.0)


def test_coarse_ecef2eci_scalar_pos_matches_internal_kernel() -> None:
    r, _, jd_ut1, jd_tt = _random_states(32, seed=7)

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
    r, v, jd_ut1, jd_tt = _random_states(32, seed=8)

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
    r, _, jd_ut1, jd_tt = _random_states(256, seed=9)

    x_ecef, y_ecef, z_ecef = coarse_eci2ecef_pos(
        r,
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
        xp_rad=1.0e-6,
        yp_rad=-2.0e-6,
    )
    x_back, y_back, z_back = coarse_ecef2eci_pos(
        np.column_stack((x_ecef, y_ecef, z_ecef)),
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
        xp_rad=1.0e-6,
        yp_rad=-2.0e-6,
    )

    np.testing.assert_allclose(np.column_stack((x_back, y_back, z_back)), r, atol=2.0e-6, rtol=0.0)


def test_coarse_ecef2eci_roundtrip_position_velocity() -> None:
    r, v, jd_ut1, jd_tt = _random_states(256, seed=10)

    state_ecef = coarse_eci2ecef_pos_vel(
        r,
        v,
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
        xp_rad=-1.0e-6,
        yp_rad=1.5e-6,
    )
    r_ecef = np.column_stack(state_ecef[:3])
    v_ecef = np.column_stack(state_ecef[3:])

    state_back = coarse_ecef2eci_pos_vel(
        r_ecef,
        v_ecef,
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
        xp_rad=-1.0e-6,
        yp_rad=1.5e-6,
    )

    np.testing.assert_allclose(np.column_stack(state_back[:3]), r, atol=3.0e-6, rtol=0.0)
    np.testing.assert_allclose(np.column_stack(state_back[3:]), v, atol=3.0e-9, rtol=0.0)


def test_unified_coarse_interfaces_work_inside_njit() -> None:
    r, v, jd_ut1, jd_tt = _random_states(32, seed=11)

    @njit
    def coarse_pos_split_jit(x_eci_m, y_eci_m, z_eci_m, jd_ut1_arr, jd_tt_arr):
        return coarse_eci2ecef_pos(x_eci_m, y_eci_m, z_eci_m, jd_ut1_arr, jd_tt_arr)

    @njit
    def coarse_pos_matrix_jit(r_eci_m, jd_ut1_arr, jd_tt_arr):
        return coarse_eci2ecef_pos(r_eci_m, jd_ut1=jd_ut1_arr, jd_tt=jd_tt_arr)

    @njit
    def coarse_state_split_jit(
        x_eci_m,
        y_eci_m,
        z_eci_m,
        vx_eci_mps,
        vy_eci_mps,
        vz_eci_mps,
        jd_ut1_arr,
        jd_tt_arr,
    ):
        return coarse_eci2ecef_pos_vel(
            x_eci_m,
            y_eci_m,
            z_eci_m,
            vx_eci_mps,
            vy_eci_mps,
            vz_eci_mps,
            jd_ut1_arr,
            jd_tt_arr,
        )

    @njit
    def coarse_state_matrix_jit(r_eci_m, v_eci_mps, jd_ut1_arr, jd_tt_arr):
        return coarse_eci2ecef_pos_vel(r_eci_m, v_eci_mps, jd_ut1=jd_ut1_arr, jd_tt=jd_tt_arr)

    @njit
    def coarse_back_pos_matrix_jit(r_ecef_m, jd_ut1_arr, jd_tt_arr):
        return coarse_ecef2eci_pos(r_ecef_m, jd_ut1=jd_ut1_arr, jd_tt=jd_tt_arr)

    @njit
    def coarse_back_state_matrix_jit(r_ecef_m, v_ecef_mps, jd_ut1_arr, jd_tt_arr):
        return coarse_ecef2eci_pos_vel(
            r_ecef_m,
            v_ecef_mps,
            jd_ut1=jd_ut1_arr,
            jd_tt=jd_tt_arr,
        )

    expected_pos_split = coarse_eci2ecef_pos(r[:, 0], r[:, 1], r[:, 2], jd_ut1, jd_tt)
    got_pos_split = coarse_pos_split_jit(r[:, 0], r[:, 1], r[:, 2], jd_ut1, jd_tt)
    for expected, got in zip(expected_pos_split, got_pos_split):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    expected_pos_matrix = coarse_eci2ecef_pos(r, jd_ut1=jd_ut1, jd_tt=jd_tt)
    got_pos_matrix = coarse_pos_matrix_jit(r, jd_ut1, jd_tt)
    for expected, got in zip(expected_pos_matrix, got_pos_matrix):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    expected_state_split = coarse_eci2ecef_pos_vel(
        r[:, 0],
        r[:, 1],
        r[:, 2],
        v[:, 0],
        v[:, 1],
        v[:, 2],
        jd_ut1,
        jd_tt,
    )
    got_state_split = coarse_state_split_jit(
        r[:, 0],
        r[:, 1],
        r[:, 2],
        v[:, 0],
        v[:, 1],
        v[:, 2],
        jd_ut1,
        jd_tt,
    )
    for expected, got in zip(expected_state_split, got_state_split):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    expected_state_matrix = coarse_eci2ecef_pos_vel(r, v, jd_ut1=jd_ut1, jd_tt=jd_tt)
    got_state_matrix = coarse_state_matrix_jit(r, v, jd_ut1, jd_tt)
    for expected, got in zip(expected_state_matrix, got_state_matrix):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    r_ecef = np.column_stack(expected_pos_matrix)
    expected_back_pos = coarse_ecef2eci_pos(r_ecef, jd_ut1=jd_ut1, jd_tt=jd_tt)
    got_back_pos = coarse_back_pos_matrix_jit(r_ecef, jd_ut1, jd_tt)
    for expected, got in zip(expected_back_pos, got_back_pos):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    r_ecef_state = np.column_stack(expected_state_matrix[:3])
    v_ecef_state = np.column_stack(expected_state_matrix[3:])
    expected_back_state = coarse_ecef2eci_pos_vel(
        r_ecef_state,
        v_ecef_state,
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
    )
    got_back_state = coarse_back_state_matrix_jit(r_ecef_state, v_ecef_state, jd_ut1, jd_tt)
    for expected, got in zip(expected_back_state, got_back_state):
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)


def test_public_names_exported_from_transform_namespace() -> None:
    assert callable(transforms.coarse_eci2ecef_pos)
    assert callable(transforms.coarse_eci2ecef_pos_vel)
    assert callable(transforms.coarse_ecef2eci_pos)
    assert callable(transforms.coarse_ecef2eci_pos_vel)
    assert not hasattr(transforms, "coarse_eci2ecef_pos_vec")
    assert not hasattr(transforms, "coarse_eci2ecef_pos_vel_vec")
    assert not hasattr(transforms, "coarse_ecef2eci")
    assert not hasattr(transforms, "coarse_ecef2eci_vec")
    assert not hasattr(transforms, "coarse_ecef2eci_pos_vec")
    assert not hasattr(transforms, "coarse_ecef2eci_pos_vel_vec")
