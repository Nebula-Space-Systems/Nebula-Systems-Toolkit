from __future__ import annotations

import numpy as np
from numba import njit

import nstk.transforms as transforms
from nstk.transforms._coarse_eci2geodetic import (
    _coarse_eci2geodetic_scalar,
    _coarse_eci2geodetic_vector,
    coarse_eci2geodetic,
)
from nstk.transforms._coarse_eci2itrf import _coarse_eci2ecef_pos_iau76_shortnut
from nstk.transforms._ecef2geodetic import ecef2geodetic


def _random_eci_samples(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rmag = rng.uniform(6.5e6, 4.3e7, size=n)
    r_eci = u * rmag[:, None]

    jd_ut1 = 2451545.0 + rng.uniform(-30000.0, 30000.0, size=n)
    jd_tt = jd_ut1 + 69.184 / 86400.0
    return r_eci.astype(np.float64), jd_ut1.astype(np.float64), jd_tt.astype(np.float64)


def test_coarse_eci2geodetic_scalar_matches_composed_path() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(32, seed=1)

    for i in range(r_eci.shape[0]):
        x, y, z = _coarse_eci2ecef_pos_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        la_ref, lo_ref, h_ref = ecef2geodetic(x, y, z)
        la, lo, h = _coarse_eci2geodetic_scalar(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose(
            [la, lo, h], [la_ref, lo_ref, h_ref], atol=1e-12, rtol=0.0
        )


def test_coarse_eci2geodetic_deg_scalar_matches_composed_path() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(32, seed=2)

    for i in range(r_eci.shape[0]):
        x, y, z = _coarse_eci2ecef_pos_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        la_ref, lo_ref, h_ref = ecef2geodetic(x, y, z, degrees=True)
        la, lo, h = _coarse_eci2geodetic_scalar(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
            True,
        )
        np.testing.assert_allclose(
            [la, lo, h], [la_ref, lo_ref, h_ref], atol=1e-12, rtol=0.0
        )


def test_coarse_eci2geodetic_vector_matches_scalar() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(256, seed=3)

    lat_v, lon_v, h_v = _coarse_eci2geodetic_vector(r_eci, jd_ut1, jd_tt)

    lat_s = np.empty_like(lat_v)
    lon_s = np.empty_like(lon_v)
    h_s = np.empty_like(h_v)
    for i in range(r_eci.shape[0]):
        la, lo, hi = _coarse_eci2geodetic_scalar(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
        )
        lat_s[i] = la
        lon_s[i] = lo
        h_s[i] = hi

    np.testing.assert_allclose(lat_v, lat_s, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(lon_v, lon_s, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(h_v, h_s, atol=1e-9, rtol=0.0)


def test_coarse_eci2geodetic_vector_deg_matches_scalar_deg() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(256, seed=4)

    lat_v, lon_v, h_v = _coarse_eci2geodetic_vector(r_eci, jd_ut1, jd_tt, degrees=True)

    lat_s = np.empty_like(lat_v)
    lon_s = np.empty_like(lon_v)
    h_s = np.empty_like(h_v)
    for i in range(r_eci.shape[0]):
        la, lo, hi = _coarse_eci2geodetic_scalar(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            degrees=True,
        )
        lat_s[i] = la
        lon_s[i] = lo
        h_s[i] = hi

    np.testing.assert_allclose(lat_v, lat_s, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(lon_v, lon_s, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(h_v, h_s, atol=1e-9, rtol=0.0)


def test_public_name_exported_from_transform_namespace() -> None:
    assert callable(transforms.coarse_eci2geodetic)


def test_public_function_matches_package_export() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(64, seed=8)

    lat0, lon0, h0 = coarse_eci2geodetic(r_eci, jd_ut1=jd_ut1, jd_tt=jd_tt)
    lat1, lon1, h1 = transforms.coarse_eci2geodetic(r_eci, jd_ut1=jd_ut1, jd_tt=jd_tt)
    np.testing.assert_allclose(lat0, lat1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lon0, lon1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(h0, h1, atol=0.0, rtol=0.0)

    i = 7
    s0 = coarse_eci2geodetic(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
    )
    s1 = transforms.coarse_eci2geodetic(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
    )
    np.testing.assert_allclose(s0, s1, atol=0.0, rtol=0.0)

    d0 = coarse_eci2geodetic(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
        degrees=True,
    )
    d1 = transforms.coarse_eci2geodetic(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
        degrees=True,
    )
    np.testing.assert_allclose(d0, d1, atol=0.0, rtol=0.0)

    vd0 = coarse_eci2geodetic(r_eci, jd_ut1=jd_ut1, jd_tt=jd_tt, degrees=True)
    vd1 = transforms.coarse_eci2geodetic(r_eci, jd_ut1=jd_ut1, jd_tt=jd_tt, degrees=True)
    np.testing.assert_allclose(vd0[0], vd1[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vd0[1], vd1[1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vd0[2], vd1[2], atol=0.0, rtol=0.0)

def test_matrix_input_form_works_inside_njit() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(16, seed=10)

    @njit
    def coarse_eci2geodetic_jit(r_eci_m, jd_ut1_arr, jd_tt_arr):
        return coarse_eci2geodetic(r_eci_m, jd_ut1=jd_ut1_arr, jd_tt=jd_tt_arr, degrees=True)

    expected = coarse_eci2geodetic(r_eci, jd_ut1=jd_ut1, jd_tt=jd_tt, degrees=True)
    got = coarse_eci2geodetic_jit(r_eci, jd_ut1, jd_tt)

    for expected_part, got_part in zip(expected, got):
        np.testing.assert_allclose(got_part, expected_part, atol=0.0, rtol=0.0)


def test_scalar_input_form_with_default_polar_motion_works_inside_njit() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(4, seed=11)

    @njit
    def coarse_eci2geodetic_scalar_jit(x_eci_m, y_eci_m, z_eci_m, jd_ut1_val, jd_tt_val):
        return coarse_eci2geodetic(
            x_eci_m,
            y_eci_m,
            z_eci_m,
            jd_ut1_val,
            jd_tt_val,
            degrees=True,
        )

    i = 2
    expected = coarse_eci2geodetic(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
        degrees=True,
    )
    got = coarse_eci2geodetic_scalar_jit(
        float(r_eci[i, 0]),
        float(r_eci[i, 1]),
        float(r_eci[i, 2]),
        float(jd_ut1[i]),
        float(jd_tt[i]),
    )
    np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)
