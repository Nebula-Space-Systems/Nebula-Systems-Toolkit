from __future__ import annotations

import numpy as np

import nebula.transform as transform
from nebula.transform._coarse_eci2geodetic import (
    _coarse_eci_to_geodetic_deg_iau76_shortnut,
    _coarse_eci_to_geodetic_iau76_shortnut,
    _coarse_eci_to_geodetic_vec_deg_iau76_shortnut,
    _coarse_eci_to_geodetic_vec_iau76_shortnut,
    coarse_eci_to_geodetic,
    coarse_eci_to_geodetic_deg,
    coarse_eci_to_geodetic_vec,
    coarse_eci_to_geodetic_vec_deg,
)
from nebula.transform._coarse_eci2itrf import _coarse_eci_to_itrf_pos_iau76_shortnut
from nebula.transform._ecef2geodetic import ecef2geodetic, ecef2geodetic_deg


def _random_eci_samples(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rmag = rng.uniform(6.5e6, 4.3e7, size=n)
    r_eci = u * rmag[:, None]

    jd_ut1 = 2451545.0 + rng.uniform(-30000.0, 30000.0, size=n)
    jd_tt = jd_ut1 + 69.184 / 86400.0
    return r_eci.astype(np.float64), jd_ut1.astype(np.float64), jd_tt.astype(np.float64)


def test_coarse_eci_to_geodetic_scalar_matches_composed_path() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(32, seed=1)

    for i in range(r_eci.shape[0]):
        x, y, z = _coarse_eci_to_itrf_pos_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        la_ref, lo_ref, h_ref = ecef2geodetic(x, y, z)
        la, lo, h = _coarse_eci_to_geodetic_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose([la, lo, h], [la_ref, lo_ref, h_ref], atol=1e-12, rtol=0.0)


def test_coarse_eci_to_geodetic_deg_scalar_matches_composed_path() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(32, seed=2)

    for i in range(r_eci.shape[0]):
        x, y, z = _coarse_eci_to_itrf_pos_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        la_ref, lo_ref, h_ref = ecef2geodetic_deg(x, y, z)
        la, lo, h = _coarse_eci_to_geodetic_deg_iau76_shortnut(
            float(r_eci[i, 0]),
            float(r_eci[i, 1]),
            float(r_eci[i, 2]),
            float(jd_ut1[i]),
            float(jd_tt[i]),
            0.0,
            0.0,
        )
        np.testing.assert_allclose([la, lo, h], [la_ref, lo_ref, h_ref], atol=1e-12, rtol=0.0)


def test_coarse_eci_to_geodetic_vector_matches_scalar() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(256, seed=3)

    lat_v, lon_v, h_v = _coarse_eci_to_geodetic_vec_iau76_shortnut(r_eci, jd_ut1, jd_tt)

    lat_s = np.empty_like(lat_v)
    lon_s = np.empty_like(lon_v)
    h_s = np.empty_like(h_v)
    for i in range(r_eci.shape[0]):
        la, lo, hi = _coarse_eci_to_geodetic_iau76_shortnut(
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


def test_coarse_eci_to_geodetic_vector_deg_matches_scalar_deg() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(256, seed=4)

    lat_v, lon_v, h_v = _coarse_eci_to_geodetic_vec_deg_iau76_shortnut(
        r_eci, jd_ut1, jd_tt
    )

    lat_s = np.empty_like(lat_v)
    lon_s = np.empty_like(lon_v)
    h_s = np.empty_like(h_v)
    for i in range(r_eci.shape[0]):
        la, lo, hi = _coarse_eci_to_geodetic_deg_iau76_shortnut(
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


def test_public_aliases_exported_from_transform_namespace() -> None:
    assert callable(transform.coarse_eci_to_geodetic)
    assert callable(transform.coarse_eci_to_geodetic_deg)
    assert callable(transform.coarse_eci_to_geodetic_vec)
    assert callable(transform.coarse_eci_to_geodetic_vec_deg)
    assert callable(transform.coarse_eci_to_geodetic_iau76_shortnut)
    assert callable(transform.coarse_eci_to_geodetic_deg_iau76_shortnut)
    assert callable(transform.coarse_eci_to_geodetic_vec_iau76_shortnut)
    assert callable(transform.coarse_eci_to_geodetic_vec_deg_iau76_shortnut)


def test_simple_public_names_match_legacy_names() -> None:
    r_eci, jd_ut1, jd_tt = _random_eci_samples(64, seed=8)

    lat0, lon0, h0 = coarse_eci_to_geodetic_vec(r_eci, jd_ut1, jd_tt)
    lat1, lon1, h1 = transform.coarse_eci_to_geodetic_vec_iau76_shortnut(r_eci, jd_ut1, jd_tt)
    np.testing.assert_allclose(lat0, lat1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(lon0, lon1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(h0, h1, atol=0.0, rtol=0.0)

    i = 7
    s0 = coarse_eci_to_geodetic(
        float(r_eci[i, 0]), float(r_eci[i, 1]), float(r_eci[i, 2]), float(jd_ut1[i]), float(jd_tt[i])
    )
    s1 = transform.coarse_eci_to_geodetic_iau76_shortnut(
        float(r_eci[i, 0]), float(r_eci[i, 1]), float(r_eci[i, 2]), float(jd_ut1[i]), float(jd_tt[i])
    )
    np.testing.assert_allclose(s0, s1, atol=0.0, rtol=0.0)

    d0 = coarse_eci_to_geodetic_deg(
        float(r_eci[i, 0]), float(r_eci[i, 1]), float(r_eci[i, 2]), float(jd_ut1[i]), float(jd_tt[i])
    )
    d1 = transform.coarse_eci_to_geodetic_deg_iau76_shortnut(
        float(r_eci[i, 0]), float(r_eci[i, 1]), float(r_eci[i, 2]), float(jd_ut1[i]), float(jd_tt[i])
    )
    np.testing.assert_allclose(d0, d1, atol=0.0, rtol=0.0)

    vd0 = coarse_eci_to_geodetic_vec_deg(r_eci, jd_ut1, jd_tt)
    vd1 = transform.coarse_eci_to_geodetic_vec_deg_iau76_shortnut(r_eci, jd_ut1, jd_tt)
    np.testing.assert_allclose(vd0[0], vd1[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vd0[1], vd1[1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(vd0[2], vd1[2], atol=0.0, rtol=0.0)
