from __future__ import annotations

import math

import numpy as np
import pytest

import nebula.transform as transform
from nebula.transform._aer2ecef import aer2ecef, aer2ecef_vec_aer, aer2ecef_vec_aer3
from nebula.transform._aer2ecef import aer2enu
from nebula.transform._aer2geodetic import (
    aer2geodetic,
    aer2geodetic_vec_aer,
    aer2geodetic_vec_aer3,
)
from nebula.transform._ecef2aer import ecef2aer, ecef2aer_vec_xyz, enu2aer
from nebula.transform._ecef2enu import (
    ecef2enu,
    ecef2enu_delta,
    ecef2enu_vec_ecef,
    ecef2enu_vec_xyz,
    enu_basis_from_ecef_xyz,
)
from nebula.transform._ecef2geodetic import (
    WGS84_A,
    _wrap_lon_pi,
    ecef2geodetic,
    ecef2geodetic_vec_ecef,
    ecef2geodetic_vec_xyz,
)
from nebula.transform._enu2ecef import (
    enu2ecef,
    enu2ecef_delta,
    enu2ecef_vec_enu,
    enu2ecef_vec_enu3,
)
from nebula.transform._enu2geodetic import (
    _enu2uvw,
    enu2geodetic,
    enu2geodetic_vec_enu,
    enu2geodetic_vec_enu3,
)
from nebula.transform._geodetic2aer import geodetic2aer, geodetic2aer_vec_llh
from nebula.transform._geodetic2ecef import (
    WGS84_E2,
    geodetic2ecef,
    geodetic2ecef_vec_lla,
    geodetic2ecef_vec_llh,
)
from nebula.transform._geodetic2enu import (
    _uvw2enu,
    enu_basis_from_latlon,
    geodetic2enu,
    geodetic2enu_vec_lla,
    geodetic2enu_vec_llh,
)


def _wrap_2pi(a):
    return a % (2.0 * np.pi)


def _az_diff(a, b):
    d = _wrap_2pi(a - b)
    return np.where(d > np.pi, d - 2.0 * np.pi, d)


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _ang_diff(a, b):
    return _wrap_pi(a - b)


def test_aer2ecef() -> None:
    rng = np.random.default_rng(12)

    _ = aer2ecef(0.1, 0.2, 100.0, 0.3, -1.0, 10.0)
    _ = ecef2aer(1.0, 2.0, 3.0, 0.3, -1.0, 10.0)

    lat0 = np.deg2rad(45.0)
    lon0 = np.deg2rad(-90.0)
    h0 = 200.0

    n = 5000
    az = rng.random(n).astype(np.float64) * 2.0 * np.pi
    el = ((rng.random(n) * 1.2 - 0.2) * (np.pi / 2.0)).astype(np.float64)
    sr = (rng.random(n) * 2.0e6).astype(np.float64)

    x, y, z = aer2ecef_vec_aer(az, el, sr, lat0, lon0, h0)
    az2, el2, sr2 = ecef2aer_vec_xyz(x, y, z, lat0, lon0, h0)

    assert np.max(np.abs(_az_diff(az2, az))) < 2e-10
    assert np.max(np.abs(el2 - el)) < 2e-10
    assert np.max(np.abs(sr2 - sr)) < 2e-3

    aer3 = np.column_stack((az, el, sr)).astype(np.float64)
    r = aer2ecef_vec_aer3(aer3, lat0, lon0, h0)
    assert np.max(np.abs(r[:, 0] - x)) < 1e-9
    assert np.max(np.abs(r[:, 1] - y)) < 1e-9
    assert np.max(np.abs(r[:, 2] - z)) < 1e-9


def test_aer2geodetic() -> None:
    rng = np.random.default_rng(13)

    _ = aer2geodetic(0.1, 0.2, 1000.0, 0.3, -1.0, 10.0)
    _ = geodetic2aer(0.1, 0.2, 10.0, 0.3, -1.0, 10.0)

    lat0 = np.deg2rad(10.0)
    lon0 = np.deg2rad(120.0)
    h0 = 20.0

    n = 4000
    az = (rng.random(n) * 2.0 * np.pi).astype(np.float64)
    el = ((rng.random(n) * 1.0 - 0.2) * (np.pi / 2.0)).astype(np.float64)
    sr = (rng.random(n) * 1.5e6).astype(np.float64)

    lat, lon, h = aer2geodetic_vec_aer(az, el, sr, lat0, lon0, h0)
    az2, el2, sr2 = geodetic2aer_vec_llh(lat, lon, h, lat0, lon0, h0)

    assert np.max(np.abs(_az_diff(az2, az))) < 5e-8
    assert np.max(np.abs(el2 - el)) < 5e-8
    assert np.max(np.abs(sr2 - sr)) < 2e-2

    aer3 = np.column_stack((az, el, sr)).astype(np.float64)
    lat3, lon3, h3 = aer2geodetic_vec_aer3(aer3, lat0, lon0, h0)

    assert np.max(np.abs(lat3 - lat)) < 1e-12
    assert np.max(np.abs(_az_diff(lon3, lon))) < 1e-12
    assert np.max(np.abs(h3 - h)) < 1e-6


def test_ecef2aer() -> None:
    rng = np.random.default_rng(11)

    _ = ecef2aer(1.0, 2.0, 3.0, 0.2, -1.0, 10.0)

    lat0 = np.deg2rad(40.0)
    lon0 = np.deg2rad(15.0)
    h0 = 120.0

    x0, y0, z0 = enu2ecef_vec_enu(
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        lat0,
        lon0,
        h0,
    )
    _, _, sr = ecef2aer(float(x0[0]), float(y0[0]), float(z0[0]), lat0, lon0, h0)
    assert abs(sr) < 1e-9

    n = 5000
    e = (rng.standard_normal(n) * 15000.0).astype(np.float64)
    n_comp = (rng.standard_normal(n) * 15000.0).astype(np.float64)
    u = (rng.standard_normal(n) * 800.0).astype(np.float64)

    x, y, z = enu2ecef_vec_enu(e, n_comp, u, lat0, lon0, h0)
    az1, el1, sr1 = ecef2aer_vec_xyz(x, y, z, lat0, lon0, h0)

    e2, n2, u2 = ecef2enu_vec_xyz(x, y, z, lat0, lon0, h0)
    az2 = np.empty(n, dtype=np.float64)
    el2 = np.empty(n, dtype=np.float64)
    sr2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, e_, r = enu2aer(float(e2[i]), float(n2[i]), float(u2[i]))
        az2[i] = a
        el2[i] = e_
        sr2[i] = r

    assert np.max(np.abs(_az_diff(az1, az2))) < 5e-12
    assert np.max(np.abs(el1 - el2)) < 5e-12
    assert np.max(np.abs(sr1 - sr2)) < 5e-6


def test_ecef2enu() -> None:
    rng = np.random.default_rng(4)

    _ = ecef2enu(1.0, 2.0, 3.0, 0.4, 1.0, 50.0)
    _ = enu2ecef(1.0, 2.0, 3.0, 0.4, 1.0, 50.0)

    lat0 = np.deg2rad(10.0)
    lon0 = np.deg2rad(-120.0)
    h0 = 500.0

    x0, y0, z0 = geodetic2ecef(lat0, lon0, h0)
    e, n_comp, u = ecef2enu(x0, y0, z0, lat0, lon0, h0)
    assert abs(e) < 1e-9 and abs(n_comp) < 1e-9 and abs(u) < 1e-9

    n = 5000
    e0 = (rng.standard_normal(n) * 20000.0).astype(np.float64)
    n0 = (rng.standard_normal(n) * 20000.0).astype(np.float64)
    u0 = (rng.standard_normal(n) * 1000.0).astype(np.float64)

    x, y, z = enu2ecef_vec_enu(e0, n0, u0, lat0, lon0, h0)
    e1, n1, u1 = ecef2enu_vec_xyz(x, y, z, lat0, lon0, h0)

    assert np.max(np.abs(e1 - e0)) < 2e-6
    assert np.max(np.abs(n1 - n0)) < 2e-6
    assert np.max(np.abs(u1 - u0)) < 2e-6

    r = np.column_stack((x, y, z)).astype(np.float64)
    e2, n2, u2 = ecef2enu_vec_ecef(r, lat0, lon0, h0)
    assert np.max(np.abs(e2 - e0)) < 2e-6
    assert np.max(np.abs(n2 - n0)) < 2e-6
    assert np.max(np.abs(u2 - u0)) < 2e-6

    enu = np.column_stack((e0, n0, u0)).astype(np.float64)
    r2 = enu2ecef_vec_enu3(enu, lat0, lon0, h0)
    assert np.max(np.abs(r2[:, 0] - x)) < 1e-9
    assert np.max(np.abs(r2[:, 1] - y)) < 1e-9
    assert np.max(np.abs(r2[:, 2] - z)) < 1e-9


def test_ecef2geodetic() -> None:
    rng = np.random.default_rng(0)

    def _geodetic2ecef_ref(lat_rad, lon_rad, h_m):
        sl = np.sin(lat_rad)
        cl = np.cos(lat_rad)
        so = np.sin(lon_rad)
        co = np.cos(lon_rad)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        x = (n + h_m) * cl * co
        y = (n + h_m) * cl * so
        z = (n * (1.0 - WGS84_E2) + h_m) * sl
        return x, y, z

    _ = ecef2geodetic(1.0, 2.0, 3.0)
    xw = np.array([1.0, 2.0], dtype=np.float64)
    yw = np.array([1.0, 2.0], dtype=np.float64)
    zw = np.array([1.0, 2.0], dtype=np.float64)
    _ = ecef2geodetic_vec_xyz(xw, yw, zw)
    _ = ecef2geodetic_vec_ecef(np.column_stack((xw, yw, zw)))

    lat, lon, h = ecef2geodetic(0.0, 0.0, 0.0)
    assert abs(lat) < 1e-15
    assert abs(lon) < 1e-15
    assert abs(h + WGS84_A) < 1e-9

    lon0 = 1.234
    h0 = 0.0

    lat0 = 0.5 * np.pi
    x, y, z = _geodetic2ecef_ref(lat0, lon0, h0)
    lat, lon, h = ecef2geodetic(float(x), float(y), float(z))
    assert abs(lat - lat0) < 1e-12
    assert abs(lon) < 1e-12
    assert abs(h) < 1e-6

    lat0 = -0.5 * np.pi
    x, y, z = _geodetic2ecef_ref(lat0, lon0, h0)
    lat, lon, h = ecef2geodetic(float(x), float(y), float(z))
    assert abs(lat - lat0) < 1e-12
    assert abs(lon) < 1e-12
    assert abs(h) < 1e-6

    lat, lon, _ = ecef2geodetic(-WGS84_A, 0.0, 0.0)
    assert abs(lat) < 1e-12
    assert abs(lon + np.pi) < 1e-12

    n = 2000
    lat_true = (rng.random(n) * 2.0 - 1.0) * np.deg2rad(85.0)
    lon_true = (rng.random(n) * 2.0 - 1.0) * np.pi
    h_true = (rng.random(n) * 1.2e6) - 1.0e3

    x, y, z = _geodetic2ecef_ref(lat_true, lon_true, h_true)

    idx = rng.choice(n, size=50, replace=False)
    for i in idx:
        la, lo, hi = ecef2geodetic(float(x[i]), float(y[i]), float(z[i]))
        assert abs(la - float(lat_true[i])) < 5e-8
        assert abs(_ang_diff(lo, float(_wrap_pi(lon_true[i])))) < 5e-8
        assert abs(hi - float(h_true[i])) < 5e-2

    lat_v, lon_v, h_v = ecef2geodetic_vec_xyz(
        x.astype(np.float64),
        y.astype(np.float64),
        z.astype(np.float64),
    )
    assert np.max(np.abs(lat_v - lat_true)) < 5e-8
    assert np.max(np.abs(_ang_diff(lon_v, _wrap_pi(lon_true)))) < 5e-8
    assert np.max(np.abs(h_v - h_true)) < 5e-2

    r = np.column_stack((x, y, z)).astype(np.float64)
    lat_m, lon_m, h_m = ecef2geodetic_vec_ecef(r)
    assert np.max(np.abs(lat_m - lat_v)) == 0.0
    assert np.max(np.abs(_ang_diff(lon_m, lon_v))) == 0.0
    assert np.max(np.abs(h_m - h_v)) == 0.0


def test_enu2ecef() -> None:
    rng = np.random.default_rng(3)

    _ = enu2ecef(1.0, 2.0, 3.0, 0.3, -1.2, 10.0)
    _ = ecef2enu(1.0, 2.0, 3.0, 0.3, -1.2, 10.0)

    lat0 = np.deg2rad(35.0)
    lon0 = np.deg2rad(10.0)
    h0 = 123.0

    x0, y0, z0 = geodetic2ecef(lat0, lon0, h0)
    x, y, z = enu2ecef(0.0, 0.0, 0.0, lat0, lon0, h0)
    assert abs(x - x0) < 1e-12 and abs(y - y0) < 1e-12 and abs(z - z0) < 1e-12

    n = 4000
    e = (rng.standard_normal(n) * 10000.0).astype(np.float64)
    n_comp = (rng.standard_normal(n) * 10000.0).astype(np.float64)
    u = (rng.standard_normal(n) * 500.0).astype(np.float64)

    x, y, z = enu2ecef_vec_enu(e, n_comp, u, lat0, lon0, h0)
    e2, n2, u2 = ecef2enu_vec_xyz(x, y, z, lat0, lon0, h0)
    assert np.max(np.abs(e2 - e)) < 2e-6
    assert np.max(np.abs(n2 - n_comp)) < 2e-6
    assert np.max(np.abs(u2 - u)) < 2e-6

    enu = np.column_stack((e, n_comp, u)).astype(np.float64)
    r = enu2ecef_vec_enu3(enu, lat0, lon0, h0)
    e3, n3, u3 = ecef2enu_vec_ecef(r, lat0, lon0, h0)
    assert np.max(np.abs(e3 - e)) < 2e-6
    assert np.max(np.abs(n3 - n_comp)) < 2e-6
    assert np.max(np.abs(u3 - u)) < 2e-6


def test_enu2geodetic() -> None:
    rng = np.random.default_rng(2)

    _ = enu2geodetic(1.0, 2.0, 3.0, 0.2, -1.0, 10.0)
    _ = geodetic2enu(0.2, -1.0, 10.0, 0.2, -1.0, 10.0)

    lat0 = np.deg2rad(-20.0)
    lon0 = np.deg2rad(110.0)
    h0 = 50.0

    la, lo, hi = enu2geodetic(0.0, 0.0, 0.0, lat0, lon0, h0)
    assert abs(la - lat0) < 1e-12
    assert abs(_ang_diff(lo, lon0)) < 1e-12
    assert abs(hi - h0) < 1e-6

    n = 4000
    e = (rng.standard_normal(n) * 5000.0).astype(np.float64)
    n_comp = (rng.standard_normal(n) * 5000.0).astype(np.float64)
    u = (rng.standard_normal(n) * 200.0).astype(np.float64)

    lat, lon, h = enu2geodetic_vec_enu(e, n_comp, u, lat0, lon0, h0)
    e2, n2, u2 = geodetic2enu_vec_llh(lat, lon, h, lat0, lon0, h0)

    assert np.max(np.abs(e2 - e)) < 2e-3
    assert np.max(np.abs(n2 - n_comp)) < 2e-3
    assert np.max(np.abs(u2 - u)) < 2e-3

    enu = np.column_stack((e, n_comp, u)).astype(np.float64)
    lat2, lon2, h2 = enu2geodetic_vec_enu3(enu, lat0, lon0, h0)
    assert np.max(np.abs(lat2 - lat)) == 0.0
    assert np.max(np.abs(_ang_diff(lon2, lon))) == 0.0
    assert np.max(np.abs(h2 - h)) == 0.0

    lla = np.column_stack((lat, lon, h)).astype(np.float64)
    enu2 = geodetic2enu_vec_lla(lla, lat0, lon0, h0)
    assert np.max(np.abs(enu2[:, 0] - e)) < 2e-3
    assert np.max(np.abs(enu2[:, 1] - n_comp)) < 2e-3
    assert np.max(np.abs(enu2[:, 2] - u)) < 2e-3


def test_geodetic2aer() -> None:
    rng = np.random.default_rng(10)

    _ = geodetic2aer(0.1, 0.2, 10.0, 0.0, 0.0, 0.0)
    _ = ecef2aer(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)

    lat0 = np.deg2rad(25.0)
    lon0 = np.deg2rad(-70.0)
    h0 = 50.0

    az, el, sr = geodetic2aer(lat0, lon0, h0, lat0, lon0, h0)
    assert abs(sr) < 1e-9
    assert abs(az) < 1e-12
    assert abs(el) < 1e-12

    n = 2000
    dlat = rng.standard_normal(n) * 2e-3
    dlon = rng.standard_normal(n) * 2e-3
    dh = rng.standard_normal(n) * 300.0

    lat = (lat0 + dlat).astype(np.float64)
    lon = (lon0 + dlon).astype(np.float64)
    h = (h0 + dh).astype(np.float64)

    az1, el1, sr1 = geodetic2aer_vec_llh(lat, lon, h, lat0, lon0, h0)

    az2 = np.empty(n, dtype=np.float64)
    el2 = np.empty(n, dtype=np.float64)
    sr2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y, z = geodetic2ecef(float(lat[i]), float(lon[i]), float(h[i]))
        a, e_, r = ecef2aer(x, y, z, lat0, lon0, h0)
        az2[i] = a
        el2[i] = e_
        sr2[i] = r

    assert np.max(np.abs(_az_diff(az1, az2))) < 5e-12
    assert np.max(np.abs(el1 - el2)) < 5e-12
    assert np.max(np.abs(sr1 - sr2)) < 5e-6


def test_geodetic2ecef() -> None:
    rng = np.random.default_rng(0)

    _ = geodetic2ecef(0.1, -0.2, 100.0)
    latw = np.array([0.1, 0.2], dtype=np.float64)
    lonw = np.array([-0.2, 0.3], dtype=np.float64)
    hw = np.array([0.0, 1000.0], dtype=np.float64)
    _ = geodetic2ecef_vec_llh(latw, lonw, hw)
    _ = geodetic2ecef_vec_lla(np.column_stack((latw, lonw, hw)).astype(np.float64))
    _ = ecef2geodetic(1.0, 2.0, 3.0)

    x, y, z = geodetic2ecef(0.0, 0.0, 0.0)
    assert abs(x - WGS84_A) < 1e-9
    assert abs(y) < 1e-12
    assert abs(z) < 1e-12

    x, y, z = geodetic2ecef(0.0, 0.5 * np.pi, 0.0)
    assert abs(x) < 1e-6
    assert abs(y - WGS84_A) < 1e-6
    assert abs(z) < 1e-12

    b = WGS84_A * math.sqrt(1.0 - WGS84_E2)
    x, y, z = geodetic2ecef(0.5 * np.pi, 0.0, 0.0)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z - b) < 1e-6

    n = 5000
    lat_true = (rng.random(n) * 2.0 - 1.0) * np.deg2rad(89.0)
    lon_true = (rng.random(n) * 2.0 - 1.0) * np.pi
    h_true = (rng.random(n) * 2.0e6) - 1.0e3

    x, y, z = geodetic2ecef_vec_llh(
        lat_true.astype(np.float64),
        lon_true.astype(np.float64),
        h_true.astype(np.float64),
    )

    idx = rng.choice(n, size=100, replace=False)
    for i in idx:
        la, lo, hi = ecef2geodetic(float(x[i]), float(y[i]), float(z[i]))
        assert abs(la - float(lat_true[i])) < 2e-7
        assert abs(_ang_diff(lo, float(_wrap_pi(lon_true[i])))) < 2e-7
        assert abs(hi - float(h_true[i])) < 5e-2

    lat_v, lon_v, h_v = ecef2geodetic_vec_xyz(x, y, z)
    assert np.max(np.abs(lat_v - lat_true)) < 2e-7
    assert np.max(np.abs(_ang_diff(lon_v, _wrap_pi(lon_true)))) < 2e-7
    assert np.max(np.abs(h_v - h_true)) < 5e-2

    r = np.column_stack((x, y, z)).astype(np.float64)
    lat_m, lon_m, h_m = ecef2geodetic_vec_ecef(r)
    assert np.max(np.abs(lat_m - lat_v)) == 0.0
    assert np.max(np.abs(_ang_diff(lon_m, lon_v))) == 0.0
    assert np.max(np.abs(h_m - h_v)) == 0.0

    lla = np.column_stack((lat_true, lon_true, h_true)).astype(np.float64)
    r2 = geodetic2ecef_vec_lla(lla)
    r_ref = np.column_stack((x, y, z))
    assert np.array_equal(r2, r_ref)


def test_geodetic2enu() -> None:
    rng = np.random.default_rng(1)

    _ = geodetic2enu(0.1, 0.2, 10.0, 0.0, 0.0, 0.0)
    _ = enu2geodetic(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)

    lat0 = np.deg2rad(30.0)
    lon0 = np.deg2rad(-80.0)
    h0 = 250.0

    e, n_comp, u = geodetic2enu(lat0, lon0, h0, lat0, lon0, h0)
    assert abs(e) < 1e-9 and abs(n_comp) < 1e-9 and abs(u) < 1e-9

    n = 3000
    dlat = rng.standard_normal(n) * 1e-3
    dlon = rng.standard_normal(n) * 1e-3
    dh = rng.standard_normal(n) * 200.0

    lat = (lat0 + dlat).astype(np.float64)
    lon = (lon0 + dlon).astype(np.float64)
    h = (h0 + dh).astype(np.float64)

    e, n_comp, u = geodetic2enu_vec_llh(lat, lon, h, lat0, lon0, h0)
    lat2, lon2, h2 = enu2geodetic_vec_enu(e, n_comp, u, lat0, lon0, h0)

    assert np.max(np.abs(lat2 - lat)) < 5e-10
    assert np.max(np.abs(_ang_diff(lon2, lon))) < 5e-10
    assert np.max(np.abs(h2 - h)) < 2e-3

    lla = np.column_stack((lat, lon, h)).astype(np.float64)
    enu = geodetic2enu_vec_lla(lla, lat0, lon0, h0)
    lat3, lon3, h3 = enu2geodetic_vec_enu3(enu, lat0, lon0, h0)
    assert np.max(np.abs(lat3 - lat)) < 5e-10
    assert np.max(np.abs(_ang_diff(lon3, lon))) < 5e-10
    assert np.max(np.abs(h3 - h)) < 2e-3


def test_aer2enu_cardinal_and_zero_cases() -> None:
    e, n, u = aer2enu(0.0, 0.0, 100.0)
    assert abs(e) < 1e-12
    assert abs(n - 100.0) < 1e-12
    assert abs(u) < 1e-12

    e, n, u = aer2enu(0.5 * np.pi, 0.0, 100.0)
    assert abs(e - 100.0) < 1e-12
    assert abs(n) < 1e-12
    assert abs(u) < 1e-12

    e, n, u = aer2enu(1.234, 0.5 * np.pi, 250.0)
    assert abs(e) < 1e-12
    assert abs(n) < 1e-12
    assert abs(u - 250.0) < 1e-12

    e, n, u = aer2enu(1.0, -0.3, 0.0)
    assert e == 0.0 and n == 0.0 and u == 0.0


def test_delta_rotations_are_inverses() -> None:
    lat0 = np.deg2rad(37.2)
    lon0 = np.deg2rad(-122.1)
    dx, dy, dz = 1234.5, -678.9, 42.0

    e, n, u = ecef2enu_delta(dx, dy, dz, lat0, lon0)
    dx2, dy2, dz2 = enu2ecef_delta(e, n, u, lat0, lon0)

    np.testing.assert_allclose([dx2, dy2, dz2], [dx, dy, dz], atol=1e-12, rtol=0.0)


def test_private_uvw_helpers_are_inverses() -> None:
    lat0 = np.deg2rad(-21.5)
    lon0 = np.deg2rad(88.0)
    dx, dy, dz = -321.0, 654.0, 987.0

    e, n, u = _uvw2enu(dx, dy, dz, lat0, lon0)
    dx2, dy2, dz2 = _enu2uvw(e, n, u, lat0, lon0)

    np.testing.assert_allclose([dx2, dy2, dz2], [dx, dy, dz], atol=1e-12, rtol=0.0)


def test_enu_basis_functions_match_and_are_orthonormal() -> None:
    lat0 = np.deg2rad(12.0)
    lon0 = np.deg2rad(-44.0)
    x0, y0, z0 = geodetic2ecef(lat0, lon0, 500.0)

    r_latlon = enu_basis_from_latlon(lat0, lon0)
    r_xyz = enu_basis_from_ecef_xyz(x0, y0, z0)

    np.testing.assert_allclose(r_latlon, r_xyz, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(r_latlon.T @ r_latlon, np.eye(3), atol=1e-12, rtol=0.0)
    assert abs(np.linalg.det(r_latlon) - 1.0) < 1e-12


def test_wrap_lon_pi_properties() -> None:
    vals = np.array(
        [-4.0 * np.pi, -3.0 * np.pi, -np.pi, -0.1, 0.0, 0.1, np.pi, 3.0 * np.pi]
    )
    wrapped = np.array([_wrap_lon_pi(float(v)) for v in vals])

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert abs(_wrap_lon_pi(np.pi) + np.pi) < 1e-12
    assert abs(_wrap_lon_pi(-np.pi) + np.pi) < 1e-12
    assert abs(_wrap_lon_pi(0.0)) < 1e-12


def test_vector_input_validation_errors() -> None:
    with pytest.raises(ValueError):
        aer2ecef_vec_aer(
            np.array([0.0]), np.array([0.0, 1.0]), np.array([1.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        aer2ecef_vec_aer3(np.zeros((2, 2), dtype=np.float64), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        aer2geodetic_vec_aer(
            np.array([0.0]), np.array([0.0]), np.array([1.0, 2.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        aer2geodetic_vec_aer3(np.zeros((2, 4), dtype=np.float64), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        ecef2aer_vec_xyz(
            np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]), 0.0, 0.0, 0.0
        )

    with pytest.raises(ValueError):
        ecef2enu_vec_xyz(
            np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        ecef2enu_vec_ecef(np.zeros((2, 2), dtype=np.float64), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        ecef2geodetic_vec_xyz(np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        ecef2geodetic_vec_ecef(np.zeros((3, 2), dtype=np.float64))

    with pytest.raises(ValueError):
        enu2ecef_vec_enu(
            np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        enu2ecef_vec_enu3(np.zeros((3, 2), dtype=np.float64), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        enu2geodetic_vec_enu(
            np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        enu2geodetic_vec_enu3(np.zeros((4, 2), dtype=np.float64), 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        geodetic2aer_vec_llh(
            np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]), 0.0, 0.0, 0.0
        )

    with pytest.raises(ValueError):
        geodetic2ecef_vec_llh(np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        geodetic2ecef_vec_lla(np.zeros((2, 2), dtype=np.float64))

    with pytest.raises(ValueError):
        geodetic2enu_vec_llh(
            np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]), 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError):
        geodetic2enu_vec_lla(np.zeros((2, 2), dtype=np.float64), 0.0, 0.0, 0.0)


def test_constants_module_consistency() -> None:
    c = transform
    assert c.WGS84_A > c.WGS84_B
    assert c.WGS84_A2 == c.WGS84_A * c.WGS84_A
    assert c.WGS84_B2 == c.WGS84_B * c.WGS84_B
    assert abs(c.WGS84_B2_OVER_A2 - (c.WGS84_B2 / c.WGS84_A2)) < 1e-15
    assert abs(c.WGS84_E2 - (1.0 - c.WGS84_B2_OVER_A2)) < 1e-15
    assert abs(c.WGS84_EP2 - ((c.WGS84_A2 - c.WGS84_B2) / c.WGS84_B2)) < 1e-15
    assert abs(c.DEG2RAD * c.RAD2DEG - 1.0) < 1e-15
    assert abs(c.TWO_PI - 2.0 * c.PI) < 1e-15
    assert abs(c.HALF_PI - 0.5 * c.PI) < 1e-15


def test_transform_init_reexports_work() -> None:
    x, y, z = transform.geodetic2ecef(0.0, 0.0, 0.0)
    assert abs(x - transform.WGS84_A) < 1e-9
    assert abs(y) < 1e-12
    assert abs(z) < 1e-12
