from __future__ import annotations

import numpy as np

from nebula.localization.aoa import aoa_az_el, aoa_jacobian_az_el_xyz


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _angle_diff(a, b):
    return _wrap_pi(a - b)


def _aoa_az_el_py(obs_pos, target_pos):
    x, y, z = target_pos - obs_pos
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x * x + y * y))
    return float(az), float(el)


def _fd_jacobian_az_el_xyz(obs_pos, target_pos, h=1.0):
    g = np.zeros(6, dtype=np.float64)

    for i in range(3):
        tp_p = target_pos.copy()
        tp_m = target_pos.copy()
        tp_p[i] += h
        tp_m[i] -= h

        az_p, el_p = aoa_az_el(obs_pos, tp_p)
        az_m, el_m = aoa_az_el(obs_pos, tp_m)

        daz = _angle_diff(az_p, az_m) / (2.0 * h)
        delv = (el_p - el_m) / (2.0 * h)

        g[i] = daz
        g[3 + i] = delv

    return g


def test_aoa_measurement_jacobian_and_invariances() -> None:
    np.random.seed(123)

    obs_pos = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0], dtype=np.float64)
    target_pos = np.array([1_200_000.0, 1_300_000.0, 3_800_000.0], dtype=np.float64)

    _ = aoa_az_el(obs_pos, target_pos)
    _ = aoa_jacobian_az_el_xyz(obs_pos, target_pos)

    az_numba, el_numba = aoa_az_el(obs_pos, target_pos)
    az_py, el_py = _aoa_az_el_py(obs_pos, target_pos)

    assert abs(_angle_diff(az_numba, az_py)) < 1e-15
    assert abs(el_numba - el_py) < 1e-15

    h = 1.0
    fd = _fd_jacobian_az_el_xyz(obs_pos, target_pos, h=h)
    an = np.array(aoa_jacobian_az_el_xyz(obs_pos, target_pos), dtype=np.float64)
    assert np.allclose(an, fd, rtol=1e-6, atol=1e-10)

    shift = np.array([9_000_000.0, -4_000_000.0, 2_000_000.0], dtype=np.float64)
    obs2 = obs_pos + shift
    tgt2 = target_pos + shift
    az2, el2 = aoa_az_el(obs2, tgt2)
    j2 = np.array(aoa_jacobian_az_el_xyz(obs2, tgt2), dtype=np.float64)
    assert abs(_angle_diff(az2, az_numba)) < 1e-15
    assert abs(el2 - el_numba) < 1e-15
    assert np.allclose(j2, an, rtol=0.0, atol=0.0)

    rho = target_pos - obs_pos
    target3 = obs_pos + 7.5 * rho
    az3, el3 = aoa_az_el(obs_pos, target3)
    assert abs(_angle_diff(az3, az_numba)) < 1e-15
    assert abs(el3 - el_numba) < 1e-15


def test_aoa_axis_sanity_cases() -> None:
    o = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    t = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    az, el = aoa_az_el(o, t)
    assert abs(_angle_diff(az, 0.0)) < 1e-15
    assert abs(el - 0.0) < 1e-15

    t = np.array([0.0, 10.0, 0.0], dtype=np.float64)
    az, el = aoa_az_el(o, t)
    assert abs(_angle_diff(az, np.pi / 2.0)) < 1e-15
    assert abs(el - 0.0) < 1e-15

    t = np.array([10.0, 10.0, 0.0], dtype=np.float64)
    az, el = aoa_az_el(o, t)
    assert abs(_angle_diff(az, np.pi / 4.0)) < 1e-15
    assert abs(el - 0.0) < 1e-15

    t = np.array([10.0, 0.0, 10.0], dtype=np.float64)
    az, el = aoa_az_el(o, t)
    assert abs(_angle_diff(az, 0.0)) < 1e-15
    assert abs(el - np.pi / 4.0) < 1e-15

