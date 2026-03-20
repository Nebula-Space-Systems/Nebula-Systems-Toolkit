from __future__ import annotations

import numpy as np

from nstk.localization.measurements.tdoa import tdoa_jacobian_xyz, tdoa_seconds


def _tdoa_seconds_py(obs1_pos, obs2_pos, target_pos, c=299_792_458.0):
    r1 = np.linalg.norm(target_pos - obs1_pos)
    r2 = np.linalg.norm(target_pos - obs2_pos)
    if r1 == 0.0 or r2 == 0.0:
        return 0.0
    return (r1 - r2) / c


def _fd_tdoa_grad_xyz(obs1_pos, obs2_pos, target_pos, h=1.0, c=299_792_458.0):
    g = np.zeros(3, dtype=np.float64)
    for i in range(3):
        tp_p = target_pos.copy()
        tp_m = target_pos.copy()
        tp_p[i] += h
        tp_m[i] -= h
        fp = tdoa_seconds(obs1_pos, obs2_pos, tp_p, c)
        fm = tdoa_seconds(obs1_pos, obs2_pos, tp_m, c)
        g[i] = (fp - fm) / (2.0 * h)
    return g


def test_tdoa_measurement_and_jacobian() -> None:
    np.random.seed(0)

    obs1_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    obs2_pos = np.array([2_000_000.0, 500_000.0, -300_000.0], dtype=np.float64)
    target_pos = np.array([1_200_000.0, -800_000.0, 900_000.0], dtype=np.float64)

    _ = tdoa_seconds(obs1_pos, obs2_pos, target_pos)
    _ = tdoa_jacobian_xyz(obs1_pos, obs2_pos, target_pos)

    tau_numba = tdoa_seconds(obs1_pos, obs2_pos, target_pos)
    tau_py = _tdoa_seconds_py(obs1_pos, obs2_pos, target_pos)
    assert abs(tau_numba - tau_py) < 1e-15

    h_pos = 1.0
    fd = _fd_tdoa_grad_xyz(obs1_pos, obs2_pos, target_pos, h=h_pos)
    an = np.array(tdoa_jacobian_xyz(obs1_pos, obs2_pos, target_pos), dtype=np.float64)
    assert np.allclose(an, fd, rtol=1e-7, atol=1e-12)


def test_tdoa_identical_observers_zero_signal_and_jacobian() -> None:
    target_pos = np.array([1_200_000.0, -800_000.0, 900_000.0], dtype=np.float64)
    obs_same = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)

    tau0 = tdoa_seconds(obs_same, obs_same, target_pos)
    g0 = np.array(tdoa_jacobian_xyz(obs_same, obs_same, target_pos), dtype=np.float64)

    assert abs(tau0) < 1e-15
    assert np.allclose(g0, 0.0, atol=1e-15)
