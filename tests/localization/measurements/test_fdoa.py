from __future__ import annotations

import numpy as np

from nstk.localization.measurements.fdoa import fdoa_hz, fdoa_jacobian_xyz_vxyz


def _fdoa_hz_py(
    obs1_pos,
    obs1_vel,
    obs2_pos,
    obs2_vel,
    target_pos,
    target_vel,
    frequency,
    c=299_792_458.0,
):
    rho1 = target_pos - obs1_pos
    rho2 = target_pos - obs2_pos
    r1 = np.linalg.norm(rho1)
    r2 = np.linalg.norm(rho2)
    if r1 == 0.0 or r2 == 0.0:
        return 0.0
    u1 = rho1 / r1
    u2 = rho2 / r2
    s1 = np.dot(target_vel - obs1_vel, u1)
    s2 = np.dot(target_vel - obs2_vel, u2)
    return (frequency / c) * (s1 - s2)


def _fd_fdoa_grad_xyz_vxyz(
    obs1_pos,
    obs1_vel,
    obs2_pos,
    obs2_vel,
    target_pos,
    target_vel,
    frequency,
    h_pos=1.0,
    h_vel=1e-2,
    c=299_792_458.0,
):
    g = np.zeros(6, dtype=np.float64)

    for i in range(3):
        tp_p = target_pos.copy()
        tp_m = target_pos.copy()
        tp_p[i] += h_pos
        tp_m[i] -= h_pos
        fp = fdoa_hz(
            obs1_pos, obs1_vel, obs2_pos, obs2_vel, tp_p, target_vel, frequency, c
        )
        fm = fdoa_hz(
            obs1_pos, obs1_vel, obs2_pos, obs2_vel, tp_m, target_vel, frequency, c
        )
        g[i] = (fp - fm) / (2.0 * h_pos)

    for i in range(3):
        tv_p = target_vel.copy()
        tv_m = target_vel.copy()
        tv_p[i] += h_vel
        tv_m[i] -= h_vel
        fp = fdoa_hz(
            obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, tv_p, frequency, c
        )
        fm = fdoa_hz(
            obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, tv_m, frequency, c
        )
        g[3 + i] = (fp - fm) / (2.0 * h_vel)

    return g


def test_fdoa_measurement_and_jacobian() -> None:
    np.random.seed(1)

    obs1_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    obs2_pos = np.array([1_500_000.0, -700_000.0, 200_000.0], dtype=np.float64)
    target_pos = np.array([900_000.0, 400_000.0, 1_100_000.0], dtype=np.float64)

    obs1_vel = np.array([0.0, 7500.0, 0.0], dtype=np.float64)
    obs2_vel = np.array([100.0, 7400.0, -50.0], dtype=np.float64)
    target_vel = np.array([200.0, -150.0, 80.0], dtype=np.float64)

    frequency = 1.0e9

    _ = fdoa_hz(
        obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, target_vel, frequency
    )
    _ = fdoa_jacobian_xyz_vxyz(
        obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, target_vel, frequency
    )

    phi_numba = fdoa_hz(
        obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, target_vel, frequency
    )
    phi_py = _fdoa_hz_py(
        obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, target_vel, frequency
    )
    assert abs(phi_numba - phi_py) < 1e-10

    fd = _fd_fdoa_grad_xyz_vxyz(
        obs1_pos,
        obs1_vel,
        obs2_pos,
        obs2_vel,
        target_pos,
        target_vel,
        frequency,
        h_pos=1.0,
        h_vel=1e-2,
    )
    an = np.array(
        fdoa_jacobian_xyz_vxyz(
            obs1_pos, obs1_vel, obs2_pos, obs2_vel, target_pos, target_vel, frequency
        ),
        dtype=np.float64,
    )
    assert np.allclose(an, fd, rtol=1e-6, atol=1e-6)


def test_fdoa_identical_observers_zero_signal_and_jacobian() -> None:
    target_pos = np.array([900_000.0, 400_000.0, 1_100_000.0], dtype=np.float64)
    target_vel = np.array([200.0, -150.0, 80.0], dtype=np.float64)
    frequency = 1.0e9

    obs_same_pos = np.array([10_000.0, -20_000.0, 30_000.0], dtype=np.float64)
    obs_same_vel = np.array([100.0, 200.0, -50.0], dtype=np.float64)

    phi0 = fdoa_hz(
        obs_same_pos,
        obs_same_vel,
        obs_same_pos,
        obs_same_vel,
        target_pos,
        target_vel,
        frequency,
    )
    j0 = np.array(
        fdoa_jacobian_xyz_vxyz(
            obs_same_pos,
            obs_same_vel,
            obs_same_pos,
            obs_same_vel,
            target_pos,
            target_vel,
            frequency,
        ),
        dtype=np.float64,
    )

    assert abs(phi0) < 1e-12
    assert np.allclose(j0, 0.0, atol=1e-12)
