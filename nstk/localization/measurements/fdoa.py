# fdoa.py

import math
import numpy as np
from numba import njit


@njit(cache=True)
def fdoa_hz(
    obs1_pos: np.ndarray,
    obs1_vel: np.ndarray,
    obs2_pos: np.ndarray,
    obs2_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    frequency: float,
    C: float = 299_792_458.0,
) -> float:
    """
    Evaluate the two-observer FDOA measurement model.

    Model (small-velocity / non-relativistic Doppler, first order in v/C):
        φ = (f / C) * [ (v_t - v_1)·u_1 - (v_t - v_2)·u_2 ]

    where:
      - φ is the frequency-difference-of-arrival (FDOA) in Hz
      - f is the nominal carrier frequency in Hz
      - r_t, v_t are target position and velocity [m], [m/s]
      - r_i, v_i are observer i position and velocity [m], [m/s]
      - u_i is the line-of-sight (LOS) unit vector from observer i to target:
            u_i = (r_t - r_i) / ||r_t - r_i||
      - · is the dot product, so (v_t - v_i)·u_i is the relative radial speed [m/s]
      - C is the propagation speed [m/s]

    Interpretation / sign convention:
      - Each term (v_t - v_i)·u_i is positive when the target is receding from observer i
        along the LOS direction u_i.
      - φ is the difference in Doppler shift between observer 1 and observer 2 under the
        above convention.

    Notes:
      - This is a first-order model; it does not include relativistic Doppler, time dilation,
        light-time iteration, or oscillator/clock frequency errors.
      - If the target position coincides exactly with either observer (range == 0),
        this function returns 0.0 to avoid division-by-zero; such geometry is invalid
        for estimation and should be avoided upstream.
      - Implemented for high-throughput @njit use: minimal temporaries and no allocations.

    Parameters
    ----------
    obs1_pos : np.ndarray
        Observer 1 position [x, y, z] in meters. Shape (3,), dtype float64.
    obs1_vel : np.ndarray
        Observer 1 velocity [vx, vy, vz] in m/s. Shape (3,), dtype float64.
    obs2_pos : np.ndarray
        Observer 2 position [x, y, z] in meters. Shape (3,), dtype float64.
    obs2_vel : np.ndarray
        Observer 2 velocity [vx, vy, vz] in m/s. Shape (3,), dtype float64.
    target_pos : np.ndarray
        Target position [x, y, z] in meters. Shape (3,), dtype float64.
    target_vel : np.ndarray
        Target velocity [vx, vy, vz] in m/s. Shape (3,), dtype float64.
    frequency : float
        Nominal carrier frequency in Hz.
    C : float, optional
        Propagation speed in m/s. Default is 299_792_458.0.

    Returns
    -------
    float
        FDOA φ in Hz.
    """
    # LOS vectors and ranges
    rho1x = target_pos[0] - obs1_pos[0]
    rho1y = target_pos[1] - obs1_pos[1]
    rho1z = target_pos[2] - obs1_pos[2]

    rho2x = target_pos[0] - obs2_pos[0]
    rho2y = target_pos[1] - obs2_pos[1]
    rho2z = target_pos[2] - obs2_pos[2]

    r1 = math.sqrt(rho1x * rho1x + rho1y * rho1y + rho1z * rho1z)
    r2 = math.sqrt(rho2x * rho2x + rho2y * rho2y + rho2z * rho2z)

    if r1 == 0.0 or r2 == 0.0:
        return 0.0

    inv_r1 = 1.0 / r1
    inv_r2 = 1.0 / r2

    u1x = rho1x * inv_r1
    u1y = rho1y * inv_r1
    u1z = rho1z * inv_r1

    u2x = rho2x * inv_r2
    u2y = rho2y * inv_r2
    u2z = rho2z * inv_r2

    # Relative velocities
    vrel1x = target_vel[0] - obs1_vel[0]
    vrel1y = target_vel[1] - obs1_vel[1]
    vrel1z = target_vel[2] - obs1_vel[2]

    vrel2x = target_vel[0] - obs2_vel[0]
    vrel2y = target_vel[1] - obs2_vel[1]
    vrel2z = target_vel[2] - obs2_vel[2]

    # Radial relative speeds (v_rel · u)
    s1 = vrel1x * u1x + vrel1y * u1y + vrel1z * u1z
    s2 = vrel2x * u2x + vrel2y * u2y + vrel2z * u2z

    return (frequency / C) * (s1 - s2)


@njit(cache=True)
def fdoa_jacobian_xyz_vxyz(
    obs1_pos: np.ndarray,
    obs1_vel: np.ndarray,
    obs2_pos: np.ndarray,
    obs2_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    frequency: float,
    C: float = 299_792_458.0,
):
    """
    Compute the FDOA Jacobian w.r.t. target position and velocity for two observers.

    This implements a small-velocity (non-relativistic) Doppler-difference model:

        φ = (f / C) * [ (v_t - v_1) · u_1  -  (v_t - v_2) · u_2 ]

    where:
        φ   : FDOA (frequency difference of arrival) [Hz]
        f   : nominal carrier frequency [Hz]
        C   : propagation speed [m/s]
        r_t : target position [m]
        v_t : target velocity [m/s]
        r_i : observer i position [m]
        v_i : observer i velocity [m/s]
        u_i : LOS unit vector from observer i to target [-]
        ·   : dot product

    Definitions:
        ρ_i     = r_t - r_i                  (LOS vector)                     [m]
        R_i     = ||ρ_i||                    (range)                          [m]
        u_i     = ρ_i / R_i                  (LOS unit vector)                [-]
        v_rel_i = v_t - v_i                  (relative velocity)              [m/s]
        s_i     = v_rel_i · u_i              (radial relative speed)          [m/s]

    With these, φ = (f/C) * (s_1 - s_2).

    Velocity Jacobian:
        ∂φ/∂v_t = (f / C) * (u_1 - u_2)

    Position Jacobian:
        Since u_i depends on r_t, we use:
            du_i/dr_t = (I - u_i u_i^T) / R_i
        which yields:
            ∂s_i/∂r_t = (v_rel_i - (v_rel_i · u_i) u_i) / R_i
        Therefore:
            ∂φ/∂r_t = (f / C) * ( ∂s_1/∂r_t - ∂s_2/∂r_t )

    Notes:
      - If the target position coincides exactly with either observer position (R_i = 0),
        this function returns zeros to avoid division-by-zero; such geometry is invalid.
      - This is a first-order, non-relativistic model (no time dilation, light-time iteration,
        or full relativistic Doppler). It is typically sufficient when |v| << C.
      - Designed for @njit usage: returns a tuple of floats (no array allocation).

    Parameters
    ----------
    obs1_pos : np.ndarray
        Observer 1 position vector [x, y, z] in meters. Shape (3,).
    obs1_vel : np.ndarray
        Observer 1 velocity vector [vx, vy, vz] in m/s. Shape (3,).
    obs2_pos : np.ndarray
        Observer 2 position vector [x, y, z] in meters. Shape (3,).
    obs2_vel : np.ndarray
        Observer 2 velocity vector [vx, vy, vz] in m/s. Shape (3,).
    target_pos : np.ndarray
        Target position vector [x, y, z] in meters. Shape (3,).
    target_vel : np.ndarray
        Target velocity vector [vx, vy, vz] in m/s. Shape (3,).
    frequency : float
        Nominal carrier frequency in Hz.
    C : float, optional
        Propagation speed in m/s. Default is 299_792_458.0.

    Returns
    -------
    (dφ_dx, dφ_dy, dφ_dz, dφ_dvx, dφ_dvy, dφ_dvz) : tuple[float, float, float, float, float, float]
        Partial derivatives of FDOA φ [Hz] with respect to target position [m] and
        target velocity [m/s]. Units:
          - dφ/dx, dφ/dy, dφ/dz are [Hz/m]
          - dφ/dvx, dφ/dvy, dφ/dvz are [Hz/(m/s)] = [Hz·s/m]
    """
    # Geometry
    rho1x = target_pos[0] - obs1_pos[0]
    rho1y = target_pos[1] - obs1_pos[1]
    rho1z = target_pos[2] - obs1_pos[2]

    rho2x = target_pos[0] - obs2_pos[0]
    rho2y = target_pos[1] - obs2_pos[1]
    rho2z = target_pos[2] - obs2_pos[2]

    r1 = math.sqrt(rho1x * rho1x + rho1y * rho1y + rho1z * rho1z)
    r2 = math.sqrt(rho2x * rho2x + rho2y * rho2y + rho2z * rho2z)

    if r1 == 0.0 or r2 == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    inv_r1 = 1.0 / r1
    inv_r2 = 1.0 / r2

    u1x = rho1x * inv_r1
    u1y = rho1y * inv_r1
    u1z = rho1z * inv_r1

    u2x = rho2x * inv_r2
    u2y = rho2y * inv_r2
    u2z = rho2z * inv_r2

    # Relative velocities
    vrel1x = target_vel[0] - obs1_vel[0]
    vrel1y = target_vel[1] - obs1_vel[1]
    vrel1z = target_vel[2] - obs1_vel[2]

    vrel2x = target_vel[0] - obs2_vel[0]
    vrel2y = target_vel[1] - obs2_vel[1]
    vrel2z = target_vel[2] - obs2_vel[2]

    # Radial components s_i = v_rel_i · u_i
    s1 = vrel1x * u1x + vrel1y * u1y + vrel1z * u1z
    s2 = vrel2x * u2x + vrel2y * u2y + vrel2z * u2z

    # ∂s_i/∂r_t = (v_rel_i - s_i u_i) / r_i
    ds1dx = (vrel1x - s1 * u1x) * inv_r1
    ds1dy = (vrel1y - s1 * u1y) * inv_r1
    ds1dz = (vrel1z - s1 * u1z) * inv_r1

    ds2dx = (vrel2x - s2 * u2x) * inv_r2
    ds2dy = (vrel2y - s2 * u2y) * inv_r2
    ds2dz = (vrel2z - s2 * u2z) * inv_r2

    scale = frequency / C

    # ∂φ/∂pos
    dphidx = scale * (ds1dx - ds2dx)
    dphidy = scale * (ds1dy - ds2dy)
    dphidz = scale * (ds1dz - ds2dz)

    # ∂φ/∂vel_t
    dphidvx = scale * (u1x - u2x)
    dphidvy = scale * (u1y - u2y)
    dphidvz = scale * (u1z - u2z)

    return dphidx, dphidy, dphidz, dphidvx, dphidvy, dphidvz
