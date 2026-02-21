# tdoa.py

import math
import numpy as np
from numba import njit


@njit(cache=True)
def tdoa_seconds(
    obs1_pos: np.ndarray,
    obs2_pos: np.ndarray,
    target_pos: np.ndarray,
    C: float = 299_792_458.0,
) -> float:
    """
    Evaluate the two-observer TDOA measurement model.

    Model (non-relativistic, straight-line propagation):
        τ = (||r_t - r_1|| - ||r_t - r_2||) / C

    where:
      - τ is the time-difference-of-arrival (TDOA) in seconds
      - r_t is the target position (ECEF/ECI/etc.; any consistent Cartesian frame) [m]
      - r_1, r_2 are the observer positions in the same frame [m]
      - C is the propagation speed (typically the speed of light) [m/s]
      - ||·|| is the Euclidean norm

    Sign convention:
      - τ > 0 means the target is farther from observer 1 than observer 2 (range1 > range2).

    Notes:
      - This is purely geometric; it does not include clock biases, atmospheric delays,
        Earth rotation corrections, or light-time iteration.
      - If the target position coincides exactly with either observer (range == 0),
        this function returns 0.0 to avoid division-by-zero; such geometry is invalid
        for estimation and should be avoided upstream.
      - Implemented for high-throughput @njit use: minimal temporaries and no allocations.

    Parameters
    ----------
    obs1_pos : np.ndarray
        Observer 1 position [x, y, z] in meters. Shape (3,), dtype float64.
    obs2_pos : np.ndarray
        Observer 2 position [x, y, z] in meters. Shape (3,), dtype float64.
    target_pos : np.ndarray
        Target position [x, y, z] in meters. Shape (3,), dtype float64.
    C : float, optional
        Propagation speed in m/s. Default is 299_792_458.0.

    Returns
    -------
    float
        TDOA τ in seconds.
    """
    dx1 = target_pos[0] - obs1_pos[0]
    dy1 = target_pos[1] - obs1_pos[1]
    dz1 = target_pos[2] - obs1_pos[2]

    dx2 = target_pos[0] - obs2_pos[0]
    dy2 = target_pos[1] - obs2_pos[1]
    dz2 = target_pos[2] - obs2_pos[2]

    r1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
    r2 = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)

    if r1 == 0.0 or r2 == 0.0:
        return 0.0

    return (r1 - r2) / C


@njit(cache=True)
def tdoa_jacobian_xyz(
    obs1_pos: np.ndarray,
    obs2_pos: np.ndarray,
    target_pos: np.ndarray,
    C: float = 299_792_458.0,
):
    """
    Compute the TDOA Jacobian w.r.t. target position (x, y, z) for two observers.

    This implements the standard small-velocity (non-relativistic) geometric TDOA model:

        τ = (||r_t - r_1|| - ||r_t - r_2||) / C

    where:
        r_t : target position [m]
        r_i : observer i position [m]
        C   : propagation speed (speed of light) [m/s]
        ||·|| : Euclidean norm

    Let:
        p_i = r_t - r_i                      (LOS vector from observer i to target) [m]
        r_i = ||p_i||                        (range from observer i to target)      [m]
        u_i = p_i / r_i                      (LOS unit vector)                      [-]

    The Jacobian of τ with respect to r_t is:

        ∂τ/∂r_t = (u_1 - u_2) / C

    Notes:
      - This model has no dependence on target velocity.
      - If the target position coincides exactly with either observer position (r_i = 0),
        this function returns zeros to avoid division-by-zero; such geometry is invalid
        for TDOA/FDOA estimation and should be avoided upstream.
      - Designed for @njit usage: returns a tuple of floats (no array allocation).

    Parameters
    ----------
    obs1_pos : np.ndarray
        Observer 1 position vector [x, y, z] in meters. Shape (3,).
    obs2_pos : np.ndarray
        Observer 2 position vector [x, y, z] in meters. Shape (3,).
    target_pos : np.ndarray
        Target position vector [x, y, z] in meters. Shape (3,).
    C : float, optional
        Propagation speed in m/s. Default is 299_792_458.0.

    Returns
    -------
    (dτ_dx, dτ_dy, dτ_dz) : tuple[float, float, float]
        Partial derivatives of TDOA τ [s] with respect to target position coordinates [m],
        with units of seconds per meter [s/m].
    """
    rho1x = target_pos[0] - obs1_pos[0]
    rho1y = target_pos[1] - obs1_pos[1]
    rho1z = target_pos[2] - obs1_pos[2]

    rho2x = target_pos[0] - obs2_pos[0]
    rho2y = target_pos[1] - obs2_pos[1]
    rho2z = target_pos[2] - obs2_pos[2]

    r1 = math.sqrt(rho1x * rho1x + rho1y * rho1y + rho1z * rho1z)
    r2 = math.sqrt(rho2x * rho2x + rho2y * rho2y + rho2z * rho2z)

    if r1 == 0.0 or r2 == 0.0:
        return 0.0, 0.0, 0.0

    inv_r1 = 1.0 / r1
    inv_r2 = 1.0 / r2

    u1x = rho1x * inv_r1
    u1y = rho1y * inv_r1
    u1z = rho1z * inv_r1

    u2x = rho2x * inv_r2
    u2y = rho2y * inv_r2
    u2z = rho2z * inv_r2

    inv_C = 1.0 / C
    return (u1x - u2x) * inv_C, (u1y - u2y) * inv_C, (u1z - u2z) * inv_C
