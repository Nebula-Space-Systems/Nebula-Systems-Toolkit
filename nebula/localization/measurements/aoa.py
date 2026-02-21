# aoa.py

import math
import numpy as np
from numba import njit


@njit(cache=True)
def aoa_az_el(obs_pos: np.ndarray, target_pos: np.ndarray):
    """
    Compute angle-of-arrival as (azimuth, elevation) from an observer to a target.

    This function interprets AOA in a *global Cartesian frame* (e.g., ECEF), using the
    line-of-sight (LOS) vector expressed in that frame:

        ρ = r_t - r_o = [x, y, z]^T

    The returned angles follow the standard spherical convention with respect to the
    global axes:

        az = atan2(y, x)                  (radians, wrapped to (-π, π])
        el = atan2(z, sqrt(x^2 + y^2))    (radians, in [-π/2, +π/2])

    Important interpretation note:
      - In an ECEF frame, this "azimuth" is the angle in the *global* XY-plane. It is
        not the same as a compass-like azimuth in the observer's local ENU/NED frame.
      - If you want sensor/body-frame az/el, rotate the LOS vector into that frame
        first, then call an identical az/el computation on the rotated components.

    Parameters
    ----------
    obs_pos : np.ndarray
        Observer position [x, y, z] in meters, expressed in a single consistent
        Cartesian frame (often ECEF). Shape (3,), dtype float64.
    target_pos : np.ndarray
        Target position [x, y, z] in meters, in the same frame as `obs_pos`.
        Shape (3,), dtype float64.

    Returns
    -------
    (az, el) : tuple[float, float]
        Azimuth and elevation in radians.

    Notes
    -----
    - Singular geometry: when x=y=0 (target exactly on the ±Z axis through the observer),
      azimuth is undefined. This function will still return `atan2(0,0)` behavior for az
      (implementation-defined) and el = ±π/2 depending on z; handle such cases upstream
      if needed.
    - Implemented for high-throughput @njit use: minimal temporaries and no allocations.
    """
    x = target_pos[0] - obs_pos[0]
    y = target_pos[1] - obs_pos[1]
    z = target_pos[2] - obs_pos[2]

    az = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    el = math.atan2(z, p)
    return az, el


@njit(cache=True)
def aoa_jacobian_az_el_xyz(obs_pos: np.ndarray, target_pos: np.ndarray):
    """
    Compute the Jacobian of (azimuth, elevation) with respect to target position.

    The AOA model is defined from the global-frame LOS vector:

        ρ = r_t - r_o = [x, y, z]^T
        p = sqrt(x^2 + y^2)
        d = x^2 + y^2 + z^2

        az = atan2(y, x)
        el = atan2(z, p)

    This function returns the partial derivatives of (az, el) with respect to the
    target position components (x_t, y_t, z_t). Since ρ = r_t - r_o, these are
    equivalent to derivatives w.r.t. the LOS components (x, y, z).

    Analytical derivatives (for x^2 + y^2 > 0):

      Azimuth:
        ∂az/∂x = -y / (x^2 + y^2)
        ∂az/∂y =  x / (x^2 + y^2)
        ∂az/∂z =  0

      Elevation (p = sqrt(x^2+y^2), d = x^2+y^2+z^2):
        ∂el/∂x = -(x z) / (p d)
        ∂el/∂y = -(y z) / (p d)
        ∂el/∂z =  p / d

    Returned units:
      - derivatives are in radians per meter [rad/m].

    Frame/attitude usage:
      - These derivatives are correct for az/el defined in the *same frame* as the input
        positions (e.g., ECEF).
      - If your measurement is in a body/sensor frame, rotate the LOS vector into that
        frame before computing az/el. For Jacobians, apply the chain rule using the
        rotation matrix R (world->sensor) since ρ_s = R ρ:
            J_pos_sensor = J_wrt_rhos @ R
        (and w.r.t. target position, because dρ/dr_t = I).
      - Do not attempt to "rotate angles" directly; rotate vectors and transform the
        Jacobian via chain rule.

    Degenerate cases:
      - If x^2 + y^2 == 0, azimuth is undefined and the elevation derivative involves
        division by p. This function returns all zeros in that case to avoid division
        by zero; treat such geometry as invalid for estimation.

    Parameters
    ----------
    obs_pos : np.ndarray
        Observer position [x, y, z] in meters in a consistent Cartesian frame.
        Shape (3,), dtype float64.
    target_pos : np.ndarray
        Target position [x, y, z] in meters in the same frame as `obs_pos`.
        Shape (3,), dtype float64.

    Returns
    -------
    (daz_dx, daz_dy, daz_dz, del_dx, del_dy, del_dz) : tuple[float, float, float, float, float, float]
        Partial derivatives of azimuth and elevation with respect to target position.
        Units are [rad/m].

    Notes
    -----
    - Implemented for high-throughput @njit use: minimal temporaries and no allocations.
    - Jacobian w.r.t. observer position is the negative of this result (since ρ = r_t - r_o).
    """
    x = target_pos[0] - obs_pos[0]
    y = target_pos[1] - obs_pos[1]
    z = target_pos[2] - obs_pos[2]

    xy2 = x * x + y * y
    if xy2 == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    inv_xy2 = 1.0 / xy2
    daz_dx = -y * inv_xy2
    daz_dy = x * inv_xy2
    daz_dz = 0.0

    p = math.sqrt(xy2)
    d = xy2 + z * z  # = r^2
    if p == 0.0 or d == 0.0:
        return daz_dx, daz_dy, daz_dz, 0.0, 0.0, 0.0

    inv_p = 1.0 / p
    inv_d = 1.0 / d

    del_dx = -(x * z) * (inv_p * inv_d)
    del_dy = -(y * z) * (inv_p * inv_d)
    del_dz = p * inv_d

    return daz_dx, daz_dy, daz_dz, del_dx, del_dy, del_dz
