import numpy as np
import math
from numba import njit

# Defaults
C_LIGHT_MPS = 299_792_458.0
OMEGA_EARTH_RADPS = 7.2921151467e-5  # WGS-84 Earth rotation rate


@njit(inline="always")
def _find_interval_sorted(t, tq):
    """
    Returns i such that t[i] <= tq <= t[i+1], clamped to [0, n-2].
    Requires t strictly increasing for best behavior (non-decreasing is tolerated).
    """
    n = t.shape[0]
    if n <= 2:
        return 0

    if tq <= t[0]:
        return 0
    if tq >= t[n - 1]:
        return n - 2

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= tq:
            lo = mid
        else:
            hi = mid
    return lo


@njit(inline="always")
def _state_cubic_hermite(tq, t, p, v):
    """
    Cubic Hermite interpolation of ECEF position/velocity.
    Inputs:
      t: (N,) seconds (sorted ascending)
      p: (N,3) meters
      v: (N,3) m/s  (time derivative of p in ECEF)

    Returns:
      (px,py,pz,vx,vy,vz) at time tq, clamped at endpoints.
    """
    n = t.shape[0]
    if n == 1:
        return p[0, 0], p[0, 1], p[0, 2], v[0, 0], v[0, 1], v[0, 2]

    i = _find_interval_sorted(t, tq)
    t0 = t[i]
    t1 = t[i + 1]

    # Clamp to endpoints
    if tq <= t0:
        return p[i, 0], p[i, 1], p[i, 2], v[i, 0], v[i, 1], v[i, 2]
    if tq >= t1:
        return (
            p[i + 1, 0],
            p[i + 1, 1],
            p[i + 1, 2],
            v[i + 1, 0],
            v[i + 1, 1],
            v[i + 1, 2],
        )

    dt = t1 - t0
    if dt <= 0.0:
        # Degenerate interval
        return p[i, 0], p[i, 1], p[i, 2], v[i, 0], v[i, 1], v[i, 2]

    s = (tq - t0) / dt
    s2 = s * s
    s3 = s2 * s

    # Hermite basis for position
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2

    p0x = p[i, 0]
    p0y = p[i, 1]
    p0z = p[i, 2]
    p1x = p[i + 1, 0]
    p1y = p[i + 1, 1]
    p1z = p[i + 1, 2]
    v0x = v[i, 0]
    v0y = v[i, 1]
    v0z = v[i, 2]
    v1x = v[i + 1, 0]
    v1y = v[i + 1, 1]
    v1z = v[i + 1, 2]

    dtv0x = dt * v0x
    dtv0y = dt * v0y
    dtv0z = dt * v0z
    dtv1x = dt * v1x
    dtv1y = dt * v1y
    dtv1z = dt * v1z

    px = h00 * p0x + h10 * dtv0x + h01 * p1x + h11 * dtv1x
    py = h00 * p0y + h10 * dtv0y + h01 * p1y + h11 * dtv1y
    pz = h00 * p0z + h10 * dtv0z + h01 * p1z + h11 * dtv1z

    # Derivatives of Hermite basis w.r.t time
    inv_dt = 1.0 / dt
    dh00 = (6.0 * s2 - 6.0 * s) * inv_dt
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = (-6.0 * s2 + 6.0 * s) * inv_dt
    dh11 = 3.0 * s2 - 2.0 * s

    vx = dh00 * p0x + dh10 * v0x + dh01 * p1x + dh11 * v1x
    vy = dh00 * p0y + dh10 * v0y + dh01 * p1y + dh11 * v1y
    vz = dh00 * p0z + dh10 * v0z + dh01 * p1z + dh11 * v1z

    return px, py, pz, vx, vy, vz


@njit(inline="always")
def _toa_doppler_one_ecef(
    t_rx_obs, t_tx, p_tx, v_tx, t_rx, p_rx, v_rx, c, omega, max_iter, tol
):
    """
    Compute one receive-time observation:
      - delay tau (seconds): solution to tau = (||r_rx(t)-r_tx(t-tau)|| + sagnac_range)/c
      - doppler factor D ~ f_rx/f_tx (dimensionless), first-order from total range-rate:
            D = 1 - (d/dt range_total)/c
        where range_total includes first-order Sagnac term in ECEF.

    Sagnac first-order (meters):
      range_sag = (omega/c) * (x_tx*y_rx - y_tx*x_rx)
      using tx at transmit time (t_rx_obs - tau) and rx at receive time (t_rx_obs).
    """
    inv_c = 1.0 / c
    omega_over_c = omega * inv_c  # omega/c

    # Receiver state at receive time
    rrx_x, rrx_y, rrx_z, vrx_x, vrx_y, vrx_z = _state_cubic_hermite(
        t_rx_obs, t_rx, p_rx, v_rx
    )

    # Initial guess: use tx state at same time (ignoring light-time), include Sagnac term
    rtx_x0, rtx_y0, rtx_z0, _, _, _ = _state_cubic_hermite(t_rx_obs, t_tx, p_tx, v_tx)
    dx0 = rrx_x - rtx_x0
    dy0 = rrx_y - rtx_y0
    dz0 = rrx_z - rtx_z0
    rho0 = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if rho0 < 1e-9:
        rho0 = 1e-9
    sag0 = omega_over_c * (rtx_x0 * rrx_y - rtx_y0 * rrx_x)  # meters
    tau = (rho0 + sag0) * inv_c

    # Iterate fixed-point for light-time with Sagnac
    rtx_x = rtx_x0
    rtx_y = rtx_y0
    rtx_z = rtx_z0
    vtx_x = 0.0
    vtx_y = 0.0
    vtx_z = 0.0
    dx = dx0
    dy = dy0
    dz = dz0
    rho = rho0

    for _ in range(max_iter):
        t_tx_emit = t_rx_obs - tau
        rtx_x, rtx_y, rtx_z, vtx_x, vtx_y, vtx_z = _state_cubic_hermite(
            t_tx_emit, t_tx, p_tx, v_tx
        )

        dx = rrx_x - rtx_x
        dy = rrx_y - rtx_y
        dz = rrx_z - rtx_z
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        if rho < 1e-9:
            rho = 1e-9

        sag = omega_over_c * (rtx_x * rrx_y - rtx_y * rrx_x)  # meters
        tau_new = (rho + sag) * inv_c
        if abs(tau_new - tau) < tol:
            tau = tau_new
            break
        tau = tau_new

    # Doppler factor from total range-rate (first order)
    inv_rho = 1.0 / rho
    ux = dx * inv_rho
    uy = dy * inv_rho
    uz = dz * inv_rho

    rel_vx = vrx_x - vtx_x
    rel_vy = vrx_y - vtx_y
    rel_vz = vrx_z - vtx_z

    # geometric range-rate (m/s)
    rho_dot_geom = ux * rel_vx + uy * rel_vy + uz * rel_vz

    # first-order sagnac range-rate (m/s):
    # d/dt [ (omega/c) * (x_tx*y_rx - y_tx*x_rx) ]
    rho_dot_sag = omega_over_c * (
        vtx_x * rrx_y + rtx_x * vrx_y - vtx_y * rrx_x - rtx_y * vrx_x
    )

    rho_dot_total = rho_dot_geom + rho_dot_sag

    # Doppler factor: f_rx/f_tx ≈ 1 - (range_rate)/c
    D = 1.0 - rho_dot_total * inv_c

    return tau, D


@njit
def toa_doppler_ecef(
    t_query,
    t_tx,
    p_tx,
    v_tx,
    t_rx,
    p_rx,
    v_rx,
    c=C_LIGHT_MPS,
    omega=OMEGA_EARTH_RADPS,
    max_iter=6,
    tol=1e-12,
):
    """
    Public njit interface (vectorized):
      t_query: (M,) receive times (seconds)
      t_tx:    (Ntx,) tx sample times (seconds), sorted
      p_tx:    (Ntx,3) tx ECEF positions (m)
      v_tx:    (Ntx,3) tx ECEF velocities (m/s)
      t_rx:    (Nrx,) rx sample times (seconds), sorted
      p_rx:    (Nrx,3) rx ECEF positions (m)
      v_rx:    (Nrx,3) rx ECEF velocities (m/s)

    Returns:
      delays:  (M,) seconds
      doppler: (M,) dimensionless Doppler factor (f_rx/f_tx)
    """
    m = t_query.shape[0]
    delays = np.empty(m, dtype=np.float64)
    doppler = np.empty(m, dtype=np.float64)

    for i in range(m):
        tau, D = _toa_doppler_one_ecef(
            t_query[i], t_tx, p_tx, v_tx, t_rx, p_rx, v_rx, c, omega, max_iter, tol
        )
        delays[i] = tau
        doppler[i] = D

    return delays, doppler
