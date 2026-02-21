import math

import numpy as np
from numba import njit, prange

_RANGE_EPS = 1e-12
_PIVOT_EPS = 1e-15


@njit(cache=True, inline="always")
def _fill_inf_4x4(m: np.ndarray) -> None:
    for i in range(4):
        for j in range(4):
            m[i, j] = np.inf


@njit(cache=True)
def _invert_4x4_gauss_jordan(a: np.ndarray, a_inv: np.ndarray) -> bool:
    """
    Invert a 4x4 matrix using Gauss-Jordan elimination with partial pivoting.
    Returns False if matrix is singular/ill-conditioned at the chosen pivot epsilon.
    """
    aug = np.empty((4, 8), dtype=np.float64)

    for i in range(4):
        for j in range(4):
            aug[i, j] = a[i, j]
            aug[i, 4 + j] = 1.0 if i == j else 0.0

    for col in range(4):
        pivot_row = col
        pivot_abs = abs(aug[col, col])
        for r in range(col + 1, 4):
            v = abs(aug[r, col])
            if v > pivot_abs:
                pivot_abs = v
                pivot_row = r

        if pivot_abs <= _PIVOT_EPS:
            return False

        if pivot_row != col:
            for j in range(8):
                tmp = aug[col, j]
                aug[col, j] = aug[pivot_row, j]
                aug[pivot_row, j] = tmp

        piv = aug[col, col]
        inv_piv = 1.0 / piv
        for j in range(8):
            aug[col, j] *= inv_piv

        for r in range(4):
            if r == col:
                continue
            f = aug[r, col]
            if f == 0.0:
                continue
            for j in range(8):
                aug[r, j] -= f * aug[col, j]

    for i in range(4):
        for j in range(4):
            a_inv[i, j] = aug[i, 4 + j]
    return True


@njit(cache=True)
def dop_covariance_matrix(observer_pos: np.ndarray, gnss_asset_positions: np.ndarray) -> np.ndarray:
    """
    Compute DOP covariance matrix Q = (H^T H)^-1 for a single observer.

    Parameters
    ----------
    observer_pos : np.ndarray
        Observer position [x, y, z], shape (3,), in a consistent Cartesian frame.
    gnss_asset_positions : np.ndarray
        GNSS asset positions [x, y, z], shape (N, 3), in the same frame.

    Returns
    -------
    np.ndarray
        4x4 DOP covariance matrix Q. Returns all `inf` when geometry is invalid.
    """
    q = np.empty((4, 4), dtype=np.float64)

    if observer_pos.shape[0] != 3:
        _fill_inf_4x4(q)
        return q
    if gnss_asset_positions.ndim != 2 or gnss_asset_positions.shape[1] != 3:
        _fill_inf_4x4(q)
        return q

    n = gnss_asset_positions.shape[0]
    if n < 4:
        _fill_inf_4x4(q)
        return q

    # A = H^T H
    a = np.zeros((4, 4), dtype=np.float64)
    valid = 0

    ox = observer_pos[0]
    oy = observer_pos[1]
    oz = observer_pos[2]

    for i in range(n):
        dx = gnss_asset_positions[i, 0] - ox
        dy = gnss_asset_positions[i, 1] - oy
        dz = gnss_asset_positions[i, 2] - oz

        r2 = dx * dx + dy * dy + dz * dz
        if r2 <= _RANGE_EPS:
            continue
        inv_r = 1.0 / math.sqrt(r2)

        # Row h = [-ux, -uy, -uz, 1]
        hx = -dx * inv_r
        hy = -dy * inv_r
        hz = -dz * inv_r
        ht = 1.0

        a[0, 0] += hx * hx
        a[0, 1] += hx * hy
        a[0, 2] += hx * hz
        a[0, 3] += hx * ht

        a[1, 1] += hy * hy
        a[1, 2] += hy * hz
        a[1, 3] += hy * ht

        a[2, 2] += hz * hz
        a[2, 3] += hz * ht

        a[3, 3] += ht * ht
        valid += 1

    if valid < 4:
        _fill_inf_4x4(q)
        return q

    # Symmetrize.
    a[1, 0] = a[0, 1]
    a[2, 0] = a[0, 2]
    a[3, 0] = a[0, 3]
    a[2, 1] = a[1, 2]
    a[3, 1] = a[1, 3]
    a[3, 2] = a[2, 3]

    ok = _invert_4x4_gauss_jordan(a, q)
    if not ok:
        _fill_inf_4x4(q)
    return q


@njit(cache=True, inline="always")
def gdop_pdop_tdop(observer_pos: np.ndarray, gnss_asset_positions: np.ndarray):
    """
    Compute GDOP, PDOP, and TDOP for one observer from GNSS asset geometry.

    Returns `(gdop, pdop, tdop)`. Returns `inf` values when geometry is invalid.
    """
    q = dop_covariance_matrix(observer_pos, gnss_asset_positions)

    qxx = q[0, 0]
    qyy = q[1, 1]
    qzz = q[2, 2]
    qtt = q[3, 3]

    if not np.isfinite(qxx) or not np.isfinite(qyy) or not np.isfinite(qzz) or not np.isfinite(qtt):
        return np.inf, np.inf, np.inf

    pd = qxx + qyy + qzz
    gd = pd + qtt
    if pd < 0.0:
        pd = 0.0
    if gd < 0.0:
        gd = 0.0
    if qtt < 0.0:
        qtt = 0.0

    return math.sqrt(gd), math.sqrt(pd), math.sqrt(qtt)


@njit(cache=True, parallel=True)
def gdop_pdop_tdop_many_observers(
    observer_positions: np.ndarray, gnss_asset_positions: np.ndarray
) -> np.ndarray:
    """
    Compute GDOP/PDOP/TDOP for many observers against a shared GNSS asset set.

    Parameters
    ----------
    observer_positions : np.ndarray
        Observer positions, shape (M, 3).
    gnss_asset_positions : np.ndarray
        GNSS asset positions, shape (N, 3).

    Returns
    -------
    np.ndarray
        DOP array with shape (M, 3), columns `[GDOP, PDOP, TDOP]`.
    """
    if observer_positions.ndim != 2 or observer_positions.shape[1] != 3:
        raise ValueError("observer_positions must have shape (M, 3)")
    if gnss_asset_positions.ndim != 2 or gnss_asset_positions.shape[1] != 3:
        raise ValueError("gnss_asset_positions must have shape (N, 3)")

    m = observer_positions.shape[0]
    out = np.empty((m, 3), dtype=np.float64)
    for i in prange(m):
        gd, pd, td = gdop_pdop_tdop(observer_positions[i], gnss_asset_positions)
        out[i, 0] = gd
        out[i, 1] = pd
        out[i, 2] = td
    return out

