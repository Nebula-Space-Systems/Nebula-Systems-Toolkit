"""
Certified wrap-induced network ambiguity PRI/PRF on the WGS84 ellipsoid.

Definition enforced by this module
----------------------------------
For reference observer 0 and TDOA vector t(x) over the common-visible footprint R:

    t_i(x) = tau_i(x) - tau_0(x),  tau_i(x) = ||x - p_i|| / c

A PRI T is SAFE iff there do not exist distinct x,y in R and an integer vector
n in Z^(N-1), with at least one nonzero component, such that:

    | t(y) - t(x) - n*T | <= eps     (componentwise)

Intrinsic unwrapped ambiguities (n=0) are intentionally allowed.

Certification strategy
----------------------
- Branch-and-bound on spherical-triangle cells mapped to WGS84 surface.
- Certified visibility bounds per cell classify nodes as:
  invisible (state=0), partial (state=1), fully-visible (state=2).
- Certified TDOA interval boxes [tlo, thi] prune impossible cell pairs via integer
  feasibility tests with conservative floating-point slack.
- Only fully-visible leaves can witness ambiguity.
- If caps are exceeded, predicate is INCONCLUSIVE and public APIs raise.

All public compute entry points are numba-njit and call only njit kernels.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numba import njit

_FP64_EPS = 2.220446049250313e-16
STATUS_AMBIG = np.int8(0)
STATUS_SAFE = np.int8(1)
STATUS_INCONCLUSIVE = np.int8(-1)
REASON_NONE = np.int8(0)
REASON_NODE_CAP = np.int8(1)
REASON_HEAP_CAP = np.int8(2)
REASON_POP_CAP = np.int8(3)
REASON_STACK_CAP = np.int8(4)

# -----------------------------
# Unit icosahedron (directions)
# -----------------------------
_ICO_V = np.array(
    [
        [-0.5257311121191336, 0.8506508083520399, 0.0],
        [0.5257311121191336, 0.8506508083520399, 0.0],
        [-0.5257311121191336, -0.8506508083520399, 0.0],
        [0.5257311121191336, -0.8506508083520399, 0.0],
        [0.0, -0.5257311121191336, 0.8506508083520399],
        [0.0, 0.5257311121191336, 0.8506508083520399],
        [0.0, -0.5257311121191336, -0.8506508083520399],
        [0.0, 0.5257311121191336, -0.8506508083520399],
        [0.8506508083520399, 0.0, -0.5257311121191336],
        [0.8506508083520399, 0.0, 0.5257311121191336],
        [-0.8506508083520399, 0.0, -0.5257311121191336],
        [-0.8506508083520399, 0.0, 0.5257311121191336],
    ],
    dtype=np.float64,
)

_ICO_F = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int64,
)


# -----------------------------
# Low-level math
# -----------------------------
@njit(cache=True, inline="always")
def _norm3(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


@njit(cache=True, inline="always")
def _normalize3(x, y, z):
    n = _norm3(x, y, z)
    return x / n, y / n, z / n


@njit(cache=True, inline="always")
def _mid_dir(ax, ay, az, bx, by, bz):
    return _normalize3(ax + bx, ay + by, az + bz)


@njit(cache=True, inline="always")
def _ellipsoid_intersect_dir(ux, uy, uz, inva2, invb2):
    s = (ux * ux + uy * uy) * inva2 + (uz * uz) * invb2
    t = 1.0 / np.sqrt(s)
    return t * ux, t * uy, t * uz


@njit(cache=True, inline="always")
def _dot3(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(cache=True)
def _closest_origin_norm_triangle(u0, u1, u2):
    # Closest point to origin on triangle (u0,u1,u2), RTCD region tests.
    ax, ay, az = u0[0], u0[1], u0[2]
    bx, by, bz = u1[0], u1[1], u1[2]
    cx, cy, cz = u2[0], u2[1], u2[2]

    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = -ax, -ay, -az
    d1 = _dot3(abx, aby, abz, apx, apy, apz)
    d2 = _dot3(acx, acy, acz, apx, apy, apz)
    if d1 <= 0.0 and d2 <= 0.0:
        return _norm3(ax, ay, az)

    bpx, bpy, bpz = -bx, -by, -bz
    d3 = _dot3(abx, aby, abz, bpx, bpy, bpz)
    d4 = _dot3(acx, acy, acz, bpx, bpy, bpz)
    if d3 >= 0.0 and d4 <= d3:
        return _norm3(bx, by, bz)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        den = d1 - d3
        if np.abs(den) < 1e-30:
            na = _norm3(ax, ay, az)
            nb = _norm3(bx, by, bz)
            return na if na < nb else nb
        v = d1 / den
        px = ax + v * abx
        py = ay + v * aby
        pz = az + v * abz
        return _norm3(px, py, pz)

    cpx, cpy, cpz = -cx, -cy, -cz
    d5 = _dot3(abx, aby, abz, cpx, cpy, cpz)
    d6 = _dot3(acx, acy, acz, cpx, cpy, cpz)
    if d6 >= 0.0 and d5 <= d6:
        return _norm3(cx, cy, cz)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        den = d2 - d6
        if np.abs(den) < 1e-30:
            na = _norm3(ax, ay, az)
            nc = _norm3(cx, cy, cz)
            return na if na < nc else nc
        w = d2 / den
        px = ax + w * acx
        py = ay + w * acy
        pz = az + w * acz
        return _norm3(px, py, pz)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        den = (d4 - d3) + (d5 - d6)
        if np.abs(den) < 1e-30:
            nb = _norm3(bx, by, bz)
            nc = _norm3(cx, cy, cz)
            return nb if nb < nc else nc
        w = (d4 - d3) / den
        bcx, bcy, bcz = cx - bx, cy - by, cz - bz
        px = bx + w * bcx
        py = by + w * bcy
        pz = bz + w * bcz
        return _norm3(px, py, pz)

    den = va + vb + vc
    if np.abs(den) < 1e-30:
        na = _norm3(ax, ay, az)
        nb = _norm3(bx, by, bz)
        nc = _norm3(cx, cy, cz)
        m = na if na < nb else nb
        return m if m < nc else nc

    inv_den = 1.0 / den
    v = vb * inv_den
    w = vc * inv_den
    px = ax + abx * v + acx * w
    py = ay + aby * v + acy * w
    pz = az + abz * v + acz * w
    return _norm3(px, py, pz)


@njit(cache=True)
def _cell_diameter_bound(u0, u1, u2, L_u_to_x):
    d01 = _norm3(u0[0] - u1[0], u0[1] - u1[1], u0[2] - u1[2])
    d12 = _norm3(u1[0] - u2[0], u1[1] - u2[1], u1[2] - u2[2])
    d20 = _norm3(u2[0] - u0[0], u2[1] - u0[1], u2[2] - u0[2])

    diam_P = d01
    if d12 > diam_P:
        diam_P = d12
    if d20 > diam_P:
        diam_P = d20

    m = _closest_origin_norm_triangle(u0, u1, u2)
    if m < 1e-15:
        dir_diam = 2.0
    else:
        dir_diam = diam_P / m
        if dir_diam > 2.0:
            dir_diam = 2.0
    return L_u_to_x * dir_diam


@njit(cache=True, inline="always")
def _g_visibility_margin(x, y, z, ps, nobs):
    g = 1e300
    for k in range(nobs):
        v = ps[k, 0] * x + ps[k, 1] * y + ps[k, 2] * z - 1.0
        if v < g:
            g = v
    return g


@njit(cache=True, inline="always")
def _round_nearest_int64(x):
    if x >= 0.0:
        return np.int64(np.floor(x + 0.5))
    return np.int64(-np.floor(-x + 0.5))


# -----------------------------
# Pair-heap (max-heap)
# -----------------------------
@njit(cache=True, inline="always")
def _pair_heap_swap(pair_a, pair_b, pair_key, i, j):
    ta = pair_a[i]
    tb = pair_b[i]
    tk = pair_key[i]
    pair_a[i] = pair_a[j]
    pair_b[i] = pair_b[j]
    pair_key[i] = pair_key[j]
    pair_a[j] = ta
    pair_b[j] = tb
    pair_key[j] = tk


@njit(cache=True)
def _pair_heap_push(pair_a, pair_b, pair_key, heap_size, a_idx, b_idx, key):
    i = heap_size
    pair_a[i] = a_idx
    pair_b[i] = b_idx
    pair_key[i] = key
    heap_size += 1

    while i > 0:
        p = (i - 1) >> 1
        if pair_key[p] >= pair_key[i]:
            break
        _pair_heap_swap(pair_a, pair_b, pair_key, p, i)
        i = p
    return heap_size


@njit(cache=True)
def _pair_heap_pop_max(pair_a, pair_b, pair_key, heap_size):
    out_a = pair_a[0]
    out_b = pair_b[0]
    out_k = pair_key[0]
    heap_size -= 1

    pair_a[0] = pair_a[heap_size]
    pair_b[0] = pair_b[heap_size]
    pair_key[0] = pair_key[heap_size]

    i = 0
    while True:
        l = (i << 1) + 1
        r = l + 1
        if l >= heap_size:
            break
        j = l
        if r < heap_size and pair_key[r] > pair_key[l]:
            j = r
        if pair_key[i] >= pair_key[j]:
            break
        _pair_heap_swap(pair_a, pair_b, pair_key, i, j)
        i = j

    return out_a, out_b, out_k, heap_size


# -----------------------------
# Generic idx-key heaps (N=2 scalar range solver)
# -----------------------------
@njit(cache=True, inline="always")
def _idx_heap_swap(heap_idx, heap_key, i, j):
    ti = heap_idx[i]
    tk = heap_key[i]
    heap_idx[i] = heap_idx[j]
    heap_key[i] = heap_key[j]
    heap_idx[j] = ti
    heap_key[j] = tk


@njit(cache=True)
def _idx_heap_push_max(heap_idx, heap_key, heap_size, idx, key):
    i = heap_size
    heap_idx[i] = idx
    heap_key[i] = key
    heap_size += 1
    while i > 0:
        p = (i - 1) >> 1
        if heap_key[p] >= heap_key[i]:
            break
        _idx_heap_swap(heap_idx, heap_key, p, i)
        i = p
    return heap_size


@njit(cache=True)
def _idx_heap_pop_max(heap_idx, heap_key, heap_size):
    out_idx = heap_idx[0]
    out_key = heap_key[0]
    heap_size -= 1
    heap_idx[0] = heap_idx[heap_size]
    heap_key[0] = heap_key[heap_size]
    i = 0
    while True:
        l = (i << 1) + 1
        r = l + 1
        if l >= heap_size:
            break
        j = l
        if r < heap_size and heap_key[r] > heap_key[l]:
            j = r
        if heap_key[i] >= heap_key[j]:
            break
        _idx_heap_swap(heap_idx, heap_key, i, j)
        i = j
    return out_idx, out_key, heap_size


@njit(cache=True)
def _idx_heap_push_min(heap_idx, heap_key, heap_size, idx, key):
    i = heap_size
    heap_idx[i] = idx
    heap_key[i] = key
    heap_size += 1
    while i > 0:
        p = (i - 1) >> 1
        if heap_key[p] <= heap_key[i]:
            break
        _idx_heap_swap(heap_idx, heap_key, p, i)
        i = p
    return heap_size


@njit(cache=True)
def _idx_heap_pop_min(heap_idx, heap_key, heap_size):
    out_idx = heap_idx[0]
    out_key = heap_key[0]
    heap_size -= 1
    heap_idx[0] = heap_idx[heap_size]
    heap_key[0] = heap_key[heap_size]
    i = 0
    while True:
        l = (i << 1) + 1
        r = l + 1
        if l >= heap_size:
            break
        j = l
        if r < heap_size and heap_key[r] < heap_key[l]:
            j = r
        if heap_key[i] <= heap_key[j]:
            break
        _idx_heap_swap(heap_idx, heap_key, i, j)
        i = j
    return out_idx, out_key, heap_size


@njit(cache=True)
def _idx_heap_peek_valid_min(heap_idx, heap_key, heap_size, active_flag):
    while heap_size > 0:
        idx = heap_idx[0]
        if active_flag[idx] == 1:
            return np.int8(1), heap_key[0], heap_size
        _idx, _key, heap_size = _idx_heap_pop_min(heap_idx, heap_key, heap_size)
    return np.int8(0), 0.0, heap_size


@njit(cache=True)
def _idx_heap_peek_valid_max(heap_idx, heap_key, heap_size, active_flag):
    while heap_size > 0:
        idx = heap_idx[0]
        if active_flag[idx] == 1:
            return np.int8(1), heap_key[0], heap_size
        _idx, _key, heap_size = _idx_heap_pop_max(heap_idx, heap_key, heap_size)
    return np.int8(0), 0.0, heap_size


# -----------------------------
# Problem setup helpers
# -----------------------------
@njit(cache=True)
def _reorder_ref_first(obs_ecef_m, sigma_t_s, ref_idx):
    nobs = obs_ecef_m.shape[0]
    obs_out = np.empty((nobs, 3), dtype=np.float64)
    sig_out = np.empty(nobs, dtype=np.float64)

    obs_out[0, 0] = obs_ecef_m[ref_idx, 0]
    obs_out[0, 1] = obs_ecef_m[ref_idx, 1]
    obs_out[0, 2] = obs_ecef_m[ref_idx, 2]
    sig_out[0] = sigma_t_s[ref_idx]

    q = 1
    for i in range(nobs):
        if i != ref_idx:
            obs_out[q, 0] = obs_ecef_m[i, 0]
            obs_out[q, 1] = obs_ecef_m[i, 1]
            obs_out[q, 2] = obs_ecef_m[i, 2]
            sig_out[q] = sigma_t_s[i]
            q += 1

    return obs_out, sig_out


@njit(cache=True)
def _prepare_eps_and_ps(obs_ecef_m, sigma_t_s, k_sigma, margin_mode, inva2, invb2):
    nobs = obs_ecef_m.shape[0]
    d = nobs - 1
    eps = np.empty(d, dtype=np.float64)

    sigma0 = sigma_t_s[0]
    for i in range(1, nobs):
        si = sigma_t_s[i]
        if margin_mode == 0:
            # hard single-measurement TDOA bound
            e = k_sigma * (si + sigma0)
        elif margin_mode == 1:
            # RSS single-measurement TDOA sigma
            e = k_sigma * np.sqrt(si * si + sigma0 * sigma0)
        else:
            raise ValueError("margin_mode must be 0 (hard) or 1 (rss_1sigma).")
        # indistinguishability tolerance between two locations
        eps[i - 1] = 2.0 * e

    ps = np.empty((nobs, 3), dtype=np.float64)
    L_g = 0.0
    for k in range(nobs):
        sx = obs_ecef_m[k, 0] * inva2
        sy = obs_ecef_m[k, 1] * inva2
        sz = obs_ecef_m[k, 2] * invb2
        ps[k, 0] = sx
        ps[k, 1] = sy
        ps[k, 2] = sz
        nr = _norm3(sx, sy, sz)
        if nr > L_g:
            L_g = nr

    return eps, ps, L_g


@njit(cache=True)
def _prepare_bdiff(obs_ecef_m, inv_c):
    nobs = obs_ecef_m.shape[0]
    d = nobs - 1
    bdiff = np.empty(d, dtype=np.float64)
    p0x = obs_ecef_m[0, 0]
    p0y = obs_ecef_m[0, 1]
    p0z = obs_ecef_m[0, 2]
    for i in range(1, nobs):
        dx = obs_ecef_m[i, 0] - p0x
        dy = obs_ecef_m[i, 1] - p0y
        dz = obs_ecef_m[i, 2] - p0z
        bdiff[i - 1] = 2.0 * _norm3(dx, dy, dz) * inv_c
    return bdiff


# -----------------------------
# Cell evaluation and pair pruning
# -----------------------------
@njit(cache=True)
def _eval_cell_tdoa_box(
    u0,
    u1,
    u2,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tlo_out,
    thi_out,
    tsamp_out,
    tvis_out,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
):
    diam = _cell_diameter_bound(u0, u1, u2, L_u_to_x)

    m01x, m01y, m01z = _mid_dir(u0[0], u0[1], u0[2], u1[0], u1[1], u1[2])
    m12x, m12y, m12z = _mid_dir(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
    m20x, m20y, m20z = _mid_dir(u2[0], u2[1], u2[2], u0[0], u0[1], u0[2])
    ccx, ccy, ccz = _normalize3(
        u0[0] + u1[0] + u2[0],
        u0[1] + u1[1] + u2[1],
        u0[2] + u1[2] + u2[2],
    )

    for k in range(nobs):
        tau_min_scratch[k] = 1e300
        tau_max_scratch[k] = -1e300

    g_max = -1e300
    g_min = 1e300

    # s=0: u0
    sx, sy, sz = _ellipsoid_intersect_dir(u0[0], u0[1], u0[2], inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[0] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[0, i - 1] = tau_tmp_scratch[i] - tref

    # s=1: u1
    sx, sy, sz = _ellipsoid_intersect_dir(u1[0], u1[1], u1[2], inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[1] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[1, i - 1] = tau_tmp_scratch[i] - tref

    # s=2: u2
    sx, sy, sz = _ellipsoid_intersect_dir(u2[0], u2[1], u2[2], inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[2] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[2, i - 1] = tau_tmp_scratch[i] - tref

    # s=3: m01
    sx, sy, sz = _ellipsoid_intersect_dir(m01x, m01y, m01z, inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[3] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[3, i - 1] = tau_tmp_scratch[i] - tref

    # s=4: m12
    sx, sy, sz = _ellipsoid_intersect_dir(m12x, m12y, m12z, inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[4] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[4, i - 1] = tau_tmp_scratch[i] - tref

    # s=5: m20
    sx, sy, sz = _ellipsoid_intersect_dir(m20x, m20y, m20z, inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[5] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[5, i - 1] = tau_tmp_scratch[i] - tref

    # s=6: center
    sx, sy, sz = _ellipsoid_intersect_dir(ccx, ccy, ccz, inva2, invb2)
    gs = _g_visibility_margin(sx, sy, sz, ps, nobs)
    if gs > g_max:
        g_max = gs
    if gs < g_min:
        g_min = gs
    tvis_out[6] = 1 if gs >= 0.0 else 0
    for k in range(nobs):
        dx = sx - obs[k, 0]
        dy = sy - obs[k, 1]
        dz = sz - obs[k, 2]
        tau = _norm3(dx, dy, dz) * inv_c
        tau_tmp_scratch[k] = tau
        if tau < tau_min_scratch[k]:
            tau_min_scratch[k] = tau
        if tau > tau_max_scratch[k]:
            tau_max_scratch[k] = tau
    tref = tau_tmp_scratch[0]
    for i in range(1, nobs):
        tsamp_out[6, i - 1] = tau_tmp_scratch[i] - tref

    # tau bounds: intersection of per-sample Lipschitz balls.
    jitter = diam * inv_c
    tau_0_lo = tau_max_scratch[0] - jitter
    tau_0_hi = tau_min_scratch[0] + jitter
    tau_0_lo_loose = tau_min_scratch[0] - jitter
    tau_0_hi_loose = tau_max_scratch[0] + jitter
    for i in range(1, nobs):
        tau_i_lo = tau_max_scratch[i] - jitter
        tau_i_hi = tau_min_scratch[i] + jitter
        tlo_out[i - 1] = tau_i_lo - tau_0_hi
        thi_out[i - 1] = tau_i_hi - tau_0_lo
        if tlo_out[i - 1] > thi_out[i - 1]:
            # Conservative fallback to looser guaranteed bounds.
            tau_i_lo_loose = tau_min_scratch[i] - jitter
            tau_i_hi_loose = tau_max_scratch[i] + jitter
            tlo_out[i - 1] = tau_i_lo_loose - tau_0_hi_loose
            thi_out[i - 1] = tau_i_hi_loose - tau_0_lo_loose
            if tlo_out[i - 1] > thi_out[i - 1]:
                # Numeric guard: widen, never shrink.
                lo = thi_out[i - 1]
                hi = tlo_out[i - 1]
                tlo_out[i - 1] = lo - 1e-15
                thi_out[i - 1] = hi + 1e-15

    gub = g_max + L_g * diam
    glb = g_min - L_g * diam
    return diam, gub, glb


@njit(cache=True, inline="always")
def _pair_overlap_possible(tlo_a, thi_a, tlo_b, thi_b, eps, pri_s):
    d = tlo_a.shape[0]
    score = 0.0
    any_nonzero_possible = 0
    for i in range(d):
        dlo = tlo_b[i] - thi_a[i]
        dhi = thi_b[i] - tlo_a[i]

        mag = np.abs(dlo)
        if np.abs(dhi) > mag:
            mag = np.abs(dhi)
        if np.abs(pri_s) > mag:
            mag = np.abs(pri_s)
        if np.abs(eps[i]) > mag:
            mag = np.abs(eps[i])
        if mag < 1.0:
            mag = 1.0

        slack = 64.0 * _FP64_EPS * mag
        nlo = int(np.ceil((dlo - eps[i] - slack) / pri_s))
        nhi = int(np.floor((dhi + eps[i] + slack) / pri_s))
        if nlo > nhi:
            return 0, 0.0
        if nlo != 0 or nhi != 0:
            any_nonzero_possible = 1

        w = (thi_a[i] - tlo_a[i]) + (thi_b[i] - tlo_b[i])
        if w > score:
            score = w
    if any_nonzero_possible == 0:
        # Only n=0 can satisfy all component intervals: intrinsic ambiguity only.
        return 0, 0.0
    return 1, score


@njit(cache=True)
def _pair_sample_witness(ts_a, ts_b, vis_a, vis_b, eps, pri_s, same_cell):
    d = eps.shape[0]

    # Fast prefilter: center-center only.
    ia = 6
    ib = 6
    if vis_a[ia] == 1 and vis_b[ib] == 1 and (same_cell == 0 or ia != ib):
        ok = 1
        has_nonzero = 0
        for j in range(d):
            diff = ts_b[ib, j] - ts_a[ia, j]
            n = _round_nearest_int64(diff / pri_s)
            r = diff - np.float64(n) * pri_s
            if np.abs(r) > eps[j]:
                ok = 0
                break
            if n != 0:
                has_nonzero = 1
        if ok == 1 and has_nonzero == 1:
            return 1

    # Full 7x7 fallback witness search (required for correctness/termination).
    for ia in range(ts_a.shape[0]):
        if vis_a[ia] == 0:
            continue
        for ib in range(ts_b.shape[0]):
            if vis_b[ib] == 0:
                continue
            if same_cell == 1 and ia == ib:
                continue

            ok = 1
            has_nonzero = 0
            for j in range(d):
                diff = ts_b[ib, j] - ts_a[ia, j]
                n = _round_nearest_int64(diff / pri_s)
                r = diff - np.float64(n) * pri_s
                if np.abs(r) > eps[j]:
                    ok = 0
                    break
                if n != 0:
                    has_nonzero = 1

            if ok == 1 and has_nonzero == 1:
                return 1

    return 0


@njit(cache=True, inline="always")
def _analytic_wrap_safe(pri_s, eps, bdiff):
    # Sufficient SAFE condition for wrap-induced ambiguity:
    # if pri_s > B_i + eps_i (+ tiny conservative slack) for all i, then
    # no nonzero n_i can be feasible in any component.
    d = eps.shape[0]
    for i in range(d):
        mag = np.abs(pri_s)
        if np.abs(bdiff[i]) > mag:
            mag = np.abs(bdiff[i])
        if np.abs(eps[i]) > mag:
            mag = np.abs(eps[i])
        if mag < 1.0:
            mag = 1.0
        slack = 64.0 * _FP64_EPS * mag
        if pri_s <= bdiff[i] + eps[i] + slack:
            return 0
    return 1


@njit(cache=True)
def _evaluate_node(
    idx,
    tri_u,
    tlo,
    thi,
    tsamp,
    tvis,
    gub_arr,
    glb_arr,
    state,
    is_leaf,
    children,
    width,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
):
    diam, gub, glb = _eval_cell_tdoa_box(
        tri_u[idx, 0],
        tri_u[idx, 1],
        tri_u[idx, 2],
        obs,
        ps,
        nobs,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tlo[idx],
        thi[idx],
        tsamp[idx],
        tvis[idx],
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    gub_arr[idx] = gub
    glb_arr[idx] = glb
    children[idx, 0] = -1
    children[idx, 1] = -1
    children[idx, 2] = -1
    children[idx, 3] = -1

    if gub < 0.0:
        state[idx] = 0
        is_leaf[idx] = 0
        width[idx] = diam
    else:
        is_leaf[idx] = 1
        if glb >= 0.0:
            state[idx] = 2
            w = 0.0
            d = tlo.shape[1]
            for j in range(d):
                ww = thi[idx, j] - tlo[idx, j]
                if ww > w:
                    w = ww
            width[idx] = w
        else:
            state[idx] = 1
            width[idx] = diam


@njit(cache=True)
def _split_node(
    idx,
    node_count,
    max_nodes,
    tri_u,
    tlo,
    thi,
    tsamp,
    tvis,
    gub_arr,
    glb_arr,
    state,
    is_leaf,
    children,
    width,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
):
    if node_count + 4 > max_nodes:
        return node_count, -1

    u0x = tri_u[idx, 0, 0]
    u0y = tri_u[idx, 0, 1]
    u0z = tri_u[idx, 0, 2]
    u1x = tri_u[idx, 1, 0]
    u1y = tri_u[idx, 1, 1]
    u1z = tri_u[idx, 1, 2]
    u2x = tri_u[idx, 2, 0]
    u2y = tri_u[idx, 2, 1]
    u2z = tri_u[idx, 2, 2]

    u01x, u01y, u01z = _mid_dir(u0x, u0y, u0z, u1x, u1y, u1z)
    u12x, u12y, u12z = _mid_dir(u1x, u1y, u1z, u2x, u2y, u2z)
    u20x, u20y, u20z = _mid_dir(u2x, u2y, u2z, u0x, u0y, u0z)

    # child0: (u0, u01, u20)
    c0 = node_count
    tri_u[c0, 0, 0], tri_u[c0, 0, 1], tri_u[c0, 0, 2] = u0x, u0y, u0z
    tri_u[c0, 1, 0], tri_u[c0, 1, 1], tri_u[c0, 1, 2] = u01x, u01y, u01z
    tri_u[c0, 2, 0], tri_u[c0, 2, 1], tri_u[c0, 2, 2] = u20x, u20y, u20z
    _evaluate_node(
        c0,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        obs,
        ps,
        nobs,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )

    # child1: (u1, u12, u01)
    c1 = node_count + 1
    tri_u[c1, 0, 0], tri_u[c1, 0, 1], tri_u[c1, 0, 2] = u1x, u1y, u1z
    tri_u[c1, 1, 0], tri_u[c1, 1, 1], tri_u[c1, 1, 2] = u12x, u12y, u12z
    tri_u[c1, 2, 0], tri_u[c1, 2, 1], tri_u[c1, 2, 2] = u01x, u01y, u01z
    _evaluate_node(
        c1,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        obs,
        ps,
        nobs,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )

    # child2: (u2, u20, u12)
    c2 = node_count + 2
    tri_u[c2, 0, 0], tri_u[c2, 0, 1], tri_u[c2, 0, 2] = u2x, u2y, u2z
    tri_u[c2, 1, 0], tri_u[c2, 1, 1], tri_u[c2, 1, 2] = u20x, u20y, u20z
    tri_u[c2, 2, 0], tri_u[c2, 2, 1], tri_u[c2, 2, 2] = u12x, u12y, u12z
    _evaluate_node(
        c2,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        obs,
        ps,
        nobs,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )

    # child3: (u01, u12, u20)
    c3 = node_count + 3
    tri_u[c3, 0, 0], tri_u[c3, 0, 1], tri_u[c3, 0, 2] = u01x, u01y, u01z
    tri_u[c3, 1, 0], tri_u[c3, 1, 1], tri_u[c3, 1, 2] = u12x, u12y, u12z
    tri_u[c3, 2, 0], tri_u[c3, 2, 1], tri_u[c3, 2, 2] = u20x, u20y, u20z
    _evaluate_node(
        c3,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        obs,
        ps,
        nobs,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )

    children[idx, 0] = c0
    children[idx, 1] = c1
    children[idx, 2] = c2
    children[idx, 3] = c3
    is_leaf[idx] = 0

    return node_count + 4, 0


@njit(cache=True)
def _push_pair_candidate(
    a_idx,
    b_idx,
    pri_s,
    eps,
    state,
    tlo,
    thi,
    pair_a,
    pair_b,
    pair_key,
    heap_size,
):
    # prunes invisible and infeasible pairs; pushes feasible ones.
    if state[a_idx] == 0 or state[b_idx] == 0:
        return heap_size, np.int8(0)

    if a_idx > b_idx:
        t = a_idx
        a_idx = b_idx
        b_idx = t

    can, score = _pair_overlap_possible(
        tlo[a_idx], thi[a_idx], tlo[b_idx], thi[b_idx], eps, pri_s
    )
    if can != 1:
        return heap_size, np.int8(0)

    if heap_size >= pair_a.shape[0]:
        return heap_size, np.int8(-1)

    heap_size = _pair_heap_push(
        pair_a, pair_b, pair_key, heap_size, a_idx, b_idx, score
    )
    return heap_size, np.int8(1)


@njit(cache=True)
def _init_root_nodes(
    obs_ecef_m,
    ps,
    max_nodes,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tri_u,
    tlo,
    thi,
    tsamp,
    tvis,
    gub_arr,
    glb_arr,
    state,
    is_leaf,
    children,
    width,
    root_ids,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
):
    nobs = obs_ecef_m.shape[0]
    node_count = 0
    root_count = 0
    for f in range(_ICO_F.shape[0]):
        if node_count >= max_nodes:
            return 0, node_count, root_count
        idx = node_count
        node_count += 1

        i0 = _ICO_F[f, 0]
        i1 = _ICO_F[f, 1]
        i2 = _ICO_F[f, 2]
        tri_u[idx, 0, 0] = _ICO_V[i0, 0]
        tri_u[idx, 0, 1] = _ICO_V[i0, 1]
        tri_u[idx, 0, 2] = _ICO_V[i0, 2]
        tri_u[idx, 1, 0] = _ICO_V[i1, 0]
        tri_u[idx, 1, 1] = _ICO_V[i1, 1]
        tri_u[idx, 1, 2] = _ICO_V[i1, 2]
        tri_u[idx, 2, 0] = _ICO_V[i2, 0]
        tri_u[idx, 2, 1] = _ICO_V[i2, 1]
        tri_u[idx, 2, 2] = _ICO_V[i2, 2]

        _evaluate_node(
            idx,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if state[idx] != 0:
            root_ids[root_count] = idx
            root_count += 1
    return 1, node_count, root_count


@njit(cache=True)
def _pri_wrap_predicate(
    obs_ecef_m,
    eps,
    ps,
    bdiff,
    pri_s,
    max_nodes,
    max_pair_pops,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tri_u,
    tlo,
    thi,
    tsamp,
    tvis,
    gub_arr,
    glb_arr,
    state,
    is_leaf,
    children,
    width,
    pair_a,
    pair_b,
    pair_key,
    root_ids,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
):
    nobs = obs_ecef_m.shape[0]

    if _analytic_wrap_safe(pri_s, eps, bdiff) == 1:
        return STATUS_SAFE, REASON_NONE

    ok_init, node_count, root_count = _init_root_nodes(
        obs_ecef_m,
        ps,
        max_nodes,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        root_ids,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    if ok_init == 0:
        return STATUS_INCONCLUSIVE, REASON_NODE_CAP
    if root_count == 0:
        return STATUS_SAFE, REASON_NONE

    heap_size = 0
    for i in range(root_count):
        for j in range(i, root_count):
            heap_size, pushed = _push_pair_candidate(
                root_ids[i],
                root_ids[j],
                pri_s,
                eps,
                state,
                tlo,
                thi,
                pair_a,
                pair_b,
                pair_key,
                heap_size,
            )
            if pushed == -1:
                return STATUS_INCONCLUSIVE, REASON_HEAP_CAP

    pops = 0
    while heap_size > 0:
        if pops >= max_pair_pops:
            return STATUS_INCONCLUSIVE, REASON_POP_CAP
        pops += 1

        a_idx, b_idx, _k, heap_size = _pair_heap_pop_max(
            pair_a, pair_b, pair_key, heap_size
        )

        if state[a_idx] == 0 or state[b_idx] == 0:
            continue

        if is_leaf[a_idx] == 0:
            for q in range(4):
                c = children[a_idx, q]
                if c >= 0:
                    heap_size, pushed = _push_pair_candidate(
                        c,
                        b_idx,
                        pri_s,
                        eps,
                        state,
                        tlo,
                        thi,
                        pair_a,
                        pair_b,
                        pair_key,
                        heap_size,
                    )
                    if pushed == -1:
                        return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            continue

        if is_leaf[b_idx] == 0:
            for q in range(4):
                c = children[b_idx, q]
                if c >= 0:
                    heap_size, pushed = _push_pair_candidate(
                        a_idx,
                        c,
                        pri_s,
                        eps,
                        state,
                        tlo,
                        thi,
                        pair_a,
                        pair_b,
                        pair_key,
                        heap_size,
                    )
                    if pushed == -1:
                        return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            continue

        # Pair feasibility already checked conservatively at push time.

        if state[a_idx] == 1:
            node_count, ok = _split_node(
                a_idx,
                node_count,
                max_nodes,
                tri_u,
                tlo,
                thi,
                tsamp,
                tvis,
                gub_arr,
                glb_arr,
                state,
                is_leaf,
                children,
                width,
                obs_ecef_m,
                ps,
                nobs,
                inva2,
                invb2,
                L_u_to_x,
                inv_c,
                L_g,
                tau_min_scratch,
                tau_max_scratch,
                tau_tmp_scratch,
            )
            if ok == -1:
                return STATUS_INCONCLUSIVE, REASON_NODE_CAP

            if a_idx == b_idx:
                for i in range(4):
                    ci = children[a_idx, i]
                    if ci < 0 or state[ci] == 0:
                        continue
                    for j in range(i, 4):
                        cj = children[a_idx, j]
                        if cj < 0 or state[cj] == 0:
                            continue
                        heap_size, pushed = _push_pair_candidate(
                            ci,
                            cj,
                            pri_s,
                            eps,
                            state,
                            tlo,
                            thi,
                            pair_a,
                            pair_b,
                            pair_key,
                            heap_size,
                        )
                        if pushed == -1:
                            return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            else:
                for q in range(4):
                    c = children[a_idx, q]
                    if c >= 0 and state[c] != 0:
                        heap_size, pushed = _push_pair_candidate(
                            c,
                            b_idx,
                            pri_s,
                            eps,
                            state,
                            tlo,
                            thi,
                            pair_a,
                            pair_b,
                            pair_key,
                            heap_size,
                        )
                        if pushed == -1:
                            return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            continue

        if state[b_idx] == 1:
            node_count, ok = _split_node(
                b_idx,
                node_count,
                max_nodes,
                tri_u,
                tlo,
                thi,
                tsamp,
                tvis,
                gub_arr,
                glb_arr,
                state,
                is_leaf,
                children,
                width,
                obs_ecef_m,
                ps,
                nobs,
                inva2,
                invb2,
                L_u_to_x,
                inv_c,
                L_g,
                tau_min_scratch,
                tau_max_scratch,
                tau_tmp_scratch,
            )
            if ok == -1:
                return STATUS_INCONCLUSIVE, REASON_NODE_CAP

            for q in range(4):
                c = children[b_idx, q]
                if c >= 0 and state[c] != 0:
                    heap_size, pushed = _push_pair_candidate(
                        a_idx,
                        c,
                        pri_s,
                        eps,
                        state,
                        tlo,
                        thi,
                        pair_a,
                        pair_b,
                        pair_key,
                        heap_size,
                    )
                    if pushed == -1:
                        return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            continue

        if state[a_idx] != 2 or state[b_idx] != 2:
            split_idx = a_idx if width[a_idx] >= width[b_idx] else b_idx
            if a_idx == b_idx:
                split_idx = a_idx
            if is_leaf[split_idx] == 0:
                continue

            node_count, ok = _split_node(
                split_idx,
                node_count,
                max_nodes,
                tri_u,
                tlo,
                thi,
                tsamp,
                tvis,
                gub_arr,
                glb_arr,
                state,
                is_leaf,
                children,
                width,
                obs_ecef_m,
                ps,
                nobs,
                inva2,
                invb2,
                L_u_to_x,
                inv_c,
                L_g,
                tau_min_scratch,
                tau_max_scratch,
                tau_tmp_scratch,
            )
            if ok == -1:
                return STATUS_INCONCLUSIVE, REASON_NODE_CAP

            if a_idx == b_idx:
                for i in range(4):
                    ci = children[split_idx, i]
                    if ci < 0 or state[ci] == 0:
                        continue
                    for j in range(i, 4):
                        cj = children[split_idx, j]
                        if cj < 0 or state[cj] == 0:
                            continue
                        heap_size, pushed = _push_pair_candidate(
                            ci,
                            cj,
                            pri_s,
                            eps,
                            state,
                            tlo,
                            thi,
                            pair_a,
                            pair_b,
                            pair_key,
                            heap_size,
                        )
                        if pushed == -1:
                            return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            else:
                other = b_idx if split_idx == a_idx else a_idx
                for q in range(4):
                    c = children[split_idx, q]
                    if c >= 0 and state[c] != 0:
                        heap_size, pushed = _push_pair_candidate(
                            c,
                            other,
                            pri_s,
                            eps,
                            state,
                            tlo,
                            thi,
                            pair_a,
                            pair_b,
                            pair_key,
                            heap_size,
                        )
                        if pushed == -1:
                            return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
            continue

        if (
            _pair_sample_witness(
                tsamp[a_idx],
                tsamp[b_idx],
                tvis[a_idx],
                tvis[b_idx],
                eps,
                pri_s,
                1 if a_idx == b_idx else 0,
            )
            == 1
        ):
            return STATUS_AMBIG, REASON_NONE

        split_idx = a_idx if width[a_idx] >= width[b_idx] else b_idx
        if a_idx == b_idx:
            split_idx = a_idx
        if is_leaf[split_idx] == 0:
            continue

        node_count, ok = _split_node(
            split_idx,
            node_count,
            max_nodes,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if ok == -1:
            return STATUS_INCONCLUSIVE, REASON_NODE_CAP

        if a_idx == b_idx:
            for i in range(4):
                ci = children[split_idx, i]
                if ci < 0 or state[ci] == 0:
                    continue
                for j in range(i, 4):
                    cj = children[split_idx, j]
                    if cj < 0 or state[cj] == 0:
                        continue
                    heap_size, pushed = _push_pair_candidate(
                        ci,
                        cj,
                        pri_s,
                        eps,
                        state,
                        tlo,
                        thi,
                        pair_a,
                        pair_b,
                        pair_key,
                        heap_size,
                    )
                    if pushed == -1:
                        return STATUS_INCONCLUSIVE, REASON_HEAP_CAP
        else:
            other = b_idx if split_idx == a_idx else a_idx
            for q in range(4):
                c = children[split_idx, q]
                if c >= 0 and state[c] != 0:
                    heap_size, pushed = _push_pair_candidate(
                        c,
                        other,
                        pri_s,
                        eps,
                        state,
                        tlo,
                        thi,
                        pair_a,
                        pair_b,
                        pair_key,
                        heap_size,
                    )
                    if pushed == -1:
                        return STATUS_INCONCLUSIVE, REASON_HEAP_CAP

    return STATUS_SAFE, REASON_NONE


@njit(cache=True)
def _n2_certified_prf_interval(
    obs_ecef_m,
    eps0,
    max_nodes,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    L_g,
    tri_u,
    tlo,
    thi,
    tsamp,
    tvis,
    gub_arr,
    glb_arr,
    state,
    is_leaf,
    children,
    width,
    root_ids,
    tau_min_scratch,
    tau_max_scratch,
    tau_tmp_scratch,
    tol_prf_hz,
    pri_safe_check_s,
):
    nobs = obs_ecef_m.shape[0]
    dx01 = obs_ecef_m[1, 0] - obs_ecef_m[0, 0]
    dy01 = obs_ecef_m[1, 1] - obs_ecef_m[0, 1]
    dz01 = obs_ecef_m[1, 2] - obs_ecef_m[0, 2]
    if _norm3(dx01, dy01, dz01) <= 1e-12:
        if eps0 <= 0.0:
            return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0
        prf = 1.0 / eps0
        return STATUS_SAFE, REASON_NONE, prf, prf, eps0, eps0

    ps_local = np.empty((nobs, 3), dtype=np.float64)
    for k in range(nobs):
        ps_local[k, 0] = obs_ecef_m[k, 0] * inva2
        ps_local[k, 1] = obs_ecef_m[k, 1] * inva2
        ps_local[k, 2] = obs_ecef_m[k, 2] * invb2

    ok_init, node_count, root_count = _init_root_nodes(
        obs_ecef_m,
        ps_local,
        max_nodes,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        root_ids,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    # NOTE: above placeholder for ps is replaced by caller via normal call path.
    if ok_init == 0:
        return STATUS_INCONCLUSIVE, REASON_NODE_CAP, 0.0, 0.0, 0.0, 0.0
    if root_count == 0:
        return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0

    cap = max_nodes
    active_flag = np.zeros(cap, dtype=np.uint8)

    min_tlo_idx = np.empty(cap, dtype=np.int32)
    min_tlo_key = np.empty(cap, dtype=np.float64)
    min_thi_idx = np.empty(cap, dtype=np.int32)
    min_thi_key = np.empty(cap, dtype=np.float64)
    max_tlo_idx = np.empty(cap, dtype=np.int32)
    max_tlo_key = np.empty(cap, dtype=np.float64)
    max_thi_idx = np.empty(cap, dtype=np.int32)
    max_thi_key = np.empty(cap, dtype=np.float64)
    width_idx = np.empty(cap, dtype=np.int32)
    width_key = np.empty(cap, dtype=np.float64)

    n_min_tlo = 0
    n_min_thi = 0
    n_max_tlo = 0
    n_max_thi = 0
    n_width = 0
    active_count = 0

    for r in range(root_count):
        idx = root_ids[r]
        if state[idx] == 0:
            continue
        if active_flag[idx] == 1:
            continue
        active_flag[idx] = 1
        active_count += 1

        if (
            n_min_tlo >= cap
            or n_min_thi >= cap
            or n_max_tlo >= cap
            or n_max_thi >= cap
            or n_width >= cap
        ):
            return STATUS_INCONCLUSIVE, REASON_HEAP_CAP, 0.0, 0.0, 0.0, 0.0

        n_min_tlo = _idx_heap_push_min(
            min_tlo_idx, min_tlo_key, n_min_tlo, idx, tlo[idx, 0]
        )
        n_min_thi = _idx_heap_push_min(
            min_thi_idx, min_thi_key, n_min_thi, idx, thi[idx, 0]
        )
        n_max_tlo = _idx_heap_push_max(
            max_tlo_idx, max_tlo_key, n_max_tlo, idx, tlo[idx, 0]
        )
        n_max_thi = _idx_heap_push_max(
            max_thi_idx, max_thi_key, n_max_thi, idx, thi[idx, 0]
        )
        w = thi[idx, 0] - tlo[idx, 0]
        n_width = _idx_heap_push_max(width_idx, width_key, n_width, idx, w)

    if active_count == 0:
        return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0

    prf_lower = 0.0
    prf_upper = np.inf
    T_lo = 0.0
    T_hi = 0.0

    while True:
        ok, tmin_lo, n_min_tlo = _idx_heap_peek_valid_min(
            min_tlo_idx, min_tlo_key, n_min_tlo, active_flag
        )
        if ok == 0:
            return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0
        ok, tmin_hi, n_min_thi = _idx_heap_peek_valid_min(
            min_thi_idx, min_thi_key, n_min_thi, active_flag
        )
        if ok == 0:
            return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0
        ok, tmax_lo, n_max_tlo = _idx_heap_peek_valid_max(
            max_tlo_idx, max_tlo_key, n_max_tlo, active_flag
        )
        if ok == 0:
            return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0
        ok, tmax_hi, n_max_thi = _idx_heap_peek_valid_max(
            max_thi_idx, max_thi_key, n_max_thi, active_flag
        )
        if ok == 0:
            return STATUS_SAFE, REASON_NONE, np.inf, np.inf, 0.0, 0.0

        range_lo = tmax_lo - tmin_hi
        if range_lo < 0.0:
            range_lo = 0.0
        range_hi = tmax_hi - tmin_lo
        if range_hi < 0.0:
            range_hi = 0.0

        T_lo = range_lo + eps0
        T_hi = range_hi + eps0

        if T_hi <= 0.0:
            prf_lower = np.inf
        else:
            prf_lower = 1.0 / T_hi
        if T_lo <= 0.0:
            prf_upper = np.inf
        else:
            prf_upper = 1.0 / T_lo

        prf_gap = prf_upper - prf_lower
        if pri_safe_check_s > 0.0 and pri_safe_check_s > T_hi:
            return STATUS_SAFE, REASON_NONE, prf_lower, prf_upper, T_lo, T_hi
        if prf_gap <= tol_prf_hz:
            return STATUS_SAFE, REASON_NONE, prf_lower, prf_upper, T_lo, T_hi

        found = 0
        split_idx = -1
        while n_width > 0:
            idx, _k, n_width = _idx_heap_pop_max(width_idx, width_key, n_width)
            if active_flag[idx] == 0:
                continue
            if state[idx] == 0:
                active_flag[idx] = 0
                active_count -= 1
                continue
            if is_leaf[idx] == 0:
                continue
            split_idx = idx
            found = 1
            break

        if found == 0:
            return STATUS_SAFE, REASON_NONE, prf_lower, prf_upper, T_lo, T_hi

        node_count, ok = _split_node(
            split_idx,
            node_count,
            max_nodes,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            obs_ecef_m,
            ps_local,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if ok == -1:
            return (
                STATUS_INCONCLUSIVE,
                REASON_NODE_CAP,
                prf_lower,
                prf_upper,
                T_lo,
                T_hi,
            )

        if active_flag[split_idx] == 1:
            active_flag[split_idx] = 0
            active_count -= 1

        for q in range(4):
            c = children[split_idx, q]
            if c < 0 or state[c] == 0:
                continue
            if active_flag[c] == 1:
                continue
            active_flag[c] = 1
            active_count += 1

            if (
                n_min_tlo >= cap
                or n_min_thi >= cap
                or n_max_tlo >= cap
                or n_max_thi >= cap
                or n_width >= cap
            ):
                return (
                    STATUS_INCONCLUSIVE,
                    REASON_HEAP_CAP,
                    prf_lower,
                    prf_upper,
                    T_lo,
                    T_hi,
                )

            n_min_tlo = _idx_heap_push_min(
                min_tlo_idx, min_tlo_key, n_min_tlo, c, tlo[c, 0]
            )
            n_min_thi = _idx_heap_push_min(
                min_thi_idx, min_thi_key, n_min_thi, c, thi[c, 0]
            )
            n_max_tlo = _idx_heap_push_max(
                max_tlo_idx, max_tlo_key, n_max_tlo, c, tlo[c, 0]
            )
            n_max_thi = _idx_heap_push_max(
                max_thi_idx, max_thi_key, n_max_thi, c, thi[c, 0]
            )
            w = thi[c, 0] - tlo[c, 0]
            n_width = _idx_heap_push_max(width_idx, width_key, n_width, c, w)


@njit(cache=True, inline="always")
def _raise_wrap_inconclusive(reason):
    if reason == REASON_NODE_CAP:
        raise ValueError(
            "wrap predicate inconclusive: increase internal caps (REASON=1)"
        )
    if reason == REASON_HEAP_CAP:
        raise ValueError(
            "wrap predicate inconclusive: increase internal caps (REASON=2)"
        )
    if reason == REASON_POP_CAP:
        raise ValueError(
            "wrap predicate inconclusive: increase internal caps (REASON=3)"
        )
    if reason == REASON_STACK_CAP:
        raise ValueError(
            "wrap predicate inconclusive: increase internal caps (REASON=4)"
        )
    raise ValueError("wrap predicate inconclusive: increase internal caps (REASON=0)")


@njit(cache=True)
def _raise_cert_inconclusive(reason):
    if reason == REASON_NODE_CAP:
        raise ValueError(
            "certification inconclusive: increase internal caps (REASON=1)"
        )
    if reason == REASON_HEAP_CAP:
        raise ValueError(
            "certification inconclusive: increase internal caps (REASON=2)"
        )
    if reason == REASON_POP_CAP:
        raise ValueError(
            "certification inconclusive: increase internal caps (REASON=3)"
        )
    if reason == REASON_STACK_CAP:
        raise ValueError(
            "certification inconclusive: increase internal caps (REASON=4)"
        )
    raise ValueError("certification inconclusive: increase internal caps (REASON=0)")


# -----------------------------
# Public API (all njit)
# -----------------------------
@njit(cache=True)
def is_pri_network_unambiguous(
    obs_ecef_m,
    sigma_t_s,
    pri_s,
    ref_idx=0,
    k_sigma=1.0,
    margin_mode=0,
    max_nodes=1_000_000,
    max_pair_pops=2_000_000,
    max_heap_pairs=4_000_000,
):
    """
    Certified predicate for wrap-induced network ambiguity at a fixed PRI.

    Returns
    -------
    bool
        True -> SAFE (no nonzero-wrap collisions possible)
        False -> AMBIGUOUS.

    Raises
    ------
    ValueError
        If certification is inconclusive under current internal caps.
    """
    if obs_ecef_m.ndim != 2 or obs_ecef_m.shape[1] != 3:
        raise ValueError("obs_ecef_m must have shape (N,3).")
    if sigma_t_s.ndim != 1 or sigma_t_s.shape[0] != obs_ecef_m.shape[0]:
        raise ValueError("sigma_t_s must have shape (N,).")
    if obs_ecef_m.shape[0] < 2:
        raise ValueError("Need at least 2 observers.")
    if ref_idx < 0 or ref_idx >= obs_ecef_m.shape[0]:
        raise ValueError("ref_idx out of range.")
    if pri_s <= 0.0:
        raise ValueError("pri_s must be > 0.")
    if k_sigma < 0.0:
        raise ValueError("k_sigma must be >= 0.")
    if max_nodes < _ICO_F.shape[0]:
        raise ValueError("max_nodes too small.")
    if max_pair_pops <= 0:
        raise ValueError("max_pair_pops must be > 0.")
    if max_heap_pairs <= 0:
        raise ValueError("max_heap_pairs must be > 0.")
    for i in range(sigma_t_s.shape[0]):
        if sigma_t_s[i] < 0.0:
            raise ValueError("sigma_t_s must be non-negative.")

    obs = np.empty((obs_ecef_m.shape[0], 3), dtype=np.float64)
    for i in range(obs_ecef_m.shape[0]):
        obs[i, 0] = obs_ecef_m[i, 0]
        obs[i, 1] = obs_ecef_m[i, 1]
        obs[i, 2] = obs_ecef_m[i, 2]
    sig = np.empty(sigma_t_s.shape[0], dtype=np.float64)
    for i in range(sigma_t_s.shape[0]):
        sig[i] = sigma_t_s[i]
    obs, sig = _reorder_ref_first(obs, sig, ref_idx)

    # WGS84 and constants.
    a = 6378137.0
    b = 6356752.3142451793
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)
    inv_c = 1.0 / 299792458.0
    L_u_to_x = (a * a * a) / (b * b)

    eps, ps, L_g = _prepare_eps_and_ps(obs, sig, k_sigma, margin_mode, inva2, invb2)
    bdiff = _prepare_bdiff(obs, inv_c)
    d = obs.shape[0] - 1

    # Workspace.
    tri_u = np.empty((max_nodes, 3, 3), dtype=np.float64)
    tlo = np.empty((max_nodes, d), dtype=np.float64)
    thi = np.empty((max_nodes, d), dtype=np.float64)
    tsamp = np.empty((max_nodes, 7, d), dtype=np.float64)
    tvis = np.empty((max_nodes, 7), dtype=np.uint8)
    gub_arr = np.empty(max_nodes, dtype=np.float64)
    glb_arr = np.empty(max_nodes, dtype=np.float64)
    state = np.zeros(max_nodes, dtype=np.uint8)
    is_leaf = np.zeros(max_nodes, dtype=np.uint8)
    children = np.full((max_nodes, 4), -1, dtype=np.int32)
    width = np.empty(max_nodes, dtype=np.float64)

    root_ids = np.empty(_ICO_F.shape[0], dtype=np.int32)
    tau_min_scratch = np.empty(obs.shape[0], dtype=np.float64)
    tau_max_scratch = np.empty(obs.shape[0], dtype=np.float64)
    tau_tmp_scratch = np.empty(obs.shape[0], dtype=np.float64)

    if obs.shape[0] == 2:
        status_n2, reason_n2, _prf_lo, _prf_hi, _T_lo, T_hi = (
            _n2_certified_prf_interval(
                obs,
                eps[0],
                max_nodes,
                inva2,
                invb2,
                L_u_to_x,
                inv_c,
                L_g,
                tri_u,
                tlo,
                thi,
                tsamp,
                tvis,
                gub_arr,
                glb_arr,
                state,
                is_leaf,
                children,
                width,
                root_ids,
                tau_min_scratch,
                tau_max_scratch,
                tau_tmp_scratch,
                0.0,
                pri_s,
            )
        )
        if status_n2 == STATUS_INCONCLUSIVE:
            _raise_wrap_inconclusive(reason_n2)
        return True if pri_s > T_hi else False

    pair_a = np.empty(max_heap_pairs, dtype=np.int32)
    pair_b = np.empty(max_heap_pairs, dtype=np.int32)
    pair_key = np.empty(max_heap_pairs, dtype=np.float64)

    status, reason = _pri_wrap_predicate(
        obs,
        eps,
        ps,
        bdiff,
        pri_s,
        max_nodes,
        max_pair_pops,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        pair_a,
        pair_b,
        pair_key,
        root_ids,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    if status == STATUS_INCONCLUSIVE:
        _raise_wrap_inconclusive(reason)
    return True if status == STATUS_SAFE else False


@njit(cache=True)
def network_unambiguous_prf(
    obs_ecef_m,
    sigma_t_s,
    tol_prf_hz=0.1,
    ref_idx=0,
    k_sigma=1.0,
    margin_mode=0,
    pri_low_s=1e-8,
    pri_high_s=5e-3,
    pri_big_s=10.0,
    max_bisect_iters=80,
    max_nodes=1_000_000,
    max_pair_pops=2_000_000,
    max_heap_pairs=40_000_000,
):
    """
    Certified critical PRF interval for wrap-induced network ambiguity.

    Returns
    -------
    tuple(float, float, float, float, float, float)
        (prf_lower_hz, prf_upper_hz, prf_gap_hz, pri_upper_us, pri_lower_us, pri_gap_us)

    Notes
    -----
    Intrinsic (n=0) ambiguities are allowed by design. The returned critical PRF
    only addresses additional ambiguity introduced by integer wrapping.

    Raises
    ------
    ValueError
        Raised with "certification inconclusive: increase internal caps" if
        internal resource limits are hit during certification.
    """
    if tol_prf_hz <= 0.0:
        raise ValueError("tol_prf_hz must be > 0.")
    if pri_low_s <= 0.0 or pri_high_s <= 0.0 or pri_big_s <= 0.0:
        raise ValueError("PRI bounds must be > 0.")
    if pri_low_s >= pri_high_s:
        raise ValueError("pri_low_s must be < pri_high_s.")
    if max_bisect_iters <= 0:
        raise ValueError("max_bisect_iters must be > 0.")
    if max_nodes < _ICO_F.shape[0]:
        raise ValueError("max_nodes too small.")
    if max_pair_pops <= 0:
        raise ValueError("max_pair_pops must be > 0.")
    if max_heap_pairs <= 0:
        raise ValueError("max_heap_pairs must be > 0.")

    if obs_ecef_m.ndim != 2 or obs_ecef_m.shape[1] != 3:
        raise ValueError("obs_ecef_m must have shape (N,3).")
    if sigma_t_s.ndim != 1 or sigma_t_s.shape[0] != obs_ecef_m.shape[0]:
        raise ValueError("sigma_t_s must have shape (N,).")
    if obs_ecef_m.shape[0] < 2:
        raise ValueError("Need at least 2 observers.")
    if ref_idx < 0 or ref_idx >= obs_ecef_m.shape[0]:
        raise ValueError("ref_idx out of range.")
    if k_sigma < 0.0:
        raise ValueError("k_sigma must be >= 0.")
    for i in range(sigma_t_s.shape[0]):
        if sigma_t_s[i] < 0.0:
            raise ValueError("sigma_t_s must be non-negative.")

    obs = np.empty((obs_ecef_m.shape[0], 3), dtype=np.float64)
    for i in range(obs_ecef_m.shape[0]):
        obs[i, 0] = obs_ecef_m[i, 0]
        obs[i, 1] = obs_ecef_m[i, 1]
        obs[i, 2] = obs_ecef_m[i, 2]
    sig = np.empty(sigma_t_s.shape[0], dtype=np.float64)
    for i in range(sigma_t_s.shape[0]):
        sig[i] = sigma_t_s[i]
    obs, sig = _reorder_ref_first(obs, sig, ref_idx)

    # WGS84 and constants.
    a = 6378137.0
    b = 6356752.3142451793
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)
    inv_c = 1.0 / 299792458.0
    L_u_to_x = (a * a * a) / (b * b)

    eps, ps, L_g = _prepare_eps_and_ps(obs, sig, k_sigma, margin_mode, inva2, invb2)
    bdiff = _prepare_bdiff(obs, inv_c)
    d = obs.shape[0] - 1

    # Workspace reused across bisection calls (each predicate call rebuilds roots).
    tri_u = np.empty((max_nodes, 3, 3), dtype=np.float64)
    tlo = np.empty((max_nodes, d), dtype=np.float64)
    thi = np.empty((max_nodes, d), dtype=np.float64)
    tsamp = np.empty((max_nodes, 7, d), dtype=np.float64)
    tvis = np.empty((max_nodes, 7), dtype=np.uint8)
    gub_arr = np.empty(max_nodes, dtype=np.float64)
    glb_arr = np.empty(max_nodes, dtype=np.float64)
    state = np.zeros(max_nodes, dtype=np.uint8)
    is_leaf = np.zeros(max_nodes, dtype=np.uint8)
    children = np.full((max_nodes, 4), -1, dtype=np.int32)
    width = np.empty(max_nodes, dtype=np.float64)

    root_ids = np.empty(_ICO_F.shape[0], dtype=np.int32)
    tau_min_scratch = np.empty(obs.shape[0], dtype=np.float64)
    tau_max_scratch = np.empty(obs.shape[0], dtype=np.float64)
    tau_tmp_scratch = np.empty(obs.shape[0], dtype=np.float64)

    if obs.shape[0] == 2:
        status_n2, reason_n2, prf_lower_hz, prf_upper_hz, T_lo, T_hi = (
            _n2_certified_prf_interval(
                obs,
                eps[0],
                max_nodes,
                inva2,
                invb2,
                L_u_to_x,
                inv_c,
                L_g,
                tri_u,
                tlo,
                thi,
                tsamp,
                tvis,
                gub_arr,
                glb_arr,
                state,
                is_leaf,
                children,
                width,
                root_ids,
                tau_min_scratch,
                tau_max_scratch,
                tau_tmp_scratch,
                tol_prf_hz,
                -1.0,
            )
        )
        if status_n2 == STATUS_INCONCLUSIVE:
            _raise_cert_inconclusive(reason_n2)
        prf_gap_hz = prf_upper_hz - prf_lower_hz
        if np.isinf(prf_upper_hz) and np.isinf(prf_lower_hz):
            prf_gap_hz = 0.0
        return (
            prf_lower_hz,
            prf_upper_hz,
            prf_gap_hz,
            T_hi * 1e6,
            T_lo * 1e6,
            (T_hi - T_lo) * 1e6,
        )

    pair_a = np.empty(max_heap_pairs, dtype=np.int32)
    pair_b = np.empty(max_heap_pairs, dtype=np.int32)
    pair_key = np.empty(max_heap_pairs, dtype=np.float64)

    # Find SAFE side.
    T_safe = pri_high_s
    status, reason = _pri_wrap_predicate(
        obs,
        eps,
        ps,
        bdiff,
        T_safe,
        max_nodes,
        max_pair_pops,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        pair_a,
        pair_b,
        pair_key,
        root_ids,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    if status == STATUS_INCONCLUSIVE:
        _raise_cert_inconclusive(reason)
    while (status == STATUS_AMBIG) and (T_safe < pri_big_s):
        T_safe = 2.0 * T_safe
        if T_safe > pri_big_s:
            T_safe = pri_big_s
        status, reason = _pri_wrap_predicate(
            obs,
            eps,
            ps,
            bdiff,
            T_safe,
            max_nodes,
            max_pair_pops,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            pair_a,
            pair_b,
            pair_key,
            root_ids,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if status == STATUS_INCONCLUSIVE:
            _raise_cert_inconclusive(reason)
    if status != STATUS_SAFE:
        raise ValueError("no SAFE PRI up to pri_big_s")

    # Find AMBIGUOUS side.
    T_amb = pri_low_s
    status_low, reason = _pri_wrap_predicate(
        obs,
        eps,
        ps,
        bdiff,
        T_amb,
        max_nodes,
        max_pair_pops,
        inva2,
        invb2,
        L_u_to_x,
        inv_c,
        L_g,
        tri_u,
        tlo,
        thi,
        tsamp,
        tvis,
        gub_arr,
        glb_arr,
        state,
        is_leaf,
        children,
        width,
        pair_a,
        pair_b,
        pair_key,
        root_ids,
        tau_min_scratch,
        tau_max_scratch,
        tau_tmp_scratch,
    )
    if status_low == STATUS_INCONCLUSIVE:
        _raise_cert_inconclusive(reason)
    while status_low == STATUS_SAFE:
        T_amb *= 0.5
        if T_amb < 1e-14:
            return (np.inf, np.inf, 0.0, 0.0, 0.0, 0.0)
        status_low, reason = _pri_wrap_predicate(
            obs,
            eps,
            ps,
            bdiff,
            T_amb,
            max_nodes,
            max_pair_pops,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            pair_a,
            pair_b,
            pair_key,
            root_ids,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if status_low == STATUS_INCONCLUSIVE:
            _raise_cert_inconclusive(reason)

    # Bisection.
    for _ in range(max_bisect_iters):
        T_mid = 0.5 * (T_safe + T_amb)
        status_mid, reason = _pri_wrap_predicate(
            obs,
            eps,
            ps,
            bdiff,
            T_mid,
            max_nodes,
            max_pair_pops,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            L_g,
            tri_u,
            tlo,
            thi,
            tsamp,
            tvis,
            gub_arr,
            glb_arr,
            state,
            is_leaf,
            children,
            width,
            pair_a,
            pair_b,
            pair_key,
            root_ids,
            tau_min_scratch,
            tau_max_scratch,
            tau_tmp_scratch,
        )
        if status_mid == STATUS_INCONCLUSIVE:
            _raise_cert_inconclusive(reason)
        if status_mid == STATUS_SAFE:
            T_safe = T_mid
        else:
            T_amb = T_mid

        prf_lower_hz = 1.0 / T_safe
        prf_upper_hz = 1.0 / T_amb
        prf_gap_hz = prf_upper_hz - prf_lower_hz
        if prf_gap_hz <= tol_prf_hz:
            pri_upper_us = T_safe * 1e6
            pri_lower_us = T_amb * 1e6
            return (
                prf_lower_hz,
                prf_upper_hz,
                prf_gap_hz,
                pri_upper_us,
                pri_lower_us,
                pri_upper_us - pri_lower_us,
            )

    raise ValueError("Tolerance not achieved before max_bisect_iters.")


if __name__ == "__main__":
    # Support direct script execution so numba cache imports can resolve `nebula`.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _repo_root not in sys.path:
        sys.path.append(_repo_root)

    from nebula.transforms import geodetic2ecef

    # Example with overlap: three near-co-located LEO observers (ECEF)
    obs = np.array(
        [
            geodetic2ecef(0.0, 0.0, 10000e3),
            geodetic2ecef(5.0, 25.0, 10000e3),
            geodetic2ecef(-10.0, -15.0, 15000e3),
            geodetic2ecef(1.0, -26.0, 12000e3),
        ],
        dtype=np.float64,
    )

    sig = np.ones_like(obs[:, 0]) * 5000e-9

    try:
        out = network_unambiguous_prf(
            obs,
            sig,
            tol_prf_hz=2.0,
            k_sigma=1.0,
            max_nodes=200_000_000,
            max_pair_pops=200_000_000,
            max_heap_pairs=300_000_000,
        )
        print("PRF_lower_hz (safe):", out[0])
        print("PRF_upper_hz:", out[1])
        print("PRF_gap_hz:", out[2])
        print("PRI_upper_us:", out[3])
        print("PRI_lower_us:", out[4])
        print("PRI_gap_us:", out[5])
    except ValueError as e:
        print(
            "Could not certify wrap-induced PRF interval under current settings:",
            str(e),
        )
