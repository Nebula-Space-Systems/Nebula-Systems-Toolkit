"""
Certified critical PRF (no-wrap TDOA criterion) over the WGS84 ellipsoid surface.

This solver computes a certified interval for the critical PRF over the common-visible
surface footprint of all observers, then returns a conservative safe PRF.

Definitions
-----------
Observers: p_i in ECEF meters.

Propagation delays:
    tau_i(x) = ||x - p_i|| / c

Inter-observer delay spread:
    W(x) = max_i tau_i(x) - min_i tau_i(x)

No-wrap condition over region R:
    W(x) + margin < PRI/2,  for all x in R
where R is the WGS84 surface region visible to all observers (ellipsoid-occluded LOS).

We define:
    F(x) = W(x) + margin
    F*   = max_{x in R} F(x)
Then:
    PRI_critical = 2*F*
    PRF_critical = 1/PRI_critical

Returned PRF interval
---------------------
The solver maintains certified bounds F_L <= F* <= F_U and returns:
    prf_lower_hz : conservative safe PRF (optionally strict-padded)
    prf_upper_hz : optimistic upper bound on critical PRF
    prf_gap_hz   : certified PRF interval width
and the corresponding PRI bounds in microseconds.

Uncertainty margin model
------------------------
Let s1 >= s2 be the two largest entries in sigma_t_s.

margin_mode = 0 (default, deterministic hard-bound interpretation):
    margin = k_sigma * (s1 + s2)
If sigma_t_s are per-observer absolute timestamp error bounds and k_sigma=1,
this is a deterministic worst-case TDOA bound.

margin_mode = 1 (probabilistic RSS interpretation):
    margin = k_sigma * sqrt(s1^2 + s2^2)
This treats sigma_t_s as independent 1-sigma timing uncertainties.

Certification details
---------------------
Branch-and-bound is performed on a subdivided icosphere parameterization.

Lipschitz constants in ECEF distance:
    W is Lipschitz with L_F = 2/c.
    g(x) = min_k (p_s,k^T x - 1), p_s = A p, A=diag(1/a^2,1/a^2,1/b^2),
    has L_g = max_k ||p_s,k||.

Cell diameter bound:
    For each direction cell, we compute a conservative ECEF diameter upper bound and use:
        ub_g(C) = max_sample(g) + L_g * diam(C)
        ub_F(C) = max_sample(F) + L_F * diam(C)
    plus a global analytic clamp:
        F(x) <= Bmax/c + margin,  Bmax = max_{i<j} ||p_i - p_j||.

Memory behavior
---------------
The solver preallocates arrays sized by max_nodes, so peak RAM is approximately:
    max_nodes * (9*8 + 8 + 8 + 4) bytes
for tri_u, gub_arr, Fub_arr, and heap_idx.
Pruned cells are recycled via a free-list, which lowers required max_nodes for
many geometries without affecting certification correctness.
The njit solver can also auto-expand max_nodes on MAX_NODES pressure by
growing internal storage and continuing from current state.

Raises ValueError if:
    - certified empty overlap,
    - invalid inputs,
    - or tolerance is not achieved before caps.
"""

import os
import sys

import numpy as np
from numba import njit, prange

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
    # radial intersection along direction u (not necessarily unit)
    s = (ux * ux + uy * uy) * inva2 + (uz * uz) * invb2
    t = 1.0 / np.sqrt(s)
    return t * ux, t * uy, t * uz


@njit(cache=True, inline="always")
def _dot3(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(cache=True)
def _closest_origin_norm_triangle(u0, u1, u2):
    # Closest-point-to-origin on triangle (u0,u1,u2), adapted from
    # "Real-Time Collision Detection" region tests.
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
    # A rigorous direction-space diameter bound for this spherical triangle:
    # S = normalize(conv(u0,u1,u2)).
    # If P = conv(u0,u1,u2) and m = min_{p in P} ||p||, then normalization is
    # (1/m)-Lipschitz on P, so diam(S) <= diam(P)/m.
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

    # Conservative global Lipschitz bound for ellipsoid radial projection map.
    return L_u_to_x * dir_diam


# -----------------------------
# Simple max-heap (no stale entries)
# heap_idx stores node indices; key array provided externally.
# -----------------------------
@njit(cache=True, inline="always")
def _heap_swap(heap_idx, i, j):
    t = heap_idx[i]
    heap_idx[i] = heap_idx[j]
    heap_idx[j] = t


@njit(cache=True)
def _heap_push(heap_idx, heap_size, idx, key):
    i = heap_size
    heap_idx[i] = idx
    heap_size += 1
    while i > 0:
        p = (i - 1) >> 1
        if key[heap_idx[p]] >= key[heap_idx[i]]:
            break
        _heap_swap(heap_idx, p, i)
        i = p
    return heap_size


@njit(cache=True)
def _heap_pop_max(heap_idx, heap_size, key):
    # assumes heap_size > 0
    out = heap_idx[0]
    heap_size -= 1
    heap_idx[0] = heap_idx[heap_size]
    i = 0
    while True:
        l = (i << 1) + 1
        r = l + 1
        if l >= heap_size:
            break
        j = l
        if r < heap_size and key[heap_idx[r]] > key[heap_idx[l]]:
            j = r
        if key[heap_idx[i]] >= key[heap_idx[j]]:
            break
        _heap_swap(heap_idx, i, j)
        i = j
    return out, heap_size


@njit(cache=True)
def _heapify(heap_idx, heap_size, key):
    # in-place heapify (max-heap) in O(n)
    for i in range((heap_size >> 1) - 1, -1, -1):
        j = i
        while True:
            l = (j << 1) + 1
            r = l + 1
            if l >= heap_size:
                break
            k = l
            if r < heap_size and key[heap_idx[r]] > key[heap_idx[l]]:
                k = r
            if key[heap_idx[j]] >= key[heap_idx[k]]:
                break
            _heap_swap(heap_idx, j, k)
            j = k


@njit(cache=True, inline="always")
def _free_push(free_idx, free_size, idx, capacity):
    # free list stored in reverse at the tail of free_idx
    free_idx[capacity - 1 - free_size] = idx
    return free_size + 1


@njit(cache=True, inline="always")
def _free_pop(free_idx, free_size, capacity):
    # assumes free_size > 0
    free_size -= 1
    return free_idx[capacity - 1 - free_size], free_size


@njit(cache=True, inline="always")
def _next_capacity(capacity, growth, hard_cap):
    new_cap = int(capacity * growth)
    if new_cap <= capacity:
        new_cap = capacity + 1
    if new_cap > hard_cap:
        new_cap = hard_cap
    return new_cap


@njit(cache=True)
def _grow_storage(
    tri_u, gub_arr, Fub_arr, idx_arr, capacity, new_capacity, heap_size, free_size
):
    tri_new = np.empty((new_capacity, 3, 3), dtype=np.float64)
    gub_new = np.full(new_capacity, -1e300, dtype=np.float64)
    Fub_new = np.full(new_capacity, -1e300, dtype=np.float64)
    idx_new = np.empty(new_capacity, dtype=np.int32)

    # Copy node geometry and bounds for existing slots.
    for i in range(capacity):
        tri_new[i, 0, 0] = tri_u[i, 0, 0]
        tri_new[i, 0, 1] = tri_u[i, 0, 1]
        tri_new[i, 0, 2] = tri_u[i, 0, 2]
        tri_new[i, 1, 0] = tri_u[i, 1, 0]
        tri_new[i, 1, 1] = tri_u[i, 1, 1]
        tri_new[i, 1, 2] = tri_u[i, 1, 2]
        tri_new[i, 2, 0] = tri_u[i, 2, 0]
        tri_new[i, 2, 1] = tri_u[i, 2, 1]
        tri_new[i, 2, 2] = tri_u[i, 2, 2]
        gub_new[i] = gub_arr[i]
        Fub_new[i] = Fub_arr[i]

    # Copy active heap prefix unchanged.
    for i in range(heap_size):
        idx_new[i] = idx_arr[i]

    # Copy free-list tail so pop order is preserved.
    for i in range(free_size):
        idx_new[new_capacity - 1 - i] = idx_arr[capacity - 1 - i]

    return tri_new, gub_new, Fub_new, idx_new


# -----------------------------
# Core per-sample evaluations
# -----------------------------
@njit(cache=True, inline="always")
def _delay_spread_seconds(x, y, z, obs, nobs, inv_c):
    tmin = 1e300
    tmax = -1e300
    for k in range(nobs):
        dx = x - obs[k, 0]
        dy = y - obs[k, 1]
        dz = z - obs[k, 2]
        r = _norm3(dx, dy, dz)
        tau = r * inv_c
        if tau < tmin:
            tmin = tau
        if tau > tmax:
            tmax = tau
    return tmax - tmin


@njit(cache=True, inline="always")
def _g_visibility_margin(x, y, z, ps, nobs):
    # g(x) = min_k (ps_k^T x - 1)
    g = 1e300
    for k in range(nobs):
        v = ps[k, 0] * x + ps[k, 1] * y + ps[k, 2] * z - 1.0
        if v < g:
            g = v
    return g


@njit(cache=True, inline="always")
def _margin_from_sigmas(sigma_t_s, nobs, k_sigma, margin_mode):
    s1 = 0.0
    s2 = 0.0
    for i in range(nobs):
        s = sigma_t_s[i]
        if s >= s1:
            s2 = s1
            s1 = s
        elif s > s2:
            s2 = s

    if margin_mode == 0:
        # Deterministic hard-bound interpretation.
        return k_sigma * (s1 + s2)
    if margin_mode == 1:
        # Probabilistic RSS interpretation (independent 1-sigma errors).
        return k_sigma * np.sqrt(s1 * s1 + s2 * s2)
    raise ValueError("margin_mode must be 0 (hard) or 1 (rss_1sigma).")


@njit(cache=True)
def _eval_cell(
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
    margin,
    L_F,
    L_g,
    F_cap,
):
    diam = _cell_diameter_bound(u0, u1, u2, L_u_to_x)

    # 7-point sampling: vertices + 3 midpoints + center direction.
    m01x, m01y, m01z = _mid_dir(u0[0], u0[1], u0[2], u1[0], u1[1], u1[2])
    m12x, m12y, m12z = _mid_dir(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
    m20x, m20y, m20z = _mid_dir(u2[0], u2[1], u2[2], u0[0], u0[1], u0[2])
    ccx, ccy, ccz = _normalize3(
        u0[0] + u1[0] + u2[0],
        u0[1] + u1[1] + u2[1],
        u0[2] + u1[2] + u2[2],
    )

    # Evaluate g on all 7 samples and compute certified possible-visibility upper bound.
    g_max = -1e300

    # Sample 0: u0
    sx0, sy0, sz0 = _ellipsoid_intersect_dir(u0[0], u0[1], u0[2], inva2, invb2)
    g0 = _g_visibility_margin(sx0, sy0, sz0, ps, nobs)
    if g0 > g_max:
        g_max = g0

    # Sample 1: u1
    sx1, sy1, sz1 = _ellipsoid_intersect_dir(u1[0], u1[1], u1[2], inva2, invb2)
    g1 = _g_visibility_margin(sx1, sy1, sz1, ps, nobs)
    if g1 > g_max:
        g_max = g1

    # Sample 2: u2
    sx2, sy2, sz2 = _ellipsoid_intersect_dir(u2[0], u2[1], u2[2], inva2, invb2)
    g2 = _g_visibility_margin(sx2, sy2, sz2, ps, nobs)
    if g2 > g_max:
        g_max = g2

    # Sample 3: m01
    sx3, sy3, sz3 = _ellipsoid_intersect_dir(m01x, m01y, m01z, inva2, invb2)
    g3 = _g_visibility_margin(sx3, sy3, sz3, ps, nobs)
    if g3 > g_max:
        g_max = g3

    # Sample 4: m12
    sx4, sy4, sz4 = _ellipsoid_intersect_dir(m12x, m12y, m12z, inva2, invb2)
    g4 = _g_visibility_margin(sx4, sy4, sz4, ps, nobs)
    if g4 > g_max:
        g_max = g4

    # Sample 5: m20
    sx5, sy5, sz5 = _ellipsoid_intersect_dir(m20x, m20y, m20z, inva2, invb2)
    g5 = _g_visibility_margin(sx5, sy5, sz5, ps, nobs)
    if g5 > g_max:
        g_max = g5

    # Sample 6: center
    sx6, sy6, sz6 = _ellipsoid_intersect_dir(ccx, ccy, ccz, inva2, invb2)
    g6 = _g_visibility_margin(sx6, sy6, sz6, ps, nobs)
    if g6 > g_max:
        g_max = g6

    gub = g_max + L_g * diam
    if gub < 0.0:
        return diam, gub, -1e300, -1e300

    # Compute F on all 7 samples, and Fvis on those with g>=0.
    F_max_all = -1e300
    Fvis = -1e300

    W = _delay_spread_seconds(sx0, sy0, sz0, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g0 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx1, sy1, sz1, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g1 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx2, sy2, sz2, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g2 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx3, sy3, sz3, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g3 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx4, sy4, sz4, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g4 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx5, sy5, sz5, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g5 >= 0.0 and F > Fvis:
        Fvis = F

    W = _delay_spread_seconds(sx6, sy6, sz6, obs, nobs, inv_c)
    F = W + margin
    if F > F_max_all:
        F_max_all = F
    if g6 >= 0.0 and F > Fvis:
        Fvis = F

    Fub = F_max_all + L_F * diam
    if Fub > F_cap:
        Fub = F_cap

    return diam, gub, Fvis, Fub


# -----------------------------
# Batch subdivision + evaluation (parallel)
# Parents are popped from the heap (so no duplicates exist in heap).
# Each parent idx is overwritten by child0; children 1..3 are appended contiguously.
# -----------------------------
@njit(cache=True, parallel=True)
def _subdivide_and_eval_batch(
    parents,
    pcount,
    child_slot1,
    child_slot2,
    child_slot3,
    tri_u,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    L_u_to_x,
    inv_c,
    margin,
    L_F,
    L_g,
    F_cap,
    gub_arr,
    Fub_arr,
    out_child_idx,
    out_child_ok,
    out_child_Fvis,
):
    for k in prange(pcount):
        idx = parents[k]
        u0 = tri_u[idx, 0]
        u1 = tri_u[idx, 1]
        u2 = tri_u[idx, 2]

        u01x, u01y, u01z = _mid_dir(u0[0], u0[1], u0[2], u1[0], u1[1], u1[2])
        u12x, u12y, u12z = _mid_dir(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
        u20x, u20y, u20z = _mid_dir(u2[0], u2[1], u2[2], u0[0], u0[1], u0[2])

        c1 = child_slot1[k]
        c2 = child_slot2[k]
        c3 = child_slot3[k]

        # child0 overwrites idx: (u0, u01, u20)
        tri_u[idx, 0, 0], tri_u[idx, 0, 1], tri_u[idx, 0, 2] = u0[0], u0[1], u0[2]
        tri_u[idx, 1, 0], tri_u[idx, 1, 1], tri_u[idx, 1, 2] = u01x, u01y, u01z
        tri_u[idx, 2, 0], tri_u[idx, 2, 1], tri_u[idx, 2, 2] = u20x, u20y, u20z

        # child1: (u1, u12, u01)
        tri_u[c1, 0, 0], tri_u[c1, 0, 1], tri_u[c1, 0, 2] = u1[0], u1[1], u1[2]
        tri_u[c1, 1, 0], tri_u[c1, 1, 1], tri_u[c1, 1, 2] = u12x, u12y, u12z
        tri_u[c1, 2, 0], tri_u[c1, 2, 1], tri_u[c1, 2, 2] = u01x, u01y, u01z

        # child2: (u2, u20, u12)
        tri_u[c2, 0, 0], tri_u[c2, 0, 1], tri_u[c2, 0, 2] = u2[0], u2[1], u2[2]
        tri_u[c2, 1, 0], tri_u[c2, 1, 1], tri_u[c2, 1, 2] = u20x, u20y, u20z
        tri_u[c2, 2, 0], tri_u[c2, 2, 1], tri_u[c2, 2, 2] = u12x, u12y, u12z

        # child3: (u01, u12, u20)
        tri_u[c3, 0, 0], tri_u[c3, 0, 1], tri_u[c3, 0, 2] = u01x, u01y, u01z
        tri_u[c3, 1, 0], tri_u[c3, 1, 1], tri_u[c3, 1, 2] = u12x, u12y, u12z
        tri_u[c3, 2, 0], tri_u[c3, 2, 1], tri_u[c3, 2, 2] = u20x, u20y, u20z

        # Evaluate 4 children
        # child0
        _, gub, Fvis, Fub = _eval_cell(
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
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[idx] = gub
        out_child_Fvis[4 * k + 0] = Fvis
        Fub_arr[idx] = Fub
        out_child_idx[4 * k + 0] = idx
        out_child_ok[4 * k + 0] = 1 if gub >= 0.0 else 0

        # child1
        _, gub, Fvis, Fub = _eval_cell(
            tri_u[c1, 0],
            tri_u[c1, 1],
            tri_u[c1, 2],
            obs,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[c1] = gub
        out_child_Fvis[4 * k + 1] = Fvis
        Fub_arr[c1] = Fub
        out_child_idx[4 * k + 1] = c1
        out_child_ok[4 * k + 1] = 1 if gub >= 0.0 else 0

        # child2
        _, gub, Fvis, Fub = _eval_cell(
            tri_u[c2, 0],
            tri_u[c2, 1],
            tri_u[c2, 2],
            obs,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[c2] = gub
        out_child_Fvis[4 * k + 2] = Fvis
        Fub_arr[c2] = Fub
        out_child_idx[4 * k + 2] = c2
        out_child_ok[4 * k + 2] = 1 if gub >= 0.0 else 0

        # child3
        _, gub, Fvis, Fub = _eval_cell(
            tri_u[c3, 0],
            tri_u[c3, 1],
            tri_u[c3, 2],
            obs,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[c3] = gub
        out_child_Fvis[4 * k + 3] = Fvis
        Fub_arr[c3] = Fub
        out_child_idx[4 * k + 3] = c3
        out_child_ok[4 * k + 3] = 1 if gub >= 0.0 else 0


# -----------------------------
# Top-level certified solver (njit)
# -----------------------------
@njit(cache=True)
def max_unambiguous_prf(
    obs_ecef_m: np.ndarray,
    sigma_t_s: np.ndarray,
    tol_prf_hz: float = 0.1,
    k_sigma: float = 1.0,
    max_nodes: int = 5_000_000,
    max_iters: int = 1_000_000,
    margin_mode: int = 0,
    strict_pri_pad_s: float = 1e-12,
    auto_expand_max_nodes: bool = True,
    max_nodes_growth: float = 2.0,
    max_nodes_hard_cap: int = 2_147_483_647,
):
    # WGS84
    a = 6378137.0
    b = 6356752.3142451793
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)

    # speed of light
    c0 = 299792458.0
    inv_c = 1.0 / c0

    nobs = obs_ecef_m.shape[0]
    if nobs < 2:
        raise ValueError("Need at least 2 observers.")

    if tol_prf_hz <= 0.0:
        raise ValueError("tol_prf_hz must be > 0.")
    if k_sigma < 0.0:
        raise ValueError("k_sigma must be >= 0.")
    if strict_pri_pad_s < 0.0:
        raise ValueError("strict_pri_pad_s must be >= 0.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be > 0.")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")
    if auto_expand_max_nodes and max_nodes_growth <= 1.0:
        raise ValueError("max_nodes_growth must be > 1 when auto expansion is enabled.")

    margin = _margin_from_sigmas(sigma_t_s, nobs, k_sigma, margin_mode)

    # ps = A p and L_g = max ||ps||
    ps = np.empty((nobs, 3), dtype=np.float64)
    L_g = 0.0
    for k in range(nobs):
        px, py, pz = obs_ecef_m[k, 0], obs_ecef_m[k, 1], obs_ecef_m[k, 2]
        sx = px * inva2
        sy = py * inva2
        sz = pz * invb2
        ps[k, 0] = sx
        ps[k, 1] = sy
        ps[k, 2] = sz
        nrm = _norm3(sx, sy, sz)
        if nrm > L_g:
            L_g = nrm

    # Lipschitz for F = W + margin: L_F = 2/c
    L_F = 2.0 * inv_c

    # Conservative global Lipschitz for radial projection u -> x(u) on WGS84.
    # For x(u)=r(u)u with r(u) in [b,a]:
    #   ||x(u)-x(v)|| <= a||u-v|| + |r(u)-r(v)|
    # and since r depends only on u_z^2 for WGS84 (a=a_eq):
    #   |r(u)-r(v)| <= (a^3/b^2 - a) ||u-v||.
    # So a valid global bound is:
    #   ||x(u)-x(v)|| <= (a^3/b^2) ||u-v||.
    L_u_to_x = (a * a * a) / (b * b)

    # Certified tolerance is on PRF (Hz) now:
    # stop when PRF_upper - PRF_lower <= tol_prf_hz,
    # where PRF_lower = 1/PRI_upper and PRF_upper = 1/PRI_lower.

    # Analytic clamp: W(x) <= Bmax/c for all x
    Bmax = 0.0
    for i in range(nobs):
        for j in range(i + 1, nobs):
            dx = obs_ecef_m[i, 0] - obs_ecef_m[j, 0]
            dy = obs_ecef_m[i, 1] - obs_ecef_m[j, 1]
            dz = obs_ecef_m[i, 2] - obs_ecef_m[j, 2]
            d = _norm3(dx, dy, dz)
            if d > Bmax:
                Bmax = d
    F_cap = Bmax * inv_c + margin

    # Caps (raise if exceeded)
    hard_cap = max_nodes_hard_cap
    if hard_cap > 2_147_483_647:
        hard_cap = 2_147_483_647
    if hard_cap < max_nodes:
        raise ValueError("max_nodes_hard_cap must be >= max_nodes.")

    capacity = max_nodes
    MAX_ITERS = max_iters
    BATCH = 2048  # increase for higher CPU utilization

    # Node storage
    tri_u = np.empty((capacity, 3, 3), dtype=np.float64)
    gub_arr = np.full(capacity, -1e300, dtype=np.float64)
    Fub_arr = np.full(capacity, -1e300, dtype=np.float64)

    # Single heap array; key depends on phase
    heap_idx = np.empty(capacity, dtype=np.int32)
    heap_size = 0
    free_size = 0
    next_free = 0

    # Bounds on F*
    F_L = -1e300
    found_feasible = False

    # Phase A key uses gub_arr; Phase B uses Fub_arr
    # Initialize with 20 base faces and insert those with gub>=0
    for f in range(_ICO_F.shape[0]):
        if next_free >= capacity:
            if not auto_expand_max_nodes:
                raise ValueError("max_nodes is too small for base mesh initialization.")
            new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
            if new_capacity <= capacity:
                raise ValueError("max_nodes is too small for base mesh initialization.")
            tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                tri_u,
                gub_arr,
                Fub_arr,
                heap_idx,
                capacity,
                new_capacity,
                heap_size,
                free_size,
            )
            capacity = new_capacity

        i0 = _ICO_F[f, 0]
        i1 = _ICO_F[f, 1]
        i2 = _ICO_F[f, 2]

        tri_u[next_free, 0, 0] = _ICO_V[i0, 0]
        tri_u[next_free, 0, 1] = _ICO_V[i0, 1]
        tri_u[next_free, 0, 2] = _ICO_V[i0, 2]
        tri_u[next_free, 1, 0] = _ICO_V[i1, 0]
        tri_u[next_free, 1, 1] = _ICO_V[i1, 1]
        tri_u[next_free, 1, 2] = _ICO_V[i1, 2]
        tri_u[next_free, 2, 0] = _ICO_V[i2, 0]
        tri_u[next_free, 2, 1] = _ICO_V[i2, 1]
        tri_u[next_free, 2, 2] = _ICO_V[i2, 2]

        _, gub, Fvis, Fub = _eval_cell(
            tri_u[next_free, 0],
            tri_u[next_free, 1],
            tri_u[next_free, 2],
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[next_free] = gub
        Fub_arr[next_free] = Fub

        if gub >= 0.0:
            heap_size = _heap_push(heap_idx, heap_size, next_free, gub_arr)
            if Fvis > F_L:
                F_L = Fvis
            if Fvis > -1e200:  # found any visible sample
                found_feasible = True
        else:
            free_size = _free_push(heap_idx, free_size, next_free, capacity)

        next_free += 1

    if heap_size == 0:
        raise ValueError("No overlapping visibility (certified empty).")

    # Workspace
    parents = np.empty(BATCH, dtype=np.int32)
    child_slot1 = np.empty(BATCH, dtype=np.int32)
    child_slot2 = np.empty(BATCH, dtype=np.int32)
    child_slot3 = np.empty(BATCH, dtype=np.int32)
    child_idx = np.empty(4 * BATCH, dtype=np.int32)
    child_ok = np.empty(4 * BATCH, dtype=np.int8)
    child_Fvis = np.empty(4 * BATCH, dtype=np.float64)

    # -----------------
    # Phase A: find any feasible visible sample (g>=0) or certify empty
    # Heap keyed by gub_arr
    # -----------------
    it = 0
    while (not found_feasible) and it < MAX_ITERS:
        it += 1

        # If best possible g upper bound is < 0, empty intersection certified.
        top = heap_idx[0]
        if gub_arr[top] < 0.0:
            raise ValueError("No overlapping visibility (certified empty).")

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, gub_arr)
            parents[k] = idx

        # Allocate 3 side-child slots per parent, preferring recycled slots.
        for k in range(pcount):
            if free_size > 0:
                c1, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c1 = next_free
                next_free += 1

            if free_size > 0:
                c2, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c2 = next_free
                next_free += 1

            if free_size > 0:
                c3, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c3 = next_free
                next_free += 1

            child_slot1[k] = c1
            child_slot2[k] = c2
            child_slot3[k] = c3

        _subdivide_and_eval_batch(
            parents,
            pcount,
            child_slot1,
            child_slot2,
            child_slot3,
            tri_u,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
            gub_arr,
            Fub_arr,
            child_idx,
            child_ok,
            child_Fvis,
        )

        # Push children back into heap (still keyed by gub)
        for q in range(4 * pcount):
            nid = child_idx[q]
            if child_ok[q] == 1:
                heap_size = _heap_push(heap_idx, heap_size, nid, gub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv
                if fv > -1e200:
                    found_feasible = True
            else:
                free_size = _free_push(heap_idx, free_size, nid, capacity)

        if heap_size == 0:
            raise ValueError("No overlapping visibility (certified empty).")

    if not found_feasible:
        raise ValueError("Failed to find feasible overlap before MAX_ITERS.")

    # -----------------
    # Phase B: tighten F bounds to tolerance
    # Rebuild heap keyed by Fub_arr.
    # -----------------
    _heapify(heap_idx, heap_size, Fub_arr)

    # Certified upper bound is max ub_F among active cells
    # Stop when F_U - F_L <= eps_F
    it = 0
    while it < MAX_ITERS:
        it += 1

        if heap_size == 0:
            raise ValueError("No overlapping visibility (no possible cells remain).")

        F_U = Fub_arr[heap_idx[0]]

        # Compute certified PRI and PRF bounds each iteration.
        pri_lower_s = 2.0 * F_L
        pri_upper_s = 2.0 * F_U

        # If pri_lower_s <= 0, PRF is unbounded (or numerically unstable).
        # In that case we cannot certify a finite PRF tolerance.
        if pri_lower_s <= 0.0:
            raise ValueError("PRI lower bound <= 0; cannot certify PRF tolerance.")

        prf_lower_cert_hz = 1.0 / pri_upper_s
        prf_lower_hz = 1.0 / (pri_upper_s + strict_pri_pad_s)  # strict-safe
        prf_upper_hz = 1.0 / pri_lower_s  # optimistic upper bound
        prf_gap_hz = prf_upper_hz - prf_lower_cert_hz

        if prf_gap_hz <= tol_prf_hz:
            pri_lower_us = pri_lower_s * 1e6
            pri_upper_us = pri_upper_s * 1e6
            pri_gap_us = pri_upper_us - pri_lower_us
            return (
                prf_lower_hz,
                prf_upper_hz,
                prf_gap_hz,
                pri_upper_us,
                pri_lower_us,
                pri_gap_us,
            )

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, Fub_arr)
            parents[k] = idx

        # Allocate 3 side-child slots per parent, preferring recycled slots.
        for k in range(pcount):
            if free_size > 0:
                c1, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c1 = next_free
                next_free += 1

            if free_size > 0:
                c2, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c2 = next_free
                next_free += 1

            if free_size > 0:
                c3, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c3 = next_free
                next_free += 1

            child_slot1[k] = c1
            child_slot2[k] = c2
            child_slot3[k] = c3

        _subdivide_and_eval_batch(
            parents,
            pcount,
            child_slot1,
            child_slot2,
            child_slot3,
            tri_u,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
            gub_arr,
            Fub_arr,
            child_idx,
            child_ok,
            child_Fvis,
        )

        # Insert possibly-visible children into heap (keyed by Fub)
        for q in range(4 * pcount):
            nid = child_idx[q]
            if child_ok[q] == 1:
                heap_size = _heap_push(heap_idx, heap_size, nid, Fub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv
            else:
                free_size = _free_push(heap_idx, free_size, nid, capacity)

    raise ValueError("Tolerance not achieved before MAX_ITERS.")


@njit(cache=True)
def _max_unambiguous_prf_reuse_workspace(
    obs_ecef_m: np.ndarray,
    sigma_t_s: np.ndarray,
    tol_prf_hz: float,
    k_sigma: float,
    max_iters: int,
    margin_mode: int,
    strict_pri_pad_s: float,
    auto_expand_max_nodes: bool,
    max_nodes_growth: float,
    max_nodes_hard_cap: int,
    use_warm_bounds: bool,
    warm_F_lower: float,
    warm_F_upper: float,
    warm_cell_valid: bool,
    warm_cell_u: np.ndarray,
    tri_u: np.ndarray,
    gub_arr: np.ndarray,
    Fub_arr: np.ndarray,
    heap_idx: np.ndarray,
    parents: np.ndarray,
    child_slot1: np.ndarray,
    child_slot2: np.ndarray,
    child_slot3: np.ndarray,
    child_idx: np.ndarray,
    child_ok: np.ndarray,
    child_Fvis: np.ndarray,
):
    # WGS84
    a = 6378137.0
    b = 6356752.3142451793
    inva2 = 1.0 / (a * a)
    invb2 = 1.0 / (b * b)

    # speed of light
    c0 = 299792458.0
    inv_c = 1.0 / c0

    nobs = obs_ecef_m.shape[0]
    if nobs < 2:
        raise ValueError("Need at least 2 observers.")

    if tol_prf_hz <= 0.0:
        raise ValueError("tol_prf_hz must be > 0.")
    if k_sigma < 0.0:
        raise ValueError("k_sigma must be >= 0.")
    if strict_pri_pad_s < 0.0:
        raise ValueError("strict_pri_pad_s must be >= 0.")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")
    if auto_expand_max_nodes and max_nodes_growth <= 1.0:
        raise ValueError("max_nodes_growth must be > 1 when auto expansion is enabled.")
    if sigma_t_s.shape[0] != nobs:
        raise ValueError("sigma_t_s length must match number of observers.")

    capacity = tri_u.shape[0]
    if capacity <= 0:
        raise ValueError("Workspace capacity must be > 0.")
    if gub_arr.shape[0] != capacity or Fub_arr.shape[0] != capacity:
        raise ValueError("Workspace bound arrays must match tri_u capacity.")
    if heap_idx.shape[0] != capacity:
        raise ValueError("Workspace heap array must match tri_u capacity.")

    BATCH = parents.shape[0]
    if BATCH <= 0:
        raise ValueError("Workspace batch size must be > 0.")
    if (
        child_slot1.shape[0] != BATCH
        or child_slot2.shape[0] != BATCH
        or child_slot3.shape[0] != BATCH
    ):
        raise ValueError("Workspace child slot arrays must match parents length.")
    if (
        child_idx.shape[0] < 4 * BATCH
        or child_ok.shape[0] < 4 * BATCH
        or child_Fvis.shape[0] < 4 * BATCH
    ):
        raise ValueError("Workspace child buffers must have length >= 4*BATCH.")

    margin = _margin_from_sigmas(sigma_t_s, nobs, k_sigma, margin_mode)

    # ps = A p and L_g = max ||ps||
    ps = np.empty((nobs, 3), dtype=np.float64)
    L_g = 0.0
    for k in range(nobs):
        px, py, pz = obs_ecef_m[k, 0], obs_ecef_m[k, 1], obs_ecef_m[k, 2]
        sx = px * inva2
        sy = py * inva2
        sz = pz * invb2
        ps[k, 0] = sx
        ps[k, 1] = sy
        ps[k, 2] = sz
        nrm = _norm3(sx, sy, sz)
        if nrm > L_g:
            L_g = nrm

    # Lipschitz for F = W + margin: L_F = 2/c
    L_F = 2.0 * inv_c

    # Conservative global Lipschitz for radial projection u -> x(u) on WGS84.
    L_u_to_x = (a * a * a) / (b * b)

    # Analytic clamp: W(x) <= Bmax/c for all x
    Bmax = 0.0
    for i in range(nobs):
        for j in range(i + 1, nobs):
            dx = obs_ecef_m[i, 0] - obs_ecef_m[j, 0]
            dy = obs_ecef_m[i, 1] - obs_ecef_m[j, 1]
            dz = obs_ecef_m[i, 2] - obs_ecef_m[j, 2]
            d = _norm3(dx, dy, dz)
            if d > Bmax:
                Bmax = d
    F_cap = Bmax * inv_c + margin
    if use_warm_bounds and warm_F_upper > 0.0 and warm_F_upper < F_cap:
        # Certified temporal upper bound from previous time step.
        F_cap = warm_F_upper

    # Caps (raise if exceeded)
    hard_cap = max_nodes_hard_cap
    if hard_cap > 2_147_483_647:
        hard_cap = 2_147_483_647
    if hard_cap < capacity:
        raise ValueError("max_nodes_hard_cap must be >= workspace capacity.")

    MAX_ITERS = max_iters

    # Per-solve state (workspace buffers are reused, not reallocated).
    heap_size = 0
    free_size = 0
    next_free = 0

    # Bounds on F*
    F_L = -1e300
    if use_warm_bounds and warm_F_lower > F_L:
        # Certified temporal lower bound from previous time step.
        F_L = warm_F_lower
    found_feasible = False

    if warm_cell_valid:
        # Evaluate prior high-priority cell immediately to seed a strong feasible lower bound.
        _, _, Fvis_warm, _ = _eval_cell(
            warm_cell_u[0],
            warm_cell_u[1],
            warm_cell_u[2],
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        if Fvis_warm > F_L:
            F_L = Fvis_warm
        if Fvis_warm > -1e200:
            found_feasible = True

    # Phase A key uses gub_arr; Phase B uses Fub_arr
    # Initialize with 20 base faces and insert those with gub>=0
    for f in range(_ICO_F.shape[0]):
        if next_free >= capacity:
            if not auto_expand_max_nodes:
                raise ValueError("max_nodes is too small for base mesh initialization.")
            new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
            if new_capacity <= capacity:
                raise ValueError("max_nodes is too small for base mesh initialization.")
            tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                tri_u,
                gub_arr,
                Fub_arr,
                heap_idx,
                capacity,
                new_capacity,
                heap_size,
                free_size,
            )
            capacity = new_capacity

        i0 = _ICO_F[f, 0]
        i1 = _ICO_F[f, 1]
        i2 = _ICO_F[f, 2]

        tri_u[next_free, 0, 0] = _ICO_V[i0, 0]
        tri_u[next_free, 0, 1] = _ICO_V[i0, 1]
        tri_u[next_free, 0, 2] = _ICO_V[i0, 2]
        tri_u[next_free, 1, 0] = _ICO_V[i1, 0]
        tri_u[next_free, 1, 1] = _ICO_V[i1, 1]
        tri_u[next_free, 1, 2] = _ICO_V[i1, 2]
        tri_u[next_free, 2, 0] = _ICO_V[i2, 0]
        tri_u[next_free, 2, 1] = _ICO_V[i2, 1]
        tri_u[next_free, 2, 2] = _ICO_V[i2, 2]

        _, gub, Fvis, Fub = _eval_cell(
            tri_u[next_free, 0],
            tri_u[next_free, 1],
            tri_u[next_free, 2],
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
        )
        gub_arr[next_free] = gub
        Fub_arr[next_free] = Fub

        if gub >= 0.0:
            heap_size = _heap_push(heap_idx, heap_size, next_free, gub_arr)
            if Fvis > F_L:
                F_L = Fvis
            if Fvis > -1e200:  # found any visible sample
                found_feasible = True
        else:
            free_size = _free_push(heap_idx, free_size, next_free, capacity)

        next_free += 1

    if heap_size == 0:
        raise ValueError("No overlapping visibility (certified empty).")

    # -----------------
    # Phase A: find any feasible visible sample (g>=0) or certify empty
    # Heap keyed by gub_arr
    # -----------------
    it = 0
    while (not found_feasible) and it < MAX_ITERS:
        it += 1

        # If best possible g upper bound is < 0, empty intersection certified.
        top = heap_idx[0]
        if gub_arr[top] < 0.0:
            raise ValueError("No overlapping visibility (certified empty).")

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, gub_arr)
            parents[k] = idx

        # Allocate 3 side-child slots per parent, preferring recycled slots.
        for k in range(pcount):
            if free_size > 0:
                c1, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c1 = next_free
                next_free += 1

            if free_size > 0:
                c2, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c2 = next_free
                next_free += 1

            if free_size > 0:
                c3, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError(
                            "Exceeded MAX_NODES before finding any feasible visible sample."
                        )
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c3 = next_free
                next_free += 1

            child_slot1[k] = c1
            child_slot2[k] = c2
            child_slot3[k] = c3

        _subdivide_and_eval_batch(
            parents,
            pcount,
            child_slot1,
            child_slot2,
            child_slot3,
            tri_u,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
            gub_arr,
            Fub_arr,
            child_idx,
            child_ok,
            child_Fvis,
        )

        # Push children back into heap (still keyed by gub)
        for q in range(4 * pcount):
            nid = child_idx[q]
            if child_ok[q] == 1:
                heap_size = _heap_push(heap_idx, heap_size, nid, gub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv
                if fv > -1e200:
                    found_feasible = True
            else:
                free_size = _free_push(heap_idx, free_size, nid, capacity)

        if heap_size == 0:
            raise ValueError("No overlapping visibility (certified empty).")

    if not found_feasible:
        raise ValueError("Failed to find feasible overlap before MAX_ITERS.")

    # -----------------
    # Phase B: tighten F bounds to tolerance
    # Rebuild heap keyed by Fub_arr.
    # -----------------
    _heapify(heap_idx, heap_size, Fub_arr)

    # Certified upper bound is max ub_F among active cells
    it = 0
    while it < MAX_ITERS:
        it += 1

        if heap_size == 0:
            raise ValueError("No overlapping visibility (no possible cells remain).")

        F_U = Fub_arr[heap_idx[0]]

        # Compute certified PRI and PRF bounds each iteration.
        pri_lower_s = 2.0 * F_L
        pri_upper_s = 2.0 * F_U

        # If pri_lower_s <= 0, PRF is unbounded (or numerically unstable).
        # In that case we cannot certify a finite PRF tolerance.
        if pri_lower_s <= 0.0:
            raise ValueError("PRI lower bound <= 0; cannot certify PRF tolerance.")

        prf_lower_cert_hz = 1.0 / pri_upper_s
        prf_lower_hz = 1.0 / (pri_upper_s + strict_pri_pad_s)  # strict-safe
        prf_upper_hz = 1.0 / pri_lower_s  # optimistic upper bound
        prf_gap_hz = prf_upper_hz - prf_lower_cert_hz

        if prf_gap_hz <= tol_prf_hz:
            pri_lower_us = pri_lower_s * 1e6
            pri_upper_us = pri_upper_s * 1e6
            pri_gap_us = pri_upper_us - pri_lower_us
            top_idx = heap_idx[0]
            warm_cell_u[0, 0] = tri_u[top_idx, 0, 0]
            warm_cell_u[0, 1] = tri_u[top_idx, 0, 1]
            warm_cell_u[0, 2] = tri_u[top_idx, 0, 2]
            warm_cell_u[1, 0] = tri_u[top_idx, 1, 0]
            warm_cell_u[1, 1] = tri_u[top_idx, 1, 1]
            warm_cell_u[1, 2] = tri_u[top_idx, 1, 2]
            warm_cell_u[2, 0] = tri_u[top_idx, 2, 0]
            warm_cell_u[2, 1] = tri_u[top_idx, 2, 1]
            warm_cell_u[2, 2] = tri_u[top_idx, 2, 2]
            return (
                tri_u,
                gub_arr,
                Fub_arr,
                heap_idx,
                True,
                prf_lower_hz,
                prf_upper_hz,
                prf_gap_hz,
                pri_upper_us,
                pri_lower_us,
                pri_gap_us,
            )

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, Fub_arr)
            parents[k] = idx

        # Allocate 3 side-child slots per parent, preferring recycled slots.
        for k in range(pcount):
            if free_size > 0:
                c1, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c1 = next_free
                next_free += 1

            if free_size > 0:
                c2, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c2 = next_free
                next_free += 1

            if free_size > 0:
                c3, free_size = _free_pop(heap_idx, free_size, capacity)
            else:
                if next_free >= capacity:
                    if not auto_expand_max_nodes:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    new_capacity = _next_capacity(capacity, max_nodes_growth, hard_cap)
                    if new_capacity <= capacity:
                        raise ValueError("Exceeded MAX_NODES before meeting tolerance.")
                    tri_u, gub_arr, Fub_arr, heap_idx = _grow_storage(
                        tri_u,
                        gub_arr,
                        Fub_arr,
                        heap_idx,
                        capacity,
                        new_capacity,
                        heap_size,
                        free_size,
                    )
                    capacity = new_capacity
                c3 = next_free
                next_free += 1

            child_slot1[k] = c1
            child_slot2[k] = c2
            child_slot3[k] = c3

        _subdivide_and_eval_batch(
            parents,
            pcount,
            child_slot1,
            child_slot2,
            child_slot3,
            tri_u,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            L_u_to_x,
            inv_c,
            margin,
            L_F,
            L_g,
            F_cap,
            gub_arr,
            Fub_arr,
            child_idx,
            child_ok,
            child_Fvis,
        )

        # Insert possibly-visible children into heap (keyed by Fub)
        for q in range(4 * pcount):
            nid = child_idx[q]
            if child_ok[q] == 1:
                heap_size = _heap_push(heap_idx, heap_size, nid, Fub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv
            else:
                free_size = _free_push(heap_idx, free_size, nid, capacity)

    raise ValueError("Tolerance not achieved before MAX_ITERS.")


@njit(cache=True)
def max_unambiguous_prf_batched(
    obs_ecef_m_series: np.ndarray,
    sigma_t_s_series: np.ndarray,
    tol_prf_hz: float = 0.1,
    k_sigma: float = 1.0,
    max_nodes: int = 5_000_000,
    max_iters: int = 1_000_000,
    margin_mode: int = 0,
    strict_pri_pad_s: float = 1e-12,
    auto_expand_max_nodes: bool = True,
    max_nodes_growth: float = 2.0,
    max_nodes_hard_cap: int = 2_147_483_647,
    temporal_warm_start: bool = True,
):
    """
    Batched certified critical-PRF solve over time.

    Parameters
    ----------
    obs_ecef_m_series : float64[T, N, 3]
        Observer ECEF positions per time step.
    sigma_t_s_series : float64[T, N]
        Per-observer timing sigma values per time step.

    Returns
    -------
    Tuple of float64[T] arrays:
        (prf_lower_hz, prf_upper_hz, prf_gap_hz, pri_upper_us, pri_lower_us, pri_gap_us)

    Notes
    -----
    This batched path reuses a single branch-and-bound workspace across all time steps.
    That avoids repeated large allocations and can significantly reduce total runtime
    versus repeatedly calling max_unambiguous_prf from Python.

    If temporal_warm_start is enabled, each step also applies certified temporal bounds
    derived from the previous step:
        F*_t in [F_L_prev - dF, F_U_prev + dF]
    where dF accounts for observer motion and sigma-margin change. This preserves
    correctness while reducing the search volume for smooth trajectories.
    """
    if obs_ecef_m_series.ndim != 3:
        raise ValueError("obs_ecef_m_series must have shape (T, N, 3).")
    if obs_ecef_m_series.shape[2] != 3:
        raise ValueError("obs_ecef_m_series must have shape (T, N, 3).")
    if sigma_t_s_series.ndim != 2:
        raise ValueError("sigma_t_s_series must have shape (T, N).")

    T = obs_ecef_m_series.shape[0]
    N = obs_ecef_m_series.shape[1]
    if sigma_t_s_series.shape[0] != T or sigma_t_s_series.shape[1] != N:
        raise ValueError("sigma_t_s_series must have shape (T, N) matching observers.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be > 0.")

    prf_lower_hz = np.empty(T, dtype=np.float64)
    prf_upper_hz = np.empty(T, dtype=np.float64)
    prf_gap_hz = np.empty(T, dtype=np.float64)
    pri_upper_us = np.empty(T, dtype=np.float64)
    pri_lower_us = np.empty(T, dtype=np.float64)
    pri_gap_us = np.empty(T, dtype=np.float64)

    BATCH = 2048
    tri_u = np.empty((max_nodes, 3, 3), dtype=np.float64)
    gub_arr = np.empty(max_nodes, dtype=np.float64)
    Fub_arr = np.empty(max_nodes, dtype=np.float64)
    heap_idx = np.empty(max_nodes, dtype=np.int32)
    parents = np.empty(BATCH, dtype=np.int32)
    child_slot1 = np.empty(BATCH, dtype=np.int32)
    child_slot2 = np.empty(BATCH, dtype=np.int32)
    child_slot3 = np.empty(BATCH, dtype=np.int32)
    child_idx = np.empty(4 * BATCH, dtype=np.int32)
    child_ok = np.empty(4 * BATCH, dtype=np.int8)
    child_Fvis = np.empty(4 * BATCH, dtype=np.float64)

    inv_c = 1.0 / 299792458.0
    warm_slack_s = 1e-12
    have_prev = False
    prev_margin = 0.0
    prev_F_L = 0.0
    prev_F_U = 0.0
    warm_cell_valid = False
    warm_cell_u = np.empty((3, 3), dtype=np.float64)

    for t in range(T):
        use_warm_bounds = False
        warm_F_lower = -1e300
        warm_F_upper = 1e300
        margin_t = _margin_from_sigmas(sigma_t_s_series[t], N, k_sigma, margin_mode)

        if temporal_warm_start and have_prev:
            max_disp = 0.0
            for i in range(N):
                dx = obs_ecef_m_series[t, i, 0] - obs_ecef_m_series[t - 1, i, 0]
                dy = obs_ecef_m_series[t, i, 1] - obs_ecef_m_series[t - 1, i, 1]
                dz = obs_ecef_m_series[t, i, 2] - obs_ecef_m_series[t - 1, i, 2]
                d = _norm3(dx, dy, dz)
                if d > max_disp:
                    max_disp = d
            # Uniform temporal Lipschitz bound:
            # |W_t(x)-W_{t-1}(x)| <= 2*max_disp/c, for all x.
            dF = 2.0 * max_disp * inv_c + np.abs(margin_t - prev_margin) + warm_slack_s
            warm_F_lower = prev_F_L - dF
            warm_F_upper = prev_F_U + dF
            use_warm_bounds = True

        (
            tri_u,
            gub_arr,
            Fub_arr,
            heap_idx,
            warm_cell_valid,
            prf_lower_hz[t],
            prf_upper_hz[t],
            prf_gap_hz[t],
            pri_upper_us[t],
            pri_lower_us[t],
            pri_gap_us[t],
        ) = _max_unambiguous_prf_reuse_workspace(
            obs_ecef_m_series[t],
            sigma_t_s_series[t],
            tol_prf_hz,
            k_sigma,
            max_iters,
            margin_mode,
            strict_pri_pad_s,
            auto_expand_max_nodes,
            max_nodes_growth,
            max_nodes_hard_cap,
            use_warm_bounds,
            warm_F_lower,
            warm_F_upper,
            warm_cell_valid,
            warm_cell_u,
            tri_u,
            gub_arr,
            Fub_arr,
            heap_idx,
            parents,
            child_slot1,
            child_slot2,
            child_slot3,
            child_idx,
            child_ok,
            child_Fvis,
        )

        prev_margin = margin_t
        prev_F_L = 0.5 * pri_lower_us[t] * 1e-6
        prev_F_U = 0.5 * pri_upper_us[t] * 1e-6
        have_prev = True

    return (
        prf_lower_hz,
        prf_upper_hz,
        prf_gap_hz,
        pri_upper_us,
        pri_lower_us,
        pri_gap_us,
    )


@njit(cache=True)
def max_unambiguous_prf_batched_const_sigma(
    obs_ecef_m_series: np.ndarray,
    sigma_t_s: np.ndarray,
    tol_prf_hz: float = 0.1,
    k_sigma: float = 1.0,
    max_nodes: int = 5_000_000,
    max_iters: int = 1_000_000,
    margin_mode: int = 0,
    strict_pri_pad_s: float = 1e-12,
    auto_expand_max_nodes: bool = True,
    max_nodes_growth: float = 2.0,
    max_nodes_hard_cap: int = 2_147_483_647,
    temporal_warm_start: bool = True,
):
    """
    Batched certified critical-PRF solve over time with constant per-observer sigma.

    Parameters
    ----------
    obs_ecef_m_series : float64[T, N, 3]
    sigma_t_s : float64[N]
    """
    if obs_ecef_m_series.ndim != 3:
        raise ValueError("obs_ecef_m_series must have shape (T, N, 3).")
    if obs_ecef_m_series.shape[2] != 3:
        raise ValueError("obs_ecef_m_series must have shape (T, N, 3).")
    if sigma_t_s.ndim != 1:
        raise ValueError("sigma_t_s must have shape (N,).")
    if sigma_t_s.shape[0] != obs_ecef_m_series.shape[1]:
        raise ValueError("sigma_t_s length must match number of observers.")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be > 0.")

    T = obs_ecef_m_series.shape[0]
    prf_lower_hz = np.empty(T, dtype=np.float64)
    prf_upper_hz = np.empty(T, dtype=np.float64)
    prf_gap_hz = np.empty(T, dtype=np.float64)
    pri_upper_us = np.empty(T, dtype=np.float64)
    pri_lower_us = np.empty(T, dtype=np.float64)
    pri_gap_us = np.empty(T, dtype=np.float64)

    BATCH = 2048
    tri_u = np.empty((max_nodes, 3, 3), dtype=np.float64)
    gub_arr = np.empty(max_nodes, dtype=np.float64)
    Fub_arr = np.empty(max_nodes, dtype=np.float64)
    heap_idx = np.empty(max_nodes, dtype=np.int32)
    parents = np.empty(BATCH, dtype=np.int32)
    child_slot1 = np.empty(BATCH, dtype=np.int32)
    child_slot2 = np.empty(BATCH, dtype=np.int32)
    child_slot3 = np.empty(BATCH, dtype=np.int32)
    child_idx = np.empty(4 * BATCH, dtype=np.int32)
    child_ok = np.empty(4 * BATCH, dtype=np.int8)
    child_Fvis = np.empty(4 * BATCH, dtype=np.float64)

    inv_c = 1.0 / 299792458.0
    warm_slack_s = 1e-12
    have_prev = False
    prev_F_L = 0.0
    prev_F_U = 0.0
    warm_cell_valid = False
    warm_cell_u = np.empty((3, 3), dtype=np.float64)

    for t in range(T):
        use_warm_bounds = False
        warm_F_lower = -1e300
        warm_F_upper = 1e300

        if temporal_warm_start and have_prev:
            max_disp = 0.0
            for i in range(sigma_t_s.shape[0]):
                dx = obs_ecef_m_series[t, i, 0] - obs_ecef_m_series[t - 1, i, 0]
                dy = obs_ecef_m_series[t, i, 1] - obs_ecef_m_series[t - 1, i, 1]
                dz = obs_ecef_m_series[t, i, 2] - obs_ecef_m_series[t - 1, i, 2]
                d = _norm3(dx, dy, dz)
                if d > max_disp:
                    max_disp = d
            dF = 2.0 * max_disp * inv_c + warm_slack_s
            warm_F_lower = prev_F_L - dF
            warm_F_upper = prev_F_U + dF
            use_warm_bounds = True

        (
            tri_u,
            gub_arr,
            Fub_arr,
            heap_idx,
            warm_cell_valid,
            prf_lower_hz[t],
            prf_upper_hz[t],
            prf_gap_hz[t],
            pri_upper_us[t],
            pri_lower_us[t],
            pri_gap_us[t],
        ) = _max_unambiguous_prf_reuse_workspace(
            obs_ecef_m_series[t],
            sigma_t_s,
            tol_prf_hz,
            k_sigma,
            max_iters,
            margin_mode,
            strict_pri_pad_s,
            auto_expand_max_nodes,
            max_nodes_growth,
            max_nodes_hard_cap,
            use_warm_bounds,
            warm_F_lower,
            warm_F_upper,
            warm_cell_valid,
            warm_cell_u,
            tri_u,
            gub_arr,
            Fub_arr,
            heap_idx,
            parents,
            child_slot1,
            child_slot2,
            child_slot3,
            child_idx,
            child_ok,
            child_Fvis,
        )

        prev_F_L = 0.5 * pri_lower_us[t] * 1e-6
        prev_F_U = 0.5 * pri_upper_us[t] * 1e-6
        have_prev = True

    return (
        prf_lower_hz,
        prf_upper_hz,
        prf_gap_hz,
        pri_upper_us,
        pri_lower_us,
        pri_gap_us,
    )


if __name__ == "__main__":
    # Support direct script execution so numba cache imports can resolve `nebula`.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _repo_root not in sys.path:
        sys.path.append(_repo_root)

    # Example with overlap: three near-co-located LEO observers (ECEF)
    a = 6378137.0
    h = 55000e3
    r = a + h
    deg = np.pi / 180.0
    lons = np.array([0.0, 1.0, -1.0]) * deg
    obs = np.stack(
        [r * np.cos(lons), r * np.sin(lons), np.zeros_like(lons)], axis=1
    ).astype(np.float64)

    sig = np.array([500e-9, 500e-9, 500e-9], dtype=np.float64)

    prf_lo, prf_hi, prf_gap, pri_u_us, pri_l_us, pri_gap_us = max_unambiguous_prf(
        obs, sig, tol_prf_hz=1.0, k_sigma=3.0
    )
    print("PRF_lower_hz (safe):", prf_lo)
    print("PRF_upper_hz:", prf_hi)
    print("PRF_gap_hz:", prf_gap)
    print("PRI_upper_us:", pri_u_us)
    print("PRI_lower_us:", pri_l_us)
    print("PRI_gap_us:", pri_gap_us)
    raise SystemExit("Example complete.")
    raise ValueError()

    # Batched examples: compare equivalence vs scalar and benchmark throughput.
    import timeit

    T = 8
    step_deg = 0.05
    lons_series = lons[None, :] + (
        np.arange(T, dtype=np.float64)[:, None] * step_deg * deg
    )
    obs_series = np.empty((T, obs.shape[0], 3), dtype=np.float64)
    obs_series[:, :, 0] = r * np.cos(lons_series)
    obs_series[:, :, 1] = r * np.sin(lons_series)
    obs_series[:, :, 2] = 0.0

    sig_const = np.array([500e-9, 500e-9, 500e-9], dtype=np.float64)
    sig_series = np.empty((T, sig_const.shape[0]), dtype=np.float64)
    for t in range(T):
        # Small deterministic sigma variation across time.
        scale = 1.0 + 0.10 * np.sin(0.2 * t)
        sig_series[t] = sig_const * scale

    bench_tol_prf_hz = 2.0

    # Warmup compile for all interfaces before timing.
    _ = max_unambiguous_prf(
        obs_series[0], sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
    )
    _ = max_unambiguous_prf_batched_const_sigma(
        obs_series[:2], sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
    )
    _ = max_unambiguous_prf_batched_const_sigma(
        obs_series[:2],
        sig_const,
        tol_prf_hz=bench_tol_prf_hz,
        k_sigma=3.0,
        temporal_warm_start=False,
    )
    _ = max_unambiguous_prf_batched(
        obs_series[:2], sig_series[:2], tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
    )
    _ = max_unambiguous_prf_batched(
        obs_series[:2],
        sig_series[:2],
        tol_prf_hz=bench_tol_prf_hz,
        k_sigma=3.0,
        temporal_warm_start=False,
    )

    # Compare scalar-loop outputs against batched APIs.
    scalar_const = np.empty((T, 6), dtype=np.float64)
    for t in range(T):
        scalar_const[t] = max_unambiguous_prf(
            obs_series[t], sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
        )
    batched_const = max_unambiguous_prf_batched_const_sigma(
        obs_series, sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
    )
    batched_const_mat = np.stack(batched_const, axis=1)
    print(
        "Batched const-sigma max abs diff vs scalar:",
        np.max(np.abs(batched_const_mat - scalar_const)),
    )

    scalar_var = np.empty((T, 6), dtype=np.float64)
    for t in range(T):
        scalar_var[t] = max_unambiguous_prf(
            obs_series[t], sig_series[t], tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
        )
    batched_var = max_unambiguous_prf_batched(
        obs_series, sig_series, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
    )
    batched_var_mat = np.stack(batched_var, axis=1)
    print(
        "Batched time-varying-sigma max abs diff vs scalar:",
        np.max(np.abs(batched_var_mat - scalar_var)),
    )

    # Steady-state timing (JIT compile cost excluded by warmup above).
    reps = 2
    scalar_const_s = timeit.timeit(
        lambda: [
            max_unambiguous_prf(
                obs_series[t], sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
            )
            for t in range(T)
        ],
        number=reps,
    )
    batched_const_s = timeit.timeit(
        lambda: max_unambiguous_prf_batched_const_sigma(
            obs_series, sig_const, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
        ),
        number=reps,
    )
    scalar_var_s = timeit.timeit(
        lambda: [
            max_unambiguous_prf(
                obs_series[t], sig_series[t], tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
            )
            for t in range(T)
        ],
        number=reps,
    )
    batched_var_s = timeit.timeit(
        lambda: max_unambiguous_prf_batched(
            obs_series, sig_series, tol_prf_hz=bench_tol_prf_hz, k_sigma=3.0
        ),
        number=reps,
    )
    reps_no_warm = 1
    batched_const_no_warm_s = timeit.timeit(
        lambda: max_unambiguous_prf_batched_const_sigma(
            obs_series,
            sig_const,
            tol_prf_hz=bench_tol_prf_hz,
            k_sigma=3.0,
            temporal_warm_start=False,
        ),
        number=reps_no_warm,
    )
    batched_var_no_warm_s = timeit.timeit(
        lambda: max_unambiguous_prf_batched(
            obs_series,
            sig_series,
            tol_prf_hz=bench_tol_prf_hz,
            k_sigma=3.0,
            temporal_warm_start=False,
        ),
        number=reps_no_warm,
    )
    batched_const_per_run = batched_const_s / reps
    batched_var_per_run = batched_var_s / reps
    batched_const_no_warm_per_run = batched_const_no_warm_s / reps_no_warm
    batched_var_no_warm_per_run = batched_var_no_warm_s / reps_no_warm

    print(f"Timing over T={T}, reps={reps}, tol_prf_hz={bench_tol_prf_hz} (seconds):")
    print(f"  scalar loop (const sigma):        {scalar_const_s:.3f}")
    print(f"  batched const-sigma API:          {batched_const_s:.3f}")
    print(
        f"  speedup const-sigma:              {scalar_const_s / batched_const_s:.2f}x"
    )
    print(f"  batched const/run (warm on):      {batched_const_per_run:.3f}")
    print(f"  batched const/run (warm off):     {batched_const_no_warm_per_run:.3f}")
    print(f"  scalar loop (time-varying sigma): {scalar_var_s:.3f}")
    print(f"  batched varying-sigma API:        {batched_var_s:.3f}")
    print(f"  speedup varying-sigma:            {scalar_var_s / batched_var_s:.2f}x")
    print(f"  batched varying/run (warm on):    {batched_var_per_run:.3f}")
    print(f"  batched varying/run (warm off):   {batched_var_no_warm_per_run:.3f}")

    sig = np.zeros(3, dtype=np.float64)  # zero noise case
    prf_lo, prf_hi, prf_gap, pri_u_us, pri_l_us, pri_gap_us = max_unambiguous_prf(
        obs, sig, tol_prf_hz=1.0, k_sigma=3.0
    )
    print("PRF_lower_hz (safe):", prf_lo)
    print("PRF_upper_hz:", prf_hi)
    print("PRF_gap_hz:", prf_gap)
    print("PRI_upper_us:", pri_u_us)
    print("PRI_lower_us:", pri_l_us)
    print("PRI_gap_us:", pri_gap_us)
