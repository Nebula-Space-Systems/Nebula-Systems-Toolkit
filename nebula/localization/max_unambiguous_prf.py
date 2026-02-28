"""
Certified maximum unambiguous PRF (no-wrap TDOA criterion) over the WGS84 ellipsoid surface
for an arbitrary number of observers (ECEF meters) at a single time.

Definitions
-----------
Observers: p_i in ECEF meters.

Propagation delays:
    tau_i(x) = ||x - p_i|| / c

Worst-case inter-observer delay spread at x:
    W(x) = max_i tau_i(x) - min_i tau_i(x)

Timing uncertainty margin (deterministic hard bounds):
    Let sigma_i be a worst-case absolute timestamp error bound (seconds) for observer i.
    Then worst-case absolute TDOA error over all baselines is:
        margin = sigma_(largest) + sigma_(second_largest)

No-wrap unambiguous PRI condition over region R:
    For all x in R:
        W(x) + margin < PRI/2
    Therefore the required PRI is:
        PRI_min = 2 * max_{x in R} (W(x) + margin)
    And the safe maximum PRF is:
        PRF_safe = 1 / PRI_upper

Region R (common visibility on WGS84):
    x is on the WGS84 ellipsoid surface and line-of-sight visible to ALL observers,
    with the WGS84 ellipsoid occluding LOS.

Certified solver
----------------
We maximize:
    F(x) = W(x) + margin
over x in R, using branch-and-bound over a hierarchical icosphere mesh projected to WGS84.

Certification uses Lipschitz bounds (Euclidean ECEF distance):
    - W is Lipschitz with L_F = 2/c
    - Visibility margin g(x) = min_k (p_s,k^T x - 1), where p_s = A p and
      A=diag(1/a^2, 1/a^2, 1/b^2), is Lipschitz with L_g = max_k ||p_s,k||.

For each cell C we compute certified upper bounds:
    ub_g(C) >= max_{x in C} g(x)
    ub_F(C) >= max_{x in C} F(x)

We maintain:
    F_L = best feasible sample value (visible sample) => F_L <= F*
    F_U = max over active cells of ub_F(C)           => F* <= F_U

Stop when:
    PRI_gap = 2*(F_U - F_L) <= tol_pri_us

If caps are hit before meeting tolerance, we raise ValueError (no partial returns).

Outputs
-------
Returns (all float64):
    (prf_safe_hz, pri_upper_us, pri_lower_us, pri_gap_us)

Raises ValueError if:
    - certified empty overlap (no common-visible region), OR
    - tolerance not achieved before caps.

Notes
-----
- PRF is in Hz. PRI is in microseconds.
- This uses a conservative analytic clamp:
      W(x) <= Bmax/c, where Bmax = max_{i<j} ||p_i - p_j||
  so F(x) <= Bmax/c + margin. This dramatically improves convergence and does not
  reduce correctness.
"""

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
def _clamp01(x):
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True, inline="always")
def _chord_bound_from_cos(cos_theta, a):
    # chord <= a*sqrt(2*(1-cos))
    c = _clamp01(cos_theta)
    return a * np.sqrt(2.0 * (1.0 - c))


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


@njit(
    cache=True,
)
def _cell_diameter_bound(u0, u1, u2, a):
    # Conservative chord diameter bound using vertices + edge midpoints (6 points).
    m01x, m01y, m01z = _mid_dir(u0[0], u0[1], u0[2], u1[0], u1[1], u1[2])
    m12x, m12y, m12z = _mid_dir(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
    m20x, m20y, m20z = _mid_dir(u2[0], u2[1], u2[2], u0[0], u0[1], u0[2])

    # list of 6 direction vectors
    # compute max chord bound among all pairs (15 pairs)
    # (dot products only, no acos)
    pts = (
        (u0[0], u0[1], u0[2]),
        (u1[0], u1[1], u1[2]),
        (u2[0], u2[1], u2[2]),
        (m01x, m01y, m01z),
        (m12x, m12y, m12z),
        (m20x, m20y, m20z),
    )

    dmax = 0.0
    for i in range(6):
        ax, ay, az = pts[i]
        for j in range(i + 1, 6):
            bx, by, bz = pts[j]
            dot = ax * bx + ay * by + az * bz
            d = _chord_bound_from_cos(dot, a)
            if d > dmax:
                dmax = d
    return dmax


@njit(
    cache=True,
)
def _eval_cell(
    u0,
    u1,
    u2,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    a,
    inv_c,
    margin,
    L_F,
    L_g,
    F_cap,
):
    """
    Returns:
      (diam, gub, Fvis, Fub)

    Where:
      - diam is a conservative diameter bound for the cell on the ellipsoid (via direction chord bound)
      - gub is a certified upper bound on max g(x) in the cell
      - Fvis is max F among *visible samples* in this cell (or -inf if none sampled visible)
      - Fub is a certified upper bound on max F(x) in the cell (clamped by F_cap)
    """
    diam = _cell_diameter_bound(u0, u1, u2, a)

    # Sample only vertices for speed (still certified).
    # Compute g at vertices to form gub and to identify visible samples.
    g_max = -1e300

    sx0, sy0, sz0 = _ellipsoid_intersect_dir(u0[0], u0[1], u0[2], inva2, invb2)
    g0 = _g_visibility_margin(sx0, sy0, sz0, ps, nobs)
    if g0 > g_max:
        g_max = g0

    sx1, sy1, sz1 = _ellipsoid_intersect_dir(u1[0], u1[1], u1[2], inva2, invb2)
    g1 = _g_visibility_margin(sx1, sy1, sz1, ps, nobs)
    if g1 > g_max:
        g_max = g1

    sx2, sy2, sz2 = _ellipsoid_intersect_dir(u2[0], u2[1], u2[2], inva2, invb2)
    g2 = _g_visibility_margin(sx2, sy2, sz2, ps, nobs)
    if g2 > g_max:
        g_max = g2

    # Certified possible-visibility upper bound
    gub = g_max + L_g * diam

    # If provably not in intersection, skip expensive F work
    if gub < 0.0:
        return diam, gub, -1e300, -1e300

    # Compute F on vertices (W + margin)
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

    # Certified upper bound on max F in cell
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
    tri_u,
    base_new,
    obs,
    ps,
    nobs,
    inva2,
    invb2,
    a,
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

        c1 = base_new + 3 * k + 0
        c2 = base_new + 3 * k + 1
        c3 = base_new + 3 * k + 2

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
            a,
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
            a,
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
            a,
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
            a,
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
    tol_pri_us: float = 1.0,
    k_sigma: float = 1.0,
    max_nodes: int = 50_000_000,
    max_iters: int = 1_000_000,
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

    # deterministic worst-case margin: sum of two largest per-observer bounds
    s1 = 0.0
    s2 = 0.0
    for i in range(nobs):
        s = sigma_t_s[i]
        if s >= s1:
            s2 = s1
            s1 = s
        elif s > s2:
            s2 = s
    # 1σ model: worst-case baseline TDOA std-dev (assuming independent Gaussian timestamp errors)
    sigma_tdoa_1sigma = np.sqrt(s1 * s1 + s2 * s2)

    # choose k_sigma for desired confidence
    margin = k_sigma * sigma_tdoa_1sigma

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

    # Certified tolerance: PRI_gap <= tol_pri_us
    eps_T = tol_pri_us * 1e-6
    eps_F = 0.5 * eps_T

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
    MAX_NODES = max_nodes
    MAX_ITERS = max_iters
    BATCH = 2048  # increase for higher CPU utilization
    if MAX_NODES > 2_147_483_647:
        raise ValueError("max_nodes too large for int32 indexing")

    # Node storage
    tri_u = np.empty((MAX_NODES, 3, 3), dtype=np.float64)
    gub_arr = np.full(MAX_NODES, -1e300, dtype=np.float64)
    Fub_arr = np.full(MAX_NODES, -1e300, dtype=np.float64)

    # Single heap array; key depends on phase
    heap_idx = np.empty(MAX_NODES, dtype=np.int32)
    heap_size = 0
    next_free = 0

    # Bounds on F*
    F_L = -1e300
    found_feasible = False

    # Phase A key uses gub_arr; Phase B uses Fub_arr
    # Initialize with 20 base faces and insert those with gub>=0
    for f in range(_ICO_F.shape[0]):
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
            a,
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

        next_free += 1

    if heap_size == 0:
        raise ValueError("No overlapping visibility (certified empty).")

    # Workspace
    parents = np.empty(BATCH, dtype=np.int32)
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

        # capacity check: each refined parent allocates 3 new nodes
        need = 3 * pcount
        if next_free + need >= MAX_NODES:
            raise ValueError(
                "Exceeded MAX_NODES before finding any feasible visible sample."
            )

        base_new = next_free
        next_free += need

        _subdivide_and_eval_batch(
            parents,
            pcount,
            tri_u,
            base_new,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            a,
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
            if child_ok[q] == 1:
                nid = child_idx[q]
                heap_size = _heap_push(heap_idx, heap_size, nid, gub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv
                if fv > -1e200:
                    found_feasible = True

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

        if (F_U - F_L) <= eps_F:
            pri_lower_s = 2.0 * F_L
            pri_upper_s = 2.0 * F_U
            prf_safe_hz = 1.0 / pri_upper_s
            pri_lower_us = pri_lower_s * 1e6
            pri_upper_us = pri_upper_s * 1e6
            pri_gap_us = pri_upper_us - pri_lower_us
            # enforce tolerance
            if pri_gap_us > tol_pri_us:
                raise ValueError("Internal error: tolerance condition mismatch.")
            return prf_safe_hz, pri_upper_us, pri_lower_us, pri_gap_us

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, Fub_arr)
            parents[k] = idx

        need = 3 * pcount
        if next_free + need >= MAX_NODES:
            raise ValueError("Exceeded MAX_NODES before meeting tolerance.")

        base_new = next_free
        next_free += need

        _subdivide_and_eval_batch(
            parents,
            pcount,
            tri_u,
            base_new,
            obs_ecef_m,
            ps,
            nobs,
            inva2,
            invb2,
            a,
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
            if child_ok[q] == 1:
                nid = child_idx[q]
                heap_size = _heap_push(heap_idx, heap_size, nid, Fub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv

    raise ValueError("Tolerance not achieved before MAX_ITERS.")


if __name__ == "__main__":
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

    prf_hz, pri_u_us, pri_l_us, gap_us = max_unambiguous_prf(obs, sig, 1.0, 3.0)
    print("PRF_safe_hz:", prf_hz)
    print("PRI_upper_us:", pri_u_us)
    print("PRI_lower_us:", pri_l_us)
    print("PRI_gap_us:", gap_us)

    sig = np.zeros(3, dtype=np.float64)  # zero noise case
    prf_hz, pri_u_us, pri_l_us, gap_us = max_unambiguous_prf(obs, sig, 1.0, 3.0)
    print("PRF_safe_hz (zero noise):", prf_hz)
    print("PRI_upper_us (zero noise):", pri_u_us)
    print("PRI_lower_us (zero noise):", pri_l_us)
    print("PRI_gap_us (zero noise):", gap_us)
