from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# resolve imports to the top level directory
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebula.localization.max_unambiguous_prf import (
    _ICO_F,
    _ICO_V,
    _ellipsoid_intersect_dir,
    _eval_cell,
    _heap_push,
    _heap_pop_max,
    _heapify,
    _norm3,
    _normalize3,
    _subdivide_and_eval_batch,
)

# Optional: use your existing transform if available
try:
    from nebula.transforms._ecef2geodetic import ecef2geodetic_vec_xyz

    def ecef_to_latlon_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat, lon, _h = ecef2geodetic_vec_xyz(
            xyz[:, 0].astype(np.float64),
            xyz[:, 1].astype(np.float64),
            xyz[:, 2].astype(np.float64),
        )
        return np.degrees(lat), np.degrees(lon)

except Exception:
    # Fallback spherical (OK for visualization)
    def ecef_to_latlon_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
        return lat, lon


@njit(cache=True)
def max_unambiguous_prf_collect_leaf_centers(
    obs_ecef_m: np.ndarray,
    sigma_t_s: np.ndarray,
    tol_pri_us: float = 1.0,
    k_sigma: float = 1.0,
    max_nodes: int = 5_000_000,
    max_iters: int = 1_000_000,
    max_plot_points: int = 200_000,
):
    """
    Same solver as max_unambiguous_prf(), but also returns a subsampled set of
    leaf-cell center points (ECEF) at termination for visualization.

    Returns:
      (prf_safe_hz, pri_upper_us, pri_lower_us, pri_gap_us, leaf_xyz, leaf_fub)

    leaf_xyz: (M,3) float64 ECEF points on WGS84 surface (center of each selected leaf cell)
    leaf_fub: (M,)  float64 corresponding F upper bound per selected leaf cell
    """
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

    # worst-case two-largest 1σ for baseline RSS
    s1 = 0.0
    s2 = 0.0
    for i in range(nobs):
        s = sigma_t_s[i]
        if s >= s1:
            s2 = s1
            s1 = s
        elif s > s2:
            s2 = s
    sigma_tdoa_1sigma = np.sqrt(s1 * s1 + s2 * s2)
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
    L_u_to_x = (a * a * a) / (b * b)

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

    MAX_NODES = max_nodes
    MAX_ITERS = max_iters
    BATCH = 2048
    if MAX_NODES > 2_147_483_647:
        raise ValueError("max_nodes too large for int32 indexing")

    # Node storage
    tri_u = np.empty((MAX_NODES, 3, 3), dtype=np.float64)
    gub_arr = np.full(MAX_NODES, -1e300, dtype=np.float64)
    Fub_arr = np.full(MAX_NODES, -1e300, dtype=np.float64)

    heap_idx = np.empty(MAX_NODES, dtype=np.int32)
    heap_size = 0
    next_free = 0

    F_L = -1e300
    found_feasible = False

    # init with 20 faces
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
            if Fvis > -1e200:
                found_feasible = True

        next_free += 1

    if heap_size == 0:
        raise ValueError("No overlapping visibility (certified empty).")

    parents = np.empty(BATCH, dtype=np.int32)
    child_slot1 = np.empty(BATCH, dtype=np.int32)
    child_slot2 = np.empty(BATCH, dtype=np.int32)
    child_slot3 = np.empty(BATCH, dtype=np.int32)
    child_idx = np.empty(4 * BATCH, dtype=np.int32)
    child_ok = np.empty(4 * BATCH, dtype=np.int8)
    child_Fvis = np.empty(4 * BATCH, dtype=np.float64)

    # Phase A: find feasible overlap
    it = 0
    while (not found_feasible) and it < MAX_ITERS:
        it += 1
        top = heap_idx[0]
        if gub_arr[top] < 0.0:
            raise ValueError("No overlapping visibility (certified empty).")

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, gub_arr)
            parents[k] = idx

        need = 3 * pcount
        if next_free + need >= MAX_NODES:
            raise ValueError("Exceeded MAX_NODES before finding feasible overlap.")

        base_new = next_free
        next_free += need
        for k in range(pcount):
            child_slot1[k] = base_new + 3 * k + 0
            child_slot2[k] = base_new + 3 * k + 1
            child_slot3[k] = base_new + 3 * k + 2

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

    # Phase B: tighten F bounds
    _heapify(heap_idx, heap_size, Fub_arr)

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
            if pri_gap_us > tol_pri_us:
                raise ValueError("Internal error: tolerance mismatch")

            # ---- Collect leaf centers from FINAL heap ----
            # Subsample so plotting is manageable
            mmax = max_plot_points
            if mmax < 1:
                mmax = 1
            stride = heap_size // mmax
            if stride < 1:
                stride = 1

            m = (heap_size + stride - 1) // stride
            if m > mmax:
                m = mmax

            leaf_xyz = np.empty((m, 3), dtype=np.float64)
            leaf_fub = np.empty(m, dtype=np.float64)

            out_i = 0
            i = 0
            while i < heap_size and out_i < m:
                idx = int(heap_idx[i])

                u0 = tri_u[idx, 0]
                u1 = tri_u[idx, 1]
                u2 = tri_u[idx, 2]

                cx, cy, cz = _normalize3(
                    u0[0] + u1[0] + u2[0],
                    u0[1] + u1[1] + u2[1],
                    u0[2] + u1[2] + u2[2],
                )
                x, y, z = _ellipsoid_intersect_dir(cx, cy, cz, inva2, invb2)

                leaf_xyz[out_i, 0] = x
                leaf_xyz[out_i, 1] = y
                leaf_xyz[out_i, 2] = z
                leaf_fub[out_i] = Fub_arr[idx]

                out_i += 1
                i += stride

            return (
                prf_safe_hz,
                pri_upper_us,
                pri_lower_us,
                pri_gap_us,
                leaf_xyz,
                leaf_fub,
            )

        pcount = BATCH if heap_size >= BATCH else heap_size
        for k in range(pcount):
            idx, heap_size = _heap_pop_max(heap_idx, heap_size, Fub_arr)
            parents[k] = idx

        need = 3 * pcount
        if next_free + need >= MAX_NODES:
            raise ValueError("Exceeded MAX_NODES before meeting tolerance.")

        base_new = next_free
        next_free += need
        for k in range(pcount):
            child_slot1[k] = base_new + 3 * k + 0
            child_slot2[k] = base_new + 3 * k + 1
            child_slot3[k] = base_new + 3 * k + 2

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

        for q in range(4 * pcount):
            if child_ok[q] == 1:
                nid = child_idx[q]
                heap_size = _heap_push(heap_idx, heap_size, nid, Fub_arr)
                fv = child_Fvis[q]
                if fv > F_L:
                    F_L = fv

    raise ValueError("Tolerance not achieved before MAX_ITERS.")


WGS84_A = 6378137.0
WGS84_B = 6356752.3142451793


def ellipsoid_intersect_dir(u: np.ndarray) -> np.ndarray:
    """Project unit direction to WGS84 ellipsoid surface by radial intersection."""
    inva2 = 1.0 / (WGS84_A * WGS84_A)
    invb2 = 1.0 / (WGS84_B * WGS84_B)
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    s = (ux * ux + uy * uy) * inva2 + (uz * uz) * invb2
    t = 1.0 / np.sqrt(s)
    return np.array([t * ux, t * uy, t * uz], dtype=np.float64)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n


def chord_bound_from_dot(dot: float) -> float:
    """Conservative chord bound on ellipsoid surface using a (max radius)."""
    d = np.clip(dot, -1.0, 1.0)
    return WGS84_A * np.sqrt(2.0 * (1.0 - d))


def cell_diameter_bound(u0: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
    """Conservative diameter bound using vertices + edge midpoints (6 points)."""
    m01 = normalize(u0 + u1)
    m12 = normalize(u1 + u2)
    m20 = normalize(u2 + u0)

    pts = np.stack([u0, u1, u2, m01, m12, m20], axis=0)
    # pairwise dot, take minimum dot -> maximum chord
    dots = pts @ pts.T
    # exclude diagonal by setting it to +1
    np.fill_diagonal(dots, 1.0)
    min_dot = float(np.min(dots))
    return chord_bound_from_dot(min_dot)


def ps_vectors(obs_ecef: np.ndarray) -> np.ndarray:
    """p_s = A p where A=diag(1/a^2,1/a^2,1/b^2)."""
    inva2 = 1.0 / (WGS84_A * WGS84_A)
    invb2 = 1.0 / (WGS84_B * WGS84_B)
    ps = np.empty_like(obs_ecef, dtype=np.float64)
    ps[:, 0] = obs_ecef[:, 0] * inva2
    ps[:, 1] = obs_ecef[:, 1] * inva2
    ps[:, 2] = obs_ecef[:, 2] * invb2
    return ps


def g_value(x_ecef: np.ndarray, ps: np.ndarray) -> float:
    """g(x) = min_k (p_s,k^T x - 1). >=0 means visible to ALL observers."""
    v = ps @ x_ecef - 1.0
    return float(np.min(v))


def common_visible_footprint_points(
    obs_ecef: np.ndarray,
    max_cell_diam_m: float = 50_000.0,
    max_triangles: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive refinement to approximate the common-visible region on WGS84.

    Uses certified bounds for visibility:
      gmax + Lg*diam < 0  => definitely blocked
      gmin - Lg*diam >=0  => definitely visible
      else uncertain; refine until diam <= max_cell_diam_m

    Returns:
      visible_pts: centers of definitely-visible leaf cells (ECEF)
      boundary_pts: centers of unresolved boundary leaf cells (ECEF)
    """
    obs = np.ascontiguousarray(obs_ecef, dtype=np.float64)
    ps = ps_vectors(obs)
    Lg = float(np.max(np.linalg.norm(ps, axis=1)))  # Lipschitz constant for g

    # Each triangle stored as 3 unit directions
    # Work queue: list of triangles to process
    queue = []
    for f in _ICO_F:
        u0 = _ICO_V[f[0]].astype(np.float64)
        u1 = _ICO_V[f[1]].astype(np.float64)
        u2 = _ICO_V[f[2]].astype(np.float64)
        queue.append((u0, u1, u2))

    visible_centers = []
    boundary_centers = []

    # Process queue
    while queue:
        u0, u1, u2 = queue.pop()

        diam = cell_diameter_bound(u0, u1, u2)

        # evaluate g at vertices (on ellipsoid)
        x0 = ellipsoid_intersect_dir(u0)
        x1 = ellipsoid_intersect_dir(u1)
        x2 = ellipsoid_intersect_dir(u2)

        g0 = g_value(x0, ps)
        g1 = g_value(x1, ps)
        g2 = g_value(x2, ps)

        gmin = min(g0, g1, g2)
        gmax = max(g0, g1, g2)

        # certified bounds on g across the whole cell
        g_upper = gmax + Lg * diam
        g_lower = gmin - Lg * diam

        if g_upper < 0.0:
            # definitely blocked
            continue

        # center point of the cell (on ellipsoid)
        uc = normalize(u0 + u1 + u2)
        xc = ellipsoid_intersect_dir(uc)

        if g_lower >= 0.0:
            # definitely visible
            visible_centers.append(xc)
            continue

        # uncertain
        if diam <= max_cell_diam_m:
            boundary_centers.append(xc)
            continue

        # refine (subdivide into 4 spherical triangles)
        m01 = normalize(u0 + u1)
        m12 = normalize(u1 + u2)
        m20 = normalize(u2 + u0)

        queue.append((u0, m01, m20))
        queue.append((u1, m12, m01))
        queue.append((u2, m20, m12))
        queue.append((m01, m12, m20))

        if len(queue) > max_triangles:
            raise RuntimeError(
                f"Exceeded max_triangles={max_triangles}. Increase limit or increase max_cell_diam_m."
            )

    visible_pts = (
        np.array(visible_centers, dtype=np.float64)
        if visible_centers
        else np.zeros((0, 3), dtype=np.float64)
    )
    boundary_pts = (
        np.array(boundary_centers, dtype=np.float64)
        if boundary_centers
        else np.zeros((0, 3), dtype=np.float64)
    )
    return visible_pts, boundary_pts


def to_latlon_deg(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, _h = ecef2geodetic_vec_xyz(
        xyz[:, 0].astype(np.float64),
        xyz[:, 1].astype(np.float64),
        xyz[:, 2].astype(np.float64),
    )
    return np.degrees(lat), np.degrees(lon)


def main() -> None:
    # Example geometry: 3 near-co-located observers (adjust to your scenario)
    a = 6378137.0
    h = 5500e3
    r = a + h
    deg = np.pi / 180.0
    lons = np.array([0.0, 1.0, -1.0]) * deg
    obs = np.stack(
        [r * np.cos(lons), r * np.sin(lons), np.zeros_like(lons)], axis=1
    ).astype(np.float64)

    sig = np.array([500e-9, 500e-9, 500e-9], dtype=np.float64)  # 1σ seconds

    prf_hz, pri_u_us, pri_l_us, gap_us, leaf_xyz, leaf_fub = (
        max_unambiguous_prf_collect_leaf_centers(
            obs,
            sig,
            tol_pri_us=1.0,
            k_sigma=3.0,
            max_nodes=50_000_000,
            max_iters=1_000_000,
            max_plot_points=10_000_000,
        )
    )

    print("PRF_safe_hz:", prf_hz)
    print("PRI_upper_us:", pri_u_us)
    print("PRI_lower_us:", pri_l_us)
    print("PRI_gap_us:", gap_us)
    print("Leaf points returned:", leaf_xyz.shape[0])

    lat_deg, lon_deg = ecef_to_latlon_deg(leaf_xyz)

    # 2D lat/lon scatter (best for coverage sanity checks)
    plt.figure()
    sc = plt.scatter(lon_deg, lat_deg, c=leaf_fub, linewidths=0)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(
        "Leaf cell centers (WGS84 surface) after PRF solve\ncolored by cell F upper bound"
    )
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linewidth=0.3)
    plt.colorbar(sc, label="F upper bound (s)")
    plt.tight_layout()

    # # Optional: 3D ECEF scatter (can be slower)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(leaf_xyz[:, 0], leaf_xyz[:, 1], leaf_xyz[:, 2], s=0.5)
    # ax.set_title("Leaf cell centers in ECEF")
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("y (m)")
    # ax.set_zlabel("z (m)")

    # # Equal-ish scaling
    # xyz = leaf_xyz
    # max_range = np.ptp(xyz, axis=0).max()
    # mid = xyz.mean(axis=0)
    # ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    # ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    # ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    visible_xyz, boundary_xyz = common_visible_footprint_points(
        obs,
        max_cell_diam_m=5_000.0,  # tighten to 10_000 for a sharper boundary
        max_triangles=2_000_000,
    )

    print("visible leaf centers:", visible_xyz.shape[0])
    print("boundary leaf centers:", boundary_xyz.shape[0])

    # Plot in lat/lon
    plt.figure()
    if visible_xyz.shape[0] > 0:
        lat, lon = to_latlon_deg(visible_xyz)
        plt.scatter(lon, lat, s=8, linewidths=0, label="definitely visible")
    if boundary_xyz.shape[0] > 0:
        latb, lonb = to_latlon_deg(boundary_xyz)
        plt.scatter(lonb, latb, s=8, linewidths=0, label="boundary/uncertain")

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Common-visible footprint on WGS84 (adaptive icosphere refinement)")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linewidth=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
