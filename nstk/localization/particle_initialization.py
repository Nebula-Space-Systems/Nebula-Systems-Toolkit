"""
Fast coarse 3D candidate (particle) initializer around the WGS84 ellipsoid.

Changes vs prior version (fixes segfault + faster)
--------------------------------------------------
1) NO np.argsort inside Numba (common segfault source on huge arrays).
2) One-per-voxel thinning via open-addressing hash table (uint64 keys).
3) Stream generation in chunks per band (lower peak memory).
4) Entire pipeline (edges->counts->generate->thin) is Numba-owned.

Notes
-----
- First run includes JIT compile cost. Benchmark the second run.
- Output ECEF points are float32 meters.
"""

import math
import time
import numpy as np
from numba import njit, prange

# =============================================================================
# WGS84 constants
# =============================================================================

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)

INV_A2 = 1.0 / (A * A)
INV_B2 = 1.0 / (B * B)

TWO_PI = 6.283185307179586476925286766559
FOUR_PI = 12.566370614359172953850573533118

# =============================================================================
# Default altitude region (meters)
# =============================================================================

H_MIN = -200.0
H_FADE_START = 8_000.0
H_DENSE_MAX = 15_000.0
H_MAX = 100_000.0

# =============================================================================
# LoS blocking sphere (optional mask later)
# =============================================================================

R_BLOCK_EARTH = B - 1000.0
R_BLOCK_SQUARED = R_BLOCK_EARTH * R_BLOCK_EARTH

# =============================================================================
# Stateless RNG
# =============================================================================


@njit(cache=True, inline="always")
def splitmix64(x: np.uint64) -> np.uint64:
    """Deterministic 64-bit SplitMix PRNG step."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def u01_from_uint64(x: np.uint64) -> float:
    """Map a uint64 to a uniform float in [0, 1)."""
    # 53-bit mantissa -> [0,1)
    return float(x >> np.uint64(11)) * 1.1102230246251565e-16


# =============================================================================
# Geometry
# =============================================================================


@njit(cache=True, inline="always", fastmath=True)
def ellipsoid_intersect_scale(ux: float, uy: float, uz: float) -> float:
    """Scale factor t such that t*(u) lies on the WGS84 ellipsoid."""
    denom = (ux * ux + uy * uy) * INV_A2 + (uz * uz) * INV_B2
    return 1.0 / math.sqrt(denom)


@njit(cache=True, inline="always", fastmath=True)
def ellipsoid_unit_normal_from_dir(ux: float, uy: float, uz: float):
    """Unit surface normal at ellipsoid intersection along direction u."""
    # normal direction at the intersection point; scale cancels
    nx = ux * INV_A2
    ny = uy * INV_A2
    nz = uz * INV_B2
    invn = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * invn, ny * invn, nz * invn


# =============================================================================
# Spacing model (njit)
# =============================================================================


@njit(cache=True, inline="always", fastmath=True)
def spacing_radius(
    h: float,
    h_fade_start: float,
    h_dense_max: float,
    r_terrain: float,
    r_air: float,
    h_grow: float,
    beta: float,
) -> float:
    """Altitude-dependent voxel spacing model (meters)."""
    if h <= h_fade_start:
        return r_terrain
    if h <= h_dense_max:
        t = (h - h_fade_start) / (h_dense_max - h_fade_start)
        return (1.0 - t) * r_terrain + t * r_air
    x = (h - h_dense_max) / h_grow
    return r_air * (1.0 + x) ** beta


# =============================================================================
# Band edges (njit)
# =============================================================================

MAX_EDGES = 4096  # safety cap


@njit(cache=True)
def make_altitude_bands_numba(
    h_min: float,
    h_fade_start: float,
    h_dense_max: float,
    h_max: float,
    dh_surface: float,
    dh0: float,
    growth: float,
    dh_max: float,
):
    """Construct monotonic altitude band edges for candidate generation."""
    edges = np.empty(MAX_EDGES, dtype=np.float32)
    k = 0

    # Surface region
    L0 = float(h_fade_start - h_min)
    n0 = max(1, int(math.ceil(L0 / dh_surface)))
    step0 = L0 / n0
    for i in range(n0 + 1):
        edges[k] = np.float32(h_min + step0 * i)
        k += 1

    # Dense region (exclude duplicate at start)
    L1 = float(h_dense_max - h_fade_start)
    n1 = max(1, int(math.ceil(L1 / dh_surface)))
    step1 = L1 / n1 if n1 > 0 else 0.0
    for i in range(1, n1 + 1):
        if k >= MAX_EDGES:
            break
        edges[k] = np.float32(h_fade_start + step1 * i)
        k += 1

    # Above dense max (geometric growth)
    h = float(edges[k - 1])
    dh = float(dh0)
    while h < h_max and k < MAX_EDGES:
        step = dh if dh < dh_max else dh_max
        h2 = h + step
        if h2 > h_max:
            h2 = h_max
        edges[k] = np.float32(h2)
        k += 1
        h = h2
        dh *= growth
        if step <= 0.0:
            break

    return edges, k


# =============================================================================
# Candidate counts (occupancy-driven) + safe keep upper bound per band (njit)
# =============================================================================


@njit(cache=True, fastmath=True)
def compute_band_counts_occ_numba(
    edges: np.ndarray,
    n_edges: int,
    occupancy: float,
    min_per_band: int,
    max_per_band: int,
    keep_inflate: float,  # safety factor for bounding unique voxels
    # spacing params
    h_fade_start: float,
    h_dense_max: float,
    r_terrain: float,
    r_air: float,
    h_grow: float,
    beta: float,
):
    """Compute per-band candidate counts and unique-voxel upper bounds."""
    nb = n_edges - 1
    counts = np.empty(nb, dtype=np.int64)
    keep_upper = np.empty(nb, dtype=np.int64)

    p = occupancy
    if p < 1e-6:
        p = 1e-6
    if p > 0.999999:
        p = 0.999999
    ln_term = -math.log(1.0 - p)  # M = -Vvox ln(1-p)

    total_cand = 0
    total_keep = 0

    for b in range(nb):
        h0 = float(edges[b])
        h1 = float(edges[b + 1])
        hc = 0.5 * (h0 + h1)
        dh = h1 - h0

        voxel = spacing_radius(
            hc, h_fade_start, h_dense_max, r_terrain, r_air, h_grow, beta
        )
        R = A + (hc if hc > 0.0 else 0.0)
        Vband = FOUR_PI * R * R * dh
        Vvox = Vband / (voxel * voxel * voxel)

        n_cand = int(math.ceil(ln_term * Vvox))
        if n_cand < min_per_band:
            n_cand = min_per_band
        elif n_cand > max_per_band:
            n_cand = max_per_band

        # Safe-ish upper bound for unique voxels hit by this shell
        # (inflate handles discretization/boundary effects)
        vk = int(math.ceil(Vvox * keep_inflate)) + 1024
        if vk > n_cand:
            vk = n_cand
        if vk < 0:
            vk = 0

        counts[b] = n_cand
        keep_upper[b] = vk

        total_cand += n_cand
        total_keep += vk

    return counts, keep_upper, total_cand, total_keep


# =============================================================================
# Voxel keying (pack 3 int indices into 64 bits)
#   - Uses 3x21-bit fields => assumes abs(index) < 2^20 for strict uniqueness.
#   - With 6km spacing, indices ~ O(1000). Safe.
# =============================================================================

SHIFT = 21
OFFSET = 1 << 20  # abs(index) must be < OFFSET for collision-free packing


@njit(cache=True, inline="always")
def pack_key(ix: int, iy: int, iz: int) -> np.uint64:
    """Pack 3 integer voxel indices into one 64-bit key."""
    return (
        np.uint64(ix + OFFSET)
        | (np.uint64(iy + OFFSET) << np.uint64(SHIFT))
        | (np.uint64(iz + OFFSET) << np.uint64(2 * SHIFT))
    )


# =============================================================================
# Hash table utilities (njit)
# =============================================================================


@njit(cache=True, inline="always")
def next_pow2_u64(x: np.uint64) -> np.uint64:
    """Return the next power of two >= x."""
    if x <= np.uint64(1):
        return np.uint64(1)
    x -= np.uint64(1)
    x |= x >> np.uint64(1)
    x |= x >> np.uint64(2)
    x |= x >> np.uint64(4)
    x |= x >> np.uint64(8)
    x |= x >> np.uint64(16)
    x |= x >> np.uint64(32)
    return x + np.uint64(1)


@njit(cache=True, inline="always")
def hash_pos(key: np.uint64, mask: np.uint64) -> np.uint64:
    """Hash key into table slot using multiplicative hashing and mask."""
    # Knuth multiplicative hashing
    return (key * np.uint64(11400714819323198485)) & mask


# =============================================================================
# Chunk generation (parallel) + streaming insert (serial), all njit
# =============================================================================


@njit(cache=True, parallel=True, fastmath=True)
def generate_chunk_points_and_keys(
    n: int,
    h0: float,
    h1: float,
    voxel: float,
    ox: float,
    oy: float,
    oz: float,
    seed: np.uint64,
    base_idx: np.uint64,
):
    """Generate one chunk of ECEF candidates and their voxel keys."""
    pts = np.empty((n, 3), dtype=np.float32)
    keys = np.empty(n, dtype=np.uint64)

    inv = 1.0 / voxel

    for j in prange(n):
        idx = base_idx + np.uint64(j)

        u1 = u01_from_uint64(splitmix64(seed ^ (idx * np.uint64(3) + np.uint64(0))))
        u2 = u01_from_uint64(splitmix64(seed ^ (idx * np.uint64(3) + np.uint64(1))))
        u3 = u01_from_uint64(splitmix64(seed ^ (idx * np.uint64(3) + np.uint64(2))))

        z = 1.0 - 2.0 * u1
        rr = 1.0 - z * z
        if rr < 0.0:
            rr = 0.0
        r = math.sqrt(rr)
        theta = TWO_PI * u2
        ux = r * math.cos(theta)
        uy = r * math.sin(theta)
        uz = z

        h = h0 + u3 * (h1 - h0)

        t = ellipsoid_intersect_scale(ux, uy, uz)
        nx, ny, nz = ellipsoid_unit_normal_from_dir(ux, uy, uz)

        x = t * ux + h * nx
        y = t * uy + h * ny
        zc = t * uz + h * nz

        pts[j, 0] = np.float32(x)
        pts[j, 1] = np.float32(y)
        pts[j, 2] = np.float32(zc)

        fx = (x + ox) * inv
        fy = (y + oy) * inv
        fz = (zc + oz) * inv

        ix = int(math.floor(fx))
        iy = int(math.floor(fy))
        iz = int(math.floor(fz))

        keys[j] = pack_key(ix, iy, iz)

    return pts, keys


@njit(cache=True, fastmath=True)
def insert_unique_append(
    pts: np.ndarray,
    keys: np.ndarray,
    table: np.ndarray,
    table_mask: np.uint64,
    empty_key: np.uint64,
    out: np.ndarray,
    out_pos: int,
):
    """
    For each (pt,key), if key not in table insert and append pt to out.
    Returns new out_pos.
    """
    n = keys.shape[0]
    for i in range(n):
        k = keys[i]
        pos = hash_pos(k, table_mask)

        while True:
            cur = table[pos]
            if cur == k:
                break
            if cur == empty_key:
                table[pos] = k
                out[out_pos, 0] = pts[i, 0]
                out[out_pos, 1] = pts[i, 1]
                out[out_pos, 2] = pts[i, 2]
                out_pos += 1
                break
            pos = (pos + np.uint64(1)) & table_mask

    return out_pos


# =============================================================================
# Full pipeline (njit)
# =============================================================================


@njit(cache=True, fastmath=True)
def make_coarse_candidates_ecef_fast(
    # altitude region
    h_min: float,
    h_fade_start: float,
    h_dense_max: float,
    h_max: float,
    # banding
    dh_surface: float,
    dh0: float,
    growth: float,
    dh_max: float,
    # occupancy-driven candidate count
    occupancy: float,
    min_per_band: int,
    max_per_band: int,
    keep_inflate: float,
    # spacing params
    r_terrain: float,
    r_air: float,
    h_grow: float,
    beta: float,
    # chunking
    chunk_size: int,
    # RNG seed
    seed_i64: int,
):
    """
    Generate coarse ECEF particle candidates with voxel-based thinning.

    This is the high-performance Numba core used by
    ``make_coarse_candidates_ecef``. It samples points over altitude bands
    around WGS84, then keeps at most one candidate per voxel (per band) using
    an open-addressing hash table.

    Returns
    -------
    np.ndarray
        ECEF points in meters with shape ``(N, 3)`` and dtype ``float32``.
    """
    seed = np.uint64(seed_i64)

    edges, n_edges = make_altitude_bands_numba(
        h_min, h_fade_start, h_dense_max, h_max, dh_surface, dh0, growth, dh_max
    )

    counts, keep_upper, total_cand, total_keep = compute_band_counts_occ_numba(
        edges,
        n_edges,
        occupancy,
        min_per_band,
        max_per_band,
        keep_inflate,
        h_fade_start,
        h_dense_max,
        r_terrain,
        r_air,
        h_grow,
        beta,
    )

    out = np.empty((total_keep, 3), dtype=np.float32)
    out_pos = 0

    nb = n_edges - 1
    global_base = np.uint64(0)

    EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)

    for b in range(nb):
        n_cand = int(counts[b])
        if n_cand <= 0:
            continue

        h0 = float(edges[b])
        h1 = float(edges[b + 1])
        hc = 0.5 * (h0 + h1)

        voxel = spacing_radius(
            hc, h_fade_start, h_dense_max, r_terrain, r_air, h_grow, beta
        )

        # deterministic per-band voxel-grid offset in [0, voxel)
        s0 = splitmix64(seed ^ np.uint64(b * 3 + 0))
        s1 = splitmix64(seed ^ np.uint64(b * 3 + 1))
        s2 = splitmix64(seed ^ np.uint64(b * 3 + 2))
        ox = u01_from_uint64(s0) * voxel
        oy = u01_from_uint64(s1) * voxel
        oz = u01_from_uint64(s2) * voxel

        # hash table sized from keep_upper[b] and a target load factor
        ku = int(keep_upper[b])
        if ku < 8:
            ku = 8
        # load factor <= ~0.65
        need = np.uint64(int(ku / 0.65) + 16)
        table_size = next_pow2_u64(need)
        if table_size < np.uint64(16):
            table_size = np.uint64(16)
        table = np.empty(int(table_size), dtype=np.uint64)
        table.fill(EMPTY)
        mask = table_size - np.uint64(1)

        # stream in chunks
        start = 0
        while start < n_cand:
            m = chunk_size
            if start + m > n_cand:
                m = n_cand - start

            pts, keys = generate_chunk_points_and_keys(
                m, h0, h1, voxel, ox, oy, oz, seed, global_base + np.uint64(start)
            )

            out_pos = insert_unique_append(pts, keys, table, mask, EMPTY, out, out_pos)

            start += m

        global_base += np.uint64(n_cand)

    return out[:out_pos]


# =============================================================================
# Optional: LoS utilities (njit)
# =============================================================================


@njit(cache=True, inline="always", fastmath=True)
def los_clear_sphere(
    ox: float, oy: float, oz: float, px: float, py: float, pz: float
) -> bool:
    """
    Return True when the segment observer->point does not intersect blocker.

    The blocker is a sphere centered at the origin with radius
    ``R_BLOCK_EARTH``.
    """
    dx = px - ox
    dy = py - oy
    dz = pz - oz

    dd = dx * dx + dy * dy + dz * dz
    if dd <= 1e-30:
        return (px * px + py * py + pz * pz) >= R_BLOCK_SQUARED

    t = -(ox * dx + oy * dy + oz * dz) / dd

    if t <= 0.0:
        cx, cy, cz = ox, oy, oz
    elif t >= 1.0:
        cx, cy, cz = px, py, pz
    else:
        cx = ox + t * dx
        cy = oy + t * dy
        cz = oz + t * dz

    return (cx * cx + cy * cy + cz * cz) >= R_BLOCK_SQUARED


@njit(cache=True, parallel=True, fastmath=True)
def los_mask_all_observers(
    points_ecef: np.ndarray, observers_ecef: np.ndarray
) -> np.ndarray:
    """
    Compute line-of-sight keep mask for candidates against all observers.

    A candidate is marked ``True`` only if it is line-of-sight clear from every
    observer according to ``los_clear_sphere``.

    Parameters
    ----------
    points_ecef : np.ndarray
        Candidate points, shape ``(N, 3)`` in ECEF meters.
    observers_ecef : np.ndarray
        Observer positions, shape ``(M, 3)`` in ECEF meters.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(N,)``.
    """
    n = points_ecef.shape[0]
    m = observers_ecef.shape[0]
    keep = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        px = float(points_ecef[i, 0])
        py = float(points_ecef[i, 1])
        pz = float(points_ecef[i, 2])
        ok = True
        for j in range(m):
            ox = float(observers_ecef[j, 0])
            oy = float(observers_ecef[j, 1])
            oz = float(observers_ecef[j, 2])
            if not los_clear_sphere(ox, oy, oz, px, py, pz):
                ok = False
                break
        keep[i] = ok

    return keep


# =============================================================================
# Thin Python wrapper
# =============================================================================


def make_coarse_candidates_ecef(
    h_min: float = H_MIN,
    h_fade_start: float = H_FADE_START,
    h_dense_max: float = H_DENSE_MAX,
    h_max: float = H_MAX,
    dh_surface: float = 1500.0,
    dh0: float = 2000.0,
    growth: float = 1.25,
    dh_max: float = 50_000.0,
    occupancy: float = 0.90,  # try 0.85–0.92
    min_per_band: int = 50_000,
    max_per_band: int = 2_000_000,
    keep_inflate: float = 1.60,  # safety factor for bounding voxels-per-shell
    r_terrain: float = 6_000.0,
    r_air: float = 10_000.0,
    h_grow: float = 60_000.0,
    beta: float = 0.75,
    chunk_size: int = 400_000,  # memory/speed trade; 200k–800k typical
    seed: int = 123,
) -> np.ndarray:
    """
    Build a coarse set of localization particles in ECEF coordinates.

    This is the public wrapper around ``make_coarse_candidates_ecef_fast`` and
    the recommended entry point for initializing a coarse candidate cloud.

    Parameters
    ----------
    h_min, h_fade_start, h_dense_max, h_max : float, optional
        Altitude model breakpoints in meters.
    dh_surface, dh0, growth, dh_max : float, optional
        Altitude band construction controls.
    occupancy : float, optional
        Target voxel occupancy used to estimate candidate counts.
    min_per_band, max_per_band : int, optional
        Candidate count clamps per altitude band.
    keep_inflate : float, optional
        Safety factor for estimating unique voxels per shell.
    r_terrain, r_air, h_grow, beta : float, optional
        Spacing model parameters controlling voxel size vs altitude.
    chunk_size : int, optional
        Streaming generation chunk size.
    seed : int, optional
        Deterministic random seed.

    Returns
    -------
    np.ndarray
        Candidate ECEF points with shape ``(N, 3)`` and dtype ``float32``
        (meters).
    """
    return make_coarse_candidates_ecef_fast(
        h_min,
        h_fade_start,
        h_dense_max,
        h_max,
        dh_surface,
        dh0,
        growth,
        dh_max,
        occupancy,
        min_per_band,
        max_per_band,
        keep_inflate,
        r_terrain,
        r_air,
        h_grow,
        beta,
        chunk_size,
        seed,
    )


# =============================================================================
# Main (benchmark)
# =============================================================================

if __name__ == "__main__":
    # Warmup compile (do not time)
    _ = make_coarse_candidates_ecef(occupancy=0.10, seed=1, chunk_size=200_000)
    print("Warmup done.", flush=True)

    t0 = time.time()
    particles = make_coarse_candidates_ecef(
        occupancy=0.99, seed=123, chunk_size=400_000, h_max=H_MAX * 2
    )
    t1 = time.time()

    print(f"Kept points: {particles.shape[0]:,}", flush=True)
    print(f"Time (post-compile): {t1 - t0:.2f} s", flush=True)

    # Example LoS usage (you provide observers_ecef)
    # observers = np.array([obs1, obs2, ...], dtype=np.float64)
    # m = los_mask_all_observers(particles, observers)
    # particles_los = particles[m]
    # print("After LoS:", particles_los.shape[0], flush=True)
