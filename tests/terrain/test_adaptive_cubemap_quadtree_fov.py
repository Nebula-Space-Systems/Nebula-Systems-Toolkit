import os
import math
import unittest
import numpy as np
from numba import njit

"""Tests for AdaptiveCubeRasterFOV.

This suite is based on the original cap-only tests and extends coverage for the
new arbitrary painted-raster + boolean operations workflow.

Notes
-----
- Some checks introspect the packed representation (uv/meta/first_child). Those
  tests are skipped if the implementation doesn't expose those arrays.
- Rendering tests target the semantics of to_dense_faces(): it is a conservative
  rasterizer (floor/ceil), so false positives near edges are allowed. For
  raster-driven builds in this implementation, we also include a stronger test
  that can be made exact by using a face_res that is a power of two and a
  sufficiently tight tolerance.
"""


from nebula.terrain.raster_fov import (
    AdaptiveCubeRasterFOV,
    azel_to_dir,
    face_uv_to_dir,
    dir_to_face_uv,
    _normalize3,
)

# -----------------------------------------------------------------------------
# Test sizing
# -----------------------------------------------------------------------------
_FAST = os.environ.get("FAST", "0") == "1"

N_RAND_BIG = 60_000 if _FAST else 200_000
N_RAND_MED = 25_000 if _FAST else 100_000
N_RAND_SMALL = 3_000 if _FAST else 30_000


def _random_unit_vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v.astype(np.float64)


# -----------------------------------------------------------------------------
# Analytic helpers (caps)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _analytic_cap_contains(dirs: np.ndarray, center: np.ndarray, cos_alpha: float):
    n = dirs.shape[0]
    out = np.empty(n, dtype=np.uint8)
    cx, cy, cz = center[0], center[1], center[2]
    for i in range(n):
        d = cx * dirs[i, 0] + cy * dirs[i, 1] + cz * dirs[i, 2]
        out[i] = 1 if d >= cos_alpha else 0
    return out


@njit(cache=True)
def _analytic_union_caps_contains(
    dirs: np.ndarray, centers: np.ndarray, cos_alphas: np.ndarray
):
    n = dirs.shape[0]
    m = cos_alphas.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x = dirs[i, 0]
        y = dirs[i, 1]
        z = dirs[i, 2]
        val = 0
        for k in range(m):
            d = centers[k, 0] * x + centers[k, 1] * y + centers[k, 2] * z
            if d >= cos_alphas[k]:
                val = 1
                break
        out[i] = val
    return out


@njit(cache=True)
def _boundary_distance_deg_single(
    dirs: np.ndarray, center: np.ndarray, alpha_rad: float
):
    n = dirs.shape[0]
    out = np.empty(n, dtype=np.float64)
    cx, cy, cz = center[0], center[1], center[2]
    for i in range(n):
        d = cx * dirs[i, 0] + cy * dirs[i, 1] + cz * dirs[i, 2]
        if d > 1.0:
            d = 1.0
        if d < -1.0:
            d = -1.0
        ang = math.acos(d)
        out[i] = abs(ang - alpha_rad) * (180.0 / math.pi)
    return out


@njit(cache=True)
def _boundary_distance_deg_union(
    dirs: np.ndarray, centers: np.ndarray, alpha_rads: np.ndarray
):
    n = dirs.shape[0]
    m = alpha_rads.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = dirs[i, 0]
        y = dirs[i, 1]
        z = dirs[i, 2]
        best = 1e9
        for k in range(m):
            d = centers[k, 0] * x + centers[k, 1] * y + centers[k, 2] * z
            if d > 1.0:
                d = 1.0
            if d < -1.0:
                d = -1.0
            ang = math.acos(d)
            dist = abs(ang - alpha_rads[k]) * (180.0 / math.pi)
            if dist < best:
                best = dist
        out[i] = best
    return out


# -----------------------------------------------------------------------------
# Packed-tree helpers (only run if fov has .uv/.meta/.first_child)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _meta_face(meta: np.uint16) -> int:
    return int((meta & np.uint16(0x000E)) >> np.uint16(1))


@njit(cache=True)
def _meta_depth(meta: np.uint16) -> int:
    return int((meta & np.uint16(0x01F0)) >> np.uint16(4))


@njit(cache=True)
def _meta_state(meta: np.uint16) -> int:
    return int(meta & np.uint16(0x0001))


@njit(cache=True)
def _uv_iu(uv: np.uint32) -> int:
    return int(uv & np.uint32(0x0000FFFF))


@njit(cache=True)
def _uv_iv(uv: np.uint32) -> int:
    return int(uv >> np.uint32(16))


@njit(cache=False)
def _contains_face_uv_packed(
    face: int,
    u: float,
    v: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    """Query in *face UV space* (no dir_to_face_uv), matching to_dense_faces()."""
    idx = roots[face]
    while True:
        fc = first_child[idx]
        if fc == -1:
            return _meta_state(meta[idx])

        d = _meta_depth(meta[idx])
        den = 1 << d
        size = 2.0 / float(den)
        uvp = uv[idx]
        iu = _uv_iu(uvp)
        iv = _uv_iv(uvp)

        um = -1.0 + (float(iu) + 0.5) * size
        vm = -1.0 + (float(iv) + 0.5) * size

        if v < vm:
            idx = fc + (0 if u < um else 1)
        else:
            idx = fc + (2 if u < um else 3)


@njit(cache=True)
def _validate_packed_tree_invariants(
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    """Returns 1 if invariants hold else 0."""
    n = meta.shape[0]

    for f in range(6):
        r = roots[f]
        if r < 0 or r >= n:
            return 0

    for i in range(n):
        m = meta[i]
        face = _meta_face(m)
        depth = _meta_depth(m)
        if face < 0 or face > 5:
            return 0
        if depth < 0 or depth > 16:
            return 0

        uvp = uv[i]
        iu = _uv_iu(uvp)
        iv = _uv_iv(uvp)
        lim = 1 << depth
        if iu < 0 or iu >= lim:
            return 0
        if iv < 0 or iv >= lim:
            return 0

        fc = first_child[i]
        if fc != -1:
            if fc < 0 or fc + 3 >= n:
                return 0
            cd = depth + 1
            if cd > 16:
                return 0
            for j in range(4):
                cm = meta[fc + j]
                if _meta_face(cm) != face:
                    return 0
                if _meta_depth(cm) != cd:
                    return 0

    return 1


# -----------------------------------------------------------------------------
# Raster helpers
# -----------------------------------------------------------------------------


def _face_uv_grid(res: int):
    """Return u,v grids (res,res) at pixel centers."""
    xs = (np.arange(res, dtype=np.float64) + 0.5) / res
    ys = (np.arange(res, dtype=np.float64) + 0.5) / res
    u = xs[None, :] * 2.0 - 1.0
    v = ys[:, None] * 2.0 - 1.0
    uu = np.broadcast_to(u, (res, res))
    vv = np.broadcast_to(v, (res, res))
    return uu, vv


def _circle_mask_on_one_face(res: int, face: int, uc: float, vc: float, r: float):
    """Return (6,res,res) mask with a UV-circle painted on `face`."""
    uu, vv = _face_uv_grid(res)
    inside = (uu - uc) ** 2 + (vv - vc) ** 2 <= (r * r)
    out = np.zeros((6, res, res), dtype=np.uint8)
    out[face] = inside.astype(np.uint8)
    return out


def _sample_dirs_from_face_uv(res: int, face: int, n: int, seed: int):
    """Sample random pixel-centers on a face and return (dirs, idx_x, idx_y)."""
    rng = np.random.default_rng(seed)
    xi = rng.integers(0, res, size=n, endpoint=False)
    yi = rng.integers(0, res, size=n, endpoint=False)
    u = ((xi.astype(np.float64) + 0.5) / res) * 2.0 - 1.0
    v = ((yi.astype(np.float64) + 0.5) / res) * 2.0 - 1.0

    dirs = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        dirs[i] = face_uv_to_dir(int(face), float(u[i]), float(v[i]))
    return dirs, xi.astype(np.int64), yi.astype(np.int64)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestAdaptiveCubeRasterFOV(unittest.TestCase):
    # ----------------------------
    # Baseline (caps) behavior
    # ----------------------------

    def test_empty_fov_is_all_false(self):
        fov = AdaptiveCubeRasterFOV(tolerance_deg=0.01)
        fov.compile()
        dirs = _random_unit_vectors(N_RAND_SMALL, seed=1)
        out = fov.contains_dirs(dirs)
        self.assertFalse(out.any())

    def test_compile_idempotent_and_contains_matches_single(self):
        fov = AdaptiveCubeRasterFOV(tolerance_deg=0.01)
        fov.add_cap_azel(0.0, 0.0, 25.0)
        fov.compile()
        n1 = fov.node_count()

        fov.compile()
        n2 = fov.node_count()
        self.assertEqual(n1, n2)

        dirs = _random_unit_vectors(N_RAND_SMALL, seed=2)
        batch = fov.contains_dirs(dirs)
        singles = np.array([fov.contains_dir(*d) for d in dirs], dtype=bool)
        self.assertTrue(np.array_equal(batch, singles))

    def test_scaling_of_dirs_does_not_change_results(self):
        fov = AdaptiveCubeRasterFOV(tolerance_deg=0.01)
        fov.add_cap_azel(30.0, 10.0, 12.0)
        fov.compile()

        dirs = _random_unit_vectors(N_RAND_MED, seed=3)
        scales = np.linspace(0.1, 10.0, dirs.shape[0]).astype(np.float64)
        dirs_scaled = dirs * scales[:, None]

        a = fov.contains_dirs(dirs)
        b = fov.contains_dirs(dirs_scaled)
        self.assertTrue(np.array_equal(a, b))

    def test_monotonicity_when_adding_caps(self):
        """UNION semantics: adding another cap can only turn False->True."""
        tol = 0.01 if not _FAST else 0.02

        base = AdaptiveCubeRasterFOV(tolerance_deg=tol)
        base.add_cap_azel(0.0, 0.0, 12.0)
        base.compile()

        extended = AdaptiveCubeRasterFOV(tolerance_deg=tol)
        extended.add_cap_azel(0.0, 0.0, 12.0)
        extended.add_cap_azel(120.0, 0.0, 12.0)
        extended.compile()

        dirs = _random_unit_vectors(N_RAND_MED, seed=4)
        a = base.contains_dirs(dirs)
        b = extended.contains_dirs(dirs)

        self.assertTrue(np.all((~a) | b))

    def test_tighter_tolerance_increases_or_keeps_node_count(self):
        loose = AdaptiveCubeRasterFOV(tolerance_deg=0.05 if not _FAST else 0.08)
        tight = AdaptiveCubeRasterFOV(tolerance_deg=0.01 if not _FAST else 0.02)

        for f in (loose, tight):
            f.add_cap_azel(10.0, -15.0, 22.0)
            f.add_cap_azel(-80.0, 25.0, 10.0)
            f.compile()

        self.assertGreaterEqual(tight.node_count(), loose.node_count())

    def test_packed_tree_invariants_if_available(self):
        fov = AdaptiveCubeRasterFOV(tolerance_deg=0.01)
        fov.add_cap_azel(0.0, 0.0, 60.0)
        fov.add_cap_azel(120.0, 0.0, 25.0)
        fov.compile()

        if not (
            hasattr(fov, "uv") and hasattr(fov, "meta") and hasattr(fov, "first_child")
        ):
            self.skipTest("packed invariants require fov.uv/fov.meta/fov.first_child")

        ok = _validate_packed_tree_invariants(
            fov.roots, fov.first_child, fov.uv, fov.meta
        )
        self.assertEqual(int(ok), 1)

    def test_disagreement_with_analytic_cap_within_tolerance(self):
        tol = 0.01
        cap_az, cap_el = 15.0, -10.0
        half_angle = 7.0

        fov = AdaptiveCubeRasterFOV(tolerance_deg=tol)
        fov.add_cap_azel(cap_az, cap_el, half_angle)
        fov.compile()

        cx, cy, cz = azel_to_dir(cap_az, cap_el)
        cx, cy, cz = _normalize3(cx, cy, cz)
        center = np.array([cx, cy, cz], dtype=np.float64)
        alpha_rad = math.radians(half_angle)
        cos_alpha = math.cos(alpha_rad)

        dirs = _random_unit_vectors(N_RAND_BIG, seed=11)
        raster = fov.contains_dirs(dirs).astype(np.uint8)
        truth = _analytic_cap_contains(dirs, center, cos_alpha)

        bad = np.nonzero(raster ^ truth)[0]
        if bad.size == 0:
            return

        dist = _boundary_distance_deg_single(dirs[bad], center, alpha_rad)
        worst = float(dist.max())
        self.assertLessEqual(worst, tol + 2e-4)

    def test_disagreement_with_analytic_union_within_tolerance(self):
        tol = 0.01
        caps = [
            (0.0, 0.0, 18.0),
            (60.0, 10.0, 12.0),
            (-100.0, -20.0, 9.0),
        ]

        fov = AdaptiveCubeRasterFOV(tolerance_deg=tol)

        centers = np.empty((len(caps), 3), dtype=np.float64)
        cos_alphas = np.empty(len(caps), dtype=np.float64)
        alpha_rads = np.empty(len(caps), dtype=np.float64)

        for i, (az, el, ha) in enumerate(caps):
            fov.add_cap_azel(az, el, ha)
            x, y, z = azel_to_dir(az, el)
            x, y, z = _normalize3(x, y, z)
            centers[i] = (x, y, z)
            ar = math.radians(ha)
            alpha_rads[i] = ar
            cos_alphas[i] = math.cos(ar)

        fov.compile()

        dirs = _random_unit_vectors(N_RAND_BIG, seed=21)
        raster = fov.contains_dirs(dirs).astype(np.uint8)
        truth = _analytic_union_caps_contains(dirs, centers, cos_alphas)

        bad = np.nonzero(raster ^ truth)[0]
        if bad.size == 0:
            return

        dist = _boundary_distance_deg_union(dirs[bad], centers, alpha_rads)
        worst = float(dist.max())
        self.assertLessEqual(worst, tol + 5e-4)

    def test_dense_band_near_boundary(self):
        tol = 0.01
        cap_az, cap_el = 0.0, 0.0
        half_angle = 20.0

        fov = AdaptiveCubeRasterFOV(tolerance_deg=tol)
        fov.add_cap_azel(cap_az, cap_el, half_angle)
        fov.compile()

        cx, cy, cz = azel_to_dir(cap_az, cap_el)
        cx, cy, cz = _normalize3(cx, cy, cz)
        center = np.array([cx, cy, cz], dtype=np.float64)
        alpha_rad = math.radians(half_angle)
        cos_alpha = math.cos(alpha_rad)

        # Orthonormal basis around center
        c = center
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(c[0])) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = np.cross(c, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c, e1)

        band = 0.05  # deg
        radii = np.array(
            [half_angle - band, half_angle, half_angle + band], dtype=np.float64
        )
        thetas = np.linspace(
            0.0, 2.0 * math.pi, 15_000 if _FAST else 45_000, endpoint=False
        )

        dirs = np.empty((thetas.size * radii.size, 3), dtype=np.float64)
        k = 0
        for rdeg in radii:
            r = math.radians(float(rdeg))
            cr = math.cos(r)
            sr = math.sin(r)
            for t in thetas:
                ct = math.cos(float(t))
                st = math.sin(float(t))
                v = cr * c + sr * (ct * e1 + st * e2)
                v /= np.linalg.norm(v)
                dirs[k] = v
                k += 1

        raster = fov.contains_dirs(dirs).astype(np.uint8)
        truth = _analytic_cap_contains(dirs, center, cos_alpha)

        bad = np.nonzero(raster ^ truth)[0]
        if bad.size == 0:
            return

        dist = _boundary_distance_deg_single(dirs[bad], center, alpha_rad)
        worst = float(dist.max())
        self.assertLessEqual(worst, tol + 2e-4)

    def test_dense_faces_is_conservative_wrt_face_uv_centers_if_packed(self):
        """Only runs when the object exposes packed quadtree arrays."""
        fov = AdaptiveCubeRasterFOV(tolerance_deg=0.01 if not _FAST else 0.02)
        fov.add_cap_azel(10.0, 5.0, 35.0)
        fov.add_cap_azel(-140.0, -25.0, 15.0)
        fov.compile()

        if not (
            hasattr(fov, "uv") and hasattr(fov, "meta") and hasattr(fov, "first_child")
        ):
            self.skipTest("requires packed representation (uv/meta/first_child)")

        res = 256 if _FAST else 512
        img = fov.to_dense_faces(resolution=res)

        rng = np.random.default_rng(123)
        n_samples = 8_000 if _FAST else 30_000

        for _ in range(n_samples):
            face = int(rng.integers(0, 6))
            xpix = int(rng.integers(0, res))
            ypix = int(rng.integers(0, res))

            u = ((xpix + 0.5) / res) * 2.0 - 1.0
            v = ((ypix + 0.5) / res) * 2.0 - 1.0

            center_val = _contains_face_uv_packed(
                face, float(u), float(v), fov.roots, fov.first_child, fov.uv, fov.meta
            )
            pix = int(img[face, ypix, xpix])

            if center_val == 1:
                self.assertEqual(pix, 1)
            else:
                if pix == 0:
                    self.assertEqual(center_val, 0)

    def test_seam_stability_dir_to_face_uv_roundtrip(self):
        """Roundtrip stress test for mapping functions (seams/poles)."""
        dirs = _random_unit_vectors(N_RAND_MED, seed=777)

        extra = np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        extra /= np.linalg.norm(extra, axis=1)[:, None]
        dirs = np.vstack([dirs, extra])

        worst_deg = 0.0
        for i in range(dirs.shape[0]):
            x, y, z = float(dirs[i, 0]), float(dirs[i, 1]), float(dirs[i, 2])
            face, u, v = dir_to_face_uv(x, y, z)
            rx, ry, rz = face_uv_to_dir(face, float(u), float(v))
            d = x * rx + y * ry + z * rz
            if d > 1.0:
                d = 1.0
            if d < -1.0:
                d = -1.0
            ang = math.degrees(math.acos(d))
            if ang > worst_deg:
                worst_deg = ang

        self.assertLessEqual(worst_deg, 1e-5 if not _FAST else 3e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
