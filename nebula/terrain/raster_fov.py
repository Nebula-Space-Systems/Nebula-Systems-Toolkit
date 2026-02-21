# -*- coding: utf-8 -*-
"""
raster_fov.py

Adaptive packed cubemap quadtree for fast boolean FOV queries on the unit sphere.

Key features
-----------
- Packed node storage (first_child:int32, uv:uint32, meta:uint16) for low memory.
- Fast contains_dir / contains_dirs via Numba.
- Multiple ways to create a mask:
    1) Analytic UNION of spherical caps (fast, conservative subdivision only near edges)
    2) From an arbitrary *painted* raster surface:
        - cubemap face masks (6, H, W)
        - az/el equirectangular masks (H, W)
- Boolean operations between compiled FOVs (union / intersection / difference / xor / invert)
  performed directly on the quadtrees, preserving adaptive refinement mainly near edges.
- Visualization helpers for cubemap faces and a single az/el raster (with optional depth panel).

Notes
-----
- Packed UV indices store (iu,iv) as uint16, so max_depth must be <= 16.
- The az/el rasterization uses the convention:
    az=0 along +X, az=+90 along +Y, el=+90 along +Z

"""
from __future__ import annotations

import math
from typing import Literal, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# --- Numba / coverage compatibility -----------------------------------------
# Some environments ship a "coverage" package without coverage.types.Tracer,
# which breaks numba import. This patch is a no-op in normal setups.
try:
    import coverage as _coverage  # type: ignore
    import types as _types

    _need_patch = (not hasattr(_coverage, "types")) or (
        not hasattr(_coverage.types, "Tracer")  # type: ignore
    )
    if _need_patch:
        _coverage.types = _types.SimpleNamespace()  # type: ignore
    for _name in (
        "Tracer",
        "TTraceData",
        "TTraceFn",
        "TFileDisposition",
        "TShouldStartContextFn",
        "TShouldTraceFn",
        "TWarnFn",
    ):
        if not hasattr(_coverage.types, _name):  # type: ignore
            setattr(_coverage.types, _name, object)  # type: ignore
except Exception:
    pass
# ---------------------------------------------------------------------------

from numba import njit, prange, types
from numba.typed import List


# ----------------------------
# Direction <-> Cube map mapping
# ----------------------------

# Face order (standard cubemap convention):
# 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
_FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

# Meta packing (uint16 per node):
# bits 0      : leaf value (0/1)  (only meaningful if first_child == -1)
# bits 1..3   : face id (0..5)
# bits 4..8   : depth (0..31)
# bits 9..15  : reserved
_META_STATE_MASK = np.uint16(0x0001)
_META_FACE_SHIFT = np.uint16(1)
_META_FACE_MASK = np.uint16(0x000E)
_META_DEPTH_SHIFT = np.uint16(4)
_META_DEPTH_MASK = np.uint16(0x01F0)

# UV packing (uint32 per node):
# low  16 bits: iu (0..65535)
# high 16 bits: iv (0..65535)
_UV_I_MASK = np.uint32(0x0000FFFF)
_UV_V_SHIFT = np.uint32(16)

# Boolean op codes (for Numba)
_OP_OR = np.int32(0)
_OP_AND = np.int32(1)
_OP_DIFF = np.int32(2)
_OP_XOR = np.int32(3)


@njit(cache=True)
def _meta_make(face: int, depth: int, state: int) -> np.uint16:
    return np.uint16((depth << 4) | (face << 1) | (state & 1))


@njit(cache=True)
def _meta_face(meta: np.uint16) -> int:
    return int((meta & _META_FACE_MASK) >> _META_FACE_SHIFT)


@njit(cache=True)
def _meta_depth(meta: np.uint16) -> int:
    return int((meta & _META_DEPTH_MASK) >> _META_DEPTH_SHIFT)


@njit(cache=True)
def _meta_state(meta: np.uint16) -> int:
    return int(meta & _META_STATE_MASK)


@njit(cache=True)
def _uv_pack(iu: int, iv: int) -> np.uint32:
    # expects iu,iv in [0, 65535]
    return np.uint32((iv << 16) | (iu & 0xFFFF))


@njit(cache=True)
def _uv_iu(uv: np.uint32) -> int:
    return int(uv & _UV_I_MASK)


@njit(cache=True)
def _uv_iv(uv: np.uint32) -> int:
    return int(uv >> _UV_V_SHIFT)


@njit(cache=True)
def _normalize3(x: float, y: float, z: float):
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    inv = 1.0 / n
    return x * inv, y * inv, z * inv


@njit(cache=True)
def azel_to_dir(az_deg: float, el_deg: float):
    """
    az = 0 points +X
    az = +90 points +Y
    el = +90 points +Z
    """
    az = az_deg * (math.pi / 180.0)
    el = el_deg * (math.pi / 180.0)
    ce = math.cos(el)
    x = ce * math.cos(az)
    y = ce * math.sin(az)
    z = math.sin(el)
    return x, y, z


@njit(cache=True)
def dir_to_azel(x: float, y: float, z: float):
    x, y, z = _normalize3(x, y, z)
    az = math.degrees(math.atan2(y, x))  # [-180, 180]
    el = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return az, el


@njit(cache=True)
def dir_to_face_uv(x: float, y: float, z: float):
    ax = abs(x)
    ay = abs(y)
    az = abs(z)

    if ax >= ay and ax >= az:
        inv = 1.0 / ax
        if x >= 0.0:
            face = 0  # +X
            u = -z * inv
            v = y * inv
        else:
            face = 1  # -X
            u = z * inv
            v = y * inv
    elif ay >= ax and ay >= az:
        inv = 1.0 / ay
        if y >= 0.0:
            face = 2  # +Y
            u = x * inv
            v = -z * inv
        else:
            face = 3  # -Y
            u = x * inv
            v = z * inv
    else:
        inv = 1.0 / az
        if z >= 0.0:
            face = 4  # +Z
            u = x * inv
            v = y * inv
        else:
            face = 5  # -Z
            u = -x * inv
            v = y * inv

    return face, u, v


@njit(cache=True)
def face_uv_to_dir(face: int, u: float, v: float):
    # Inverse mapping consistent with dir_to_face_uv above
    if face == 0:  # +X
        x, y, z = 1.0, v, -u
    elif face == 1:  # -X
        x, y, z = -1.0, v, u
    elif face == 2:  # +Y
        x, y, z = u, 1.0, -v
    elif face == 3:  # -Y
        x, y, z = u, -1.0, v
    elif face == 4:  # +Z
        x, y, z = u, v, 1.0
    else:  # -Z
        x, y, z = -u, v, -1.0

    return _normalize3(x, y, z)


# ----------------------------
# Cap sampling (union of caps)
# ----------------------------


@njit(cache=True)
def _inside_any_cap(
    px: float, py: float, pz: float, cap_centers: np.ndarray, cap_cos: np.ndarray
) -> int:
    m = cap_cos.shape[0]
    for i in range(m):
        d = cap_centers[i, 0] * px + cap_centers[i, 1] * py + cap_centers[i, 2] * pz
        if d >= cap_cos[i]:
            return 1
    return 0


@njit(cache=True)
def _cell_center_and_cosr(face: int, u0: float, u1: float, v0: float, v1: float):
    # center
    uc = 0.5 * (u0 + u1)
    vc = 0.5 * (v0 + v1)
    px, py, pz = face_uv_to_dir(face, uc, vc)

    # corners
    p0x, p0y, p0z = face_uv_to_dir(face, u0, v0)
    p1x, p1y, p1z = face_uv_to_dir(face, u1, v0)
    p2x, p2y, p2z = face_uv_to_dir(face, u0, v1)
    p3x, p3y, p3z = face_uv_to_dir(face, u1, v1)

    d0 = px * p0x + py * p0y + pz * p0z
    d1 = px * p1x + py * p1y + pz * p1z
    d2 = px * p2x + py * p2y + pz * p2z
    d3 = px * p3x + py * p3y + pz * p3z

    cos_r = d0
    if d1 < cos_r:
        cos_r = d1
    if d2 < cos_r:
        cos_r = d2
    if d3 < cos_r:
        cos_r = d3

    if cos_r > 1.0:
        cos_r = 1.0
    elif cos_r < -1.0:
        cos_r = -1.0

    return px, py, pz, cos_r


@njit(cache=True)
def _classify_node_conservative_union_caps(
    face: int,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    cap_centers: np.ndarray,
    cap_cos: np.ndarray,
):
    """
    Conservative classification for UNION of spherical caps.

    Returns:
      kind:
        0 = proven outside union
        1 = proven inside union (cell fully inside at least one cap)
        2 = mixed / unknown
      center_val: union membership at center (sampling) used only when mixed & small
      cos_r: cos(cell angular radius)
    """
    px, py, pz, cos_r = _cell_center_and_cosr(face, u0, u1, v0, v1)
    center_val = _inside_any_cap(px, py, pz, cap_centers, cap_cos)

    sin_r = math.sqrt(max(0.0, 1.0 - cos_r * cos_r))

    proven_inside = 0
    proven_outside_all = 1

    m = cap_cos.shape[0]
    for i in range(m):
        dot_cp = (
            cap_centers[i, 0] * px + cap_centers[i, 1] * py + cap_centers[i, 2] * pz
        )

        cos_a = cap_cos[i]
        sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))

        # PROVEN INSIDE requires r <= alpha  <=> cos(r) >= cos(alpha)
        if cos_r >= cos_a:
            cos_a_minus_r = cos_a * cos_r + sin_a * sin_r
            if dot_cp >= cos_a_minus_r:
                proven_inside = 1
                break

        cos_a_plus_r = cos_a * cos_r - sin_a * sin_r
        if dot_cp > cos_a_plus_r:
            proven_outside_all = 0

    if proven_inside == 1:
        return 1, center_val, cos_r
    if proven_outside_all == 1:
        return 0, center_val, cos_r
    return 2, center_val, cos_r


# ----------------------------
# Quadtree build (NJIT) - packed UV indices
# ----------------------------


@njit(cache=True)
def _build_quadtree_union_caps_packed(
    cap_centers: np.ndarray,
    cap_cos: np.ndarray,
    cos_tol: float,
    max_depth: int,
):
    """
    Build a packed quadtree for UNION(caps).

    Returns:
      roots: (6,) int32
      first_child: (N,) int32   (-1 for leaf; else first of 4 contiguous children)
      uv: (N,) uint32           (iu low16, iv high16) at node depth
      meta: (N,) uint16         (face, depth, leaf state)
    """
    roots = np.empty(6, dtype=np.int32)

    first_child_l = List.empty_list(types.int32)
    uv_l = List.empty_list(types.uint32)
    meta_l = List.empty_list(types.uint16)

    stack_out = List.empty_list(types.int32)

    for face in range(6):
        root_idx = len(meta_l)
        roots[face] = root_idx

        first_child_l.append(np.int32(-1))
        uv_l.append(_uv_pack(0, 0))  # depth 0 => cell is whole face
        meta_l.append(_meta_make(face, 0, 0))

        stack_out.append(np.int32(root_idx))

        while len(stack_out) > 0:
            idx = stack_out.pop()

            meta = meta_l[idx]
            f = _meta_face(meta)
            depth = _meta_depth(meta)

            uvp = uv_l[idx]
            iu = _uv_iu(uvp)
            iv = _uv_iv(uvp)

            den = 1 << depth
            size = 2.0 / float(den)

            u0 = -1.0 + float(iu) * size
            v0 = -1.0 + float(iv) * size
            u1 = u0 + size
            v1 = v0 + size

            kind, center_val, cos_r = _classify_node_conservative_union_caps(
                f, u0, u1, v0, v1, cap_centers, cap_cos
            )

            if kind == 0:
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, 0)
                continue
            if kind == 1:
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, 1)
                continue

            # mixed / unknown
            if cos_r >= cos_tol or depth >= max_depth:
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, int(center_val))
                continue

            # subdivide
            meta_l[idx] = _meta_make(f, depth, 0)

            child0 = len(meta_l)
            first_child_l[idx] = np.int32(child0)
            nd = depth + 1

            # children order must match traversal:
            # 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
            for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
                first_child_l.append(np.int32(-1))
                uv_l.append(_uv_pack((iu << 1) + du, (iv << 1) + dv))
                meta_l.append(_meta_make(f, nd, 0))
                stack_out.append(np.int32(len(meta_l) - 1))

    n = len(meta_l)
    first_child = np.empty(n, dtype=np.int32)
    uv = np.empty(n, dtype=np.uint32)
    meta = np.empty(n, dtype=np.uint16)
    for i in range(n):
        first_child[i] = first_child_l[i]
        uv[i] = uv_l[i]
        meta[i] = meta_l[i]
    return roots, first_child, uv, meta


# ----------------------------
# Quadtree build from a dense face mask (NJIT)
# ----------------------------


@njit(cache=True)
def _rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    # ii shape: (H+1, W+1)
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


@njit(cache=True)
def _build_quadtree_from_integrals_packed(
    ii_faces: np.ndarray,  # (6, res+1, res+1) int32
    cos_tol: float,
    max_depth: int,
):
    """
    Build a packed quadtree from a painted cubemap raster mask, using integral images
    for fast conservative uniform tests.

    A node is:
      - leaf if the covered pixel block is all-0 or all-1
      - otherwise it subdivides while the cell is larger than tolerance and depth<max_depth

    Leaf value for mixed-but-stopped nodes is chosen by majority vote on the block.
    """
    roots = np.empty(6, dtype=np.int32)

    first_child_l = List.empty_list(types.int32)
    uv_l = List.empty_list(types.uint32)
    meta_l = List.empty_list(types.uint16)

    stack_i = List.empty_list(types.int32)

    res = ii_faces.shape[1] - 1  # face resolution of the painted mask

    for face in range(6):
        root_idx = len(meta_l)
        roots[face] = root_idx

        first_child_l.append(np.int32(-1))
        uv_l.append(_uv_pack(0, 0))
        meta_l.append(_meta_make(face, 0, 0))

        stack_i.append(np.int32(root_idx))

        while len(stack_i) > 0:
            idx = stack_i.pop()

            meta = meta_l[idx]
            f = _meta_face(meta)
            depth = _meta_depth(meta)

            uvp = uv_l[idx]
            iu = _uv_iu(uvp)
            iv = _uv_iv(uvp)

            den = 1 << depth
            # map node to pixel rectangle on the source mask (conservative)
            x0 = (iu * res) // den
            x1 = ((iu + 1) * res + den - 1) // den
            y0 = (iv * res) // den
            y1 = ((iv + 1) * res + den - 1) // den

            # clamp
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > res:
                x1 = res
            if y1 > res:
                y1 = res

            w = x1 - x0
            h = y1 - y0
            area = w * h
            if area <= 0:
                # should not happen, but keep safe
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, 0)
                continue

            s = _rect_sum(ii_faces[f], x0, y0, x1, y1)

            if s == 0:
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, 0)
                continue

            if s == area:
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, 1)
                continue

            # mixed
            if depth >= max_depth:
                # majority vote
                state = 1 if s * 2 >= area else 0
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, state)
                continue

            # angular tolerance stop test
            size_uv = 2.0 / float(den)
            u0 = -1.0 + float(iu) * size_uv
            v0 = -1.0 + float(iv) * size_uv
            u1 = u0 + size_uv
            v1 = v0 + size_uv
            _, _, _, cos_r = _cell_center_and_cosr(f, u0, u1, v0, v1)

            if cos_r >= cos_tol:
                # small enough; store majority value
                state = 1 if s * 2 >= area else 0
                first_child_l[idx] = np.int32(-1)
                meta_l[idx] = _meta_make(f, depth, state)
                continue

            # subdivide
            meta_l[idx] = _meta_make(f, depth, 0)

            child0 = len(meta_l)
            first_child_l[idx] = np.int32(child0)
            nd = depth + 1

            for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
                first_child_l.append(np.int32(-1))
                uv_l.append(_uv_pack((iu << 1) + du, (iv << 1) + dv))
                meta_l.append(_meta_make(f, nd, 0))
                stack_i.append(np.int32(len(meta_l) - 1))

    n = len(meta_l)
    first_child = np.empty(n, dtype=np.int32)
    uv = np.empty(n, dtype=np.uint32)
    meta = np.empty(n, dtype=np.uint16)
    for i in range(n):
        first_child[i] = first_child_l[i]
        uv[i] = uv_l[i]
        meta[i] = meta_l[i]
    return roots, first_child, uv, meta


# ----------------------------
# Quadtree boolean combine (NJIT)
# ----------------------------


@njit(cache=True)
def _op_apply(a: int, b: int, op: int) -> int:
    a1 = 1 if a != 0 else 0
    b1 = 1 if b != 0 else 0
    if op == _OP_OR:
        return 1 if (a1 == 1 or b1 == 1) else 0
    if op == _OP_AND:
        return 1 if (a1 == 1 and b1 == 1) else 0
    if op == _OP_DIFF:
        return 1 if (a1 == 1 and b1 == 0) else 0
    # _OP_XOR
    return 1 if (a1 != b1) else 0


@njit(cache=True)
def _combine_quadtrees_packed(
    roots_a: np.ndarray,
    first_child_a: np.ndarray,
    meta_a: np.ndarray,
    roots_b: np.ndarray,
    first_child_b: np.ndarray,
    meta_b: np.ndarray,
    op: int,
):
    """
    Combine two packed quadtrees with a boolean op, producing a new packed quadtree.

    The output refinement follows the union of input refinements, but can early-out
    to constant leaves when the operation becomes constant over a region.

    op:
      0 OR
      1 AND
      2 DIFF (A \\ B) == A & ~B
      3 XOR
    """
    roots = np.empty(6, dtype=np.int32)

    first_child_l = List.empty_list(types.int32)
    uv_l = List.empty_list(types.uint32)
    meta_l = List.empty_list(types.uint16)

    # stack entries: out_idx, idx_a, idx_b, depth, iu, iv
    stack_out = List.empty_list(types.int32)
    stack_a = List.empty_list(types.int32)
    stack_b = List.empty_list(types.int32)
    stack_depth = List.empty_list(types.int32)
    stack_iu = List.empty_list(types.int32)
    stack_iv = List.empty_list(types.int32)

    for face in range(6):
        out_root = len(meta_l)
        roots[face] = out_root

        first_child_l.append(np.int32(-1))
        uv_l.append(_uv_pack(0, 0))
        meta_l.append(_meta_make(face, 0, 0))

        stack_out.append(np.int32(out_root))
        stack_a.append(np.int32(roots_a[face]))
        stack_b.append(np.int32(roots_b[face]))
        stack_depth.append(np.int32(0))
        stack_iu.append(np.int32(0))
        stack_iv.append(np.int32(0))

        while len(stack_out) > 0:
            out_idx = stack_out.pop()
            idxa = stack_a.pop()
            idxb = stack_b.pop()
            depth = stack_depth.pop()
            iu = stack_iu.pop()
            iv = stack_iv.pop()

            fca = int(first_child_a[idxa])
            fcb = int(first_child_b[idxb])
            leafa = 1 if fca == -1 else 0
            leafb = 1 if fcb == -1 else 0

            statea = _meta_state(meta_a[idxa]) if leafa == 1 else -1  # type: ignore
            stateb = _meta_state(meta_b[idxb]) if leafb == 1 else -1  # type: ignore

            # constant short-circuits
            decided = 0
            out_state = 0

            if op == _OP_OR:
                if (leafa == 1 and statea == 1) or (leafb == 1 and stateb == 1):
                    decided = 1
                    out_state = 1
                elif leafa == 1 and leafb == 1:
                    decided = 1
                    out_state = 1 if (statea != 0 or stateb != 0) else 0
            elif op == _OP_AND:
                if (leafa == 1 and statea == 0) or (leafb == 1 and stateb == 0):
                    decided = 1
                    out_state = 0
                elif leafa == 1 and leafb == 1:
                    decided = 1
                    out_state = 1 if (statea != 0 and stateb != 0) else 0
            elif op == _OP_DIFF:
                # A & ~B
                if leafa == 1 and statea == 0:
                    decided = 1
                    out_state = 0
                elif leafb == 1 and stateb == 1:
                    decided = 1
                    out_state = 0
                elif leafa == 1 and leafb == 1:
                    decided = 1
                    out_state = 1 if (statea != 0 and stateb == 0) else 0
            else:  # XOR
                if leafa == 1 and leafb == 1:
                    decided = 1
                    out_state = 1 if (statea != stateb) else 0

            if decided == 1:
                first_child_l[out_idx] = np.int32(-1)
                uv_l[out_idx] = _uv_pack(iu, iv)
                meta_l[out_idx] = _meta_make(face, depth, out_state)
                continue

            # Need to descend. If a side is leaf, treat it as constant over children.
            first_child_l[out_idx] = np.int32(len(meta_l))
            uv_l[out_idx] = _uv_pack(iu, iv)
            meta_l[out_idx] = _meta_make(face, depth, 0)

            nd = depth + 1

            for c, (du, dv) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                ciu = (iu << 1) + du
                civ = (iv << 1) + dv

                # output child node placeholder
                first_child_l.append(np.int32(-1))
                uv_l.append(_uv_pack(ciu, civ))
                meta_l.append(_meta_make(face, nd, 0))
                out_child = len(meta_l) - 1

                # pick corresponding child in A/B if internal, else keep same leaf idx
                if leafa == 1:
                    idxa_child = idxa
                else:
                    idxa_child = fca + c

                if leafb == 1:
                    idxb_child = idxb
                else:
                    idxb_child = fcb + c

                stack_out.append(np.int32(out_child))
                stack_a.append(np.int32(idxa_child))
                stack_b.append(np.int32(idxb_child))
                stack_depth.append(np.int32(nd))
                stack_iu.append(np.int32(ciu))
                stack_iv.append(np.int32(civ))

    n = len(meta_l)
    first_child = np.empty(n, dtype=np.int32)
    uv = np.empty(n, dtype=np.uint32)
    meta = np.empty(n, dtype=np.uint16)
    for i in range(n):
        first_child[i] = first_child_l[i]
        uv[i] = uv_l[i]
        meta[i] = meta_l[i]
    return roots, first_child, uv, meta


# ----------------------------
# Quadtree lookup (NJIT)
# ----------------------------


@njit(cache=True)
def _contains_face_uv_packed(
    face: int,
    u: float,
    v: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    idx = int(roots[face])
    while True:
        fc = int(first_child[idx])
        if fc == -1:
            return _meta_state(meta[idx])

        m = meta[idx]
        d = _meta_depth(m)
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
def _leaf_depth_face_uv_packed(
    face: int,
    u: float,
    v: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    idx = int(roots[face])
    while True:
        fc = int(first_child[idx])
        m = meta[idx]
        d = _meta_depth(m)
        if fc == -1:
            return d

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
def _contains_dir_packed(
    x: float,
    y: float,
    z: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    face, u, v = dir_to_face_uv(x, y, z)
    return _contains_face_uv_packed(face, u, v, roots, first_child, uv, meta)


@njit(cache=True)
def _leaf_depth_dir_packed(
    x: float,
    y: float,
    z: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> int:
    face, u, v = dir_to_face_uv(x, y, z)
    return _leaf_depth_face_uv_packed(face, u, v, roots, first_child, uv, meta)


@njit(cache=True)
def _contains_dirs_packed(
    dirs: np.ndarray,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
) -> np.ndarray:
    n = dirs.shape[0]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = _contains_dir_packed(
            dirs[i, 0], dirs[i, 1], dirs[i, 2], roots, first_child, uv, meta
        )
    return out


@njit(cache=True)
def _contains_dirs_and_depth_packed(
    dirs: np.ndarray,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
):
    n = dirs.shape[0]
    out_mask = np.empty(n, dtype=np.uint8)
    out_depth = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x = dirs[i, 0]
        y = dirs[i, 1]
        z = dirs[i, 2]
        face, u, v = dir_to_face_uv(x, y, z)

        idx = int(roots[face])
        while True:
            fc = int(first_child[idx])
            m = meta[idx]
            d = _meta_depth(m)
            if fc == -1:
                out_mask[i] = _meta_state(m)
                out_depth[i] = np.uint8(d if d < 255 else 255)
                break

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

    return out_mask, out_depth


@njit(cache=True, parallel=True)
def _dense_faces_from_tree_centers(
    resolution: int,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
):
    """
    Render a dense (6,res,res) mask by evaluating the tree at pixel centers on each face.
    This is exact w.r.t the tree's own evaluation and avoids "leaf smaller than pixel"
    overwrites.
    """
    res = int(resolution)
    out = np.empty((6, res, res), dtype=np.uint8)

    inv = 2.0 / float(res)
    for face in prange(6):
        for y in range(res):
            v = -1.0 + (float(y) + 0.5) * inv
            for x in range(res):
                u = -1.0 + (float(x) + 0.5) * inv
                out[face, y, x] = _contains_face_uv_packed(
                    face, u, v, roots, first_child, uv, meta
                )
    return out


@njit(cache=True, parallel=True)
def _dense_faces_from_tree_centers_and_depth(
    resolution: int,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
):
    res = int(resolution)
    out_mask = np.empty((6, res, res), dtype=np.uint8)
    out_depth = np.empty((6, res, res), dtype=np.uint8)

    inv = 2.0 / float(res)
    for face in prange(6):
        for y in range(res):
            v = -1.0 + (float(y) + 0.5) * inv
            for x in range(res):
                u = -1.0 + (float(x) + 0.5) * inv

                idx = int(roots[face])
                while True:
                    fc = int(first_child[idx])
                    m = meta[idx]
                    d = _meta_depth(m)
                    if fc == -1:
                        out_mask[face, y, x] = _meta_state(m)
                        out_depth[face, y, x] = np.uint8(d if d < 255 else 255)
                        break

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

    return out_mask, out_depth


@njit(cache=True, parallel=True)
def _rasterize_azel_mask_and_depth(
    az_res: int,
    el_res: int,
    center_az_deg: float,
    roots: np.ndarray,
    first_child: np.ndarray,
    uv: np.ndarray,
    meta: np.ndarray,
):
    """
    Produce a single az/el raster mask (el_res, az_res) by sampling at pixel centers.

    x-axis spans [-180, 180] and is centered on center_az_deg.
    """
    AZ = int(az_res)
    EL = int(el_res)
    out_mask = np.empty((EL, AZ), dtype=np.uint8)
    out_depth = np.empty((EL, AZ), dtype=np.uint8)

    daz = 360.0 / float(AZ)
    delv = 180.0 / float(EL)

    for j in prange(EL):
        el = -90.0 + (float(j) + 0.5) * delv
        for i in range(AZ):
            az = -180.0 + (float(i) + 0.5) * daz + center_az_deg
            x, y, z = azel_to_dir(az, el)

            face, u, v = dir_to_face_uv(x, y, z)

            idx = int(roots[face])
            while True:
                fc = int(first_child[idx])
                m = meta[idx]
                d = _meta_depth(m)
                if fc == -1:
                    out_mask[j, i] = _meta_state(m)
                    out_depth[j, i] = np.uint8(d if d < 255 else 255)
                    break

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

    return out_mask, out_depth


# ----------------------------
# Raster import helpers (NJIT)
# ----------------------------


@njit(cache=True, parallel=True)
def _sample_azel_mask_to_faces_nn(
    mask_azel: np.ndarray,  # (EL, AZ) uint8/bool
    face_res: int,
    az_min_deg: float,
    az_max_deg: float,
    el_min_deg: float,
    el_max_deg: float,
):
    """
    Sample an equirectangular az/el mask onto cubemap faces (nearest-neighbor).

    az_min_deg/az_max_deg define the horizontal span; typical choices:
      [-180, 180] or [0, 360]
    el_min_deg/el_max_deg define the vertical span; typical [-90, 90]
    """
    src_el = mask_azel.shape[0]
    src_az = mask_azel.shape[1]

    out = np.empty((6, face_res, face_res), dtype=np.uint8)

    az_span = az_max_deg - az_min_deg
    el_span = el_max_deg - el_min_deg

    inv = 2.0 / float(face_res)
    for face in prange(6):
        for y in range(face_res):
            v = -1.0 + (float(y) + 0.5) * inv
            for x in range(face_res):
                u = -1.0 + (float(x) + 0.5) * inv
                dx, dy, dz = face_uv_to_dir(face, u, v)
                az, el = dir_to_azel(dx, dy, dz)

                # map az into [az_min, az_max)
                t = (az - az_min_deg) / az_span
                # wrap for az
                t = t - math.floor(t)
                ix = int(t * src_az)
                if ix < 0:
                    ix = 0
                elif ix >= src_az:
                    ix = src_az - 1

                # map el into [el_min, el_max]
                s = (el - el_min_deg) / el_span
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
                iy = int(s * src_el)
                if iy >= src_el:
                    iy = src_el - 1

                out[face, y, x] = 1 if mask_azel[iy, ix] != 0 else 0

    return out


@njit(cache=True)
def _point_in_poly_2d(x: float, y: float, poly: np.ndarray) -> int:
    """
    Ray casting point-in-polygon for 2D polygon poly[:,0]=x, poly[:,1]=y.
    Returns 1 if inside, 0 if outside.
    """
    inside = 0
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi = poly[i, 0]
        yi = poly[i, 1]
        xj = poly[j, 0]
        yj = poly[j, 1]

        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersect:
            inside = 1 - inside
        j = i
    return inside


@njit(cache=True, parallel=True)
def _rasterize_azel_polygon_mask(
    poly_azel: np.ndarray,  # (N,2) in degrees, az in [az_min,az_max) continuous
    az_res: int,
    el_res: int,
    az_min_deg: float,
    az_max_deg: float,
    el_min_deg: float,
    el_max_deg: float,
):
    """
    Rasterize a 2D polygon in az/el degrees onto an az/el grid (nearest coverage by center samples).

    This treats az/el as a flat equirectangular plane (not a true spherical polygon).
    """
    AZ = int(az_res)
    EL = int(el_res)
    out = np.zeros((EL, AZ), dtype=np.uint8)

    daz = (az_max_deg - az_min_deg) / float(AZ)
    delv = (el_max_deg - el_min_deg) / float(EL)

    for j in prange(EL):
        el = el_min_deg + (float(j) + 0.5) * delv
        for i in range(AZ):
            az = az_min_deg + (float(i) + 0.5) * daz
            out[j, i] = _point_in_poly_2d(az, el, poly_azel)

    return out


# ----------------------------
# Adaptive cube raster FOV
# ----------------------------


class AdaptiveCubeRasterFOV:
    """
    Adaptive cubemap quadtree raster for a boolean FOV mask.

    Two primary creation paths:
      1) Analytic union-of-caps via add_cap_* + compile()
      2) From painted rasters via from_faces_mask() / from_azel_mask()

    Compiled representation:
      roots:       (6,) int32
      first_child: (N,) int32
      uv:          (N,) uint32
      meta:        (N,) uint16
    """

    def __init__(
        self,
        tolerance_deg: float = 0.01,
        max_depth: int | None = None,
    ):
        if tolerance_deg <= 0:
            raise ValueError("tolerance_deg must be > 0")

        self.tolerance_deg = float(tolerance_deg)

        if max_depth is None:
            est = math.log2(90.0 / self.tolerance_deg)
            md = int(math.ceil(est))
            md = max(0, min(md, 30))
        else:
            md = int(max_depth)

        # Packed UV indices require depth <= 16 (iu,iv are uint16)
        if md > 16:
            raise ValueError(
                f"max_depth={md} exceeds packed UV limit (16). "
                "Increase tolerance_deg or set max_depth<=16."
            )

        self.max_depth = md

        # Analytic cap list (union)
        self._caps_centers: list[tuple[float, float, float]] = []
        self._caps_cos: list[float] = []

        self._compiled = False
        self.roots: np.ndarray | None = None
        self.first_child: np.ndarray | None = None
        self.uv: np.ndarray | None = None
        self.meta: np.ndarray | None = None

    # ---- construction / compilation ----

    def clear(self):
        self._caps_centers.clear()
        self._caps_cos.clear()
        self._compiled = False
        self.roots = None
        self.first_child = None
        self.uv = None
        self.meta = None

    def add_cap_azel(
        self, center_az_deg: float, center_el_deg: float, half_angle_deg: float
    ):
        """Add a spherical cap (union)."""
        if half_angle_deg <= 0:
            raise ValueError("half_angle_deg must be > 0")

        x, y, z = azel_to_dir(float(center_az_deg), float(center_el_deg))
        x, y, z = _normalize3(x, y, z)

        ha = float(half_angle_deg) * (math.pi / 180.0)
        self._caps_centers.append((x, y, z))
        self._caps_cos.append(math.cos(ha))
        self._compiled = False

    def add_cap_solid_angle_azel(
        self, center_az_deg: float, center_el_deg: float, omega_sr: float
    ):
        """Add a spherical cap by solid angle (steradians)."""
        if omega_sr <= 0:
            raise ValueError("omega_sr must be > 0")
        alpha = math.acos(1.0 - (omega_sr / (2.0 * math.pi)))
        self.add_cap_azel(center_az_deg, center_el_deg, alpha * (180.0 / math.pi))

    def compile(self):
        """Compile the current analytic cap-union into a packed quadtree."""
        if len(self._caps_centers) == 0:
            self._build_trivial_empty()
            self._compiled = True
            return

        cap_centers = np.asarray(self._caps_centers, dtype=np.float64)
        cap_cos = np.asarray(self._caps_cos, dtype=np.float64)

        tol_rad = self.tolerance_deg * (math.pi / 180.0)
        cos_tol = math.cos(tol_rad)

        roots, first_child, uv, meta = _build_quadtree_union_caps_packed(
            cap_centers, cap_cos, cos_tol, int(self.max_depth)
        )

        self.roots = roots
        self.first_child = first_child
        self.uv = uv
        self.meta = meta
        self._compiled = True

    def _build_trivial_empty(self):
        self.roots = np.arange(6, dtype=np.int32)
        self.first_child = np.full(6, -1, dtype=np.int32)
        self.uv = np.empty(6, dtype=np.uint32)
        self.meta = np.empty(6, dtype=np.uint16)
        for f in range(6):
            self.uv[f] = _uv_pack(0, 0)
            self.meta[f] = _meta_make(f, 0, 0)

    @property
    def compiled(self) -> bool:
        return self._compiled

    # ---- alternate constructors (painted rasters) ----

    @classmethod
    def from_faces_mask(
        cls,
        mask_faces: np.ndarray,
        *,
        tolerance_deg: float = 0.01,
        max_depth: int | None = None,
    ) -> "AdaptiveCubeRasterFOV":
        """
        Build a FOV from an arbitrary painted cubemap mask.

        mask_faces: (6, H, W) (bool/uint8). H and W must match (square faces).
        """
        mask_faces = np.asarray(mask_faces)
        if mask_faces.ndim != 3 or mask_faces.shape[0] != 6:
            raise ValueError("mask_faces must have shape (6, H, W)")
        if mask_faces.shape[1] != mask_faces.shape[2]:
            raise ValueError("mask_faces must be square per face (H==W)")

        res = int(mask_faces.shape[1])
        # integral images (6, res+1, res+1)
        mf = (mask_faces != 0).astype(np.int32)
        ii = np.zeros((6, res + 1, res + 1), dtype=np.int32)
        # numpy cumsum is fast
        ii[:, 1:, 1:] = mf.cumsum(axis=1).cumsum(axis=2)

        obj = cls(tolerance_deg=tolerance_deg, max_depth=max_depth)

        tol_rad = obj.tolerance_deg * (math.pi / 180.0)
        cos_tol = math.cos(tol_rad)

        roots, first_child, uv, meta = _build_quadtree_from_integrals_packed(
            ii, cos_tol, int(obj.max_depth)
        )

        obj.roots = roots
        obj.first_child = first_child
        obj.uv = uv
        obj.meta = meta
        obj._compiled = True
        return obj

    @classmethod
    def from_azel_mask(
        cls,
        mask_azel: np.ndarray,
        *,
        face_res: int = 512,
        az_min_deg: float = -180.0,
        az_max_deg: float = 180.0,
        el_min_deg: float = -90.0,
        el_max_deg: float = 90.0,
        tolerance_deg: float = 0.01,
        max_depth: int | None = None,
    ) -> "AdaptiveCubeRasterFOV":
        """
        Build a FOV from an arbitrary painted equirectangular az/el mask.

        mask_azel: (EL, AZ) (bool/uint8). Coordinates assumed:
          az spans [az_min_deg, az_max_deg)
          el spans [el_min_deg, el_max_deg]
        """
        mask_azel = np.asarray(mask_azel)
        if mask_azel.ndim != 2:
            raise ValueError("mask_azel must be a 2D array (EL, AZ)")

        m = (mask_azel != 0).astype(np.uint8)
        faces = _sample_azel_mask_to_faces_nn(
            m,
            int(face_res),
            float(az_min_deg),
            float(az_max_deg),
            float(el_min_deg),
            float(el_max_deg),
        )
        return cls.from_faces_mask(
            faces, tolerance_deg=tolerance_deg, max_depth=max_depth
        )

    @classmethod
    def from_azel_polygon(
        cls,
        vertices_az_el_deg: Sequence[Sequence[float]],
        *,
        az_res: int = 720,
        el_res: int = 360,
        face_res: int = 512,
        az_min_deg: float = -180.0,
        az_max_deg: float = 180.0,
        el_min_deg: float = -90.0,
        el_max_deg: float = 90.0,
        tolerance_deg: float = 0.01,
        max_depth: int | None = None,
    ) -> "AdaptiveCubeRasterFOV":
        """
        Convenience: rasterize a 2D az/el polygon into an az/el mask, then build a FOV.

        This polygon fill is done in the equirectangular az/el plane (not a true spherical polygon).
        """
        poly = np.asarray(vertices_az_el_deg, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ValueError("vertices_az_el_deg must be (N,2) with N>=3")

        mask_azel = _rasterize_azel_polygon_mask(
            poly,
            int(az_res),
            int(el_res),
            float(az_min_deg),
            float(az_max_deg),
            float(el_min_deg),
            float(el_max_deg),
        )
        return cls.from_azel_mask(
            mask_azel,
            face_res=face_res,
            az_min_deg=az_min_deg,
            az_max_deg=az_max_deg,
            el_min_deg=el_min_deg,
            el_max_deg=el_max_deg,
            tolerance_deg=tolerance_deg,
            max_depth=max_depth,
        )

    # ---- boolean ops between FOVs ----

    def _ensure_compiled(self):
        if not self._compiled:
            self.compile()

    def _binary_op(
        self, other: "AdaptiveCubeRasterFOV", op_code: int
    ) -> "AdaptiveCubeRasterFOV":
        if not isinstance(other, AdaptiveCubeRasterFOV):
            raise TypeError("other must be an AdaptiveCubeRasterFOV")
        self._ensure_compiled()
        other._ensure_compiled()

        roots, first_child, uv, meta = _combine_quadtrees_packed(
            self.roots,  # type: ignore
            self.first_child,  # type: ignore
            self.meta,  # type: ignore
            other.roots,  # type: ignore
            other.first_child,  # type: ignore
            other.meta,  # type: ignore
            int(op_code),
        )

        out = AdaptiveCubeRasterFOV(
            tolerance_deg=min(self.tolerance_deg, other.tolerance_deg),
            max_depth=max(self.max_depth, other.max_depth),
        )
        out.roots = roots
        out.first_child = first_child
        out.uv = uv
        out.meta = meta
        out._compiled = True
        return out

    def union(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self._binary_op(other, _OP_OR)  # type: ignore

    def intersection(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self._binary_op(other, _OP_AND)  # type: ignore

    def difference(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self._binary_op(other, _OP_DIFF)  # type: ignore

    def xor(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self._binary_op(other, _OP_XOR)  # type: ignore

    def invert(self) -> "AdaptiveCubeRasterFOV":
        """Return NOT(self) as a new FOV (same structure, flipped leaf states)."""
        self._ensure_compiled()
        out = AdaptiveCubeRasterFOV(
            tolerance_deg=self.tolerance_deg, max_depth=self.max_depth
        )
        out.roots = self.roots.copy()  # type: ignore
        out.first_child = self.first_child.copy()  # type: ignore
        out.uv = self.uv.copy()  # type: ignore
        out.meta = self.meta.copy()  # type: ignore
        # Flip state bit (bit0). Internal nodes' state is ignored.
        out.meta ^= np.uint16(1)
        out._compiled = True
        return out

    # Python operator sugar
    def __or__(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self.union(other)

    def __and__(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self.intersection(other)

    def __sub__(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self.difference(other)

    def __add__(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self.union(other)

    def __xor__(self, other: "AdaptiveCubeRasterFOV") -> "AdaptiveCubeRasterFOV":
        return self.xor(other)

    def __invert__(self) -> "AdaptiveCubeRasterFOV":
        return self.invert()

    # ---- queries ----

    def node_count(self) -> int:
        return 0 if self.meta is None else int(self.meta.size)

    def memory_bytes(self, include_python_overhead: bool = False) -> int:
        import sys

        total = 0

        def add(arr):
            nonlocal total
            if arr is None:
                return
            total += int(arr.nbytes)
            if include_python_overhead:
                total += sys.getsizeof(arr)

        add(self.roots)
        add(self.first_child)
        add(self.uv)
        add(self.meta)

        if include_python_overhead:
            total += sys.getsizeof(self)

        return total

    def contains_dir(self, x: float, y: float, z: float) -> bool:
        self._ensure_compiled()
        x, y, z = _normalize3(float(x), float(y), float(z))
        v = _contains_dir_packed(
            x, y, z, self.roots, self.first_child, self.uv, self.meta  # type: ignore
        )
        return bool(v)

    def contains_azel(self, az_deg: float, el_deg: float) -> bool:
        x, y, z = azel_to_dir(float(az_deg), float(el_deg))
        return self.contains_dir(x, y, z)

    def contains_dirs(self, dirs: np.ndarray) -> np.ndarray:
        self._ensure_compiled()
        dirs = np.asarray(dirs, dtype=np.float64)
        nrm = np.linalg.norm(dirs, axis=1)
        nrm[nrm == 0] = 1.0
        d = dirs / nrm[:, None]
        out_u8 = _contains_dirs_packed(
            d, self.roots, self.first_child, self.uv, self.meta  # type: ignore
        )
        return out_u8.astype(bool)

    def leaf_depth_dirs(self, dirs: np.ndarray) -> np.ndarray:
        """Return leaf depth for each direction (same shape as dirs[:,0])."""
        self._ensure_compiled()
        dirs = np.asarray(dirs, dtype=np.float64)
        nrm = np.linalg.norm(dirs, axis=1)
        nrm[nrm == 0] = 1.0
        d = dirs / nrm[:, None]
        out_mask, out_depth = _contains_dirs_and_depth_packed(
            d, self.roots, self.first_child, self.uv, self.meta  # type: ignore
        )
        return out_depth.astype(np.int32)

    # ---- visualization / export ----

    def to_dense_faces(self, resolution: int = 512) -> np.ndarray:
        """Dense faces mask by evaluating pixel centers (exact wrt tree)."""
        self._ensure_compiled()
        return _dense_faces_from_tree_centers(
            int(resolution), self.roots, self.first_child, self.uv, self.meta  # type: ignore
        )

    def to_dense_faces_and_depth(self, resolution: int = 512):
        """Dense faces (mask, depth) by evaluating pixel centers (exact wrt tree)."""
        self._ensure_compiled()
        return _dense_faces_from_tree_centers_and_depth(
            int(resolution), self.roots, self.first_child, self.uv, self.meta  # type: ignore
        )

    def to_dense_faces_aa(
        self, resolution: int = 512, supersample: int = 4
    ) -> np.ndarray:
        """
        Anti-aliased dense faces mask by supersampling and box filtering.
        Returns float32 in [0,1].
        """
        if supersample <= 1:
            return self.to_dense_faces(resolution=resolution).astype(np.float32)

        res = int(resolution)
        ss = int(supersample)
        hi = self.to_dense_faces(resolution=res * ss).astype(np.float32)
        hi = hi.reshape(6, res, ss, res, ss).mean(axis=(2, 4))
        return hi

    def plot_azel_with_detail(
        self,
        az_res: int = 720,
        el_res: int = 360,
        *,
        center_az_deg: float = 0.0,
        show_detail: bool = True,
        detail_mode: Literal["depth", "radius"] = "depth",
    ):
        """
        Plot a single az/el raster of the FOV mask, with an optional second panel showing
        adaptive detail.

        The x-axis is labeled [-180, 180] and centered on center_az_deg.

        detail_mode:
          - "depth": leaf depth (higher = more subdivision)
          - "radius": approximate angular radius (deg) derived from depth only (rough)
        """
        self._ensure_compiled()

        az_res = int(az_res)
        el_res = int(el_res)

        mask_u8, depth_u8 = _rasterize_azel_mask_and_depth(
            az_res,
            el_res,
            float(center_az_deg),
            self.roots,  # type: ignore
            self.first_child,  # type: ignore
            self.uv,  # type: ignore
            self.meta,  # type: ignore
        )

        if not show_detail:
            fig, ax0 = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
        else:
            fig, (ax0, ax1) = plt.subplots(
                2,
                1,
                figsize=(14, 6),
                constrained_layout=True,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

        extent = [-180.0, 180.0, -90.0, 90.0]
        ax0.imshow(
            mask_u8,
            origin="lower",
            interpolation="nearest",
            extent=extent,  # type: ignore
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        ax0.set_ylabel("Elevation (deg)")
        ax0.set_title(f"FOV az/el raster {az_res}×{el_res}")

        if show_detail:
            if detail_mode == "depth":
                detail = depth_u8.astype(np.float32)
                im = ax1.imshow(
                    detail,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    aspect="auto",
                )
                ax1.set_title("Adaptive detail: leaf depth (higher = more subdivision)")
                cbar = fig.colorbar(
                    im, ax=ax1, orientation="horizontal", pad=0.25, fraction=0.25
                )
                cbar.set_label("depth")
            else:
                # Rough depth->radius mapping: size in uv ~ 2/2^d; half-diagonal ~ sqrt(2)*size/2
                d = depth_u8.astype(np.float32)
                size_uv = 2.0 / (2.0**d)
                half_diag_uv = 0.5 * math.sqrt(2.0) * size_uv
                radius_deg = (half_diag_uv * (180.0 / math.pi)).astype(np.float32)
                im = ax1.imshow(
                    radius_deg,
                    origin="lower",
                    interpolation="nearest",
                    extent=extent,
                    aspect="auto",
                )
                ax1.set_title("Adaptive detail: approx leaf angular radius (deg)")
                cbar = fig.colorbar(
                    im, ax=ax1, orientation="horizontal", pad=0.25, fraction=0.25
                )
                cbar.set_label("deg")

            ax1.set_ylabel("Elevation (deg)")

        ax0.set_xlim(-180, 180)
        ax0.set_ylim(-90, 90)
        ax0.set_yticks(np.arange(-90, 91, 15))
        ax0.set_xticks(np.arange(-180, 181, 30))
        ax0.set_xlabel("Azimuth (deg)")

        if show_detail:
            ax1.set_ylim(-90, 90)
            ax1.set_yticks(np.arange(-90, 91, 30))

        return fig

    def plot_faces(
        self,
        resolution: int = 512,
        *,
        show_depth: bool = True,
        show_quadtree: bool = True,
        quadtree_min_depth: int | None = None,
        aa_supersample: int = 2,
    ):
        """
        Plot the 6 cubemap faces, with optional leaf depth and optional quadtree overlay.

        The quadtree overlay is based on leaf rectangles (visual only).
        """
        self._ensure_compiled()

        res = int(resolution)
        mask_vis = self.to_dense_faces_aa(resolution=res, supersample=aa_supersample)
        _, dmap = (
            self.to_dense_faces_and_depth(resolution=res)
            if show_depth
            else (None, None)
        )

        nrows = 4 if show_depth else 2
        fig, axes = plt.subplots(
            nrows, 3, figsize=(12, 4 * (nrows / 2)), constrained_layout=True
        )
        axes = np.asarray(axes).reshape(nrows, 3)

        if quadtree_min_depth is None:
            quadtree_min_depth = max(0, int(self.max_depth) - 3)

        def _overlay_leaf_rects(ax, face: int):
            if not show_quadtree:
                return
            for i in range(self.meta.shape[0]):  # type: ignore
                if self.first_child[i] != -1:  # type: ignore
                    continue
                if _meta_face(self.meta[i]) != face:  # type: ignore
                    continue
                depth = _meta_depth(self.meta[i])  # type: ignore
                if depth < quadtree_min_depth:
                    continue

                uvp = self.uv[i]  # type: ignore
                iu = _uv_iu(uvp)
                iv = _uv_iv(uvp)

                den = 1 << depth
                x0 = (iu * res) // den
                x1 = ((iu + 1) * res + den - 1) // den
                y0 = (iv * res) // den
                y1 = ((iv + 1) * res + den - 1) // den

                if x1 <= x0 or y1 <= y0:
                    continue

                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        linewidth=0.35,
                        alpha=0.6,
                    )
                )

        for face in range(6):
            r = 0 if face < 3 else 1
            c = face % 3
            ax = axes[r, c]
            ax.imshow(
                mask_vis[face],
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_title(
                f"{_FACE_NAMES[face]}  mask (AA x{aa_supersample})"
                if aa_supersample > 1
                else f"{_FACE_NAMES[face]}  mask"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            _overlay_leaf_rects(ax, face)

        if show_depth:
            for face in range(6):
                r = 2 + (0 if face < 3 else 1)
                c = face % 3
                ax = axes[r, c]
                ax.imshow(dmap[face], origin="lower", interpolation="nearest")  # type: ignore
                ax.set_title(f"{_FACE_NAMES[face]}  leaf depth")
                ax.set_xticks([])
                ax.set_yticks([])

        return fig
