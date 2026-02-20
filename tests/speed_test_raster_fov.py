import numpy as np

from nebula.utils.raster_fov import AdaptiveCubeRasterFOV


def speed_test(
    *,
    seed: int = 0,
    number: int = 10,
    repeats: int = 7,
    n_points: int = 200_000,
    face_res: int = 256,
    az_res: int = 720,
    el_res: int = 360,
):
    """
    Speed tests for AdaptiveCubeRasterFOV without plotting.
    """
    import timeit

    rng = np.random.default_rng(seed)

    def bench(stmt, glb, *, label, number=number, repeats=repeats):
        t = timeit.Timer(stmt, globals=glb)
        samples = np.array(
            t.repeat(repeat=repeats, number=number), dtype=float
        ) / float(number)
        return {
            "label": label,
            "best_s": float(samples.min()),
            "mean_s": float(samples.mean()),
            "std_s": float(samples.std(ddof=1)) if repeats > 1 else 0.0,
        }

    def print_results(results):
        name_w = max(len(r["label"]) for r in results)
        print(f"{'test':<{name_w}}  best (ms)   mean (ms)   std (ms)")
        print("-" * (name_w + 33))
        for r in results:
            print(
                f"{r['label']:<{name_w}}  "
                f"{1e3*r['best_s']:>9.3f}  {1e3*r['mean_s']:>9.3f}  {1e3*r['std_s']:>8.3f}"
            )

    a = AdaptiveCubeRasterFOV(tolerance_deg=0.05)
    a.add_cap_azel(0.0, 0.0, 55.0)
    a.add_cap_azel(40.0, 0.0, 35.0)
    a.add_cap_azel(-30.0, 10.0, 25.0)
    a.compile()

    b = AdaptiveCubeRasterFOV(tolerance_deg=0.05)
    b.add_cap_azel(20.0, -5.0, 40.0)
    b.add_cap_azel(-70.0, 15.0, 30.0)
    b.compile()

    v = rng.normal(size=(n_points, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    az = rng.uniform(-180.0, 180.0, size=(n_points,))
    el = rng.uniform(-90.0, 90.0, size=(n_points,))

    faces_mask = (rng.random((6, face_res, face_res)) > 0.70).astype(np.uint8)
    azel_mask = (rng.random((el_res, az_res)) > 0.75).astype(np.uint8)
    poly = [(-60, -30), (60, -30), (60, 30), (-60, 30)]

    _ = (a | b).node_count()
    _ = (a & b).node_count()
    _ = (a - b).node_count()
    _ = (a ^ b).node_count()
    _ = (~a).node_count()

    _ = a.contains_azel(float(az[0]), float(el[0]))
    _ = a.contains_dir(float(v[0, 0]), float(v[0, 1]), float(v[0, 2]))
    _ = a.contains_dirs(v[:1000])
    _ = a.leaf_depth_dirs(v[:1000])

    _ = a.to_dense_faces(64)
    _ = a.to_dense_faces_and_depth(64)
    _ = a.to_dense_faces_aa(64, supersample=2)

    _ = AdaptiveCubeRasterFOV.from_faces_mask(faces_mask, tolerance_deg=0.05)
    _ = AdaptiveCubeRasterFOV.from_azel_mask(
        azel_mask, face_res=face_res, tolerance_deg=0.05
    )
    _ = AdaptiveCubeRasterFOV.from_azel_polygon(
        poly,
        az_res=az_res,
        el_res=el_res,
        face_res=face_res,
        tolerance_deg=0.05,
    )

    results = []

    glb = {"AdaptiveCubeRasterFOV": AdaptiveCubeRasterFOV}
    results.append(
        bench("AdaptiveCubeRasterFOV(tolerance_deg=0.05)", glb, label="create()")
    )
    results.append(
        bench(
            "f=AdaptiveCubeRasterFOV(tolerance_deg=0.05);"
            "f.add_cap_azel(0,0,55);f.add_cap_azel(40,0,35);f.add_cap_azel(-30,10,25)",
            glb,
            label="add_cap_azel x3",
        )
    )
    results.append(
        bench(
            "f=AdaptiveCubeRasterFOV(tolerance_deg=0.05);"
            "f.add_cap_azel(0,0,55);f.add_cap_azel(40,0,35);f.add_cap_azel(-30,10,25);"
            "f.compile()",
            glb,
            label="create+caps+compile",
        )
    )

    glb = {"a": a}
    results.append(bench("a.compile()", glb, label="compile() (recompile same caps)"))

    glb = {"a": a, "b": b}
    results.append(bench("a.union(b)", glb, label="union()"))
    results.append(bench("a.intersection(b)", glb, label="intersection()"))
    results.append(bench("a.difference(b)", glb, label="difference()"))
    results.append(bench("a.xor(b)", glb, label="xor()"))
    results.append(bench("a.invert()", glb, label="invert()"))
    results.append(bench("a | b", glb, label="operator: a | b"))
    results.append(bench("a & b", glb, label="operator: a & b"))
    results.append(bench("a - b", glb, label="operator: a - b"))
    results.append(bench("a ^ b", glb, label="operator: a ^ b"))
    results.append(bench("~a", glb, label="operator: ~a"))

    az0, el0 = float(az[0]), float(el[0])
    x0, y0, z0 = map(float, v[0])
    glb = {"a": a, "az0": az0, "el0": el0}
    results.append(
        bench("a.contains_azel(az0, el0)", glb, label="contains_azel (single)")
    )
    glb = {"a": a, "x0": x0, "y0": y0, "z0": z0}
    results.append(
        bench("a.contains_dir(x0, y0, z0)", glb, label="contains_dir (single)")
    )

    glb = {"a": a, "v": v}
    results.append(
        bench(
            "a.contains_dirs(v)", glb, label=f"contains_dirs (N={n_points})", number=1
        )
    )
    results.append(
        bench(
            "a.leaf_depth_dirs(v)",
            glb,
            label=f"leaf_depth_dirs (N={n_points})",
            number=1,
        )
    )

    glb = {"a": a}
    results.append(
        bench("a.to_dense_faces(256)", glb, label="to_dense_faces(256)", number=1)
    )
    results.append(
        bench(
            "a.to_dense_faces_and_depth(256)",
            glb,
            label="to_dense_faces_and_depth(256)",
            number=1,
        )
    )
    results.append(
        bench(
            "a.to_dense_faces_aa(256, supersample=2)",
            glb,
            label="to_dense_faces_aa(256,2)",
            number=1,
        )
    )

    glb = {"AdaptiveCubeRasterFOV": AdaptiveCubeRasterFOV, "faces_mask": faces_mask}
    results.append(
        bench(
            "AdaptiveCubeRasterFOV.from_faces_mask(faces_mask, tolerance_deg=0.05)",
            glb,
            label=f"from_faces_mask(face_res={face_res})",
            number=1,
        )
    )

    glb = {
        "AdaptiveCubeRasterFOV": AdaptiveCubeRasterFOV,
        "azel_mask": azel_mask,
        "face_res": face_res,
    }
    results.append(
        bench(
            "AdaptiveCubeRasterFOV.from_azel_mask(azel_mask, face_res=face_res, tolerance_deg=0.05)",
            glb,
            label=f"from_azel_mask(ELxAZ={el_res}x{az_res}, face_res={face_res})",
            number=1,
        )
    )

    glb = {
        "AdaptiveCubeRasterFOV": AdaptiveCubeRasterFOV,
        "poly": poly,
        "az_res": az_res,
        "el_res": el_res,
        "face_res": face_res,
    }
    results.append(
        bench(
            "AdaptiveCubeRasterFOV.from_azel_polygon(poly, az_res=az_res, el_res=el_res, face_res=face_res, tolerance_deg=0.05)",
            glb,
            label=f"from_azel_polygon(ELxAZ={el_res}x{az_res}, face_res={face_res})",
            number=1,
        )
    )

    print_results(results)
    return results


if __name__ == "__main__":
    speed_test()
