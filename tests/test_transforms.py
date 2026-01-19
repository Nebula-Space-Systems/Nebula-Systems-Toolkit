import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nebula import ensure_setup

ensure_setup()

import os
import timeit
import unittest

import astropy.units as u  # type: ignore
import numpy as np
from astropy.coordinates import ITRS  # type: ignore
from astropy.coordinates import TEME  # type: ignore
from astropy.coordinates import GCRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time as AstropyTime  # type: ignore
from astropy.utils import iers  # type: ignore
from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
from org.orekit.bodies import GeodeticPoint  # type: ignore
from org.orekit.bodies import OneAxisEllipsoid  # type: ignore
from org.orekit.frames import FramesFactory  # type: ignore
from org.orekit.time import AbsoluteDate  # type: ignore
from org.orekit.time import TimeScalesFactory  # type: ignore
from org.orekit.utils import Constants  # type: ignore
from org.orekit.utils import IERSConventions  # type: ignore
from org.orekit.utils import PVCoordinates  # type: ignore

from nebula.transform import *


class TransformTests(unittest.TestCase):

    # -----------------------------
    # Tests / benchmarks
    # -----------------------------
    def test_general(self):
        utc = TimeScalesFactory.getUTC()
        date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
        date1 = date0.shiftedBy(123.456)
        date2 = date0.shiftedBy(9876.0)

        def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        gcrf = FramesFactory.getGCRF()
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

        N = 1000
        rng = np.random.default_rng(0)
        r_gcrf = rng.normal(size=(N, 3)).astype(np.float64)
        r_gcrf *= 7_000_000.0 / np.linalg.norm(r_gcrf, axis=1, keepdims=True)

        # Single date
        r_itrf = gcrf_to_itrf_pos(r_gcrf, date0)
        r_back = itrf_to_gcrf_pos(r_itrf, date0)
        err = max_abs_err(r_gcrf, r_back)
        print(f"[Test 1] Position roundtrip (single date) max abs err: {err:.6e} m")
        assert err < 1e-6

        # Validate vs Orekit direct
        t = gcrf.getTransformTo(itrf, date0)
        idx = rng.integers(0, N, size=10)
        for i in idx:
            v = Vector3D(r_gcrf[i, 0], r_gcrf[i, 1], r_gcrf[i, 2])
            w = t.transformPosition(v)
            ref = np.array([w.getX(), w.getY(), w.getZ()], dtype=np.float64)
            e = float(np.max(np.abs(ref - r_itrf[i])))
            assert e < 1e-8
        print("[Test 1b] Position vs Orekit transformPosition (sample) OK")

        # PV single date
        v_gcrf = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0
        r_itrf2, v_itrf2 = gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, date0)
        r_back2, v_back2 = itrf_to_gcrf_pos_vel(r_itrf2, v_itrf2, date0)
        err_r = max_abs_err(r_gcrf, r_back2)
        err_v = max_abs_err(v_gcrf, v_back2)
        print(
            f"[Test 2] PV roundtrip (single date) max abs err r: {err_r:.6e} m, v: {err_v:.6e} m/s"
        )
        assert err_r < 1e-6
        assert err_v < 1e-9

        t = gcrf.getTransformTo(itrf, date0)
        for i in idx:
            pv = PVCoordinates(
                Vector3D(r_gcrf[i, 0], r_gcrf[i, 1], r_gcrf[i, 2]),
                Vector3D(v_gcrf[i, 0], v_gcrf[i, 1], v_gcrf[i, 2]),
            )
            pv2 = t.transformPVCoordinates(pv)
            ref_r = np.array(
                [
                    pv2.getPosition().getX(),
                    pv2.getPosition().getY(),
                    pv2.getPosition().getZ(),
                ],
                dtype=np.float64,
            )
            ref_v = np.array(
                [
                    pv2.getVelocity().getX(),
                    pv2.getVelocity().getY(),
                    pv2.getVelocity().getZ(),
                ],
                dtype=np.float64,
            )
            assert np.max(np.abs(ref_r - r_itrf2[i])) < 1e-8
            assert np.max(np.abs(ref_v - v_itrf2[i])) < 1e-8
        print("[Test 2b] PV vs Orekit transformPVCoordinates (sample) OK")

        # Per-row dates with repetition (identity-groupable)
        dates = [date0, date1, date2] * (N // 3) + [date0] * (N % 3)
        dates = dates[:N]
        r_itrf3 = gcrf_to_itrf_pos(r_gcrf, dates)
        r_back3 = itrf_to_gcrf_pos(r_itrf3, dates)
        err3 = max_abs_err(r_gcrf, r_back3)
        print(f"[Test 3] Position roundtrip (per-row dates) max abs err: {err3:.6e} m")
        assert err3 < 1e-6

        r_itrf4, v_itrf4 = gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, dates)
        r_back4, v_back4 = itrf_to_gcrf_pos_vel(r_itrf4, v_itrf4, dates)
        err4r = max_abs_err(r_gcrf, r_back4)
        err4v = max_abs_err(v_gcrf, v_back4)
        print(
            f"[Test 4] PV roundtrip (per-row dates) max abs err r: {err4r:.6e} m, v: {err4v:.6e} m/s"
        )
        assert err4r < 1e-6
        assert err4v < 1e-9

        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        r_itrf_a = gcrf_to_itrf_pos(r_gcrf, t0)
        r_back_a = itrf_to_gcrf_pos(r_itrf_a, t0)
        erra = max_abs_err(r_gcrf, r_back_a)
        print(
            f"[Test 5] Astropy scalar time position roundtrip max abs err: {erra:.6e} m"
        )
        assert erra < 1e-6

        t_arr = AstropyTime(
            [
                "2026-01-16T12:00:00",
                "2026-01-16T12:02:03.456",
                "2026-01-16T14:44:36.0",
            ],
            scale="utc",
        )
        # Repeat to length N (array-valued astropy Time)
        t_arrN = AstropyTime(np.resize(t_arr.utc.isot, N), scale="utc")

        r_itrf_b = gcrf_to_itrf_pos(r_gcrf, t_arrN)
        r_back_b = itrf_to_gcrf_pos(r_itrf_b, t_arrN)
        errb = max_abs_err(r_gcrf, r_back_b)
        print(
            f"[Test 5b] Astropy per-row time position roundtrip max abs err: {errb:.6e} m"
        )
        assert errb < 1e-6

        r_itrf_c, v_itrf_c = gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, t0)
        r_back_c, v_back_c = itrf_to_gcrf_pos_vel(r_itrf_c, v_itrf_c, t0)
        errc_r = max_abs_err(r_gcrf, r_back_c)
        errc_v = max_abs_err(v_gcrf, v_back_c)
        print(
            f"[Test 5c] Astropy scalar time PV roundtrip max abs err r: {errc_r:.6e} m, v: {errc_v:.6e} m/s"
        )
        assert errc_r < 1e-6
        assert errc_v < 1e-9

        print("All tests passed.")

    def speed_test(self):
        """
        Fair speed comparison between:
        - Orekit (this library): gcrf_to_itrf_pos / gcrf_to_itrf_pos_vel
        - Astropy: GCRS -> ITRS positions, and (optionally) velocities via CartesianDifferential

        Fairness rules applied:
        1) Exactly ONE frame transform call per benchmark iteration for both libraries.
        2) Same N, same input arrays.
        3) For multi-time cases, both libraries are grouped by the same K unique times and transform each group once.
        4) Output extraction is included for both (materialize Nx3 numpy arrays).
        5) Astropy single-date benchmark is fixed (previous version called transform_to() 3x).

        Notes:
        - Orekit uses GCRF<->ITRF (IERS, EOP handling depends on Orekit config).
        - Astropy uses GCRS<->ITRS. These are not strictly identical frames, but this is the closest practical match.
        - If you want Astropy to use IERS-A tables, ensure astropy-iers-data is installed and IERS is configured.
        """

        # Report thread env that can affect NumPy matmul performance
        thread_env = {
            k: os.environ.get(k)
            for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        }
        print(f"Thread env: {thread_env}")

        utc = TimeScalesFactory.getUTC()
        date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

        rng = np.random.default_rng(0)

        def make_states(N: int):
            r = rng.normal(size=(N, 3)).astype(np.float64)
            r *= 7_000_000.0 / np.linalg.norm(r, axis=1, keepdims=True)
            v = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0
            return r, v

        def bench_min(fn, repeat=5, number=1) -> float:
            return min(timeit.repeat(fn, repeat=repeat, number=number))

        REPEATS = 3
        NUMBER = 1

        # ---------------------------------------------------------------------
        # Helpers: Astropy transforms (single-date + grouped multi-date)
        # ---------------------------------------------------------------------
        def _astropy_pos_single(r_gcrf: np.ndarray, t_ast):
            rep = CartesianRepresentation(
                r_gcrf[:, 0] * u.m, r_gcrf[:, 1] * u.m, r_gcrf[:, 2] * u.m
            )
            g = GCRS(rep, obstime=t_ast)
            itrs = g.transform_to(ITRS(obstime=t_ast))  # ONE transform_to call
            c = itrs.cartesian
            return np.stack(
                [c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], axis=1
            )

        def _astropy_pos_vel_single(r_gcrf: np.ndarray, v_gcrf: np.ndarray, t_ast):
            rep = CartesianRepresentation(
                r_gcrf[:, 0] * u.m,
                r_gcrf[:, 1] * u.m,
                r_gcrf[:, 2] * u.m,
                differentials=CartesianDifferential(
                    v_gcrf[:, 0] * (u.m / u.s),
                    v_gcrf[:, 1] * (u.m / u.s),
                    v_gcrf[:, 2] * (u.m / u.s),
                ),
            )
            g = GCRS(rep, obstime=t_ast)
            itrs = g.transform_to(ITRS(obstime=t_ast))  # ONE transform_to call
            c = itrs.cartesian
            r = np.stack(
                [c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], axis=1
            )
            # velocity in astropy is in the differential
            d = c.differentials["s"]
            v = np.stack(
                [
                    d.d_x.to_value(u.m / u.s),
                    d.d_y.to_value(u.m / u.s),
                    d.d_z.to_value(u.m / u.s),
                ],
                axis=1,
            )
            return r, v

        def _astropy_pos_grouped(r_gcrf: np.ndarray, times_k, date_idx: np.ndarray):
            # group indices once
            K = len(times_k)
            groups = [np.where(date_idx == i)[0] for i in range(K)]
            out = np.empty_like(r_gcrf)
            for i in range(K):
                g = groups[i]
                rep = CartesianRepresentation(
                    r_gcrf[g, 0] * u.m, r_gcrf[g, 1] * u.m, r_gcrf[g, 2] * u.m
                )
                gcrs = GCRS(rep, obstime=times_k[i])
                itrs = gcrs.transform_to(
                    ITRS(obstime=times_k[i])
                )  # ONE transform_to call per group
                c = itrs.cartesian
                out[g, 0] = c.x.to_value(u.m)
                out[g, 1] = c.y.to_value(u.m)
                out[g, 2] = c.z.to_value(u.m)
            return out

        def _astropy_pos_vel_grouped(
            r_gcrf: np.ndarray, v_gcrf: np.ndarray, times_k, date_idx: np.ndarray
        ):
            K = len(times_k)
            groups = [np.where(date_idx == i)[0] for i in range(K)]
            r_out = np.empty_like(r_gcrf)
            v_out = np.empty_like(v_gcrf)
            for i in range(K):
                g = groups[i]
                rep = CartesianRepresentation(
                    r_gcrf[g, 0] * u.m,
                    r_gcrf[g, 1] * u.m,
                    r_gcrf[g, 2] * u.m,
                    differentials=CartesianDifferential(
                        v_gcrf[g, 0] * (u.m / u.s),
                        v_gcrf[g, 1] * (u.m / u.s),
                        v_gcrf[g, 2] * (u.m / u.s),
                    ),
                )
                gcrs = GCRS(rep, obstime=times_k[i])
                itrs = gcrs.transform_to(
                    ITRS(obstime=times_k[i])
                )  # ONE transform_to call per group
                c = itrs.cartesian
                r_out[g, 0] = c.x.to_value(u.m)
                r_out[g, 1] = c.y.to_value(u.m)
                r_out[g, 2] = c.z.to_value(u.m)
                d = c.differentials["s"]
                v_out[g, 0] = d.d_x.to_value(u.m / u.s)
                v_out[g, 1] = d.d_y.to_value(u.m / u.s)
                v_out[g, 2] = d.d_z.to_value(u.m / u.s)
            return r_out, v_out

        # ---------------------------------------------------------------------
        # Bench configurations
        # ---------------------------------------------------------------------
        Ns = [10_000, 50_000, 200_000]
        K_fixed = 16
        dt_seconds = 60.0
        orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K_fixed)]

        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        ast_times_k = t0 + (np.arange(K_fixed) * dt_seconds) * u.s  # (K,)

        # ---------------------------------------------------------------------
        # A) N sweep
        # ---------------------------------------------------------------------
        print("\n=== N sweep (fair, single-date + K-times) ===")
        for N in Ns:
            r_gcrf, v_gcrf = make_states(N)
            date_idx = np.arange(N) % K_fixed
            orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

            # Orekit single-date
            t_ok_pos_single = bench_min(
                lambda: gcrf_to_itrf_pos(r_gcrf, date0), REPEATS, NUMBER
            )
            t_ok_pos_vel_single = bench_min(
                lambda: gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, date0), REPEATS, NUMBER
            )

            # Orekit K-times grouped (AbsoluteDate list with reused objects)
            t_ok_pos_k = bench_min(
                lambda: gcrf_to_itrf_pos(r_gcrf, orekit_dates_per_row),
                REPEATS,
                NUMBER,
            )
            t_ok_pos_vel_k = bench_min(
                lambda: gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, orekit_dates_per_row),
                REPEATS,
                NUMBER,
            )

            print(
                f"N={N:>8} | Orekit POS single {t_ok_pos_single:.4f}s | PV single {t_ok_pos_vel_single:.4f}s"
                f" | POS K {t_ok_pos_k:.4f}s | PV K {t_ok_pos_vel_k:.4f}s"
            )

            # Astropy single-date
            t_ast_pos_single = bench_min(
                lambda: _astropy_pos_single(r_gcrf, ast_times_k[0]), REPEATS, NUMBER
            )
            t_ast_pos_vel_single = bench_min(
                lambda: _astropy_pos_vel_single(r_gcrf, v_gcrf, ast_times_k[0]),
                REPEATS,
                NUMBER,
            )

            # Astropy K-times grouped
            t_ast_pos_k = bench_min(
                lambda: _astropy_pos_grouped(r_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            t_ast_pos_vel_k = bench_min(
                lambda: _astropy_pos_vel_grouped(r_gcrf, v_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )

            print(
                f"          | Astropy POS single {t_ast_pos_single:.4f}s | PV single {t_ast_pos_vel_single:.4f}s"
                f" | POS K {t_ast_pos_k:.4f}s | PV K {t_ast_pos_vel_k:.4f}s"
            )

            print(
                f"          | Ratios Astropy/Orekit: POS single {t_ast_pos_single / t_ok_pos_single:.2f}x,"
                f" PV single {t_ast_pos_vel_single / t_ok_pos_vel_single:.2f}x,"
                f" POS K {t_ast_pos_k / t_ok_pos_k:.2f}x,"
                f" PV K {t_ast_pos_vel_k / t_ok_pos_vel_k:.2f}x"
            )

        # ---------------------------------------------------------------------
        # B) K sweep at fixed N
        # ---------------------------------------------------------------------
        print("\n=== K sweep (fixed N=200k, fair grouped multi-time) ===")
        N = 200_000
        r_gcrf, v_gcrf = make_states(N)

        for K in [1, 4, 16, 64, 128]:
            orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K)]
            date_idx = np.arange(N) % K
            orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

            t_ok_pos_k = bench_min(
                lambda: gcrf_to_itrf_pos(r_gcrf, orekit_dates_per_row),
                REPEATS,
                NUMBER,
            )
            t_ok_pos_vel_k = bench_min(
                lambda: gcrf_to_itrf_pos_vel(r_gcrf, v_gcrf, orekit_dates_per_row),
                REPEATS,
                NUMBER,
            )

            line = f"K={K:>4} | Orekit POS {t_ok_pos_k:.4f}s | PV {t_ok_pos_vel_k:.4f}s"

            t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
            ast_times_k = t0 + (np.arange(K) * dt_seconds) * u.s
            t_ast_pos_k = bench_min(
                lambda: _astropy_pos_grouped(r_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            t_ast_pos_vel_k = bench_min(
                lambda: _astropy_pos_vel_grouped(r_gcrf, v_gcrf, ast_times_k, date_idx),
                REPEATS,
                NUMBER,
            )
            line += f" | Astropy POS {t_ast_pos_k:.4f}s | PV {t_ast_pos_vel_k:.4f}s | POS ratio {t_ast_pos_k / t_ok_pos_k:.2f}x | PV ratio {t_ast_pos_vel_k / t_ok_pos_vel_k:.2f}x"

            print(line)

        # ---------------------------------------------------------------------
        # C) Worst-case unique times (small N)
        # ---------------------------------------------------------------------
        print("\n=== Worst-case unique times (small N) ===")
        Nw = 1_000
        r_w, v_w = make_states(Nw)

        unique_dates = [date0.shiftedBy(float(i)) for i in range(Nw)]
        t_ok_pos_unique = bench_min(
            lambda: gcrf_to_itrf_pos(r_w, unique_dates), repeat=3, number=1
        )
        t_ok_pos_vel_unique = bench_min(
            lambda: gcrf_to_itrf_pos_vel(r_w, v_w, unique_dates), repeat=3, number=1
        )
        print(
            f"N={Nw} unique | Orekit POS {t_ok_pos_unique:.4f}s | PV {t_ok_pos_vel_unique:.4f}s"
        )

        # Astropy unique-time: group size is 1 per time, so this is very expensive;
        # keep Nw small and do 1 repeat.
        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        times_unique = t0 + (np.arange(Nw) * 1.0) * u.s

        # Implement as a loop of single-point transforms (fair but slow)
        def astropy_unique_pos():
            out = np.empty_like(r_w)
            for i in range(Nw):
                rep = CartesianRepresentation(
                    r_w[i, 0] * u.m, r_w[i, 1] * u.m, r_w[i, 2] * u.m
                )
                g = GCRS(rep, obstime=times_unique[i])
                itrs = g.transform_to(ITRS(obstime=times_unique[i]))
                c = itrs.cartesian
                out[i, 0] = c.x.to_value(u.m)
                out[i, 1] = c.y.to_value(u.m)
                out[i, 2] = c.z.to_value(u.m)
            return out

        t_ast_pos_unique = bench_min(astropy_unique_pos, repeat=1, number=1)
        print(
            f"N={Nw} unique | Astropy POS {t_ast_pos_unique:.4f}s | POS ratio {t_ast_pos_unique / t_ok_pos_unique:.2f}x"
        )

        # ---------------------------------------------------------------------
        # D) Input overhead for Orekit: AbsoluteDate list vs astropy Time array
        # ---------------------------------------------------------------------
        print(
            "\n=== Input overhead (Orekit): AbsoluteDate list vs astropy Time array ==="
        )
        N = 200_000
        K = 16
        r_gcrf, _ = make_states(N)

        orekit_dates_k = [date0.shiftedBy(i * dt_seconds) for i in range(K)]
        date_idx = np.arange(N) % K
        orekit_dates_per_row = [orekit_dates_k[int(i)] for i in date_idx]

        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        ast_times_k = t0 + (np.arange(K) * dt_seconds) * u.s
        ast_times_per_row = ast_times_k[date_idx]  # array-valued Time

        t_abs = bench_min(
            lambda: gcrf_to_itrf_pos(r_gcrf, orekit_dates_per_row),
            REPEATS,
            NUMBER,
        )
        t_ast = bench_min(
            lambda: gcrf_to_itrf_pos(r_gcrf, ast_times_per_row), REPEATS, NUMBER
        )

        print(f"Orekit POS K-times with AbsoluteDate list: {t_abs:.4f}s")
        print(f"Orekit POS K-times with astropy Time array: {t_ast:.4f}s")
        print(f"Overhead factor: {t_ast / t_abs:.2f}x")

    # =============================
    # TEST CODE: add this test function (and call it from __main__)
    # =============================
    def test_geodetic(self):
        # Helpers
        def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        def ang_err_rad(a: np.ndarray, b: np.ndarray) -> float:
            # minimal angular difference
            d = a - b
            d = np.arctan2(np.sin(d), np.cos(d))
            return float(np.max(np.abs(d)))

        utc = TimeScalesFactory.getUTC()
        date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            itrf,
        )

        rng = np.random.default_rng(123)
        N = 50_000

        # Random geodetic (avoid exactly +/-90 deg to keep conditioning clean)
        lat = rng.uniform(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, size=N)
        lon = rng.uniform(-np.pi, np.pi, size=N)
        alt = rng.uniform(-1000.0, 1_000_000.0, size=N)  # -1 km to 1000 km
        lla = np.stack([lat, lon, alt], axis=1)

        # 1) Roundtrip geodetic <-> ITRF (NumPy)
        r_itrf = geodetic_to_itrf_pos(lla)
        lla_back = itrf_to_geodetic_pos(r_itrf)

        e_lat = ang_err_rad(lla[:, 0], lla_back[:, 0])
        e_lon = ang_err_rad(lla[:, 1], lla_back[:, 1])
        e_alt = max_abs_err(lla[:, 2], lla_back[:, 2])

        print(
            f"[Geo Test 1] lla -> itrf -> lla max err lat: {e_lat:.3e} rad, lon: {e_lon:.3e} rad, alt: {e_alt:.3e} m"
        )
        assert e_lat < 1e-8
        assert e_lon < 1e-8
        assert e_alt < 1e-2  # ~0.1m typical

        # 2) Cross-check vs Orekit OneAxisEllipsoid for a small sample (positions)
        M = 200
        idx = rng.integers(0, N, size=M)

        max_pos = 0.0
        max_lat = 0.0
        max_lon = 0.0
        max_alt = 0.0

        for i in idx:
            gp = GeodeticPoint(float(lla[i, 0]), float(lla[i, 1]), float(lla[i, 2]))
            v = earth.transform(gp)  # Vector3D in body frame (ITRF)
            ref = np.array([v.getX(), v.getY(), v.getZ()], dtype=np.float64)
            max_pos = max(max_pos, float(np.max(np.abs(ref - r_itrf[i]))))

            # inverse: Orekit cartesian -> geodetic (needs frame+date)
            gp2 = earth.transform(Vector3D(ref[0], ref[1], ref[2]), itrf, date0)
            ref_lla = np.array(
                [gp2.getLatitude(), gp2.getLongitude(), gp2.getAltitude()],
                dtype=np.float64,
            )

            max_lat = max(
                max_lat,
                abs(
                    float(
                        np.arctan2(
                            np.sin(ref_lla[0] - lla_back[i, 0]),
                            np.cos(ref_lla[0] - lla_back[i, 0]),
                        )
                    )
                ),
            )
            max_lon = max(
                max_lon,
                abs(
                    float(
                        np.arctan2(
                            np.sin(ref_lla[1] - lla_back[i, 1]),
                            np.cos(ref_lla[1] - lla_back[i, 1]),
                        )
                    )
                ),
            )
            max_alt = max(max_alt, abs(float(ref_lla[2] - lla_back[i, 2])))

        print(
            f"[Geo Test 2] vs Orekit (sample) max |pos| component err: {max_pos:.3e} m"
        )
        print(
            f"[Geo Test 2] vs Orekit (sample) max err lat: {max_lat:.3e} rad, lon: {max_lon:.3e} rad, alt: {max_alt:.3e} m"
        )
        assert max_pos < 1e-4  # conservative
        assert max_lat < 1e-8
        assert max_lon < 1e-8
        assert max_alt < 1e-2

        # 3) Roundtrip through GCRF with time
        r_gcrf = itrf_to_gcrf_pos(r_itrf, date0)
        lla_from_gcrf = gcrf_to_geodetic_pos(r_gcrf, date0)

        e_lat2 = ang_err_rad(lla[:, 0], lla_from_gcrf[:, 0])
        e_lon2 = ang_err_rad(lla[:, 1], lla_from_gcrf[:, 1])
        e_alt2 = max_abs_err(lla[:, 2], lla_from_gcrf[:, 2])
        print(
            f"[Geo Test 3] lla -> itrf -> gcrf -> itrf -> lla max err lat: {e_lat2:.3e} rad, lon: {e_lon2:.3e} rad, alt: {e_alt2:.3e} m"
        )
        assert e_lat2 < 1e-9
        assert e_lon2 < 1e-9
        assert (
            e_alt2 < 1e-2
        )  # frame pipeline + EOP may introduce tiny mm–cm-level differences

        # 4) Degrees interface sanity
        lla_deg = np.stack([np.rad2deg(lat), np.rad2deg(lon), alt], axis=1)
        r_itrf_deg = geodetic_to_itrf_pos(lla_deg, degrees=True)
        assert max_abs_err(r_itrf_deg, r_itrf) < 1e-8

        lla_deg_back = itrf_to_geodetic_pos(r_itrf, degrees=True)
        # compare degrees for lat/lon, meters for alt
        e_latd = max_abs_err(lla_deg[:, 0], lla_deg_back[:, 0])
        e_lond = max_abs_err(
            ((lla_deg[:, 1] - lla_deg_back[:, 1] + 180.0) % 360.0) - 180.0, 0.0
        )
        e_altd = max_abs_err(lla_deg[:, 2], lla_deg_back[:, 2])
        print(
            f"[Geo Test 4] degrees interface max err lat: {e_latd:.3e} deg, lon (wrapped): {e_lond:.3e} deg, alt: {e_altd:.3e} m"
        )

        # 5) Speed sanity (vectorized numpy vs Orekit point-loop for geodetic->ITRF)
        # Vectorized (full N)
        t_vec = min(
            timeit.repeat(lambda: geodetic_to_itrf_pos(lla), repeat=5, number=1)
        )
        print(f"[Geo Speed] vectorized geodetic->ITRF: {t_vec:.4f}s for N={N}")

        # Orekit point-loop (small M to avoid huge runtimes)
        M2 = 10_000
        lla2 = lla[:M2]

        def orekit_loop():
            out = np.empty((M2, 3), dtype=np.float64)
            for i in range(M2):
                gp = GeodeticPoint(
                    float(lla2[i, 0]), float(lla2[i, 1]), float(lla2[i, 2])
                )
                v = earth.transform(gp)
                out[i, 0] = v.getX()
                out[i, 1] = v.getY()
                out[i, 2] = v.getZ()
            return out

        t_loop = min(timeit.repeat(orekit_loop, repeat=3, number=1))
        print(f"[Geo Speed] Orekit loop geodetic->ITRF: {t_loop:.4f}s for N={M2}")

        print("Geodetic tests passed.")

    # =============================
    # TEST CODE: add this test function and call it from __main__
    # =============================
    def test_j2000_gcrf(self):
        def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        utc = TimeScalesFactory.getUTC()
        date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
        date1 = date0.shiftedBy(86400.0)

        j2000 = FramesFactory.getEME2000()
        gcrf = FramesFactory.getGCRF()

        N = 10000
        rng = np.random.default_rng(0)

        r = rng.normal(size=(N, 3)).astype(np.float64)
        r *= 7_000_000.0 / np.linalg.norm(r, axis=1, keepdims=True)
        v = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

        # Roundtrip POS
        r_g = j2000_to_gcrf_pos(r)
        r_back = gcrf_to_j2000_pos(r_g)
        e = max_abs_err(r, r_back)
        print(f"[J2000/GCRF Test 1] POS roundtrip max abs err: {e:.3e} m")
        assert e < 1e-8

        # Roundtrip PV
        r_g2, v_g2 = j2000_to_gcrf_pos_vel(r, v)
        r_back2, v_back2 = gcrf_to_j2000_pos_vel(r_g2, v_g2)
        er = max_abs_err(r, r_back2)
        ev = max_abs_err(v, v_back2)
        print(
            f"[J2000/GCRF Test 2] PV roundtrip max abs err r: {er:.3e} m, v: {ev:.3e} m/s"
        )
        assert er < 1e-8
        assert ev < 1e-10

        # Validate vs Orekit Transform at two different dates (should match; frames are inertial)
        idx = rng.integers(0, N, size=20)

        for d in [date0, date1]:
            t = j2000.getTransformTo(gcrf, d)

            # position
            for i in idx:
                vv = Vector3D(r[i, 0], r[i, 1], r[i, 2])
                ww = t.transformPosition(vv)
                ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
                assert np.max(np.abs(ref - r_g[i])) < 1e-8

            # PV
            for i in idx:
                pv = PVCoordinates(
                    Vector3D(r[i, 0], r[i, 1], r[i, 2]),
                    Vector3D(v[i, 0], v[i, 1], v[i, 2]),
                )
                pv2 = t.transformPVCoordinates(pv)
                ref_r = np.array(
                    [
                        pv2.getPosition().getX(),
                        pv2.getPosition().getY(),
                        pv2.getPosition().getZ(),
                    ],
                    dtype=np.float64,
                )
                ref_v = np.array(
                    [
                        pv2.getVelocity().getX(),
                        pv2.getVelocity().getY(),
                        pv2.getVelocity().getZ(),
                    ],
                    dtype=np.float64,
                )
                assert np.max(np.abs(ref_r - r_g2[i])) < 1e-8
                assert np.max(np.abs(ref_v - v_g2[i])) < 1e-8

        print(
            "[J2000/GCRF Test 3] Matches Orekit transformPosition/transformPVCoordinates at multiple dates"
        )
        print("J2000/GCRF tests passed.")

    # =============================
    # TESTS: TEME transforms
    # =============================
    def test_teme(self):
        def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(a - b)))

        utc = TimeScalesFactory.getUTC()
        date0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)
        date1 = date0.shiftedBy(123.456)
        date2 = date0.shiftedBy(9876.0)

        teme = FramesFactory.getTEME()
        gcrf = FramesFactory.getGCRF()
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

        N = 5000
        rng = np.random.default_rng(42)

        r_teme = rng.normal(size=(N, 3)).astype(np.float64)
        r_teme *= 7_000_000.0 / np.linalg.norm(r_teme, axis=1, keepdims=True)
        v_teme = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

        # -------------------------
        # 1) Roundtrip: TEME <-> GCRF (single date)
        # -------------------------
        r_g = teme_to_gcrf_pos(r_teme, date0)
        r_back = gcrf_to_teme_pos(r_g, date0)
        e = max_abs_err(r_teme, r_back)
        print(
            f"[TEME Test 1] POS TEME->GCRF->TEME (single date) max abs err: {e:.3e} m"
        )
        assert e < 1e-6

        r_g2, v_g2 = teme_to_gcrf_pos_vel(r_teme, v_teme, date0)
        r_back2, v_back2 = gcrf_to_teme_pos_vel(r_g2, v_g2, date0)
        er = max_abs_err(r_teme, r_back2)
        ev = max_abs_err(v_teme, v_back2)
        print(
            f"[TEME Test 2] PV TEME->GCRF->TEME (single date) max abs err r: {er:.3e} m, v: {ev:.3e} m/s"
        )
        assert er < 1e-6
        assert ev < 1e-6

        # -------------------------
        # 2) Roundtrip: TEME <-> ITRF (single date)
        # -------------------------
        r_i = teme_to_itrf_pos(r_teme, date0)
        r_back3 = itrf_to_teme_pos(r_i, date0)
        e3 = max_abs_err(r_teme, r_back3)
        print(
            f"[TEME Test 3] POS TEME->ITRF->TEME (single date) max abs err: {e3:.3e} m"
        )
        assert e3 < 1e-6

        r_i2, v_i2 = teme_to_itrf_pos_vel(r_teme, v_teme, date0)
        r_back4, v_back4 = itrf_to_teme_pos_vel(r_i2, v_i2, date0)
        er4 = max_abs_err(r_teme, r_back4)
        ev4 = max_abs_err(v_teme, v_back4)
        print(
            f"[TEME Test 4] PV TEME->ITRF->TEME (single date) max abs err r: {er4:.3e} m, v: {ev4:.3e} m/s"
        )
        assert er4 < 1e-6
        assert ev4 < 1e-6

        # -------------------------
        # 3) Grouped multi-date path (repeated dates)
        # -------------------------
        dates = [date0, date1, date2] * (N // 3) + [date0] * (N % 3)
        dates = dates[:N]

        r_gk = teme_to_gcrf_pos(r_teme, dates)
        r_backk = gcrf_to_teme_pos(r_gk, dates)
        ek = max_abs_err(r_teme, r_backk)
        print(
            f"[TEME Test 5] POS TEME<->GCRF (multi-date grouped) max abs err: {ek:.3e} m"
        )
        assert ek < 1e-6

        r_ik, v_ik = teme_to_itrf_pos_vel(r_teme, v_teme, dates)
        r_backk2, v_backk2 = itrf_to_teme_pos_vel(r_ik, v_ik, dates)
        erk = max_abs_err(r_teme, r_backk2)
        evk = max_abs_err(v_teme, v_backk2)
        print(
            f"[TEME Test 6] PV TEME<->ITRF (multi-date grouped) max abs err r: {erk:.3e} m, v: {evk:.3e} m/s"
        )
        assert erk < 1e-6
        assert evk < 1e-6

        # -------------------------
        # 4) Spot-check vs Orekit direct transforms (sample)
        # -------------------------
        idx = rng.integers(0, N, size=20)

        # TEME -> GCRF spot-checks at date0
        t_tg = teme.getTransformTo(gcrf, date0)
        for i in idx:
            vv = Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2])
            ww = t_tg.transformPosition(vv)
            ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
            assert np.max(np.abs(ref - r_g[i])) < 1e-8

            pv = PVCoordinates(
                Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2]),
                Vector3D(v_teme[i, 0], v_teme[i, 1], v_teme[i, 2]),
            )
            pv2 = t_tg.transformPVCoordinates(pv)
            ref_r = np.array(
                [
                    pv2.getPosition().getX(),
                    pv2.getPosition().getY(),
                    pv2.getPosition().getZ(),
                ],
                dtype=np.float64,
            )
            ref_v = np.array(
                [
                    pv2.getVelocity().getX(),
                    pv2.getVelocity().getY(),
                    pv2.getVelocity().getZ(),
                ],
                dtype=np.float64,
            )
            assert np.max(np.abs(ref_r - r_g2[i])) < 1e-8
            assert np.max(np.abs(ref_v - v_g2[i])) < 1e-8

        # TEME -> ITRF spot-checks at date0
        t_ti = teme.getTransformTo(itrf, date0)
        for i in idx:
            vv = Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2])
            ww = t_ti.transformPosition(vv)
            ref = np.array([ww.getX(), ww.getY(), ww.getZ()], dtype=np.float64)
            assert np.max(np.abs(ref - r_i[i])) < 1e-8

            pv = PVCoordinates(
                Vector3D(r_teme[i, 0], r_teme[i, 1], r_teme[i, 2]),
                Vector3D(v_teme[i, 0], v_teme[i, 1], v_teme[i, 2]),
            )
            pv2 = t_ti.transformPVCoordinates(pv)
            ref_r = np.array(
                [
                    pv2.getPosition().getX(),
                    pv2.getPosition().getY(),
                    pv2.getPosition().getZ(),
                ],
                dtype=np.float64,
            )
            ref_v = np.array(
                [
                    pv2.getVelocity().getX(),
                    pv2.getVelocity().getY(),
                    pv2.getVelocity().getZ(),
                ],
                dtype=np.float64,
            )
            assert np.max(np.abs(ref_r - r_i2[i])) < 1e-8
            assert np.max(np.abs(ref_v - v_i2[i])) < 1e-8

        print(
            "[TEME Test 7] Matches Orekit transformPosition/transformPVCoordinates (sample) OK"
        )
        print("TEME tests passed.")

    def test_teme_vs_astropy(self):
        """
        Cross-check Orekit TEME transforms against Astropy's TEME transforms.

        This is an *inter-library* comparison and is sensitive to EOP/IERS configuration.
        Your TEME implementation is already validated against Orekit itself (TEME Test 7);
        this test is mainly to catch gross mistakes, not to guarantee mm-level agreement.

        Strategy:
        - Force Astropy to use IERS-B (offline, stable) and disable downloads.
        - Use Orekit ITRF with simple_eop=False for closer parity with Astropy.
        - Assert only loose bounds (meters / cm/s). Print guidance if larger.
        """
        # -------------------------
        # Pin Astropy IERS behavior (offline, stable)
        # -------------------------
        try:
            iers.conf.auto_download = False
            iers.conf.iers_degraded_accuracy = "warn"

            iers_b = None
            try:
                iers_b = iers.IERS_B.open(iers.IERS_B_FILE)
            except Exception:
                iers_b = None

            # Use the global Earth orientation table if available (Astropy versions vary)
            # If this fails, Astropy will still run but may use a different EOP source/behavior.
            if iers_b is not None:
                try:
                    # Astropy >= 5-ish
                    iers.earth_orientation_table.set(iers_b)
                except Exception:
                    pass
        except Exception:
            pass

        rng = np.random.default_rng(123)

        N = 50_000
        K = 16
        dt = 60.0

        t0 = AstropyTime("2026-01-16T12:00:00", scale="utc")
        times_k = t0 + (np.arange(K) * dt) * u.s
        date_idx = np.arange(N) % K
        times_per_row = times_k[date_idx]

        # Random TEME states
        r_teme = rng.normal(size=(N, 3)).astype(np.float64)
        r_teme *= 7_000_000.0 / np.linalg.norm(r_teme, axis=1, keepdims=True)
        v_teme = rng.normal(size=(N, 3)).astype(np.float64) * 7500.0

        # -------------------------
        # Orekit results
        # Use simple_eop=False here (closer parity to typical Astropy behavior)
        # -------------------------
        r_itrf_ok = teme_to_itrf_pos(r_teme, times_per_row, simple_eop=False)
        r_gcrf_ok = teme_to_gcrf_pos(r_teme, times_per_row)

        r_itrf_ok_pos_vel, v_itrf_ok_pos_vel = teme_to_itrf_pos_vel(
            r_teme, v_teme, times_per_row, simple_eop=False
        )
        r_gcrf_ok_pos_vel, v_gcrf_ok_pos_vel = teme_to_gcrf_pos_vel(
            r_teme, v_teme, times_per_row
        )

        # -------------------------
        # Astropy results (grouped by K)
        # -------------------------
        r_itrf_ast = np.empty_like(r_teme)
        r_gcrs_ast = np.empty_like(r_teme)
        v_itrf_ast = np.empty_like(v_teme)
        v_gcrs_ast = np.empty_like(v_teme)

        groups = [np.where(date_idx == i)[0] for i in range(K)]
        for i in range(K):
            idxs = groups[i]
            ti = times_k[i]

            rep = CartesianRepresentation(
                r_teme[idxs, 0] * u.m,
                r_teme[idxs, 1] * u.m,
                r_teme[idxs, 2] * u.m,
                differentials=CartesianDifferential(
                    v_teme[idxs, 0] * (u.m / u.s),
                    v_teme[idxs, 1] * (u.m / u.s),
                    v_teme[idxs, 2] * (u.m / u.s),
                ),
            )

            teme = TEME(rep, obstime=ti)

            itrs = teme.transform_to(ITRS(obstime=ti))
            c = itrs.cartesian
            r_itrf_ast[idxs, 0] = c.x.to_value(u.m)
            r_itrf_ast[idxs, 1] = c.y.to_value(u.m)
            r_itrf_ast[idxs, 2] = c.z.to_value(u.m)
            d = c.differentials["s"]
            v_itrf_ast[idxs, 0] = d.d_x.to_value(u.m / u.s)
            v_itrf_ast[idxs, 1] = d.d_y.to_value(u.m / u.s)
            v_itrf_ast[idxs, 2] = d.d_z.to_value(u.m / u.s)

            gcrs = teme.transform_to(GCRS(obstime=ti))
            c = gcrs.cartesian
            r_gcrs_ast[idxs, 0] = c.x.to_value(u.m)
            r_gcrs_ast[idxs, 1] = c.y.to_value(u.m)
            r_gcrs_ast[idxs, 2] = c.z.to_value(u.m)
            d = c.differentials["s"]
            v_gcrs_ast[idxs, 0] = d.d_x.to_value(u.m / u.s)
            v_gcrs_ast[idxs, 1] = d.d_y.to_value(u.m / u.s)
            v_gcrs_ast[idxs, 2] = d.d_z.to_value(u.m / u.s)

        def max_abs(a: np.ndarray) -> float:
            return float(np.max(np.abs(a)))

        pos_itrf_err = max_abs(r_itrf_ok - r_itrf_ast)
        pos_gcrf_err = max_abs(r_gcrf_ok - r_gcrs_ast)

        pv_itrf_r_err = max_abs(r_itrf_ok_pos_vel - r_itrf_ast)
        pv_itrf_v_err = max_abs(v_itrf_ok_pos_vel - v_itrf_ast)

        pv_gcrf_r_err = max_abs(r_gcrf_ok_pos_vel - r_gcrs_ast)
        pv_gcrf_v_err = max_abs(v_gcrf_ok_pos_vel - v_gcrs_ast)

        print(
            f"[TEME vs Astropy] POS TEME->ITRF(ITRS) max abs err: {pos_itrf_err:.6e} m"
        )
        print(
            f"[TEME vs Astropy] POS TEME->GCRF(GCRS) max abs err: {pos_gcrf_err:.6e} m"
        )
        print(
            f"[TEME vs Astropy] PV  TEME->ITRF(ITRS) max abs err: r {pv_itrf_r_err:.6e} m, v {pv_itrf_v_err:.6e} m/s"
        )
        print(
            f"[TEME vs Astropy] PV  TEME->GCRF(GCRS) max abs err: r {pv_gcrf_r_err:.6e} m, v {pv_gcrf_v_err:.6e} m/s"
        )

        # Loose bounds for inter-library checks.
        # If these fail, it *still* may be EOP/IERS differences; inspect the printed errors.
        POS_TOL_M = 20.0
        VEL_TOL_MPS = 0.10

        assert pos_itrf_err < POS_TOL_M, (
            f"TEME->ITRF mismatch too large ({pos_itrf_err:.3f} m). "
            "Likely EOP/IERS mismatch; verify both libraries use comparable EOP sources."
        )
        assert pos_gcrf_err < POS_TOL_M, (
            f"TEME->GCRF mismatch too large ({pos_gcrf_err:.3f} m). "
            "Likely frame/EOP differences (GCRF vs GCRS)."
        )
        assert pv_itrf_v_err < VEL_TOL_MPS, (
            f"TEME->ITRF velocity mismatch too large ({pv_itrf_v_err:.3f} m/s). "
            "Likely EOP/IERS mismatch."
        )
        assert pv_gcrf_v_err < VEL_TOL_MPS, (
            f"TEME->GCRF velocity mismatch too large ({pv_gcrf_v_err:.3f} m/s). "
            "Likely frame/EOP differences (GCRF vs GCRS)."
        )

        print("TEME vs Astropy cross-check passed (loose inter-library tolerances).")


if __name__ == "__main__":
    unittest.main(exit=False)
