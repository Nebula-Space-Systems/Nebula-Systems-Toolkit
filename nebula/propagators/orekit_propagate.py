from __future__ import annotations
import os
from math import radians

import numpy as np
import jdk4py
import orekit_jpype
from enum import Enum

# 1. Setup JVM and Data
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
vm = orekit_jpype.initVM()
from orekit_jpype.pyhelpers import setup_orekit_curdir
import matplotlib.pyplot as plt

setup_orekit_curdir(
    filename=os.path.join(os.path.dirname(__file__), "..", "data", "orekit-data")
)

from typing import Any, Mapping, Optional, Sequence, Tuple, Union, Literal

from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # type: ignore

# Bodies / shapes
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid  # type: ignore

# Drag
from org.orekit.forces.drag import DragForce, IsotropicDrag  # type: ignore

# Gravity
from org.orekit.forces.gravity import (  # type: ignore
    DeSitterRelativity,
    HolmesFeatherstoneAttractionModel,
    LenseThirringRelativity,
    OceanTides,
    Relativity,
    SolidTides,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import (  # type: ignore
    GravityFieldFactory,
    TideSystem,
)

# Radiation
from org.orekit.forces.radiation import (  # type: ignore
    IsotropicRadiationSingleCoefficient,
    KnockeRediffusedForceModel,
    SolarRadiationPressure,
)

# Orekit core
from org.orekit.frames import FramesFactory  # type: ignore
from org.orekit.models.earth.atmosphere import NRLMSISE00  # type: ignore
from org.orekit.models.earth.atmosphere.data import (  # type: ignore
    MarshallSolarActivityFutureEstimation,
)
from org.orekit.orbits import (  # type: ignore
    KeplerianOrbit,
    OrbitType,
    PositionAngleType,
)
from org.orekit.propagation import SpacecraftState  # type: ignore
from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
from org.orekit.utils import Constants, IERSConventions  # type: ignore


def profile_force_models(
    *,
    initial_orbit,
    initial_date,
    duration_s: float,
    mass_kg: float,
    itrf,
    earth,  # OneAxisEllipsoid
    mu: float,
    ae: float,
    utc,
    # integrator config
    min_step: float = 0.001,
    max_step: float = 1000.0,
    initial_step: float = 60.0,
    position_tolerance_m: float = 1.0,
    # profiling config
    repeats: int = 3,
    warmup: int = 1,
    verbose: bool = True,
):
    """
    Profile incremental runtime cost of each Orekit force model by measuring:
        T(baseline + model) - T(baseline)

    Notes on fairness:
      - Uses a fresh NumericalPropagator for each timed run.
      - Same integrator/tolerances/state/time span for all runs.
      - Repeats each run and reports min time (less sensitive to OS jitter).
      - JVM is warmed up before timings.

    Returns
    -------
    results : list[dict]
        Sorted list of dicts with keys:
          - name
          - baseline_s
          - total_s
          - delta_s
          - delta_pct
    """
    import gc
    import time

    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import (
        DeSitterRelativity,
        HolmesFeatherstoneAttractionModel,
        LenseThirringRelativity,
        OceanTides,
        Relativity,
        SolidTides,
        ThirdBodyAttraction,
    )
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
    from org.orekit.forces.radiation import (
        IsotropicRadiationSingleCoefficient,
        KnockeRediffusedForceModel,
        SolarRadiationPressure,
    )
    from org.orekit.models.earth.atmosphere import NRLMSISE00
    from org.orekit.models.earth.atmosphere.data import (
        MarshallSolarActivityFutureEstimation,
    )
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.time import TimeScalesFactory
    from org.orekit.utils import Constants, IERSConventions

    # -----------------------------
    # Shared bodies and aux objects
    # -----------------------------
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    earth_body = CelestialBodyFactory.getEarth()

    # UT1 for tides/ocean tides
    ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True)
    tidesystem = TideSystem.ZERO_TIDE

    # Atmosphere + drag sensitive
    msafe = MarshallSolarActivityFutureEstimation(
        MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
        MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE,
    )
    atmosphere = NRLMSISE00(msafe, sun, earth)
    drag_sensitive = IsotropicDrag(1.0, 2.2)  # area=1 m^2, Cd=2.2

    # Radiation sensitive + SRP + ERP
    radiation_sensitive = IsotropicRadiationSingleCoefficient(
        1.0, 1.2
    )  # area=1 m^2, Cr=1.2

    # Gravity field for "main geopotential" model (you can change degrees to see scaling)
    degree, order = 20, 20
    gravity_provider = GravityFieldFactory.getNormalizedProvider(degree, order)
    gravity_model = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)

    # Ocean tides config
    ocean_degree, ocean_order = 8, 8

    # SRP + moon occultation
    srp = SolarRadiationPressure(sun, earth, radiation_sensitive)
    srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)

    # ERP (Knocke)
    from math import radians

    angular_resolution_rad = radians(1.0)
    erp = KnockeRediffusedForceModel(
        sun,
        radiation_sensitive,
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        angular_resolution_rad,
        utc,
    )

    # Third bodies
    venus = CelestialBodyFactory.getVenus()
    mars = CelestialBodyFactory.getMars()
    jupiter = CelestialBodyFactory.getJupiter()
    saturn = CelestialBodyFactory.getSaturn()

    # -----------------------------
    # Build model list (callables)
    # -----------------------------
    # Each entry returns a *new* force model instance (or a safe reusable one).
    models = [
        ("HolmesFeatherstoneAttractionModel (EGM, 20x20)", lambda: gravity_model),
        ("DragForce (NRLMSISE00)", lambda: DragForce(atmosphere, drag_sensitive)),
        ("ThirdBodyAttraction (Sun)", lambda: ThirdBodyAttraction(sun)),
        ("ThirdBodyAttraction (Moon)", lambda: ThirdBodyAttraction(moon)),
        ("ThirdBodyAttraction (Venus)", lambda: ThirdBodyAttraction(venus)),
        ("ThirdBodyAttraction (Mars)", lambda: ThirdBodyAttraction(mars)),
        ("ThirdBodyAttraction (Jupiter)", lambda: ThirdBodyAttraction(jupiter)),
        ("ThirdBodyAttraction (Saturn)", lambda: ThirdBodyAttraction(saturn)),
        (
            "SolidTides (Sun)",
            lambda: SolidTides(
                itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, sun
            ),
        ),
        (
            "SolidTides (Moon)",
            lambda: SolidTides(
                itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, moon
            ),
        ),
        (
            "OceanTides (8x8)",
            lambda: OceanTides(
                itrf, ae, mu, ocean_degree, ocean_order, IERSConventions.IERS_2010, ut1
            ),
        ),
        ("Relativity (Schwarzschild)", lambda: Relativity(mu)),
        ("DeSitterRelativity", lambda: DeSitterRelativity(earth_body, sun)),
        ("LenseThirringRelativity", lambda: LenseThirringRelativity(mu, itrf)),
        ("SolarRadiationPressure (+Moon occult.)", lambda: srp),
        ("KnockeRediffusedForceModel (ERP)", lambda: erp),
    ]

    # -----------------------------
    # Propagator builder
    # -----------------------------
    def build_propagator(force_models):
        tolerances = NumericalPropagator.tolerances(
            position_tolerance_m, initial_orbit, initial_orbit.getType()
        )
        integrator = DormandPrince853Integrator(
            min_step, max_step, tolerances[0], tolerances[1]
        )
        integrator.setInitialStepSize(initial_step)

        prop = NumericalPropagator(integrator)
        prop.setInitialState(SpacecraftState(initial_orbit, mass_kg))

        for fm in force_models:
            prop.addForceModel(fm)

        return prop

    tf = initial_date.shiftedBy(float(duration_s))

    def time_run(force_models):
        # fresh propagator each timed run
        prop = build_propagator(force_models)

        # Avoid ephemeris generation overhead for pure runtime profiling unless you want it
        # (ephemeris storage can dominate for long runs).
        # gen = prop.getEphemerisGenerator()

        t0 = time.perf_counter()
        prop.propagate(tf)
        t1 = time.perf_counter()
        return t1 - t0

    def best_of_n(force_models, n):
        best = None
        for _ in range(n):
            gc.collect()
            dt = time_run(force_models)
            best = dt if best is None else min(best, dt)
        return float(best)

    # -----------------------------
    # Warm up JVM/JIT
    # -----------------------------
    if warmup > 0:
        if verbose:
            print(f"[warmup] running {warmup} short runs...")
        short_tf = initial_date.shiftedBy(min(60.0, float(duration_s) * 0.1))

        def warmup_run():
            prop = build_propagator([gravity_model])  # small-but-nontrivial
            prop.propagate(short_tf)

        for _ in range(warmup):
            warmup_run()

    # -----------------------------
    # Baseline: no force models
    # -----------------------------
    # This measures only integrator + propagation plumbing.
    baseline_s = best_of_n([], repeats)
    if verbose:
        print(f"[baseline] {baseline_s:.6f} s (no force models)")

    # -----------------------------
    # Profile each model: baseline + model
    # -----------------------------
    results = []
    for name, ctor in models:
        fm = ctor()
        total_s = best_of_n([fm], repeats)
        delta_s = total_s - baseline_s
        delta_pct = (delta_s / baseline_s * 100.0) if baseline_s > 0 else float("inf")
        results.append(
            {
                "name": name,
                "baseline_s": baseline_s,
                "total_s": total_s,
                "delta_s": delta_s,
                "delta_pct": delta_pct,
            }
        )
        if verbose:
            print(
                f"[{name}] total={total_s:.6f}s  delta={delta_s:.6f}s  ({delta_pct:+.1f}%)"
            )

    # Sort by incremental cost
    results.sort(key=lambda d: d["delta_s"], reverse=True)

    if verbose:
        print("\n=== Force model incremental cost ranking (highest first) ===")
        for i, r in enumerate(results, 1):
            print(
                f"{i:>2}. {r['name']:<45} delta={r['delta_s']:.6f}s  total={r['total_s']:.6f}s"
            )

    return results


def main():
    # 2. Time / frames / constants
    utc = TimeScalesFactory.getUTC()
    initial_date = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

    inertial_frame = FramesFactory.getGCRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    mu = Constants.WGS84_EARTH_MU
    ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    # 3. Define Initial Orbit (Geostationary)
    initial_orbit = KeplerianOrbit(
        42164000.0,  # semi-major axis (m) for GEO
        0.0015,  # eccentricity (circular)
        radians(10.0),  # inclination
        0.0,  # RAAN
        0.0,  # argument of perigee
        0.0,  # true anomaly
        PositionAngleType.TRUE,
        inertial_frame,
        initial_date,
        mu,
    )

    # 4. Setup Numerical Integrator (Dormand-Prince 8(5,3))
    min_step, max_step, initial_step = 0.001, 1000.0, 60.0
    position_tolerance = 1.0  # meters
    tolerances = NumericalPropagator.tolerances(
        position_tolerance, initial_orbit, OrbitType.KEPLERIAN
    )

    integrator = DormandPrince853Integrator(
        min_step, max_step, tolerances[0], tolerances[1]
    )
    integrator.setInitialStepSize(initial_step)

    # 5. Initialize Numerical Propagator
    propagator = NumericalPropagator(integrator)
    gen = propagator.getEphemerisGenerator()

    # Set explicit mass (drag + radiation acceleration depends on mass)
    mass_kg = 1000.0
    propagator.setInitialState(SpacecraftState(initial_orbit, mass_kg))

    # -------------------------------------------------------------------------
    # Force models
    # -------------------------------------------------------------------------

    # A) Earth gravity field (EGM2008 via Orekit data), including J2/Jn harmonics
    degree, order = 20, 20
    gravity_provider = GravityFieldFactory.getNormalizedProvider(degree, order)
    gravity_model = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)
    propagator.addForceModel(gravity_model)

    # B) Atmospheric drag (NRLMSISE-00)
    msafe = MarshallSolarActivityFutureEstimation(
        MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
        MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE,
    )
    sun = CelestialBodyFactory.getSun()
    atmosphere = NRLMSISE00(msafe, sun, earth)

    earth_body = CelestialBodyFactory.getEarth()  # CelestialBody (not OneAxisEllipsoid)

    # De Sitter (geodetic) relativity: constructor takes (Earth, Sun)
    propagator.addForceModel(DeSitterRelativity(earth_body, sun))

    # Lense–Thirring (frame-dragging): constructor takes (Earth GM, central body frame)
    propagator.addForceModel(LenseThirringRelativity(mu, itrf))

    cross_section_m2 = 1.0  # drag area
    cd = 2.2  # drag coefficient
    drag_sensitive = IsotropicDrag(cross_section_m2, cd)
    propagator.addForceModel(DragForce(atmosphere, drag_sensitive))

    # C) Third-body gravity (always include Sun + Moon; others are usually small but valid)
    moon = CelestialBodyFactory.getMoon()
    propagator.addForceModel(ThirdBodyAttraction(sun))
    propagator.addForceModel(ThirdBodyAttraction(moon))

    # Optional additional planets (small for most Earth orbits; more relevant for high-altitude / long arcs)
    venus = CelestialBodyFactory.getVenus()
    mars = CelestialBodyFactory.getMars()
    jupiter = CelestialBodyFactory.getJupiter()
    saturn = CelestialBodyFactory.getSaturn()
    propagator.addForceModel(ThirdBodyAttraction(venus))
    propagator.addForceModel(ThirdBodyAttraction(mars))
    propagator.addForceModel(ThirdBodyAttraction(jupiter))
    propagator.addForceModel(ThirdBodyAttraction(saturn))

    # D) Solid Earth tides (Sun + Moon)
    # Use UT1 for Earth rotation effects in tides
    ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True)
    tidesystem = TideSystem.ZERO_TIDE
    propagator.addForceModel(
        SolidTides(itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, sun)
    )
    propagator.addForceModel(
        SolidTides(itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, moon)
    )

    # E) Ocean tides (pick a modest degree/order; increase if you need more fidelity)
    ocean_degree, ocean_order = 8, 8
    propagator.addForceModel(
        OceanTides(
            itrf, ae, mu, ocean_degree, ocean_order, IERSConventions.IERS_2010, ut1
        )
    )

    # F) General relativity correction for Earth orbit (post-Newtonian)
    propagator.addForceModel(Relativity(mu))

    # G) Solar Radiation Pressure (SRP) with eclipses by Earth, plus Moon occultation
    cr = 1.2  # reflectivity coefficient
    srp_area_m2 = 1.0  # SRP area (often not the same as drag area)
    radiation_sensitive = IsotropicRadiationSingleCoefficient(srp_area_m2, cr)

    srp = SolarRadiationPressure(sun, earth, radiation_sensitive)
    srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)
    propagator.addForceModel(srp)

    # H) Earth radiation pressure (albedo + IR), Knocke model
    # Angular resolution trades accuracy vs speed (1 degree is a common starting point)
    # angular_resolution_rad = radians(3)
    # propagator.addForceModel(
    #     KnockeRediffusedForceModel(
    #         sun,
    #         radiation_sensitive,
    #         Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    #         angular_resolution_rad,
    #         utc,
    #     )
    # )

    # 6. Propagate
    import time

    t0 = time.time()
    duration = 86400.0 * 1  # 30 days
    final_state = propagator.propagate(initial_date.shiftedBy(duration))
    t1 = time.time()
    print(f"Propagation time: {t1 - t0:.3f} seconds")
    print(
        "Final Position with EGM2008 + tides + drag + SRP + ERP + 3rd bodies + relativity: "
        f"{final_state.getPVCoordinates().getPosition()}"
    )
    ephem = gen.getGeneratedEphemeris()
    # sample ephemeris every 10 minutes
    _sample_times = [
        initial_date.shiftedBy(dt) for dt in range(0, int(duration) + 1, 600 * 6)
    ]
    pos = [ephem.propagate(t).getPVCoordinates().getPosition() for t in _sample_times]
    from nebula.transform._transform import gcrf_to_geodetic_positions

    lla = gcrf_to_geodetic_positions(
        [p.toArray() for p in pos],
        _sample_times,
        degrees=True,
    )
    fig, ax, crs, gl = make_basemap(
        projection="PlateCarree",
        draw_coastlines=True,
        texture=False,
        style=LIGHT_THEME,
        use_raster_background=False,
        draw_countries=False,
    )
    add_orbit_trace_cartopy(ax, lla[:, 0], lla[:, 1], color="C1", label="Orbit Track")
    plt.show()
    pass

    # results = profile_force_models(
    #     initial_orbit=initial_orbit,
    #     initial_date=initial_date,
    #     duration_s=3600.0 * 4,  # 1 hour
    #     mass_kg=mass_kg,
    #     itrf=itrf,
    #     earth=earth,
    #     mu=mu,
    #     ae=ae,
    #     utc=utc,
    #     repeats=3,
    #     warmup=1,
    #     verbose=True,
    # )


if __name__ == "__main__":
    main()
