"""Internal Orekit propagation-construction helpers.

This module keeps propagator-builder logic out of :mod:`nstk.propagation.orbit`
so that :class:`nstk.propagation.orbit.Orbit` remains a thin wrapper around an
already-configured Orekit propagator.
"""

from __future__ import annotations

import math
from typing import Any

import nstk._orekit_frames as _orekit_frames


_RUNTIME_BOUND = False
FramesFactory = None
PositionAngleType = None
TimeScalesFactory = None
OneAxisEllipsoid = None


def _bind_java() -> None:
    """Bind Orekit propagation-construction types lazily."""

    global _RUNTIME_BOUND
    global FramesFactory, PositionAngleType, TimeScalesFactory, OneAxisEllipsoid

    if _RUNTIME_BOUND:
        return

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()
    _orekit_frames._bind_java()

    from org.orekit.bodies import OneAxisEllipsoid as _OneAxisEllipsoid  # type: ignore
    from org.orekit.orbits import PositionAngleType as _PositionAngleType  # type: ignore
    from org.orekit.time import TimeScalesFactory as _TimeScalesFactory  # type: ignore

    FramesFactory = _orekit_frames.FramesFactory
    PositionAngleType = _PositionAngleType
    TimeScalesFactory = _TimeScalesFactory
    OneAxisEllipsoid = _OneAxisEllipsoid
    _RUNTIME_BOUND = True


def _coerce_position_angle_type(anomaly_type: Any):
    """Normalize anomaly-type input to Orekit ``PositionAngleType``."""

    _bind_java()
    if anomaly_type is None:
        return PositionAngleType.MEAN
    if isinstance(anomaly_type, str):
        key = anomaly_type.strip().lower()
        if key == "mean":
            return PositionAngleType.MEAN
        if key == "true":
            return PositionAngleType.TRUE
        if key == "eccentric":
            return PositionAngleType.ECCENTRIC
        raise ValueError(
            "anomaly_type must be 'mean', 'true', 'eccentric', or PositionAngleType"
        )
    if anomaly_type in (
        PositionAngleType.MEAN,
        PositionAngleType.TRUE,
        PositionAngleType.ECCENTRIC,
    ):
        return anomaly_type
    raise ValueError("Unsupported anomaly_type")


def _validate_kepler(a: float, e: float, mass: float) -> None:
    """Validate basic Keplerian constructor inputs."""

    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("semi-major axis 'a' must be finite and > 0")
    if not math.isfinite(e) or e < 0.0:
        raise ValueError("eccentricity 'e' must be finite and >= 0")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and > 0")


def _resolve_solar_activity_strength(level: str):
    """Map user-friendly solar-activity label to the Orekit enum."""

    _bind_java()
    from org.orekit.models.earth.atmosphere.data import (  # type: ignore
        MarshallSolarActivityFutureEstimation,
    )

    key = str(level).strip().lower()
    if key == "average":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
    if key == "weak":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.WEAK
    if key == "strong":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.STRONG
    raise ValueError("solar_activity_strength must be 'average', 'weak', or 'strong'")


def _resolve_third_body(name: str):
    """Resolve a third-body label to an Orekit celestial body."""

    _bind_java()
    from org.orekit.bodies import CelestialBodyFactory  # type: ignore

    key = str(name).strip().lower()
    if key == "sun":
        return CelestialBodyFactory.getSun()
    if key == "moon":
        return CelestialBodyFactory.getMoon()
    if key == "earth":
        return CelestialBodyFactory.getEarth()
    if key == "mercury":
        return CelestialBodyFactory.getMercury()
    if key == "venus":
        return CelestialBodyFactory.getVenus()
    if key == "mars":
        return CelestialBodyFactory.getMars()
    if key == "jupiter":
        return CelestialBodyFactory.getJupiter()
    if key == "saturn":
        return CelestialBodyFactory.getSaturn()
    if key == "uranus":
        return CelestialBodyFactory.getUranus()
    if key == "neptune":
        return CelestialBodyFactory.getNeptune()
    if key == "pluto":
        return CelestialBodyFactory.getPluto()
    raise ValueError(f"Unsupported third body: '{name}'")


def _build_earth_shape(itrf: Any):
    """Create a WGS84 Earth shape in the requested Earth-fixed frame."""

    _bind_java()
    from org.orekit.utils import Constants  # type: ignore

    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _build_numerical_propagator(
    *,
    initial_orbit: Any,
    initial_state: Any,
    position_tolerance_m: float,
    min_step_s: float,
    max_step_s: float,
    initial_step_s: float,
):
    """Create a DP853-based Orekit ``NumericalPropagator`` in Cartesian form."""

    _bind_java()
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # type: ignore
    from org.orekit.orbits import OrbitType  # type: ignore
    from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore

    tolerances = NumericalPropagator.tolerances(
        float(position_tolerance_m),
        initial_orbit,
        OrbitType.CARTESIAN,
    )

    integrator = DormandPrince853Integrator(
        float(min_step_s),
        float(max_step_s),
        tolerances[0],
        tolerances[1],
    )
    integrator.setInitialStepSize(float(initial_step_s))

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)
    return propagator


def _configure_numerical_force_models(
    *,
    propagator: Any,
    itrf: Any,
    utc: Any,
    iers: Any,
    simple_eop: bool,
    mu: float,
    ae: float,
    earth_shape: Any,
    gravity_degree: int = 20,
    gravity_order: int = 20,
    enable_drag: bool = False,
    drag_area_m2: float = 1.0,
    drag_cd: float = 2.2,
    solar_activity_strength: str = "average",
    enable_third_body: bool = True,
    third_bodies: tuple[str, ...] = ("sun", "moon"),
    enable_solid_tides: bool = False,
    solid_tides_bodies: tuple[str, ...] = ("sun", "moon"),
    enable_ocean_tides: bool = False,
    ocean_degree: int = 8,
    ocean_order: int = 8,
    enable_relativity: bool = False,
    enable_de_sitter: bool = False,
    enable_lense_thirring: bool = False,
    enable_srp: bool = False,
    srp_area_m2: float = 1.0,
    srp_cr: float = 1.2,
    srp_occult_moon: bool = True,
    enable_erp: bool = False,
    erp_angular_resolution_deg: float = 1.0,
) -> None:
    """Attach selected high-fidelity force models to a numerical propagator."""

    _bind_java()
    from org.orekit.forces.drag import DragForce, IsotropicDrag  # type: ignore
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
    from org.orekit.forces.radiation import (  # type: ignore
        IsotropicRadiationSingleCoefficient,
        KnockeRediffusedForceModel,
        SolarRadiationPressure,
    )
    from org.orekit.models.earth.atmosphere import NRLMSISE00  # type: ignore
    from org.orekit.models.earth.atmosphere.data import (  # type: ignore
        MarshallSolarActivityFutureEstimation,
    )
    from org.orekit.utils import Constants  # type: ignore

    provider = GravityFieldFactory.getNormalizedProvider(
        int(gravity_degree), int(gravity_order)
    )
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, provider))

    sun = _resolve_third_body("sun")
    moon = _resolve_third_body("moon")
    earth_body = _resolve_third_body("earth")

    if enable_drag:
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            _resolve_solar_activity_strength(solar_activity_strength),
        )
        atmosphere = NRLMSISE00(msafe, sun, earth_shape)
        drag_sensitive = IsotropicDrag(float(drag_area_m2), float(drag_cd))
        propagator.addForceModel(DragForce(atmosphere, drag_sensitive))

    if enable_third_body:
        for name in third_bodies:
            propagator.addForceModel(ThirdBodyAttraction(_resolve_third_body(name)))

    if enable_solid_tides or enable_ocean_tides:
        ut1 = TimeScalesFactory.getUT1(iers, bool(simple_eop))
        tide_system = TideSystem.ZERO_TIDE

        if enable_solid_tides:
            for name in solid_tides_bodies:
                propagator.addForceModel(
                    SolidTides(
                        itrf,
                        float(ae),
                        float(mu),
                        tide_system,
                        iers,
                        ut1,
                        _resolve_third_body(name),
                    )
                )

        if enable_ocean_tides:
            propagator.addForceModel(
                OceanTides(
                    itrf,
                    float(ae),
                    float(mu),
                    int(ocean_degree),
                    int(ocean_order),
                    iers,
                    ut1,
                )
            )

    if enable_relativity:
        propagator.addForceModel(Relativity(float(mu)))
    if enable_de_sitter:
        propagator.addForceModel(DeSitterRelativity(earth_body, sun))
    if enable_lense_thirring:
        propagator.addForceModel(LenseThirringRelativity(float(mu), itrf))

    radiation_sensitive = None
    if enable_srp or enable_erp:
        radiation_sensitive = IsotropicRadiationSingleCoefficient(
            float(srp_area_m2), float(srp_cr)
        )

    if enable_srp:
        srp = SolarRadiationPressure(sun, earth_shape, radiation_sensitive)
        if bool(srp_occult_moon):
            srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)
        propagator.addForceModel(srp)

    if enable_erp:
        if radiation_sensitive is None:
            radiation_sensitive = IsotropicRadiationSingleCoefficient(
                float(srp_area_m2), float(srp_cr)
            )
        ang_res_rad = math.radians(float(erp_angular_resolution_deg))
        propagator.addForceModel(
            KnockeRediffusedForceModel(
                sun,
                radiation_sensitive,
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                ang_res_rad,
                utc,
            )
        )


__all__ = [
    "PositionAngleType",
    "_bind_java",
    "_build_earth_shape",
    "_build_numerical_propagator",
    "_coerce_position_angle_type",
    "_configure_numerical_force_models",
    "_resolve_solar_activity_strength",
    "_resolve_third_body",
    "_validate_kepler",
]
