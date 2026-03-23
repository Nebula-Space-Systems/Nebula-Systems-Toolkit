"""High-level Python interface for the standalone Java-first Orekit orbit wrapper.

This module intentionally keeps Python-side work minimal. Heavy propagation,
ephemeris interpolation, frame transforms, geodetic conversion, and attitude
queries are executed in Java via ``OrekitOrbitPropagationBridge``.

Design goals:
- Thin, ergonomic Python API for Orekit users.
- Lazy JVM startup (no VM init at module import time).
- Efficient vector queries with Java-side loops.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from typing import Any, Union

import astropy.units as u
import numpy as np
from astropy.time import Time

from nstk.time_utils import (
    astropy_time_to_orekit_date as _astropy_time_to_orekit_date,
    is_orekit_absolute_date as _is_orekit_absolute_date_shared,
    normalize_time_to_epoch_seconds as _normalize_time_to_epoch_seconds,
    orekit_date_to_astropy_time as _orekit_date_to_astropy_time_shared,
)

from ._orbit_propagation_bridge import (
    get_orbit_propagation_bridge_class,
)


# Lazy-initialized Java bindings
_RUNTIME_BOUND = False
_JavaOrbitPropagationBridge = None
FramesFactory = None
PositionAngleType = None
TimeScalesFactory = None
AbsoluteDate = None
IERSConventions = None
OneAxisEllipsoid = None

# Lazily built WGS84 ellipsoid in ITRF/IERS2010/simpleEOP
_WGS84_ELLIPSOID_CACHE = None

def _bind_java() -> None:
    """Bind Orekit/bridge classes lazily after starting the JVM.

    This avoids JVM startup as an import side-effect and keeps CLI/tools that
    merely import this module lightweight.
    """

    global _RUNTIME_BOUND
    global _JavaOrbitPropagationBridge
    global FramesFactory, PositionAngleType, TimeScalesFactory
    global AbsoluteDate, IERSConventions, OneAxisEllipsoid

    if _RUNTIME_BOUND:
        return

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.bodies import OneAxisEllipsoid as _OneAxisEllipsoid
    from org.orekit.frames import FramesFactory as _FramesFactory
    from org.orekit.orbits import PositionAngleType as _PositionAngleType
    from org.orekit.time import AbsoluteDate as _AbsoluteDate
    from org.orekit.time import TimeScalesFactory as _TimeScalesFactory
    from org.orekit.utils import IERSConventions as _IERSConventions

    FramesFactory = _FramesFactory
    PositionAngleType = _PositionAngleType
    TimeScalesFactory = _TimeScalesFactory
    AbsoluteDate = _AbsoluteDate
    IERSConventions = _IERSConventions
    OneAxisEllipsoid = _OneAxisEllipsoid

    _JavaOrbitPropagationBridge = get_orbit_propagation_bridge_class()
    _RUNTIME_BOUND = True


def _iers_default():
    """Return the default Earth orientation convention (IERS 2010)."""

    _bind_java()
    return IERSConventions.IERS_2010


def _get_wgs84_ellipsoid():
    """Return module-cached WGS84 ellipsoid in ITRF(IERS2010, simpleEOP=True)."""

    global _WGS84_ELLIPSOID_CACHE
    if _WGS84_ELLIPSOID_CACHE is not None:
        return _WGS84_ELLIPSOID_CACHE

    _bind_java()
    a_m = 6378137.0
    b_m = 6356752.314245
    f = (a_m - b_m) / a_m
    _WGS84_ELLIPSOID_CACHE = OneAxisEllipsoid(
        a_m,
        f,
        FramesFactory.getITRF(IERSConventions.IERS_2010, True),
    )
    return _WGS84_ELLIPSOID_CACHE


def __getattr__(name: str):
    """Expose lazily materialized module attributes.

    Supported dynamic attribute:
    - ``WGS84_ELLIPSOID``
    """

    if name == "WGS84_ELLIPSOID":
        return _get_wgs84_ellipsoid()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def astropy_time_to_orekit_date(time: Time):
    """Convert scalar ``astropy.time.Time`` (UTC) to Orekit ``AbsoluteDate``."""
    _bind_java()
    return _astropy_time_to_orekit_date(
        time,
        absolute_date_cls=AbsoluteDate,
        time_scales_factory=TimeScalesFactory,
    )


def _orekit_date_to_astropy_time(date) -> Time:
    """Convert Orekit ``AbsoluteDate`` to scalar UTC ``astropy.time.Time``."""

    _bind_java()
    return _orekit_date_to_astropy_time_shared(
        date,
        time_scales_factory=TimeScalesFactory,
    )


def _normalize_frame_name(name: str) -> str:
    """Normalize frame-like strings to a compact comparison key."""

    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _is_orekit_absolute_date(obj: Any) -> bool:
    """Return True when ``obj`` is an Orekit ``AbsoluteDate`` instance."""

    _bind_java()
    return _is_orekit_absolute_date_shared(obj, absolute_date_cls=AbsoluteDate)


def _resolve_named_frame(name: str, *, iers, simple_eop: bool):
    """Resolve a supported frame name string into an Orekit ``Frame`` object."""

    _bind_java()
    key = _normalize_frame_name(name)

    if key == "gcrf":
        return FramesFactory.getGCRF()
    if key == "icrf":
        return FramesFactory.getICRF()
    if key in ("eme2000", "j2000"):
        return FramesFactory.getEME2000()
    if key == "teme":
        return FramesFactory.getTEME()
    if key == "mod":
        return FramesFactory.getMOD(iers)
    if key == "tod":
        return FramesFactory.getTOD(iers, bool(simple_eop))
    if key == "cirf":
        return FramesFactory.getCIRF(iers, bool(simple_eop))
    if key in ("veis", "veis1950", "veis50"):
        return FramesFactory.getVeis1950()
    if key == "ecliptic":
        return FramesFactory.getEcliptic(iers)
    if key in ("itrf", "ecef"):
        return FramesFactory.getITRF(iers, bool(simple_eop))

    raise ValueError(f"Unsupported frame string: '{name}'")


def _normalize_time_input(time_like: Any, epoch: Time) -> tuple[np.ndarray, bool]:
    """Normalize time-like input to seconds-from-epoch."""

    _bind_java()
    return _normalize_time_to_epoch_seconds(
        time_like,
        epoch,
        absolute_date_cls=AbsoluteDate,
        astropy_to_orekit=astropy_time_to_orekit_date,
        time_scales_factory=TimeScalesFactory,
    )


def _reshape_xyz(flat: np.ndarray, is_scalar: bool):
    """Reshape flattened xyz triplets to ``(N, 3)`` or ``(3,)`` for scalar input."""

    arr = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    return arr[0] if is_scalar else arr


def _reshape_1d(arr: np.ndarray, is_scalar: bool):
    """Return 1D output as scalar float when scalar query was requested."""

    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    return float(out[0]) if is_scalar else out


def _reshape_quat(flat: np.ndarray, is_scalar: bool):
    """Reshape flattened quaternion output to ``(N, 4)`` or ``(4,)``."""

    out = np.asarray(flat, dtype=np.float64).reshape(-1, 4)
    return out[0] if is_scalar else out


def _coerce_iers(iers_convention):
    """Return provided IERS convention or module default when ``None``."""

    _bind_java()
    return _iers_default() if iers_convention is None else iers_convention


def _coerce_position_angle_type(anomaly_type):
    """Normalize anomaly type input to Orekit ``PositionAngleType`` enum."""

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

    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("semi-major axis 'a' must be finite and > 0")
    if not np.isfinite(e) or e < 0.0:
        raise ValueError("eccentricity 'e' must be finite and >= 0")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and > 0")


def _resolve_solar_activity_strength(level: str):
    """Map user-friendly solar activity label to Orekit enum."""

    _bind_java()
    from org.orekit.models.earth.atmosphere.data import (
        MarshallSolarActivityFutureEstimation,  # type: ignore
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
    """Resolve a third-body name to Orekit celestial body object."""

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


def _build_earth_shape(itrf):
    """Create a WGS84 Earth shape in the requested Earth-fixed frame."""

    _bind_java()
    from org.orekit.bodies import OneAxisEllipsoid  # type: ignore
    from org.orekit.utils import Constants  # type: ignore

    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _build_numerical_propagator(
    *,
    initial_orbit,
    initial_state,
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


def _is_attitude_provider_instance(obj: Any) -> bool:
    """Return True if ``obj`` is an Orekit ``AttitudeProvider`` instance."""

    _bind_java()
    from org.orekit.attitudes import AttitudeProvider  # type: ignore

    try:
        return bool(AttitudeProvider.class_.isInstance(obj))
    except Exception:
        return False


def _resolve_lof_type(name: str):
    """Resolve LOF type name to Orekit ``LOFType`` enum."""

    _bind_java()
    from org.orekit.frames import LOFType  # type: ignore

    key = _normalize_frame_name(name)
    mapping = {
        "lvlh": LOFType.LVLH_CCSDS,
        "lvlhlegacy": LOFType.LVLH,
        "lvlhccsds": LOFType.LVLH_CCSDS,
        "vvlh": LOFType.VVLH,
        "tnw": LOFType.TNW,
        "ntw": LOFType.NTW,
        "qsw": LOFType.QSW,
        "vnc": LOFType.VNC,
        "eqw": LOFType.EQW,
        "enu": LOFType.ENU,
        "ned": LOFType.NED,
    }
    if key not in mapping:
        raise ValueError(
            "Unsupported LOF type. Use one of: "
            "'lvlh_ccsds', 'lvlh', 'lvlh_legacy', 'vvlh', "
            "'tnw', 'ntw', 'qsw', 'vnc', 'eqw', 'enu', 'ned'"
        )
    return mapping[key]


def _coerce_attitude_provider(
    attitude: Any,
    *,
    inertial_frame,
    iers,
    simple_eop: bool,
):
    """Resolve a user-facing attitude spec into an Orekit ``AttitudeProvider``.

    Supported inputs
    ----------------
    - ``None`` / ``"default"`` / ``"lvlh"`` / ``"lvlh_ccsds"``
      -> ``LofOffset(inertial_frame, LOFType.LVLH_CCSDS)``
    - LOF string names -> ``LofOffset(inertial_frame, <LOFType>)``
    - ``"nadir"`` -> ``NadirPointing(inertial_frame, WGS84 ellipsoid)``
    - ``"body_center"`` -> ``BodyCenterPointing(inertial_frame, WGS84 ellipsoid)``
    - dict forms:
      ``{"provider": <AttitudeProvider>}``
      ``{"type": "lof", "lof": "tnw"}``
      ``{"type": "nadir"}``
      ``{"type": "body_center"}``
    - direct Orekit ``AttitudeProvider`` object
    - callable ``(inertial_frame, iers, simple_eop) -> AttitudeProvider``

    Returns
    -------
    org.orekit.attitudes.AttitudeProvider
        Provider instance that can be attached to a propagator.
    """

    _bind_java()
    from org.orekit.attitudes import (  # type: ignore
        BodyCenterPointing,
        LofOffset,
        NadirPointing,
    )
    from org.orekit.frames import LOFType  # type: ignore

    def _default_provider():
        return LofOffset(inertial_frame, LOFType.LVLH_CCSDS)

    if attitude is None:
        return _default_provider()

    if _is_attitude_provider_instance(attitude):
        return attitude

    if callable(attitude):
        candidate = attitude(inertial_frame, iers, bool(simple_eop))
        if not _is_attitude_provider_instance(candidate):
            raise TypeError("attitude callable must return an Orekit AttitudeProvider")
        return candidate

    if isinstance(attitude, str):
        key = _normalize_frame_name(attitude)
        if key in ("default", "lvlhccsds", "lvlhccsdsoffset"):
            return _default_provider()
        if key in ("nadir", "nadirpointing"):
            itrf = FramesFactory.getITRF(iers, bool(simple_eop))
            return NadirPointing(inertial_frame, _build_earth_shape(itrf))
        if key in ("bodycenter", "bodycenterpointing"):
            itrf = FramesFactory.getITRF(iers, bool(simple_eop))
            return BodyCenterPointing(inertial_frame, _build_earth_shape(itrf))
        return LofOffset(inertial_frame, _resolve_lof_type(key))

    if isinstance(attitude, Mapping):
        if "provider" in attitude:
            provider = attitude["provider"]
            if not _is_attitude_provider_instance(provider):
                raise TypeError("'provider' must be an Orekit AttitudeProvider")
            return provider

        kind = _normalize_frame_name(str(attitude.get("type", "lof")))
        if kind in ("default", "lvlh", "lvlhccsds", "lof", "lofoffset"):
            lof_name = attitude.get("lof", attitude.get("lof_type", "lvlh_ccsds"))
            return LofOffset(inertial_frame, _resolve_lof_type(str(lof_name)))
        if kind in ("nadir", "nadirpointing"):
            itrf = FramesFactory.getITRF(iers, bool(simple_eop))
            return NadirPointing(inertial_frame, _build_earth_shape(itrf))
        if kind in ("bodycenter", "bodycenterpointing"):
            itrf = FramesFactory.getITRF(iers, bool(simple_eop))
            return BodyCenterPointing(inertial_frame, _build_earth_shape(itrf))
        raise ValueError("Unsupported attitude mapping type")

    raise TypeError(
        "attitude must be None, str, mapping, callable, or an Orekit AttitudeProvider"
    )


def _configure_numerical_force_models(
    *,
    propagator,
    itrf,
    utc,
    iers,
    simple_eop: bool,
    mu: float,
    ae: float,
    earth_shape,
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
    """Attach selected high-fidelity force models to a numerical propagator.

    Notes
    -----
    Harmonic gravity is always configured via
    ``HolmesFeatherstoneAttractionModel`` using ``gravity_degree`` and
    ``gravity_order``. Other models are optional toggles.
    """

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
    from org.orekit.time import TimeScalesFactory  # type: ignore
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


class OrbitCreationMixin:
    """Factory constructors for creating Java-backed :class:`Orbit` objects."""

    @classmethod
    def from_spacecraft_state(
        cls,
        state,
        iers_convention=None,
        simple_eop: bool = True,
        attitude: Any = None,
    ) -> "Orbit":
        """Construct an :class:`Orbit` from an existing ``SpacecraftState``.

        Parameters
        ----------
        state : org.orekit.propagation.SpacecraftState
            Initial state used to seed a Java-side propagator.
        iers_convention : org.orekit.utils.IERSConventions, optional
            Earth orientation convention used for Earth-fixed frame resolution.
            Defaults to ``IERS_2010`` when omitted.
        simple_eop : bool, default True
            Whether to use simple EOP mode when resolving Earth-fixed frames.
        attitude : optional
            Attitude law spec applied immediately after construction.
            See :meth:`Orbit.set_attitude_law`.

        Returns
        -------
        Orbit
            New Java-backed orbit wrapper.
        """

        _bind_java()
        iers = _coerce_iers(iers_convention)
        bridge = _JavaOrbitPropagationBridge.fromSpacecraftState(state)
        orbit = cls._from_bridge(
            bridge,
            iers,
            bool(simple_eop),
        )
        orbit.set_attitude_law(attitude)
        return orbit

    @classmethod
    def from_kepler_two_body(
        cls,
        epoch: Time,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type=None,
        mass: float = 1000.0,
        inertial_frame=None,
        iers_convention=None,
        simple_eop: bool = True,
        attitude: Any = None,
    ) -> "Orbit":
        """Build a two-body analytical orbit using ``KeplerianPropagator``.

        Parameters
        ----------
        epoch : astropy.time.Time
            Initial orbit epoch (scalar UTC-compatible time).
        a, e, i, raan, argp, anomaly : float
            Keplerian elements in SI/radian units:
            semi-major axis [m], eccentricity [-], inclination [rad],
            right ascension of ascending node [rad], argument of perigee [rad],
            and anomaly [rad].
        anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
            Type of the supplied anomaly. Defaults to ``"mean"``.
        mass : float, default 1000.0
            Spacecraft mass in kilograms.
        inertial_frame : Frame | str | None, optional
            Propagation frame. Must be pseudo-inertial. If ``None``, uses GCRF.
            String aliases are accepted (for example ``"gcrf"``, ``"eme2000"``).
        iers_convention : IERSConventions, optional
            Convention used for Earth-fixed frame resolution.
        simple_eop : bool, default True
            Whether to use simple EOP mode for Earth-fixed frames.
        attitude : optional
            Attitude law spec applied immediately after construction.
            See :meth:`Orbit.set_attitude_law`.

        Returns
        -------
        Orbit
            New analytical two-body orbit wrapper.
        """

        _bind_java()
        _validate_kepler(float(a), float(e), float(mass))

        iers = _coerce_iers(iers_convention)
        if inertial_frame is None:
            inertial_frame = FramesFactory.getGCRF()
        elif isinstance(inertial_frame, str):
            inertial_frame = _resolve_named_frame(
                inertial_frame,
                iers=iers,
                simple_eop=bool(simple_eop),
            )
        if not bool(inertial_frame.isPseudoInertial()):
            raise ValueError("inertial_frame must be pseudo-inertial")

        bridge = _JavaOrbitPropagationBridge.fromKeplerTwoBody(
            astropy_time_to_orekit_date(epoch),
            float(a),
            float(e),
            float(i),
            float(raan),
            float(argp),
            float(anomaly),
            _coerce_position_angle_type(anomaly_type),
            float(mass),
            inertial_frame,
        )
        orbit = cls._from_bridge(
            bridge,
            iers,
            bool(simple_eop),
        )
        orbit.set_attitude_law(attitude)
        return orbit

    @classmethod
    def from_kepler_numerical(
        cls,
        epoch: Time,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type=None,
        mass: float = 1000.0,
        inertial_frame=None,
        iers_convention=None,
        simple_eop: bool = True,
        attitude: Any = None,
        *,
        mu: float | None = None,
        position_tolerance_m: float = 0.1,
        min_step_s: float = 1.0e-3,
        max_step_s: float = 180.0,
        initial_step_s: float = 20.0,
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
    ) -> "Orbit":
        """Build a high-fidelity numerical orbit from Keplerian elements.

        Uses Orekit ``NumericalPropagator`` with ``DormandPrince853Integrator``
        and optional force-model toggles for harmonic gravity (degree/order),
        third-body attraction, tides, drag, SRP, and relativistic terms.

        Parameters
        ----------
        epoch : astropy.time.Time
            Initial orbit epoch (scalar UTC-compatible time).
        a, e, i, raan, argp, anomaly : float
            Keplerian elements in SI/radian units.
        anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
            Type of supplied anomaly. Defaults to ``"mean"``.
        mass : float, default 1000.0
            Spacecraft mass in kilograms.
        inertial_frame : Frame | str | None, optional
            Propagation frame. Must be pseudo-inertial.
        iers_convention : IERSConventions, optional
            Convention used when resolving Earth-fixed frames.
        simple_eop : bool, default True
            Whether Earth-fixed frame resolution should use simple EOP mode.
        attitude : optional
            Attitude law spec. See :meth:`Orbit.set_attitude_law`.
        mu : float | None, optional
            Gravitational parameter [m^3/s^2]. Defaults to WGS84 Earth ``mu``.
        position_tolerance_m : float, default 0.1
            Position tolerance for Orekit integrator tolerance construction.
        min_step_s, max_step_s, initial_step_s : float
            DP853 integration step controls [s].
        gravity_degree, gravity_order : int, default 20
            Spherical harmonic degree/order for Earth gravity.
        enable_drag, enable_third_body, enable_solid_tides, enable_ocean_tides,
        enable_relativity, enable_de_sitter, enable_lense_thirring, enable_srp,
        enable_erp : bool
            Toggles for optional force models.

        Returns
        -------
        Orbit
            New numerical orbit wrapper configured with selected perturbations.
        """

        _bind_java()
        _validate_kepler(float(a), float(e), float(mass))

        if float(min_step_s) <= 0.0 or float(max_step_s) <= 0.0:
            raise ValueError("min_step_s and max_step_s must be > 0")
        if float(min_step_s) >= float(max_step_s):
            raise ValueError("min_step_s must be < max_step_s")
        if int(gravity_degree) < 0 or int(gravity_order) < 0:
            raise ValueError("gravity_degree and gravity_order must be >= 0")

        iers = _coerce_iers(iers_convention)

        if inertial_frame is None:
            inertial_frame = FramesFactory.getGCRF()
        elif isinstance(inertial_frame, str):
            inertial_frame = _resolve_named_frame(
                inertial_frame,
                iers=iers,
                simple_eop=bool(simple_eop),
            )

        if not bool(inertial_frame.isPseudoInertial()):
            raise ValueError("inertial_frame must be pseudo-inertial")

        from org.orekit.orbits import KeplerianOrbit  # type: ignore
        from org.orekit.propagation import SpacecraftState  # type: ignore
        from org.orekit.time import TimeScalesFactory  # type: ignore
        from org.orekit.utils import Constants  # type: ignore

        mu_val = float(Constants.WGS84_EARTH_MU if mu is None else mu)
        ae = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
        date0 = astropy_time_to_orekit_date(epoch)

        orbit0 = KeplerianOrbit(
            float(a),
            float(e),
            float(i),
            float(argp),
            float(raan),
            float(anomaly),
            _coerce_position_angle_type(anomaly_type),
            inertial_frame,
            date0,
            mu_val,
        )
        state0 = SpacecraftState(orbit0, float(mass))

        propagator = _build_numerical_propagator(
            initial_orbit=orbit0,
            initial_state=state0,
            position_tolerance_m=float(position_tolerance_m),
            min_step_s=float(min_step_s),
            max_step_s=float(max_step_s),
            initial_step_s=float(initial_step_s),
        )

        itrf = FramesFactory.getITRF(iers, bool(simple_eop))
        utc = TimeScalesFactory.getUTC()
        earth_shape = _build_earth_shape(itrf)

        _configure_numerical_force_models(
            propagator=propagator,
            itrf=itrf,
            utc=utc,
            iers=iers,
            simple_eop=bool(simple_eop),
            mu=mu_val,
            ae=ae,
            earth_shape=earth_shape,
            gravity_degree=int(gravity_degree),
            gravity_order=int(gravity_order),
            enable_drag=bool(enable_drag),
            drag_area_m2=float(drag_area_m2),
            drag_cd=float(drag_cd),
            solar_activity_strength=solar_activity_strength,
            enable_third_body=bool(enable_third_body),
            third_bodies=tuple(third_bodies),
            enable_solid_tides=bool(enable_solid_tides),
            solid_tides_bodies=tuple(solid_tides_bodies),
            enable_ocean_tides=bool(enable_ocean_tides),
            ocean_degree=int(ocean_degree),
            ocean_order=int(ocean_order),
            enable_relativity=bool(enable_relativity),
            enable_de_sitter=bool(enable_de_sitter),
            enable_lense_thirring=bool(enable_lense_thirring),
            enable_srp=bool(enable_srp),
            srp_area_m2=float(srp_area_m2),
            srp_cr=float(srp_cr),
            srp_occult_moon=bool(srp_occult_moon),
            enable_erp=bool(enable_erp),
            erp_angular_resolution_deg=float(erp_angular_resolution_deg),
        )

        orbit = cls(
            propagator,
            iers=iers,
            simple_eop=bool(simple_eop),
        )
        orbit.set_attitude_law(attitude)
        return orbit


class Orbit(OrbitCreationMixin):
    """Thin Python wrapper over a Java ``OrekitOrbitPropagationBridge`` instance.

    Notes
    -----
    - Propagation, interpolation, frame transforms, and geodetic conversion run
      in Java for performance.
    - Query methods accept absolute times as ``astropy.Time`` or Orekit
      ``AbsoluteDate`` (scalar or 1D collections), plus seconds-from-epoch
      numeric inputs.
    - String frame names are supported in addition to Orekit ``Frame`` objects.
    """

    def __init__(
        self,
        propagator,
        iers=None,
        simple_eop: bool = True,
    ):
        """Create an ``Orbit`` from an existing Orekit ``Propagator``.

        Parameters
        ----------
        propagator : org.orekit.propagation.Propagator
            Orekit propagator instance.
        iers : org.orekit.utils.IERSConventions, optional
            Earth orientation convention for Earth-fixed frame resolution.
        simple_eop : bool, default True
            Whether Earth-fixed frame resolution should use simple EOP mode.
        """

        _bind_java()

        self.propagator = propagator
        self.iers = _coerce_iers(iers)
        self.simple_eop = bool(simple_eop)

        self._bridge = _JavaOrbitPropagationBridge.fromPropagator(propagator)
        self._native_frame = self._bridge.getNativeFrame()
        self._epoch = _orekit_date_to_astropy_time(
            self.propagator.getInitialState().getDate()
        )
        self._frame_cache: dict[str, Any] = {
            "native": self._native_frame,
        }

    @classmethod
    def _from_bridge(
        cls,
        bridge,
        iers,
        simple_eop: bool,
    ) -> "Orbit":
        """Internal constructor from a pre-built Java bridge object."""

        _bind_java()

        obj = cls.__new__(cls)
        obj._bridge = bridge
        obj.propagator = bridge.getPropagator()
        obj.iers = _coerce_iers(iers)
        obj.simple_eop = bool(simple_eop)
        obj._native_frame = bridge.getNativeFrame()
        obj._epoch = _orekit_date_to_astropy_time(
            obj.propagator.getInitialState().getDate()
        )
        obj._frame_cache = {
            "native": obj._native_frame,
        }
        return obj

    @property
    def epoch(self) -> Time:
        """Reference epoch of the propagator as scalar UTC ``astropy.Time``."""

        return self._epoch

    def coverage(self) -> tuple[float, float]:
        """Return cached ephemeris coverage as ``(t_min_s, t_max_s)``.

        The returned values are seconds relative to :attr:`epoch`.
        """

        cov = np.asarray(self._bridge.coverage(), dtype=np.float64).reshape(2)
        return float(cov[0]), float(cov[1])

    def precompute(self, t_min_s: float, t_max_s: float) -> None:
        """Precompute/expand Java ephemeris coverage over a time window.

        Parameters
        ----------
        t_min_s, t_max_s : float
            Coverage bounds in seconds from :attr:`epoch`.
        """

        self._bridge.precompute(float(t_min_s), float(t_max_s))

    def get_native_frame(self):
        """Return the native propagation frame used by the underlying propagator."""

        return self._native_frame

    def set_attitude_law(self, attitude: Any = None) -> "Orbit":
        """Set or override the propagator attitude law.

        Parameters
        ----------
        attitude : optional
            Supported forms:
            - ``None`` / ``"default"`` / ``"lvlh_ccsds"``: LVLH_CCSDS ``LofOffset``.
            - LOF strings: ``"lvlh"``, ``"lvlh_legacy"``, ``"vvlh"``, ``"tnw"``, ``"ntw"``,
              ``"qsw"``, ``"vnc"``, ``"eqw"``, ``"enu"``, ``"ned"``.
            - ``"nadir"`` or ``"body_center"``.
            - dict specs, e.g. ``{"type": "lof", "lof": "tnw"}``,
              ``{"type": "nadir"}``, ``{"provider": <AttitudeProvider>}``.
            - direct Orekit ``AttitudeProvider`` object.
            - callable ``(inertial_frame, iers, simple_eop) -> AttitudeProvider``.

        Returns
        -------
        Orbit
            ``self`` for fluent chaining.
        """

        provider = _coerce_attitude_provider(
            attitude,
            inertial_frame=self._native_frame,
            iers=self.iers,
            simple_eop=self.simple_eop,
        )
        self._bridge.setAttitudeProvider(provider)
        return self

    def _resolve_frame(self, frame: Union[Any, str, None]):
        """Resolve output frame from Frame object, name string, or ``None``.

        Supported frame strings: ``native``, ``gcrf``, ``icrf``, ``eme2000``,
        ``teme``, ``mod``, ``tod``, ``cirf``, ``veis1950``, ``ecliptic``,
        ``itrf``/``ecef``.
        """

        _bind_java()

        if frame is None:
            return self._native_frame

        if isinstance(frame, str):
            key = _normalize_frame_name(frame)
            if key in self._frame_cache:
                return self._frame_cache[key]

            if key == "native":
                out = self._native_frame
            else:
                out = _resolve_named_frame(
                    frame,
                    iers=self.iers,
                    simple_eop=self.simple_eop,
                )

            self._frame_cache[key] = out
            return out

        return frame

    def get_p_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ):
        """Return position in meters as numpy output (no units wrapper).

        The ``time`` input accepts ``astropy.Time``, Orekit ``AbsoluteDate``,
        seconds from epoch (scalar/array), or a time ``Quantity``.

        Scalar time input returns shape ``(3,)``. Vector time input returns
        shape ``(N, 3)``.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryPosition(dt_s, self._resolve_frame(frame))
        return _reshape_xyz(out, is_scalar)

    def get_v_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ):
        """Return velocity in m/s as numpy output (no units wrapper).

        ``time`` supports the same formats as :meth:`get_p_np`, including
        Orekit ``AbsoluteDate``.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryVelocity(dt_s, self._resolve_frame(frame))
        return _reshape_xyz(out, is_scalar)

    def get_a_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ):
        """Return acceleration in m/s^2 as numpy output (no units wrapper).

        ``time`` supports the same formats as :meth:`get_p_np`, including
        Orekit ``AbsoluteDate``.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryAcceleration(dt_s, self._resolve_frame(frame))
        return _reshape_xyz(out, is_scalar)

    def get_pv_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position, velocity)`` in SI units as numpy outputs.

        ``time`` supports the same formats as :meth:`get_p_np`, including
        Orekit ``AbsoluteDate``.

        Each component is shape ``(3,)`` for scalar queries or ``(N, 3)``
        for vector queries.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryPV(dt_s, self._resolve_frame(frame))
        return _reshape_xyz(out.p, is_scalar), _reshape_xyz(out.v, is_scalar)

    def get_pva_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(position, velocity, acceleration)`` in SI units as numpy outputs.

        ``time`` supports the same formats as :meth:`get_p_np`, including
        Orekit ``AbsoluteDate``.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryPVA(dt_s, self._resolve_frame(frame))
        return (
            _reshape_xyz(out.p, is_scalar),
            _reshape_xyz(out.v, is_scalar),
            _reshape_xyz(out.a, is_scalar),
        )

    def get_geodetic_np(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        ellipsoid=None,
    ) -> tuple[
        Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]
    ]:
        """Return geodetic ``(lat_deg, lon_deg, alt_m)`` as raw numeric outputs.

        Parameters
        ----------
        time : Time | AbsoluteDate | float | int | ndarray | Quantity
            Query times. Numeric values are interpreted as seconds from epoch.
            Orekit ``AbsoluteDate`` is accepted as a scalar or 1D collection.
        ellipsoid : OneAxisEllipsoid, optional
            Earth/body shape used for conversion. Defaults to cached WGS84.
        """

        if ellipsoid is None:
            ellipsoid = _get_wgs84_ellipsoid()

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryGeodetic(dt_s, ellipsoid)
        return (
            _reshape_1d(out.latDeg, is_scalar),
            _reshape_1d(out.lonDeg, is_scalar),
            _reshape_1d(out.altM, is_scalar),
        )

    def get_attitude_np(
        self, time: Union[Time, float, int, np.ndarray, u.Quantity]
    ) -> np.ndarray:
        """Return attitude quaternions as numpy array(s) ``[q0, q1, q2, q3]``.

        ``time`` supports the same formats as :meth:`get_p_np`, including
        Orekit ``AbsoluteDate``.
        """

        dt_s, is_scalar = _normalize_time_input(time, self._epoch)
        out = self._bridge.queryAttitudeQuaternion(dt_s)
        return _reshape_quat(out, is_scalar)

    def get_p(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> u.Quantity:
        """Return position as ``astropy.Quantity`` in meters."""

        return self.get_p_np(time, frame=frame) * u.m

    def get_v(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> u.Quantity:
        """Return velocity as ``astropy.Quantity`` in meters/second."""

        return self.get_v_np(time, frame=frame) * (u.m / u.s)

    def get_a(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> u.Quantity:
        """Return acceleration as ``astropy.Quantity`` in meters/second^2."""

        return self.get_a_np(time, frame=frame) * (u.m / (u.s**2))

    def get_pv(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Return ``(position, velocity)`` as ``astropy.Quantity`` objects."""

        p, v = self.get_pv_np(time, frame=frame)
        return p * u.m, v * (u.m / u.s)

    def get_pva(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Return ``(position, velocity, acceleration)`` as quantities."""

        p, v, a = self.get_pva_np(time, frame=frame)
        return p * u.m, v * (u.m / u.s), a * (u.m / (u.s**2))

    def get_geodetic(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        ellipsoid=None,
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Return geodetic ``(lat, lon, alt)`` as ``(deg, deg, m)`` quantities."""

        lat, lon, alt = self.get_geodetic_np(time, ellipsoid=ellipsoid)
        return lat * u.deg, lon * u.deg, alt * u.m

    def get_attitude(
        self, time: Union[Time, float, int, np.ndarray, u.Quantity]
    ) -> np.ndarray:
        """Return attitude quaternions as numpy array(s) ``[q0, q1, q2, q3]``."""

        return self.get_attitude_np(time)

    def plot(self, **kwargs):
        """Plot this orbit using :func:`nstk.plotting.plot_orbits`.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`nstk.plotting.orbits.plot_orbits`,
            including ``view="3d"``, ``view="2d"``, ``opacity``,
            ``line_width``, and ``marker_size``.

        Returns
        -------
        tuple
            ``(figure, axis)`` from the plotting helper.
        """

        from nstk.plotting.orbits import plot_orbits

        return plot_orbits(self, **kwargs)

    def get_state(
        self,
        time: Union[Time, float, int, np.ndarray, u.Quantity],
        frame: Union[Any, str, None] = None,
        fields: str = "pva",
        as_quantity: bool = True,
    ) -> dict[str, Any]:
        """Query one or more Cartesian state components with a single API call.

        Parameters
        ----------
        time : Time | AbsoluteDate | float | ndarray | Quantity
            Query time(s). Numeric values are seconds from orbit epoch.
            Orekit ``AbsoluteDate`` is accepted as scalar or 1D collection.
        frame : Frame | str | None, optional
            Output frame. ``None`` means native frame.
        fields : {"p", "v", "a", "pv", "pva"}, default "pva"
            Requested state components.
        as_quantity : bool, default True
            If True, values are returned as ``astropy.Quantity``.
            If False, values are raw numpy arrays/scalars in SI units.

        Returns
        -------
        dict
            Dictionary containing requested keys from ``{"p", "v", "a"}``.

        Examples
        --------
        ``orbit.get_state(t, fields="pv", as_quantity=False)`` returns
        ``{"p": ndarray, "v": ndarray}``.
        """

        key = fields.strip().lower()

        if key == "p":
            out = (
                self.get_p(time, frame=frame)
                if as_quantity
                else self.get_p_np(time, frame=frame)
            )
            return {"p": out}
        if key == "v":
            out = (
                self.get_v(time, frame=frame)
                if as_quantity
                else self.get_v_np(time, frame=frame)
            )
            return {"v": out}
        if key == "a":
            out = (
                self.get_a(time, frame=frame)
                if as_quantity
                else self.get_a_np(time, frame=frame)
            )
            return {"a": out}
        if key == "pv":
            if as_quantity:
                p, v = self.get_pv(time, frame=frame)
            else:
                p, v = self.get_pv_np(time, frame=frame)
            return {"p": p, "v": v}
        if key == "pva":
            if as_quantity:
                p, v, a = self.get_pva(time, frame=frame)
            else:
                p, v, a = self.get_pva_np(time, frame=frame)
            return {"p": p, "v": v, "a": a}

        raise ValueError("fields must be one of: 'p', 'v', 'a', 'pv', 'pva'")


if __name__ == "__main__":
    from time import perf_counter

    epoch = Time("2026-01-01T00:00:00", scale="utc")

    orbit = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )

    # Small sanity check
    ts_small = Time(
        epoch.unix + np.array([0.0, 60.0, 120.0], dtype=np.float64),
        format="unix",
        scale="utc",
    )
    state = orbit.get_state(ts_small, frame="gcrf", fields="pva", as_quantity=False)
    print("sanity p/v/a shapes:", state["p"].shape, state["v"].shape, state["a"].shape)

    # Speed micro-bench
    dt_s = np.arange(0, 14 * 86400.0, 30, dtype=np.float64)
    ts = Time(epoch.unix + dt_s, format="unix", scale="utc")

    # Warmup
    orbit.get_pv_np(dt_s[:128], frame="gcrf")
    orbit.get_geodetic_np(dt_s[:128])
    orbit.get_attitude_np(dt_s[:128])

    t0 = perf_counter()
    p_np, v_np = orbit.get_pv_np(dt_s, frame="gcrf")
    t1 = perf_counter()

    t2 = perf_counter()
    p_q, v_q = orbit.get_pv(ts, frame="gcrf")
    t3 = perf_counter()

    t4 = perf_counter()
    lat, lon, alt = orbit.get_geodetic_np(dt_s)
    t5 = perf_counter()

    t6 = perf_counter()
    q = orbit.get_attitude_np(dt_s)
    t7 = perf_counter()

    n = len(dt_s)
    print(f"get_pv_np ({n} pts): {t1 - t0:.3f} s")
    print(f"get_pv quantity ({n} pts): {t3 - t2:.3f} s")
    print(f"get_geodetic_np ({n} pts): {t5 - t4:.3f} s")
    print(f"get_attitude_np ({n} pts): {t7 - t6:.3f} s")
    print("outputs:", p_np.shape, v_np.shape, p_q.shape, v_q.shape, lat.shape, q.shape)
    # ------------------------------------------------------------------
    # Benchmark section: analytical vs numerical (DP853 + force models)
    # ------------------------------------------------------------------
    print("\n=== Benchmark Section: Two-Body vs Numerical ===")

    bench_dt_s = np.arange(0.0, 2.0 * 86400.0, 60.0, dtype=np.float64)
    bench_n = bench_dt_s.size

    common = dict(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )

    tb0 = perf_counter()
    orb_two_body = Orbit.from_kepler_two_body(**common)
    tb1 = perf_counter()

    orb_two_body.get_pv_np(bench_dt_s[:64], frame="gcrf")
    tb2 = perf_counter()
    r_tb, v_tb = orb_two_body.get_pv_np(bench_dt_s, frame="gcrf")
    tb3 = perf_counter()

    num0 = perf_counter()
    orb_num = Orbit.from_kepler_numerical(
        **common,
        gravity_degree=12,
        gravity_order=12,
        enable_third_body=True,
        third_bodies=("sun", "moon"),
        enable_srp=True,
        srp_area_m2=1.0,
        srp_cr=1.2,
        enable_drag=False,
    )
    num1 = perf_counter()

    orb_num.get_pv_np(bench_dt_s[:64], frame="gcrf")
    num2 = perf_counter()
    r_num, v_num = orb_num.get_pv_np(bench_dt_s, frame="gcrf")
    num3 = perf_counter()

    print(f"samples: {bench_n}")
    print(f"two-body build: {tb1 - tb0:.3f} s")
    print(f"two-body pv query: {tb3 - tb2:.3f} s")
    print(f"numerical build: {num1 - num0:.3f} s")
    print(f"numerical pv query: {num3 - num2:.3f} s")
    print("two-body output:", r_tb.shape, v_tb.shape)
    print("numerical output:", r_num.shape, v_num.shape)

    # Optional quick consistency indicator at epoch over shared frame.
    p0_tb, v0_tb = orb_two_body.get_pv_np(0.0, frame="gcrf")
    p0_num, v0_num = orb_num.get_pv_np(0.0, frame="gcrf")
    print("epoch |dr| (m):", float(np.linalg.norm(p0_tb - p0_num)))
    print("epoch |dv| (m/s):", float(np.linalg.norm(v0_tb - v0_num)))
