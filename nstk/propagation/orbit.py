"""High-level Python interface for the standalone Java-first Orekit orbit wrapper.

This module intentionally keeps Python-side work minimal. Heavy propagation,
ephemeris interpolation, frame transforms, geodetic conversion, attitude
queries, and attitude-kinematics queries are executed in Java via
``OrekitOrbitPropagationBridge``.

Design goals:
- Thin, ergonomic Python API for Orekit users.
- Lazy JVM startup (no VM init at module import time).
- Efficient vector queries with Java-side loops.

Attitude conventions
--------------------
Convenience constructors in this module create propagators with the Java
bridge's default VVLH attitude provider. When callers construct Orekit
propagators directly and wrap them with :class:`Orbit`, the attitude provider
is whatever the caller already configured on that propagator.

Common local-orbital-frame axis conventions used in NSTK/Orekit are:

- ``"vvlh"`` / ``"lvlh_ccsds"``: ``+Z`` points opposite position and ``+Y``
  points opposite orbital momentum.
- ``"lvlh"`` / ``"qsw"``: ``+X`` points along position and ``+Z`` points along
  orbital momentum.
- ``"tnw"``: ``+X`` points along velocity and ``+Z`` points along orbital
  momentum.
- ``"ntw"``: ``+Y`` points along velocity and ``+Z`` points along orbital
  momentum.
- ``"vnc"``: ``+X`` points along velocity and ``+Y`` points along orbital
  momentum.

For each local orbital frame, the remaining axis is the one required to close a
right-handed triad.

Common Earth-pointing providers that are not pure LOF axis conventions are:

- ``"nadir"``: Orekit ``NadirPointing`` using the WGS84 Earth shape. This
  points the spacecraft toward the sub-satellite point on the reference
  ellipsoid, so it accounts for Earth shape rather than only instantaneous LOF
  geometry.
- ``"body_center"``: Orekit ``BodyCenterPointing`` using the WGS84 Earth shape.
  This points directly toward the Earth's center, which is simpler than nadir
  pointing and differs slightly from ``"nadir"`` away from a spherical model.

Quaternion convention
---------------------
``Orbit.get_attitude_quat(..., quaternion_convention="scalar_last")`` returns
attitude quaternions in STK-style scalar-last ordering ``[q1, q2, q3, q4]``:

- ``q1``, ``q2``, and ``q3`` are the vector terms.
- ``q4`` is the scalar term.
- The quaternion represents the rotation from the attitude reference frame into
  the spacecraft/body frame.

Internally, Orekit exposes these same quaternions as ``[q0, q1, q2, q3]`` with
the scalar term first, and NSTK reorders them on output to match STK
conventions. In normal NSTK orbit usage, the attitude reference frame is
typically the orbit native propagation frame returned by
:meth:`Orbit.get_native_frame`. As with any unit quaternion representation,
``q`` and ``-q`` represent the same physical orientation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields as dataclass_fields

from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, TypeAlias, Union

import numpy as np
from astropy.time import Time

import nstk._orekit_frames as _orekit_frames
from nstk.time_utils import (
    astropy_time_to_orekit_date,
    make_times_astropy,
    normalize_time_to_epoch_seconds,
    orekit_date_to_astropy_time,
    safe_orekit_date_to_astropy_time,
)
from nstk._orekit_frames import (
    _coerce_iers,
    _coerce_iers_from_suffix,
    _iers_default,
    _latest_iers_convention,
    _latest_itrf_version,
    _normalize_frame_name,
    _resolve_iers_versioned_frame,
    _resolve_itrf_frame,
    _resolve_named_frame,
    _resolve_predefined_frame,
    _supported_iers_suffixes,
    _supported_itrf_suffixes,
)

from ._orbit_propagation_bridge import (
    get_orbit_propagation_bridge_class,
)

if TYPE_CHECKING:
    from org.orekit.propagation import Propagator as OrekitPropagator
    from org.orekit.propagation import SpacecraftState as OrekitSpacecraftState
    from org.orekit.time import AbsoluteDate as OrekitAbsoluteDate
else:
    OrekitPropagator = Any
    OrekitSpacecraftState = Any
    OrekitAbsoluteDate = Any


# Lazy-initialized Java bindings
_RUNTIME_BOUND = False
_JavaOrbitPropagationBridge = None
FramesFactory = None
ITRFVersion = None
Predefined = None
PositionAngleType = None
TimeScalesFactory = None
AbsoluteDate = None
IERSConventions = None
OneAxisEllipsoid = None

# Lazily built WGS84 ellipsoid in ITRF/IERS2010/simpleEOP
_WGS84_ELLIPSOID_CACHE = None


class SupportsFrame(Protocol):
    """Structural subset of an Orekit ``Frame`` used by this module."""

    def getName(self) -> str: ...

    def isPseudoInertial(self) -> bool: ...


class SupportsSpacecraftState(Protocol):
    """Structural subset of an Orekit ``SpacecraftState`` used by this module."""

    def getOrbit(self) -> Any: ...

    def getMass(self) -> float: ...

    def getFrame(self) -> SupportsFrame: ...


class SupportsPropagator(Protocol):
    """Structural subset of an Orekit ``Propagator`` used by this module."""

    def getInitialState(self) -> SupportsSpacecraftState: ...


FrameLike: TypeAlias = SupportsFrame | str | None
"""Accepted frame input for public APIs.

Frame parameters accept:

- an Orekit ``Frame`` object
- a supported string alias such as ``"gcrf"`` or ``"itrf2014"``
- ``None`` to use the method's documented default frame
"""

TimeLike: TypeAlias = Any
"""Accepted time input for public vectorized sampling APIs.

Time parameters accept:

- a scalar ``float`` giving seconds since :attr:`Orbit.epoch_orekit`
- an array-like of ``float`` values in seconds since :attr:`Orbit.epoch_orekit`
- an Orekit ``AbsoluteDate``
- a sequence of Orekit ``AbsoluteDate`` objects
- an ``astropy.time.Time`` scalar or vector
"""

AngleType: TypeAlias = Literal["mean", "eccentric", "true"]
LongitudeType: TypeAlias = Literal["mean", "eccentric", "true"]
QuaternionConvention: TypeAlias = Literal["scalar_first", "scalar_last"]


@dataclass(slots=True)
class SampledStates:
    """Vectorized outputs returned by :meth:`Orbit.sample`.

    Arrays are always vectorized, even when the input time is scalar. Vector
    outputs have shape ``(N, 3)``, quaternion outputs ``(N, 4)``, rotation
    matrices ``(N, 3, 3)``, and scalar outputs ``(N,)``. The Astropy time
    axis is constructed lazily on first access to :attr:`times_astropy`.
    """

    delta_times_sec: np.ndarray
    epoch_orekit: OrekitAbsoluteDate
    epoch_astropy: Optional[Time] = None
    _times_astropy_cache: Optional[Time] = field(default=None, init=False, repr=False)

    input_was_scalar: bool = False
    requested_fields: tuple[str, ...] = ()

    cartesian_frame: Optional[FrameLike] = None
    attitude_reference_frame: Optional[FrameLike] = None
    elements_frame: Optional[FrameLike] = None

    quaternion_convention: QuaternionConvention = "scalar_first"
    attitude_euler_sequence: Optional[str] = None
    attitude_euler_degrees: Optional[bool] = None
    anomaly_type: Optional[AngleType] = None
    longitude_type: Optional[LongitudeType] = None
    elements_angles_degrees: Optional[bool] = None
    geodetic_degrees: Optional[bool] = None
    ellipsoid_a_m: Optional[float] = None
    ellipsoid_b_m: Optional[float] = None

    position_m: Optional[np.ndarray] = None
    velocity_mps: Optional[np.ndarray] = None
    acceleration_mps2: Optional[np.ndarray] = None

    attitude_quat_ref_to_body: Optional[np.ndarray] = None
    attitude_matrix_ref_to_body: Optional[np.ndarray] = None
    attitude_euler_ref_to_body: Optional[np.ndarray] = None
    attitude_spin_body_rad_s: Optional[np.ndarray] = None
    attitude_accel_body_rad_s2: Optional[np.ndarray] = None

    semi_major_axis_m: Optional[np.ndarray] = None
    eccentricity: Optional[np.ndarray] = None
    inclination: Optional[np.ndarray] = None
    raan: Optional[np.ndarray] = None
    argp: Optional[np.ndarray] = None
    anomaly: Optional[np.ndarray] = None

    equinoctial_a_m: Optional[np.ndarray] = None
    equinoctial_ex: Optional[np.ndarray] = None
    equinoctial_ey: Optional[np.ndarray] = None
    equinoctial_hx: Optional[np.ndarray] = None
    equinoctial_hy: Optional[np.ndarray] = None
    equinoctial_longitude: Optional[np.ndarray] = None

    mass_kg: Optional[np.ndarray] = None

    additional: dict[str, np.ndarray] = field(default_factory=dict)
    additional_derivatives: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def times_astropy(self) -> Optional[Time]:
        """Sample epochs as an Astropy ``Time`` vector.

        Returns
        -------
        astropy.time.Time or None
            Vector of sample epochs with shape ``(N,)``. The value is derived
            lazily from :attr:`epoch_astropy` and :attr:`delta_times_sec` on
            first access and then cached. Returns ``None`` when
            :attr:`epoch_astropy` is unavailable.
        """
        if self._times_astropy_cache is None and self.epoch_astropy is not None:
            self._times_astropy_cache = make_times_astropy(
                self.epoch_astropy,
                self.delta_times_sec,
            )
        return self._times_astropy_cache

    @property
    def n(self) -> int:
        return int(self.delta_times_sec.shape[0])

    @property
    def available_fields(self) -> tuple[str, ...]:
        ignored = {
            "delta_times_sec",
            "epoch_orekit",
            "epoch_astropy",
            "_times_astropy_cache",
            "input_was_scalar",
            "requested_fields",
            "cartesian_frame",
            "attitude_reference_frame",
            "elements_frame",
            "quaternion_convention",
            "attitude_euler_sequence",
            "attitude_euler_degrees",
            "anomaly_type",
            "longitude_type",
            "elements_angles_degrees",
            "geodetic_degrees",
            "ellipsoid_a_m",
            "ellipsoid_b_m",
        }
        names: list[str] = []
        for field_info in dataclass_fields(self):
            name = field_info.name
            if name in ignored:
                continue
            value = getattr(self, name)
            if isinstance(value, dict):
                if value:
                    names.append(name)
            elif value is not None:
                names.append(name)
        return tuple(names)


def _bind_orbit_java() -> None:
    """Bind orbit-module Java bridge classes and supporting Orekit types lazily.

    This avoids JVM startup as an import side-effect and keeps CLI/tools that
    merely import this module lightweight.
    """

    global _RUNTIME_BOUND
    global _JavaOrbitPropagationBridge
    global FramesFactory, ITRFVersion, Predefined, PositionAngleType, TimeScalesFactory
    global AbsoluteDate, IERSConventions, OneAxisEllipsoid

    if _RUNTIME_BOUND:
        return

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()
    _orekit_frames._bind_java()

    from org.orekit.bodies import OneAxisEllipsoid as _OneAxisEllipsoid
    from org.orekit.orbits import PositionAngleType as _PositionAngleType
    from org.orekit.time import AbsoluteDate as _AbsoluteDate
    from org.orekit.time import TimeScalesFactory as _TimeScalesFactory

    FramesFactory = _orekit_frames.FramesFactory
    ITRFVersion = _orekit_frames.ITRFVersion
    Predefined = _orekit_frames.Predefined
    PositionAngleType = _PositionAngleType
    TimeScalesFactory = _TimeScalesFactory
    AbsoluteDate = _AbsoluteDate
    IERSConventions = _orekit_frames.IERSConventions
    OneAxisEllipsoid = _OneAxisEllipsoid

    _JavaOrbitPropagationBridge = get_orbit_propagation_bridge_class()
    _RUNTIME_BOUND = True


def _get_wgs84_ellipsoid():
    """Return module-cached WGS84 ellipsoid in ITRF(IERS2010, simpleEOP=True)."""

    global _WGS84_ELLIPSOID_CACHE
    if _WGS84_ELLIPSOID_CACHE is not None:
        return _WGS84_ELLIPSOID_CACHE

    _bind_orbit_java()
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


def _normalize_time_input(time_like: Any, epoch: Time) -> tuple[np.ndarray, bool]:
    """Normalize time-like input to seconds-from-epoch."""

    _bind_orbit_java()
    return normalize_time_to_epoch_seconds(
        time_like,
        epoch,
        bind_java=_bind_orbit_java,
        absolute_date_cls=AbsoluteDate,
        astropy_to_orekit=astropy_time_to_orekit_date,
        time_scales_factory=TimeScalesFactory,
    )


def _reshape_vectorized_xyz(flat: Any) -> np.ndarray:
    """Reshape flattened xyz triplets to canonical vectorized shape ``(N, 3)``."""

    return np.asarray(flat, dtype=np.float64).reshape(-1, 3)


def _reshape_vectorized_quat(flat: Any) -> np.ndarray:
    """Reshape flattened quaternions to canonical vectorized shape ``(N, 4)``."""

    return np.asarray(flat, dtype=np.float64).reshape(-1, 4)


def _reshape_vectorized_matrix(flat: Any) -> np.ndarray:
    """Reshape flattened rotation matrices to canonical shape ``(N, 3, 3)``."""

    return np.asarray(flat, dtype=np.float64).reshape(-1, 3, 3)


def _reshape_vectorized_scalar(values: Any) -> np.ndarray:
    """Reshape scalar outputs to canonical vectorized shape ``(N,)``."""

    return np.asarray(values, dtype=np.float64).reshape(-1)


def _reshape_additional_payload(names: Any, values: Any, widths: Any) -> dict[str, np.ndarray]:
    """Convert Java additional-state payloads into numpy arrays."""

    out: dict[str, np.ndarray] = {}
    py_names = list(names) if names is not None else []
    py_values = list(values) if values is not None else []
    py_widths = [int(width) for width in list(widths)] if widths is not None else []

    for name, flat, width in zip(py_names, py_values, py_widths, strict=False):
        arr = np.asarray(flat, dtype=np.float64)
        width = int(width)
        if width <= 0:
            continue
        if width == 1:
            out[str(name)] = arr.reshape(-1)
        else:
            out[str(name)] = arr.reshape(-1, width)
    return out


def _build_reference_ellipsoid(
    *,
    a_m: float,
    b_m: float,
    iers: Any,
    simple_eop: bool,
):
    """Create a one-axis ellipsoid in the repository's standard Earth-fixed frame."""

    _bind_orbit_java()
    f = (float(a_m) - float(b_m)) / float(a_m)
    return OneAxisEllipsoid(
        float(a_m),
        f,
        FramesFactory.getITRF(iers, bool(simple_eop)),
    )


def _coerce_position_angle_type(anomaly_type):
    """Normalize anomaly type input to Orekit ``PositionAngleType`` enum."""

    _bind_orbit_java()
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

    _bind_orbit_java()
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

    _bind_orbit_java()
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

    _bind_orbit_java()
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

    _bind_orbit_java()
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

    _bind_orbit_java()
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
        state: SupportsSpacecraftState,
        iers_convention: Any | None = None,
        simple_eop: bool = True,
        should_cache: bool = True,
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
        Returns
        -------
        Orbit
            New Java-backed orbit wrapper.
        """

        _bind_orbit_java()
        iers = _coerce_iers(iers_convention)
        bridge = _JavaOrbitPropagationBridge.fromSpacecraftState(state, bool(should_cache))
        orbit = cls._from_bridge(
            bridge,
            iers,
            bool(simple_eop),
        )
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
        anomaly_type: str | Any | None = None,
        mass: float = 1000.0,
        inertial_frame: FrameLike = None,
        iers_convention: Any | None = None,
        simple_eop: bool = True,
        should_cache: bool = True,
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
        Returns
        -------
        Orbit
            New analytical two-body orbit wrapper.
        """

        _bind_orbit_java()
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
            astropy_time_to_orekit_date(
                epoch,
                bind_java=_bind_orbit_java,
                absolute_date_cls=AbsoluteDate,
                time_scales_factory=TimeScalesFactory,
            ),
            float(a),
            float(e),
            float(i),
            float(raan),
            float(argp),
            float(anomaly),
            _coerce_position_angle_type(anomaly_type),
            float(mass),
            inertial_frame,
            bool(should_cache),
        )
        orbit = cls._from_bridge(
            bridge,
            iers,
            bool(simple_eop),
        )
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
        anomaly_type: str | Any | None = None,
        mass: float = 1000.0,
        inertial_frame: FrameLike = None,
        iers_convention: Any | None = None,
        simple_eop: bool = True,
        should_cache: bool = True,
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
        mu : float | None, optional
            Gravitational parameter [m^3/s^2]. Defaults to WGS84 Earth ``mu``.
        position_tolerance_m : float, default 0.1
            Position tolerance for Orekit integrator tolerance construction.
        min_step_s, max_step_s, initial_step_s : float
            DP853 integration step controls [s].
        gravity_degree, gravity_order : int, default 20
            Spherical harmonic degree/order for Earth gravity.
        enable_drag : bool, default False
            If True, attach an atmospheric drag model.
        drag_area_m2 : float, default 1.0
            Cross-sectional area used by the drag model [m^2].
        drag_cd : float, default 2.2
            Drag coefficient used by the drag model.
        solar_activity_strength : {"low", "average", "high"}, default "average"
            Solar-activity preset used by the drag atmosphere model.
        enable_third_body : bool, default True
            If True, attach third-body attraction for the bodies listed in
            ``third_bodies``.
        third_bodies : tuple[str, ...], default ("sun", "moon")
            Named bodies used when ``enable_third_body`` is enabled.
        enable_solid_tides : bool, default False
            If True, add solid Earth tides from ``solid_tides_bodies``.
        solid_tides_bodies : tuple[str, ...], default ("sun", "moon")
            Bodies that raise solid Earth tides when enabled.
        enable_ocean_tides : bool, default False
            If True, add ocean tide gravity corrections.
        ocean_degree, ocean_order : int, default 8
            Degree/order used by the ocean tide model.
        enable_relativity : bool, default False
            If True, add Schwarzschild relativity.
        enable_de_sitter : bool, default False
            If True, add de Sitter relativity.
        enable_lense_thirring : bool, default False
            If True, add Lense-Thirring relativity.
        enable_srp : bool, default False
            If True, add direct solar-radiation pressure.
        srp_area_m2 : float, default 1.0
            Effective SRP cross-sectional area [m^2].
        srp_cr : float, default 1.2
            Solar-radiation pressure coefficient.
        srp_occult_moon : bool, default True
            If True, include lunar occultation for SRP.
        enable_erp : bool, default False
            If True, add Earth rediffused radiation pressure.
        erp_angular_resolution_deg : float, default 1.0
            Angular resolution used by the ERP force model [deg].

        Returns
        -------
        Orbit
            New numerical orbit wrapper configured with selected perturbations.
        """

        _bind_orbit_java()
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
        date0 = astropy_time_to_orekit_date(
            epoch,
            bind_java=_bind_orbit_java,
            absolute_date_cls=AbsoluteDate,
            time_scales_factory=TimeScalesFactory,
        )

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

        orbit = cls(propagator, should_cache=bool(should_cache))
        orbit._set_frame_resolution_config(iers, bool(simple_eop))
        return orbit


class Orbit(OrbitCreationMixin):
    """High-performance vectorized wrapper around a Java Orekit propagator.

    This class keeps repeated propagation, frame transforms, attitude
    extraction, orbit-element conversion, and additional-state extraction on
    the Java side so that bulk queries return NumPy arrays with minimal
    Python-Java call overhead.

    Notes
    -----
    - Public sampling methods accept seconds since :attr:`epoch_orekit`,
      Orekit ``AbsoluteDate`` objects, or ``astropy.time.Time`` values.
    - Array-returning getters are always vectorized. A scalar time input still
      returns arrays with a leading length-1 sample dimension.
    - Caching is optional. For analytical propagators, orbit-only bulk queries
      may use a direct fast path that is faster than ephemeris caching and
      therefore does not expand :meth:`coverage`.
    """

    def __init__(
        self,
        propagator: SupportsPropagator | OrekitPropagator,
        should_cache: bool = True,
    ) -> None:
        """Wrap an Orekit propagator with the Java vectorized sampling bridge.

        Parameters
        ----------
        propagator
            Orekit propagator that defines the propagated orbit, attitude law,
            and any additional states. The propagator itself is retained and is
            exposed on :attr:`propagator`.
        should_cache
            Whether the Java bridge should build and retain cached ephemerides
            when that is beneficial. Caching primarily helps repeated queries
            that require full ``SpacecraftState`` evaluation, especially for
            numerical propagators. Analytical orbit-only sampling may use a
            faster direct path even when caching is enabled.
        """
        _bind_orbit_java()

        self.propagator: SupportsPropagator | OrekitPropagator = propagator
        self.should_cache = bool(should_cache)
        self._initialize_from_bridge(
            _JavaOrbitPropagationBridge.fromPropagator(propagator, self.should_cache)
        )

    def _initialize_from_bridge(self, bridge: Any) -> None:
        """Populate shared wrapper state from a Java bridge instance."""

        self._bridge = bridge
        self.propagator = bridge.getPropagator()
        self.should_cache = bool(bridge.isCachingEnabled())
        self._native_frame = bridge.getNativeFrame()
        self._epoch_orekit = bridge.getEpoch()
        self._epoch = orekit_date_to_astropy_time(
            self._epoch_orekit,
            bind_java=_bind_orbit_java,
            time_scales_factory=TimeScalesFactory,
        )
        self._frame_cache: dict[str, Any] = {"native": self._native_frame}
        self._frame_resolution_iers = _iers_default()
        self._frame_resolution_simple_eop = True

    @property
    def epoch_astropy(self) -> Time:
        """Reference epoch as an Astropy ``Time`` scalar.

        All scalar ``float`` time inputs are interpreted as seconds relative to
        this epoch.
        """
        return self._epoch

    @property
    def epoch_orekit(self) -> OrekitAbsoluteDate:
        """Reference epoch as an Orekit ``AbsoluteDate``."""
        return self._epoch_orekit

    @property
    def epoch(self) -> Time:
        """Reference epoch as a scalar UTC ``astropy.time.Time`` value."""
        return self._epoch

    @property
    def start_epoch_astropy(self) -> Time:
        """Earliest valid sample epoch as an Astropy ``Time`` scalar.

        For unbounded propagators, Astropy cannot represent Orekit infinity
        dates, so this property falls back to :attr:`epoch_astropy`.
        """
        return safe_orekit_date_to_astropy_time(
            self.start_epoch_orekit,
            self._epoch,
            bind_java=_bind_orbit_java,
            time_scales_factory=TimeScalesFactory,
        )

    @property
    def start_epoch_orekit(self) -> OrekitAbsoluteDate:
        """Earliest valid sample epoch as an Orekit ``AbsoluteDate``.

        Unbounded propagators return Orekit past infinity.
        """
        return self._bridge.getStartDate()

    @property
    def stop_epoch_astropy(self) -> Time:
        """Latest valid sample epoch as an Astropy ``Time`` scalar.

        For unbounded propagators, Astropy cannot represent Orekit infinity
        dates, so this property falls back to :attr:`epoch_astropy`.
        """
        return safe_orekit_date_to_astropy_time(
            self.stop_epoch_orekit,
            self._epoch,
            bind_java=_bind_orbit_java,
            time_scales_factory=TimeScalesFactory,
        )

    @property
    def stop_epoch_orekit(self) -> OrekitAbsoluteDate:
        """Latest valid sample epoch as an Orekit ``AbsoluteDate``.

        Unbounded propagators return Orekit future infinity.
        """
        return self._bridge.getStopDate()

    @property
    def native_frame(self) -> SupportsFrame:
        """Native Orekit frame in which the underlying propagator is defined."""
        return self._native_frame

    def get_native_frame(self) -> SupportsFrame:
        """Return the native Orekit propagation frame.

        Returns
        -------
        org.orekit.frames.Frame
            Native frame used internally by the wrapped propagator.
        """
        return self._native_frame

    def coverage(self) -> tuple[float, float]:
        """Return cached ephemeris coverage in seconds since :attr:`epoch_orekit`.

        Returns
        -------
        tuple[float, float]
            ``(t_min_s, t_max_s)`` describing the currently cached ephemeris
            interval in seconds relative to :attr:`epoch_orekit`.

        Notes
        -----
        - Returns ``(0.0, 0.0)`` when caching is disabled or no cached
          ephemeris has been materialized yet.
        - Analytical orbit-only sampling may bypass ephemeris generation for
          performance, so calling :meth:`sample` does not guarantee that
          :meth:`coverage` expands even when ``should_cache=True``.
        """
        cov = np.asarray(self._bridge.coverage(), dtype=np.float64).reshape(2)
        return float(cov[0]), float(cov[1])

    def precompute(self, t_min_s: float, t_max_s: float) -> None:
        """Populate cache coverage over a requested epoch-relative time span.

        Parameters
        ----------
        t_min_s
            Start of the requested cache interval in seconds since
            :attr:`epoch_orekit`.
        t_max_s
            End of the requested cache interval in seconds since
            :attr:`epoch_orekit`.

        Notes
        -----
        This is a no-op when caching is disabled. When caching is enabled, the
        Java bridge ensures that later cache-backed queries over the requested
        interval can reuse the materialized ephemeris.
        """
        self._bridge.precompute(float(t_min_s), float(t_max_s))

    def clear_cache(self) -> None:
        """Discard any cached ephemeris held by the Java bridge."""
        self._bridge.clearCache()

    @classmethod
    def _from_bridge(
        cls,
        bridge: Any,
        iers: Any | None = None,
        simple_eop: bool = True,
    ) -> "Orbit":
        """Internal constructor from a pre-built Java bridge object."""

        _bind_orbit_java()

        obj = cls.__new__(cls)
        obj._initialize_from_bridge(bridge)
        obj._set_frame_resolution_config(iers, simple_eop)
        return obj

    def _set_frame_resolution_config(
        self,
        iers: Any | None = None,
        simple_eop: bool = True,
    ) -> None:
        """Store frame-resolution defaults used by named-frame helpers."""

        self._frame_resolution_iers = _coerce_iers(iers)
        self._frame_resolution_simple_eop = bool(simple_eop)

    def _iers_for_resolution(self):
        return _coerce_iers(
            getattr(
                self,
                "_frame_resolution_iers",
                getattr(self, "iers", None),
            )
        )

    def _simple_eop_for_resolution(self) -> bool:
        return bool(
            getattr(
                self,
                "_frame_resolution_simple_eop",
                getattr(self, "simple_eop", True),
            )
        )

    def _resolve_frame(self, frame: FrameLike) -> SupportsFrame:
        """Resolve an output frame from Frame object, name string, or ``None``."""

        _bind_orbit_java()

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
                    iers=self._iers_for_resolution(),
                    simple_eop=self._simple_eop_for_resolution(),
                )

            self._frame_cache[key] = out
            return out

        return frame

    def sample(
        self,
        times: TimeLike,
        *,
        cartesian_frame: Optional[FrameLike] = None,
        position: bool = False,
        velocity: bool = False,
        acceleration: bool = False,
        attitude_reference_frame: Optional[FrameLike] = None,
        attitude: bool = False,
        attitude_quat: bool = False,
        attitude_matrix: bool = False,
        attitude_euler: bool = False,
        attitude_spin: bool = False,
        attitude_acceleration: bool = False,
        attitude_euler_sequence: str = "xyz",
        attitude_euler_degrees: bool = False,
        quaternion_convention: QuaternionConvention = "scalar_first",
        elements_frame: Optional[FrameLike] = None,
        keplerian: bool = False,
        anomaly_type: AngleType = "mean",
        equinoctial: bool = False,
        longitude_type: LongitudeType = "mean",
        elements_angles_degrees: bool = False,
        mass: bool = False,
        additional_states: Sequence[str] = (),
        additional_state_derivatives: Sequence[str] = (),
        strict: bool = True,
    ) -> SampledStates:
        """Sample one or more epochs and return structured vectorized outputs.

        Parameters
        ----------
        times
            Sample epochs. Accepted forms are:

            - scalar ``float`` or array-like ``float`` values measured in
              seconds since :attr:`epoch_orekit`
            - an Orekit ``AbsoluteDate``
            - a sequence of Orekit ``AbsoluteDate`` objects
            - an ``astropy.time.Time`` scalar or vector

        cartesian_frame
            Output frame for Cartesian translational fields. Accepts an Orekit
            ``Frame``, a supported frame name string, or ``None`` to use
            :attr:`native_frame`.
        position
            If ``True``, populate :attr:`SampledStates.position_m` with shape
            ``(N, 3)`` in meters.
        velocity
            If ``True``, populate :attr:`SampledStates.velocity_mps` with shape
            ``(N, 3)`` in meters per second.
        acceleration
            If ``True``, populate :attr:`SampledStates.acceleration_mps2` with
            shape ``(N, 3)`` in meters per second squared.

        attitude_reference_frame
            Reference frame from which attitude is expressed. Quaternion,
            matrix, and Euler outputs describe the rotation from this frame
            into the spacecraft body frame. ``None`` uses :attr:`native_frame`.
        attitude
            Convenience flag equivalent to requesting both
            ``attitude_quat=True`` and ``attitude_spin=True``.
        attitude_quat
            If ``True``, populate
            :attr:`SampledStates.attitude_quat_ref_to_body` with shape
            ``(N, 4)``.
        attitude_matrix
            If ``True``, populate
            :attr:`SampledStates.attitude_matrix_ref_to_body` with shape
            ``(N, 3, 3)``.
        attitude_euler
            If ``True``, populate
            :attr:`SampledStates.attitude_euler_ref_to_body` with shape
            ``(N, 3)`` using ``attitude_euler_sequence``.
        attitude_spin
            If ``True``, populate
            :attr:`SampledStates.attitude_spin_body_rad_s` with body-frame
            angular-rate vectors of shape ``(N, 3)`` in radians per second.
        attitude_acceleration
            If ``True``, populate
            :attr:`SampledStates.attitude_accel_body_rad_s2` with body-frame
            angular-acceleration vectors of shape ``(N, 3)`` in radians per
            second squared.
        attitude_euler_sequence
            Euler axis sequence used when ``attitude_euler=True``. The value is
            forwarded to the Java bridge in lowercase form.
        attitude_euler_degrees
            If ``True``, Euler outputs are in degrees. Otherwise they are in
            radians.
        quaternion_convention
            Quaternion ordering for quaternion outputs. ``"scalar_first"``
            returns ``(q0, q1, q2, q3)``. ``"scalar_last"`` returns
            ``(q1, q2, q3, q0)``.

        elements_frame
            Defining frame used for derived orbit-element outputs. Accepts an
            Orekit ``Frame``, a supported frame name string, or ``None`` to use
            :attr:`native_frame`. This frame must be pseudo-inertial when any
            element set is requested.
        keplerian
            If ``True``, populate classical Keplerian element outputs:
            ``semi_major_axis_m``, ``eccentricity``, ``inclination``, ``raan``,
            ``argp``, and ``anomaly``.
        anomaly_type
            Angular anomaly to return with ``keplerian=True``. Supported values
            are ``"mean"``, ``"eccentric"``, and ``"true"``.
        equinoctial
            If ``True``, populate equinoctial outputs:
            ``equinoctial_a_m``, ``equinoctial_ex``, ``equinoctial_ey``,
            ``equinoctial_hx``, ``equinoctial_hy``, and
            ``equinoctial_longitude``.
        longitude_type
            Longitude type to return with ``equinoctial=True``. Supported
            values are ``"mean"``, ``"eccentric"``, and ``"true"``.
        elements_angles_degrees
            If ``True``, orbit-element angular outputs are returned in degrees.
            Otherwise they are returned in radians.

        mass
            If ``True``, populate :attr:`SampledStates.mass_kg` with shape
            ``(N,)`` in kilograms.
        additional_states
            Names of Orekit additional states to extract. Each returned entry is
            stored in :attr:`SampledStates.additional` under the same name. A
            scalar-valued state is shaped ``(N,)`` and a width-``k`` state is
            shaped ``(N, k)``.
        additional_state_derivatives
            Names of Orekit additional-state derivatives to extract. Returned
            entries are stored in :attr:`SampledStates.additional_derivatives`
            with the same shape rules as ``additional_states``.
        strict
            If ``True``, raise when a requested field is unavailable or cannot
            be represented. If ``False``, unavailable array fields are left as
            ``None`` and unavailable additional-state entries are omitted from
            their output dictionaries.

        Returns
        -------
        SampledStates
            Structured vectorized outputs plus metadata describing the sampled
            time axis, frames used, requested fields, and angle conventions.

        Notes
        -----
        All array fields in the returned :class:`SampledStates` object remain
        vectorized even when ``times`` is a single instant.
        """

        dt_s, input_was_scalar = _normalize_time_input(times, self._epoch)
        dt_s = np.asarray(dt_s, dtype=np.float64).reshape(-1)

        if attitude:
            attitude_quat = True
            attitude_spin = True

        cartesian_requested = bool(position or velocity or acceleration)
        attitude_requested = bool(
            attitude_quat
            or attitude_matrix
            or attitude_euler
            or attitude_spin
            or attitude_acceleration
        )
        elements_requested = bool(keplerian or equinoctial)

        cartesian_target = self._resolve_frame(cartesian_frame) if cartesian_requested else None
        attitude_target = (
            self._resolve_frame(attitude_reference_frame) if attitude_requested else None
        )
        elements_target = self._resolve_frame(elements_frame) if elements_requested else None

        if elements_target is not None and not bool(elements_target.isPseudoInertial()):
            raise ValueError("elements_frame must be pseudo-inertial")

        if quaternion_convention not in ("scalar_first", "scalar_last"):
            raise ValueError("quaternion_convention must be 'scalar_first' or 'scalar_last'")

        bridge_result = self._bridge.sample(
            dt_s,
            cartesian_target,
            bool(position),
            bool(velocity),
            bool(acceleration),
            attitude_target,
            bool(attitude_quat),
            bool(attitude_matrix),
            bool(attitude_euler),
            bool(attitude_spin),
            bool(attitude_acceleration),
            str(attitude_euler_sequence).lower(),
            bool(attitude_euler_degrees),
            quaternion_convention == "scalar_last",
            elements_target,
            bool(keplerian),
            _coerce_position_angle_type(anomaly_type),
            bool(equinoctial),
            _coerce_position_angle_type(longitude_type),
            bool(elements_angles_degrees),
            bool(mass),
            tuple(str(name) for name in additional_states),
            tuple(str(name) for name in additional_state_derivatives),
            bool(strict),
        )

        requested_fields: list[str] = []
        if position:
            requested_fields.append("position_m")
        if velocity:
            requested_fields.append("velocity_mps")
        if acceleration:
            requested_fields.append("acceleration_mps2")
        if attitude_quat:
            requested_fields.append("attitude_quat_ref_to_body")
        if attitude_matrix:
            requested_fields.append("attitude_matrix_ref_to_body")
        if attitude_euler:
            requested_fields.append("attitude_euler_ref_to_body")
        if attitude_spin:
            requested_fields.append("attitude_spin_body_rad_s")
        if attitude_acceleration:
            requested_fields.append("attitude_accel_body_rad_s2")
        if keplerian:
            requested_fields.extend(
                [
                    "semi_major_axis_m",
                    "eccentricity",
                    "inclination",
                    "raan",
                    "argp",
                    "anomaly",
                ]
            )
        if equinoctial:
            requested_fields.extend(
                [
                    "equinoctial_a_m",
                    "equinoctial_ex",
                    "equinoctial_ey",
                    "equinoctial_hx",
                    "equinoctial_hy",
                    "equinoctial_longitude",
                ]
            )
        if mass:
            requested_fields.append("mass_kg")
        if additional_states:
            requested_fields.append("additional")
        if additional_state_derivatives:
            requested_fields.append("additional_derivatives")

        return SampledStates(
            delta_times_sec=dt_s,
            epoch_orekit=self._epoch_orekit,
            epoch_astropy=self._epoch,
            input_was_scalar=bool(input_was_scalar),
            requested_fields=tuple(requested_fields),
            cartesian_frame=cartesian_target,
            attitude_reference_frame=attitude_target,
            elements_frame=elements_target,
            quaternion_convention=quaternion_convention,
            attitude_euler_sequence=str(attitude_euler_sequence).lower()
            if attitude_euler
            else None,
            attitude_euler_degrees=bool(attitude_euler_degrees) if attitude_euler else None,
            anomaly_type=str(anomaly_type) if keplerian else None,
            longitude_type=str(longitude_type) if equinoctial else None,
            elements_angles_degrees=bool(elements_angles_degrees) if elements_requested else None,
            position_m=_reshape_vectorized_xyz(bridge_result.positionM)
            if position
            else None,
            velocity_mps=_reshape_vectorized_xyz(bridge_result.velocityMps)
            if velocity
            else None,
            acceleration_mps2=_reshape_vectorized_xyz(bridge_result.accelerationMps2)
            if acceleration
            else None,
            attitude_quat_ref_to_body=_reshape_vectorized_quat(
                bridge_result.attitudeQuatRefToBody
            )
            if attitude_quat
            else None,
            attitude_matrix_ref_to_body=_reshape_vectorized_matrix(
                bridge_result.attitudeMatrixRefToBody
            )
            if attitude_matrix
            else None,
            attitude_euler_ref_to_body=_reshape_vectorized_xyz(
                bridge_result.attitudeEulerRefToBody
            )
            if attitude_euler
            else None,
            attitude_spin_body_rad_s=_reshape_vectorized_xyz(bridge_result.attitudeSpinBodyRadS)
            if attitude_spin
            else None,
            attitude_accel_body_rad_s2=_reshape_vectorized_xyz(
                bridge_result.attitudeAccelBodyRadS2
            )
            if attitude_acceleration
            else None,
            semi_major_axis_m=_reshape_vectorized_scalar(bridge_result.semiMajorAxisM)
            if keplerian
            else None,
            eccentricity=_reshape_vectorized_scalar(bridge_result.eccentricity)
            if keplerian
            else None,
            inclination=_reshape_vectorized_scalar(bridge_result.inclination)
            if keplerian
            else None,
            raan=_reshape_vectorized_scalar(bridge_result.raan) if keplerian else None,
            argp=_reshape_vectorized_scalar(bridge_result.argp) if keplerian else None,
            anomaly=_reshape_vectorized_scalar(bridge_result.anomaly) if keplerian else None,
            equinoctial_a_m=_reshape_vectorized_scalar(bridge_result.equinoctialAM)
            if equinoctial
            else None,
            equinoctial_ex=_reshape_vectorized_scalar(bridge_result.equinoctialEx)
            if equinoctial
            else None,
            equinoctial_ey=_reshape_vectorized_scalar(bridge_result.equinoctialEy)
            if equinoctial
            else None,
            equinoctial_hx=_reshape_vectorized_scalar(bridge_result.equinoctialHx)
            if equinoctial
            else None,
            equinoctial_hy=_reshape_vectorized_scalar(bridge_result.equinoctialHy)
            if equinoctial
            else None,
            equinoctial_longitude=_reshape_vectorized_scalar(
                bridge_result.equinoctialLongitude
            )
            if equinoctial
            else None,
            mass_kg=_reshape_vectorized_scalar(bridge_result.massKg) if mass else None,
            additional=_reshape_additional_payload(
                bridge_result.additionalNames,
                bridge_result.additionalValues,
                bridge_result.additionalWidths,
            ),
            additional_derivatives=_reshape_additional_payload(
                bridge_result.additionalDerivativeNames,
                bridge_result.additionalDerivativeValues,
                bridge_result.additionalDerivativeWidths,
            ),
        )

    def get_position(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled position vectors.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for the returned position vectors. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Position array with shape ``(N, 3)`` in meters.
        """
        return self.sample(times, cartesian_frame=frame, position=True).position_m

    def get_velocity(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled velocity vectors.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for the returned velocity vectors. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Velocity array with shape ``(N, 3)`` in meters per second.
        """
        return self.sample(times, cartesian_frame=frame, velocity=True).velocity_mps

    def get_acceleration(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled acceleration vectors.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for the returned acceleration vectors. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Acceleration array with shape ``(N, 3)`` in meters per second
            squared.
        """
        return self.sample(times, cartesian_frame=frame, acceleration=True).acceleration_mps2

    def get_pva(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled position, velocity, and acceleration arrays.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for all returned Cartesian arrays. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple ``(position_m, velocity_mps, acceleration_mps2)`` with each
            array shaped ``(N, 3)``.
        """
        sampled = self.sample(
            times,
            cartesian_frame=frame,
            position=True,
            velocity=True,
            acceleration=True,
        )
        return sampled.position_m, sampled.velocity_mps, sampled.acceleration_mps2

    def get_pv(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sampled position and velocity arrays.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for both returned Cartesian arrays. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Tuple ``(position_m, velocity_mps)`` with each array shaped
            ``(N, 3)``.
        """
        sampled = self.sample(
            times,
            cartesian_frame=frame,
            position=True,
            velocity=True,
        )
        return sampled.position_m, sampled.velocity_mps

    def get_attitude_quat(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
        quaternion_convention: QuaternionConvention = "scalar_first",
    ) -> np.ndarray:
        """Return sampled attitude quaternions.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        reference_frame
            Reference frame from which the quaternion rotates into the
            spacecraft body frame. ``None`` uses :attr:`native_frame`.
        quaternion_convention
            Output ordering convention. ``"scalar_first"`` returns
            ``(q0, q1, q2, q3)`` and ``"scalar_last"`` returns
            ``(q1, q2, q3, q0)``.

        Returns
        -------
        numpy.ndarray
            Quaternion array with shape ``(N, 4)`` representing the rotation
            ``reference_frame -> body``.
        """
        return self.sample(
            times,
            attitude_reference_frame=reference_frame,
            attitude_quat=True,
            quaternion_convention=quaternion_convention,
        ).attitude_quat_ref_to_body

    def get_attitude_matrix(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled attitude rotation matrices.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        reference_frame
            Reference frame from which each rotation matrix maps coordinates
            into the spacecraft body frame. ``None`` uses
            :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Rotation-matrix array with shape ``(N, 3, 3)`` representing the
            rotation ``reference_frame -> body``.
        """
        return self.sample(
            times,
            attitude_reference_frame=reference_frame,
            attitude_matrix=True,
        ).attitude_matrix_ref_to_body

    def get_attitude_euler(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
        sequence: str = "xyz",
        degrees: bool = False,
    ) -> np.ndarray:
        """Return sampled attitude Euler angles.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        reference_frame
            Reference frame from which the sampled attitude rotates into the
            spacecraft body frame. ``None`` uses :attr:`native_frame`.
        sequence
            Euler axis sequence, for example ``"xyz"`` or ``"zyx"``.
        degrees
            If ``True``, return angles in degrees. Otherwise return radians.

        Returns
        -------
        numpy.ndarray
            Euler-angle array with shape ``(N, 3)`` using the requested axis
            sequence.
        """
        return self.sample(
            times,
            attitude_reference_frame=reference_frame,
            attitude_euler=True,
            attitude_euler_sequence=sequence,
            attitude_euler_degrees=degrees,
        ).attitude_euler_ref_to_body

    def get_attitude_spin(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled body angular-rate vectors.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        reference_frame
            Attitude reference frame used for the underlying attitude
            evaluation. ``None`` uses :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Body-frame angular-rate array with shape ``(N, 3)`` in radians per
            second.
        """
        return self.sample(
            times,
            attitude_reference_frame=reference_frame,
            attitude_spin=True,
        ).attitude_spin_body_rad_s

    def get_attitude_acceleration(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled body angular-acceleration vectors.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        reference_frame
            Attitude reference frame used for the underlying attitude
            evaluation. ``None`` uses :attr:`native_frame`.

        Returns
        -------
        numpy.ndarray
            Body-frame angular-acceleration array with shape ``(N, 3)`` in
            radians per second squared.
        """
        return self.sample(
            times,
            attitude_reference_frame=reference_frame,
            attitude_acceleration=True,
        ).attitude_accel_body_rad_s2

    def get_keplerian_classical(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
        anomaly_type: AngleType = "mean",
        degrees: bool = False,
    ) -> np.ndarray:
        """Return sampled classical Keplerian elements.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Pseudo-inertial defining frame used to derive the element set.
            ``None`` uses :attr:`native_frame`.
        anomaly_type
            Angular anomaly to include in the sixth output column. Supported
            values are ``"mean"``, ``"eccentric"``, and ``"true"``.
        degrees
            If ``True``, angular columns are returned in degrees. Otherwise
            they are returned in radians.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(N, 6)`` ordered as
            ``a_m, e, i, raan, argp, anomaly``.
        """
        sampled = self.sample(
            times,
            elements_frame=frame,
            keplerian=True,
            anomaly_type=anomaly_type,
            elements_angles_degrees=degrees,
        )
        return np.column_stack(
            [
                sampled.semi_major_axis_m,
                sampled.eccentricity,
                sampled.inclination,
                sampled.raan,
                sampled.argp,
                sampled.anomaly,
            ]
        )

    def get_equinoctial(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
        longitude_type: LongitudeType = "mean",
        degrees: bool = False,
    ) -> np.ndarray:
        """Return sampled equinoctial elements.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Pseudo-inertial defining frame used to derive the element set.
            ``None`` uses :attr:`native_frame`.
        longitude_type
            Longitude type to include in the sixth output column. Supported
            values are ``"mean"``, ``"eccentric"``, and ``"true"``.
        degrees
            If ``True``, the longitude column is returned in degrees.
            Otherwise it is returned in radians.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(N, 6)`` ordered as
            ``a_m, ex, ey, hx, hy, longitude``.
        """
        sampled = self.sample(
            times,
            elements_frame=frame,
            equinoctial=True,
            longitude_type=longitude_type,
            elements_angles_degrees=degrees,
        )
        return np.column_stack(
            [
                sampled.equinoctial_a_m,
                sampled.equinoctial_ex,
                sampled.equinoctial_ey,
                sampled.equinoctial_hx,
                sampled.equinoctial_hy,
                sampled.equinoctial_longitude,
            ]
        )

    def get_mass(self, times: TimeLike) -> np.ndarray:
        """Return sampled spacecraft mass.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.

        Returns
        -------
        numpy.ndarray
            Mass array with shape ``(N,)`` in kilograms.
        """
        return self.sample(times, mass=True).mass_kg

    def get_geodetic(
        self,
        times: TimeLike,
        *,
        degrees: bool = True,
        ellipsoid_a_m: float = 6378137.0,
        ellipsoid_b_m: float = 6356752.314245,
    ) -> np.ndarray:
        """Return geodetic latitude, longitude, and altitude.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        degrees
            If ``True``, latitude and longitude are returned in degrees.
            Otherwise they are returned in radians.
        ellipsoid_a_m
            Semimajor axis of the reference ellipsoid in meters.
        ellipsoid_b_m
            Semiminor axis of the reference ellipsoid in meters.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(N, 3)`` ordered as
            ``latitude, longitude, altitude_m``.

        Notes
        -----
        The geodetic transform uses an Orekit ``OneAxisEllipsoid`` built in
        the repository's configured Earth-fixed frame for the current IERS and
        EOP settings.
        """
        dt_s, _ = _normalize_time_input(times, self._epoch)
        dt_s = np.asarray(dt_s, dtype=np.float64).reshape(-1)
        ellipsoid = _build_reference_ellipsoid(
            a_m=float(ellipsoid_a_m),
            b_m=float(ellipsoid_b_m),
            iers=self._iers_for_resolution(),
            simple_eop=self._simple_eop_for_resolution(),
        )
        geodetic = self._bridge.queryGeodetic(dt_s, ellipsoid)
        lat = np.asarray(geodetic.latDeg, dtype=np.float64).reshape(-1)
        lon = np.asarray(geodetic.lonDeg, dtype=np.float64).reshape(-1)
        alt = np.asarray(geodetic.altM, dtype=np.float64).reshape(-1)
        if not bool(degrees):
            lat = np.deg2rad(lat)
            lon = np.deg2rad(lon)
        return np.column_stack([lat, lon, alt])

    def get_java_states(
        self, times: TimeLike
    ) -> OrekitSpacecraftState | list[OrekitSpacecraftState]:
        """Return raw Orekit ``SpacecraftState`` objects.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.

        Returns
        -------
        org.orekit.propagation.SpacecraftState or list[org.orekit.propagation.SpacecraftState]
            A single ``SpacecraftState`` when ``times`` is scalar, otherwise a
            Python ``list`` of states in the same order as the requested input
            epochs.

        Notes
        -----
        This is a lower-level escape hatch for advanced Orekit workflows that
        need direct access to Java state objects rather than NumPy arrays.
        """
        dt_s, input_was_scalar = _normalize_time_input(times, self._epoch)
        states = self._bridge.queryStates(np.asarray(dt_s, dtype=np.float64).reshape(-1))
        if input_was_scalar:
            return states[0]
        return list(states)

    def list_additional_states(self) -> list[str]:
        """Return available Orekit additional-state names.

        Returns
        -------
        list[str]
            Additional state names currently exposed by the wrapped propagator.
        """
        return [str(name) for name in list(self._bridge.listAdditionalDataNames())]

    def list_additional_state_derivatives(self) -> list[str]:
        """Return available Orekit additional-state-derivative names.

        Returns
        -------
        list[str]
            Additional-state derivative names currently exposed by the wrapped
            propagator.
        """
        return [str(name) for name in list(self._bridge.listAdditionalStateDerivativeNames())]

    def get_additional_state(self, times: TimeLike, name: str) -> np.ndarray:
        """Return a sampled Orekit additional state by name.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        name
            Additional state name to extract.

        Returns
        -------
        numpy.ndarray
            Sampled additional-state array. Scalar-valued states are shaped
            ``(N,)`` and width-``k`` states are shaped ``(N, k)``.
        """
        return self.sample(times, additional_states=(name,)).additional[str(name)]

    def get_additional_state_derivative(self, times: TimeLike, name: str) -> np.ndarray:
        """Return a sampled Orekit additional-state derivative by name.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        name
            Additional-state derivative name to extract.

        Returns
        -------
        numpy.ndarray
            Sampled additional-state-derivative array. Scalar-valued entries
            are shaped ``(N,)`` and width-``k`` entries are shaped ``(N, k)``.
        """
        return self.sample(
            times,
            additional_state_derivatives=(name,),
        ).additional_derivatives[str(name)]

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Plot this orbit with :func:`nstk.plotting.orbits.plot_orbits`.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to
            :func:`nstk.plotting.orbits.plot_orbits`, such as ``view``,
            ``opacity``, ``line_width``, ``marker_size``, and ``show_info``.

        Returns
        -------
        tuple[Any, Any]
            ``(figure, axis)`` produced by the plotting helper.
        """

        from nstk.plotting.orbits import plot_orbits

        return plot_orbits(self, **kwargs)


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
    pos, vel, acc = orbit.get_pva(ts_small, frame="gcrf")
    print("sanity p/v/a shapes:", pos.shape, vel.shape, acc.shape)

    # Speed micro-bench
    dt_s = np.arange(0, 14 * 86400.0, 30, dtype=np.float64)
    ts = Time(epoch.unix + dt_s, format="unix", scale="utc")

    # Warmup
    orbit.get_pv(dt_s[:128], frame="gcrf")
    orbit.get_geodetic(dt_s[:128])
    orbit.get_attitude_quat(dt_s[:128], quaternion_convention="scalar_last")

    t0 = perf_counter()
    p_np, v_np = orbit.get_pv(dt_s, frame="gcrf")
    t1 = perf_counter()

    t2 = perf_counter()
    p_t, v_t = orbit.get_pv(ts, frame="gcrf")
    t3 = perf_counter()

    t4 = perf_counter()
    geodetic = orbit.get_geodetic(dt_s)
    t5 = perf_counter()

    t6 = perf_counter()
    q = orbit.get_attitude_quat(dt_s, quaternion_convention="scalar_last")
    t7 = perf_counter()

    n = len(dt_s)
    print(f"get_pv raw ({n} pts): {t1 - t0:.3f} s")
    print(f"get_pv astropy-time ({n} pts): {t3 - t2:.3f} s")
    print(f"get_geodetic raw ({n} pts): {t5 - t4:.3f} s")
    print(f"get_attitude raw ({n} pts): {t7 - t6:.3f} s")
    print("outputs:", p_np.shape, v_np.shape, p_t.shape, v_t.shape, geodetic.shape, q.shape)
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

    orb_two_body.get_pv(bench_dt_s[:64], frame="gcrf")
    tb2 = perf_counter()
    r_tb, v_tb = orb_two_body.get_pv(bench_dt_s, frame="gcrf")
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

    orb_num.get_pv(bench_dt_s[:64], frame="gcrf")
    num2 = perf_counter()
    r_num, v_num = orb_num.get_pv(bench_dt_s, frame="gcrf")
    num3 = perf_counter()

    print(f"samples: {bench_n}")
    print(f"two-body build: {tb1 - tb0:.3f} s")
    print(f"two-body pv query: {tb3 - tb2:.3f} s")
    print(f"numerical build: {num1 - num0:.3f} s")
    print(f"numerical pv query: {num3 - num2:.3f} s")
    print("two-body output:", r_tb.shape, v_tb.shape)
    print("numerical output:", r_num.shape, v_num.shape)

    # Optional quick consistency indicator at epoch over shared frame.
    p0_tb, v0_tb = orb_two_body.get_pv(0.0, frame="gcrf")
    p0_num, v0_num = orb_num.get_pv(0.0, frame="gcrf")
    print("epoch |dr| (m):", float(np.linalg.norm(p0_tb - p0_num)))
    print("epoch |dv| (m/s):", float(np.linalg.norm(v0_tb - v0_num)))
