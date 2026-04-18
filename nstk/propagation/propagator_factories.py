"""Propagator builder functions and callable factories for Orekit workflows.

This module separates propagator construction from the thin :class:`Orbit`
wrapper. Users can either build a propagator directly from Keplerian elements
and then wrap it with :class:`nstk.propagation.orbit.Orbit`, or create a
callable factory object that accepts a Walkerized ``SpacecraftState`` and
returns a new Orekit propagator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from astropy.time import Time

from nstk._orekit_frames import _coerce_iers

from . import orbit as orbit_module
from .attitude_providers import RateLimitedYawSteeringProvider, _coerce_attitude_provider
from ._propagator_utils import (
    _build_earth_shape,
    _build_numerical_propagator,
    _coerce_position_angle_type,
    _configure_numerical_force_models,
    _resolve_inertial_frame,
    _validate_kepler,
)

if TYPE_CHECKING:
    from org.orekit.propagation import Propagator as OrekitPropagator
else:
    OrekitPropagator = Any


class PropagatorFactory(Protocol):
    """Callable protocol for building propagators from initial states.

    Implementations accept an Orekit ``SpacecraftState`` and return a new
    Orekit propagator configured from that state.
    """

    def __call__(self, initial_state: orbit_module.SupportsSpacecraftState) -> OrekitPropagator: ...


def _apply_attitude_provider(
    propagator: Any,
    inertial_frame: orbit_module.SupportsFrame,
    attitude_provider: Any | None,
    *,
    pv_provider: Any | None = None,
) -> None:
    """Attach an attitude provider, defaulting to VVLH when none is supplied."""

    orbit_module._bind_orbit_java()
    resolved_provider = _coerce_attitude_provider(attitude_provider, pv_provider=pv_provider)

    if resolved_provider is not None:
        propagator.setAttitudeProvider(resolved_provider)
        return

    from org.orekit.attitudes import LofOffset  # type: ignore
    from org.orekit.frames import LOFType  # type: ignore

    propagator.setAttitudeProvider(LofOffset(inertial_frame, LOFType.VVLH))


def _build_eckstein_hechler_propagator(
    initial_state: orbit_module.SupportsSpacecraftState,
) -> OrekitPropagator:
    """Create an ``EcksteinHechlerPropagator`` from ``initial_state``."""

    orbit_module._bind_orbit_java()

    from org.orekit.forces.gravity.potential import GravityFieldFactory  # type: ignore
    from org.orekit.propagation.analytical import EcksteinHechlerPropagator  # type: ignore

    inertial_frame = initial_state.getOrbit().getFrame()
    if not bool(inertial_frame.isPseudoInertial()):
        raise ValueError("initial_state orbit frame must be pseudo-inertial")

    provider = GravityFieldFactory.getUnnormalizedProvider(6, 0)
    harmonics = provider.onDate(initial_state.getDate())
    propagator = EcksteinHechlerPropagator(
        initial_state.getOrbit(),
        float(initial_state.getMass()),
        float(provider.getAe()),
        float(provider.getMu()),
        float(harmonics.getUnnormalizedCnm(2, 0)),
        float(harmonics.getUnnormalizedCnm(3, 0)),
        float(harmonics.getUnnormalizedCnm(4, 0)),
        0.0,
        0.0,
    )
    propagator.resetInitialState(initial_state)
    return propagator


def _build_keplerian_state(
    *,
    epoch: Time,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: str | Any | None = None,
    mass: float = 1000.0,
    inertial_frame: orbit_module.FrameLike = None,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    mu: float | None = None,
) -> orbit_module.SupportsSpacecraftState:
    """Build an Orekit ``SpacecraftState`` from Keplerian elements."""

    orbit_module._bind_orbit_java()
    _validate_kepler(float(a), float(e), float(mass))

    from org.orekit.orbits import KeplerianOrbit  # type: ignore
    from org.orekit.propagation import SpacecraftState  # type: ignore
    from org.orekit.utils import Constants  # type: ignore

    frame = _resolve_inertial_frame(
        inertial_frame,
        iers_convention=iers_convention,
        simple_eop=bool(simple_eop),
    )
    mu_value = float(Constants.WGS84_EARTH_MU if mu is None else mu)
    date0 = orbit_module.astropy_time_to_orekit_date(
        epoch,
        bind_java=orbit_module._bind_orbit_java,
        absolute_date_cls=orbit_module.AbsoluteDate,
        time_scales_factory=orbit_module.TimeScalesFactory,
    )

    orbit0 = KeplerianOrbit(
        float(a),
        float(e),
        float(i),
        float(argp),
        float(raan),
        float(anomaly),
        _coerce_position_angle_type(anomaly_type),
        frame,
        date0,
        mu_value,
    )
    return SpacecraftState(orbit0, float(mass))


@dataclass(slots=True)
class TwoBodyPropagatorFactory:
    """Build Keplerian two-body propagators from ``SpacecraftState`` objects.

    This is the state-based construction path for analytical two-body
    propagation. It is useful for Walker generation and other workflows that
    start from an already-prepared Orekit ``SpacecraftState``.

    Parameters
    ----------
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to each built propagator. When omitted, the
        factory applies a default VVLH attitude provider in the state's orbit
        frame. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.
    """

    attitude_provider: Any | None = None

    def __call__(self, initial_state: orbit_module.SupportsSpacecraftState) -> OrekitPropagator:
        """Create a ``KeplerianPropagator`` seeded from ``initial_state``."""

        orbit_module._bind_orbit_java()

        bridge = orbit_module._JavaOrbitPropagationBridge.fromSpacecraftState(
            initial_state,
            False,
        )
        propagator = bridge.getPropagator()

        pv_provider = None
        if isinstance(self.attitude_provider, RateLimitedYawSteeringProvider):
            pv_provider = orbit_module._JavaOrbitPropagationBridge.fromSpacecraftState(
                initial_state,
                False,
            ).getPropagator()
        _apply_attitude_provider(
            propagator,
            initial_state.getFrame(),
            self.attitude_provider,
            pv_provider=pv_provider,
        )

        return propagator


@dataclass(slots=True)
class J2J3J4PropagatorFactory:
    """Build fast zonal analytical propagators using J2, J3, and J4 only.

    This factory uses Orekit ``EcksteinHechlerPropagator`` with an
    coefficient-based constructor seeded from an unnormalized gravity provider.
    Only the zonal ``C20``, ``C30``, and ``C40`` terms are retained; ``C50``
    and ``C60`` are set to zero explicitly. It is intended for fast
    Earth-orbit propagation when the J2/J3/J4 zonal terms are a reasonable
    approximation.

    Parameters
    ----------
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to each built propagator. When omitted, the
        factory applies a default VVLH attitude provider in the state's orbit
        frame. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.
    """

    attitude_provider: Any | None = None

    def __call__(self, initial_state: orbit_module.SupportsSpacecraftState) -> OrekitPropagator:
        """Create an ``EcksteinHechlerPropagator`` from ``initial_state``."""

        inertial_frame = initial_state.getOrbit().getFrame()
        propagator = _build_eckstein_hechler_propagator(initial_state)
        pv_provider = None
        if isinstance(self.attitude_provider, RateLimitedYawSteeringProvider):
            pv_provider = _build_eckstein_hechler_propagator(initial_state)
        _apply_attitude_provider(
            propagator,
            inertial_frame,
            self.attitude_provider,
            pv_provider=pv_provider,
        )
        return propagator


@dataclass(slots=True)
class NumericalPropagatorFactory:
    """Build DP853-based numerical propagators from ``SpacecraftState`` objects.

    This is the state-based construction path for NSTK's high-fidelity
    numerical propagator configuration. It preserves the old ability to start
    from an existing Orekit ``SpacecraftState`` without putting that builder
    logic on :class:`nstk.propagation.orbit.Orbit`.

    Parameters
    ----------
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used to resolve Earth-fixed force-model frames.
        Defaults to NSTK's propagation convention when omitted.
    simple_eop : bool, default True
        Whether Earth-fixed frame resolution should use Orekit simple-EOP mode.
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to each built propagator. When omitted, the
        factory applies a default VVLH attitude provider in the state's orbit
        frame. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.
    mu : float | None, optional
        Central-body gravitational parameter in m^3/s^2. When omitted, the
        initial state's orbit ``mu`` value is reused.
    position_tolerance_m : float, default 0.1
        Position tolerance used to derive DP853 integrator tolerances.
    min_step_s, max_step_s, initial_step_s : float
        DP853 integration step controls in seconds.
    gravity_degree, gravity_order : int, default 20
        Spherical-harmonic degree and order for the Holmes-Featherstone Earth
        gravity model.
    enable_drag : bool, default False
        Whether to attach atmospheric drag.
    drag_area_m2 : float, default 1.0
        Drag reference area in square meters.
    drag_cd : float, default 2.2
        Drag coefficient.
    solar_activity_strength : {"weak", "average", "strong"}, default "average"
        Solar-activity preset for the drag atmosphere model.
    enable_third_body : bool, default True
        Whether to add third-body attraction for ``third_bodies``.
    third_bodies : tuple[str, ...], default ("sun", "moon")
        Third bodies to include when third-body attraction is enabled.
    enable_solid_tides : bool, default False
        Whether to add solid Earth tides from ``solid_tides_bodies``.
    solid_tides_bodies : tuple[str, ...], default ("sun", "moon")
        Bodies that raise solid Earth tides when enabled.
    enable_ocean_tides : bool, default False
        Whether to add ocean tides.
    ocean_degree, ocean_order : int, default 8
        Ocean-tide model degree and order.
    enable_relativity : bool, default False
        Whether to add Schwarzschild relativity.
    enable_de_sitter : bool, default False
        Whether to add de Sitter relativity.
    enable_lense_thirring : bool, default False
        Whether to add Lense-Thirring relativity.
    enable_srp : bool, default False
        Whether to add direct solar-radiation pressure.
    srp_area_m2 : float, default 1.0
        Solar-radiation pressure area in square meters.
    srp_cr : float, default 1.2
        Solar-radiation pressure coefficient.
    srp_occult_moon : bool, default True
        Whether SRP should include lunar occultation.
    enable_erp : bool, default False
        Whether to add Earth rediffused radiation pressure.
    erp_angular_resolution_deg : float, default 1.0
        ERP angular resolution in degrees.
    """

    iers_convention: Any | None = None
    simple_eop: bool = True
    attitude_provider: Any | None = None
    mu: float | None = None
    position_tolerance_m: float = 0.1
    min_step_s: float = 1.0e-3
    max_step_s: float = 180.0
    initial_step_s: float = 20.0
    gravity_degree: int = 20
    gravity_order: int = 20
    enable_drag: bool = False
    drag_area_m2: float = 1.0
    drag_cd: float = 2.2
    solar_activity_strength: str = "average"
    enable_third_body: bool = True
    third_bodies: tuple[str, ...] = ("sun", "moon")
    enable_solid_tides: bool = False
    solid_tides_bodies: tuple[str, ...] = ("sun", "moon")
    enable_ocean_tides: bool = False
    ocean_degree: int = 8
    ocean_order: int = 8
    enable_relativity: bool = False
    enable_de_sitter: bool = False
    enable_lense_thirring: bool = False
    enable_srp: bool = False
    srp_area_m2: float = 1.0
    srp_cr: float = 1.2
    srp_occult_moon: bool = True
    enable_erp: bool = False
    erp_angular_resolution_deg: float = 1.0

    def _build_core_propagator(
        self,
        initial_state: orbit_module.SupportsSpacecraftState,
    ) -> tuple[OrekitPropagator, orbit_module.SupportsFrame]:
        """Build the numerical propagator core without applying a custom attitude provider."""

        orbit_module._bind_orbit_java()

        from org.orekit.utils import Constants  # type: ignore

        if float(self.min_step_s) <= 0.0 or float(self.max_step_s) <= 0.0:
            raise ValueError("min_step_s and max_step_s must be > 0")
        if float(self.min_step_s) >= float(self.max_step_s):
            raise ValueError("min_step_s must be < max_step_s")
        if int(self.gravity_degree) < 0 or int(self.gravity_order) < 0:
            raise ValueError("gravity_degree and gravity_order must be >= 0")

        initial_orbit = initial_state.getOrbit()
        inertial_frame = initial_orbit.getFrame()
        if not bool(inertial_frame.isPseudoInertial()):
            raise ValueError("initial_state orbit frame must be pseudo-inertial")

        mu_value = float(initial_orbit.getMu() if self.mu is None else self.mu)
        ae = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
        iers = _coerce_iers(self.iers_convention)

        propagator = _build_numerical_propagator(
            initial_orbit=initial_orbit,
            initial_state=initial_state,
            position_tolerance_m=float(self.position_tolerance_m),
            min_step_s=float(self.min_step_s),
            max_step_s=float(self.max_step_s),
            initial_step_s=float(self.initial_step_s),
        )

        itrf = orbit_module.FramesFactory.getITRF(iers, bool(self.simple_eop))
        utc = orbit_module.TimeScalesFactory.getUTC()
        earth_shape = _build_earth_shape(itrf)

        _configure_numerical_force_models(
            propagator=propagator,
            itrf=itrf,
            utc=utc,
            iers=iers,
            simple_eop=bool(self.simple_eop),
            mu=mu_value,
            ae=ae,
            earth_shape=earth_shape,
            gravity_degree=int(self.gravity_degree),
            gravity_order=int(self.gravity_order),
            enable_drag=bool(self.enable_drag),
            drag_area_m2=float(self.drag_area_m2),
            drag_cd=float(self.drag_cd),
            solar_activity_strength=self.solar_activity_strength,
            enable_third_body=bool(self.enable_third_body),
            third_bodies=tuple(self.third_bodies),
            enable_solid_tides=bool(self.enable_solid_tides),
            solid_tides_bodies=tuple(self.solid_tides_bodies),
            enable_ocean_tides=bool(self.enable_ocean_tides),
            ocean_degree=int(self.ocean_degree),
            ocean_order=int(self.ocean_order),
            enable_relativity=bool(self.enable_relativity),
            enable_de_sitter=bool(self.enable_de_sitter),
            enable_lense_thirring=bool(self.enable_lense_thirring),
            enable_srp=bool(self.enable_srp),
            srp_area_m2=float(self.srp_area_m2),
            srp_cr=float(self.srp_cr),
            srp_occult_moon=bool(self.srp_occult_moon),
            enable_erp=bool(self.enable_erp),
            erp_angular_resolution_deg=float(self.erp_angular_resolution_deg),
        )
        return propagator, inertial_frame

    def __call__(self, initial_state: orbit_module.SupportsSpacecraftState) -> OrekitPropagator:
        """Create a configured ``NumericalPropagator`` from ``initial_state``."""

        propagator, inertial_frame = self._build_core_propagator(initial_state)
        pv_provider = None
        if isinstance(self.attitude_provider, RateLimitedYawSteeringProvider):
            pv_provider, _ = self._build_core_propagator(initial_state)
        _apply_attitude_provider(
            propagator,
            inertial_frame,
            self.attitude_provider,
            pv_provider=pv_provider,
        )
        return propagator


def build_two_body_propagator(
    epoch: Time,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: str | Any | None = None,
    mass: float = 1000.0,
    inertial_frame: orbit_module.FrameLike = None,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    attitude_provider: Any | None = None,
) -> OrekitPropagator:
    """Build a Keplerian two-body Orekit propagator from Keplerian elements.

    Parameters
    ----------
    epoch : astropy.time.Time
        Initial orbit epoch as a scalar UTC-compatible Astropy time.
    a, e, i, raan, argp, anomaly : float
        Keplerian elements in SI/radian units: semi-major axis [m],
        eccentricity [-], inclination [rad], RAAN [rad], argument of perigee
        [rad], and anomaly [rad].
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
        Type of the supplied anomaly. Defaults to mean anomaly.
    mass : float, default 1000.0
        Spacecraft mass in kilograms.
    inertial_frame : Frame | str | None, optional
        Propagation frame. Must be pseudo-inertial. ``None`` selects GCRF.
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used when resolving Earth-fixed frame strings during
        input parsing.
    simple_eop : bool, default True
        Whether Earth-fixed frame parsing should use Orekit simple-EOP mode.
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to the propagator. When omitted, a default
        VVLH provider is applied. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.

    Returns
    -------
    org.orekit.propagation.analytical.KeplerianPropagator
        Configured Orekit analytical two-body propagator.
    """

    state0 = _build_keplerian_state(
        epoch=epoch,
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        anomaly=anomaly,
        anomaly_type=anomaly_type,
        mass=mass,
        inertial_frame=inertial_frame,
        iers_convention=iers_convention,
        simple_eop=simple_eop,
    )
    return TwoBodyPropagatorFactory(attitude_provider=attitude_provider)(state0)


def build_j2_j3_j4_propagator(
    epoch: Time,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: str | Any | None = None,
    mass: float = 1000.0,
    inertial_frame: orbit_module.FrameLike = None,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    attitude_provider: Any | None = None,
) -> OrekitPropagator:
    """Build a fast J2/J3/J4 analytical Orekit propagator from Keplerian elements.

    This builder uses Orekit ``EcksteinHechlerPropagator`` with only the
    unnormalized ``C20``, ``C30``, and ``C40`` zonal coefficients retained.
    The higher ``C50`` and ``C60`` terms required by the constructor are set
    to zero.

    Parameters
    ----------
    epoch : astropy.time.Time
        Initial orbit epoch as a scalar UTC-compatible Astropy time.
    a, e, i, raan, argp, anomaly : float
        Keplerian elements in SI/radian units.
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
        Type of the supplied anomaly. Defaults to mean anomaly.
    mass : float, default 1000.0
        Spacecraft mass in kilograms.
    inertial_frame : Frame | str | None, optional
        Propagation frame. Must be pseudo-inertial. ``None`` selects GCRF.
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used when resolving Earth-fixed frame strings during
        input parsing.
    simple_eop : bool, default True
        Whether Earth-fixed frame parsing should use Orekit simple-EOP mode.
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to the propagator. When omitted, a default
        VVLH provider is applied. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.

    Returns
    -------
    org.orekit.propagation.analytical.EcksteinHechlerPropagator
        Configured Orekit analytical zonal propagator.
    """

    state0 = _build_keplerian_state(
        epoch=epoch,
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        anomaly=anomaly,
        anomaly_type=anomaly_type,
        mass=mass,
        inertial_frame=inertial_frame,
        iers_convention=iers_convention,
        simple_eop=simple_eop,
    )
    return J2J3J4PropagatorFactory(attitude_provider=attitude_provider)(state0)


def build_numerical_propagator(
    epoch: Time,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: str | Any | None = None,
    mass: float = 1000.0,
    inertial_frame: orbit_module.FrameLike = None,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    attitude_provider: Any | None = None,
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
) -> OrekitPropagator:
    """Build a high-fidelity numerical Orekit propagator from Keplerian elements.

    Parameters
    ----------
    epoch : astropy.time.Time
        Initial orbit epoch as a scalar UTC-compatible Astropy time.
    a, e, i, raan, argp, anomaly : float
        Keplerian elements in SI/radian units.
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
        Type of the supplied anomaly. Defaults to mean anomaly.
    mass : float, default 1000.0
        Spacecraft mass in kilograms.
    inertial_frame : Frame | str | None, optional
        Propagation frame. Must be pseudo-inertial. ``None`` selects GCRF.
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used when resolving Earth-fixed frames.
    simple_eop : bool, default True
        Whether Earth-fixed frame resolution should use Orekit simple-EOP mode.
    attitude_provider : org.orekit.attitudes.AttitudeProvider, optional
        Attitude provider to attach to the propagator. When omitted, a default
        VVLH provider is applied. Use
        :func:`nstk.propagation.attitude_providers.build_ideal_nadir_sun_constrained_attitude_provider`
        for the ideal STK-style nadir-aligned, Sun-constrained geometric law,
        or :class:`nstk.propagation.RateLimitedYawSteeringProvider` for the
        rate-limited controller.
    mu : float | None, optional
        Central-body gravitational parameter [m^3/s^2]. Defaults to WGS84
        Earth ``mu``.
    position_tolerance_m : float, default 0.1
        Position tolerance used to derive integrator tolerances [m].
    min_step_s, max_step_s, initial_step_s : float
        DP853 integration step controls [s].
    gravity_degree, gravity_order : int, default 20
        Spherical-harmonic degree and order for Earth gravity.
    enable_drag : bool, default False
        Whether to add atmospheric drag.
    drag_area_m2 : float, default 1.0
        Drag area [m^2].
    drag_cd : float, default 2.2
        Drag coefficient.
    solar_activity_strength : {"weak", "average", "strong"}, default "average"
        Solar-activity preset for the atmosphere model.
    enable_third_body : bool, default True
        Whether to add third-body attraction for ``third_bodies``.
    third_bodies : tuple[str, ...], default ("sun", "moon")
        Third bodies to include when enabled.
    enable_solid_tides : bool, default False
        Whether to add solid Earth tides.
    solid_tides_bodies : tuple[str, ...], default ("sun", "moon")
        Bodies that raise solid Earth tides when enabled.
    enable_ocean_tides : bool, default False
        Whether to add ocean tides.
    ocean_degree, ocean_order : int, default 8
        Ocean-tide model degree and order.
    enable_relativity : bool, default False
        Whether to add Schwarzschild relativity.
    enable_de_sitter : bool, default False
        Whether to add de Sitter relativity.
    enable_lense_thirring : bool, default False
        Whether to add Lense-Thirring relativity.
    enable_srp : bool, default False
        Whether to add solar-radiation pressure.
    srp_area_m2 : float, default 1.0
        Solar-radiation pressure area [m^2].
    srp_cr : float, default 1.2
        Solar-radiation pressure coefficient.
    srp_occult_moon : bool, default True
        Whether SRP should include lunar occultation.
    enable_erp : bool, default False
        Whether to add Earth rediffused radiation pressure.
    erp_angular_resolution_deg : float, default 1.0
        ERP angular resolution [deg].

    Returns
    -------
    org.orekit.propagation.numerical.NumericalPropagator
        Configured Orekit numerical propagator.
    """

    state0 = _build_keplerian_state(
        epoch=epoch,
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        anomaly=anomaly,
        anomaly_type=anomaly_type,
        mass=mass,
        inertial_frame=inertial_frame,
        iers_convention=iers_convention,
        simple_eop=simple_eop,
        mu=mu,
    )
    factory = NumericalPropagatorFactory(
        iers_convention=iers_convention,
        simple_eop=bool(simple_eop),
        attitude_provider=attitude_provider,
        mu=mu,
        position_tolerance_m=float(position_tolerance_m),
        min_step_s=float(min_step_s),
        max_step_s=float(max_step_s),
        initial_step_s=float(initial_step_s),
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
    return factory(state0)


__all__ = [
    "J2J3J4PropagatorFactory",
    "NumericalPropagatorFactory",
    "PropagatorFactory",
    "TwoBodyPropagatorFactory",
    "build_j2_j3_j4_propagator",
    "build_numerical_propagator",
    "build_two_body_propagator",
]
