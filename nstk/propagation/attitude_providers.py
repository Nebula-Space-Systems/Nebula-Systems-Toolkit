"""Reusable Orekit attitude-provider builders and wrappers.

This module contains public attitude-law helpers for NSTK propagation
workflows. It includes both lightweight builders that return raw Orekit
``AttitudeProvider`` objects and Python wrapper classes around NSTK's custom
Java attitude providers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
from astropy.time import Time

from nstk._orekit_frames import _coerce_iers

from . import orbit as orbit_module
from ._attitude_provider_java import get_rate_limited_yaw_provider_class
from ._propagator_utils import _build_earth_shape, _resolve_inertial_frame

if TYPE_CHECKING:
    from org.hipparchus.geometry.euclidean.threed import Vector3D as OrekitVector3D
    from org.orekit.attitudes import AttitudeProvider as OrekitAttitudeProvider
    from org.orekit.bodies import BodyShape as OrekitBodyShape
    from org.orekit.time import AbsoluteDate as OrekitAbsoluteDate
else:
    OrekitVector3D = Any
    OrekitAttitudeProvider = Any
    OrekitBodyShape = Any
    OrekitAbsoluteDate = Any


_RUNTIME_BOUND = False
_JavaRateLimitedYawSteeringProvider = None
Vector3D = None
CelestialBodyFactory = None
AlignedAndConstrained = None
PredefinedTarget = None


def _bind_attitude_provider_java() -> None:
    """Bind Java classes used by NSTK's attitude-provider wrappers."""

    global _RUNTIME_BOUND
    global _JavaRateLimitedYawSteeringProvider, Vector3D, CelestialBodyFactory
    global AlignedAndConstrained, PredefinedTarget

    if _RUNTIME_BOUND:
        return

    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Vector3D as _Vector3D  # type: ignore
    from org.orekit.attitudes import (  # type: ignore
        AlignedAndConstrained as _AlignedAndConstrained,
        PredefinedTarget as _PredefinedTarget,
    )
    from org.orekit.bodies import CelestialBodyFactory as _CelestialBodyFactory  # type: ignore

    _JavaRateLimitedYawSteeringProvider = get_rate_limited_yaw_provider_class()
    Vector3D = _Vector3D
    CelestialBodyFactory = _CelestialBodyFactory
    AlignedAndConstrained = _AlignedAndConstrained
    PredefinedTarget = _PredefinedTarget
    _RUNTIME_BOUND = True


def _coerce_absolute_date(value: Time | OrekitAbsoluteDate) -> OrekitAbsoluteDate:
    """Normalize a scalar time input to an Orekit ``AbsoluteDate``."""

    orbit_module._bind_orbit_java()

    if isinstance(value, Time):
        if not value.isscalar:
            raise ValueError("reference_epoch must be a scalar time")
        return orbit_module.astropy_time_to_orekit_date(
            value,
            bind_java=orbit_module._bind_orbit_java,
            absolute_date_cls=orbit_module.AbsoluteDate,
            time_scales_factory=orbit_module.TimeScalesFactory,
        )
    if isinstance(value, orbit_module.AbsoluteDate):
        return value
    raise TypeError("reference_epoch must be an astropy.time.Time scalar or Orekit AbsoluteDate")


def _coerce_provider_query_date(
    value: float | Time | OrekitAbsoluteDate,
    *,
    reference_epoch: OrekitAbsoluteDate,
) -> OrekitAbsoluteDate:
    """Normalize a diagnostic query date for an attitude-provider wrapper."""

    orbit_module._bind_orbit_java()

    if isinstance(value, (float, int, np.floating, np.integer)):
        return reference_epoch.shiftedBy(float(value))
    if isinstance(value, Time):
        if not value.isscalar:
            raise ValueError("diagnostic date inputs must be scalar")
        return orbit_module.astropy_time_to_orekit_date(
            value,
            bind_java=orbit_module._bind_orbit_java,
            absolute_date_cls=orbit_module.AbsoluteDate,
            time_scales_factory=orbit_module.TimeScalesFactory,
        )
    if isinstance(value, orbit_module.AbsoluteDate):
        return value
    raise TypeError(
        "diagnostic date must be a scalar seconds offset from reference_epoch, "
        "an astropy.time.Time scalar, or an Orekit AbsoluteDate"
    )


def _coerce_vector3d(axis: OrekitVector3D | Sequence[float] | str | None) -> OrekitVector3D:
    """Normalize a body-axis specification to a unit Orekit ``Vector3D``."""

    _bind_attitude_provider_java()

    if axis is None:
        return Vector3D.PLUS_I
    if isinstance(axis, str):
        key = axis.strip().lower().replace(" ", "").replace("_", "")
        aliases = {
            "x": Vector3D.PLUS_I,
            "+x": Vector3D.PLUS_I,
            "plusx": Vector3D.PLUS_I,
            "y": Vector3D.PLUS_J,
            "+y": Vector3D.PLUS_J,
            "plusy": Vector3D.PLUS_J,
            "z": Vector3D.PLUS_K,
            "+z": Vector3D.PLUS_K,
            "plusz": Vector3D.PLUS_K,
            "-x": Vector3D.MINUS_I,
            "minusx": Vector3D.MINUS_I,
            "-y": Vector3D.MINUS_J,
            "minusy": Vector3D.MINUS_J,
            "-z": Vector3D.MINUS_K,
            "minusz": Vector3D.MINUS_K,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError("axis string must identify one of +/-x, +/-y, or +/-z")

    if hasattr(axis, "getX") and hasattr(axis, "getY") and hasattr(axis, "getZ"):
        norm = float(axis.getNorm())
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("axis must be finite and non-zero")
        return axis.normalize()

    arr = np.asarray(axis, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("axis must be a length-3 vector")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("axis must be finite and non-zero")
    arr = arr / norm
    return Vector3D(float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_yaw_steering_axis_pair(
    *,
    nadir_axis: OrekitVector3D,
    sun_axis: OrekitVector3D,
) -> tuple[OrekitVector3D, OrekitVector3D]:
    """Validate user nadir/sun body axes for yaw-steering convention mapping."""

    nadir_norm = float(nadir_axis.getNorm())
    sun_norm = float(sun_axis.getNorm())
    if not np.isfinite(nadir_norm) or nadir_norm <= 0.0:
        raise ValueError("nadir_axis must be finite and non-zero")
    if not np.isfinite(sun_norm) or sun_norm <= 0.0:
        raise ValueError("sun_axis must be finite and non-zero")

    nadir_unit = nadir_axis.normalize()
    sun_unit = sun_axis.normalize()
    cross_norm = float(nadir_unit.crossProduct(sun_unit).getNorm())
    if cross_norm <= 1.0e-15:
        raise ValueError("sun_axis must not be parallel to nadir_axis")
    if abs(float(nadir_unit.dotProduct(sun_unit))) > 1.0e-12:
        raise ValueError("sun_axis must be orthogonal to nadir_axis")

    return nadir_unit, sun_unit


def _build_axis_mapped_attitude_provider_proxy(
    base_provider: Any,
    canonical_to_body_rotation: Any,
) -> Any:
    """Wrap an Orekit provider with a fixed canonical-to-body axis mapping."""

    import jpype

    orbit_module._bind_orbit_java()
    from org.hipparchus.geometry.euclidean.threed import Vector3D as _Vector3D  # type: ignore
    from org.orekit.attitudes import Attitude as _Attitude  # type: ignore
    from org.orekit.utils import AngularCoordinates as _AngularCoordinates  # type: ignore

    axis_offset = _AngularCoordinates(canonical_to_body_rotation, _Vector3D.ZERO, _Vector3D.ZERO)
    attitude_provider_iface = jpype.JClass("org.orekit.attitudes.AttitudeProvider")

    @jpype.JImplements(attitude_provider_iface)
    class _AxisMappedAttitudeProviderProxy:
        def __init__(self, delegate: Any, offset: Any):
            self._delegate = delegate
            self._offset = offset

        @jpype.JOverride
        def getAttitude(self, *args):
            attitude = self._delegate.getAttitude(*args)
            # Keep field-attitude path delegated as-is; runtime propagators in
            # NSTK use the regular Attitude path.
            class_name = str(attitude.getClass().getName())
            if "FieldAttitude" in class_name:
                return attitude
            mapped_orientation = self._offset.addOffset(attitude.getOrientation())
            return _Attitude(
                attitude.getDate(),
                attitude.getReferenceFrame(),
                mapped_orientation,
            )

        @jpype.JOverride
        def getEventDetectors(self):
            return self._delegate.getEventDetectors()

        @jpype.JOverride
        def getFieldEventDetectors(self, field):
            return self._delegate.getFieldEventDetectors(field)

    return _AxisMappedAttitudeProviderProxy(base_provider, axis_offset)


def _coerce_attitude_provider(
    attitude_provider: Any | None,
    *,
    pv_provider: Any | None = None,
) -> Any | None:
    """Normalize user attitude-provider inputs to a raw Orekit provider object."""

    if attitude_provider is None:
        return None
    if isinstance(attitude_provider, RateLimitedYawSteeringProvider):
        return attitude_provider.to_orekit(pv_provider=pv_provider)
    to_orekit = getattr(attitude_provider, "to_orekit", None)
    if callable(to_orekit):
        try:
            return to_orekit(pv_provider=pv_provider)
        except TypeError:
            return to_orekit()
    java_obj = getattr(attitude_provider, "java", None)
    if java_obj is not None:
        return java_obj
    return attitude_provider


def build_ideal_nadir_sun_constrained_attitude_provider(
    inertial_frame: orbit_module.FrameLike = None,
    *,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    earth_shape: OrekitBodyShape | None = None,
    sun_provider: Any | None = None,
) -> OrekitAttitudeProvider:
    """Build the ideal geometric Orekit nadir-plus-Sun constrained attitude law.

    This returns Orekit ``AlignedAndConstrained`` and therefore represents the
    unrestricted geometric attitude law itself, not NSTK's rate-limited yaw
    controller. It matches the common STK-style "nadir aligned with Sun
    constraint" geometry:

    - spacecraft body ``+Z`` points to nadir
    - spacecraft body ``+X`` points along the local nadir tangent toward the
      Sun, i.e. the Sun direction projected into the plane orthogonal to nadir
    - spacecraft body ``+Y`` completes the right-handed triad

    Parameters
    ----------
    inertial_frame : Frame | str | None, optional
        Pseudo-inertial frame used by the attitude law. This should normally
        match the propagation frame of the propagator that will use the
        returned provider. ``None`` selects GCRF.
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used when building the default Earth ellipsoid.
        Ignored when ``earth_shape`` is supplied explicitly.
    simple_eop : bool, default True
        Whether the default Earth ellipsoid should use Orekit simple-EOP mode.
        Ignored when ``earth_shape`` is supplied explicitly.
    earth_shape : org.orekit.bodies.BodyShape, optional
        Earth reference shape used for the nadir target. When omitted, a WGS84
        ellipsoid in ``ITRF(iers_convention, simple_eop)`` is built.
    sun_provider : org.orekit.utils.ExtendedPositionProvider, optional
        Sun ephemeris provider used for the Sun constraint. When omitted,
        Orekit ``CelestialBodyFactory.getSun()`` is used.

    Returns
    -------
    org.orekit.attitudes.AttitudeProvider
        Orekit ``AlignedAndConstrained`` provider implementing the ideal
        nadir/Sun-constrained body-frame geometry described above.
    """

    _bind_attitude_provider_java()

    frame = _resolve_inertial_frame(
        inertial_frame,
        iers_convention=iers_convention,
        simple_eop=bool(simple_eop),
    )
    iers = _coerce_iers(iers_convention)

    resolved_earth_shape = earth_shape
    if resolved_earth_shape is None:
        itrf = orbit_module.FramesFactory.getITRF(iers, bool(simple_eop))
        resolved_earth_shape = _build_earth_shape(itrf)

    resolved_sun_provider = sun_provider
    if resolved_sun_provider is None:
        resolved_sun_provider = CelestialBodyFactory.getSun()

    return AlignedAndConstrained(
        Vector3D.PLUS_K,
        PredefinedTarget.NADIR,
        Vector3D.PLUS_I,
        PredefinedTarget.SUN,
        frame,
        resolved_sun_provider,
        resolved_earth_shape,
    )


def build_nadir_sun_constrained_attitude_provider(
    inertial_frame: orbit_module.FrameLike = None,
    *,
    iers_convention: Any | None = None,
    simple_eop: bool = True,
    earth_shape: OrekitBodyShape | None = None,
    sun_provider: Any | None = None,
) -> OrekitAttitudeProvider:
    """Build the ideal geometric nadir-plus-Sun constrained Orekit attitude law.

    This compatibility helper returns the same unrestricted geometric
    ``AlignedAndConstrained`` provider as
    :func:`build_ideal_nadir_sun_constrained_attitude_provider`. It does not
    apply NSTK's rate-limited yaw controller. For the rate-limited controller,
    use :class:`RateLimitedYawSteeringProvider`.
    """

    return build_ideal_nadir_sun_constrained_attitude_provider(
        inertial_frame,
        iers_convention=iers_convention,
        simple_eop=simple_eop,
        earth_shape=earth_shape,
        sun_provider=sun_provider,
    )


@dataclass(slots=True)
class RateLimitedYawSteeringProvider:
    """Deterministic yaw-steering attitude provider with yaw-rate and yaw-acceleration limits.

    The provider uses Orekit's ``NadirPointing`` as its base law and
    ``YawSteering`` as its ideal reference attitude. The actual commanded yaw
    tracks the ideal yaw through an internal 2-state yaw ODE integrated from a
    fixed reference epoch and initial yaw state:

    - ``psi``: actual yaw angle [rad]
    - ``omega``: actual yaw rate [rad/s]

    The commanded yaw acceleration uses PD tracking plus ideal feed-forward
    acceleration and is saturated to ``max_yaw_acceleration_rad_s2``. The yaw
    rate is saturated to ``max_yaw_rate_rad_s``. Internally, the Java
    implementation can use a deterministic fixed-grid checkpoint cache, but the
    computed yaw state still depends only on the fixed reference epoch, initial
    state, requested date, and control settings, so it remains safe for Orekit
    propagators that request attitudes out of chronological order.

    Parameters
    ----------
    inertial_frame : Frame | str | None, optional
        Pseudo-inertial frame used by the underlying Orekit ``NadirPointing``
        and ``YawSteering`` laws. This should normally match the propagator's
        inertial frame. ``None`` selects GCRF.
    earth_shape : org.orekit.bodies.BodyShape, optional
        Central-body shape used by the nadir-pointing base law. When omitted, a
        WGS84 Earth ellipsoid in ``ITRF(iers_convention, simple_eop)`` is
        created.
    sun_provider : org.orekit.utils.ExtendedPositionProvider, optional
        Sun ephemeris provider used by Orekit ``YawSteering``. When omitted,
        Orekit ``CelestialBodyFactory.getSun()`` is used.
    nadir_axis : Vector3D | sequence[float] | str | None, optional
        Spacecraft body-fixed axis that should point to nadir for this
        provider's user-facing body-axis convention. ``None`` selects body
        ``+Z``.
    sun_axis : Vector3D | sequence[float] | str | None, optional
        Spacecraft body-fixed axis that should be Sun-constrained by Orekit
        ``YawSteering`` for this provider's user-facing body-axis convention.
        ``None`` selects body ``+X``. This axis must not be parallel to
        ``nadir_axis`` and must be orthogonal to it.
    max_yaw_rate_rad_s : float
        Maximum allowed yaw-rate magnitude [rad/s].
    max_yaw_acceleration_rad_s2 : float
        Maximum allowed yaw-acceleration magnitude [rad/s^2].
    kp : float
        Proportional yaw-angle tracking gain. Units are effectively 1/s^2
        because it multiplies a yaw-angle error [rad] to produce an angular
        acceleration command [rad/s^2].
    kd : float
        Derivative yaw-rate tracking gain. Units are effectively 1/s because it
        multiplies a yaw-rate error [rad/s] to produce an angular acceleration
        command [rad/s^2].
    reference_epoch : astropy.time.Time | AbsoluteDate
        Fixed reference epoch from which the internal yaw ODE is integrated.
        This must be a scalar absolute time.
    initial_yaw_rad : float, default 0.0
        Actual yaw angle at ``reference_epoch`` [rad] relative to the base
        nadir attitude.
    initial_yaw_rate_rad_s : float, default 0.0
        Actual yaw rate at ``reference_epoch`` [rad/s].
    finite_difference_step_s : float, default 0.25
        Centered finite-difference step [s] used to derive ideal yaw-rate and
        yaw-acceleration reference terms from Orekit's ideal ``YawSteering``
        law.
    enable_cache : bool, default True
        Whether to enable deterministic fixed-grid checkpoint caching inside
        the Java provider. When enabled and the provider is bound to a global
        PV source, yaw integration no longer restarts from ``reference_epoch``
        on every query. Disabling this can substantially increase runtime for
        long spans; it is mainly useful for memory-constrained workflows and
        low-level debugging.
    cache_step_s : float, default 1.0
        Fixed checkpoint spacing [s] used by the deterministic cache lattice
        when ``enable_cache=True``. Smaller values trade memory and upfront
        checkpoint generation for faster dense sampling. For roughly 1 Hz
        attitude sampling, the default ``cache_step_s=1.0`` is usually the
        right starting point. For sparser queries, increase this value to
        reduce checkpoint buildup.
    iers_convention : org.orekit.utils.IERSConventions, optional
        IERS convention used only when ``earth_shape`` is omitted and NSTK
        needs to build a default WGS84 Earth ellipsoid.
    simple_eop : bool, default True
        Whether default Earth-fixed frame construction should use Orekit
        simple-EOP mode. Ignored when ``earth_shape`` is supplied explicitly.

    Notes
    -----
    Instances of this wrapper can be passed directly to NSTK propagator
    factories via ``attitude_provider=...``. The factories unwrap the
    underlying Java ``AttitudeProvider`` automatically.

    This class implements NSTK's deterministic rate-limited yaw controller.
    If you instead want the ideal unrestricted geometric nadir/Sun law, use
    :func:`build_ideal_nadir_sun_constrained_attitude_provider`.

    The user-facing body-axis convention is treated as right-handed by
    construction: with normalized ``nadir_axis`` and ``sun_axis``, the third
    body axis is defined from ``nadir_axis × sun_axis``.
    """

    inertial_frame: orbit_module.FrameLike = None
    earth_shape: OrekitBodyShape | None = None
    sun_provider: Any | None = None
    nadir_axis: OrekitVector3D | Sequence[float] | str | None = None
    sun_axis: OrekitVector3D | Sequence[float] | str | None = None
    max_yaw_rate_rad_s: float = 0.05
    max_yaw_acceleration_rad_s2: float = 0.01
    kp: float = 0.05
    kd: float = 0.25
    reference_epoch: Time | OrekitAbsoluteDate | None = None
    initial_yaw_rad: float = 0.0
    initial_yaw_rate_rad_s: float = 0.0
    finite_difference_step_s: float = 0.25
    enable_cache: bool = True
    cache_step_s: float = 1.0
    iers_convention: Any | None = None
    simple_eop: bool = True
    _inertial_frame: Any = field(init=False, repr=False)
    _reference_epoch: Any = field(init=False, repr=False)
    _sun_axis: Any = field(init=False, repr=False)
    _nadir_axis: Any = field(init=False, repr=False)
    _earth_shape: Any = field(init=False, repr=False)
    _sun_provider: Any = field(init=False, repr=False)
    _axis_map_rotation: Any = field(init=False, repr=False)
    _use_axis_map_proxy: bool = field(init=False, repr=False)
    _java_provider: Any = field(init=False, repr=False)
    _bound_provider_cache: dict[int, tuple[Any, Any, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and build the underlying Java provider."""

        _bind_attitude_provider_java()

        if self.reference_epoch is None:
            raise ValueError("reference_epoch is required")

        if not np.isfinite(float(self.max_yaw_rate_rad_s)) or float(self.max_yaw_rate_rad_s) < 0.0:
            raise ValueError("max_yaw_rate_rad_s must be finite and >= 0")
        if (
            not np.isfinite(float(self.max_yaw_acceleration_rad_s2))
            or float(self.max_yaw_acceleration_rad_s2) < 0.0
        ):
            raise ValueError("max_yaw_acceleration_rad_s2 must be finite and >= 0")
        if not np.isfinite(float(self.kp)) or float(self.kp) < 0.0:
            raise ValueError("kp must be finite and >= 0")
        if not np.isfinite(float(self.kd)) or float(self.kd) < 0.0:
            raise ValueError("kd must be finite and >= 0")
        if not np.isfinite(float(self.initial_yaw_rad)):
            raise ValueError("initial_yaw_rad must be finite")
        if not np.isfinite(float(self.initial_yaw_rate_rad_s)):
            raise ValueError("initial_yaw_rate_rad_s must be finite")
        if abs(float(self.initial_yaw_rate_rad_s)) > float(self.max_yaw_rate_rad_s) + 1.0e-12:
            raise ValueError("initial_yaw_rate_rad_s magnitude must not exceed max_yaw_rate_rad_s")
        if (
            not np.isfinite(float(self.finite_difference_step_s))
            or float(self.finite_difference_step_s) <= 0.0
        ):
            raise ValueError("finite_difference_step_s must be finite and > 0")
        if not np.isfinite(float(self.cache_step_s)) or float(self.cache_step_s) <= 0.0:
            raise ValueError("cache_step_s must be finite and > 0")

        self._inertial_frame = _resolve_inertial_frame(
            self.inertial_frame,
            iers_convention=self.iers_convention,
            simple_eop=bool(self.simple_eop),
        )
        self._reference_epoch = _coerce_absolute_date(self.reference_epoch)
        self._nadir_axis, self._sun_axis = _validate_yaw_steering_axis_pair(
            nadir_axis=_coerce_vector3d("z" if self.nadir_axis is None else self.nadir_axis),
            sun_axis=_coerce_vector3d("x" if self.sun_axis is None else self.sun_axis),
        )

        resolved_earth_shape = self.earth_shape
        if resolved_earth_shape is None:
            iers = _coerce_iers(self.iers_convention)
            itrf = orbit_module.FramesFactory.getITRF(iers, bool(self.simple_eop))
            resolved_earth_shape = _build_earth_shape(itrf)
        self._earth_shape = resolved_earth_shape

        resolved_sun_provider = self.sun_provider
        if resolved_sun_provider is None:
            resolved_sun_provider = CelestialBodyFactory.getSun()
        self._sun_provider = resolved_sun_provider
        self._axis_map_rotation = None
        self._use_axis_map_proxy = False

        try:
            self._java_provider = _JavaRateLimitedYawSteeringProvider(
                self._inertial_frame,
                self._earth_shape,
                self._sun_provider,
                self._sun_axis,
                self._nadir_axis,
                float(self.max_yaw_rate_rad_s),
                float(self.max_yaw_acceleration_rad_s2),
                float(self.kp),
                float(self.kd),
                self._reference_epoch,
                float(self.initial_yaw_rad),
                float(self.initial_yaw_rate_rad_s),
                float(self.finite_difference_step_s),
                bool(self.enable_cache),
                float(self.cache_step_s),
            )
        except TypeError:
            # Older prebuilt Java artifacts may not expose the nadir-axis
            # constructor overload yet.
            if (
                abs(float(self._nadir_axis.getX())) <= 1.0e-15
                and abs(float(self._nadir_axis.getY())) <= 1.0e-15
                and float(self._nadir_axis.getZ()) > 0.0
            ):
                self._java_provider = _JavaRateLimitedYawSteeringProvider(
                    self._inertial_frame,
                    self._earth_shape,
                    self._sun_provider,
                    self._sun_axis,
                    float(self.max_yaw_rate_rad_s),
                    float(self.max_yaw_acceleration_rad_s2),
                    float(self.kp),
                    float(self.kd),
                    self._reference_epoch,
                    float(self.initial_yaw_rad),
                    float(self.initial_yaw_rate_rad_s),
                    float(self.finite_difference_step_s),
                    bool(self.enable_cache),
                    float(self.cache_step_s),
                )
            else:
                from org.hipparchus.geometry.euclidean.threed import Rotation as _Rotation  # type: ignore

                body_to_canonical = _Rotation(
                    self._nadir_axis,
                    self._sun_axis,
                    Vector3D.PLUS_K,
                    Vector3D.PLUS_I,
                )
                canonical_sun_axis = body_to_canonical.applyTo(self._sun_axis)
                self._java_provider = _JavaRateLimitedYawSteeringProvider(
                    self._inertial_frame,
                    self._earth_shape,
                    self._sun_provider,
                    canonical_sun_axis,
                    float(self.max_yaw_rate_rad_s),
                    float(self.max_yaw_acceleration_rad_s2),
                    float(self.kp),
                    float(self.kd),
                    self._reference_epoch,
                    float(self.initial_yaw_rad),
                    float(self.initial_yaw_rate_rad_s),
                    float(self.finite_difference_step_s),
                    bool(self.enable_cache),
                    float(self.cache_step_s),
                )
                self._axis_map_rotation = body_to_canonical.revert()
                self._use_axis_map_proxy = True
        self._bound_provider_cache = {}

    @property
    def java(self) -> OrekitAttitudeProvider:
        """Return the underlying Java Orekit ``AttitudeProvider`` instance."""

        return self._java_provider

    @property
    def reference_epoch_orekit(self) -> OrekitAbsoluteDate:
        """Return the fixed yaw-integration reference epoch as ``AbsoluteDate``."""

        return self._reference_epoch

    def to_orekit(self, *, pv_provider: Any | None = None) -> OrekitAttitudeProvider:
        """Return the underlying Java Orekit ``AttitudeProvider``.

        Parameters
        ----------
        pv_provider : org.orekit.utils.PVCoordinatesProvider, optional
            Global PV provider to bind into the returned Java provider for
            internal fixed-epoch yaw integration. NSTK propagator builders pass
            the just-built propagator here automatically so the provider can use
            a global PV source instead of Orekit's per-call local provider.

        Returns
        -------
        org.orekit.attitudes.AttitudeProvider
            Underlying Java attitude provider, optionally rebound to
            ``pv_provider``.
        """

        if pv_provider is None:
            if not self._use_axis_map_proxy:
                return self._java_provider
            return _build_axis_mapped_attitude_provider_proxy(
                self._java_provider,
                self._axis_map_rotation,
            )
        key = id(pv_provider)
        cached = self._bound_provider_cache.get(key)
        if cached is None:
            bound_raw = self._java_provider.withPVProvider(pv_provider)
            bound = bound_raw
            if self._use_axis_map_proxy:
                bound = _build_axis_mapped_attitude_provider_proxy(
                    bound_raw,
                    self._axis_map_rotation,
                )
            self._bound_provider_cache[key] = (pv_provider, bound, bound_raw)
            return bound
        return cached[1]

    def get_actual_yaw_state(
        self,
        pv_provider: Any,
        date: float | Time | OrekitAbsoluteDate,
        *,
        frame: orbit_module.FrameLike = None,
    ) -> np.ndarray:
        """Return the tracked yaw state ``[psi, omega, alpha]`` at a requested date.

        Parameters
        ----------
        pv_provider : org.orekit.utils.PVCoordinatesProvider
            Orbit/PV provider used to evaluate the underlying Orekit base and
            ideal attitude laws.
        date : float | astropy.time.Time | AbsoluteDate
            Requested scalar date. A ``float`` is interpreted as seconds since
            :attr:`reference_epoch_orekit`; an Astropy time or Orekit
            ``AbsoluteDate`` is interpreted as an absolute date.
        frame : Frame | str | None, optional
            Reference frame in which the base and ideal attitudes are evaluated.
            ``None`` uses the wrapper's inertial frame.

        Returns
        -------
        numpy.ndarray
            Length-3 vector ``[psi, omega, alpha]`` with yaw angle [rad],
            yaw rate [rad/s], and yaw acceleration [rad/s^2].
        """

        query_date = _coerce_provider_query_date(date, reference_epoch=self._reference_epoch)
        query_frame = self._inertial_frame if frame is None else _resolve_inertial_frame(
            frame,
            iers_convention=self.iers_convention,
            simple_eop=bool(self.simple_eop),
        )
        bound_provider = self.to_orekit(pv_provider=pv_provider)
        return np.asarray(
            bound_provider.getTrackedYawState(pv_provider, query_date, query_frame).toArray(),
            dtype=np.float64,
        )

    def get_reference_yaw_state(
        self,
        pv_provider: Any,
        date: float | Time | OrekitAbsoluteDate,
        *,
        frame: orbit_module.FrameLike = None,
    ) -> np.ndarray:
        """Return the ideal reference yaw state ``[psi_ref, omega_ref, alpha_ref]``.

        Parameters
        ----------
        pv_provider : org.orekit.utils.PVCoordinatesProvider
            Orbit/PV provider used to evaluate Orekit's ideal ``YawSteering``
            law.
        date : float | astropy.time.Time | AbsoluteDate
            Requested scalar date. A ``float`` is interpreted as seconds since
            :attr:`reference_epoch_orekit`; an Astropy time or Orekit
            ``AbsoluteDate`` is interpreted as an absolute date.
        frame : Frame | str | None, optional
            Reference frame in which the base and ideal attitudes are evaluated.
            ``None`` uses the wrapper's inertial frame.

        Returns
        -------
        numpy.ndarray
            Length-3 vector ``[psi_ref, omega_ref, alpha_ref]`` with reference
            yaw angle [rad], yaw rate [rad/s], and yaw acceleration [rad/s^2].
        """

        query_date = _coerce_provider_query_date(date, reference_epoch=self._reference_epoch)
        query_frame = self._inertial_frame if frame is None else _resolve_inertial_frame(
            frame,
            iers_convention=self.iers_convention,
            simple_eop=bool(self.simple_eop),
        )
        bound_provider = self.to_orekit(pv_provider=pv_provider)
        return np.asarray(
            bound_provider.getReferenceYawState(pv_provider, query_date, query_frame).toArray(),
            dtype=np.float64,
        )

    def precompute_cache(
        self,
        end_date: float | Time | OrekitAbsoluteDate,
        pv_provider: Any | None = None,
        *,
        frame: orbit_module.FrameLike = None,
    ) -> None:
        """Precompute yaw checkpoints up to ``end_date`` for faster long runs.

        Parameters
        ----------
        end_date : float | astropy.time.Time | AbsoluteDate
            End date to precompute up to. A ``float`` is interpreted as seconds
            since :attr:`reference_epoch_orekit`; an Astropy time or Orekit
            ``AbsoluteDate`` is interpreted as an absolute date.
        pv_provider : org.orekit.utils.PVCoordinatesProvider | None, optional
            Global PV provider to use for deterministic yaw integration.
            Passing the same propagator/orbit provider that will later evaluate
            this attitude law is recommended. ``None`` precomputes for all
            currently bound providers known to this wrapper.
        frame : Frame | str | None, optional
            Reference frame used for yaw reference evaluation. ``None`` uses
            the wrapper's inertial frame.

        Notes
        -----
        This method is effective only when caching is enabled and the provider
        is bound to a global PV source via :meth:`to_orekit`/factory wiring.
        Otherwise it is a safe no-op.
        """

        query_date = _coerce_provider_query_date(end_date, reference_epoch=self._reference_epoch)
        query_frame = self._inertial_frame if frame is None else _resolve_inertial_frame(
            frame,
            iers_convention=self.iers_convention,
            simple_eop=bool(self.simple_eop),
        )
        bound_pairs: list[tuple[Any, Any]]
        if pv_provider is None:
            bound_pairs = [(entry[0], entry[2]) for entry in self._bound_provider_cache.values()]
            if not bound_pairs:
                return
        else:
            cached = self._bound_provider_cache.get(id(pv_provider))
            if cached is not None:
                bound_pairs = [(cached[0], cached[2])]
            else:
                bound_pairs = [(pv_provider, self._java_provider.withPVProvider(pv_provider))]

        for provider_for_eval, bound_provider in bound_pairs:
            precompute_to_date = getattr(bound_provider, "precomputeToDate", None)
            if callable(precompute_to_date):
                precompute_to_date(provider_for_eval, query_date, query_frame)
                continue

            # Backward-compatible fallback for older Java artifacts.
            bound_provider.getTrackedYawState(provider_for_eval, query_date, query_frame)

    @staticmethod
    def extract_relative_yaw(base_reference_to_body: Any, target_reference_to_body: Any) -> float:
        """Extract the relative yaw angle from two reference-to-body rotations.

        Parameters
        ----------
        base_reference_to_body : org.hipparchus.geometry.euclidean.threed.Rotation
            Rotation from the common reference frame into the spacecraft body
            frame for the base attitude.
        target_reference_to_body : org.hipparchus.geometry.euclidean.threed.Rotation
            Rotation from the common reference frame into the spacecraft body
            frame for the target attitude.

        Returns
        -------
        float
            Relative yaw angle [rad] about the spacecraft body ``+Z`` axis.
        """

        _bind_attitude_provider_java()
        return float(
            _JavaRateLimitedYawSteeringProvider.extractRelativeYaw(
                base_reference_to_body,
                target_reference_to_body,
            )
        )


__all__ = [
    "RateLimitedYawSteeringProvider",
    "build_ideal_nadir_sun_constrained_attitude_provider",
    "build_nadir_sun_constrained_attitude_provider",
]
