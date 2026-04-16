from __future__ import annotations

"""High-level Python interface for vectorized Orekit state sampling.

This module defines a thin Python wrapper API intended to sit on top of a
Java-side vectorized sampling engine. The Python layer focuses on a clean,
explicit user interface, stable return shapes, and clear naming around frames,
attitude conventions, and orbit-element outputs.

The actual heavy work is expected to happen in Java:
- bulk propagation / ephemeris lookup
- bulk frame transforms
- bulk attitude extraction
- bulk additional-state extraction
- bulk geodetic conversion

The Python layer is intentionally interface-first. Most methods either delegate
through :meth:`VectorizedPropagator.sample` or raise ``NotImplementedError``
until the Java bridge is connected.

When no frame is provided, the wrapper should use the native propagation / state-defining frame for
Cartesian outputs and the native attitude reference frame for attitude outputs. When an elements frame 
is needed for orbit-element outputs, the wrapper should use the native orbit-defining frame if no explicit 
frame is provided.
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np
from astropy.time import Time

if TYPE_CHECKING:
    from org.orekit.frames import Frame as OrekitFrame
    from org.orekit.propagation import Propagator as OrekitPropagator
    from org.orekit.propagation import SpacecraftState as OrekitSpacecraftState
    from org.orekit.time import AbsoluteDate as OrekitAbsoluteDate
else:
    OrekitFrame = Any
    OrekitPropagator = Any
    OrekitSpacecraftState = Any
    OrekitAbsoluteDate = Any


TimeLike = Any
"""Accepted time input for public methods.

The wrapper is expected to support:
- ``float`` or array-like ``float``: seconds since :attr:`epoch_orekit`
- Orekit ``AbsoluteDate``
- sequence of Orekit ``AbsoluteDate``
- ``astropy.time.Time``
"""

FrameLike = Any
"""Accepted frame input for public methods.

The wrapper is expected to support Orekit ``Frame`` objects directly and may
also support selected string aliases resolved by the wrapper.
"""

AngleType = Literal["mean", "eccentric", "true"]
LongitudeType = Literal["mean", "eccentric", "true"]
QuaternionConvention = Literal["scalar_first", "scalar_last"]


@dataclass(slots=True)
class SampledStates:
    """Vectorized outputs returned by :meth:`VectorizedPropagator.sample`.

    This object stores bulk-sampled arrays and the metadata required to
    interpret them correctly.

    Shape conventions
    -----------------
    - Arrays are always vectorized, even when the input time is scalar.
    - Vector outputs use shape ``(N, 3)``.
    - Quaternion outputs use shape ``(N, 4)``.
    - Rotation-matrix outputs use shape ``(N, 3, 3)``.
    - Scalar outputs use shape ``(N,)``.
    - Geodetic outputs use shape ``(N, 3)`` ordered as
      ``latitude, longitude, altitude``.

    Frame conventions
    -----------------
    - Cartesian outputs are expressed in :attr:`cartesian_frame`.
    - Attitude quaternion / matrix / Euler outputs represent the rotation from
      :attr:`attitude_reference_frame` to the spacecraft body frame.
    - Attitude spin and attitude acceleration are expressed in the spacecraft
      body frame.
    - Orbit-element outputs are derived in :attr:`elements_frame`.

    Time conventions
    ----------------
    :attr:`delta_times_sec` is the canonical vectorized time axis and is always
    measured relative to :attr:`epoch_orekit`.
    """

    # Canonical time basis
    delta_times_sec: np.ndarray
    """Seconds since :attr:`epoch_orekit`, shape ``(N,)``."""

    epoch_orekit: OrekitAbsoluteDate | Any
    """Reference epoch as an Orekit ``AbsoluteDate``."""

    epoch_astropy: Optional[Time] = None
    """Reference epoch as an Astropy ``Time`` when available."""

    times_astropy: Optional[Time] = None
    """Sample times as Astropy ``Time`` when available."""

    input_was_scalar: bool = False
    """Whether the original ``times`` input represented a single instant."""

    requested_fields: tuple[str, ...] = ()
    """Field names explicitly requested by the caller."""

    # Frames actually used
    cartesian_frame: Optional[FrameLike] = None
    attitude_reference_frame: Optional[FrameLike] = None
    elements_frame: Optional[FrameLike] = None

    # Output metadata / conventions
    quaternion_convention: QuaternionConvention = "scalar_first"
    """Quaternion ordering convention for attitude quaternion outputs."""

    attitude_euler_sequence: Optional[str] = None
    attitude_euler_degrees: Optional[bool] = None
    anomaly_type: Optional[AngleType] = None
    longitude_type: Optional[LongitudeType] = None
    elements_angles_degrees: Optional[bool] = None
    geodetic_degrees: Optional[bool] = None
    ellipsoid_a_m: Optional[float] = None
    ellipsoid_b_m: Optional[float] = None

    # Cartesian outputs
    position_m: Optional[np.ndarray] = None
    velocity_mps: Optional[np.ndarray] = None
    acceleration_mps2: Optional[np.ndarray] = None

    # Attitude outputs
    attitude_quat_ref_to_body: Optional[np.ndarray] = None
    attitude_matrix_ref_to_body: Optional[np.ndarray] = None
    attitude_euler_ref_to_body: Optional[np.ndarray] = None
    attitude_spin_body_rad_s: Optional[np.ndarray] = None
    attitude_accel_body_rad_s2: Optional[np.ndarray] = None

    # Classical Keplerian outputs
    semi_major_axis_m: Optional[np.ndarray] = None
    eccentricity: Optional[np.ndarray] = None
    inclination: Optional[np.ndarray] = None
    raan: Optional[np.ndarray] = None
    argp: Optional[np.ndarray] = None
    anomaly: Optional[np.ndarray] = None

    # Equinoctial outputs
    equinoctial_a_m: Optional[np.ndarray] = None
    equinoctial_ex: Optional[np.ndarray] = None
    equinoctial_ey: Optional[np.ndarray] = None
    equinoctial_hx: Optional[np.ndarray] = None
    equinoctial_hy: Optional[np.ndarray] = None
    equinoctial_longitude: Optional[np.ndarray] = None

    # Scalar outputs
    mass_kg: Optional[np.ndarray] = None

    # Additional Orekit state data
    additional: dict[str, np.ndarray] = field(default_factory=dict)
    additional_derivatives: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of sampled instants."""
        return int(self.delta_times_sec.shape[0])

    @property
    def available_fields(self) -> tuple[str, ...]:
        """Names of populated output fields.

        This excludes metadata-only fields and includes ``additional`` and
        ``additional_derivatives`` only when those dictionaries are non-empty.
        """
        ignored = {
            "delta_times_sec",
            "epoch_orekit",
            "epoch_astropy",
            "times_astropy",
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


class VectorizedPropagator:
    """Vectorized state-sampling wrapper around an Orekit propagator.

    The intent of this class is to expose a Pythonic, NumPy-friendly API while
    keeping repeated propagation, frame transforms, and state extraction inside
    Java for performance.

    Notes
    -----
    - This wrapper is interface-focused. The actual vectorized implementation is
      expected to be provided by a Java-side engine bridged into Python.
    - The wrapper assumes Orekit / orekit-jpype objects are used to construct
      the underlying propagator and any attitude profile or related state logic.
    - The public API is built around explicit keyword arguments rather than
      field-name strings. This keeps the interface discoverable and easy to use.
    """

    def __init__(self, propagator: OrekitPropagator | Any, should_cache: bool = True) -> None:
        """Create a vectorized wrapper around an Orekit propagator.

        Parameters
        ----------
        propagator
            The underlying Orekit propagator or a compatible Java-side wrapper
            object used to generate sampled states.
        should_cache
            Whether the wrapper should cache internal EphemerisGenerators to optimize repeated sampling over the same propagator. 
        """
        self.propagator = propagator
        self.should_cache = should_cache

    def clear_cache(self) -> None:
        """Clear any cached vectorized sampling results.

        Implementations may cache normalized times, generated ephemerides,
        frame transforms, or previously sampled arrays.
        """
        raise NotImplementedError

    @property
    def epoch_astropy(self) -> Time:
        """Reference epoch as an Astropy ``Time`` object."""
        raise NotImplementedError

    @property
    def epoch_orekit(self) -> OrekitAbsoluteDate | Any:
        """Reference epoch as an Orekit ``AbsoluteDate``."""
        raise NotImplementedError

    @property
    def start_epoch_astropy(self) -> Time:
        """Earliest valid sample epoch as an Astropy ``Time`` object."""
        raise NotImplementedError

    @property
    def start_epoch_orekit(self) -> OrekitAbsoluteDate | Any:
        """Earliest valid sample epoch as an Orekit ``AbsoluteDate``."""
        raise NotImplementedError

    @property
    def stop_epoch_astropy(self) -> Time:
        """Latest valid sample epoch as an Astropy ``Time`` object."""
        raise NotImplementedError

    @property
    def stop_epoch_orekit(self) -> OrekitAbsoluteDate | Any:
        """Latest valid sample epoch as an Orekit ``AbsoluteDate``."""
        raise NotImplementedError

    @property
    def native_frame(self) -> FrameLike:
        """Native propagation / state-defining frame used by the wrapper."""
        raise NotImplementedError

    def sample(
        self,
        times: TimeLike,
        *,
        # Cartesian outputs
        cartesian_frame: Optional[FrameLike] = None,
        position: bool = False,
        velocity: bool = False,
        acceleration: bool = False,
        # Attitude outputs
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
        # Orbit-element outputs
        elements_frame: Optional[FrameLike] = None,
        keplerian: bool = False,
        anomaly_type: AngleType = "mean",
        equinoctial: bool = False,
        longitude_type: LongitudeType = "mean",
        elements_angles_degrees: bool = False,
        # Scalar outputs
        mass: bool = False,
        # Additional Orekit state data
        additional_states: Sequence[str] = (),
        additional_state_derivatives: Sequence[str] = (),
        # Behavior
        strict: bool = True,
    ) -> SampledStates:
        """Vectorized bulk sampler over one or more times.

        Parameters
        ----------
        times
            Supported inputs:

            - ``float`` or array-like ``float``:
              seconds since :attr:`epoch_orekit`
            - Orekit ``AbsoluteDate``
            - sequence of Orekit ``AbsoluteDate``
            - ``astropy.time.Time``

        cartesian_frame
            Output frame for ``position``, ``velocity``, and ``acceleration``.
            If ``None``, use the native propagation / state-defining frame.

        position, velocity, acceleration
            Request Cartesian translational outputs. Returned arrays are shaped
            ``(N, 3)`` and use SI units.

        attitude_reference_frame
            Reference frame from which body attitude is expressed.
            Quaternion, matrix, and Euler outputs represent the rotation
            ``attitude_reference_frame -> spacecraft body``.
            If ``None``, use the native state / attitude reference frame.

        attitude
            Convenience flag equivalent to requesting ``attitude_quat=True`` and
            ``attitude_spin=True``.

        attitude_quat, attitude_matrix, attitude_euler
            Request attitude orientation outputs. Quaternion ordering is
            controlled by ``quaternion_convention``.

        attitude_spin, attitude_acceleration
            Request spacecraft body angular-rate and angular-acceleration
            outputs, expressed in body coordinates.

        attitude_euler_sequence
            Euler-axis sequence used when ``attitude_euler=True``.

        attitude_euler_degrees
            If ``True``, Euler angles are returned in degrees. Otherwise they
            are returned in radians.

        quaternion_convention
            Quaternion ordering convention used for quaternion outputs.
            ``"scalar_first"`` returns ``(q0, q1, q2, q3)``.
            ``"scalar_last"`` returns ``(q1, q2, q3, q0)``.

        elements_frame
            Defining frame used for derived orbit elements.
            If ``None``, use the native orbit-defining frame.
            When orbit elements are requested, this frame should be
            pseudo-inertial.

        keplerian
            Request the classical Keplerian element set.
            The returned arrays are:
            ``semi_major_axis_m``, ``eccentricity``, ``inclination``, ``raan``,
            ``argp``, and ``anomaly``.

        anomaly_type
            Angular anomaly returned when ``keplerian=True``.
            Valid values are ``"mean"``, ``"eccentric"``, and ``"true"``.

        equinoctial
            Request the equinoctial element set.
            The returned arrays are:
            ``equinoctial_a_m``, ``equinoctial_ex``, ``equinoctial_ey``,
            ``equinoctial_hx``, ``equinoctial_hy``, and
            ``equinoctial_longitude``.

        longitude_type
            Longitude type returned when ``equinoctial=True``.
            Valid values are ``"mean"``, ``"eccentric"``, and ``"true"``.

        elements_angles_degrees
            If ``True``, all orbit-element angular outputs are returned in
            degrees. Otherwise they are returned in radians.

        mass
            Request spacecraft mass, returned in kilograms with shape ``(N,)``.

        additional_states
            Names of Orekit additional state entries to extract into the
            returned :attr:`SampledStates.additional` dictionary.

        additional_state_derivatives
            Names of Orekit additional-state derivatives to extract into the
            returned :attr:`SampledStates.additional_derivatives` dictionary.

        strict
            If ``True``, raise when a requested output is unavailable or cannot
            be represented. If ``False``, leave unavailable outputs as ``None``
            or omit them from the corresponding dictionaries.

        Returns
        -------
        SampledStates
            Structured vectorized outputs and metadata.

        Notes
        -----
        - All arrays are returned in vectorized form, even when ``times`` is a
          single instant.
        - This method is intended to be the single authoritative bulk-sampling
          path for the wrapper. Narrow convenience getters should delegate to
          it internally.
        """
        raise NotImplementedError

    def get_position(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled position vectors in meters.

        Parameters
        ----------
        times
            Time input accepted by :meth:`sample`.
        frame
            Output frame for the sampled position vectors.
            If ``None``, use the native propagation / state-defining frame.

        Returns
        -------
        ndarray
            Position array with shape ``(N, 3)`` in meters.
        """
        return self.sample(times, cartesian_frame=frame, position=True).position_m

    def get_velocity(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled velocity vectors in meters per second."""
        return self.sample(times, cartesian_frame=frame, velocity=True).velocity_mps

    def get_acceleration(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> np.ndarray:
        """Return sampled acceleration vectors in meters per second squared."""
        return self.sample(times, cartesian_frame=frame, acceleration=True).acceleration_mps2

    def get_pva(
        self,
        times: TimeLike,
        *,
        frame: Optional[FrameLike] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled position, velocity, and acceleration arrays.

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            ``(position_m, velocity_mps, acceleration_mps2)`` with shapes
            ``(N, 3)``, ``(N, 3)``, and ``(N, 3)``.
        """
        sampled = self.sample(times, cartesian_frame=frame, position=True, velocity=True, acceleration=True)
        return sampled.position_m, sampled.velocity_mps, sampled.acceleration_mps2

    def get_attitude_quat(
        self,
        times: TimeLike,
        *,
        reference_frame: Optional[FrameLike] = None,
        quaternion_convention: QuaternionConvention = "scalar_first",
    ) -> np.ndarray:
        """Return sampled attitude quaternions.

        The returned quaternion rotates coordinates from ``reference_frame`` to
        spacecraft body coordinates.
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

        The returned matrices rotate coordinates from ``reference_frame`` to
        spacecraft body coordinates.
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
        """Return sampled attitude Euler angles."""
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
        """Return sampled body angular-rate vectors in radians per second."""
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
        """Return sampled body angular-acceleration vectors in radians per second squared."""
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

        Returns
        -------
        ndarray
            Array with shape ``(N, 6)`` ordered as:
            ``a, e, i, raan, argp, anomaly``.

        Notes
        -----
        ``frame`` should be pseudo-inertial when supplied.
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

        Returns
        -------
        ndarray
            Array with shape ``(N, 6)`` ordered as:
            ``a, ex, ey, hx, hy, longitude``.

        Notes
        -----
        ``frame`` should be pseudo-inertial when supplied.
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
        """Return sampled spacecraft mass in kilograms."""
        return self.sample(times, mass=True).mass_kg

    def get_geodetic(
        self,
        times: TimeLike,
        *,
        degrees: bool = True,
        ellipsoid_a_m: float = 6378137.0,
        ellipsoid_b_m: float = 6356752.314245,
    ) -> np.ndarray:
        """Return Earth geodetic latitude, longitude, and altitude.

        This helper is intentionally outside :meth:`sample` because geodetic
        coordinates are a body-shape-specific derived product rather than a
        direct state field.

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
        ndarray
            Array with shape ``(N, 3)`` ordered as:
            ``latitude, longitude, altitude``.
            Altitude is in meters.

        Notes
        -----
        The intended implementation should perform the correct Orekit body-shape
        transform using the sampled position, frame, and date internally.
        """
        raise NotImplementedError

    def get_java_states(
        self,
        times: TimeLike,
    ) -> OrekitSpacecraftState | list[OrekitSpacecraftState] | Any:
        """Return raw Orekit ``SpacecraftState`` objects.

        This is a lower-level escape hatch for advanced use cases where callers
        need direct access to the underlying Orekit state objects rather than
        the NumPy-based sampled outputs.
        """
        raise NotImplementedError

    def list_additional_states(self) -> list[str]:
        """Return available additional Orekit state names."""
        raise NotImplementedError

    def list_additional_state_derivatives(self) -> list[str]:
        """Return available additional-state-derivative names."""
        raise NotImplementedError

    def get_additional_state(self, times: TimeLike, name: str) -> np.ndarray:
        """Return a sampled additional state by name."""
        return self.sample(times, additional_states=(name,)).additional[name]

    def get_additional_state_derivative(self, times: TimeLike, name: str) -> np.ndarray:
        """Return a sampled additional-state derivative by name."""
        return self.sample(times, additional_state_derivatives=(name,)).additional_derivatives[name]


__all__ = [
    "AngleType",
    "FrameLike",
    "LongitudeType",
    "QuaternionConvention",
    "SampledStates",
    "TimeLike",
    "VectorizedPropagator",
]
