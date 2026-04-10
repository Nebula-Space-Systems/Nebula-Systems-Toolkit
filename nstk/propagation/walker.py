from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Literal, Protocol, TypeAlias

import numpy as np

from nstk.propagation import orbit as orbit_module
from nstk.propagation.orbit import (
    Orbit,
    SupportsFrame,
    SupportsSpacecraftState,
    _bind_java,
    _coerce_position_angle_type,
)


class SupportsWalkerState(SupportsSpacecraftState, Protocol):
    """Structural subset of an Orekit ``SpacecraftState`` used in Walker helpers."""

    def getDate(self) -> Any: ...

    def getAttitude(self) -> Any: ...

    def getPVCoordinates(self, frame: SupportsFrame | None = None) -> Any: ...

    def getAdditionalDataValues(self) -> Any: ...

    def getAdditionalStatesDerivatives(self) -> Any: ...


WalkerOrbitFactory = Callable[[SupportsWalkerState], Orbit]
WalkerSeed: TypeAlias = Orbit | SupportsWalkerState
DEFAULT_WALKER_RAAN_SPAN = 2.0 * math.pi


def _coerce_exact_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    try:
        coerced = int(value)
    except Exception as exc:
        raise TypeError(f"{name} must be an integer") from exc

    try:
        if float(value) != float(coerced):
            raise ValueError(f"{name} must be an integer")
    except Exception:
        if value != coerced:
            raise ValueError(f"{name} must be an integer")
    return coerced


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _coerce_raan_span(raan_span: float) -> float:
    span = float(raan_span)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("raan_span must be finite and > 0")
    if span > (2.0 * math.pi + 1.0e-12):
        raise ValueError("raan_span must be <= 2*pi radians")
    return span


def _normalize_anomaly_type_label(anomaly_type: Any) -> Literal["mean", "true", "eccentric"]:
    _bind_java()
    pa_type = _coerce_position_angle_type(anomaly_type)

    if pa_type == orbit_module.PositionAngleType.MEAN:
        return "mean"
    if pa_type == orbit_module.PositionAngleType.TRUE:
        return "true"
    if pa_type == orbit_module.PositionAngleType.ECCENTRIC:
        return "eccentric"
    raise ValueError("Unsupported anomaly_type")


def _validate_walker_geometry(
    total_satellites: int,
    num_planes: int,
    phasing: int,
    *,
    initial_raan_offset: float,
    initial_anomaly_offset: float,
    raan_span: float,
) -> tuple[int, int, int]:
    t = _coerce_exact_int("total_satellites", total_satellites)
    p = _coerce_exact_int("num_planes", num_planes)
    f = _coerce_exact_int("phasing", phasing)

    if t <= 0:
        raise ValueError("total_satellites must be >= 1")
    if p <= 0:
        raise ValueError("num_planes must be >= 1")
    if t % p != 0:
        raise ValueError("total_satellites must be divisible by num_planes")
    if not math.isfinite(float(initial_raan_offset)):
        raise ValueError("initial_raan_offset must be finite")
    if not math.isfinite(float(initial_anomaly_offset)):
        raise ValueError("initial_anomaly_offset must be finite")
    _coerce_raan_span(raan_span)

    return t, p, f


def _iter_walker_raan_anomaly(
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int,
    raan_span: float,
    include_seed: bool,
    base_raan: float,
    base_anomaly: float,
) -> Iterator[tuple[float, float]]:
    sats_per_plane = total_satellites // num_planes
    two_pi = 2.0 * math.pi

    for plane_idx in range(num_planes):
        d_raan = raan_span * (float(plane_idx) / float(num_planes))
        for slot_idx in range(sats_per_plane):
            if (not include_seed) and plane_idx == 0 and slot_idx == 0:
                continue
            d_anomaly = two_pi * (
                float(slot_idx) / float(sats_per_plane)
                + float(phasing * plane_idx) / float(total_satellites)
            )
            yield (
                _wrap_pm_pi(base_raan + d_raan),
                _wrap_pm_pi(base_anomaly + d_anomaly),
            )


def _is_spacecraft_state_instance(value: Any) -> bool:
    _bind_java()
    from org.orekit.propagation import SpacecraftState  # type: ignore

    try:
        return bool(SpacecraftState.class_.isInstance(value))
    except Exception:
        return False


def _coerce_seed_state(seed: WalkerSeed) -> SupportsWalkerState:
    if isinstance(seed, Orbit):
        return seed.propagator.getInitialState()
    if _is_spacecraft_state_instance(seed):
        return seed
    raise TypeError("seed must be an nstk.propagation.Orbit or Orekit SpacecraftState")


def _extract_state_kepler(
    state: SupportsWalkerState,
    *,
    anomaly_type: Any = "mean",
) -> tuple[Any, float, float, float, float, float, float, SupportsFrame]:
    """Return inertial seed geometry as ``(date, a, e, i, raan, argp, anomaly, frame)``."""

    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore

    date0 = state.getDate()
    orbit0 = state.getOrbit()
    mu = float(orbit0.getMu())
    frame0 = orbit0.getFrame()
    pv0 = state.getPVCoordinates(frame0)

    if frame0.isPseudoInertial():
        inertial = frame0
        pv_inertial = pv0
    else:
        inertial = FramesFactory.getGCRF()
        tr = frame0.getTransformTo(inertial, date0)
        pv_inertial = tr.transformPVCoordinates(pv0)

    kep = KeplerianOrbit(CartesianOrbit(pv_inertial, inertial, date0, mu))
    anomaly_label = _normalize_anomaly_type_label(anomaly_type)
    if anomaly_label == "mean":
        anomaly = float(kep.getMeanAnomaly())
    elif anomaly_label == "true":
        anomaly = float(kep.getTrueAnomaly())
    else:
        anomaly = float(kep.getEccentricAnomaly())
    return (
        date0,
        float(kep.getA()),
        float(kep.getE()),
        float(kep.getI()),
        float(kep.getRightAscensionOfAscendingNode()),
        float(kep.getPerigeeArgument()),
        anomaly,
        inertial,
    )


def _copy_data_dictionary(source):
    _bind_java()
    from org.orekit.utils import DataDictionary  # type: ignore

    out = DataDictionary()
    for entry in source.getData():
        out.put(entry.getKey(), entry.getValue())
    return out


def _copy_double_array_dictionary(source):
    _bind_java()
    from org.orekit.utils import DoubleArrayDictionary  # type: ignore

    out = DoubleArrayDictionary()
    for entry in source.getData():
        out.put(entry.getKey(), np.asarray(entry.getValue(), dtype=np.float64).tolist())
    return out


def _state_with_kepler_angles(
    seed_state: SupportsWalkerState,
    *,
    inertial_frame: SupportsFrame,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: Any,
) -> SupportsWalkerState:
    _bind_java()
    from org.orekit.orbits import KeplerianOrbit  # type: ignore
    from org.orekit.propagation import SpacecraftState  # type: ignore

    orbit0 = seed_state.getOrbit()
    new_orbit = KeplerianOrbit(
        float(a),
        float(e),
        float(i),
        float(argp),
        float(raan),
        float(anomaly),
        _coerce_position_angle_type(anomaly_type),
        inertial_frame,
        seed_state.getDate(),
        float(orbit0.getMu()),
    )
    return SpacecraftState(
        new_orbit,
        seed_state.getAttitude(),
        float(seed_state.getMass()),
        _copy_data_dictionary(seed_state.getAdditionalDataValues()),
        _copy_double_array_dictionary(seed_state.getAdditionalStatesDerivatives()),
    )


def build_walker_initial_states(
    seed: WalkerSeed,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 1,
    raan_span: float = DEFAULT_WALKER_RAAN_SPAN,
    initial_raan_offset: float = 0.0,
    initial_anomaly_offset: float = 0.0,
    anomaly_type: Any = "mean",
    include_seed: bool = True,
) -> list[SupportsWalkerState]:
    """Build Walkerized Orekit ``SpacecraftState`` objects from a seed state.

    Parameters
    ----------
    seed : Orbit | SpacecraftState
        Seed state source. Normal user workflows should start from an NSTK
        :class:`~nstk.propagation.orbit.Orbit` and call
        ``seed_orbit.build_walker_initial_states(...)``. Raw Orekit
        ``SpacecraftState`` inputs are retained for advanced interop only.
    total_satellites : int
        Total number of satellites ``T`` in the constellation.
    num_planes : int
        Number of orbital planes ``P``. ``total_satellites`` must be divisible
        by ``num_planes``.
    phasing : int, default 1
        Walker phasing factor ``F``. Adjacent planes are shifted by
        ``F * 2*pi / T`` in the selected anomaly coordinate.
    raan_span : float, default ``2*pi``
        RAAN span in radians across all planes. A full ``2*pi`` span matches a
        classic Walker-delta style spread; a ``pi`` span matches the older
        star-style spread. Values must lie in ``(0, 2*pi]``.
    initial_raan_offset : float, default 0.0
        Extra offset in radians applied to the seed RAAN before plane spacing
        is generated. This shifts the whole constellation in RAAN.
    initial_anomaly_offset : float, default 0.0
        Extra offset in radians applied to the seed anomaly before slot spacing
        and inter-plane phasing are generated.
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, default "mean"
        Keplerian anomaly coordinate used for ``initial_anomaly_offset`` and
        Walker phasing. The selected anomaly is also what gets written into each
        returned state.
    include_seed : bool, default True
        If True, include plane 0 / slot 0 in the returned list. When both
        offsets are zero, that member matches the seed geometry. When either
        offset is nonzero, the first member is an offset version of the seed.

    Returns
    -------
    list[SpacecraftState]
        New initial states with the seed's epoch, mass, attitude snapshot, and
        additional state dictionaries preserved. Only RAAN and the selected
        anomaly coordinate are changed to satisfy the Walker geometry. Members
        are returned in plane-major, slot-major order.

    Notes
    -----
    This function preserves state data, not propagator configuration. It does
    not clone force models, integrator settings, event detectors, or attitude
    providers. Use the returned states to build those pieces explicitly when you
    need custom propagators.
    """

    resolved_span = _coerce_raan_span(raan_span)
    anomaly_label = _normalize_anomaly_type_label(anomaly_type)
    t, p, f = _validate_walker_geometry(
        total_satellites,
        num_planes,
        phasing,
        initial_raan_offset=initial_raan_offset,
        initial_anomaly_offset=initial_anomaly_offset,
        raan_span=resolved_span,
    )
    seed_state = _coerce_seed_state(seed)

    _, a0, e0, i0, seed_raan, argp0, seed_anomaly, inertial_frame = _extract_state_kepler(
        seed_state,
        anomaly_type=anomaly_label,
    )
    base_raan = _wrap_pm_pi(seed_raan + float(initial_raan_offset))
    base_anomaly = _wrap_pm_pi(seed_anomaly + float(initial_anomaly_offset))

    out: list[SupportsWalkerState] = []
    for raan_i, anomaly_i in _iter_walker_raan_anomaly(
        total_satellites=t,
        num_planes=p,
        phasing=f,
        raan_span=resolved_span,
        include_seed=include_seed,
        base_raan=base_raan,
        base_anomaly=base_anomaly,
    ):
        out.append(
            _state_with_kepler_angles(
                seed_state,
                inertial_frame=inertial_frame,
                a=a0,
                e=e0,
                i=i0,
                raan=raan_i,
                argp=argp0,
                anomaly=anomaly_i,
                anomaly_type=anomaly_label,
            )
        )
    return out


def build_walker_constellation(
    seed: WalkerSeed,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 1,
    raan_span: float = DEFAULT_WALKER_RAAN_SPAN,
    initial_raan_offset: float = 0.0,
    initial_anomaly_offset: float = 0.0,
    anomaly_type: Any = "mean",
    include_seed: bool = True,
    orbit_factory: WalkerOrbitFactory | None = None,
) -> list[Orbit]:
    """Build Walker satellites as NSTK ``Orbit`` objects.

    Parameters
    ----------
    seed : Orbit | SpacecraftState
        Seed source for the Walker geometry. Normal user workflows should start
        from an NSTK :class:`~nstk.propagation.orbit.Orbit` and call
        ``seed_orbit.build_walker_constellation(...)``. Raw Orekit
        ``SpacecraftState`` seeds are supported when ``orbit_factory`` is
        provided for advanced interop.
    total_satellites : int
        Total number of satellites ``T`` in the constellation.
    num_planes : int
        Number of planes ``P``.
    phasing : int, default 1
        Walker phasing factor ``F``. Adjacent planes are shifted by
        ``F * 2*pi / T`` in the selected anomaly coordinate.
    raan_span : float, default ``2*pi``
        RAAN span in radians across all planes. A full ``2*pi`` span matches a
        classic Walker-delta style spread; a ``pi`` span matches the older
        star-style spread. Values must lie in ``(0, 2*pi]``.
    initial_raan_offset : float, default 0.0
        Extra offset in radians applied to the seed RAAN before plane spacing
        is generated.
    initial_anomaly_offset : float, default 0.0
        Extra offset in radians applied to the seed anomaly before slot spacing
        and inter-plane phasing are generated.
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, default "mean"
        Keplerian anomaly coordinate used for the Walker slot spacing and
        offsets.
    include_seed : bool, default True
        If True, include plane 0 / slot 0 in the returned list. When both
        offsets are zero, that member matches the seed geometry.
    orbit_factory : callable, optional
        Callback of the form ``orbit_factory(state) -> Orbit``. When supplied,
        this function first creates Walkerized ``SpacecraftState`` objects and
        then lets the callback build custom propagators or NSTK ``Orbit``
        wrappers from each state.

    Returns
    -------
    list[Orbit]
        Walker constellation members as NSTK orbit wrappers in plane-major,
        slot-major order.

    Notes
    -----
    If ``orbit_factory`` is omitted, this function only works for ``Orbit``
    seeds that expose a reproducible NSTK construction recipe. That path keeps
    the seed's supported propagator configuration intact. For generic custom
    propagators, pass ``orbit_factory`` or call
    :func:`build_walker_initial_states` directly.
    """

    if orbit_factory is not None:
        if not callable(orbit_factory):
            raise TypeError("orbit_factory must be callable")
        states = build_walker_initial_states(
            seed,
            total_satellites=total_satellites,
            num_planes=num_planes,
            phasing=phasing,
            raan_span=raan_span,
            initial_raan_offset=initial_raan_offset,
            initial_anomaly_offset=initial_anomaly_offset,
            anomaly_type=anomaly_type,
            include_seed=include_seed,
        )
        out: list[Orbit] = []
        for state in states:
            orbit = orbit_factory(state)
            if not isinstance(orbit, Orbit):
                raise TypeError("orbit_factory must return nstk.propagation.Orbit instances")
            out.append(orbit)
        return out

    if not isinstance(seed, Orbit):
        raise ValueError(
            "seed must be an Orbit when orbit_factory is omitted; "
            "otherwise pass orbit_factory=... or use build_walker_initial_states(...)."
        )

    resolved_span = _coerce_raan_span(raan_span)
    anomaly_label = _normalize_anomaly_type_label(anomaly_type)
    t, p, f = _validate_walker_geometry(
        total_satellites,
        num_planes,
        phasing,
        initial_raan_offset=initial_raan_offset,
        initial_anomaly_offset=initial_anomaly_offset,
        raan_span=resolved_span,
    )
    seed_state = seed.propagator.getInitialState()
    _, _, _, _, seed_raan, _, seed_anomaly, _ = _extract_state_kepler(
        seed_state,
        anomaly_type=anomaly_label,
    )
    base_raan = _wrap_pm_pi(seed_raan + float(initial_raan_offset))
    base_anomaly = _wrap_pm_pi(seed_anomaly + float(initial_anomaly_offset))
    out: list[Orbit] = []

    for raan_i, anomaly_i in _iter_walker_raan_anomaly(
        total_satellites=t,
        num_planes=p,
        phasing=f,
        raan_span=resolved_span,
        include_seed=include_seed,
        base_raan=base_raan,
        base_anomaly=base_anomaly,
    ):
        try:
            out.append(
                seed._clone_for_walker(
                    raan=raan_i,
                    anomaly=anomaly_i,
                    anomaly_type=anomaly_label,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "This seed Orbit does not expose a reproducible NSTK construction recipe. "
                "Pass orbit_factory=... to build custom propagators from "
                "build_walker_initial_states(...)."
            ) from exc

    return out


__all__ = [
    "WalkerOrbitFactory",
    "DEFAULT_WALKER_RAAN_SPAN",
    "build_walker_initial_states",
    "build_walker_constellation",
]
