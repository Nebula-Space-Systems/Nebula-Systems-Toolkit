from __future__ import annotations

import math
from typing import Any, Callable, Literal

import numpy as np
from astropy.time import Time

from nstk.propagation.orbit import (
    DEFAULT_ATTITUDE,
    Orbit,
    _bind_java,
    _coerce_attitude_provider,
    _coerce_iers,
    _coerce_position_angle_type,
    _resolve_named_frame,
    astropy_time_to_orekit_date,
)

WalkerPattern = Literal["delta", "star"]
WalkerOrbitFactory = Callable[[Any], Orbit]


def _wrap_pm_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _validate_walker_inputs(
    total_satellites: int,
    num_planes: int,
    phasing: int,
    pattern: WalkerPattern,
) -> tuple[int, int, int, WalkerPattern]:
    t = int(total_satellites)
    p = int(num_planes)
    f = int(phasing)
    pat = str(pattern).strip().lower()

    if t <= 0:
        raise ValueError("total_satellites must be >= 1")
    if p <= 0:
        raise ValueError("num_planes must be >= 1")
    if t % p != 0:
        raise ValueError("total_satellites must be divisible by num_planes")
    if pat not in ("delta", "star"):
        raise ValueError("pattern must be 'delta' or 'star'")

    pattern_out: WalkerPattern = "delta" if pat == "delta" else "star"
    return t, p, f, pattern_out


def _iter_walker_raan_mean(
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int,
    pattern: WalkerPattern,
    include_seed: bool,
    seed_raan: float,
    seed_mean: float,
):
    sats_per_plane = total_satellites // num_planes
    raan_span = (2.0 * math.pi) if pattern == "delta" else math.pi
    two_pi = 2.0 * math.pi

    for plane_idx in range(num_planes):
        d_raan = raan_span * (float(plane_idx) / float(num_planes))
        for slot_idx in range(sats_per_plane):
            if (not include_seed) and plane_idx == 0 and slot_idx == 0:
                continue
            d_mean = two_pi * (
                float(slot_idx) / float(sats_per_plane)
                + float(phasing * plane_idx) / float(total_satellites)
            )
            yield (
                _wrap_pm_pi(seed_raan + d_raan),
                _wrap_pm_pi(seed_mean + d_mean),
            )


def _is_spacecraft_state_instance(value: Any) -> bool:
    _bind_java()
    from org.orekit.propagation import SpacecraftState  # type: ignore

    try:
        return bool(SpacecraftState.class_.isInstance(value))
    except Exception:
        return False


def _coerce_seed_state(seed: Orbit | Any):
    if isinstance(seed, Orbit):
        return seed.propagator.getInitialState()
    if _is_spacecraft_state_instance(seed):
        return seed
    raise TypeError("seed must be an nstk.propagation.Orbit or Orekit SpacecraftState")


def _extract_state_kepler(state) -> tuple[Any, float, float, float, float, float, float, Any]:
    """Return inertial seed geometry as ``(date, a, e, i, raan, argp, mean, frame)``."""

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
    return (
        date0,
        float(kep.getA()),
        float(kep.getE()),
        float(kep.getI()),
        float(kep.getRightAscensionOfAscendingNode()),
        float(kep.getPerigeeArgument()),
        float(kep.getMeanAnomaly()),
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
    seed_state,
    *,
    inertial_frame,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
):
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
        _coerce_position_angle_type("mean"),
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


def spacecraft_state_from_kepler(
    epoch: Time,
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    anomaly: float,
    anomaly_type: Any = None,
    mass: float = 1000.0,
    inertial_frame: Any = None,
    iers_convention: Any = None,
    simple_eop: bool = True,
    attitude: Any = DEFAULT_ATTITUDE,
    *,
    mu: float | None = None,
):
    """Build an Orekit ``SpacecraftState`` from Keplerian elements.

    Parameters
    ----------
    epoch : astropy.time.Time
        Initial epoch for the returned state.
    a, e, i, raan, argp, anomaly : float
        Classical Keplerian elements in SI/radian units.
    anomaly_type : {"mean", "true", "eccentric"} or PositionAngleType, optional
        Interpretation of ``anomaly``. Defaults to mean anomaly when omitted.
    mass : float, default 1000.0
        Spacecraft mass in kilograms.
    inertial_frame : Frame | str | None, optional
        Pseudo-inertial frame for the constructed orbit. ``None`` defaults to
        GCRF, and common NSTK frame strings are accepted.
    iers_convention : IERSConventions, optional
        Earth orientation convention used if frame-name resolution needs it.
    simple_eop : bool, default True
        Whether Earth-fixed frame resolution should use simple EOP mode.
    attitude : optional, default ``"vvlh"``
        Attitude law specification evaluated at ``epoch``. The built-in
        default is STK-style ``"vvlh"``, which resolves to
        ``LofOffset(inertial_frame, LOFType.VVLH)``. The returned
        ``SpacecraftState`` stores the resulting attitude snapshot directly.
    mu : float, optional
        Gravitational parameter used to construct the Keplerian orbit. Defaults
        to ``Constants.WGS84_EARTH_MU``.

    Returns
    -------
    SpacecraftState
        Orekit initial state that can seed custom propagator creation or Walker
        geometry generation workflows.
    """

    _bind_java()
    from org.orekit.orbits import KeplerianOrbit  # type: ignore
    from org.orekit.propagation import SpacecraftState  # type: ignore
    from org.orekit.utils import Constants  # type: ignore

    iers = _coerce_iers(iers_convention)
    if inertial_frame is None:
        from org.orekit.frames import FramesFactory  # type: ignore

        inertial_frame = FramesFactory.getGCRF()
    elif isinstance(inertial_frame, str):
        inertial_frame = _resolve_named_frame(
            inertial_frame,
            iers=iers,
            simple_eop=bool(simple_eop),
        )
    if not bool(inertial_frame.isPseudoInertial()):
        raise ValueError("inertial_frame must be pseudo-inertial")

    mu_val = float(Constants.WGS84_EARTH_MU if mu is None else mu)
    orbit0 = KeplerianOrbit(
        float(a),
        float(e),
        float(i),
        float(argp),
        float(raan),
        float(anomaly),
        _coerce_position_angle_type(anomaly_type),
        inertial_frame,
        astropy_time_to_orekit_date(epoch),
        mu_val,
    )
    provider = _coerce_attitude_provider(
        attitude,
        inertial_frame=inertial_frame,
        iers=iers,
        simple_eop=bool(simple_eop),
    )
    return SpacecraftState(
        orbit0,
        provider.getAttitude(orbit0, orbit0.getDate(), inertial_frame),
        float(mass),
    )


def build_walker_initial_states(
    seed: Orbit | Any,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 0,
    pattern: WalkerPattern = "delta",
    include_seed: bool = True,
) -> list[Any]:
    """Build Walkerized Orekit ``SpacecraftState`` objects from a seed state.

    Parameters
    ----------
    seed : Orbit | SpacecraftState
        Seed state source. ``Orbit`` inputs are normalized via
        ``orbit.propagator.getInitialState()``.
    total_satellites, num_planes, phasing, pattern, include_seed
        Standard Walker ``T/P/F`` geometry controls. ``pattern="delta"`` uses
        a full ``2*pi`` RAAN span, while ``pattern="star"`` uses a ``pi`` span.

    Returns
    -------
    list[SpacecraftState]
        New initial states with the seed's epoch, mass, attitude snapshot, and
        additional state dictionaries preserved. Only RAAN and mean anomaly are
        changed to satisfy the Walker geometry.

    Notes
    -----
    This function preserves state data, not propagator configuration. It does
    not clone force models, integrator settings, event detectors, or attitude
    providers. Use the returned states to build those pieces explicitly when you
    need custom propagators.
    """

    t, p, f, pat = _validate_walker_inputs(total_satellites, num_planes, phasing, pattern)
    seed_state = _coerce_seed_state(seed)

    _, a0, e0, i0, seed_raan, argp0, seed_mean, inertial_frame = _extract_state_kepler(seed_state)

    out: list[Any] = []
    for raan_i, mean_i in _iter_walker_raan_mean(
        total_satellites=t,
        num_planes=p,
        phasing=f,
        pattern=pat,
        include_seed=include_seed,
        seed_raan=seed_raan,
        seed_mean=seed_mean,
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
                anomaly=mean_i,
            )
        )
    return out


def build_walker_constellation(
    seed: Orbit | Any,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 0,
    pattern: WalkerPattern = "delta",
    include_seed: bool = True,
    orbit_factory: WalkerOrbitFactory | None = None,
) -> list[Orbit]:
    """Build Walker satellites as NSTK ``Orbit`` objects.

    Parameters
    ----------
    seed : Orbit | SpacecraftState
        Seed source for the Walker geometry. Raw ``SpacecraftState`` seeds are
        supported when ``orbit_factory`` is provided.
    total_satellites, num_planes, phasing, pattern, include_seed
        Standard Walker ``T/P/F`` geometry controls.
    orbit_factory : callable, optional
        Callback of the form ``orbit_factory(state) -> Orbit``. When supplied,
        this function first creates Walkerized ``SpacecraftState`` objects and
        then lets the callback build custom propagators or NSTK ``Orbit``
        wrappers from each state.

    Returns
    -------
    list[Orbit]
        Walker constellation members as NSTK orbit wrappers.

    Notes
    -----
    If ``orbit_factory`` is omitted, this function only works for ``Orbit``
    seeds that expose a reproducible NSTK construction recipe. That path keeps
    the seed's supported propagator configuration intact. For generic custom
    propagators, pass ``orbit_factory`` or call
    :func:`build_walker_initial_states` directly.
    """

    if orbit_factory is not None:
        states = build_walker_initial_states(
            seed,
            total_satellites=total_satellites,
            num_planes=num_planes,
            phasing=phasing,
            pattern=pattern,
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

    t, p, f, pat = _validate_walker_inputs(total_satellites, num_planes, phasing, pattern)
    seed_state = seed.propagator.getInitialState()
    _, _, _, _, seed_raan, _, seed_mean, _ = _extract_state_kepler(seed_state)
    out: list[Orbit] = []

    for raan_i, mean_i in _iter_walker_raan_mean(
        total_satellites=t,
        num_planes=p,
        phasing=f,
        pattern=pat,
        include_seed=include_seed,
        seed_raan=seed_raan,
        seed_mean=seed_mean,
    ):
        try:
            out.append(
                seed._clone_for_walker(
                    raan=raan_i,
                    anomaly=mean_i,
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
    "WalkerPattern",
    "WalkerOrbitFactory",
    "spacecraft_state_from_kepler",
    "build_walker_initial_states",
    "build_walker_constellation",
]
