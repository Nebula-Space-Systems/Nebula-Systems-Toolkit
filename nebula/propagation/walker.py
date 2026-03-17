from __future__ import annotations

import math
from typing import Any, Literal, Optional

from nebula.propagation.orbit import Orbit

WalkerPattern = Literal["delta", "star"]
WalkerConstructor = Literal["auto", "two_body", "numerical"]


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

    return t, p, f, pat  # type: ignore[return-value]


def _extract_seed_kepler(seed: Orbit) -> tuple[dict[str, Any], float, float]:
    """Extract mean Keplerian elements from the seed at epoch."""

    from org.orekit.frames import FramesFactory  # type: ignore
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore

    state0 = seed.propagator.getInitialState()
    date0 = state0.getDate()
    mu = float(state0.getOrbit().getMu())
    mass = float(state0.getMass())
    frame0 = state0.getFrame()
    pv0 = state0.getPVCoordinates(frame0)

    if frame0.isPseudoInertial():
        inertial = frame0
        pv_inertial = pv0
    else:
        inertial = FramesFactory.getGCRF()
        tr = frame0.getTransformTo(inertial, date0)
        pv_inertial = tr.transformPVCoordinates(pv0)

    cart = CartesianOrbit(pv_inertial, inertial, date0, mu)
    kep = KeplerianOrbit(cart)

    base_kwargs: dict[str, Any] = {
        "epoch": seed.epoch,
        "a": float(kep.getA()),
        "e": float(kep.getE()),
        "i": float(kep.getI()),
        "argp": float(kep.getPerigeeArgument()),
        "anomaly_type": "mean",
        "mass": mass,
        "inertial_frame": inertial,
        "iers_convention": seed.iers,
        "simple_eop": bool(seed.simple_eop),
    }
    return base_kwargs, float(kep.getRightAscensionOfAscendingNode()), float(kep.getMeanAnomaly())


def _seed_is_numerical(seed: Orbit) -> bool:
    name = str(seed.propagator.getClass().getSimpleName())
    return name == "NumericalPropagator"


def build_walker_constellation(
    seed: Orbit,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 0,
    pattern: WalkerPattern = "delta",
    include_seed: bool = True,
    constructor: WalkerConstructor = "auto",
    constructor_kwargs: Optional[dict[str, Any]] = None,
) -> list[Orbit]:
    """Build a Walker T/P/F constellation from a seed orbit.

    In ``auto`` mode, seed numerical propagators produce numerical members;
    all other seeds produce two-body analytical members.
    """

    t, p, f, pat = _validate_walker_inputs(total_satellites, num_planes, phasing, pattern)
    sats_per_plane = t // p
    raan_span = (2.0 * math.pi) if pat == "delta" else math.pi

    base_kwargs, seed_raan, seed_mean = _extract_seed_kepler(seed)
    extra = dict(constructor_kwargs or {})

    ctor_mode = str(constructor).strip().lower()
    if ctor_mode not in ("auto", "two_body", "numerical"):
        raise ValueError("constructor must be 'auto', 'two_body', or 'numerical'")

    if ctor_mode == "auto":
        ctor_mode = "numerical" if _seed_is_numerical(seed) else "two_body"

    if ctor_mode == "numerical":
        make_orbit = Orbit.from_kepler_numerical
    else:
        make_orbit = Orbit.from_kepler_two_body

    out: list[Orbit] = []
    two_pi = 2.0 * math.pi

    for plane_idx in range(p):
        d_raan = raan_span * (float(plane_idx) / float(p))
        for slot_idx in range(sats_per_plane):
            if (not include_seed) and plane_idx == 0 and slot_idx == 0:
                continue

            d_mean = two_pi * (
                float(slot_idx) / float(sats_per_plane)
                + float(f * plane_idx) / float(t)
            )

            kwargs = dict(base_kwargs)
            kwargs.update(extra)
            kwargs["raan"] = _wrap_pm_pi(seed_raan + d_raan)
            kwargs["anomaly"] = _wrap_pm_pi(seed_mean + d_mean)
            out.append(make_orbit(**kwargs))

    return out
