from __future__ import annotations

import math
from typing import Any, Literal, Optional

from nebula.propagation.orbit import FramesFactory, Orbit, initialize_orekit

WalkerPattern = Literal["delta", "star"]


_FRAME_NAME_TO_KEY = {
    "GCRF": "gcrf",
    "ICRF": "icrf",
    "EME2000": "eme2000",
    "MOD": "mod",
    "TOD": "tod",
    "TEME": "teme",
    "CIRF": "cirf",
    "VEIS1950": "veis1950",
    "ECLIPTIC": "ecliptic",
}


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


def _infer_precision_gravity_kwargs(seed: Orbit) -> dict[str, Any]:
    """
    Best-effort gravity-model cloning from an Orekit seed propagator.
    """
    gravity_model: Literal["newtonian", "harmonic"] = "newtonian"
    gravity_degree = 20
    gravity_order = 20

    try:
        for fm in seed.propagator.getAllForceModels():  # type: ignore[attr-defined]
            simple = str(fm.getClass().getSimpleName())
            if simple == "HolmesFeatherstoneAttractionModel":
                gravity_model = "harmonic"
                try:
                    provider = fm.getProvider()
                    gravity_degree = int(provider.getMaxDegree())
                    gravity_order = int(provider.getMaxOrder())
                except Exception:
                    pass
                break
    except Exception:
        pass

    return {
        "gravity_model": gravity_model,
        "gravity_degree": gravity_degree,
        "gravity_order": gravity_order,
    }


def _extract_precision_seed(seed: Orbit) -> tuple[dict[str, Any], float, float]:
    """
    Extract mean-element seed and reproducible constructor kwargs for precision mode.
    """
    initialize_orekit()
    from org.orekit.orbits import CartesianOrbit, KeplerianOrbit  # type: ignore

    state0 = seed.propagator.getInitialState()  # type: ignore[attr-defined]
    date0 = state0.getDate()
    mu = float(state0.getOrbit().getMu())
    mass_kg = float(state0.getMass())
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
    frame_name = str(inertial.getName()).upper()
    inertial_key = _FRAME_NAME_TO_KEY.get(frame_name, "gcrf")

    base_kwargs: dict[str, Any] = {
        "epoch": seed.epoch,
        "a_m": float(kep.getA()),
        "e": float(kep.getE()),
        "i": float(kep.getI()),
        "argp": float(kep.getPerigeeArgument()),
        "anomaly_type": "mean",
        "degrees": False,
        "inertial_frame": inertial_key,
        "mass_kg": mass_kg,
        "mu": float(kep.getMu()),
        "dt_save_s": float(seed.dt),
        "iers": seed.iers,
        "simple_eop": bool(seed.simple_eop),
        "itrf_query_mode": seed.itrf_query_mode,
        "interpolation_mode": seed.interpolation_mode,
    }
    base_kwargs.update(_infer_precision_gravity_kwargs(seed))

    return base_kwargs, float(kep.getRightAscensionOfAscendingNode()), float(
        kep.getMeanAnomaly()
    )


def _extract_fast_seed(seed: Orbit) -> tuple[dict[str, Any], float, float]:
    """
    Extract seed and constructor kwargs for efficiency mode.
    """
    impl = seed._fast_impl  # type: ignore[attr-defined]
    base_kwargs: dict[str, Any] = {
        "epoch": seed.epoch,
        "a_m": float(impl.a_m),
        "e": float(impl.e),
        "i": float(impl.i_rad),
        "argp": float(impl.argp_rad),
        "anomaly_type": "mean",
        "degrees": False,
        "mu": float(impl.mu),
        "dt_save_s": float(impl.dt),
        "enable_j2": bool(impl.enable_j2),
        "j2_mode": str(impl.j2_mode),
        "j2_substeps": int(impl.j2_substeps),
        "J2": float(impl.J2),
        "Re_m": float(impl.Re_m),
        "use_polar_motion": bool(impl.use_polar_motion),
    }
    return base_kwargs, float(impl.raan_rad), float(impl.M0_rad)


def build_walker_constellation(
    seed: Orbit,
    *,
    total_satellites: int,
    num_planes: int,
    phasing: int = 0,
    pattern: WalkerPattern = "delta",
    include_seed: bool = True,
    precision_overrides: Optional[dict[str, Any]] = None,
) -> list[Orbit]:
    """
    Build a Walker constellation from a seed orbit.

    This helper generates a classical Walker ``T/P/F`` constellation from
    a reference (seed) satellite:

    - ``T`` = ``total_satellites`` (total members)
    - ``P`` = ``num_planes`` (orbital planes)
    - ``F`` = ``phasing`` (inter-plane slot offset)

    All generated members inherit the seed's core orbital shape
    (semi-major axis, eccentricity, inclination, argument of perigee,
    epoch, and propagation mode). Only RAAN and mean anomaly are shifted
    according to the selected Walker geometry.

    Mean-anomaly placement follows:

    ``M = M0 + slot*(2π/s) + plane*(F*2π/T)``

    where ``s = T/P`` satellites per plane.

    Parameters
    ----------
    seed : Orbit
        Reference orbit used to derive the constellation. The output mode matches
        the seed mode:

        - efficiency seed -> uses fast constructor
        - precision seed -> uses precision constructor

        The seed's dynamic configuration (for example gravity/J2 options) is
        cloned as closely as possible for consistency across all generated members.
    total_satellites : int
        Total number of satellites ``T`` in the constellation. Must be a positive
        integer and divisible by ``num_planes``.
    num_planes : int
        Number of orbital planes ``P``. Must be a positive integer that divides
        ``total_satellites`` exactly. Satellites per plane are ``s = T/P``.
    phasing : int, default 0
        Walker phasing parameter ``F`` controlling along-track offset between
        adjacent planes. ``F=0`` aligns equivalent slots across planes; nonzero
        values introduce systematic inter-plane phase shifts.
    pattern : {"delta","star"}, default "delta"
        RAAN distribution pattern:

        - ``"delta"``: planes span 360° in RAAN
        - ``"star"``: planes span 180° in RAAN

        This changes only RAAN spacing, not the in-plane slot spacing.
    include_seed : bool, default True
        If ``True``, the first generated member corresponds to the seed-equivalent
        slot (plane 0, slot 0). If ``False``, that member is omitted from output.
    precision_overrides : dict[str, Any] | None, default None
        Optional keyword overrides applied only when ``seed`` is in precision mode.
        Use this to force specific precision-constructor options (for example,
        gravity model settings) when you need strict reproducibility with custom
        force-model assumptions.

    Returns
    -------
    list[Orbit]
        Generated ``Orbit`` objects in plane-major order (all slots for plane 0,
        then plane 1, etc.). Output length is:

        - ``T`` when ``include_seed=True``
        - ``T - 1`` when ``include_seed=False``

    Raises
    ------
    ValueError
        If ``total_satellites <= 0``, ``num_planes <= 0``, ``total_satellites``
        is not divisible by ``num_planes``, or ``pattern`` is invalid.
    RuntimeError
        If a precision constructor is required but unavailable.
    """
    t, p, f, pat = _validate_walker_inputs(total_satellites, num_planes, phasing, pattern)
    sats_per_plane = t // p
    raan_span = (2.0 * math.pi) if pat == "delta" else math.pi

    if seed.is_efficiency:
        base_kwargs, seed_raan, seed_mean = _extract_fast_seed(seed)
        make_orbit = Orbit.from_kepler_fast
    elif seed.is_precision:
        base_kwargs, seed_raan, seed_mean = _extract_precision_seed(seed)
        if precision_overrides:
            base_kwargs.update(dict(precision_overrides))
        if hasattr(Orbit, "from_kepler_precision"):
            make_orbit = Orbit.from_kepler_precision  # type: ignore[attr-defined]
        elif hasattr(Orbit, "from_kepler_precise"):
            make_orbit = Orbit.from_kepler_precise  # type: ignore[attr-defined]
        elif hasattr(Orbit, "from_kepler"):
            make_orbit = Orbit.from_kepler  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Orbit precision constructor is unavailable")
    else:
        raise ValueError(f"Unsupported seed mode: {seed.mode!r}")

    out: list[Orbit] = []
    two_pi = 2.0 * math.pi

    for plane_idx in range(p):
        d_raan = raan_span * (float(plane_idx) / float(p))
        for slot_idx in range(sats_per_plane):
            if (not include_seed) and plane_idx == 0 and slot_idx == 0:
                continue

            # Walker Delta/Star mean anomaly phasing:
            # M = M0 + slot*(2π/s) + plane*(F*2π/T)
            d_mean = two_pi * (
                float(slot_idx) / float(sats_per_plane)
                + float(f * plane_idx) / float(t)
            )
            kwargs = dict(base_kwargs)
            kwargs["raan"] = _wrap_pm_pi(seed_raan + d_raan)
            kwargs["anomaly"] = _wrap_pm_pi(seed_mean + d_mean)
            out.append(make_orbit(**kwargs))

    return out
