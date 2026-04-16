"""Shared Orekit frame-resolution helpers used across NSTK modules.

This module centralizes Orekit runtime binding for frame enums/factories and
the string-to-frame resolution logic used by both propagation and transform
APIs. Keeping one implementation avoids semantic drift between modules.
"""

from __future__ import annotations

from typing import Any, Literal


_RUNTIME_BOUND = False
FramesFactory = None
ITRFVersion = None
Predefined = None
IERSConventions = None


def _bind_java() -> None:
    """Bind Orekit frame classes lazily after starting the JVM."""

    global _RUNTIME_BOUND
    global FramesFactory, ITRFVersion, Predefined, IERSConventions

    if _RUNTIME_BOUND:
        return

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory as _FramesFactory  # type: ignore
    from org.orekit.frames import ITRFVersion as _ITRFVersion  # type: ignore
    from org.orekit.frames import Predefined as _Predefined  # type: ignore
    from org.orekit.utils import IERSConventions as _IERSConventions  # type: ignore

    FramesFactory = _FramesFactory
    ITRFVersion = _ITRFVersion
    Predefined = _Predefined
    IERSConventions = _IERSConventions
    _RUNTIME_BOUND = True


def _normalize_frame_name(name: str) -> str:
    """Normalize frame-like strings to a compact comparison key."""

    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _iers_default():
    """Return NSTK's default IERS convention for propagation workflows."""

    _bind_java()
    return IERSConventions.IERS_2010


def _supported_iers_suffixes() -> list[str]:
    """Return sorted IERS convention year suffixes exposed by Orekit."""

    _bind_java()
    return sorted(
        (
            attr.removeprefix("IERS_")
            for attr in dir(IERSConventions)
            if attr.startswith("IERS_") and attr.removeprefix("IERS_").isdigit()
        ),
        key=int,
    )


def _latest_iers_convention():
    """Return the newest available Orekit IERS convention enum."""

    suffixes = _supported_iers_suffixes()
    if not suffixes:
        raise RuntimeError("No IERS conventions are available from the Orekit runtime.")
    return getattr(IERSConventions, f"IERS_{suffixes[-1]}")


def _coerce_iers(
    iers_convention: Any,
    *,
    when_none: Literal["iers_2010", "latest"] = "iers_2010",
) -> Any:
    """Resolve a possibly-missing IERS convention using an explicit default policy."""

    if iers_convention is not None:
        return iers_convention

    if when_none == "iers_2010":
        return _iers_default()
    if when_none == "latest":
        return _latest_iers_convention()
    raise ValueError("when_none must be 'iers_2010' or 'latest'")


def _resolve_predefined_frame(key: str):
    """Resolve an Orekit ``Predefined`` frame constant by normalized name."""

    _bind_java()
    for attr in dir(Predefined):
        if not attr.isupper():
            continue
        if _normalize_frame_name(attr) == key:
            return FramesFactory.getFrame(getattr(Predefined, attr))
    return None


def _supported_itrf_suffixes() -> list[str]:
    """Return sorted ITRF version year suffixes exposed by Orekit."""

    _bind_java()
    return sorted(
        (
            attr.removeprefix("ITRF_")
            for attr in dir(ITRFVersion)
            if attr.startswith("ITRF_") and attr.removeprefix("ITRF_").isdigit()
        ),
        key=int,
    )


def _latest_itrf_version():
    """Return the newest available Orekit ITRF version enum."""

    suffixes = _supported_itrf_suffixes()
    if not suffixes:
        raise RuntimeError("No ITRF versions are available from the Orekit runtime.")
    return getattr(ITRFVersion, f"ITRF_{suffixes[-1]}")


def _coerce_iers_from_suffix(name: str, suffix: str):
    """Resolve an explicit IERS suffix from a frame-string suffix."""

    _bind_java()
    attr = f"IERS_{suffix}"
    convention = getattr(IERSConventions, attr, None)
    if convention is None:
        raise ValueError(
            f"Unsupported IERS convention in frame string: '{name}'. "
            f"Supported convention suffixes are {', '.join(_supported_iers_suffixes())}."
        )
    return convention


def _resolve_iers_versioned_frame(name: str, key: str, *, simple_eop: bool):
    """Resolve frame families parameterized by IERS convention year."""

    _bind_java()

    prefix_map = {
        "itrfequinox": lambda iers: FramesFactory.getITRFEquinox(iers, bool(simple_eop)),
        "itrfcio": lambda iers: FramesFactory.getITRF(iers, bool(simple_eop)),
        "ecliptic": lambda iers: FramesFactory.getEcliptic(iers),
        "cirf": lambda iers: FramesFactory.getCIRF(iers, bool(simple_eop)),
        "gtod": lambda iers: FramesFactory.getGTOD(iers, bool(simple_eop)),
        "tirf": lambda iers: FramesFactory.getTIRF(iers, bool(simple_eop)),
        "mod": lambda iers: FramesFactory.getMOD(iers),
        "tod": lambda iers: FramesFactory.getTOD(iers, bool(simple_eop)),
    }

    for prefix, factory in prefix_map.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix:
            return factory(_latest_iers_convention())
        if suffix.isdigit():
            return factory(_coerce_iers_from_suffix(name, suffix))
    return None


def _resolve_itrf_frame(name: str, key: str, *, iers, simple_eop: bool):
    """Resolve ITRF/ITRS/ECEF aliases with optional explicit ITRF versions."""

    _bind_java()

    for prefix in ("itrf", "itrs", "ecef"):
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix) :]
        if not suffix:
            return FramesFactory.getITRF(_latest_itrf_version(), iers, bool(simple_eop))

        if suffix.isdigit() and len(suffix) == 4:
            version = getattr(ITRFVersion, f"ITRF_{suffix}", None)
            if version is not None:
                return FramesFactory.getITRF(version, iers, bool(simple_eop))

            raise ValueError(
                f"Unsupported ITRF frame string: '{name}'. "
                f"Supported ITRF versions are: {', '.join(_supported_itrf_suffixes())}."
            )

    return None


def _resolve_named_frame(name: str, *, iers, simple_eop: bool):
    """Resolve a supported frame name string into an Orekit ``Frame`` object.

    Supported string forms include:
    - inertial aliases like ``gcrf``, ``icrf``, ``eme2000``, ``teme``
    - ITRF-family aliases, optionally versioned:
      ``itrf``, ``itrs``, ``ecef``, ``itrf2014``
    - IERS-versioned frame families:
      ``mod2003``, ``tod2010``, ``cirf1996``, ``gtod2010``,
      ``tirf2003``, ``ecliptic2010``, ``itrfcio2010``,
      ``itrfequinox2003``
    - Orekit ``Predefined`` enum names directly

    When a versioned family is requested without an explicit suffix, the newest
    available Orekit version is used for that family.
    """

    _bind_java()
    key = _normalize_frame_name(name)

    aliases = {
        "j2000": "eme2000",
        "gcrs": "gcrf",
        "itrs": "itrf",
        "ecef": "itrf",
        "eci": "gcrf",
        "veis": "veis1950",
        "veis50": "veis1950",
        "meanofdate": "mod",
        "trueofdate": "tod",
    }
    key = aliases.get(key, key)

    predefined_frame = _resolve_predefined_frame(key)
    if predefined_frame is not None:
        return predefined_frame

    iers_versioned_frame = _resolve_iers_versioned_frame(
        name,
        key,
        simple_eop=bool(simple_eop),
    )
    if iers_versioned_frame is not None:
        return iers_versioned_frame

    itrf_frame = _resolve_itrf_frame(name, key, iers=iers, simple_eop=bool(simple_eop))
    if itrf_frame is not None:
        return itrf_frame

    if key == "gcrf":
        return FramesFactory.getGCRF()
    if key == "icrf":
        return FramesFactory.getICRF()
    if key == "eme2000":
        return FramesFactory.getEME2000()
    if key == "teme":
        return FramesFactory.getTEME()
    if key in ("veis1950",):
        return FramesFactory.getVeis1950()
    raise ValueError(
        f"Unsupported frame string: '{name}'. "
        "Use Orekit Frame objects for custom frames."
    )


def _resolve_frame(frame_like: Any, *, iers, simple_eop: bool):
    """Resolve a frame-like input from string alias or pass through an Orekit frame."""

    if isinstance(frame_like, str):
        return _resolve_named_frame(frame_like, iers=iers, simple_eop=bool(simple_eop))
    return frame_like


__all__ = [
    "FramesFactory",
    "IERSConventions",
    "ITRFVersion",
    "Predefined",
    "_bind_java",
    "_coerce_iers",
    "_coerce_iers_from_suffix",
    "_iers_default",
    "_latest_iers_convention",
    "_latest_itrf_version",
    "_normalize_frame_name",
    "_resolve_frame",
    "_resolve_iers_versioned_frame",
    "_resolve_itrf_frame",
    "_resolve_named_frame",
    "_resolve_predefined_frame",
    "_supported_iers_suffixes",
    "_supported_itrf_suffixes",
]
