"""Centralized Orekit/JVM runtime initialization for Nebula."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import threading

import jdk4py
import jpype
import orekit_jpype


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_READY = False
_RUNTIME_FAULTHANDLER_DISABLED = False


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "orekit-data"


def _unique_classpaths(*paths: Optional[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        p = str(path)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def ensure_orekit_runtime(*, data_path: Optional[str] = None) -> None:
    """Initialize JVM + Orekit data and register all Nebula Java bridge classpaths.

    This function is idempotent and is the single runtime initialization path
    for Orekit-backed features in Nebula.
    """

    global _RUNTIME_READY, _RUNTIME_FAULTHANDLER_DISABLED

    if _RUNTIME_READY:
        return

    with _RUNTIME_LOCK:
        if _RUNTIME_READY:
            return

        os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))

        # On Windows, CPython faulthandler + embedded JVM (JPype/Orekit)
        # can produce spurious access-violation crashes during teardown.
        if os.name == "nt":
            try:
                import faulthandler

                if faulthandler.is_enabled():
                    faulthandler.disable()
                    _RUNTIME_FAULTHANDLER_DISABLED = True
            except Exception:
                pass

        from nebula.propagation._orbit_propagation_bridge import (
            prepare_orbit_propagation_bridge_classpath,
        )
        from nebula.transforms._timed_rotations_java_bridge import (
            prepare_timed_rotations_bridge_classpath,
        )

        cp_orbit = prepare_orbit_propagation_bridge_classpath()
        cp_timed = prepare_timed_rotations_bridge_classpath()
        classpaths = _unique_classpaths(cp_orbit, cp_timed)

        # Ensure bridge classpaths are visible even when JVM was already started.
        if jpype.isJVMStarted():
            for cp in classpaths:
                try:
                    jpype.addClassPath(cp)
                except Exception:
                    pass

        orekit_jpype.initVM(additional_classpaths=classpaths if classpaths else None)

        from orekit_jpype.pyhelpers import setup_orekit_curdir

        setup_orekit_curdir(
            filename=str(Path(data_path).resolve() if data_path else _default_data_path().resolve())
        )

        # Redundant but harmless: keep classpaths attached post-init too.
        if jpype.isJVMStarted():
            for cp in classpaths:
                try:
                    jpype.addClassPath(cp)
                except Exception:
                    pass

        _RUNTIME_READY = True
