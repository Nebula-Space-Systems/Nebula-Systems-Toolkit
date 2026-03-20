"""Centralized Orekit/JVM runtime initialization for Nebula Space Toolkit."""

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
_CONFIGURED_DATA_PATH: Optional[Path] = None
_ACTIVE_DATA_PATH: Optional[Path] = None


def _default_data_path() -> Path:
    from nstk._data_dependency import get_installed_orekit_data_dir

    return get_installed_orekit_data_dir()


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


def _coerce_data_path(data_path: str | os.PathLike[str]) -> Path:
    path = Path(data_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Expected Orekit data directory at '{path}', but it was not found.")
    return path


def _resolve_requested_data_path(data_path: Optional[str] = None) -> Path:
    if data_path is not None:
        return _coerce_data_path(data_path)
    if _CONFIGURED_DATA_PATH is not None:
        return _CONFIGURED_DATA_PATH
    return _default_data_path().resolve()


def _ensure_runtime_data_path_compatible(requested_path: Path) -> None:
    if _ACTIVE_DATA_PATH is None or _ACTIVE_DATA_PATH == requested_path:
        return
    raise RuntimeError(
        "Nebula Space Toolkit's Orekit runtime is already initialized with data path "
        f"'{_ACTIVE_DATA_PATH}', so it cannot be switched to '{requested_path}' "
        "in the same Python process. Configure the custom data path before first use."
    )


def configure_orekit_data_path(data_path: str | os.PathLike[str]) -> Path:
    """Set a custom Orekit data directory for future lazy runtime initialization.

    This is intended for advanced users. Basic users normally do not need to call
    this, because Nebula Space Toolkit initializes Orekit lazily on first use with the
    installed ``nstk-data`` package.
    """

    global _CONFIGURED_DATA_PATH

    resolved = _coerce_data_path(data_path)
    with _RUNTIME_LOCK:
        _ensure_runtime_data_path_compatible(resolved)
        _CONFIGURED_DATA_PATH = resolved
        return resolved


def ensure_orekit_runtime(*, data_path: Optional[str] = None) -> None:
    """Initialize JVM + Orekit data and register all Nebula Space Toolkit bridge classpaths.

    This function is idempotent and is the single runtime initialization path
    for Orekit-backed features in Nebula Space Toolkit.
    """

    global _RUNTIME_READY, _RUNTIME_FAULTHANDLER_DISABLED
    global _CONFIGURED_DATA_PATH, _ACTIVE_DATA_PATH

    requested_data_path = _resolve_requested_data_path(data_path)

    if _RUNTIME_READY:
        _ensure_runtime_data_path_compatible(requested_data_path)
        return

    with _RUNTIME_LOCK:
        requested_data_path = _resolve_requested_data_path(data_path)
        if _RUNTIME_READY:
            _ensure_runtime_data_path_compatible(requested_data_path)
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

        from nstk.propagation._orbit_propagation_bridge import (
            prepare_orbit_propagation_bridge_classpath,
        )
        from nstk.transforms._timed_rotations_java_bridge import (
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

        setup_orekit_curdir(filename=str(requested_data_path))

        # Redundant but harmless: keep classpaths attached post-init too.
        if jpype.isJVMStarted():
            for cp in classpaths:
                try:
                    jpype.addClassPath(cp)
                except Exception:
                    pass

        _CONFIGURED_DATA_PATH = requested_data_path
        _ACTIVE_DATA_PATH = requested_data_path
        _RUNTIME_READY = True
