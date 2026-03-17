"""Runtime and build utilities for the standalone orbit design Java bridge.

This module owns:
- Java source compilation for ``OrekitOrbitDesignBridge``.
- Packaging compiled classes into a local prebuilt JAR.
- JVM startup and Orekit data initialization.

It is intentionally isolated from the rest of ``nebula.propagation`` so the
Java bridge runtime/build logic remains focused and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import shutil
import subprocess
import threading
import warnings
import zipfile

import jdk4py
import jpype
import orekit_jpype


JAVA_ORBIT_DESIGN_CLASS = "com.nebula.orbitdesign.OrekitOrbitDesignBridge"

_BUILD_LOCK = threading.Lock()
_BUILD_DONE = False
_BUILD_CLASSPATH: Optional[str] = None
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_READY = False
_RUNTIME_FAULTHANDLER_DISABLED = False


def _source_file() -> Path:
    return (
        Path(__file__).resolve().parent
        / "java_orbit_design"
        / "com"
        / "nebula"
        / "orbitdesign"
        / "OrekitOrbitDesignBridge.java"
    )


def _classes_dir() -> Path:
    return Path(__file__).resolve().parent / "java_orbit_design" / ".build_classes"


def _prebuilt_jar() -> Path:
    return Path(__file__).resolve().parent / "java_orbit_design" / "OrekitOrbitDesignBridge.jar"


def _class_file(classes_dir: Path) -> Path:
    return classes_dir / "com" / "nebula" / "orbitdesign" / "OrekitOrbitDesignBridge.class"


def _javac_path() -> Optional[Path]:
    exe = "javac.exe" if os.name == "nt" else "javac"
    java_home = Path(str(jdk4py.JAVA_HOME))
    candidate = java_home / "bin" / exe
    if candidate.exists():
        return candidate

    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home:
        candidate = Path(env_java_home) / "bin" / exe
        if candidate.exists():
            return candidate

    found = shutil.which("javac")
    return Path(found) if found else None


def _jar_path() -> Optional[Path]:
    exe = "jar.exe" if os.name == "nt" else "jar"
    java_home = Path(str(jdk4py.JAVA_HOME))
    candidate = java_home / "bin" / exe
    if candidate.exists():
        return candidate

    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home:
        candidate = Path(env_java_home) / "bin" / exe
        if candidate.exists():
            return candidate

    found = shutil.which("jar")
    return Path(found) if found else None


def _orekit_jars_glob() -> Optional[str]:
    jars_dir = Path(orekit_jpype.__file__).resolve().parent / "jars"
    if not jars_dir.exists():
        return None
    return str(jars_dir / "*")


def _needs_rebuild(source: Path, class_file: Path, prebuilt: Path) -> bool:
    if not prebuilt.exists() or not class_file.exists():
        return True
    try:
        src_time = source.stat().st_mtime_ns
        class_time = class_file.stat().st_mtime_ns
        jar_time = prebuilt.stat().st_mtime_ns
        return src_time > class_time or src_time > jar_time
    except Exception:
        return True


def _create_jar_with_zipfile(classes_dir: Path, jar_path: Path) -> None:
    with zipfile.ZipFile(jar_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in classes_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(classes_dir).as_posix()
                zf.write(file_path, arcname)


def prepare_orbit_design_bridge_classpath() -> Optional[str]:
    """Build (if needed) and return classpath entry for the Java bridge.

    Returns
    -------
    str | None
        Path to prebuilt JAR (preferred) or compiled classes directory.
        Returns ``None`` if build artifacts are unavailable.
    """

    global _BUILD_DONE, _BUILD_CLASSPATH

    if _BUILD_DONE:
        return _BUILD_CLASSPATH

    with _BUILD_LOCK:
        if _BUILD_DONE:
            return _BUILD_CLASSPATH

        source = _source_file()
        prebuilt = _prebuilt_jar()

        if not source.exists():
            _BUILD_DONE = True
            _BUILD_CLASSPATH = None
            return None

        classes_dir = _classes_dir()
        class_file = _class_file(classes_dir)

        javac = _javac_path()
        jar_cmd = _jar_path()
        jars_glob = _orekit_jars_glob()

        try:
            if javac and jars_glob and _needs_rebuild(source, class_file, prebuilt):
                classes_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        str(javac),
                        "-encoding",
                        "UTF-8",
                        "-cp",
                        jars_glob,
                        "-d",
                        str(classes_dir),
                        str(source),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if jar_cmd:
                    subprocess.run(
                        [
                            str(jar_cmd),
                            "--create",
                            "--file",
                            str(prebuilt),
                            "-C",
                            str(classes_dir),
                            ".",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                else:
                    _create_jar_with_zipfile(classes_dir, prebuilt)

        except Exception as exc:
            warnings.warn(
                f"Failed to build orbit design Java bridge: {exc}",
                RuntimeWarning,
            )

        if prebuilt.exists():
            _BUILD_CLASSPATH = str(prebuilt)
        elif class_file.exists():
            _BUILD_CLASSPATH = str(classes_dir)
        else:
            _BUILD_CLASSPATH = None

        _BUILD_DONE = True
        return _BUILD_CLASSPATH


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "orekit-data"


def initialize_orbit_design_runtime(*, data_path: Optional[str] = None) -> None:
    """Initialize JVM + Orekit data for the standalone orbit design stack.

    This function is idempotent.
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

        cp = prepare_orbit_design_bridge_classpath()
        if cp:
            try:
                # If JVM is already running, dynamically append classpath.
                jpype.addClassPath(cp)
            except Exception:
                pass

        orekit_jpype.initVM(additional_classpaths=[cp] if cp else None)

        from orekit_jpype.pyhelpers import setup_orekit_curdir

        setup_orekit_curdir(
            filename=str(Path(data_path).resolve() if data_path else _default_data_path().resolve())
        )

        _RUNTIME_READY = True


def get_orbit_design_bridge_class():
    """Return JPype class handle for ``OrekitOrbitDesignBridge``."""

    initialize_orbit_design_runtime()
    return jpype.JClass(JAVA_ORBIT_DESIGN_CLASS)
