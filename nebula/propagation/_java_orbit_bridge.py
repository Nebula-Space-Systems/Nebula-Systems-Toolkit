from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import shutil
import subprocess
import threading
import warnings

import jdk4py


JAVA_ORBIT_BRIDGE_CLASS = "com.nebula.orekit.OrekitOrbitBridge"

_BUILD_LOCK = threading.Lock()
_BUILD_DONE = False
_BUILD_CLASSPATH: Optional[str] = None
_BUILD_ERROR: Optional[Exception] = None


def _source_file() -> Path:
    return (
        Path(__file__).resolve().parent
        / "java"
        / "com"
        / "nebula"
        / "orekit"
        / "OrekitOrbitBridge.java"
    )


def _classes_dir() -> Path:
    base = os.environ.get("NEBULA_OREKIT_JAVA_BUILD_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "classes"

    if os.name == "nt":
        root = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
        return Path(root) / "nebula" / "orekit-java-bridge" / "classes"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "nebula" / "orekit-java-bridge" / "classes"
    return Path.home() / ".cache" / "nebula" / "orekit-java-bridge" / "classes"


def _prebuilt_jar() -> Optional[Path]:
    jar = Path(__file__).resolve().parent / "java" / "OrekitOrbitBridge.jar"
    return jar if jar.exists() else None


def _class_file(classes_dir: Path) -> Path:
    return classes_dir / "com" / "nebula" / "orekit" / "OrekitOrbitBridge.class"


def _javac_path() -> Optional[Path]:
    exe = "javac.exe" if os.name == "nt" else "javac"

    # 1) Preferred: co-located JDK in jdk4py (if present in the environment).
    java_home = Path(str(jdk4py.JAVA_HOME))
    p = java_home / "bin" / exe
    if p.exists():
        return p

    # 2) User/system JAVA_HOME.
    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home:
        p = Path(env_java_home) / "bin" / exe
        if p.exists():
            return p

    # 3) PATH lookup.
    found = shutil.which("javac")
    if found:
        return Path(found)
    return None


def _orekit_jars_glob() -> Optional[str]:
    try:
        import orekit_jpype
    except Exception:
        return None

    jars_dir = Path(orekit_jpype.__file__).resolve().parent / "jars"
    if not jars_dir.exists():
        return None
    return str(jars_dir / "*")


def _needs_rebuild(source: Path, class_file: Path) -> bool:
    if not class_file.exists():
        return True
    try:
        return source.stat().st_mtime_ns > class_file.stat().st_mtime_ns
    except Exception:
        return True


def prepare_java_orbit_bridge_classpath() -> Optional[str]:
    """
    Compile and return classpath for the optional Java orbit bridge backend.

    Returns
    -------
    str | None
        Directory containing compiled classes to be appended to JPype classpath,
        or ``None`` when the backend is unavailable.
    """
    global _BUILD_DONE, _BUILD_CLASSPATH, _BUILD_ERROR

    if os.environ.get("NEBULA_OREKIT_JAVA_BACKEND", "1").strip() in {"0", "false", "False"}:
        return None

    if _BUILD_DONE:
        return _BUILD_CLASSPATH

    with _BUILD_LOCK:
        if _BUILD_DONE:
            return _BUILD_CLASSPATH

        source = _source_file()

        # Prefer shipped JARs so end users do not need a local Java compiler.
        prebuilt = _prebuilt_jar()
        force_compile = os.environ.get("NEBULA_OREKIT_JAVA_FORCE_COMPILE", "").strip() in {
            "1",
            "true",
            "True",
        }
        if prebuilt is not None and not force_compile:
            _BUILD_DONE = True
            _BUILD_CLASSPATH = str(prebuilt)
            return _BUILD_CLASSPATH

        classes_dir = _classes_dir()
        class_file = _class_file(classes_dir)

        if not source.exists():
            _BUILD_DONE = True
            _BUILD_CLASSPATH = None
            return None

        javac = _javac_path()
        jars_glob = _orekit_jars_glob()
        if javac is None or jars_glob is None:
            _BUILD_DONE = True
            _BUILD_CLASSPATH = None
            return None

        try:
            classes_dir.mkdir(parents=True, exist_ok=True)
            if _needs_rebuild(source, class_file):
                cmd = [
                    str(javac),
                    "-encoding",
                    "UTF-8",
                    "-cp",
                    jars_glob,
                    "-d",
                    str(classes_dir),
                    str(source),
                ]
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            _BUILD_CLASSPATH = str(classes_dir)
        except Exception as exc:  # pragma: no cover
            _BUILD_ERROR = exc
            _BUILD_CLASSPATH = None
            warnings.warn(
                "Nebula Java Orekit bridge could not be compiled; "
                "falling back to Python-side Orekit loops.",
                RuntimeWarning,
            )
        finally:
            _BUILD_DONE = True

        return _BUILD_CLASSPATH
