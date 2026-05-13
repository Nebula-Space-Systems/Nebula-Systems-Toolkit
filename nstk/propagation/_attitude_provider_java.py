"""Build/class-loading helpers for NSTK Java attitude providers."""

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


JAVA_RATE_LIMITED_YAW_PROVIDER_CLASS = "com.nstk.attitudes.RateLimitedYawSteeringProvider"

_BUILD_LOCK = threading.Lock()
_BUILD_DONE = False
_BUILD_CLASSPATH: Optional[str] = None
_JAVA_RELEASE = "17"
_MAX_SUPPORTED_CLASS_MAJOR = 61


def _source_file() -> Path:
    return (
        Path(__file__).resolve().parent
        / "_java_attitude_providers"
        / "com"
        / "nstk"
        / "attitudes"
        / "RateLimitedYawSteeringProvider.java"
    )


def _classes_dir() -> Path:
    return Path(__file__).resolve().parent / "_java_attitude_providers" / ".build_classes"


def _prebuilt_jar() -> Path:
    return Path(__file__).resolve().parent / "_java_attitude_providers" / "NSTKAttitudeProviders.jar"


def _class_file(classes_dir: Path) -> Path:
    return classes_dir / "com" / "nstk" / "attitudes" / "RateLimitedYawSteeringProvider.class"


def _javac_path() -> Optional[Path]:
    exe = "javac.exe" if os.name == "nt" else "javac"
    candidates: list[Path] = []
    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home:
        candidates.append(Path(env_java_home) / "bin" / exe)

    java_home = Path(str(jdk4py.JAVA_HOME))
    candidates.append(java_home / "bin" / exe)

    found = shutil.which("javac")
    if found:
        candidates.append(Path(found))

    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and _command_is_usable(candidate):
            return candidate

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return None


def _jar_path() -> Optional[Path]:
    exe = "jar.exe" if os.name == "nt" else "jar"
    candidates: list[Path] = []
    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home:
        candidates.append(Path(env_java_home) / "bin" / exe)

    java_home = Path(str(jdk4py.JAVA_HOME))
    candidates.append(java_home / "bin" / exe)

    found = shutil.which("jar")
    if found:
        candidates.append(Path(found))

    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and _command_is_usable(candidate):
            return candidate

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return None


def _orekit_jars_glob() -> Optional[str]:
    jars_dir = Path(orekit_jpype.__file__).resolve().parent / "jars"
    if not jars_dir.exists():
        return None
    return str(jars_dir / "*")


def _command_is_usable(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        probe = subprocess.run(
            [str(path), "-version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False
    return probe.returncode == 0


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


def _read_class_major_from_bytes(data: bytes) -> Optional[int]:
    if len(data) < 8:
        return None
    if data[0:4] != b"\xca\xfe\xba\xbe":
        return None
    return int.from_bytes(data[6:8], byteorder="big", signed=False)


def _read_class_major_from_class_file(class_file: Path) -> Optional[int]:
    try:
        data = class_file.read_bytes()
    except Exception:
        return None
    return _read_class_major_from_bytes(data)


def _read_class_major_from_jar(jar_path: Path) -> Optional[int]:
    if not jar_path.exists():
        return None
    try:
        with zipfile.ZipFile(jar_path, mode="r") as zf:
            with zf.open("com/nstk/attitudes/RateLimitedYawSteeringProvider.class") as member:
                return _read_class_major_from_bytes(member.read(8))
    except Exception:
        return None


def _has_incompatible_artifact(class_file: Path, prebuilt: Path) -> bool:
    for major in (
        _read_class_major_from_jar(prebuilt),
        _read_class_major_from_class_file(class_file),
    ):
        if major is not None and major > _MAX_SUPPORTED_CLASS_MAJOR:
            return True
    return False


def _create_jar_with_zipfile(classes_dir: Path, jar_path: Path) -> None:
    with zipfile.ZipFile(jar_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in classes_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(classes_dir).as_posix()
                zf.write(file_path, arcname)


def prepare_attitude_providers_classpath() -> Optional[str]:
    """Build, if needed, and return the classpath entry for NSTK attitude providers."""

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
        needs_rebuild = _needs_rebuild(source, class_file, prebuilt)
        has_incompatible_artifact = _has_incompatible_artifact(class_file, prebuilt)
        needs_rebuild = needs_rebuild or has_incompatible_artifact
        can_compile = _command_is_usable(javac)

        try:
            if needs_rebuild and not can_compile:
                if not prebuilt.exists() and not class_file.exists():
                    warnings.warn(
                        "Failed to build NSTK Java attitude providers: no usable javac was found, "
                        "and no prebuilt attitude-provider artifact is available.",
                        RuntimeWarning,
                    )
                elif has_incompatible_artifact:
                    warnings.warn(
                        "NSTK Java attitude provider artifact is compiled for a newer Java runtime "
                        f"(class major > {_MAX_SUPPORTED_CLASS_MAJOR}). Install a Java 17-compatible "
                        "build toolchain to rebuild it.",
                        RuntimeWarning,
                    )
            elif can_compile and jars_glob and needs_rebuild:
                classes_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        str(javac),
                        "--release",
                        _JAVA_RELEASE,
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
                f"Failed to build NSTK Java attitude providers: {exc}",
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


def get_rate_limited_yaw_provider_class():
    """Return the JPype class handle for ``RateLimitedYawSteeringProvider``."""

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()
    return jpype.JClass(JAVA_RATE_LIMITED_YAW_PROVIDER_CLASS)
