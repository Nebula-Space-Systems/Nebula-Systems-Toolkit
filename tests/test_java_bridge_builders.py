from __future__ import annotations

from pathlib import Path
import warnings
import zipfile

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from nstk import _orekit_runtime
from nstk.propagation import _attitude_provider_java as attitude_bridge
from nstk.propagation import _orbit_propagation_bridge as orbit_bridge
from nstk.transforms import _timed_rotations_java_bridge as timed_bridge


def test_orbit_bridge_uses_prebuilt_jar_without_warning_when_javac_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(orbit_bridge, "_BUILD_DONE", False)
    monkeypatch.setattr(orbit_bridge, "_BUILD_CLASSPATH", None)
    monkeypatch.setattr(orbit_bridge, "_javac_path", lambda: Path("/usr/bin/javac"))
    monkeypatch.setattr(orbit_bridge, "_command_is_usable", lambda path: False)
    monkeypatch.setattr(orbit_bridge, "_needs_rebuild", lambda source, class_file, prebuilt: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classpath = orbit_bridge.prepare_orbit_propagation_bridge_classpath()

    assert classpath == str(orbit_bridge._prebuilt_jar())
    assert caught == []


def test_timed_rotation_bridge_uses_prebuilt_jar_without_warning_when_javac_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(timed_bridge, "_BUILD_DONE", False)
    monkeypatch.setattr(timed_bridge, "_BUILD_CLASSPATH", None)
    monkeypatch.setattr(timed_bridge, "_javac_path", lambda: Path("/usr/bin/javac"))
    monkeypatch.setattr(timed_bridge, "_command_is_usable", lambda path: False)
    monkeypatch.setattr(timed_bridge, "_needs_rebuild", lambda source, class_file, prebuilt: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classpath = timed_bridge.prepare_timed_rotations_bridge_classpath()

    assert classpath == str(timed_bridge._prebuilt_jar())
    assert caught == []


def test_attitude_provider_bridge_uses_prebuilt_jar_without_warning_when_javac_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(attitude_bridge, "_BUILD_DONE", False)
    monkeypatch.setattr(attitude_bridge, "_BUILD_CLASSPATH", None)
    monkeypatch.setattr(attitude_bridge, "_javac_path", lambda: Path("/usr/bin/javac"))
    monkeypatch.setattr(attitude_bridge, "_command_is_usable", lambda path: False)
    monkeypatch.setattr(attitude_bridge, "_needs_rebuild", lambda source, class_file, prebuilt: True)
    monkeypatch.setattr(attitude_bridge, "_has_incompatible_artifact", lambda class_file, prebuilt: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classpath = attitude_bridge.prepare_attitude_providers_classpath()

    assert classpath == str(attitude_bridge._prebuilt_jar())
    assert caught == []


def test_orbit_bridge_falls_back_to_legacy_java_package(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", lambda: None)

    def fake_jclass(name: str):
        calls.append(name)
        if name == orbit_bridge.JAVA_ORBIT_PROPAGATION_CLASS:
            raise TypeError("missing renamed class")
        return f"class:{name}"

    monkeypatch.setattr(orbit_bridge.jpype, "JClass", fake_jclass)

    result = orbit_bridge.get_orbit_propagation_bridge_class()

    assert result == f"class:{orbit_bridge._LEGACY_JAVA_ORBIT_PROPAGATION_CLASS}"
    assert calls == [
        orbit_bridge.JAVA_ORBIT_PROPAGATION_CLASS,
        orbit_bridge._LEGACY_JAVA_ORBIT_PROPAGATION_CLASS,
    ]


def test_timed_rotation_bridge_falls_back_to_legacy_java_package(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", lambda: None)

    def fake_jclass(name: str):
        calls.append(name)
        if name == timed_bridge.JAVA_TIMED_ROTATION_CLASS:
            raise TypeError("missing renamed class")
        return f"class:{name}"

    monkeypatch.setattr(timed_bridge.jpype, "JClass", fake_jclass)

    result = timed_bridge.get_timed_rotation_bridge_class()

    assert result == f"class:{timed_bridge._LEGACY_JAVA_TIMED_ROTATION_CLASS}"
    assert calls == [
        timed_bridge.JAVA_TIMED_ROTATION_CLASS,
        timed_bridge._LEGACY_JAVA_TIMED_ROTATION_CLASS,
    ]


def test_attitude_provider_bridge_loads_java_class(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", lambda: None)

    def fake_jclass(name: str):
        calls.append(name)
        return f"class:{name}"

    monkeypatch.setattr(attitude_bridge.jpype, "JClass", fake_jclass)

    result = attitude_bridge.get_rate_limited_yaw_provider_class()

    assert result == f"class:{attitude_bridge.JAVA_RATE_LIMITED_YAW_PROVIDER_CLASS}"
    assert calls == [attitude_bridge.JAVA_RATE_LIMITED_YAW_PROVIDER_CLASS]


def test_attitude_provider_bridge_rebuilds_when_prebuilt_jar_is_newer_java(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(attitude_bridge, "_BUILD_DONE", False)
    monkeypatch.setattr(attitude_bridge, "_BUILD_CLASSPATH", None)

    source = tmp_path / "RateLimitedYawSteeringProvider.java"
    source.write_text("class X {}", encoding="utf-8")
    classes_dir = tmp_path / "classes"
    class_file = classes_dir / "com" / "nstk" / "attitudes" / "RateLimitedYawSteeringProvider.class"
    class_file.parent.mkdir(parents=True, exist_ok=True)
    class_file.write_bytes(b"\xca\xfe\xba\xbe\x00\x00\x00\x41")

    prebuilt = tmp_path / "NSTKAttitudeProviders.jar"
    with zipfile.ZipFile(prebuilt, mode="w") as zf:
        zf.writestr(
            "com/nstk/attitudes/RateLimitedYawSteeringProvider.class",
            b"\xca\xfe\xba\xbe\x00\x00\x00\x41",
        )

    monkeypatch.setattr(attitude_bridge, "_source_file", lambda: source)
    monkeypatch.setattr(attitude_bridge, "_classes_dir", lambda: classes_dir)
    monkeypatch.setattr(attitude_bridge, "_class_file", lambda _: class_file)
    monkeypatch.setattr(attitude_bridge, "_prebuilt_jar", lambda: prebuilt)
    monkeypatch.setattr(attitude_bridge, "_javac_path", lambda: Path("/usr/bin/javac"))
    monkeypatch.setattr(attitude_bridge, "_command_is_usable", lambda path: True)
    monkeypatch.setattr(attitude_bridge, "_orekit_jars_glob", lambda: "/tmp/fake-jars/*")
    monkeypatch.setattr(attitude_bridge, "_jar_path", lambda: None)

    compile_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        compile_calls.append(cmd)

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    monkeypatch.setattr(attitude_bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(
        attitude_bridge, "_create_jar_with_zipfile", lambda classes, jar: jar.write_bytes(b"jar")
    )

    classpath = attitude_bridge.prepare_attitude_providers_classpath()

    assert classpath == str(prebuilt)
    assert len(compile_calls) == 1
    assert compile_calls[0][0] == "/usr/bin/javac"
    assert "--release" in compile_calls[0]
    assert "17" in compile_calls[0]


def test_pyproject_force_includes_all_prebuilt_java_bridges() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert (
        "nstk/propagation/_java_orbit_propagation/OrekitOrbitPropagationBridge.jar"
        in force_include
    )
    assert "nstk/propagation/_java_attitude_providers/NSTKAttitudeProviders.jar" in force_include
    assert "nstk/transforms/java_timed_rotations/TimedRotationBridge.jar" in force_include


@pytest.mark.parametrize(
    "bridge_module, tool_name",
    [
        (attitude_bridge, "javac"),
        (orbit_bridge, "javac"),
        (timed_bridge, "javac"),
        (attitude_bridge, "jar"),
        (orbit_bridge, "jar"),
        (timed_bridge, "jar"),
    ],
)
def test_java_bridge_prefers_usable_env_java_home_tools(
    monkeypatch,
    tmp_path,
    bridge_module,
    tool_name: str,
) -> None:
    exe = f"{tool_name}.exe" if bridge_module.os.name == "nt" else tool_name

    env_home = tmp_path / "env_home"
    env_tool = env_home / "bin" / exe
    env_tool.parent.mkdir(parents=True, exist_ok=True)
    env_tool.write_text("placeholder", encoding="utf-8")

    jdk4py_home = tmp_path / "jdk4py_home"
    jdk4py_tool = jdk4py_home / "bin" / exe
    jdk4py_tool.parent.mkdir(parents=True, exist_ok=True)
    jdk4py_tool.write_text("placeholder", encoding="utf-8")

    monkeypatch.setenv("JAVA_HOME", str(env_home))
    monkeypatch.setattr(bridge_module.jdk4py, "JAVA_HOME", str(jdk4py_home))
    monkeypatch.setattr(
        bridge_module,
        "_command_is_usable",
        lambda path: Path(path) == env_tool,
    )
    monkeypatch.setattr(bridge_module.shutil, "which", lambda name: None)

    resolver = bridge_module._javac_path if tool_name == "javac" else bridge_module._jar_path
    assert resolver() == env_tool
