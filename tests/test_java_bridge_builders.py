from __future__ import annotations

from pathlib import Path
import warnings

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
