from __future__ import annotations

from pathlib import Path
import warnings

from nebula.propagation import _orbit_propagation_bridge as orbit_bridge
from nebula.transforms import _timed_rotations_java_bridge as timed_bridge


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
