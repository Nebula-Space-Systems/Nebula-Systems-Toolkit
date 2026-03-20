from __future__ import annotations

from pathlib import Path

import pytest

import nstk
from nstk import _orekit_runtime


def test_set_orekit_data_path_configures_lazy_runtime(monkeypatch, tmp_path: Path) -> None:
    orekit_dir = tmp_path / "orekit-data"
    orekit_dir.mkdir()

    monkeypatch.setattr(_orekit_runtime, "_RUNTIME_READY", False)
    monkeypatch.setattr(_orekit_runtime, "_CONFIGURED_DATA_PATH", None)
    monkeypatch.setattr(_orekit_runtime, "_ACTIVE_DATA_PATH", None)

    nstk.set_orekit_data_path(str(orekit_dir))

    assert _orekit_runtime._CONFIGURED_DATA_PATH == orekit_dir.resolve()
    assert _orekit_runtime._ACTIVE_DATA_PATH is None


def test_set_orekit_data_path_rejects_switch_after_runtime_init(monkeypatch, tmp_path: Path) -> None:
    active_dir = tmp_path / "active"
    new_dir = tmp_path / "new"
    active_dir.mkdir()
    new_dir.mkdir()

    monkeypatch.setattr(_orekit_runtime, "_RUNTIME_READY", True)
    monkeypatch.setattr(_orekit_runtime, "_CONFIGURED_DATA_PATH", active_dir.resolve())
    monkeypatch.setattr(_orekit_runtime, "_ACTIVE_DATA_PATH", active_dir.resolve())

    with pytest.raises(RuntimeError, match="already initialized"):
        nstk.set_orekit_data_path(str(new_dir))
