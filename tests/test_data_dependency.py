from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nstk import _data_dependency


def test_installed_data_dependency_resolves_directories(monkeypatch, tmp_path: Path) -> None:
    cartopy_dir = tmp_path / "cartopy"
    orekit_dir = tmp_path / "orekit-data"
    cartopy_dir.mkdir()
    orekit_dir.mkdir()

    fake_module = SimpleNamespace(
        get_cartopy_data_dir=lambda: cartopy_dir,
        get_orekit_data_dir=lambda: orekit_dir,
    )
    monkeypatch.setattr(_data_dependency, "import_module", lambda name: fake_module)

    assert _data_dependency.get_installed_cartopy_data_dir() == cartopy_dir.resolve()
    assert _data_dependency.get_installed_orekit_data_dir() == orekit_dir.resolve()


def test_missing_data_dependency_raises_helpful_error(monkeypatch) -> None:
    def _raise(name: str):
        raise ModuleNotFoundError("nstk_data")

    monkeypatch.setattr(_data_dependency, "import_module", _raise)

    with pytest.raises(ModuleNotFoundError, match="nstk-data"):
        _data_dependency.get_installed_orekit_data_dir()
