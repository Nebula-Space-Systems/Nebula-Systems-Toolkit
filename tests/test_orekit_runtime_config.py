from __future__ import annotations

from pathlib import Path

import pytest

import nstk
from nstk import _orekit_runtime
from nstk.plotting import _cartopy_data


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


def test_initialize_bootstraps_orekit_and_cartopy(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_ensure_runtime(*, data_path=None) -> None:
        calls["orekit_data_path"] = data_path

    def _fake_configure_cartopy(*, data_dir=None) -> Path:
        calls["cartopy_data_path"] = data_dir
        return Path("/tmp/fake-cartopy")

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", _fake_ensure_runtime)
    monkeypatch.setattr(_cartopy_data, "configure_cartopy_data_dir", _fake_configure_cartopy)

    nstk.initialize()

    assert calls == {
        "orekit_data_path": None,
        "cartopy_data_path": None,
    }


def test_initialize_accepts_explicit_orekit_and_cartopy_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orekit_dir = tmp_path / "orekit-data"
    cartopy_dir = tmp_path / "cartopy-data"
    orekit_dir.mkdir()
    cartopy_dir.mkdir()

    calls: dict[str, Path] = {}

    def _fake_ensure_runtime(*, data_path=None) -> None:
        calls["orekit_data_path"] = Path(data_path)

    def _fake_configure_cartopy(*, data_dir=None) -> Path:
        calls["cartopy_data_path"] = Path(data_dir)
        return Path(data_dir)

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", _fake_ensure_runtime)
    monkeypatch.setattr(_cartopy_data, "configure_cartopy_data_dir", _fake_configure_cartopy)

    nstk.initialize(
        orekit_data_path=orekit_dir,
        cartopy_data_path=cartopy_dir,
    )

    assert calls == {
        "orekit_data_path": orekit_dir,
        "cartopy_data_path": cartopy_dir,
    }


def test_initialize_offline_configures_astropy_iers(monkeypatch) -> None:
    iers = pytest.importorskip("astropy.utils.iers")
    original = {
        "auto_download": iers.conf.auto_download,
        "auto_max_age": iers.conf.auto_max_age,
        "remote_timeout": iers.conf.remote_timeout,
        "iers_degraded_accuracy": iers.conf.iers_degraded_accuracy,
    }

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", lambda *, data_path=None: None)
    monkeypatch.setattr(
        _cartopy_data,
        "configure_cartopy_data_dir",
        lambda *, data_dir=None: Path("/tmp/fake-cartopy"),
    )
    monkeypatch.setattr(nstk, "_ASTROPY_IERS_DEFAULTS", None)

    try:
        nstk.initialize(offline=True)

        assert iers.conf.auto_download is False
        assert iers.conf.auto_max_age is None
        assert float(iers.conf.remote_timeout) == pytest.approx(1.0)
        assert iers.conf.iers_degraded_accuracy == "warn"
    finally:
        iers.conf.auto_download = original["auto_download"]
        iers.conf.auto_max_age = original["auto_max_age"]
        iers.conf.remote_timeout = original["remote_timeout"]
        iers.conf.iers_degraded_accuracy = original["iers_degraded_accuracy"]


def test_initialize_repeated_calls_restore_astropy_iers_defaults(monkeypatch) -> None:
    iers = pytest.importorskip("astropy.utils.iers")
    original = {
        "auto_download": iers.conf.auto_download,
        "auto_max_age": iers.conf.auto_max_age,
        "remote_timeout": iers.conf.remote_timeout,
        "iers_degraded_accuracy": iers.conf.iers_degraded_accuracy,
    }

    monkeypatch.setattr(_orekit_runtime, "ensure_orekit_runtime", lambda *, data_path=None: None)
    monkeypatch.setattr(
        _cartopy_data,
        "configure_cartopy_data_dir",
        lambda *, data_dir=None: Path("/tmp/fake-cartopy"),
    )
    monkeypatch.setattr(nstk, "_ASTROPY_IERS_DEFAULTS", None)

    try:
        nstk.initialize(offline=True)
        nstk.initialize(offline=True)
        nstk.initialize(offline=False)

        assert iers.conf.auto_download == original["auto_download"]
        assert iers.conf.auto_max_age == original["auto_max_age"]
        assert float(iers.conf.remote_timeout) == float(original["remote_timeout"])
        assert iers.conf.iers_degraded_accuracy == original["iers_degraded_accuracy"]
    finally:
        iers.conf.auto_download = original["auto_download"]
        iers.conf.auto_max_age = original["auto_max_age"]
        iers.conf.remote_timeout = original["remote_timeout"]
        iers.conf.iers_degraded_accuracy = original["iers_degraded_accuracy"]
