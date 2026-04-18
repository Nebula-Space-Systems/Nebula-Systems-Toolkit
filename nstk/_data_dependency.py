from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType


_MISSING_NSTK_DATA_MESSAGE = (
    "Nebula Space Toolkit's offline data package 'nstk-data' is not installed. "
    "Install it in the same Python environment as nstk with "
    "'python -m pip install nstk-data', or pass an explicit local data path via "
    "nstk.initialize(...), nstk.set_orekit_data_path(...), or the relevant initializer."
)


def _load_nstk_data_module() -> ModuleType:
    try:
        return import_module("nstk_data")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via callers/tests
        raise ModuleNotFoundError(_MISSING_NSTK_DATA_MESSAGE) from exc


def _coerce_directory(path_like: object, description: str) -> Path:
    path = Path(path_like).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Expected {description} at '{path}', but it was not found.")
    return path


def get_installed_cartopy_data_dir() -> Path:
    module = _load_nstk_data_module()
    return _coerce_directory(
        module.get_cartopy_data_dir(),
        "installed nstk-data Cartopy directory",
    )


def get_installed_orekit_data_dir() -> Path:
    module = _load_nstk_data_module()
    return _coerce_directory(
        module.get_orekit_data_dir(),
        "installed nstk-data Orekit directory",
    )


__all__ = [
    "get_installed_cartopy_data_dir",
    "get_installed_orekit_data_dir",
]
