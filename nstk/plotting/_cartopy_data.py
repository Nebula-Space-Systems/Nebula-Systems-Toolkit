from __future__ import annotations

import os
from pathlib import Path

import cartopy

from nstk._data_dependency import get_installed_cartopy_data_dir


def _resolve_cartopy_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    if data_dir is None:
        return get_installed_cartopy_data_dir()

    path = Path(data_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Expected Cartopy data directory at '{path}', but it was not found.")
    return path


def get_cartopy_data_dir(*, data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the Cartopy data directory used for NSTK offline plotting assets."""

    return _resolve_cartopy_data_dir(data_dir=data_dir)


def configure_cartopy_data_dir(*, data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Point Cartopy at NSTK's offline data directory and return the resolved path."""

    resolved = _resolve_cartopy_data_dir(data_dir=data_dir)
    cartopy.config["pre_existing_data_dir"] = str(resolved)
    cartopy.config["data_dir"] = str(resolved)
    return resolved


def get_cartopy_raster_path(filename: str) -> Path:
    path = configure_cartopy_data_dir() / "raster" / str(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected local Cartopy raster '{path.name}' at '{path}', but it was not found."
        )
    return path


def get_natural_earth_shapefile(
    *,
    resolution: str,
    category: str,
    name: str,
) -> Path:
    path = (
        configure_cartopy_data_dir()
        / "shapefiles"
        / "natural_earth"
        / str(category)
        / f"ne_{resolution}_{name}.shp"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Required local Cartopy Natural Earth shapefile was not found at "
            f"'{path}'. Orbit plotting is configured to work offline and will not "
            "download Cartopy resources automatically."
        )
    return path


__all__ = [
    "configure_cartopy_data_dir",
    "get_cartopy_data_dir",
    "get_cartopy_raster_path",
    "get_natural_earth_shapefile",
]
