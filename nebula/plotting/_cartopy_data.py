from __future__ import annotations

from pathlib import Path

import cartopy


def get_cartopy_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "cartopy"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Expected local Cartopy data directory at '{data_dir}', but it was not found."
        )
    return data_dir


def configure_cartopy_data_dir() -> Path:
    data_dir = get_cartopy_data_dir()
    cartopy.config["pre_existing_data_dir"] = str(data_dir)
    cartopy.config["data_dir"] = str(data_dir)
    return data_dir


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
