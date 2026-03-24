from __future__ import annotations

from importlib import import_module
from typing import Any

_MAP_EXPORTS = {
    "MapStyle",
    "MapView",
    "CFeatureScale",
    "DARK_DETAILED_NO_GRID_STYLE",
    "DARK_DETAILED_STYLE",
    "DARK",
    "DARK_DETAILED",
    "DARK_DETAILED_NO_GRID",
    "DARK_NO_GRID",
    "DARK_NO_GRID_STYLE",
    "DARK_RASTER",
    "DARK_RASTER_NO_GRID",
    "DARK_RASTER_NO_GRID_STYLE",
    "DARK_RASTER_STYLE",
    "DARK_STYLE",
    "ExtentConfig",
    "LIGHT_DETAILED_NO_GRID_STYLE",
    "LIGHT_DETAILED_STYLE",
    "LIGHT",
    "LIGHT_DETAILED",
    "LIGHT_DETAILED_NO_GRID",
    "LIGHT_NO_GRID",
    "LIGHT_NO_GRID_STYLE",
    "LIGHT_RASTER",
    "LIGHT_RASTER_NO_GRID",
    "LIGHT_RASTER_NO_GRID_STYLE",
    "LIGHT_RASTER_STYLE",
    "LIGHT_STYLE",
    "MapConfig",
    "ProjectionConfig",
    "add_geodesic_trace",
    "available_map_styles",
    "compile_map_config",
    "get_map_style",
    "make_basemap",
    "register_map_style",
}
_GEO_EXPORTS = {"GeoMap", "MapLayer", "get_map_preset"}
_COUNTRY_EXPORTS = {
    "country_geometries",
    "country_geometry",
    "fuzzy_find_countries",
    "fuzzy_find_country_record",
}
_ORBIT_EXPORTS = {"plot_orbits"}
_COVERAGE_EXPORTS = {
    "plot_coverage_ecdf",
    "plot_coverage_histogram",
    "plot_coverage_map",
    "plot_coverage_small_multiples",
    "plot_target_timeline",
}

__all__ = sorted(_MAP_EXPORTS | _GEO_EXPORTS | _COUNTRY_EXPORTS | _ORBIT_EXPORTS | _COVERAGE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _MAP_EXPORTS:
        mod = import_module("nstk.plotting.map")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _GEO_EXPORTS:
        mod = import_module("nstk.plotting.geo")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _COUNTRY_EXPORTS:
        mod = import_module("nstk.plotting.country_shapes")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _ORBIT_EXPORTS:
        mod = import_module("nstk.plotting.orbits")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _COVERAGE_EXPORTS:
        mod = import_module("nstk.plotting.coverage")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'nstk.plotting' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
