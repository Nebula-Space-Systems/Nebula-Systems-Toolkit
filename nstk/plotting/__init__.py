from __future__ import annotations

from importlib import import_module
from typing import Any

_MAP_EXPORTS = {
    "DARK",
    "DARK_DETAILED",
    "DARK_DETAILED_NO_GRID",
    "DARK_NO_GRID",
    "DARK_RASTER",
    "DARK_RASTER_NO_GRID",
    "LIGHT",
    "LIGHT_DETAILED",
    "LIGHT_DETAILED_NO_GRID",
    "LIGHT_NO_GRID",
    "LIGHT_RASTER",
    "LIGHT_RASTER_NO_GRID",
    "add_geodesic_trace",
    "make_basemap",
}
_ORBIT_EXPORTS = {"plot_orbits"}

__all__ = sorted(_MAP_EXPORTS | _ORBIT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _MAP_EXPORTS:
        mod = import_module("nstk.plotting.map")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    if name in _ORBIT_EXPORTS:
        mod = import_module("nstk.plotting.orbits")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'nstk.plotting' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
