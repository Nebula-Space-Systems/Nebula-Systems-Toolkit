from importlib import import_module

_FAST_EXPORTS = {
    "FastOrbit",
    "j2_secular_rates",
    "propagate_constellation_pv",
    "julian_date",
    "sun_position_eci",
    "sun_position_ecef",
    "gmst_angle",
    "EARTH_MU",
    "WGS84_A",
}
_OREKIT_EXPORTS = {
    "Orbit",
    "initialize_orekit",
}

__all__ = sorted(_FAST_EXPORTS | _OREKIT_EXPORTS | {"fast", "orekit"})


def __getattr__(name: str):
    if name == "fast":
        mod = import_module("nebula.propagation.fast")
        globals()[name] = mod
        return mod
    if name == "orekit":
        mod = import_module("nebula.propagation.orekit")
        globals()[name] = mod
        return mod
    if name in _FAST_EXPORTS:
        mod = import_module("nebula.propagation.fast")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name in _OREKIT_EXPORTS:
        mod = import_module("nebula.propagation.orekit")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'nebula.propagation' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
