from importlib import import_module

from nebula.propagation.orbit import Orbit, initialize_orekit

__all__ = ["Orbit", "initialize_orekit", "orbit"]


def __getattr__(name: str):
    if name == "orbit":
        mod = import_module("nebula.propagation.orbit")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nebula.propagation' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))

