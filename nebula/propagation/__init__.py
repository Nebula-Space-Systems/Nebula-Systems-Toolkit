from importlib import import_module
from typing import Any

from nebula.propagation.orbit import Orbit, initialize_orekit
from nebula.propagation.walker import build_walker_constellation

__all__ = ["Orbit", "initialize_orekit", "build_walker_constellation", "orbit"]


def __getattr__(name: str) -> Any:
    if name == "orbit":
        mod = import_module("nebula.propagation.orbit")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nebula.propagation' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
