from importlib import import_module
from typing import Any

from nstk.propagation.orbit import Orbit
from nstk.propagation.walker import build_walker_constellation

__all__ = ["Orbit", "build_walker_constellation", "orbit"]


def __getattr__(name: str) -> Any:
    if name == "orbit":
        mod = import_module("nstk.propagation.orbit")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nstk.propagation' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
