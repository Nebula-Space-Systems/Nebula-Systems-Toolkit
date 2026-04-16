from importlib import import_module
from typing import Any

from nstk.propagation.orbit import Orbit, SampledStates
from nstk.propagation.walker import (
    build_numerical_walker_constellation,
    build_two_body_walker_constellation,
    build_walker_constellation,
    build_walker_initial_states,
)

__all__ = [
    "Orbit",
    "SampledStates",
    "build_walker_initial_states",
    "build_walker_constellation",
    "build_two_body_walker_constellation",
    "build_numerical_walker_constellation",
    "orbit",
]


def __getattr__(name: str) -> Any:
    if name == "orbit":
        mod = import_module("nstk.propagation.orbit")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nstk.propagation' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
