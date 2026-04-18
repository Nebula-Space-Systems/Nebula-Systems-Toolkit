from importlib import import_module
from typing import Any

from nstk.propagation.orbit import Orbit, SampledStates
from nstk.propagation.attitude_providers import (
    RateLimitedYawSteeringProvider,
    build_ideal_nadir_sun_constrained_attitude_provider,
    build_nadir_sun_constrained_attitude_provider,
)
from nstk.propagation.propagator_factories import (
    J2J3J4PropagatorFactory,
    NumericalPropagatorFactory,
    PropagatorFactory,
    TwoBodyPropagatorFactory,
    build_j2_j3_j4_propagator,
    build_numerical_propagator,
    build_two_body_propagator,
)
from nstk.propagation.walker import (
    build_j2_j3_j4_walker_constellation,
    build_numerical_walker_constellation,
    build_two_body_walker_constellation,
    build_walker_constellation,
    build_walker_initial_states,
    build_walker_propagators,
)

__all__ = [
    "Orbit",
    "SampledStates",
    "PropagatorFactory",
    "TwoBodyPropagatorFactory",
    "J2J3J4PropagatorFactory",
    "NumericalPropagatorFactory",
    "RateLimitedYawSteeringProvider",
    "build_ideal_nadir_sun_constrained_attitude_provider",
    "build_nadir_sun_constrained_attitude_provider",
    "build_two_body_propagator",
    "build_j2_j3_j4_propagator",
    "build_numerical_propagator",
    "build_walker_initial_states",
    "build_walker_propagators",
    "build_walker_constellation",
    "build_two_body_walker_constellation",
    "build_j2_j3_j4_walker_constellation",
    "build_numerical_walker_constellation",
    "attitude_providers",
    "orbit",
]


def __getattr__(name: str) -> Any:
    if name == "orbit":
        mod = import_module("nstk.propagation.orbit")
        globals()[name] = mod
        return mod
    if name == "attitude_providers":
        mod = import_module("nstk.propagation.attitude_providers")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nstk.propagation' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
