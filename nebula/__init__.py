from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("nebula")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

_SUBPACKAGES = {
    "coverage",
    "localization",
    "geometry",
    "transforms",
    "propagation",
    "plotting",
}

__all__ = [
    "__version__",
    "coverage",
    "localization",
    "geometry",
    "transforms",
    "propagation",
    "plotting",
]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        mod = import_module(f"nebula.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nebula' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _SUBPACKAGES)
