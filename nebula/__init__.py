from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nebula")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

_SUBPACKAGES = {"coverage", "localization", "terrain", "transform", "propagation", "plotting"}

__all__ = ["__version__", "coverage", "localization", "terrain", "transform", "propagation", "plotting"]


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        mod = import_module(f"nebula.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nebula' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | _SUBPACKAGES)
