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
_MODULE_EXPORTS = {"time_utils"}

__all__ = [
    "__version__",
    "initialize_orekit",
    "coverage",
    "localization",
    "geometry",
    "transforms",
    "propagation",
    "plotting",
    "time_utils",
]


def initialize_orekit(*, data_path: str | None = None) -> None:
    """Initialize Orekit/JVM runtime for Nebula.

    This is the single package-level initializer for Orekit-backed features.
    It prepares all bridge classpaths, starts the JVM, and configures Orekit data.
    """

    from nebula._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime(data_path=data_path)


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES or name in _MODULE_EXPORTS:
        mod = import_module(f"nebula.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nebula' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _SUBPACKAGES | _MODULE_EXPORTS)
