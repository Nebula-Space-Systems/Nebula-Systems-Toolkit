from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class CompiledMetric:
    """A compiled per-target coverage metric.

    The kernel must return a 1D array with one value per target in the current
    coverage object. Multi-dimensional metric outputs are not supported.
    """

    name: str
    kernel: Callable[..., np.ndarray]
    unit: str | None = None
    label: str | None = None
    params: Any = None


__all__ = ["CompiledMetric"]
