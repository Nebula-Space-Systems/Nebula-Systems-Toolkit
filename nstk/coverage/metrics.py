from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class CompiledMetric:
    name: str
    kernel: Callable[..., np.ndarray]
    unit: str | None = None
    label: str | None = None
    params: Any = None
    dims: tuple[str, ...] = ("target",)
    coords: dict[str, np.ndarray] | None = None


__all__ = ["CompiledMetric"]
