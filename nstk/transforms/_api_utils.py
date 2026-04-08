"""Shared helpers for the simplified transform APIs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import types


def is_numba_absent(value: Any) -> bool:
    """Return ``True`` when a Numba overload argument was omitted or is ``None``."""

    return value is None or isinstance(value, (types.NoneType, types.Omitted))


def is_numba_scalar(value: Any) -> bool:
    """Return ``True`` for numeric scalar types seen by Numba overloads."""

    return isinstance(value, (types.Integer, types.Float, int, float))


def is_numba_array1d(value: Any) -> bool:
    """Return ``True`` for one-dimensional Numba array arguments."""

    return isinstance(value, types.Array) and value.ndim == 1


def is_numba_array2d(value: Any) -> bool:
    """Return ``True`` for two-dimensional Numba array arguments."""

    return isinstance(value, types.Array) and value.ndim == 2


def as_1d_array(value: Any, name: str) -> np.ndarray:
    """Convert a Python value to a 1D ``ndarray`` or raise a clear error."""

    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def as_nx3_array(value: Any, name: str) -> np.ndarray:
    """Convert a Python value to an ``(N, 3)`` ``ndarray`` or raise a clear error."""

    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return arr


def validate_matching_lengths(*named_arrays: tuple[str, np.ndarray]) -> int:
    """Validate that a set of 1D arrays all share the same leading length."""

    if not named_arrays:
        raise ValueError("At least one array is required")

    n = named_arrays[0][1].shape[0]
    for name, arr in named_arrays[1:]:
        if arr.shape[0] != n:
            raise ValueError(
                ", ".join(item[0] for item in named_arrays) + " must have the same length"
            )
    return n


def require_not_none(value: Any, name: str) -> Any:
    """Raise a clear ``TypeError`` when a required argument is omitted."""

    if value is None:
        raise TypeError(f"`{name}` is required for this input form")
    return value
