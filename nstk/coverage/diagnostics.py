from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .results import CoverageField
from .store import interval_count_by_pair


@dataclass
class CoverageDiagnostics:
    coverage: Any

    def interval_counts(self, *, by: str = "target") -> CoverageField | np.ndarray:
        counts = interval_count_by_pair(self.coverage.store)
        if by == "pair":
            return counts
        if by != "target":
            raise ValueError("by must be 'target' or 'pair'")
        values = counts.sum(axis=0).astype(np.float64)
        return CoverageField(
            targets=self.coverage.target_set,
            values=values,
            metric_name="interval_count",
            window_start_s=float(self.coverage.store.time_start),
            window_stop_s=float(self.coverage.store.time_stop),
            unit="count",
            label="Interval Count",
            fill_value=0.0,
        )

    def constraint_loss_breakdown(self) -> dict[str, float]:
        raw = float(self.coverage.raw_store.start_times.size)
        final = float(self.coverage.store.start_times.size)
        return {
            "raw_interval_count": raw,
            "filtered_interval_count": final,
            "removed_interval_count": raw - final,
            "fraction_retained": (final / raw) if raw > 0 else 1.0,
        }

    def marginal(
        self,
        *,
        metric: str = "access_duration",
        reduction: str = "mean",
        **metric_kwargs: Any,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for obs in self.coverage.observer_items:
            view = self.coverage.observers.only([obs.name] if obs.name is not None else [self.coverage.observer_items.index(obs)])
            field = getattr(view, metric)(**metric_kwargs)
            out[obs.name or f"observer_{len(out)}"] = field.reduce_targets(reduction)
        return out

    def leave_one_out(
        self,
        *,
        metric: str = "access_duration",
        reduction: str = "mean",
        **metric_kwargs: Any,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        names = [obs.name or f"observer_{idx}" for idx, obs in enumerate(self.coverage.observer_items)]
        for name in names:
            view = self.coverage.observers.exclude([name])
            field = getattr(view, metric)(**metric_kwargs)
            out[name] = field.reduce_targets(reduction)
        return out


__all__ = ["CoverageDiagnostics"]
