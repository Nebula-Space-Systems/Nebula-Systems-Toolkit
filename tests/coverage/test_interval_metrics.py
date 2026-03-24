from __future__ import annotations

import numpy as np
import pytest
from numba import njit

astropy_time = pytest.importorskip("astropy.time")
astropy_units = pytest.importorskip("astropy.units")

from nstk.coverage import (
    AzimuthConstraint,
    BBoxDomain,
    CompiledMetric,
    CoverageTargets,
    CoverageTimeline,
    IntervalCoverage,
    LatitudeLongitudeSampler,
    MinAccessDurationConstraint,
    Observer,
    RangeConstraint,
    TargetLocalTimeConstraint,
    TargetSunElevationConstraint,
)

Time = astropy_time.Time
u = astropy_units


def _targets() -> CoverageTargets:
    return CoverageTargets.from_domain(
        BBoxDomain(west_deg=-20.0, east_deg=20.0, south_deg=-10.0, north_deg=10.0),
        sampler=LatitudeLongitudeSampler(nlats=7, nlons=9),
    )


def _positions(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs0 = np.tile(np.array([[7_000_000.0, 0.0, 0.0]], dtype=np.float64), (time_s.size, 1))
    obs1 = np.tile(np.array([[0.0, 7_000_000.0, 0.0]], dtype=np.float64), (time_s.size, 1))
    return obs0, obs1


def _coverage() -> IntervalCoverage:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 1200.0, 13))
    obs0, obs1 = _positions(timeline.seconds)
    return IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_samples(obs0, name="A", tags=("alpha",)),
            Observer.from_samples(obs1, name="B", tags=("beta",)),
        ],
        targets=_targets(),
        interpolation="linear",
    )


def test_metric_methods_return_inspectable_result_objects() -> None:
    coverage = _coverage()

    duration = coverage.access_duration(min_assets=1, unit="minutes")
    stack = coverage.access_duration(min_assets=[1, 2], unit="minutes")
    max_asset = coverage.max_asset()
    min_asset = coverage.min_asset()
    mtta = coverage.mtta(unit="minutes")
    revisit = coverage.revisit_time(unit="minutes")
    gap = coverage.gap_duration(unit="minutes")
    separation = coverage.access_separation(max_separation_s=2000.0)

    assert duration.values.shape == (coverage.target_set.n_targets,)
    assert stack.values.shape == (2, coverage.target_set.n_targets)
    assert stack.dims == ("min_assets", "target")
    assert max_asset.metric_name == "max_asset"
    assert min_asset.values.shape == duration.values.shape
    assert mtta.unit == "minutes"
    assert revisit.unit == "minutes"
    assert gap.unit == "minutes"
    assert separation.values.shape == duration.values.shape
    assert len(duration.to_records()) == coverage.target_set.n_targets
    assert np.isfinite(duration.reduce_targets("mean"))


def test_observer_subset_rescoring_matches_fresh_recompute() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 1200.0, 13))
    obs0, obs1 = _positions(timeline.seconds)
    targets = _targets()

    full = IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_samples(obs0, name="A"),
            Observer.from_samples(obs1, name="B"),
        ],
        targets=targets,
        interpolation="linear",
    )
    reduced_view = full.observers.only(["A"])
    reduced_fresh = IntervalCoverage.compute(
        timeline=timeline,
        observers=[Observer.from_samples(obs0, name="A")],
        targets=targets,
        interpolation="linear",
    )

    np.testing.assert_allclose(
        reduced_view.access_duration(unit="seconds").values,
        reduced_fresh.access_duration(unit="seconds").values,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        reduced_view.max_asset().values,
        reduced_fresh.max_asset().values,
        atol=1e-9,
    )


def test_post_hoc_constraints_can_be_applied_without_recomputing_base_intervals() -> None:
    coverage = _coverage()

    zero_range = coverage.with_constraints(RangeConstraint(max_m=1.0))
    zero_duration = zero_range.access_duration(unit="seconds")
    assert np.all(zero_duration.values == 0.0)

    az_limited = coverage.with_constraints(AzimuthConstraint(min_deg=0.0, max_deg=45.0))
    assert np.all(
        az_limited.access_duration(unit="seconds").values
        <= coverage.access_duration(unit="seconds").values + 1e-9
    )

    long_only = coverage.with_constraints(MinAccessDurationConstraint(min_seconds=1.0e9))
    assert np.all(long_only.access_duration(unit="seconds").values == 0.0)


def test_absolute_time_constraints_and_channels_are_available() -> None:
    epoch = Time("2025-01-01T00:00:00", scale="utc")
    timeline = CoverageTimeline.absolute(epoch + np.arange(0, 3600 + 1, 600) * u.s)
    obs0, _ = _positions(timeline.seconds)

    coverage = IntervalCoverage.compute(
        timeline=timeline,
        observers=[Observer.from_samples(obs0, name="A")],
        targets=_targets(),
        constraints=[
            TargetLocalTimeConstraint(start_hour=0.0, stop_hour=12.0),
            TargetSunElevationConstraint(min_deg=-90.0, max_deg=90.0),
        ],
        interpolation="linear",
    )

    assert "target_local_time" in coverage.channels
    assert "target_sun_elevation" in coverage.channels
    assert coverage.channels["target_local_time"].values.shape[1] == coverage.target_set.n_targets
    assert coverage.channels["target_sun_elevation"].values.shape[1] == coverage.target_set.n_targets


@njit(cache=True)
def _interval_count_kernel(
    pair_offsets: np.ndarray,
    start_times: np.ndarray,
    stop_times: np.ndarray,
    n_observers: int,
    n_targets: int,
    time_start: float,
    time_stop: float,
) -> np.ndarray:
    out = np.zeros(n_targets, dtype=np.float64)
    for obs_idx in range(n_observers):
        for target_idx in range(n_targets):
            pair_idx = obs_idx * n_targets + target_idx
            out[target_idx] += pair_offsets[pair_idx + 1] - pair_offsets[pair_idx]
    return out


def test_compiled_metric_and_python_side_analysis_hooks_work() -> None:
    coverage = _coverage()

    metric = CompiledMetric(
        name="interval_count",
        kernel=_interval_count_kernel,
        unit="count",
        label="Interval Count",
    )
    field = coverage.evaluate(metric)

    assert field.metric_name == "interval_count"
    assert field.values.shape == (coverage.target_set.n_targets,)
    assert np.all(field.values >= 0.0)

    max_concurrency = coverage.analyze(lambda cov: cov.max_asset().reduce_targets("max", weights=None))
    assert max_concurrency >= 1.0


def test_result_objects_support_regional_reductions_and_target_timeline_analysis() -> None:
    coverage = _coverage()
    field = coverage.access_duration(unit="seconds")
    region_mean = field.reduce_region(
        BBoxDomain(west_deg=-5.0, east_deg=5.0, south_deg=-5.0, north_deg=5.0),
        op="mean",
    )

    assert np.isfinite(region_mean)
    assert 0.0 <= field.covered_fraction() <= 1.0

    timeline = coverage.target(index=0).timeline()
    t, n = timeline.concurrency_profile()
    assert t.ndim == 1
    assert n.ndim == 1
    assert t.size == n.size
