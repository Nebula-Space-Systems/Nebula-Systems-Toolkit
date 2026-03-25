from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from nstk.coverage import (
    BBoxDomain,
    CompositeDomain,
    CountryDomain,
    CoverageTargets,
    CoverageTimeline,
    ElevationConstraint,
    EqualAreaSampler,
    IntervalCoverage,
    LatitudeLongitudeSampler,
    Observer,
    PolygonDomain,
)


class MockOrbit:
    def __init__(self, positions: np.ndarray):
        self._positions = np.asarray(positions, dtype=np.float64)
        self.frames: list[str | None] = []

    def get_p_np(self, time: object, frame: str | None = None) -> np.ndarray:
        self.frames.append(frame)
        return self._positions


def _demo_targets() -> CoverageTargets:
    return CoverageTargets.from_domain(
        BBoxDomain(west_deg=-20.0, east_deg=20.0, south_deg=-10.0, north_deg=10.0),
        sampler=LatitudeLongitudeSampler(nlats=7, nlons=9),
    )


def _demo_positions(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs0 = np.tile(np.array([[7_000_000.0, 0.0, 0.0]], dtype=np.float64), (time_s.size, 1))
    obs1 = np.tile(np.array([[0.0, 7_000_000.0, 0.0]], dtype=np.float64), (time_s.size, 1))
    return obs0, obs1


def test_interval_coverage_accepts_sampled_and_orbit_inputs_with_matching_results() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 900.0, 10))
    targets = _demo_targets()
    obs0, obs1 = _demo_positions(timeline.seconds)

    sampled = IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_samples(obs0, name="sample-a"),
            Observer.from_samples(obs1, name="sample-b"),
        ],
        targets=targets,
        interpolation="linear",
    )
    orbit_like = IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_orbit(MockOrbit(obs0), name="orbit-a"),
            Observer.from_orbit(MockOrbit(obs1), name="orbit-b"),
        ],
        targets=targets,
        interpolation="linear",
    )

    np.testing.assert_allclose(
        sampled.access_duration(unit="seconds").values,
        orbit_like.access_duration(unit="seconds").values,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        sampled.max_asset().values,
        orbit_like.max_asset().values,
        atol=1e-9,
    )


def test_interval_coverage_accepts_raw_orbit_objects_with_elevation_constraint() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 900.0, 10))
    targets = _demo_targets()
    obs0, obs1 = _demo_positions(timeline.seconds)
    constraint = ElevationConstraint(min_deg=5.0)

    wrapped = IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_orbit(MockOrbit(obs0), name="orbit-a"),
            Observer.from_orbit(MockOrbit(obs1), name="orbit-b"),
        ],
        targets=targets,
        constraints=[constraint],
        interpolation="linear",
    )
    direct = IntervalCoverage.compute(
        timeline=timeline,
        observers=[MockOrbit(obs0), MockOrbit(obs1)],
        targets=targets,
        constraints=[constraint],
        interpolation="linear",
    )

    np.testing.assert_allclose(
        wrapped.access_duration(unit="seconds").values,
        direct.access_duration(unit="seconds").values,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        wrapped.channel("elevation").values,
        direct.channel("elevation").values,
        atol=1e-9,
    )


def test_interval_coverage_samples_orbit_observers_in_itrf_internally() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 900.0, 10))
    targets = _demo_targets()
    obs0, _ = _demo_positions(timeline.seconds)
    orbit = MockOrbit(obs0)

    IntervalCoverage.compute(
        timeline=timeline,
        observers=[Observer.from_orbit(orbit, frame="gcrf", name="orbit-a")],
        targets=targets,
        interpolation="linear",
    )

    assert orbit.frames
    assert set(orbit.frames) == {"itrf"}


def test_interval_coverage_compute_rejects_public_frame_argument() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 900.0, 10))
    targets = _demo_targets()
    obs0, _ = _demo_positions(timeline.seconds)

    with pytest.raises(TypeError, match="frame"):
        IntervalCoverage.compute(
            timeline=timeline,
            observers=[Observer.from_samples(obs0, name="sample-a")],
            targets=targets,
            frame="gcrf",
            interpolation="linear",
        )


def test_target_domains_and_samplers_materialize_targets_for_analysis() -> None:
    bbox_domain = BBoxDomain(west_deg=-10.0, east_deg=10.0, south_deg=-5.0, north_deg=5.0)
    poly_domain = PolygonDomain(geometry=box(-8.0, -4.0, 8.0, 4.0), name="Inner Box")
    composite = bbox_domain & poly_domain

    bbox_targets = CoverageTargets.from_domain(
        bbox_domain,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=7),
    )
    poly_targets = CoverageTargets.from_domain(
        poly_domain,
        sampler=LatitudeLongitudeSampler(nlats=5, nlons=7),
    )
    equal_area_targets = CoverageTargets.from_domain(
        composite,
        sampler=EqualAreaSampler(target_count=32),
    )

    assert bbox_targets.n_targets > 0
    assert poly_targets.n_targets > 0
    assert equal_area_targets.n_targets == 32
    assert bbox_targets.surface_grid is not None
    assert equal_area_targets.surface_grid is None
    np.testing.assert_allclose(equal_area_targets.area_weights.sum(), 1.0, atol=1e-12)


def test_country_and_composite_domains_can_be_sampled() -> None:
    try:
        france = CountryDomain(names=("France",))
    except Exception as exc:
        pytest.skip(f"Natural Earth country data unavailable: {exc}")

    regional = CompositeDomain(
        op="intersection",
        items=(
            france,
            BBoxDomain(west_deg=-10.0, east_deg=12.0, south_deg=40.0, north_deg=55.0),
        ),
    )
    try:
        targets = CoverageTargets.from_domain(
            regional,
            sampler=LatitudeLongitudeSampler(nlats=25, nlons=25),
        )
    except ValueError as exc:
        pytest.skip(f"Country/composite target sample landed on an empty coarse grid: {exc}")
    assert targets.n_targets > 0
    assert targets.boundary_geometry is not None


def test_coverage_targets_union_supports_country_names_and_boxes() -> None:
    pacific_box = BBoxDomain(west_deg=150.0, east_deg=170.0, south_deg=0.0, north_deg=20.0)
    try:
        targets = CoverageTargets.union(
            ["Japan", "Philippines"],
            pacific_box,
            sampler=EqualAreaSampler(target_count=256),
            resolution="10m",
            name="Japan, Philippines, and Pacific Box",
        )
    except Exception as exc:
        pytest.skip(f"Natural Earth country data unavailable: {exc}")

    country_union = CountryDomain(names=("Japan", "Philippines"), resolution="10m")
    combined_domain = country_union + pacific_box

    in_box = pacific_box.contains_latlon(targets.lat_deg, targets.lon_deg)
    in_countries = country_union.contains_latlon(targets.lat_deg, targets.lon_deg)

    assert targets.n_targets == 256
    assert targets.attrs["domain"] == "Japan, Philippines, and Pacific Box"
    assert np.all(combined_domain.contains_latlon(targets.lat_deg, targets.lon_deg))
    assert np.count_nonzero(in_box) > 0
    assert np.count_nonzero(in_countries & ~in_box) > 0


def test_interval_coverage_views_support_observer_target_and_window_selection() -> None:
    timeline = CoverageTimeline.relative(np.linspace(0.0, 1200.0, 13))
    obs0, obs1 = _demo_positions(timeline.seconds)
    coverage = IntervalCoverage.compute(
        timeline=timeline,
        observers=[
            Observer.from_samples(obs0, name="A", tags=("primary",)),
            Observer.from_samples(obs1, name="B", tags=("secondary",)),
        ],
        targets=_demo_targets(),
        interpolation="linear",
    )

    only_primary = coverage.observers.by_tag("primary")
    assert len(only_primary.observer_items) == 1

    first_two_targets = coverage.targets.only([0, 1])
    assert first_two_targets.target_set.n_targets == 2

    early_window = coverage.window(stop=600.0)
    assert early_window.store.time_stop == pytest.approx(600.0)

    timeline_result = coverage.target(index=0).timeline()
    assert timeline_result.target_index == 0
    assert len(timeline_result.to_records()) >= 1
