# Coverage Redesign Prototype

## Goals

The coverage API should feel excellent in two very different modes:

1. easy mode
   - a user can hand NSTK a list of `Orbit` objects, a time axis, and a target
     definition, then immediately ask for metrics and plots

2. power-user mode
   - a user can mix observer types, supply custom ephemerides, define new
     metrics, inspect exact interval timelines, and target either Numba or a
     future native backend

The public API should be flexible, but all computation should collapse down to a
small compiled intermediate representation with contiguous arrays and primitive
scalars only.


## Design Principles

- Keep the "happy path" tiny.
- Make extensibility live at the edges: inputs, metrics, exports.
- Keep plotting as a convenience layer, not the core interface.
- Separate user metadata from kernel-facing numeric arrays.
- Make all built-in metrics operate on a stable low-level interval store.
- Allow future backends (`numba`, `native`) to share the same low-level ABI.
- Let advanced users write custom metrics against that ABI when they need speed.


## High-Level Shape

The recommended architecture is:

1. public API layer
   - `IntervalCoverage`
   - `CoverageTimeline`
   - `TargetDomain`
   - `TargetSampler`
   - `CoverageTargets`
   - `Observer`
   - `CoverageArray`
   - `CoverageField`
   - `CoverageStack`
   - `TargetTimeline`

2. adapter layer
   - coerce `Orbit`, sampled arrays, tabulated ephemerides, callables, and user
     source objects into a common observer interface

3. compiled IR layer
   - resolve all inputs into contiguous numeric arrays
   - build exact interval store
   - run built-in or user-provided compiled metrics over that store

4. presentation / export layer
   - plots
   - tables / records
   - numpy-friendly inspection
   - optional `pandas`/`xarray` export helpers if installed


## Core Public Types

### `CoverageTimeline`

This should be a concrete object, not a bare time array.

Responsibilities:

- hold the analysis time axis
- normalize absolute and relative time inputs
- provide a common epoch-seconds representation for kernels
- preserve original time semantics for user inspection

Prototype:

```python
@dataclass(frozen=True)
class CoverageTimeline:
    epoch: Time | None
    seconds: np.ndarray          # (nt,) float64, strictly increasing
    absolute_time: Time | None   # optional original absolute times
    label: str | None = None

    @classmethod
    def absolute(cls, time: Time) -> "CoverageTimeline": ...

    @classmethod
    def relative(
        cls,
        seconds: np.ndarray | u.Quantity,
        *,
        epoch: Time | None = None,
    ) -> "CoverageTimeline": ...

    @classmethod
    def linspace(
        cls,
        start: Time | float,
        stop: Time | float,
        step: float | u.Quantity,
    ) -> "CoverageTimeline": ...

    @property
    def start(self) -> float: ...

    @property
    def stop(self) -> float: ...

    @property
    def duration(self) -> float: ...
```

Why this matters:

- `Orbit` objects often want absolute times.
- raw ephemerides may already live on some custom numeric time axis.
- compiled kernels want a plain `float64` seconds vector.


### `CoverageTargets`

This should be the common target container for all target-wise outputs.

Responsibilities:

- represent the exact target set used in the study
- hold kernel-facing target geometry
- optionally hold surface-grid metadata for rasterization and map plots
- support arbitrary points, not just lat-lon surface grids

Prototype:

```python
@dataclass(frozen=True)
class SurfaceGridMetadata:
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    row_offsets: np.ndarray
    lat_rows_deg: np.ndarray
    shape: tuple[int, int] | None


@dataclass(frozen=True)
class CoverageTargets:
    positions_ecef_m: np.ndarray     # (n_targets, 3)
    up_vectors_ecef: np.ndarray      # (n_targets, 3)
    labels: list[str] | None = None
    surface_grid: SurfaceGridMetadata | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: ExactCoverageConfig) -> "CoverageTargets": ...

    @classmethod
    def surface_grid(
        cls,
        *,
        nlats: int = 181,
        nlons_equator: int = 361,
        min_elevation_deg: float = 0.0,
        max_elevation_deg: float = 90.0,
        ...
    ) -> "CoverageTargets": ...

    @classmethod
    def points(
        cls,
        positions_ecef_m: np.ndarray,
        up_vectors_ecef: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> "CoverageTargets": ...

    @classmethod
    def geodetic_points(
        cls,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: np.ndarray | float = 0.0,
        *,
        labels: list[str] | None = None,
    ) -> "CoverageTargets": ...

    def nearest_target_index(self, *, lat_deg: float, lon_deg: float) -> int: ...
```

Design note:

- `CoverageTargets` is the static-target container for the first redesign pass.
- if NSTK later supports moving targets or target-vs-target analyses, add a
  `TargetSource` protocol parallel to `ObserverSource` and lower it into the
  same compiled IR without changing the result model.


### `TargetDomain` And `TargetSampler`

This is one of the most important additions for long-term flexibility.

Users often do not want "all Earth points on a grid". They want:

- global Earth
- a latitude/longitude box
- a list of countries
- land only / ocean only
- a polygon or multipolygon
- a shapefile / GeoJSON region
- a custom mask from some raster
- a union / intersection / difference of multiple regions

That means target definition should be split into:

1. `TargetDomain`
   - the geometric region or logical subset to cover

2. `TargetSampler`
   - how points/cells are chosen inside that domain

3. `CoverageTargets`
   - the materialized numeric target set actually used by kernels

Prototype:

```python
class TargetDomain(Protocol):
    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray: ...
    def bounds(self) -> tuple[float, float, float, float] | None: ...
    def boundary_geometry(self) -> Any | None: ...


@dataclass(frozen=True)
class GlobalEarthDomain: ...


@dataclass(frozen=True)
class BBoxDomain:
    west_deg: float
    east_deg: float
    south_deg: float
    north_deg: float


@dataclass(frozen=True)
class CountryDomain:
    names: tuple[str, ...]
    resolution: str = "110m"


@dataclass(frozen=True)
class PolygonDomain:
    geometry: Any


@dataclass(frozen=True)
class RasterMaskDomain:
    mask: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray


@dataclass(frozen=True)
class CompositeDomain:
    op: str
    items: tuple[TargetDomain, ...]


class TargetSampler(Protocol):
    def materialize(self, domain: TargetDomain) -> CoverageTargets: ...


@dataclass(frozen=True)
class LatitudeLongitudeSampler:
    nlats: int = 181
    nlons_equator: int = 361
    scale_longitude_by_latitude: bool = True


@dataclass(frozen=True)
class EqualAreaSampler:
    target_count: int
    family: str = "cubed_sphere"


@dataclass(frozen=True)
class AdaptiveSampler:
    base: TargetSampler
    refine_boundaries: bool = True
    refine_hotspots: bool = False
```

Recommended conveniences:

```python
CoverageTargets.from_domain(domain, sampler=LatitudeLongitudeSampler(...))
CoverageTargets.global_earth(...)
CoverageTargets.country("Japan", ...)
CoverageTargets.countries(["France", "Germany", "Italy"], ...)
CoverageTargets.region_bbox(...)
CoverageTargets.from_geojson(path, ...)
CoverageTargets.from_shapefile(path, ...)
CoverageTargets.land(...)
CoverageTargets.ocean(...)
```

This design works very well with the existing fuzzy country-shape helpers in
`nstk.plotting.country_shapes` and keeps target generation separate from the
kernel ABI.

Important design note:

- region logic should be composable:
  - `domain_a | domain_b`
  - `domain_a & domain_b`
  - `domain_a - domain_b`
- sampling should be independent of region choice
- every materialized target set should carry area weights when meaningful


### Region Weights And Regional Analysis

Target sets should retain enough metadata to support weighted summaries.

For surface coverage this usually means:

- point area weight
- source row/cell id
- region membership tags

That enables analysis like:

- fraction of a country above a threshold
- area-weighted average revisit time over Europe
- percentile availability across only coastal cells
- compare land vs ocean vs EEZ masks from the same underlying coverage run


### `Observer`

This is where flexibility should live.

`Observer` should be a public adapter/factory type that normalizes many user
inputs into a stable source interface.

Prototype:

```python
@runtime_checkable
class ObserverSource(Protocol):
    def sample_positions(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray:
        """Return shape (nt, 3) in requested frame."""

    def sample_velocities(
        self,
        timeline: CoverageTimeline,
        *,
        frame: str,
    ) -> np.ndarray | None:
        """Optional. Return shape (nt, 3) or None."""

    @property
    def name(self) -> str | None: ...


@dataclass(frozen=True)
class Observer:
    source: ObserverSource

    @classmethod
    def from_orbit(
        cls,
        orbit: Orbit,
        *,
        frame: str = "itrf",
        name: str | None = None,
        use_velocity: bool = True,
        precompute: bool = False,
    ) -> "Observer": ...

    @classmethod
    def from_samples(
        cls,
        positions: np.ndarray,
        *,
        timeline: CoverageTimeline | None = None,
        frame: str = "itrf",
        velocities: np.ndarray | None = None,
        name: str | None = None,
    ) -> "Observer": ...

    @classmethod
    def from_tabulated(
        cls,
        time: Any,
        positions: np.ndarray,
        *,
        frame: str = "itrf",
        velocities: np.ndarray | None = None,
        interpolation: str = "linear",
        name: str | None = None,
    ) -> "Observer": ...

    @classmethod
    def from_callable(
        cls,
        sampler: Callable[..., np.ndarray],
        *,
        velocity_sampler: Callable[..., np.ndarray] | None = None,
        frame: str = "itrf",
        name: str | None = None,
    ) -> "Observer": ...
```

Important design note:

- `Orbit` should be just one built-in observer source.
- external propagators and ephemeris stores should fit the same interface.
- if velocities are available, keep them; exact cubic interpolation can use
  them instead of estimating derivatives from sampled positions.


### `IntervalCoverage`

This should be the main user-facing object.

Responsibilities:

- accept flexible inputs
- coerce them into resolved numeric arrays
- build the low-level interval store
- expose built-in metrics as methods
- expose user-defined metrics through `evaluate(...)`
- expose exact target-level inspection

Prototype:

```python
@dataclass
class IntervalCoverage:
    timeline: CoverageTimeline
    targets: CoverageTargets
    observers: list[Observer]
    store: IntervalStore
    constraints: "ConstraintSet | None" = None
    backend: str = "numba"
    analysis_frame: str = "itrf"
    observer_names: list[str] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        *,
        observers: Any,
        timeline: Any,
        targets: Any,
        interpolation: str = "cubic",
        min_elevation_deg: float | None = None,
        max_elevation_deg: float | None = None,
        backend: str = "numba",
        analysis_frame: str = "itrf",
        cache_tracks: bool = False,
    ) -> "IntervalCoverage": ...

    @classmethod
    def from_orbits(
        cls,
        orbits: Sequence[Orbit],
        *,
        timeline: Any,
        targets: Any,
        frame: str = "itrf",
        interpolation: str = "cubic",
        ...
    ) -> "IntervalCoverage": ...

    def window(
        self,
        start: float | Time | None = None,
        stop: float | Time | None = None,
    ) -> "IntervalCoverageView": ...

    def target(
        self,
        *,
        target_index: int | None = None,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        name: str | None = None,
    ) -> "TargetSelection": ...

    def evaluate(self, metric: "MetricSpec | CompiledMetric") -> "CoverageResult": ...

    def access_duration(... ) -> "CoverageField | CoverageStack": ...
    def max_asset(... ) -> "CoverageField": ...
    def min_asset(... ) -> "CoverageField": ...
    def mtta(... ) -> "CoverageField": ...
    def revisit_time(... ) -> "CoverageField": ...
    def gap_duration(... ) -> "CoverageField": ...
    def access_separation(... ) -> "CoverageField": ...
```


## Constraints And Coverage Views

This redesign should treat constraints as first-class objects.

The key idea is that not all constraints belong at the same stage:

1. build-time pair constraints
   - affect the exact observer-target access predicate itself
   - examples:
     - min/max elevation
     - min/max azimuth
     - min/max range
     - FOV / boresight / attitude constraints

2. reusable time gates
   - depend on time plus observer/target/global context and can often be applied
     by interval intersection after pair access is already known
   - examples:
     - target sun elevation angle
     - observer eclipse state
     - local solar time at target
     - local clock time at observer
     - global mission blackout windows

3. post-interval filters
   - operate on already-built access intervals
   - examples:
     - minimum access duration
     - maximum access duration
     - merge gaps shorter than X
     - erode/dilate intervals by margins

4. analysis-time selections
   - do not change pair intervals at all; they change what is counted in a metric
   - examples:
     - only observers 0, 2, and 4
     - exclude failed satellites
     - only targets in a region
     - only a time sub-window

This staging is important because it determines what can be changed without
rebuilding raw geometry.


### Constraint Model

Prototype:

```python
@dataclass(frozen=True)
class ConstraintSet:
    items: tuple["Constraint", ...] = ()

    def pair(self) -> tuple["PairConstraint", ...]: ...
    def observer_time(self) -> tuple["ObserverTimeConstraint", ...]: ...
    def target_time(self) -> tuple["TargetTimeConstraint", ...]: ...
    def global_time(self) -> tuple["GlobalTimeConstraint", ...]: ...
    def post_interval(self) -> tuple["PostIntervalConstraint", ...]: ...


class Constraint(Protocol):
    name: str
    stage: str
    def lower(self, context: "ConstraintLoweringContext") -> "LoweredConstraint": ...
```

Built-in constraint types should be plain concrete dataclasses:

```python
@dataclass(frozen=True)
class ElevationConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class AzimuthConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class RangeConstraint:
    min_m: float | None = None
    max_m: float | None = None


@dataclass(frozen=True)
class TargetSunElevationConstraint:
    min_deg: float | None = None
    max_deg: float | None = None


@dataclass(frozen=True)
class TargetLocalTimeConstraint:
    start_hour: float
    stop_hour: float
    solar: bool = True


@dataclass(frozen=True)
class MinAccessDurationConstraint:
    min_seconds: float
```


### Constraint Usage

Easy mode:

```python
coverage = IntervalCoverage.compute(
    observers=sats,
    timeline=times,
    targets=config,
    constraints=[
        ElevationConstraint(min_deg=10.0),
        RangeConstraint(max_m=2_000_000.0),
        TargetSunElevationConstraint(max_deg=-6.0),
        MinAccessDurationConstraint(min_seconds=30.0),
    ],
)
```

Power-user mode:

```python
constraints = ConstraintSet(
    items=(
        ElevationConstraint(min_deg=10.0, max_deg=80.0),
        AzimuthConstraint(min_deg=20.0, max_deg=160.0),
        TargetLocalTimeConstraint(start_hour=6.0, stop_hour=18.0),
        MinAccessDurationConstraint(min_seconds=45.0),
    )
)

coverage = IntervalCoverage.compute(
    observers=observers,
    timeline=timeline,
    targets=targets,
    constraints=constraints,
)
```


### Constraint Lowering

The public API can accept rich objects, but the backend should only see compact
numeric representations.

Prototype:

```python
@dataclass(frozen=True)
class LoweredConstraintProgram:
    pair_codes: np.ndarray                 # (n_pair_rules,), int32
    pair_param_offsets: np.ndarray         # (n_pair_rules + 1,), int32
    pair_params: np.ndarray                # (n_pair_params,), float64

    observer_gate_store: "TimeGateStore | None" = None
    target_gate_store: "TimeGateStore | None" = None
    global_gate_store: "TimeGateStore | None" = None

    post_codes: np.ndarray | None = None
    post_param_offsets: np.ndarray | None = None
    post_params: np.ndarray | None = None
```

Design note:

- pair constraints should be evaluated inside the exact interval builder
- observer/target/global time gates should preferably lower to interval stores
  that can be intersected without rebuilding pair geometry
- post-interval filters should lower to interval transforms over already-built
  pair intervals


### Reusable Time Gates

To support constraints like sun angle and local time well, NSTK should have
small gate stores parallel to the pair interval store.

Prototype:

```python
@dataclass(frozen=True)
class TimeGateStore:
    scope: str                     # "observer", "target", or "global"
    item_offsets: np.ndarray       # CSR-like offsets
    start_times: np.ndarray
    stop_times: np.ndarray
```

Examples:

- target sun elevation gate:
  - for each target, intervals where sun elevation is within limits

- observer local-time gate:
  - for each observer, intervals where the observer is active

- global blackout gate:
  - intervals where the mission is allowed to score access

These can be intersected with pair access intervals after the pair store is
built, which means many environmental or schedule-style constraints can be
toggled without recomputing geometry.


### Important Constraint Honesty

Some constraints can be changed after raw access intervals are computed, and
some cannot.

Usually safe to change post hoc:

- selected observer subset
- selected target subset
- scoring time window
- minimum access duration
- target sun-elevation gates
- observer schedule gates
- global blackout windows

Usually requires rebuilding the pair access store:

- min/max elevation
- min/max azimuth
- min/max range
- sensor FOV / boresight conditions

Unless NSTK explicitly cached those specific pair-gate channels in advance.

This distinction should be documented clearly in the public API.


### Coverage Views

`IntervalCoverage` should support cheap derived views without rebuilding the raw
pair interval store.

Prototype:

```python
@dataclass(frozen=True)
class IntervalCoverageView:
    parent: IntervalCoverage
    observer_indices: np.ndarray | None = None
    target_indices: np.ndarray | None = None
    time_window: tuple[float, float] | None = None
    gate_overrides: tuple[TimeGateStore, ...] = ()
    post_constraints: tuple["PostIntervalConstraint", ...] = ()

    def access_duration(... ) -> "CoverageField | CoverageStack": ...
    def max_asset(... ) -> "CoverageField": ...
    def mtta(... ) -> "CoverageField": ...
    def target(... ) -> "TargetSelection": ...
```

This makes all of these natural:

```python
coverage.window(stop=6 * 3600.0).access_duration()
coverage.observers.include(["sat-0", "sat-2", "sat-4"]).access_duration(min_assets=3)
coverage.observers.exclude(group="failed").mtta()
coverage.targets.select(indices=my_idx).revisit_time()
```


### Observer Activation / Deactivation

Post-hoc contributor control should be a core feature, not an afterthought.

Recommended public shape:

```python
coverage.observers.include([...])
coverage.observers.exclude([...])
coverage.observers.only([...])
coverage.observers.by_tag("plane-a")
coverage.observers.active(mask)
```

Each of these should return an `IntervalCoverageView` that reuses the original
pair intervals.

Design note:

- the interval store should keep intervals for every observer-target pair
- metric kernels should accept an optional selected-observer index array
- repeated work on the same subset can optionally be compacted into a materialized
  subset store later, but the no-copy view should be the default


### Observer Subset Example

This should be cheap and not require recomputing coverage:

```python
coverage = IntervalCoverage.compute(
    observers=[sat0, sat1, sat2, sat3, sat4],
    timeline=times,
    targets=config,
)

field = (
    coverage
    .observers.only(["sat0", "sat2", "sat4"])
    .access_duration(min_assets=3, normalize="day", unit="hours")
)
```

What happens internally:

1. the original pair interval store for all five observers is reused
2. the metric kernel scans only the selected three observer indices
3. no geometry propagation or exact root-finding is rerun

This is the right architecture for "3 out of 5 after compute" workflows.


## Result Types

The outputs should be concrete and inspectable. Plotting should be a method, not
the only reason they exist.


### `CoverageArray`

Under the hood, NSTK should probably have one general labeled-array result type.

`CoverageField` and `CoverageStack` can then be thin convenience wrappers around
the most common shapes.

Prototype:

```python
@dataclass(frozen=True)
class CoverageArray:
    values: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    targets: CoverageTargets | None = None
    unit: str | None = None
    label: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def sel(self, **coords) -> "CoverageArray": ...
    def reduce(self, dim: str, op: str, *, weights=None) -> "CoverageArray": ...
    def to_numpy(self) -> np.ndarray: ...
    def to_xarray(self): ...
```

This becomes very useful once you start supporting:

- target
- time
- min_assets
- observer_subset
- region
- scenario

without inventing a separate custom class for every dimensionality.


### `CoverageResult`

This can be a very small nominal base class or protocol shared by all result
objects.

Prototype:

```python
@dataclass(frozen=True)
class CoverageResult:
    targets: CoverageTargets
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_numpy(self, **kwargs): ...
    def to_records(self): ...
```

Design note:

- this can be a tiny shared base if you want one
- `CoverageField` should still remain concrete and not abstract
- the important abstraction is the stable result family, not a deep OO hierarchy


### `CoverageField`

One scalar value per target.

Prototype:

```python
@dataclass(frozen=True)
class CoverageField(CoverageResult):
    values: np.ndarray             # (n_targets,)
    metric_name: str
    window_start_s: float
    window_stop_s: float
    unit: str | None = None
    label: str | None = None
    fill_value: float | int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def plot(self, **kwargs): ...
    def to_numpy(self, *, reshape: bool = False) -> np.ndarray: ...
    def to_records(self) -> list[dict[str, Any]]: ...
    def at_index(self, target_index: int) -> float | int: ...
    def at(self, *, lat_deg: float, lon_deg: float) -> float | int: ...
    def summary(self) -> dict[str, float]: ...
```

Design note:

- `CoverageField` is a concrete value object.
- it is not the extension seam.
- the extension seams are the input adapters and the metric runtime.


### `CoverageStack`

For multi-parameter results like `min_assets=[1, 2, 3]`.

Prototype:

```python
@dataclass(frozen=True)
class CoverageStack(CoverageResult):
    values: np.ndarray                 # (..., n_targets)
    dims: tuple[str, ...]              # e.g. ("min_assets", "target")
    coords: dict[str, np.ndarray]      # e.g. {"min_assets": array([1, 2, 3])}
    metric_name: str
    unit: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def sel(self, **coords) -> CoverageField | "CoverageStack": ...
    def plot(self, **kwargs): ...
    def to_numpy(self, *, reshape_targets: bool = False) -> np.ndarray: ...
```

This replaces awkward dict outputs like:

```python
{1: values_for_n1, 2: values_for_n2}
```

with a more inspectable structure:

```python
stack = coverage.access_duration(min_assets=[1, 2, 3], normalize="day", unit="hours")
stack.coords["min_assets"]   # array([1, 2, 3])
stack.sel(min_assets=2)      # CoverageField
```


### Reductions And Aggregations

The module will feel much more complete if it supports not only per-target
fields, but also region-wise and time-wise reductions directly.

Recommended examples:

```python
field.reduce_targets("mean", weights="area")
field.reduce_region(CountryDomain("Japan"), op="p95", weights="area")
field.threshold(12.0).covered_fraction(weights="area")
stack.reduce("min_assets", op="argmax")
coverage.access_duration_time_series(region="US")
```

Suggested built-in reductions:

- mean / min / max / std
- p50 / p90 / p95 / p99
- covered fraction above threshold
- histogram / CDF / exceedance curve
- worst-k / best-k targets
- grouped summaries by country, tag, region, observer plane, etc.

These are extremely useful for analysis and report generation.


### `TargetTimeline`

Exact intervals and concurrency information for a selected target.

Prototype:

```python
@dataclass(frozen=True)
class TargetTimeline(CoverageResult):
    target_index: int
    target_label: str | None
    observer_names: list[str] | None
    starts_by_observer: list[np.ndarray]
    stops_by_observer: list[np.ndarray]
    time_start_s: float
    time_stop_s: float
    attrs: dict[str, Any] = field(default_factory=dict)

    def plot(self, **kwargs): ...
    def concurrency_profile(self) -> tuple[np.ndarray, np.ndarray]: ...
    def to_records(self) -> list[dict[str, Any]]: ...
```


## Low-Level Compiled IR

This is the most important part of the design.

The public API can be flexible, but kernels should only see compact arrays and
primitive scalars.


### 1. Resolved Inputs

Before interval construction, all inputs are normalized into:

```python
@dataclass(frozen=True)
class ResolvedObserverTracks:
    positions: np.ndarray         # (nt, n_obs, 3), float64, contiguous
    velocities: np.ndarray | None # (nt, n_obs, 3), float64, contiguous
    names: list[str] | None


@dataclass(frozen=True)
class ResolvedTargets:
    positions: np.ndarray         # (n_targets, 3), float64, contiguous
    up_vectors: np.ndarray        # (n_targets, 3), float64, contiguous
    metadata: CoverageTargets
```

These objects are Python-side assembly only. Kernels should not depend on them
directly.


### 2. Interval Store

This should be the stable metric ABI.

Prototype:

```python
@dataclass(frozen=True)
class IntervalStore:
    time_start: float
    time_stop: float
    n_observers: int
    n_targets: int
    pair_offsets: np.ndarray      # (n_pairs + 1,), int64
    start_times: np.ndarray       # (n_intervals,), float64
    stop_times: np.ndarray        # (n_intervals,), float64
    interpolation_code: int       # 0 linear, 1 cubic
    root_tolerance_s: float
```

Important change from the current shape:

- kernel-facing interval arrays should be separated from plotting and target
  metadata
- user metadata belongs on `IntervalCoverage` / `CoverageTargets`, not inside the
  kernel ABI

This separation makes it easier to:

- keep kernels simple
- support a native backend later
- serialize / cache the store
- expose a stable low-level interface for custom compiled metrics


### 2b. Pair Channel Stores

If you want the system to be truly extensible for advanced metrics and
post-processing, the raw interval store alone is not enough.

Many interesting analyses depend on quantities like:

- elevation
- azimuth
- slant range
- range rate
- off-nadir angle
- sun elevation at target
- local solar time

NSTK should support optional sidecar channel stores that can be requested at
build time and reused later.

Prototype:

```python
@dataclass(frozen=True)
class PairChannelStore:
    name: str
    scope: str                     # "pair", "observer", "target", "global"
    sample_times: np.ndarray       # channel knot times
    values: np.ndarray             # packed values
    index_offsets: np.ndarray      # CSR-like lookup
    interpolation: str             # "step", "linear", "cubic"
```

Examples:

- pair elevation channel
- pair azimuth channel
- pair range channel
- target sun-angle channel
- target local-time channel

Why this matters:

- it enables much richer custom metrics
- it makes some constraints toggleable after the main compute
- it avoids forcing users to rebuild the entire geometry pipeline every time

Recommended policy:

- keep the default build lightweight
- let users opt into extra channel families explicitly
- document the memory/performance tradeoff clearly


### 3. Metric Scan ABI

Built-in metrics and advanced user-defined metrics should target a stable scan
API for "iterate exact event timeline for one target".

Prototype concept:

```python
@njit(cache=True)
def metric_kernel(
    pair_offsets,
    start_times,
    stop_times,
    n_observers,
    n_targets,
    t0,
    t1,
    selected_observer_indices,   # int32[:] or empty sentinel
    active_target_indices,       # int32[:] or empty sentinel
    observer_gate_store,
    target_gate_store,
    global_gate_store,
    params,
    out,
):
    ...
```

NSTK should provide public low-level helper functions for custom compiled
metrics, mirroring the exact event-scan helpers already used internally:

- initialize target state
- find next event time
- apply all events at a time
- scan piecewise-constant concurrency segments

That gives advanced users a supported way to build their own Numba metrics
without copy-pasting internals.

Design note:

- observer subseting belongs in the metric scan layer, not in the geometry build
- observer/target/global time gates should be intersected during scan or via a
  lightweight pre-pass over intervals
- this keeps the raw pair interval store reusable across many scoring views


### Diagnostics And Explainability

A best-in-class coverage module should also help the user understand why a
result looks the way it does.

Recommended diagnostics:

- interval count per pair / target
- access lost by constraint stage
- no-access reason maps
- observer contribution deltas
- marginal gain from adding each observer
- overlap / redundancy decomposition

Examples:

```python
coverage.diagnostics.interval_counts()
coverage.diagnostics.constraint_loss_breakdown()
coverage.observers.marginal(metric="access_duration", min_assets=1)
coverage.observers.leave_one_out(metric="mtta")
```

These features make the system much more useful for design trades and
constellation analysis.


## Metric Architecture

There should be three supported levels of metric customization.


### Level 1: Built-In Metrics

Easy mode:

```python
coverage.access_duration()
coverage.mtta()
coverage.revisit_time(statistic="maximum")
coverage.max_asset()
```

These return `CoverageField` or `CoverageStack`.


### Level 2: Python-Side Custom Analysis

Easy to write, slower than compiled, still very inspectable.

Example:

```python
timeline = coverage.target(lat_deg=34.05, lon_deg=-118.25).timeline()
records = timeline.to_records()
```

This mode should let users do whatever they want with:

- exact intervals
- target-wise interval collections
- exported records / arrays
- direct access to the `IntervalStore`

This is the "maximum flexibility, not necessarily maximum speed" tier.


### Level 3: Compiled Custom Metrics

Advanced users should be able to define a compiled metric against the interval
ABI.

Prototype:

```python
@dataclass(frozen=True)
class CompiledMetric:
    name: str
    output_dtype: np.dtype
    unit: str | None
    label: str | None
    kernel: Callable[..., None]   # expected to be njit-compiled
    params: Any = None
```

Usage:

```python
metric = CompiledMetric(
    name="time_with_three_plus",
    output_dtype=np.float64,
    unit="hours/day",
    label="Hours/Day With >=3 Assets",
    kernel=my_numba_kernel,
    params={"min_assets": 3},
)

field = coverage.evaluate(metric)
```

Important honesty about this layer:

- NSTK can make compiled custom metrics possible and pleasant.
- NSTK cannot make arbitrary user Python automatically fast.
- the contract should be:
  - built-ins are fully compiled
  - user Python analysis is flexible but slower
  - user compiled metrics can match built-in performance if they target the
    metric ABI


## Backend Design

The system should be backend-neutral above the compiled IR.

Prototype:

```python
coverage = IntervalCoverage.compute(..., backend="numba")
coverage = IntervalCoverage.compute(..., backend="native")
```

Recommended contract:

- `numba`
  - reference backend
  - fastest to iterate on
  - has warm-up cost

- `native`
  - optional compiled extension later
  - same interval store ABI
  - no JIT warm-up
  - likely C++/Rust/Cython/pybind11 implementation of the same kernels

If the low-level ABI stays clean, backends can swap without changing the public
API or result objects.


## Input Coercion Rules

The public `compute(...)` path should coerce these inputs automatically:

- `targets`
  - `ExactCoverageConfig` -> `CoverageTargets.from_config(...)`
  - `CoverageTargets` -> use as-is

- `timeline`
  - `astropy.time.Time` -> `CoverageTimeline.absolute(...)`
  - numeric seconds / quantity -> `CoverageTimeline.relative(...)`
  - `CoverageTimeline` -> use as-is

- `observers`
  - `Orbit` -> `Observer.from_orbit(...)`
  - list of `Orbit` -> same
  - `Observer` -> use as-is
  - `(time, positions)` tuple -> `Observer.from_tabulated(...)`
  - raw `(nt, 3)` positions -> `Observer.from_samples(...)` if a common timeline
    is provided

NSTK should support automatic coercion for the obvious cases, but still expose
explicit constructors for clarity and advanced control.


## Plotting Design

Plotting should become generic.

Recommended rule:

- `CoverageField.plot(...)` handles maps when `targets.surface_grid is not None`
- `TargetTimeline.plot(...)` handles exact interval timeline views
- separate free functions may still exist internally, but they should not be the
  main public interface

This means the user learns one concept:

```python
result = coverage.mtta(unit="minutes")
result.plot(...)
```

instead of one plot wrapper per metric.


## Visualization Architecture

If you want the visuals to feel genuinely polished, make plotting a coherent
layer rather than a handful of wrappers.

Recommended additions:

- automatic extent from `TargetDomain`
- boundary overlays from region/country shapes
- hatching or alpha masks for no-data / no-access cells
- small multiples for `CoverageStack`
- histogram / ECDF / exceedance plots from any `CoverageArray`
- timeline and concurrency plots from `TargetTimeline`
- observer contribution plots from diagnostics
- region-summary bar charts / tables

Prototype examples:

```python
field.plot_map(boundary="target_domain", theme=LIGHT_DETAILED)
stack.plot_small_multiples(dim="min_assets", ncols=2)
field.plot_histogram()
field.plot_ecdf(weights="area")
timeline.plot()
diagnostics.plot_constraint_loss()
```

Strong recommendation:

- plotting functions should accept either raw result objects or explicit map
  domains/extents
- every plotting method should return handles cleanly for further styling
- plotting should preserve provenance in titles/labels automatically, but allow
  easy overrides


## API Naming And Structural Changes

If the redesign is meant to be a fresh start, I would also make these changes:

- de-emphasize `ExactCoverageConfig` as the main user-facing target type
- remove metric-specific plotting entrypoints as the primary teaching API
- stop returning bare dicts for multi-parameter metrics
- standardize on:
  - `min_assets`
  - `statistic`
  - `unit`
  - `normalize`
  - `fill_value`
  - `constraints`
  - `domain`
  - `sampler`
- keep free functions only as thin convenience aliases over the object API

One additional naming thought:

- if you expect one platform to host multiple sensors or beam modes, consider
  using `Contributor` or `Asset` as the broader public concept, with `Observer`
  as one geometry-oriented implementation detail


## Proposed Public API Examples

### Simplest Possible Usage

```python
coverage = IntervalCoverage.compute(
    observers=sats,          # list[Orbit]
    timeline=times,          # astropy Time array
    targets=config,          # ExactCoverageConfig
)

coverage.access_duration().plot(title="Availability")
coverage.max_asset().plot(title="Peak Overlap")
coverage.target(lat_deg=51.51, lon_deg=-0.13, name="London").timeline().plot()
```


### Mixed Observer Sources

```python
coverage = IntervalCoverage.compute(
    observers=[
        Observer.from_orbit(leo_a, name="LEO-A"),
        Observer.from_orbit(leo_b, name="LEO-B"),
        Observer.from_tabulated(spice_times, spice_pos, frame="itrf", name="SPK-1"),
        Observer.from_samples(pos_uav, timeline=my_timeline, frame="ecef", name="UAV"),
        Observer.from_callable(my_sampler, name="custom"),
    ],
    timeline=my_timeline,
    targets=CoverageTargets.surface_grid(nlats=181, nlons_equator=361),
)
```


### Parameter Sweep

```python
stack = coverage.access_duration(
    min_assets=[1, 2, 3],
    normalize="day",
    unit="hours",
)

stack.sel(min_assets=2).plot(title="N>=2")
```


### Python-Side Custom Analysis

```python
field = coverage.access_duration(min_assets=1, normalize="day", unit="hours")
records = field.to_records()
summary = field.summary()

london = coverage.target(lat_deg=51.51, lon_deg=-0.13, name="London")
timeline = london.timeline()
concurrency_t, concurrency_n = timeline.concurrency_profile()
```


### Compiled Custom Metric

```python
from numba import njit


@njit(cache=True)
def my_metric_kernel(
    pair_offsets,
    start_times,
    stop_times,
    n_observers,
    n_targets,
    t0,
    t1,
    params,
    out,
):
    ...


field = coverage.evaluate(
    CompiledMetric(
        name="my_metric",
        output_dtype=np.float64,
        unit="s",
        label="My Custom Metric",
        kernel=my_metric_kernel,
        params={"threshold": 2},
    )
)
```


## Proposed Module Layout

```text
nstk/coverage/
    __init__.py
    api.py                     # IntervalCoverage, CoverageTimeline, window views
    targets.py                 # CoverageTargets, SurfaceGridMetadata
    observers.py               # Observer, ObserverSource adapters
    results.py                 # CoverageField, CoverageStack, TargetTimeline
    metrics.py                 # MetricSpec, CompiledMetric, built-in metric specs
    plotting.py                # generic result plotting helpers

    low_level/
        __init__.py
        store.py               # IntervalStore
        build.py               # build exact interval store from resolved arrays
        scan.py                # public numba-friendly event scan helpers
        backends.py            # numba/native backend dispatch

    adapters/
        orbit.py               # Orbit observer adapter
        sampled.py             # sampled / tabulated ephemeris adapters
        callable.py            # callable observer adapter
```


## Suggested Rules For The Refactor

1. `IntervalCoverage` becomes the main entrypoint.
2. `CoverageField` stays concrete.
3. `ObserverSource` is the primary extensibility seam.
4. built-in metrics return rich result objects, not bare arrays or dicts.
5. plotting consumes result objects, not raw stores.
6. low-level kernels only see contiguous arrays and primitive scalars.
7. user-defined compiled metrics get a supported stable ABI.
8. user-defined Python analysis remains possible through inspectable results and
   direct access to exact timelines.


## Most Important Decisions

If this redesign is adopted, the clearest architecture choices are:

- `CoverageField` is not an abstract base class
- `IntervalCoverage` is the public face of the coverage system
- `Orbit` is one observer adapter, not the privileged propagation model
- exact interval kernels target a backend-neutral low-level store
- the output model is analysis-first, with plotting layered on top
