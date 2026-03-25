from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nstk.transforms.constants import WGS84_A, WGS84_E2

from .domains import (
    BBoxDomain,
    CountryDomain,
    GlobalEarthDomain,
    LandDomain,
    OceanDomain,
    PolygonDomain,
    TargetDomain,
    _combine_domains,
    coerce_domain,
)

_PLASTIC_CONSTANT = 1.3247179572447458
_R2_ALPHA_1 = 1.0 / _PLASTIC_CONSTANT
_R2_ALPHA_2 = 1.0 / (_PLASTIC_CONSTANT * _PLASTIC_CONSTANT)


def _geodetic_to_surface(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    prime_vertical_radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    xy_radius = prime_vertical_radius * cos_lat
    z_ecef = (1.0 - WGS84_E2) * prime_vertical_radius * sin_lat

    position = np.column_stack(
        (
            xy_radius * cos_lon,
            xy_radius * sin_lon,
            z_ecef,
        )
    ).astype(np.float64)
    up = np.column_stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)).astype(np.float64)
    return position, up


def _haversine_score(lat0_deg: float, lon0_deg: float, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    dlat = lat - lat0
    dlon = (lon - lon0 + np.pi) % (2.0 * np.pi) - np.pi
    return np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2


def _wrap_longitude(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _continuous_longitude_bounds(west_deg: float, east_deg: float) -> tuple[float, float]:
    west = float(west_deg)
    east = float(east_deg)
    raw = east - west
    if abs(raw - 360.0) < 1e-9:
        return west, west + 360.0
    if east <= west:
        east += 360.0
    span = east - west
    if span <= 0.0 or span > 360.0 + 1e-9:
        raise ValueError("Longitude span must be in (0, 360] degrees")
    return west, east


def _axis_edges(axis: np.ndarray, *, clip_min: float | None = None, clip_max: float | None = None) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64)
    if arr.size == 1:
        width = 1.0
        if clip_min is not None and clip_max is not None:
            width = float(clip_max) - float(clip_min)
        edges = np.asarray([arr[0] - 0.5 * width, arr[0] + 0.5 * width], dtype=np.float64)
    else:
        edges = np.empty(arr.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
        edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
        edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    if clip_min is not None:
        edges = np.maximum(edges, float(clip_min))
    if clip_max is not None:
        edges = np.minimum(edges, float(clip_max))
    return edges


def _bounded_equal_area_candidates(
    bounds: tuple[float, float, float, float],
    total: int,
) -> tuple[np.ndarray, np.ndarray]:
    west, east, south, north = bounds
    lon_span = float(east) - float(west)
    if lon_span <= 0.0:
        lon_span += 360.0

    idx = np.arange(int(total), dtype=np.float64)
    u = np.mod(0.5 + _R2_ALPHA_1 * idx, 1.0)
    v = np.mod(0.5 + _R2_ALPHA_2 * idx, 1.0)

    sin_south = np.sin(np.deg2rad(float(south)))
    sin_north = np.sin(np.deg2rad(float(north)))
    sin_lat = sin_south + (sin_north - sin_south) * u
    lat = np.rad2deg(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))
    lon = _wrap_longitude(float(west) + lon_span * v)
    return lat, lon


def _spread_sample_indices(n_keep: int, n_requested: int) -> np.ndarray:
    if n_requested < 1:
        raise ValueError("n_requested must be >= 1")
    if n_keep < n_requested:
        raise ValueError("n_keep must be >= n_requested")
    positions = (np.arange(n_requested, dtype=np.float64) + 0.5) * (n_keep / float(n_requested))
    return np.floor(positions).astype(np.int64)


def _domain_bounds(domain: TargetDomain) -> tuple[float, float, float, float]:
    bounds = domain.bounds() or (-180.0, 180.0, -90.0, 90.0)
    west, east, south, north = bounds
    return float(west), float(east), float(south), float(north)


def _require_positive_int(value: int, *, field_name: str) -> int:
    count = int(value)
    if count < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return count


@dataclass(frozen=True)
class SurfaceGridMetadata:
    """Structured latitude/longitude grid metadata for gridded target sets."""

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    target_index_grid: np.ndarray
    boundary_geometry: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the ``(n_lat, n_lon)`` shape of the source surface grid."""
        return tuple(self.target_index_grid.shape)  # type: ignore[return-value]


@runtime_checkable
class TargetSampler(Protocol):
    """Protocol for objects that can materialize a domain into coverage targets."""

    def materialize(self, domain: TargetDomain) -> "CoverageTargets":
        """Return a concrete :class:`CoverageTargets` sample for ``domain``."""
        ...


def _coerce_target_sampler(sampler: TargetSampler) -> TargetSampler:
    if not isinstance(sampler, TargetSampler):
        raise TypeError("sampler must implement TargetSampler.materialize(domain)")
    return sampler


def _build_materialized_targets(
    *,
    domain: TargetDomain,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    area_weights: np.ndarray,
    sampler_name: str,
    surface_grid: SurfaceGridMetadata | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> "CoverageTargets":
    positions, up = _geodetic_to_surface(lat_deg, lon_deg)
    attrs = {"sampler": sampler_name, "domain": domain.name}
    if extra_attrs:
        attrs.update(extra_attrs)
    return CoverageTargets(
        positions_ecef_m=positions,
        up_vectors_ecef=up,
        lat_deg=np.asarray(lat_deg, dtype=np.float64),
        lon_deg=np.asarray(lon_deg, dtype=np.float64),
        area_weights=np.asarray(area_weights, dtype=np.float64),
        surface_grid=surface_grid,
        boundary_geometry=domain.boundary_geometry(),
        outline_geometry=domain.outline_geometry(),
        attrs=attrs,
    )


@dataclass(frozen=True)
class CoverageTargets:
    """Materialized Earth-surface coverage targets in WGS84 Earth-fixed coordinates.

    Instances store one surface sample per target, including geodetic latitude and
    longitude, Earth-fixed Cartesian position, local surface ``up`` direction, and
    area weights for aggregation.
    """

    positions_ecef_m: np.ndarray
    up_vectors_ecef: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    area_weights: np.ndarray
    labels: list[str] | None = None
    surface_grid: SurfaceGridMetadata | None = None
    boundary_geometry: Any | None = None
    outline_geometry: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and normalize array-backed fields."""
        pos = np.asarray(self.positions_ecef_m, dtype=np.float64)
        up = np.asarray(self.up_vectors_ecef, dtype=np.float64)
        lat = np.asarray(self.lat_deg, dtype=np.float64)
        lon = np.asarray(self.lon_deg, dtype=np.float64)
        weights = np.asarray(self.area_weights, dtype=np.float64)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions_ecef_m must have shape (n_targets, 3)")
        if up.shape != pos.shape:
            raise ValueError("up_vectors_ecef must have shape (n_targets, 3)")
        if lat.shape != (pos.shape[0],) or lon.shape != (pos.shape[0],):
            raise ValueError("lat_deg and lon_deg must match n_targets")
        if weights.shape != (pos.shape[0],):
            raise ValueError("area_weights must match n_targets")
        if pos.shape[0] == 0:
            raise ValueError("CoverageTargets must contain at least one target")

        object.__setattr__(self, "positions_ecef_m", np.ascontiguousarray(pos))
        object.__setattr__(self, "up_vectors_ecef", np.ascontiguousarray(up))
        object.__setattr__(self, "lat_deg", np.ascontiguousarray(lat))
        object.__setattr__(self, "lon_deg", np.ascontiguousarray(lon))
        object.__setattr__(self, "area_weights", np.ascontiguousarray(weights))

    @property
    def n_targets(self) -> int:
        """Return the number of materialized target points."""
        return int(self.positions_ecef_m.shape[0])

    def nearest_target_index(self, *, lat_deg: float, lon_deg: float) -> int:
        """Return the index of the sampled target nearest to ``(lat_deg, lon_deg)``."""
        return int(np.argmin(_haversine_score(lat_deg, lon_deg, self.lat_deg, self.lon_deg)))

    def to_grid(self, values: np.ndarray, *, fill_value: float = np.nan) -> np.ndarray:
        """Project one value per target back onto the original structured surface grid."""
        if self.surface_grid is None:
            raise ValueError("CoverageTargets does not include structured surface grid metadata")
        vals = np.asarray(values)
        if vals.shape != (self.n_targets,):
            raise ValueError(f"values must have shape ({self.n_targets},)")
        grid = np.full(self.surface_grid.shape, fill_value, dtype=np.result_type(vals, np.float64))
        mask = self.surface_grid.target_index_grid >= 0
        grid[mask] = vals[self.surface_grid.target_index_grid[mask]]
        return grid

    def extent(
        self,
        *,
        pad_deg: float = 0.0,
        pad_lon_deg: float | None = None,
        pad_lat_deg: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return a map extent as ``(west, east, south, north)`` in degrees."""
        lon_pad = float(pad_deg if pad_lon_deg is None else pad_lon_deg)
        lat_pad = float(pad_deg if pad_lat_deg is None else pad_lat_deg)
        if lon_pad < 0.0 or lat_pad < 0.0:
            raise ValueError("extent padding must be >= 0")

        if self.boundary_geometry is not None and not getattr(self.boundary_geometry, "is_empty", False):
            min_lon, min_lat, max_lon, max_lat = self.boundary_geometry.bounds
        else:
            min_lon = float(np.min(self.lon_deg))
            max_lon = float(np.max(self.lon_deg))
            min_lat = float(np.min(self.lat_deg))
            max_lat = float(np.max(self.lat_deg))

        west = max(-180.0, float(min_lon) - lon_pad)
        east = min(180.0, float(max_lon) + lon_pad)
        south = max(-90.0, float(min_lat) - lat_pad)
        north = min(90.0, float(max_lat) + lat_pad)
        return (west, east, south, north)

    def subset(self, indices: np.ndarray | list[int]) -> "CoverageTargets":
        """Return a new target set containing only the selected target indices."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be 1D")
        new_surface = None
        if self.surface_grid is not None:
            remap = np.full(self.n_targets, -1, dtype=np.int64)
            remap[idx] = np.arange(idx.size, dtype=np.int64)
            new_grid = remap[self.surface_grid.target_index_grid.clip(min=0)]
            new_grid[self.surface_grid.target_index_grid < 0] = -1
            new_surface = SurfaceGridMetadata(
                lat_deg=self.surface_grid.lat_deg,
                lon_deg=self.surface_grid.lon_deg,
                target_index_grid=new_grid,
                boundary_geometry=self.surface_grid.boundary_geometry,
            )
        labels = None if self.labels is None else [self.labels[int(i)] for i in idx]
        return CoverageTargets(
            positions_ecef_m=self.positions_ecef_m[idx],
            up_vectors_ecef=self.up_vectors_ecef[idx],
            lat_deg=self.lat_deg[idx],
            lon_deg=self.lon_deg[idx],
            area_weights=self.area_weights[idx],
            labels=labels,
            surface_grid=new_surface,
            boundary_geometry=self.boundary_geometry,
            outline_geometry=self.outline_geometry,
            attrs=dict(self.attrs),
        )

    def select_domain(self, domain: TargetDomain) -> np.ndarray:
        """Return indices of targets whose sampled lat/lon fall inside ``domain``."""
        mask = domain.contains_latlon(self.lat_deg, self.lon_deg)
        return np.flatnonzero(mask)

    @classmethod
    def from_domain(
        cls,
        domain: Any,
        *,
        sampler: TargetSampler,
    ) -> "CoverageTargets":
        """Materialize a domain-like object with the provided target sampler."""
        domain_obj = coerce_domain(domain) if not isinstance(domain, TargetDomain) else domain
        return _coerce_target_sampler(sampler).materialize(domain_obj)

    @classmethod
    def global_earth(
        cls,
        *,
        sampler: TargetSampler,
    ) -> "CoverageTargets":
        """Sample the full Earth surface."""
        return cls.from_domain(GlobalEarthDomain(), sampler=sampler)

    @classmethod
    def country(
        cls,
        name: str,
        *,
        sampler: TargetSampler,
        resolution: str = "110m",
    ) -> "CoverageTargets":
        """Sample one Natural Earth country polygon by name."""
        return cls.from_domain(CountryDomain(names=(name,), resolution=resolution), sampler=sampler)

    @classmethod
    def countries(
        cls,
        names: list[str],
        *,
        sampler: TargetSampler,
        resolution: str = "110m",
    ) -> "CoverageTargets":
        """Sample the union of multiple Natural Earth countries."""
        return cls.from_domain(CountryDomain(names=tuple(names), resolution=resolution), sampler=sampler)

    @classmethod
    def region_bbox(
        cls,
        west_deg: float,
        east_deg: float,
        south_deg: float,
        north_deg: float,
        *,
        sampler: TargetSampler,
    ) -> "CoverageTargets":
        """Sample a rectangular latitude/longitude region."""
        return cls.from_domain(
            BBoxDomain(west_deg=west_deg, east_deg=east_deg, south_deg=south_deg, north_deg=north_deg),
            sampler=sampler,
        )

    @classmethod
    def from_geojson(cls, path: str, *, sampler: TargetSampler) -> "CoverageTargets":
        """Sample a polygonal region loaded from a GeoJSON file."""
        return cls.from_domain(PolygonDomain.from_geojson(path), sampler=sampler)

    @classmethod
    def from_shapefile(cls, path: str, *, sampler: TargetSampler) -> "CoverageTargets":
        """Sample a polygonal region loaded from a shapefile."""
        return cls.from_domain(PolygonDomain.from_shapefile(path), sampler=sampler)

    @classmethod
    def land(cls, *, sampler: TargetSampler, resolution: str = "110m") -> "CoverageTargets":
        """Sample the Natural Earth land mask."""
        return cls.from_domain(LandDomain(resolution=resolution), sampler=sampler)

    @classmethod
    def ocean(cls, *, sampler: TargetSampler, resolution: str = "110m") -> "CoverageTargets":
        """Sample the Natural Earth ocean mask."""
        return cls.from_domain(OceanDomain(resolution=resolution), sampler=sampler)

    @classmethod
    def union(
        cls,
        *parts: Any,
        sampler: TargetSampler,
        resolution: str = "110m",
        name: str | None = None,
    ) -> "CoverageTargets":
        """Materialize one target set from the union of domain-like inputs."""
        return cls.from_domain(
            _combine_target_parts(*parts, resolution=resolution, name=name),
            sampler=sampler,
        )


def _coerce_target_domain_part(part: Any, *, resolution: str) -> TargetDomain:
    if isinstance(part, CoverageTargets):
        if part.boundary_geometry is None:
            raise ValueError(
                "CoverageTargets.union requires CoverageTargets inputs to carry boundary_geometry"
            )
        domain_name = part.attrs.get("domain") if isinstance(part.attrs, dict) else None
        return PolygonDomain(
            geometry=part.boundary_geometry,
            name=domain_name if isinstance(domain_name, str) else None,
        )
    if isinstance(part, str):
        return CountryDomain(names=(part,), resolution=resolution)
    if isinstance(part, Sequence) and not isinstance(part, (str, bytes)):
        if len(part) == 0:
            raise ValueError("CoverageTargets.union requires at least one domain-like item")
        if all(isinstance(item, str) for item in part):
            return CountryDomain(names=tuple(str(item) for item in part), resolution=resolution)
        nested = [_coerce_target_domain_part(item, resolution=resolution) for item in part]
        return _combine_domains(*nested, op="union")
    return part if isinstance(part, TargetDomain) else coerce_domain(part)


def _combine_target_parts(*parts: Any, resolution: str, name: str | None = None) -> TargetDomain:
    domains = [_coerce_target_domain_part(part, resolution=resolution) for part in parts]
    return _combine_domains(*domains, op="union", name=name)


@dataclass(frozen=True)
class LatitudeLongitudeSampler:
    """Sample a domain on a structured latitude/longitude grid."""

    nlats: int = 181
    nlons: int = 361

    def materialize(self, domain: TargetDomain) -> CoverageTargets:
        """Return structured grid targets clipped to ``domain``."""
        nlats = _require_positive_int(self.nlats, field_name="LatitudeLongitudeSampler.nlats")
        nlons = _require_positive_int(self.nlons, field_name="LatitudeLongitudeSampler.nlons")

        west, east, south, north = _domain_bounds(domain)
        lat_axis = np.linspace(south, north, nlats, dtype=np.float64)
        lon_axis = np.linspace(west, east, nlons, dtype=np.float64)
        lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
        include = domain.contains_latlon(lat_grid, lon_grid)

        target_index_grid = np.full(lat_grid.shape, -1, dtype=np.int64)
        selected = np.flatnonzero(include.ravel())
        target_index_grid.ravel()[selected] = np.arange(selected.size, dtype=np.int64)

        lat_flat = lat_grid.ravel()[selected]
        lon_flat = lon_grid.ravel()[selected]

        dlat = abs(lat_axis[1] - lat_axis[0]) if lat_axis.size > 1 else 180.0
        dlon = abs(lon_axis[1] - lon_axis[0]) if lon_axis.size > 1 else 360.0
        weights = np.cos(np.deg2rad(lat_flat)) * np.deg2rad(dlat) * np.deg2rad(dlon)
        weights = np.abs(weights)

        return _build_materialized_targets(
            domain=domain,
            lat_deg=lat_flat,
            lon_deg=lon_flat,
            area_weights=weights,
            sampler_name="latitude_longitude",
            surface_grid=SurfaceGridMetadata(
                lat_deg=lat_axis,
                lon_deg=lon_axis,
                target_index_grid=target_index_grid,
                boundary_geometry=domain.boundary_geometry(),
            ),
        )


@dataclass(frozen=True)
class LatitudeAdaptiveSampler:
    """Sample latitude rows with longitude density that scales by latitude."""

    nlats: int = 181
    nlons_equator: int = 361
    min_lon_points_per_row: int = 1
    include_lat_endpoints: bool = True
    include_lon_endpoints: bool = False

    def materialize(self, domain: TargetDomain) -> CoverageTargets:
        """Return targets from an equal-latitude-row sampler clipped to ``domain``."""
        nlats = _require_positive_int(self.nlats, field_name="LatitudeAdaptiveSampler.nlats")
        nlons_equator = _require_positive_int(
            self.nlons_equator,
            field_name="LatitudeAdaptiveSampler.nlons_equator",
        )
        min_lon_points_per_row = _require_positive_int(
            self.min_lon_points_per_row,
            field_name="LatitudeAdaptiveSampler.min_lon_points_per_row",
        )

        west, east, south, north = _domain_bounds(domain)

        if nlats == 1:
            lat_rows = np.asarray([south], dtype=np.float64)
        elif self.include_lat_endpoints:
            lat_rows = np.linspace(south, north, nlats, dtype=np.float64)
        else:
            dlat = (north - south) / float(nlats)
            lat_rows = south + (np.arange(nlats, dtype=np.float64) + 0.5) * dlat

        lon_min_cont, lon_max_cont = _continuous_longitude_bounds(west, east)
        lon_span = float(lon_max_cont - lon_min_cont)
        base_n_lon = max(1, int(np.round(float(nlons_equator) * (lon_span / 360.0))))
        min_per_row = min(min_lon_points_per_row, base_n_lon)
        lat_edges = _axis_edges(lat_rows, clip_min=-90.0, clip_max=90.0)

        lat_chunks: list[np.ndarray] = []
        lon_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        lat_rows_used: list[float] = []
        row_counts: list[int] = []

        for row_idx, lat in enumerate(lat_rows):
            if abs(abs(float(lat)) - 90.0) <= 1e-12:
                n_lon = 1
            else:
                n_lon = int(np.round(float(base_n_lon) * max(0.0, np.cos(np.deg2rad(float(lat))))))
                n_lon = max(min_per_row, n_lon)
                n_lon = max(1, n_lon)

            if n_lon == 1:
                lon_cont = np.asarray([0.5 * (lon_min_cont + lon_max_cont)], dtype=np.float64)
                dlon_deg = lon_span
            elif self.include_lon_endpoints:
                lon_cont = np.linspace(lon_min_cont, lon_max_cont, n_lon, endpoint=True, dtype=np.float64)
                dlon_deg = lon_span / float(max(1, n_lon - 1))
            else:
                dlon_deg = lon_span / float(n_lon)
                lon_cont = lon_min_cont + (np.arange(n_lon, dtype=np.float64) + 0.5) * dlon_deg

            lon_row = _wrap_longitude(lon_cont)
            lat_row = np.full(n_lon, float(lat), dtype=np.float64)
            include = np.asarray(domain.contains_latlon(lat_row, lon_row), dtype=bool)
            if not np.any(include):
                continue

            sin_band = np.sin(np.deg2rad(float(lat_edges[row_idx + 1]))) - np.sin(np.deg2rad(float(lat_edges[row_idx])))
            cell_area = abs(float(sin_band) * np.deg2rad(float(dlon_deg)))
            n_keep = int(np.count_nonzero(include))

            lat_chunks.append(lat_row[include])
            lon_chunks.append(lon_row[include])
            weight_chunks.append(np.full(n_keep, cell_area, dtype=np.float64))
            lat_rows_used.append(float(lat))
            row_counts.append(n_keep)

        if not lat_chunks:
            raise ValueError("LatitudeAdaptiveSampler produced no targets for the requested domain")

        lat = np.concatenate(lat_chunks)
        lon = np.concatenate(lon_chunks)
        weights = np.concatenate(weight_chunks)
        row_offsets = np.empty(len(row_counts) + 1, dtype=np.int64)
        row_offsets[0] = 0
        np.cumsum(np.asarray(row_counts, dtype=np.int64), out=row_offsets[1:])
        return _build_materialized_targets(
            domain=domain,
            lat_deg=lat,
            lon_deg=lon,
            area_weights=weights,
            sampler_name="latitude_adaptive",
            extra_attrs={
                "nlats": int(self.nlats),
                "nlons_equator": int(self.nlons_equator),
                "lat_rows_deg": np.asarray(lat_rows_used, dtype=np.float64),
                "row_offsets": row_offsets,
                "lon_min_cont_deg": float(lon_min_cont),
                "lon_max_cont_deg": float(lon_max_cont),
            },
        )


@dataclass(frozen=True)
class EqualAreaSampler:
    """Sample approximately equal-area target points within a domain."""

    target_count: int = 4096

    def materialize(self, domain: TargetDomain) -> CoverageTargets:
        """Return approximately equal-area targets clipped to ``domain``."""
        n_requested = _require_positive_int(
            self.target_count,
            field_name="EqualAreaSampler.target_count",
        )

        bounds = _domain_bounds(domain)
        total = max(n_requested * 2, n_requested + 64)
        max_total = max(total, min(4_194_304, n_requested * 16_384))
        lat: np.ndarray | None = None
        lon: np.ndarray | None = None

        for _ in range(12):
            cand_lat, cand_lon = _bounded_equal_area_candidates(bounds, total)
            keep = np.flatnonzero(domain.contains_latlon(cand_lat, cand_lon))
            if keep.size >= n_requested:
                selected = keep[_spread_sample_indices(int(keep.size), n_requested)]
                lat = cand_lat[selected]
                lon = cand_lon[selected]
                break

            if total >= max_total:
                break

            if keep.size == 0:
                next_total = total * 4
            else:
                acceptance = keep.size / float(total)
                next_total = int(np.ceil((n_requested / acceptance) * 1.5))
            total = min(max_total, max(total * 2, next_total))

        if lat is None or lon is None:
            raise RuntimeError(
                "Unable to materialize enough equal-area targets for the requested domain"
            )

        weights = np.full(lat.shape, 1.0 / max(1, lat.size), dtype=np.float64)
        return _build_materialized_targets(
            domain=domain,
            lat_deg=lat,
            lon_deg=lon,
            area_weights=weights,
            sampler_name="equal_area",
        )


__all__ = [
    "SurfaceGridMetadata",
    "TargetSampler",
    "CoverageTargets",
    "LatitudeLongitudeSampler",
    "LatitudeAdaptiveSampler",
    "EqualAreaSampler",
]
