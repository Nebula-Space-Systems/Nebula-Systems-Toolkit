from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Sequence

import numpy as np
import shapefile
from shapely import contains_xy
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.ops import unary_union

from nstk._data_dependency import get_installed_cartopy_data_dir


def _norm_text(s: str) -> str:
    text = "" if s is None else str(s)
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set(s: str) -> set[str]:
    return {token for token in _norm_text(s).split(" ") if token}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(len(a | b))


def _natural_earth_shapefile(*, resolution: str, category: str, name: str) -> Path:
    return (
        get_installed_cartopy_data_dir()
        / "shapefiles"
        / "natural_earth"
        / category
        / f"ne_{resolution}_{name}.shp"
    )


def _load_shapefile_records(path: Path) -> list[tuple[dict[str, Any], Any]]:
    reader = shapefile.Reader(str(path))
    field_names = [field[0] for field in reader.fields[1:]]
    out: list[tuple[dict[str, Any], Any]] = []
    for shape_record in reader.iterShapeRecords():
        attrs = {
            field_names[idx]: value
            for idx, value in enumerate(shape_record.record)
        }
        geom = shape(shape_record.shape.__geo_interface__)
        out.append((attrs, geom))
    return out


@lru_cache(maxsize=None)
def _country_records(resolution: str) -> tuple[tuple[dict[str, Any], Any], ...]:
    path = _natural_earth_shapefile(
        resolution=resolution,
        category="cultural",
        name="admin_0_countries",
    )
    return tuple(_load_shapefile_records(path))


@lru_cache(maxsize=None)
def _land_geometry(resolution: str) -> Any:
    path = _natural_earth_shapefile(
        resolution=resolution,
        category="physical",
        name="land",
    )
    geoms = [geom for _, geom in _load_shapefile_records(path)]
    return unary_union(geoms)


@lru_cache(maxsize=None)
def _coastline_geometry(resolution: str) -> Any:
    path = _natural_earth_shapefile(
        resolution=resolution,
        category="physical",
        name="coastline",
    )
    geoms = [geom for _, geom in _load_shapefile_records(path)]
    return unary_union(geoms)


def _global_geometry() -> Polygon:
    return box(-180.0, -90.0, 180.0, 90.0)


def _best_country_geometry(names: Sequence[str], resolution: str) -> Any:
    records = _country_records(resolution)
    geometries: list[Any] = []
    keys = (
        "ADMIN",
        "NAME",
        "NAME_LONG",
        "FORMAL_EN",
        "SOVEREIGNT",
        "ABBREV",
        "ISO_A2",
        "ISO_A3",
        "ADM0_A3",
    )
    for query in names:
        query_norm = _norm_text(query)
        query_tokens = _token_set(query)
        best_score = -1.0
        best_geom = None
        for attrs, geom in records:
            local_best = 0.0
            for key in keys:
                value = attrs.get(key)
                if not value:
                    continue
                val_norm = _norm_text(str(value))
                score = _jaccard(query_tokens, _token_set(str(value)))
                if query_norm == val_norm:
                    score = 1.0
                elif query_norm in val_norm or val_norm in query_norm:
                    score = max(score, 0.85)
                if score > local_best:
                    local_best = score
            if local_best > best_score:
                best_score = local_best
                best_geom = geom
        if best_geom is None or best_score < 0.35:
            raise ValueError(f"No Natural Earth country match found for {query!r}")
        geometries.append(best_geom)
    return unary_union(geometries)


@dataclass(frozen=True, kw_only=True)
class TargetDomain:
    """Abstract geographic target domain."""

    name: str | None = None

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def bounds(self) -> tuple[float, float, float, float] | None:
        raise NotImplementedError

    def boundary_geometry(self) -> Any | None:
        return None

    def outline_geometry(self) -> Any | None:
        return self.boundary_geometry()

    def __or__(self, other: "TargetDomain") -> "CompositeDomain":
        return self.union(other)

    def __add__(self, other: Any) -> "TargetDomain":
        return self.union(other)

    def __and__(self, other: "TargetDomain") -> "CompositeDomain":
        return self.intersection(other)

    def __sub__(self, other: "TargetDomain") -> "CompositeDomain":
        return self.difference(other)

    def __radd__(self, other: Any) -> "TargetDomain":
        if other == 0:
            return self
        return _combine_domains(other, self, op="union")

    def union(self, *others: Any, name: str | None = None) -> "TargetDomain":
        """Return the union of this domain and the provided domains."""
        return _combine_domains(self, *others, op="union", name=name)

    def intersection(self, *others: Any, name: str | None = None) -> "TargetDomain":
        """Return the intersection of this domain and the provided domains."""
        return _combine_domains(self, *others, op="intersection", name=name)

    def difference(self, *others: Any, name: str | None = None) -> "TargetDomain":
        """Return this domain with the provided domains removed."""
        return _combine_domains(self, *others, op="difference", name=name)


def _combine_domains(*items: Any, op: str, name: str | None = None) -> TargetDomain:
    domains: list[TargetDomain] = []
    for item in items:
        domain = item if isinstance(item, TargetDomain) else coerce_domain(item)
        if isinstance(domain, CompositeDomain) and domain.op == op:
            domains.extend(domain.items)
        else:
            domains.append(domain)
    if len(domains) == 0:
        raise ValueError("At least one target domain is required")
    if len(domains) == 1:
        return domains[0]
    return CompositeDomain(op=op, items=tuple(domains), name=name)


@dataclass(frozen=True, kw_only=True)
class GlobalEarthDomain(TargetDomain):
    name: str | None = "Global Earth"

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        return (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= -90.0)
            & (lat <= 90.0)
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        return (-180.0, 180.0, -90.0, 90.0)

    def boundary_geometry(self) -> Any | None:
        return _global_geometry()

    def outline_geometry(self) -> Any | None:
        return None


@dataclass(frozen=True, kw_only=True)
class BBoxDomain(TargetDomain):
    west_deg: float = -180.0
    east_deg: float = 180.0
    south_deg: float = -90.0
    north_deg: float = 90.0
    name: str | None = None

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        return (
            (lat >= self.south_deg)
            & (lat <= self.north_deg)
            & (lon >= self.west_deg)
            & (lon <= self.east_deg)
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        return (self.west_deg, self.east_deg, self.south_deg, self.north_deg)

    def boundary_geometry(self) -> Any | None:
        return box(self.west_deg, self.south_deg, self.east_deg, self.north_deg)


@dataclass(frozen=True, kw_only=True)
class PolygonDomain(TargetDomain):
    geometry: Any = field(default_factory=GeometryCollection)
    name: str | None = None

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        return np.asarray(
            contains_xy(self.geometry, np.asarray(lon_deg), np.asarray(lat_deg)),
            dtype=bool,
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        if self.geometry.is_empty:
            return None
        minx, miny, maxx, maxy = self.geometry.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def boundary_geometry(self) -> Any | None:
        return self.geometry

    @classmethod
    def from_geojson(cls, path: str | Path, *, name: str | None = None) -> "PolygonDomain":
        payload = json.loads(Path(path).read_text())
        if payload.get("type") == "FeatureCollection":
            geoms = [shape(feature["geometry"]) for feature in payload["features"]]
            geom = unary_union(geoms)
        elif payload.get("type") == "Feature":
            geom = shape(payload["geometry"])
        else:
            geom = shape(payload)
        return cls(geometry=geom, name=name or Path(path).stem)

    @classmethod
    def from_shapefile(cls, path: str | Path, *, name: str | None = None) -> "PolygonDomain":
        geoms = [geom for _, geom in _load_shapefile_records(Path(path))]
        return cls(geometry=unary_union(geoms), name=name or Path(path).stem)


@dataclass(frozen=True, kw_only=True)
class CountryDomain(TargetDomain):
    names: tuple[str, ...] = ()
    resolution: str = "110m"
    name: str | None = None
    geometry: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.names) == 0:
            raise ValueError("CountryDomain requires at least one country name")
        geom = _best_country_geometry(self.names, self.resolution)
        object.__setattr__(self, "geometry", geom)
        if self.name is None:
            object.__setattr__(self, "name", ", ".join(self.names))

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        return np.asarray(
            contains_xy(self.geometry, np.asarray(lon_deg), np.asarray(lat_deg)),
            dtype=bool,
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        minx, miny, maxx, maxy = self.geometry.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def boundary_geometry(self) -> Any | None:
        return self.geometry


@dataclass(frozen=True, kw_only=True)
class LandDomain(TargetDomain):
    resolution: str = "110m"
    name: str | None = "Land"
    geometry: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", _land_geometry(self.resolution))

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        return np.asarray(
            contains_xy(self.geometry, np.asarray(lon_deg), np.asarray(lat_deg)),
            dtype=bool,
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        minx, miny, maxx, maxy = self.geometry.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def boundary_geometry(self) -> Any | None:
        return self.geometry

    def outline_geometry(self) -> Any | None:
        return _coastline_geometry(self.resolution)


@dataclass(frozen=True, kw_only=True)
class OceanDomain(TargetDomain):
    resolution: str = "110m"
    name: str | None = "Ocean"
    geometry: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        geom = _global_geometry().difference(_land_geometry(self.resolution))
        object.__setattr__(self, "geometry", geom)

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        return np.asarray(
            contains_xy(self.geometry, np.asarray(lon_deg), np.asarray(lat_deg)),
            dtype=bool,
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        minx, miny, maxx, maxy = self.geometry.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def boundary_geometry(self) -> Any | None:
        return self.geometry

    def outline_geometry(self) -> Any | None:
        return _coastline_geometry(self.resolution)


@dataclass(frozen=True, kw_only=True)
class CompositeDomain(TargetDomain):
    op: str = "union"
    items: tuple[TargetDomain, ...] = ()
    name: str | None = None
    geometry: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.items) == 0:
            raise ValueError("CompositeDomain requires at least one item")
        geoms = [item.boundary_geometry() for item in self.items]
        if any(geom is None for geom in geoms):
            raise ValueError("All composite domains must expose boundary geometry")
        if self.op == "union":
            geom = unary_union(geoms)
        elif self.op == "intersection":
            geom = geoms[0]
            for other in geoms[1:]:
                geom = geom.intersection(other)
        elif self.op == "difference":
            geom = geoms[0]
            for other in geoms[1:]:
                geom = geom.difference(other)
        else:
            raise ValueError("CompositeDomain.op must be one of: union, intersection, difference")
        object.__setattr__(self, "geometry", geom)
        if self.name is None:
            joined = f" {self.op} ".join(item.name or "domain" for item in self.items)
            object.__setattr__(self, "name", joined)

    def contains_latlon(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        return np.asarray(
            contains_xy(self.geometry, np.asarray(lon_deg), np.asarray(lat_deg)),
            dtype=bool,
        )

    def bounds(self) -> tuple[float, float, float, float] | None:
        minx, miny, maxx, maxy = self.geometry.bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def boundary_geometry(self) -> Any | None:
        return self.geometry


def coerce_domain(value: Any) -> TargetDomain:
    if isinstance(value, TargetDomain):
        return value
    if isinstance(value, (Polygon, MultiPolygon, GeometryCollection)):
        return PolygonDomain(geometry=value)
    if isinstance(value, dict):
        return PolygonDomain(geometry=shape(value))
    raise TypeError(f"Unsupported target domain: {type(value)!r}")


__all__ = [
    "TargetDomain",
    "GlobalEarthDomain",
    "BBoxDomain",
    "PolygonDomain",
    "CountryDomain",
    "LandDomain",
    "OceanDomain",
    "CompositeDomain",
    "coerce_domain",
]
