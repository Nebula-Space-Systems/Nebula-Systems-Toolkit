from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ExactCoverageConfig:
    """
    Coverage target definition for interval-based coverage computation.

    Unlike fixed-dt cap/raster methods, this config supports non-uniform longitude
    density by latitude to avoid pole oversampling.
    """

    nlats: int = 181
    nlons_equator: int = 361
    scale_longitude_by_latitude: bool = True
    min_lon_points_per_row: int = 1

    min_lat: float = -90.0
    max_lat: float = 90.0
    min_lon: float = -180.0
    max_lon: float = 180.0

    include_lat_endpoints: bool = True
    include_lon_endpoints: bool = False

    min_elevation_deg: float = 0.0
    max_elevation_deg: float = 90.0

    lat_deg_rows: np.ndarray = field(init=False, repr=False)
    lat_rad_rows: np.ndarray = field(init=False, repr=False)
    lon_deg_flat: np.ndarray = field(init=False, repr=False)
    lat_deg_flat: np.ndarray = field(init=False, repr=False)
    row_sizes: np.ndarray = field(init=False, repr=False)
    row_offsets: np.ndarray = field(init=False, repr=False)
    n_targets: int = field(init=False)
    target_shape: tuple[int, int] | None = field(init=False)
    lon_min_deg_cont: float = field(init=False)
    lon_max_deg_cont: float = field(init=False)
    lon_span_deg: float = field(init=False)

    def __post_init__(self) -> None:
        if not (isinstance(self.nlats, int) and self.nlats >= 1):
            raise ValueError("nlats must be an integer >= 1")
        if not (isinstance(self.nlons_equator, int) and self.nlons_equator >= 1):
            raise ValueError("nlons_equator must be an integer >= 1")
        if not (
            isinstance(self.min_lon_points_per_row, int)
            and self.min_lon_points_per_row >= 1
        ):
            raise ValueError("min_lon_points_per_row must be an integer >= 1")

        if self.nlats == 1:
            if abs(float(self.max_lat) - float(self.min_lat)) > 1e-12:
                raise ValueError("If nlats==1, require min_lat == max_lat")
        else:
            if not (-90.0 <= self.min_lat < self.max_lat <= 90.0):
                raise ValueError("Require -90 <= min_lat < max_lat <= 90")

        if not np.isfinite(self.min_lon) or not np.isfinite(self.max_lon):
            raise ValueError("Longitude bounds must be finite")

        raw = float(self.max_lon - self.min_lon)
        if abs(raw - 360.0) < 1e-9:
            lon_min_cont = float(self.min_lon)
            lon_max_cont = float(self.min_lon + 360.0)
        else:
            crosses = self.max_lon <= self.min_lon
            lon_min_cont = float(self.min_lon)
            lon_max_cont = float(self.max_lon + (360.0 if crosses else 0.0))

        lon_span = float(lon_max_cont - lon_min_cont)
        if lon_span <= 0.0 or lon_span > 360.0 + 1e-9:
            raise ValueError("Longitude span must be in (0, 360] degrees")

        if self.nlats == 1:
            lat_rows = np.array([float(self.min_lat)], dtype=np.float64)
        else:
            if self.include_lat_endpoints:
                lat_rows = np.linspace(
                    float(self.min_lat),
                    float(self.max_lat),
                    int(self.nlats),
                    endpoint=True,
                    dtype=np.float64,
                )
            else:
                dlat = (float(self.max_lat) - float(self.min_lat)) / float(self.nlats)
                lat_rows = float(self.min_lat) + (np.arange(self.nlats) + 0.5) * dlat
                lat_rows = lat_rows.astype(np.float64)

        span_frac = lon_span / 360.0
        base_n_lon = max(1, int(np.round(float(self.nlons_equator) * span_frac)))
        min_per_row = min(int(self.min_lon_points_per_row), base_n_lon)

        row_sizes = np.empty(self.nlats, dtype=np.int32)
        for j in range(self.nlats):
            lat_j = float(lat_rows[j])

            if self.scale_longitude_by_latitude:
                # Exact poles are degenerate in longitude: keep one point.
                if abs(abs(lat_j) - 90.0) <= 1e-12:
                    row_sizes[j] = 1
                    continue

                w = np.cos(np.deg2rad(lat_j))
                if w < 0.0:
                    w = 0.0
                n_j = int(np.round(float(base_n_lon) * w))
                if n_j < min_per_row:
                    n_j = min_per_row
                if n_j < 1:
                    n_j = 1
            else:
                n_j = int(base_n_lon)
            row_sizes[j] = n_j

        row_offsets = np.empty(self.nlats + 1, dtype=np.int64)
        row_offsets[0] = 0
        np.cumsum(row_sizes, out=row_offsets[1:])
        n_targets = int(row_offsets[-1])

        lon_flat = np.empty(n_targets, dtype=np.float64)
        lat_flat = np.empty(n_targets, dtype=np.float64)

        idx = 0
        for j in range(self.nlats):
            n_j = int(row_sizes[j])
            if n_j == 1:
                lon_row = np.array([0.5 * (lon_min_cont + lon_max_cont)], dtype=np.float64)
            else:
                if self.include_lon_endpoints:
                    lon_row = np.linspace(
                        lon_min_cont, lon_max_cont, n_j, endpoint=True, dtype=np.float64
                    )
                else:
                    dlon_j = lon_span / float(n_j)
                    lon_row = lon_min_cont + (np.arange(n_j) + 0.5) * dlon_j
                    lon_row = lon_row.astype(np.float64)

            lon_row = ((lon_row + 180.0) % 360.0) - 180.0
            lon_flat[idx : idx + n_j] = lon_row
            lat_flat[idx : idx + n_j] = lat_rows[j]
            idx += n_j

        target_shape = None
        if not self.scale_longitude_by_latitude:
            target_shape = (int(self.nlats), int(base_n_lon))

        object.__setattr__(self, "lat_deg_rows", lat_rows)
        object.__setattr__(self, "lat_rad_rows", np.deg2rad(lat_rows).astype(np.float64))
        object.__setattr__(self, "lon_deg_flat", lon_flat)
        object.__setattr__(self, "lat_deg_flat", lat_flat)
        object.__setattr__(self, "row_sizes", row_sizes)
        object.__setattr__(self, "row_offsets", row_offsets)
        object.__setattr__(self, "n_targets", n_targets)
        object.__setattr__(self, "target_shape", target_shape)
        object.__setattr__(self, "lon_min_deg_cont", lon_min_cont)
        object.__setattr__(self, "lon_max_deg_cont", lon_max_cont)
        object.__setattr__(self, "lon_span_deg", lon_span)
