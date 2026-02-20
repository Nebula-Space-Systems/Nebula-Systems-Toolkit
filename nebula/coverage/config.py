# config.py

from dataclasses import dataclass, field
import numpy as np
from nebula.coverage._empirical_core import coverage_stamp_kernel_empirical
from nebula.transform import geodetic2ecef
from nebula.transform.constants import WGS84_A, WGS84_E2


@dataclass(frozen=True)
class CoverageConfig:
    # sampling grid
    nlats: int = 181
    nlons: int = 361
    include_lat_endpoints: bool = False
    include_lon_endpoints: bool = False

    # optional visibility constraints on observers
    min_elevation_deg: float = 0.0
    max_elevation_deg: float = 90.0

    # region bounds in degrees
    min_lat: float = -90.0
    max_lat: float = 90.0
    min_lon: float = -180.0
    max_lon: float = 180.0

    # axes / grids
    lat_deg_1d: np.ndarray = field(init=False, repr=False)
    lon_deg_1d: np.ndarray = field(init=False, repr=False)
    lat_rad_1d: np.ndarray = field(init=False, repr=False)
    lon_rad_1d: np.ndarray = field(init=False, repr=False)

    lon_grid_deg: np.ndarray = field(init=False, repr=False)
    lat_grid_deg: np.ndarray = field(init=False, repr=False)

    lat_edges_deg: np.ndarray = field(init=False, repr=False)
    lon_edges_deg: np.ndarray = field(init=False, repr=False)

    dlat_deg: float = field(init=False)
    dlon_deg: float = field(init=False)
    dlon_rad: float = field(init=False)
    half_dlon_rad: float = field(init=False)

    # constraints in radians
    min_elevation_rad: float = field(init=False)
    max_elevation_rad: float = field(init=False)

    # lon extent bookkeeping for rectangular support
    lon_min_deg_cont: float = field(init=False)
    lon_max_deg_cont: float = field(init=False)
    crosses_dateline: bool = field(init=False)
    is_global_lon: bool = field(init=False)

    # row latitudes used by the mask
    lat_row_geod_rad: np.ndarray = field(init=False, repr=False)  # (ny,)
    lat_row_gc_rad: np.ndarray = field(init=False, repr=False)  # (ny,)
    sin_lat_row_gc: np.ndarray = field(init=False, repr=False)  # (ny,)
    cos_lat_row_gc: np.ndarray = field(init=False, repr=False)  # (ny,)

    # lon-domain scalars for the njit kernel
    lon_start_2pi_rad: float = field(init=False)
    lon_span_rad: float = field(init=False)
    lon_base_rad: float = field(init=False)

    # elevation values to pass WITHOUT wrapper logic
    min_el_eff_rad: float = field(init=False)
    max_el_eff_rad: float = field(init=False)

    # ---- compact geometry tables for refine predicate ----
    sin_lon_col: np.ndarray = field(init=False, repr=False)  # (nx,)
    cos_lon_col: np.ndarray = field(init=False, repr=False)  # (nx,)
    sin_lat_row_geod: np.ndarray = field(init=False, repr=False)  # (ny,)
    cos_lat_row_geod: np.ndarray = field(init=False, repr=False)  # (ny,)
    Ncos_row_m: np.ndarray = field(init=False, repr=False)  # (ny,)
    Nz_row_m: np.ndarray = field(init=False, repr=False)  # (ny,)

    # cached stamp and epoch for internal methods
    cached_stamp: np.ndarray = field(init=False, repr=False)
    cached_epoch: np.uint8 = field(init=False, repr=False)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        # optional compatibility with old code
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)

    def __post_init__(self):
        # ---- basic type / value checks ----
        if not (isinstance(self.nlats, int) and self.nlats >= 1):
            raise ValueError("nlats must be an integer >= 1")
        if not (isinstance(self.nlons, int) and self.nlons >= 1):
            raise ValueError("nlons must be an integer >= 1")

        min_lat, max_lat, min_lon, max_lon = (
            self.min_lat,
            self.max_lat,
            self.min_lon,
            self.max_lon,
        )  # after you refactor extent->fields

        # if min_lat > max_lat:
        # raise ValueError("min_lat cannot be greater than max_lat")

        # latitude extent rules
        if self.nlats == 1:
            if not (-90.0 <= min_lat <= 90.0) or not (-90.0 <= max_lat <= 90.0):
                raise ValueError("Latitude must be within [-90, 90]")
            if abs(max_lat - min_lat) > 1e-12:
                raise ValueError("If nlats==1, require min_lat == max_lat")
        else:
            if not (-90.0 <= min_lat < max_lat <= 90.0):
                raise ValueError(
                    "Latitude extent must satisfy -90 <= min_lat < max_lat <= 90"
                )

        # longitude extent rules (supports dateline crossing and full-globe spans)
        if self.nlons == 1:
            if not np.isfinite(min_lon) or not np.isfinite(max_lon):
                raise ValueError("Longitude must be finite")
            if abs(max_lon - min_lon) > 1e-12:
                raise ValueError("If nlons==1, require min_lon == max_lon")
        else:
            if not np.isfinite(min_lon) or not np.isfinite(max_lon):
                raise ValueError("Longitude must be finite")

            raw = float(max_lon - min_lon)

            # If user provided a full-globe span explicitly (e.g., -180..180 or 0..360),
            # raw will be ~360 and should be accepted as 360 (not 0).
            if abs(raw - 360.0) < 1e-9:
                lon_span = 360.0
                crosses = False
                max_lon_cont = min_lon + 360.0
            else:
                # Dateline-crossing allowed: if max<=min, treat as crossing and add 360.
                crosses = max_lon <= min_lon
                max_lon_cont = max_lon + (360.0 if crosses else 0.0)
                lon_span = float(max_lon_cont - min_lon)

            if lon_span <= 0.0 or lon_span > 360.0 + 1e-9:
                raise ValueError(
                    "Longitude span must be in (0, 360] degrees (dateline-crossing allowed)."
                )

            # Store these (or compute them later consistently)
            object.__setattr__(self, "crosses_dateline", bool(crosses))
            object.__setattr__(self, "lon_min_deg_cont", float(min_lon))
            object.__setattr__(self, "lon_max_deg_cont", float(max_lon_cont))
            object.__setattr__(
                self, "is_global_lon", bool(abs(lon_span - 360.0) < 1e-9)
            )

        # ---- 1D sample axes (deg) ----
        if self.nlats == 1:
            lat_deg = np.array([float(min_lat)], dtype=np.float64)
        else:
            lat_deg = np.linspace(
                min_lat, max_lat, self.nlats, endpoint=self.include_lat_endpoints
            )

        if self.nlons == 1:
            lon_deg = np.array([float(min_lon)], dtype=np.float64)
        else:
            lon_deg = np.linspace(
                min_lon, max_lon_cont, self.nlons, endpoint=self.include_lon_endpoints
            )
        object.__setattr__(self, "lat_deg_1d", lat_deg)
        object.__setattr__(self, "lon_deg_1d", lon_deg)

        # ---- radians versions ----
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        object.__setattr__(self, "lat_rad_1d", lat_rad)
        object.__setattr__(self, "lon_rad_1d", lon_rad)

        # ---- step sizes + edges (deg) ----
        # latitude edges
        if self.nlats == 1:
            # choose a sensible default "cell height" in degrees; 1.0 is usually fine for plotting
            dlat = 1.0
            lat_edges = np.array(
                [min_lat - 0.5 * dlat, min_lat + 0.5 * dlat], dtype=np.float64
            )
        else:
            if self.include_lat_endpoints:
                dlat = (max_lat - min_lat) / (self.nlats - 1)
                lat_edges = (min_lat - 0.5 * dlat) + np.arange(self.nlats + 1) * dlat
                lat_edges[0] = min_lat
                lat_edges[-1] = max_lat
            else:
                dlat = (max_lat - min_lat) / self.nlats
                lat_edges = min_lat + np.arange(self.nlats + 1) * dlat

        # longitude edges
        if self.nlons == 1:
            dlon = 1.0
            lon_edges = np.array(
                [min_lon - 0.5 * dlon, min_lon + 0.5 * dlon], dtype=np.float64
            )
        else:
            if self.include_lon_endpoints:
                dlon = (max_lon_cont - min_lon) / (self.nlons - 1)
                lon_edges = (min_lon - 0.5 * dlon) + np.arange(self.nlons + 1) * dlon
                lon_edges[0] = min_lon
                lon_edges[-1] = max_lon_cont
            else:
                dlon = (max_lon_cont - min_lon) / self.nlons
                lon_edges = min_lon + np.arange(self.nlons + 1) * dlon

        object.__setattr__(self, "dlat_deg", float(dlat))
        object.__setattr__(self, "dlon_deg", float(dlon))
        object.__setattr__(self, "lat_edges_deg", lat_edges)
        object.__setattr__(self, "lon_edges_deg", lon_edges)

        if abs(dlat) < 1e-12:
            raise ValueError(f"dlat is too small or zero ({dlat}).")
        if abs(dlon) < 1e-12:
            raise ValueError(f"dlon is too small or zero ({dlon}).")

        dlon_rad = float(np.deg2rad(dlon))
        object.__setattr__(self, "dlon_rad", dlon_rad)
        object.__setattr__(self, "half_dlon_rad", 0.5 * dlon_rad)

        # ---- constraints in radians (convert once) ----
        min_el = float(np.deg2rad(self.min_elevation_deg))
        max_el = float(np.deg2rad(self.max_elevation_deg))
        object.__setattr__(self, "min_elevation_rad", min_el)
        object.__setattr__(self, "max_elevation_rad", max_el)
        object.__setattr__(self, "min_el_eff_rad", float(min_el))
        object.__setattr__(self, "max_el_eff_rad", float(max_el))

        # ---- latitude rows used by the mask (geodetic) ----
        if self.include_lat_endpoints:
            lat_row_geod_rad = np.asarray(self.lat_rad_1d, dtype=np.float64)
        else:
            lat_cent_deg = 0.5 * (lat_edges[:-1] + lat_edges[1:])
            lat_row_geod_rad = np.deg2rad(lat_cent_deg).astype(np.float64)

        ONE_MINUS_E2 = 1.0 - WGS84_E2
        lat_row_gc_rad = np.arctan(ONE_MINUS_E2 * np.tan(lat_row_geod_rad)).astype(
            np.float64
        )

        object.__setattr__(self, "lat_row_geod_rad", lat_row_geod_rad)
        object.__setattr__(self, "lat_row_gc_rad", lat_row_gc_rad)
        object.__setattr__(
            self, "sin_lat_row_gc", np.sin(lat_row_gc_rad).astype(np.float64)
        )
        object.__setattr__(
            self, "cos_lat_row_gc", np.cos(lat_row_gc_rad).astype(np.float64)
        )

        # ---- precompute lon-domain scalars for njit kernel ----
        lon_min_rad = float(self.lon_min_deg_cont) * (np.pi / 180.0)
        lon_span_rad = float(self.lon_max_deg_cont - self.lon_min_deg_cont) * (
            np.pi / 180.0
        )
        lon_start_2pi = (lon_min_rad + np.pi) % (2.0 * np.pi)

        if self.include_lon_endpoints:
            lon_base = lon_start_2pi
        else:
            lon_base = lon_start_2pi + float(self.half_dlon_rad)

        object.__setattr__(self, "lon_start_2pi_rad", float(lon_start_2pi))
        object.__setattr__(self, "lon_span_rad", float(lon_span_rad))
        object.__setattr__(self, "lon_base_rad", float(lon_base))

        # ---- 2D grids corresponding to mask rows/cols (kept) ----
        if self.include_lon_endpoints:
            lon_cent_deg = lon_deg
        else:
            lon_cent_deg = 0.5 * (lon_edges[:-1] + lon_edges[1:])

        if self.include_lat_endpoints:
            lat_cent_deg = lat_deg
        else:
            lat_cent_deg = 0.5 * (lat_edges[:-1] + lat_edges[1:])

        lon_grid, lat_grid = np.meshgrid(lon_cent_deg, lat_cent_deg)
        object.__setattr__(self, "lon_grid_deg", lon_grid)
        object.__setattr__(self, "lat_grid_deg", lat_grid)

        # ---- compact per-row/per-col geometry for refine predicate ----
        lon_col_rad = np.deg2rad(lon_cent_deg).astype(np.float64)
        object.__setattr__(self, "sin_lon_col", np.sin(lon_col_rad).astype(np.float64))
        object.__setattr__(self, "cos_lon_col", np.cos(lon_col_rad).astype(np.float64))

        lat_row = np.asarray(lat_row_geod_rad, dtype=np.float64)
        sin_lat = np.sin(lat_row).astype(np.float64)
        cos_lat = np.cos(lat_row).astype(np.float64)
        object.__setattr__(self, "sin_lat_row_geod", sin_lat)
        object.__setattr__(self, "cos_lat_row_geod", cos_lat)

        N_row = (WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)).astype(
            np.float64
        )
        object.__setattr__(self, "Ncos_row_m", (N_row * cos_lat).astype(np.float64))
        object.__setattr__(
            self, "Nz_row_m", (ONE_MINUS_E2 * N_row * sin_lat).astype(np.float64)
        )

        object.__setattr__(self, "cached_stamp", self.make_stamp())
        object.__setattr__(self, "cached_epoch", self.make_epoch())

    def lon_grid_for_plot(self) -> np.ndarray:
        lon = self.lon_grid_deg
        return ((lon + 180.0) % 360.0) - 180.0

    def make_stamp(self) -> np.ndarray:
        return np.zeros((self.nlats, self.nlons), dtype=np.uint8)

    def make_epoch(self) -> np.uint8:
        return np.uint8(0)

    @staticmethod
    def mask_from_stamp(stamp: np.ndarray, epoch: np.uint8) -> np.ndarray:
        return stamp == epoch

    def get_visibility_mask_ecef(
        self,
        obs_x_m: float,
        obs_y_m: float,
        obs_z_m: float,
    ) -> np.ndarray:
        object.__setattr__(self, "cached_epoch", self.cached_epoch + np.uint8(1))
        if self.cached_epoch == 0:
            self.cached_stamp.fill(0)
            object.__setattr__(self, "cached_epoch", np.uint8(1))

        coverage_stamp_kernel_empirical(
            float(obs_x_m),
            float(obs_y_m),
            float(obs_z_m),
            self.lat_row_gc_rad,
            self.lon_start_2pi_rad,
            self.lon_span_rad,
            self.dlon_rad,
            self.lon_base_rad,
            self.nlons,
            self.min_el_eff_rad,
            self.max_el_eff_rad,
            self.cached_stamp,
            self.cached_epoch,
            self.Ncos_row_m,
            self.Nz_row_m,
            self.cos_lat_row_geod,
            self.sin_lat_row_geod,
            self.cos_lon_col,
            self.sin_lon_col,
        )

        return self.cached_stamp == self.cached_epoch

    def get_visibility_mask_geodetic(
        self,
        obs_lat: float,
        obs_lon: float,
        obs_height_m: float,
        degrees: bool = True,
    ) -> np.ndarray:
        if degrees:
            obs_lat = np.deg2rad(obs_lat)
            obs_lon = np.deg2rad(obs_lon)

        x, y, z = geodetic2ecef(float(obs_lat), float(obs_lon), float(obs_height_m))
        return self.get_visibility_mask_ecef(x, y, z)
