import time
from typing import Union

import numpy as np
import rasterio
from rasterio.windows import Window
from numba import njit

ArrayLike = Union[float, np.ndarray]


@njit
def _bilinear_interp_block(block, rows_f, cols_f, fill_value):
    """
    Numba-accelerated bilinear interpolation on a single in-memory block.

    Parameters
    ----------
    block : 2D float64 array
        DEM values (with nodata already converted to NaN and scale applied).
    rows_f, cols_f : 1D float64 arrays
        Fractional row/col coordinates in the block (same length).
    fill_value : float
        Value to use if interpolation is not possible.

    Returns
    -------
    out : 1D float64 array
        Interpolated values, same length as rows_f/cols_f.
    """
    n = rows_f.shape[0]
    out = np.empty(n, dtype=np.float32)
    h, w = block.shape

    for i in range(n):
        r = rows_f[i]
        c = cols_f[i]

        # Integer top-left indices in the block
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))

        # Check bounds for a full 2x2 neighborhood
        if r0 < 0 or r0 + 1 >= h or c0 < 0 or c0 + 1 >= w:
            out[i] = fill_value
            continue

        dr = r - r0
        dc = c - c0

        f00 = block[r0, c0]
        f10 = block[r0, c0 + 1]
        f01 = block[r0 + 1, c0]
        f11 = block[r0 + 1, c0 + 1]

        num = 0.0
        den = 0.0

        # Corner 00
        if not np.isnan(f00):
            w00 = (1.0 - dc) * (1.0 - dr)
            num += w00 * f00
            den += w00

        # Corner 10
        if not np.isnan(f10):
            w10 = dc * (1.0 - dr)
            num += w10 * f10
            den += w10

        # Corner 01
        if not np.isnan(f01):
            w01 = (1.0 - dc) * dr
            num += w01 * f01
            den += w01

        # Corner 11
        if not np.isnan(f11):
            w11 = dc * dr
            num += w11 * f11
            den += w11

        if den > 0.0:
            out[i] = num / den
        else:
            out[i] = fill_value

    return out


class RasterDEM:
    """
    DEM wrapper that reads a single-band GeoTIFF and provides
    bilinear interpolation at arbitrary lat/lon positions.

    Assumes the raster CRS is geographic (lon, lat), e.g. EPSG:4326.
    It does NOT load the whole raster into memory.
    """

    def __init__(
        self,
        tif_path: str,
        band: int = 1,
        scale: float = 1.0,
        nodata: float | None = None,
    ):
        self.dataset = rasterio.open(tif_path)
        self.band = band

        # Scale factor applied to raw pixel values (e.g. 0.1 if stored in decimeters).
        self.scale = float(scale)

        # Nodata value (override or from dataset). If None and dataset has one, use it.
        if nodata is not None:
            self.nodata = nodata
        else:
            self.nodata = self.dataset.nodata

        self.transform = self.dataset.transform
        self.inv_transform = ~self.transform  # (x=lon, y=lat) -> (col, row)

        self.height = self.dataset.height
        self.width = self.dataset.width

    def get_height(
        self, lats: ArrayLike, lons: ArrayLike, fill_value=np.nan
    ) -> np.ndarray:
        """
        Bilinearly interpolate elevations at given lat/lon(s).

        Parameters
        ----------
        lats, lons : float or array-like
            Latitude(s) and longitude(s) in degrees in the raster CRS (e.g. EPSG:4326).
            They will be broadcast to a common shape.
        fill_value : float, optional
            Value to use for points falling outside the raster or where interpolation
            is not possible.

        Returns
        -------
        elevations : np.ndarray
            Elevations with the same shape as the broadcast lat/lon.
        """
        # Broadcast inputs
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        lats, lons = np.broadcast_arrays(lats, lons)
        original_shape = lats.shape

        # Flatten for vectorized operations
        lats_flat = lats.ravel()
        lons_flat = lons.ravel()

        # Convert lon/lat to fractional pixel coordinates (col, row)
        cols, rows = self.inv_transform * (lons_flat, lats_flat)
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)

        # Integer indices of the top-left neighbor
        col0 = np.floor(cols).astype(np.int64)
        row0 = np.floor(rows).astype(np.int64)
        col1 = col0 + 1
        row1 = row0 + 1

        # Valid where we have a full 2x2 neighborhood inside the raster bounds
        valid = (row0 >= 0) & (row1 < self.height) & (col0 >= 0) & (col1 < self.width)

        elev_flat = np.full(cols.shape, fill_value, dtype=np.float64)

        if not np.any(valid):
            # Everything is out-of-bounds
            return elev_flat.reshape(original_shape)

        # Indices of valid points in the flattened arrays
        valid_idx = np.nonzero(valid)[0]

        # For valid points, compute the bounding window we need to read once
        row0_v = row0[valid]
        row1_v = row1[valid]
        col0_v = col0[valid]
        col1_v = col1[valid]

        row_off = int(row0_v.min())
        col_off = int(col0_v.min())
        row_max = int(row1_v.max())
        col_max = int(col1_v.max())

        win_height = row_max - row_off + 1
        win_width = col_max - col_off + 1

        # Read a single block covering all valid points
        window = Window(
            col_off=col_off,  # type: ignore
            row_off=row_off,  # type: ignore
            width=win_width,  # type: ignore
            height=win_height,  # type: ignore
        )
        arr = self.dataset.read(self.band, window=window)

        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(
                    "Expected a single band but got count != 1 in window read"
                )
            arr2 = arr[0]
        else:
            arr2 = arr

        # Convert to float32 and apply scale
        vals = arr2.astype(np.float32)
        if self.scale != 1.0:
            vals *= self.scale

        # Apply nodata -> NaN if we have a nodata sentinel
        if self.nodata is not None:
            nodata_mask = (
                arr2 == self.nodata
            )  # use original integer array for comparison
            vals[nodata_mask] = np.nan

        # Local fractional coordinates inside the block for valid points
        rows_local = rows[valid_idx] - row_off
        cols_local = cols[valid_idx] - col_off

        # Numba-accelerated bilinear interpolation on the in-memory block
        out_vals = _bilinear_interp_block(vals, rows_local, cols_local, fill_value)

        # Scatter back into the full flat output
        elev_flat[valid_idx] = out_vals

        return elev_flat.reshape(original_shape)

    def get_height_native(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        fill_value=np.nan,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the *native* DEM samples (no interpolation) and their
        lat/lon grids within the given bounding box.

        The returned lat/lon correspond to pixel centers of the DEM.

        Parameters
        ----------
        lat_min, lat_max : float
            Latitude bounds in degrees.
        lon_min, lon_max : float
            Longitude bounds in degrees.
        fill_value : float, optional
            Value to use for pixels with nodata. Defaults to NaN.

        Returns
        -------
        lat_grid : 2D np.ndarray
            Latitudes of pixel centers (deg), shape (H, W).
        lon_grid : 2D np.ndarray
            Longitudes of pixel centers (deg), shape (H, W).
        elev_grid : 2D np.ndarray
            Elevations (scaled, nodata -> NaN), shape (H, W).
        """
        # Ensure ordering
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        if lon_min > lon_max:
            lon_min, lon_max = lon_max, lon_min

        # Clip to dataset bounds
        bounds = self.dataset.bounds  # left, bottom, right, top
        lon_min_c = max(lon_min, bounds.left)
        lon_max_c = min(lon_max, bounds.right)
        lat_min_c = max(lat_min, bounds.bottom)
        lat_max_c = min(lat_max, bounds.top)

        if lon_min_c >= lon_max_c or lat_min_c >= lat_max_c:
            raise ValueError("Requested bounding box does not intersect DEM extent.")

        # Get pixel indices for the corners (rasterio: index(x, y) = (row, col))
        row_ul, col_ul = self.dataset.index(lon_min_c, lat_max_c)
        row_lr, col_lr = self.dataset.index(lon_max_c, lat_min_c)

        # Ensure correct ordering
        row_off = max(0, min(row_ul, row_lr))
        col_off = max(0, min(col_ul, col_lr))
        row_max = min(max(row_ul, row_lr), self.height - 1)
        col_max = min(max(col_ul, col_lr), self.width - 1)

        win_height = row_max - row_off + 1
        win_width = col_max - col_off + 1

        # Read the native block
        window = Window(
            col_off=col_off,  # type: ignore
            row_off=row_off,  # type: ignore
            width=win_width,  # type: ignore
            height=win_height,  # type: ignore
        )
        arr = self.dataset.read(self.band, window=window)

        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(
                    "Expected a single band but got count != 1 in window read"
                )
            arr2 = arr[0]
        else:
            arr2 = arr

        # Scale and apply nodata -> NaN
        elev = arr2.astype(np.float32)
        if self.scale != 1.0:
            elev *= self.scale
        if self.nodata is not None:
            nodata_mask = arr2 == self.nodata
            elev[nodata_mask] = fill_value

        # Build lat/lon grids for pixel centers.
        # Handle north-up Affine safely using attributes.
        transform = self.transform
        try:
            a = transform.a
            b = transform.b
            c = transform.c
            d = transform.d
            e = transform.e
            f = transform.f
        except AttributeError:
            # Fallback if transform is array-like instead of Affine
            a, b, c, d, e, f = tuple(transform)[:6]

        if abs(b) > 1e-12 or abs(d) > 1e-12:
            raise ValueError(
                "get_height_native assumes a north-up (unrotated) transform."
            )

        cols_idx = np.arange(col_off, col_off + win_width, dtype=np.float64)
        rows_idx = np.arange(row_off, row_off + win_height, dtype=np.float64)

        # Pixel center coordinates
        lon_1d = c + (cols_idx + 0.5) * a
        lat_1d = f + (rows_idx + 0.5) * e  # e is negative for north-up rasters

        lat_grid, lon_grid = np.meshgrid(lat_1d, lon_1d, indexing="ij")

        return lat_grid, lon_grid, elev

    def close(self):
        self.dataset.close()


if __name__ == "__main__":
    tif_path = r"J:\downloads\terrain\gedtm30\gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v20250611.tif"

    dem = RasterDEM(tif_path, scale=0.1, nodata=-2147483648)

    lat_min, lat_max = 10.5, 11.5
    lon_min, lon_max = 0.0, 1.0

    n = 2000
    lats = np.linspace(lat_min, lat_max, n)
    lons = np.linspace(lon_min, lon_max, n)
    lat_grid, lon_grid = np.meshgrid(lats, lons)

    # Example using the original grid API
    t0 = time.time()
    elev_grid = dem.get_height(lat_grid, lon_grid)
    t1 = time.time()
    print(f"[grid] Sampled {elev_grid.size} points in {t1 - t0:.3f} s")

    t0 = time.time()
    elev_grid = dem.get_height(lat_grid, lon_grid)
    t1 = time.time()
    print(f"[grid] Sampled {elev_grid.size} points in {t1 - t0:.3f} s")

    dem.close()
