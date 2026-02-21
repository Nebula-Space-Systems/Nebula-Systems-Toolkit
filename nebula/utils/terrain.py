from read_dem import RasterDEM
from read_geoid import EGM2008Geoid
import numpy as np


class Terrain:
    def __init__(self, dem: RasterDEM, geoid: EGM2008Geoid):
        self.dem = dem
        self.geoid = geoid

    def get_orthometric_height(
        self, latitudes: np.ndarray, longitudes: np.ndarray, fill_value=0.0
    ) -> np.ndarray:
        """
        Get orthometric heights (elevations above mean sea level) for given coordinates.

        This method retrieves the orthometric height by querying the DEM (Digital Elevation Model)
        for the specified latitude and longitude coordinates. Orthometric height represents the
        height above the geoid (mean sea level), as opposed to ellipsoidal height.

        Args:
            latitudes (np.ndarray): Array of latitude values in decimal degrees.
            longitudes (np.ndarray): Array of longitude values in decimal degrees.
            fill_value (float, optional): Value to use for locations where height data is not
                available or invalid. Defaults to 0.0.

        Returns:
            np.ndarray: Array of orthometric heights in meters (or the DEM's native units)
                corresponding to the input coordinates. Returns fill_value for invalid locations.

        Example:
            >>> lats = np.array([47.6062, 47.6101])
            >>> lons = np.array([-122.3321, -122.3421])
            >>> heights = terrain.get_orthometric_height(lats, lons)
        """
        height = self.dem.get_height(latitudes, longitudes, fill_value=fill_value)
        return height

    def get_undulation(
        self, latitudes: np.ndarray, longitudes: np.ndarray, fill_value=0.0
    ) -> np.ndarray:
        """
        Get geoid undulation values for given coordinates.

        This method retrieves the geoid undulation (height difference between the geoid
        and the reference ellipsoid) for arrays of latitude and longitude coordinates.

        Args:
            latitudes (np.ndarray): Array of latitude values in decimal degrees.
            longitudes (np.ndarray): Array of longitude values in decimal degrees.
            fill_value (float, optional): Value to use for invalid or out-of-bounds
                coordinates. Defaults to 0.0.

        Returns:
            np.ndarray: Array of geoid undulation values in meters, corresponding to
                the input coordinate arrays.

        Example:
            >>> lats = np.array([40.7128, 51.5074])
            >>> lons = np.array([-74.0060, -0.1278])
            >>> undulations = terrain.get_undulation(lats, lons)
        """
        undulation = self.geoid.get_undulation(
            latitudes, longitudes, fill_value=fill_value
        )
        return undulation

    def get_ellipsoidal_height(
        self, latitudes: np.ndarray, longitudes: np.ndarray, fill_value=0.0
    ) -> np.ndarray:
        """
        Get ellipsoidal heights for given coordinates.

        This method calculates ellipsoidal heights (elevations above the WGS84 reference
        ellipsoid) by combining Digital Elevation Model (DEM) heights with geoid undulations.

        Args:
            latitudes (np.ndarray): Array of latitude values in decimal degrees.
            longitudes (np.ndarray): Array of longitude values in decimal degrees.
            fill_value (float, optional): Value to use for locations where data is unavailable.
                Defaults to 0.0.

        Returns:
            np.ndarray: Array of ellipsoidal heights in meters, with the same shape as the
                input coordinate arrays. Each value represents the height above the WGS84
                ellipsoid at the corresponding latitude/longitude position.

        Notes:
            The ellipsoidal height is calculated as:
            ellipsoidal_height = DEM_height + geoid_undulation

            Where:
            - DEM_height is the orthometric height (height above mean sea level)
            - geoid_undulation is the separation between the geoid and ellipsoid
        """
        dem_heights = self.dem.get_height(latitudes, longitudes, fill_value=fill_value)
        geoid_heights = self.geoid.get_undulation(
            latitudes, longitudes, fill_value=fill_value
        )
        ellipsoidal_heights = dem_heights + geoid_heights
        return ellipsoidal_heights

    def get_native_ellipsoidal_height(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        fill_value=0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get native resolution ellipsoidal heights for a bounding box.

        This method retrieves elevation data from the DEM and converts it to ellipsoidal
        heights by adding geoid undulations.

        Args:
            lat_min (float): Minimum latitude of the bounding box in degrees.
            lat_max (float): Maximum latitude of the bounding box in degrees.
            lon_min (float): Minimum longitude of the bounding box in degrees.
            lon_max (float): Maximum longitude of the bounding box in degrees.
            fill_value (float, optional): Value to use for areas with no data. Defaults to 0.0.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - lat_grid: 2D array of latitude values
                - lon_grid: 2D array of longitude values
                - orthometric_heights: 2D array of ellipsoidal heights in meters
        """
        lat_grid, lon_grid, elev_grid = self.dem.get_height_native(
            lat_min, lat_max, lon_min, lon_max, fill_value=fill_value
        )
        undulations = self.geoid.get_undulation(
            lat_grid, lon_grid, fill_value=fill_value
        )
        orthometric_heights = elev_grid + undulations
        return lat_grid, lon_grid, orthometric_heights

    def get_native_orthometric_height(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        fill_value=0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get native resolution orthometric height data for a specified bounding box.

        This method retrieves elevation data at the dataset's native resolution for the
        specified geographic bounds. The returned data is in orthometric height (height
        above the geoid).

        Args:
            lat_min (float): Minimum latitude of the bounding box in decimal degrees.
            lat_max (float): Maximum latitude of the bounding box in decimal degrees.
            lon_min (float): Minimum longitude of the bounding box in decimal degrees.
            lon_max (float): Maximum longitude of the bounding box in decimal degrees.
            fill_value (float, optional): Value to use for missing or invalid data points.
                Defaults to 0.0.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
                - lat_grid: 2D array of latitude values corresponding to each grid point.
                - lon_grid: 2D array of longitude values corresponding to each grid point.
                - elev_grid: 2D array of elevation values in meters above the geoid.
        """
        lat_grid, lon_grid, elev_grid = self.dem.get_height_native(
            lat_min, lat_max, lon_min, lon_max, fill_value=fill_value
        )
        return lat_grid, lon_grid, elev_grid

    def close(self):
        self.dem.close()
