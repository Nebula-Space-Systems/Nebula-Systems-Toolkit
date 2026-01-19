import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nebula.plots.map import (
    make_basemap,
    DARK_DETAILED,
    LIGHT_DETAILED,
    DARK_RASTER,
    LIGHT_RASTER,
    DARK,
    LIGHT,
    DARK_DETAILED_NO_GRID,
    LIGHT_DETAILED_NO_GRID,
    DARK_NO_GRID,
    LIGHT_NO_GRID,
    DARK_RASTER_NO_GRID,
    LIGHT_RASTER_NO_GRID,
    plt,
)

if __name__ == "__main__":
    # 1) Light + raster background (prebuilt config)
    make_basemap(LIGHT_RASTER)
    make_basemap(DARK_RASTER)

    # 2) Light + bump texture + borders enabled
    make_basemap(LIGHT_DETAILED)
    make_basemap(DARK_DETAILED)

    # 3) dark + bump texture + borders enabled
    make_basemap(LIGHT)
    make_basemap(DARK)

    # no all the no grid variants
    make_basemap(LIGHT_DETAILED_NO_GRID)
    make_basemap(DARK_DETAILED_NO_GRID)

    make_basemap(LIGHT_NO_GRID)
    make_basemap(DARK_NO_GRID)

    make_basemap(LIGHT_RASTER_NO_GRID)
    make_basemap(DARK_RASTER_NO_GRID)

    plt.show()
