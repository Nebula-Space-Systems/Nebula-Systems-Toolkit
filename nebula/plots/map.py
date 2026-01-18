import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Literal
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

ProjectionName = Literal[
    "PlateCarree", "Robinson", "Mollweide", "Mercator", "Orthographic"
]
NEScale = Literal["110m", "50m", "10m"]


@dataclass(frozen=True)
class MapStyle:
    """Colors and line widths for the dark map theme."""

    face: str = "#171717"
    land_fc: str = "#4D4D4D"
    ocean_fc_plain: str = "#1C1C1C"
    ocean_fc_textured: str = "#0F0F0F"
    coast_edge: str = "#0F0F0F"
    borders_edge: str = "#707070"
    grid_color: str = "#CFCFCF"
    grid_label_color: str = "#DFDFDF"
    axes_edge: str = "#616161"
    track_default: str = "C1"
    coast_lw: float = 0.4
    border_lw: float = 0.4


DARK_THEME = MapStyle(
    face="#171717",
    land_fc="#4D4D4D",
    ocean_fc_plain="#1C1C1C",
    ocean_fc_textured="#0F0F0F",
    coast_edge="#0F0F0F",
    borders_edge="#707070",
    grid_color="#CFCFCF",
    grid_label_color="#DFDFDF",
    axes_edge="#616161",
    track_default="C1",
    coast_lw=0.4,
    border_lw=0.4,
)

LIGHT_THEME = MapStyle(
    face="#FFFFFF",
    land_fc="#E6E6E6",
    ocean_fc_plain="#FFFFFF",
    ocean_fc_textured="#FFFFFF",
    coast_edge="#B5B5B5",
    borders_edge="#555555",
    grid_color="#B5B5B5",
    grid_label_color="#222222",
    axes_edge="#FFFFFF",
    track_default="C0",
    coast_lw=0.4,
    border_lw=0.4,
)


def make_basemap(
    *,
    projection: ProjectionName = "PlateCarree",
    figsize: Tuple[float, float] = (12.0, 6.0),
    draw_coastlines: bool = True,
    draw_countries: bool = False,
    draw_parallels: bool = True,
    draw_meridians: bool = True,
    gridline_kwargs: Optional[Mapping[str, Any]] = None,
    ne_scale: NEScale = "110m",
    add_land: bool = True,
    add_ocean: bool = True,
    zorder_land: float = 0.0,
    texture: bool = True,
    texture_path: Optional[str] = None,
    texture_opacity: float = 0.2,
    style: MapStyle = DARK_THEME,
    use_raster_background: bool = False,
    raster_background_path: Optional[str] = None,
):
    """
    Create a Cartopy base map (figure + GeoAxes).

    If `texture=True`, a topo+bathy *bump* texture is applied in a color-safe way:
    it only modulates brightness (highlights + shadows) and does not introduce gray
    tinting or rely on semi-transparent land/ocean fills.

    Notes
    -----
    - `texture_path` should point to an equirectangular global image (lon: -180..180, lat: 90..-90).
      For your case, set `texture_path="topo_and_bathy.png"` (or leave None; it will default to that name).
    """

    def _load_luminance(path: str) -> np.ndarray:
        img = mpimg.imread(path)
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[-1] >= 3:
            # sRGB-ish luminance (good enough for bump/shade use)
            lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        else:
            lum = img.astype(np.float32)
        lum = np.asarray(lum, dtype=np.float32)
        # Some loaders return uint8 in [0,255]
        if lum.max() > 1.0:
            lum = lum / 255.0
        return np.clip(lum, 0.0, 1.0)

    def _add_bump_shading(
        ax,
        *,
        path: str,
        strength: float = 1.15,
        gamma: float = 1.0,
        zorder: float = 1.25,
        opacity: float = 0.4,
    ) -> None:
        """
        Apply bump shading without changing hue:
        - highlights: white overlay where bump > mid
        - shadows:   black overlay where bump < mid
        This changes brightness only (no gray tint from a single semi-transparent raster).
        """

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Texture file not found: {path!r}. "
                "Set texture_path to your topo+bathy bump map (e.g. 'topo_and_bathy.png')."
            )

        lum = _load_luminance(path)
        if gamma != 1.0:
            lum = np.clip(lum, 0.0, 1.0) ** gamma

        # Center around mid-gray and build separate highlight/shadow alpha masks.
        d = lum - 0.5

        # Alpha masks (tuned to be subtle but visible; adjust strength if desired)
        a_hi = np.clip(d * strength * opacity, 0.0, 1.0)
        a_lo = np.clip((-d) * strength * opacity, 0.0, 1.0)

        h, w = lum.shape
        white = np.ones((h, w, 3), dtype=np.float32)
        black = np.zeros((h, w, 3), dtype=np.float32)

        img_hi = np.dstack([white, a_hi]).astype(np.float32)
        img_lo = np.dstack([black, a_lo]).astype(np.float32)

        common = dict(
            extent=[-180, 180, -90, 90],
            transform=ccrs.PlateCarree(),
            origin="upper",
            interpolation="bilinear",
            zorder=zorder,
        )

        # Draw highlight then shadow; net effect is relief without hue shift.
        ax.imshow(img_hi, **common)
        ax.imshow(img_lo, **common)

    # --- Matplotlib rcParams for the dark look (kept local and minimal) ---
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "figure.facecolor": style.face,
            "figure.edgecolor": style.face,
            "axes.facecolor": style.face,
            "axes.edgecolor": style.axes_edge,
            "text.color": style.grid_label_color,
            "xtick.color": style.grid_label_color,
            "ytick.color": style.grid_label_color,
        }
    )

    # --- Projection selection ---
    proj_map = {
        "PlateCarree": ccrs.PlateCarree(),
        "Robinson": ccrs.Robinson(),
        "Mollweide": ccrs.Mollweide(),
        "Mercator": ccrs.Mercator(),
        "Orthographic": ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0),
    }
    crs = proj_map[projection]

    fig = plt.figure(figsize=figsize, facecolor=style.face)
    ax = plt.axes(projection=crs)
    ax.set_facecolor(style.face)
    ax.set_global()

    # --- Background ---
    if use_raster_background:
        if raster_background_path is None:
            raster_background_path = os.path.join(
                cartopy.config["data_dir"], "raster", "shadedrelief.jpg"
            )
        img = mpimg.imread(raster_background_path)
        ax.imshow(
            img,
            extent=[-180, 180, -90, 90],
            transform=ccrs.PlateCarree(),
            origin="upper",
            interpolation="bilinear",
            zorder=-10,
            alpha=1.0,
        )
    else:
        # Solid-color base fills (no transparency tricks that alter perceived color)
        ocean_fc = style.ocean_fc_textured if texture else style.ocean_fc_plain

        if add_ocean:
            ax.add_feature(
                cfeature.OCEAN.with_scale(ne_scale),
                facecolor=ocean_fc,
                edgecolor="none",
                alpha=1.0,
                zorder=zorder_land + 0.0,
            )

        if add_land:
            ax.add_feature(
                cfeature.LAND.with_scale(ne_scale),
                facecolor=style.land_fc,
                edgecolor="none",
                alpha=1.0,
                zorder=zorder_land + 1.0,
            )

        # Apply topo+bathy bump shading ABOVE land/ocean fills (brightness-only modulation)
        if texture:
            if texture_path is None:
                texture_path = os.path.join(
                    cartopy.config["data_dir"], "raster", "topo_and_bathy.png"
                )
            _add_bump_shading(
                ax,
                path=texture_path,
                strength=1.15,
                gamma=1.0,
                zorder=zorder_land + 1.25,
                opacity=texture_opacity,
            )

        # Optional subtle land shadow (kept very light; bump provides the main relief)
        if add_land:
            land_shadow = ax.add_feature(
                cfeature.LAND.with_scale(ne_scale),
                facecolor="black",
                edgecolor="none",
                alpha=0.12 if texture else 0.18,
                zorder=zorder_land + 1.30,
            )
            land_shadow.set_path_effects(
                [
                    pe.SimplePatchShadow(
                        offset=(-0.6, -0.6),
                        alpha=0.25 if texture else 0.35,
                        shadow_rgbFace="black",
                    )
                ]
            )

        if draw_coastlines:
            ax.coastlines(
                resolution=ne_scale,
                linewidth=style.coast_lw,
                edgecolor=style.coast_edge,
                zorder=zorder_land + 3.0,
            )

        if draw_countries:
            ax.add_feature(
                cfeature.BORDERS.with_scale(ne_scale),
                linewidth=style.border_lw,
                edgecolor=style.borders_edge,
                zorder=zorder_land + 3.0,
            )

    # --- Gridlines ---
    gl = None
    if draw_parallels or draw_meridians:
        gl_defaults: dict[str, Any] = {
            "draw_labels": True,
            "linewidth": 0.4,
            "linestyle": "-",
            "alpha": 0.25,
            "color": style.grid_color,
        }
        if gridline_kwargs:
            gl_defaults.update(dict(gridline_kwargs))

        gl = ax.gridlines(**gl_defaults)
        gl.top_labels = False
        gl.right_labels = False
        if not draw_parallels:
            gl.left_labels = False
        if not draw_meridians:
            gl.bottom_labels = False

        gl.xlabel_style = {"size": 11, "alpha": 1.0, "color": style.grid_label_color}
        gl.ylabel_style = {"size": 11, "alpha": 1.0, "color": style.grid_label_color}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        if draw_parallels:
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
        if draw_meridians:
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))

        # Make gridlines visually uniform over land/ocean via under-stroke
        fig.canvas.draw()
        for coll in getattr(gl, "xline_artists", []) + getattr(gl, "yline_artists", []):
            coll.set_path_effects(
                [
                    pe.Stroke(linewidth=0.8, foreground="black", alpha=0.35),
                    pe.Normal(),
                ]
            )

    plt.tight_layout()
    return fig, ax, crs, gl


def add_trace(
    ax,
    lat_deg: Sequence[float],
    lon_deg: Sequence[float],
    *,
    linewidth: float = 1.5,
    marker: Optional[str] = None,
    markersize: float = 2.0,
    color: str = "C1",
    zorder: float = 10.0,
    wrap_lon: bool = True,
    dateline_threshold_deg: float = 180.0,
    glow: bool = False,
    glow_color: str = "white",
    glow_alpha: float = 0.6,
    glow_width: float = 1.0,
    label: Optional[str] = None,
):
    """
    Add a single trace (lat/lon in degrees) to an existing Cartopy GeoAxes.

    Parameters
    ----------
    ax:
        A Cartopy GeoAxes (from `make_basemap()` or otherwise).
    lat_deg, lon_deg:
        Latitude/longitude samples in degrees.
    linewidth, marker, markersize, color:
        Line/marker styling.
    zorder:
        Z-order for the trace.
    wrap_lon:
        If True, wrap longitudes to [-180, 180).
    dateline_threshold_deg:
        Split the line when |Δlon| exceeds this threshold (prevents wraparound streaks).
    glow:
        If True, adds a subtle halo using path effects.
    glow_color, glow_alpha, glow_width:
        Glow styling.
    label:
        Optional label for the line (legend).

    Returns
    -------
    artists:
        List of Line2D artists added to the axes (one per segment).
    """

    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    if lat.shape != lon.shape:
        raise ValueError(
            f"lat_deg and lon_deg must have same shape; got {lat.shape} vs {lon.shape}"
        )
    if lat.size == 0:
        return []

    if wrap_lon:
        lon = (lon + 180.0) % 360.0 - 180.0

    if lon.size < 2:
        segments = [np.arange(lon.size)]
    else:
        dlon = np.abs(np.diff(lon))
        breaks = np.where(dlon > dateline_threshold_deg)[0] + 1
        segments = [
            seg for seg in np.split(np.arange(lon.size), breaks) if seg.size > 0
        ]

    src = ccrs.PlateCarree()

    effects = None
    if glow:
        effects = [
            pe.Stroke(
                linewidth=linewidth + glow_width,
                foreground=glow_color,
                alpha=glow_alpha,
            ),
            pe.Normal(),
        ]

    artists = []
    for seg in segments:
        if seg.size < 2:
            continue
        line = ax.plot(
            lon[seg],
            lat[seg],
            linewidth=linewidth,
            transform=src,
            zorder=zorder,
            color=color,
            marker=marker,
            markersize=markersize,
            label=label,
        )[0]
        if effects is not None:
            line.set_path_effects(effects)
        artists.append(line)

    return artists
