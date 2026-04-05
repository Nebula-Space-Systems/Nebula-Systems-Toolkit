"""Shared basemap, style, and view primitives for NSTK geographic plotting."""

import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from ._cartopy_data import configure_cartopy_data_dir, get_cartopy_raster_path

# -----------------------------------------------------------------------------
# Cartopy data directory setup
# -----------------------------------------------------------------------------

configure_cartopy_data_dir()

ProjectionName = Literal[
    "PlateCarree", "Robinson", "Mollweide", "Mercator", "Orthographic"
]

# Natural Earth feature scale options
CFeatureScale = Literal["10m", "50m", "110m"]


# -----------------------------------------------------------------------------
# Config objects
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MatplotlibConfig:
    """Local rcParams applied by make_basemap()."""

    figure_dpi: int = 120
    savefig_dpi: int = 200
    font_size: int = 11
    tight_layout: bool = True
    tight_layout_pad: float = 0.4


@dataclass(frozen=True)
class ProjectionConfig:
    name: ProjectionName = "PlateCarree"
    central_longitude: float = 0.0
    central_latitude: float = 0.0


@dataclass(frozen=True)
class ExtentConfig:
    """
    Map extent control.

    If extent is provided, it takes precedence and defines the map bounds as
    (west, east, south, north) in degrees using the specified crs.

    Otherwise, if global_map=True, the map is set to show the full globe.

    Examples:
        # Full earth (default)
        ExtentConfig()

        # Zoom to specific region
        ExtentConfig(extent=(-10, 30, 35, 60))  # Europe
        ExtentConfig(extent=(-125, -66, 24, 49))  # Continental US

        # Explicit global
        ExtentConfig(global_map=True)
    """

    global_map: bool = True
    extent: Optional[Tuple[float, float, float, float]] = None
    crs: Any = field(default_factory=ccrs.PlateCarree)


@dataclass(frozen=True)
class ThemeConfig:
    """
    High-level visual defaults.

    Keep this to colors that are shared across multiple components.
    Component-specific widths/colors belong in their respective configs below.
    """

    figure_face: str = "#171717"
    axes_face: str = "#171717"
    outline_edge: str = "#616161"

    grid_color: str = "#CFCFCF"
    grid_label_color: str = "#DFDFDF"
    grid_label_fontsize: float = 11.0

    # Default trace color (used by callers if they want a theme default)
    trace_default: str = "C1"


@dataclass(frozen=True)
class RasterBackgroundConfig:
    enabled: bool = False
    path: Optional[str] = None
    alpha: float = 1.0
    zorder: float = -10.0
    interpolation: str = "bilinear"
    origin: str = "upper"


@dataclass(frozen=True)
class FillRenderConfig:
    """
    Rendering controls for filled polygon features (LAND/OCEAN/LAKES).

    Important:
    - Opacity is always 1.0 (solid fill). Not configurable by design.
    - `seam_fix=True` prevents the “white halo/seam” around polygon edges by
      forcing a zero-width edge colored the same as the face. This produces
      no visible stroke but eliminates antialias blending against the background.
    """

    enabled: bool = True
    facecolor: str = "#FFFFFF"
    zorder: float = 0.0

    seam_fix: bool = True
    antialiased: bool = True
    rasterized: bool = True


@dataclass(frozen=True)
class BumpTextureConfig:
    """
    Topo+bathy bump shading that changes brightness only (no hue shift).

    This is drawn above land/ocean fills and below linework (coastlines/borders).
    """

    enabled: bool = True
    path: Optional[str] = None  # defaults to cartopy data_dir/raster/topo_and_bathy.png
    opacity: float = 0.25  # overall intensity
    strength: float = 1.15  # contrast around mid-gray
    gamma: float = 1.0
    zorder: float = 1.25
    interpolation: str = "bilinear"
    origin: str = "upper"


@dataclass(frozen=True)
class ShadowConfig:
    """
    Drop shadow for a filled feature (applied as a path-effect, not a second polygon).

    This avoids alpha “interplay” with other layers because the fill remains solid
    and the shadow is rendered by the artist itself.
    """

    enabled: bool = True
    offset: Tuple[float, float] = (-1.2, -1.2)  # points
    color: str = "black"
    alpha: float = 0.30
    zorder_boost: float = (
        0.01  # tiny bump to keep it deterministic above same-zorder fills
    )


@dataclass(frozen=True)
class CoastlinesConfig:
    enabled: bool = True
    color: str = "#B5B5B5"
    linewidth: float = 0.2
    zorder: float = 3.0
    alpha: float = 1.0


@dataclass(frozen=True)
class BordersConfig:
    enabled: bool = False
    color: str = "#555555"
    linewidth: float = 0.4
    zorder: float = 3.0
    alpha: float = 1.0


@dataclass(frozen=True)
class OutlineConfig:
    enabled: bool = True
    linewidth: float = 0.8
    color: Optional[str] = None  # defaults to theme.outline_edge


@dataclass(frozen=True)
class GridStrokeConfig:
    enabled: bool = False
    linewidth: float = 0.8
    foreground: str = "black"
    alpha: float = 0.35


@dataclass(frozen=True)
class GridlinesConfig:
    """
    Gridline control.

    Duplicate-meridian fix:
    - Default meridians are generated in (-180, 180] so you don't get both -180 and +180.
    """

    enabled: bool = True

    draw_parallels: bool = True
    draw_meridians: bool = True
    draw_labels: bool = True

    top_labels: bool = False
    right_labels: bool = False
    bottom_labels: bool = True
    left_labels: bool = True

    linewidth: float = 0.3
    linestyle: str = "-"
    alpha: float = 0.8
    color: Optional[str] = None  # defaults to theme.grid_color
    zorder: float = 2.5

    xlabel_style: Mapping[str, Any] = field(default_factory=dict)
    ylabel_style: Mapping[str, Any] = field(default_factory=dict)
    xformatter: Any = LONGITUDE_FORMATTER
    yformatter: Any = LATITUDE_FORMATTER

    # Locators:
    xlocator: Optional[mticker.Locator] = None
    ylocator: Optional[mticker.Locator] = None
    xlocator_values: Optional[Sequence[float]] = None
    ylocator_values: Optional[Sequence[float]] = None

    meridian_step_deg: float = 60.0
    parallel_step_deg: float = 30.0

    meridian_min_deg: float = -180.0
    meridian_max_deg: float = 180.0
    parallel_min_deg: float = -90.0
    parallel_max_deg: float = 90.0

    include_dateline: bool = True
    dateline_value_deg: float = 180.0  # keep +180 by default

    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)
    stroke: GridStrokeConfig = field(default_factory=GridStrokeConfig)


@dataclass(frozen=True)
class TitleConfig:
    enabled: bool = False
    text: str = ""
    color: Optional[str] = None  # defaults to theme.grid_label_color
    pad: float = 10.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MapConfig:
    """
    Single configuration object controlling *all* basemap appearance.
    """

    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    extent: ExtentConfig = field(default_factory=ExtentConfig)
    figsize: Tuple[float, float] = (12.0, 6.0)

    # Optional Natural Earth feature scale for cfeatures
    # (LAND/OCEAN/LAKES/BORDERS/COASTLINE)
    # If None, use default behavior. Otherwise one of: "10m", "50m", "110m".
    cfeature_scale: Optional[CFeatureScale] = None

    mpl: MatplotlibConfig = field(default_factory=MatplotlibConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)

    raster_background: RasterBackgroundConfig = field(
        default_factory=RasterBackgroundConfig
    )

    ocean: FillRenderConfig = field(
        default_factory=lambda: FillRenderConfig(
            enabled=True, facecolor="#FFFFFF", zorder=0.0
        )
    )
    land: FillRenderConfig = field(
        default_factory=lambda: FillRenderConfig(
            enabled=True, facecolor="#E6E6E6", zorder=1.0
        )
    )

    # Shadow is applied to the LAND artist only (path effect), so it never changes fill alpha.
    land_shadow: ShadowConfig = field(default_factory=ShadowConfig)

    bump_texture: BumpTextureConfig = field(default_factory=BumpTextureConfig)

    coastlines: CoastlinesConfig = field(default_factory=CoastlinesConfig)
    borders: BordersConfig = field(default_factory=BordersConfig)
    outline: OutlineConfig = field(default_factory=OutlineConfig)

    gridlines: GridlinesConfig = field(default_factory=GridlinesConfig)
    title: TitleConfig = field(default_factory=TitleConfig)


# -----------------------------------------------------------------------------
# User-Facing Styles
# -----------------------------------------------------------------------------

def _replace_section(current: Any, value: Any) -> Any:
    if isinstance(value, Mapping):
        return replace(current, **dict(value))
    return value


@dataclass(frozen=True)
class MapView:
    """Reusable view/layout controls kept separate from visual style.

    `MapView` collects the non-visual map decisions that users often want to
    reuse across figures: projection, extent, figure size, Cartopy feature
    scale, title, and local Matplotlib defaults.

    Typical usage:

    ```python
    from nstk.plotting import GeoMap, MapView

    conus = (
        MapView()
        .with_projection(name="Mercator")
        .with_extent((-125.0, -66.0, 24.0, 50.0), global_map=False)
    )
    m = GeoMap(style="light_detailed", view=conus)
    ```
    """

    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    extent: ExtentConfig = field(default_factory=ExtentConfig)
    figsize: Tuple[float, float] = (12.0, 6.0)
    cfeature_scale: Optional[CFeatureScale] = None
    mpl: MatplotlibConfig = field(default_factory=MatplotlibConfig)
    title: TitleConfig = field(default_factory=TitleConfig)

    def update(self, **changes: Any) -> "MapView":
        aliases = {"size": "figsize"}
        nested = {"projection", "extent", "mpl", "title"}
        resolved: dict[str, Any] = {}
        for key, value in changes.items():
            name = aliases.get(key, key)
            if not hasattr(self, name):
                raise KeyError(f"Unknown MapView section {key!r}")
            current = getattr(self, name)
            resolved[name] = _replace_section(current, value) if name in nested else value
        return replace(self, **resolved)

    def with_projection(self, **changes: Any) -> "MapView":
        return self.update(projection=changes)

    def with_extent(
        self,
        extent: Tuple[float, float, float, float] | None = None,
        *,
        global_map: bool | None = None,
        crs: Any | None = None,
    ) -> "MapView":
        updates: dict[str, Any] = {}
        if extent is not None:
            updates["extent"] = tuple(float(v) for v in extent)
        if global_map is not None:
            updates["global_map"] = bool(global_map)
        if crs is not None:
            updates["crs"] = crs
        return self.update(extent=updates)

    def with_title(self, text: str, **changes: Any) -> "MapView":
        return self.update(title={"enabled": True, "text": text, **changes})

    def to_map_config(self, *, style: "MapStyle | str | None" = None) -> "MapConfig":
        return compile_map_config(style=style, view=self)


@dataclass(frozen=True)
class MapStyle:
    """User-facing visual style definition for 2D maps.

    `MapStyle` is the main customization surface for NSTK basemaps. It keeps
    the exact visual presets used internally, but exposes them in a form that
    is much easier to inspect, copy, and modify than hand-editing `MapConfig`.

    Start from a preset with `get_map_style(...)`, then tweak only the pieces
    you care about:

    ```python
    from nstk.plotting import GeoMap, get_map_style

    paper = (
        get_map_style("light_detailed")
        .with_land(facecolor="#d8d0c4")
        .with_grid(alpha=0.12, draw_labels=False)
        .with_borders(enabled=False)
    )
    m = GeoMap(style=paper, extent="auto")
    ```
    """

    name: str | None = None
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    raster_background: RasterBackgroundConfig = field(default_factory=RasterBackgroundConfig)
    ocean: FillRenderConfig = field(
        default_factory=lambda: FillRenderConfig(enabled=True, facecolor="#FFFFFF", zorder=0.0)
    )
    land: FillRenderConfig = field(
        default_factory=lambda: FillRenderConfig(enabled=True, facecolor="#E6E6E6", zorder=1.0)
    )
    land_shadow: ShadowConfig = field(default_factory=ShadowConfig)
    bump_texture: BumpTextureConfig = field(default_factory=BumpTextureConfig)
    coastlines: CoastlinesConfig = field(default_factory=CoastlinesConfig)
    borders: BordersConfig = field(default_factory=BordersConfig)
    outline: OutlineConfig = field(default_factory=OutlineConfig)
    gridlines: GridlinesConfig = field(default_factory=GridlinesConfig)

    def update(self, **sections: Any) -> "MapStyle":
        aliases = {
            "palette": "theme",
            "background": "raster_background",
            "grid": "gridlines",
            "frame": "outline",
            "shadow": "land_shadow",
            "bump": "bump_texture",
        }
        nested = {
            "theme",
            "raster_background",
            "ocean",
            "land",
            "land_shadow",
            "bump_texture",
            "coastlines",
            "borders",
            "outline",
            "gridlines",
        }
        resolved: dict[str, Any] = {}
        for key, value in sections.items():
            name = aliases.get(key, key)
            if not hasattr(self, name):
                raise KeyError(f"Unknown MapStyle section {key!r}")
            current = getattr(self, name)
            resolved[name] = _replace_section(current, value) if name in nested else value
        return replace(self, **resolved)

    def with_palette(self, **changes: Any) -> "MapStyle":
        return self.update(theme=changes)

    def with_background(self, **changes: Any) -> "MapStyle":
        return self.update(raster_background=changes)

    def with_ocean(self, **changes: Any) -> "MapStyle":
        return self.update(ocean=changes)

    def with_land(self, **changes: Any) -> "MapStyle":
        return self.update(land=changes)

    def with_surfaces(
        self,
        *,
        land: Mapping[str, Any] | FillRenderConfig | None = None,
        ocean: Mapping[str, Any] | FillRenderConfig | None = None,
        background: Mapping[str, Any] | RasterBackgroundConfig | None = None,
    ) -> "MapStyle":
        updates: dict[str, Any] = {}
        if land is not None:
            updates["land"] = land
        if ocean is not None:
            updates["ocean"] = ocean
        if background is not None:
            updates["raster_background"] = background
        return self.update(**updates)

    def with_coastlines(self, **changes: Any) -> "MapStyle":
        return self.update(coastlines=changes)

    def with_borders(self, **changes: Any) -> "MapStyle":
        return self.update(borders=changes)

    def with_frame(self, **changes: Any) -> "MapStyle":
        return self.update(outline=changes)

    def with_grid(self, **changes: Any) -> "MapStyle":
        return self.update(gridlines=changes)

    def with_lines(
        self,
        *,
        coastlines: Mapping[str, Any] | CoastlinesConfig | None = None,
        borders: Mapping[str, Any] | BordersConfig | None = None,
        frame: Mapping[str, Any] | OutlineConfig | None = None,
    ) -> "MapStyle":
        updates: dict[str, Any] = {}
        if coastlines is not None:
            updates["coastlines"] = coastlines
        if borders is not None:
            updates["borders"] = borders
        if frame is not None:
            updates["outline"] = frame
        return self.update(**updates)

    def with_effects(
        self,
        *,
        land_shadow: Mapping[str, Any] | ShadowConfig | None = None,
        bump_texture: Mapping[str, Any] | BumpTextureConfig | None = None,
    ) -> "MapStyle":
        updates: dict[str, Any] = {}
        if land_shadow is not None:
            updates["land_shadow"] = land_shadow
        if bump_texture is not None:
            updates["bump_texture"] = bump_texture
        return self.update(**updates)

    def renamed(self, name: str | None) -> "MapStyle":
        """Return a copy with a new display/registry name."""
        return replace(self, name=None if name is None else str(name))

    def register(self, name: str | None = None, *, overwrite: bool = False) -> "MapStyle":
        """Register this style under `name` and return the registered copy."""
        style_name = self.name if name is None else name
        if style_name is None:
            raise ValueError("A style name is required before registering a MapStyle")
        return register_map_style(str(style_name), self, overwrite=overwrite)

    def to_map_config(self, *, view: MapView | None = None) -> "MapConfig":
        return compile_map_config(style=self, view=view)


def _normalize_style_key(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


_STYLE_PRESETS: dict[str, MapStyle] = {}


def register_map_style(name: str, style: MapStyle, *, overwrite: bool = False) -> MapStyle:
    """Register a named style preset for later use by string name."""
    key = _normalize_style_key(name)
    if not overwrite and key in _STYLE_PRESETS:
        raise ValueError(f"Map style {name!r} is already registered")
    style_obj = style if style.name == name else replace(style, name=str(name))
    _STYLE_PRESETS[key] = style_obj
    return style_obj


def get_map_style(style: str | MapStyle | None = None) -> MapStyle:
    """Return a registered `MapStyle` or pass through an existing style object."""
    if isinstance(style, MapStyle):
        return style
    if style is None:
        style = "light_detailed"
    key = _normalize_style_key(style)
    if key not in _STYLE_PRESETS:
        available = ", ".join(sorted(_STYLE_PRESETS))
        raise ValueError(f"Unknown map style {style!r}. Available: {available}")
    return _STYLE_PRESETS[key]


def available_map_styles() -> tuple[str, ...]:
    """List the currently registered public style names."""
    return tuple(sorted(style.name or key for key, style in _STYLE_PRESETS.items()))


def compile_map_config(
    *,
    style: str | MapStyle | None = None,
    view: MapView | None = None,
) -> MapConfig:
    """Compile a user-facing `MapStyle` and `MapView` into a renderer config."""
    style_obj = get_map_style(style)
    view_obj = MapView() if view is None else view
    return MapConfig(
        projection=view_obj.projection,
        extent=view_obj.extent,
        figsize=view_obj.figsize,
        cfeature_scale=view_obj.cfeature_scale,
        mpl=view_obj.mpl,
        theme=style_obj.theme,
        raster_background=style_obj.raster_background,
        ocean=style_obj.ocean,
        land=style_obj.land,
        land_shadow=style_obj.land_shadow,
        bump_texture=style_obj.bump_texture,
        coastlines=style_obj.coastlines,
        borders=style_obj.borders,
        outline=style_obj.outline,
        gridlines=style_obj.gridlines,
        title=view_obj.title,
    )


_DARK_THEME = ThemeConfig(
    figure_face="#171717",
    axes_face="#171717",
    outline_edge="#616161",
    grid_color="#CFCFCF",
    grid_label_color="#DFDFDF",
    trace_default="C1",
)

_LIGHT_THEME = ThemeConfig(
    figure_face="#FFFFFF",
    axes_face="#FFFFFF",
    outline_edge="#FFFFFF",
    grid_color="#424242",
    grid_label_color="#222222",
    trace_default="C0",
)

DARK_DETAILED_STYLE = MapStyle(
    name="dark_detailed",
    theme=_DARK_THEME,
    ocean=FillRenderConfig(
        enabled=True,
        facecolor="#1E1E1E",
        zorder=0.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    land=FillRenderConfig(
        enabled=True,
        facecolor="#9A9A9A",
        zorder=1.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    coastlines=CoastlinesConfig(enabled=True, color="#0F0F0F", linewidth=0.4, zorder=3.0),
    borders=BordersConfig(enabled=True, color="#0F0F0F", linewidth=0.4, zorder=3.0),
    gridlines=GridlinesConfig(alpha=0.35),
    bump_texture=BumpTextureConfig(opacity=0.32),
    land_shadow=ShadowConfig(enabled=True, offset=(-0.8, -0.8), color="black", alpha=0.7),
)

DARK_DETAILED_NO_GRID_STYLE = DARK_DETAILED_STYLE.with_grid(alpha=0.0).update(name="dark_detailed_no_grid")
DARK_STYLE = (
    DARK_DETAILED_STYLE.with_effects(
        bump_texture={"enabled": False},
        # land_shadow={"enabled": False},
    )
    .update(name="dark")
)
DARK_NO_GRID_STYLE = DARK_STYLE.with_grid(alpha=0.0).update(name="dark_no_grid")

LIGHT_DETAILED_STYLE = MapStyle(
    name="light_detailed",
    theme=_LIGHT_THEME,
    ocean=FillRenderConfig(
        enabled=True,
        facecolor="#FFFFFF",
        zorder=0.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    land=FillRenderConfig(
        enabled=True,
        facecolor="#C2C2C2",
        zorder=1.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    coastlines=CoastlinesConfig(enabled=True, color="#B5B5B5", linewidth=0.2, zorder=3.0, alpha=0.3),
    borders=BordersConfig(enabled=True, color="#FFFFFF", linewidth=0.4, zorder=3.0),
    gridlines=GridlinesConfig(color="#222222", alpha=0.5),
    bump_texture=BumpTextureConfig(opacity=0.15),
    land_shadow=ShadowConfig(enabled=True, offset=(-0.8, -0.8), color="black", alpha=0.18),
)
LIGHT_DETAILED_NO_GRID_STYLE = LIGHT_DETAILED_STYLE.with_grid(alpha=0.0).update(name="light_detailed_no_grid")
LIGHT_STYLE = (
    LIGHT_DETAILED_STYLE.with_effects(bump_texture={"enabled": False}, land_shadow={"enabled": True})
    .update(name="light")
)
LIGHT_NO_GRID_STYLE = (
    LIGHT_STYLE.with_grid(alpha=0.0)
    .update(name="light_no_grid")
)

DARK_RASTER_STYLE = (
    DARK_DETAILED_STYLE.with_background(enabled=True)
    .with_effects(bump_texture={"enabled": False})
    .with_coastlines(enabled=False)
    .with_grid(color="#424242", alpha=0.7, linewidth=0.5)
    .with_borders(enabled=False)
    .update(name="dark_raster")
)
DARK_RASTER_NO_GRID_STYLE = DARK_RASTER_STYLE.with_grid(alpha=0.0).update(name="dark_raster_no_grid")

LIGHT_RASTER_STYLE = (
    LIGHT_DETAILED_STYLE.with_background(enabled=True)
    .with_effects(bump_texture={"enabled": False})
    .with_coastlines(enabled=False)
    .with_grid(color="#EEEEEE", alpha=0.85, linewidth=0.5)
    .with_borders(enabled=False)
    .update(name="light_raster")
)
LIGHT_RASTER_NO_GRID_STYLE = LIGHT_RASTER_STYLE.with_grid(alpha=0.0).update(name="light_raster_no_grid")

for _style in (
    DARK_DETAILED_STYLE,
    DARK_DETAILED_NO_GRID_STYLE,
    DARK_STYLE,
    DARK_NO_GRID_STYLE,
    LIGHT_DETAILED_STYLE,
    LIGHT_DETAILED_NO_GRID_STYLE,
    LIGHT_STYLE,
    LIGHT_NO_GRID_STYLE,
    DARK_RASTER_STYLE,
    DARK_RASTER_NO_GRID_STYLE,
    LIGHT_RASTER_STYLE,
    LIGHT_RASTER_NO_GRID_STYLE,
):
    register_map_style(_style.name or "default", _style, overwrite=True)

DARK_DETAILED = compile_map_config(style=DARK_DETAILED_STYLE)
DARK_DETAILED_NO_GRID = compile_map_config(style=DARK_DETAILED_NO_GRID_STYLE)
DARK = compile_map_config(style=DARK_STYLE)
DARK_NO_GRID = compile_map_config(style=DARK_NO_GRID_STYLE)
LIGHT_DETAILED = compile_map_config(style=LIGHT_DETAILED_STYLE)
LIGHT_DETAILED_NO_GRID = compile_map_config(style=LIGHT_DETAILED_NO_GRID_STYLE)
LIGHT = compile_map_config(style=LIGHT_STYLE)
LIGHT_NO_GRID = compile_map_config(style=LIGHT_NO_GRID_STYLE)
DARK_RASTER = compile_map_config(style=DARK_RASTER_STYLE)
DARK_RASTER_NO_GRID = compile_map_config(style=DARK_RASTER_NO_GRID_STYLE)
LIGHT_RASTER = compile_map_config(style=LIGHT_RASTER_STYLE)
LIGHT_RASTER_NO_GRID = compile_map_config(style=LIGHT_RASTER_NO_GRID_STYLE)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _projection_from_config(cfg: ProjectionConfig) -> ccrs.Projection:
    if cfg.name == "PlateCarree":
        return ccrs.PlateCarree(central_longitude=cfg.central_longitude)
    if cfg.name == "Robinson":
        return ccrs.Robinson(central_longitude=int(cfg.central_longitude))
    if cfg.name == "Mollweide":
        return ccrs.Mollweide(central_longitude=int(cfg.central_longitude))
    if cfg.name == "Mercator":
        return ccrs.Mercator(central_longitude=int(cfg.central_longitude))
    if cfg.name == "Orthographic":
        return ccrs.Orthographic(
            central_longitude=float(cfg.central_longitude),
            central_latitude=float(cfg.central_latitude),
        )
    raise ValueError(f"Unknown projection: {cfg.name!r}")


def _default_bump_path() -> str:
    return str(get_cartopy_raster_path("topo_and_bathy.png"))


def _default_raster_path() -> str:
    return str(get_cartopy_raster_path("shadedrelief.jpg"))


def _load_luminance(path: str) -> np.ndarray:
    img = mpimg.imread(path)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.astype(np.float32)
    lum = np.asarray(lum, dtype=np.float32)
    if lum.max() > 1.0:
        lum = lum / 255.0
    return np.clip(lum, 0.0, 1.0)


def _add_bump_shading(
    ax,
    *,
    path: str,
    opacity: float,
    strength: float,
    gamma: float,
    zorder: float,
    interpolation: str,
    origin: str,
) -> None:
    """
    Brightness-only bump shading:
      - white overlay for highlights
      - black overlay for shadows
    Keeps hue intact (no gray tinting from semi-transparent grayscale rasters).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Texture file not found: {path!r}. Set MapConfig.bump_texture.path."
        )

    lum = _load_luminance(path)
    if gamma != 1.0:
        lum = np.clip(lum, 0.0, 1.0) ** gamma

    d = lum - 0.5
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
        origin=origin,
        interpolation=interpolation,
        zorder=zorder,
    )
    try:
        ax.imshow(img_hi, **common)
        ax.imshow(img_lo, **common)
    except ImportError:
        # Cartopy image reprojection for non-native projections depends on
        # optional spatial-index packages (SciPy or pykdtree). If they are not
        # available, keep the map usable and just skip the bump texture layer.
        return


def _apply_fill_rendering(artist, cfg: FillRenderConfig) -> None:
    """
    Apply seam/antialias/rasterize controls consistently.

    Seam fix strategy (while still "no visible stroke"):
    - edgecolor == facecolor
    - linewidth == 0
    This avoids the ocean/background bleeding through polygon edges due to antialiasing.
    """
    try:
        artist.set_facecolor(cfg.facecolor)
    except Exception:
        pass

    try:
        if cfg.seam_fix:
            artist.set_edgecolor(cfg.facecolor)
            artist.set_linewidth(0.0)
        else:
            artist.set_edgecolor("none")
            artist.set_linewidth(0.0)
    except Exception:
        pass

    try:
        artist.set_antialiased(bool(cfg.antialiased))
    except Exception:
        pass

    try:
        artist.set_rasterized(bool(cfg.rasterized))
    except Exception:
        pass


def _add_lakes_fill(ax, cfg: MapConfig, *, zorder: float) -> None:
    """Draw lakes using the ocean fill styling so inland water stays visible."""
    if not cfg.ocean.enabled:
        return

    feat_lakes = (
        cfeature.LAKES
        if cfg.cfeature_scale is None
        else cfeature.LAKES.with_scale(cfg.cfeature_scale)
    )
    lakes_artist = ax.add_feature(  # type: ignore
        feat_lakes,
        facecolor=cfg.ocean.facecolor,
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        zorder=zorder,
    )
    _apply_fill_rendering(lakes_artist, cfg.ocean)


def _safe_gridlines(ax, kwargs: dict) -> Any:
    """
    Cartopy versions differ slightly; drop unsupported kwargs instead of failing.
    """
    try:
        return ax.gridlines(**kwargs)
    except TypeError:
        for k in ("x_inline", "y_inline", "rotate_labels", "dms"):
            if k in kwargs:
                kwargs.pop(k, None)
                try:
                    return ax.gridlines(**kwargs)
                except TypeError:
                    continue
        minimal = {
            k: kwargs[k]
            for k in ("draw_labels", "linewidth", "linestyle", "alpha", "color")
            if k in kwargs
        }
        return ax.gridlines(**minimal)


def _generate_meridians(cfg: GridlinesConfig) -> np.ndarray:
    """
    Default meridians in (-180, 180] to avoid -180/+180 duplicate line.
    """
    step = float(cfg.meridian_step_deg)
    mn = float(cfg.meridian_min_deg)
    mx = float(cfg.meridian_max_deg)
    if step <= 0:
        return np.array([], dtype=float)

    vals = np.arange(mn, mx + 1e-9, step, dtype=float)

    # Wrap to (-180, 180] with preference for +180
    def wrap180(x: float) -> float:
        y = (x + 180.0) % 360.0 - 180.0
        if abs(y + 180.0) < 1e-9 and x > 0:
            return 180.0
        return y

    wrapped = np.array([wrap180(v) for v in vals], dtype=float)

    # Remove dateline duplicates; keep only the configured one (default +180)
    out: list[float] = []
    for v in wrapped:
        out.append(float(v))

    return np.array(out, dtype=float)


def _generate_parallels(cfg: GridlinesConfig) -> np.ndarray:
    step = float(cfg.parallel_step_deg)
    mn = float(cfg.parallel_min_deg)
    mx = float(cfg.parallel_max_deg)
    if step <= 0:
        return np.array([], dtype=float)

    vals = np.arange(mn, mx + 1e-9, step, dtype=float)
    vals = np.clip(vals, -90.0, 90.0)

    out: list[float] = []
    seen = set()
    for v in vals:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return np.array(out, dtype=float)


# -----------------------------------------------------------------------------
# Basemap
# -----------------------------------------------------------------------------


def _configure_basemap_axes(fig: Any, ax: Any, cfg: MapConfig) -> tuple[Any, Any]:
    """
    Apply MapConfig-driven basemap styling and features onto an existing GeoAxes.

    Returns the axes projection and the gridliner artist (if any).
    """
    t = cfg.theme

    # Matplotlib rcParams (local)
    plt.rcParams.update(
        {
            "figure.dpi": cfg.mpl.figure_dpi,
            "savefig.dpi": cfg.mpl.savefig_dpi,
            "font.size": cfg.mpl.font_size,
            "figure.facecolor": t.figure_face,
            "figure.edgecolor": t.figure_face,
            "axes.facecolor": t.axes_face,
            "axes.edgecolor": t.outline_edge,
            "text.color": t.grid_label_color,
            "xtick.color": t.grid_label_color,
            "ytick.color": t.grid_label_color,
        }
    )

    crs = getattr(ax, "projection", None)
    if crs is None:
        crs = _projection_from_config(cfg.projection)

    fig.set_facecolor(t.figure_face)
    ax.set_facecolor(t.axes_face)

    # Extent/global (extent takes precedence if provided)
    if cfg.extent.extent is not None:
        # Explicit extent provided - use it to zoom into specific region
        ax.set_extent(cfg.extent.extent, crs=cfg.extent.crs)  # type: ignore
    elif cfg.extent.global_map:
        # No extent specified - show full globe
        ax.set_global()  # type: ignore
    else:
        raise ValueError("ExtentConfig: either set global_map=True or provide extent")

    # Outline/frame
    if hasattr(ax, "outline_patch") and ax.outline_patch is not None:  # type: ignore
        if cfg.outline.enabled:
            ax.outline_patch.set_visible(True)  # type: ignore
            ax.outline_patch.set_linewidth(cfg.outline.linewidth)  # type: ignore
            ax.outline_patch.set_edgecolor(cfg.outline.color or t.outline_edge)  # type: ignore
        else:
            ax.outline_patch.set_visible(False)  # type: ignore

    # Background: raster OR vector fills (+ bump)
    land_artist = None
    lakes_zorder = max(cfg.ocean.zorder, cfg.land.zorder)

    if cfg.raster_background.enabled:
        path = cfg.raster_background.path or _default_raster_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Raster background not found: {path!r}. Set MapConfig.raster_background.path."
            )
        img = mpimg.imread(path)
        try:
            ax.imshow(
                img,
                extent=[-180.0, 180.0, -90.0, 90.0],  # type: ignore
                transform=ccrs.PlateCarree(),
                origin=cfg.raster_background.origin,  # type: ignore
                interpolation=cfg.raster_background.interpolation,
                zorder=cfg.raster_background.zorder,
                alpha=cfg.raster_background.alpha,
            )
        except ImportError:
            pass
    else:
        # OCEAN (solid, alpha=1.0)
        if cfg.ocean.enabled:
            feat_ocean = (
                cfeature.OCEAN
                if cfg.cfeature_scale is None
                else cfeature.OCEAN.with_scale(cfg.cfeature_scale)
            )
            ocean_artist = ax.add_feature(  # type: ignore
                feat_ocean,
                facecolor=cfg.ocean.facecolor,
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=cfg.ocean.zorder,
            )
            _apply_fill_rendering(ocean_artist, cfg.ocean)

        # LAND (solid, alpha=1.0)
        if cfg.land.enabled:
            feat_land = (
                cfeature.LAND
                if cfg.cfeature_scale is None
                else cfeature.LAND.with_scale(cfg.cfeature_scale)
            )
            land_artist = ax.add_feature(  # type: ignore
                feat_land,
                facecolor=cfg.land.facecolor,
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=cfg.land.zorder,
            )
            _apply_fill_rendering(land_artist, cfg.land)

            # Shadow as path effect (no extra polygon layer)
            if cfg.land_shadow.enabled:
                try:
                    land_artist.set_zorder(
                        cfg.land.zorder + cfg.land_shadow.zorder_boost
                    )
                    land_artist.set_path_effects(
                        [
                            pe.SimplePatchShadow(
                                offset=cfg.land_shadow.offset,
                                alpha=cfg.land_shadow.alpha,
                                shadow_rgbFace=cfg.land_shadow.color,
                            ),
                            pe.Normal(),
                        ]
                    )
                except Exception:
                    # If a backend/cartopy version can't apply path effects to this artist, ignore silently.
                    pass

        # Bump texture (brightness-only)
        if cfg.bump_texture.enabled:
            bump_path = cfg.bump_texture.path or _default_bump_path()
            _add_bump_shading(
                ax,
                path=bump_path,
                opacity=cfg.bump_texture.opacity,
                strength=cfg.bump_texture.strength,
                gamma=cfg.bump_texture.gamma,
                zorder=cfg.bump_texture.zorder,
                interpolation=cfg.bump_texture.interpolation,
                origin=cfg.bump_texture.origin,
            )
            lakes_zorder = max(lakes_zorder, cfg.bump_texture.zorder)

    _add_lakes_fill(ax, cfg, zorder=lakes_zorder + 0.01)

    # Coastlines / borders
    if cfg.coastlines.enabled:
        if cfg.cfeature_scale is None:
            ax.coastlines(  # type: ignore
                resolution="auto",
                linewidth=cfg.coastlines.linewidth,
                edgecolor=cfg.coastlines.color,
                zorder=cfg.coastlines.zorder,
                alpha=cfg.coastlines.alpha,
            )
        else:
            feat_coast = cfeature.COASTLINE.with_scale(cfg.cfeature_scale)
            ax.add_feature(  # type: ignore
                feat_coast,
                linewidth=cfg.coastlines.linewidth,
                edgecolor=cfg.coastlines.color,
                zorder=cfg.coastlines.zorder,
                alpha=cfg.coastlines.alpha,
            )

    if cfg.borders.enabled:
        feat_borders = (
            cfeature.BORDERS
            if cfg.cfeature_scale is None
            else cfeature.BORDERS.with_scale(cfg.cfeature_scale)
        )
        ax.add_feature(  # type: ignore
            feat_borders,
            linewidth=cfg.borders.linewidth,
            edgecolor=cfg.borders.color,
            zorder=cfg.borders.zorder,
            alpha=cfg.borders.alpha,
        )

    # Gridlines
    gl = None
    gcfg = cfg.gridlines
    if gcfg.enabled and (gcfg.draw_parallels or gcfg.draw_meridians):
        grid_color = gcfg.color or t.grid_color
        gl_kwargs: dict[str, Any] = {
            "draw_labels": gcfg.draw_labels,
            "linewidth": gcfg.linewidth,
            "linestyle": gcfg.linestyle,
            "alpha": gcfg.alpha,
            "color": grid_color,
            "zorder": gcfg.zorder,
        }
        gl_kwargs.update(dict(gcfg.extra_kwargs))

        gl = _safe_gridlines(ax, gl_kwargs)

        gl.top_labels = bool(gcfg.top_labels)
        gl.right_labels = bool(gcfg.right_labels)
        gl.bottom_labels = bool(gcfg.bottom_labels) if gcfg.draw_meridians else False
        gl.left_labels = bool(gcfg.left_labels) if gcfg.draw_parallels else False

        gl.xformatter = gcfg.xformatter
        gl.yformatter = gcfg.yformatter

        # Label styles
        gl.xlabel_style = (
            dict(gcfg.xlabel_style)
            if gcfg.xlabel_style
            else {
                "size": t.grid_label_fontsize,
                "alpha": 1.0,
                "color": t.grid_label_color,
            }
        )
        gl.ylabel_style = (
            dict(gcfg.ylabel_style)
            if gcfg.ylabel_style
            else {
                "size": t.grid_label_fontsize,
                "alpha": 1.0,
                "color": t.grid_label_color,
            }
        )

        # Locators
        if gcfg.draw_parallels:
            if gcfg.ylocator is not None:
                gl.ylocator = gcfg.ylocator
            else:
                vals = np.asarray(
                    (
                        gcfg.ylocator_values
                        if gcfg.ylocator_values is not None
                        else _generate_parallels(gcfg)
                    ),
                    dtype=float,
                )
                gl.ylocator = mticker.FixedLocator(vals)  # type: ignore

        if gcfg.draw_meridians:
            if gcfg.xlocator is not None:
                gl.xlocator = gcfg.xlocator
            else:
                vals = np.asarray(
                    (
                        gcfg.xlocator_values
                        if gcfg.xlocator_values is not None
                        else _generate_meridians(gcfg)
                    ),
                    dtype=float,
                )
                gl.xlocator = mticker.FixedLocator(vals)  # type: ignore

        # Under-stroke (optional)
        if gcfg.stroke.enabled:
            fig.canvas.draw()
            effects = [
                pe.Stroke(
                    linewidth=gcfg.stroke.linewidth,
                    foreground=gcfg.stroke.foreground,
                    alpha=gcfg.stroke.alpha,
                ),
                pe.Normal(),
            ]
            for coll in getattr(gl, "xline_artists", []) + getattr(
                gl, "yline_artists", []
            ):
                coll.set_path_effects(effects)

    # Title
    if cfg.title.enabled:
        ax.set_title(
            cfg.title.text,
            pad=cfg.title.pad,
            color=cfg.title.color or t.grid_label_color,
            **dict(cfg.title.kwargs),
        )

    if cfg.mpl.tight_layout:
        plt.tight_layout(pad=cfg.mpl.tight_layout_pad)

    return crs, gl


def make_basemap(cfg: MapConfig = DARK) -> tuple:
    """
    Create a Cartopy basemap (figure + GeoAxes) driven entirely by MapConfig.

    Guarantees:
    - LAND and OCEAN fills are always alpha=1.0 (solid).
    - Shadow (if enabled) is applied as a path-effect to the LAND artist, not as a second polygon.
    - Seam fix is handled via rendering controls (no visible strokes).
    """
    crs = _projection_from_config(cfg.projection)

    fig = plt.figure(figsize=cfg.figsize, facecolor=cfg.theme.figure_face)
    ax = plt.axes(projection=crs)
    crs, gl = _configure_basemap_axes(fig, ax, cfg)
    return fig, ax, crs, gl


# -----------------------------------------------------------------------------
# Trace drawing
# -----------------------------------------------------------------------------


def add_geodesic_trace(
    ax,
    lat_deg,
    lon_deg,
    *,
    # primary styling (common)
    linewidth: float = 1.5,
    color: str | None = "C1",
    linestyle: str = "-",
    marker: str | None = None,
    markersize: float = 2.0,
    zorder: float = 10.0,
    label: str | None = None,
    # behavior
    src_crs=None,
    densify: bool = False,
    densify_step_deg: float = 0.25,
    # wrapping / basemap-like shifting
    split_on_wrap: bool = True,
    wrap_center_deg: float | None = None,  # if None, inferred from ax.projection
    shift_to_extent: bool = True,  # basemap-like: shift data into current view lon window
    # splitting controls
    split_max_jump_frac: float = 0.12,  # fraction of projected map size (smaller than your 0.25)
    split_max_jump_world: float | None = None,
    # NaN handling
    allow_nan: bool = True,
    # latitude safety near singular projections (mercator, etc.)
    clip_lat_deg: float | None = None,  # if None, picks a safe default
    # effects
    glow: bool = False,
    glow_color: str = "white",
    glow_alpha: float = 0.6,
    glow_width: float = 1.0,
    # passthrough for uncommon kwargs (alpha, solid_capstyle, etc.)
    plot_kwargs=None,
):
    """
    Plot a trace on a Cartopy GeoAxes with Basemap-like robustness.

    Strategy:
      1) Split on NaNs (optional).
      2) For each run, unwrap longitudes as a *sequence* (prevents fake 350° jumps).
      3) Optionally shift the entire run into the axes' current lon extent
         (Basemap shiftdata behavior for cylindrical/pseudo-cylindrical use-cases).
      4) Project points (Basemap m(lon,lat) style).
      5) Split in projected space on large jumps and plot each segment.

    Returns a list of Line2D artists (one per rendered segment).
    """
    import numpy as np
    import matplotlib.patheffects as pe
    import cartopy.crs as ccrs

    if src_crs is None:
        src_crs = ccrs.PlateCarree()

    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()

    if lat.shape != lon.shape:
        raise ValueError(
            f"lat_deg and lon_deg must have same shape; got {lat.shape} vs {lon.shape}"
        )
    if lat.size == 0:
        return []

    finite = np.isfinite(lat) & np.isfinite(lon)
    if not allow_nan and not np.all(finite):
        raise ValueError("Input contains NaN/Inf but allow_nan=False")
    if not np.any(finite):
        return []

    # Split into contiguous finite runs
    idx = np.arange(lat.size)
    breaks = np.where(~finite)[0]
    runs = []
    start = 0
    for b in breaks:
        if b > start:
            runs.append(idx[start:b])
        start = b + 1
    if start < lat.size:
        runs.append(idx[start:])

    # Pick wrap center from the axes projection if not specified.
    if wrap_center_deg is None:
        wrap_center_deg = float(getattr(ax.projection, "central_longitude", 0.0))

    # Latitude clipping:
    # - Mercator-like projections blow up at the poles.
    # - Basemap silently clips; do the same by default.
    if clip_lat_deg is None:
        # Safe default. (If you want tighter for Mercator, pass clip_lat_deg=85 or similar.)
        clip_lat_deg = 89.999999

    def densify_run(
        lat_run: np.ndarray, lon_run: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if lat_run.size < 2:
            return lat_run, lon_run
        out_lat = [float(lat_run[0])]
        out_lon = [float(lon_run[0])]
        step = max(float(densify_step_deg), 1e-9)
        for i in range(1, lat_run.size):
            la0, lo0 = float(lat_run[i - 1]), float(lon_run[i - 1])
            la1, lo1 = float(lat_run[i]), float(lon_run[i])
            d = max(abs(la1 - la0), abs(lo1 - lo0))
            n = max(1, int(np.ceil(d / step)))
            if n == 1:
                out_lat.append(la1)
                out_lon.append(lo1)
            else:
                t = np.linspace(0.0, 1.0, n + 1)[1:]
                out_lat.extend((la0 + (la1 - la0) * t).tolist())
                out_lon.extend((lo0 + (lo1 - lo0) * t).tolist())
        return np.asarray(out_lat), np.asarray(out_lon)

    def unwrap_longitudes(lon_seq: np.ndarray, center: float) -> np.ndarray:
        # unwrap around `center` so point-to-point jumps are minimized
        rel = lon_seq - center
        rel_unw = np.rad2deg(np.unwrap(np.deg2rad(rel)))
        return rel_unw + center

    def extent_lon_window():
        # Get current axes extent in PlateCarree lon/lat.
        # This is how we emulate Basemap's "shiftdata into map window".
        try:
            lon_min, lon_max, _, _ = ax.get_extent(crs=ccrs.PlateCarree())
            lon_min = float(lon_min)
            lon_max = float(lon_max)
            # If the extent crosses the seam, Cartopy can return lon_max < lon_min.
            # Interpret it in a continuous domain.
            if lon_max < lon_min:
                lon_max += 360.0
            return lon_min, lon_max
        except Exception:
            return None

    # Use projection native limits if available (more stable than get_xlim early in lifecycle).
    def projected_span():
        try:
            x0, x1 = ax.projection.x_limits
            y0, y1 = ax.projection.y_limits
            w = float(abs(x1 - x0))
            h = float(abs(y1 - y0))
            if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return float(max(abs(x1 - x0), 1e-12)), float(max(abs(y1 - y0), 1e-12))

    w_proj, h_proj = projected_span()
    base_thresh = float(split_max_jump_frac) * max(w_proj, h_proj)

    common_kwargs = dict(
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        zorder=zorder,
        label=label,
    )
    if plot_kwargs:
        common_kwargs.update(plot_kwargs)

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

    target_crs = ax.projection
    artists = []
    did_label = False

    win = extent_lon_window() if (split_on_wrap and shift_to_extent) else None

    for run in runs:
        la = np.asarray(lat[run], dtype=np.float64)
        lo = np.asarray(lon[run], dtype=np.float64)

        # Clip latitudes (Basemap-style safety)
        if clip_lat_deg is not None:
            la = np.clip(la, -float(clip_lat_deg), float(clip_lat_deg))

        # Densify before projection if requested
        if densify:
            la, lo = densify_run(la, lo)

        if split_on_wrap:
            # Unwrap as a sequence (prevents fake 350° jumps caused by modulo at 180)
            lo = unwrap_longitudes(lo, wrap_center_deg)

            # Shift the whole run into the current axes lon window (Basemap shiftdata behavior)
            if win is not None:
                lon_min, lon_max = win
                lon_mid = 0.5 * (lon_min + lon_max)

                # Choose k so mean longitude is closest to the window center
                lo_mean = float(np.nanmean(lo))
                k = int(np.round((lon_mid - lo_mean) / 360.0))
                lo = lo + 360.0 * k

                # Optional: if still mostly outside, try neighbor shifts and pick best
                # (helps small circles / pole-adjacent paths)
                def score(arr):
                    inside = (arr >= lon_min - 1e-9) & (arr <= lon_max + 1e-9)
                    # primary: maximize inside count; secondary: minimize distance to center
                    return (
                        int(inside.sum()),
                        -float(np.nanmean(np.abs(arr - lon_mid))),
                    )

                best = lo
                best_s = score(lo)
                for dk in (-1, 1):
                    cand = lo + 360.0 * dk
                    s = score(cand)
                    if s > best_s:
                        best = cand
                        best_s = s
                lo = best

        # Project (Basemap m(lon,lat) style)
        xy = target_crs.transform_points(src_crs, lo, la)
        x = xy[:, 0]
        y = xy[:, 1]

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        x = x[ok]
        y = y[ok]

        if x.size < 2:
            continue

        dx = np.diff(x)
        dy = np.diff(y)
        jump = np.hypot(dx, dy)

        thresh = base_thresh
        if split_max_jump_world is not None:
            thresh = min(thresh, float(split_max_jump_world))

        cut = np.where(jump > thresh)[0] + 1
        segs = np.split(np.arange(x.size), cut)

        for si in segs:
            if si.size < 2:
                continue

            # only label once
            if did_label:
                common_kwargs["label"] = None

            line = ax.plot(
                x[si],
                y[si],
                transform=target_crs,  # already projected
                **common_kwargs,
            )[0]

            if effects is not None:
                line.set_path_effects(effects)

            artists.append(line)
            did_label = True

    return artists


def add_polygon(
    ax,
    geom,
    *,
    src_crs=None,
    # styling
    facecolor: str | None = "red",
    fill_alpha: float = 0.35,
    edgecolor: str | None = "red",
    edge_alpha: float = 0.85,
    linewidth: float = 1.0,
    linestyle: str = "-",
    zorder: float = 5.0,
    # robustness / seam handling
    split_on_wrap: bool = True,
    wrap_center_deg: float | None = None,  # defaults to ax.projection.central_longitude
    shift_to_extent: bool = True,  # basemap-like: shift geometry into current lon window
    clip_lat_deg: float = 89.999999,  # safety near poles / mercator singularities
    pin_pole_longitudes: bool = True,  # if lat is clipped to pole, force lon to wrap_center
    fix_invalid: bool = True,  # buffer(0) cleanups
    clip_to_view: bool = False,  # optionally intersect with current view extent (reduces proj NaNs)
    # rendering hints
    antialiased: bool = True,
    rasterized: bool = False,
    wrap_split_parts: bool = True,
):
    """
    Plot an arbitrary Polygon/MultiPolygon (or anything coercible to one) onto a Cartopy GeoAxes,
    with Basemap-like handling for antimeridian/seam crossings and polar singularities.

    Accepts `geom` as:
      - shapely Polygon/MultiPolygon/GeometryCollection
      - cartopy.io.shapereader.Record (uses `.geometry`)
      - GeoJSON-like mapping dict (uses shapely.geometry.shape)
      - Nx2 array/list of (lon,lat) coords (interpreted as a polygon shell)

    Key features:
      - Unwraps polygon rings as sequences around `wrap_center_deg` to avoid fake 360° jumps.
      - Optionally shifts the geometry by ±360*k into the current axes lon extent (Basemap shiftdata behavior),
        so zoomed-in dateline views don’t show an object “straddling” the seam.
      - Optionally splits at the current seam (wrap_center±180) to prevent a single polygon from drawing
        a long edge across the map.
      - Clips latitudes to `clip_lat_deg` and can pin longitudes at the pole to avoid undefined-lon artifacts.
      - Separate fill/edge alpha via RGBA colors (no shared `alpha=`).

    Returns:
      The FeatureArtist created by ax.add_geometries().
    """
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.colors import to_rgba

    # Shapely imports (Cartopy depends on shapely; keep local to avoid import-time issues elsewhere)
    from shapely.geometry import (
        Polygon,
        MultiPolygon,
        GeometryCollection,
        LineString,
        shape as shp_shape,
        box as shp_box,
    )
    from shapely.ops import split as shp_split
    from shapely.affinity import translate as shp_translate

    if src_crs is None:
        src_crs = ccrs.PlateCarree()

    # -----------------------------
    # Coercion helpers
    # -----------------------------
    def _coerce_geometry(obj):
        if obj is None:
            raise ValueError("geom is None")

        # Cartopy Record
        if hasattr(obj, "geometry"):
            obj = obj.geometry

        # GeoJSON-like
        if isinstance(obj, dict):
            return shp_shape(obj)

        # Nx2 coordinate array/list -> Polygon shell
        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
                shell = [(float(x), float(y)) for x, y in arr[:, :2]]
                if shell[0] != shell[-1]:
                    shell.append(shell[0])
                return Polygon(shell)
        return obj  # assume shapely geom already

    g0 = _coerce_geometry(geom)

    # -----------------------------
    # CRS / seam configuration
    # -----------------------------
    if wrap_center_deg is None:
        wrap_center_deg = float(getattr(ax.projection, "central_longitude", 0.0))

    def _wrap180(x: float) -> float:
        return (x + 180.0) % 360.0 - 180.0

    # Seam longitude for the map (where it “wraps”)
    seam_lon = _wrap180(wrap_center_deg + 180.0)

    # -----------------------------
    # Ring unwrap + lat safety
    # -----------------------------
    def _unwrap_ring(coords) -> list[tuple[float, float]] | None:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return None

        lon = arr[:, 0].copy()
        lat = arr[:, 1].copy()

        finite = np.isfinite(lon) & np.isfinite(lat)
        lon = lon[finite]
        lat = lat[finite]
        if lon.size < 3:
            return None

        # drop closing point for unwrap stability, re-close later
        is_closed = (
            (lon.size >= 2)
            and (abs(lon[0] - lon[-1]) < 1e-12)
            and (abs(lat[0] - lat[-1]) < 1e-12)
        )
        if is_closed and lon.size >= 4:
            lon0 = lon[:-1]
            lat0 = lat[:-1]
        else:
            lon0 = lon
            lat0 = lat

        # clip latitudes
        lat0 = np.clip(lat0, -float(clip_lat_deg), float(clip_lat_deg))

        # unwrap longitudes around wrap_center
        rel = lon0 - float(wrap_center_deg)
        rel_unw = np.rad2deg(np.unwrap(np.deg2rad(rel)))
        lon_unw = rel_unw + float(wrap_center_deg)

        # pole pinning (lon undefined at poles; prevents “spokes”)
        if pin_pole_longitudes:
            pole = np.abs(lat0) >= float(clip_lat_deg) - 1e-6
            if np.any(pole):
                lon_unw[pole] = float(wrap_center_deg)

        # re-close ring
        if lon_unw.size < 3:
            return None
        lon_out = np.r_[lon_unw, lon_unw[0]]
        lat_out = np.r_[lat0, lat0[0]]

        # must have >= 4 points for a valid ring
        if lon_out.size < 4:
            return None

        return [(float(x), float(y)) for x, y in zip(lon_out, lat_out)]

    def _unwrap_polygon(poly: Polygon) -> Polygon | None:
        shell = _unwrap_ring(poly.exterior.coords)
        if shell is None:
            return None

        holes = []
        for ring in poly.interiors:
            h = _unwrap_ring(ring.coords)
            if h is not None and len(h) >= 4:
                holes.append(h)

        out = Polygon(shell, holes=holes)
        if fix_invalid and (not out.is_valid):
            # common fix for self-crossings / minor issues after unwrap
            out = out.buffer(0)
        if out.is_empty:
            return None
        return out

    def _unwrap_geometry(obj):
        if obj is None:
            return None
        if obj.is_empty:
            return obj

        gt = getattr(obj, "geom_type", None)

        if gt == "Polygon":
            return _unwrap_polygon(obj)

        if gt == "MultiPolygon":
            polys = []
            for p in obj.geoms:
                pu = _unwrap_polygon(p)
                if pu is None or pu.is_empty:
                    continue
                if getattr(pu, "geom_type", None) == "Polygon":
                    polys.append(pu)
                elif getattr(pu, "geom_type", None) == "MultiPolygon":
                    polys.extend(pp for pp in pu.geoms if not pp.is_empty)
                else:
                    raise TypeError(f"Unsupported geometry returned from unwrap: {type(pu)!r}")
            if not polys:
                return None
            out = MultiPolygon(polys)
            if fix_invalid and (not out.is_valid):
                out = out.buffer(0)
            return out

        if gt == "GeometryCollection":
            parts = []
            for gg in obj.geoms:
                uu = _unwrap_geometry(gg)
                if uu is not None and (not uu.is_empty):
                    parts.append(uu)
            if not parts:
                return None
            out = GeometryCollection(parts)
            return out

        # If it's not a polygon-ish geometry, return as-is (or you can raise)
        return obj

    g = _unwrap_geometry(g0)
    if g is None or getattr(g, "is_empty", False):
        # nothing to draw
        return ax.add_geometries([], src_crs)

    # -----------------------------
    # Optional: shift into current lon window (Basemap-like)
    # -----------------------------
    def _extent_lon_window():
        try:
            lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
            lon_min = float(lon_min)
            lon_max = float(lon_max)
            lat_min = float(lat_min)
            lat_max = float(lat_max)
            if lon_max < lon_min:
                lon_max += 360.0
            return lon_min, lon_max, lat_min, lat_max
        except Exception:
            return None

    win = _extent_lon_window() if shift_to_extent else None

    def _choose_best_shift(gg):
        if win is None:
            return gg
        lon_min, lon_max, _, _ = win
        lon_mid = 0.5 * (lon_min + lon_max)
        win_w = max(1e-9, lon_max - lon_min)

        # reference longitude from representative point (more stable than centroid)
        try:
            ref_x = float(gg.representative_point().x)
        except Exception:
            ref_x = float((gg.bounds[0] + gg.bounds[2]) * 0.5)

        k0 = int(np.round((lon_mid - ref_x) / 360.0))
        candidates = [k0 - 1, k0, k0 + 1]

        def score(gg2):
            minx, _, maxx, _ = gg2.bounds
            # overlap of bounds with window (rough but effective)
            overlap = max(0.0, min(maxx, lon_max) - max(minx, lon_min))
            overlap_frac = overlap / win_w
            center = 0.5 * (minx + maxx)
            dist = abs(center - lon_mid)
            # prefer overlap, then closeness
            return (overlap_frac, -dist)

        best = gg
        best_s = score(gg)
        for k in candidates:
            gg2 = shp_translate(gg, xoff=360.0 * float(k), yoff=0.0)
            s = score(gg2)
            if s > best_s:
                best_s = s
                best = gg2
        return best

    g = _choose_best_shift(g)

    # -----------------------------
    # Optional: clip to view extent in lon/lat to reduce projection NaNs/warnings
    # -----------------------------
    if clip_to_view and win is not None:
        lon_min, lon_max, lat_min, lat_max = win
        # handle wrapped window by clipping against up to two boxes
        boxes = []
        if lon_max - lon_min <= 360.0 + 1e-9:
            boxes.append(shp_box(lon_min, lat_min, lon_max, lat_max))
        else:
            # extremely rare; just don't clip
            boxes = []

        if boxes:
            clip_geom = boxes[0]
            try:
                g2 = g.intersection(clip_geom)
                if not g2.is_empty:
                    g = g2
            except Exception:
                pass

    # -----------------------------
    # Optional: split at seam to avoid long edges across the wrap boundary
    # -----------------------------
    geoms_to_draw = [g]

    if split_on_wrap:
        out_parts = []
        # Build seam lines at seam_lon + 360*k so it works regardless of the chosen shift
        seam_lines = [
            LineString([(seam_lon + 360.0 * k, -90.0), (seam_lon + 360.0 * k, 90.0)])
            for k in (-1, 0, 1)
        ]

        def _split_one(gg):
            parts = [gg]
            for ln in seam_lines:
                new_parts = []
                for p in parts:
                    try:
                        minx, _, maxx, _ = p.bounds
                        sx = ln.coords[0][0]
                        # quick reject if seam not within bounds
                        if not (minx < sx < maxx):
                            new_parts.append(p)
                            continue
                        res = shp_split(p, ln)
                        # shp_split returns GeometryCollection
                        for r in getattr(res, "geoms", []):
                            if not r.is_empty:
                                new_parts.append(r)
                    except Exception:
                        new_parts.append(p)
                parts = new_parts
            return parts

        for gg in geoms_to_draw:
            out_parts.extend(_split_one(gg))

        # keep only polygonal parts
        geoms_to_draw = [
            p
            for p in out_parts
            if (p is not None)
            and (not p.is_empty)
            and getattr(p, "geom_type", "") in ("Polygon", "MultiPolygon")
        ]

    # Final validity cleanup (optional)
    if fix_invalid:
        cleaned = []
        for gg in geoms_to_draw:
            try:
                if not gg.is_valid:
                    gg = gg.buffer(0)
            except Exception:
                pass
            if gg is not None and (not gg.is_empty):
                cleaned.append(gg)
        geoms_to_draw = cleaned

    if not geoms_to_draw:
        return ax.add_geometries([], src_crs)

    def _wrap_part_into_world(gg):
        # Bring a geometry into [wrap_center-180, wrap_center+180] by translating lon.
        try:
            x = float(gg.representative_point().x)
        except Exception:
            x = 0.5 * (gg.bounds[0] + gg.bounds[2])

        lo = float(wrap_center_deg) - 180.0
        hi = float(wrap_center_deg) + 180.0

        # Choose integer shift so representative lon lands in the interval.
        k = int(np.floor((x - lo) / 360.0))
        x_in = x - 360.0 * k
        if x_in < lo:
            x_in += 360.0
            k -= 1
        elif x_in > hi:
            x_in -= 360.0
            k += 1

        return shp_translate(gg, xoff=-360.0 * float(k), yoff=0.0)

    if split_on_wrap and wrap_split_parts:
        wrapped_parts = []
        for gg in geoms_to_draw:
            try:
                wrapped_parts.append(_wrap_part_into_world(gg))
            except Exception:
                wrapped_parts.append(gg)
        geoms_to_draw = wrapped_parts

    # -----------------------------
    # Draw with independent edge/fill alpha via RGBA
    # -----------------------------
    fc = "none" if (facecolor is None) else facecolor
    ec = "none" if (edgecolor is None) else edgecolor

    fc_rgba = "none" if fc == "none" else to_rgba(fc, float(fill_alpha))
    ec_rgba = "none" if ec == "none" else to_rgba(ec, float(edge_alpha))

    artist = ax.add_geometries(
        geoms_to_draw,
        crs=src_crs,
        facecolor=fc_rgba,
        edgecolor=ec_rgba,
        linewidth=float(linewidth),
        linestyle=linestyle,
        zorder=float(zorder),
        antialiased=bool(antialiased),
        rasterized=bool(rasterized),
    )
    return artist
