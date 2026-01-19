import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from nebula import NEBULA_ROOT_DIR

# -----------------------------------------------------------------------------
# Cartopy data directory setup
# -----------------------------------------------------------------------------

project_root = os.path.dirname(NEBULA_ROOT_DIR)
cartopy.config["pre_existing_data_dir"] = os.path.join(project_root, "data", "cartopy")
cartopy.config["data_dir"] = os.path.join(project_root, "data", "cartopy")

ProjectionName = Literal[
    "PlateCarree", "Robinson", "Mollweide", "Mercator", "Orthographic"
]


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
    # Used for Orthographic only
    central_longitude: float = 0.0
    central_latitude: float = 0.0


@dataclass(frozen=True)
class ExtentConfig:
    """
    Map extent control.

    If global_map=True, the map is set to global and extent is ignored.
    Otherwise extent=(west, east, south, north) in degrees and `crs` is used.
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
    Rendering controls for filled polygon features (LAND/OCEAN).

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


@dataclass(frozen=True)
class BordersConfig:
    enabled: bool = False
    color: str = "#555555"
    linewidth: float = 0.4
    zorder: float = 3.0


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
# Presets
# -----------------------------------------------------------------------------

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

DARK_DETAILED = MapConfig(
    theme=_DARK_THEME,
    ocean=FillRenderConfig(
        enabled=True,
        facecolor="#0F0F0F",
        zorder=0.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    land=FillRenderConfig(
        enabled=True,
        facecolor="#4D4D4D",
        zorder=1.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    coastlines=CoastlinesConfig(
        enabled=True, color="#0F0F0F", linewidth=0.4, zorder=3.0
    ),
    borders=BordersConfig(enabled=True, color="#707070", linewidth=0.4, zorder=3.0),
    gridlines=GridlinesConfig(alpha=0.35),
    bump_texture=BumpTextureConfig(opacity=0.32),
)
DARK_DETAILED_NO_GRID = replace(
    DARK_DETAILED,
    gridlines=replace(DARK_DETAILED.gridlines, alpha=0.0),
    borders=replace(DARK_DETAILED.borders, enabled=False),
)

DARK = replace(
    DARK_DETAILED,
    borders=replace(DARK_DETAILED.borders, enabled=False),
    bump_texture=replace(DARK_DETAILED.bump_texture, enabled=False),
    land_shadow=replace(DARK_DETAILED.land_shadow, enabled=False),
)

DARK_NO_GRID = replace(
    DARK,
    gridlines=replace(DARK_DETAILED.gridlines, alpha=0.0),
)

LIGHT_DETAILED = MapConfig(
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
        facecolor="#E6E6E6",
        zorder=1.0,
        seam_fix=True,
        antialiased=True,
        rasterized=True,
    ),
    coastlines=CoastlinesConfig(
        enabled=True, color="#B5B5B5", linewidth=0.2, zorder=3.0
    ),
    borders=BordersConfig(enabled=True, color="#555555", linewidth=0.4, zorder=3.0),
    gridlines=GridlinesConfig(color="#222222", alpha=0.5),
    bump_texture=BumpTextureConfig(opacity=0.18),
    land_shadow=ShadowConfig(
        enabled=True, offset=(-0.8, -0.8), color="black", alpha=0.22
    ),
)

LIGHT_DETAILED_NO_GRID = replace(
    LIGHT_DETAILED,
    gridlines=replace(LIGHT_DETAILED.gridlines, alpha=0.0),
    borders=replace(LIGHT_DETAILED.borders, enabled=False),
)

LIGHT = replace(
    LIGHT_DETAILED,
    borders=replace(LIGHT_DETAILED.borders, enabled=False),
    bump_texture=replace(LIGHT_DETAILED.bump_texture, enabled=False),
    land_shadow=replace(LIGHT_DETAILED.land_shadow, enabled=True),
)

LIGHT_NO_GRID = replace(
    LIGHT,
    gridlines=replace(LIGHT_DETAILED.gridlines, alpha=0.0),
    borders=replace(LIGHT_DETAILED.borders, enabled=False),
)

DARK_RASTER = replace(
    DARK_DETAILED,
    raster_background=replace(DARK_DETAILED.raster_background, enabled=True),
    bump_texture=replace(DARK_DETAILED.bump_texture, enabled=False),
    coastlines=replace(DARK_DETAILED.coastlines, enabled=False),
    gridlines=replace(DARK_DETAILED.gridlines, color="#424242", alpha=0.7),
    borders=replace(DARK_DETAILED.borders, enabled=False),
)

DARK_RASTER_NO_GRID = replace(
    DARK_RASTER,
    gridlines=replace(DARK_RASTER.gridlines, alpha=0.0),
)

LIGHT_RASTER = replace(
    LIGHT_DETAILED,
    raster_background=replace(LIGHT_DETAILED.raster_background, enabled=True),
    bump_texture=replace(LIGHT_DETAILED.bump_texture, enabled=False),
    coastlines=replace(LIGHT_DETAILED.coastlines, enabled=False),
    gridlines=replace(LIGHT_DETAILED.gridlines, color="#EEEEEE", alpha=0.85),
    borders=replace(LIGHT_DETAILED.borders, enabled=False),
)

LIGHT_RASTER_NO_GRID = replace(
    LIGHT_RASTER,
    gridlines=replace(LIGHT_RASTER.gridlines, alpha=0.0),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _projection_from_config(cfg: ProjectionConfig) -> ccrs.Projection:
    if cfg.name == "PlateCarree":
        return ccrs.PlateCarree()
    if cfg.name == "Robinson":
        return ccrs.Robinson()
    if cfg.name == "Mollweide":
        return ccrs.Mollweide()
    if cfg.name == "Mercator":
        return ccrs.Mercator()
    if cfg.name == "Orthographic":
        return ccrs.Orthographic(
            central_longitude=float(cfg.central_longitude),
            central_latitude=float(cfg.central_latitude),
        )
    raise ValueError(f"Unknown projection: {cfg.name!r}")


def _default_bump_path() -> str:
    # Your file name
    return os.path.join(cartopy.config["data_dir"], "raster", "topo_and_bathy.png")


def _default_raster_path() -> str:
    return os.path.join(cartopy.config["data_dir"], "raster", "shadedrelief.jpg")


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
    ax.imshow(img_hi, **common)
    ax.imshow(img_lo, **common)


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
    seen = set()
    for v in wrapped:
        # optionally remove dateline entirely
        if not cfg.include_dateline and abs(abs(v) - 180.0) < 1e-9:
            continue
        # normalize dateline to requested sign/value
        if cfg.include_dateline and abs(abs(v) - 180.0) < 1e-9:
            v = float(cfg.dateline_value_deg)

        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))

    # Ensure dateline present exactly once if requested
    if cfg.include_dateline:
        dl = float(cfg.dateline_value_deg)
        out = [v for v in out if abs(abs(v) - 180.0) > 1e-9] + [dl]

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


def make_basemap(cfg: MapConfig = DARK) -> tuple:
    """
    Create a Cartopy basemap (figure + GeoAxes) driven entirely by MapConfig.

    Guarantees:
    - LAND and OCEAN fills are always alpha=1.0 (solid).
    - Shadow (if enabled) is applied as a path-effect to the LAND artist, not as a second polygon.
    - Seam fix is handled via rendering controls (no visible strokes).
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

    crs = _projection_from_config(cfg.projection)

    fig = plt.figure(figsize=cfg.figsize, facecolor=t.figure_face)
    ax = plt.axes(projection=crs)
    ax.set_facecolor(t.axes_face)

    # Extent/global
    if cfg.extent.global_map:
        ax.set_global()
    else:
        if cfg.extent.extent is None:
            raise ValueError("ExtentConfig.extent must be set when global_map=False")
        ax.set_extent(cfg.extent.extent, crs=cfg.extent.crs)

    # Outline/frame
    if hasattr(ax, "outline_patch") and ax.outline_patch is not None:
        if cfg.outline.enabled:
            ax.outline_patch.set_visible(True)
            ax.outline_patch.set_linewidth(cfg.outline.linewidth)
            ax.outline_patch.set_edgecolor(cfg.outline.color or t.outline_edge)
        else:
            ax.outline_patch.set_visible(False)

    # Background: raster OR vector fills (+ bump)
    land_artist = None

    if cfg.raster_background.enabled:
        path = cfg.raster_background.path or _default_raster_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Raster background not found: {path!r}. Set MapConfig.raster_background.path."
            )
        img = mpimg.imread(path)
        ax.imshow(
            img,
            extent=[-180, 180, -90, 90],
            transform=ccrs.PlateCarree(),
            origin=cfg.raster_background.origin,
            interpolation=cfg.raster_background.interpolation,
            zorder=cfg.raster_background.zorder,
            alpha=cfg.raster_background.alpha,
        )
    else:
        # OCEAN (solid, alpha=1.0)
        if cfg.ocean.enabled:
            ocean_artist = ax.add_feature(
                cfeature.OCEAN,
                facecolor=cfg.ocean.facecolor,
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=cfg.ocean.zorder,
            )
            _apply_fill_rendering(ocean_artist, cfg.ocean)

        # LAND (solid, alpha=1.0)
        if cfg.land.enabled:
            land_artist = ax.add_feature(
                cfeature.LAND,
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

    # Coastlines / borders
    if cfg.coastlines.enabled:
        ax.coastlines(
            resolution="auto",
            linewidth=cfg.coastlines.linewidth,
            edgecolor=cfg.coastlines.color,
            zorder=cfg.coastlines.zorder,
        )

    if cfg.borders.enabled:
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=cfg.borders.linewidth,
            edgecolor=cfg.borders.color,
            zorder=cfg.borders.zorder,
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
            else {"size": 11, "alpha": 1.0, "color": t.grid_label_color}
        )
        gl.ylabel_style = (
            dict(gcfg.ylabel_style)
            if gcfg.ylabel_style
            else {"size": 11, "alpha": 1.0, "color": t.grid_label_color}
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
                gl.ylocator = mticker.FixedLocator(vals)

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
                gl.xlocator = mticker.FixedLocator(vals)

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

    return fig, ax, crs, gl


# -----------------------------------------------------------------------------
# Trace drawing
# -----------------------------------------------------------------------------


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
        breaks = np.where(dlon > float(dateline_threshold_deg))[0] + 1
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
