# Nebula Space Toolkit

`nstk` is the Python package for Nebula Space Toolkit, a library for space
systems engineering and analysis. It is designed for users who need reusable
building blocks for orbit propagation, coordinate transforms, geometry,
coverage analysis, localization workflows, and mission-oriented plotting.

## What NSTK Provides

- Orbit construction and propagation interfaces built around Orekit
- Coordinate and frame transforms for common astrodynamics workflows
- Coverage and access analysis utilities
- Line-of-sight, terrain, and other geometric analysis tools
- Localization and measurement helpers for angle, time-difference, and
  frequency-difference studies
- 2D and 3D plotting tools for orbits and geospatial views

## Package Layout

The top-level package is organized into the following modules:

- `nstk.propagation`: orbit creation, propagation, and Walker constellations
- `nstk.transforms`: geodetic, ECEF, ENU, AER, and timed frame transforms
- `nstk.geometry`: line-of-sight, terrain, raster-mask, and sun-position tools
- `nstk.coverage`: interval and fixed-step coverage analysis
- `nstk.localization`: measurement models and localization helpers
- `nstk.plotting`: basemaps and orbit visualization
- `nstk.time_utils`: shared Orekit and Astropy time conversion helpers

## Installation

Install the base package with:

```bash
pip install nstk
```

The base install includes the core package plus the offline data dependency
`nstk-data`.

Optional extras are available for heavier workflows:

```bash
pip install "nstk[propagation]"
pip install "nstk[plotting]"
pip install "nstk[all]"
```

Extras are intended for:

- `propagation`: Orekit-backed propagation and timed frame operations
- `plotting`: Cartopy and related geospatial plotting support
- `all`: all optional dependencies used by the project

## Installing From Source

For local development, install from the repository root:

```bash
python -m pip install -e .
```

To include optional features during development:

```bash
python -m pip install -e ".[propagation]"
python -m pip install -e ".[plotting]"
python -m pip install -e ".[all]"
```

If Orekit-backed features later report that `nstk-data` is missing, install it
into the same environment as `nstk` and your Jupyter kernel:

```bash
python -m pip install nstk-data
```

## Data and Runtime Behavior

Nebula Space Toolkit uses the separate `nstk-data` package for bundled Orekit
and Cartopy assets. This keeps the code package smaller and allows data updates
to be managed independently.

Orekit-backed features initialize lazily on first use. Most users do not need
to do anything beyond installing the appropriate extra.

If you want to use a custom Orekit data directory instead of the bundled data
package, configure it before first use:

```python
import nstk

nstk.set_orekit_data_path("/path/to/orekit-data")
```

If you prefer eager initialization, you can explicitly initialize the runtime:

```python
import nstk

nstk.initialize_orekit()
```

## Minimal Example

The example below shows a simple Orekit-backed orbit workflow:

```python
from astropy.time import Time
from nstk.propagation import Orbit

orbit = Orbit.from_kepler_two_body(
    epoch=Time("2026-01-01T00:00:00", scale="utc"),
    a=7000e3,
    e=0.001,
    i=0.9,
    raan=0.1,
    argp=0.2,
    anomaly=0.3,
    anomaly_type="mean",
)

position_m = orbit.get_p(0.0, frame="gcrf", as_quantity=False)
print(position_m)
```

This example requires the `propagation` extra.

## Orbit Attitude Defaults

NSTK orbit propagators default to `attitude="vvlh"`, which is implemented as
Orekit `LofOffset(native_frame, LOFType.VVLH)`. This is a reasonable default
for a general Earth-observing satellite because it keeps the spacecraft body in
a local orbital, nadir-pointing attitude law.

Supported local-orbital attitude names include:

- `vvlh` and `lvlh_ccsds` for the CCSDS-style local orbital frame family
- `lvlh` and `qsw` for Orekit's STK/Vallado-style LVLH family
- `tnw`, `ntw`, `vnc`, `eqw`, `enu`, and `ned` for other built-in Orekit
  local orbital frames

You can also pass an Orekit `LOFType`, a prebuilt Orekit `AttitudeProvider`,
or a callable accepted by `Orbit.set_attitude_law(...)`.

## Examples

Additional usage examples are provided in the `examples/` directory, including:

- orbit usage
- orbit attitude configuration
- transforms
- Walker constellations
- interval coverage

## Project Identity

- Brand name: `Nebula Space Toolkit`
- PyPI package: `nstk`
- Python import: `import nstk`


## Attribution

NSTK is a part of Nebula Space Systems
