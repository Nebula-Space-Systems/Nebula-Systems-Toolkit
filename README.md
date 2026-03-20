# Nebula Space Toolkit

`nstk` is the Python package for Nebula Space Toolkit, a library for space systems engineering and analysis.

## Package Name

- Brand name: `Nebula Space Toolkit`
- PyPI package: `nstk`
- Python import: `import nstk`

## Data Packaging

Nebula Space Toolkit's offline Orekit and Cartopy assets live in the separate
`nstk-data` package. The main `nstk` package depends on `nstk-data`, which
lets code releases move independently from the larger bundled data payload.

Orekit-backed features initialize automatically on first use. If you want to
use a custom Orekit data directory, call `nstk.set_orekit_data_path(...)`
before using those features.

## Goals

- Build and propagate satellite orbits with configurable propagators and force models
- Model spacecraft attitude and frame behavior
- Compute coverage, access, and visibility metrics over local regions or the full globe
- Support systems analysis workflows with reusable building blocks instead of one-off scripts

## Example Analysis Studies

- Intersatellite range and range-rate analysis
- Relative angle and angle-rate analysis in spacecraft body frames
- Doppler and geometry-driven measurement studies
- Coverage and revisit analysis for single spacecraft and constellations
