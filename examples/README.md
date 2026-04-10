# Nebula Space Toolkit Example Learning Path

## Setup (once per environment)

From the repository root, install Nebula Space Toolkit into your active environment:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[all]"
```

If an Orekit-backed notebook later reports `No module named 'nstk_data'`,
install the bundled data package into the same environment as the notebook
kernel:

```powershell
.venv\Scripts\python.exe -m pip install nstk-data
```

If you prefer to manage Orekit data yourself, call
`nstk.set_orekit_data_path(...)` before using Orekit-backed examples.

If you see `ModuleNotFoundError: No module named 'nstk'` while running a
notebook from this `examples/` folder, restart the kernel after install.

Use these notebooks in order:

1. `01_orbit_usage.ipynb`
   - Build analytical and numerical orbits
   - Query state vectors and geodetic outputs
   - Use different time input types

2. `02_transforms_usage.ipynb`
   - Geodetic, ECEF, ENU, AER transforms
   - Coarse ECI/ECEF transforms and roundtrip checks
   - Timed Orekit frame transforms with `transform(...)`

3. `03_walker_constellation.ipynb`
   - Build Walker constellations from seed `Orbit` objects with explicit RAAN spans, offsets, and anomaly controls
   - Use two-body and numerical member construction

4. `04_coverage.ipynb`
   - Use the new `IntervalCoverage` object API
   - Mix orbit-backed and sampled observers
   - Build regional targets from domains and samplers
   - Apply constraints, observer subsets, and custom metrics
   - Make map plots, histograms, ECDFs, and target timelines

5. `05_plotting.ipynb`
   - Use `GeoMap` for direct map composition
   - Reuse `MapStyle` and `MapView` across plots
   - Mix geographic layers, coverage plots, and orbit plots
   - Drop to `MapConfig` only when you need exact renderer-level control

6. `06_attitude.ipynb`
   - Understand NSTK orbit attitude defaults and naming conventions
   - Compare `vvlh`, `lvlh_ccsds`, `lvlh`, `qsw`, and other Orekit LOFs
   - Configure attitudes with strings, `LOFType`, mappings, providers, and callables
   - Query quaternions, body-frame angular rates, and angular accelerations

These notebooks are the canonical examples for the repository.
