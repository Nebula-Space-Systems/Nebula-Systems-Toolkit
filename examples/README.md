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
   - Build Walker delta/star constellations
   - Use two-body and numerical member construction

4. `04_interval_coverage.ipynb`
   - Short migration note pointing to the redesigned coverage workflow

5. `05_coverage.ipynb`
   - Use the new `IntervalCoverage` object API
   - Mix orbit-backed and sampled observers
   - Build regional targets from domains and samplers
   - Apply constraints, observer subsets, and custom metrics
   - Make map plots, histograms, ECDFs, and target timelines

Legacy script examples are still available in this folder for direct `python` execution.
