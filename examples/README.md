# Nebula Space Toolkit Example Learning Path

## Setup (once per environment)

From the repository root, install Nebula Space Toolkit into your active environment:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[all]"
```

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
   - Build interval coverage target grids
   - Compute exact access intervals
   - Compute N-of-M access duration metrics

Legacy script examples are still available in this folder for direct `python` execution.
