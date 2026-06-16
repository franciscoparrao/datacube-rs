# datacube-rs (Python)

Python bindings for [datacube-rs](https://github.com/franciscoparrao/datacube-rs):
the temporal-cube model and per-pixel trend / seasonality / break statistics,
with the Rust core doing the heavy lifting (Rayon-parallel, NaN-aware) and
NumPy arrays at the boundary.

This is the analytics layer — the natural complement to gdalcubes / stars for
people who already have a cube in NumPy.

## Build

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release        # from crates/datacube-python/
```

## Use

```python
import numpy as np
import datacube_rs as dc

# --- per-series statistics (1-D arrays; NaN = missing, dropped pairwise) ---
t = np.arange(120) / 23.0                     # fractional years
y = 0.4 + 0.01 * t + 0.2 * np.cos(2 * np.pi * t)

dc.linear_trend(t, y)                          # {slope, intercept, r_squared, ...}
dc.theil_sen(t, y)                             # robust slope
dc.mann_kendall(y, alpha=0.05)                 # {trend, tau, p_value, ...}
dc.harmonic_regression(t, y, period=1.0, n_harmonics=2)
dc.detect_breaks(t, y, n_harmonics=1)          # BFAST-style structural breaks

# --- cube operations ((band, y, x, time) array) ---
cube = dc.Cube(data, time, ["ndvi"])           # data: 4-D float64, NaN = nodata
slope, pvalue = cube.trend_map(band=0, method="theil_sen")   # two (y, x) arrays
monthly = cube.composite("monthly", "median")  # cloud-robust temporal composite
filled  = cube.gapfill(max_gap=0.25)           # linear gap-fill, ≤ 0.25 yr gaps
arr = cube.to_numpy()                          # back to a NumPy array
```

`trend_map`, `composite` and `gapfill` run the Rust core in parallel across
all cores (the worker threads never touch Python objects, so the GIL is not a
bottleneck): a Theil-Sen + Mann-Kendall trend map of a 256×256×60 cube is
sub-second.

Numerical parity of the underlying statistics is validated against
pyMannKendall, scipy, NumPy and statsmodels (see the main repo).
