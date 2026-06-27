# datacube-rs

Temporal data cubes for remote sensing time series in Rust — per-pixel trend,
seasonality and structural-break analysis (OLS, Theil-Sen, Mann-Kendall,
harmonic regression, OLS-CUSUM structural breaks) over `(band, y, x, time)`
cubes with streaming, Rayon-parallel iteration, temporal compositing and
gap-filling.

Part of the SurtGIS family of Rust geospatial engines.

## Workspace

- `crates/datacube-core` — cube model and statistics (no I/O).
- `crates/datacube-io` — STAC/COG temporal stacking into cubes. Reuses the
  SurtGIS cloud stack (STAC client, COG reader, SAS signing, UTM
  reprojection), so it requires a **sibling checkout of `surtgis`** next to
  this repository.
- `crates/datacube-cli` — `datacube` binary; `datacube trend series.csv`
  reports all three estimators as JSON. Build with `--features stac` to
  enable `datacube stack`.
- `crates/datacube-python` — PyO3 bindings (`datacube_rs` module): the cube
  model and statistics over NumPy arrays. See its
  [README](crates/datacube-python/README.md). I/O stacking is not exposed to
  Python yet — use the CLI (`datacube stack`) for STAC/COG ingestion.
- `crates/datacube-wasm` — WebAssembly bindings + a browser demo that fits
  harmonics and detects breaks live. See its
  [README](crates/datacube-wasm/README.md).

## Quick start

```bash
cargo test                                   # unit + doc tests
cargo run -p datacube-cli -- trend ndvi.csv  # CSV: "value" or "t,value"
cargo run -p datacube-cli -- harmonic ndvi.csv --period 1 --harmonics 2
cargo run -p datacube-cli -- breaks ndvi.csv --harmonics 1 --period 1

# Sentinel-2 trend map straight from Planetary Computer (needs --features stac).
# Optional: monthly median composite + gap-fill + reflectance scaling.
cargo run -p datacube-cli --features stac -- stack \
  --collection sentinel-2-l2a --assets B04 \
  --bbox -70.70,-33.50,-70.68,-33.48 --datetime 2024-01-01/2024-06-30 \
  --max-cloud 30 --overview 3 --scale 0.0001 --offset -0.1 \
  --composite monthly --composite-method median --gapfill 0.25 \
  --output slope.tif --pvalue-output pvalue.tif \
  --breaks-output nbreaks.tif --first-break-output firstbreak.tif
```

```rust
use datacube_core::{Cube, stats};

let cube = Cube::new(data, time, bands)?;            // (band, y, x, time)
let slopes = cube.par_map_series(0, |t, y| {
    stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN)
})?;
```

Missing observations are `NaN` and dropped pairwise; Theil-Sen and OLS use the
real time coordinates, so irregular sampling (cloud-masked scenes) is handled
correctly.

## Numerical parity

`scripts/validate_stats.py` cross-checks every reported field against
`pyMannKendall` (original_test, sens_slope), `scipy.stats.linregress`,
`numpy.linalg.lstsq` (harmonic design matrix) and
`statsmodels.breaks_cusumolsresid` (OLS-CUSUM break statistic) within `1e-9`
relative tolerance — 103 checks total.

statsmodels needs a pandas-compatible environment, so the script runs in a
dedicated venv:

```bash
python3 -m venv .venv-validate
.venv-validate/bin/pip install numpy scipy pymannkendall statsmodels
.venv-validate/bin/python scripts/validate_stats.py
```

Documented divergences from the references:

- Constant series: scipy reports `NaN` for r²/std_err/p; we define the perfect
  fit (`r² = 1`, `p = 1`).
- `pymannkendall.sens_slope` assumes unit spacing after dropping NaN; we keep
  the true time gaps.

## Roadmap

- [x] Cube model + streaming per-pixel/chunk iterators
- [x] OLS linear trend, Theil-Sen, Mann-Kendall (tie-corrected)
- [x] Harmonic regression with trend (seasonality/phenology, amplitude/phase)
- [x] STAC/COG temporal stacking (Planetary Computer / Earth Search, via
  SurtGIS): cloud filter, grid alignment, fractional-year time axis,
  reflectance scaling, GeoTIFF trend maps
- [x] Structural break detection (OLS-CUSUM + binary segmentation, in the
  spirit of BFAST), as a per-series stat and as per-pixel break-count /
  first-break-time maps over a stacked cube
- [x] Temporal compositing (same-time / period, median·mean·min·max) and
  linear gap-filling
- [x] Criterion benchmarks (`BENCHMARKS.md`)
- [x] Rayon-parallel compositing / gap-filling
- [x] Cross-UTM-zone mosaicking (reproject neighbouring-zone scenes onto the
  reference grid instead of skipping them)
- [x] PyO3 bindings (`datacube_rs` module: cube + statistics over NumPy)
- [x] WASM bindings + browser time-series demo (harmonic fit + live breaks)

## Performance

See [`BENCHMARKS.md`](BENCHMARKS.md). `par_map_series` scales linearly with
pixel count across the Rayon pool; Theil-Sen and Mann-Kendall are O(n²) per
pixel and dominate long records.

## Citation

If you use datacube-rs, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository"). An archived, DOI-bearing version is deposited
on Zenodo (see `.zenodo.json`).

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your
option.
