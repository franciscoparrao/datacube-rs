# datacube-rs

Temporal data cubes for remote sensing time series in Rust — per-pixel trend
and seasonality analysis (OLS, Theil-Sen, Mann-Kendall, harmonic regression)
over `(band, y, x, time)` cubes with streaming, Rayon-parallel iteration.

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

## Quick start

```bash
cargo test                                   # unit + doc tests
cargo run -p datacube-cli -- trend ndvi.csv  # CSV: "value" or "t,value"
cargo run -p datacube-cli -- harmonic ndvi.csv --period 1 --harmonics 2

# Sentinel-2 trend map straight from Planetary Computer (needs --features stac)
cargo run -p datacube-cli --features stac -- stack \
  --collection sentinel-2-l2a --assets B04 \
  --bbox -70.70,-33.50,-70.68,-33.48 --datetime 2024-01-01/2024-06-30 \
  --max-cloud 30 --overview 3 \
  --output slope.tif --pvalue-output pvalue.tif
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

`scripts/validate_pymannkendall.py` cross-checks every reported field against
`pyMannKendall` (original_test, sens_slope), `scipy.stats.linregress` and
`numpy.linalg.lstsq` (harmonic design matrix) within `1e-9` relative
tolerance.

Documented divergences from the references:

- Constant series: scipy reports `NaN` for r²/std_err/p; we define the perfect
  fit (`r² = 1`, `p = 1`).
- `pymannkendall.sens_slope` assumes unit spacing after dropping NaN; we keep
  the true time gaps.

## Roadmap (v0.1 → v0.2)

- [x] Cube model + streaming per-pixel/chunk iterators
- [x] OLS linear trend, Theil-Sen, Mann-Kendall (tie-corrected)
- [x] Harmonic regression with trend (seasonality/phenology, amplitude/phase)
- [x] STAC/COG temporal stacking (Planetary Computer / Earth Search, via
  SurtGIS): cloud filter, grid alignment, fractional-year time axis,
  GeoTIFF trend maps
- [ ] BFAST-style break detection, temporal compositing, gap-filling
- [ ] Cross-UTM-zone mosaicking; same-date tile compositing

## License

MIT OR Apache-2.0
