# datacube-rs

Temporal data cubes for remote sensing time series in Rust — per-pixel trend
analysis (OLS, Theil-Sen, Mann-Kendall) over `(band, y, x, time)` cubes with
streaming, Rayon-parallel iteration.

Part of the SurtGIS family of Rust geospatial engines.

## Workspace

- `crates/datacube-core` — cube model and statistics (no I/O).
- `crates/datacube-cli` — `datacube` binary; `datacube trend series.csv`
  reports all three estimators as JSON.

## Quick start

```bash
cargo test                                   # unit + doc tests
cargo run -p datacube-cli -- trend ndvi.csv  # CSV: "value" or "t,value"
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
`pyMannKendall` (original_test, sens_slope) and `scipy.stats.linregress`
within `1e-9` relative tolerance.

Documented divergences from the references:

- Constant series: scipy reports `NaN` for r²/std_err/p; we define the perfect
  fit (`r² = 1`, `p = 1`).
- `pymannkendall.sens_slope` assumes unit spacing after dropping NaN; we keep
  the true time gaps.

## Roadmap (v0.1 → v0.2)

- [x] Cube model + streaming per-pixel/chunk iterators
- [x] OLS linear trend, Theil-Sen, Mann-Kendall (tie-corrected)
- [ ] Harmonic regression (seasonality/phenology)
- [ ] STAC/COG temporal stacking (via SurtGIS STAC client)
- [ ] BFAST-style break detection, temporal compositing, gap-filling

## License

MIT OR Apache-2.0
