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
  enable `datacube stack` and `datacube zonal`.
- `crates/datacube-python` — PyO3 bindings (`datacube_rs` module): the cube
  model and statistics over NumPy arrays. See its
  [README](crates/datacube-python/README.md). STAC/COG ingestion (`dc.stack`)
  and zonal aggregation (`Cube.zonal`) are exposed when the wheel is built
  with the `stac` feature (needs the SurtGIS sibling checkout).
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

## Zonal aggregation by polygon

When the unit of analysis is a polygon (a wetland, a field, a catchment) rather
than the cube's bounding box, `datacube zonal` reduces each polygon of a vector
layer to a tidy scalar series `(polygon_id, time, band, reducer, value,
n_valid, n_total)`:

```bash
# annual median NDVI per wetland, straight from a stacked Sentinel-2 cube
cargo run -p datacube-cli --features stac -- zonal \
  --collection sentinel-2-l2a --assets B04,B08 \
  --bbox -71.0,-33.9,-70.9,-33.8 --datetime 2022-01-01/2023-12-31 \
  --mask scl --index ndvi --nir B08 --red B04 \
  --vector wetlands.shp --id-field ID \
  --composite yearly --reduce median --inclusion center \
  --format csv --out ndvi_by_wetland.csv

# ...or over a cube already materialized to GeoZarr (offline)
cargo run -p datacube-cli --features stac -- zonal --cube cube.zarr \
  --vector wetlands.geojson --id-field ID --reduce mean --format json
```

- **Reading** (`.shp` / `.geojson`) and **rasterization** reuse SurtGIS's
  vector stack; polygons are **reprojected to the cube CRS** (pure-Rust
  WGS84↔UTM / UTM↔UTM — the geometry is reprojected, never the raster).
- **`--inclusion`** picks how a polygon selects pixels — `center` (pixel
  centre inside, the rasterio default), `all-touched` (any cell the polygon
  touches), or `area-fraction` (area-weighted coverage). This is semantically
  significant, so it is explicit.
- **`--reduce`** is NaN-aware: `mean median min max std sum count
  fraction-above`; `n_valid`/`n_total` report coverage for QC.
- **`--composite`** bins time with the same calendar windows as `composite`.
- **`--format`** is `csv` or `json`; `parquet` is available when the CLI is
  built with `--features parquet` (`ZonalTable::write_parquet`, a lightweight
  Snappy Parquet writer with no Arrow dependency).
- In Python: `cube.zonal("wetlands.shp", id_field="ID", reducer="median",
  inclusion="center", window="yearly")` returns a dict of columns ready for
  `pandas.DataFrame`.

## Cloud/quality masking

`--mask` decodes the per-scene quality band before grid alignment, so rejected
pixels never bleed into their neighbours:

- **`scl`** — Sentinel-2 L2A scene classification; keep the clear classes with
  `--mask-keep` (default `4,5,6,7,11`).
- **`qa-pixel`** — Landsat Collection-2 Level-2 `QA_PIXEL` bitmask; reject bits
  with `--qa-reject-bits` (default `fill,dilated-cloud,cirrus,cloud,cloud-shadow`)
  and optionally a cloud-confidence floor with `--qa-min-confidence`.
- **`auto`** — SCL for Sentinel-2 collections, QA_PIXEL for Landsat.

Band asset keys differ between sensors: Sentinel-2 uses `B04`/`B08` (red/NIR),
Landsat C2 L2 uses `SR_B4`/`SR_B5`. The band→role mapping is the caller's
(`--red`/`--nir`/… or the `--assets` order); set it to match the collection.

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

`scripts/validate_zonal.py` cross-checks zonal aggregation against
`rasterio.features.geometry_mask` (Center / AllTouched pixel sets, exact) and
`shapely` intersection areas (AreaFraction weights, `1e-6` relative — geo's
`BooleanOps` snaps to a fixed-precision integer grid), across every reducer.
Needs the `stac`-enabled binding plus `rasterio`/`shapely` in the venv.

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
- [x] Zonal aggregation by polygon (Shapefile/GeoJSON → tidy per-polygon
  series; centre / all-touched / area-fraction inclusion; NaN-aware reducers;
  CLI + library + Python)
- [x] Landsat Collection-2 `QA_PIXEL` bitmask masking, alongside Sentinel-2
  SCL (auto-selected by collection)

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
