# datacube-rs — Data cubes temporales de teledetección en Rust ("gdalcubes moderno")

> **Estado:** EN DESARROLLO (v0.2: core+io+CLI, breaks/compositing/gapfill). Creado 2026-06-10.
> Familia de motores Rust del autor: SurtGIS, Hydroflux, Smelt, Anvil, Cantus, Criterium.
> Doc madre: `~/proyectos/ideas-motores-rust.md` (idea C1; también extensión SurtGIS #1).

## Qué es
Motor para apilar y analizar series temporales de rásters (Sentinel/Landsat):
tendencias, fenología y detección de quiebres, con streaming.

## El gap que llena
SurtGIS es **mono-temporal**. El análisis temporal de data cubes vive en
**gdalcubes** (R/C++), **stars**, **BFAST**. No hay motor Rust single-binary que
lo haga aprovechando STAC.

## Alcance MVP (v0.1)
- [x] Apilado temporal desde STAC/COG (reusa el cliente STAC de SurtGIS).
      `datacube-io::stack()` probado end-to-end contra Planetary Computer
      (Sentinel-2 Santiago, UTM 19S, filtro de nubes, GeoTIFF de salida).
- [x] Tendencia por píxel: regresión lineal, Theil-Sen + Mann-Kendall.
      Validado contra pyMannKendall/scipy: 85/85 checks, tol 1e-9
      (`scripts/validate_stats.py`).
- [x] Regresión armónica (estacionalidad/fenología). Validada contra
      numpy.linalg.lstsq (tol 1e-9).
- [x] (v0.2) Detección de quiebres OLS-CUSUM + binary segmentation
      (`stats::detect_breaks`, inspirado en BFAST pero NO el test de BFAST:
      BFAST usa OLS-MOSUM + Bai-Perron; validado vs statsmodels, no vs R bfast);
      compositing temporal (`Cube::composite`);
      gap-filling lineal (`Cube::gapfill_linear`); scale/offset en stack.
      Validación total 103/103 checks tol 1e-9, breaks vs
      statsmodels.breaks_cusumolsresid.

## Arquitectura tentativa
- `datacube-core`: modelo de cubo (x,y,t,band), iteradores streaming por píxel/chunk.
- Targets: native (Rayon) + Python (PyO3) + CLI; WASM como demo de series.
- Apóyate en el STAC composite + COG reader ya existentes en SurtGIS.

## Validación / paridad numérica
Cross-check Mann-Kendall/Theil-Sen contra **pyMannKendall**; armónicos contra
implementaciones de referencia.

## Venue objetivo
**Computers & Geosciences** o **Environmental Modelling & Software**.

## Conexiones con tu ecosistema
- **SurtGIS**: reusa STAC/COG; podría empezar como `surtgis temporal` y graduarse.
- Casos: NDVI multianual, sequía, deforestación (líneas RS/forestal).

## Refinamiento SOTA (2026-06-10)
Cloud-native ARD es la dirección dominante: usar **GeoZarr** como backing store
del cubo (Sentinel/Landsat ya lo adoptan) y **STAC-Zarr** para indexar; salida
opcional **GeoParquet** (queryable con DuckDB). Integrar **GeoRust** (geozero,
proj) en vez de reinventar I/O. Diferenciador: cubo Rust nativo sobre GeoZarr.

## Estado del código (2026-06-10)
- Workspace edition 2024: `crates/datacube-core` (modelo `Cube` (band,y,x,t)
  con eje temporal contiguo, `iter_series`/`par_map_series` (Rayon)/`chunks`,
  stats: OLS + Theil-Sen + Mann-Kendall tie-corrected estilo pyMannKendall,
  funciones especiales propias con libm) + `crates/datacube-cli`
  (`datacube trend serie.csv` → JSON).
- NaN = nodata, filtrado pairwise; Theil-Sen/OLS usan coordenadas t reales
  (muestreo irregular por nubes OK — diverge a propósito de sens_slope).
- v0.2 añade en core: `stats::detect_breaks` (OLS-CUSUM, p-value Brownian
  bridge = kstwobign.sf, binary segmentation; inspirado en BFAST, ver nota
  de fidelidad arriba) y `temporal.rs`
  (`Cube::composite` SameTime/Period × median/mean/min/max,
  `Cube::gapfill_linear` con max_gap y sin extrapolar bordes).
- El modelo trend+armónicos (lstsq por ecuaciones normales + solver con
  pivoteo) vive en `stats/lstsq.rs` (`HarmonicModel::fit`/`predict`),
  compartido por `harmonic_regression` y el modelo de segmento de breaks.

## datacube-io (2026-06-11)
- Depende de surtgis-core/surtgis-cloud por **path** (`../surtgis` sibling
  checkout obligatorio). API blocking (feature `native` de surtgis-cloud).
- `stack(StackConfig)` → `StackedCube { cube, slices, skipped, transform, epsg }`:
  busca STAC (pc/es/URL), filtra nubes, firma SAS de PC, lee COG por bbox
  (overview opcional), alinea con `resample_to_grid`, nodata→NaN, tiempo en
  años fraccionales (`fractional_year`, sin chrono).
- v0.2: `StackConfig::scaling(scale, offset)` aplica transform lineal post-
  máscara (S2 L2A: 1e-4, -0.1 baseline ≥04.00). CLI stack acepta --scale
  --offset --composite (same-time|monthly) --composite-method --gapfill.
- v0.3: mosaico **cross-UTM-zone**. `StackConfig::cross_zone_mosaic` (default
  on): escenas en otra zona UTM se reproyectan a la zona de referencia con
  `reproject::reproject_raster_utm` (UTM↔UTM bilineal de surtgis-cloud) antes
  de `resample_to_grid`. Non-UTM → StackError::Reproject → skip con razón.
  CLI: `--no-cross-zone` para volver al comportamiento anterior. Probado en
  frontera zona 18/19 (-72° Chile): 15→31 escenas (16 tiles T18 reproyectadas
  a EPSG 32719).
- CLI: `datacube stack` tras `--features stac` (CLI default sigue standalone).

## datacube-python (PyO3, 2026-06-16)
- Crate `crates/datacube-python`, módulo `datacube_rs`. pyo3 0.29 + numpy 0.29
  (abi3-py39). crate-type `["cdylib","rlib"]` + feature `extension-module`
  (off para `cargo test --workspace`, on para maturin) → no rompe los tests.
- Expone: `linear_trend`, `theil_sen`, `mann_kendall`, `harmonic_regression`,
  `detect_breaks` (toman np.ndarray 1-D) y clase `Cube` (data 4-D + time +
  bands) con `.dims/.bands/.time/.to_numpy()`, `.trend_map(band, method)` →
  (slope, pvalue) 2-D, `.composite(window, method)`, `.gapfill(max_gap)`.
- par_map_series corre Rayon a través del binding (los hilos no tocan objetos
  Python → GIL no estorba); trend_map 256x256x60 ~780ms desde Python.
- Build: `VIRTUAL_ENV=.venv-validate maturin develop --release` desde el crate.
  Tests: `.venv-validate/bin/python -m pytest crates/datacube-python/tests`
  (10 tests). datacube-io NO se expone aún (pulls surtgis/red).

## Validación (venv obligatorio para statsmodels)
- `.venv-validate/` (gitignored): numpy/scipy/pymannkendall/statsmodels.
  statsmodels del sistema roto por pandas 3.0 (`deprecate_kwarg`); el venv usa
  statsmodels 0.14.6 que sí importa. Correr con
  `.venv-validate/bin/python scripts/validate_stats.py` → 103/103.
- OJO numpy 2.x: `repr(np.float64)` da "np.float64(0.0)"; el script castea a
  float() antes de escribir CSV.

## Mapas de breaks por píxel (2026-06-13)
- `datacube stack --breaks-output N.tif --first-break-output T.tif`
  (`--break-harmonics`, `--break-alpha`): corre `detect_breaks` por píxel vía
  `par_map_series` → GeoTIFF de conteo de breaks y de tiempo del primer break
  (NaN donde hay pocas obs). min_segment se sube a max(12, 2*K+4).
- Verificado end-to-end contra PC (Santiago); con pocos composites el mapa
  queda NaN como corresponde (algoritmo validado aparte 103/103).

## datacube-wasm (2026-06-18)
- Crate `crates/datacube-wasm`, wasm-bindgen 0.2 + serde-wasm-bindgen. Expone
  las stats por-serie (linear_trend, theil_sen, mann_kendall,
  harmonic_regression, detect_breaks) sobre Float64Array → objetos JS.
- crate-type cdylib+rlib; `tests/web.rs` con `#![cfg(target_arch="wasm32")]`
  para no romper `cargo test --workspace` en host. wasm-pack test --node: 3 ok.
- Demo `web/index.html` (canvas vanilla, sin deps): serie NDVI sintética con
  break inyectado + gaps; ajuste armónico + breaks en vivo con sliders.
  Verificada con screenshot headless (break detectado donde cae el nivel).
  Build: `wasm-pack build --target web --out-dir web/pkg` (pkg/ gitignored).

## Estado (2026-06-18) — ROADMAP COMPLETO
5 targets: core (stats+temporal), io (STAC/COG + cross-zone), CLI, PyO3, WASM.
Validación 103/103 a 1e-9. cargo test --workspace 10 suites verdes.

## Próximos pasos al retomar
1. Pensar el paper (venue: Computers & Geosciences o EMS). Material listo:
   paridad numérica documentada, benchmarks, 5 targets, demo web.
2. Opcional: exponer datacube-io (stack STAC) a Python; GeoZarr backing store.
