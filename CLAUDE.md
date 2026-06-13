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
      (`scripts/validate_pymannkendall.py`).
- [x] Regresión armónica (estacionalidad/fenología). Validada contra
      numpy.linalg.lstsq (tol 1e-9).
- [x] (v0.2) Break-point estilo BFAST (OLS-CUSUM + binary segmentation,
      `stats::detect_breaks`); compositing temporal (`Cube::composite`);
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
- v0.2 añade en core: `stats::detect_breaks` (BFAST-style OLS-CUSUM, p-value
  Brownian bridge = kstwobign.sf, binary segmentation; modelo de segmento
  trend+armónicos reusa `solve_symmetric`) y `temporal.rs`
  (`Cube::composite` SameTime/Period × median/mean/min/max,
  `Cube::gapfill_linear` con max_gap y sin extrapolar bordes).

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
- Limitaciones: escenas con EPSG distinto al de referencia se saltan; mosaico
  cross-UTM-zone pendiente.
- CLI: `datacube stack` tras `--features stac` (CLI default sigue standalone).

## Validación (venv obligatorio para statsmodels)
- `.venv-validate/` (gitignored): numpy/scipy/pymannkendall/statsmodels.
  statsmodels del sistema roto por pandas 3.0 (`deprecate_kwarg`); el venv usa
  statsmodels 0.14.6 que sí importa. Correr con
  `.venv-validate/bin/python scripts/validate_pymannkendall.py` → 103/103.
- OJO numpy 2.x: `repr(np.float64)` da "np.float64(0.0)"; el script castea a
  float() antes de escribir CSV.

## Próximos pasos al retomar
1. Mapa de breaks por píxel sobre un cubo (hoy detect_breaks es por-serie + CLI).
2. Benchmarks criterion para `par_map_series` sobre cubos grandes.
3. Mosaico cross-UTM-zone; bindings PyO3 / WASM demo.
