# datacube-rs — Data cubes temporales de teledetección en Rust ("gdalcubes moderno")

> **Estado:** EN DESARROLLO (workspace v0.1 con core + CLI funcionales). Creado 2026-06-10.
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
- [ ] Apilado temporal desde STAC/COG (reusa el cliente STAC de SurtGIS).
- [x] Tendencia por píxel: regresión lineal, Theil-Sen + Mann-Kendall.
      Validado contra pyMannKendall/scipy: 85/85 checks, tol 1e-9
      (`scripts/validate_pymannkendall.py`).
- [x] Regresión armónica (estacionalidad/fenología). Validada contra
      numpy.linalg.lstsq (97/97 checks totales, tol 1e-9).
- [ ] (v0.2) Break-point estilo BFAST; compositing temporal; gap-filling.

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
- 30 tests + 4 doctests; clippy -D warnings limpio.
- NaN = nodata, filtrado pairwise; Theil-Sen/OLS usan coordenadas t reales
  (muestreo irregular por nubes OK — diverge a propósito de sens_slope).

## Próximos pasos al retomar
1. Conectar al STAC de SurtGIS para armar un cubo Sentinel-2 real (I/O en
   crate aparte, `datacube-io`; core queda sin I/O).
2. Benchmarks criterion para `par_map_series` sobre cubos grandes.
3. v0.2: break-points BFAST (la armónica ya da el modelo de estación),
   compositing temporal, gap-filling.
