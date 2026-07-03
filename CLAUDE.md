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

## Band-math / índices espectrales (v0.4, 2026-06-30)
- Workspace bump a **0.4.0** (estaba estancado en 0.1.0 pese a v0.2/v0.3).
- `datacube-core::bandmath` (módulo privado, `pub use indices`): álgebra de
  bandas por celda `(y,x,t)` → cubo de 1 banda en la misma grilla.
  - `Cube::band(name)` lookup → `CubeError::BandNotFound`.
  - `Cube::normalized_difference(a,b,label)` = (a−b)/(a+b); NaN si algún input
    es NaN o el denominador es 0. Workhorse de NDVI/NDWI/NBR/NDBI.
  - `Cube::band_ratio(a,b,label)`, `Cube::combine_bands(label, f)` (primitiva
    general con closure sobre las bandas de la celda; fast-path stack-buffer
    para ≤16 bandas, fallback heap). Paraleliza sobre celdas (Rayon).
  - `indices::{ndvi,ndwi,nbr,ndbi,evi,savi}` (EVI/SAVI vía combine_bands).
  - Layout (band,y,x,t) → cada volumen de banda es contiguo: nd usa dos slices
    elementwise; combine_bands gatherea banda*cells+i.
- `datacube-python`: `Cube.{ndvi,ndwi,nbr,evi,savi,normalized_difference,
  band_index}` (por nombre de banda). 4 tests pytest nuevos validan NDVI/EVI/
  SAVI vs numpy closed-form a 1e-12 + propagación de NaN (14/14 total).
- CLI `datacube stack --index ndvi|ndwi|nbr|ndbi|evi|savi` (feature `stac`):
  computa el índice desde las bandas apiladas tras composite/gapfill y corre
  trend/breaks sobre él. Roles de banda: `--nir/--red/--green/--blue/--swir`
  (defaults S2: B08/B04/B03/B02/B11), `--savi-l` (0.5). El índice colapsa el
  cubo a 1 banda → la selección de banda del análisis pasa a ser esa.
- Cierra el gap del paper: el NDVI del caso de estudio se calculaba fuera del
  motor (xarray en case_study.py); ahora el cubo lo computa nativo.
- WASM no toca band-math (no expone `Cube`; consistente).

## datacube-zarr — GeoZarr backing store (v0.5, 2026-06-30)
- Crate nuevo `crates/datacube-zarr` (6º target): serialización nativa del cubo
  en **Zarr V3** vía `zarrs 0.23` (FilesystemStore, puro-Rust, sin surtgis →
  testeable offline). Es la dirección cloud-native ARD del roadmap.
- `write_zarr(cube, path, &GeoRef)` / `read_zarr(path) -> (Cube, GeoRef)`:
  array 4-D f64 `(band,y,x,time)` en `/cube` bajo un grupo raíz Zarr V3, con
  `dimension_names=["band","y","x","time"]` y atributos `bands`/`time`/`epsg`/
  `geotransform`. Chunk espacial 256 (bandas y tiempo single-chunk).
  `GeoRef { epsg: Option<u32>, transform: Option<[f64;6]> }`.
- OJO choque de versiones: zarrs usa **ndarray 0.17**, el workspace **0.16**.
  No se pasan tipos ndarray a través del límite → se usa la API por-bytes
  no-deprecada `store_array_subset(&subset, &[f64])` /
  `retrieve_array_subset::<Vec<f64>>(&subset)` y se reconstruye `Array4` con
  nuestra 0.16 (layout std row-major = C-order de zarr, 1:1).
- API zarrs 0.23 (afinada contra el crate): `data_type::float64()`,
  `ArrayBuilder::new(shape, chunk, dtype, fill)` (fill acepta `f64` directo),
  `ArraySubset` en `zarrs::array`, `GroupBuilder::new().build(store,"/")`.
- Tests: roundtrip (multi-chunk y=300, NaN, georef) + error al abrir ausente.
- **Interop probada** (el diferenciador): `examples/write_sample.rs` escribe un
  store y `scripts/zarr_interop.py` lo lee con **zarr 3.2.1 desde Python**
  (dims, dimension_names, atributos, valores); NDVI computado en Rust coincide
  con el recalculado en Python. zarr instalado en `.venv-validate` (xarray ya
  estaba). Correr: `cargo run -p datacube-zarr --example write_sample -- P` +
  `.venv-validate/bin/python scripts/zarr_interop.py P`.
- Falta para GeoZarr-CF pleno (refinamiento): variables-coordenada separadas,
  `grid_mapping`/CRS WKT, atributos CF por-banda. Hoy es cubo-en-Zarr-V3 fiel.

## Estado (2026-07-02) — v0.8
**6 targets**: core (stats+temporal+bandmath+GeoRef), io (STAC/COG+cross-zone
+mask SCL+GridSpec), CLI, PyO3, WASM, zarr (**comprimido zstd+f32 opcional,
lectura/escritura por chunks**). Validación estadística 103/103 a 1e-9;
band-math vs numpy 1e-12; pytest 16/16; zarr 8 tests + interop Python real
(zstd y f32 decodifican transparentemente); core 61 unit + 9 doctests, io 15.
cargo test --workspace verde.

## Auditoría + quick wins (2026-07-02)
- `AUDIT.md` (raíz): auditoría completa del motor — 0 critical, 5 HIGH de
  diseño (composite calendario, máscara SCL, GridSpec, georef en core,
  ejecución por chunks), roadmap priorizado hacia "mejor motor de cubos".
- Quick wins aplicados: `CubeError::{UnsortedTime, DegenerateInput}` +
  `#[non_exhaustive]`; PyO3 libera el GIL (`py.detach`) en trend_map/
  composite/gapfill/índices; CLI stack ya no clona el cubo (destructure);
  copia escena→cubo por slices; items STAC ordenados por año fraccional (no
  string); alpha estricto en (0,1); bump workspace 0.5.0; tags v0.1.0–v0.4.0;
  CI GitHub Actions (fmt+clippy -D warnings+test, clona surtgis sibling);
  `cargo fmt` aplicado a todo el workspace (antes no estaba formateado).
- OJO pyo3 0.29: el método es `py.detach(...)`, NO `allow_threads` (renombrado).

## Paridad ARD — AUDIT grupo 2 (v0.6, 2026-07-02)
- **H1 composite calendario**: `CompositeWindow::{CalendarMonth, CalendarYear}`
  en core (`temporal.rs`): bin por `(año, mes)` recuperado con la inversa
  exacta de `fractional_year` (tabla CUM_DAYS + leap shift, tolerancia 1e-6
  días para bordes exactos de mes/año). `--composite monthly` del CLI y
  `"monthly"` de Python ahora son calendario (antes `Period(1/12)` anclado en
  la 1ª observación → no reproducible); `yearly` nuevo en ambos; `Period`
  queda para ventanas físicas (16 días etc.). `group_by_key` rechaza tiempos
  no finitos.
- **H2 máscara por píxel**: `MaskConfig { asset, keep, resample }` +
  `StackConfig::mask()`; `MaskConfig::scl()` = SCL keep [4,5,6,7,11] nearest.
  El asset de máscara se lee 1×/escena (`read_asset`, helper extraído) y
  `apply_mask` lo resamplea nearest a la grilla de cada banda y pone NaN
  donde la clase no está en keep — ANTES del resample bilineal a la
  referencia (no sangra nubes a vecinos). Nodata/NaN del mask → masked.
  CLI: `--mask-scl` (+ `--mask-asset`, `--mask-keep`).
- **H3 GridSpec**: `GridSpec { epsg, resolution, bbox: Option<[f64;4]> en CRS
  destino, align: Option<f64> }` + `StackConfig::grid()`. Cuando está seteado,
  `reference_from_grid` construye la referencia sintética (Raster vacío con
  transform/CRS del spec, origen = esquina sup-izq, snap outward con align,
  guard 1..=100_000 px/eje) ANTES del loop → grilla independiente del orden
  del catálogo/nubes/red. bbox None → deriva reproyectando el bbox WGS84
  (`reproject_bbox_to_cog`; EPSG no-UTM requiere bbox explícito). CLI:
  `--grid-epsg --grid-res` (juntos) + `--grid-bbox --grid-align`.
- stack() ahora chequea `scenes.is_empty()` (antes el error Empty salía de
  reference=None, que con GridSpec ya no ocurre).
- Verificación: core 55 unit (5 nuevos calendario), io 15 (7 nuevos
  mask/grid), pytest 15/15, clippy limpio, e2e vs PC con
  `--mask-scl --grid-epsg/--grid-res --composite monthly`.

## AUDIT grupo 3 — H4 georef en core (v0.7, 2026-07-02)
- `datacube_core::GeoRef { epsg: Option<u32>, transform: Option<[f64;6]> }`
  (Copy) ahora vive como campo privado `Option<GeoRef>` de `Cube`
  (`cube.rs`): `Cube::with_georef(geo)` (builder), `.georef()` (getter),
  `pub(crate) inherit_georef(&self, from: &Cube)` para que las operaciones
  internas copien el georef del cubo fuente sin exponer el campo.
- Propagación automática en las operaciones que preservan la grilla espacial:
  `composite`/`gapfill_linear` (`temporal.rs`) y `combine_bands`/
  `binary_band` — o sea todos los índices espectrales (`bandmath.rs`).
  Operaciones sin georef en el cubo fuente producen cubos geo-blind (`None`),
  sin cambio de comportamiento para el código previo a v0.7.
- `datacube-io::stack()`: adjunta `GeoRef{epsg, transform: reference
  .transform().to_gdal()}` al cubo devuelto (`StackedCube.cube.georef()`),
  además de seguir exponiendo `StackedCube.transform/epsg` para compat.
- `datacube-zarr`: borró su copia duplicada del struct — `pub use
  datacube_core::GeoRef;`. `read_zarr` además adjunta el GeoRef leído al
  `Cube` devuelto (antes solo iba en la tupla separada).
- PyO3: `Cube.epsg`/`Cube.transform` (getters) + `Cube.with_georef(epsg=,
  transform=)` (copia); `transform` se lee desde Python como lista de 6
  floats. `composite`/`gapfill` ya lo propagan gratis vía core.
- CLI `datacube stack`: dejó de acarrear `transform`/`epsg` manualmente por
  toda la función — `cube_georef()` los lee de `cube.georef()` justo antes
  de escribir el GeoTIFF, después de mask/grid/composite/index. Es la prueba
  de que H4 cierra el problema real: un solo `cube.georef()` reemplaza el
  hilo de dos variables que antes había que mantener sincronizado a mano.
- Verificación: core 61 unit (+6 georef), zarr 2 (dedup sin romper roundtrip
  ni interop Python), pytest 16/16, clippy limpio, e2e vs PC: mismo GeoTIFF
  (origen/pixel size/EPSG) que antes de H4 con `--mask-scl --grid-epsg
  --grid-res --composite monthly --index ndvi`.
- Workspace bump 0.7.0.

## AUDIT grupo 3 — M2 (Zarr zstd+f32) + H5 parcial (chunks sobre GeoZarr) (v0.8, 2026-07-02)
- **M2**: `ZarrOptions { compression_level: Option<i32>, dtype: ZarrDType }`
  (`ZarrDType::{F64,F32}`). Default `Default for ZarrOptions` = zstd nivel 5 +
  f64 — `write_zarr`/`read_zarr` (firmas sin cambios) ahora comprimen por
  defecto de forma transparente (lossless; ningún test de igualdad exacta se
  rompió). `write_zarr_with_options(cube, path, geo, options)` para f32 u
  override de compresión. `read_zarr` detecta el dtype real vía
  `array.data_type()` comparado contra `data_type::float32()`/`float64()` (
  `DataType: PartialEq`, confirmado en el fuente vendored de zarrs 0.23.13) y
  sube f32→f64 al leer (`Vec<f32>` → `.map(f64::from)`).
  API de zarrs usada: `ArrayBuilder::bytes_to_bytes_codecs(vec![Arc::new(
  zarrs::array::codec::ZstdCodec::new(level, checksum))])` — zstd viene en
  el feature-set default de `zarrs` (`Cargo.toml` de datacube-zarr no
  desactiva default-features, no hizo falta tocarlo).
- **H5 (parcial)**: `read_zarr_chunked(path, chunk_y, chunk_x) -> impl
  Iterator<Item = Result<(Cube, ChunkPos), ZarrError>>` — solo la metadata se
  lee al abrir; cada `.next()` trae UN tile espacial (todas las bandas/tiempo,
  `y0..y1 × x0..x1`) vía `ArraySubset::new_with_ranges` con offset no-cero
  (infalible, a diferencia de `new_with_start_shape` que devuelve Result).
  `GeoRef` de cada tile tiene el origen del transform desplazado
  (`shift_georef`: `a' = a + x0·b + y0·c`, `d' = d + x0·e + y0·f`, convención
  GDAL) — el tile es georreferenciable de forma independiente.
  `ZarrCubeWriter::create(path, dims, bands, time, geo, options)` +
  `.write_chunk(view, y0, x0)`: contraparte de escritura, tiles en cualquier
  orden (zarrs re-codifica solo los chunks Zarr que el subset toca — API
  confirmada en el ejemplo oficial `array_write_read.rs` del crate).
  `create_array`/`store_region`/`retrieve_region` factorizados y compartidos
  entre `write_zarr`, `read_zarr`, `read_zarr_chunked` y `ZarrCubeWriter`.
- Verificación: zarr 8 tests (roundtrip f64/f32, compresión on/off idénticas,
  chunked read reensambla == cubo completo, georef desplazado por tile,
  writer por chunks == write_zarr, rechazo chunk_size=0), interop Python real
  (zarr 3.2.1) confirmando que `zarr-python` decodifica zstd y f32
  transparentemente (`arr.metadata.codecs` muestra `ZstdCodec(level=5,
  checksum=False)`; ejemplo sintético comprimió 98304→2578 bytes). Ejemplo
  `write_sample.rs` acepta un segundo arg `f32` para demostrarlo.
- Pendiente de H5: paso 1 (streaming directo en `stack()`, elimina el pico
  2× reteniendo `Vec<Raster>`) y paso 3 (grafo lazy `stack→mask→composite→
  index→trend` evaluado por chunk, v0.7+ en el roadmap original).
- Workspace bump 0.8.0. `approx` añadido como dev-dependency de
  `datacube-zarr` (ya estaba en `workspace.dependencies`).

## Próximos pasos al retomar
1. Paper (C&G/EMS): material listo + band-math + GeoZarr (comprimido+chunks)
   + ARD (mask/grid) + georef unificado. Opciones: §4.4 NDVI in-engine;
   sección GeoZarr/arquitectura citando `read_zarr_chunked` como el
   "cube_view + chunk streaming" de gdalcubes hecho en Rust sobre Zarr V3.
2. Pendiente Zenodo DOI (gated en ORCID).
3. AUDIT grupo 3 restante: M3 (lecturas STAC paralelas en `datacube-io`,
   independiente de zarr); dentro de H5, streaming en `stack()` y grafo lazy.
4. Bug externo detectado en sesión anterior: paginación de `search_all` en
   surtgis-cloud repite items en Earth Search (dedup por id ya puesto como
   guard en `stack()`, pero el fix real es en surtgis).
5. Opcional: GeoZarr-CF pleno (coord vars, grid_mapping); object-store
   (S3/HTTP) vía zarrs async; exponer datacube-io (stack STAC) a Python;
   sharding Zarr para object store (M2.c, no implementado — solo relevante
   para S3/HTTP, no filesystem local).
