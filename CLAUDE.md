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

## Estado (2026-07-03) — v0.12
**6 targets**: core (stats+temporal+bandmath+GeoRef+**pipeline chunked**), io
(STAC/COG+cross-zone+mask SCL+GridSpec+lecturas paralelas+escritura directa
por-slot), CLI (pipeline post-stack por chunk + **volcado a GeoZarr**), PyO3,
WASM, zarr (comprimido zstd+f32 opcional, lectura/escritura por chunks).
Validación estadística 103/103 a 1e-9; band-math vs numpy 1e-12; pytest
16/16; zarr 8 tests + interop Python real; core 68 unit + 9 doctests, io 23.
cargo test --workspace verde. Stacking ~6× más rápido en escenarios de red
reales (M3) sin el pico de memoria 2× de antes (H5 paso 1); pipeline
post-stack (composite/gapfill/index/trend/breaks) acotado a ~1x + un chunk
en vez de ~4x (H5 paso 3). AUDIT.md grupo 3 queda completamente cerrado.

## `datacube stack --zarr-output` (v0.12.0, 2026-07-03)
- Motivado por el paper: el caso de estudio (case_study.py) usaba odc-stac +
  máscara/NDVI manuales en Python porque la ingesta no exponía el cubo
  procesado; con `--index`/`--mask-scl` ya nativos (v0.4/v0.6) faltaba una
  forma de sacar el cubo resultante (post mask/composite/gapfill/index) de
  vuelta a Python sin materializarlo completo en memoria.
- `ChunkPipeline::transform(&self, cube: &Cube) -> Result<Cube, CubeError>`
  (`datacube-core::pipeline`): extrae el prefijo composite→gapfill→index de
  `run_on` como método público reusable — `run_on` ahora es
  `self.transform(cube)?` + banda + stats, mismo comportamiento (tests
  existentes sin cambios, +1 test nuevo que compara `transform` contra
  encadenar `gapfill_linear`+`indices::ndvi` a mano).
- CLI: nuevo flag `--zarr-output <path>` en `datacube stack`. Cuando está
  seteado, `write_processed_zarr` (`stack_cmd.rs`) crea un
  `datacube_zarr::ZarrCubeWriter` con las dims/bands/time que
  `ChunkPipeline::output_time`/el label del índice ya dan analíticamente, e
  itera `cube.chunks(chunk_size, chunk_size)` escribiendo cada tile
  procesado (`pipeline.transform(&sub)`) — es la contraparte de escritura de
  `Cube::run_chunked`, ya que el paso de stats por diseño nunca materializa
  un cubo procesado completo (H5 paso 3). `datacube-zarr` pasa a ser
  dependencia opcional de `datacube-cli` bajo el feature `stac`.
- Verificado e2e contra Planetary Computer: `--mask-scl --grid-epsg
  --grid-res --index ndvi --zarr-output cube.zarr` produce un store leído
  correctamente con `zarr` desde Python (shape/attrs/geotransform/EPSG
  correctos, NDVI en rango esperado, fracción finita ~83% consistente con
  la máscara).
- Workspace bump 0.12.0.

## Paper: motor real reflejado + caso de estudio nativo (2026-07-03)
- `papers/draft/datacube-rs.tex` (target C&G) estaba desactualizado desde
  2026-06-29 (Limitations decía "NDVI planned"/"GeoZarr planned but not
  implemented", ambos hechos desde v0.4/v0.5). Reescritas Abstract/Highlights/
  Limitations/Outlook y agregadas 4 subsecciones nuevas a §4 (Spectral
  indices, Georeferenced results across transforms, GeoZarr backing store,
  Bounded-memory chunked analysis) + actualizada Cube ingestion (máscara SCL,
  GridSpec, M3). Encuadre: máscara/índices se presentan como "cierre de
  brecha ARD" vs gdalcubes/FORCE, no como diferenciador nuevo (evita
  sobreventa ante reviewer).
- `scripts/case_study.py` reescrito: ya no usa odc-stac/xarray para
  ingesta+máscara+NDVI (motivo original del hedge en el paper) — corre
  `datacube stack` dos veces (período completo con `--mask-scl --mask-keep
  2,4,5,6,7 --grid-epsg 32718 --grid-bbox <fijo> --composite same-time
  --index ndvi --breaks-output --first-break-output --zarr-output`; ventana
  post-incendio con `--stat theil-sen --output` para la recuperación, mismo
  grid vía `--grid-bbox` fijo → alineación garantizada entre ambas corridas)
  y lee el Zarr/GeoTIFFs resultantes solo para lo que el motor no expone
  como stat (magnitud de caída NDVI en el quiebre, series de 2 píxeles para
  el panel a). `--limit 500` obligatorio (default 100 trunca un archivo de
  19 meses — encontrado al regenerar: el primer intento con default dio un
  cubo empezando en 2023.38 en vez de 2022.69, silenciosamente cortado).
  Resultado casi idéntico al original (Python manual): 28% píxeles con
  quiebre (antes 31%), mediana 2023.09 (igual), 94% ene-mar 2023 (igual),
  caída NDVI -0.32 (antes -0.31), recuperación +0.062 NDVI/yr (antes +0.06).
  Fig. 4 y prosa de `sec:casestudy` actualizadas con los números nuevos;
  quitado el hedge "reproducible independently of the optional ingestion
  crate" — la ingesta ahora es 100% nativa.
- Nueva tabla en Validation/Performance (`tab:memory`): RSS medido
  (`/usr/bin/time -v`) del mismo run real (Sentinel-2 1465×1418px, 24
  escenas, 3 bandas, 10m nativo, máscara+NDVI+Theil-Sen) a `--chunk-size
  4000` (1 chunk, todo el cubo) vs `--chunk-size 64` (multi-chunk): 4.10GB
  vs 1.60GB, ~2.6x — evidencia cuantitativa nueva para el pipeline chunked
  (H5 paso 3), con la limitación honesta de que el piso restante es la
  ingesta STAC/COG no chunked. También agregada mención del ~6x de M3 en
  Performance (ya medido en v0.9, no re-corrido).
- Sin cambios de código en el motor (solo `.tex`/`.py`); no ameritó bump de
  versión del workspace.

## AUDIT grupo 3 — H5 paso 3: pipeline chunked post-stack (v0.11.0, 2026-07-03)
- Al retomar el único ítem abierto del grupo 3 (grafo lazy `stack→mask→
  composite→index→trend` por chunk) se descubrió que el techo de RAM real
  hoy no es el que describía el planteamiento original de H5: `stack()`
  (desde v0.10) ya escribe cada escena directo en su slot final, así que el
  ingest STAC/COG sigue siendo O(1x cubo) — sin cambios con este paso, y
  bajarlo requeriría lecturas COG en ventana por chunk con `GridSpec`
  obligatorio (fuera de alcance, riesgo alto: toca M3/cross-zone/firma SAS).
  El problema real y sin resolver estaba **después** del stack:
  `stack_cmd.rs::run()` encadenaba `composite→gapfill_linear→compute_index`,
  cada uno asignando un `Array4` completo nuevo — pico real ~4x el cubo.
- Nuevo módulo `datacube_core::pipeline`: `ChunkPipeline { composite,
  gapfill: Option<GapfillSpec>, index: Option<IndexSpec>, stat: StatSpec }`
  + `Cube::run_chunked(chunk_y, chunk_x, &pipeline) -> impl Iterator<Item =
  Result<ChunkResult, CubeError>>` corre la cadena completa de forma
  independiente en cada `CubeChunk` de `Cube::chunks()` (secuencial entre
  chunks; el paralelismo real sigue viniendo de `par_map_series`/
  `combine_bands` dentro de cada chunk). `StatSpec` calcula trend Y breaks
  del mismo cubo procesado por chunk sin recomputar composite/gapfill/index
  dos veces. `ChunkPipeline::output_time` (+ `temporal::composite_time_axis`,
  extraída de `Cube::composite`) da el eje de tiempo post-composite sin
  tocar datos de píxel — el reporte JSON del CLI ya no materializa el cubo
  cuando no se pide ningún `--output`.
  Todas las etapas (composite, gapfill_linear, band-math, par_map_series/
  stats) son puramente por-píxel: trocear por `(y,x)` no cambia ningún
  resultado numérico. Verificado con test de invarianza chunk-vs-cubo-
  completo (core) y e2e vs Planetary Computer comparando `--chunk-size`
  grande vs chico: slope/pvalue idénticos byte a byte, breaks/first
  idénticos (incluida la máscara NaN).
- CLI `datacube stack`: reemplazó por completo el pipeline imperativo
  (`compute_index`/`trend_maps`/`break_maps` borrados, movidos a core) por
  la cadena chunked; nuevo flag `--chunk-size` (default 256, igual al chunk
  espacial de Zarr). Cambio de comportamiento documentado: si no se pide
  ningún `--output`/`--pvalue-output`/`--breaks-output`/`--first-break-
  output`, errores de `--index`/`--nir`/etc. ya no se validan eagerly (antes
  se corría el índice igual con o sin output; caso de uso degenerado, se
  documenta en vez de agregar validación extra para un camino que nadie usa).
- **Limitación conocida, a propósito**: esto NO baja el techo de RAM de
  `stack()` — el ingest STAC/COG sigue siendo O(1x cubo). Bajarlo es un
  rediseño de mayor riesgo, descartado en esta sesión (ver arriba).
- Verificación: 6 tests nuevos en `pipeline.rs` (core 61→67), clippy limpio,
  cargo fmt aplicado, e2e Planetary Computer (Santiago, ene-abr 2024,
  `--mask-scl --grid-epsg --grid-res --composite monthly --index ndvi
  --breaks-output`) con `--chunk-size 100000` (1 chunk) vs `--chunk-size 17`
  (multi-chunk): mismo reporte JSON, GeoTIFFs idénticos.
- Workspace bump 0.11.0.

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

## AUDIT grupo 3 — M3 lecturas STAC paralelas (v0.9, 2026-07-02)
- `stack()` en dos fases. **Bootstrap** (solo si `cfg.grid` es `None`): lee
  ítems uno a uno hasta que el primero exitoso fija la grilla de referencia
  (`bootstrapped` cuenta cuántos ítems ya se consumieron, éxito o fracaso).
  **Paralela**: el resto (`items[bootstrapped..]`) — o TODOS si `GridSpec`
  ya fijó la grilla de antemano, caso común dado H3 — se filtra con
  `plan_scene(item, cfg, ref_epsg) -> Result<SliceMeta, String>` (chequeo
  barato sin red: datetime válido, cloud cover, cross-zone-off EPSG
  mismatch; antes vivía inline en el loop, ahora factorizado y testeado
  solo con 4 tests de unidad construyendo `StacItem` a mano) y se lee con
  `rayon::ThreadPoolBuilder::new().num_threads(cfg.concurrency).build()`
  — pool dedicado y acotado, separado del pool global de Rayon (que sigue
  usándose para cómputo por-píxel en core). `candidates.into_par_iter()
  .collect()` sobre un `Vec` preserva el orden de entrada (ya ordenado por
  tiempo), así el eje temporal del cubo sale ordenado sin trabajo extra.
- `StacClientBlocking` resultó `Sync` sin ningún cambio: runtime tokio
  compartido `&'static Runtime` + `Mutex<HashMap<...>>` interno para el
  cache SAS — compilo a la primera compartiendo `&client` entre closures
  paralelas.
- `StackConfig::concurrency` (default 8, validado > 0) + CLI `--concurrency`.
  `rayon` pasó a dependencia directa de `datacube-io` (antes solo
  transitiva vía datacube-core).
- Verificado e2e vs Planetary Computer (49 ítems, Santiago ene-abr 2024):
  **4m41s → 47s (~6×)** con `--concurrency 8` vs `1`; mismo orden de
  escenas, mismos 5 skips, mismo `time_range`. Repetido con `GridSpec` fijo
  (`--mask-scl --grid-epsg --grid-res --composite monthly --index ndvi`):
  toda la lectura entra a la fase paralela desde el ítem 0 (sin bootstrap),
  mismas dims/EPSG que el run pre-M3.
- Tests: io 20 (antes 15) — `plan_scene` (4 nuevos: item limpio, datetime
  faltante/malformado, cloud cover, cross-zone solo cuando mosaicking está
  off) + `concurrency` default/validación. `StacItem`/`StacItemProperties`
  se construyen a mano en tests (no hay `StacItem::new()` en surtgis-cloud;
  `item.epsg()` lee `properties.extra["proj:epsg"]`, un `HashMap` capturado
  por `#[serde(flatten)]`).
- Workspace bump 0.9.0.

## AUDIT grupo 3 — H5 paso 1: escritura directa por-slot en stack() (v0.10, 2026-07-03)
- `stack()` ya no acumula `scenes: Vec<(SliceMeta, Vec<Raster<f64>>)>` para
  todo el lote antes de copiarlo al `Array4` del cubo (el pico de memoria
  ~2× que señalaba H5). Cada candidato (post `plan_scene`, ver M3) recibe un
  slot de tiempo pre-asignado en un `Array4` dimensionado a una cota
  superior (`bootstrap_count + candidates.len()` — solo un fallo de I/O real
  dentro de `read_scene` la reduce, ya que todo lo demás fue filtrado antes
  sin red) y escribe ahí apenas se lee.
- Los slots se reparten entre las tareas paralelas de M3 vía
  `data.axis_iter_mut(Axis(3)).skip(bootstrap_count).collect::<Vec<_>>()`
  zippeado con `candidates` — cada `ArrayViewMut` es una sub-vista disjunta
  del mismo `Array4`, así que N hilos escriben simultáneamente **sin**
  `unsafe` ni sincronización (compila porque `ArrayViewMut<f64,_>: Send`;
  no hizo falta el feature "rayon" de ndarray, basta el `Vec<T: Send>` +
  `IntoParallelIterator` de rayon).
  `outcomes` sigue devolviendo `(SliceMeta, Result<(), StackError>)` por
  candidato en el orden original (temporal), igual que antes de este cambio.
- Caso común (0 fallos — lo esperado, dado que `plan_scene` ya descartó todo
  lo demás por adelantado): `compact_time_axis` es un no-op literal, **el
  mismo buffer se reusa sin copiar nada** (test dedicado compara el puntero
  del buffer antes/después). Caso con fallos reales de I/O: una única copia
  de compactación al tamaño *final* (nunca al tamaño del lote completo) vía
  la función pura `compact_time_axis(data, slot_ok) -> Array4<f64>`
  (extraída para poder testearla sin red: construye un `Array4` sintético y
  un `&[bool]`, sin `StacItem` ni `read_scene` de por medio).
- Verificación: io 23 tests (+3: no-op sin copia, compactación preserva
  orden, todos fallidos → 0 tiempos), `cargo test --workspace` verde,
  clippy limpio. E2E vs Planetary Computer (mismo escenario de M3): cubo
  idéntico (dims/orden de escenas/skips/`time_range`) y mismo wall-clock
  (~48s) — la paralelización de M3 no se vio afectada por el cambio.
- Workspace bump 0.10.0.

## GeoZarr-CF pleno (v0.13.0, 2026-07-09)
- Primer opcional del roadmap post-AUDIT resuelto: `datacube-zarr` ya no solo
  guarda `bands`/`time`/`epsg`/`geotransform` como atributos planos del array
  `/cube` — ahora escribe las tres piezas que le faltaban para GeoZarr-CF
  real (ver nota en el doc del crate, antes decía "planned refinement"):
  1. **Variables-coordenada separadas**: `/y`, `/x` (cuando el transform es
     axis-aligned, `c==0 && e==0` en la convención GDAL — chequeo explícito,
     se saltea sin fallar si algún día hay rotación) y `/time` siempre, cada
     una como array Zarr propio de 1-D con `dimension_names` igual al nombre
     de la dimensión que representa — la convención que el backend Zarr V3
     de xarray usa para reconocer automáticamente una coordenada sin flags
     extra. Atributos CF (`standard_name`/`axis`/`units`) por eje;
     `time` documenta honestamente que sus unidades son "year" fraccional,
     no un `"days since ..."` CF-estándar (el crate evita `chrono` a
     propósito, ver nota histórica de `datacube-io`).
  2. **`grid_mapping`/CRS WKT real**: `/spatial_ref` (variable CF
     grid-mapping, convención rioxarray/GDAL — un array dummy de 1 elemento,
     solo importan sus atributos) con `crs_wkt`/`spatial_ref`/
     `GeoTransform`/`grid_mapping_name`. El WKT sale de la tabla estática
     **`crs-definitions`** (crate pure-Rust, `no_std`, ~5000 EPSG codes con
     WKT/PROJ4 reales tomados del registro EPSG — sin libproj/GDAL, mantiene
     el crate offline-testable) en vez de reinventar un generador de WKT a
     mano; `/cube` gana el atributo `grid_mapping: "spatial_ref"` apuntando
     a la variable (convención CF que xarray/rioxarray usan para ubicar el
     CRS de una data variable).
  3. **Atributos CF por-banda** (alcance modesto, honesto): `band_long_names`
     en `/cube`, un `long_name` legible para los seis índices espectrales
     que el motor sabe computar nativamente (ndvi/ndwi/nbr/ndbi/evi/savi);
     cualquier otro nombre de banda (asset key crudo, etc.) cae de vuelta a
     sí mismo — no se inventan `standard_name` CF que no existen para
     índices espectrales.
- Todo esto es **aditivo sobre el `create_array` compartido** por
  `write_zarr_with_options` y `ZarrCubeWriter::create` — ni `read_zarr` ni
  `read_zarr_chunked` cambiaron (siguen leyendo solo los atributos planos de
  `/cube`), así que el roundtrip Rust↔Rust y los 8 tests previos pasan sin
  tocarlos. CLI (`--zarr-output`) hereda el enriquecimiento gratis, sin
  cambios propios, porque solo llama a `ZarrCubeWriter::create`.
- Verificación real (no solo tests unitarios): `scripts/zarr_interop.py`
  ahora además abre `/y`/`/x`/`/time`/`spatial_ref` con `zarr` 3.2.1, parsea
  `crs_wkt` con **pyproj** (`CRS.from_wkt(...).to_epsg() == 32719`, nombre
  "WGS 84 / UTM zone 19S" correcto) y abre el store completo con
  **`xr.open_zarr()`** — xarray reconoce `y`/`x`/`time` como coordenadas
  automáticamente sin ningún hint de decodificación manual (el pago real de
  las variables-coordenada separadas). Encontrado en el camino: `zarrs`
  exige `dimension_names` en *todo* array del grupo para que el backend
  Zarr V3 de xarray no falle con `KeyError` al abrir — `/spatial_ref`
  necesitó su propia dimensión dummy (`dimension_names: ["spatial_ref"]`)
  aunque su dato no signifique nada.
- Tests: zarr 8→14 (+6: coordenadas coinciden con centros de píxel y
  declaran ejes CF, grid_mapping trae WKT/GeoTransform correctos,
  EPSG geográfico → `latitude_longitude` vs UTM → `transverse_mercator`,
  `band_long_names` rellena conocidos y cae a sí mismo en desconocidos,
  transform rotado saltea `/y`/`/x` sin romper `/time`/`spatial_ref`, sin
  georef saltea `spatial_ref` pero igual escribe `/time`). `cargo test
  --workspace` y `clippy -D warnings` verdes.
- **Nota de sesión, no de código**: agregar `crs-definitions` disparó un
  re-resuelto completo de `Cargo.lock` que además — sin relación con el
  cambio real — reasignó el edge `numpy → ndarray` de `0.16.1` a `0.17.2`
  (ambas versiones ya convivían en el grafo por `zarrs`; el rango de
  `numpy` 0.29 es `>=0.15,<=0.17` así que las dos satisfacen). Eso rompía
  `datacube-python` (`into_pyarray` no encontrado, porque `datacube-core`
  sigue fijo en ndarray 0.16 vía `workspace.dependencies`) en
  `cargo test --workspace`, aunque compilaba bien aislado
  (`cargo test -p datacube-python`). Fix: edición manual del lockfile para
  devolver ese edge a `0.16.1` (confirmado estable con `cargo build/test
  --workspace --locked`, no se revirtió solo). Vale la pena recordarlo si
  vuelve a pasar al tocar `datacube-zarr`/`datacube-python` en la misma
  sesión — revisar `git diff Cargo.lock` completo, no asumir que solo
  cambió lo que uno tocó.
- Workspace bump 0.13.0.

## Próximos pasos al retomar
1. Paper (C&G): draft con la pasada de estilo de `/paper-style audit`
   commiteada (2026-07-05, prosa más corta/menos run-ons en Abstract/§4/
   Performance/case study, sin cambios de contenido/números). Pendiente:
   revisión final de estilo/longitud antes de someter (ver `/paper-style` y
   `/paper-review-computers-geosciences` para una pasada de calibración) —
   y decidir si vale la pena mencionar GeoZarr-CF en §4.4 (hoy dice
   "planned refinement", ya no es cierto tras v0.13.0).
2. Pendiente Zenodo DOI (gated en ORCID).
3. AUDIT grupo 3 queda **completamente cerrado** (H5 paso 3 resuelto en
   v0.11.0, alcance revisado — ver sección arriba). El techo de RAM de la
   ingesta STAC/COG en sí (`stack()`, O(1x cubo)) queda documentado como
   limitación conocida, a propósito: bajarlo exigiría lecturas COG en
   ventana por chunk con `GridSpec` obligatorio, un rediseño de mayor
   riesgo que toca M3/cross-zone/firma SAS — no se justificó en esta sesión.
4. Bug externo detectado en sesión anterior: paginación de `search_all` en
   surtgis-cloud repite items en Earth Search (dedup por id ya puesto como
   guard en `stack()`, pero el fix real es en surtgis).
5. Opcionales post-AUDIT, 1 de 4 resuelto: **GeoZarr-CF pleno ✓ (v0.13.0)**.
   Quedan: object-store (S3/HTTP) vía zarrs async; exponer datacube-io
   (stack STAC) a Python; sharding Zarr para object store (M2.c — solo
   relevante una vez exista el backend object-store, no antes). Si algún
   día se quiere bajar el techo de RAM del ingest STAC/COG, ver punto 3.
