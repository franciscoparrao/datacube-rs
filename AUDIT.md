# Auditoría de código — datacube-rs v0.5

**Fecha**: 2026-07-02
**Reviewer**: Claude Code Reviewer (skill /review)
**Scope**: los 6 crates del workspace (`datacube-core`, `datacube-io`, `datacube-cli`,
`datacube-python`, `datacube-wasm`, `datacube-zarr`), ~4.500 líneas de Rust.
**Objetivo**: qué cambiar para perfeccionar el motor y posicionarlo como el mejor
motor de data cubes temporales existente (referentes: gdalcubes, Open Data Cube,
xarray/stackstac, sits).

## Resumen

- Critical: 0 | High: 5 | Medium: 9 | Low/Nit: 7
- **Calidad general: Excelente.** Clippy limpio en los 6 crates (`--all-targets` y
  `--features stac`), cero TODOs/FIXMEs, cero `unwrap()` en código de producción
  (los `expect` existentes documentan invariantes reales garantizadas por
  `Cube::new`), errores tipados con `thiserror` en cada frontera, rustdoc con
  doctests en casi toda la API pública, validación numérica externa (103/103 vs
  statsmodels/pyMannKendall, band-math 1e-12 vs numpy).
- **Deuda técnica: Baja.** Los hallazgos HIGH no son bugs de código sino brechas
  de diseño/funcionalidad entre "motor correcto y validado" (lo que ya es) y
  "mejor motor de cubos" (lo que se quiere ser).

---

## Hallazgos

### [HIGH] H1 — El composite "monthly" no agrupa por meses calendario

- **Archivo**: `crates/datacube-core/src/temporal.rs:156-172` (`group_times`),
  `crates/datacube-cli/src/stack_cmd.rs:126-131` (`CompositeKind::Monthly`)
- **Dimensión**: D6 Naming & clarity / corrección de intención
- **Descripción**: `CompositeWindow::Period(w)` ancla los bins en la **primera
  observación** (`bin = floor((t - t0)/w)`), y el CLI expone `--composite monthly`
  como `Period(1/12)`. Dos problemas: (a) la agrupación depende de la fecha de la
  primera escena sin nubes — dos ejecuciones sobre bboxes o filtros distintos
  agrupan las mismas escenas en composites distintos (no reproducible); (b) bins
  de ancho fijo `1/12` de año fraccional no coinciden con meses calendario (feb 1
  = `+31/365 ≈ 0.0849`, no `1/12 ≈ 0.0833`), así que escenas de fin de mes caen
  en el bin del mes vecino. Está documentado en el rustdoc, pero el nombre
  "monthly" promete calendario y los usuarios de gdalcubes/ODC esperan calendario.
- **Impacto**: composites que mezclan meses adyacentes en cualquier serie
  multianual; resultados de trend/breaks levemente dependientes de la primera
  escena. Para un motor con paper de validación numérica, es la clase de detalle
  que un reviewer o usuario avanzado va a encontrar.
- **Fix propuesto**: agregar una ventana calendario que consuma las fechas reales
  (ya están en `SliceMeta.datetime`):
  ```rust
  pub enum CompositeWindow {
      SameTime,
      Period(f64),
      /// Bins de calendario derivados del tiempo fraccional:
      /// año + mes se recuperan exactamente del eje de años fraccionales.
      CalendarMonth,
      CalendarYear,
  }
  ```
  Para `CalendarMonth` el bin se calcula invirtiendo `fractional_year` (año =
  `t.floor()`, día-del-año = `frac * days_in_year`, mes por tabla `CUM_DAYS`) —
  la inversa exacta de `datacube-io::time`. Mapear `--composite monthly` a esto y
  dejar `Period` para ventanas físicas (16 días, etc.).

### [HIGH] H2 — Sin enmascaramiento de nubes por píxel (SCL/QA)

- **Archivo**: `crates/datacube-io/src/stack.rs` (gap funcional, no bug)
- **Dimensión**: D4 Abstracciones (falta la abstracción central del dominio ARD)
- **Descripción**: el filtro de nubes es por **escena** (`eo:cloud_cover > max`
  → skip). No hay forma de enmascarar píxeles nublados dentro de una escena
  aceptada. gdalcubes (`image_mask`), ODC y sits aplican máscara por píxel con la
  banda de calidad (SCL en S2 L2A, QA_PIXEL en Landsat); es el paso #1 de
  cualquier pipeline ARD real.
- **Impacto**: los NDVI apilados llevan píxeles de nube/sombra como valores
  válidos; el gap-filling los interpola y el trend/breaks los modela. Es la
  brecha funcional más grande frente a los motores de referencia.
- **Fix propuesto**: en `StackConfig`, una máscara declarativa que lee un asset
  extra y lo aplica antes del resample:
  ```rust
  pub struct MaskConfig {
      pub asset: String,          // "SCL"
      pub keep: Vec<u16>,         // [4, 5, 6, 7, 11] (veg, bare, agua, unclass, nieve)
      pub resample: ResampleMethod, // NearestNeighbor (categórica)
  }
  pub fn mask(mut self, m: MaskConfig) -> Self { ... }
  ```
  En `read_scene`: leer el asset de máscara con la misma ventana, resamplear
  nearest a la grilla del asset, y `NaN` donde la clase no está en `keep`.
  CLI: `--mask-scl` con el default S2. Con esto el motor cierra el caso de uso
  end-to-end sin salir de Rust.

### [HIGH] H3 — La grilla del cubo es implícita (la define "la primera escena legible")

- **Archivo**: `crates/datacube-io/src/stack.rs:186-253`
- **Dimensión**: D4 Abstracciones / reproducibilidad
- **Descripción**: la grilla de referencia (CRS, resolución, extent, alineación)
  la define el primer scene+asset que se logra leer, que depende del orden del
  catálogo, del filtro de nubes y de fallas de red. No se puede pedir "dame el
  cubo en EPSG:32719, 10 m, este bbox exacto, alineado a la grilla MGRS".
  gdalcubes tiene el `cube_view` como concepto de primera clase precisamente por
  esto; es su mejor idea de diseño.
- **Impacto**: dos corridas con distinto `--max-cloud` pueden producir cubos en
  distinta zona UTM y distinta resolución (si la primera escena legible cambia);
  imposible garantizar grillas idénticas entre sitios o períodos para comparar
  mapas. Limita el uso científico serio del motor.
- **Fix propuesto**: un `GridSpec` opcional en `StackConfig`:
  ```rust
  pub struct GridSpec {
      pub epsg: u32,
      pub resolution: f64,          // unidades del CRS
      pub bbox: Option<[f64; 4]>,   // en el CRS destino; None = derivar del bbox WGS84
      pub align: Option<f64>,       // snap de origen (p.ej. 60.0 para MGRS)
  }
  ```
  Cuando `grid: Some(spec)`, construir el raster de referencia sintético desde el
  spec (sin depender de ninguna escena) y alinear todo contra él; cuando `None`,
  conservar el comportamiento actual. Es el cambio de API con mejor razón
  costo/beneficio de toda la auditoría.

### [HIGH] H4 — El `Cube` es geo-ciego: el georef vive repetido en 3 crates

- **Archivo**: `crates/datacube-io/src/stack.rs:137-151` (`StackedCube.transform/epsg`),
  `crates/datacube-zarr/src/lib.rs:32-39` (`GeoRef`),
  `crates/datacube-cli/src/stack_cmd.rs:336-350` (los pasea a mano)
- **Dimensión**: D3 Cohesión / D2 Acoplamiento
- **Descripción**: `datacube_core::Cube` no sabe dónde está en el espacio. Cada
  consumidor inventó su propio acarreo del georef: `StackedCube` (transform +
  epsg), `datacube-zarr::GeoRef` (epsg + transform), y el CLI los copia del stack
  al GeoTIFF. Toda operación que produce un cubo nuevo (`composite`, `gapfill`,
  `ndvi`) pierde el georef y el llamador debe re-adjuntarlo. En Python el `Cube`
  ni siquiera puede expresarlo, así que exportar un trend map georreferenciado
  desde Python hoy es imposible sin pasar el transform por fuera.
- **Impacto**: API frágil (fácil escribir un GeoTIFF con el transform equivocado
  tras un composite que cambió dims), duplicación creciente con cada target
  nuevo, y el binding Python queda cojo para el caso de uso principal.
- **Fix propuesto**: mover `GeoRef { epsg: Option<u32>, transform: Option<[f64;6]> }`
  a `datacube-core` como campo opcional del `Cube` (default `None` para cubos
  puramente numéricos; los tests actuales no cambian). Las operaciones lo
  propagan automáticamente (composite/gapfill/bandmath no cambian la grilla
  espacial). `StackedCube` y `datacube-zarr` pasan a consumirlo; `datacube-zarr`
  borra su copia. Exponer `.epsg`/`.transform` en PyO3.

### [HIGH] H5 — Ejecución enteramente materializada: el streaming es solo de vistas ✅ resuelto v0.8.0 (parcial)

- **Archivo**: `crates/datacube-io/src/stack.rs:255-274`,
  `crates/datacube-zarr/src/lib.rs:99-155`
- **Dimensión**: D4 Abstracciones / escalabilidad (arquitectura)
- **Descripción**: `stack()` retiene **todas** las escenas (`Vec<Raster>`) y
  después las copia al `Array4` (pico de memoria ~2× el cubo); `read_zarr` lee
  el array completo en un solo `retrieve_array_subset`. `Cube::chunks()` existe
  pero nadie lo usa como unidad de ejecución. El área procesable está acotada por
  la RAM: un año de S2 a 10 m sobre 100×100 km ≈ (1 banda) 10960²·70·8 B ≈ 67 GB.
  gdalcubes ejecuta chunk a chunk precisamente para no tener este techo.
- **Impacto**: el motor hoy es "de escritorio para ROIs medianas"; el título de
  "mejor motor de cubos" exige procesar regiones grandes en hardware modesto.
- **Fix propuesto** (incremental, no big-bang):
  1. *Quick win*: en `stack()`, escribir cada escena al `Array4` apenas se lee y
     soltarla (elimina el 2×; ver M4 para la copia por slices).
  2. `read_zarr_chunked(path, chunk_y, chunk_x) -> impl Iterator<Item = (Cube, ChunkPos)>`
     leyendo subsets espaciales (los chunks ya son espaciales-256, el layout
     coopera) + `write_zarr` por subsets. Con eso `trend_map` sobre un store
     GeoZarr corre acotado en memoria: `for chunk in read_zarr_chunked(...) { par_map_series(...) }`.
  3. (v0.7+) un grafo lazy estilo gdalcubes sobre esa primitiva
     (`stack → mask → composite → index → trend` evaluado por chunk). El
     diferenciador "cubo Rust nativo sobre GeoZarr" se concreta aquí.
- **Resuelto (parcial) en v0.8.0**: pasos 2 hecho — `read_zarr_chunked(path,
  chunk_y, chunk_x)` (iterador perezoso, memoria acotada a un tile, `GeoRef`
  desplazado por tile) + `ZarrCubeWriter::create/.write_chunk` (contraparte de
  escritura, tiles en cualquier orden). Patrón `for chunk in
  read_zarr_chunked(...) { cube.par_map_series(...) }` verificado con tests
  que reconstruyen el cubo completo desde tiles y comparan igualdad exacta.
  Paso 1 (streaming en `stack()` mismo) y paso 3 (grafo lazy) quedan
  pendientes — este primitivo GeoZarr es independiente de `datacube-io` y no
  requiere red, por eso se priorizó.

### [MEDIUM] M1 — Los bindings Python retienen el GIL durante todo el cómputo Rayon

- **Archivo**: `crates/datacube-python/src/lib.rs:203-228` (`trend_map`), ídem
  `composite`, `gapfill`, `ndvi`/`evi`/... y las 5 funciones por-serie
- **Dimensión**: D5 Error handling/concurrencia (calidad del binding)
- **Descripción**: `par_map_series` corre en el pool de Rayon, pero el hilo
  llamador **mantiene el GIL** hasta retornar (~780 ms para 256²×60). Cualquier
  otro hilo Python (un dashboard, un executor de dask, un servidor) queda
  bloqueado ese tiempo. El fix es la práctica estándar PyO3:
- **Fix propuesto**:
  ```rust
  let grid = py.allow_threads(|| {
      self.inner.par_map_series(band, |t, y| { ... })
  }).map_err(err)?;
  ```
  Aplicarlo a todo método cuyo closure no toque objetos Python (todos los de
  cómputo cumplen). Cero costo, gran diferencia en integración real.

### [MEDIUM] M2 — Zarr sin compresión, sin f32 y con I/O de un solo bloque ✅ resuelto v0.8.0

- **Archivo**: `crates/datacube-zarr/src/lib.rs:86-92, 104-107, 148-151`
- **Dimensión**: D7 Deuda técnica (para la promesa "cloud-native ARD")
- **Descripción**: `ArrayBuilder` no configura codecs → f64 crudo. Un cubo de
  reflectancias (que cabe cómodo en f32 y comprime muy bien) ocupa >8× lo
  necesario en disco/objeto; en cloud eso es costo y latencia de transferencia.
  Además write y read pasan por un único subset del array completo (refuerza H5).
- **Fix propuesto**: (a) codec `zstd` por defecto (zarrs lo trae) con
  `ZarrOptions { compression, chunk, dtype }` para override; (b) opción de
  dtype f32 en escritura (`write_zarr_f32` o campo en options) manteniendo f64
  el modelo en memoria; (c) sharding cuando se apunte a object store. Interop:
  zarr-python/xarray leen zstd sin fricción.
- **Resuelto en v0.8.0**: (a) y (b) hechos — `ZarrOptions { compression_level:
  Option<i32>, dtype: ZarrDType::{F64,F32} }`, default zstd nivel 5 + f64
  (compresión transparente y sin pérdida; antes el default era f64 crudo).
  `write_zarr_with_options`/`ZarrCubeWriter::create` la aceptan; `read_zarr`
  detecta el dtype real (`array.data_type()`) y sube f32→f64 al leer. Interop
  verificada desde Python real (zarr 3.2.1): store default (`bytes`+`zstd`,
  level 5) y store f32 ambos decodifican transparentemente, NDVI recalculado
  coincide. (c) sharding queda pendiente (solo relevante para object store).

### [MEDIUM] M3 — Lecturas STAC secuenciales ✅ resuelto v0.9.0

- **Archivo**: `crates/datacube-io/src/stack.rs:193-245`
- **Dimensión**: D1 (latencia, no complejidad)
- **Descripción**: las escenas se leen una por una; el stacking es I/O-bound de
  red (range requests HTTP). Con 30-100 escenas, paralelizar descargas es la
  mayor ganancia de wall-clock disponible en todo el motor (~Nx con N workers).
- **Fix propuesto**: leer con un pool acotado conservando el orden temporal:
  ```rust
  let results: Vec<_> = items.par_iter()  // o un semáforo de 4-8 permits
      .map(|item| read_scene(...))
      .collect();
  ```
  Cuidado: la escena de referencia debe resolverse primero (leer la primera
  válida secuencialmente, luego el resto en paralelo), y `StacClientBlocking` /
  firma SAS deben ser `Sync` o clonables por worker. Exponer `concurrency: usize`
  en `StackConfig`.
- **Resuelto en v0.9.0**: `stack()` se dividió en dos fases. *Bootstrap*
  (solo si `cfg.grid` es `None`): lee ítems uno a uno, secuencial, hasta que
  el primero exitoso fija la grilla de referencia. *Paralela*: el resto de
  los ítems (o **todos**, si `GridSpec` ya fijó la referencia de antemano —
  caso común dado H3) se filtran con `plan_scene` (chequeo barato sin red:
  datetime/nubes/cross-zone) y se leen con `rayon::ThreadPoolBuilder::new()
  .num_threads(cfg.concurrency)` — un pool acotado dedicado, separado del
  pool global de Rayon usado para cómputo por-píxel. `into_par_iter()
  .collect()` sobre un `Vec` preserva el orden de entrada, así que el eje
  temporal del cubo sigue ordenado sin trabajo extra. `StacClientBlocking`
  resultó `Sync` sin cambios (runtime tokio compartido `'static` + `Mutex`
  interno para el cache SAS). Nuevo `StackConfig::concurrency` (default 8) +
  CLI `--concurrency`. Verificado e2e contra Planetary Computer (49 ítems,
  Santiago ene-abr 2024): **4m41s → 47s** (~6×) con `--concurrency 8` vs `1`,
  mismo orden de escenas, mismos skips, mismo cubo resultante byte a byte
  (comparado vía el reporte JSON). También probado con `GridSpec` fijo (toda
  la lectura va a la fase paralela desde el ítem 0, sin bootstrap).

### [MEDIUM] M4 — Copia escena→cubo elemento a elemento y clon completo del cubo en el CLI

- **Archivo**: `crates/datacube-io/src/stack.rs:261-269`,
  `crates/datacube-cli/src/stack_cmd.rs:167`
- **Dimensión**: D1 Complejidad/perf
- **Descripción**: (a) el volcado usa `data[[bi, r, c, ti]] = src[[r, c]]` — 4
  índices con bounds-check por celda y patrón de escritura strided (t es el eje
  contiguo del destino pero se escribe con stride nt); (b) el CLI hace
  `stacked.cube.clone()` del cubo completo solo para poder mutarlo, duplicando el
  pico de memoria.
- **Fix propuesto**: (a) escribir por lanes contiguas del origen:
  `data.slice_mut(s![bi, r, .., ti]).assign(&src.row(r))` (o el volcado directo
  propuesto en H5.1); (b) en el CLI, desestructurar:
  ```rust
  let StackedCube { cube, slices, skipped, transform, epsg } = stacked;
  let mut cube = cube;  // move, no clone
  ```

### [MEDIUM] M5 — Versionado desincronizado, sin tags y sin CI

- **Archivo**: `Cargo.toml:14` (`version = "0.4.0"` con v0.5 ya commiteada),
  `git tag` vacío, sin `.github/workflows/`
- **Dimensión**: D7 Deuda técnica
- **Descripción**: el commit `3e3af4f` se anuncia como v0.5 pero el workspace
  sigue en 0.4.0 (y `datacube_rs.__version__` en Python reporta 0.4.0). No hay
  tags que anclen los releases que cita el paper, ni CI que proteja el "cargo
  test --workspace verde" entre sesiones.
- **Fix propuesto**: bump a `0.5.0` + `git tag v0.1..v0.5` retroactivos sobre los
  commits de release; GitHub Actions mínimo: `fmt --check`, `clippy -D warnings`,
  `test --workspace` (core/cli/zarr/python compilan sin surtgis; excluir
  `datacube-io` o cachear el sibling), y `wasm-pack test --node`. Para el paper
  (reproducibilidad/Zenodo) esto es casi obligatorio.

### [MEDIUM] M6 — Dependencia por path a `../surtgis` bloquea publicación y colaboración

- **Archivo**: `Cargo.toml:22-25`
- **Dimensión**: D2 Acoplamiento (build-level)
- **Descripción**: `surtgis-core`/`surtgis-cloud` por path relativo obligan a un
  checkout hermano no versionado (¿qué commit de surtgis?). Nadie puede `cargo
  build --features stac` sin reproducir tu layout de disco; `datacube-io` no es
  publicable en crates.io; el Cargo.lock no fija a surtgis.
- **Fix propuesto**: corto plazo, cambiar a dependencia git con rev pineado
  (`surtgis-cloud = { git = "...", rev = "abc123" }`) — reproducible y sin tocar
  código. Mediano plazo: publicar `surtgis-cloud`/`surtgis-core` en crates.io
  (también le sirve a SurtGIS) o extraer el subconjunto STAC+COG+reproject a un
  crate compartido pequeño.

### [MEDIUM] M7 — `CubeError::DimensionMismatch` está sobrecargado

- **Archivo**: `crates/datacube-core/src/error.rs`, usos en `temporal.rs:100,141`,
  `linear.rs:55`, `theil_sen.rs:46`, `breaks.rs:117`
- **Dimensión**: D5 Error handling
- **Descripción**: "eje temporal desordenado", "t constante" y errores de shape
  de `from_shape_vec` se reportan todos como `DimensionMismatch`. Un consumidor
  (p.ej. el binding Python, que mapea todo a `ValueError` con el mensaje) no
  puede distinguir programáticamente "ordena tu eje" de "tus shapes no calzan".
- **Fix propuesto**: dos variantes nuevas y honestas:
  ```rust
  #[error("time axis must be ascending: {0}")]
  UnsortedTime(String),
  #[error("degenerate input: {0}")]   // t constante, etc.
  DegenerateInput(String),
  ```
  Es un cambio semver-minor (enum ya es `non_exhaustive`-able; aprovechar de
  marcarlo `#[non_exhaustive]` antes de 1.0).

### [MEDIUM] M8 — Theil-Sen O(n²) en tiempo y memoria por píxel

- **Archivo**: `crates/datacube-core/src/stats/theil_sen.rs:36-44`
- **Dimensión**: D1 Complejidad algorítmica
- **Descripción**: materializa las n(n−1)/2 pendientes por píxel. Para series de
  60-100 composites (el caso actual) es irrelevante, pero un cubo Landsat de 40
  años (~800 obs) son ~320k f64 = 2.5 MB **por píxel por hilo**, y el cómputo
  crece cuadrático. No es bug; es un techo documentable.
- **Fix propuesto**: corto plazo, documentar el costo en el rustdoc (está
  parcialmente: "O(n²) pairs") y reusar un buffer por hilo
  (`rayon::map_with(Vec::new(), ...)` en los llamadores). Largo plazo, el
  estimador exacto O(n log n) (Katz–Cole / Chan & Pătraşcu) o el aproximado por
  muestreo de pares para n > ~500.

### [MEDIUM] M9 — WASM duplica a mano todos los structs de resultado

- **Archivo**: `crates/datacube-wasm/src/lib.rs:20-190`
- **Dimensión**: D4 Abstracciones (duplicación)
- **Descripción**: ~100 líneas de structs espejo (`LinearTrend`, `HarmonicFit`,
  `BreakResult`, ...) que copian campo a campo los tipos de core solo para
  derivar `Serialize`. Cada campo nuevo en core exige tocar el espejo (ya pasó
  con band-math: WASM quedó atrás).
- **Fix propuesto**: feature opcional en core:
  ```toml
  # datacube-core/Cargo.toml
  [features]
  serde = ["dep:serde"]
  ```
  con `#[cfg_attr(feature = "serde", derive(serde::Serialize))]` en los tipos de
  resultado. `datacube-wasm` pasa a serializar los tipos de core directamente y
  borra los espejos; `Trend` gana un `Display`/`as_str()` compartido que también
  desduplica el `match` repetido en CLI, Python y WASM (3 copias hoy).

### [LOW] L1 — `fractional_year` ignora offsets de zona horaria

- **Archivo**: `crates/datacube-io/src/time.rs:36-49`
- **Descripción**: `"2024-06-15T23:30:00+05:00"` se interpreta como si fuera UTC
  (el offset se descarta). STAC exige UTC así que en la práctica no muerde, pero
  el parser acepta el input y devuelve un valor sutilmente corrido (< 1 día).
  Fix: rechazar (`None`) offsets distintos de `Z`/`+00:00`, o aplicarlos.

### [LOW] L2 — Escenas sin CRS identificable se resamplean asumiendo el CRS de referencia

- **Archivo**: `crates/datacube-io/src/stack.rs:320-346`
- **Descripción**: si `item.epsg()` y el CRS del COG son ambos `None`, no hay
  reproyección posible y la escena cae directo a `resample_to_grid` contra la
  referencia, que asume mismo CRS → desalineación silenciosa si en realidad era
  otra proyección. Fix: si el CRS es desconocido y hay referencia con EPSG,
  skip con razón explícita ("unknown CRS") en `skipped`.

### [LOW] L3 — Validación de `alpha` acepta 0.0 pero el mensaje dice `(0, 1)`

- **Archivo**: `crates/datacube-core/src/stats/breaks.rs:88-93`
- **Descripción**: `(0.0..1.0).contains(&alpha)` incluye `0.0` (rango
  semiabierto por la izquierda al revés de lo anunciado). Inofensivo
  (`alpha = 0` → nunca hay breaks) pero inconsistente. Fix:
  `if !(alpha > 0.0 && alpha < 1.0)`.

### [LOW] L4 — Doc desactualizada en `write_zarr`

- **Archivo**: `crates/datacube-zarr/src/lib.rs:54-55`
- **Descripción**: "returning the stored georeference echo" — la función retorna
  `Result<(), _>`. Resto del rustdoc excelente; este quedó de una firma anterior.

### [LOW] L5 — Orden temporal por comparación de strings ISO

- **Archivo**: `crates/datacube-io/src/stack.rs:185`
- **Descripción**: correcto mientras todos los datetimes sean UTC-`Z` del mismo
  formato (el caso STAC), frágil ante offsets mixtos. Barato de blindar:
  ordenar por el `fractional_year` ya calculado en vez del string.

### [LOW] L6 — Binding Python sin type stubs ni clases de resultado

- **Archivo**: `crates/datacube-python/`
- **Descripción**: las funciones retornan `dict` sin `.pyi`, así que IDEs y
  mypy no ven nada. Para adopción: generar `datacube_rs.pyi` (a mano, son ~8
  firmas) y empaquetarlo; considerar dataclasses de resultado en una capa
  Python pura sobre el módulo nativo. Relacionado: no hay CI de wheels
  (maturin-action) — sin wheels en PyPI el target Python no existe para el
  usuario típico.

### [LOW] L7 — MSRV no declarada

- **Archivo**: `Cargo.toml` (workspace)
- **Descripción**: edition 2024 + let-chains fijan un rustc reciente de facto;
  declarar `rust-version = "1.88"` (o la que corresponda) hace el fallo de
  compilación diagnóstico en vez de críptico.

---

## Arquitectura (Fase 4)

**Lo que está bien y no hay que tocar:**

- **Layering ejemplar**: `core` (cómputo puro, 4 deps), `io` (STAC/COG),
  targets delgados (CLI/PyO3/WASM/Zarr) que solo adaptan. Flujo unidireccional,
  cero imports circulares, cero módulos "utils".
- **La decisión de layout `(band, y, x, t)` con t contiguo** está bien razonada,
  documentada y explotada consistentemente (slices por serie en
  `par_map_series`, `par_chunks_mut(nt)` en temporal, volúmenes de banda
  contiguos en bandmath, C-order 1:1 con Zarr).
- **Semántica NaN-como-nodata** uniforme en todo el motor, con filtrado pairwise
  documentado y testeado en cada estadístico.
- **Error handling**: sin panics en producción; los 3 `expect` de layout citan
  el invariante que los garantiza. `thiserror` en libs, `anyhow` + context solo
  en el binario. Correcto de libro.
- **Testing**: unit + doctests + benches (criterion) + validación numérica
  contra referencias externas + test de red `#[ignore]` + interop Python real
  para Zarr. Para un motor científico, el estándar de validación es el punto
  más fuerte del proyecto.

**El gap arquitectónico central** (síntesis de H3+H4+H5): el motor tiene un
*modelo* de cubo excelente pero aún no tiene un *concepto de cubo virtual*: la
grilla es un accidente de la primera escena (H3), la geografía no viaja con el
cubo (H4) y la evaluación es siempre materializada (H5). Los tres se resuelven
con la misma inversión conceptual — un `CubeSpec/GridSpec` de primera clase +
georef en core + ejecución por chunks sobre GeoZarr — y esa inversión es
exactamente el diferenciador prometido en el roadmap ("cubo Rust nativo sobre
GeoZarr"). gdalcubes lo hizo con `cube_view` + chunk streaming en C++; hacerlo
en Rust con Zarr V3 nativo, sin GDAL en el hot path, es una contribución
publicable por sí sola.

## Deuda técnica

- TODOs/FIXMEs: **0**. Código muerto: no detectado. Clippy: **0 warnings** en
  los 6 crates.
- Deuda real: versionado/tags/CI (M5), path-dependency surtgis (M6), duplicación
  WASM (M9), stubs Python (L6). Toda acotada y barata.

## Recomendaciones priorizadas

1. **Quick wins de una sesión** — ✅ **ejecutados el 2026-07-02**:
   M1 (GIL: `py.detach` en trend_map/composite/gapfill/índices), M4 (destructure
   sin clone + copia por slices), M7 (`UnsortedTime`/`DegenerateInput` +
   `#[non_exhaustive]`), L3, L4, L5 (sort por año fraccional), M5 (bump 0.5.0,
   tags v0.1.0–v0.4.0, CI en `.github/workflows/ci.yml`, `cargo fmt` aplicado
   a todo el workspace). Verificado: cargo test 12 suites ok, clippy limpio
   (incl. `--features stac`), pytest 14/14 con el módulo reconstruido.
2. **Paridad funcional ARD** — ✅ **ejecutado el 2026-07-02 (v0.6.0)**:
   H2 (`MaskConfig` en `StackConfig`: asset de calidad leído 1×/escena,
   resample nearest a la grilla de cada banda, NaN fuera de `keep`, aplicado
   antes del resample bilineal; CLI `--mask-scl --mask-asset --mask-keep`),
   H1 (`CompositeWindow::{CalendarMonth, CalendarYear}` con inversa exacta de
   `fractional_year`; `--composite monthly` ahora es calendario, `yearly`
   nuevo, `Period` queda para ventanas físicas; Python `"monthly"`/`"yearly"`),
   H3 (`GridSpec { epsg, resolution, bbox, align }` en `StackConfig`:
   referencia sintética construida antes del loop → grilla reproducible;
   CLI `--grid-epsg --grid-res --grid-bbox --grid-align`).
   Verificado: tests core 55 + io 15, pytest 15/15, clippy limpio.
3. **El diferenciador** — la tesis "mejor motor de cubos":
   H4 (georef en core) → M2 (Zarr comprimido + f32) → H5 (ejecución por chunks
   sobre GeoZarr) → M3 (lecturas paralelas). En ese orden: cada paso habilita el
   siguiente y todos suman a la sección de arquitectura del paper.
   **H4 ejecutado el 2026-07-02 (v0.7.0)**: `GeoRef { epsg, transform }` ahora
   vive en `datacube_core::Cube` (`.georef()`/`.with_georef()`), propagado
   automáticamente por `composite`/`gapfill_linear`/band-math
   (`inherit_georef` interno); `datacube-io::stack()` lo adjunta al cubo;
   `datacube-zarr` re-exporta el tipo de core en vez de duplicarlo; PyO3
   expone `.epsg`/`.transform`/`.with_georef()`; el CLI ya no acarrea
   `transform`/`epsg` a mano por toda la función — los lee de `cube.georef()`
   justo antes de escribir el GeoTIFF, después de mask/grid/composite/index.
   Verificado e2e: mismo GeoTIFF (origen, pixel size, EPSG) que antes de H4.
   **M2 + H5(parcial) ejecutados el 2026-07-02 (v0.8.0)**: ver detalle en los
   hallazgos M2/H5 arriba — Zarr comprimido zstd + f32 opcional, y
   `read_zarr_chunked`/`ZarrCubeWriter` para ejecución por chunks acotada en
   memoria sobre GeoZarr.
   **M3 ejecutado el 2026-07-02 (v0.9.0)**: lecturas STAC en pool paralelo
   acotado (`StackConfig::concurrency`, default 8) — ver detalle arriba;
   ~6× de aceleración medido e2e. AUDIT grupo 3 queda con dos ítems abiertos,
   ambos dentro de H5: streaming en `stack()` mismo (paso 1, elimina el
   `Vec<Raster>` intermedio) y el grafo lazy `stack→mask→composite→index→
   trend` evaluado por chunk (paso 3, v0.7+ en el roadmap original).
4. **Ecosistema/adopción**: M6 (desacoplar surtgis), L6 (stubs + wheels PyPI),
   M9 (feature serde en core). Sin esto el motor es excelente pero solo tuyo.
