# datacube-rs → Computers & Geosciences — plan de paper

> Síntesis de la lectura de los 13 PDFs en `papers/`. Venue: **Computers & Geosciences**
> (software paper). Fecha: 2026-06-25.

## 1. Tabla comparativa VERIFICADA (datacube-rs vs el campo)

Corregida tras leer los papers — varias celdas de mi borrador inicial eran demasiado
fuertes. Esta es la versión defendible ante un reviewer.

| Dimensión | gdalcubes | Open Data Cube | openEO | Google Earth Engine | FORCE | xarray | **datacube-rs** |
|---|---|---|---|---|---|---|---|
| Lenguaje | R + C++ | Python | API (clientes R/Py/JS) | JS/Py → cloud (JVM) | C/C++ | Python | **Rust** |
| Deployment | librería | Python + HPC | API sobre backends cloud | servicio cloud propietario | suite CLI | librería | **single-binary: CLI/lib/PyO3/WASM** |
| STAC-native | no (paper 2019) | no (índice propio) | **sí** | no (catálogo privado) | no | no | **sí** |
| COG-native | endorsado/parcial | no (netCDF4) | **sí** | no (tile-store) | no (GeoTIFF/binario) | no | **sí** |
| Offline / sin lock-in | sí | sí (setup pesado) | depende del backend | **no** | sí | sí | **sí** |
| Theil-Sen / tendencia | no | no | no (UDF) | Sen-slope (reducer) | OLS trend (no MK) | no | **Theil-Sen + test Mann-Kendall** |
| Quiebres estructurales | no | no | no | no | CAT change | no | **OLS-CUSUM + binary seg** |
| Fenología armónica | no | no | no | hecho a mano | SPLITS phenometrics | no | **regresión armónica** |
| Paridad numérica validada | — | — | — | — | — | — | **103/103 a 1e-9** |
| Corre en browser | no | no | no | no | no | no | **sí (WASM)** |
| Licencia | open (CRAN) | Apache-2.0 | open standard | propietario | GPLv3 | Apache-2.0 | MIT/Apache-2.0 |

**Frase-diferenciador honesta** (no sobre-vender): datacube-rs es el único que ofrece el
**conjunto unificado** —test de significancia Mann-Kendall + Theil-Sen + armónicos + quiebres—
como operadores nativos, numéricamente validados, en **un binario portable** que corre offline
y en el browser. FORCE es el competidor más cercano en profundidad temporal (fenología+trend+CAT)
pero es C/C++/HPC local, sin MK/Theil-Sen ni portabilidad; GEE tiene Sen-slope como reducer pero
es un servicio cloud cerrado.

## 2. ⚠ Reviewer landmines (correcciones de honestidad — CRÍTICO)

1. **NO decir "reproduce BFAST".** BFAST (Verbesselt 2010) usa **OLS-MOSUM + Bai-Perron + BIC**;
   nuestro `detect_breaks` usa **OLS-CUSUM + binary segmentation**. Son tests distintos. Frasear:
   *"in the spirit of BFAST's decompose-then-detect approach"*, y citar el test real que usamos
   (OLS-CUSUM, validado vs `statsmodels.breaks_cusumolsresid`). Un reviewer que conoce BFAST lo va a chequear.
2. **GEE NO es un blanco total en tendencia.** Tiene Sen-slope + correlación (Kendall/Spearman/Pearson)
   + regresión lineal/robusta como reducers. El claim correcto NO es "los competidores no tienen trend";
   es "ninguno ofrece el **test de significancia MK + el conjunto unificado validado**".
3. **FORCE es el competidor temporal más fuerte** — reconocerlo explícitamente (fenología SPLITS,
   trend lineal, CAT). Nuestro diferenciador vs FORCE = portabilidad (single-binary vs C/C++/HPC/OpenMP),
   STAC/COG-native (vs cubo local), y MK/Theil-Sen con significancia (FORCE no los tiene).
4. **Validación: sé preciso sobre contra QUÉ.** Validamos vs **pyMannKendall** (MK/Theil-Sen),
   **scipy** (OLS), **numpy.lstsq** (armónicos), **statsmodels** (CUSUM). **NO** validamos contra R `bfast`.
   Entonces los breaks tienen paridad vs el estadístico CUSUM de statsmodels, no vs bfast.
5. **El paper de pyMannKendall (JOSS, 3 pág) no trae ecuaciones.** Citar **Kendall 1975 / Mann 1945**
   para la fórmula de varianza tie-corrected, y pyMannKendall solo como el target de paridad bit-a-bit.

## 3. Procedencia de métodos (qué citar para cada estimador)

- **Mann-Kendall / Theil-Sen** → Kendall 1975, Mann 1945, Theil 1950, Sen 1968 (métodos);
  Hussain & Mahmud 2019 (pyMannKendall, target de paridad).
- **Regresión armónica (fenología)** → Zhu & Woodcock 2014 (CCDC: modelo `a0 + a1cos + b1sin + c1·x`,
  1 armónico anual + trend lineal). Es el modelo que nuestro `harmonic_regression` reproduce.
- **Quiebres** → Verbesselt 2010 (BFAST, la idea decompose-then-detect) + el test OLS-CUSUM real que usamos.

## 4. Estructura del paper (molde C&G, de los 4 exemplars)

Convención C&G rígida: **título = nombre del software** (`datacube-rs: An open-source Rust engine…`),
"open-source" en la primera frase del abstract, **Fig. 1 = diagrama de arquitectura** (obligatorio),
validación **sintético-primero-luego-real**, y back-matter fijo con **Code availability section**
(caja de metadata: lenguaje, tamaño, deps, licencia, repo) + **Data availability** separada.
Longitud ~12-14 pág, 8-11 figuras, ~1 tabla. Sin code listings largos.

1. **Abstract** — gap + qué es + headline (103/103, benchmarks, 5 targets incl. WASM). "open-source Rust" al frente.
2. **Introduction** — necesidad de cubos temporales RS; survey por nombre (gdalcubes, ODC, openEO, GEE,
   FORCE, BFAST, pyMannKendall) + el hueco (ningún motor Rust single-binary sobre ARD cloud-native con
   stats temporales validadas first-class); "Here we present datacube-rs"; bullets de contribuciones; "paper organized as follows".
3. **Methods / teoría de las stats temporales** — ecuaciones numeradas: OLS, Theil-Sen + MK (con varianza
   tie-corrected), armónica, quiebres OLS-CUSUM. Implementaciones de referencia que validamos.
4. **The datacube-rs engine (Software)** — **abre con Fig. 1 (arquitectura)**: STAC/COG/GeoZarr → modelo
   de cubo (band,y,x,t) → iterador streaming por píxel/chunk (Rayon) → estimadores → salidas. Sub: 4.1 modelo
   de cubo; 4.2 iterador streaming + chunking; 4.3 ingest STAC/COG (reusa SurtGIS); 4.4 targets de deployment
   (CLI/lib/PyO3/WASM) como "multi-target usability".
5. **Validation and benchmarking** — dos niveles (estilo TomoATT): 5.1 **paridad numérica** (103/103 vs
   pyMannKendall/scipy/numpy/statsmodels — tabla); 5.2 **performance** (criterion: throughput, scaling vs cores,
   footprint streaming); 5.3 **caso real** (cubo Sentinel-2 multianual NDVI: trend/fenología/break — ya corrido:
   Santiago / frontera UTM 18-19).
6. **Discussion** — posicionamiento vs el campo; qué compra Rust single-binary + WASM + GeoZarr; limitaciones + roadmap.
7. **Conclusions**.
- **Back-matter (orden exacto)**: CRediT → **Code availability section** (caja metadata + **Zenodo DOI** vía
  `/zenodo`, over-delivers vs los 4 exemplars) → Data availability → Competing interest → Acknowledgements →
  Appendix A → References.

**Figuras objetivo (8-11)**: Fig.1 arquitectura · esquema iterador/chunking · scatter/Bland-Altman paridad vs
pyMannKendall · plot de scaling de benchmarks · mapas + series del caso Sentinel-2.

## 5. Citas exactas (related work)

- gdalcubes — Appel & Pebesma (2019), *Data* 4(3):92, doi:10.3390/data4030092
- Open Data Cube — Lewis et al. (2017), *RSE* 202:276-292, doi:10.1016/j.rse.2017.03.015
- openEO — Schramm et al. (2021), *Remote Sensing* 13(6):1125, doi:10.3390/rs13061125
- Google Earth Engine — Gorelick et al. (2017), *RSE* 202:18-27, doi:10.1016/j.rse.2017.06.031
- FORCE — Frantz (2019), *Remote Sensing* 11(9):1124, doi:10.3390/rs11091124
- xarray — Hoyer & Hamman (2017), *JORS* 5(1):10, doi:10.5334/jors.148
- BFAST — Verbesselt et al. (2010), *RSE* 114(1):106-115, doi:10.1016/j.rse.2009.08.014
- CCDC — Zhu & Woodcock (2014), *RSE* 144:152-171, doi:10.1016/j.rse.2014.01.011
- pyMannKendall — Hussain & Mahmud (2019), *JOSS* 4(39):1556, doi:10.21105/joss.01556
- (stars — Pebesma & Bivand 2023, *Spatial Data Science*, CRC Press / paquete R; sin paper único)
