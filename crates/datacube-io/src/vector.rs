//! Zonal aggregation of a temporal cube by polygon.
//!
//! Given a vector layer (Shapefile or GeoJSON) whose polygons are identified
//! by an attribute field, and a georeferenced [`Cube`], this produces a tidy
//! scalar series **per polygon, per temporal window and per band**:
//! `(polygon_id, time, band, reducer, value, n_valid, n_total)`.
//!
//! The wetland-monitoring use case that motivates it: the unit of analysis is
//! the wetland *polygon*, not the bounding box the cube is defined over.
//!
//! Design (each step reuses the engine rather than reinventing it):
//! 1. **Read** the vector with `surtgis_core::vector::read_vector` (GeoJSON
//!    always; `.shp` via the `shapefile` feature). Geometries come back as
//!    `geo_types` values, re-exported from `surtgis_core`.
//! 2. **Reproject** each polygon to the cube's CRS with the pure-Rust
//!    WGS84↔UTM / UTM↔UTM point transforms in `surtgis_cloud::reproject` —
//!    the geometry is reprojected, never the raster.
//! 3. **Rasterize** onto the cube grid with a configurable [`PixelInclusion`]
//!    rule (centre, any-touch, or area-fraction weights).
//! 4. **Reduce** each polygon's covered pixels, pooled over every slice in the
//!    temporal bin, with a NaN-aware [`Reducer`]; `n_valid`/`n_total` report
//!    coverage for quality control.
//! 5. **Bin** time with the same windows as [`datacube_core::CompositeWindow`]
//!    (shared via [`datacube_core::time_bins`]), so an annual median per
//!    wetland lines up with `composite`'s calendar bins exactly.

use std::io::Write;
use std::path::Path;

use datacube_core::{CompositeWindow, Cube, bin_time, time_bins};
use surtgis_cloud::reproject;
use surtgis_core::geo::{Area, BooleanOps, Contains, Intersects};
use surtgis_core::geo_types::{Coord, Geometry, LineString, Polygon, Rect};
use surtgis_core::vector::{AttributeValue, FeatureCollection, read_vector};

use crate::StackError;

/// How a polygon selects the cube pixels it covers.
///
/// The choice is semantically significant (it changes which border pixels are
/// counted and how they are weighted), so it is explicit rather than
/// hard-coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelInclusion {
    /// A pixel is included iff its **centre** falls inside the polygon (the
    /// GDAL/`rasterio` default `all_touched=False` rule). Weight 1.
    Center,
    /// A pixel is included iff the polygon touches **any** part of its cell
    /// (`rasterio`'s `all_touched=True`). Weight 1.
    AllTouched,
    /// A pixel is weighted by the **fraction of its area** covered by the
    /// polygon (weight in `(0, 1]`), for area-weighted reductions — the
    /// semantics of `exactextract`.
    AreaFraction,
}

/// A NaN-aware spatial reduction over a polygon's covered pixels.
///
/// Reductions are area-weighted when [`PixelInclusion::AreaFraction`] is used
/// (each pixel's weight is its coverage fraction); with [`PixelInclusion::Center`]
/// / [`PixelInclusion::AllTouched`] every weight is 1, so they collapse to the
/// plain statistics. `Min`/`Max` ignore weights (an extreme is an extreme);
/// `Median` is unweighted (a weighted median is non-standard) and documented
/// as such.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reducer {
    /// Weighted arithmetic mean `Σ(w·v) / Σw`.
    Mean,
    /// Unweighted median (weights ignored).
    Median,
    /// Minimum finite value.
    Min,
    /// Maximum finite value.
    Max,
    /// Weighted population standard deviation `sqrt(Σw·(v-mean)² / Σw)`.
    Std,
    /// Weighted sum `Σ(w·v)`.
    Sum,
    /// Effective covered-pixel count `Σw` over finite pixels (integer for
    /// centre/all-touched; fractional for area-fraction).
    Count,
    /// Weighted fraction of finite values strictly above the threshold,
    /// `Σ(w·[v>t]) / Σw`.
    FractionAbove(f64),
}

impl Reducer {
    /// The lowercase name written into the `reducer` output column.
    pub fn name(&self) -> &'static str {
        match self {
            Reducer::Mean => "mean",
            Reducer::Median => "median",
            Reducer::Min => "min",
            Reducer::Max => "max",
            Reducer::Std => "std",
            Reducer::Sum => "sum",
            Reducer::Count => "count",
            Reducer::FractionAbove(_) => "fraction_above",
        }
    }
}

/// Configuration for [`zonal_reduce`].
#[derive(Debug, Clone)]
pub struct ZonalConfig {
    /// Attribute field identifying each polygon (its value becomes
    /// `polygon_id`). Int/float/string/bool attributes are all accepted and
    /// stringified.
    pub id_field: String,
    /// Pixel-inclusion rule.
    pub inclusion: PixelInclusion,
    /// Spatial reduction over each polygon's covered pixels.
    pub reducer: Reducer,
    /// Temporal binning window (`SameTime` keeps one row per original slice).
    pub window: CompositeWindow,
    /// EPSG of the vector's CRS, overriding whatever the file declares. When
    /// both this and the file's CRS are unknown, the polygons are assumed to
    /// already be in the cube's CRS.
    pub source_epsg: Option<u32>,
}

impl ZonalConfig {
    pub fn new(id_field: impl Into<String>, inclusion: PixelInclusion, reducer: Reducer) -> Self {
        Self {
            id_field: id_field.into(),
            inclusion,
            reducer,
            window: CompositeWindow::SameTime,
            source_epsg: None,
        }
    }

    pub fn window(mut self, window: CompositeWindow) -> Self {
        self.window = window;
        self
    }

    pub fn source_epsg(mut self, epsg: Option<u32>) -> Self {
        self.source_epsg = epsg;
        self
    }
}

/// One row of the tidy zonal table.
#[derive(Debug, Clone)]
pub struct ZonalRow {
    pub polygon_id: String,
    /// Representative time of the temporal bin (mean of member slice times).
    pub time: f64,
    pub band: String,
    /// The reducer's name (constant per table).
    pub reducer: &'static str,
    /// The reduced value, or `NaN` when no finite observation was covered.
    pub value: f64,
    /// Covered (pixel, slice) pairs with a finite value.
    pub n_valid: u64,
    /// Covered (pixel, slice) pairs total (mask size × slices in the bin).
    pub n_total: u64,
}

/// The result of [`zonal_reduce`]: one row per `(polygon, time bin, band)`.
#[derive(Debug, Clone)]
pub struct ZonalTable {
    pub rows: Vec<ZonalRow>,
}

impl ZonalTable {
    /// Writes the table as CSV with a header row. `NaN` values are written as
    /// the string `NaN`.
    pub fn write_csv(&self, path: &Path) -> Result<(), StackError> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| StackError::Zonal(format!("cannot create {}: {e}", path.display())))?;
        file.write_all(self.to_csv_string().as_bytes())
            .map_err(|e| StackError::Zonal(format!("writing {} failed: {e}", path.display())))?;
        Ok(())
    }

    /// The table as a CSV string (header + rows).
    pub fn to_csv_string(&self) -> String {
        let mut out = String::from("polygon_id,time,band,reducer,value,n_valid,n_total\n");
        for r in &self.rows {
            let value = if r.value.is_finite() {
                r.value.to_string()
            } else {
                "NaN".to_string()
            };
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                csv_escape(&r.polygon_id),
                r.time,
                csv_escape(&r.band),
                r.reducer,
                value,
                r.n_valid,
                r.n_total,
            ));
        }
        out
    }

    /// The table as a pretty-printed JSON array of row objects.
    pub fn to_json_string(&self) -> Result<String, StackError> {
        let rows: Vec<serde_json::Value> = self
            .rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "polygon_id": r.polygon_id,
                    "time": r.time,
                    "band": r.band,
                    "reducer": r.reducer,
                    // JSON has no NaN literal; emit null so the file stays valid.
                    "value": if r.value.is_finite() { serde_json::json!(r.value) } else { serde_json::Value::Null },
                    "n_valid": r.n_valid,
                    "n_total": r.n_total,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .map_err(|e| StackError::Zonal(format!("serializing JSON failed: {e}")))
    }

    /// Writes the table as a JSON array of row objects.
    pub fn write_json(&self, path: &Path) -> Result<(), StackError> {
        std::fs::write(path, self.to_json_string()?)
            .map_err(|e| StackError::Zonal(format!("writing {} failed: {e}", path.display())))
    }

    /// Writes the table as a Snappy-compressed Parquet file (one row group,
    /// columns `polygon_id, time, band, reducer, value, n_valid, n_total`).
    ///
    /// Only available with the `parquet` Cargo feature. `value` is a DOUBLE
    /// column carrying `NaN` for empty reductions (Parquet stores NaN
    /// natively, so the column stays non-null); counts are INT64. Uses the
    /// low-level `parquet` writer (no Arrow), matching `surtgis-core`'s
    /// GeoParquet pin so the graph keeps a single `parquet` version.
    #[cfg(feature = "parquet")]
    pub fn write_parquet(&self, path: &Path) -> Result<(), StackError> {
        use std::sync::Arc;

        use parquet::basic::{Compression, ConvertedType, LogicalType, Repetition, Type as Phys};
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let zerr = |e: parquet::errors::ParquetError| StackError::Zonal(format!("parquet: {e}"));

        let utf8 = |name: &str| {
            SchemaType::primitive_type_builder(name, Phys::BYTE_ARRAY)
                .with_logical_type(Some(LogicalType::String))
                .with_converted_type(ConvertedType::UTF8)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .map(Arc::new)
                .map_err(zerr)
        };
        let prim = |name: &str, ty: Phys| {
            SchemaType::primitive_type_builder(name, ty)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .map(Arc::new)
                .map_err(zerr)
        };
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                utf8("polygon_id")?,
                prim("time", Phys::DOUBLE)?,
                utf8("band")?,
                utf8("reducer")?,
                prim("value", Phys::DOUBLE)?,
                prim("n_valid", Phys::INT64)?,
                prim("n_total", Phys::INT64)?,
            ])
            .build()
            .map(Arc::new)
            .map_err(zerr)?;

        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        );
        let file = std::fs::File::create(path)
            .map_err(|e| StackError::Zonal(format!("cannot create {}: {e}", path.display())))?;
        let mut writer = SerializedFileWriter::new(file, schema, props).map_err(zerr)?;
        let mut rg = writer.next_row_group().map_err(zerr)?;

        // Build each column's buffer once (schema order), then write and close
        // it against the next column writer — the low-level column-oriented
        // pattern surtgis-core's GeoParquet writer uses.
        let strings = |get: &dyn Fn(&ZonalRow) -> &str| -> Vec<ByteArray> {
            self.rows.iter().map(|r| ByteArray::from(get(r))).collect()
        };
        let polygon_id = strings(&|r| r.polygon_id.as_str());
        let band = strings(&|r| r.band.as_str());
        let reducer = strings(&|r| r.reducer);
        let time: Vec<f64> = self.rows.iter().map(|r| r.time).collect();
        let value: Vec<f64> = self.rows.iter().map(|r| r.value).collect();
        let n_valid: Vec<i64> = self.rows.iter().map(|r| r.n_valid as i64).collect();
        let n_total: Vec<i64> = self.rows.iter().map(|r| r.n_total as i64).collect();

        // One helper per physical type; each opens the next column, writes the
        // whole buffer (all REQUIRED, no def/rep levels) and closes it.
        macro_rules! write_col {
            ($ty:ty, $buf:expr) => {{
                let mut col = rg
                    .next_column()
                    .map_err(zerr)?
                    .ok_or_else(|| StackError::Zonal("parquet: missing column writer".into()))?;
                col.typed::<$ty>()
                    .write_batch($buf, None, None)
                    .map_err(zerr)?;
                col.close().map_err(zerr)?;
            }};
        }
        write_col!(ByteArrayType, &polygon_id);
        write_col!(DoubleType, &time);
        write_col!(ByteArrayType, &band);
        write_col!(ByteArrayType, &reducer);
        write_col!(DoubleType, &value);
        write_col!(Int64Type, &n_valid);
        write_col!(Int64Type, &n_total);

        rg.close().map_err(zerr)?;
        writer.close().map_err(zerr)?;
        Ok(())
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Reads a vector layer (auto-detected by extension: `.geojson`/`.json`
/// always, `.shp` with the `shapefile` feature) into a [`FeatureCollection`].
pub fn read_zones(path: &Path) -> Result<FeatureCollection, StackError> {
    Ok(read_vector(path)?)
}

/// An axis-aligned georeferenced grid derived from a cube's [`GeoRef`].
struct Grid {
    origin_x: f64,
    origin_y: f64,
    px_w: f64,
    px_h: f64,
    ny: usize,
    nx: usize,
}

impl Grid {
    /// Centre world coordinate of pixel `(row, col)`.
    fn center(&self, row: usize, col: usize) -> (f64, f64) {
        (
            self.origin_x + (col as f64 + 0.5) * self.px_w,
            self.origin_y + (row as f64 + 0.5) * self.px_h,
        )
    }

    /// World-space rectangle covered by pixel `(row, col)`.
    fn cell_rect(&self, row: usize, col: usize) -> Rect<f64> {
        let x0 = self.origin_x + col as f64 * self.px_w;
        let x1 = self.origin_x + (col as f64 + 1.0) * self.px_w;
        let y0 = self.origin_y + row as f64 * self.px_h;
        let y1 = self.origin_y + (row as f64 + 1.0) * self.px_h;
        Rect::new(
            Coord {
                x: x0.min(x1),
                y: y0.min(y1),
            },
            Coord {
                x: x0.max(x1),
                y: y0.max(y1),
            },
        )
    }

    fn pixel_area(&self) -> f64 {
        (self.px_w * self.px_h).abs()
    }

    /// Half-open pixel index ranges `(r0..r1, c0..c1)` covering a world bbox,
    /// clamped to the grid.
    fn pixel_window(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> (usize, usize, usize, usize) {
        let col_a = (min_x - self.origin_x) / self.px_w;
        let col_b = (max_x - self.origin_x) / self.px_w;
        let row_a = (min_y - self.origin_y) / self.px_h;
        let row_b = (max_y - self.origin_y) / self.px_h;
        // A one-pixel halo so cells that only *touch* the bbox boundary (e.g.
        // a polygon edge lying exactly on a grid line, for AllTouched) are
        // still tested; the per-pixel predicate drops any that don't qualify.
        let c0 = (col_a.min(col_b).floor() - 1.0).clamp(0.0, self.nx as f64) as usize;
        let c1 = ((col_a.max(col_b).ceil() + 1.0).clamp(0.0, self.nx as f64) as usize).max(c0);
        let r0 = (row_a.min(row_b).floor() - 1.0).clamp(0.0, self.ny as f64) as usize;
        let r1 = ((row_a.max(row_b).ceil() + 1.0).clamp(0.0, self.ny as f64) as usize).max(r0);
        (r0, r1, c0, c1)
    }
}

/// A polygon zone in the cube's CRS, ready to rasterize.
struct Zone {
    id: String,
    /// Flattened polygons (a `MultiPolygon` becomes several entries); assumed
    /// non-overlapping, so coverage is additive across them.
    polygons: Vec<Polygon<f64>>,
}

/// Computes zonal statistics of `cube` over the polygons in `features`.
///
/// The cube must carry a [`datacube_core::GeoRef`] with an axis-aligned
/// (north-up or flipped, no rotation) transform; rotated grids are rejected.
/// Polygons are reprojected from their CRS (`cfg.source_epsg`, else the
/// layer's declared CRS, else assumed already matching the cube) to the cube's
/// CRS before rasterization — only WGS84↔UTM and UTM↔UTM are supported without
/// GDAL, matching the rest of the engine; any other pair is an error asking
/// the caller to pre-reproject.
///
/// Rows come back ordered by feature, then time bin, then band.
pub fn zonal_reduce(
    cube: &Cube,
    features: &FeatureCollection,
    cfg: &ZonalConfig,
) -> Result<ZonalTable, StackError> {
    let georef = cube
        .georef()
        .ok_or_else(|| StackError::Zonal("cube has no georeference".into()))?;
    let transform = georef
        .transform
        .ok_or_else(|| StackError::Zonal("cube georeference has no affine transform".into()))?;
    let [a, b, c, d, e, f] = transform;
    if c != 0.0 || e != 0.0 {
        return Err(StackError::Zonal(
            "zonal aggregation requires an axis-aligned (non-rotated) cube grid".into(),
        ));
    }
    if b == 0.0 || f == 0.0 {
        return Err(StackError::Zonal(
            "cube transform has a zero pixel size".into(),
        ));
    }
    let (nb, ny, nx, _nt) = cube.dims();
    let grid = Grid {
        origin_x: a,
        origin_y: d,
        px_w: b,
        px_h: f,
        ny,
        nx,
    };

    let dst_epsg = georef.epsg;
    let src_epsg = cfg
        .source_epsg
        .or_else(|| features.crs().and_then(|c| c.epsg()));

    // Bin the time axis exactly as `composite` would, and precompute each
    // bin's representative time.
    let time = cube.time();
    let bins = time_bins(time, cfg.window)?;
    let bin_times: Vec<f64> = bins.iter().map(|g| bin_time(time, g)).collect();

    let zones = build_zones(features, cfg, src_epsg, dst_epsg)?;

    let bands = cube.bands();
    let data = cube.data();
    let mut rows = Vec::with_capacity(zones.len() * bins.len() * nb);

    for zone in &zones {
        // Rasterize the zone once (shared by every band and time bin).
        let mask = rasterize_zone(&zone.polygons, &grid, cfg.inclusion);
        let mask_pixels = mask.len() as u64;

        for (bin, &t) in bins.iter().zip(&bin_times) {
            let n_total = mask_pixels * bin.len() as u64;
            for (bi, band) in bands.iter().enumerate() {
                let (value, n_valid) = reduce_bin(&data, bi, &mask, bin, cfg.reducer);
                rows.push(ZonalRow {
                    polygon_id: zone.id.clone(),
                    time: t,
                    band: band.clone(),
                    reducer: cfg.reducer.name(),
                    value,
                    n_valid,
                    n_total,
                });
            }
        }
    }

    Ok(ZonalTable { rows })
}

/// Extracts, identifies and reprojects the polygon zones from a collection.
fn build_zones(
    features: &FeatureCollection,
    cfg: &ZonalConfig,
    src_epsg: Option<u32>,
    dst_epsg: Option<u32>,
) -> Result<Vec<Zone>, StackError> {
    let mut zones = Vec::new();
    for (idx, feature) in features.iter().enumerate() {
        let id = feature_id(feature, &cfg.id_field, idx)?;
        let Some(geom) = &feature.geometry else {
            continue;
        };
        let mut polygons = Vec::new();
        collect_polygons(geom, &mut polygons);
        if polygons.is_empty() {
            continue; // non-polygon geometry: nothing to aggregate
        }
        let reprojected = polygons
            .iter()
            .map(|p| reproject_polygon(p, src_epsg, dst_epsg))
            .collect::<Result<Vec<_>, _>>()?;
        zones.push(Zone {
            id,
            polygons: reprojected,
        });
    }
    Ok(zones)
}

/// Reads the id attribute of a feature, stringifying whatever type it holds.
fn feature_id(
    feature: &surtgis_core::vector::Feature,
    id_field: &str,
    idx: usize,
) -> Result<String, StackError> {
    match feature.get_property(id_field) {
        Some(AttributeValue::Int(v)) => Ok(v.to_string()),
        Some(AttributeValue::Float(v)) => Ok(v.to_string()),
        Some(AttributeValue::String(s)) => Ok(s.clone()),
        Some(AttributeValue::Bool(v)) => Ok(v.to_string()),
        Some(AttributeValue::Null) | None => Err(StackError::Zonal(format!(
            "feature {idx} has no usable value for id field '{id_field}'"
        ))),
    }
}

/// Flattens `Polygon`/`MultiPolygon` geometries into a polygon list; other
/// geometry types contribute nothing.
fn collect_polygons(geom: &Geometry<f64>, out: &mut Vec<Polygon<f64>>) {
    match geom {
        Geometry::Polygon(p) => out.push(p.clone()),
        Geometry::MultiPolygon(mp) => out.extend(mp.0.iter().cloned()),
        Geometry::GeometryCollection(gc) => {
            for g in &gc.0 {
                collect_polygons(g, out);
            }
        }
        _ => {}
    }
}

/// Reprojects a polygon from `src` to `dst` CRS (pure-Rust WGS84↔UTM /
/// UTM↔UTM). Identity when the CRSs match or are unknown.
fn reproject_polygon(
    poly: &Polygon<f64>,
    src: Option<u32>,
    dst: Option<u32>,
) -> Result<Polygon<f64>, StackError> {
    if src == dst || src.is_none() || dst.is_none() {
        return Ok(poly.clone());
    }
    let (src, dst) = (src.unwrap(), dst.unwrap());
    let exterior = reproject_ring(poly.exterior(), src, dst)?;
    let interiors = poly
        .interiors()
        .iter()
        .map(|r| reproject_ring(r, src, dst))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Polygon::new(exterior, interiors))
}

fn reproject_ring(
    ring: &LineString<f64>,
    src: u32,
    dst: u32,
) -> Result<LineString<f64>, StackError> {
    let coords = ring
        .coords()
        .map(|coord| reproject_coord(coord.x, coord.y, src, dst).map(|(x, y)| Coord { x, y }))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LineString::new(coords))
}

/// Reprojects one coordinate between the CRS pairs the engine supports without
/// GDAL: WGS84↔UTM and UTM↔UTM. For a WGS84 point, `x` is longitude and `y`
/// latitude; for UTM, easting and northing.
fn reproject_coord(x: f64, y: f64, src: u32, dst: u32) -> Result<(f64, f64), StackError> {
    if src == dst {
        return Ok((x, y));
    }
    let src_utm = reproject::parse_utm_epsg(src);
    let dst_utm = reproject::parse_utm_epsg(dst);
    match (
        reproject::is_wgs84(src),
        src_utm,
        reproject::is_wgs84(dst),
        dst_utm,
    ) {
        // WGS84 -> UTM
        (true, _, false, Some((zone, north))) => Ok(reproject::wgs84_to_utm(x, y, zone, north)),
        // UTM -> WGS84
        (false, Some((zone, north)), true, _) => Ok(reproject::utm_to_wgs84(x, y, zone, north)),
        // UTM -> UTM
        (false, Some((sz, sn)), false, Some((dz, dn))) => {
            Ok(reproject::reproject_utm_to_utm(x, y, sz, sn, dz, dn))
        }
        _ => Err(StackError::Zonal(format!(
            "reprojection EPSG:{src} -> EPSG:{dst} is not supported without GDAL \
             (only WGS84↔UTM and UTM↔UTM); reproject the vector to the cube CRS first"
        ))),
    }
}

/// Rasterizes a zone onto the grid, returning `(row, col, weight)` for every
/// covered pixel (`weight > 0`).
fn rasterize_zone(
    polygons: &[Polygon<f64>],
    grid: &Grid,
    inclusion: PixelInclusion,
) -> Vec<(usize, usize, f64)> {
    // Union of polygon bounding boxes → the only pixels worth testing.
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for poly in polygons {
        for coord in poly.exterior().coords() {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
        }
    }
    if !min_x.is_finite() || !max_x.is_finite() {
        return Vec::new();
    }
    let (r0, r1, c0, c1) = grid.pixel_window(min_x, min_y, max_x, max_y);
    let pixel_area = grid.pixel_area();

    let mut mask = Vec::new();
    for row in r0..r1 {
        for col in c0..c1 {
            let weight = match inclusion {
                PixelInclusion::Center => {
                    let (cx, cy) = grid.center(row, col);
                    let pt = surtgis_core::geo_types::Point::new(cx, cy);
                    if polygons.iter().any(|p| p.contains(&pt)) {
                        1.0
                    } else {
                        0.0
                    }
                }
                PixelInclusion::AllTouched => {
                    let cell = grid.cell_rect(row, col).to_polygon();
                    if polygons.iter().any(|p| p.intersects(&cell)) {
                        1.0
                    } else {
                        0.0
                    }
                }
                PixelInclusion::AreaFraction => {
                    let cell = grid.cell_rect(row, col).to_polygon();
                    let covered: f64 = polygons
                        .iter()
                        .map(|p| cell.intersection(p).unsigned_area())
                        .sum();
                    (covered / pixel_area).clamp(0.0, 1.0)
                }
            };
            if weight > 0.0 {
                mask.push((row, col, weight));
            }
        }
    }
    mask
}

/// Reduces one band over one temporal bin, pooling every covered pixel across
/// every slice in the bin. Returns `(value, n_valid)`.
fn reduce_bin(
    data: &ndarray::ArrayView4<'_, f64>,
    band: usize,
    mask: &[(usize, usize, f64)],
    bin: &[usize],
    reducer: Reducer,
) -> (f64, u64) {
    // Accumulators for the weighted statistics.
    let mut sum_w = 0.0;
    let mut sum_wv = 0.0;
    let mut sum_wv2 = 0.0;
    let mut sum_w_above = 0.0;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut n_valid: u64 = 0;
    // Only Median needs the raw values.
    let mut values: Vec<f64> = if matches!(reducer, Reducer::Median) {
        Vec::with_capacity(mask.len() * bin.len())
    } else {
        Vec::new()
    };
    let threshold = if let Reducer::FractionAbove(t) = reducer {
        t
    } else {
        f64::NAN
    };

    for &(row, col, w) in mask {
        for &t in bin {
            let v = data[[band, row, col, t]];
            if !v.is_finite() {
                continue;
            }
            n_valid += 1;
            sum_w += w;
            sum_wv += w * v;
            sum_wv2 += w * v * v;
            if v > threshold {
                sum_w_above += w;
            }
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
            if matches!(reducer, Reducer::Median) {
                values.push(v);
            }
        }
    }

    if n_valid == 0 {
        return (f64::NAN, 0);
    }

    let value = match reducer {
        Reducer::Mean => sum_wv / sum_w,
        Reducer::Sum => sum_wv,
        Reducer::Count => sum_w,
        Reducer::Min => min_v,
        Reducer::Max => max_v,
        Reducer::Std => {
            let mean = sum_wv / sum_w;
            let var = (sum_wv2 / sum_w) - mean * mean;
            var.max(0.0).sqrt()
        }
        Reducer::FractionAbove(_) => sum_w_above / sum_w,
        Reducer::Median => median(&mut values),
    };
    (value, n_valid)
}

/// Median of finite values (input reordered). Empty → NaN.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datacube_core::GeoRef;
    use ndarray::Array4;
    use surtgis_core::crs::CRS;
    use surtgis_core::geo_types::{Geometry, LineString, Polygon};
    use surtgis_core::vector::Feature;

    /// A 1-band cube on a 4×4 grid at 10 m, origin (0, 40) north-up (UTM 19S),
    /// values = row*10 + col across a single time step.
    fn grid_cube(nt: usize) -> Cube {
        let (ny, nx) = (4, 4);
        let mut data = Array4::from_elem((1, ny, nx, nt), f64::NAN);
        for row in 0..ny {
            for col in 0..nx {
                for t in 0..nt {
                    data[[0, row, col, t]] = (row * 10 + col) as f64;
                }
            }
        }
        let time: Vec<f64> = (0..nt).map(|t| 2024.0 + t as f64).collect();
        Cube::new(data, time, vec!["b1".into()])
            .unwrap()
            .with_georef(GeoRef {
                epsg: Some(32719),
                // GDAL: [origin_x, px_w, 0, origin_y, 0, px_h]
                transform: Some([0.0, 10.0, 0.0, 40.0, 0.0, -10.0]),
            })
    }

    /// A one-feature collection with a single square polygon (cube CRS).
    fn square_zone(id: &str, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> FeatureCollection {
        let ring = LineString::from(vec![
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]);
        let poly = Polygon::new(ring, vec![]);
        let mut feature = Feature::new(Geometry::Polygon(poly));
        feature.set_property("ID", AttributeValue::String(id.to_string()));
        let mut fc = FeatureCollection::with_crs(Some(CRS::from_epsg(32719)));
        fc.push(feature);
        fc
    }

    fn cfg(inclusion: PixelInclusion, reducer: Reducer) -> ZonalConfig {
        ZonalConfig::new("ID", inclusion, reducer)
    }

    #[test]
    fn center_mean_over_known_square() {
        // Square [5,35]x[5,35] (world). North-up: y=40 at top row.
        // Pixel centres inside: cols 1,2 (x=15,25), rows 1,2 (y=25,15).
        // → 2x2 block: rows 1,2, cols 1,2. values = row*10+col.
        // {11,12,21,22} mean = 16.5, count = 4.
        let cube = grid_cube(1);
        let fc = square_zone("w1", 5.0, 5.0, 35.0, 35.0);
        let table = zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)).unwrap();
        assert_eq!(table.rows.len(), 1);
        let r = &table.rows[0];
        assert_eq!(r.polygon_id, "w1");
        assert!((r.value - 16.5).abs() < 1e-12, "value {}", r.value);
        assert_eq!(r.n_valid, 4);
        assert_eq!(r.n_total, 4);

        let count = zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Count)).unwrap();
        assert_eq!(count.rows[0].value, 4.0);
    }

    #[test]
    fn all_touched_includes_more_border_pixels_than_center() {
        // A square snapped to exactly cover cells: [10,30]x[10,30].
        // Center rule: centres at 15,25 inside → cols 1,2 rows 1,2 = 4 pixels.
        // All-touched: the polygon edges lie on grid lines 10/30, touching
        // rows/cols 0..3 → strictly more pixels.
        let cube = grid_cube(1);
        let fc = square_zone("w", 10.0, 10.0, 30.0, 30.0);
        let center =
            zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Count)).unwrap();
        let touched =
            zonal_reduce(&cube, &fc, &cfg(PixelInclusion::AllTouched, Reducer::Count)).unwrap();
        assert!(
            touched.rows[0].value > center.rows[0].value,
            "all_touched {} should exceed center {}",
            touched.rows[0].value,
            center.rows[0].value
        );
    }

    #[test]
    fn area_fraction_half_covered_pixel() {
        // Square covering cols 1,2 fully (x in [10,30]) but only the top half
        // of rows: y in [25,40] → row 0 full, row 1 half (y 20..30 → covered
        // 25..30 = 0.5). Check a partial weight appears and mean is weighted.
        let cube = grid_cube(1);
        let fc = square_zone("w", 10.0, 25.0, 30.0, 40.0);
        let table = zonal_reduce(
            &cube,
            &fc,
            &cfg(PixelInclusion::AreaFraction, Reducer::Count),
        )
        .unwrap();
        // Count = Σ weights. Row 0 (y 30..40) cols 1,2 fully covered (w=1 each).
        // Row 1 (y 20..30) cols 1,2 half covered (w=0.5 each). → 2 + 1 = 3.
        assert!(
            (table.rows[0].value - 3.0).abs() < 1e-9,
            "weighted count {}",
            table.rows[0].value
        );
    }

    #[test]
    fn nan_pixels_excluded_and_reflected_in_n_valid() {
        let mut cube = grid_cube(1);
        // Poke a NaN into pixel (row 1, col 1) which the centre square covers.
        {
            let data = cube.data();
            assert_eq!(data[[0, 1, 1, 0]], 11.0);
        }
        // Rebuild with a NaN there.
        let mut arr = cube.data().to_owned();
        arr[[0, 1, 1, 0]] = f64::NAN;
        cube = Cube::new(arr, cube.time().to_vec(), cube.bands().to_vec())
            .unwrap()
            .with_georef(cube.georef().unwrap());
        let fc = square_zone("w", 5.0, 5.0, 35.0, 35.0);
        let table = zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)).unwrap();
        let r = &table.rows[0];
        // Mask still 4 pixels, but one is NaN.
        assert_eq!(r.n_total, 4);
        assert_eq!(r.n_valid, 3);
        // mean of {12, 21, 22} = 18.333...
        assert!((r.value - (12.0 + 21.0 + 22.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn multi_polygon_and_temporal_binning() {
        // Two time steps in 2024 and 2025; yearly window → 2 bins per zone.
        let cube = grid_cube(2); // times 2024, 2025
        let fc = square_zone("w1", 5.0, 5.0, 35.0, 35.0);
        let table = zonal_reduce(
            &cube,
            &fc,
            &cfg(PixelInclusion::Center, Reducer::Mean).window(CompositeWindow::CalendarYear),
        )
        .unwrap();
        // 1 zone × 2 year-bins × 1 band = 2 rows; both means 16.5 (same data).
        assert_eq!(table.rows.len(), 2);
        assert!((table.rows[0].value - 16.5).abs() < 1e-12);
        assert!((table.rows[1].value - 16.5).abs() < 1e-12);
        assert_eq!(table.rows[0].n_total, 4);
    }

    #[test]
    fn reprojects_polygon_from_wgs84() {
        // Cube in UTM 19S; polygon given in WGS84 (EPSG:4326). The zonal
        // reducer must reproject the polygon, not the raster, and land on the
        // right pixels. We place the cube at a real UTM origin and derive the
        // WGS84 square by inverse-projecting the target extent.
        let (ny, nx) = (4, 4);
        let mut data = Array4::from_elem((1, ny, nx, 1), 0.0);
        for row in 0..ny {
            for col in 0..nx {
                data[[0, row, col, 0]] = (row * 10 + col) as f64;
            }
        }
        // origin easting 300000, northing 6300000, 10 m pixels.
        let (ox, oy, res) = (300_000.0, 6_300_000.0, 10.0);
        let cube = Cube::new(data, vec![2024.0], vec!["b1".into()])
            .unwrap()
            .with_georef(GeoRef {
                epsg: Some(32719),
                transform: Some([ox, res, 0.0, oy, 0.0, -res]),
            });
        // Target UTM square covering the 2x2 centre block (cols 1,2 rows 1,2)
        // with margin, so sub-millimetre reprojection round-trip error can't
        // flip a pixel centre that sits exactly on a cell boundary.
        let (uz, un) = reproject::parse_utm_epsg(32719).unwrap();
        let corners_utm = [
            (ox + 12.0, oy - 28.0),
            (ox + 28.0, oy - 28.0),
            (ox + 28.0, oy - 12.0),
            (ox + 12.0, oy - 12.0),
        ];
        let ring_ll: Vec<(f64, f64)> = corners_utm
            .iter()
            .map(|&(e, n)| reproject::utm_to_wgs84(e, n, uz, un)) // (lon, lat)
            .collect();
        let mut coords = ring_ll.clone();
        coords.push(ring_ll[0]);
        let poly = Polygon::new(LineString::from(coords), vec![]);
        let mut feature = Feature::new(Geometry::Polygon(poly));
        feature.set_property("ID", AttributeValue::Int(7));
        let mut fc = FeatureCollection::with_crs(Some(CRS::from_epsg(4326)));
        fc.push(feature);

        let table = zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)).unwrap();
        let r = &table.rows[0];
        assert_eq!(r.polygon_id, "7");
        // Same 2x2 block {11,12,21,22} → 16.5 (reprojection round-trips to the
        // right pixels within sub-pixel accuracy).
        assert!((r.value - 16.5).abs() < 1e-9, "value {}", r.value);
        assert_eq!(r.n_valid, 4);
    }

    #[test]
    fn rejects_rotated_grid_and_missing_georef() {
        let plain = Cube::new(
            Array4::from_elem((1, 2, 2, 1), 1.0),
            vec![2024.0],
            vec!["b".into()],
        )
        .unwrap();
        let fc = square_zone("w", 0.0, 0.0, 1.0, 1.0);
        assert!(matches!(
            zonal_reduce(&plain, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)),
            Err(StackError::Zonal(_))
        ));

        let rotated = plain.with_georef(GeoRef {
            epsg: Some(32719),
            transform: Some([0.0, 10.0, 1.0, 0.0, 1.0, -10.0]), // c,e != 0
        });
        assert!(matches!(
            zonal_reduce(&rotated, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)),
            Err(StackError::Zonal(_))
        ));
    }

    #[test]
    fn missing_id_field_is_an_error() {
        let cube = grid_cube(1);
        let mut fc = square_zone("w", 5.0, 5.0, 35.0, 35.0);
        // Wrong id field name.
        let bad = ZonalConfig::new("NOPE", PixelInclusion::Center, Reducer::Mean);
        assert!(matches!(
            zonal_reduce(&cube, &fc, &bad),
            Err(StackError::Zonal(_))
        ));
        // touch fc so it is used
        fc.set_crs(Some(CRS::from_epsg(32719)));
    }

    #[test]
    fn fraction_above_threshold() {
        let cube = grid_cube(1);
        let fc = square_zone("w", 5.0, 5.0, 35.0, 35.0);
        // block values {11,12,21,22}; fraction above 15 = {21,22} / 4 = 0.5.
        let table = zonal_reduce(
            &cube,
            &fc,
            &cfg(PixelInclusion::Center, Reducer::FractionAbove(15.0)),
        )
        .unwrap();
        assert!((table.rows[0].value - 0.5).abs() < 1e-12);
        assert_eq!(table.rows[0].reducer, "fraction_above");
    }

    #[test]
    fn csv_roundtrip_shape() {
        let cube = grid_cube(1);
        let fc = square_zone("w1", 5.0, 5.0, 35.0, 35.0);
        let table = zonal_reduce(&cube, &fc, &cfg(PixelInclusion::Center, Reducer::Mean)).unwrap();
        let csv = table.to_csv_string();
        assert!(csv.starts_with("polygon_id,time,band,reducer,value,n_valid,n_total\n"));
        assert_eq!(csv.lines().count(), 2); // header + 1 row
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_roundtrip() {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::RowAccessor;

        let cube = grid_cube(2); // two yearly bins
        let fc = square_zone("w1", 5.0, 5.0, 35.0, 35.0);
        let table = zonal_reduce(
            &cube,
            &fc,
            &cfg(PixelInclusion::Center, Reducer::Mean).window(CompositeWindow::CalendarYear),
        )
        .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join(format!("datacube_zonal_{}.parquet", std::process::id()));
        table.write_parquet(&path).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        assert_eq!(
            reader.metadata().file_metadata().num_rows() as usize,
            table.rows.len()
        );
        let first = reader.get_row_iter(None).unwrap().next().unwrap().unwrap();
        assert_eq!(first.get_string(0).unwrap(), "w1");
        assert_eq!(first.get_string(2).unwrap(), "b1");
        assert_eq!(first.get_string(3).unwrap(), "mean");
        assert!((first.get_double(4).unwrap() - 16.5).abs() < 1e-12);
        assert_eq!(first.get_long(5).unwrap(), 4); // n_valid
        assert_eq!(first.get_long(6).unwrap(), 4); // n_total

        let _ = std::fs::remove_file(&path);
    }
}
