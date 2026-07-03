//! # datacube-zarr
//!
//! Zarr V3 backing store for [`datacube_core::Cube`]: a native, cloud-ready
//! serialization of a temporal data cube. The cube is written as a single
//! 4-D array `(band, y, x, time)` with named dimensions, plus the time
//! coordinates, band labels and optional georeference (EPSG + affine
//! geotransform) carried in the array attributes. Storage defaults to
//! zstd-compressed `f64`; [`ZarrOptions`] can opt into `f32` on disk while
//! the in-memory [`Cube`] stays `f64`.
//!
//! This is the cloud-native ARD direction for datacube-rs (Sentinel/Landsat
//! are adopting GeoZarr): a Rust cube that reads and writes Zarr V3 directly,
//! without a Python/C++ stack. [`read_zarr_chunked`] and [`ZarrCubeWriter`]
//! stream a store one spatial tile at a time, so processing a cube is
//! bounded by tile size rather than the whole store's size — the primitive
//! behind "cubo Rust nativo sobre GeoZarr" executing on modest hardware.
//! Full GeoZarr CF conventions (separate coordinate variables, `grid_mapping`)
//! are a planned refinement; this layer round-trips a cube losslessly (at
//! `f64`) and stores enough metadata to relocate it in space and time.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use datacube_core::Cube;
pub use datacube_core::GeoRef;
use ndarray::{Array4, ArrayView4};
use thiserror::Error;
use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

/// Path of the cube array within the Zarr group.
const CUBE_PATH: &str = "/cube";
/// Default spatial chunk edge (pixels); bands and time are single-chunked.
const DEFAULT_CHUNK: u64 = 256;

/// On-disk element type for a Zarr-backed cube (see [`ZarrOptions::dtype`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrDType {
    /// Store as `f64`, matching the in-memory [`Cube`] exactly (lossless).
    F64,
    /// Store as `f32`, halving the array size on disk/object storage; values
    /// are downcast on write and upcast back to `f64` on read (the in-memory
    /// model stays `f64`, per the crate's cloud-native-ARD scale/precision
    /// tradeoff — reflectances and indices comfortably fit `f32`).
    F32,
}

/// Options for [`write_zarr_with_options`] and [`ZarrCubeWriter::create`].
#[derive(Debug, Clone, Copy)]
pub struct ZarrOptions {
    /// zstd compression level (1-22), or `None` to store uncompressed.
    /// Compression is lossless and transparent to readers (zarr-python/
    /// xarray decode it automatically); there is no reason to disable it
    /// outside of benchmarking.
    pub compression_level: Option<i32>,
    /// On-disk element type.
    pub dtype: ZarrDType,
}

impl Default for ZarrOptions {
    /// zstd level 5, `f64` (lossless — matches [`write_zarr`]'s prior
    /// uncompressed-`f64` behavior except for the (transparent) compression).
    fn default() -> Self {
        Self {
            compression_level: Some(5),
            dtype: ZarrDType::F64,
        }
    }
}

/// Errors from reading or writing a Zarr-backed cube.
#[derive(Debug, Error)]
pub enum ZarrError {
    #[error("zarr store error: {0}")]
    Store(String),
    #[error("zarr array error: {0}")]
    Array(String),
    #[error("malformed cube metadata: {0}")]
    Metadata(String),
    #[error(transparent)]
    Cube(#[from] datacube_core::CubeError),
}

/// Writes `cube` to a new Zarr V3 store rooted at `path` with the default
/// [`ZarrOptions`] (zstd-compressed `f64`); `geo` is persisted in the array
/// attributes alongside the band labels and time coordinates.
///
/// `geo` is taken explicitly (rather than read from `cube.georef()`) so a
/// store can be relocated in space without re-attaching anything to the
/// cube first; pass `cube.georef().unwrap_or_default()` to persist whatever
/// georeference the cube already carries. See [`write_zarr_with_options`]
/// for compression/dtype control.
pub fn write_zarr(cube: &Cube, path: &Path, geo: &GeoRef) -> Result<(), ZarrError> {
    write_zarr_with_options(cube, path, geo, ZarrOptions::default())
}

/// Like [`write_zarr`], with explicit [`ZarrOptions`] (compression level,
/// on-disk dtype).
pub fn write_zarr_with_options(
    cube: &Cube,
    path: &Path,
    geo: &GeoRef,
    options: ZarrOptions,
) -> Result<(), ZarrError> {
    let array = create_array(path, cube.dims(), cube.bands(), cube.time(), geo, options)?;
    store_region(&array, options.dtype, [0, 0, 0, 0], cube.data())
}

/// Reads a Zarr V3 cube written by [`write_zarr`]/[`write_zarr_with_options`]
/// back into a [`Cube`] and its georeference; the same `GeoRef` is also
/// attached to the returned cube (`cube.georef()`) when it carries an EPSG or
/// transform. The on-disk dtype (`f32` or `f64`) is detected automatically.
pub fn read_zarr(path: &Path) -> Result<(Cube, GeoRef), ZarrError> {
    let store = Arc::new(FilesystemStore::new(path).map_err(|e| ZarrError::Store(e.to_string()))?);
    let array = Array::open(store, CUBE_PATH).map_err(|e| ZarrError::Array(e.to_string()))?;
    let (bands, time, geo, dims) = read_meta(&array)?;
    let dtype = dtype_of(&array)?;

    let (nb, ny, nx, nt) = dims;
    let subset =
        ArraySubset::new_with_ranges(&[0..nb as u64, 0..ny as u64, 0..nx as u64, 0..nt as u64]);
    let flat = retrieve_region(&array, dtype, &subset)?;
    let data =
        Array4::from_shape_vec(dims, flat).map_err(|e| ZarrError::Metadata(e.to_string()))?;
    let mut cube = Cube::new(data, time, bands)?;
    if geo.epsg.is_some() || geo.transform.is_some() {
        cube = cube.with_georef(geo);
    }
    Ok((cube, geo))
}

/// Position of a tile yielded by [`read_zarr_chunked`], as a pixel offset
/// from the array's spatial origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPos {
    pub y0: usize,
    pub x0: usize,
}

/// Reads a Zarr V3 cube one spatial tile at a time, bounding memory to a
/// single tile regardless of the store's total size — the read-side
/// counterpart of [`ZarrCubeWriter`].
///
/// Every band and the full time axis are read for each `chunk_y` × `chunk_x`
/// tile (edge tiles are smaller); a tile's [`GeoRef`] has its transform
/// origin shifted to the tile's own top-left corner, so per-tile output
/// (e.g. a trend map) can be written out already georeferenced. This is the
/// primitive behind bounded-memory execution over a GeoZarr store:
///
/// ```no_run
/// # use datacube_zarr::read_zarr_chunked;
/// # use std::path::Path;
/// for tile in read_zarr_chunked(Path::new("cube.zarr"), 256, 256).unwrap() {
///     let (cube, _pos) = tile.unwrap();
///     // only this tile is resident in memory
///     let _ = cube.par_map_series(0, |t, y| y.len() as f64 * t.len() as f64);
/// }
/// ```
pub fn read_zarr_chunked(
    path: &Path,
    chunk_y: usize,
    chunk_x: usize,
) -> Result<impl Iterator<Item = Result<(Cube, ChunkPos), ZarrError>>, ZarrError> {
    if chunk_y == 0 || chunk_x == 0 {
        return Err(ZarrError::Metadata(format!(
            "chunk sizes must be > 0, got ({chunk_y}, {chunk_x})"
        )));
    }
    let store = Arc::new(FilesystemStore::new(path).map_err(|e| ZarrError::Store(e.to_string()))?);
    let array = Array::open(store, CUBE_PATH).map_err(|e| ZarrError::Array(e.to_string()))?;
    let (bands, time, geo, dims) = read_meta(&array)?;
    let dtype = dtype_of(&array)?;
    let (nb, ny, nx, nt) = dims;

    let tiles: Vec<(usize, usize)> = (0..ny)
        .step_by(chunk_y)
        .flat_map(|y0| (0..nx).step_by(chunk_x).map(move |x0| (y0, x0)))
        .collect();

    Ok(tiles.into_iter().map(move |(y0, x0)| {
        let y1 = (y0 + chunk_y).min(ny);
        let x1 = (x0 + chunk_x).min(nx);
        let subset = ArraySubset::new_with_ranges(&[
            0..nb as u64,
            y0 as u64..y1 as u64,
            x0 as u64..x1 as u64,
            0..nt as u64,
        ]);
        let flat = retrieve_region(&array, dtype, &subset)?;
        let tile_dims = (nb, y1 - y0, x1 - x0, nt);
        let data = Array4::from_shape_vec(tile_dims, flat)
            .map_err(|e| ZarrError::Metadata(e.to_string()))?;
        let mut cube = Cube::new(data, time.clone(), bands.clone())?;
        let tile_geo = shift_georef(&geo, x0, y0);
        if tile_geo.epsg.is_some() || tile_geo.transform.is_some() {
            cube = cube.with_georef(tile_geo);
        }
        Ok((cube, ChunkPos { y0, x0 }))
    }))
}

/// Incrementally writes a large cube to a new Zarr V3 store one spatial tile
/// at a time, bounding write-side memory the same way [`read_zarr_chunked`]
/// bounds reads — the write-side counterpart, for producing a store without
/// ever materializing the whole cube (e.g. writing per-chunk trend maps or
/// stacking scenes region by region).
///
/// The full dimensions (bands, spatial extent, time steps) and coordinates
/// must be known upfront — only the tiling of the *write* is streamed, not
/// the shape. Regions never written keep the array's fill value (`NaN`).
pub struct ZarrCubeWriter {
    array: Array<FilesystemStore>,
    dtype: ZarrDType,
}

impl ZarrCubeWriter {
    /// Creates a new (empty, fill-valued) Zarr V3 store with the given
    /// shape, ready for [`write_chunk`](Self::write_chunk) calls.
    pub fn create(
        path: &Path,
        dims: (usize, usize, usize, usize),
        bands: &[String],
        time: &[f64],
        geo: &GeoRef,
        options: ZarrOptions,
    ) -> Result<Self, ZarrError> {
        let array = create_array(path, dims, bands, time, geo, options)?;
        Ok(Self {
            array,
            dtype: options.dtype,
        })
    }

    /// Writes `chunk` (all bands, all times, a `(y0, x0)`-offset spatial
    /// tile) into the store. Tiles may be written in any order, more than
    /// once, and need not align with the underlying Zarr chunk grid — zarrs
    /// re-encodes only the chunks the write touches.
    pub fn write_chunk(
        &self,
        chunk: ArrayView4<'_, f64>,
        y0: usize,
        x0: usize,
    ) -> Result<(), ZarrError> {
        store_region(&self.array, self.dtype, [0, y0 as u64, x0 as u64, 0], chunk)
    }
}

/// Builds and persists the array + group metadata for a new store; no cell
/// data is written. Shared by [`write_zarr_with_options`] and
/// [`ZarrCubeWriter::create`].
fn create_array(
    path: &Path,
    dims: (usize, usize, usize, usize),
    bands: &[String],
    time: &[f64],
    geo: &GeoRef,
    options: ZarrOptions,
) -> Result<Array<FilesystemStore>, ZarrError> {
    let (nb, ny, nx, nt) = dims;
    let store = Arc::new(FilesystemStore::new(path).map_err(|e| ZarrError::Store(e.to_string()))?);

    // root group so the store opens as a Zarr V3 group (xarray/zarr-python)
    let root = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| ZarrError::Store(e.to_string()))?;
    root.store_metadata()
        .map_err(|e| ZarrError::Store(e.to_string()))?;

    let shape: Vec<u64> = vec![nb as u64, ny as u64, nx as u64, nt as u64];
    let chunk: Vec<u64> = vec![
        nb as u64,
        (ny as u64).clamp(1, DEFAULT_CHUNK),
        (nx as u64).clamp(1, DEFAULT_CHUNK),
        nt as u64,
    ];

    let mut attrs = serde_json::Map::new();
    attrs.insert("bands".into(), serde_json::to_value(bands).unwrap());
    attrs.insert("time".into(), serde_json::to_value(time).unwrap());
    if let Some(epsg) = geo.epsg {
        attrs.insert("epsg".into(), serde_json::json!(epsg));
    }
    if let Some(t) = geo.transform {
        attrs.insert("geotransform".into(), serde_json::json!(t));
    }

    // ArrayBuilder::new(shape, chunk_shape, data_type, fill_value)
    let mut builder = match options.dtype {
        ZarrDType::F64 => ArrayBuilder::new(shape, chunk, data_type::float64(), f64::NAN),
        ZarrDType::F32 => ArrayBuilder::new(shape, chunk, data_type::float32(), f32::NAN),
    };
    builder
        .dimension_names(Some(["band", "y", "x", "time"]))
        .attributes(attrs);
    if let Some(level) = options.compression_level {
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(level, false))]);
    }
    let array = builder
        .build(store, CUBE_PATH)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    array
        .store_metadata()
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    Ok(array)
}

/// Writes `view` (in the cube's `(band, y, x, time)` element order) into the
/// array at the given start offset, converting to `f32` first if `dtype`
/// calls for it. `view` need not be contiguous (spatial-tile sub-views from
/// [`Cube::chunks`](datacube_core::Cube::chunks) aren't).
fn store_region(
    array: &Array<FilesystemStore>,
    dtype: ZarrDType,
    start: [u64; 4],
    view: ArrayView4<'_, f64>,
) -> Result<(), ZarrError> {
    let ranges: Vec<Range<u64>> = start
        .iter()
        .zip(view.shape())
        .map(|(&s, &len)| s..s + len as u64)
        .collect();
    let subset = ArraySubset::new_with_ranges(&ranges);
    match dtype {
        ZarrDType::F64 => match view.as_slice() {
            Some(flat) => array.store_array_subset(&subset, flat),
            None => {
                let flat: Vec<f64> = view.iter().copied().collect();
                array.store_array_subset(&subset, flat.as_slice())
            }
        },
        ZarrDType::F32 => {
            let flat: Vec<f32> = view.iter().map(|&v| v as f32).collect();
            array.store_array_subset(&subset, flat.as_slice())
        }
    }
    .map_err(|e| ZarrError::Array(e.to_string()))
}

/// Reads `subset` back as `f64`, upcasting from `f32` storage if needed.
fn retrieve_region(
    array: &Array<FilesystemStore>,
    dtype: ZarrDType,
    subset: &ArraySubset,
) -> Result<Vec<f64>, ZarrError> {
    match dtype {
        ZarrDType::F64 => array
            .retrieve_array_subset::<Vec<f64>>(subset)
            .map_err(|e| ZarrError::Array(e.to_string())),
        ZarrDType::F32 => {
            let flat: Vec<f32> = array
                .retrieve_array_subset::<Vec<f32>>(subset)
                .map_err(|e| ZarrError::Array(e.to_string()))?;
            Ok(flat.into_iter().map(f64::from).collect())
        }
    }
}

/// The on-disk element type, detected from the array's Zarr data type.
fn dtype_of(array: &Array<FilesystemStore>) -> Result<ZarrDType, ZarrError> {
    let dt = array.data_type();
    if *dt == data_type::float64() {
        Ok(ZarrDType::F64)
    } else if *dt == data_type::float32() {
        Ok(ZarrDType::F32)
    } else {
        Err(ZarrError::Metadata(
            "unsupported array data type (expected float32 or float64)".into(),
        ))
    }
}

/// Bands, time coordinates, georeference and `(band, y, x, time)` dims.
type CubeMeta = (Vec<String>, Vec<f64>, GeoRef, (usize, usize, usize, usize));

/// Bands, time, georeference and `(band, y, x, time)` dims from an opened
/// array's metadata (no cell data read). Shared by [`read_zarr`] and
/// [`read_zarr_chunked`].
fn read_meta(array: &Array<FilesystemStore>) -> Result<CubeMeta, ZarrError> {
    let shape = array.shape();
    if shape.len() != 4 {
        return Err(ZarrError::Metadata(format!(
            "expected a 4-D (band,y,x,time) array, got {} dims",
            shape.len()
        )));
    }
    let dims = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );

    let attrs = array.attributes();
    let bands: Vec<String> = attrs
        .get("bands")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or_else(|| ZarrError::Metadata("missing 'bands' attribute".into()))?;
    let time: Vec<f64> = attrs
        .get("time")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or_else(|| ZarrError::Metadata("missing 'time' attribute".into()))?;
    let geo = GeoRef {
        epsg: attrs.get("epsg").and_then(|v| v.as_u64()).map(|v| v as u32),
        transform: attrs
            .get("geotransform")
            .and_then(|v| serde_json::from_value::<[f64; 6]>(v.clone()).ok()),
    };
    Ok((bands, time, geo, dims))
}

/// Shifts a georeference's transform origin to a tile starting at pixel
/// `(x0, y0)` (GDAL convention: `x = a + col·b + row·c`, `y = d + col·e +
/// row·f`; only `a`/`d` move, the pixel size/rotation terms are unchanged).
fn shift_georef(geo: &GeoRef, x0: usize, y0: usize) -> GeoRef {
    GeoRef {
        epsg: geo.epsg,
        transform: geo.transform.map(|[a, b, c, d, e, f]| {
            let (x0, y0) = (x0 as f64, y0 as f64);
            [a + x0 * b + y0 * c, b, c, d + x0 * e + y0 * f, e, f]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array4, s};

    fn sample_cube() -> Cube {
        // 2 bands, 300x4 px (forces multi-chunk on y), 5 t; with a NaN
        let (nb, ny, nx, nt) = (2, 300, 4, 5);
        let mut data = Array4::zeros((nb, ny, nx, nt));
        for ((b, y, x, t), v) in data.indexed_iter_mut() {
            *v = (b * 1000 + y * 10 + x) as f64 + t as f64 * 0.5;
        }
        data[[1, 7, 2, 3]] = f64::NAN;
        Cube::new(
            data,
            (0..nt).map(|t| 2020.0 + t as f64).collect(),
            vec!["red".into(), "nir".into()],
        )
        .unwrap()
    }

    fn sample_geo() -> GeoRef {
        GeoRef {
            epsg: Some(32719),
            transform: Some([300000.0, 10.0, 0.0, 6200000.0, 0.0, -10.0]),
        }
    }

    #[test]
    fn roundtrip_preserves_cube_and_georef() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.zarr");
        let cube = sample_cube();
        let geo = sample_geo();

        write_zarr(&cube, &path, &geo).unwrap();
        let (back, back_geo) = read_zarr(&path).unwrap();

        assert_eq!(back.dims(), cube.dims());
        assert_eq!(back.bands(), cube.bands());
        assert_eq!(back.time(), cube.time());
        assert_eq!(back_geo, geo);
        assert_eq!(back.georef(), Some(geo));

        let (a, b) = (cube.data(), back.data());
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn read_missing_array_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_zarr(&dir.path().join("nope.zarr"));
        assert!(matches!(err, Err(ZarrError::Array(_))));
    }

    #[test]
    fn f32_roundtrip_is_detected_and_upcast() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube_f32.zarr");
        let cube = sample_cube(); // values are small integers/.5 -> exact in f32
        let options = ZarrOptions {
            compression_level: Some(9),
            dtype: ZarrDType::F32,
        };

        write_zarr_with_options(&cube, &path, &sample_geo(), options).unwrap();
        let (back, _) = read_zarr(&path).unwrap();

        assert_eq!(back.dims(), cube.dims());
        for (x, y) in cube.data().iter().zip(back.data().iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_abs_diff_eq!(x, y, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn uncompressed_and_compressed_stores_read_back_identically() {
        let dir = tempfile::tempdir().unwrap();
        let cube = sample_cube();
        let geo = sample_geo();

        let plain = dir.path().join("plain.zarr");
        write_zarr_with_options(
            &cube,
            &plain,
            &geo,
            ZarrOptions {
                compression_level: None,
                dtype: ZarrDType::F64,
            },
        )
        .unwrap();
        let compressed = dir.path().join("compressed.zarr");
        write_zarr_with_options(
            &cube,
            &compressed,
            &geo,
            ZarrOptions {
                compression_level: Some(19),
                dtype: ZarrDType::F64,
            },
        )
        .unwrap();

        let (a, _) = read_zarr(&plain).unwrap();
        let (b, _) = read_zarr(&compressed).unwrap();
        for (x, y) in a.data().iter().zip(b.data().iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn chunked_read_reassembles_into_the_full_cube() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.zarr");
        let cube = sample_cube(); // 300x4 px -> chunk_y=128 yields 3 y-tiles, 1 x-tile
        write_zarr(&cube, &path, &sample_geo()).unwrap();

        let tiles: Vec<_> = read_zarr_chunked(&path, 128, 128)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tiles.len(), 3); // ceil(300/128) * ceil(4/128)

        let mut reassembled = Array4::from_elem(cube.dims(), f64::NAN);
        for (chunk, pos) in &tiles {
            let (_, cy, cx, _) = chunk.dims();
            reassembled
                .slice_mut(s![.., pos.y0..pos.y0 + cy, pos.x0..pos.x0 + cx, ..])
                .assign(&chunk.data());
        }
        for (x, y) in cube.data().iter().zip(reassembled.iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn chunked_read_shifts_georef_transform_per_tile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.zarr");
        let cube = sample_cube();
        let geo = sample_geo(); // pixel size 10, origin (300000, 6200000)
        write_zarr(&cube, &path, &geo).unwrap();

        let tiles: Vec<_> = read_zarr_chunked(&path, 128, 128)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let (_, second_tile) = &tiles[1]; // y0 = 128, x0 = 0
        assert_eq!(second_tile.y0, 128);
        let second_cube = &tiles[1].0;
        let t = second_cube.georef().unwrap().transform.unwrap();
        assert_abs_diff_eq!(t[0], 300000.0, epsilon = 1e-9); // x origin unshifted (x0=0)
        assert_abs_diff_eq!(t[3], 6200000.0 - 128.0 * 10.0, epsilon = 1e-9); // y origin shifts
        assert_eq!(second_cube.georef().unwrap().epsg, geo.epsg);
    }

    #[test]
    fn chunk_writer_matches_whole_cube_write() {
        let dir = tempfile::tempdir().unwrap();
        let cube = sample_cube();
        let geo = sample_geo();

        let whole = dir.path().join("whole.zarr");
        write_zarr(&cube, &whole, &geo).unwrap();

        let chunked = dir.path().join("chunked.zarr");
        let writer = ZarrCubeWriter::create(
            &chunked,
            cube.dims(),
            cube.bands(),
            cube.time(),
            &geo,
            ZarrOptions::default(),
        )
        .unwrap();
        let (_, ny, nx, _) = cube.dims();
        for y0 in (0..ny).step_by(128) {
            for x0 in (0..nx).step_by(128) {
                let y1 = (y0 + 128).min(ny);
                let x1 = (x0 + 128).min(nx);
                let tile = cube.data().slice(s![.., y0..y1, x0..x1, ..]).to_owned();
                writer.write_chunk(tile.view(), y0, x0).unwrap();
            }
        }

        let (a, _) = read_zarr(&whole).unwrap();
        let (b, _) = read_zarr(&chunked).unwrap();
        assert_eq!(a.dims(), b.dims());
        for (x, y) in a.data().iter().zip(b.data().iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn read_zarr_chunked_rejects_zero_chunk_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.zarr");
        write_zarr(&sample_cube(), &path, &sample_geo()).unwrap();
        assert!(matches!(
            read_zarr_chunked(&path, 0, 10),
            Err(ZarrError::Metadata(_))
        ));
    }
}
