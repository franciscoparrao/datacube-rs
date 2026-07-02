//! # datacube-zarr
//!
//! Zarr V3 backing store for [`datacube_core::Cube`]: a native, cloud-ready
//! serialization of a temporal data cube. The cube is written as a single
//! 4-D `f64` array `(band, y, x, time)` with named dimensions, plus the time
//! coordinates, band labels and optional georeference (EPSG + affine
//! geotransform) carried in the array attributes.
//!
//! This is the cloud-native ARD direction for datacube-rs (Sentinel/Landsat
//! are adopting GeoZarr): a Rust cube that reads and writes Zarr V3 directly,
//! without a Python/C++ stack. Full GeoZarr CF conventions (separate
//! coordinate variables, `grid_mapping`) are a planned refinement; this layer
//! round-trips a cube losslessly and stores enough metadata to relocate it in
//! space and time.

use std::path::Path;
use std::sync::Arc;

use datacube_core::Cube;
use ndarray::Array4;
use thiserror::Error;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

/// Path of the cube array within the Zarr group.
const CUBE_PATH: &str = "/cube";
/// Default spatial chunk edge (pixels); bands and time are single-chunked.
const DEFAULT_CHUNK: u64 = 256;

/// Optional spatial reference stored alongside the cube.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GeoRef {
    /// EPSG code of the cube CRS.
    pub epsg: Option<u32>,
    /// Affine geotransform `[a, b, c, d, e, f]` mapping pixel `(col, row)` to
    /// world `(x, y)` as `x = a + col·b + row·c`, `y = d + col·e + row·f`.
    pub transform: Option<[f64; 6]>,
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

/// Writes `cube` to a Zarr V3 store rooted at `path`; `geo` is persisted in
/// the array attributes alongside the band labels and time coordinates.
pub fn write_zarr(cube: &Cube, path: &Path, geo: &GeoRef) -> Result<(), ZarrError> {
    let store = Arc::new(FilesystemStore::new(path).map_err(|e| ZarrError::Store(e.to_string()))?);

    // root group so the store opens as a Zarr V3 group (xarray/zarr-python)
    let root = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| ZarrError::Store(e.to_string()))?;
    root.store_metadata()
        .map_err(|e| ZarrError::Store(e.to_string()))?;

    let (nb, ny, nx, nt) = cube.dims();
    let shape: Vec<u64> = vec![nb as u64, ny as u64, nx as u64, nt as u64];
    let chunk: Vec<u64> = vec![
        nb as u64,
        (ny as u64).clamp(1, DEFAULT_CHUNK),
        (nx as u64).clamp(1, DEFAULT_CHUNK),
        nt as u64,
    ];

    let mut attrs = serde_json::Map::new();
    attrs.insert("bands".into(), serde_json::to_value(cube.bands()).unwrap());
    attrs.insert("time".into(), serde_json::to_value(cube.time()).unwrap());
    if let Some(epsg) = geo.epsg {
        attrs.insert("epsg".into(), serde_json::json!(epsg));
    }
    if let Some(t) = geo.transform {
        attrs.insert("geotransform".into(), serde_json::json!(t));
    }

    // ArrayBuilder::new(shape, chunk_shape, data_type, fill_value)
    let mut builder = ArrayBuilder::new(shape.clone(), chunk, data_type::float64(), f64::NAN);
    builder
        .dimension_names(Some(["band", "y", "x", "time"]))
        .attributes(attrs);
    let array = builder
        .build(store.clone(), CUBE_PATH)
        .map_err(|e| ZarrError::Array(e.to_string()))?;

    array
        .store_metadata()
        .map_err(|e| ZarrError::Array(e.to_string()))?;

    // The cube is standard layout (band,y,x,time) row-major, matching the Zarr
    // C-order array, so the flat slice maps 1:1 to the whole-array subset.
    let view = cube.data();
    let flat = view
        .as_slice()
        .expect("cube data is standard layout (enforced by Cube::new)");
    let subset = ArraySubset::new_with_shape(shape);
    array
        .store_array_subset(&subset, flat)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    Ok(())
}

/// Reads a Zarr V3 cube written by [`write_zarr`] back into a [`Cube`] and its
/// georeference.
pub fn read_zarr(path: &Path) -> Result<(Cube, GeoRef), ZarrError> {
    let store = Arc::new(FilesystemStore::new(path).map_err(|e| ZarrError::Store(e.to_string()))?);
    let array = Array::open(store, CUBE_PATH).map_err(|e| ZarrError::Array(e.to_string()))?;

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

    let subset = ArraySubset::new_with_shape(shape.to_vec());
    let flat: Vec<f64> = array
        .retrieve_array_subset::<Vec<f64>>(&subset)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    let data =
        Array4::from_shape_vec(dims, flat).map_err(|e| ZarrError::Metadata(e.to_string()))?;
    let cube = Cube::new(data, time, bands)?;
    Ok((cube, geo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

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

    #[test]
    fn roundtrip_preserves_cube_and_georef() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cube.zarr");
        let cube = sample_cube();
        let geo = GeoRef {
            epsg: Some(32719),
            transform: Some([300000.0, 10.0, 0.0, 6200000.0, 0.0, -10.0]),
        };

        write_zarr(&cube, &path, &geo).unwrap();
        let (back, back_geo) = read_zarr(&path).unwrap();

        assert_eq!(back.dims(), cube.dims());
        assert_eq!(back.bands(), cube.bands());
        assert_eq!(back.time(), cube.time());
        assert_eq!(back_geo, geo);

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
}
