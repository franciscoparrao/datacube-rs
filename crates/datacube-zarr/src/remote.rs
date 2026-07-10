//! S3/HTTP (or any [`object_store::ObjectStore`]) backing store, via zarrs'
//! async storage API.
//!
//! Every public function here is synchronous — like the rest of this crate,
//! and like `datacube-io`'s STAC/COG blocking API (`StacClientBlocking`),
//! which uses the same shared-runtime-behind-a-sync-API pattern for the
//! same reason: nothing else in datacube-rs (core, CLI, PyO3) wants to be
//! async, so the async I/O this needs stays an implementation detail behind
//! [`shared_runtime`].
//!
//! The store itself is *not* something this crate builds for you — `object_store`
//! (re-exported here) already has ergonomic per-backend builders
//! (`object_store::aws::AmazonS3Builder`, `object_store::http::HttpBuilder`,
//! ...) covering credentials, regions, endpoints and retry policy; wrapping
//! all of that again would just be a worse copy of an API that already
//! exists. Build a store with `object_store`, then pass it to
//! [`write_zarr_to_store`]/[`read_zarr_from_store`].
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use datacube_zarr::remote::{object_store, write_zarr_to_store};
//! use datacube_zarr::GeoRef;
//! use datacube_core::Cube;
//!
//! let store = object_store::aws::AmazonS3Builder::new()
//!     .with_bucket_name("my-bucket")
//!     .with_region("us-east-1")
//!     .build()?;
//! # let cube: Cube = unimplemented!();
//! write_zarr_to_store(&cube, store, &GeoRef::default())?;
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, OnceLock};

use datacube_core::Cube;
use ndarray::Array4;
use zarrs_object_store::AsyncObjectStore;
pub use zarrs_object_store::object_store;

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
use zarrs::group::GroupBuilder;
use zarrs::storage::AsyncReadableWritableListableStorage;

use crate::{
    CUBE_PATH, GRID_MAPPING_PATH, GeoRef, TIME_PATH, X_PATH, Y_PATH, ZarrDType, ZarrError,
    ZarrOptions, axis_aligned_coords, coord_attrs, cube_attrs, dtype_of, grid_mapping_attrs,
    read_meta,
};

/// A store opened via [`Array::async_open`]; every function in this module
/// works against this one storage type.
type RemoteArray = Array<dyn zarrs::storage::AsyncReadableWritableListableStorageTraits>;

/// Shared Tokio runtime for this module's blocking wrappers.
///
/// A single lazily-initialised, process-wide runtime (2 worker threads) is
/// reused across calls, mirroring `surtgis_cloud::blocking`'s
/// `shared_runtime` — same rationale (avoid one runtime per call, keep HTTP
/// connection pools warm).
///
/// # Do not call the public functions of this module from async contexts
///
/// They drive futures with `Runtime::block_on`, which **panics** when
/// invoked from within an async runtime (e.g. inside a `tokio::spawn` task).
fn shared_runtime() -> Result<&'static tokio::runtime::Runtime, ZarrError> {
    static RT: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("datacube-zarr-object-store")
            .enable_all()
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| ZarrError::Store(format!("failed to build shared tokio runtime: {e}")))
}

/// Writes `cube` to a new Zarr V3 store backed by `store` (e.g. an S3 bucket
/// or an HTTP endpoint) with the default [`ZarrOptions`]; see
/// [`write_zarr`](crate::write_zarr) for the filesystem equivalent — same
/// GeoZarr-CF companions (`/y`/`/x`/`/time`, `/spatial_ref`), same
/// attributes, just written over the network instead of to disk.
pub fn write_zarr_to_store<S: object_store::ObjectStore>(
    cube: &Cube,
    store: S,
    geo: &GeoRef,
) -> Result<(), ZarrError> {
    write_zarr_to_store_with_options(cube, store, geo, ZarrOptions::default())
}

/// Like [`write_zarr_to_store`], with explicit [`ZarrOptions`] (compression
/// level, on-disk dtype).
pub fn write_zarr_to_store_with_options<S: object_store::ObjectStore>(
    cube: &Cube,
    store: S,
    geo: &GeoRef,
    options: ZarrOptions,
) -> Result<(), ZarrError> {
    let rt = shared_runtime()?;
    rt.block_on(async_write_zarr_to_store(cube, store, geo, options))
}

async fn async_write_zarr_to_store<S: object_store::ObjectStore>(
    cube: &Cube,
    store: S,
    geo: &GeoRef,
    options: ZarrOptions,
) -> Result<(), ZarrError> {
    let store: AsyncReadableWritableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let array =
        create_array_async(store, cube.dims(), cube.bands(), cube.time(), geo, options).await?;
    store_region_async(&array, options.dtype, [0, 0, 0, 0], cube.data()).await
}

/// Reads a Zarr V3 cube written by [`write_zarr_to_store`] back into a
/// [`Cube`] and its georeference; see [`read_zarr`](crate::read_zarr) for
/// the filesystem equivalent. The whole cube is read into memory at once —
/// there is no object-store equivalent of
/// [`read_zarr_chunked`](crate::read_zarr_chunked) yet.
pub fn read_zarr_from_store<S: object_store::ObjectStore>(
    store: S,
) -> Result<(Cube, GeoRef), ZarrError> {
    let rt = shared_runtime()?;
    rt.block_on(async_read_zarr_from_store(store))
}

async fn async_read_zarr_from_store<S: object_store::ObjectStore>(
    store: S,
) -> Result<(Cube, GeoRef), ZarrError> {
    let store: AsyncReadableWritableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let array: RemoteArray = Array::async_open(store, CUBE_PATH)
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    let (bands, time, geo, dims) = read_meta(&array)?;
    let dtype = dtype_of(&array)?;

    let (nb, ny, nx, nt) = dims;
    #[allow(clippy::single_range_in_vec_init)]
    let subset =
        ArraySubset::new_with_ranges(&[0..nb as u64, 0..ny as u64, 0..nx as u64, 0..nt as u64]);
    let flat = retrieve_region_async(&array, dtype, &subset).await?;
    let data =
        Array4::from_shape_vec(dims, flat).map_err(|e| ZarrError::Metadata(e.to_string()))?;
    let mut cube = Cube::new(data, time, bands)?;
    if geo.epsg.is_some() || geo.transform.is_some() {
        cube = cube.with_georef(geo);
    }
    Ok((cube, geo))
}

/// Async counterpart of `create_array` (private, sync, in `lib.rs`) — same
/// GeoZarr-CF metadata (via the shared, storage-agnostic [`cube_attrs`]/
/// [`grid_mapping_attrs`]/[`axis_aligned_coords`]/[`coord_attrs`] helpers),
/// just `.async_*` I/O calls against an object-store-backed group instead
/// of a filesystem one.
async fn create_array_async(
    store: AsyncReadableWritableListableStorage,
    dims: (usize, usize, usize, usize),
    bands: &[String],
    time: &[f64],
    geo: &GeoRef,
    options: ZarrOptions,
) -> Result<RemoteArray, ZarrError> {
    let (nb, ny, nx, nt) = dims;

    let root = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| ZarrError::Store(e.to_string()))?;
    root.async_store_metadata()
        .await
        .map_err(|e| ZarrError::Store(e.to_string()))?;

    let shape: Vec<u64> = vec![nb as u64, ny as u64, nx as u64, nt as u64];
    let chunk: Vec<u64> = vec![
        nb as u64,
        (ny as u64).clamp(1, crate::DEFAULT_CHUNK),
        (nx as u64).clamp(1, crate::DEFAULT_CHUNK),
        nt as u64,
    ];

    let (attrs, has_grid_mapping) = cube_attrs(bands, time, geo);

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
    let array: RemoteArray = builder
        .build(store.clone(), CUBE_PATH)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    array
        .async_store_metadata()
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;

    write_coord_array_async(&store, TIME_PATH, "time", time, coord_attrs("time")).await?;
    if let Some(transform) = geo.transform
        && let Some((y, x)) = axis_aligned_coords(transform, ny, nx)
    {
        write_coord_array_async(&store, Y_PATH, "y", &y, coord_attrs("y")).await?;
        write_coord_array_async(&store, X_PATH, "x", &x, coord_attrs("x")).await?;
    }
    if has_grid_mapping {
        write_grid_mapping_async(&store, geo).await?;
    }

    Ok(array)
}

async fn write_coord_array_async(
    store: &AsyncReadableWritableListableStorage,
    path: &str,
    dim_name: &str,
    values: &[f64],
    attrs: serde_json::Map<String, serde_json::Value>,
) -> Result<(), ZarrError> {
    let n = values.len() as u64;
    let mut builder = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::float64(), f64::NAN);
    builder.dimension_names(Some([dim_name])).attributes(attrs);
    let array: RemoteArray = builder
        .build(store.clone(), path)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    array
        .async_store_metadata()
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    #[allow(clippy::single_range_in_vec_init)]
    let subset = ArraySubset::new_with_ranges(&[0..n]);
    array
        .async_store_array_subset(&subset, values)
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    Ok(())
}

async fn write_grid_mapping_async(
    store: &AsyncReadableWritableListableStorage,
    geo: &GeoRef,
) -> Result<(), ZarrError> {
    let attrs = grid_mapping_attrs(geo);
    let mut builder = ArrayBuilder::new(vec![1u64], vec![1u64], data_type::int32(), 0i32);
    builder
        .dimension_names(Some(["spatial_ref"]))
        .attributes(attrs);
    let array: RemoteArray = builder
        .build(store.clone(), GRID_MAPPING_PATH)
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    array
        .async_store_metadata()
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    #[allow(clippy::single_range_in_vec_init)]
    let subset = ArraySubset::new_with_ranges(&[0..1]);
    array
        .async_store_array_subset(&subset, [0i32].as_slice())
        .await
        .map_err(|e| ZarrError::Array(e.to_string()))?;
    Ok(())
}

/// Async counterpart of `store_region` (private, sync, in `lib.rs`).
async fn store_region_async(
    array: &RemoteArray,
    dtype: ZarrDType,
    start: [u64; 4],
    view: ndarray::ArrayView4<'_, f64>,
) -> Result<(), ZarrError> {
    let ranges: Vec<std::ops::Range<u64>> = start
        .iter()
        .zip(view.shape())
        .map(|(&s, &len)| s..s + len as u64)
        .collect();
    let subset = ArraySubset::new_with_ranges(&ranges);
    match dtype {
        ZarrDType::F64 => match view.as_slice() {
            Some(flat) => array.async_store_array_subset(&subset, flat).await,
            None => {
                let flat: Vec<f64> = view.iter().copied().collect();
                array
                    .async_store_array_subset(&subset, flat.as_slice())
                    .await
            }
        },
        ZarrDType::F32 => {
            let flat: Vec<f32> = view.iter().map(|&v| v as f32).collect();
            array
                .async_store_array_subset(&subset, flat.as_slice())
                .await
        }
    }
    .map_err(|e| ZarrError::Array(e.to_string()))
}

/// Async counterpart of `retrieve_region` (private, sync, in `lib.rs`).
async fn retrieve_region_async(
    array: &RemoteArray,
    dtype: ZarrDType,
    subset: &ArraySubset,
) -> Result<Vec<f64>, ZarrError> {
    match dtype {
        ZarrDType::F64 => array
            .async_retrieve_array_subset::<Vec<f64>>(subset)
            .await
            .map_err(|e| ZarrError::Array(e.to_string())),
        ZarrDType::F32 => {
            let flat: Vec<f32> = array
                .async_retrieve_array_subset::<Vec<f32>>(subset)
                .await
                .map_err(|e| ZarrError::Array(e.to_string()))?;
            Ok(flat.into_iter().map(f64::from).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array4;
    use object_store::memory::InMemory;

    // Real object_store I/O (in-memory backend, no network) — not a mock of
    // the write/read path, so this exercises the same async_* calls a real
    // S3/HTTP store would see, just offline-testable like the rest of the
    // crate. `InMemory` is `Clone` (cheap, shares the underlying store), so
    // tests can write through one handle and inspect through another.

    fn sample_cube() -> Cube {
        let (nb, ny, nx, nt) = (2, 40, 4, 5);
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
        let store = InMemory::new();
        let cube = sample_cube();
        let geo = sample_geo();

        write_zarr_to_store(&cube, store.clone(), &geo).unwrap();
        let (back, back_geo) = read_zarr_from_store(store).unwrap();

        assert_eq!(back.dims(), cube.dims());
        assert_eq!(back.bands(), cube.bands());
        assert_eq!(back.time(), cube.time());
        assert_eq!(back_geo, geo);
        assert_eq!(back.georef(), Some(geo));
        for (x, y) in cube.data().iter().zip(back.data().iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn f32_option_roundtrips_and_is_detected() {
        let store = InMemory::new();
        let cube = sample_cube();
        let options = ZarrOptions {
            compression_level: Some(9),
            dtype: ZarrDType::F32,
        };

        write_zarr_to_store_with_options(&cube, store.clone(), &sample_geo(), options).unwrap();
        let (back, _) = read_zarr_from_store(store).unwrap();

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
    fn geozarr_cf_metadata_is_written_alongside_cube() {
        let store = InMemory::new();
        let cube = sample_cube();
        let geo = sample_geo();
        write_zarr_to_store(&cube, store.clone(), &geo).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let async_store: AsyncReadableWritableListableStorage =
                Arc::new(AsyncObjectStore::new(store));

            let spatial_ref: RemoteArray =
                Array::async_open(async_store.clone(), GRID_MAPPING_PATH)
                    .await
                    .unwrap();
            let attrs = spatial_ref.attributes();
            assert_eq!(attrs.get("epsg").unwrap(), 32719);
            assert_eq!(
                attrs.get("grid_mapping_name").unwrap(),
                "transverse_mercator"
            );
            assert!(
                attrs
                    .get("crs_wkt")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .contains("32719")
            );

            let y: RemoteArray = Array::async_open(async_store.clone(), Y_PATH)
                .await
                .unwrap();
            assert_eq!(y.attributes().get("axis").unwrap(), "Y");
            let time: RemoteArray = Array::async_open(async_store.clone(), TIME_PATH)
                .await
                .unwrap();
            assert_eq!(time.attributes().get("standard_name").unwrap(), "time");

            let cube_array: RemoteArray = Array::async_open(async_store, CUBE_PATH).await.unwrap();
            assert_eq!(
                cube_array.attributes().get("grid_mapping").unwrap(),
                "spatial_ref"
            );
        });
    }

    #[test]
    fn read_missing_array_errors() {
        let store = InMemory::new();
        assert!(matches!(
            read_zarr_from_store(store),
            Err(ZarrError::Array(_))
        ));
    }
}
