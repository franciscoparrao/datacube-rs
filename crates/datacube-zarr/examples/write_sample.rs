//! Writes a small sample cube to a Zarr V3 store for interoperability checks.
//!
//! Run: cargo run -q -p datacube-zarr --example write_sample -- /tmp/sample.zarr
//! Pass `f32` as a second arg to store as f32 instead of the zstd-compressed
//! f64 default (e.g. `... -- /tmp/sample.zarr f32`).
//! Then read it back from Python (see scripts/zarr_interop.py) to confirm the
//! Rust-written store is consumable by the zarr/xarray ecosystem.

use datacube_core::{Cube, GeoRef, indices};
use datacube_zarr::{ZarrDType, ZarrOptions, write_zarr_with_options};
use ndarray::Array4;
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/sample.zarr".to_string()),
    );
    let dtype = if std::env::args().nth(2).as_deref() == Some("f32") {
        ZarrDType::F32
    } else {
        ZarrDType::F64
    };

    // red, nir over a 32x32 grid, 6 dates; a synthetic greening trend
    let (nb, ny, nx, nt) = (2, 32, 32, 6);
    let mut data = Array4::zeros((nb, ny, nx, nt));
    for y in 0..ny {
        for x in 0..nx {
            for t in 0..nt {
                let nir = 0.4 + 0.02 * t as f64 + 0.001 * (x + y) as f64;
                let red = 0.15 - 0.005 * t as f64;
                data[[0, y, x, t]] = red;
                data[[1, y, x, t]] = nir;
            }
        }
    }
    let time: Vec<f64> = (0..nt).map(|t| 2020.0 + t as f64 * 0.5).collect();
    let cube = Cube::new(data, time, vec!["red".into(), "nir".into()])
        .unwrap()
        .with_georef(GeoRef {
            epsg: Some(32719),
            transform: Some([300000.0, 10.0, 0.0, 6200000.0, 0.0, -10.0]),
        });

    // NDVI derived in-engine inherits the cube's georef automatically
    let ndvi = indices::ndvi(&cube, "nir", "red").unwrap();
    assert_eq!(ndvi.georef(), cube.georef());

    let options = ZarrOptions {
        dtype,
        ..ZarrOptions::default()
    };
    write_zarr_with_options(&cube, &path, &cube.georef().unwrap_or_default(), options).unwrap();
    println!(
        "wrote {} ({nb} bands {ny}x{nx} px, {nt} t, dtype {dtype:?}); NDVI[0,0,0]={:.4}",
        path.display(),
        ndvi.data()[[0, 0, 0, 0]]
    );
}
