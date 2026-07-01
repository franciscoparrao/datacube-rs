//! Writes a small sample cube to a Zarr V3 store for interoperability checks.
//!
//! Run: cargo run -q -p datacube-zarr --example write_sample -- /tmp/sample.zarr
//! Then read it back from Python (see scripts/zarr_interop.py) to confirm the
//! Rust-written store is consumable by the zarr/xarray ecosystem.

use datacube_core::{Cube, indices};
use datacube_zarr::{GeoRef, write_zarr};
use ndarray::Array4;
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/sample.zarr".to_string()),
    );

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
    let cube = Cube::new(data, time, vec!["red".into(), "nir".into()]).unwrap();

    // also derive NDVI in-engine and stack it as a third band for the demo
    let ndvi = indices::ndvi(&cube, "nir", "red").unwrap();
    let geo = GeoRef {
        epsg: Some(32719),
        transform: Some([300000.0, 10.0, 0.0, 6200000.0, 0.0, -10.0]),
    };

    write_zarr(&cube, &path, &geo).unwrap();
    println!(
        "wrote {} ({nb} bands {ny}x{nx} px, {nt} t); NDVI[0,0,0]={:.4}",
        path.display(),
        ndvi.data()[[0, 0, 0, 0]]
    );
}
