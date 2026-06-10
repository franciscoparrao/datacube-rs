//! # datacube-core
//!
//! Temporal data cube model `(band, y, x, time)` and per-pixel time-series
//! statistics for remote sensing: OLS linear trend, Theil-Sen slope and the
//! Mann-Kendall trend test (tie-corrected, following `pyMannKendall`).
//!
//! The time axis is the innermost (contiguous) dimension so that per-pixel
//! series can be streamed as plain slices without copying.
//!
//! ```
//! use datacube_core::{Cube, stats};
//! use ndarray::Array4;
//!
//! // a 1-band, 2x2 pixel cube with 5 time steps where value = t
//! let mut data = Array4::zeros((1, 2, 2, 5));
//! for t in 0..5 {
//!     data.slice_mut(ndarray::s![.., .., .., t]).fill(t as f64);
//! }
//! let cube = Cube::new(data, (0..5).map(f64::from).collect(), vec!["ndvi".into()]).unwrap();
//!
//! let trends = cube.par_map_series(0, |t, y| {
//!     stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN)
//! }).unwrap();
//! assert!((trends[[0, 0]] - 1.0).abs() < 1e-12);
//! ```

mod cube;
mod error;
pub mod stats;

pub use cube::{Cube, CubeChunk, PixelSeries};
pub use error::CubeError;
