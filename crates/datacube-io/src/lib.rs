//! # datacube-io
//!
//! Temporal stacking of STAC/COG imagery into [`datacube_core::Cube`]s.
//!
//! Searches a STAC catalog (Planetary Computer, Earth Search or any STAC API)
//! for items intersecting a bbox and date range, reads the requested COG
//! assets windowed to the bbox (signing Planetary Computer URLs
//! automatically), aligns every scene to a common grid, and assembles a
//! `(band, y, x, time)` cube with `NaN` nodata and fractional-year time
//! coordinates — ready for the trend and phenology statistics in
//! `datacube_core::stats`.
//!
//! I/O lives here so that `datacube-core` stays pure compute. The heavy
//! lifting (HTTP range reads, TIFF decode, SAS signing, UTM reprojection,
//! grid resampling) is reused from the SurtGIS engine (`surtgis-cloud`,
//! `surtgis-core`), which must be checked out as a sibling of this repo.
//!
//! ```no_run
//! use datacube_io::{StackConfig, stack};
//!
//! let cfg = StackConfig::new("pc", "sentinel-2-l2a", &["B04"])
//!     .bbox(-70.75, -33.55, -70.65, -33.45)
//!     .datetime("2024-01-01/2024-12-31")
//!     .max_cloud_cover(30.0)
//!     .overview(Some(3));
//! let stacked = stack(&cfg).unwrap();
//! println!("{} scenes -> dims {:?}", stacked.slices.len(), stacked.cube.dims());
//! ```

mod stack;
mod time;

pub use stack::{SliceMeta, StackConfig, StackedCube, stack};
pub use time::fractional_year;

use thiserror::Error;

/// Errors produced while searching, reading or assembling a stack.
#[derive(Debug, Error)]
pub enum StackError {
    /// STAC search, signing, HTTP or COG decode failure (from surtgis-cloud).
    #[error("cloud I/O error: {0}")]
    Cloud(#[from] surtgis_cloud::error::CloudError),

    /// Resampling/raster failure (from surtgis-core).
    #[error("raster error: {0}")]
    Raster(#[from] surtgis_core::Error),

    /// Cube assembly failure (from datacube-core).
    #[error(transparent)]
    Cube(#[from] datacube_core::CubeError),

    /// Invalid stack configuration.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A scene could not be reprojected onto the reference grid (e.g. a
    /// non-UTM CRS, which the UTM↔UTM mosaicker does not handle).
    #[error("reprojection failed: {0}")]
    Reproject(String),

    /// The search returned no items, or none survived the filters.
    #[error("no usable scenes: {0}")]
    Empty(String),
}
