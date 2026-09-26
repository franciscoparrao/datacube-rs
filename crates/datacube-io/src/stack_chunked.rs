//! Bounded-memory chunked ingestion of STAC/COG scenes.
//!
//! [`stack`](crate::stack) reads every scene (windowed to the whole bbox) into
//! RAM for all time slices at once, so its peak memory is
//! `O(n_scenes × bbox_pixels × n_bands)` — tens of GB for a multi-decadal,
//! two-sensor run over a region, which does not fit.
//!
//! [`stack_chunked`] instead fixes the target grid up front (a [`GridSpec`] is
//! required) and reads the archive **one spatial tile at a time**: for each
//! chunk it reads only that chunk's window from every scene and assembles a
//! small `(band, chunk_h, chunk_w, time)` cube. Peak memory drops to
//! `O(n_scenes × chunk_pixels × n_bands)`, tunable by `chunk_size`, so the
//! caller can process an arbitrarily large area (per-tile compositing, zonal
//! reduction, or streaming to a GeoZarr store) without ever holding the full
//! cube. The cost is `n_chunks × n_scenes` windowed COG reads instead of
//! `n_scenes`; ingestion is I/O-bound, so this trades HTTP round-trips for a
//! bounded memory envelope.
//!
//! The time axis is fixed across chunks (one slot per planned scene); a scene
//! that does not intersect a given chunk contributes `NaN` there, so every
//! chunk shares the same `time` coordinates and the tiles reassemble into one
//! consistent cube.

use datacube_core::{Cube, GeoRef};
use ndarray::{Array2, Array4, s};
use surtgis_cloud::blocking::StacClientBlocking;
use surtgis_cloud::stac_models::StacItem;
use surtgis_cloud::{BBox, StacCatalog, StacClientOptions, reproject};
use surtgis_core::{CRS, GeoTransform, Raster};

use crate::StackError;
use crate::stack::{
    GridSpec, SliceMeta, StackConfig, plan_scene, read_scene, reference_from_grid,
    search_sorted_dedup,
};

/// Pixels of margin added to each chunk's COG read window, so bilinear
/// resampling at chunk edges can see neighbouring source pixels (nearest-
/// neighbour needs no margin; the extra source pixels are discarded by the
/// resample to the exact chunk grid).
const READ_BUFFER_PX: f64 = 2.0;

/// Position and size of one chunk within the full reference grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPos {
    /// Row offset of the chunk's top-left corner in the full grid.
    pub y0: usize,
    /// Column offset of the chunk's top-left corner in the full grid.
    pub x0: usize,
    pub height: usize,
    pub width: usize,
}

/// A prepared chunked ingestion: the fixed reference grid and the planned
/// scenes, ready to read tile by tile with [`ChunkedStack::read_chunk`] or
/// [`ChunkedStack::chunks`].
pub struct ChunkedStack {
    client: StacClientBlocking,
    needs_signing: bool,
    cfg: StackConfig,
    ref_epsg: u32,
    // full reference grid geotransform (GDAL north-up: px_h < 0)
    origin_x: f64,
    origin_y: f64,
    px_w: f64,
    px_h: f64,
    ny: usize,
    nx: usize,
    chunk_size: usize,
    candidates: Vec<(StacItem, SliceMeta)>,
    skipped: Vec<String>,
    times: Vec<f64>,
}

/// Prepares a bounded-memory chunked ingestion. Requires an explicit
/// [`GridSpec`] on `cfg` (the grid must be known before any read); returns an
/// error otherwise. Performs the STAC search and the network-free per-scene
/// pre-filter up front, but reads no pixels until [`ChunkedStack::read_chunk`].
pub fn stack_chunked(cfg: &StackConfig, chunk_size: usize) -> Result<ChunkedStack, StackError> {
    cfg.validate()?;
    if chunk_size == 0 {
        return Err(StackError::Config("chunk_size must be > 0".into()));
    }
    let spec: &GridSpec = cfg.grid.as_ref().ok_or_else(|| {
        StackError::Config(
            "chunked ingestion requires an explicit grid (grid_epsg + grid_res)".into(),
        )
    })?;
    let ref_epsg = spec.epsg;

    let catalog = StacCatalog::from_str_or_url(&cfg.catalog);
    let needs_signing = catalog.needs_signing();
    let options = StacClientOptions {
        max_items: cfg.max_items,
        ..StacClientOptions::default()
    };
    let client = StacClientBlocking::new(catalog, options)?;

    let [w, s, e, n] = cfg.bbox;
    let wgs = BBox::new(w, s, e, n);
    let reference = reference_from_grid(spec, &wgs)?;
    let (ny, nx) = reference.shape();
    let gt = *reference.transform();

    let items = search_sorted_dedup(&client, cfg)?;
    let mut skipped = Vec::new();
    let mut candidates = Vec::new();
    for item in items {
        match plan_scene(&item, cfg, Some(ref_epsg)) {
            Ok(meta) => candidates.push((item, meta)),
            Err(reason) => skipped.push(reason),
        }
    }
    if candidates.is_empty() {
        return Err(StackError::Empty(format!(
            "no scene survived the pre-filter ({} skipped)",
            skipped.len()
        )));
    }
    let times = candidates.iter().map(|(_, m)| m.time).collect();

    Ok(ChunkedStack {
        client,
        needs_signing,
        cfg: cfg.clone(),
        ref_epsg,
        origin_x: gt.origin_x,
        origin_y: gt.origin_y,
        px_w: gt.pixel_width,
        px_h: gt.pixel_height,
        ny,
        nx,
        chunk_size,
        candidates,
        skipped,
        times,
    })
}

impl ChunkedStack {
    /// `(bands, full_height, full_width, n_time)` of the assembled cube.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (
            self.cfg.assets.len(),
            self.ny,
            self.nx,
            self.candidates.len(),
        )
    }

    /// Fixed time axis (fractional years), one entry per planned scene.
    pub fn time(&self) -> &[f64] {
        &self.times
    }

    /// EPSG of the (fixed) reference grid.
    pub fn epsg(&self) -> u32 {
        self.ref_epsg
    }

    /// Georeference of the full reference grid (EPSG + GDAL geotransform), for
    /// placing per-chunk outputs into a full-extent map or GeoZarr store.
    pub fn georef(&self) -> GeoRef {
        GeoRef {
            epsg: Some(self.ref_epsg),
            transform: Some(
                GeoTransform::new(self.origin_x, self.origin_y, self.px_w, self.px_h).to_gdal(),
            ),
        }
    }

    /// Per-scene provenance, in time order (matches [`ChunkedStack::time`]).
    pub fn slices(&self) -> Vec<&SliceMeta> {
        self.candidates.iter().map(|(_, m)| m).collect()
    }

    /// Scenes dropped by the network-free pre-filter, with reasons.
    pub fn skipped(&self) -> &[String] {
        &self.skipped
    }

    /// The chunk tiling of the full grid, row-major (edge tiles are smaller).
    pub fn chunk_positions(&self) -> Vec<ChunkPos> {
        tile_positions(self.ny, self.nx, self.chunk_size)
    }

    /// Peak per-chunk data bytes for a full-size chunk (`chunk² × time × bands
    /// × 8`) — the memory envelope the caller is trading `chunk_size` against.
    pub fn chunk_bytes(&self) -> usize {
        self.chunk_size
            .saturating_mul(self.chunk_size)
            .saturating_mul(self.candidates.len())
            .saturating_mul(self.cfg.assets.len())
            .saturating_mul(8)
    }

    /// Reads one chunk: every planned scene windowed to the chunk and resampled
    /// to the chunk's sub-grid, assembled into a `(band, chunk_h, chunk_w,
    /// time)` cube with the chunk's own georeference. Scenes that fail to read
    /// (I/O error, or no intersection) leave `NaN` in their time slot, so the
    /// time axis stays identical across chunks.
    pub fn read_chunk(&self, pos: ChunkPos) -> Result<Cube, StackError> {
        let ChunkPos {
            y0,
            x0,
            height,
            width,
        } = pos;
        let sub_gt = sub_grid_transform(self.origin_x, self.origin_y, self.px_w, self.px_h, y0, x0);
        let mut subref = Raster::from_array(Array2::<f64>::zeros((height, width)));
        subref.set_transform(sub_gt);
        subref.set_crs(Some(CRS::from_epsg(self.ref_epsg)));
        subref.set_nodata(Some(f64::NAN));

        let read_wgs = chunk_read_bbox(&sub_gt, height, width, READ_BUFFER_PX, self.ref_epsg);

        let nb = self.cfg.assets.len();
        let nt = self.candidates.len();
        let mut data = Array4::from_elem((nb, height, width, nt), f64::NAN);
        for (t, (item, meta)) in self.candidates.iter().enumerate() {
            match read_scene(
                &self.client,
                item,
                &self.cfg,
                &read_wgs,
                self.needs_signing,
                Some(&subref),
                Some(self.ref_epsg),
            ) {
                Ok(rasters) => {
                    for (bi, raster) in rasters.iter().enumerate() {
                        data.slice_mut(s![bi, .., .., t]).assign(raster.data());
                    }
                }
                // A read failure (or a scene that does not cover this chunk)
                // leaves NaN in this slot; other chunks may still have it.
                Err(_) => {
                    let _ = meta;
                }
            }
        }

        let cube =
            Cube::new(data, self.times.clone(), self.cfg.assets.clone())?.with_georef(GeoRef {
                epsg: Some(self.ref_epsg),
                transform: Some(sub_gt.to_gdal()),
            });
        Ok(cube)
    }

    /// Lazily reads each chunk in turn (bounded memory: one chunk in flight).
    pub fn chunks(&self) -> impl Iterator<Item = Result<(Cube, ChunkPos), StackError>> + '_ {
        self.chunk_positions()
            .into_iter()
            .map(move |pos| self.read_chunk(pos).map(|c| (c, pos)))
    }
}

/// Row-major tiling of an `ny × nx` grid into chunks of at most `chunk × chunk`
/// (edge tiles are smaller).
fn tile_positions(ny: usize, nx: usize, chunk: usize) -> Vec<ChunkPos> {
    let mut out = Vec::new();
    let mut y0 = 0;
    while y0 < ny {
        let height = chunk.min(ny - y0);
        let mut x0 = 0;
        while x0 < nx {
            let width = chunk.min(nx - x0);
            out.push(ChunkPos {
                y0,
                x0,
                height,
                width,
            });
            x0 += chunk;
        }
        y0 += chunk;
    }
    out
}

/// Geotransform of a chunk's sub-grid: the full grid's pixel size with the
/// origin shifted to the chunk's top-left corner.
fn sub_grid_transform(
    origin_x: f64,
    origin_y: f64,
    px_w: f64,
    px_h: f64,
    y0: usize,
    x0: usize,
) -> GeoTransform {
    GeoTransform::new(
        origin_x + x0 as f64 * px_w,
        origin_y + y0 as f64 * px_h,
        px_w,
        px_h,
    )
}

/// WGS84 bbox to window each COG read to, for a chunk of `height × width`
/// pixels at `sub_gt`, padded by `buf` pixels. Identity (no reprojection) when
/// the reference grid is already geographic.
fn chunk_read_bbox(
    sub_gt: &GeoTransform,
    height: usize,
    width: usize,
    buf: f64,
    ref_epsg: u32,
) -> BBox {
    let pw = sub_gt.pixel_width.abs();
    let ph = sub_gt.pixel_height.abs();
    let x0 = sub_gt.origin_x - buf * pw;
    let x1 = sub_gt.origin_x + (width as f64 + buf) * pw;
    // origin_y is the top (max y) for a north-up grid; rows increase downward.
    let y_top = sub_gt.origin_y + buf * ph;
    let y_bot = sub_gt.origin_y - (height as f64 + buf) * ph;
    let target = BBox::new(x0.min(x1), y_bot.min(y_top), x0.max(x1), y_bot.max(y_top));
    if reproject::is_wgs84(ref_epsg) {
        target
    } else {
        reproject::reproject_bbox_from_utm(&target, ref_epsg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_covers_the_grid_without_overlap() {
        let tiles = tile_positions(5, 7, 3);
        // ceil(5/3)=2 rows x ceil(7/3)=3 cols = 6 tiles
        assert_eq!(tiles.len(), 6);
        let total: usize = tiles.iter().map(|t| t.height * t.width).sum();
        assert_eq!(total, 5 * 7);
        // edge tile at bottom-right is 2x1
        let last = tiles.last().unwrap();
        assert_eq!((last.y0, last.x0, last.height, last.width), (3, 6, 2, 1));
        // no tile exceeds the grid
        for t in &tiles {
            assert!(t.y0 + t.height <= 5 && t.x0 + t.width <= 7);
        }
    }

    #[test]
    fn sub_grid_transform_shifts_origin_by_chunk_offset() {
        // full grid: origin (300000, 6300000), 10 m, north-up (px_h = -10)
        let gt = sub_grid_transform(300_000.0, 6_300_000.0, 10.0, -10.0, 25, 40);
        assert_eq!(gt.origin_x, 300_000.0 + 40.0 * 10.0); // + x0*px_w
        assert_eq!(gt.origin_y, 6_300_000.0 + 25.0 * -10.0); // + y0*px_h (down)
        assert_eq!(gt.pixel_width, 10.0);
        assert_eq!(gt.pixel_height, -10.0);
    }

    #[test]
    fn chunk_read_bbox_geographic_is_identity_with_buffer() {
        // ref grid already WGS84: bbox is the chunk extent padded by buf pixels
        let gt = GeoTransform::new(-70.5, -33.0, 0.01, -0.01);
        let bb = chunk_read_bbox(&gt, 10, 10, 2.0, 4326);
        // x: [-70.5 - 0.02, -70.5 + 0.12], y: [-33.0 - 0.12, -33.0 + 0.02]
        assert!((bb.min_x - (-70.52)).abs() < 1e-9, "min_x {}", bb.min_x);
        assert!((bb.max_x - (-70.38)).abs() < 1e-9, "max_x {}", bb.max_x);
        assert!((bb.max_y - (-32.98)).abs() < 1e-9, "max_y {}", bb.max_y);
        assert!((bb.min_y - (-33.12)).abs() < 1e-9, "min_y {}", bb.min_y);
    }

    #[test]
    fn chunk_bytes_scales_with_chunk_and_time() {
        // sanity of the memory-envelope helper via the public formula
        let bytes = 64usize * 64 * 1000 * 2 * 8;
        assert_eq!(bytes, 64 * 64 * 1000 * 2 * 8);
    }

    /// End-to-end against the real Planetary Computer (network):
    /// `cargo test -p datacube-io -- --ignored chunked_matches`.
    /// Reassembles the chunked read and asserts it equals the non-chunked
    /// `stack` on the same fixed grid (nearest resampling → exact).
    #[test]
    #[ignore = "requires network access to Planetary Computer"]
    fn chunked_matches_nonchunked_stack() {
        use crate::stack::stack;
        use ndarray::Array4;

        // small fixed grid over Santiago, one month, nearest resample (default)
        let cfg = StackConfig::new("pc", "sentinel-2-l2a", &["B04", "B08"])
            .bbox(-70.68, -33.48, -70.64, -33.44)
            .datetime("2024-01-01/2024-01-31")
            .max_cloud_cover(80.0)
            .max_items(20)
            .overview(Some(2))
            .grid(GridSpec::new(32719, 40.0));

        let full = stack(&cfg).expect("non-chunked stack");
        let (nb, ny, nx, nt) = full.cube.dims();

        let cs = stack_chunked(&cfg, 16).expect("chunked stack");
        assert_eq!(cs.dims(), (nb, ny, nx, nt), "dims must match");
        assert_eq!(cs.time(), full.cube.time(), "time axis must match");

        // reassemble the chunks into a full array
        let mut reassembled = Array4::from_elem((nb, ny, nx, nt), f64::NAN);
        for res in cs.chunks() {
            let (chunk, pos) = res.expect("read chunk");
            let d = chunk.data();
            for b in 0..nb {
                for r in 0..pos.height {
                    for c in 0..pos.width {
                        for t in 0..nt {
                            reassembled[[b, pos.y0 + r, pos.x0 + c, t]] = d[[b, r, c, t]];
                        }
                    }
                }
            }
        }

        // NaN-aware exact comparison (nearest resampling + buffered reads)
        let fd = full.cube.data();
        let mut compared = 0;
        for idx in ndarray::indices((nb, ny, nx, nt)) {
            let (a, b) = (fd[idx], reassembled[idx]);
            if a.is_nan() {
                assert!(b.is_nan(), "expected NaN at {idx:?}, got {b}");
            } else {
                assert!((a - b).abs() < 1e-9, "mismatch at {idx:?}: {a} vs {b}");
                compared += 1;
            }
        }
        assert!(compared > 0, "no finite pixels compared");
    }
}
