use datacube_core::Cube;
use ndarray::{Array2, Array4};
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::stac_models::StacItem;
use surtgis_cloud::{
    BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacSearchParams, reproject,
};
use surtgis_core::{CRS, GeoTransform, Raster, ResampleMethod, resample_to_grid};

use crate::StackError;
use crate::time::fractional_year;

/// Per-pixel quality masking applied to every scene before grid alignment.
///
/// The mask asset (e.g. the Sentinel-2 `SCL` scene-classification band) is
/// read with the same window as the data assets, resampled to each band's
/// grid, and every pixel whose class is **not** in [`keep`] becomes `NaN`.
/// Masking happens before the (NaN-tolerant) resampling to the reference
/// grid, so cloudy values never bleed into their neighbours.
///
/// [`keep`]: MaskConfig::keep
#[derive(Debug, Clone)]
pub struct MaskConfig {
    /// Asset key of the quality band, e.g. `"SCL"`.
    pub asset: String,
    /// Class values to keep; every other class (and mask nodata) → `NaN`.
    pub keep: Vec<u16>,
    /// How the mask is aligned to each band's grid. Class bands are
    /// categorical, so this should stay [`ResampleMethod::NearestNeighbor`].
    pub resample: ResampleMethod,
}

impl MaskConfig {
    /// Sentinel-2 L2A scene classification mask keeping vegetation (4),
    /// bare soil (5), water (6), unclassified (7) and snow/ice (11).
    pub fn scl() -> Self {
        Self {
            asset: "SCL".to_string(),
            keep: vec![4, 5, 6, 7, 11],
            resample: ResampleMethod::NearestNeighbor,
        }
    }
}

/// Explicit target grid for [`stack`], à la gdalcubes' `cube_view`.
///
/// When set, the reference grid is built from this spec instead of from the
/// first readable scene, so the cube's CRS, resolution, extent and alignment
/// are fully reproducible — independent of catalog order, cloud filtering
/// and network failures. Scenes in other UTM zones are reprojected onto it
/// (see [`StackConfig::cross_zone_mosaic`]).
#[derive(Debug, Clone)]
pub struct GridSpec {
    /// EPSG of the target grid (UTM codes are fully supported end-to-end).
    pub epsg: u32,
    /// Pixel size in CRS units (e.g. metres for UTM).
    pub resolution: f64,
    /// `[min_x, min_y, max_x, max_y]` in the target CRS. `None` derives the
    /// extent by reprojecting the WGS84 search bbox.
    pub bbox: Option<[f64; 4]>,
    /// Snap the grid origin outward to a multiple of this value (e.g.
    /// `60.0` to align with the Sentinel-2 MGRS grid).
    pub align: Option<f64>,
}

impl GridSpec {
    pub fn new(epsg: u32, resolution: f64) -> Self {
        Self {
            epsg,
            resolution,
            bbox: None,
            align: None,
        }
    }

    /// Target-CRS extent override.
    pub fn bbox(mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        self.bbox = Some([min_x, min_y, max_x, max_y]);
        self
    }

    /// Snap the grid origin outward to a multiple of `step`.
    pub fn align(mut self, step: f64) -> Self {
        self.align = Some(step);
        self
    }
}

/// Configuration for [`stack`].
#[derive(Debug, Clone)]
pub struct StackConfig {
    /// Catalog: `"pc"` (Planetary Computer), `"es"` (Earth Search) or a
    /// full STAC API URL.
    pub catalog: String,
    /// Collection id, e.g. `"sentinel-2-l2a"`.
    pub collection: String,
    /// Asset keys to stack as cube bands, e.g. `["B04", "B08"]`.
    pub assets: Vec<String>,
    /// WGS84 `[west, south, east, north]`.
    pub bbox: [f64; 4],
    /// STAC datetime range, e.g. `"2024-01-01/2024-12-31"`.
    pub datetime: String,
    /// Maximum items fetched from the search (across pages).
    pub max_items: usize,
    /// Skip scenes with `eo:cloud_cover` above this percentage.
    pub max_cloud_cover: Option<f64>,
    /// COG overview level to read (`None` = full resolution; higher levels
    /// are coarser and much faster).
    pub overview: Option<usize>,
    /// How scenes are aligned to the reference grid.
    pub resample: ResampleMethod,
    /// Multiplicative factor applied to every value after nodata masking
    /// (e.g. `1e-4` for Sentinel-2 L2A DN → reflectance).
    pub scale: f64,
    /// Additive offset applied after `scale` (e.g. `-0.1` for Sentinel-2
    /// L2A processing baseline ≥ 04.00).
    pub offset: f64,
    /// Mosaic scenes from other UTM zones onto the reference grid by UTM↔UTM
    /// reprojection (default `true`). When `false`, scenes whose CRS differs
    /// from the reference are skipped instead.
    pub cross_zone_mosaic: bool,
    /// Per-pixel quality masking (e.g. Sentinel-2 SCL). `None` = no masking.
    pub mask: Option<MaskConfig>,
    /// Explicit target grid. `None` = the first readable scene defines it.
    pub grid: Option<GridSpec>,
}

impl StackConfig {
    pub fn new(catalog: &str, collection: &str, assets: &[&str]) -> Self {
        Self {
            catalog: catalog.to_string(),
            collection: collection.to_string(),
            assets: assets.iter().map(|s| s.to_string()).collect(),
            bbox: [0.0; 4],
            datetime: String::new(),
            max_items: 100,
            max_cloud_cover: None,
            overview: None,
            resample: ResampleMethod::NearestNeighbor,
            scale: 1.0,
            offset: 0.0,
            cross_zone_mosaic: true,
            mask: None,
            grid: None,
        }
    }

    pub fn bbox(mut self, west: f64, south: f64, east: f64, north: f64) -> Self {
        self.bbox = [west, south, east, north];
        self
    }

    pub fn datetime(mut self, range: &str) -> Self {
        self.datetime = range.to_string();
        self
    }

    pub fn max_items(mut self, n: usize) -> Self {
        self.max_items = n;
        self
    }

    pub fn max_cloud_cover(mut self, pct: f64) -> Self {
        self.max_cloud_cover = Some(pct);
        self
    }

    pub fn overview(mut self, level: Option<usize>) -> Self {
        self.overview = level;
        self
    }

    /// Linear value transform `v·scale + offset`, applied after nodata
    /// masking (e.g. `.scaling(1e-4, -0.1)` for Sentinel-2 L2A reflectance,
    /// processing baseline ≥ 04.00).
    pub fn scaling(mut self, scale: f64, offset: f64) -> Self {
        self.scale = scale;
        self.offset = offset;
        self
    }

    /// Whether to mosaic scenes from other UTM zones onto the reference grid
    /// (default `true`). Set `false` to skip cross-zone scenes instead.
    pub fn cross_zone_mosaic(mut self, on: bool) -> Self {
        self.cross_zone_mosaic = on;
        self
    }

    /// Per-pixel quality masking ([`MaskConfig::scl`] for Sentinel-2 L2A).
    pub fn mask(mut self, mask: MaskConfig) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Explicit target grid (see [`GridSpec`]).
    pub fn grid(mut self, grid: GridSpec) -> Self {
        self.grid = Some(grid);
        self
    }

    fn validate(&self) -> Result<(), StackError> {
        if self.assets.is_empty() {
            return Err(StackError::Config(
                "at least one asset key is required".into(),
            ));
        }
        let [w, s, e, n] = self.bbox;
        if !(w < e && s < n) {
            return Err(StackError::Config(format!(
                "bbox must satisfy west < east and south < north, got [{w}, {s}, {e}, {n}]"
            )));
        }
        if self.datetime.is_empty() {
            return Err(StackError::Config("datetime range is required".into()));
        }
        if let Some(mask) = &self.mask {
            if mask.asset.is_empty() {
                return Err(StackError::Config("mask asset key is empty".into()));
            }
            if mask.keep.is_empty() {
                return Err(StackError::Config(
                    "mask keep-list is empty (every pixel would be masked)".into(),
                ));
            }
        }
        if let Some(grid) = &self.grid {
            if !grid.resolution.is_finite() || grid.resolution <= 0.0 {
                return Err(StackError::Config(format!(
                    "grid resolution must be finite and > 0, got {}",
                    grid.resolution
                )));
            }
            if let Some([min_x, min_y, max_x, max_y]) = grid.bbox
                && !(min_x < max_x && min_y < max_y)
            {
                return Err(StackError::Config(format!(
                    "grid bbox must satisfy min < max, got [{min_x}, {min_y}, {max_x}, {max_y}]"
                )));
            }
            if let Some(step) = grid.align
                && (!step.is_finite() || step <= 0.0)
            {
                return Err(StackError::Config(format!(
                    "grid align step must be finite and > 0, got {step}"
                )));
            }
        }
        Ok(())
    }
}

/// Provenance of one time slice of the stacked cube.
#[derive(Debug, Clone)]
pub struct SliceMeta {
    pub item_id: String,
    /// Original ISO 8601 datetime from the STAC item.
    pub datetime: String,
    /// Fractional-year time coordinate used in the cube.
    pub time: f64,
    pub cloud_cover: Option<f64>,
}

/// A cube assembled from STAC scenes, with its geospatial context.
#[derive(Debug)]
pub struct StackedCube {
    /// `(band, y, x, time)` cube; nodata is `NaN`, time is fractional years.
    pub cube: Cube,
    /// One entry per time slice, in cube time order.
    pub slices: Vec<SliceMeta>,
    /// Scenes that were skipped, with the reason (cloud filter, missing
    /// asset, read failure, CRS mismatch, ...).
    pub skipped: Vec<String>,
    /// Geotransform of the common grid (from the reference scene).
    pub transform: GeoTransform,
    /// EPSG of the common grid, if known.
    pub epsg: Option<u32>,
}

/// Searches the catalog and stacks the matching scenes into a cube.
///
/// The reference grid comes from [`StackConfig::grid`] when set (explicit,
/// reproducible); otherwise the first successfully-read scene defines it.
/// Every scene is resampled onto it (`cfg.resample`). Scenes in a different
/// UTM zone are reprojected onto the reference zone first (UTM↔UTM
/// mosaicking, unless [`StackConfig::cross_zone_mosaic`] is off); scenes
/// that still cannot be reprojected (non-UTM CRS) are skipped and reported
/// in [`StackedCube::skipped`]. Nodata becomes `NaN`, pixels rejected by
/// [`StackConfig::mask`] become `NaN`, and values stay raw unless
/// [`StackConfig::scaling`] is set.
pub fn stack(cfg: &StackConfig) -> Result<StackedCube, StackError> {
    cfg.validate()?;

    let catalog = StacCatalog::from_str_or_url(&cfg.catalog);
    let needs_signing = catalog.needs_signing();
    let options = StacClientOptions {
        max_items: cfg.max_items,
        ..StacClientOptions::default()
    };
    let client = StacClientBlocking::new(catalog, options)?;

    let [w, s, e, n] = cfg.bbox;
    let params = StacSearchParams::new()
        .bbox(w, s, e, n)
        .datetime(&cfg.datetime)
        .collections(&[cfg.collection.as_str()]);
    let mut items = client.search_all(&params)?;
    if items.is_empty() {
        return Err(StackError::Empty(format!(
            "search returned no items for {} in {}",
            cfg.collection, cfg.datetime
        )));
    }
    // sort by the parsed time coordinate (robust to mixed datetime formats);
    // items whose datetime cannot be parsed sort first and are skipped below
    let time_key = |item: &StacItem| {
        item.properties
            .datetime
            .as_deref()
            .and_then(fractional_year)
            .unwrap_or(f64::NEG_INFINITY)
    };
    items.sort_by(|a, b| time_key(a).total_cmp(&time_key(b)));
    // some catalogs (Earth Search) repeat items across result pages; a
    // duplicated scene would otherwise become a duplicated time slice
    let mut seen_ids = std::collections::HashSet::new();
    items.retain(|item| seen_ids.insert(item.id.clone()));

    let wgs_bbox = BBox::new(w, s, e, n);
    let mut skipped = Vec::new();
    // an explicit GridSpec fixes the reference grid up front; otherwise the
    // first successfully-read scene defines it.
    let mut reference: Option<Raster<f64>> = match &cfg.grid {
        Some(spec) => Some(reference_from_grid(spec, &wgs_bbox)?),
        None => None,
    };
    let mut ref_epsg: Option<u32> = cfg.grid.as_ref().map(|g| g.epsg);
    let mut scenes: Vec<(SliceMeta, Vec<Raster<f64>>)> = Vec::new();

    for item in &items {
        let Some(datetime) = item.properties.datetime.clone() else {
            skipped.push(format!("{}: item has no datetime", item.id));
            continue;
        };
        let Some(time) = fractional_year(&datetime) else {
            skipped.push(format!("{}: unparseable datetime '{datetime}'", item.id));
            continue;
        };
        if let (Some(max), Some(cc)) = (cfg.max_cloud_cover, item.properties.eo_cloud_cover)
            && cc > max
        {
            skipped.push(format!("{}: cloud cover {cc:.0}% > {max:.0}%", item.id));
            continue;
        }
        // when cross-zone mosaicking is off, keep the old behaviour: scenes
        // in a different CRS than the reference are skipped.
        if !cfg.cross_zone_mosaic
            && let (Some(re), Some(ie)) = (ref_epsg, item.epsg())
            && re != ie
        {
            skipped.push(format!(
                "{}: EPSG {ie} differs from reference EPSG {re} (cross-zone mosaic off)",
                item.id
            ));
            continue;
        }

        match read_scene(
            &client,
            item,
            cfg,
            &wgs_bbox,
            needs_signing,
            reference.as_ref(),
            ref_epsg,
        ) {
            Ok(rasters) => {
                if reference.is_none() {
                    reference = Some(rasters[0].clone());
                    ref_epsg = item.epsg();
                }
                let meta = SliceMeta {
                    item_id: item.id.clone(),
                    datetime,
                    time,
                    cloud_cover: item.properties.eo_cloud_cover,
                };
                scenes.push((meta, rasters));
            }
            Err(err) => skipped.push(format!("{}: {err}", item.id)),
        }
    }

    if scenes.is_empty() {
        return Err(StackError::Empty(format!(
            "no scene could be read ({} skipped: {})",
            skipped.len(),
            skipped.join("; ")
        )));
    }
    let reference = reference.expect("a read scene implies a reference grid");

    let (ny, nx) = reference.shape();
    let nb = cfg.assets.len();
    let nt = scenes.len();
    let mut data = Array4::from_elem((nb, ny, nx, nt), f64::NAN);
    let mut times = Vec::with_capacity(nt);
    let mut slices = Vec::with_capacity(nt);
    for (ti, (meta, rasters)) in scenes.into_iter().enumerate() {
        for (bi, raster) in rasters.iter().enumerate() {
            data.slice_mut(ndarray::s![bi, .., .., ti])
                .assign(raster.data());
        }
        times.push(meta.time);
        slices.push(meta);
    }

    let cube = Cube::new(data, times, cfg.assets.clone())?;
    Ok(StackedCube {
        cube,
        slices,
        skipped,
        transform: *reference.transform(),
        epsg: ref_epsg.or_else(|| reference.crs().and_then(|c| c.epsg())),
    })
}

/// Reads every requested asset of one item, aligned to the reference grid
/// (or defining it, for the first scene when no [`GridSpec`] is set).
fn read_scene(
    client: &StacClientBlocking,
    item: &StacItem,
    cfg: &StackConfig,
    wgs_bbox: &BBox,
    needs_signing: bool,
    reference: Option<&Raster<f64>>,
    ref_epsg: Option<u32>,
) -> Result<Vec<Raster<f64>>, StackError> {
    let collection = item.collection.as_deref().unwrap_or(&cfg.collection);
    let mut rasters: Vec<Raster<f64>> = Vec::with_capacity(cfg.assets.len());

    // the quality mask is read once per scene (same CRS as the data assets,
    // possibly a coarser grid) and applied to each band at its native grid.
    let mask = cfg
        .mask
        .as_ref()
        .map(|mc| {
            read_asset(
                client,
                item,
                collection,
                &mc.asset,
                cfg,
                wgs_bbox,
                needs_signing,
            )
            .map(|raster| (raster, mc))
        })
        .transpose()?;

    for key in &cfg.assets {
        let mut raster = read_asset(client, item, collection, key, cfg, wgs_bbox, needs_signing)?;
        nodata_to_nan(&mut raster);
        if cfg.scale != 1.0 || cfg.offset != 0.0 {
            let (scale, offset) = (cfg.scale, cfg.offset);
            raster.data_mut().mapv_inplace(|v| v * scale + offset);
        }

        // per-pixel quality masking, before any reprojection/resampling so
        // masked values never bleed into their neighbours.
        if let Some((mask_raster, mc)) = &mask {
            apply_mask(&mut raster, mask_raster, mc)?;
        }

        // cross-UTM-zone mosaicking: if this scene sits in a different zone
        // than the reference, reproject it onto the reference zone (UTM↔UTM,
        // bilinear) before grid alignment. NaN nodata is preserved.
        if cfg.cross_zone_mosaic
            && let Some(ref_epsg) = ref_epsg
        {
            let scene_epsg = item.epsg().or_else(|| raster.crs().and_then(|c| c.epsg()));
            if let Some(scene_epsg) = scene_epsg
                && scene_epsg != ref_epsg
            {
                raster = reproject::reproject_raster_utm(&raster, scene_epsg, ref_epsg)
                    .ok_or_else(|| {
                        StackError::Reproject(format!(
                            "EPSG {scene_epsg} -> {ref_epsg} (non-UTM or degenerate extent)"
                        ))
                    })?;
            }
        }

        // first asset of the first scene defines the grid; everything else
        // (other bands at other resolutions, later scenes, reprojected tiles)
        // aligns to it
        let target = reference.or(rasters.first());
        if let Some(target) = target
            && needs_resample(&raster, target)
        {
            raster = resample_to_grid(&raster, target, cfg.resample)?;
        }
        rasters.push(raster);
    }
    Ok(rasters)
}

/// Opens one COG asset of an item and reads it windowed to the search bbox.
fn read_asset(
    client: &StacClientBlocking,
    item: &StacItem,
    collection: &str,
    key: &str,
    cfg: &StackConfig,
    wgs_bbox: &BBox,
    needs_signing: bool,
) -> Result<Raster<f64>, StackError> {
    let asset = item
        .asset(key)
        .ok_or_else(|| StackError::Config(format!("asset '{key}' not found in item")))?;
    let href = if needs_signing {
        client.sign_asset_href(&asset.href, collection)?
    } else {
        asset.href.clone()
    };
    let mut reader = CogReaderBlocking::open(&href, CogReaderOptions::default())?;
    let read_bbox = resolve_read_bbox(wgs_bbox, item, &reader);
    Ok(reader.read_bbox(&read_bbox, cfg.overview)?)
}

/// Masks `band` in place: pixels whose mask class is not in the keep-list
/// become `NaN`. The mask is aligned to the band's grid first (nearest
/// neighbour for categorical bands); mask nodata/NaN also masks the pixel.
fn apply_mask(
    band: &mut Raster<f64>,
    mask: &Raster<f64>,
    cfg: &MaskConfig,
) -> Result<(), StackError> {
    let aligned;
    let mask = if needs_resample(mask, band) {
        aligned = resample_to_grid(mask, band, cfg.resample)?;
        &aligned
    } else {
        mask
    };
    let keep = &cfg.keep;
    band.data_mut().zip_mut_with(mask.data(), |v, &class| {
        if !(class.is_finite() && keep.contains(&(class as u16))) {
            *v = f64::NAN;
        }
    });
    Ok(())
}

/// Largest per-axis size a [`GridSpec`] may produce (same guard as the
/// UTM↔UTM reprojector).
const MAX_GRID_DIM: usize = 100_000;

/// Builds the synthetic reference raster that carries a [`GridSpec`]'s grid
/// (shape + transform + CRS); its values are never read.
fn reference_from_grid(spec: &GridSpec, wgs_bbox: &BBox) -> Result<Raster<f64>, StackError> {
    let bbox = match spec.bbox {
        Some([min_x, min_y, max_x, max_y]) => BBox::new(min_x, min_y, max_x, max_y),
        None => reproject::reproject_bbox_to_cog(wgs_bbox, spec.epsg),
    };
    // grid origin = upper-left corner, optionally snapped outward
    let (mut x0, mut y1) = (bbox.min_x, bbox.max_y);
    if let Some(step) = spec.align {
        x0 = (x0 / step).floor() * step;
        y1 = (y1 / step).ceil() * step;
    }
    let res = spec.resolution;
    let nx = ((bbox.max_x - x0) / res).ceil() as usize;
    let ny = ((y1 - bbox.min_y) / res).ceil() as usize;
    if !(1..=MAX_GRID_DIM).contains(&nx) || !(1..=MAX_GRID_DIM).contains(&ny) {
        return Err(StackError::Config(format!(
            "grid spec yields a {ny}x{nx} px grid (allowed: 1..={MAX_GRID_DIM} per axis); \
             check the resolution/bbox units against EPSG {}",
            spec.epsg
        )));
    }
    let mut raster = Raster::from_array(Array2::zeros((ny, nx)));
    raster.set_transform(GeoTransform::new(x0, y1, res, -res));
    raster.set_crs(Some(CRS::from_epsg(spec.epsg)));
    raster.set_nodata(Some(f64::NAN));
    Ok(raster)
}

/// Same bbox resolution as `surtgis_cloud::stac_reader`: prefer `proj:epsg`
/// from the item, fall back to the COG metadata CRS.
fn resolve_read_bbox(bbox: &BBox, item: &StacItem, reader: &CogReaderBlocking) -> BBox {
    if let Some(epsg) = item.epsg()
        && !reproject::is_wgs84(epsg)
    {
        return reproject::reproject_bbox_to_cog(bbox, epsg);
    }
    if let Some(epsg) = reader.metadata().crs.as_ref().and_then(|c| c.epsg())
        && !reproject::is_wgs84(epsg)
    {
        return reproject::reproject_bbox_to_cog(bbox, epsg);
    }
    *bbox
}

fn nodata_to_nan(raster: &mut Raster<f64>) {
    if let Some(nd) = raster.nodata() {
        raster
            .data_mut()
            .mapv_inplace(|v| if v == nd { f64::NAN } else { v });
        raster.set_nodata(Some(f64::NAN));
    }
}

fn needs_resample(raster: &Raster<f64>, target: &Raster<f64>) -> bool {
    raster.shape() != target.shape() || raster.transform() != target.transform()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        let base = StackConfig::new("pc", "sentinel-2-l2a", &["B04"]);
        assert!(matches!(
            base.clone().validate(),
            Err(StackError::Config(_))
        )); // no bbox

        let no_assets = StackConfig::new("pc", "sentinel-2-l2a", &[])
            .bbox(-70.0, -34.0, -69.0, -33.0)
            .datetime("2024-01-01/2024-12-31");
        assert!(matches!(no_assets.validate(), Err(StackError::Config(_))));

        let bad_bbox = StackConfig::new("pc", "sentinel-2-l2a", &["B04"])
            .bbox(-69.0, -34.0, -70.0, -33.0)
            .datetime("2024-01-01/2024-12-31");
        assert!(matches!(bad_bbox.validate(), Err(StackError::Config(_))));

        let ok = StackConfig::new("pc", "sentinel-2-l2a", &["B04"])
            .bbox(-70.0, -34.0, -69.0, -33.0)
            .datetime("2024-01-01/2024-12-31");
        assert!(ok.validate().is_ok());
    }

    fn raster_at(data: Array2<f64>, x0: f64, y1: f64, res: f64) -> Raster<f64> {
        let mut r = Raster::from_array(data);
        r.set_transform(GeoTransform::new(x0, y1, res, -res));
        r.set_crs(Some(CRS::from_epsg(32719)));
        r
    }

    #[test]
    fn mask_and_grid_validation() {
        let ok = || {
            StackConfig::new("pc", "sentinel-2-l2a", &["B04"])
                .bbox(-70.0, -34.0, -69.0, -33.0)
                .datetime("2024-01-01/2024-12-31")
        };
        assert!(ok().mask(MaskConfig::scl()).validate().is_ok());

        let empty_keep = MaskConfig {
            keep: vec![],
            ..MaskConfig::scl()
        };
        assert!(matches!(
            ok().mask(empty_keep).validate(),
            Err(StackError::Config(_))
        ));

        assert!(ok().grid(GridSpec::new(32719, 10.0)).validate().is_ok());
        assert!(matches!(
            ok().grid(GridSpec::new(32719, 0.0)).validate(),
            Err(StackError::Config(_))
        ));
        assert!(matches!(
            ok().grid(GridSpec::new(32719, 10.0).bbox(10.0, 0.0, 0.0, 5.0))
                .validate(),
            Err(StackError::Config(_))
        ));
        assert!(matches!(
            ok().grid(GridSpec::new(32719, 10.0).align(-60.0))
                .validate(),
            Err(StackError::Config(_))
        ));
    }

    #[test]
    fn mask_keeps_only_listed_classes() {
        // band and mask share the same 2x2 grid
        let mut band = raster_at(
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            0.0,
            20.0,
            10.0,
        );
        let mask = raster_at(
            Array2::from_shape_vec((2, 2), vec![4.0, 8.0, f64::NAN, 5.0]).unwrap(),
            0.0,
            20.0,
            10.0,
        );
        apply_mask(&mut band, &mask, &MaskConfig::scl()).unwrap();
        let d = band.data();
        assert_eq!(d[[0, 0]], 1.0); // class 4 kept
        assert!(d[[0, 1]].is_nan()); // class 8 (cloud) masked
        assert!(d[[1, 0]].is_nan()); // mask NaN masks the pixel
        assert_eq!(d[[1, 1]], 4.0); // class 5 kept
    }

    #[test]
    fn mask_resamples_coarser_grid_nearest() {
        // 4x4 band at 10 m, 2x2 mask at 20 m over the same extent (like
        // S2 10 m bands vs the 20 m SCL): each mask cell covers 2x2 pixels
        let mut band = raster_at(Array2::from_elem((4, 4), 1.0), 0.0, 40.0, 10.0);
        let mask = raster_at(
            Array2::from_shape_vec((2, 2), vec![4.0, 9.0, 9.0, 4.0]).unwrap(),
            0.0,
            40.0,
            20.0,
        );
        apply_mask(&mut band, &mask, &MaskConfig::scl()).unwrap();
        let d = band.data();
        for r in 0..4 {
            for c in 0..4 {
                let clear = (r < 2 && c < 2) || (r >= 2 && c >= 2);
                assert_eq!(d[[r, c]].is_nan(), !clear, "pixel ({r},{c})");
            }
        }
    }

    #[test]
    fn grid_reference_is_deterministic() {
        let spec = GridSpec::new(32719, 10.0).bbox(300_000.0, 6_280_000.0, 300_100.0, 6_280_050.0);
        let r = reference_from_grid(&spec, &BBox::new(0.0, 0.0, 1.0, 1.0)).unwrap();
        assert_eq!(r.shape(), (5, 10));
        let gt = r.transform();
        assert_eq!(gt.origin_x, 300_000.0);
        assert_eq!(gt.origin_y, 6_280_050.0);
        assert_eq!(gt.pixel_width, 10.0);
        assert_eq!(gt.pixel_height, -10.0);
        assert_eq!(r.crs().and_then(|c| c.epsg()), Some(32719));
    }

    #[test]
    fn grid_reference_snaps_origin_outward_with_align() {
        let spec = GridSpec::new(32719, 10.0)
            .bbox(300_010.0, 6_280_000.0, 300_100.0, 6_280_015.0)
            .align(60.0);
        let r = reference_from_grid(&spec, &BBox::new(0.0, 0.0, 1.0, 1.0)).unwrap();
        let gt = r.transform();
        assert_eq!(gt.origin_x, 300_000.0); // floor(300010/60)*60 — origin moves west
        assert_eq!(gt.origin_y, 6_280_020.0); // ceil(6280015/60)*60 — origin moves north
        // the extent still covers the requested bbox
        assert_eq!(r.shape(), (2, 10));
    }

    #[test]
    fn grid_reference_derives_extent_from_wgs84_bbox() {
        // Santiago-area bbox → UTM 19S: easting/northing in plausible ranges
        let spec = GridSpec::new(32719, 100.0);
        let wgs = BBox::new(-70.70, -33.50, -70.68, -33.48);
        let r = reference_from_grid(&spec, &wgs).unwrap();
        let gt = r.transform();
        assert!(
            (100_000.0..900_000.0).contains(&gt.origin_x),
            "easting {}",
            gt.origin_x
        );
        assert!(
            (6_000_000.0..6_500_000.0).contains(&gt.origin_y),
            "northing {}",
            gt.origin_y
        );
        // ~2 km x ~2 km at 100 m → about 20x20 px
        let (ny, nx) = r.shape();
        assert!(
            (15..=30).contains(&ny) && (15..=30).contains(&nx),
            "{ny}x{nx}"
        );
    }

    #[test]
    fn grid_reference_rejects_oversized_grids() {
        // sub-millimetre resolution over 100 m → >100k px per axis
        let spec = GridSpec::new(32719, 1e-4).bbox(0.0, 0.0, 100.0, 100.0);
        assert!(matches!(
            reference_from_grid(&spec, &BBox::new(0.0, 0.0, 1.0, 1.0)),
            Err(StackError::Config(_))
        ));
    }

    #[test]
    fn cross_zone_default_on_and_toggle() {
        let cfg = StackConfig::new("pc", "sentinel-2-l2a", &["B04"]);
        assert!(cfg.cross_zone_mosaic, "mosaicking should default to on");
        assert!(!cfg.cross_zone_mosaic(false).cross_zone_mosaic);
    }

    /// End-to-end against the real Planetary Computer (network):
    /// `cargo test -p datacube-io -- --ignored`
    #[test]
    #[ignore = "requires network access to Planetary Computer"]
    fn stacks_sentinel2_red_band() {
        let cfg = StackConfig::new("pc", "sentinel-2-l2a", &["B04"])
            .bbox(-70.70, -33.50, -70.68, -33.48)
            .datetime("2024-01-01/2024-03-31")
            .max_cloud_cover(40.0)
            .max_items(10)
            .overview(Some(3));
        let stacked = stack(&cfg).expect("stack should succeed");
        let (nb, ny, nx, nt) = stacked.cube.dims();
        assert_eq!(nb, 1);
        assert!(nt >= 2, "expected at least 2 scenes, got {nt}");
        assert!(ny > 0 && nx > 0);
        assert_eq!(stacked.slices.len(), nt);
        assert!(stacked.cube.time().windows(2).all(|w| w[0] <= w[1]));
    }
}
