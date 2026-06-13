use datacube_core::Cube;
use ndarray::Array4;
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking};
use surtgis_cloud::stac_models::StacItem;
use surtgis_cloud::{
    BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacSearchParams, reproject,
};
use surtgis_core::{GeoTransform, Raster, ResampleMethod, resample_to_grid};

use crate::StackError;
use crate::time::fractional_year;

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
/// The first successfully-read scene defines the reference grid; every other
/// scene is resampled onto it (`cfg.resample`). Scenes in a different CRS
/// than the reference are skipped (cross-UTM-zone mosaicking is out of
/// scope) and reported in [`StackedCube::skipped`]. Nodata becomes `NaN`;
/// values stay raw unless [`StackConfig::scaling`] is set.
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
    items.sort_by(|a, b| a.properties.datetime.cmp(&b.properties.datetime));

    let wgs_bbox = BBox::new(w, s, e, n);
    let mut skipped = Vec::new();
    let mut reference: Option<Raster<f64>> = None;
    let mut ref_epsg: Option<u32> = None;
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
        if let (Some(re), Some(ie)) = (ref_epsg, item.epsg())
            && re != ie
        {
            skipped.push(format!(
                "{}: EPSG {ie} differs from reference EPSG {re}",
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

    let Some(reference) = reference else {
        return Err(StackError::Empty(format!(
            "no scene could be read ({} skipped: {})",
            skipped.len(),
            skipped.join("; ")
        )));
    };

    let (ny, nx) = reference.shape();
    let nb = cfg.assets.len();
    let nt = scenes.len();
    let mut data = Array4::from_elem((nb, ny, nx, nt), f64::NAN);
    let mut times = Vec::with_capacity(nt);
    let mut slices = Vec::with_capacity(nt);
    for (ti, (meta, rasters)) in scenes.into_iter().enumerate() {
        for (bi, raster) in rasters.iter().enumerate() {
            let src = raster.data();
            for r in 0..ny {
                for c in 0..nx {
                    data[[bi, r, c, ti]] = src[[r, c]];
                }
            }
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
/// (or defining it, for the first scene).
fn read_scene(
    client: &StacClientBlocking,
    item: &StacItem,
    cfg: &StackConfig,
    wgs_bbox: &BBox,
    needs_signing: bool,
    reference: Option<&Raster<f64>>,
) -> Result<Vec<Raster<f64>>, StackError> {
    let collection = item.collection.as_deref().unwrap_or(&cfg.collection);
    let mut rasters: Vec<Raster<f64>> = Vec::with_capacity(cfg.assets.len());

    for key in &cfg.assets {
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
        let mut raster: Raster<f64> = reader.read_bbox(&read_bbox, cfg.overview)?;
        nodata_to_nan(&mut raster);
        if cfg.scale != 1.0 || cfg.offset != 0.0 {
            let (scale, offset) = (cfg.scale, cfg.offset);
            raster.data_mut().mapv_inplace(|v| v * scale + offset);
        }

        // first asset of the first scene defines the grid; everything else
        // (other bands at other resolutions, later scenes) aligns to it
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
