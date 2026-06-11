//! `datacube stack`: STAC search → COG reads → cube → per-pixel trend maps.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use datacube_core::stats;
use datacube_io::{StackConfig, StackedCube, stack};
use surtgis_core::io::write_geotiff;
use surtgis_core::{CRS, Raster};

#[derive(clap::Args)]
pub struct StackArgs {
    /// STAC catalog: "pc" (Planetary Computer), "es" (Earth Search) or a URL
    #[arg(long, default_value = "pc")]
    catalog: String,
    /// Collection id
    #[arg(long, default_value = "sentinel-2-l2a")]
    collection: String,
    /// Comma-separated asset keys stacked as cube bands (e.g. B04,B08)
    #[arg(long, value_delimiter = ',', default_value = "B04")]
    assets: Vec<String>,
    /// WGS84 bbox: west,south,east,north
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    bbox: Vec<f64>,
    /// Datetime range, e.g. 2023-01-01/2024-12-31
    #[arg(long)]
    datetime: String,
    /// Skip scenes with eo:cloud_cover above this percentage
    #[arg(long)]
    max_cloud: Option<f64>,
    /// Maximum scenes to fetch from the search
    #[arg(long, default_value_t = 100)]
    limit: usize,
    /// COG overview level (higher = coarser & faster; omit for full res)
    #[arg(long)]
    overview: Option<usize>,
    /// Band (asset key) for the trend statistic
    #[arg(long)]
    band: Option<String>,
    /// Trend estimator for --output
    #[arg(long, value_enum, default_value_t = TrendStat::TheilSen)]
    stat: TrendStat,
    /// Write the per-pixel slope map to this GeoTIFF
    #[arg(long)]
    output: Option<PathBuf>,
    /// Write the per-pixel p-value map (Mann-Kendall for theil-sen, t-test
    /// for ols) to this GeoTIFF
    #[arg(long)]
    pvalue_output: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrendStat {
    TheilSen,
    Ols,
}

pub fn run(args: &StackArgs) -> Result<()> {
    if args.bbox.len() != 4 {
        bail!("--bbox needs west,south,east,north");
    }
    let asset_refs: Vec<&str> = args.assets.iter().map(String::as_str).collect();
    let mut cfg = StackConfig::new(&args.catalog, &args.collection, &asset_refs)
        .bbox(args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
        .datetime(&args.datetime)
        .max_items(args.limit)
        .overview(args.overview);
    if let Some(mc) = args.max_cloud {
        cfg = cfg.max_cloud_cover(mc);
    }

    eprintln!("searching {} in {} ...", args.collection, args.catalog);
    let stacked = stack(&cfg).context("stacking failed")?;
    let (nb, ny, nx, nt) = stacked.cube.dims();
    eprintln!(
        "stacked {nt} scenes ({nb} bands, {ny}x{nx} px), {} skipped",
        stacked.skipped.len()
    );

    let mut maps_written = Vec::new();
    if args.output.is_some() || args.pvalue_output.is_some() {
        let band_key = args.band.as_deref().unwrap_or(&args.assets[0]);
        let band = stacked
            .cube
            .bands()
            .iter()
            .position(|b| b == band_key)
            .with_context(|| format!("band '{band_key}' is not in the stacked assets"))?;

        let (slope, pvalue) = trend_maps(&stacked, band, args.stat)?;
        if let Some(path) = &args.output {
            write_map(&slope, &stacked, path)?;
            maps_written.push(path.display().to_string());
        }
        if let Some(path) = &args.pvalue_output {
            write_map(&pvalue, &stacked, path)?;
            maps_written.push(path.display().to_string());
        }
    }

    let report = serde_json::json!({
        "scenes": stacked.slices.iter().map(|s| serde_json::json!({
            "id": s.item_id,
            "datetime": s.datetime,
            "time": s.time,
            "cloud_cover": s.cloud_cover,
        })).collect::<Vec<_>>(),
        "skipped": stacked.skipped,
        "dims": { "bands": nb, "height": ny, "width": nx, "times": nt },
        "bands": stacked.cube.bands(),
        "time_range": [stacked.cube.time().first(), stacked.cube.time().last()],
        "epsg": stacked.epsg,
        "maps_written": maps_written,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Per-pixel slope and p-value grids for the selected band.
fn trend_maps(
    stacked: &StackedCube,
    band: usize,
    stat: TrendStat,
) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>)> {
    let results = stacked
        .cube
        .par_map_series(band, |t, y| match stat {
            TrendStat::TheilSen => {
                let slope = stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN);
                let p = stats::mann_kendall(y)
                    .map(|r| r.p_value)
                    .unwrap_or(f64::NAN);
                (slope, p)
            }
            TrendStat::Ols => stats::linear_trend(t, y)
                .map(|r| (r.slope, r.p_value))
                .unwrap_or((f64::NAN, f64::NAN)),
        })
        .context("per-pixel trend computation failed")?;
    let slope = results.mapv(|(s, _)| s);
    let pvalue = results.mapv(|(_, p)| p);
    Ok((slope, pvalue))
}

/// Writes a float map on the stack's grid as GeoTIFF (f32, NaN nodata).
fn write_map(values: &ndarray::Array2<f64>, stacked: &StackedCube, path: &PathBuf) -> Result<()> {
    let (ny, nx) = values.dim();
    let mut raster = Raster::<f32>::new(ny, nx);
    {
        let data = raster.data_mut();
        for ((r, c), v) in values.indexed_iter() {
            data[[r, c]] = *v as f32;
        }
    }
    raster.set_transform(stacked.transform);
    raster.set_crs(stacked.epsg.map(CRS::from_epsg));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None)
        .map_err(|e| anyhow::anyhow!("writing {} failed: {e}", path.display()))
}
