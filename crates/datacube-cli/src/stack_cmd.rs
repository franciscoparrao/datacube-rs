//! `datacube stack`: STAC search → COG reads → cube → per-pixel trend maps.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use datacube_core::{CompositeMethod, CompositeWindow, stats};
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
    /// Multiply values by this factor (e.g. 0.0001 for S2 L2A reflectance)
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Add this offset after --scale (e.g. -0.1 for S2 baseline >= 04.00)
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    offset: f64,
    /// Skip scenes from other UTM zones instead of reprojecting them onto the
    /// reference grid (cross-zone mosaicking is on by default)
    #[arg(long)]
    no_cross_zone: bool,
    /// Composite slices before analysis
    #[arg(long, value_enum)]
    composite: Option<CompositeKind>,
    /// Aggregation for --composite
    #[arg(long, value_enum, default_value_t = CompositeAgg::Median)]
    composite_method: CompositeAgg,
    /// Fill temporal NaN gaps by linear interpolation, skipping gaps wider
    /// than this many time units (in fractional years; 0 = no limit)
    #[arg(long)]
    gapfill: Option<f64>,
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
    /// Write a per-pixel break-count map (BFAST-style) to this GeoTIFF.
    /// Uses --band, --harmonics and --break-alpha.
    #[arg(long)]
    breaks_output: Option<PathBuf>,
    /// Write a per-pixel map of the first break time (fractional years; NaN
    /// where no break) to this GeoTIFF
    #[arg(long)]
    first_break_output: Option<PathBuf>,
    /// Fourier pairs in the per-pixel break model (0 = trend only)
    #[arg(long, default_value_t = 1)]
    break_harmonics: usize,
    /// Significance level for per-pixel break detection
    #[arg(long, default_value_t = 0.05)]
    break_alpha: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrendStat {
    TheilSen,
    Ols,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CompositeKind {
    /// Merge tiles acquired at the same instant
    SameTime,
    /// Monthly bins (1/12 of a fractional year)
    Monthly,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CompositeAgg {
    Median,
    Mean,
    Min,
    Max,
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
        .overview(args.overview)
        .scaling(args.scale, args.offset)
        .cross_zone_mosaic(!args.no_cross_zone);
    if let Some(mc) = args.max_cloud {
        cfg = cfg.max_cloud_cover(mc);
    }

    eprintln!("searching {} in {} ...", args.collection, args.catalog);
    let stacked = stack(&cfg).context("stacking failed")?;
    {
        let (nb, ny, nx, nt) = stacked.cube.dims();
        eprintln!(
            "stacked {nt} scenes ({nb} bands, {ny}x{nx} px), {} skipped",
            stacked.skipped.len()
        );
    }

    let mut cube = stacked.cube.clone();
    if let Some(kind) = args.composite {
        let window = match kind {
            CompositeKind::SameTime => CompositeWindow::SameTime,
            CompositeKind::Monthly => CompositeWindow::Period(1.0 / 12.0),
        };
        let method = match args.composite_method {
            CompositeAgg::Median => CompositeMethod::Median,
            CompositeAgg::Mean => CompositeMethod::Mean,
            CompositeAgg::Min => CompositeMethod::Min,
            CompositeAgg::Max => CompositeMethod::Max,
        };
        cube = cube
            .composite(window, method)
            .context("compositing failed")?;
        eprintln!("composited to {} slices", cube.dims().3);
    }
    if let Some(mg) = args.gapfill {
        let max_gap = if mg > 0.0 { Some(mg) } else { None };
        cube = cube.gapfill_linear(max_gap).context("gap-filling failed")?;
    }
    let (nb, ny, nx, nt) = cube.dims();

    let wants_trend = args.output.is_some() || args.pvalue_output.is_some();
    let wants_breaks = args.breaks_output.is_some() || args.first_break_output.is_some();

    let mut maps_written = Vec::new();
    if wants_trend || wants_breaks {
        let band_key = args.band.as_deref().unwrap_or(&args.assets[0]);
        let band = cube
            .bands()
            .iter()
            .position(|b| b == band_key)
            .with_context(|| format!("band '{band_key}' is not in the stacked assets"))?;

        if wants_trend {
            let (slope, pvalue) = trend_maps(&cube, band, args.stat)?;
            if let Some(path) = &args.output {
                write_map(&slope, &stacked, path)?;
                maps_written.push(path.display().to_string());
            }
            if let Some(path) = &args.pvalue_output {
                write_map(&pvalue, &stacked, path)?;
                maps_written.push(path.display().to_string());
            }
        }
        if wants_breaks {
            let (count, first) = break_maps(&cube, band, args.break_harmonics, args.break_alpha)?;
            if let Some(path) = &args.breaks_output {
                write_map(&count, &stacked, path)?;
                maps_written.push(path.display().to_string());
            }
            if let Some(path) = &args.first_break_output {
                write_map(&first, &stacked, path)?;
                maps_written.push(path.display().to_string());
            }
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
        "bands": cube.bands(),
        "time_range": [cube.time().first(), cube.time().last()],
        "epsg": stacked.epsg,
        "maps_written": maps_written,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Per-pixel slope and p-value grids for the selected band.
fn trend_maps(
    cube: &datacube_core::Cube,
    band: usize,
    stat: TrendStat,
) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>)> {
    let results = cube
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

/// Per-pixel break-count and first-break-time grids (BFAST-style).
/// Pixels with too few finite observations yield `NaN`.
fn break_maps(
    cube: &datacube_core::Cube,
    band: usize,
    harmonics: usize,
    alpha: f64,
) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>)> {
    let opts = stats::BreakOptions {
        alpha,
        n_harmonics: harmonics,
        period: 1.0,
        min_segment: stats::BreakOptions::default()
            .min_segment
            .max(2 * harmonics + 4),
    };
    let results = cube
        .par_map_series(band, |t, y| match stats::detect_breaks(t, y, &opts) {
            Ok(r) => {
                let first = r.breaks.first().map(|b| b.time).unwrap_or(f64::NAN);
                (r.breaks.len() as f64, first)
            }
            // too few observations / degenerate series → no break info
            Err(_) => (f64::NAN, f64::NAN),
        })
        .context("per-pixel break detection failed")?;
    let count = results.mapv(|(c, _)| c);
    let first = results.mapv(|(_, f)| f);
    Ok((count, first))
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
