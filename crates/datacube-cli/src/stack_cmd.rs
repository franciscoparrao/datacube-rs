//! `datacube stack`: STAC search → COG reads → cube → per-pixel trend maps.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use datacube_core::{
    ChunkPipeline, ChunkResult, CompositeMethod, CompositeWindow, Cube, GapfillSpec, GeoRef,
    IndexSpec, StatSpec, TrendMethod, stats::BreakOptions,
};
use datacube_io::{GridSpec, MaskConfig, StackConfig, StackedCube, stack};
use datacube_zarr::ZarrCubeWriter;
use ndarray::{Array2, s};
use surtgis_core::io::write_geotiff;
use surtgis_core::{CRS, GeoTransform, Raster};

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
    /// Scenes read concurrently once the reference grid is known (stacking
    /// is network-bound; higher values cut wall-clock time for 30-100 scene
    /// stacks, within the catalog's rate limits)
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    /// Mask pixels per scene with the S2 SCL band, keeping only clear classes
    /// (vegetation, bare, water, unclassified, snow); see --mask-asset/--mask-keep
    #[arg(long)]
    mask_scl: bool,
    /// Quality-band asset key for --mask-scl
    #[arg(long, default_value = "SCL")]
    mask_asset: String,
    /// Comma-separated class values kept by --mask-scl
    #[arg(long, value_delimiter = ',', default_values_t = [4u16, 5, 6, 7, 11])]
    mask_keep: Vec<u16>,
    /// Explicit target grid EPSG (requires --grid-res); the cube grid then no
    /// longer depends on which scene is read first
    #[arg(long)]
    grid_epsg: Option<u32>,
    /// Explicit grid resolution in CRS units (requires --grid-epsg)
    #[arg(long)]
    grid_res: Option<f64>,
    /// Explicit grid extent in the target CRS: minx,miny,maxx,maxy
    /// (default: derived from --bbox)
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    grid_bbox: Option<Vec<f64>>,
    /// Snap the grid origin outward to a multiple of this value
    /// (e.g. 60 to align with the Sentinel-2 MGRS grid)
    #[arg(long)]
    grid_align: Option<f64>,
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
    /// Compute a spectral index from the stacked bands before analysis; the
    /// trend/break maps then run on the index. Band roles come from the
    /// --nir/--red/--green/--blue/--swir flags (Sentinel-2 defaults).
    #[arg(long, value_enum)]
    index: Option<IndexKind>,
    /// NIR asset key for --index
    #[arg(long, default_value = "B08")]
    nir: String,
    /// Red asset key for --index
    #[arg(long, default_value = "B04")]
    red: String,
    /// Green asset key for --index
    #[arg(long, default_value = "B03")]
    green: String,
    /// Blue asset key for --index (EVI)
    #[arg(long, default_value = "B02")]
    blue: String,
    /// SWIR asset key for --index (NBR/NDBI)
    #[arg(long, default_value = "B11")]
    swir: String,
    /// Soil-brightness factor L for --index savi
    #[arg(long, default_value_t = 0.5)]
    savi_l: f64,
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
    /// Write a per-pixel break-count map (OLS-CUSUM) to this GeoTIFF.
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
    /// Spatial tile size for the composite/gapfill/index/trend/breaks chain:
    /// each tile runs the whole chain independently, bounding peak memory to
    /// one tile instead of one full-cube copy per stage (same default as the
    /// GeoZarr chunk size)
    #[arg(long, default_value_t = 256)]
    chunk_size: usize,
    /// Persist the processed cube (after mask/composite/gapfill/index, before
    /// the trend/breaks statistics) to a GeoZarr store at this path, written
    /// one spatial tile at a time (--chunk-size)
    #[arg(long)]
    zarr_output: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrendStat {
    TheilSen,
    Ols,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum IndexKind {
    Ndvi,
    Ndwi,
    Nbr,
    Ndbi,
    Evi,
    Savi,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CompositeKind {
    /// Merge tiles acquired at the same instant
    SameTime,
    /// Calendar-month bins (year + month recovered from the time axis)
    Monthly,
    /// Calendar-year bins
    Yearly,
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
        .cross_zone_mosaic(!args.no_cross_zone)
        .concurrency(args.concurrency);
    if let Some(mc) = args.max_cloud {
        cfg = cfg.max_cloud_cover(mc);
    }
    if args.mask_scl {
        cfg = cfg.mask(MaskConfig {
            asset: args.mask_asset.clone(),
            keep: args.mask_keep.clone(),
            ..MaskConfig::scl()
        });
    }
    match (args.grid_epsg, args.grid_res) {
        (Some(epsg), Some(res)) => {
            let mut spec = GridSpec::new(epsg, res);
            if let Some(gb) = &args.grid_bbox {
                if gb.len() != 4 {
                    bail!("--grid-bbox needs minx,miny,maxx,maxy");
                }
                spec = spec.bbox(gb[0], gb[1], gb[2], gb[3]);
            }
            if let Some(step) = args.grid_align {
                spec = spec.align(step);
            }
            cfg = cfg.grid(spec);
        }
        (None, None) => {
            if args.grid_bbox.is_some() || args.grid_align.is_some() {
                bail!("--grid-bbox/--grid-align require --grid-epsg and --grid-res");
            }
        }
        _ => bail!("--grid-epsg and --grid-res must be given together"),
    }

    eprintln!("searching {} in {} ...", args.collection, args.catalog);
    // destructure to take ownership of the cube (no full-cube clone); the
    // grid (transform/EPSG) travels with the cube itself as a GeoRef and
    // survives composite/gapfill/index, so it's read back from the cube at
    // write time instead of being threaded through this function by hand.
    let StackedCube {
        cube,
        slices,
        skipped,
        ..
    } = stack(&cfg).context("stacking failed")?;
    let (_, ny, nx, _) = cube.dims();
    eprintln!(
        "stacked {} scenes ({} bands, {ny}x{nx} px), {} skipped",
        cube.dims().3,
        cube.dims().0,
        skipped.len()
    );
    let (transform, epsg) = cube_georef(&cube)?;

    let wants_trend = args.output.is_some() || args.pvalue_output.is_some();
    let wants_breaks = args.breaks_output.is_some() || args.first_break_output.is_some();

    let pipeline = build_pipeline(args, wants_trend, wants_breaks)?;

    // Report shape analytically: composite/gapfill/index never change the
    // spatial extent, and the derived band set/time axis are known without
    // running the pipeline (see `ChunkPipeline::output_time`) — so a report-
    // only invocation (no --output/--*-output/--zarr-output) never
    // materializes the cube.
    let out_bands: Vec<String> = match &pipeline.index {
        Some(index) => vec![index.label().to_string()],
        None => cube.bands().to_vec(),
    };
    let out_time = pipeline.output_time(&cube)?;

    if let Some(path) = &args.zarr_output {
        write_processed_zarr(
            &cube,
            &pipeline,
            args.chunk_size,
            &out_bands,
            &out_time,
            path,
        )
        .context("writing --zarr-output failed")?;
    }

    let mut maps_written = Vec::new();
    if wants_trend || wants_breaks {
        let mut slope = wants_trend.then(|| Array2::from_elem((ny, nx), f64::NAN));
        let mut pvalue = wants_trend.then(|| Array2::from_elem((ny, nx), f64::NAN));
        let mut count = wants_breaks.then(|| Array2::from_elem((ny, nx), f64::NAN));
        let mut first = wants_breaks.then(|| Array2::from_elem((ny, nx), f64::NAN));

        for result in cube
            .run_chunked(args.chunk_size, args.chunk_size, &pipeline)
            .context("chunked pipeline failed")?
        {
            let ChunkResult { y0, x0, stat } = result.context("chunked pipeline failed")?;
            if let Some((s, p)) = stat.trend {
                let (ch, cw) = s.dim();
                slope
                    .as_mut()
                    .unwrap()
                    .slice_mut(s![y0..y0 + ch, x0..x0 + cw])
                    .assign(&s);
                pvalue
                    .as_mut()
                    .unwrap()
                    .slice_mut(s![y0..y0 + ch, x0..x0 + cw])
                    .assign(&p);
            }
            if let Some((c, f)) = stat.breaks {
                let (ch, cw) = c.dim();
                count
                    .as_mut()
                    .unwrap()
                    .slice_mut(s![y0..y0 + ch, x0..x0 + cw])
                    .assign(&c);
                first
                    .as_mut()
                    .unwrap()
                    .slice_mut(s![y0..y0 + ch, x0..x0 + cw])
                    .assign(&f);
            }
        }

        if let Some(path) = &args.output {
            write_map(slope.as_ref().unwrap(), transform, epsg, path)?;
            maps_written.push(path.display().to_string());
        }
        if let Some(path) = &args.pvalue_output {
            write_map(pvalue.as_ref().unwrap(), transform, epsg, path)?;
            maps_written.push(path.display().to_string());
        }
        if let Some(path) = &args.breaks_output {
            write_map(count.as_ref().unwrap(), transform, epsg, path)?;
            maps_written.push(path.display().to_string());
        }
        if let Some(path) = &args.first_break_output {
            write_map(first.as_ref().unwrap(), transform, epsg, path)?;
            maps_written.push(path.display().to_string());
        }
    }

    let report = serde_json::json!({
        "scenes": slices.iter().map(|s| serde_json::json!({
            "id": s.item_id,
            "datetime": s.datetime,
            "time": s.time,
            "cloud_cover": s.cloud_cover,
        })).collect::<Vec<_>>(),
        "skipped": skipped,
        "dims": { "bands": out_bands.len(), "height": ny, "width": nx, "times": out_time.len() },
        "bands": out_bands,
        "time_range": [out_time.first(), out_time.last()],
        "epsg": epsg,
        "maps_written": maps_written,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Builds the chunked pipeline (composite/gapfill/index/stat) from CLI args.
/// The stat band is the index's fixed label when `--index` is given
/// (`cube.bands()[0]` after applying it), otherwise `--band` or the first
/// stacked asset — resolved against the cube as it exists right before the
/// stat step, matching `stack_cmd`'s historical band-selection behavior.
fn build_pipeline(
    args: &StackArgs,
    wants_trend: bool,
    wants_breaks: bool,
) -> Result<ChunkPipeline> {
    let composite = args.composite.map(|kind| {
        let window = match kind {
            CompositeKind::SameTime => CompositeWindow::SameTime,
            CompositeKind::Monthly => CompositeWindow::CalendarMonth,
            CompositeKind::Yearly => CompositeWindow::CalendarYear,
        };
        let method = match args.composite_method {
            CompositeAgg::Median => CompositeMethod::Median,
            CompositeAgg::Mean => CompositeMethod::Mean,
            CompositeAgg::Min => CompositeMethod::Min,
            CompositeAgg::Max => CompositeMethod::Max,
        };
        (window, method)
    });
    let gapfill = args.gapfill.map(|mg| GapfillSpec {
        max_gap: if mg > 0.0 { Some(mg) } else { None },
    });
    let index = args.index.map(|kind| index_spec(kind, args));
    let band = match &index {
        Some(spec) => spec.label().to_string(),
        None => args.band.clone().unwrap_or_else(|| args.assets[0].clone()),
    };
    let trend = wants_trend.then_some(match args.stat {
        TrendStat::TheilSen => TrendMethod::TheilSenMannKendall,
        TrendStat::Ols => TrendMethod::Ols,
    });
    let breaks = wants_breaks.then(|| BreakOptions {
        alpha: args.break_alpha,
        n_harmonics: args.break_harmonics,
        period: 1.0,
        min_segment: BreakOptions::default()
            .min_segment
            .max(2 * args.break_harmonics + 4),
    });
    Ok(ChunkPipeline {
        composite,
        gapfill,
        index,
        stat: StatSpec {
            band,
            trend,
            breaks,
        },
    })
}

/// Maps `--index` and the `--nir/--red/…` band-role flags to an [`IndexSpec`].
fn index_spec(kind: IndexKind, args: &StackArgs) -> IndexSpec {
    let (nir, red, green, blue, swir) = (
        args.nir.clone(),
        args.red.clone(),
        args.green.clone(),
        args.blue.clone(),
        args.swir.clone(),
    );
    match kind {
        IndexKind::Ndvi => IndexSpec::Ndvi { nir, red },
        IndexKind::Ndwi => IndexSpec::Ndwi { green, nir },
        IndexKind::Nbr => IndexSpec::Nbr { nir, swir },
        IndexKind::Ndbi => IndexSpec::Ndbi { swir, nir },
        IndexKind::Evi => IndexSpec::Evi { nir, red, blue },
        IndexKind::Savi => IndexSpec::Savi {
            nir,
            red,
            l: args.savi_l,
        },
    }
}

/// Writes a float map on the stack's grid as GeoTIFF (f32, NaN nodata).
/// Recovers the GeoTIFF-writable georeference from a cube's `GeoRef`. `stack`
/// always attaches one and every transform applied above preserves the
/// spatial grid, so this only fails if that invariant is somehow broken.
fn cube_georef(cube: &datacube_core::Cube) -> Result<(GeoTransform, Option<u32>)> {
    let geo = cube
        .georef()
        .context("stacked cube unexpectedly has no georeference")?;
    Ok((
        geo.transform
            .map(GeoTransform::from_gdal)
            .unwrap_or_default(),
        geo.epsg,
    ))
}

fn write_map(
    values: &ndarray::Array2<f64>,
    transform: GeoTransform,
    epsg: Option<u32>,
    path: &std::path::Path,
) -> Result<()> {
    let (ny, nx) = values.dim();
    let mut raster = Raster::<f32>::new(ny, nx);
    {
        let data = raster.data_mut();
        for ((r, c), v) in values.indexed_iter() {
            data[[r, c]] = *v as f32;
        }
    }
    raster.set_transform(transform);
    raster.set_crs(epsg.map(CRS::from_epsg));
    raster.set_nodata(Some(f32::NAN));
    write_geotiff(&raster, path, None)
        .map_err(|e| anyhow::anyhow!("writing {} failed: {e}", path.display()))
}

/// Persists the pipeline's processed cube (composite/gapfill/index, no stat)
/// to a GeoZarr store, one spatial tile at a time — the write-side
/// counterpart of `Cube::run_chunked`, since the stat step never
/// materializes a full processed cube by design (Section 4.8 of the paper).
fn write_processed_zarr(
    cube: &Cube,
    pipeline: &ChunkPipeline,
    chunk_size: usize,
    out_bands: &[String],
    out_time: &[f64],
    path: &std::path::Path,
) -> Result<()> {
    let (_, ny, nx, _) = cube.dims();
    let geo = cube.georef().unwrap_or(GeoRef {
        epsg: None,
        transform: None,
    });
    let writer = ZarrCubeWriter::create(
        path,
        (out_bands.len(), ny, nx, out_time.len()),
        out_bands,
        out_time,
        &geo,
        Default::default(),
    )
    .map_err(|e| anyhow::anyhow!("creating {} failed: {e}", path.display()))?;
    for chunk in cube
        .chunks(chunk_size, chunk_size)
        .context("chunking failed")?
    {
        let sub = Cube::new(
            chunk.data.to_owned(),
            cube.time().to_vec(),
            cube.bands().to_vec(),
        )?;
        let processed = pipeline.transform(&sub)?;
        writer
            .write_chunk(processed.data(), chunk.y0, chunk.x0)
            .map_err(|e| anyhow::anyhow!("writing tile ({},{}) failed: {e}", chunk.y0, chunk.x0))?;
    }
    Ok(())
}
