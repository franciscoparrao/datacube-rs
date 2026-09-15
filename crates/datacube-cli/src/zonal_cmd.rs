//! `datacube zonal`: polygon zonal statistics over a temporal cube.
//!
//! Sources the cube either from a GeoZarr store (`--cube`) or by stacking from
//! STAC (the shared ingestion flags), optionally computes a spectral index /
//! gap-fill, then reduces each polygon of a vector layer to a tidy scalar
//! series `(polygon_id, time, band, reducer, value, n_valid, n_total)`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use datacube_core::Cube;
use datacube_io::{
    PixelInclusion, Reducer, StackedCube, ZonalConfig, read_zones, stack, zonal_reduce,
};

use crate::stack_cmd::{StackSourceArgs, build_stack_config};

#[derive(clap::Args)]
pub struct ZonalArgs {
    #[command(flatten)]
    source: StackSourceArgs,
    /// Read the cube from a GeoZarr store instead of stacking from STAC.
    /// Mutually exclusive with the STAC ingestion flags.
    #[arg(long)]
    cube: Option<PathBuf>,
    /// Vector file of polygons (.shp or .geojson).
    #[arg(long)]
    vector: PathBuf,
    /// Attribute field identifying each polygon (its value becomes polygon_id).
    #[arg(long)]
    id_field: String,
    /// Pixel-inclusion rule: how a polygon selects the pixels it covers.
    #[arg(long, value_enum, default_value_t = InclusionArg::Center)]
    inclusion: InclusionArg,
    /// Spatial reducer applied to each polygon's covered pixels.
    #[arg(long, value_enum, default_value_t = ReducerArg::Mean)]
    reduce: ReducerArg,
    /// Threshold for `--reduce fraction-above`.
    #[arg(long, allow_hyphen_values = true)]
    threshold: Option<f64>,
    /// EPSG of the vector's CRS, overriding whatever the file declares.
    #[arg(long)]
    vector_epsg: Option<u32>,
    /// Output format for the tidy table.
    #[arg(long, value_enum, default_value_t = FormatArg::Csv)]
    format: FormatArg,
    /// Output file (stdout if omitted).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum InclusionArg {
    /// Pixel included iff its centre is inside the polygon (rasterio default).
    Center,
    /// Pixel included iff the polygon touches any part of its cell.
    AllTouched,
    /// Pixel weighted by the fraction of its area the polygon covers.
    AreaFraction,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ReducerArg {
    Mean,
    Median,
    Min,
    Max,
    Std,
    Sum,
    Count,
    /// Fraction of finite values strictly above `--threshold`.
    FractionAbove,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FormatArg {
    Csv,
    Json,
    /// Parquet (requires building with `--features parquet`); needs `--out`.
    Parquet,
}

impl From<InclusionArg> for PixelInclusion {
    fn from(a: InclusionArg) -> Self {
        match a {
            InclusionArg::Center => PixelInclusion::Center,
            InclusionArg::AllTouched => PixelInclusion::AllTouched,
            InclusionArg::AreaFraction => PixelInclusion::AreaFraction,
        }
    }
}

pub fn run(args: &ZonalArgs) -> Result<()> {
    // 1. Get the cube: from a GeoZarr store, or by stacking from STAC.
    let cube: Cube = if let Some(path) = &args.cube {
        let (cube, _geo) = datacube_zarr::read_zarr(path)
            .map_err(|e| anyhow::anyhow!("reading {} failed: {e}", path.display()))?;
        cube
    } else {
        let cfg = build_stack_config(&args.source)?;
        eprintln!("searching {} ...", args.source.collection());
        let StackedCube { cube, skipped, .. } = stack(&cfg).context("stacking failed")?;
        eprintln!(
            "stacked {} scenes, {} skipped",
            cube.dims().3,
            skipped.len()
        );
        cube
    };

    // 2. Optional gap-fill + spectral index (temporal pooling is the zonal
    //    reducer's job, so composite is deliberately not applied here).
    let pipeline = args.source.zonal_transform();
    let cube = pipeline
        .transform(&cube)
        .context("preparing cube (gapfill/index) failed")?;

    // 3. Resolve the reducer.
    let reducer = match args.reduce {
        ReducerArg::Mean => Reducer::Mean,
        ReducerArg::Median => Reducer::Median,
        ReducerArg::Min => Reducer::Min,
        ReducerArg::Max => Reducer::Max,
        ReducerArg::Std => Reducer::Std,
        ReducerArg::Sum => Reducer::Sum,
        ReducerArg::Count => Reducer::Count,
        ReducerArg::FractionAbove => {
            let t = args
                .threshold
                .ok_or_else(|| anyhow::anyhow!("--reduce fraction-above needs --threshold"))?;
            Reducer::FractionAbove(t)
        }
    };

    let cfg = ZonalConfig::new(&args.id_field, args.inclusion.into(), reducer)
        .window(args.source.zonal_window())
        .source_epsg(args.vector_epsg);

    // 4. Read the vector and reduce.
    let features = read_zones(&args.vector)
        .with_context(|| format!("reading vector {} failed", args.vector.display()))?;
    let table = zonal_reduce(&cube, &features, &cfg).context("zonal reduction failed")?;

    // 5. Emit.
    match args.format {
        FormatArg::Csv => match &args.out {
            Some(path) => table.write_csv(path)?,
            None => print!("{}", table.to_csv_string()),
        },
        FormatArg::Json => match &args.out {
            Some(path) => table.write_json(path)?,
            None => println!("{}", table.to_json_string()?),
        },
        FormatArg::Parquet => {
            let path = args
                .out
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--format parquet requires --out <file>"))?;
            #[cfg(feature = "parquet")]
            table.write_parquet(path)?;
            #[cfg(not(feature = "parquet"))]
            {
                let _ = path;
                anyhow::bail!("parquet output requires building the CLI with --features parquet");
            }
        }
    }
    if let Some(path) = &args.out {
        eprintln!("wrote {} rows to {}", table.rows.len(), path.display());
    }
    Ok(())
}
