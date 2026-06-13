use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use datacube_core::stats;

#[cfg(feature = "stac")]
mod stack_cmd;

#[derive(Parser)]
#[command(
    name = "datacube",
    version,
    about = "Temporal data cube analysis (datacube-rs)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Trend analysis of a time series CSV: OLS, Theil-Sen and Mann-Kendall.
    ///
    /// The CSV may have one column (values; t = 0,1,2,...) or two columns
    /// (t,value). A non-numeric first line is treated as a header. Empty,
    /// "NA" and "NaN" values are missing data.
    Trend {
        /// Input CSV file
        input: PathBuf,
        /// Significance level for the Mann-Kendall test
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
    },
    /// Harmonic (Fourier) regression with linear trend, for seasonality
    /// and phenology. Same CSV format as `trend`.
    Harmonic {
        /// Input CSV file
        input: PathBuf,
        /// Season length in the units of t (e.g. 1 for fractional years,
        /// 365.25 for days, 12 for monthly indices)
        #[arg(long)]
        period: f64,
        /// Number of Fourier pairs
        #[arg(long, default_value_t = 2)]
        harmonics: usize,
    },
    /// BFAST-style structural break detection (OLS-CUSUM + binary
    /// segmentation). Same CSV format as `trend`.
    Breaks {
        /// Input CSV file
        input: PathBuf,
        /// Significance level for the CUSUM test
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
        /// Fourier pairs in the segment model (0 = trend only)
        #[arg(long, default_value_t = 0)]
        harmonics: usize,
        /// Season length in the units of t (used when --harmonics > 0)
        #[arg(long, default_value_t = 1.0)]
        period: f64,
        /// Minimum observations per segment
        #[arg(long, default_value_t = 12)]
        min_segment: usize,
    },
    /// Stack STAC/COG scenes into a temporal cube and compute trend maps
    /// (requires building with `--features stac`).
    #[cfg(feature = "stac")]
    Stack(Box<stack_cmd::StackArgs>),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Trend { input, alpha } => trend(&input, alpha),
        Command::Harmonic {
            input,
            period,
            harmonics,
        } => harmonic(&input, period, harmonics),
        Command::Breaks {
            input,
            alpha,
            harmonics,
            period,
            min_segment,
        } => breaks(&input, alpha, harmonics, period, min_segment),
        #[cfg(feature = "stac")]
        Command::Stack(args) => stack_cmd::run(&args),
    }
}

fn breaks(
    input: &PathBuf,
    alpha: f64,
    harmonics: usize,
    period: f64,
    min_segment: usize,
) -> Result<()> {
    let raw =
        fs::read_to_string(input).with_context(|| format!("cannot read {}", input.display()))?;
    let (t, y) = parse_series(&raw)?;
    let opts = stats::BreakOptions {
        alpha,
        n_harmonics: harmonics,
        period,
        min_segment,
    };
    let result = stats::detect_breaks(&t, &y, &opts).context("break detection failed")?;

    let report = serde_json::json!({
        "input": input.display().to_string(),
        "n": result.n,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "alpha": alpha,
        "breaks": result.breaks.iter().map(|b| serde_json::json!({
            "index": b.index,
            "time": b.time,
            "statistic": b.statistic,
            "p_value": b.p_value,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn harmonic(input: &PathBuf, period: f64, harmonics: usize) -> Result<()> {
    let raw =
        fs::read_to_string(input).with_context(|| format!("cannot read {}", input.display()))?;
    let (t, y) = parse_series(&raw)?;
    let fit = stats::harmonic_regression(&t, &y, period, harmonics)
        .context("harmonic regression failed")?;

    let report = serde_json::json!({
        "input": input.display().to_string(),
        "n": fit.n,
        "period": fit.period,
        "intercept": fit.intercept,
        "slope": fit.slope,
        "r_squared": fit.r_squared,
        "rmse": fit.rmse,
        "components": fit.components.iter().map(|c| serde_json::json!({
            "harmonic": c.harmonic,
            "cos_coef": c.cos_coef,
            "sin_coef": c.sin_coef,
            "amplitude": c.amplitude,
            "phase": c.phase,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn trend(input: &PathBuf, alpha: f64) -> Result<()> {
    let raw =
        fs::read_to_string(input).with_context(|| format!("cannot read {}", input.display()))?;
    let (t, y) = parse_series(&raw)?;

    let ols = stats::linear_trend(&t, &y).context("OLS fit failed")?;
    let sen = stats::theil_sen(&t, &y).context("Theil-Sen fit failed")?;
    let mk = stats::mann_kendall_alpha(&y, alpha).context("Mann-Kendall test failed")?;

    let report = serde_json::json!({
        "input": input.display().to_string(),
        "n": ols.n,
        "ols": {
            "slope": ols.slope,
            "intercept": ols.intercept,
            "r_squared": ols.r_squared,
            "std_err": ols.std_err,
            "p_value": ols.p_value,
        },
        "theil_sen": {
            "slope": sen.slope,
            "intercept": sen.intercept,
        },
        "mann_kendall": {
            "trend": match mk.trend {
                stats::Trend::Increasing => "increasing",
                stats::Trend::Decreasing => "decreasing",
                stats::Trend::NoTrend => "no trend",
            },
            "alpha": alpha,
            "s": mk.s,
            "var_s": mk.var_s,
            "z": mk.z,
            "tau": mk.tau,
            "p_value": mk.p_value,
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_series(raw: &str) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut t = Vec::new();
    let mut y = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let parsed: Vec<f64> = fields.iter().map(|f| parse_value(f)).collect();
        if lineno == 0 && parsed.iter().all(|v| v.is_nan()) && fields.iter().all(|f| !f.is_empty())
        {
            continue; // header line
        }
        match parsed.as_slice() {
            [v] => {
                t.push(y.len() as f64);
                y.push(*v);
            }
            [ti, v] => {
                if ti.is_nan() {
                    bail!(
                        "line {}: time coordinate '{}' is not numeric",
                        lineno + 1,
                        fields[0]
                    );
                }
                t.push(*ti);
                y.push(*v);
            }
            _ => bail!(
                "line {}: expected 1 or 2 columns, got {}",
                lineno + 1,
                fields.len()
            ),
        }
    }
    if y.is_empty() {
        bail!("no data rows found");
    }
    Ok((t, y))
}

fn parse_value(field: &str) -> f64 {
    if field.is_empty() || field.eq_ignore_ascii_case("na") || field.eq_ignore_ascii_case("nan") {
        return f64::NAN;
    }
    field.parse().unwrap_or(f64::NAN)
}
