//! WebAssembly bindings for the per-series statistics of `datacube-core`.
//!
//! Exposes the trend / seasonality / break estimators to JavaScript over
//! `Float64Array`s, returning plain JS objects (via `serde-wasm-bindgen`).
//! Powers the browser time-series demo in `web/`.

use datacube_core::{ChunkPipeline, Cube, StatSpec, TrendMethod, stats};
use js_sys::Float64Array;
use ndarray::{Array2, Array3, Axis};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Maps a core error to a JS exception.
fn js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
struct LinearTrend {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    std_err: f64,
    p_value: f64,
    n: usize,
}

/// OLS linear trend of `y` over `t`.
#[wasm_bindgen]
pub fn linear_trend(t: &[f64], y: &[f64]) -> Result<JsValue, JsError> {
    let r = stats::linear_trend(t, y).map_err(js_err)?;
    to_js(&LinearTrend {
        slope: r.slope,
        intercept: r.intercept,
        r_squared: r.r_squared,
        std_err: r.std_err,
        p_value: r.p_value,
        n: r.n,
    })
}

#[derive(Serialize)]
struct TheilSen {
    slope: f64,
    intercept: f64,
    n: usize,
}

/// Theil-Sen robust slope of `y` over `t`.
#[wasm_bindgen]
pub fn theil_sen(t: &[f64], y: &[f64]) -> Result<JsValue, JsError> {
    let r = stats::theil_sen(t, y).map_err(js_err)?;
    to_js(&TheilSen {
        slope: r.slope,
        intercept: r.intercept,
        n: r.n,
    })
}

#[derive(Serialize)]
struct MannKendall {
    trend: String,
    s: f64,
    var_s: f64,
    z: f64,
    tau: f64,
    p_value: f64,
    n: usize,
}

/// Mann-Kendall trend test on `y` at significance `alpha`.
#[wasm_bindgen]
pub fn mann_kendall(y: &[f64], alpha: f64) -> Result<JsValue, JsError> {
    let r = stats::mann_kendall_alpha(y, alpha).map_err(js_err)?;
    let trend = match r.trend {
        stats::Trend::Increasing => "increasing",
        stats::Trend::Decreasing => "decreasing",
        stats::Trend::NoTrend => "no trend",
    };
    to_js(&MannKendall {
        trend: trend.to_string(),
        s: r.s,
        var_s: r.var_s,
        z: r.z,
        tau: r.tau,
        p_value: r.p_value,
        n: r.n,
    })
}

#[derive(Serialize)]
struct HarmonicComponent {
    harmonic: usize,
    cos_coef: f64,
    sin_coef: f64,
    amplitude: f64,
    phase: f64,
}

#[derive(Serialize)]
struct HarmonicFit {
    intercept: f64,
    slope: f64,
    period: f64,
    r_squared: f64,
    rmse: f64,
    n: usize,
    components: Vec<HarmonicComponent>,
}

/// Harmonic (Fourier) regression with trend.
#[wasm_bindgen]
pub fn harmonic_regression(
    t: &[f64],
    y: &[f64],
    period: f64,
    n_harmonics: usize,
) -> Result<JsValue, JsError> {
    let r = stats::harmonic_regression(t, y, period, n_harmonics).map_err(js_err)?;
    to_js(&HarmonicFit {
        intercept: r.intercept,
        slope: r.slope,
        period,
        r_squared: r.r_squared,
        rmse: r.rmse,
        n: r.n,
        components: r
            .components
            .iter()
            .map(|c| HarmonicComponent {
                harmonic: c.harmonic,
                cos_coef: c.cos_coef,
                sin_coef: c.sin_coef,
                amplitude: c.amplitude,
                phase: c.phase,
            })
            .collect(),
    })
}

#[derive(Serialize)]
struct BreakPoint {
    index: usize,
    time: f64,
    statistic: f64,
    p_value: f64,
}

#[derive(Serialize)]
struct BreakResult {
    statistic: f64,
    p_value: f64,
    n: usize,
    breaks: Vec<BreakPoint>,
}

/// Structural break detection (OLS-CUSUM + binary segmentation).
#[wasm_bindgen]
pub fn detect_breaks(
    t: &[f64],
    y: &[f64],
    alpha: f64,
    n_harmonics: usize,
    period: f64,
    min_segment: usize,
) -> Result<JsValue, JsError> {
    let opts = stats::BreakOptions {
        alpha,
        n_harmonics,
        period,
        min_segment,
    };
    let r = stats::detect_breaks(t, y, &opts).map_err(js_err)?;
    to_js(&BreakResult {
        statistic: r.statistic,
        p_value: r.p_value,
        n: r.n,
        breaks: r
            .breaks
            .iter()
            .map(|b| BreakPoint {
                index: b.index,
                time: b.time,
                statistic: b.statistic,
                p_value: b.p_value,
            })
            .collect(),
    })
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsError> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), value)
        .map_err(|_| JsError::new("cannot set result field"))?;
    Ok(())
}

/// Flattens a `(y, x)` grid row-major, masking pixels with fewer than
/// `min_valid` finite observations.
fn grid_to_js(grid: &Array2<f64>, valid: &[usize], min_valid: usize) -> Float64Array {
    let mut v: Vec<f64> = grid.iter().copied().collect();
    for (out, &n) in v.iter_mut().zip(valid) {
        if n < min_valid {
            *out = f64::NAN;
        }
    }
    Float64Array::from(&v[..])
}

/// Per-pixel trend (and optionally break) statistics over a whole single-band
/// cube in one JS↔WASM crossing, instead of one call per pixel.
///
/// Contract:
/// - `values` is the cube flattened **time-major, row-major within each time
///   slice**: `values[t * height * width + y * width + x]` — i.e. the
///   concatenation of the per-date grids, each in row order. Length must be
///   `n_times * height * width`.
/// - `times` is the shared time axis (decimal years), `n_times` long.
/// - `NaN` marks nodata (e.g. SCL cloud masking); each pixel is computed on
///   its remaining finite observations. Pixels with fewer than `min_valid`
///   finite observations get `NaN` in every output grid.
/// - `method` is `"theil_sen"` (Theil-Sen slope + Mann-Kendall p-value) or
///   `"ols"` (OLS slope + t-test p-value).
/// - `breaks_alpha` enables OLS-CUSUM break detection (trend-only segment
///   model) at that significance; `0` disables it and the break grids come
///   back `null`. `min_segment` is the minimum observations per segment.
///
/// Returns `{ slope, p_value, break_count, first_break, width, height }`,
/// each grid a `Float64Array` of `height * width` in the same row order as
/// the input slices. Pixels where a statistic could not be computed are
/// `NaN` (matching the per-series functions' error cases).
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn cube_stats(
    values: &[f64],
    n_times: usize,
    height: usize,
    width: usize,
    times: &[f64],
    method: &str,
    breaks_alpha: f64,
    min_segment: usize,
    min_valid: usize,
) -> Result<JsValue, JsError> {
    if values.len() != n_times * height * width {
        return Err(JsError::new(&format!(
            "values has {} elements but n_times * height * width = {}",
            values.len(),
            n_times * height * width
        )));
    }
    let trend = match method {
        "theil_sen" => TrendMethod::TheilSenMannKendall,
        "ols" => TrendMethod::Ols,
        other => return Err(JsError::new(&format!("unknown method '{other}'"))),
    };
    if !(0.0..1.0).contains(&breaks_alpha) {
        return Err(JsError::new("breaks_alpha must be in [0, 1)"));
    }

    // Finite observations per pixel, in the output grids' row order.
    let mut valid = vec![0usize; height * width];
    for slice in values.chunks_exact(height * width) {
        for (n, v) in valid.iter_mut().zip(slice) {
            if v.is_finite() {
                *n += 1;
            }
        }
    }

    // (time, y, x) as delivered -> (band, y, x, time) as the core expects;
    // Cube::new copies the permuted view back into standard layout.
    let data = Array3::from_shape_vec((n_times, height, width), values.to_vec())
        .map_err(js_err)?
        .permuted_axes([1, 2, 0])
        .insert_axis(Axis(0));
    let cube = Cube::new(data, times.to_vec(), vec!["b".to_string()]).map_err(js_err)?;

    let pipeline = ChunkPipeline {
        composite: None,
        gapfill: None,
        index: None,
        stat: StatSpec {
            band: "b".to_string(),
            trend: Some(trend),
            breaks: (breaks_alpha > 0.0).then_some(stats::BreakOptions {
                alpha: breaks_alpha,
                n_harmonics: 0,
                period: 1.0,
                min_segment,
            }),
        },
    };
    let stat = pipeline.run_on(&cube).map_err(js_err)?;

    let out = js_sys::Object::new();
    let (slope, p_value) = stat.trend.expect("trend was requested");
    set(&out, "slope", &grid_to_js(&slope, &valid, min_valid).into())?;
    set(
        &out,
        "p_value",
        &grid_to_js(&p_value, &valid, min_valid).into(),
    )?;
    match stat.breaks {
        Some((count, first)) => {
            set(
                &out,
                "break_count",
                &grid_to_js(&count, &valid, min_valid).into(),
            )?;
            set(
                &out,
                "first_break",
                &grid_to_js(&first, &valid, min_valid).into(),
            )?;
        }
        None => {
            set(&out, "break_count", &JsValue::NULL)?;
            set(&out, "first_break", &JsValue::NULL)?;
        }
    }
    set(&out, "width", &JsValue::from_f64(width as f64))?;
    set(&out, "height", &JsValue::from_f64(height as f64))?;
    Ok(out.into())
}
