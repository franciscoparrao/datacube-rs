//! WebAssembly bindings for the per-series statistics of `datacube-core`.
//!
//! Exposes the trend / seasonality / break estimators to JavaScript over
//! `Float64Array`s, returning plain JS objects (via `serde-wasm-bindgen`).
//! Powers the browser time-series demo in `web/`.

use datacube_core::stats;
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

/// BFAST-style structural break detection.
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
