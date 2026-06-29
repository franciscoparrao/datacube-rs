//! Cross-target reproducibility runner (native).
//!
//! Reads the shared input series, runs the five per-series estimators, and
//! prints their key results plus per-estimator timing as JSON. The Python and
//! WebAssembly runners (scripts/cross_target.py, scripts/cross_target.mjs) do
//! the same over the identical input; all three share the same Rust core
//! (including the `libm` crate for the special functions), so the results are
//! bit-identical across native, Python and the browser.
//!
//! Run: cargo run -q -p datacube-cli --example cross_target -- /tmp/xt_input.json

use std::time::Instant;

use datacube_core::stats;
use serde_json::{Value, json};

fn main() {
    let path = std::env::args().nth(1).expect("usage: cross_target <input.json>");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let t: Vec<f64> = v["t"].as_array().unwrap().iter().map(num).collect();
    let y: Vec<f64> = v["y"].as_array().unwrap().iter().map(num).collect();

    let lt = stats::linear_trend(&t, &y).unwrap();
    let ts = stats::theil_sen(&t, &y).unwrap();
    let mk = stats::mann_kendall(&y).unwrap();
    let hr = stats::harmonic_regression(&t, &y, 1.0, 2).unwrap();
    let br = stats::detect_breaks(&t, &y, &stats::BreakOptions { n_harmonics: 1, ..Default::default() })
        .unwrap();

    // timing: median over repeats
    let reps = 2000;
    let time = |f: &dyn Fn()| {
        let s = Instant::now();
        for _ in 0..reps {
            f();
        }
        s.elapsed().as_secs_f64() * 1e6 / reps as f64 // µs per call
    };
    let timing = json!({
        "linear_trend_us": time(&|| { stats::linear_trend(&t, &y).ok(); }),
        "theil_sen_us":    time(&|| { stats::theil_sen(&t, &y).ok(); }),
        "mann_kendall_us": time(&|| { stats::mann_kendall(&y).ok(); }),
        "harmonic_us":     time(&|| { stats::harmonic_regression(&t, &y, 1.0, 2).ok(); }),
        "detect_breaks_us":time(&|| { stats::detect_breaks(&t, &y, &stats::BreakOptions { n_harmonics: 1, ..Default::default() }).ok(); }),
    });

    let out = json!({
        "target": "native",
        "results": {
            "linear_trend.slope": lt.slope,
            "linear_trend.p_value": lt.p_value,
            "theil_sen.slope": ts.slope,
            "mann_kendall.tau": mk.tau,
            "mann_kendall.z": mk.z,
            "mann_kendall.p_value": mk.p_value,
            "harmonic.slope": hr.slope,
            "harmonic.amplitude_1": hr.components[0].amplitude,
            "harmonic.rmse": hr.rmse,
            "detect_breaks.statistic": br.statistic,
            "detect_breaks.p_value": br.p_value,
            "detect_breaks.n_breaks": br.breaks.len(),
        },
        "timing": timing,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}
