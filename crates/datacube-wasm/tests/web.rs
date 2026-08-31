//! wasm-bindgen tests; run with `wasm-pack test --node` from the crate.
//! Compiled only for wasm32 so `cargo test --workspace` (host) skips them.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn series(n: usize) -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = t.iter().map(|&x| 2.0 * x + 1.0).collect();
    (t, y)
}

fn field(obj: &JsValue, key: &str) -> f64 {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .unwrap()
        .as_f64()
        .unwrap()
}

#[wasm_bindgen_test]
fn linear_trend_exact() {
    let (t, y) = series(10);
    let r = datacube_wasm::linear_trend(&t, &y).unwrap();
    assert!((field(&r, "slope") - 2.0).abs() < 1e-12);
    assert!((field(&r, "intercept") - 1.0).abs() < 1e-12);
    assert_eq!(field(&r, "n") as usize, 10);
}

#[wasm_bindgen_test]
fn mann_kendall_increasing() {
    let (_, y) = series(12);
    let r = datacube_wasm::mann_kendall(&y, 0.05).unwrap();
    let trend = js_sys::Reflect::get(&r, &JsValue::from_str("trend"))
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(trend, "increasing");
    assert!((field(&r, "tau") - 1.0).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn detect_breaks_level_shift() {
    let t: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let y: Vec<f64> = t
        .iter()
        .map(|&x| if x < 30.0 { 1.0 } else { 6.0 } + 0.05 * (x * 12.9898).sin())
        .collect();
    let r = datacube_wasm::detect_breaks(&t, &y, 0.05, 0, 1.0, 12).unwrap();
    let breaks = js_sys::Reflect::get(&r, &JsValue::from_str("breaks")).unwrap();
    let len = js_sys::Array::from(&breaks).length();
    assert_eq!(len, 1);
}

// --- cube_stats -----------------------------------------------------------

/// Flat time-major cube: values[t*h*w + y*w + x] = series(y, x) at t.
fn flat_cube<F: Fn(usize, usize, usize) -> f64>(nt: usize, h: usize, w: usize, f: F) -> Vec<f64> {
    let mut v = Vec::with_capacity(nt * h * w);
    for t in 0..nt {
        for y in 0..h {
            for x in 0..w {
                v.push(f(t, y, x));
            }
        }
    }
    v
}

fn grid(obj: &JsValue, key: &str) -> Vec<f64> {
    let arr = js_sys::Reflect::get(obj, &JsValue::from_str(key)).unwrap();
    js_sys::Float64Array::from(arr).to_vec()
}

#[wasm_bindgen_test]
fn cube_stats_matches_per_series_path() {
    let (nt, h, w) = (12, 3, 4);
    let times: Vec<f64> = (0..nt).map(|i| 2022.0 + i as f64 / 12.0).collect();
    // Distinct slope and offset per pixel, plus a deterministic wiggle.
    let series_at = |t: usize, y: usize, x: usize| {
        let slope = (y * w + x) as f64 - 5.0;
        slope * (times[t] - 2022.0) + 0.1 * ((t * 7 + y + x) as f64).sin()
    };
    let values = flat_cube(nt, h, w, series_at);
    let r = datacube_wasm::cube_stats(&values, nt, h, w, &times, "theil_sen", 0.0, 12, 5).unwrap();
    let slope = grid(&r, "slope");
    let p_value = grid(&r, "p_value");
    assert_eq!(slope.len(), h * w);
    for y in 0..h {
        for x in 0..w {
            let series: Vec<f64> = (0..nt).map(|t| series_at(t, y, x)).collect();
            let ts = datacube_wasm::theil_sen(&times, &series).unwrap();
            let mk = datacube_wasm::mann_kendall(&series, 0.05).unwrap();
            assert_eq!(slope[y * w + x], field(&ts, "slope"), "slope at ({y},{x})");
            assert_eq!(
                p_value[y * w + x],
                field(&mk, "p_value"),
                "p_value at ({y},{x})"
            );
        }
    }
    // breaks_alpha = 0 disables break detection.
    let bc = js_sys::Reflect::get(&r, &JsValue::from_str("break_count")).unwrap();
    assert!(bc.is_null());
}

#[wasm_bindgen_test]
fn cube_stats_nan_masked_pixel_matches_per_series() {
    let (nt, h, w) = (12, 2, 2);
    let times: Vec<f64> = (0..nt).map(|i| 2022.0 + i as f64 / 12.0).collect();
    // Pixel (0, 1) is cloud-masked in two dates (10 valid observations, the
    // Maipo-basin situation); pixel (1, 1) keeps only 3 (below min_valid=5).
    let series_at = |t: usize, y: usize, x: usize| {
        if y == 0 && x == 1 && (t == 3 || t == 8) {
            return f64::NAN;
        }
        if y == 1 && x == 1 && t >= 3 {
            return f64::NAN;
        }
        (1 + y + x) as f64 * (times[t] - 2022.0) + 0.05 * (t as f64).cos()
    };
    let values = flat_cube(nt, h, w, series_at);
    let r = datacube_wasm::cube_stats(&values, nt, h, w, &times, "ols", 0.0, 12, 5).unwrap();
    let slope = grid(&r, "slope");
    let p_value = grid(&r, "p_value");

    let series: Vec<f64> = (0..nt).map(|t| series_at(t, 0, 1)).collect();
    let lt = datacube_wasm::linear_trend(&times, &series).unwrap();
    assert_eq!(field(&lt, "n") as usize, 10);
    assert_eq!(slope[1], field(&lt, "slope"));
    assert_eq!(p_value[1], field(&lt, "p_value"));

    // Below min_valid -> NaN, even though OLS on 3 points would return numbers.
    assert!(slope[w + 1].is_nan());
    assert!(p_value[w + 1].is_nan());
    // Fully valid pixel unaffected.
    assert!(slope[0].is_finite());
}

#[wasm_bindgen_test]
fn cube_stats_breaks_level_shift() {
    let (nt, h, w) = (60, 2, 2);
    let times: Vec<f64> = (0..nt).map(|i| i as f64).collect();
    // Pixel (0, 0) carries the level shift from `detect_breaks_level_shift`;
    // the rest stay flat.
    let series_at = |t: usize, y: usize, x: usize| {
        let wiggle = 0.05 * (t as f64 * 12.9898).sin();
        if y == 0 && x == 0 && t >= 30 {
            6.0 + wiggle
        } else {
            1.0 + wiggle
        }
    };
    let values = flat_cube(nt, h, w, series_at);
    let r = datacube_wasm::cube_stats(&values, nt, h, w, &times, "theil_sen", 0.05, 12, 5).unwrap();
    let count = grid(&r, "break_count");
    let first = grid(&r, "first_break");
    assert_eq!(count[0], 1.0);
    // detect_breaks anchors a break on the last index of the segment before
    // the shift (same convention as the per-series binding).
    assert_eq!(first[0], 29.0);
    assert_eq!(count[1], 0.0);
    assert!(first[1].is_nan());
}

#[wasm_bindgen_test]
fn cube_stats_rejects_bad_input() {
    let times = [0.0, 1.0, 2.0];
    assert!(
        datacube_wasm::cube_stats(&[0.0; 11], 3, 2, 2, &times, "theil_sen", 0.0, 12, 5).is_err()
    );
    assert!(datacube_wasm::cube_stats(&[0.0; 12], 3, 2, 2, &times, "median", 0.0, 12, 5).is_err());
    assert!(datacube_wasm::cube_stats(&[0.0; 12], 3, 2, 2, &times, "ols", 1.5, 12, 5).is_err());
}
