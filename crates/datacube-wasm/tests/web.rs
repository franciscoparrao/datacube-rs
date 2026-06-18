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
