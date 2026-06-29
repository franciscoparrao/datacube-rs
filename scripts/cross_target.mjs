// Cross-target reproducibility runner (WebAssembly via Node).
//
// Loads the datacube-rs WASM module (the same one that powers the browser
// demo) and runs the five per-series estimators over the shared input. The
// WASM, native and Python targets share one Rust core, so results agree to
// machine precision. Prints results + per-estimator timing as JSON.
//
// Build:  wasm-pack build --target nodejs --out-dir /tmp/wasm-node  (from crates/datacube-wasm)
// Run:    node scripts/cross_target.mjs /tmp/wasm-node /tmp/xt_input.json

import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

const pkgDir = process.argv[2] || "/tmp/wasm-node";
const inputPath = process.argv[3] || "/tmp/xt_input.json";
const wasm = await import(`${pkgDir}/datacube_wasm.js`);

const inp = JSON.parse(readFileSync(inputPath, "utf8"));
const nan = (a) => Float64Array.from(a.map((v) => (v === null ? NaN : v)));
const t = nan(inp.t), y = nan(inp.y);

const lt = wasm.linear_trend(t, y);
const ts = wasm.theil_sen(t, y);
const mk = wasm.mann_kendall(y, 0.05);
const hr = wasm.harmonic_regression(t, y, 1.0, 2);
const br = wasm.detect_breaks(t, y, 0.05, 1, 1.0, 12);

const reps = 2000;
const us = (f) => {
  const s = performance.now();
  for (let i = 0; i < reps; i++) f();
  return (performance.now() - s) * 1e3 / reps; // µs/call
};

const out = {
  target: "wasm",
  results: {
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
    "detect_breaks.n_breaks": br.breaks.length,
  },
  timing: {
    linear_trend_us: us(() => wasm.linear_trend(t, y)),
    theil_sen_us: us(() => wasm.theil_sen(t, y)),
    mann_kendall_us: us(() => wasm.mann_kendall(y, 0.05)),
    harmonic_us: us(() => wasm.harmonic_regression(t, y, 1.0, 2)),
    detect_breaks_us: us(() => wasm.detect_breaks(t, y, 0.05, 1, 1.0, 12)),
  },
};
console.log(JSON.stringify(out, null, 2));
