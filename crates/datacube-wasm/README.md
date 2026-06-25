# datacube-rs (WebAssembly)

WebAssembly bindings for the per-series statistics of
[datacube-rs](https://github.com/franciscoparrao/datacube-rs) — run the trend,
seasonality and structural-break estimators in the browser, with the same Rust
core (and the same 1e-9 numerical parity) as the native build.

The `web/` demo fits a harmonic+trend model and detects OLS-CUSUM breaks on a
synthetic NDVI series, live as you move the sliders — no server-side compute.

## Build & run the demo

```bash
# from crates/datacube-wasm/
wasm-pack build --target web --out-dir web/pkg
python3 -m http.server -d web 8731        # then open http://localhost:8731
```

## API

The module (`datacube_wasm.js`) exports, over `Float64Array`s, returning plain
JS objects:

```js
import init, { linear_trend, theil_sen, mann_kendall,
               harmonic_regression, detect_breaks } from "./pkg/datacube_wasm.js";
await init();

linear_trend(t, y);                          // {slope, intercept, r_squared, ...}
theil_sen(t, y);                             // {slope, intercept, n}
mann_kendall(y, 0.05);                       // {trend, tau, p_value, ...}
harmonic_regression(t, y, 1.0, 2);           // {intercept, slope, components:[...], ...}
detect_breaks(t, y, 0.05, 1, 1.0, 12);       // {statistic, p_value, breaks:[...]}
```

`NaN` marks missing observations (dropped pairwise), exactly as in the native
and Python builds.

## Tests

```bash
wasm-pack test --node     # runs tests/web.rs under Node
```
