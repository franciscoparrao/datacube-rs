#!/usr/bin/env python3
"""Cross-target reproducibility runner (Python / PyO3).

Reads the shared input and runs the five per-series estimators through the
datacube_rs Python extension. Companion to the native example and the Node
WebAssembly runner; all three share one Rust core, so the results are
bit-identical. Prints results + per-estimator timing as JSON.

Run: .venv-validate/bin/python scripts/cross_target.py /tmp/xt_input.json
"""
import json
import sys
import time

import numpy as np
import datacube_rs as dc

inp = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/xt_input.json"))
t = np.array([np.nan if v is None else v for v in inp["t"]], dtype=np.float64)
y = np.array([np.nan if v is None else v for v in inp["y"]], dtype=np.float64)

lt = dc.linear_trend(t, y)
ts = dc.theil_sen(t, y)
mk = dc.mann_kendall(y, 0.05)
hr = dc.harmonic_regression(t, y, 1.0, 2)
br = dc.detect_breaks(t, y, 0.05, 1, 1.0, 12)

reps = 2000
def us(f):
    s = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - s) * 1e6 / reps

out = {
    "target": "python",
    "results": {
        "linear_trend.slope": lt["slope"],
        "linear_trend.p_value": lt["p_value"],
        "theil_sen.slope": ts["slope"],
        "mann_kendall.tau": mk["tau"],
        "mann_kendall.z": mk["z"],
        "mann_kendall.p_value": mk["p_value"],
        "harmonic.slope": hr["slope"],
        "harmonic.amplitude_1": hr["components"][0]["amplitude"],
        "harmonic.rmse": hr["rmse"],
        "detect_breaks.statistic": br["statistic"],
        "detect_breaks.p_value": br["p_value"],
        "detect_breaks.n_breaks": len(br["breaks"]),
    },
    "timing": {
        "linear_trend_us": us(lambda: dc.linear_trend(t, y)),
        "theil_sen_us": us(lambda: dc.theil_sen(t, y)),
        "mann_kendall_us": us(lambda: dc.mann_kendall(y, 0.05)),
        "harmonic_us": us(lambda: dc.harmonic_regression(t, y, 1.0, 2)),
        "detect_breaks_us": us(lambda: dc.detect_breaks(t, y, 0.05, 1, 1.0, 12)),
    },
}
print(json.dumps(out, indent=2, sort_keys=True))
