#!/usr/bin/env python3
"""Numerical parity of the autocorrelation/seasonal Mann-Kendall variants.

Cross-checks datacube_rs.seasonal_mann_kendall and .mann_kendall_hamed_rao
against pymannkendall.seasonal_test and .hamed_rao_modification_test on
deterministic series (a seasonal NDVI-like signal and an AR(1) process), which
are the corrections the C&G review demanded for satellite time series.

Requires the validation venv with the (default) binding + pymannkendall:
    VIRTUAL_ENV=.venv-validate maturin develop --release \\
        -m crates/datacube-python/Cargo.toml
    .venv-validate/bin/python scripts/validate_mk_corrected.py
"""

import sys

import numpy as np
import pymannkendall as mk

import datacube_rs as dc

TOL = 1e-9


def series():
    rng = np.random.default_rng(7)
    cases = {}
    # seasonal + trend (monthly, 6 years) — the NDVI-like regime
    t = np.arange(72, dtype=float)
    cases["seasonal_trend"] = (
        0.01 * t + 0.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, 72)
    )
    # pure seasonal, no interannual trend
    cases["seasonal_flat"] = 0.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.05, 72)
    # AR(1) phi=0.8 (strong serial correlation)
    ar = np.empty(120)
    ar[0] = rng.normal()
    for i in range(1, 120):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.5)
    cases["ar1"] = ar
    # AR(1) + trend
    cases["ar1_trend"] = ar + 0.02 * np.arange(120)
    # near-white noise
    cases["white"] = rng.normal(0, 1.0, 100)
    return cases


def cmp(name, got, ref_trend, ref_s, ref_var, ref_z, ref_p, ref_tau):
    fails = 0
    checks = [
        ("s", got["s"], ref_s),
        ("var_s", got["var_s"], ref_var),
        ("z", got["z"], ref_z),
        ("p_value", got["p_value"], ref_p),
        ("tau", got["tau"], ref_tau),
    ]
    for field, a, b in checks:
        rel = abs(a - b) / max(1.0, abs(b))
        if rel > TOL:
            fails += 1
            print(f"FAIL {name}.{field}: rust={a!r} ref={b!r} rel={rel:.2e}")
    tmap = {"increasing": "increasing", "decreasing": "decreasing", "no trend": "no trend"}
    if tmap[got["trend"]] != ref_trend:
        fails += 1
        print(f"FAIL {name}.trend: rust={got['trend']} ref={ref_trend}")
    return len(checks) + 1, fails


def main():
    if not hasattr(dc, "seasonal_mann_kendall"):
        print("datacube_rs lacks seasonal_mann_kendall; rebuild the binding.")
        return 2
    total = fails = 0
    for name, x in series().items():
        # --- seasonal test (period=12) ---
        r = mk.seasonal_test(x, period=12)
        got = dc.seasonal_mann_kendall(x, period=12)
        c, f = cmp(f"seasonal[{name}]", got, r.trend, r.s, r.var_s, r.z, r.p, r.Tau)
        total += c
        fails += f
        # --- hamed-rao (all lags, and first-3-lags) ---
        for lag_ref, lag_arg in [(None, None), (3, 3)]:
            r = mk.hamed_rao_modification_test(x, lag=lag_ref)
            got = dc.mann_kendall_hamed_rao(x, lag=lag_arg)
            c, f = cmp(
                f"hamed_rao[{name},lag={lag_arg}]", got, r.trend, r.s, r.var_s, r.z, r.p, r.Tau
            )
            total += c
            fails += f
        print(f"ok   {name}: n={len(x)}")

    print(f"\n{total - fails}/{total} checks passed (tol={TOL})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
