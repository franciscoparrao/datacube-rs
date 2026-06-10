#!/usr/bin/env python3
"""Cross-validate datacube-rs statistics against pyMannKendall and scipy.

Generates deterministic test series, runs the reference implementations
(pymannkendall.original_test, pymannkendall.sens_slope,
scipy.stats.linregress), runs `datacube trend` on the same series, and
compares every reported field within tolerance.

Usage: python3 scripts/validate_pymannkendall.py [--tol 1e-9]
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymannkendall as mk
import scipy.stats as st

REPO = Path(__file__).resolve().parent.parent

CASES = {
    "strict_increase": np.arange(10, dtype=float),
    "strict_decrease": np.arange(20, dtype=float)[::-1] * 0.5 + 3.0,
    "with_ties": np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0]),
    "constant": np.full(15, 2.5),
    "noisy_trend": None,  # filled below (deterministic RNG)
    "seasonal_plus_trend": None,
    "with_nan": None,
}

rng = np.random.default_rng(42)
t = np.arange(60, dtype=float)
CASES["noisy_trend"] = 0.02 * t + rng.normal(0, 0.3, 60)
CASES["seasonal_plus_trend"] = (
    0.01 * t + 0.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, 60)
)
wn = 0.05 * t + rng.normal(0, 0.2, 60)
wn[[3, 17, 41, 42]] = np.nan
CASES["with_nan"] = wn


def run_datacube(values: np.ndarray, *cmd: str, t=None) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        for i, v in enumerate(values):
            sv = "NaN" if np.isnan(v) else repr(v)
            f.write(f"{t[i]!r},{sv}\n" if t is not None else f"{sv}\n")
        path = f.name
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "datacube-cli", "--", *(cmd or ["trend"]), path],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def validate_harmonic(tol: float) -> tuple[int, int]:
    """Cross-check `datacube harmonic` against numpy.linalg.lstsq on the
    same design matrix (intercept + trend + K Fourier pairs)."""
    rng = np.random.default_rng(7)
    t = np.arange(72, dtype=float) / 12.0  # 6 years, monthly, fractional years
    y = (0.5 + 0.02 * t
         + 0.25 * np.cos(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * t)
         - 0.05 * np.cos(4 * np.pi * t)
         + rng.normal(0, 0.03, t.size))
    y[[7, 30]] = np.nan  # cloud gaps
    n_harmonics = 2

    got = run_datacube(y, "harmonic", "--period", "1.0",
                       "--harmonics", str(n_harmonics), t=t)

    mask = ~np.isnan(y)
    tc, yc = t[mask], y[mask]
    cols = [np.ones_like(tc), tc]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2 * np.pi * k * tc), np.sin(2 * np.pi * k * tc)]
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, yc, rcond=None)
    resid = yc - design @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((yc - yc.mean()) ** 2).sum())

    expected = {
        "intercept": beta[0],
        "slope": beta[1],
        "r_squared": 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / len(yc)),
    }
    checks = failures = 0
    for field, ref in expected.items():
        checks += 1
        if not close(got[field], float(ref), tol):
            failures += 1
            print(f"FAIL harmonic: {field} rust={got[field]!r} ref={float(ref)!r}")
    for k in range(n_harmonics):
        a, b = beta[2 + 2 * k], beta[3 + 2 * k]
        comp = got["components"][k]
        for field, ref in (("cos_coef", a), ("sin_coef", b),
                           ("amplitude", math.hypot(a, b)),
                           ("phase", math.atan2(b, a))):
            checks += 1
            if not close(comp[field], float(ref), tol):
                failures += 1
                print(f"FAIL harmonic k={k + 1}: {field} "
                      f"rust={comp[field]!r} ref={float(ref)!r}")
    print(f"ok   harmonic: n={got['n']}")
    return checks, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    failures = 0
    checks = 0
    for name, values in CASES.items():
        got = run_datacube(values)
        clean = values[~np.isnan(values)]
        idx = np.arange(len(values), dtype=float)[~np.isnan(values)]

        ref_mk = mk.original_test(values)  # pyMannKendall drops NaN itself
        ref_sen = mk.sens_slope(clean)
        expected = {
            ("mann_kendall", "s"): ref_mk.s,
            ("mann_kendall", "var_s"): ref_mk.var_s,
            ("mann_kendall", "z"): ref_mk.z,
            ("mann_kendall", "tau"): ref_mk.Tau,
            ("mann_kendall", "p_value"): ref_mk.p,
        }
        if np.isnan(values).any():
            # sens_slope assumes unit spacing after dropping NaN; with gaps the
            # correct reference is the median pairwise slope on the real t
            ii, jj = np.triu_indices(len(clean), k=1)
            expected[("theil_sen", "slope")] = np.median(
                (clean[jj] - clean[ii]) / (idx[jj] - idx[ii])
            )
        else:
            expected[("theil_sen", "slope")] = ref_sen.slope
            expected[("theil_sen", "intercept")] = ref_sen.intercept

        if not np.all(clean == clean[0]):
            # constant input: scipy reports NaN r/stderr/p, we define a
            # perfect fit (r²=1, p=1) -> only compare when non-constant
            ols = st.linregress(idx, clean)
            expected.update({
                ("ols", "slope"): ols.slope,
                ("ols", "intercept"): ols.intercept,
                ("ols", "r_squared"): ols.rvalue ** 2,
                ("ols", "std_err"): ols.stderr,
                ("ols", "p_value"): ols.pvalue,
            })

        for (section, field), ref in expected.items():
            checks += 1
            val = got[section][field]
            if not close(val, float(ref), args.tol):
                failures += 1
                print(f"FAIL {name}: {section}.{field} rust={val!r} ref={float(ref)!r}")

        trend_map = {"increasing": "increasing", "decreasing": "decreasing",
                     "no trend": "no trend"}
        checks += 1
        if trend_map[got["mann_kendall"]["trend"]] != ref_mk.trend:
            failures += 1
            print(f"FAIL {name}: trend rust={got['mann_kendall']['trend']} "
                  f"ref={ref_mk.trend}")
        print(f"ok   {name}: n={got['n']}")

    hc, hf = validate_harmonic(args.tol)
    checks += hc
    failures += hf

    print(f"\n{checks - failures}/{checks} checks passed (tol={args.tol})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
