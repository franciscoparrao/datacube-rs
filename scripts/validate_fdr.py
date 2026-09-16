#!/usr/bin/env python3
"""Numerical parity of the FDR field-significance control.

Cross-checks datacube_rs.fdr (Benjamini-Hochberg / Benjamini-Yekutieli) against
statsmodels.stats.multitest.multipletests(method='fdr_bh' | 'fdr_by') on
deterministic p-value fields, including NaN handling (masked pixels). This is
the multiple-testing control the C&G review demanded for per-pixel maps.

Requires the validation venv with the (default) binding + statsmodels:
    VIRTUAL_ENV=.venv-validate maturin develop --release \\
        -m crates/datacube-python/Cargo.toml
    .venv-validate/bin/python scripts/validate_fdr.py
"""

import sys

import numpy as np
from statsmodels.stats.multitest import multipletests

import datacube_rs as dc

TOL = 1e-12


def fields():
    rng = np.random.default_rng(11)
    cases = {}
    # mostly null with a handful of strong signals (typical trend map)
    p = rng.uniform(0, 1, 500)
    p[:20] = rng.uniform(0, 0.001, 20)
    cases["sparse_signal"] = p
    # dense signal
    cases["dense_signal"] = rng.uniform(0, 0.03, 200)
    # all null
    cases["all_null"] = rng.uniform(0.2, 1.0, 300)
    # tiny p-values and ties
    cases["ties"] = np.array([0.01, 0.01, 0.04, 0.04, 0.2, 0.2, 0.5, 0.9] * 5, dtype=float)
    return cases


def main():
    if not hasattr(dc, "fdr"):
        print("datacube_rs lacks fdr; rebuild the binding.")
        return 2
    total = fails = 0
    q = 0.05
    for name, p in fields().items():
        for method in ("fdr_bh", "fdr_by"):
            key = "bh" if method == "fdr_bh" else "by"
            # --- no NaN ---
            reject, padj, _, _ = multipletests(p, alpha=q, method=method)
            got = dc.fdr(p, q=q, method=key)
            total += 1
            if not np.allclose(got["adjusted"], padj, atol=TOL, rtol=0):
                fails += 1
                d = np.max(np.abs(got["adjusted"] - padj))
                print(f"FAIL {name}/{method}: adjusted max|Δ|={d:.2e}")
            total += 1
            if not np.array_equal(got["rejected"].astype(bool), reject):
                fails += 1
                print(f"FAIL {name}/{method}: reject mask differs "
                      f"({got['rejected'].sum()} vs {reject.sum()})")
            total += 1
            if int(got["n_significant"]) != int(reject.sum()):
                fails += 1
                print(f"FAIL {name}/{method}: n_significant {got['n_significant']} vs {reject.sum()}")

            # --- with NaN interleaved (statsmodels can't take NaN: run on finite) ---
            pn = p.copy()
            pn[1::7] = np.nan
            finite = np.isfinite(pn)
            reject_f, padj_f, _, _ = multipletests(pn[finite], alpha=q, method=method)
            gotn = dc.fdr(pn, q=q, method=key)
            total += 1
            if not np.allclose(gotn["adjusted"][finite], padj_f, atol=TOL, rtol=0):
                fails += 1
                print(f"FAIL {name}/{method}/NaN: adjusted differs")
            total += 1
            if gotn["adjusted"][~finite].size and not np.isnan(gotn["adjusted"][~finite]).all():
                fails += 1
                print(f"FAIL {name}/{method}/NaN: NaN positions not preserved")
            total += 1
            if not np.array_equal(gotn["rejected"][finite].astype(bool), reject_f):
                fails += 1
                print(f"FAIL {name}/{method}/NaN: reject mask differs")
        print(f"ok   {name}: m={len(p)}")

    print(f"\n{total - fails}/{total} checks passed (tol={TOL})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
