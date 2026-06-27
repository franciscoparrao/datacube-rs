#!/usr/bin/env python3
"""Controlled recovery experiment for datacube-rs (Computers & Geosciences §4).

Generates synthetic NDVI-like series with a KNOWN linear trend and a KNOWN
abrupt break, under a grid of noise levels and cloud-gap fractions, runs the
datacube-rs estimators through the Python bindings, and measures how well the
ground truth is recovered:

  (A) trend recovery   — Theil-Sen slope bias/RMSE and Mann-Kendall detection
                         rate vs noise (no break);
  (B) break recovery   — OLS-CUSUM detection rate and break-time error vs
                         noise and gap fraction (known break).

This is a controlled validation with ground truth, complementing the
numerical-parity check (which tests implementation fidelity, not recovery).

Run:  .venv-validate/bin/python scripts/synthetic_recovery.py
Outputs: papers/draft/figures/recovery.pdf|.png and a printed summary table.
"""

from pathlib import Path

import numpy as np
import datacube_rs as dc

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "papers" / "draft" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PER_YEAR = 12          # monthly
YEARS = 6
N = YEARS * PER_YEAR
T = np.arange(N) / PER_YEAR            # fractional years
TRUE_SLOPE = 0.02                      # NDVI/yr
AMP = 0.20                             # annual amplitude
BREAK_T = 3.0                          # year of the injected step
N_REAL = 300                           # realisations per cell
RNG = np.random.default_rng(20260627)


def make_series(sigma, gap_frac, break_mag, rng):
    """One synthetic series with known trend (+ optional break) and gaps."""
    y = (0.45 + TRUE_SLOPE * T + AMP * np.cos(2 * np.pi * T)
         + rng.normal(0, sigma, N))
    if break_mag:
        y = y + np.where(T >= BREAK_T, break_mag, 0.0)
    if gap_frac:
        y = y.copy()
        y[rng.random(N) < gap_frac] = np.nan
    return y


def experiment_trend(sigmas):
    """Theil-Sen slope error vs noise on a realistic seasonal NDVI series."""
    rows = []
    for s in sigmas:
        slopes = []
        for _ in range(N_REAL):
            y = make_series(s, gap_frac=0.10, break_mag=0.0, rng=RNG)
            slopes.append(dc.theil_sen(T, y)["slope"])
        slopes = np.array(slopes)
        rows.append(dict(sigma=s,
                         bias=float(np.mean(slopes) - TRUE_SLOPE),
                         rmse=float(np.sqrt(np.mean((slopes - TRUE_SLOPE) ** 2)))))
    return rows


def experiment_mk_power(sigmas):
    """Mann-Kendall power vs noise on a MONOTONIC series (its intended use;
    strongly seasonal series must be deseasonalised or use seasonal MK first)."""
    rows = []
    for s in sigmas:
        det = 0
        for _ in range(N_REAL):
            y = 0.45 + TRUE_SLOPE * T + RNG.normal(0, s, N)
            det += (dc.mann_kendall(y, 0.05)["trend"] == "increasing")
        rows.append(dict(sigma=s, mk_power=det / N_REAL))
    return rows


def experiment_break_vs_mag(mags, sigma, gap):
    """OLS-CUSUM detection rate vs disturbance magnitude (realistic NDVI drops
    are 0.2-0.5), at fixed noise and gap fraction."""
    rows = []
    for m in mags:
        hit, t_err = 0, []
        for _ in range(N_REAL):
            y = make_series(sigma, gap_frac=gap, break_mag=m, rng=RNG)
            try:
                r = dc.detect_breaks(T, y, 0.05, 1, 1.0, 12)
            except Exception:
                continue
            if r["breaks"]:
                hit += 1
                times = [b["time"] for b in r["breaks"]]
                t_err.append(min(abs(np.array(times) - BREAK_T)))
        rows.append(dict(mag=m, gap=gap,
                         detect_rate=hit / N_REAL,
                         t_err_med=float(np.median(t_err)) if t_err else np.nan))
    return rows


def false_positive_rate(sigmas):
    """Break false-positive rate on stable (no-break) series."""
    rows = []
    for s in sigmas:
        fp = 0
        for _ in range(N_REAL):
            y = make_series(s, gap_frac=0.10, break_mag=0.0, rng=RNG)
            try:
                r = dc.detect_breaks(T, y, 0.05, 1, 1.0, 12)
                fp += bool(r["breaks"])
            except Exception:
                pass
        rows.append(dict(sigma=s, fp_rate=fp / N_REAL))
    return rows


def main():
    sigmas = np.round(np.linspace(0.02, 0.14, 7), 3)
    mags = np.round(np.linspace(0.10, 0.50, 9), 2)
    trend = experiment_trend(sigmas)
    mk = experiment_mk_power(sigmas)
    fp = false_positive_rate(sigmas)
    brk10 = experiment_break_vs_mag(mags, sigma=0.05, gap=0.10)
    brk40 = experiment_break_vs_mag(mags, sigma=0.05, gap=0.40)

    print(f"\nControlled recovery (N={N} monthly obs, true slope={TRUE_SLOPE}/yr, "
          f"{N_REAL} realisations/cell)\n")
    print(f"{'sigma':>6} | {'TS bias':>9} {'TS rmse':>9} | {'MK power':>9} | {'FP rate':>8}")
    for a, m, f in zip(trend, mk, fp):
        print(f"{a['sigma']:>6.3f} | {a['bias']:>9.4f} {a['rmse']:>9.4f} | "
              f"{m['mk_power']:>9.2f} | {f['fp_rate']:>8.3f}")
    print(f"\nBreak detection vs magnitude (sigma=0.05):")
    print(f"{'mag':>6} | {'10% gaps':>9} {'40% gaps':>9} {'t-err(yr)':>9}")
    for b1, b4 in zip(brk10, brk40):
        print(f"{b1['mag']:>6.2f} | {b1['detect_rate']:>9.2f} {b4['detect_rate']:>9.2f} "
              f"{b1['t_err_med']:>9.3f}")

    plot(sigmas, mags, trend, mk, fp, brk10, brk40)


def plot(sigmas, mags, trend, mk, fp, brk10, brk40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) Theil-Sen slope RMSE vs noise
    ax[0].plot(sigmas, [r["rmse"] for r in trend], "o-", color="#c1666b", lw=1.5, ms=4)
    ax[0].set_xlabel("noise $\\sigma$"); ax[0].set_ylabel("Theil--Sen slope RMSE (/yr)")
    ax[0].set_ylim(0, None)
    ax[0].set_title("(a) Trend slope recovery", fontsize=9, loc="left")

    # (b) Mann-Kendall power (monotonic) + break false-positive rate
    ax[1].plot(sigmas, [r["mk_power"] for r in mk], "s-", color="#2e86ab",
               lw=1.5, ms=4, label="MK power")
    ax[1].plot(sigmas, [r["fp_rate"] for r in fp], "^--", color="#9aa0a6",
               lw=1.3, ms=4, label="break false-pos.")
    ax[1].axhline(0.05, color="k", lw=0.7, ls=":", label="$\\alpha=0.05$")
    ax[1].set_xlabel("noise $\\sigma$"); ax[1].set_ylabel("rate")
    ax[1].set_ylim(-0.03, 1.05)
    ax[1].legend(fontsize=6.5, frameon=False, loc="center left")
    ax[1].set_title("(b) MK power & break specificity", fontsize=9, loc="left")

    # (c) break detection rate vs disturbance magnitude, two gap levels
    ax[2].plot(mags, [r["detect_rate"] for r in brk10], "o-", color="#3a7d44",
               lw=1.5, ms=4, label="10% gaps")
    ax[2].plot(mags, [r["detect_rate"] for r in brk40], "o--", color="#7cae7a",
               lw=1.3, ms=4, label="40% gaps")
    ax[2].set_xlabel("break magnitude (NDVI)"); ax[2].set_ylabel("detection rate")
    ax[2].set_ylim(-0.03, 1.05)
    ax[2].legend(fontsize=7, frameon=False, loc="lower right")
    ax[2].set_title("(c) Break recovery ($\\sigma{=}0.05$)", fontsize=9, loc="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"recovery.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR/'recovery.pdf'}")


if __name__ == "__main__":
    main()
