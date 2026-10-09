#!/usr/bin/env python3
"""Merge per-year/per-shard zonal NDWI tables into per-wetland trends.

The full-extent, multi-decadal fused run is sharded by year (each shard a
`fuse_landsat_s2.py --zonal-table-out` table of per-(wetland, month) NDWI).
Because the calendar months are disjoint across years, concatenating the shard
tables reconstructs each wetland's full monthly series exactly. This script
concatenates them and runs the same per-wetland seasonal Mann-Kendall + FDR the
single-shot path runs — so the sharded result is identical to one run over the
whole period.

Usage:
    merge_fused_trends.py --out fused_wetland_trends.csv \\
        [--min-months 24] [--fdr-q 0.05] [--period 12] \\
        shard_2000.csv shard_2001.csv ... shard_2024.csv
"""
import argparse
import collections
import csv
import sys

import numpy as np
import datacube_rs as dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+", help="per-shard zonal table CSVs")
    ap.add_argument("--out", default="fused_wetland_trends.csv")
    ap.add_argument("--min-months", type=int, default=24,
                    help="minimum finite monthly observations to test a wetland")
    ap.add_argument("--fdr-q", type=float, default=0.05)
    ap.add_argument("--period", type=int, default=12,
                    help="season length for seasonal Mann-Kendall (12 = monthly)")
    args = ap.parse_args()

    # (polygon_id, time) -> value, keeping the last non-empty. Shards are
    # disjoint in time, so there is normally one entry per (pid, month); a
    # duplicate (overlapping shard windows) just overwrites with the same bin.
    by = collections.defaultdict(dict)
    n_rows = 0
    for path in args.tables:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                n_rows += 1
                v = row["value"]
                val = float(v) if v not in ("", "nan", "NaN") else np.nan
                t = float(row["time"])
                by[row["polygon_id"]][t] = val
    print(f"read {n_rows} rows from {len(args.tables)} shards; "
          f"{len(by)} wetlands", file=sys.stderr)

    res = []
    for pid, series in by.items():
        ts = sorted(series)
        y = np.array([series[t] for t in ts], float)
        if np.sum(np.isfinite(y)) < args.min_months:
            continue
        smk = dc.seasonal_mann_kendall(y, period=args.period)
        res.append((pid, smk["p_value"], smk["trend"]))

    n_sig = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["polygon_id", "smk_p", "smk_trend", "fdr_adj", "fdr_sig"])
        if res:
            pv = np.array([r[1] for r in res])
            fdr = dc.fdr(pv, q=args.fdr_q, method="bh")
            for (pid, p, trend), adj, rej in zip(res, fdr["adjusted"], fdr["rejected"]):
                w.writerow([pid, p, trend, adj, bool(rej)])
            n_sig = int(fdr["n_significant"])

    all_t = sorted({t for s in by.values() for t in s})
    span = f"{all_t[0]:.2f}..{all_t[-1]:.2f}" if all_t else "empty"
    print(f"merged span {span}, {len(all_t)} bins; {len(res)} wetlands tested; "
          f"{n_sig} significant (FDR q={args.fdr_q})", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
