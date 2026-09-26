#!/usr/bin/env python3
"""Cross-sensor Landsat + Sentinel-2 fusion → per-wetland trajectories.

The reproducible orchestration behind the wetlands paper: it fuses the
multi-decadal Landsat Collection-2 record with Sentinel-2 into one consistent
per-wetland NDWI trajectory and runs the significance-controlled trend
analysis. It uses only engine primitives (no GDAL): datacube_rs.stack (per
collection), Cube.rename_bands, Cube.harmonize, Cube.concat_time, Cube.ndwi,
Cube.zonal, seasonal_mann_kendall and fdr.

Recipe:
  1. Stack Landsat C2 L2 and Sentinel-2 L2A SEPARATELY on the SAME fixed grid
     (grid_epsg/grid_res/grid_bbox identical → aligned, concat-able cubes),
     each with its own bands, quality mask and reflectance scaling.
  2. Relabel both to common role names (green, nir).
  3. Harmonize Sentinel-2 into Landsat OLI spectral space (per-band bandpass
     adjustment; coefficients from --coeffs, e.g. Claverie et al. 2018 HLS
     Table 5 / Roy et al. 2016). Landsat is the reference (identity).
  4. Fuse in time (concat_time → ascending), compute NDWI.
  5. Zonal-reduce per wetland (monthly), then per-wetland seasonal Mann-Kendall
     + FDR across wetlands.

Requires the stac-enabled binding:
    VIRTUAL_ENV=.venv-validate maturin develop --release \\
        -m crates/datacube-python/Cargo.toml --features stac,extension-module
    .venv-validate/bin/python scripts/fuse_landsat_s2.py --vector wetlands.geojson \\
        --grid-bbox <minx,miny,maxx,maxy in EPSG:32719> [--coeffs coeffs.json]
"""

import argparse
import csv
import json
import sys

import numpy as np
import datacube_rs as dc

# Reflectance scaling to physical [0,1] surface reflectance.
LANDSAT_C2_SCALE, LANDSAT_C2_OFFSET = 2.75e-5, -0.2  # USGS C2 L2 SR
S2_L2A_SCALE, S2_L2A_OFFSET = 1e-4, -0.1  # baseline >= 04.00

# Role -> asset key per sensor.
LANDSAT_BANDS = {"green": "green", "nir": "nir08"}  # PC landsat-c2-l2 common names
S2_BANDS = {"green": "B03", "nir": "B08"}


def stack_sensor(collection, assets, bbox, datetime, grid_epsg, grid_res, grid_bbox,
                 scale, offset, mask, limit, overview):
    return dc.stack(
        catalog="pc", collection=collection, assets=assets,
        bbox=bbox, datetime=datetime,
        grid_epsg=grid_epsg, grid_res=grid_res, grid_bbox=grid_bbox,
        scale=scale, offset=offset, mask=mask, max_items=limit, overview=overview,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True, help="wetland polygons (.shp/.geojson)")
    ap.add_argument("--id-field", default="objectid")
    ap.add_argument("--bbox", required=True, help="WGS84 west,south,east,north")
    ap.add_argument("--grid-epsg", type=int, default=32719)
    ap.add_argument("--grid-res", type=float, default=30.0)
    ap.add_argument("--grid-bbox", required=True, help="target-CRS minx,miny,maxx,maxy")
    ap.add_argument("--landsat-datetime", default="2000-01-01/2024-12-31")
    ap.add_argument("--s2-datetime", default="2015-06-01/2024-12-31")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--overview", type=int, default=None)
    ap.add_argument("--coeffs", help="JSON {band:[slope,offset]} S2->OLI; identity if omitted")
    ap.add_argument("--out", default="fused_wetland_trends.csv")
    ap.add_argument("--fdr-q", type=float, default=0.05)
    args = ap.parse_args()

    if not hasattr(dc, "stack"):
        print("datacube_rs built without the 'stac' feature; rebuild (see docstring).")
        return 2

    bbox = tuple(float(x) for x in args.bbox.split(","))
    grid_bbox = tuple(float(x) for x in args.grid_bbox.split(","))
    coeffs = json.load(open(args.coeffs)) if args.coeffs else {}
    if not coeffs:
        print("WARNING: no --coeffs given; harmonization is IDENTITY (no bandpass "
              "adjustment). Supply Claverie 2018 / Roy 2016 S2->OLI coefficients "
              "for a scientifically valid fusion.", file=sys.stderr)

    common = dict(bbox=bbox, grid_epsg=args.grid_epsg, grid_res=args.grid_res,
                  grid_bbox=grid_bbox, limit=args.limit, overview=args.overview)

    print("stacking Landsat C2 L2 ...", file=sys.stderr)
    ls = stack_sensor("landsat-c2-l2", list(LANDSAT_BANDS.values()),
                      datetime=args.landsat_datetime, scale=LANDSAT_C2_SCALE,
                      offset=LANDSAT_C2_OFFSET, mask="qa_pixel", **common)["cube"]
    ls = ls.rename_bands(list(LANDSAT_BANDS.keys()))  # -> [green, nir]

    print("stacking Sentinel-2 L2A ...", file=sys.stderr)
    s2 = stack_sensor("sentinel-2-l2a", list(S2_BANDS.values()),
                      datetime=args.s2_datetime, scale=S2_L2A_SCALE,
                      offset=S2_L2A_OFFSET, mask="scl", **common)["cube"]
    s2 = s2.rename_bands(list(S2_BANDS.keys()))  # -> [green, nir]
    if coeffs:
        s2 = s2.harmonize({b: tuple(v) for b, v in coeffs.items()})  # S2 -> OLI

    print("fusing (concat_time) and computing NDWI ...", file=sys.stderr)
    fused = ls.concat_time(s2)
    ndwi = fused.ndwi("green", "nir")  # McFeeters NDWI

    print("zonal reduction per wetland (monthly) ...", file=sys.stderr)
    tbl = ndwi.zonal(args.vector, id_field=args.id_field, reducer="mean",
                     inclusion="center", window="monthly",
                     source_epsg=4326 if args.vector.endswith((".geojson", ".json")) else None)

    # per-wetland seasonal MK + FDR across wetlands
    import collections as _c
    by = _c.defaultdict(list)
    for pid, t, v in zip(tbl["polygon_id"], tbl["time"], tbl["value"]):
        by[pid].append((t, v))
    res = []
    for pid, series in by.items():
        series.sort()
        y = np.array([vv if vv is not None and vv == vv else np.nan for _, vv in series])
        if np.sum(np.isfinite(y)) < 24:
            continue
        smk = dc.seasonal_mann_kendall(y, period=12)
        res.append((pid, smk["p_value"], smk["trend"]))
    pv = np.array([r[1] for r in res])
    fdr = dc.fdr(pv, q=args.fdr_q, method="bh")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["polygon_id", "smk_p", "smk_trend", "fdr_adj", "fdr_sig"])
        for (pid, p, trend), adj, rej in zip(res, fdr["adjusted"], fdr["rejected"]):
            w.writerow([pid, p, trend, adj, bool(rej)])
    n_sig = int(fdr["n_significant"])
    print(f"fused span {fused.time[0]:.2f}..{fused.time[-1]:.2f}, "
          f"{fused.dims[3]} obs; {len(res)} wetlands; {n_sig} significant (FDR q={args.fdr_q})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
