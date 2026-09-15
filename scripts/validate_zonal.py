#!/usr/bin/env python3
"""Cross-validate datacube-rs zonal aggregation against rasterio + shapely.

Builds a small georeferenced cube and a polygon, runs the shipped Python
binding (`datacube_rs.Cube.zonal`, built with the `stac` feature), and compares
each pixel-inclusion / reducer combination against a reference computed with
`rasterio.features.geometry_mask` (Center / AllTouched) and `shapely`
intersection areas (AreaFraction), within a documented tolerance.

Requires the validation venv with the stac-enabled binding:
    VIRTUAL_ENV=.venv-validate maturin develop --release \\
        -m crates/datacube-python/Cargo.toml --features stac,extension-module
    .venv-validate/bin/pip install rasterio shapely      # numpy already present
    .venv-validate/bin/python scripts/validate_zonal.py

Reference semantics matched:
  - Center      = rasterio geometry_mask(all_touched=False)  (pixel centre rule)
  - AllTouched  = rasterio geometry_mask(all_touched=True)
  - AreaFraction= shapely  area(pixel ∩ polygon) / pixel_area  (exactextract rule)
Tolerance: Center/AllTouched are exact set operations → 1e-9; AreaFraction is
1e-6 relative, because geo's BooleanOps (i_overlay) snaps coordinates to a
fixed-precision integer grid and so differs from shapely's exact floating
intersection area by ~1e-7. The pixel *sets* of Center/AllTouched match exactly.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import datacube_rs as dc

try:
    import rasterio.features
    from rasterio.transform import Affine
    from shapely.geometry import box, shape
except ImportError as e:  # pragma: no cover
    print(f"missing dependency: {e}\n  .venv-validate/bin/pip install rasterio shapely")
    sys.exit(2)

# ── Synthetic grid ─────────────────────────────────────────────────────────
# 20x20 grid, 1-unit pixels, origin (0, 20) north-up, EPSG:32719 (values are
# arbitrary; we only ever compare against the same numpy array).
NY = NX = 20
EPSG = 32719
GDAL = [0.0, 1.0, 0.0, 20.0, 0.0, -1.0]  # [origin_x, px_w, 0, origin_y, 0, px_h]
AFFINE = Affine.from_gdal(*GDAL)
ORIGIN_X, PX_W, _, ORIGIN_Y, _, PX_H = GDAL

# A rectangle whose edges fall inside cells (not on grid lines) so that
# Center and AllTouched select genuinely different pixel sets.
POLY_COORDS = [(2.6, 2.6), (12.4, 2.6), (12.4, 12.4), (2.6, 12.4), (2.6, 2.6)]


def make_data(with_nan: bool) -> np.ndarray:
    data = np.zeros((1, NY, NX, 1))
    for r in range(NY):
        for c in range(NX):
            data[0, r, c, 0] = r * NX + c
    if with_nan:
        # a covered interior pixel → excluded from reductions, drops n_valid
        data[0, 9, 5, 0] = np.nan
    return data


def geojson_path(tmp: Path) -> str:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ID": "w1"},
                "geometry": {"type": "Polygon", "coordinates": [list(map(list, POLY_COORDS))]},
            }
        ],
    }
    p = tmp / "poly.geojson"
    p.write_text(json.dumps(fc))
    return str(p)


def cube(data: np.ndarray):
    return dc.Cube(data, np.array([2024.0]), ["b1"]).with_georef(epsg=EPSG, transform=GDAL)


def rasterio_mask(all_touched: bool) -> np.ndarray:
    geom = {"type": "Polygon", "coordinates": [list(map(list, POLY_COORDS))]}
    # invert=True → True where inside the polygon.
    return rasterio.features.geometry_mask(
        [geom], out_shape=(NY, NX), transform=AFFINE, invert=True, all_touched=all_touched
    )


def area_fraction_weights() -> np.ndarray:
    poly = shape({"type": "Polygon", "coordinates": [list(map(list, POLY_COORDS))]})
    w = np.zeros((NY, NX))
    pixel_area = abs(PX_W * PX_H)
    for r in range(NY):
        for c in range(NX):
            x0 = ORIGIN_X + c * PX_W
            x1 = ORIGIN_X + (c + 1) * PX_W
            y0 = ORIGIN_Y + r * PX_H
            y1 = ORIGIN_Y + (r + 1) * PX_H
            cell = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            inter = cell.intersection(poly).area
            w[r, c] = inter / pixel_area
    return w


def reduce_ref(values: np.ndarray, weights: np.ndarray, reducer: str, threshold=None):
    """Weighted reference reduction over finite values (weights>0)."""
    finite = np.isfinite(values) & (weights > 0)
    v = values[finite]
    w = weights[finite]
    n_valid = int(finite.sum())
    n_total = int((weights > 0).sum())
    if v.size == 0:
        return float("nan"), n_valid, n_total
    sw = w.sum()
    if reducer == "mean":
        val = float((w * v).sum() / sw)
    elif reducer == "sum":
        val = float((w * v).sum())
    elif reducer == "count":
        val = float(sw)
    elif reducer == "min":
        val = float(v.min())
    elif reducer == "max":
        val = float(v.max())
    elif reducer == "std":
        mean = (w * v).sum() / sw
        val = float(np.sqrt(max((w * v * v).sum() / sw - mean * mean, 0.0)))
    elif reducer == "median":
        val = float(np.median(v))  # unweighted, matching the engine
    elif reducer == "fraction_above":
        val = float((w * (v > threshold)).sum() / sw)
    else:
        raise ValueError(reducer)
    return val, n_valid, n_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    tol = args.tol

    if not hasattr(dc.Cube, "zonal"):
        print("datacube_rs built without the 'stac' feature; rebuild with it (see docstring)")
        return 2

    checks = failures = 0

    def check(name, got, ref, atol=tol):
        nonlocal checks, failures
        checks += 1
        ok = (np.isnan(got) and np.isnan(ref)) or abs(got - ref) <= atol
        if not ok:
            failures += 1
            print(f"FAIL {name}: got={got!r} ref={ref!r}")

    data = make_data(with_nan=False)
    values = data[0, :, :, 0]
    c = cube(data)

    center_mask = rasterio_mask(all_touched=False).astype(float)
    touched_mask = rasterio_mask(all_touched=True).astype(float)
    frac = area_fraction_weights()

    with tempfile.TemporaryDirectory() as td:
        vec = geojson_path(Path(td))

        combos = [
            ("center", center_mask, ["mean", "count", "min", "max", "std", "sum", "median"]),
            ("all_touched", touched_mask, ["mean", "count", "sum"]),
            ("area_fraction", frac, ["mean", "count", "sum", "std"]),
        ]
        for inclusion, weights, reducers in combos:
            # geo's BooleanOps (i_overlay) snaps coordinates to a fixed-precision
            # integer grid, so AreaFraction weights differ from shapely's exact
            # floating intersection by ~1e-7 relative — Center/AllTouched are
            # exact set operations and match to 1e-9.
            for reducer in reducers:
                got = c.zonal(vec, id_field="ID", reducer=reducer, inclusion=inclusion,
                              source_epsg=EPSG)
                ref_val, ref_nvalid, ref_ntotal = reduce_ref(values, weights, reducer)
                atol = 1e-6 * max(1.0, abs(ref_val)) if inclusion == "area_fraction" else tol
                check(f"{inclusion}.{reducer}.value", got["value"][0], ref_val, atol=atol)
                check(f"{inclusion}.{reducer}.n_valid", got["n_valid"][0], ref_nvalid, atol=0)
                check(f"{inclusion}.{reducer}.n_total", got["n_total"][0], ref_ntotal, atol=0)
            print(f"ok   {inclusion}: n_total ref={int((weights>0).sum())}")

        # fraction_above with a threshold (center)
        thr = float(values[center_mask > 0].mean())
        got = c.zonal(vec, id_field="ID", reducer="fraction_above", inclusion="center",
                      source_epsg=EPSG, threshold=thr)
        ref_val, _, _ = reduce_ref(values, center_mask, "fraction_above", threshold=thr)
        check(f"center.fraction_above(>{thr:.1f}).value", got["value"][0], ref_val)

        # NaN handling: an interior covered pixel is NaN → n_valid drops by 1.
        cn = cube(make_data(with_nan=True))
        got = cn.zonal(vec, id_field="ID", reducer="mean", inclusion="center", source_epsg=EPSG)
        vals_nan = make_data(with_nan=True)[0, :, :, 0]
        ref_val, ref_nvalid, ref_ntotal = reduce_ref(vals_nan, center_mask, "mean")
        check("center.mean.nan.value", got["value"][0], ref_val)
        check("center.mean.nan.n_valid", got["n_valid"][0], ref_nvalid, atol=0)
        check("center.mean.nan.n_total", got["n_total"][0], ref_ntotal, atol=0)

    print(f"\n{checks - failures}/{checks} checks passed (tol={tol})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
