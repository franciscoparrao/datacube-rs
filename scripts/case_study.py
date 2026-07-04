#!/usr/bin/env python3
"""Real case study for datacube-rs (Computers & Geosciences, Section 4.8 /
case study). Builds an NDVI data cube over an area of the central-southern
Chile forest landscape affected by the catastrophic February 2023 wildfires,
runs the OLS-CUSUM break detector to date the disturbance per pixel and the
Theil-Sen estimator on the post-disturbance window to map recovery.

Ingestion, per-pixel cloud/shadow masking, NDVI computation and break
detection all run natively in `datacube stack` (the CLI, `--features stac`);
this script only invokes the CLI twice (full period for masking+NDVI+breaks,
post-fire window for the recovery slope), reads the resulting GeoZarr cube
and GeoTIFF maps back into NumPy, and does the two bespoke per-pixel
computations the engine does not expose as a built-in statistic: the NDVI
drop magnitude across each pixel's detected break, and picking the two
example pixels for the time-series panel. No Python geospatial stack
(odc-stac/xarray) is used for the analysis; only `zarr`/`rasterio` to read
back what `datacube-rs` already computed.

Run: .venv-validate/bin/python scripts/case_study.py [--modis]
Requires a `datacube` binary built with `--features stac`
(`cargo build --release -p datacube-cli --features stac`); override its path
with the DATACUBE_BIN environment variable if not at the default location.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import zarr

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "papers" / "draft" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Forest landscape near Santa Juana (Biobio, Chile), severely burned Feb 2023.
BBOX = [-72.96, -37.20, -72.90, -37.15]           # WGS84 west,south,east,north
UTM_BBOX = [681045.05, 5880875.12, 686493.47, 5886539.28]  # same, in EPSG:32718
GRID_EPSG = 32718
RES = 100             # metres; coarse for a tractable demo
FIRE_T = 2023.10      # early February 2023
POST_FIRE_START = "2023-03-01"     # safely after FIRE_T + 0.05 (recovery window)
FULL_RANGE = "2022-09-01/2024-03-31"

DATACUBE_BIN = os.environ.get(
    "DATACUBE_BIN", str(REPO / "target" / "release" / "datacube")
)


def run_stack(datetime_range, extra_args, workdir):
    """Invokes `datacube stack` and returns its parsed JSON report."""
    cmd = [
        DATACUBE_BIN, "stack",
        "--catalog", "pc", "--collection", "sentinel-2-l2a",
        "--assets", "B04,B08,SCL",
        "--bbox", ",".join(str(v) for v in BBOX),
        "--datetime", datetime_range,
        "--max-cloud", "40",
        "--mask-scl", "--mask-keep", "2,4,5,6,7",
        "--grid-epsg", str(GRID_EPSG), "--grid-res", str(RES),
        "--grid-bbox", ",".join(str(v) for v in UTM_BBOX),
        "--composite", "same-time",
        "--index", "ndvi", "--nir", "B08", "--red", "B04",
        "--chunk-size", "24",
        "--limit", "500",  # default 100 truncates a 19-month S2 archive
    ] + extra_args
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"datacube stack failed:\n{result.stderr}")
    print(result.stderr.strip())
    return json.loads(result.stdout)


def read_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1)


def read_ndvi_zarr(path):
    z = zarr.open(str(path), mode="r")
    arr = z["cube"]
    data = np.asarray(arr[0])          # single band (ndvi): (y, x, time)
    t = np.asarray(arr.attrs["time"], dtype=np.float64)
    return data, t


def validate_modis():
    """External check: MODIS MCD64A1 burn dates over the AOI (independent
    product, different sensor/algorithm). Confirms the datacube-rs break
    dates. Uses standard STAC tooling directly -- MODIS is not part of the
    datacube-rs pipeline being validated, so no native ingestion applies."""
    import planetary_computer as pc
    import pystac_client
    from odc.stac import load as odc_load

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace)
    items = list(cat.search(collections=["modis-64A1-061"], bbox=BBOX,
                             datetime="2023-01-01/2023-06-30").items())
    ds = odc_load(items, bands=["Burn_Date"], bbox=BBOX, resolution=500,
                  groupby="solar_day")
    bd = ds["Burn_Date"].values
    burned = bd[bd > 0].astype(float)
    ty = 2023 + (burned - 1) / 365
    print(f"MODIS MCD64A1: burn DOY {burned.min():.0f}-{burned.max():.0f} "
          f"(median {np.median(burned):.0f}); fractional-year median {np.median(ty):.3f}; "
          f"AOI burned {100*(bd > 0).any(axis=0).mean():.0f}%")


def main():
    if "--modis" in sys.argv:
        validate_modis()
        return

    with tempfile.TemporaryDirectory() as workdir:
        workdir = Path(workdir)

        report = run_stack(FULL_RANGE, [
            "--breaks-output", str(workdir / "breaks_count.tif"),
            "--first-break-output", str(workdir / "first_break.tif"),
            "--break-harmonics", "1", "--break-alpha", "0.05",
            "--zarr-output", str(workdir / "ndvi_cube.zarr"),
        ], workdir)
        arr, t = read_ndvi_zarr(workdir / "ndvi_cube.zarr")
        first_break = read_geotiff(workdir / "first_break.tif")
        ny, nx, nt = arr.shape
        print(f"NDVI cube: {ny}x{nx} px, {nt} time steps, {t.min():.2f}-{t.max():.2f} "
              f"({len(report['scenes'])} scenes, {len(report['skipped'])} skipped)")

        run_stack(f"{POST_FIRE_START}/2024-03-31", [
            "--stat", "theil-sen",
            "--output", str(workdir / "recovery_slope.tif"),
        ], workdir)
        recov = read_geotiff(workdir / "recovery_slope.tif")
        if recov.shape != (ny, nx):
            sys.exit(f"recovery grid {recov.shape} does not match NDVI grid {(ny, nx)}")

        # NDVI drop magnitude across each pixel's first break: the one
        # per-pixel computation the engine does not expose as a built-in
        # statistic (mean NDVI in a +/-0.3 yr window around the break time).
        drop_mag = np.full((ny, nx), np.nan)
        for y in range(ny):
            for x in range(nx):
                bt = first_break[y, x]
                if not np.isfinite(bt):
                    continue
                s = arr[y, x, :]
                before = np.nanmean(s[(t > bt - 0.3) & (t <= bt)])
                after = np.nanmean(s[(t > bt) & (t <= bt + 0.3)])
                drop_mag[y, x] = after - before

        nb = np.isfinite(first_break)
        print(f"pixels with a detected break: {nb.sum()}/{ny*nx} ({100*nb.sum()/(ny*nx):.0f}%)")
        if nb.sum():
            bt = first_break[nb]
            feb = ((bt >= 2023.0) & (bt <= 2023.25)).sum()
            print(f"first-break median {np.median(bt):.2f}; {100*feb/nb.sum():.0f}% in Jan-Mar 2023")
            print(f"NDVI drop at break: median {np.nanmedian(drop_mag):.2f}")
            print(f"post-fire recovery slope: median {np.nanmedian(recov):.3f} NDVI/yr")

        np.savez(FIGDIR / "case_arrays.npz",
                 arr=arr, t=t, first_break=first_break, drop_mag=drop_mag, recov=recov)
        plot(arr, t, first_break, drop_mag, recov)


def plot(arr, t, first_break, drop_mag, recov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    ny, nx, nt = arr.shape
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))

    # (a) burned vs unburned NDVI series: strongest drop, and an unburned
    # pixel (no detected break) with the highest mean NDVI (healthy, stable)
    drop_flat = np.where(np.isfinite(drop_mag), drop_mag, np.inf)
    yb, xb = np.unravel_index(np.argmin(drop_flat), drop_flat.shape)
    mean_ndvi = np.nanmean(arr, axis=2)
    unburned = np.where(np.isfinite(first_break), -np.inf, mean_ndvi)
    ys, xs = np.unravel_index(np.nanargmax(unburned), unburned.shape)
    for (yy, xx), c, lab in [((yb, xb), "#c1666b", "burned"), ((ys, xs), "#3a7d44", "unburned")]:
        s = arr[yy, xx, :]; ok = np.isfinite(s)
        ax[0].plot(t[ok], s[ok], "o-", color=c, ms=2.5, lw=1, label=lab)
    ax[0].axvline(2023.10, color="k", ls=":", lw=0.8)
    ax[0].text(2023.10, ax[0].get_ylim()[0], " Feb 2023", fontsize=6, va="bottom")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("NDVI")
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax[0].set_title("(a) Pixel time series", fontsize=9, loc="left")

    def decorate(a):
        # scale bar (10 px = 1 km at 100 m) and north arrow
        ny_, nx_ = first_break.shape
        x0, yb_ = nx_ * 0.06, ny_ * 0.92
        a.plot([x0, x0 + 10], [yb_, yb_], "-", color="k", lw=2)
        a.text(x0 + 5, yb_ - ny_ * 0.04, "1 km", ha="center", va="bottom",
               fontsize=6, color="k")
        a.annotate("N", xy=(nx_ * 0.93, ny_ * 0.07), xytext=(nx_ * 0.93, ny_ * 0.22),
                   ha="center", fontsize=7, color="k",
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
        a.set_xticks([]); a.set_yticks([])

    # (b) first-break-time map
    im = ax[1].imshow(first_break, cmap="inferno", vmin=2022.9, vmax=2023.6)
    decorate(ax[1])
    ax[1].set_title("(b) First-break time", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)

    # (c) post-fire recovery slope
    im2 = ax[2].imshow(recov, cmap="BrBG", vmin=-0.3, vmax=0.3)
    decorate(ax[2])
    ax[2].set_title("(c) Recovery slope (NDVI/yr)", fontsize=9, loc="left")
    cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"casestudy.{ext}", dpi=200, bbox_inches="tight")
    print(f"figure -> {FIGDIR/'casestudy.pdf'}")


if __name__ == "__main__":
    main()
