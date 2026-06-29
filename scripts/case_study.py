#!/usr/bin/env python3
"""Real case study for datacube-rs (Computers & Geosciences §4.4).

Builds an NDVI data cube over an area of the central-southern Chile forest
landscape affected by the catastrophic February 2023 wildfires, and uses the
datacube-rs estimators (via the Python bindings) to (i) date the disturbance
per pixel with the OLS-CUSUM break detector and (ii) map the post-disturbance
recovery trend with Theil-Sen. Ingestion uses standard STAC tooling so the case
study is reproducible without the optional datacube-io path; the analysis is
datacube-rs.

Run: .venv-validate/bin/python scripts/case_study.py [--verify]
"""

import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
from odc.stac import load as odc_load

import datacube_rs as dc

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "papers" / "draft" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Forest landscape near Santa Juana (Biobio, Chile), severely burned Feb 2023.
BBOX = [-72.96, -37.20, -72.90, -37.15]
DATE = "2022-09-01/2024-03-31"
RES = 100             # metres (native UTM grid); coarse for a tractable demo
FIRE_T = 2023.10      # early February 2023


def load_ndvi_cube():
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                            datetime=DATE,
                            query={"eo:cloud_cover": {"lt": 40}}).items())
    ds = odc_load(items, bands=["B04", "B08", "SCL"], bbox=BBOX,
                  resolution=RES, groupby="solar_day")
    # cloud / shadow / snow mask from the scene classification layer
    scl = ds["SCL"]
    clear = ~scl.isin([0, 1, 3, 8, 9, 10, 11])
    red = ds["B04"].where(clear).astype("float32")
    nir = ds["B08"].where(clear).astype("float32")
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.where(np.isfinite(ndvi))
    t = np.array([np.datetime64(v) for v in ndvi.time.values])
    # fractional years
    year = t.astype("datetime64[Y]").astype(int) + 1970
    frac = ((t - t.astype("datetime64[Y]")) / np.timedelta64(1, "D"))
    days = np.where((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 366, 365)
    tfrac = year + frac / days
    return ndvi, tfrac


def validate_modis():
    """External check: MODIS MCD64A1 burn dates over the AOI (independent
    product). Confirms the datacube-rs break dates (run with --modis)."""
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
    verify = "--verify" in sys.argv
    ndvi, t = load_ndvi_cube()
    # estimators expect f64; NDVI is computed as f32
    arr = np.ascontiguousarray(ndvi.transpose("y", "x", "time").values, dtype=np.float64)
    t = t.astype(np.float64)
    ny, nx, nt = arr.shape
    print(f"NDVI cube: {ny}x{nx} px, {nt} time steps, {t.min():.2f}-{t.max():.2f}")

    if verify:
        # area-mean NDVI vs time, and the pre/post-fire change, to confirm signal
        m = np.nanmean(arr.reshape(-1, nt), axis=0)
        pre = np.nanmean(m[t < FIRE_T]); post = np.nanmean(m[(t > FIRE_T) & (t < FIRE_T + 0.5)])
        print(f"area-mean NDVI pre-fire {pre:.2f} -> 6 months post {post:.2f} (drop {pre-post:.2f})")
        for ti, mi in zip(t, m):
            print(f"  {ti:.2f}: {mi:.2f}" + ("  <-- fire" if abs(ti - FIRE_T) < 0.05 else ""))
        return

    # datacube-rs analysis: build a (band, y, x, time) cube and run the estimators
    data = arr[None, :, :, :].astype("float64")          # 1 band
    cube = dc.Cube(np.ascontiguousarray(data), t.astype("float64"), ["ndvi"])

    band = 0
    first_break = np.full((ny, nx), np.nan)
    drop_mag = np.full((ny, nx), np.nan)
    recov = np.full((ny, nx), np.nan)
    for y in range(ny):
        for x in range(nx):
            s = arr[y, x, :]
            tv = t[np.isfinite(s)]; sv = s[np.isfinite(s)]
            if sv.size < 20:
                continue
            try:
                r = dc.detect_breaks(tv, sv, 0.05, 1, 1.0, 12)
            except Exception:
                continue
            if r["breaks"]:
                bt = r["breaks"][0]["time"]
                first_break[y, x] = bt
                # NDVI change across the first break (mean 0.3 yr each side)
                before = np.nanmean(sv[(tv > bt - 0.3) & (tv <= bt)])
                after = np.nanmean(sv[(tv > bt) & (tv <= bt + 0.3)])
                drop_mag[y, x] = after - before
            # post-fire recovery slope (Theil-Sen on the post-window)
            post = (tv > FIRE_T + 0.05)
            if post.sum() >= 6:
                recov[y, x] = dc.theil_sen(tv[post], sv[post])["slope"]

    nb = np.isfinite(first_break)
    print(f"pixels with a detected break: {nb.sum()}/{ny*nx} ({100*nb.sum()/(ny*nx):.0f}%)")
    if nb.sum():
        bt = first_break[nb]
        feb = ((bt >= 2023.0) & (bt <= 2023.25)).sum()
        print(f"first-break median {np.median(bt):.2f}; {100*feb/nb.sum():.0f}% in Jan-Mar 2023")
        print(f"NDVI drop at break: median {np.nanmedian(drop_mag):.2f}")
        print(f"post-fire recovery slope: median {np.nanmedian(recov):.3f} NDVI/yr")

    np.savez(REPO / "papers" / "draft" / "figures" / "case_arrays.npz",
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
