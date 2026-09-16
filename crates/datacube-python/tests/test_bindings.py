"""Smoke + parity tests for the datacube_rs Python bindings.

Run with the build venv:
    VIRTUAL_ENV=.venv-validate maturin develop --release
    .venv-validate/bin/python -m pytest crates/datacube-python/tests

The stack() test additionally needs a build with the 'stac' feature and
network access to Planetary Computer (see its skip reason):
    VIRTUAL_ENV=.venv-validate maturin develop --release --features stac,extension-module
"""

import math

import numpy as np
import pytest
import datacube_rs as dc


def test_linear_trend_exact():
    t = np.arange(10, dtype=float)
    y = 2.0 * t + 1.0
    r = dc.linear_trend(t, y)
    assert math.isclose(r["slope"], 2.0, abs_tol=1e-12)
    assert math.isclose(r["intercept"], 1.0, abs_tol=1e-12)
    assert math.isclose(r["r_squared"], 1.0, abs_tol=1e-12)
    assert r["n"] == 10


def test_theil_sen_robust_to_outlier():
    t = np.arange(11, dtype=float)
    y = 2.0 * t + 1.0
    y[5] = 100.0
    r = dc.theil_sen(t, y)
    assert math.isclose(r["slope"], 2.0, abs_tol=1e-12)


def test_mann_kendall_monotonic():
    y = np.arange(12, dtype=float)
    r = dc.mann_kendall(y)
    assert r["trend"] == "increasing"
    assert math.isclose(r["tau"], 1.0, abs_tol=1e-12)


def test_harmonic_recovers_amplitude():
    t = np.arange(48, dtype=float) / 12.0
    y = 0.5 + 0.01 * t + 0.2 * np.cos(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * t)
    r = dc.harmonic_regression(t, y, 1.0, 1)
    assert math.isclose(r["slope"], 0.01, abs_tol=1e-9)
    c = r["components"][0]
    assert math.isclose(c["amplitude"], math.hypot(0.2, 0.1), abs_tol=1e-9)


def test_detect_breaks_level_shift():
    t = np.arange(60, dtype=float)
    y = np.where(t < 30, 1.0, 6.0) + 0.05 * np.sin(t * 12.9898)
    r = dc.detect_breaks(t, y)
    assert len(r["breaks"]) == 1
    assert abs(r["breaks"][0]["index"] - 29) <= 1


def test_nan_dropped_pairwise():
    t = np.arange(6, dtype=float)
    y = np.array([0.0, 2.0, np.nan, 6.0, 8.0, np.nan])
    r = dc.linear_trend(t, y)
    assert r["n"] == 4
    assert math.isclose(r["slope"], 2.0, abs_tol=1e-12)


def test_seasonal_mann_kendall_removes_cycle():
    # pure repeating 4-step cycle, no interannual trend → S = 0
    y = np.tile([1.0, 5.0, 3.0, 8.0], 6)
    r = dc.seasonal_mann_kendall(y, period=4)
    assert r["s"] == 0.0
    assert r["trend"] == "no trend"
    # period=1 collapses to the plain test
    y2 = np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 10.0])
    assert math.isclose(
        dc.seasonal_mann_kendall(y2, period=1)["z"], dc.mann_kendall(y2)["z"], abs_tol=1e-12
    )


def test_hamed_rao_inflates_variance_under_autocorrelation():
    rng = np.random.default_rng(3)
    ar = np.empty(120)
    ar[0] = rng.normal()
    for i in range(1, 120):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.5)
    plain = dc.mann_kendall(ar)
    hr = dc.mann_kendall_hamed_rao(ar)
    assert hr["s"] == plain["s"]
    assert hr["var_s"] > plain["var_s"]  # correction raises variance
    # lag=0 → no correction, matches the plain test
    assert math.isclose(dc.mann_kendall_hamed_rao(ar, lag=0)["var_s"], plain["var_s"], abs_tol=1e-9)


def test_fdr_bh_controls_and_carries_nan():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, np.nan, 0.9])
    r = dc.fdr(p, q=0.05, method="bh")
    assert r["n_tested"] == 7
    assert np.isnan(r["adjusted"][6])
    assert not r["rejected"][6]
    assert r["n_significant"] == int(np.sum(r["rejected"]))
    # BY is at least as conservative as BH
    by = dc.fdr(p, q=0.05, method="by")
    assert by["n_significant"] <= r["n_significant"]


def _ramp_cube():
    # (1 band, 2x2, 5 t); value = t * (1 + y + x), distinct slope per pixel
    nb, ny, nx, nt = 1, 2, 2, 5
    data = np.zeros((nb, ny, nx, nt))
    for y in range(ny):
        for x in range(nx):
            for t in range(nt):
                data[0, y, x, t] = t * (1.0 + y + x)
    return dc.Cube(data, np.arange(nt, dtype=float), ["b1"])


def test_cube_dims_and_roundtrip():
    cube = _ramp_cube()
    assert cube.dims == (1, 2, 2, 5)
    assert cube.bands == ["b1"]
    back = cube.to_numpy()
    assert back.shape == (1, 2, 2, 5)
    assert math.isclose(back[0, 1, 1, 4], 4.0 * 3.0)


def test_cube_trend_map_matches_per_pixel_slope():
    cube = _ramp_cube()
    slope, pvalue = cube.trend_map(0, "theil_sen")
    assert slope.shape == (2, 2)
    # pixel (y,x) has slope (1+y+x)
    assert math.isclose(slope[0, 0], 1.0, abs_tol=1e-12)
    assert math.isclose(slope[1, 1], 3.0, abs_tol=1e-12)
    assert pvalue.shape == (2, 2)


def test_cube_composite_same_time_merges_tiles():
    # two tiles at t=0 (complementary coverage), one slice at t=1
    data = np.full((1, 1, 2, 3), np.nan)
    data[0, 0, 0, 0] = 1.0
    data[0, 0, 1, 1] = 3.0
    data[0, 0, 0, 2] = 5.0
    data[0, 0, 1, 2] = 7.0
    cube = dc.Cube(data, np.array([0.0, 0.0, 1.0]), ["b"])
    merged = cube.composite("same_time", "median")
    assert merged.dims == (1, 1, 2, 2)
    out = merged.to_numpy()
    assert out[0, 0, 0, 0] == 1.0
    assert out[0, 0, 1, 0] == 3.0


def test_cube_georef_is_none_by_default_and_propagates_through_ops():
    data = np.array([1.0, np.nan, 3.0, 4.0]).reshape(1, 1, 1, 4)
    cube = dc.Cube(data, np.arange(4, dtype=float), ["b"])
    assert cube.epsg is None
    assert cube.transform is None

    transform = [300000.0, 10.0, 0.0, 6200000.0, 0.0, -10.0]
    geo_cube = cube.with_georef(epsg=32719, transform=transform)
    assert geo_cube.epsg == 32719
    assert list(geo_cube.transform) == transform
    # original is untouched (with_georef returns a copy)
    assert cube.epsg is None

    # composite/gapfill preserve the spatial grid, so georef propagates
    composited = geo_cube.composite("same_time", "median")
    assert composited.epsg == 32719
    assert list(composited.transform) == transform
    filled = geo_cube.gapfill()
    assert filled.epsg == 32719
    assert list(filled.transform) == transform


def test_cube_composite_monthly_uses_calendar_months():
    # Jan 20 and Feb 5, 2023: 16 days apart, but distinct calendar months
    times = np.array([2023.0 + 19.5 / 365.0, 2023.0 + 35.5 / 365.0])
    data = np.array([1.0, 3.0]).reshape(1, 1, 1, 2)
    cube = dc.Cube(data, times, ["b"])
    monthly = cube.composite("monthly", "mean")
    assert monthly.dims == (1, 1, 1, 2)
    out = monthly.to_numpy()
    assert out[0, 0, 0, 0] == 1.0
    assert out[0, 0, 0, 1] == 3.0


def test_cube_gapfill_interpolates():
    data = np.array([1.0, np.nan, np.nan, 7.0]).reshape(1, 1, 1, 4)
    cube = dc.Cube(data, np.arange(4, dtype=float), ["b"])
    filled = cube.gapfill().to_numpy()
    assert math.isclose(filled[0, 0, 0, 1], 3.0, abs_tol=1e-12)
    assert math.isclose(filled[0, 0, 0, 2], 5.0, abs_tol=1e-12)


def _rgbn_cube():
    # red, nir, blue; 2x2 px, 1 time; second pixel column has a masked NIR
    nb, ny, nx, nt = 3, 2, 2, 1
    data = np.zeros((nb, ny, nx, nt))
    red = np.array([[0.10, 0.20], [0.15, 0.25]])
    nir = np.array([[0.50, np.nan], [0.40, 0.60]])
    blue = np.full((ny, nx), 0.05)
    data[0, :, :, 0] = red
    data[1, :, :, 0] = nir
    data[2, :, :, 0] = blue
    return dc.Cube(data, np.array([0.0]), ["red", "nir", "blue"])


def test_cube_ndvi_matches_numpy_and_propagates_nan():
    cube = _rgbn_cube()
    nd = cube.ndvi("nir", "red")
    assert nd.dims == (1, 2, 2, 1)
    assert nd.bands == ["ndvi"]
    out = nd.to_numpy()[0, :, :, 0]
    red = np.array([[0.10, 0.20], [0.15, 0.25]])
    nir = np.array([[0.50, np.nan], [0.40, 0.60]])
    ref = (nir - red) / (nir + red)
    # finite cells agree to machine precision; masked cell stays NaN both sides
    finite = np.isfinite(ref)
    assert np.allclose(out[finite], ref[finite], atol=1e-12)
    assert np.isnan(out[~finite]).all()


def test_cube_band_index_and_missing():
    cube = _rgbn_cube()
    assert cube.band_index("nir") == 1
    try:
        cube.band_index("swir")
    except Exception as e:  # noqa: BLE001 - binding maps to a Python error
        assert "swir" in str(e)
    else:
        raise AssertionError("expected band_index('swir') to raise")


def test_cube_savi_l0_equals_ndvi():
    cube = _rgbn_cube()
    savi = cube.savi("nir", "red", 0.0).to_numpy()[0, :, :, 0]
    ndvi = cube.ndvi("nir", "red").to_numpy()[0, :, :, 0]
    finite = np.isfinite(ndvi)
    assert np.allclose(savi[finite], ndvi[finite], atol=1e-12)


def test_cube_normalized_difference_generic():
    cube = _rgbn_cube()
    nd = cube.normalized_difference("nir", "red", "myidx")
    assert nd.bands == ["myidx"]


@pytest.mark.skipif(
    not hasattr(dc.Cube, "zonal"),
    reason="built without the 'stac' feature "
    "(VIRTUAL_ENV=... maturin develop --release --features stac,extension-module)",
)
def test_cube_zonal_over_geojson_square(tmp_path):
    """Offline zonal: a 4x4 UTM cube + a GeoJSON square → per-polygon mean.

    Values are row*10 + col; a square covering pixel centres of rows/cols 1,2
    selects {11, 12, 21, 22}, mean 16.5, over 4 pixels.
    """
    import json

    ny = nx = 4
    data = np.zeros((1, ny, nx, 1))
    for r in range(ny):
        for c in range(nx):
            data[0, r, c, 0] = r * 10 + c
    cube = dc.Cube(data, np.array([2024.0]), ["b1"]).with_georef(
        epsg=32719, transform=[0.0, 10.0, 0.0, 40.0, 0.0, -10.0]
    )

    # Square [5,35] x [5,35] in the cube's own CRS (source_epsg overrides the
    # GeoJSON default of WGS84 so no reprojection is attempted).
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ID": "w1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[5, 5], [35, 5], [35, 35], [5, 35], [5, 5]]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps(geojson))

    table = cube.zonal(
        str(path),
        id_field="ID",
        reducer="mean",
        inclusion="center",
        source_epsg=32719,
    )
    assert table["polygon_id"] == ["w1"]
    assert table["band"] == ["b1"]
    assert table["reducer"] == ["mean"]
    assert math.isclose(table["value"][0], 16.5, abs_tol=1e-12)
    assert table["n_valid"] == [4]
    assert table["n_total"] == [4]


@pytest.mark.skipif(
    not hasattr(dc, "stack"),
    reason="built without the 'stac' feature "
    "(VIRTUAL_ENV=... maturin develop --release --features stac,extension-module)",
)
def test_stack_against_planetary_computer():
    """Network e2e: STAC search -> COG reads -> masked cube -> NDVI -> trend.

    Mirrors the Rust-side ignored network test (datacube-io's
    stacks_sentinel2_red_band) and the CLI's e2e verification convention —
    run manually, not part of `cargo test`/CI.
    """
    result = dc.stack(
        catalog="pc",
        collection="sentinel-2-l2a",
        assets=["B04", "B08"],
        bbox=(-70.75, -33.55, -70.65, -33.45),
        datetime="2024-01-01/2024-02-28",
        max_cloud_cover=30.0,
        max_items=20,
        overview=4,
        mask_scl=True,
    )
    cube = result["cube"]
    assert cube.bands == ["B04", "B08"]
    assert cube.epsg == 32719
    assert len(result["scenes"]) > 0
    assert all({"id", "datetime", "time", "cloud_cover"} <= s.keys() for s in result["scenes"])

    ndvi = cube.ndvi("B08", "B04")
    assert ndvi.bands == ["ndvi"]
    slope, pvalue = ndvi.trend_map(0, method="theil_sen")
    assert slope.shape == cube.dims[1:3]
    assert np.isfinite(slope).any()
