"""Smoke + parity tests for the datacube_rs Python bindings.

Run with the build venv:
    VIRTUAL_ENV=.venv-validate maturin develop --release
    .venv-validate/bin/python -m pytest crates/datacube-python/tests
"""

import math

import numpy as np
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


def test_cube_gapfill_interpolates():
    data = np.array([1.0, np.nan, np.nan, 7.0]).reshape(1, 1, 1, 4)
    cube = dc.Cube(data, np.arange(4, dtype=float), ["b"])
    filled = cube.gapfill().to_numpy()
    assert math.isclose(filled[0, 0, 0, 1], 3.0, abs_tol=1e-12)
    assert math.isclose(filled[0, 0, 0, 2], 5.0, abs_tol=1e-12)
