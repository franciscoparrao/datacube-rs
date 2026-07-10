//! Python bindings for datacube-rs.
//!
//! Exposes the per-series statistics and the temporal `Cube` of
//! `datacube-core` to Python, with NumPy array interop. Built as the
//! `datacube_rs` extension module (see the `python/` package and
//! `maturin develop`).
//!
//! `dc.stack(...)` (STAC/COG ingestion, [`stack`]) is opt-in behind the
//! `stac` Cargo feature — it needs the SurtGIS sibling checkout and a
//! system GDAL, so the default build (and the wheel `pyproject.toml`
//! builds) stays standalone. Opt in with:
//! `VIRTUAL_ENV=.venv maturin develop --release --features stac,extension-module`.

use datacube_core::{CompositeMethod, CompositeWindow, Cube as CoreCube, GeoRef, indices, stats};
use ndarray::Array4;
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray4};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A pair of `(height, width)` NumPy grids (e.g. slope + p-value).
type GridPair<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// OLS linear trend → dict(slope, intercept, r_squared, std_err, p_value, n).
#[pyfunction]
fn linear_trend<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::linear_trend(t.as_slice()?, y.as_slice()?).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("slope", r.slope)?;
    d.set_item("intercept", r.intercept)?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("std_err", r.std_err)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

/// Theil-Sen robust slope → dict(slope, intercept, n).
#[pyfunction]
fn theil_sen<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::theil_sen(t.as_slice()?, y.as_slice()?).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("slope", r.slope)?;
    d.set_item("intercept", r.intercept)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

/// Mann-Kendall trend test → dict(trend, s, var_s, z, tau, p_value, n).
#[pyfunction]
#[pyo3(signature = (y, alpha=0.05))]
fn mann_kendall<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::mann_kendall_alpha(y.as_slice()?, alpha).map_err(err)?;
    let trend = match r.trend {
        stats::Trend::Increasing => "increasing",
        stats::Trend::Decreasing => "decreasing",
        stats::Trend::NoTrend => "no trend",
    };
    let d = PyDict::new(py);
    d.set_item("trend", trend)?;
    d.set_item("s", r.s)?;
    d.set_item("var_s", r.var_s)?;
    d.set_item("z", r.z)?;
    d.set_item("tau", r.tau)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

/// Harmonic regression with trend → dict(intercept, slope, r_squared, rmse,
/// n, components=[dict(harmonic, cos_coef, sin_coef, amplitude, phase), ...]).
#[pyfunction]
#[pyo3(signature = (t, y, period, n_harmonics=2))]
fn harmonic_regression<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    period: f64,
    n_harmonics: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::harmonic_regression(t.as_slice()?, y.as_slice()?, period, n_harmonics)
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("intercept", r.intercept)?;
    d.set_item("slope", r.slope)?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("rmse", r.rmse)?;
    d.set_item("n", r.n)?;
    let comps: Vec<Bound<'py, PyDict>> = r
        .components
        .iter()
        .map(|c| {
            let cd = PyDict::new(py);
            cd.set_item("harmonic", c.harmonic)?;
            cd.set_item("cos_coef", c.cos_coef)?;
            cd.set_item("sin_coef", c.sin_coef)?;
            cd.set_item("amplitude", c.amplitude)?;
            cd.set_item("phase", c.phase)?;
            Ok(cd)
        })
        .collect::<PyResult<_>>()?;
    d.set_item("components", comps)?;
    Ok(d)
}

/// Structural break detection (OLS-CUSUM) → dict(statistic, p_value, n,
/// breaks=[dict(index, time, statistic, p_value), ...]).
#[pyfunction]
#[pyo3(signature = (t, y, alpha=0.05, n_harmonics=0, period=1.0, min_segment=12))]
fn detect_breaks<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    n_harmonics: usize,
    period: f64,
    min_segment: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = stats::BreakOptions {
        alpha,
        n_harmonics,
        period,
        min_segment,
    };
    let r = stats::detect_breaks(t.as_slice()?, y.as_slice()?, &opts).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n", r.n)?;
    let breaks: Vec<Bound<'py, PyDict>> = r
        .breaks
        .iter()
        .map(|b| {
            let bd = PyDict::new(py);
            bd.set_item("index", b.index)?;
            bd.set_item("time", b.time)?;
            bd.set_item("statistic", b.statistic)?;
            bd.set_item("p_value", b.p_value)?;
            Ok(bd)
        })
        .collect::<PyResult<_>>()?;
    d.set_item("breaks", breaks)?;
    Ok(d)
}

/// A temporal data cube `(band, y, x, time)`.
///
/// Wraps `datacube_core::Cube`. Construct from a 4-D NumPy array, a 1-D time
/// array and a list of band names; missing values are `NaN`.
#[pyclass(name = "Cube")]
struct PyCube {
    inner: CoreCube,
}

#[pymethods]
impl PyCube {
    #[new]
    fn new(
        data: PyReadonlyArray4<'_, f64>,
        time: PyReadonlyArray1<'_, f64>,
        bands: Vec<String>,
    ) -> PyResult<Self> {
        let data: Array4<f64> = data.as_array().to_owned();
        let inner = CoreCube::new(data, time.as_slice()?.to_vec(), bands).map_err(err)?;
        Ok(Self { inner })
    }

    /// `(bands, height, width, time)`.
    #[getter]
    fn dims(&self) -> (usize, usize, usize, usize) {
        self.inner.dims()
    }

    #[getter]
    fn bands(&self) -> Vec<String> {
        self.inner.bands().to_vec()
    }

    #[getter]
    fn time<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.time().to_vec().into_pyarray(py)
    }

    /// The raw cube as a `(band, y, x, time)` NumPy array (copy).
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        self.inner.data().to_owned().into_pyarray(py)
    }

    /// EPSG code of the cube's grid, or `None` if it has no georeference.
    #[getter]
    fn epsg(&self) -> Option<u32> {
        self.inner.georef().and_then(|g| g.epsg)
    }

    /// Affine geotransform `[a, b, c, d, e, f]` (GDAL convention: `x = a +
    /// col*b + row*c`, `y = d + col*e + row*f`), or `None` if the cube has
    /// no georeference or no transform was set.
    #[getter]
    fn transform(&self) -> Option<[f64; 6]> {
        self.inner.georef().and_then(|g| g.transform)
    }

    /// Returns a copy of this cube with the given georeference attached
    /// (`epsg` and/or `transform`, either may be omitted/`None`).
    #[pyo3(signature = (epsg=None, transform=None))]
    fn with_georef(&self, epsg: Option<u32>, transform: Option<[f64; 6]>) -> Self {
        Self {
            inner: self.inner.clone().with_georef(GeoRef { epsg, transform }),
        }
    }

    /// Per-pixel trend maps for `band` (default 0): returns `(slope, p_value)`
    /// as two `(height, width)` NumPy arrays. `method` is "theil_sen"
    /// (slope + Mann-Kendall p) or "ols" (slope + t-test p).
    ///
    /// The GIL is released while the Rayon compute runs.
    #[pyo3(signature = (band=0, method="theil_sen"))]
    fn trend_map<'py>(
        &self,
        py: Python<'py>,
        band: usize,
        method: &str,
    ) -> PyResult<GridPair<'py>> {
        let inner = &self.inner;
        let grid = match method {
            "theil_sen" => py.detach(|| {
                inner.par_map_series(band, |t, y| {
                    let slope = stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN);
                    let p = stats::mann_kendall(y)
                        .map(|r| r.p_value)
                        .unwrap_or(f64::NAN);
                    (slope, p)
                })
            }),
            "ols" => py.detach(|| {
                inner.par_map_series(band, |t, y| {
                    stats::linear_trend(t, y)
                        .map(|r| (r.slope, r.p_value))
                        .unwrap_or((f64::NAN, f64::NAN))
                })
            }),
            other => return Err(PyValueError::new_err(format!("unknown method '{other}'"))),
        }
        .map_err(err)?;
        let slope = grid.mapv(|(s, _)| s);
        let pvalue = grid.mapv(|(_, p)| p);
        Ok((slope.into_pyarray(py), pvalue.into_pyarray(py)))
    }

    /// Aggregate time slices into composites. `window` is "same_time",
    /// "monthly" (calendar months), "yearly" (calendar years) or
    /// "period:<width>" (fixed bins in time units, anchored on the first
    /// observation); `method` is one of median, mean, min, max.
    #[pyo3(signature = (window="monthly", method="median"))]
    fn composite(&self, py: Python<'_>, window: &str, method: &str) -> PyResult<Self> {
        let win = match window {
            "same_time" => CompositeWindow::SameTime,
            "monthly" => CompositeWindow::CalendarMonth,
            "yearly" => CompositeWindow::CalendarYear,
            other => other
                .strip_prefix("period:")
                .and_then(|w| w.parse::<f64>().ok())
                .map(CompositeWindow::Period)
                .ok_or_else(|| PyValueError::new_err(format!("bad window '{other}'")))?,
        };
        let m = match method {
            "median" => CompositeMethod::Median,
            "mean" => CompositeMethod::Mean,
            "min" => CompositeMethod::Min,
            "max" => CompositeMethod::Max,
            other => return Err(PyKeyError::new_err(format!("bad method '{other}'"))),
        };
        let inner = &self.inner;
        Ok(Self {
            inner: py.detach(|| inner.composite(win, m)).map_err(err)?,
        })
    }

    /// Fill temporal NaN gaps by linear interpolation; gaps wider than
    /// `max_gap` time units (None = unlimited) and edges are left as NaN.
    #[pyo3(signature = (max_gap=None))]
    fn gapfill(&self, py: Python<'_>, max_gap: Option<f64>) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py.detach(|| inner.gapfill_linear(max_gap)).map_err(err)?,
        })
    }

    /// Index of the band labelled `name`.
    fn band_index(&self, name: &str) -> PyResult<usize> {
        self.inner.band(name).map_err(err)
    }

    /// Normalized difference `(a - b) / (a + b)` of two bands (given by label),
    /// returned as a new single-band cube. NaN where either input is NaN or the
    /// denominator is zero.
    #[pyo3(signature = (a, b, label="nd"))]
    fn normalized_difference(
        &self,
        py: Python<'_>,
        a: &str,
        b: &str,
        label: &str,
    ) -> PyResult<Self> {
        let (ai, bi) = (
            self.inner.band(a).map_err(err)?,
            self.inner.band(b).map_err(err)?,
        );
        let inner = &self.inner;
        Ok(Self {
            inner: py
                .detach(|| inner.normalized_difference(ai, bi, label))
                .map_err(err)?,
        })
    }

    /// NDVI = (NIR − Red) / (NIR + Red) as a new single-band cube.
    fn ndvi(&self, py: Python<'_>, nir: &str, red: &str) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py.detach(|| indices::ndvi(inner, nir, red)).map_err(err)?,
        })
    }

    /// NDWI = (Green − NIR) / (Green + NIR) (McFeeters).
    fn ndwi(&self, py: Python<'_>, green: &str, nir: &str) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py
                .detach(|| indices::ndwi(inner, green, nir))
                .map_err(err)?,
        })
    }

    /// NBR = (NIR − SWIR) / (NIR + SWIR).
    fn nbr(&self, py: Python<'_>, nir: &str, swir: &str) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py.detach(|| indices::nbr(inner, nir, swir)).map_err(err)?,
        })
    }

    /// EVI = 2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1).
    fn evi(&self, py: Python<'_>, nir: &str, red: &str, blue: &str) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py
                .detach(|| indices::evi(inner, nir, red, blue))
                .map_err(err)?,
        })
    }

    /// SAVI = (1 + L)·(NIR − Red) / (NIR + Red + L); `l` is soil brightness.
    #[pyo3(signature = (nir, red, l=0.5))]
    fn savi(&self, py: Python<'_>, nir: &str, red: &str, l: f64) -> PyResult<Self> {
        let inner = &self.inner;
        Ok(Self {
            inner: py
                .detach(|| indices::savi(inner, nir, red, l))
                .map_err(err)?,
        })
    }
}

/// Searches a STAC catalog and stacks the matching scenes into a [`Cube`]
/// (only built with the `stac` feature — needs the SurtGIS sibling checkout
/// and a system GDAL, so it's off by default and the wheel still builds
/// standalone otherwise; see `datacube-io::stack` for the full semantics of
/// each option).
///
/// Returns `dict(cube=Cube, scenes=[dict(id, datetime, time, cloud_cover),
/// ...], skipped=[str, ...])` — the same shape as `datacube stack`'s JSON
/// report, minus the fields that come from post-processing the CLI does
/// itself (composite/index/trend/breaks all live as `Cube` methods, so
/// chain them on `result["cube"]` instead of re-exposing them here).
#[cfg(feature = "stac")]
#[pyfunction]
#[pyo3(signature = (
    catalog, collection, assets, bbox, datetime,
    max_cloud_cover=None, max_items=100, overview=None,
    scale=1.0, offset=0.0, cross_zone_mosaic=true, concurrency=8,
    mask_scl=false, mask_asset="SCL", mask_keep=vec![4, 5, 6, 7, 11],
    grid_epsg=None, grid_res=None, grid_bbox=None, grid_align=None,
))]
#[allow(clippy::too_many_arguments)]
fn stack<'py>(
    py: Python<'py>,
    catalog: &str,
    collection: &str,
    assets: Vec<String>,
    bbox: [f64; 4],
    datetime: &str,
    max_cloud_cover: Option<f64>,
    max_items: usize,
    overview: Option<usize>,
    scale: f64,
    offset: f64,
    cross_zone_mosaic: bool,
    concurrency: usize,
    mask_scl: bool,
    mask_asset: &str,
    mask_keep: Vec<u16>,
    grid_epsg: Option<u32>,
    grid_res: Option<f64>,
    grid_bbox: Option<[f64; 4]>,
    grid_align: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let asset_refs: Vec<&str> = assets.iter().map(String::as_str).collect();
    let [w, s, e, n] = bbox;
    let mut cfg = datacube_io::StackConfig::new(catalog, collection, &asset_refs)
        .bbox(w, s, e, n)
        .datetime(datetime)
        .max_items(max_items)
        .overview(overview)
        .scaling(scale, offset)
        .cross_zone_mosaic(cross_zone_mosaic)
        .concurrency(concurrency);
    if let Some(pct) = max_cloud_cover {
        cfg = cfg.max_cloud_cover(pct);
    }
    if mask_scl {
        cfg = cfg.mask(datacube_io::MaskConfig {
            asset: mask_asset.to_string(),
            keep: mask_keep,
            resample: surtgis_core::ResampleMethod::NearestNeighbor,
        });
    }
    if let (Some(epsg), Some(res)) = (grid_epsg, grid_res) {
        let mut grid = datacube_io::GridSpec::new(epsg, res);
        if let Some([gw, gs, ge, gn]) = grid_bbox {
            grid = grid.bbox(gw, gs, ge, gn);
        }
        if let Some(step) = grid_align {
            grid = grid.align(step);
        }
        cfg = cfg.grid(grid);
    }

    let stacked = py.detach(|| datacube_io::stack(&cfg)).map_err(err)?;

    let d = PyDict::new(py);
    d.set_item(
        "cube",
        PyCube {
            inner: stacked.cube,
        },
    )?;
    let scenes: Vec<Bound<'py, PyDict>> = stacked
        .slices
        .iter()
        .map(|s| {
            let sd = PyDict::new(py);
            sd.set_item("id", &s.item_id)?;
            sd.set_item("datetime", &s.datetime)?;
            sd.set_item("time", s.time)?;
            sd.set_item("cloud_cover", s.cloud_cover)?;
            Ok(sd)
        })
        .collect::<PyResult<_>>()?;
    d.set_item("scenes", scenes)?;
    d.set_item("skipped", stacked.skipped)?;
    Ok(d)
}

#[pymodule]
fn datacube_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(linear_trend, m)?)?;
    m.add_function(wrap_pyfunction!(theil_sen, m)?)?;
    m.add_function(wrap_pyfunction!(mann_kendall, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(detect_breaks, m)?)?;
    m.add_class::<PyCube>()?;
    #[cfg(feature = "stac")]
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    Ok(())
}
