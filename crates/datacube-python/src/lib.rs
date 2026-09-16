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
use numpy::{
    PyArray1, PyArray2, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A pair of `(height, width)` NumPy grids (e.g. slope + p-value).
// The `numpy` crate resolves its own `ndarray` (any of 0.15..=0.17 satisfy
// its range), while this workspace is pinned to the `ndarray` that
// `datacube-core` and the `surtgis` path deps share. Only plain `Vec`s and
// shapes cross the boundary here, so whichever `ndarray` the lockfile ends
// up assigning to `numpy` cannot break the build.
fn array4_from_py(data: &PyReadonlyArray4<'_, f64>) -> PyResult<Array4<f64>> {
    let shape = data.shape();
    let dims = (shape[0], shape[1], shape[2], shape[3]);
    let flat: Vec<f64> = data.as_array().iter().copied().collect();
    Array4::from_shape_vec(dims, flat)
        .map_err(|e| PyValueError::new_err(format!("cannot build cube from array: {e}")))
}

fn array2_to_py<'py>(
    py: Python<'py>,
    a: &ndarray::Array2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ny, nx) = a.dim();
    PyArray1::from_vec(py, a.iter().copied().collect()).reshape([ny, nx])
}

fn array4_to_py<'py>(
    py: Python<'py>,
    a: ndarray::ArrayView4<'_, f64>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let (b, ny, nx, nt) = a.dim();
    PyArray1::from_vec(py, a.iter().copied().collect()).reshape([b, ny, nx, nt])
}

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

/// Seasonal Mann-Kendall test (Hirsch & Slack 1984), matching
/// `pymannkendall.seasonal_test` → dict(trend, s, var_s, z, tau, p_value, n).
#[pyfunction]
#[pyo3(signature = (y, period=12, alpha=0.05))]
fn seasonal_mann_kendall<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    period: usize,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::seasonal_mann_kendall(y.as_slice()?, period, alpha).map_err(err)?;
    mk_dict(py, &r)
}

/// Modified Mann-Kendall test with the Hamed & Rao (1998) autocorrelation
/// correction, matching `pymannkendall.hamed_rao_modification_test`. `lag`
/// limits the number of first lags considered (None = all).
#[pyfunction]
#[pyo3(signature = (y, alpha=0.05, lag=None))]
fn mann_kendall_hamed_rao<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    lag: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = stats::mann_kendall_hamed_rao(y.as_slice()?, alpha, lag).map_err(err)?;
    mk_dict(py, &r)
}

/// Shared dict builder for Mann-Kendall-family results.
fn mk_dict<'py>(py: Python<'py>, r: &stats::MannKendallResult) -> PyResult<Bound<'py, PyDict>> {
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

/// False-discovery-rate control over a p-value field (Benjamini-Hochberg or
/// Benjamini-Yekutieli), matching `statsmodels multipletests(method='fdr_bh' |
/// 'fdr_by')`. `method` is "bh"/"fdr_bh" (default) or "by"/"fdr_by".
///
/// Takes a 1-D array of p-values (flatten a map first); non-finite entries are
/// carried through as `NaN`/not-rejected. Returns dict(rejected: bool array,
/// adjusted: float array, n_tested, n_significant, threshold).
#[pyfunction]
#[pyo3(signature = (pvalues, q=0.05, method="bh"))]
fn fdr<'py>(
    py: Python<'py>,
    pvalues: PyReadonlyArray1<'py, f64>,
    q: f64,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let m = match method {
        "bh" | "fdr_bh" => stats::FdrMethod::BenjaminiHochberg,
        "by" | "fdr_by" => stats::FdrMethod::BenjaminiYekutieli,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown FDR method '{other}'"
            )));
        }
    };
    let r = stats::fdr(pvalues.as_slice()?, q, m).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("rejected", PyArray1::from_vec(py, r.rejected))?;
    d.set_item("adjusted", PyArray1::from_vec(py, r.adjusted))?;
    d.set_item("n_tested", r.n_tested)?;
    d.set_item("n_significant", r.n_significant)?;
    d.set_item("threshold", r.threshold)?;
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
        let data = array4_from_py(&data)?;
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
    fn time<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.time().to_vec())
    }

    /// The raw cube as a `(band, y, x, time)` NumPy array (copy).
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<f64>>> {
        array4_to_py(py, self.inner.data().view())
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
        Ok((array2_to_py(py, &slope)?, array2_to_py(py, &pvalue)?))
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

    /// Reduce each polygon of a vector layer (`.shp`/`.geojson`) to a tidy
    /// table, returned as a dict of equal-length columns (feed straight to
    /// `pandas.DataFrame`): `polygon_id`, `time`, `band`, `reducer`, `value`,
    /// `n_valid`, `n_total`.
    ///
    /// `reducer`: mean|median|min|max|std|sum|count|fraction_above (the last
    /// needs `threshold`). `inclusion`: center|all_touched|area_fraction.
    /// `window`: same_time|monthly|yearly|period:<w>. `source_epsg` overrides
    /// the vector's declared CRS. The cube must carry a georeference.
    /// (Only available when built with the `stac` feature.)
    #[cfg(feature = "stac")]
    #[pyo3(signature = (vector_path, id_field, reducer="mean", inclusion="center",
                        window="same_time", source_epsg=None, threshold=None))]
    #[allow(clippy::too_many_arguments)]
    fn zonal<'py>(
        &self,
        py: Python<'py>,
        vector_path: &str,
        id_field: &str,
        reducer: &str,
        inclusion: &str,
        window: &str,
        source_epsg: Option<u32>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        use datacube_io::{PixelInclusion, Reducer, ZonalConfig, read_zones, zonal_reduce};

        let incl = match inclusion {
            "center" => PixelInclusion::Center,
            "all_touched" => PixelInclusion::AllTouched,
            "area_fraction" => PixelInclusion::AreaFraction,
            other => return Err(PyValueError::new_err(format!("bad inclusion '{other}'"))),
        };
        let red = match reducer {
            "mean" => Reducer::Mean,
            "median" => Reducer::Median,
            "min" => Reducer::Min,
            "max" => Reducer::Max,
            "std" => Reducer::Std,
            "sum" => Reducer::Sum,
            "count" => Reducer::Count,
            "fraction_above" => Reducer::FractionAbove(threshold.ok_or_else(|| {
                PyValueError::new_err("reducer 'fraction_above' needs a threshold")
            })?),
            other => return Err(PyValueError::new_err(format!("bad reducer '{other}'"))),
        };
        let cfg = ZonalConfig::new(id_field, incl, red)
            .window(parse_window(window)?)
            .source_epsg(source_epsg);

        let features = read_zones(std::path::Path::new(vector_path)).map_err(err)?;
        let inner = &self.inner;
        let table = py
            .detach(|| zonal_reduce(inner, &features, &cfg))
            .map_err(err)?;

        let d = PyDict::new(py);
        d.set_item(
            "polygon_id",
            table
                .rows
                .iter()
                .map(|r| r.polygon_id.clone())
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "time",
            table.rows.iter().map(|r| r.time).collect::<Vec<f64>>(),
        )?;
        d.set_item(
            "band",
            table
                .rows
                .iter()
                .map(|r| r.band.clone())
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "reducer",
            table
                .rows
                .iter()
                .map(|r| r.reducer.to_string())
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "value",
            table.rows.iter().map(|r| r.value).collect::<Vec<f64>>(),
        )?;
        d.set_item(
            "n_valid",
            table.rows.iter().map(|r| r.n_valid).collect::<Vec<u64>>(),
        )?;
        d.set_item(
            "n_total",
            table.rows.iter().map(|r| r.n_total).collect::<Vec<u64>>(),
        )?;
        Ok(d)
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
    mask=None, mask_scl=false, mask_asset=None, mask_keep=vec![4, 5, 6, 7, 11],
    qa_reject_bits=None, qa_min_confidence=None,
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
    mask: Option<&str>,
    mask_scl: bool,
    mask_asset: Option<&str>,
    mask_keep: Vec<u16>,
    qa_reject_bits: Option<Vec<String>>,
    qa_min_confidence: Option<&str>,
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
    // `mask` ("scl" | "qa_pixel" | "auto") wins; `mask_scl=True` is the legacy
    // alias for `mask="scl"`.
    let mask_kind = mask.or(if mask_scl { Some("scl") } else { None });
    if let Some(kind) = mask_kind {
        let mc = build_mask_py(
            kind,
            collection,
            mask_asset,
            mask_keep,
            qa_reject_bits,
            qa_min_confidence,
        )?;
        cfg = cfg.mask(mc);
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

/// Resolves the Python masking kwargs into a [`datacube_io::MaskConfig`].
#[cfg(feature = "stac")]
fn build_mask_py(
    kind: &str,
    collection: &str,
    asset: Option<&str>,
    keep: Vec<u16>,
    qa_reject_bits: Option<Vec<String>>,
    qa_min_confidence: Option<&str>,
) -> PyResult<datacube_io::MaskConfig> {
    use datacube_io::MaskConfig;
    let nn = surtgis_core::ResampleMethod::NearestNeighbor;
    let mask = match kind.to_ascii_lowercase().as_str() {
        "scl" => MaskConfig::Scl {
            asset: asset.unwrap_or("SCL").to_string(),
            keep,
            resample: nn,
        },
        "qa_pixel" | "qa-pixel" => MaskConfig::QaBits {
            asset: asset.unwrap_or("QA_PIXEL").to_string(),
            reject: parse_qa_flags(qa_reject_bits.as_deref())?,
            min_confidence: parse_confidence(qa_min_confidence)?,
            resample: nn,
        },
        "auto" => {
            let mut m = MaskConfig::for_collection(collection).ok_or_else(|| {
                PyValueError::new_err(format!("mask='auto' has no default for '{collection}'"))
            })?;
            if let Some(a) = asset {
                match &mut m {
                    MaskConfig::Scl { asset, .. } | MaskConfig::QaBits { asset, .. } => {
                        *asset = a.to_string();
                    }
                }
            }
            m
        }
        other => return Err(PyValueError::new_err(format!("unknown mask '{other}'"))),
    };
    Ok(mask)
}

#[cfg(feature = "stac")]
fn parse_qa_flags(names: Option<&[String]>) -> PyResult<datacube_io::QaFlags> {
    use datacube_io::QaFlags;
    let Some(names) = names else {
        return Ok(QaFlags::default_reject());
    };
    let mut bits = 0u16;
    for name in names {
        bits |= match name.trim().to_ascii_lowercase().as_str() {
            "fill" => QaFlags::FILL,
            "dilated-cloud" | "dilated_cloud" => QaFlags::DILATED_CLOUD,
            "cirrus" => QaFlags::CIRRUS,
            "cloud" => QaFlags::CLOUD,
            "cloud-shadow" | "cloud_shadow" => QaFlags::CLOUD_SHADOW,
            "snow" => QaFlags::SNOW,
            "clear" => QaFlags::CLEAR,
            "water" => QaFlags::WATER,
            other => return Err(PyValueError::new_err(format!("unknown QA flag '{other}'"))),
        };
    }
    Ok(QaFlags(bits))
}

#[cfg(feature = "stac")]
fn parse_confidence(level: Option<&str>) -> PyResult<Option<datacube_io::Confidence>> {
    use datacube_io::Confidence;
    match level {
        None => Ok(None),
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "low" => Ok(Some(Confidence::Low)),
            "medium" => Ok(Some(Confidence::Medium)),
            "high" => Ok(Some(Confidence::High)),
            other => Err(PyValueError::new_err(format!(
                "unknown confidence '{other}'"
            ))),
        },
    }
}

/// Maps a window string ("same_time" | "monthly" | "yearly" | "period:<w>")
/// to a [`CompositeWindow`].
#[cfg(feature = "stac")]
fn parse_window(window: &str) -> PyResult<CompositeWindow> {
    match window {
        "same_time" => Ok(CompositeWindow::SameTime),
        "monthly" => Ok(CompositeWindow::CalendarMonth),
        "yearly" => Ok(CompositeWindow::CalendarYear),
        other => other
            .strip_prefix("period:")
            .and_then(|w| w.parse::<f64>().ok())
            .map(CompositeWindow::Period)
            .ok_or_else(|| PyValueError::new_err(format!("bad window '{other}'"))),
    }
}

#[pymodule]
fn datacube_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(linear_trend, m)?)?;
    m.add_function(wrap_pyfunction!(theil_sen, m)?)?;
    m.add_function(wrap_pyfunction!(mann_kendall, m)?)?;
    m.add_function(wrap_pyfunction!(seasonal_mann_kendall, m)?)?;
    m.add_function(wrap_pyfunction!(mann_kendall_hamed_rao, m)?)?;
    m.add_function(wrap_pyfunction!(fdr, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(detect_breaks, m)?)?;
    m.add_class::<PyCube>()?;
    #[cfg(feature = "stac")]
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    Ok(())
}
