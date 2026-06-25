//! Python bindings for datacube-rs.
//!
//! Exposes the per-series statistics and the temporal `Cube` of
//! `datacube-core` to Python, with NumPy array interop. Built as the
//! `datacube_rs` extension module (see the `python/` package and
//! `maturin develop`).

use datacube_core::{CompositeMethod, CompositeWindow, Cube as CoreCube, stats};
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

    /// Per-pixel trend maps for `band` (default 0): returns `(slope, p_value)`
    /// as two `(height, width)` NumPy arrays. `method` is "theil_sen"
    /// (slope + Mann-Kendall p) or "ols" (slope + t-test p).
    #[pyo3(signature = (band=0, method="theil_sen"))]
    fn trend_map<'py>(
        &self,
        py: Python<'py>,
        band: usize,
        method: &str,
    ) -> PyResult<GridPair<'py>> {
        let grid = match method {
            "theil_sen" => self.inner.par_map_series(band, |t, y| {
                let slope = stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN);
                let p = stats::mann_kendall(y)
                    .map(|r| r.p_value)
                    .unwrap_or(f64::NAN);
                (slope, p)
            }),
            "ols" => self.inner.par_map_series(band, |t, y| {
                stats::linear_trend(t, y)
                    .map(|r| (r.slope, r.p_value))
                    .unwrap_or((f64::NAN, f64::NAN))
            }),
            other => return Err(PyValueError::new_err(format!("unknown method '{other}'"))),
        }
        .map_err(err)?;
        let slope = grid.mapv(|(s, _)| s);
        let pvalue = grid.mapv(|(_, p)| p);
        Ok((slope.into_pyarray(py), pvalue.into_pyarray(py)))
    }

    /// Aggregate time slices into composites. `window` is "same_time" or
    /// "monthly" (or "period:<width>" in time units); `method` is one of
    /// median, mean, min, max.
    #[pyo3(signature = (window="monthly", method="median"))]
    fn composite(&self, window: &str, method: &str) -> PyResult<Self> {
        let win = match window {
            "same_time" => CompositeWindow::SameTime,
            "monthly" => CompositeWindow::Period(1.0 / 12.0),
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
        Ok(Self {
            inner: self.inner.composite(win, m).map_err(err)?,
        })
    }

    /// Fill temporal NaN gaps by linear interpolation; gaps wider than
    /// `max_gap` time units (None = unlimited) and edges are left as NaN.
    #[pyo3(signature = (max_gap=None))]
    fn gapfill(&self, max_gap: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.gapfill_linear(max_gap).map_err(err)?,
        })
    }
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
    Ok(())
}
