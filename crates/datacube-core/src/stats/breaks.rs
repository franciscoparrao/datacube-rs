//! BFAST-style structural break detection.
//!
//! Fits a trend (+ optional harmonic season) model per segment, tests its
//! OLS residuals with the OLS-CUSUM statistic (`sup |W|`, asymptotically the
//! supremum of a Brownian bridge — the same test as
//! `statsmodels.stats.diagnostic.breaks_cusumolsresid`), and locates breaks
//! by recursive binary segmentation.

use crate::error::CubeError;

use super::filter_finite_pairs;
use super::lstsq::HarmonicModel;

/// Options for [`detect_breaks`].
#[derive(Debug, Clone, Copy)]
pub struct BreakOptions {
    /// Significance level for the CUSUM test at each segmentation step.
    pub alpha: f64,
    /// Harmonic pairs in the segment model (`0` = trend-only).
    pub n_harmonics: usize,
    /// Season length in the units of `t` (used when `n_harmonics > 0`).
    pub period: f64,
    /// Minimum observations per segment (each side of a break).
    pub min_segment: usize,
}

impl Default for BreakOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            n_harmonics: 0,
            period: 1.0,
            min_segment: 12,
        }
    }
}

/// One detected break.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreakPoint {
    /// Index (into the NaN-filtered series) of the last observation of the
    /// left segment; the regime change happens after this observation.
    pub index: usize,
    /// Time coordinate of that observation.
    pub time: f64,
    /// `sup |W|` of the segment whose split produced this break.
    pub statistic: f64,
    /// Brownian-bridge p-value of that statistic.
    pub p_value: f64,
}

/// Result of [`detect_breaks`].
#[derive(Debug, Clone)]
pub struct BreakResult {
    /// `sup |W|` of the OLS-CUSUM process over the full series.
    pub statistic: f64,
    /// Brownian-bridge p-value of the full-series statistic.
    pub p_value: f64,
    /// Breaks in time order (empty if the series is stable).
    pub breaks: Vec<BreakPoint>,
    /// Number of finite observations used.
    pub n: usize,
}

/// Detects structural breaks in a time series.
///
/// Non-finite pairs are dropped; `t` must be ascending. Within each segment
/// a least-squares model `intercept + slope·t (+ K Fourier pairs)` is fitted
/// and its residuals tested with OLS-CUSUM; significant segments are split
/// at the CUSUM maximum and both halves are re-tested (binary segmentation)
/// while segments keep at least `min_segment` observations.
///
/// ```
/// use datacube_core::stats::{BreakOptions, detect_breaks};
///
/// // stable left regime, abrupt level shift after index 29
/// let t: Vec<f64> = (0..60).map(f64::from).collect();
/// let y: Vec<f64> = t.iter().map(|&t| if t < 30.0 { 1.0 } else { 6.0 }
///     + 0.05 * (t * 12.9898).sin()).collect();
/// let r = detect_breaks(&t, &y, &BreakOptions::default()).unwrap();
/// assert_eq!(r.breaks.len(), 1);
/// assert!((r.breaks[0].index as i64 - 29).abs() <= 1);
/// ```
pub fn detect_breaks(t: &[f64], y: &[f64], opts: &BreakOptions) -> Result<BreakResult, CubeError> {
    if !(0.0..1.0).contains(&opts.alpha) {
        return Err(CubeError::InvalidParameter(format!(
            "alpha must be in (0, 1), got {}",
            opts.alpha
        )));
    }
    let nparams = 2 + 2 * opts.n_harmonics;
    if opts.min_segment < nparams + 2 {
        return Err(CubeError::InvalidParameter(format!(
            "min_segment must be >= nparams + 2 = {}, got {}",
            nparams + 2,
            opts.min_segment
        )));
    }
    if opts.n_harmonics > 0 && (!opts.period.is_finite() || opts.period <= 0.0) {
        return Err(CubeError::InvalidParameter(format!(
            "period must be finite and > 0, got {}",
            opts.period
        )));
    }
    let (t, y) = filter_finite_pairs(t, y)?;
    let n = t.len();
    if n < opts.min_segment {
        return Err(CubeError::InsufficientData {
            needed: opts.min_segment,
            got: n,
        });
    }
    if t.windows(2).any(|w| w[1] < w[0]) {
        return Err(CubeError::DimensionMismatch(
            "detect_breaks requires an ascending time axis".into(),
        ));
    }

    let (statistic, p_value, _) = cusum_test(&t, &y, opts)?;
    let mut breaks = Vec::new();
    segment(&t, &y, 0, n, opts, &mut breaks)?;
    breaks.sort_by_key(|b| b.index);

    Ok(BreakResult {
        statistic,
        p_value,
        breaks,
        n,
    })
}

/// Recursively tests `[lo, hi)` and records significant splits.
fn segment(
    t: &[f64],
    y: &[f64],
    lo: usize,
    hi: usize,
    opts: &BreakOptions,
    out: &mut Vec<BreakPoint>,
) -> Result<(), CubeError> {
    let len = hi - lo;
    if len < 2 * opts.min_segment {
        return Ok(()); // no room for two valid segments
    }
    let (stat, p, w) = cusum_test(&t[lo..hi], &y[lo..hi], opts)?;
    if p >= opts.alpha {
        return Ok(());
    }
    // split at the CUSUM maximum, constrained so both sides stay valid
    let k_lo = opts.min_segment - 1;
    let k_hi = len - opts.min_segment - 1;
    let mut k_best = k_lo;
    for k in k_lo..=k_hi {
        if w[k].abs() > w[k_best].abs() {
            k_best = k;
        }
    }
    out.push(BreakPoint {
        index: lo + k_best,
        time: t[lo + k_best],
        statistic: stat,
        p_value: p,
    });
    segment(t, y, lo, lo + k_best + 1, opts, out)?;
    segment(t, y, lo + k_best + 1, hi, opts, out)
}

/// OLS-CUSUM test of one segment: fits the segment model, returns
/// `(sup |W|, p-value, W)` where `W_k = Σ_{i<=k} e_i / (σ̂ √n)` and
/// `σ̂² = SSE / (n − nparams)`.
fn cusum_test(
    t: &[f64],
    y: &[f64],
    opts: &BreakOptions,
) -> Result<(f64, f64, Vec<f64>), CubeError> {
    let residuals = fit_residuals(t, y, opts)?;
    let n = residuals.len();
    let nparams = 2 + 2 * opts.n_harmonics;
    let sse: f64 = residuals.iter().map(|e| e * e).sum();
    if sse <= 0.0 {
        // perfect fit: no evidence of instability
        return Ok((0.0, 1.0, vec![0.0; n]));
    }
    let sigma = (sse / (n - nparams) as f64).sqrt();
    let scale = sigma * (n as f64).sqrt();

    let mut w = Vec::with_capacity(n);
    let mut cum = 0.0;
    let mut sup = 0.0_f64;
    for e in &residuals {
        cum += e;
        let wk = cum / scale;
        sup = sup.max(wk.abs());
        w.push(wk);
    }
    Ok((sup, sup_brownian_bridge_pvalue(sup), w))
}

/// Least-squares residuals of the segment model
/// `intercept + slope·(t − t̄) + K Fourier pairs`.
fn fit_residuals(t: &[f64], y: &[f64], opts: &BreakOptions) -> Result<Vec<f64>, CubeError> {
    let model = HarmonicModel::fit(t, y, opts.n_harmonics, opts.period)?;
    Ok(t.iter()
        .zip(y)
        .map(|(ti, yi)| yi - model.predict(*ti))
        .collect())
}

/// `P(sup |BB| > s)` for a Brownian bridge: `2 Σ_{j≥1} (−1)^{j+1} e^{−2 j² s²}`.
fn sup_brownian_bridge_pvalue(s: f64) -> f64 {
    if s <= 0.0 {
        return 1.0;
    }
    let mut p = 0.0;
    let mut sign = 1.0;
    for j in 1..=100 {
        let term = (-2.0 * (j as f64) * (j as f64) * s * s).exp();
        p += 2.0 * sign * term;
        sign = -sign;
        if term < 1e-18 {
            break;
        }
    }
    p.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Deterministic pseudo-noise in roughly [-0.5, 0.5].
    fn noise(i: usize) -> f64 {
        let x = (i as f64 * 12.9898).sin() * 43758.5453;
        x - x.round()
    }

    #[test]
    fn brownian_bridge_pvalue_known_values() {
        // 1.358 is the classic 5% critical value of sup |BB|
        assert_abs_diff_eq!(sup_brownian_bridge_pvalue(1.358), 0.0502, epsilon = 1e-3);
        assert!(sup_brownian_bridge_pvalue(0.5) > 0.95);
        assert!(sup_brownian_bridge_pvalue(3.0) < 1e-6);
        assert_eq!(sup_brownian_bridge_pvalue(0.0), 1.0);
    }

    #[test]
    fn stable_series_has_no_breaks() {
        let t: Vec<f64> = (0..80).map(f64::from).collect();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| 2.0 + 0.05 * t + 0.3 * noise(i))
            .collect();
        let r = detect_breaks(&t, &y, &BreakOptions::default()).unwrap();
        assert!(r.breaks.is_empty(), "false break: {:?}", r.breaks);
        assert!(r.p_value >= 0.05);
    }

    #[test]
    fn level_shift_is_located() {
        let t: Vec<f64> = (0..60).map(f64::from).collect();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| if t < 30.0 { 1.0 } else { 6.0 } + 0.2 * noise(i))
            .collect();
        let r = detect_breaks(&t, &y, &BreakOptions::default()).unwrap();
        assert_eq!(r.breaks.len(), 1, "breaks: {:?}", r.breaks);
        assert!((r.breaks[0].index as i64 - 29).unsigned_abs() <= 1);
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn two_breaks_via_binary_segmentation() {
        let t: Vec<f64> = (0..120).map(f64::from).collect();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let level = if t < 40.0 {
                    0.0
                } else if t < 80.0 {
                    5.0
                } else {
                    -3.0
                };
                level + 0.2 * noise(i)
            })
            .collect();
        let r = detect_breaks(&t, &y, &BreakOptions::default()).unwrap();
        assert_eq!(r.breaks.len(), 2, "breaks: {:?}", r.breaks);
        assert!((r.breaks[0].index as i64 - 39).unsigned_abs() <= 1);
        assert!((r.breaks[1].index as i64 - 79).unsigned_abs() <= 1);
    }

    #[test]
    fn seasonal_series_needs_harmonics() {
        use core::f64::consts::PI;
        // monthly data, strong annual cycle, level shift at i = 36
        let t: Vec<f64> = (0..72).map(|m| m as f64 / 12.0).collect();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let shift = if i >= 36 { 2.0 } else { 0.0 };
                1.0 + 0.8 * (2.0 * PI * t).sin() + shift + 0.05 * noise(i)
            })
            .collect();
        let opts = BreakOptions {
            n_harmonics: 1,
            period: 1.0,
            ..BreakOptions::default()
        };
        let r = detect_breaks(&t, &y, &opts).unwrap();
        assert_eq!(r.breaks.len(), 1, "breaks: {:?}", r.breaks);
        assert!((r.breaks[0].index as i64 - 35).unsigned_abs() <= 1);
    }

    #[test]
    fn nan_filtered_and_options_validated() {
        let t: Vec<f64> = (0..40).map(f64::from).collect();
        let mut y: Vec<f64> = t
            .iter()
            .map(|&t| if t < 20.0 { 0.0 } else { 4.0 })
            .collect();
        y[5] = f64::NAN;
        let r = detect_breaks(&t, &y, &BreakOptions::default()).unwrap();
        assert_eq!(r.n, 39);
        assert_eq!(r.breaks.len(), 1);

        let bad_alpha = BreakOptions {
            alpha: 1.5,
            ..BreakOptions::default()
        };
        assert!(detect_breaks(&t, &y, &bad_alpha).is_err());
        let bad_min = BreakOptions {
            min_segment: 2,
            ..BreakOptions::default()
        };
        assert!(detect_breaks(&t, &y, &bad_min).is_err());
    }
}
