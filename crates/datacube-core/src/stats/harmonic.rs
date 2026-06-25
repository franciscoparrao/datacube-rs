use crate::error::CubeError;

use super::filter_finite_pairs;
use super::lstsq::HarmonicModel;

/// One fitted harmonic term `a·cos(ωkt) + b·sin(ωkt)`, also expressed as
/// amplitude/phase: `A·cos(ωkt − φ)` with `A = √(a² + b²)`, `φ = atan2(b, a)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicComponent {
    /// Harmonic order `k` (1 = the base period).
    pub harmonic: usize,
    pub cos_coef: f64,
    pub sin_coef: f64,
    pub amplitude: f64,
    /// Phase in radians within `(−π, π]`; the k-th harmonic peaks at
    /// `t = period · φ / (2πk)` (mod `period / k`).
    pub phase: f64,
}

/// Least-squares fit of `y = intercept + slope·t + Σ_k a_k cos(2πkt/P) + b_k sin(2πkt/P)`.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicFit {
    pub intercept: f64,
    /// Linear trend in units of `y` per unit of `t`.
    pub slope: f64,
    /// One entry per harmonic, ordered `k = 1..=n_harmonics`.
    pub components: Vec<HarmonicComponent>,
    /// Base period `P`, in the same units as `t`.
    pub period: f64,
    pub r_squared: f64,
    /// Root mean squared residual, `√(ss_res / n)`.
    pub rmse: f64,
    /// Number of finite observations used.
    pub n: usize,
}

impl HarmonicFit {
    /// Evaluates the fitted model at time `t`.
    pub fn predict(&self, t: f64) -> f64 {
        let omega = 2.0 * core::f64::consts::PI / self.period;
        let mut y = self.intercept + self.slope * t;
        for c in &self.components {
            let kt = omega * c.harmonic as f64 * t;
            y += c.cos_coef * kt.cos() + c.sin_coef * kt.sin();
        }
        y
    }
}

/// Harmonic (Fourier) regression with linear trend, the standard model for
/// seasonality and phenology in remote sensing time series (the season model
/// used by BFAST and CCDC, Zhu & Woodcock 2014).
///
/// `period` is the length of one season in the units of `t` (e.g. `1.0` for
/// fractional years, `365.25` for days); `n_harmonics` is the number of
/// Fourier pairs (2 is typical for vegetation phenology). Non-finite pairs
/// are dropped; at least `2·n_harmonics + 3` finite observations are
/// required. The normal equations are solved with the linear term centered
/// for conditioning.
///
/// ```
/// use std::f64::consts::PI;
///
/// // monthly NDVI-like series: trend + one annual harmonic
/// let t: Vec<f64> = (0..48).map(|m| m as f64 / 12.0).collect();
/// let y: Vec<f64> = t.iter()
///     .map(|&t| 0.5 + 0.01 * t + 0.2 * (2.0 * PI * t).cos())
///     .collect();
/// let fit = datacube_core::stats::harmonic_regression(&t, &y, 1.0, 1).unwrap();
/// assert!((fit.slope - 0.01).abs() < 1e-9);
/// assert!((fit.components[0].amplitude - 0.2).abs() < 1e-9);
/// ```
pub fn harmonic_regression(
    t: &[f64],
    y: &[f64],
    period: f64,
    n_harmonics: usize,
) -> Result<HarmonicFit, CubeError> {
    if !period.is_finite() || period <= 0.0 {
        return Err(CubeError::InvalidParameter(format!(
            "period must be finite and > 0, got {period}"
        )));
    }
    if n_harmonics == 0 {
        return Err(CubeError::InvalidParameter(
            "n_harmonics must be >= 1 (use linear_trend for a pure trend)".into(),
        ));
    }
    let (t, y) = filter_finite_pairs(t, y)?;
    let n = t.len();
    let p = 2 + 2 * n_harmonics;
    if n < p + 1 {
        return Err(CubeError::InsufficientData {
            needed: p + 1,
            got: n,
        });
    }

    let model = HarmonicModel::fit(&t, &y, n_harmonics, period)?;
    let beta = &model.beta;

    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_yy = 0.0;
    for (ti, yi) in t.iter().zip(&y) {
        let pred = model.predict(*ti);
        ss_res += (yi - pred) * (yi - pred);
        ss_yy += (yi - mean_y) * (yi - mean_y);
    }
    let r_squared = if ss_yy == 0.0 {
        1.0
    } else {
        (1.0 - ss_res / ss_yy).max(0.0)
    };

    let slope = beta[1];
    let intercept = beta[0] - slope * model.mean_t; // undo the centering
    let components = (1..=n_harmonics)
        .map(|k| {
            let (a, b) = (beta[2 * k], beta[2 * k + 1]);
            HarmonicComponent {
                harmonic: k,
                cos_coef: a,
                sin_coef: b,
                amplitude: a.hypot(b),
                phase: b.atan2(a),
            }
        })
        .collect();

    Ok(HarmonicFit {
        intercept,
        slope,
        components,
        period,
        r_squared,
        rmse: (ss_res / n as f64).sqrt(),
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use core::f64::consts::PI;

    fn monthly_t(n: usize) -> Vec<f64> {
        (0..n).map(|m| m as f64 / 12.0).collect()
    }

    #[test]
    fn recovers_exact_single_harmonic() {
        let t = monthly_t(48);
        let y: Vec<f64> = t
            .iter()
            .map(|&t| 2.0 + 0.1 * t + 1.5 * (2.0 * PI * t).cos() + 0.8 * (2.0 * PI * t).sin())
            .collect();
        let fit = harmonic_regression(&t, &y, 1.0, 1).unwrap();
        assert_abs_diff_eq!(fit.intercept, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.slope, 0.1, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[0].cos_coef, 1.5, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[0].sin_coef, 0.8, epsilon = 1e-9);
        assert_abs_diff_eq!(
            fit.components[0].amplitude,
            (1.5f64.powi(2) + 0.8f64.powi(2)).sqrt(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(fit.components[0].phase, 0.8f64.atan2(1.5), epsilon = 1e-9);
        assert_abs_diff_eq!(fit.r_squared, 1.0, epsilon = 1e-12);
        assert!(fit.rmse < 1e-9);
    }

    #[test]
    fn recovers_two_harmonics() {
        let t = monthly_t(60);
        let y: Vec<f64> = t
            .iter()
            .map(|&t| {
                1.0 - 0.05 * t + 0.6 * (2.0 * PI * t).sin() + 0.25 * (4.0 * PI * t).cos()
                    - 0.1 * (4.0 * PI * t).sin()
            })
            .collect();
        let fit = harmonic_regression(&t, &y, 1.0, 2).unwrap();
        assert_abs_diff_eq!(fit.slope, -0.05, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[0].sin_coef, 0.6, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[0].cos_coef, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[1].cos_coef, 0.25, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.components[1].sin_coef, -0.1, epsilon = 1e-9);
    }

    #[test]
    fn predict_matches_model() {
        let t = monthly_t(36);
        let f = |t: f64| 0.4 + 0.02 * t + 0.3 * (2.0 * PI * t - 1.0).cos();
        let y: Vec<f64> = t.iter().map(|&t| f(t)).collect();
        let fit = harmonic_regression(&t, &y, 1.0, 1).unwrap();
        for &ti in &[0.0, 0.31, 1.7, 5.0] {
            assert_abs_diff_eq!(fit.predict(ti), f(ti), epsilon = 1e-9);
        }
    }

    #[test]
    fn nan_dropped() {
        let t = monthly_t(40);
        let mut y: Vec<f64> = t.iter().map(|&t| 1.0 + (2.0 * PI * t).sin()).collect();
        y[5] = f64::NAN;
        y[20] = f64::NAN;
        let fit = harmonic_regression(&t, &y, 1.0, 1).unwrap();
        assert_eq!(fit.n, 38);
        assert_abs_diff_eq!(fit.components[0].sin_coef, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn aliased_harmonic_is_singular() {
        // annual sampling of an annual cycle: cos column is constant, sin is 0
        let t: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = t.iter().map(|&t| 1.0 + 0.1 * t).collect();
        assert!(matches!(
            harmonic_regression(&t, &y, 1.0, 1),
            Err(CubeError::SingularSystem(_))
        ));
    }

    #[test]
    fn parameter_validation() {
        let t = monthly_t(24);
        let y = vec![1.0; 24];
        assert!(matches!(
            harmonic_regression(&t, &y, 0.0, 1),
            Err(CubeError::InvalidParameter(_))
        ));
        assert!(matches!(
            harmonic_regression(&t, &y, 1.0, 0),
            Err(CubeError::InvalidParameter(_))
        ));
        assert!(matches!(
            harmonic_regression(&t[..5], &y[..5], 1.0, 2),
            Err(CubeError::InsufficientData { needed: 7, got: 5 })
        ));
    }
}
