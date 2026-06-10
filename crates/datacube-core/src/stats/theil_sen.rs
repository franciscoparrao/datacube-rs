use crate::error::CubeError;

use super::{filter_finite_pairs, median_mut};

/// Theil-Sen robust slope estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TheilSenResult {
    /// Median of all pairwise slopes, in units of `y` per unit of `t`.
    pub slope: f64,
    /// `median(y) - slope * median(t)`, as in `pyMannKendall.sens_slope`.
    pub intercept: f64,
    /// Number of finite observations used.
    pub n: usize,
}

/// Theil-Sen estimator: median of the slopes of all point pairs.
///
/// Robust to ~29% of outliers, the standard companion of the Mann-Kendall
/// test for monotonic trends. Pairs with identical `t` are skipped; at least
/// 2 finite observations with distinct `t` are required. O(n²) pairs.
///
/// ```
/// let t: Vec<f64> = (0..10).map(f64::from).collect();
/// let mut y: Vec<f64> = t.iter().map(|x| 2.0 * x + 1.0).collect();
/// y[4] = 100.0; // outlier does not move the median slope
/// let fit = datacube_core::stats::theil_sen(&t, &y).unwrap();
/// assert!((fit.slope - 2.0).abs() < 1e-12);
/// ```
pub fn theil_sen(t: &[f64], y: &[f64]) -> Result<TheilSenResult, CubeError> {
    let (mut t, mut y) = filter_finite_pairs(t, y)?;
    let n = t.len();
    if n < 2 {
        return Err(CubeError::InsufficientData { needed: 2, got: n });
    }

    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dt = t[j] - t[i];
            if dt != 0.0 {
                slopes.push((y[j] - y[i]) / dt);
            }
        }
    }
    if slopes.is_empty() {
        return Err(CubeError::DimensionMismatch(
            "time coordinate is constant; slope is undefined".into(),
        ));
    }

    let slope = median_mut(&mut slopes);
    let intercept = median_mut(&mut y) - slope * median_mut(&mut t);
    Ok(TheilSenResult {
        slope,
        intercept,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn exact_line() {
        let t: Vec<f64> = (0..15).map(f64::from).collect();
        let y: Vec<f64> = t.iter().map(|x| 3.0 * x - 2.0).collect();
        let fit = theil_sen(&t, &y).unwrap();
        assert_abs_diff_eq!(fit.slope, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.intercept, -2.0, epsilon = 1e-12);
    }

    #[test]
    fn robust_to_outliers() {
        let t: Vec<f64> = (0..21).map(f64::from).collect();
        let mut y: Vec<f64> = t.iter().map(|x| 0.5 * x).collect();
        y[3] = 1000.0;
        y[17] = -1000.0;
        let fit = theil_sen(&t, &y).unwrap();
        assert_abs_diff_eq!(fit.slope, 0.5, epsilon = 1e-9);
    }

    #[test]
    fn irregular_sampling() {
        // gaps in t must use real time distances, not indices
        let t = [0.0, 1.0, 2.0, 10.0];
        let y = [0.0, 2.0, 4.0, 20.0]; // exactly y = 2t
        let fit = theil_sen(&t, &y).unwrap();
        assert_abs_diff_eq!(fit.slope, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn nan_dropped_and_n_reported() {
        let t = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, f64::NAN, 4.0, 6.0];
        let fit = theil_sen(&t, &y).unwrap();
        assert_eq!(fit.n, 3);
        assert_abs_diff_eq!(fit.slope, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn constant_time_is_error() {
        assert!(theil_sen(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
    }
}
