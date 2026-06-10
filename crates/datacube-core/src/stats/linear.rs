use crate::error::CubeError;

use super::filter_finite_pairs;
use super::special::student_t_two_sided;

/// Ordinary least squares trend of a time series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearTrend {
    /// Slope in units of `y` per unit of `t`.
    pub slope: f64,
    pub intercept: f64,
    /// Coefficient of determination in `[0, 1]`.
    pub r_squared: f64,
    /// Standard error of the slope.
    pub std_err: f64,
    /// Two-sided p-value of the slope (t-test, `n - 2` degrees of freedom).
    pub p_value: f64,
    /// Number of finite observations used.
    pub n: usize,
}

/// Fits `y = intercept + slope * t` by ordinary least squares.
///
/// Non-finite pairs are dropped; at least 3 finite observations and a
/// non-constant `t` are required.
///
/// ```
/// let t: Vec<f64> = (0..10).map(f64::from).collect();
/// let y: Vec<f64> = t.iter().map(|x| 2.0 * x + 1.0).collect();
/// let fit = datacube_core::stats::linear_trend(&t, &y).unwrap();
/// assert!((fit.slope - 2.0).abs() < 1e-12);
/// assert!((fit.r_squared - 1.0).abs() < 1e-12);
/// ```
pub fn linear_trend(t: &[f64], y: &[f64]) -> Result<LinearTrend, CubeError> {
    let (t, y) = filter_finite_pairs(t, y)?;
    let n = t.len();
    if n < 3 {
        return Err(CubeError::InsufficientData { needed: 3, got: n });
    }
    let nf = n as f64;
    let mean_t = t.iter().sum::<f64>() / nf;
    let mean_y = y.iter().sum::<f64>() / nf;

    let mut ss_tt = 0.0;
    let mut ss_ty = 0.0;
    let mut ss_yy = 0.0;
    for (ti, yi) in t.iter().zip(&y) {
        let dt = ti - mean_t;
        let dy = yi - mean_y;
        ss_tt += dt * dt;
        ss_ty += dt * dy;
        ss_yy += dy * dy;
    }
    if ss_tt == 0.0 {
        return Err(CubeError::DimensionMismatch(
            "time coordinate is constant; slope is undefined".into(),
        ));
    }

    let slope = ss_ty / ss_tt;
    let intercept = mean_y - slope * mean_t;
    let ss_res = (ss_yy - slope * ss_ty).max(0.0);
    let r_squared = if ss_yy == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_yy
    };

    let df = nf - 2.0;
    let var_res = ss_res / df;
    let std_err = (var_res / ss_tt).sqrt();
    let p_value = if std_err == 0.0 {
        // perfect fit: the slope is exact
        if slope == 0.0 { 1.0 } else { 0.0 }
    } else {
        student_t_two_sided(slope / std_err, df)
    };

    Ok(LinearTrend {
        slope,
        intercept,
        r_squared,
        std_err,
        p_value,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn exact_line() {
        let t: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = t.iter().map(|x| -0.5 * x + 3.0).collect();
        let fit = linear_trend(&t, &y).unwrap();
        assert_abs_diff_eq!(fit.slope, -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.intercept, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.r_squared, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.p_value, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn known_noisy_fit() {
        // y = [1, 3, 2, 5, 4] over t = 0..5: slope = 0.8, intercept = 1.4
        let t: Vec<f64> = (0..5).map(f64::from).collect();
        let y = [1.0, 3.0, 2.0, 5.0, 4.0];
        let fit = linear_trend(&t, &y).unwrap();
        assert_abs_diff_eq!(fit.slope, 0.8, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.intercept, 1.4, epsilon = 1e-12);
        // scipy.stats.linregress: r = 0.8, p = 0.10408803866182788
        assert_abs_diff_eq!(fit.r_squared, 0.64, epsilon = 1e-12);
        assert_abs_diff_eq!(fit.p_value, 0.104088, epsilon = 1e-5);
    }

    #[test]
    fn nan_pairs_dropped() {
        let t: Vec<f64> = (0..6).map(f64::from).collect();
        let y = [0.0, 2.0, f64::NAN, 6.0, 8.0, f64::NAN];
        let fit = linear_trend(&t, &y).unwrap();
        assert_eq!(fit.n, 4);
        assert_abs_diff_eq!(fit.slope, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn too_few_points() {
        assert!(matches!(
            linear_trend(&[0.0, 1.0], &[1.0, 2.0]),
            Err(CubeError::InsufficientData { needed: 3, got: 2 })
        ));
    }
}
