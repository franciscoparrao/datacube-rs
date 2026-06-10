use crate::error::CubeError;

use super::special::norm_sf;

/// Direction of a monotonic trend at the chosen significance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Increasing,
    Decreasing,
    NoTrend,
}

/// Result of the (original, tie-corrected) Mann-Kendall test, matching the
/// fields reported by `pyMannKendall.original_test`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MannKendallResult {
    pub trend: Trend,
    /// Mann-Kendall statistic `S = Σ sign(x_j − x_i)` over all pairs `j > i`.
    pub s: f64,
    /// Tie-corrected variance of `S`.
    pub var_s: f64,
    /// Continuity-corrected normal score.
    pub z: f64,
    /// Kendall's tau, `S / (n(n−1)/2)`.
    pub tau: f64,
    /// Two-sided p-value from the normal approximation.
    pub p_value: f64,
    /// Number of finite observations used.
    pub n: usize,
}

/// Mann-Kendall trend test with the conventional `alpha = 0.05`.
///
/// Non-finite values are dropped; at least 3 finite observations are
/// required (the normal approximation is recommended for `n >= 10`).
///
/// ```
/// use datacube_core::stats::{mann_kendall, Trend};
///
/// let y: Vec<f64> = (0..10).map(f64::from).collect();
/// let mk = mann_kendall(&y).unwrap();
/// assert_eq!(mk.trend, Trend::Increasing);
/// assert_eq!(mk.tau, 1.0);
/// ```
pub fn mann_kendall(values: &[f64]) -> Result<MannKendallResult, CubeError> {
    mann_kendall_alpha(values, 0.05)
}

/// Mann-Kendall trend test at significance level `alpha`.
///
/// Implements `pyMannKendall.original_test`: tie-corrected variance
/// `var(S) = [n(n−1)(2n+5) − Σ_t t(t−1)(2t+5)] / 18` and continuity-corrected
/// `z = (S ∓ 1)/√var(S)`.
pub fn mann_kendall_alpha(values: &[f64], alpha: f64) -> Result<MannKendallResult, CubeError> {
    let y: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = y.len();
    if n < 3 {
        return Err(CubeError::InsufficientData { needed: 3, got: n });
    }
    let nf = n as f64;

    // note: f64::signum(0.0) == 1.0, so ties need an explicit zero branch
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            s += match y[j].partial_cmp(&y[i]) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Less) => -1.0,
                _ => 0.0,
            };
        }
    }

    // tie correction over groups of equal values
    let mut sorted = y.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let mut tie_term = 0.0;
    let mut run = 1.0_f64;
    for k in 1..=n {
        if k < n && sorted[k] == sorted[k - 1] {
            run += 1.0;
        } else {
            if run > 1.0 {
                tie_term += run * (run - 1.0) * (2.0 * run + 5.0);
            }
            run = 1.0;
        }
    }
    let var_s = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - tie_term) / 18.0;

    let z = if s > 0.0 {
        (s - 1.0) / var_s.sqrt()
    } else if s < 0.0 {
        (s + 1.0) / var_s.sqrt()
    } else {
        0.0
    };

    let p_value = 2.0 * norm_sf(z.abs());
    let tau = s / (0.5 * nf * (nf - 1.0));
    let trend = if p_value < alpha {
        if z > 0.0 {
            Trend::Increasing
        } else {
            Trend::Decreasing
        }
    } else {
        Trend::NoTrend
    };

    Ok(MannKendallResult {
        trend,
        s,
        var_s,
        z,
        tau,
        p_value,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn strictly_increasing() {
        let y: Vec<f64> = (0..10).map(f64::from).collect();
        let mk = mann_kendall(&y).unwrap();
        assert_eq!(mk.s, 45.0);
        // var = 10*9*25/18 = 125, z = 44/sqrt(125)
        assert_abs_diff_eq!(mk.var_s, 125.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mk.z, 44.0 / 125.0_f64.sqrt(), epsilon = 1e-12);
        assert_eq!(mk.tau, 1.0);
        assert_eq!(mk.trend, Trend::Increasing);
        assert!(mk.p_value < 1e-4 && mk.p_value > 1e-5);
    }

    #[test]
    fn strictly_decreasing_is_symmetric() {
        let up: Vec<f64> = (0..10).map(f64::from).collect();
        let down: Vec<f64> = up.iter().rev().copied().collect();
        let mk_up = mann_kendall(&up).unwrap();
        let mk_down = mann_kendall(&down).unwrap();
        assert_eq!(mk_down.s, -mk_up.s);
        assert_abs_diff_eq!(mk_down.z, -mk_up.z, epsilon = 1e-12);
        assert_abs_diff_eq!(mk_down.p_value, mk_up.p_value, epsilon = 1e-15);
        assert_eq!(mk_down.trend, Trend::Decreasing);
    }

    #[test]
    fn ties_reduce_variance() {
        // [1, 2, 2, 3]: S = 5, var = (4*3*13 - 2*1*9)/18 = 138/18
        let mk = mann_kendall(&[1.0, 2.0, 2.0, 3.0]).unwrap();
        assert_eq!(mk.s, 5.0);
        assert_abs_diff_eq!(mk.var_s, 138.0 / 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mk.tau, 5.0 / 6.0, epsilon = 1e-12);
        assert_eq!(mk.trend, Trend::NoTrend); // n too small for significance
    }

    #[test]
    fn constant_series_no_trend() {
        let mk = mann_kendall(&[2.0; 12]).unwrap();
        assert_eq!(mk.s, 0.0);
        assert_eq!(mk.z, 0.0);
        assert_abs_diff_eq!(mk.p_value, 1.0, epsilon = 1e-15);
        assert_eq!(mk.trend, Trend::NoTrend);
    }

    #[test]
    fn nan_dropped() {
        let y = [1.0, f64::NAN, 2.0, 3.0, f64::NAN, 4.0];
        let mk = mann_kendall(&y).unwrap();
        assert_eq!(mk.n, 4);
        assert_eq!(mk.s, 6.0);
    }

    #[test]
    fn too_short() {
        assert!(matches!(
            mann_kendall(&[1.0, 2.0]),
            Err(CubeError::InsufficientData { needed: 3, got: 2 })
        ));
    }
}
