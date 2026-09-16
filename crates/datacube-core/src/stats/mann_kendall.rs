use crate::error::CubeError;

use super::special::{norm_ppf, norm_sf};

/// Mann-Kendall statistic `S = Σ sign(x_j − x_i)` over pairs `j > i`, on
/// already-finite data.
fn mk_score(y: &[f64]) -> f64 {
    let n = y.len();
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
    s
}

/// Tie-corrected variance of `S`, `[n(n−1)(2n+5) − Σ_t t(t−1)(2t+5)] / 18`,
/// on already-finite data.
fn variance_s(y: &[f64]) -> f64 {
    let n = y.len();
    let nf = n as f64;
    let mut sorted = y.to_vec();
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
    (nf * (nf - 1.0) * (2.0 * nf + 5.0) - tie_term) / 18.0
}

/// Continuity-corrected normal score `z = (S ∓ 1)/√var(S)`.
fn z_score(s: f64, var_s: f64) -> f64 {
    if s > 0.0 {
        (s - 1.0) / var_s.sqrt()
    } else if s < 0.0 {
        (s + 1.0) / var_s.sqrt()
    } else {
        0.0
    }
}

/// Trend direction from a two-sided p-value at level `alpha`.
fn trend_at(z: f64, p_value: f64, alpha: f64) -> Trend {
    if p_value < alpha {
        if z > 0.0 {
            Trend::Increasing
        } else {
            Trend::Decreasing
        }
    } else {
        Trend::NoTrend
    }
}

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
    let s = mk_score(&y);
    let var_s = variance_s(&y);
    let z = z_score(s, var_s);
    let p_value = 2.0 * norm_sf(z.abs());
    let tau = s / (0.5 * nf * (nf - 1.0));
    let trend = trend_at(z, p_value, alpha);

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

/// Seasonal Mann-Kendall test (Hirsch & Slack 1984), matching
/// `pymannkendall.seasonal_test`.
///
/// The series is split into `period` seasons (season `i` = elements at indices
/// `i, i+period, i+2·period, …`, with non-finite values skipped per season);
/// the per-season statistic `S` and tie-corrected variance are summed, and the
/// continuity-corrected `z` is formed from the totals. This removes the
/// seasonal cycle's contribution to apparent trend, which the plain test
/// conflates with a monotonic signal — the correct estimator for the strongly
/// seasonal NDVI/NDWI series this engine targets.
///
/// `var_s` is the summed seasonal variance and `tau = S / Σ_seasons n_s(n_s−1)/2`.
/// Requires `period ≥ 1` and at least 3 finite observations overall.
pub fn seasonal_mann_kendall(
    values: &[f64],
    period: usize,
    alpha: f64,
) -> Result<MannKendallResult, CubeError> {
    if period == 0 {
        return Err(CubeError::InvalidParameter("period must be >= 1".into()));
    }
    let n_finite = values.iter().filter(|v| v.is_finite()).count();
    if n_finite < 3 {
        return Err(CubeError::InsufficientData {
            needed: 3,
            got: n_finite,
        });
    }
    let mut s = 0.0;
    let mut var_s = 0.0;
    let mut denom = 0.0;
    for i in 0..period {
        let season: Vec<f64> = values
            .iter()
            .skip(i)
            .step_by(period)
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let ns = season.len();
        if ns == 0 {
            continue;
        }
        s += mk_score(&season);
        var_s += variance_s(&season);
        denom += 0.5 * ns as f64 * (ns as f64 - 1.0);
    }
    if denom == 0.0 {
        return Err(CubeError::InsufficientData {
            needed: 3,
            got: n_finite,
        });
    }
    let z = z_score(s, var_s);
    let p_value = 2.0 * norm_sf(z.abs());
    let tau = s / denom;
    let trend = trend_at(z, p_value, alpha);
    Ok(MannKendallResult {
        trend,
        s,
        var_s,
        z,
        tau,
        p_value,
        n: n_finite,
    })
}

/// Modified Mann-Kendall test with the Hamed & Rao (1998) autocorrelation
/// correction, matching `pymannkendall.hamed_rao_modification_test`.
///
/// Serially correlated observations inflate the significance of the plain
/// test (its variance assumes independence); satellite NDVI/NDWI series are
/// strongly autocorrelated even after de-seasonalisation, so the plain test
/// over-declares trends. This variant rescales `var(S)` by a factor derived
/// from the significant autocorrelations of the Theil-Sen-detrended ranks:
/// `var(S)* = var(S) · [1 + 2/(n(n−1)(n−2)) · Σ_i (n−i)(n−i−1)(n−i−2) ρ_i]`,
/// summing only lags whose rank autocorrelation `ρ_i` exceeds the `alpha`-level
/// significance bound. `lag = Some(L)` uses the first `L` lags; `None` uses all.
pub fn mann_kendall_hamed_rao(
    values: &[f64],
    alpha: f64,
    lag: Option<usize>,
) -> Result<MannKendallResult, CubeError> {
    let y: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = y.len();
    if n < 3 {
        return Err(CubeError::InsufficientData { needed: 3, got: n });
    }
    let nf = n as f64;
    let s = mk_score(&y);
    let var0 = variance_s(&y);
    let tau = s / (0.5 * nf * (nf - 1.0));

    // Detrend with the Theil-Sen slope over unit-spaced indices (matching
    // pymannkendall's sens_slope, which uses index differences), then rank the
    // residuals; the correction is computed on the ranks' autocorrelation.
    let t_idx: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let slope = super::theil_sen::theil_sen(&t_idx, &y)?.slope;
    let detrended: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(k, &v)| v - (k as f64 + 1.0) * slope)
        .collect();
    let ranks = rankdata_average(&detrended);

    // lag = n considers every lag (pymannkendall default); Some(L) → first L.
    let lag = match lag {
        None => n,
        Some(l) => l + 1,
    };
    let acf = acf_biased(&ranks, lag.saturating_sub(1));
    let interval = norm_ppf(1.0 - alpha / 2.0) / nf.sqrt();

    let mut sni = 0.0;
    for (i, &rho) in acf.iter().enumerate().take(lag).skip(1) {
        if rho > interval || rho < -interval {
            let ni = (n - i) as f64;
            sni += ni * (ni - 1.0) * (ni - 2.0) * rho;
        }
    }
    let n_ns = 1.0 + (2.0 / (nf * (nf - 1.0) * (nf - 2.0))) * sni;
    let var_s = var0 * n_ns;

    let z = z_score(s, var_s);
    let p_value = 2.0 * norm_sf(z.abs());
    let trend = trend_at(z, p_value, alpha);
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

/// Average ranks (ties share the mean of their positions), like
/// `scipy.stats.rankdata` with the default `'average'` method.
fn rankdata_average(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && x[idx[j]] == x[idx[k]] {
            j += 1;
        }
        // positions k..j are tied; average of 1-based ranks (k+1 .. j)
        let avg = ((k + 1 + j) as f64) / 2.0;
        for &m in &idx[k..j] {
            ranks[m] = avg;
        }
        k = j;
    }
    ranks
}

/// Biased autocorrelation `ρ_k = c_k / c_0`, `c_k = Σ_t (x_t−x̄)(x_{t+k}−x̄)`,
/// for lags `0..=nlags` — the normalisation `pymannkendall.__acf` uses.
fn acf_biased(x: &[f64], nlags: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let c0: f64 = y.iter().map(|v| v * v).sum();
    (0..=nlags)
        .map(|k| {
            if c0 == 0.0 {
                return if k == 0 { 1.0 } else { 0.0 };
            }
            let ck: f64 = (0..n - k).map(|t| y[t] * y[t + k]).sum();
            ck / c0
        })
        .collect()
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

    #[test]
    fn seasonal_period_one_equals_original() {
        let y = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 9.0, 6.0, 10.0];
        let orig = mann_kendall(&y).unwrap();
        let seas = seasonal_mann_kendall(&y, 1, 0.05).unwrap();
        assert_abs_diff_eq!(seas.s, orig.s, epsilon = 1e-12);
        assert_abs_diff_eq!(seas.var_s, orig.var_s, epsilon = 1e-12);
        assert_abs_diff_eq!(seas.z, orig.z, epsilon = 1e-12);
        assert_abs_diff_eq!(seas.tau, orig.tau, epsilon = 1e-12);
    }

    #[test]
    fn seasonal_removes_pure_cycle() {
        // A pure repeating seasonal cycle with no inter-year trend: the plain
        // test may see structure, but the seasonal test sees S = 0 (each
        // season is constant across years).
        let cycle = [1.0, 5.0, 3.0, 8.0];
        let mut y = Vec::new();
        for _ in 0..6 {
            y.extend_from_slice(&cycle);
        }
        let seas = seasonal_mann_kendall(&y, 4, 0.05).unwrap();
        assert_eq!(seas.s, 0.0);
        assert_eq!(seas.trend, Trend::NoTrend);
    }

    #[test]
    fn seasonal_detects_interannual_trend_under_a_cycle() {
        // Rising year-on-year with a strong seasonal cycle on top.
        let cycle = [0.0, 4.0, 1.0, 6.0];
        let mut y = Vec::new();
        for year in 0..8 {
            for &c in &cycle {
                y.push(c + 2.0 * year as f64);
            }
        }
        let seas = seasonal_mann_kendall(&y, 4, 0.05).unwrap();
        assert!(seas.s > 0.0);
        assert_eq!(seas.trend, Trend::Increasing);
    }

    #[test]
    fn hamed_rao_lag0_leaves_variance_unchanged() {
        // With no lags considered (Some(0)) the correction factor is exactly 1,
        // so the result matches the plain test.
        let y = [
            1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0,
        ];
        let plain = mann_kendall(&y).unwrap();
        let hr = mann_kendall_hamed_rao(&y, 0.05, Some(0)).unwrap();
        assert_abs_diff_eq!(hr.var_s, plain.var_s, epsilon = 1e-12);
        assert_abs_diff_eq!(hr.z, plain.z, epsilon = 1e-12);
        assert_eq!(hr.s, plain.s);
    }

    #[test]
    fn hamed_rao_inflates_variance_under_ar1() {
        // Deterministic AR(1) with phi = 0.85: strong positive serial
        // correlation, so the Hamed-Rao factor must raise var(S) above the
        // independence value and shrink |z|.
        let n = 120;
        let phi = 0.85;
        // fixed LCG noise, zero-ish mean
        let mut state: u64 = 0x1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0 // in [-1, 1)
        };
        let mut y = Vec::with_capacity(n);
        let mut prev = 0.0;
        for _ in 0..n {
            prev = phi * prev + next();
            y.push(prev);
        }
        let plain = mann_kendall(&y).unwrap();
        let hr = mann_kendall_hamed_rao(&y, 0.05, None).unwrap();
        assert_eq!(hr.s, plain.s); // same S; only the variance changes
        assert!(
            hr.var_s > plain.var_s,
            "hamed-rao var {} should exceed plain var {}",
            hr.var_s,
            plain.var_s
        );
        assert!(hr.z.abs() < plain.z.abs()); // inflated variance → smaller |z|
    }

    #[test]
    fn rankdata_average_handles_ties() {
        // scipy.stats.rankdata([3,1,4,1,5,9,2,6]) with 'average'
        let r = rankdata_average(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(r, vec![4.0, 1.5, 5.0, 1.5, 6.0, 8.0, 3.0, 7.0]);
    }
}
