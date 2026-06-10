//! Per-pixel time-series statistics.
//!
//! All functions treat non-finite values (`NaN`, `±inf`) as missing data and
//! drop them (pairwise with their time coordinate) before computing, matching
//! how `pyMannKendall` preprocesses series.

mod linear;
mod mann_kendall;
pub(crate) mod special;
mod theil_sen;

pub use linear::{LinearTrend, linear_trend};
pub use mann_kendall::{MannKendallResult, Trend, mann_kendall, mann_kendall_alpha};
pub use theil_sen::{TheilSenResult, theil_sen};

use crate::error::CubeError;

/// Drops pairs where either coordinate is non-finite.
pub(crate) fn filter_finite_pairs(t: &[f64], y: &[f64]) -> Result<(Vec<f64>, Vec<f64>), CubeError> {
    if t.len() != y.len() {
        return Err(CubeError::DimensionMismatch(format!(
            "time has {} values but series has {}",
            t.len(),
            y.len()
        )));
    }
    let (ft, fy): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(y)
        .filter(|(ti, yi)| ti.is_finite() && yi.is_finite())
        .map(|(ti, yi)| (*ti, *yi))
        .unzip();
    Ok((ft, fy))
}

/// Median by partial selection; `v` must be non-empty and all-finite.
pub(crate) fn median_mut(v: &mut [f64]) -> f64 {
    debug_assert!(!v.is_empty());
    let n = v.len();
    let mid = n / 2;
    let (lo_part, m, _) = v.select_nth_unstable_by(mid, f64::total_cmp);
    let hi = *m;
    if n % 2 == 1 {
        hi
    } else {
        let lo = lo_part.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median_mut(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_mut(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_mut(&mut [5.0]), 5.0);
    }

    #[test]
    fn filter_drops_nan_pairs() {
        let (t, y) =
            filter_finite_pairs(&[0.0, 1.0, 2.0, 3.0], &[1.0, f64::NAN, 3.0, 4.0]).unwrap();
        assert_eq!(t, vec![0.0, 2.0, 3.0]);
        assert_eq!(y, vec![1.0, 3.0, 4.0]);
    }
}
