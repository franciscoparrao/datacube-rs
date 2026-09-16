//! False-discovery-rate control for per-pixel / per-polygon p-value fields.
//!
//! A trend or break test applied independently to every pixel (or every
//! polygon) of a map produces one p-value per unit. Declaring "significant"
//! wherever `p < α` ignores multiplicity: over `m` truly-null units one expects
//! `α·m` false positives, so a raw-threshold map of a large cube is
//! uninterpretable. Controlling the false discovery rate (Benjamini &
//! Hochberg 1995; Benjamini & Yekutieli 2001 under dependence) fixes the
//! expected fraction of false discoveries among the declared-significant units,
//! and — following Wilks (2006, 2016) — is the recommended way to assign
//! **field significance** to a spatial map of tests.

use crate::error::CubeError;

/// FDR procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdrMethod {
    /// Benjamini-Hochberg (1995): controls FDR under independence or positive
    /// regression dependence. The usual choice for spatial fields.
    BenjaminiHochberg,
    /// Benjamini-Yekutieli (2001): controls FDR under *arbitrary* dependence
    /// (adds the `Σ 1/i` penalty). More conservative; robust to the negative
    /// spatial dependence trend fields can show.
    BenjaminiYekutieli,
}

/// Outcome of an FDR correction over a p-value field.
#[derive(Debug, Clone)]
pub struct FdrResult {
    /// Per input: `true` where the unit is declared significant. Non-finite
    /// inputs are never rejected.
    pub rejected: Vec<bool>,
    /// Per input: FDR-adjusted p-value (monotone step-up), `NaN` where the
    /// input was non-finite.
    pub adjusted: Vec<f64>,
    /// Number of finite p-values tested.
    pub n_tested: usize,
    /// Number declared significant.
    pub n_significant: usize,
    /// Largest raw p-value that is still rejected (the effective threshold),
    /// or `NaN` if nothing is significant.
    pub threshold: f64,
}

/// Applies FDR control at level `q` to a p-value field, matching
/// `statsmodels.stats.multitest.multipletests(method='fdr_bh' | 'fdr_by')`.
///
/// Non-finite entries (masked pixels, insufficient-data polygons) are carried
/// through as `NaN`/not-rejected and excluded from the count `m`. The adjusted
/// p-values are the standard monotone step-up values, clipped to `≤ 1`; a unit
/// is significant iff its adjusted p-value is `≤ q`.
pub fn fdr(pvalues: &[f64], q: f64, method: FdrMethod) -> Result<FdrResult, CubeError> {
    if !(q.is_finite() && q > 0.0 && q <= 1.0) {
        return Err(CubeError::InvalidParameter(format!(
            "FDR level q must be in (0, 1], got {q}"
        )));
    }
    let n = pvalues.len();
    let mut adjusted = vec![f64::NAN; n];
    let mut rejected = vec![false; n];

    // finite p-values with their original positions, sorted ascending
    let mut order: Vec<usize> = (0..n).filter(|&i| pvalues[i].is_finite()).collect();
    let m = order.len();
    if m == 0 {
        return Ok(FdrResult {
            rejected,
            adjusted,
            n_tested: 0,
            n_significant: 0,
            threshold: f64::NAN,
        });
    }
    order.sort_by(|&a, &b| pvalues[a].total_cmp(&pvalues[b]));

    let mf = m as f64;
    // dependence penalty c(m): 1 for BH, Σ_{i=1..m} 1/i for BY
    let cm = match method {
        FdrMethod::BenjaminiHochberg => 1.0,
        FdrMethod::BenjaminiYekutieli => (1..=m).map(|i| 1.0 / i as f64).sum::<f64>(),
    };

    // step-up adjusted p: from the largest rank down, adj = min(running,
    // p·m·c(m)/rank), clipped to 1.
    let mut running = f64::INFINITY;
    for rank in (1..=m).rev() {
        let idx = order[rank - 1];
        let raw = pvalues[idx] * mf * cm / rank as f64;
        running = running.min(raw);
        adjusted[idx] = running.min(1.0);
    }

    // reject where adjusted <= q; threshold = largest raw p among rejected
    let mut n_significant = 0;
    let mut threshold = f64::NAN;
    for &idx in &order {
        if adjusted[idx] <= q {
            rejected[idx] = true;
            n_significant += 1;
            // order is ascending, so the last rejected has the largest raw p
            threshold = pvalues[idx];
        }
    }

    Ok(FdrResult {
        rejected,
        adjusted,
        n_tested: m,
        n_significant,
        threshold,
    })
}

/// Whether the p-value field is **field-significant** at FDR level `q`
/// (Wilks 2006): at least one unit survives the FDR correction. Wilks
/// recommends `q = 2·α_global`; the locally-significant units are exactly
/// those in [`FdrResult::rejected`].
pub fn field_significant(pvalues: &[f64], q: f64, method: FdrMethod) -> Result<bool, CubeError> {
    Ok(fdr(pvalues, q, method)?.n_significant > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn bh_known_example() {
        // classic BH example (Benjamini-Hochberg 1995, Table 1 style)
        let p = [
            0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324,
        ];
        let r = fdr(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        // m=10; adjusted step-up of p*10/rank. rank8: 0.0344*10/8=0.043<=0.05;
        // rank9: 0.0459*10/9=0.051>0.05 → reject ranks 1..8 (8 significant).
        assert_eq!(r.n_tested, 10);
        assert_eq!(r.n_significant, 8);
        assert!(r.rejected[7]); // p=0.0344 rejected
        assert!(!r.rejected[8]); // p=0.0459 not
        assert_abs_diff_eq!(r.threshold, 0.0344, epsilon = 1e-15);
        assert_abs_diff_eq!(r.adjusted[0], 0.001, epsilon = 1e-12);
    }

    #[test]
    fn by_is_more_conservative_than_bh() {
        let p = [0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.7, 0.9];
        let bh = fdr(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        let by = fdr(&p, 0.05, FdrMethod::BenjaminiYekutieli).unwrap();
        assert!(by.n_significant <= bh.n_significant);
        // BY adjusted >= BH adjusted everywhere (extra Σ1/i penalty)
        for i in 0..p.len() {
            assert!(by.adjusted[i] >= bh.adjusted[i] - 1e-15);
        }
    }

    #[test]
    fn adjusted_is_monotone_and_clipped() {
        let p = [0.9, 0.01, 0.5, 0.001, 0.3];
        let r = fdr(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        // sort by raw p and check adjusted is non-decreasing in raw-p order
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
        let mut prev = -1.0;
        for &i in &idx {
            assert!(r.adjusted[i] >= prev - 1e-15);
            assert!(r.adjusted[i] <= 1.0);
            prev = r.adjusted[i];
        }
    }

    #[test]
    fn nan_carried_through_and_excluded() {
        let p = [0.001, f64::NAN, 0.04, f64::NAN, 0.9];
        let r = fdr(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        assert_eq!(r.n_tested, 3);
        assert!(r.adjusted[1].is_nan() && r.adjusted[3].is_nan());
        assert!(!r.rejected[1] && !r.rejected[3]);
        // m=3: adjusted[0]=0.001*3/1=0.003 significant
        assert!(r.rejected[0]);
        assert_abs_diff_eq!(r.adjusted[0], 0.003, epsilon = 1e-12);
    }

    #[test]
    fn all_null_none_significant() {
        let p = [0.4, 0.5, 0.6, 0.7, 0.8, 0.99];
        let r = fdr(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        assert_eq!(r.n_significant, 0);
        assert!(r.threshold.is_nan());
        assert!(!field_significant(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap());
    }

    #[test]
    fn field_significance_flag() {
        let p = [0.0001, 0.2, 0.3, 0.4, 0.5];
        assert!(field_significant(&p, 0.05, FdrMethod::BenjaminiHochberg).unwrap());
    }

    #[test]
    fn rejects_bad_q() {
        assert!(matches!(
            fdr(&[0.1], 0.0, FdrMethod::BenjaminiHochberg),
            Err(CubeError::InvalidParameter(_))
        ));
        assert!(matches!(
            fdr(&[0.1], 1.5, FdrMethod::BenjaminiHochberg),
            Err(CubeError::InvalidParameter(_))
        ));
    }

    #[test]
    fn empty_and_all_nan() {
        let r = fdr(&[f64::NAN, f64::NAN], 0.05, FdrMethod::BenjaminiHochberg).unwrap();
        assert_eq!(r.n_tested, 0);
        assert_eq!(r.n_significant, 0);
        assert!(r.threshold.is_nan());
    }
}
