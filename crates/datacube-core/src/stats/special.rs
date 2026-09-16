//! Minimal special functions for p-values (no heavy stats dependency).

/// Survival function of the standard normal, `P(Z > z)`.
pub(crate) fn norm_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / core::f64::consts::SQRT_2)
}

/// Inverse CDF (quantile) of the standard normal: the `z` with `P(Z ≤ z) = p`,
/// for `p ∈ (0, 1)`. Acklam's rational approximation refined by one Halley
/// step, giving full double precision — matches `scipy.stats.norm.ppf`.
///
/// The coefficients are Acklam's published constants; some carry more digits
/// than an `f64` distinguishes, so the precision lint is silenced for them.
#[allow(clippy::excessive_precision)]
pub(crate) fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Acklam coefficients.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement step against the true CDF for full precision.
    let e = 0.5 * libm::erfc(-x / core::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * core::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

/// Two-sided p-value of a Student-t statistic with `df` degrees of freedom:
/// `p = I_{df/(df+t²)}(df/2, 1/2)`.
pub(crate) fn student_t_two_sided(t: f64, df: f64) -> f64 {
    if !t.is_finite() {
        return 0.0;
    }
    betai(0.5 * df, 0.5, df / (df + t * t))
}

/// Regularized incomplete beta function `I_x(a, b)`
/// (Numerical Recipes §6.4, continued-fraction evaluation).
pub(crate) fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt =
        libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn norm_ppf_known_values() {
        use super::norm_ppf;
        // scipy.stats.norm.ppf values
        assert_abs_diff_eq!(norm_ppf(0.975), 1.959963984540054, epsilon = 1e-12);
        assert_abs_diff_eq!(norm_ppf(0.5), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(norm_ppf(0.025), -1.959963984540054, epsilon = 1e-12);
        assert_abs_diff_eq!(norm_ppf(0.95), 1.6448536269514722, epsilon = 1e-12);
        // round-trip against the survival function
        assert_abs_diff_eq!(super::norm_sf(norm_ppf(0.9)), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn norm_sf_known_values() {
        assert_abs_diff_eq!(norm_sf(0.0), 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(norm_sf(1.959963984540054), 0.025, epsilon = 1e-12);
        assert_abs_diff_eq!(norm_sf(1.0), 0.15865525393145707, epsilon = 1e-12);
    }

    #[test]
    fn t_two_sided_known_value() {
        // scipy.stats.t.sf(2.0, 10) * 2 = 0.07338803...
        assert_abs_diff_eq!(student_t_two_sided(2.0, 10.0), 0.073388, epsilon = 1e-5);
        assert_abs_diff_eq!(student_t_two_sided(0.0, 10.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn betai_bounds_and_symmetry() {
        assert_eq!(betai(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betai(2.0, 3.0, 1.0), 1.0);
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        let x = 0.3;
        assert_abs_diff_eq!(
            betai(2.5, 1.5, x),
            1.0 - betai(1.5, 2.5, 1.0 - x),
            epsilon = 1e-12
        );
        // I_x(1,1) = x (uniform CDF)
        assert_abs_diff_eq!(betai(1.0, 1.0, 0.42), 0.42, epsilon = 1e-12);
    }
}
