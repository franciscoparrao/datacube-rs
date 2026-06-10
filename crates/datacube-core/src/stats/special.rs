//! Minimal special functions for p-values (no heavy stats dependency).

/// Survival function of the standard normal, `P(Z > z)`.
pub(crate) fn norm_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / core::f64::consts::SQRT_2)
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
