//! Shared least-squares fit of the harmonic-with-trend model, used by both
//! [`super::harmonic_regression`] and the break-detection segment model.
//!
//! The model is `intercept + slope·(t − t̄) + Σ_k aₖcos(kωt) + bₖsin(kωt)`
//! with the linear term centered on `t̄` for conditioning. It is fitted by
//! normal equations and a partial-pivot solve (no linear-algebra dependency).

use crate::error::CubeError;

/// A fitted harmonic-with-trend model. Coefficients (`beta`) are in the
/// centered basis: `beta[0]` is the model value at `t̄`, `beta[1]` the slope,
/// then `(aₖ, bₖ)` pairs for `k = 1..=n_harmonics`.
pub(super) struct HarmonicModel {
    pub beta: Vec<f64>,
    pub mean_t: f64,
    pub omega: f64,
    pub n_harmonics: usize,
}

impl HarmonicModel {
    /// Fits the model to `(t, y)` (equal length, finite) with `period > 0`.
    /// Errors if the design matrix is rank-deficient (aliased harmonic or
    /// constant time).
    pub fn fit(t: &[f64], y: &[f64], n_harmonics: usize, period: f64) -> Result<Self, CubeError> {
        let n = t.len();
        let p = 2 + 2 * n_harmonics;
        let mean_t = t.iter().sum::<f64>() / n as f64;
        let omega = 2.0 * core::f64::consts::PI / period;

        // normal equations X'X β = X'y, accumulating the upper triangle
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];
        let mut r = Vec::with_capacity(p);
        for (&ti, &yi) in t.iter().zip(y) {
            fill_row(&mut r, ti, mean_t, omega, n_harmonics);
            for i in 0..p {
                xty[i] += r[i] * yi;
                for (x, rj) in xtx[i][i..].iter_mut().zip(&r[i..]) {
                    *x += r[i] * rj;
                }
            }
        }
        // mirror into the lower triangle
        for i in 1..p {
            let (head, tail) = xtx.split_at_mut(i);
            for (j, row) in head.iter().enumerate() {
                tail[0][j] = row[i];
            }
        }
        let beta = solve_symmetric(xtx, xty)?;
        Ok(Self {
            beta,
            mean_t,
            omega,
            n_harmonics,
        })
    }

    /// Fitted value at time `ti`.
    pub fn predict(&self, ti: f64) -> f64 {
        let mut acc = self.beta[0] + self.beta[1] * (ti - self.mean_t);
        for k in 1..=self.n_harmonics {
            let kt = self.omega * k as f64 * ti;
            acc += self.beta[2 * k] * kt.cos() + self.beta[2 * k + 1] * kt.sin();
        }
        acc
    }
}

/// Writes the design row `[1, t−t̄, cos(kωt), sin(kωt), …]` into `out`.
fn fill_row(out: &mut Vec<f64>, ti: f64, mean_t: f64, omega: f64, n_harmonics: usize) {
    out.clear();
    out.push(1.0);
    out.push(ti - mean_t);
    for k in 1..=n_harmonics {
        let kt = omega * k as f64 * ti;
        out.push(kt.cos());
        out.push(kt.sin());
    }
}

/// Solves a symmetric positive (semi-)definite system by Gaussian
/// elimination with partial pivoting; errors on rank deficiency.
fn solve_symmetric(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, CubeError> {
    let p = b.len();
    let scale: f64 = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = scale * 1e-12 * p as f64;
    for col in 0..p {
        let pivot_row = (col..p)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        if a[pivot_row][col].abs() <= tol {
            return Err(CubeError::SingularSystem(
                "design matrix is rank-deficient (aliased harmonic or constant time?)".into(),
            ));
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let (pivot_rows, rest) = a.split_at_mut(col + 1);
        let pivot = &pivot_rows[col];
        for (i, row) in rest.iter_mut().enumerate() {
            let f = row[col] / pivot[col];
            if f != 0.0 {
                for (rj, pj) in row[col..].iter_mut().zip(&pivot[col..]) {
                    *rj -= f * pj;
                }
                b[col + 1 + i] -= f * b[col];
            }
        }
    }
    for col in (0..p).rev() {
        let mut s = b[col];
        for j in (col + 1)..p {
            s -= a[col][j] * b[j];
        }
        b[col] = s / a[col][col];
    }
    Ok(b)
}
