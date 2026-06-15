//! Temporal transforms of a cube: compositing and gap-filling.
//!
//! Both operate purely on the public cube API and return new cubes; the
//! time axis must be sorted ascending (as produced by stacking).

use ndarray::Array4;
use rayon::prelude::*;

use crate::cube::Cube;
use crate::error::CubeError;

/// NaN-aware aggregation used by [`Cube::composite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMethod {
    /// Median of finite values (the standard cloud-robust composite).
    Median,
    Mean,
    Min,
    Max,
}

/// How time slices are grouped by [`Cube::composite`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeWindow {
    /// Merge slices with an identical time coordinate (e.g. adjacent
    /// satellite tiles acquired at the same instant).
    SameTime,
    /// Fixed-width bins starting at the first time (e.g. `1.0 / 12.0` for
    /// monthly composites on a fractional-year axis).
    Period(f64),
}

impl Cube {
    /// Aggregates time slices into composites.
    ///
    /// Slices are grouped by `window`; within each group every pixel is
    /// reduced with `method` over its finite values (all-NaN groups stay
    /// NaN). The composite time coordinate is the mean of the member times.
    ///
    /// ```
    /// # use datacube_core::{Cube, CompositeMethod, CompositeWindow};
    /// # use ndarray::Array4;
    /// // two tiles at t=0 covering complementary halves, one slice at t=1
    /// let mut data = Array4::from_elem((1, 1, 2, 3), f64::NAN);
    /// data[[0, 0, 0, 0]] = 1.0;             // tile A covers x=0
    /// data[[0, 0, 1, 1]] = 3.0;             // tile B covers x=1
    /// data[[0, 0, 0, 2]] = 5.0;
    /// data[[0, 0, 1, 2]] = 7.0;
    /// let cube = Cube::new(data, vec![0.0, 0.0, 1.0], vec!["b".into()]).unwrap();
    /// let merged = cube.composite(CompositeWindow::SameTime, CompositeMethod::Median).unwrap();
    /// assert_eq!(merged.dims().3, 2);
    /// assert_eq!(merged.data()[[0, 0, 0, 0]], 1.0);
    /// assert_eq!(merged.data()[[0, 0, 1, 0]], 3.0);
    /// ```
    pub fn composite(
        &self,
        window: CompositeWindow,
        method: CompositeMethod,
    ) -> Result<Cube, CubeError> {
        let groups = group_times(self.time(), window)?;
        let (nb, ny, nx, nt) = self.dims();
        let ng = groups.len();
        let src = self.data();
        let src = src
            .as_slice()
            .expect("cube data is standard layout (enforced by Cube::new)");

        let times: Vec<f64> = groups
            .iter()
            .map(|g| g.iter().map(|&i| self.time()[i]).sum::<f64>() / g.len() as f64)
            .collect();

        // every pixel reduces its own series independently — parallelize over
        // pixels (the source and output pixel series are both contiguous).
        let mut out = vec![f64::NAN; nb * ny * nx * ng];
        out.par_chunks_mut(ng).enumerate().for_each(|(pp, dst)| {
            let series = &src[pp * nt..(pp + 1) * nt];
            let mut values = Vec::new();
            for (gi, group) in groups.iter().enumerate() {
                values.clear();
                values.extend(group.iter().map(|&ti| series[ti]).filter(|v| v.is_finite()));
                dst[gi] = reduce(&mut values, method);
            }
        });

        let data = Array4::from_shape_vec((nb, ny, nx, ng), out)
            .map_err(|e| CubeError::DimensionMismatch(e.to_string()))?;
        Cube::new(data, times, self.bands().to_vec())
    }

    /// Fills temporal NaN gaps per pixel by linear interpolation between the
    /// nearest finite observations.
    ///
    /// Gaps wider than `max_gap` (in time units, measured between the two
    /// bracketing observations) are left as NaN, as are leading/trailing
    /// NaNs (no extrapolation). Requires an ascending time axis.
    pub fn gapfill_linear(&self, max_gap: Option<f64>) -> Result<Cube, CubeError> {
        let time = self.time();
        if time.windows(2).any(|w| w[1] < w[0]) {
            return Err(CubeError::DimensionMismatch(
                "gapfill_linear requires an ascending time axis".into(),
            ));
        }
        let (_, _, _, nt) = self.dims();
        let mut data = self.data().to_owned();
        let flat = data
            .as_slice_mut()
            .expect("cube data is standard layout (enforced by Cube::new)");

        // each pixel's series is a contiguous nt-slice; fill them in parallel.
        flat.par_chunks_mut(nt).for_each(|series| {
            let mut prev: Option<usize> = None;
            for i in 0..nt {
                if !series[i].is_finite() {
                    continue;
                }
                // close a gap (prev, i) if one was open
                if let Some(p) = prev
                    && i > p + 1
                {
                    let dt = time[i] - time[p];
                    if max_gap.is_none_or(|mg| dt <= mg) && dt > 0.0 {
                        let (v0, v1) = (series[p], series[i]);
                        for k in (p + 1)..i {
                            let f = (time[k] - time[p]) / dt;
                            series[k] = v0 + f * (v1 - v0);
                        }
                    }
                }
                prev = Some(i);
            }
        });

        Cube::new(data, time.to_vec(), self.bands().to_vec())
    }
}

/// Groups time indices according to the window; groups preserve time order.
fn group_times(time: &[f64], window: CompositeWindow) -> Result<Vec<Vec<usize>>, CubeError> {
    if time.windows(2).any(|w| w[1] < w[0]) {
        return Err(CubeError::DimensionMismatch(
            "composite requires an ascending time axis".into(),
        ));
    }
    match window {
        CompositeWindow::SameTime => {
            let mut groups: Vec<Vec<usize>> = Vec::new();
            for (i, &t) in time.iter().enumerate() {
                match groups.last_mut() {
                    Some(g) if time[g[0]] == t => g.push(i),
                    _ => groups.push(vec![i]),
                }
            }
            Ok(groups)
        }
        CompositeWindow::Period(width) => {
            if !width.is_finite() || width <= 0.0 {
                return Err(CubeError::InvalidParameter(format!(
                    "composite period must be finite and > 0, got {width}"
                )));
            }
            let t0 = time.first().copied().unwrap_or(0.0);
            let mut groups: Vec<(u64, Vec<usize>)> = Vec::new();
            for (i, &t) in time.iter().enumerate() {
                let bin = ((t - t0) / width).floor() as u64;
                match groups.last_mut() {
                    Some((b, g)) if *b == bin => g.push(i),
                    _ => groups.push((bin, vec![i])),
                }
            }
            Ok(groups.into_iter().map(|(_, g)| g).collect())
        }
    }
}

/// NaN-free reduction; `values` may be reordered. Empty input → NaN.
fn reduce(values: &mut [f64], method: CompositeMethod) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    match method {
        CompositeMethod::Median => crate::stats::median_mut(values),
        CompositeMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
        CompositeMethod::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        CompositeMethod::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array4;

    fn cube_1px(values: &[f64], times: &[f64]) -> Cube {
        let mut data = Array4::zeros((1, 1, 1, values.len()));
        for (i, v) in values.iter().enumerate() {
            data[[0, 0, 0, i]] = *v;
        }
        Cube::new(data, times.to_vec(), vec!["b".into()]).unwrap()
    }

    #[test]
    fn monthly_median_composite() {
        // fractional years: three obs in month 0, two in month 1
        let times = [2024.0, 2024.01, 2024.02, 2024.09, 2024.10];
        let cube = cube_1px(&[1.0, 9.0, 2.0, 4.0, 6.0], &times);
        let c = cube
            .composite(CompositeWindow::Period(1.0 / 12.0), CompositeMethod::Median)
            .unwrap();
        assert_eq!(c.dims().3, 2);
        assert_eq!(c.data()[[0, 0, 0, 0]], 2.0); // median(1, 9, 2)
        assert_eq!(c.data()[[0, 0, 0, 1]], 5.0); // median(4, 6)
        assert_abs_diff_eq!(c.time()[0], 2024.01, epsilon = 1e-12);
    }

    #[test]
    fn composite_ignores_nan_and_keeps_all_nan_groups() {
        let cube = cube_1px(&[f64::NAN, 3.0, f64::NAN, f64::NAN], &[0.0, 0.0, 1.0, 1.0]);
        let c = cube
            .composite(CompositeWindow::SameTime, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(c.dims().3, 2);
        assert_eq!(c.data()[[0, 0, 0, 0]], 3.0);
        assert!(c.data()[[0, 0, 0, 1]].is_nan());
    }

    #[test]
    fn composite_methods() {
        let cube = cube_1px(&[1.0, 4.0, 2.0], &[0.0, 0.0, 0.0]);
        let get = |m| cube.composite(CompositeWindow::SameTime, m).unwrap().data()[[0, 0, 0, 0]];
        assert_eq!(get(CompositeMethod::Min), 1.0);
        assert_eq!(get(CompositeMethod::Max), 4.0);
        assert_eq!(get(CompositeMethod::Median), 2.0);
        assert_abs_diff_eq!(get(CompositeMethod::Mean), 7.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn gapfill_interpolates_with_real_time_distances() {
        let cube = cube_1px(&[1.0, f64::NAN, f64::NAN, 7.0], &[0.0, 1.0, 2.0, 3.0]);
        let filled = cube.gapfill_linear(None).unwrap();
        assert_abs_diff_eq!(filled.data()[[0, 0, 0, 1]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(filled.data()[[0, 0, 0, 2]], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn gapfill_respects_max_gap_and_edges() {
        let cube = cube_1px(
            &[
                f64::NAN,
                1.0,
                f64::NAN,
                5.0,
                f64::NAN,
                f64::NAN,
                8.0,
                f64::NAN,
            ],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        );
        let filled = cube.gapfill_linear(Some(2.0)).unwrap();
        let d = filled.data().to_owned();
        assert!(d[[0, 0, 0, 0]].is_nan()); // leading edge untouched
        assert_abs_diff_eq!(d[[0, 0, 0, 2]], 3.0, epsilon = 1e-12); // gap of 2.0 <= max
        assert!(d[[0, 0, 0, 4]].is_nan()); // gap of 3.0 > max stays
        assert!(d[[0, 0, 0, 5]].is_nan());
        assert!(d[[0, 0, 0, 7]].is_nan()); // trailing edge untouched
    }

    #[test]
    fn unsorted_time_is_rejected() {
        let cube = cube_1px(&[1.0, 2.0], &[1.0, 0.0]);
        assert!(
            cube.composite(CompositeWindow::SameTime, CompositeMethod::Mean)
                .is_err()
        );
        assert!(cube.gapfill_linear(None).is_err());
    }
}
