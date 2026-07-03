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
    /// Fixed-width bins starting at the first time (physical windows such
    /// as 16-day composites: `16.0 / 365.25` on a fractional-year axis).
    /// Bins are anchored on the first observation, so the grouping depends
    /// on when the series starts; for calendar months use [`CalendarMonth`].
    ///
    /// [`CalendarMonth`]: CompositeWindow::CalendarMonth
    Period(f64),
    /// Calendar-month bins recovered exactly from a fractional-year time
    /// axis (the inverse of `datacube_io::fractional_year`): each slice is
    /// keyed by its `(year, month)`, independent of when the series starts.
    CalendarMonth,
    /// Calendar-year bins on a fractional-year time axis.
    CalendarYear,
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

        let times = group_means(self.time(), &groups);

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
        Ok(Cube::new(data, times, self.bands().to_vec())?.inherit_georef(self))
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
            return Err(CubeError::UnsortedTime(
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

        Ok(Cube::new(data, time.to_vec(), self.bands().to_vec())?.inherit_georef(self))
    }
}

/// The mean time coordinate of each group, in group order.
fn group_means(time: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|g| g.iter().map(|&i| time[i]).sum::<f64>() / g.len() as f64)
        .collect()
}

/// The time coordinates [`Cube::composite`] would produce for `time` under
/// `window`, without touching any pixel data. Composite/gapfill/index never
/// change the spatial extent, so this plus the original `(height, width)`
/// fully describe a pipeline's output shape without running it — used by
/// [`crate::pipeline::ChunkPipeline::output_time`] to report chunked-pipeline
/// output shape cheaply.
pub(crate) fn composite_time_axis(
    time: &[f64],
    window: CompositeWindow,
) -> Result<Vec<f64>, CubeError> {
    let groups = group_times(time, window)?;
    Ok(group_means(time, &groups))
}

/// Groups time indices according to the window; groups preserve time order.
fn group_times(time: &[f64], window: CompositeWindow) -> Result<Vec<Vec<usize>>, CubeError> {
    if time.windows(2).any(|w| w[1] < w[0]) {
        return Err(CubeError::UnsortedTime(
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
            group_by_key(time, |t| ((t - t0) / width).floor() as i64)
        }
        CompositeWindow::CalendarMonth => group_by_key(time, |t| {
            let (year, month) = calendar_year_month(t);
            i64::from(year) * 12 + i64::from(month) - 1
        }),
        CompositeWindow::CalendarYear => {
            group_by_key(time, |t| i64::from(calendar_year_month(t).0))
        }
    }
}

/// Groups consecutive indices whose times map to the same bin key (the time
/// axis is ascending, so equal keys are always adjacent).
fn group_by_key(time: &[f64], key: impl Fn(f64) -> i64) -> Result<Vec<Vec<usize>>, CubeError> {
    if time.iter().any(|t| !t.is_finite()) {
        return Err(CubeError::InvalidParameter(
            "calendar/period composites require finite time coordinates".into(),
        ));
    }
    let mut groups: Vec<(i64, Vec<usize>)> = Vec::new();
    for (i, &t) in time.iter().enumerate() {
        let bin = key(t);
        match groups.last_mut() {
            Some((b, g)) if *b == bin => g.push(i),
            _ => groups.push((bin, vec![i])),
        }
    }
    Ok(groups.into_iter().map(|(_, g)| g).collect())
}

/// Cumulative days before each month in a non-leap year.
const CUM_DAYS: [f64; 12] = [
    0.0, 31.0, 59.0, 90.0, 120.0, 151.0, 181.0, 212.0, 243.0, 273.0, 304.0, 334.0,
];

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Recovers `(year, month)` from a fractional-year coordinate — the exact
/// inverse of the month binning in `datacube_io::fractional_year`.
///
/// A tolerance of ~0.09 s absorbs the floating-point round-trip so that
/// coordinates computed for exact month boundaries (e.g. `2024 + 31/366`
/// for 2024-02-01T00:00Z) land in the month they name.
fn calendar_year_month(t: f64) -> (i32, u32) {
    const EPS: f64 = 1e-6; // days
    let mut year = t.floor() as i32;
    let days_in_year = if is_leap(year) { 366.0 } else { 365.0 };
    // 0-based day-of-year (+ intra-day fraction)
    let mut doy = (t - f64::from(year)) * days_in_year + EPS;
    if doy >= days_in_year {
        // t sat a hair below an exact year boundary: it names Jan 1 of the
        // next year
        year += 1;
        doy = 0.0;
    }
    let leap_shift = if is_leap(year) { 1.0 } else { 0.0 };
    let month = (0..12)
        .rev()
        .find(|&m| {
            let start = CUM_DAYS[m] + if m >= 2 { leap_shift } else { 0.0 };
            doy >= start
        })
        .unwrap_or(0);
    (year, month as u32 + 1)
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
    use crate::cube::GeoRef;
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
    fn composite_and_gapfill_propagate_georef() {
        let geo = GeoRef {
            epsg: Some(32719),
            transform: Some([300_000.0, 10.0, 0.0, 6_200_000.0, 0.0, -10.0]),
        };
        let cube = cube_1px(&[1.0, f64::NAN, 3.0], &[0.0, 1.0, 2.0]).with_georef(geo);
        let composited = cube
            .composite(CompositeWindow::SameTime, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(composited.georef(), Some(geo));
        let filled = cube.gapfill_linear(None).unwrap();
        assert_eq!(filled.georef(), Some(geo));
    }

    #[test]
    fn composite_without_georef_stays_geo_blind() {
        let cube = cube_1px(&[1.0, 2.0], &[0.0, 1.0]);
        let c = cube
            .composite(CompositeWindow::SameTime, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(c.georef(), None);
    }

    /// Fractional year of a calendar date, mirroring `datacube_io::fractional_year`.
    fn fy(year: i32, doy0: u32, day_frac: f64) -> f64 {
        let days = if is_leap(year) { 366.0 } else { 365.0 };
        f64::from(year) + (f64::from(doy0) + day_frac) / days
    }

    #[test]
    fn calendar_month_bins_are_anchor_independent() {
        // Jan 20 and Feb 5: 16 days apart, so Period(1/12) anchored on the
        // first observation lumps them together; CalendarMonth must not.
        let times = [fy(2023, 19, 0.5), fy(2023, 35, 0.5)];
        let cube = cube_1px(&[1.0, 3.0], &times);
        let p = cube
            .composite(CompositeWindow::Period(1.0 / 12.0), CompositeMethod::Mean)
            .unwrap();
        assert_eq!(p.dims().3, 1, "period bins anchor on the first obs");
        let c = cube
            .composite(CompositeWindow::CalendarMonth, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(c.dims().3, 2, "calendar bins split Jan from Feb");
        assert_eq!(c.data()[[0, 0, 0, 0]], 1.0);
        assert_eq!(c.data()[[0, 0, 0, 1]], 3.0);
    }

    #[test]
    fn calendar_month_handles_exact_boundaries_and_leap_years() {
        // exact midnight coordinates around the Jan/Feb boundary, leap year
        assert_eq!(calendar_year_month(fy(2024, 30, 0.0)), (2024, 1)); // Jan 31
        assert_eq!(calendar_year_month(fy(2024, 31, 0.0)), (2024, 2)); // Feb 1
        assert_eq!(calendar_year_month(fy(2024, 59, 0.0)), (2024, 2)); // Feb 29
        assert_eq!(calendar_year_month(fy(2024, 60, 0.0)), (2024, 3)); // Mar 1
        // non-leap year: Mar 1 is doy0 59
        assert_eq!(calendar_year_month(fy(2023, 59, 0.0)), (2023, 3));
        // late in the day stays in its month
        assert_eq!(calendar_year_month(fy(2023, 30, 0.99)), (2023, 1));
        // a coordinate a hair below an exact year boundary is Jan 1
        assert_eq!(calendar_year_month(2024.0 - 1e-14), (2024, 1));
        assert_eq!(calendar_year_month(2023.0), (2023, 1));
        assert_eq!(calendar_year_month(fy(2023, 364, 0.5)), (2023, 12));
    }

    #[test]
    fn calendar_month_separates_same_month_across_years() {
        let times = [fy(2023, 10, 0.0), fy(2024, 10, 0.0)];
        let cube = cube_1px(&[1.0, 5.0], &times);
        let c = cube
            .composite(CompositeWindow::CalendarMonth, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(c.dims().3, 2, "Jan 2023 and Jan 2024 are distinct bins");
    }

    #[test]
    fn calendar_year_composite() {
        let times = [
            fy(2023, 5, 0.0),
            fy(2023, 200, 0.0),
            fy(2024, 5, 0.0),
            fy(2024, 300, 0.0),
        ];
        let cube = cube_1px(&[1.0, 3.0, 10.0, 20.0], &times);
        let c = cube
            .composite(CompositeWindow::CalendarYear, CompositeMethod::Mean)
            .unwrap();
        assert_eq!(c.dims().3, 2);
        assert_abs_diff_eq!(c.data()[[0, 0, 0, 0]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.data()[[0, 0, 0, 1]], 15.0, epsilon = 1e-12);
    }

    #[test]
    fn calendar_composite_rejects_non_finite_times() {
        let cube = cube_1px(&[1.0, 2.0], &[2023.0, f64::NAN]);
        assert!(
            cube.composite(CompositeWindow::CalendarMonth, CompositeMethod::Mean)
                .is_err()
        );
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
