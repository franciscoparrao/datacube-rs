//! ISO 8601 datetimes → fractional years, without a chrono dependency.

const CUM_DAYS: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Converts `"YYYY-MM-DD"` or `"YYYY-MM-DDTHH:MM:SS[.fff][Z]"` to a
/// fractional year (e.g. `2023-07-02T12:00:00Z` → exactly `2023.5` in a
/// non-leap year). Returns `None` on malformed input.
///
/// Fractional years make the annual cycle have period `1.0`, which is the
/// natural time coordinate for `datacube_core::stats::harmonic_regression`.
pub fn fractional_year(iso: &str) -> Option<f64> {
    let bytes = iso.as_bytes();
    if bytes.len() < 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year: i32 = iso.get(0..4)?.parse().ok()?;
    let month: u32 = iso.get(5..7)?.parse().ok()?;
    let day: u32 = iso.get(8..10)?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut doy = CUM_DAYS[(month - 1) as usize] + day;
    if month > 2 && is_leap(year) {
        doy += 1;
    }

    let mut day_frac = 0.0;
    if bytes.len() > 10 {
        if bytes[10] != b'T' && bytes[10] != b' ' {
            return None;
        }
        let hour: f64 = iso.get(11..13)?.parse().ok()?;
        let minute: f64 = iso.get(14..16).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        // seconds may carry decimals and a trailing Z / offset
        let second: f64 = iso
            .get(17..)
            .map(|rest| {
                let end = rest
                    .find(|c: char| c != '.' && !c.is_ascii_digit())
                    .unwrap_or(rest.len());
                rest[..end].parse().unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        day_frac = (hour * 3600.0 + minute * 60.0 + second) / 86_400.0;
    }

    let days_in_year = if is_leap(year) { 366.0 } else { 365.0 };
    Some(f64::from(year) + (f64::from(doy) - 1.0 + day_frac) / days_in_year)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn year_boundaries() {
        assert_abs_diff_eq!(
            fractional_year("2024-01-01").unwrap(),
            2024.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            fractional_year("2023-12-31T00:00:00Z").unwrap(),
            2023.0 + 364.0 / 365.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn midyear_noon() {
        // non-leap year: Jul 2 12:00 is exactly half the year
        assert_abs_diff_eq!(
            fractional_year("2023-07-02T12:00:00Z").unwrap(),
            2023.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn leap_year_shift() {
        // 2024 is leap: Mar 1 is doy 61
        assert_abs_diff_eq!(
            fractional_year("2024-03-01").unwrap(),
            2024.0 + 60.0 / 366.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn fractional_seconds_and_offsets() {
        let a = fractional_year("2024-06-15T10:30:15.500Z").unwrap();
        let b = fractional_year("2024-06-15T10:30:15.500+00:00").unwrap();
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        assert!(a > 2024.45 && a < 2024.46);
    }

    #[test]
    fn ordering_is_monotonic() {
        let seq = [
            "2023-01-05T14:00:00Z",
            "2023-01-15T14:00:00Z",
            "2023-02-04T14:00:00Z",
            "2024-01-05T14:00:00Z",
        ];
        let ts: Vec<f64> = seq.iter().map(|s| fractional_year(s).unwrap()).collect();
        assert!(ts.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn malformed_inputs() {
        assert!(fractional_year("").is_none());
        assert!(fractional_year("2024").is_none());
        assert!(fractional_year("2024-13-01").is_none());
        assert!(fractional_year("not-a-date").is_none());
    }
}
