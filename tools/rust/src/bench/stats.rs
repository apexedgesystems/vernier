//! Basic statistics: median and percentiles.
//!
//! Hand-rolled to avoid pulling in a stats crate for two functions.
//! All functions operate on `&[f64]` slices and allocate only for sorting.

/* ----------------------------- Descriptive Stats ----------------------------- */

/// Compute the median of a slice. Returns 0.0 for empty input.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute the p-th percentile (0..100) using linear interpolation.
/// Returns 0.0 for empty input.
pub fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test Median of odd-length array.
    #[test]
    fn median_odd() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-10);
    }

    /// @test Median of even-length array.
    #[test]
    fn median_even() {
        assert!((median(&[4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-10);
    }

    /// @test Median of single element.
    #[test]
    fn median_single() {
        assert!((median(&[42.0]) - 42.0).abs() < 1e-10);
    }

    /// @test Median of empty slice returns 0.
    #[test]
    fn median_empty() {
        assert!((median(&[]) - 0.0).abs() < 1e-10);
    }

    /// @test 50th percentile equals median.
    #[test]
    fn percentile_50_is_median() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - median(&data)).abs() < 1e-10);
    }

    /// @test 0th percentile is minimum.
    #[test]
    fn percentile_0_is_min() {
        let data = [5.0, 3.0, 1.0, 4.0, 2.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
    }

    /// @test 100th percentile is maximum.
    #[test]
    fn percentile_100_is_max() {
        let data = [5.0, 3.0, 1.0, 4.0, 2.0];
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-10);
    }
}
