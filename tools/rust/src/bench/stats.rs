//! Basic statistics: median, percentiles, and Mann-Whitney U test.
//!
//! Hand-rolled to avoid pulling in a stats crate for 3 functions.
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

/* ----------------------------- Mann-Whitney U ----------------------------- */

/// Two-tailed Mann-Whitney U test. Returns an approximate p-value.
///
/// Uses the normal approximation with continuity correction, which is
/// adequate for n >= 8 (our typical repeat count is 5-20).
///
/// Returns 1.0 if either sample is empty or all values are identical.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return 1.0;
    }

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(na + nb);
    for &v in a {
        combined.push((v, 0));
    }
    for &v in b {
        combined.push((v, 1));
    }
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (handle ties by averaging)
    let n = combined.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (combined[j].0 - combined[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for item in ranks.iter_mut().take(j).skip(i) {
            *item = avg_rank;
        }
        i = j;
    }

    // Sum ranks for group A
    let rank_sum_a: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|(item, _)| item.1 == 0)
        .map(|(_, &r)| r)
        .sum();

    let u_a = rank_sum_a - (na as f64 * (na as f64 + 1.0)) / 2.0;
    let u_b = (na as f64) * (nb as f64) - u_a;
    let u = u_a.min(u_b);

    let mean_u = (na as f64 * nb as f64) / 2.0;
    let std_u = ((na as f64 * nb as f64 * (na + nb + 1) as f64) / 12.0).sqrt();

    if std_u < f64::EPSILON {
        return 1.0; // All identical values
    }

    // Normal approximation with continuity correction
    let z = ((u - mean_u).abs() - 0.5) / std_u;

    // Two-tailed p-value from z-score (using rational approximation of erfc)
    two_tailed_p(z)
}

/// Approximate two-tailed p-value from a z-score using the Abramowitz & Stegun
/// rational approximation (formula 26.2.17). Accurate to ~1e-5.
fn two_tailed_p(z: f64) -> f64 {
    let z = z.abs();
    if z > 8.0 {
        return 0.0; // Effectively zero
    }

    const B1: f64 = 0.319381530;
    const B2: f64 = -0.356563782;
    const B3: f64 = 1.781477937;
    const B4: f64 = -1.821255978;
    const B5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    let t = 1.0 / (1.0 + P * z);
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * B5))));
    let one_tail = 1.0 - cdf;

    (2.0 * one_tail).min(1.0)
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

    /// @test Mann-Whitney U with identical samples gives p near 1.
    #[test]
    fn mann_whitney_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = mann_whitney_u(&a, &b);
        assert!(p > 0.05, "p={p} should be > 0.05 for identical samples");
    }

    /// @test Mann-Whitney U with clearly separated samples gives p near 0.
    #[test]
    fn mann_whitney_separated() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let p = mann_whitney_u(&a, &b);
        assert!(p < 0.01, "p={p} should be < 0.01 for separated samples");
    }

    /// @test Mann-Whitney U with empty input returns 1.0.
    #[test]
    fn mann_whitney_empty() {
        assert!((mann_whitney_u(&[], &[1.0, 2.0]) - 1.0).abs() < 1e-10);
        assert!((mann_whitney_u(&[1.0, 2.0], &[]) - 1.0).abs() < 1e-10);
    }
}
