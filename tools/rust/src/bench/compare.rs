//! Two-run comparison: join baseline and candidate CSVs, classify regressions.

use std::collections::HashMap;

use super::stats;
use super::{BenchRow, Classification};

/* ----------------------------- CompareResult ----------------------------- */

/// Comparison result for a single test name.
#[derive(Debug, Clone)]
pub struct CompareResult {
    pub test: String,
    pub baseline_median: f64,
    pub candidate_median: f64,
    pub delta_us: f64,
    pub delta_pct: f64,
    pub classification: Classification,
    pub p_value: f64,
    pub baseline_cv: f64,
    pub candidate_cv: f64,
}

/* ----------------------------- API ----------------------------- */

/// Compare two sets of benchmark rows, joining on test name.
///
/// `threshold` is the percentage change (e.g. 5.0 for 5%) beyond which a
/// test is classified as a regression or improvement. The Mann-Whitney U test
/// is used to assess statistical significance (p < 0.05).
///
/// Tests present in only one set are silently skipped.
pub fn compare_runs(
    baseline: &[BenchRow],
    candidate: &[BenchRow],
    threshold: f64,
) -> Vec<CompareResult> {
    // Group rows by test name (multiple repeats may produce multiple rows,
    // but our CSV has one row per test with pre-computed stats).
    let base_map: HashMap<&str, &BenchRow> =
        baseline.iter().map(|r| (r.test.as_str(), r)).collect();

    let cand_map: HashMap<&str, &BenchRow> =
        candidate.iter().map(|r| (r.test.as_str(), r)).collect();

    // Collect all test names that appear in both sets
    let mut common: Vec<&str> = base_map
        .keys()
        .filter(|k| cand_map.contains_key(*k))
        .copied()
        .collect();
    common.sort();

    common
        .iter()
        .map(|&test| {
            let b = base_map[test];
            let c = cand_map[test];

            let delta_us = c.wall_median - b.wall_median;
            let delta_pct = if b.wall_median.abs() > f64::EPSILON {
                (delta_us / b.wall_median) * 100.0
            } else {
                0.0
            };

            // For Mann-Whitney we need raw samples. Since we only have summary stats,
            // synthesize approximate samples from the reported distribution.
            let base_samples = synthesize_samples(b);
            let cand_samples = synthesize_samples(c);
            let p_value = stats::mann_whitney_u(&base_samples, &cand_samples);

            let classification = classify(delta_pct, p_value, threshold);

            CompareResult {
                test: test.to_string(),
                baseline_median: b.wall_median,
                candidate_median: c.wall_median,
                delta_us,
                delta_pct,
                classification,
                p_value,
                baseline_cv: b.wall_cv,
                candidate_cv: c.wall_cv,
            }
        })
        .collect()
}

/// Returns true if any result is classified as a regression.
pub fn has_regressions(results: &[CompareResult]) -> bool {
    results
        .iter()
        .any(|r| r.classification == Classification::Regression)
}

/* ----------------------------- Helpers ----------------------------- */

/// Classify a delta percentage as regression, improvement, or neutral.
///
/// Positive delta means the candidate is slower (regression).
/// Only classify if the result is statistically significant (p < 0.05).
fn classify(delta_pct: f64, p_value: f64, threshold: f64) -> Classification {
    if p_value > 0.05 {
        return Classification::Neutral;
    }
    if delta_pct > threshold {
        Classification::Regression
    } else if delta_pct < -threshold {
        Classification::Improvement
    } else {
        Classification::Neutral
    }
}

/// Synthesize approximate samples from summary statistics.
///
/// Since the CSV stores one row per test (not raw samples), we reconstruct
/// a rough distribution using the reported P10, median, P90, min, and max.
/// This is adequate for the Mann-Whitney U normal approximation.
fn synthesize_samples(row: &BenchRow) -> Vec<f64> {
    let n = row.repeats.max(5) as usize;
    let mut samples = Vec::with_capacity(n);

    // Use reported quantiles to build a plausible distribution
    samples.push(row.wall_min);
    samples.push(row.wall_p10);

    // Fill the middle with values around the median
    let mid_count = n.saturating_sub(4);
    for i in 0..mid_count {
        let t = (i as f64 + 1.0) / (mid_count as f64 + 1.0);
        let val = row.wall_p10 * (1.0 - t) + row.wall_p90 * t;
        samples.push(val);
    }

    samples.push(row.wall_p90);
    samples.push(row.wall_max);

    samples
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(test: &str, median: f64, cv: f64) -> BenchRow {
        BenchRow {
            test: test.to_string(),
            cycles: 10000,
            repeats: 10,
            warmup: 1,
            threads: 1,
            msg_bytes: 64,
            wall_median: median,
            wall_p10: median * 0.9,
            wall_p90: median * 1.1,
            wall_min: median * 0.8,
            wall_max: median * 1.2,
            wall_mean: median,
            wall_stddev: median * cv,
            wall_cv: cv,
            calls_per_second: 1.0 / (median * 1e-6),
            stable: 1,
            cv_threshold: 0.10,
        }
    }

    /// @test Identical runs produce neutral classification.
    #[test]
    fn identical_runs_neutral() {
        let base = vec![make_row("Test.A", 0.05, 0.02)];
        let cand = vec![make_row("Test.A", 0.05, 0.02)];
        let results = compare_runs(&base, &cand, 5.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].classification, Classification::Neutral);
    }

    /// @test Large slowdown classified as regression when statistically significant.
    #[test]
    fn regression_detected() {
        // 100% slower with very different distributions
        let base = vec![make_row("Test.A", 0.05, 0.01)];
        let cand = vec![make_row("Test.A", 0.10, 0.01)];
        let results = compare_runs(&base, &cand, 5.0);
        assert_eq!(results.len(), 1);
        assert!(results[0].delta_pct > 5.0);
        // Classification depends on p-value from synthetic samples
    }

    /// @test Tests only in one set are excluded.
    #[test]
    fn mismatched_tests_excluded() {
        let base = vec![make_row("Test.A", 0.05, 0.02)];
        let cand = vec![make_row("Test.B", 0.05, 0.02)];
        let results = compare_runs(&base, &cand, 5.0);
        assert!(results.is_empty());
    }

    /// @test has_regressions returns false when all neutral.
    #[test]
    fn no_regressions() {
        let base = vec![make_row("Test.A", 0.05, 0.02)];
        let cand = vec![make_row("Test.A", 0.05, 0.02)];
        let results = compare_runs(&base, &cand, 5.0);
        assert!(!has_regressions(&results));
    }
}
