//! Two-run comparison: join baseline and candidate CSVs, label median changes.
//!
//! A comparison labels each test by the percentage change of its reported
//! median against a configured threshold. That label describes an observed
//! summary difference between two runs; it is not an inference from
//! observations, and the CSV carries summary statistics rather than the
//! samples such an inference would need. The two CV values travel with the
//! result as context: each is a within-run spread, not a measurement of the
//! spread between the two runs.
//!
//! Every problem is found before a `Comparison` exists, so a comparison either
//! describes both runs or names why it cannot. That holds for a value a
//! CSV does not hold as a number when both runs are loaded with
//! `measured_columns` strict; the lenient loader reads one as zero.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{BenchRow, Classification};

/* ----------------------------- Constants ----------------------------- */

/// The columns a comparison computes from, each with the row field it is read
/// into. `measured_columns` names them for the strict load and `index_rows`
/// refuses a value in them that is not finite, so the two checks cover the
/// same columns.
const MEASUREMENTS: [(&str, Getter); 2] = [
    ("wallMedian", |row| row.wall_median),
    ("wallCV", |row| row.wall_cv),
];

/* ----------------------------- Types ----------------------------- */

/// Reads the value of one measured column out of a row.
type Getter = fn(&BenchRow) -> f64;

/// Which of the two runs a value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    Baseline,
    Candidate,
}

impl fmt::Display for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Run::Baseline => "baseline",
            Run::Candidate => "candidate",
        })
    }
}

/// A comparison that cannot be performed, named by its cause.
#[derive(Debug, Clone, PartialEq)]
pub enum CompareError {
    /// The threshold is negative or not a number.
    InvalidThreshold(f64),
    /// One run reports the same test name more than once.
    DuplicateTest { run: Run, test: String },
    /// A reported measurement is NaN or infinite.
    NonFinite {
        run: Run,
        test: String,
        field: &'static str,
        value: f64,
    },
    /// A reported median is zero or negative.
    NonPositiveMedian { run: Run, test: String, value: f64 },
    /// A reported coefficient of variation is negative.
    NegativeCv { run: Run, test: String, value: f64 },
    /// The two runs share no test name.
    NoComparableTests {
        baseline_only: Vec<String>,
        candidate_only: Vec<String>,
    },
    /// The candidate's median is so many times the baseline's that the
    /// percentage change overflows.
    UnrepresentableChange {
        test: String,
        baseline: f64,
        candidate: f64,
    },
}

impl std::error::Error for CompareError {}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareError::InvalidThreshold(value) => write!(
                f,
                "--threshold {value} is not a usable percentage: expected a finite value of zero or more"
            ),
            CompareError::DuplicateTest { run, test } => write!(
                f,
                "duplicate test identity '{test}' in the {run} results: a comparison needs one row per test"
            ),
            CompareError::NonFinite {
                run,
                test,
                field,
                value,
            } => write!(
                f,
                "{run} {field} for test '{test}' is not a finite number ({value})"
            ),
            CompareError::NonPositiveMedian { run, test, value } => match run {
                Run::Baseline => write!(
                    f,
                    "baseline wallMedian for test '{test}' is {value}: the relative change against a baseline of zero or less is undefined"
                ),
                Run::Candidate => write!(
                    f,
                    "candidate wallMedian for test '{test}' is {value}: a measured median is greater than zero"
                ),
            },
            CompareError::NegativeCv { run, test, value } => write!(
                f,
                "{run} wallCV for test '{test}' is negative ({value})"
            ),
            CompareError::NoComparableTests {
                baseline_only,
                candidate_only,
            } => write!(
                f,
                "no tests are present in both runs: {} only in the baseline ({}), {} only in the candidate ({})",
                baseline_only.len(),
                baseline_only.join(", "),
                candidate_only.len(),
                candidate_only.join(", ")
            ),
            CompareError::UnrepresentableChange {
                test,
                baseline,
                candidate,
            } => write!(
                f,
                "wallMedian for test '{test}' goes from {baseline:e} in the baseline to {candidate:e} in the candidate: the percentage change is too large to represent"
            ),
        }
    }
}

/// Comparison result for one test that both runs report.
#[derive(Debug, Clone)]
pub struct CompareResult {
    pub test: String,
    pub baseline_median: f64,
    pub candidate_median: f64,
    pub delta_us: f64,
    pub delta_pct: f64,
    pub classification: Classification,
    /// The p-value of a test for statistical significance, when one is
    /// available. A comparison of two summary CSVs has no observations to
    /// run such a test on, so this is `None`.
    pub p_value: Option<f64>,
    pub baseline_cv: f64,
    pub candidate_cv: f64,
}

/// One comparison of two runs: the tests both report, and the tests only one
/// of them reports.
///
/// The table, the Markdown table, the JSON document and the exit decision are
/// all read from this one value.
#[derive(Debug, Clone)]
pub struct Comparison {
    /// The percentage change beyond which a test is labelled.
    pub threshold_pct: f64,
    /// One result per test present in both runs, ordered by test name.
    pub results: Vec<CompareResult>,
    /// Tests the baseline reports and the candidate does not, ordered by
    /// name. Under `--fail-on-regression` these fail the gate.
    pub baseline_only: Vec<String>,
    /// Tests the candidate reports and the baseline does not, ordered by
    /// name. These are new and have no baseline to be judged against.
    pub candidate_only: Vec<String>,
}

impl Comparison {
    /// The number of results labelled as a regression.
    pub fn regression_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.classification == Classification::Regression)
            .count()
    }
}

/* ----------------------------- Helpers ----------------------------- */

/// Label a percentage change against the threshold.
///
/// A positive change means the candidate's median is the larger of the two.
/// A change of exactly the threshold, judged on the medians as the CSV
/// reports them, is inside it and stays neutral. Binary floating point holds
/// most decimals only approximately (1 to 1.05 computes as
/// 5.000000000000004%), and rounding the two medians, the threshold and the
/// arithmetic moves the change by no more than about
/// `EPSILON * (100 + 3 * |change|)`. A change is labelled only when it passes
/// the threshold by more than twice that: under 2e-13 percentage points for
/// thresholds up to 100%, far finer than the six significant digits a CSV
/// gives a median. The margin never exceeds the threshold, so with a zero
/// threshold only medians that parse to the same value are neutral.
fn classify(delta_pct: f64, threshold: f64) -> Classification {
    let margin = (2.0 * f64::EPSILON * (100.0 + 3.0 * delta_pct.abs())).min(threshold);
    if delta_pct > threshold + margin {
        Classification::Regression
    } else if delta_pct < -(threshold + margin) {
        Classification::Improvement
    } else {
        Classification::Neutral
    }
}

/// Index one run's rows by test name, rejecting unusable input.
fn index_rows(rows: &[BenchRow], run: Run) -> Result<BTreeMap<&str, &BenchRow>, CompareError> {
    let mut indexed: BTreeMap<&str, &BenchRow> = BTreeMap::new();

    for row in rows {
        for (field, value_of) in MEASUREMENTS {
            let value = value_of(row);
            if !value.is_finite() {
                return Err(CompareError::NonFinite {
                    run,
                    test: row.test.clone(),
                    field,
                    value,
                });
            }
        }
        if row.wall_median <= 0.0 {
            return Err(CompareError::NonPositiveMedian {
                run,
                test: row.test.clone(),
                value: row.wall_median,
            });
        }
        if row.wall_cv < 0.0 {
            return Err(CompareError::NegativeCv {
                run,
                test: row.test.clone(),
                value: row.wall_cv,
            });
        }
        if indexed.insert(row.test.as_str(), row).is_some() {
            return Err(CompareError::DuplicateTest {
                run,
                test: row.test.clone(),
            });
        }
    }

    Ok(indexed)
}

/// The names in `from` that `other` does not report, ordered by name.
fn only_in(from: &BTreeMap<&str, &BenchRow>, other: &BTreeMap<&str, &BenchRow>) -> Vec<String> {
    from.keys()
        .filter(|name| !other.contains_key(*name))
        .map(|name| (*name).to_string())
        .collect()
}

/* ----------------------------- API ----------------------------- */

/// The CSV columns a comparison computes from.
///
/// Load both runs with these columns strict (`load_csv_strict`): the lenient
/// loader reads a missing, empty or unparsable field as 0.0, which a
/// comparison cannot tell from a measured value.
pub fn measured_columns() -> Vec<&'static str> {
    MEASUREMENTS.iter().map(|(column, _)| *column).collect()
}

/// Compare two runs, joining on test name.
///
/// `threshold` is the percentage change (for example 5.0 for 5%) beyond which
/// a test is labelled a regression or an improvement.
///
/// Returns the named cause when the two runs cannot be compared: an unusable
/// threshold, a duplicate test identity, a measurement that is not a finite
/// positive number, no test in common, or a percentage change too large to
/// represent.
pub fn compare_runs(
    baseline: &[BenchRow],
    candidate: &[BenchRow],
    threshold: f64,
) -> Result<Comparison, CompareError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(CompareError::InvalidThreshold(threshold));
    }

    let base_map = index_rows(baseline, Run::Baseline)?;
    let cand_map = index_rows(candidate, Run::Candidate)?;

    let common: BTreeSet<&str> = base_map
        .keys()
        .filter(|name| cand_map.contains_key(*name))
        .copied()
        .collect();

    let baseline_only = only_in(&base_map, &cand_map);
    let candidate_only = only_in(&cand_map, &base_map);

    if common.is_empty() {
        return Err(CompareError::NoComparableTests {
            baseline_only,
            candidate_only,
        });
    }

    let results = common
        .iter()
        .map(|&test| {
            let b = base_map[test];
            let c = cand_map[test];

            // Two finite positive medians differ by a finite amount, and a
            // fall is at most 100%, but a rise can overflow the percentage.
            let delta_us = c.wall_median - b.wall_median;
            let delta_pct = (delta_us / b.wall_median) * 100.0;
            if !delta_pct.is_finite() {
                return Err(CompareError::UnrepresentableChange {
                    test: test.to_string(),
                    baseline: b.wall_median,
                    candidate: c.wall_median,
                });
            }

            Ok(CompareResult {
                test: test.to_string(),
                baseline_median: b.wall_median,
                candidate_median: c.wall_median,
                delta_us,
                delta_pct,
                classification: classify(delta_pct, threshold),
                p_value: None,
                baseline_cv: b.wall_cv,
                candidate_cv: c.wall_cv,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Comparison {
        threshold_pct: threshold,
        results,
        baseline_only,
        candidate_only,
    })
}

/// Returns true if any result is labelled as a regression.
pub fn has_regressions(results: &[CompareResult]) -> bool {
    results
        .iter()
        .any(|r| r.classification == Classification::Regression)
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
            wall_p99: median * 1.18,
            wall_p999: median * 1.19,
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

    fn compare_one(baseline: f64, candidate: f64, threshold: f64) -> CompareResult {
        let base = vec![make_row("Test.A", baseline, 0.02)];
        let cand = vec![make_row("Test.A", candidate, 0.03)];
        let mut comparison = compare_runs(&base, &cand, threshold).expect("comparable");
        assert_eq!(comparison.results.len(), 1);
        comparison.results.remove(0)
    }

    /// @test A change of exactly the threshold, either way, stays neutral.
    #[test]
    fn classify_at_the_threshold_is_neutral() {
        assert_eq!(classify(5.0, 5.0), Classification::Neutral);
        assert_eq!(classify(-5.0, 5.0), Classification::Neutral);
        assert_eq!(classify(0.0, 5.0), Classification::Neutral);
        assert_eq!(classify(-0.0, 5.0), Classification::Neutral);
    }

    /// @test A change beyond the threshold is labelled; rounding past it is not.
    #[test]
    fn classify_separates_a_real_change_from_rounding() {
        // 1 -> 1.05 computes as 5.000000000000004%: rounding, not a change.
        assert_eq!(classify(5.000000000000004, 5.0), Classification::Neutral);
        assert_eq!(classify(-5.000000000000004, 5.0), Classification::Neutral);
        // 1 -> 1.0500001 is +5.00001%, beyond the threshold.
        assert_eq!(classify(5.00001, 5.0), Classification::Regression);
        assert_eq!(classify(-5.00001, 5.0), Classification::Improvement);
    }

    /// @test A zero threshold labels every change that is not exactly zero.
    #[test]
    fn classify_with_a_zero_threshold() {
        assert_eq!(classify(0.0, 0.0), Classification::Neutral);
        assert_eq!(classify(f64::MIN_POSITIVE, 0.0), Classification::Regression);
        assert_eq!(
            classify(-f64::MIN_POSITIVE, 0.0),
            Classification::Improvement
        );
    }

    /// Parse `mantissa`e`exponent` as the loader parses a CSV field.
    fn decimal(mantissa: u64, exponent: i32) -> f64 {
        format!("{mantissa}e{exponent}").parse().expect("a decimal")
    }

    /// The label `compare_runs` gives the change between two medians.
    fn label(baseline: f64, candidate: f64, threshold: f64) -> Classification {
        compare_one(baseline, candidate, threshold).classification
    }

    /// @test A median exactly at a decimal boundary is neutral; one unit past it is labelled.
    #[test]
    fn decimal_threshold_boundary_holds() {
        // (mantissa, decimal places): 1%, 2.5%, 5%, 7.5%, 10%, 12.5%, 33.3%, 50%, 99%.
        let thresholds: [(u64, u32); 9] = [
            (1, 0),
            (25, 1),
            (5, 0),
            (75, 1),
            (10, 0),
            (125, 1),
            (333, 1),
            (50, 0),
            (99, 0),
        ];
        // Six significant digits, as the CSV writer prints a median.
        let baselines: Vec<u64> = (100_003..1_000_000).step_by(8_999).collect();
        for (t_mantissa, t_places) in thresholds {
            let threshold = decimal(t_mantissa, -(t_places as i32));
            let hundred = 100 * 10u64.pow(t_places);
            for &b_mantissa in &baselines {
                for b_exponent in [-9, -3, 0, 3] {
                    let baseline = decimal(b_mantissa, b_exponent);
                    // candidate = baseline * (100 +/- T) / 100, written exactly.
                    let exponent = b_exponent - t_places as i32 - 2;
                    let up = b_mantissa * (hundred + t_mantissa);
                    let down = b_mantissa * (hundred - t_mantissa);
                    for (mantissa, want) in [
                        (up, Classification::Neutral),
                        (up + 1, Classification::Regression),
                        (up - 1, Classification::Neutral),
                        (down, Classification::Neutral),
                        (down - 1, Classification::Improvement),
                        (down + 1, Classification::Neutral),
                    ] {
                        let candidate = decimal(mantissa, exponent);
                        assert_eq!(
                            label(baseline, candidate, threshold),
                            want,
                            "{b_mantissa}e{b_exponent} -> {mantissa}e{exponent} at {threshold}%"
                        );
                    }
                }
            }
        }
    }

    /// @test The rounding margin hides no change a fifteen-digit median can express.
    #[test]
    fn rounding_margin_hides_no_real_change() {
        for b_mantissa in [
            100_000_000_001u64,
            123_456_789_012,
            555_555_555_557,
            999_999_999_989,
        ] {
            let baseline = decimal(b_mantissa, -12);
            let (up, down) = (b_mantissa * 105, b_mantissa * 95);
            let at = |mantissa: u64| label(baseline, decimal(mantissa, -14), 5.0);
            assert_eq!(at(up), Classification::Neutral, "{b_mantissa}");
            assert_eq!(at(up + 1), Classification::Regression, "{b_mantissa}");
            assert_eq!(at(down), Classification::Neutral, "{b_mantissa}");
            assert_eq!(at(down - 1), Classification::Improvement, "{b_mantissa}");
        }
    }

    /// @test With a zero threshold the smallest representable change is labelled.
    #[test]
    fn zero_threshold_labels_the_smallest_change() {
        let one = decimal(1, 0);
        let above = decimal(10_000_000_000_000_002, -16);
        let below = decimal(9_999_999_999_999_999, -16);
        assert_eq!(above, 1.0 + f64::EPSILON);
        assert_eq!(below, 1.0 - f64::EPSILON / 2.0);
        assert_eq!(label(one, above, 0.0), Classification::Regression);
        assert_eq!(label(one, below, 0.0), Classification::Improvement);
        assert_eq!(label(one, one, 0.0), Classification::Neutral);
        // The same changes are rounding-sized against any threshold above zero.
        assert_eq!(label(one, above, 5.0), Classification::Neutral);
        assert_eq!(label(one, below, 5.0), Classification::Neutral);
    }

    /// @test A larger threshold covers a change a smaller one labels.
    #[test]
    fn classify_follows_the_configured_threshold() {
        assert_eq!(classify(10.0, 5.0), Classification::Regression);
        assert_eq!(classify(10.0, 10.0), Classification::Neutral);
        assert_eq!(classify(10.0, 20.0), Classification::Neutral);
        assert_eq!(classify(-10.0, 5.0), Classification::Improvement);
        assert_eq!(classify(-10.0, 20.0), Classification::Neutral);
    }

    /// @test A slower median is a regression and carries its percentage.
    #[test]
    fn slower_median_is_a_regression() {
        let result = compare_one(0.05, 0.10, 5.0);
        assert_eq!(result.classification, Classification::Regression);
        assert!((result.delta_pct - 100.0).abs() < 1e-9, "{result:?}");
        assert!((result.delta_us - 0.05).abs() < 1e-12, "{result:?}");
    }

    /// @test A faster median is an improvement.
    #[test]
    fn faster_median_is_an_improvement() {
        let result = compare_one(0.10, 0.05, 5.0);
        assert_eq!(result.classification, Classification::Improvement);
        assert!((result.delta_pct + 50.0).abs() < 1e-9, "{result:?}");
    }

    /// @test An equal median is neutral and its change is zero.
    #[test]
    fn equal_medians_are_neutral() {
        let result = compare_one(0.05, 0.05, 5.0);
        assert_eq!(result.classification, Classification::Neutral);
        assert!(result.delta_pct.abs() < 1e-12, "{result:?}");
    }

    /// @test A risen median with unchanged outer quantiles is a regression.
    #[test]
    fn quantile_contradiction_follows_the_median() {
        // The reported median moves; the reported outer quantiles do not.
        let mut base = make_row("Test.A", 100.0, 0.15);
        let mut cand = make_row("Test.A", 120.0, 0.15);
        for row in [&mut base, &mut cand] {
            row.wall_p10 = 90.0;
            row.wall_p90 = 130.0;
            row.wall_min = 80.0;
            row.wall_max = 140.0;
        }
        let comparison = compare_runs(&[base], &[cand], 5.0).expect("comparable");
        assert_eq!(
            comparison.results[0].classification,
            Classification::Regression
        );
        assert!((comparison.results[0].delta_pct - 20.0).abs() < 1e-9);
    }

    /// @test A result carries both CV values and no p-value.
    #[test]
    fn result_carries_cv_context_and_no_p_value() {
        let result = compare_one(0.05, 0.10, 5.0);
        assert_eq!(result.p_value, None);
        assert!((result.baseline_cv - 0.02).abs() < 1e-12);
        assert!((result.candidate_cv - 0.03).abs() < 1e-12);
    }

    /// @test A baseline median of zero is an error, not a neutral 0%.
    #[test]
    fn zero_baseline_median_is_an_error() {
        let base = vec![make_row("Test.A", 0.0, 0.02)];
        let cand = vec![make_row("Test.A", 0.05, 0.02)];
        assert_eq!(
            compare_runs(&base, &cand, 5.0).unwrap_err(),
            CompareError::NonPositiveMedian {
                run: Run::Baseline,
                test: "Test.A".to_string(),
                value: 0.0,
            }
        );
    }

    /// @test A candidate median of zero is an invalid measurement.
    #[test]
    fn zero_candidate_median_is_an_error() {
        let base = vec![make_row("Test.A", 0.05, 0.02)];
        let cand = vec![make_row("Test.A", 0.0, 0.02)];
        assert_eq!(
            compare_runs(&base, &cand, 5.0).unwrap_err(),
            CompareError::NonPositiveMedian {
                run: Run::Candidate,
                test: "Test.A".to_string(),
                value: 0.0,
            }
        );
    }

    /// @test A negative median is an invalid measurement.
    #[test]
    fn negative_median_is_an_error() {
        let base = vec![make_row("Test.A", -0.05, 0.02)];
        let cand = vec![make_row("Test.A", 0.05, 0.02)];
        assert!(matches!(
            compare_runs(&base, &cand, 5.0),
            Err(CompareError::NonPositiveMedian {
                run: Run::Baseline,
                ..
            })
        ));
    }

    /// @test A non-finite measurement is an error, whichever run reports it.
    #[test]
    fn non_finite_measurement_is_an_error() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let good = vec![make_row("Test.A", 0.05, 0.02)];

            let mut base = vec![make_row("Test.A", 0.05, 0.02)];
            base[0].wall_median = bad;
            assert!(
                matches!(
                    compare_runs(&base, &good, 5.0).unwrap_err(),
                    CompareError::NonFinite {
                        run: Run::Baseline,
                        field: "wallMedian",
                        ..
                    }
                ),
                "baseline median {bad}"
            );

            let mut cand = vec![make_row("Test.A", 0.05, 0.02)];
            cand[0].wall_cv = bad;
            assert!(
                matches!(
                    compare_runs(&good, &cand, 5.0),
                    Err(CompareError::NonFinite {
                        run: Run::Candidate,
                        field: "wallCV",
                        ..
                    })
                ),
                "candidate CV {bad}"
            );
        }
    }

    /// @test A negative CV is an invalid measurement.
    #[test]
    fn negative_cv_is_an_error() {
        let base = vec![make_row("Test.A", 0.05, -0.02)];
        let cand = vec![make_row("Test.A", 0.05, 0.02)];
        assert!(matches!(
            compare_runs(&base, &cand, 5.0),
            Err(CompareError::NegativeCv {
                run: Run::Baseline,
                ..
            })
        ));
    }

    /// @test A test name reported twice in one run is an error.
    #[test]
    fn duplicate_test_identity_is_an_error() {
        let twice = vec![
            make_row("Test.A", 0.05, 0.02),
            make_row("Test.A", 0.10, 0.02),
        ];
        let once = vec![make_row("Test.A", 0.05, 0.02)];
        assert_eq!(
            compare_runs(&twice, &once, 5.0).unwrap_err(),
            CompareError::DuplicateTest {
                run: Run::Baseline,
                test: "Test.A".to_string(),
            }
        );
        assert_eq!(
            compare_runs(&once, &twice, 5.0).unwrap_err(),
            CompareError::DuplicateTest {
                run: Run::Candidate,
                test: "Test.A".to_string(),
            }
        );
    }

    /// @test An unusable threshold is an error.
    #[test]
    fn unusable_threshold_is_an_error() {
        let rows = vec![make_row("Test.A", 0.05, 0.02)];
        for bad in [-5.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    compare_runs(&rows, &rows, bad),
                    Err(CompareError::InvalidThreshold(_))
                ),
                "threshold {bad}"
            );
        }
    }

    /// @test A rise too large for a percentage is an error; the same fall is -100%.
    #[test]
    fn unrepresentable_change_is_an_error() {
        let tiny = vec![make_row("Test.A", 1e-300, 0.02)];
        let huge = vec![make_row("Test.A", 1e300, 0.02)];
        assert_eq!(
            compare_runs(&tiny, &huge, 5.0).unwrap_err(),
            CompareError::UnrepresentableChange {
                test: "Test.A".to_string(),
                baseline: 1e-300,
                candidate: 1e300,
            }
        );
        let fall = compare_runs(&huge, &tiny, 5.0).expect("a fall is representable");
        assert_eq!(fall.results[0].delta_pct, -100.0);
        assert_eq!(fall.results[0].classification, Classification::Improvement);
    }

    /// @test Two runs with no test in common are an error naming both sides.
    #[test]
    fn disjoint_runs_are_an_error() {
        let base = vec![
            make_row("Test.A", 0.05, 0.02),
            make_row("Test.B", 0.05, 0.02),
        ];
        let cand = vec![make_row("Test.C", 0.05, 0.02)];
        assert_eq!(
            compare_runs(&base, &cand, 5.0).unwrap_err(),
            CompareError::NoComparableTests {
                baseline_only: vec!["Test.A".to_string(), "Test.B".to_string()],
                candidate_only: vec!["Test.C".to_string()],
            }
        );
    }

    /// @test A partial overlap compares the intersection and reports the rest.
    #[test]
    fn partial_overlap_reports_both_unmatched_lists() {
        let base = vec![
            make_row("Test.Gone", 0.05, 0.02),
            make_row("Test.Kept", 0.05, 0.02),
        ];
        let cand = vec![
            make_row("Test.Kept", 0.05, 0.02),
            make_row("Test.New", 0.05, 0.02),
        ];
        let comparison = compare_runs(&base, &cand, 5.0).expect("comparable");
        assert_eq!(comparison.results.len(), 1);
        assert_eq!(comparison.results[0].test, "Test.Kept");
        assert_eq!(comparison.baseline_only, vec!["Test.Gone".to_string()]);
        assert_eq!(comparison.candidate_only, vec!["Test.New".to_string()]);
        assert_eq!(comparison.regression_count(), 0);
    }

    /// @test A renamed test is one missing test plus one new test.
    #[test]
    fn rename_is_missing_plus_new() {
        let base = vec![
            make_row("Test.Kept", 0.05, 0.02),
            make_row("Test.OldName", 0.05, 0.02),
        ];
        let cand = vec![
            make_row("Test.Kept", 0.05, 0.02),
            make_row("Test.NewName", 0.05, 0.02),
        ];
        let comparison = compare_runs(&base, &cand, 5.0).expect("comparable");
        assert_eq!(comparison.baseline_only, vec!["Test.OldName".to_string()]);
        assert_eq!(comparison.candidate_only, vec!["Test.NewName".to_string()]);
    }

    /// @test Results are ordered by test name whatever order the rows arrive in.
    #[test]
    fn results_are_ordered_by_test_name() {
        let base = vec![
            make_row("Test.C", 0.05, 0.02),
            make_row("Test.A", 0.05, 0.02),
            make_row("Test.B", 0.05, 0.02),
        ];
        let cand = base.clone();
        let comparison = compare_runs(&base, &cand, 5.0).expect("comparable");
        let names: Vec<&str> = comparison.results.iter().map(|r| r.test.as_str()).collect();
        assert_eq!(names, ["Test.A", "Test.B", "Test.C"]);
    }

    /// @test has_regressions follows the labels.
    #[test]
    fn has_regressions_follows_the_labels() {
        let base = vec![make_row("Test.A", 0.05, 0.02)];
        let slower = vec![make_row("Test.A", 0.10, 0.02)];
        let same = vec![make_row("Test.A", 0.05, 0.02)];

        let regressed = compare_runs(&base, &slower, 5.0).expect("comparable");
        assert!(has_regressions(&regressed.results));
        assert_eq!(regressed.regression_count(), 1);

        let neutral = compare_runs(&base, &same, 5.0).expect("comparable");
        assert!(!has_regressions(&neutral.results));
        assert_eq!(neutral.regression_count(), 0);
    }
}
