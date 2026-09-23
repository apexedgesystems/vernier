//! Output formatting: terminal tables, markdown, and JSON.
//!
//! Uses ANSI escape codes for colored terminal output.

use super::{BenchRow, Classification, Comparison, Need, SortColumn};

/* ----------------------------- ANSI Colors ----------------------------- */

const GREEN: &str = "\x1b[92m";
const RED: &str = "\x1b[91m";
const YELLOW: &str = "\x1b[93m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/* ----------------------------- Comparison Table ----------------------------- */

/// What the labels mean, in the lines a terminal shows them on.
fn labelling_rule(threshold_pct: f64) -> [String; 3] {
    [
        format!("Labels compare the median change against the {threshold_pct:.1}% threshold."),
        "They describe the difference between two runs, not a significance test;".to_string(),
        "the CV of each run is its own spread, not the spread between the runs.".to_string(),
    ]
}

/// Print a colored comparison table to stdout.
pub fn print_comparison_table(comparison: &Comparison) {
    let results = &comparison.results;

    // Calculate column widths
    let name_width = results
        .iter()
        .map(|r| r.test.len())
        .max()
        .unwrap_or(4)
        .max(4);

    // Header
    println!();
    println!(
        "{BOLD}{:<width$}  {:>12}  {:>12}  {:>10}  {:>8}  {:>8}  {:>8}  {:>12}{RESET}",
        "Test",
        "Baseline",
        "Candidate",
        "Delta",
        "%",
        "Base CV",
        "Cand CV",
        "Result",
        width = name_width
    );
    println!(
        "{:-<width$}  {:-<12}  {:-<12}  {:-<10}  {:-<8}  {:-<8}  {:-<8}  {:-<12}",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        width = name_width
    );

    for r in results {
        let (color, label) = match r.classification {
            Classification::Regression => (RED, "REGRESSION"),
            Classification::Improvement => (GREEN, "IMPROVEMENT"),
            Classification::Neutral => ("", "neutral"),
        };

        println!(
            "{color}{:<width$}  {:>12.5}  {:>12.5}  {:>+10.5}  {:>+7.1}%  {:>7.1}%  {:>7.1}%  {:<12}{RESET}",
            r.test,
            r.baseline_median,
            r.candidate_median,
            r.delta_us,
            r.delta_pct,
            r.baseline_cv * 100.0,
            r.candidate_cv * 100.0,
            label,
            width = name_width,
            color = color,
        );
    }

    // Summary line
    let reg_count = comparison.regression_count();
    let imp_count = results
        .iter()
        .filter(|r| r.classification == Classification::Improvement)
        .count();
    let neu_count = results
        .iter()
        .filter(|r| r.classification == Classification::Neutral)
        .count();

    println!();
    print!("  ");
    if reg_count > 0 {
        print!("{RED}{reg_count} regression(s){RESET}  ");
    }
    if imp_count > 0 {
        print!("{GREEN}{imp_count} improvement(s){RESET}  ");
    }
    println!("{neu_count} neutral");

    if !comparison.baseline_only.is_empty() {
        println!(
            "  {YELLOW}Missing from the candidate ({}): {}{RESET}",
            comparison.baseline_only.len(),
            comparison.baseline_only.join(", ")
        );
    }
    if !comparison.candidate_only.is_empty() {
        println!(
            "  New in the candidate ({}): {}",
            comparison.candidate_only.len(),
            comparison.candidate_only.join(", ")
        );
    }

    println!();
    for line in labelling_rule(comparison.threshold_pct) {
        println!("  {line}");
    }
}

/* ----------------------------- Summary Table ----------------------------- */

/// The columns the summary table and the summary JSON present, and how each
/// is needed. A results CSV carries the three measurements in every row; the
/// others are shown where a row gives them and take their defaults where a
/// layout or a row leaves them out, but a value that is there must be a
/// finite number of its kind. Rows for a summary are loaded with these.
pub const SUMMARY_COLUMNS: [(&str, Need); 9] = [
    ("wallMedian", Need::Required),
    ("wallCV", Need::Required),
    ("callsPerSecond", Need::Required),
    ("wallP10", Need::IfPresent),
    ("wallP90", Need::IfPresent),
    ("stable", Need::IfPresent),
    ("cvThreshold", Need::IfPresent),
    ("cycles", Need::IfPresent),
    ("repeats", Need::IfPresent),
];

/// Print a single-CSV summary table to stdout.
pub fn print_summary_table(rows: &[BenchRow], sort: SortColumn) {
    if rows.is_empty() {
        println!("No rows to display.");
        return;
    }

    let mut sorted: Vec<&BenchRow> = rows.iter().collect();
    match sort {
        SortColumn::Name => sorted.sort_by(|a, b| a.test.cmp(&b.test)),
        SortColumn::Median => sorted.sort_by(|a, b| {
            a.wall_median
                .partial_cmp(&b.wall_median)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        SortColumn::Cv => sorted.sort_by(|a, b| {
            b.wall_cv
                .partial_cmp(&a.wall_cv)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        SortColumn::Throughput => sorted.sort_by(|a, b| {
            b.calls_per_second
                .partial_cmp(&a.calls_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
    }

    let name_width = sorted
        .iter()
        .map(|r| r.test.len())
        .max()
        .unwrap_or(4)
        .max(4);

    println!();
    println!(
        "{BOLD}{:<width$}  {:>12}  {:>8}  {:>8}  {:>8}  {:>14}  {:>6}{RESET}",
        "Test",
        "Median (us)",
        "P10",
        "P90",
        "CV",
        "Calls/sec",
        "Stable",
        width = name_width
    );
    println!(
        "{:-<width$}  {:-<12}  {:-<8}  {:-<8}  {:-<8}  {:-<14}  {:-<6}",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        width = name_width
    );

    for r in &sorted {
        let stable_str = if r.stable != 0 { "yes" } else { "NO" };
        let cv_color = if r.wall_cv > r.cv_threshold {
            YELLOW
        } else {
            ""
        };
        let stable_color = if r.stable == 0 { RED } else { "" };

        println!(
            "{:<width$}  {:>12.5}  {:>8.5}  {:>8.5}  {cv_color}{:>7.1}%{RESET}  {:>14.0}  {stable_color}{:<6}{RESET}",
            r.test,
            r.wall_median,
            r.wall_p10,
            r.wall_p90,
            r.wall_cv * 100.0,
            r.calls_per_second,
            stable_str,
            width = name_width,
        );
    }

    println!();
    println!(
        "  {} tests, sorted by {}",
        sorted.len(),
        match sort {
            SortColumn::Name => "name",
            SortColumn::Median => "median",
            SortColumn::Cv => "CV (highest first)",
            SortColumn::Throughput => "throughput (highest first)",
        }
    );
}

/* ----------------------------- Markdown ----------------------------- */

/// Format a comparison as a markdown table with its unmatched tests.
pub fn to_markdown(comparison: &Comparison) -> String {
    let mut out = String::new();

    out.push_str("| Test | Baseline | Candidate | Delta | % | Base CV | Cand CV | Result |\n");
    out.push_str("|------|----------|-----------|-------|---|---------|---------|--------|\n");

    for r in &comparison.results {
        let marker = match r.classification {
            Classification::Regression => "[!]",
            Classification::Improvement => "[+]",
            Classification::Neutral => "[ ]",
        };

        out.push_str(&format!(
            "| {} | {:.5} | {:.5} | {:+.5} | {:+.1}% | {:.1}% | {:.1}% | {} {} |\n",
            r.test,
            r.baseline_median,
            r.candidate_median,
            r.delta_us,
            r.delta_pct,
            r.baseline_cv * 100.0,
            r.candidate_cv * 100.0,
            marker,
            r.classification,
        ));
    }

    if !comparison.baseline_only.is_empty() {
        out.push_str(&format!(
            "\nMissing from the candidate ({}): {}\n",
            comparison.baseline_only.len(),
            comparison.baseline_only.join(", ")
        ));
    }
    if !comparison.candidate_only.is_empty() {
        out.push_str(&format!(
            "\nNew in the candidate ({}): {}\n",
            comparison.candidate_only.len(),
            comparison.candidate_only.join(", ")
        ));
    }

    out.push_str(&format!(
        "\n{}\n",
        labelling_rule(comparison.threshold_pct).join(" ")
    ));

    out
}

/* ----------------------------- JSON ----------------------------- */

/// Format a comparison as a JSON document.
///
/// `p_value` is null: the comparison reads summary statistics, so no test for
/// statistical significance was run. `baseline_only` holds the tests the
/// candidate does not run, `candidate_only` the tests that have no baseline.
pub fn to_json(comparison: &Comparison) -> String {
    let entries: Vec<serde_json::Value> = comparison
        .results
        .iter()
        .map(|r| {
            serde_json::json!({
                "test": r.test,
                "baseline_median": r.baseline_median,
                "candidate_median": r.candidate_median,
                "delta_us": r.delta_us,
                "delta_pct": r.delta_pct,
                "classification": format!("{}", r.classification),
                "p_value": r.p_value,
                "baseline_cv": r.baseline_cv,
                "candidate_cv": r.candidate_cv,
            })
        })
        .collect();

    let document = serde_json::json!({
        "threshold_pct": comparison.threshold_pct,
        "results": entries,
        "baseline_only": comparison.baseline_only,
        "candidate_only": comparison.candidate_only,
    });

    serde_json::to_string_pretty(&document).unwrap_or_else(|_| "{}".to_string())
}

/// Format summary rows as a JSON array.
pub fn summary_to_json(rows: &[BenchRow]) -> String {
    let entries: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "test": r.test,
                "wall_median": r.wall_median,
                "wall_p10": r.wall_p10,
                "wall_p90": r.wall_p90,
                "wall_cv": r.wall_cv,
                "calls_per_second": r.calls_per_second,
                "stable": r.stable != 0,
                "cycles": r.cycles,
                "repeats": r.repeats,
            })
        })
        .collect();

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Format GPU environment results as a JSON array.
pub fn gpu_env_to_json(checks: &[super::CheckResult]) -> String {
    validate_to_json(checks)
}

/// Format validation results as a JSON array.
pub fn validate_to_json(checks: &[super::CheckResult]) -> String {
    let entries: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "label": c.label,
                "status": match c.status {
                    super::CheckStatus::Ok => "ok",
                    super::CheckStatus::Warn => "warn",
                    super::CheckStatus::Fail => "fail",
                },
                "detail": c.detail,
            })
        })
        .collect();

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::CompareResult;

    fn make_result(test: &str, base: f64, cand: f64) -> CompareResult {
        let delta = cand - base;
        let pct = (delta / base) * 100.0;
        CompareResult {
            test: test.to_string(),
            baseline_median: base,
            candidate_median: cand,
            delta_us: delta,
            delta_pct: pct,
            classification: if pct > 5.0 {
                Classification::Regression
            } else if pct < -5.0 {
                Classification::Improvement
            } else {
                Classification::Neutral
            },
            p_value: None,
            baseline_cv: 0.02,
            candidate_cv: 0.03,
        }
    }

    fn make_comparison(results: Vec<CompareResult>) -> Comparison {
        Comparison {
            threshold_pct: 5.0,
            results,
            baseline_only: Vec::new(),
            candidate_only: Vec::new(),
        }
    }

    /// @test Markdown carries the header, the CV context and the rule.
    #[test]
    fn markdown_has_header_and_rule() {
        let comparison = make_comparison(vec![make_result("Foo", 1.0, 1.5)]);
        let md = to_markdown(&comparison);
        assert!(
            md.contains("| Test | Baseline | Candidate | Delta | % | Base CV | Cand CV | Result |")
        );
        assert!(md.contains(
            "| Foo | 1.00000 | 1.50000 | +0.50000 | +50.0% | 2.0% | 3.0% | [!] REGRESSION |"
        ));
        assert!(md.contains("median change against the 5.0% threshold"));
        assert!(!md.to_lowercase().contains("p-value"), "{md}");
    }

    /// @test Markdown names the tests only one run reports.
    #[test]
    fn markdown_lists_unmatched_tests() {
        let mut comparison = make_comparison(vec![make_result("Foo", 1.0, 1.0)]);
        comparison.baseline_only = vec!["Gone".to_string()];
        comparison.candidate_only = vec!["New".to_string()];
        let md = to_markdown(&comparison);
        assert!(
            md.contains("\nMissing from the candidate (1): Gone\n"),
            "{md}"
        );
        assert!(md.contains("\nNew in the candidate (1): New\n"), "{md}");
    }

    /// @test JSON carries the results, the threshold and an explicit null p-value.
    #[test]
    fn json_valid() {
        let comparison = make_comparison(vec![make_result("Bar", 2.0, 1.8)]);
        let json = to_json(&comparison);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["threshold_pct"], 5.0);
        assert_eq!(parsed["results"][0]["test"], "Bar");
        assert_eq!(parsed["results"][0]["classification"], "IMPROVEMENT");
        assert!(parsed["results"][0]["p_value"].is_null(), "{json}");
        assert!(parsed["baseline_only"].as_array().unwrap().is_empty());
    }

    /// @test JSON names the tests only one run reports.
    #[test]
    fn json_lists_unmatched_tests() {
        let mut comparison = make_comparison(vec![make_result("Foo", 1.0, 1.0)]);
        comparison.baseline_only = vec!["Gone".to_string()];
        comparison.candidate_only = vec!["New".to_string()];
        let parsed: serde_json::Value = serde_json::from_str(&to_json(&comparison)).unwrap();
        assert_eq!(parsed["baseline_only"][0], "Gone");
        assert_eq!(parsed["candidate_only"][0], "New");
    }
}
