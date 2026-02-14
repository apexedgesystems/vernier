//! Output formatting: terminal tables, markdown, and JSON.
//!
//! Uses ANSI escape codes for colored terminal output.

use super::{BenchRow, Classification, CompareResult, SortColumn};

/* ----------------------------- ANSI Colors ----------------------------- */

const GREEN: &str = "\x1b[92m";
const RED: &str = "\x1b[91m";
const YELLOW: &str = "\x1b[93m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/* ----------------------------- Comparison Table ----------------------------- */

/// Print a colored comparison table to stdout.
pub fn print_comparison_table(results: &[CompareResult]) {
    if results.is_empty() {
        println!("No common tests to compare.");
        return;
    }

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
        "{BOLD}{:<width$}  {:>12}  {:>12}  {:>10}  {:>8}  {:>8}  {:>12}{RESET}",
        "Test",
        "Baseline",
        "Candidate",
        "Delta",
        "%",
        "p-value",
        "Result",
        width = name_width
    );
    println!(
        "{:-<width$}  {:-<12}  {:-<12}  {:-<10}  {:-<8}  {:-<8}  {:-<12}",
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
            "{color}{:<width$}  {:>12.5}  {:>12.5}  {:>+10.5}  {:>+7.1}%  {:>8.4}  {:<12}{RESET}",
            r.test,
            r.baseline_median,
            r.candidate_median,
            r.delta_us,
            r.delta_pct,
            r.p_value,
            label,
            width = name_width,
            color = color,
        );
    }

    // Summary line
    let reg_count = results
        .iter()
        .filter(|r| r.classification == Classification::Regression)
        .count();
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
    print!("{neu_count} neutral");
    println!();
}

/* ----------------------------- Summary Table ----------------------------- */

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

/// Format comparison results as a markdown table.
pub fn to_markdown(results: &[CompareResult]) -> String {
    let mut out = String::new();

    out.push_str("| Test | Baseline | Candidate | Delta | % | Result |\n");
    out.push_str("|------|----------|-----------|-------|---|--------|\n");

    for r in results {
        let marker = match r.classification {
            Classification::Regression => "[!]",
            Classification::Improvement => "[+]",
            Classification::Neutral => "[ ]",
        };

        out.push_str(&format!(
            "| {} | {:.5} | {:.5} | {:+.5} | {:+.1}% | {} {} |\n",
            r.test,
            r.baseline_median,
            r.candidate_median,
            r.delta_us,
            r.delta_pct,
            marker,
            r.classification,
        ));
    }

    out
}

/* ----------------------------- JSON ----------------------------- */

/// Format comparison results as a JSON array.
pub fn to_json(results: &[CompareResult]) -> String {
    let entries: Vec<serde_json::Value> = results
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

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
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

    fn make_result(test: &str, base: f64, cand: f64) -> CompareResult {
        let delta = cand - base;
        let pct = if base > 0.0 {
            (delta / base) * 100.0
        } else {
            0.0
        };
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
            p_value: 0.01,
            baseline_cv: 0.02,
            candidate_cv: 0.03,
        }
    }

    /// @test Markdown output contains header row.
    #[test]
    fn markdown_has_header() {
        let results = vec![make_result("Foo", 1.0, 1.5)];
        let md = to_markdown(&results);
        assert!(md.contains("| Test |"));
        assert!(md.contains("Foo"));
    }

    /// @test JSON output is valid and contains test name.
    #[test]
    fn json_valid() {
        let results = vec![make_result("Bar", 2.0, 1.8)];
        let json = to_json(&results);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed[0]["test"], "Bar");
    }

    /// @test Empty results produce valid output.
    #[test]
    fn empty_results() {
        assert_eq!(to_markdown(&[]), "| Test | Baseline | Candidate | Delta | % | Result |\n|------|----------|-----------|-------|---|--------|\n");
        let json = to_json(&[]);
        assert_eq!(json, "[]");
    }
}
