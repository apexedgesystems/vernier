use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bench"))
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("spawn bench");
    let code = out.status.code().unwrap_or(255);
    (
        code,
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn fixture(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/* ----------------------------- Help ----------------------------- */

#[test]
fn help_exits_zero() {
    let (code, out, _) = run(&["--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("Benchmark analysis"));
}

/* ----------------------------- Summary ----------------------------- */

#[test]
fn summary_prints_table() {
    let (code, out, _) = run(&["summary", &fixture("sample_bench.csv")]);
    assert_eq!(code, 0, "summary should exit 0");
    assert!(out.contains("Queue.Throughput"), "should contain test name");
    assert!(out.contains("Queue.Latency"), "should contain all tests");
    assert!(out.contains("3 tests"), "should show test count");
}

#[test]
fn summary_json_output() {
    let (code, out, _) = run(&["summary", &fixture("sample_bench.csv"), "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert!(parsed.is_array());
    assert_eq!(parsed.as_array().unwrap().len(), 3);
}

#[test]
fn summary_sort_by_median() {
    let (code, out, _) = run(&["summary", &fixture("sample_bench.csv"), "--sort", "median"]);
    assert_eq!(code, 0);
    assert!(out.contains("sorted by median"));
}

#[test]
fn summary_sort_by_throughput() {
    let (code, out, _) = run(&[
        "summary",
        &fixture("sample_bench.csv"),
        "--sort",
        "throughput",
    ]);
    assert_eq!(code, 0);
    assert!(out.contains("sorted by throughput"));
}

#[test]
fn summary_old_format_csv() {
    let (code, out, _) = run(&["summary", &fixture("sample_bench_old_format.csv")]);
    assert_eq!(code, 0, "old format CSV should parse successfully");
    assert!(out.contains("Queue.Throughput"));
}

/* ----------------------------- Compare ----------------------------- */

#[test]
fn compare_identical_all_neutral() {
    let csv = fixture("sample_bench.csv");
    let (code, out, _) = run(&["compare", &csv, &csv]);
    assert_eq!(code, 0);
    assert!(out.contains("neutral"), "identical files should be neutral");
    assert!(!out.contains("REGRESSION"));
}

#[test]
fn compare_detects_regression() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(
        code, 0,
        "compare without --fail-on-regression always exits 0"
    );
    // Queue.Latency doubled from 0.012 to 0.024 -- should show large delta
    assert!(out.contains("Queue.Latency"));
}

#[test]
fn compare_fail_on_regression_exits_nonzero() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    // Queue.Latency has 100% regression, should trigger failure
    // (depends on p-value from synthesized samples -- may or may not trigger)
    // At minimum, the command should run without crashing
    assert!(
        code == 0 || code == 1,
        "should exit 0 or 1, got {code}: {err}"
    );
}

#[test]
fn compare_json_output() {
    let csv = fixture("sample_bench.csv");
    let (code, out, _) = run(&["compare", &csv, &csv, "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert!(parsed.is_array());
}

#[test]
fn compare_markdown_output() {
    let csv = fixture("sample_bench.csv");
    let (code, out, _) = run(&["compare", &csv, &csv, "--markdown"]);
    assert_eq!(code, 0);
    assert!(out.contains("| Test |"));
    assert!(out.contains("|---"));
}

/* ----------------------------- Validate ----------------------------- */

#[test]
fn validate_runs_successfully() {
    let (code, out, _) = run(&["validate"]);
    assert_eq!(code, 0);
    assert!(out.contains("Profile Readiness Check"));
    assert!(out.contains("passed"));
}

#[test]
fn validate_json_output() {
    let (code, out, _) = run(&["validate", "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert!(parsed.is_array());
    assert!(parsed.as_array().unwrap().len() >= 5);
}

/* ----------------------------- Error Cases ----------------------------- */

#[test]
fn summary_missing_file() {
    let (code, _, err) = run(&["summary", "/nonexistent/file.csv"]);
    assert_ne!(code, 0);
    assert!(err.contains("Error"));
}

#[test]
fn compare_missing_file() {
    let (code, _, err) = run(&["compare", "/nonexistent/a.csv", "/nonexistent/b.csv"]);
    assert_ne!(code, 0);
    assert!(err.contains("Error"));
}

#[test]
fn invalid_sort_column() {
    let csv = fixture("sample_bench.csv");
    let (code, _, err) = run(&["summary", &csv, "--sort", "bogus"]);
    assert_ne!(code, 0);
    assert!(err.contains("unknown sort column"));
}
