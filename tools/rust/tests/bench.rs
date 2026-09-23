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

/// @test Two identical runs label every test neutral and exit 0.
#[test]
fn compare_identical_all_neutral() {
    let csv = fixture("sample_bench.csv");
    let (code, out, _) = run(&["compare", &csv, &csv]);
    assert_eq!(code, 0);
    assert!(out.contains("neutral"), "identical files should be neutral");
    assert!(!out.contains("REGRESSION"));
    assert!(!out.contains("IMPROVEMENT"));
}

/// @test A slower and a faster median are labelled against the threshold.
#[test]
fn compare_labels_both_directions() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(
        code, 0,
        "a comparison that ran exits 0 without --fail-on-regression"
    );
    // Queue.Latency 0.012 -> 0.024 (+100%), Lock.Contended 0.080 -> 0.070
    // (-12.5%), Queue.Throughput 0.045 -> 0.046 (+2.2%, inside the 5%
    // threshold).
    assert!(out.contains("REGRESSION"), "Queue.Latency is +100%: {out}");
    assert!(
        out.contains("IMPROVEMENT"),
        "Lock.Contended is -12.5%: {out}"
    );
    assert!(out.contains("1 regression(s)"), "{out}");
    assert!(out.contains("1 improvement(s)"), "{out}");
    assert!(out.contains("1 neutral"), "{out}");
}

/// @test The table shows both CV values and no p-value.
#[test]
fn compare_table_shows_cv_not_p_value() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(code, 0);
    assert!(out.contains("Base CV"), "CV context column missing: {out}");
    assert!(out.contains("Cand CV"), "CV context column missing: {out}");
    assert!(
        !out.to_lowercase().contains("p-value"),
        "the table still presents a p-value: {out}"
    );
    // Queue.Throughput baseline CV 0.0667 -> 6.7%.
    assert!(out.contains("6.7%"), "CV values are not shown: {out}");
}

/// @test The output states the rule the labels come from.
#[test]
fn compare_states_the_labelling_rule() {
    let csv = fixture("sample_bench.csv");
    let (code, out, _) = run(&["compare", &csv, &csv, "--threshold", "3"]);
    assert_eq!(code, 0);
    assert!(
        out.contains("median change against the 3.0% threshold"),
        "the labelling rule is not stated: {out}"
    );
}

/// @test The gate exits 1 and names the flag when a test is a regression.
#[test]
fn compare_fail_on_regression_exits_one() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    assert_eq!(code, 1, "Queue.Latency is +100%: {err}");
    assert!(err.contains("--fail-on-regression"), "{err}");
    assert!(err.contains("1 regression(s)"), "{err}");
    assert!(err.contains("0 baseline test(s) missing"), "{err}");
}

/// @test A change of exactly the threshold is neutral and passes the gate.
#[test]
fn compare_threshold_boundary_is_neutral() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("sample_bench_regressed.csv");
    // Queue.Latency is exactly +100.0%.
    let (code, out, err) = run(&[
        "compare",
        &base,
        &cand,
        "--threshold",
        "100",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 0, "+100.0% is not beyond a 100% threshold: {err}");
    assert!(!out.contains("REGRESSION"), "{out}");

    let (code, out, _) = run(&[
        "compare",
        &base,
        &cand,
        "--threshold",
        "99.9",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 1, "+100.0% is beyond a 99.9% threshold");
    assert!(out.contains("REGRESSION"), "{out}");
}

/// `text` without its ANSI colour sequences.
fn strip_ansi(text: &str) -> String {
    let mut plain = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            chars.by_ref().find(|&c| c == 'm');
        } else {
            plain.push(c);
        }
    }
    plain
}

/// Each test's label, as JSON, the table and the Markdown table print it.
/// The three must agree.
fn labels(base: &str, cand: &str, threshold: &str) -> Vec<(String, String)> {
    let (code, out, err) = run(&["compare", base, cand, "--threshold", threshold, "--json"]);
    assert_eq!(code, 0, "advisory comparison exits 0: {err}");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    let json: Vec<(String, String)> = parsed["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|r| {
            (
                r["test"].as_str().expect("test").to_string(),
                r["classification"].as_str().expect("label").to_string(),
            )
        })
        .collect();

    let (_, out, _) = run(&["compare", base, cand, "--threshold", threshold]);
    let table = strip_ansi(&out);
    let (_, markdown, _) = run(&[
        "compare",
        base,
        cand,
        "--threshold",
        threshold,
        "--markdown",
    ]);
    for (test, label) in &json {
        let row = table
            .lines()
            .find(|line| line.split_whitespace().next() == Some(test.as_str()))
            .expect("table row");
        assert_eq!(
            row.split_whitespace().last(),
            Some(label.as_str()),
            "{table}"
        );
        let row = markdown
            .lines()
            .find(|line| line.starts_with(&format!("| {test} |")))
            .expect("Markdown row");
        assert!(row.ends_with(&format!(" {label} |")), "{markdown}");
    }
    json
}

fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
    expected
        .iter()
        .map(|(test, label)| (test.to_string(), label.to_string()))
        .collect()
}

/// @test A median change of exactly the threshold in the CSV's decimals is
/// neutral; a change just past it is labelled, in every format and at the gate.
#[test]
fn compare_decimal_threshold_boundary() {
    let base = fixture("compare_boundary_baseline.csv");
    for (file, down, up, gate) in [
        ("compare_boundary_at.csv", "neutral", "neutral", 0),
        ("compare_boundary_inside.csv", "neutral", "neutral", 0),
        (
            "compare_boundary_outside.csv",
            "IMPROVEMENT",
            "REGRESSION",
            1,
        ),
    ] {
        let cand = fixture(file);
        assert_eq!(
            labels(&base, &cand, "5"),
            pairs(&[("Edge.Down", down), ("Edge.Up", up)]),
            "{file}"
        );
        let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
        assert_eq!(code, gate, "{file}: {err}");
    }

    // 1 -> 1.05 prints as +5.0% and is neutral beside it.
    let (_, out, _) = run(&["compare", &base, &fixture("compare_boundary_at.csv")]);
    let up: Vec<String> = strip_ansi(&out)
        .lines()
        .find(|line| line.starts_with("Edge.Up"))
        .expect("Edge.Up row")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    assert_eq!(up[4..], ["+5.0%", "1.0%", "1.0%", "neutral"], "{out}");
}

/// @test A zero threshold labels any change in a reported median and nothing else.
#[test]
fn compare_zero_threshold_labels_every_change() {
    let base = fixture("compare_boundary_baseline.csv");
    assert_eq!(
        labels(&base, &base, "0"),
        pairs(&[("Edge.Down", "neutral"), ("Edge.Up", "neutral")])
    );
    let (code, _, err) = run(&[
        "compare",
        &base,
        &base,
        "--threshold",
        "0",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 0, "{err}");

    let tiny = fixture("compare_boundary_tiny.csv");
    assert_eq!(
        labels(&base, &tiny, "0"),
        pairs(&[("Edge.Down", "IMPROVEMENT"), ("Edge.Up", "REGRESSION")])
    );
    let (code, _, _) = run(&[
        "compare",
        &base,
        &tiny,
        "--threshold",
        "0",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 1, "a 0.0001% rise fails a zero-threshold gate");
    assert_eq!(
        labels(&base, &tiny, "5"),
        pairs(&[("Edge.Down", "neutral"), ("Edge.Up", "neutral")])
    );
}

/// @test A median that rose while the outer quantiles held still is a regression.
#[test]
fn compare_quantile_contradiction_is_a_regression() {
    let base = fixture("compare_quantile_baseline.csv");
    let cand = fixture("compare_quantile_candidate.csv");
    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(code, 0);
    assert!(out.contains("+20.0%"), "median 100 -> 120 is +20%: {out}");
    assert!(
        out.contains("REGRESSION"),
        "a +20% median change is beyond the 5% threshold: {out}"
    );
    assert!(
        !out.to_lowercase().contains("p-value"),
        "unchanged outer quantiles must not buy a significance claim: {out}"
    );

    let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    assert_eq!(code, 1, "the gate must fail on a +20% median: {err}");
}

/// @test Two runs with no test in common are an error, not a pass.
#[test]
fn compare_disjoint_inputs_are_an_error() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_disjoint.csv");
    let (base, cand) = (base.as_str(), cand.as_str());
    for args in [
        vec!["compare", base, cand],
        vec!["compare", base, cand, "--fail-on-regression"],
        vec!["compare", base, cand, "--json"],
    ] {
        let (code, out, err) = run(&args);
        assert_eq!(code, 1, "{args:?} exited {code}: {err}");
        assert!(
            err.contains("no tests are present in both runs"),
            "{args:?}: {err}"
        );
        assert!(err.contains("Queue.Latency"), "baseline names: {err}");
        assert!(err.contains("Cache.Warm"), "candidate names: {err}");
        assert!(
            out.trim().is_empty(),
            "{args:?} printed a comparison anyway: {out}"
        );
    }
}

/// @test A baseline median of zero is an input error, not a neutral 0%.
#[test]
fn compare_zero_baseline_median_is_an_error() {
    let base = fixture("compare_zero_median.csv");
    let cand = fixture("sample_bench.csv");
    let (code, out, err) = run(&["compare", &base, &cand]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("baseline wallMedian"), "{err}");
    assert!(err.contains("Queue.Latency"), "{err}");
    assert!(err.contains("undefined"), "{err}");
    assert!(out.trim().is_empty(), "{out}");
}

/// @test A candidate median of zero is an invalid measurement.
#[test]
fn compare_zero_candidate_median_is_an_error() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_zero_median.csv");
    let (code, _, err) = run(&["compare", &base, &cand]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("candidate wallMedian"), "{err}");
    assert!(err.contains("Queue.Latency"), "{err}");
}

/// @test The same test name twice in one file is an input error.
#[test]
fn compare_duplicate_identity_is_an_error() {
    let dup = fixture("compare_duplicate_test.csv");
    let ok = fixture("sample_bench.csv");

    let (code, out, err) = run(&["compare", &dup, &ok]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("duplicate test identity"), "{err}");
    assert!(err.contains("Queue.Latency"), "{err}");
    assert!(err.contains("baseline"), "{err}");
    assert!(out.trim().is_empty(), "{out}");

    let (code, _, err) = run(&["compare", &ok, &dup]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("duplicate test identity"), "{err}");
    assert!(err.contains("candidate"), "{err}");
}

/// @test A non-finite measurement is an input error.
#[test]
fn compare_non_finite_measurement_is_an_error() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_non_finite.csv");
    let (code, out, err) = run(&["compare", &base, &cand]);
    assert_eq!(code, 1, "{err}");
    assert!(err.contains("not a finite number"), "{err}");
    assert!(err.contains("Queue.Latency"), "{err}");
    assert!(out.trim().is_empty(), "{out}");
}

/// The ways a comparison can be run: advisory or gated, in each output format.
const COMPARE_MODES: [&[&str]; 6] = [
    &[],
    &["--fail-on-regression"],
    &["--json"],
    &["--json", "--fail-on-regression"],
    &["--markdown"],
    &["--markdown", "--fail-on-regression"],
];

/// @test A measured field the CSV does not hold as a number fails every mode, on either side.
#[test]
fn compare_unreadable_measurement_is_an_error() {
    let good = fixture("sample_bench.csv");
    for (file, found) in [
        (
            "compare_cv_malformed.csv",
            "has wallCV 'garbage', which is not a number",
        ),
        (
            "compare_cv_blank.csv",
            "has no wallCV value: the field is empty",
        ),
        (
            "compare_cv_absent.csv",
            "has no wallCV value: the row ends after 16 of 20 columns",
        ),
        (
            "compare_median_malformed.csv",
            "has wallMedian 'n/a', which is not a number",
        ),
        (
            "compare_median_blank.csv",
            "has no wallMedian value: the field is empty",
        ),
        (
            "compare_median_absent.csv",
            "has no wallMedian value: the row ends after 9 of 20 columns",
        ),
    ] {
        let bad = fixture(file);
        let expected = format!("{bad}, line 3: test 'Queue.Latency' {found}");
        for (base, cand) in [(&good, &bad), (&bad, &good)] {
            for mode in COMPARE_MODES {
                let mut args = vec!["compare", base.as_str(), cand.as_str()];
                args.extend_from_slice(mode);
                let (code, out, err) = run(&args);
                assert_eq!(code, 1, "{args:?}: {err}");
                assert!(out.is_empty(), "{args:?} printed a comparison: {out}");
                assert!(err.contains(&expected), "{args:?}: {err}");
            }
        }
    }
}

/// @test A wallCV of 0 is a measured value, not a failed parse: the comparison runs.
#[test]
fn compare_zero_cv_is_a_value() {
    let good = fixture("sample_bench.csv");
    let zero = fixture("compare_cv_zero.csv");
    for (base, cand, field) in [
        (&good, &zero, "candidate_cv"),
        (&zero, &good, "baseline_cv"),
    ] {
        let (code, out, err) = run(&["compare", base, cand, "--json", "--fail-on-regression"]);
        assert_eq!(code, 0, "{err}");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        let latency = parsed["results"]
            .as_array()
            .expect("results array")
            .iter()
            .find(|r| r["test"] == "Queue.Latency")
            .expect("Queue.Latency is compared");
        assert_eq!(latency[field], 0.0, "{out}");
        assert_eq!(latency["classification"], "neutral", "{out}");
    }

    let (code, out, _) = run(&["compare", &good, &zero]);
    assert_eq!(code, 0);
    let row: Vec<&str> = out
        .lines()
        .find(|line| line.starts_with("Queue.Latency"))
        .expect("Queue.Latency row")
        .split_whitespace()
        .collect();
    assert_eq!(row[5..7], ["8.3%", "0.0%"], "Base CV, Cand CV: {out}");
}

/// @test A CSV without the stable and cvThreshold columns still compares.
#[test]
fn compare_older_csv_format_still_compares() {
    let old = fixture("sample_bench_old_format.csv");
    let new = fixture("sample_bench.csv");
    let (code, out, err) = run(&["compare", &old, &new, "--fail-on-regression"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("2 neutral"), "{out}");
}

/// @test A baseline test the candidate does not run is reported and fails the gate.
#[test]
fn compare_missing_baseline_test_fails_the_gate() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_missing_test.csv");

    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(code, 0, "advisory comparison reports, it does not fail");
    assert!(
        out.contains("Missing from the candidate (1): Lock.Contended"),
        "{out}"
    );

    let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    assert_eq!(code, 1, "a missing baseline test fails the gate: {err}");
    assert!(err.contains("0 regression(s)"), "{err}");
    assert!(err.contains("1 baseline test(s) missing"), "{err}");
}

/// @test A candidate-only test is reported as new and does not fail the gate.
#[test]
fn compare_new_test_alone_passes_the_gate() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_added_test.csv");

    let (code, out, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    assert_eq!(code, 0, "a new test does not fail on its own: {err}");
    assert!(
        out.contains("New in the candidate (1): Queue.Drain"),
        "a new test must not be certified silently: {out}"
    );
}

/// @test A renamed test is reported as one missing plus one new.
#[test]
fn compare_rename_is_missing_plus_new() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_renamed_test.csv");

    let (code, out, _) = run(&["compare", &base, &cand]);
    assert_eq!(code, 0);
    assert!(
        out.contains("Missing from the candidate (1): Lock.Contended"),
        "{out}"
    );
    assert!(
        out.contains("New in the candidate (1): Lock.Uncontended"),
        "{out}"
    );

    let (code, _, err) = run(&["compare", &base, &cand, "--fail-on-regression"]);
    assert_eq!(
        code, 1,
        "the missing half of a rename fails the gate: {err}"
    );
}

/// @test JSON carries the labels, an explicit null p-value and both unmatched lists.
#[test]
fn compare_json_output() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_renamed_test.csv");
    let (code, out, _) = run(&["compare", &base, &cand, "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");

    let results = parsed["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "two tests are in both runs: {out}");
    assert_eq!(results[0]["test"], "Queue.Latency");
    assert_eq!(results[0]["classification"], "neutral");
    assert!(
        results[0]["p_value"].is_null(),
        "an unavailable p-value must be explicit: {out}"
    );
    assert_eq!(parsed["threshold_pct"], 5.0);
    assert_eq!(
        parsed["baseline_only"].as_array().expect("baseline_only"),
        &vec![serde_json::json!("Lock.Contended")]
    );
    assert_eq!(
        parsed["candidate_only"].as_array().expect("candidate_only"),
        &vec![serde_json::json!("Lock.Uncontended")]
    );
}

/// @test Markdown carries the CV columns, the unmatched lists and the rule.
#[test]
fn compare_markdown_output() {
    let base = fixture("sample_bench.csv");
    let cand = fixture("compare_renamed_test.csv");
    let (code, out, _) = run(&["compare", &base, &cand, "--markdown"]);
    assert_eq!(code, 0);
    assert!(out.contains("| Test |"));
    assert!(out.contains("|---"));
    assert!(out.contains("| Base CV | Cand CV |"), "{out}");
    assert!(
        !out.to_lowercase().contains("p-value"),
        "markdown still presents a p-value: {out}"
    );
    assert!(
        out.contains("Missing from the candidate (1): Lock.Contended"),
        "{out}"
    );
    assert!(
        out.contains("New in the candidate (1): Lock.Uncontended"),
        "{out}"
    );
    assert!(
        out.contains("median change against the 5.0% threshold"),
        "{out}"
    );
}

/// @test A threshold that is not a usable percentage is an error.
#[test]
fn compare_invalid_threshold_is_an_error() {
    let csv = fixture("sample_bench.csv");
    for threshold in ["--threshold=-5", "--threshold=nan"] {
        let (code, out, err) = run(&["compare", &csv, &csv, threshold]);
        assert_eq!(code, 1, "{threshold}: {err}");
        assert!(err.contains("--threshold"), "{threshold}: {err}");
        assert!(out.trim().is_empty(), "{threshold}: {out}");
    }
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

/* ----------------------------- GPU Env ----------------------------- */

#[test]
fn gpu_env_runs_successfully() {
    let (code, out, _) = run(&["gpu-env"]);
    assert_eq!(code, 0);
    assert!(out.contains("GPU Environment Check"));
    assert!(out.contains("passed"));
}

#[test]
fn gpu_env_json_output() {
    let (code, out, _) = run(&["gpu-env", "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert!(parsed.is_array());
    // Should always have at least nvidia-smi, driver, toolkit checks
    assert!(parsed.as_array().unwrap().len() >= 3);
}

/* ----------------------------- GPU Lock ----------------------------- */

#[test]
fn gpu_lock_help_exits_zero() {
    let (code, out, _) = run(&["gpu-lock", "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("Lock") || out.contains("lock") || out.contains("reset"));
}

#[test]
fn gpu_lock_lock_help() {
    let (code, out, _) = run(&["gpu-lock", "lock", "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("freq") || out.contains("device"));
}

#[test]
fn gpu_lock_reset_help() {
    let (code, out, _) = run(&["gpu-lock", "reset", "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("device"));
}

/* ----------------------------- GPU Monitor ----------------------------- */

#[test]
fn gpu_monitor_snapshot_help() {
    let (code, out, _) = run(&["gpu-monitor", "snapshot", "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("output") || out.contains("JSON"));
}

#[test]
fn gpu_monitor_diff_help() {
    let (code, out, _) = run(&["gpu-monitor", "diff", "--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("BEFORE") || out.contains("AFTER") || out.contains("snapshot"));
}

#[test]
fn gpu_monitor_snapshot_to_stdout() {
    let (code, out, _) = run(&["gpu-monitor", "snapshot"]);
    // Will succeed on GPU machines, fail gracefully otherwise
    if code == 0 {
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(parsed.get("timestamp").is_some());
        assert!(parsed.get("devices").is_some());
    }
}

#[test]
fn gpu_monitor_snapshot_to_file_and_diff() {
    // Create a snapshot, save it, then diff it with itself
    let (code, out, _) = run(&["gpu-monitor", "snapshot"]);
    if code != 0 {
        return; // No GPU, skip
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("snap.json");
    std::fs::write(&path, &out).expect("write");

    let snap_path = path.to_string_lossy().to_string();
    let (diff_code, diff_out, _) = run(&["gpu-monitor", "diff", &snap_path, &snap_path]);
    assert_eq!(diff_code, 0);
    assert!(diff_out.contains("No significant changes"));
}

#[test]
fn gpu_monitor_diff_json_output() {
    let (code, out, _) = run(&["gpu-monitor", "snapshot"]);
    if code != 0 {
        return; // No GPU, skip
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("snap.json");
    std::fs::write(&path, &out).expect("write");

    let snap_path = path.to_string_lossy().to_string();
    let (diff_code, diff_out, _) = run(&["gpu-monitor", "diff", &snap_path, &snap_path, "--json"]);
    assert_eq!(diff_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&diff_out).expect("valid JSON");
    assert!(parsed.is_array());
    assert!(parsed.as_array().unwrap().is_empty()); // No diffs for self-compare
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

/* ----------------------------- Run: Missing Wrapper ----------------------------- */

/// Run `bench` with a PATH that resolves nothing, from an empty working
/// directory so a relative `bench-out/` cannot land in the source tree.
fn run_without_tools(cwd: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let empty_path = cwd.join("empty-path");
    std::fs::create_dir_all(&empty_path).expect("create empty PATH dir");
    let out = Command::new(bin())
        .args(args)
        .env("PATH", &empty_path)
        .current_dir(cwd)
        .output()
        .expect("spawn bench");
    (
        out.status.code().unwrap_or(255),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// @test A wrapped profile whose wrapper is absent fails by naming the wrapper and the profile.
#[test]
fn run_profile_names_missing_wrapper() {
    // The benchmark argument only has to be an existing file: the wrapper is
    // checked before anything is spawned.
    let target = bin().to_string_lossy().into_owned();
    for (profile, program) in [
        ("callgrind", "valgrind"),
        ("massif", "valgrind"),
        ("memcheck", "valgrind"),
        ("helgrind", "valgrind"),
        ("heaptrack", "heaptrack"),
        ("compute-sanitizer", "compute-sanitizer"),
        ("nsight", "nsys"),
        ("ncu", "ncu"),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let (code, _, err) = run_without_tools(dir.path(), &["run", &target, "--profile", profile]);
        assert_ne!(code, 0, "--profile {profile}: exit status");
        assert!(
            err.contains(&format!("'{program}'")),
            "--profile {profile}: stderr should name '{program}': {err}"
        );
        assert!(
            err.contains(&format!("--profile {profile}")),
            "--profile {profile}: stderr should name the profile: {err}"
        );
        assert!(
            !err.contains("No such file or directory"),
            "--profile {profile}: raw OS error leaked: {err}"
        );
    }
}

/// @test A missing wrapper leaves no artifact directory behind.
#[test]
fn run_profile_missing_wrapper_creates_no_artifacts() {
    let target = bin().to_string_lossy().into_owned();
    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = dir.path().join("artifacts");
    let out_arg = out_dir.to_string_lossy().into_owned();
    let (code, _, _) = run_without_tools(
        dir.path(),
        &[
            "run",
            &target,
            "--profile",
            "callgrind",
            "--profile-output-dir",
            &out_arg,
        ],
    );
    assert_ne!(code, 0);
    assert!(
        !out_dir.exists(),
        "artifact directory created for a run that never started"
    );
    assert!(!dir.path().join("bench-out").exists());
}

/// @test --taskset without the taskset program fails by naming it.
#[test]
fn run_taskset_names_missing_program() {
    let target = bin().to_string_lossy().into_owned();
    let dir = tempfile::tempdir().expect("tempdir");
    let (code, _, err) = run_without_tools(dir.path(), &["run", &target, "--taskset", "0"]);
    assert_ne!(code, 0);
    assert!(
        err.contains("'taskset'"),
        "stderr should name taskset: {err}"
    );
    assert!(
        !err.contains("No such file or directory"),
        "raw OS error leaked: {err}"
    );
}

/* ----------------------------- Run: Wrap Folder ----------------------------- */

/// @test A wrapped run tells the benchmark which folder holds the wrap's output.
#[test]
fn run_wrapped_exports_wrap_folder_to_child() {
    use std::os::unix::fs::PermissionsExt;

    // A stand-in for valgrind that records what the runner exported and
    // exits cleanly; built from shell builtins because PATH holds only it.
    let dir = tempfile::tempdir().expect("tempdir");
    let tools = dir.path().join("tools");
    std::fs::create_dir_all(&tools).expect("create tools dir");
    let record = dir.path().join("exported.txt");
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n%s\\n' \"$VERNIER_EXTERNAL_WRAP\" \"$VERNIER_EXTERNAL_WRAP_DIR\" > '{}'\n",
        record.display()
    );
    let fake = tools.join("valgrind");
    std::fs::write(&fake, script).expect("write fake valgrind");
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).expect("chmod");

    let target = bin();
    let stem = target.file_stem().unwrap().to_string_lossy().into_owned();
    let out = Command::new(bin())
        .args(["run", &target.to_string_lossy(), "--profile", "massif"])
        .env("PATH", &tools)
        .current_dir(dir.path())
        .output()
        .expect("spawn bench");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let exported = std::fs::read_to_string(&record).expect("fake valgrind ran");
    let mut lines = exported.lines();
    assert_eq!(lines.next(), Some("massif"));
    assert_eq!(
        lines.next(),
        Some(format!("bench-out/{stem}.massif").as_str()),
        "VERNIER_EXTERNAL_WRAP_DIR must name the folder the wrap writes into"
    );
    assert!(dir.path().join(format!("bench-out/{stem}.massif")).is_dir());
}
