use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::RwLock;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bench"))
}

/// Writing a stand-in program and starting a process never overlap. A process
/// started while a thread of this test binary holds a stand-in open for
/// writing inherits that descriptor until it execs, and running the stand-in
/// meanwhile fails with "Text file busy" (ETXTBSY). A writer takes the gate
/// exclusively; every start takes it shared until the child has exec'd.
static START_GATE: RwLock<()> = RwLock::new(());

/// Write an executable stand-in while no process is being started.
fn write_executable(path: &Path, script: &str) {
    use std::os::unix::fs::PermissionsExt;

    let _gate = START_GATE.write().unwrap_or_else(|e| e.into_inner());
    std::fs::write(path, script).expect("write the stand-in");
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
}

/// `command.output()`, started while no stand-in is being written.
fn output_of(command: &mut Command) -> Output {
    let child = {
        let _gate = START_GATE.read().unwrap_or_else(|e| e.into_inner());
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn bench")
    };
    child.wait_with_output().expect("wait for bench")
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = output_of(Command::new(bin()).args(args));
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

/// Summary JSON rows by test name.
fn summary_json(csv: &str) -> serde_json::Map<String, serde_json::Value> {
    let (code, out, err) = run(&["summary", csv, "--json"]);
    assert_eq!(code, 0, "{csv}: {err}");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    parsed
        .as_array()
        .expect("an array of rows")
        .iter()
        .map(|row| (row["test"].as_str().expect("test").to_string(), row.clone()))
        .collect()
}

/// @test A value summary cannot show truthfully fails it, as text and as JSON,
/// naming the file, line, test, column and value, with nothing on stdout.
#[test]
fn summary_unreadable_measurement_is_an_error() {
    let modes: [&[&str]; 2] = [&[], &["--json"]];
    for (file, found) in [
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
            "summary_throughput_malformed.csv",
            "has callsPerSecond 'garbage', which is not a number",
        ),
        (
            "summary_throughput_blank.csv",
            "has no callsPerSecond value: the field is empty",
        ),
        (
            "summary_throughput_absent.csv",
            "has no callsPerSecond value: the row ends after 17 of 20 columns",
        ),
        (
            "compare_non_finite.csv",
            "has wallMedian 'nan', which is not a finite number",
        ),
        (
            "summary_cv_infinite.csv",
            "has wallCV 'inf', which is not a finite number",
        ),
        (
            "summary_throughput_infinite.csv",
            "has callsPerSecond '-inf', which is not a finite number",
        ),
        (
            "summary_p10_malformed.csv",
            "has wallP10 'garbage', which is not a number",
        ),
        (
            "summary_p90_not_finite.csv",
            "has wallP90 'NaN', which is not a finite number",
        ),
        (
            "summary_cycles_malformed.csv",
            "has cycles '1.5', which is not a whole number from 0 to 4294967295",
        ),
        (
            "summary_stable_malformed.csv",
            "has stable 'yes', which is not a whole number from 0 to 255",
        ),
    ] {
        let csv = fixture(file);
        let expected = format!("{csv}, line 3: test 'Queue.Latency' {found}");
        for mode in modes {
            let mut args = vec!["summary", csv.as_str()];
            args.extend_from_slice(mode);
            let (code, out, err) = run(&args);
            assert_eq!(code, 1, "{args:?}: {err}");
            assert!(out.is_empty(), "{args:?} printed a summary: {out}");
            assert!(err.contains(&expected), "{args:?}: {err}");
        }
    }
}

/// @test Zeros are values, and optional columns a layout or a row leaves out
/// take their defaults, as do the columns a GPU CSV's CPU rows stop before.
#[test]
fn summary_accepts_zeros_and_left_out_columns() {
    let rows = summary_json(&fixture("compare_cv_zero.csv"));
    assert_eq!(rows["Queue.Latency"]["wall_cv"], 0.0);

    // A median, CV and throughput of 0: summary shows them though compare
    // would refuse the median.
    let zeros = fixture("summary_zeros.csv");
    let rows = summary_json(&zeros);
    for field in ["wall_median", "wall_p10", "wall_cv", "calls_per_second"] {
        assert_eq!(rows["Queue.Latency"][field], 0.0, "{field}");
    }
    let (code, out, err) = run(&["summary", &zeros]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("3 tests"), "{out}");

    // No cycles, repeats or cvThreshold column; an empty wallP10; a row that
    // stops before stable.
    let rows = summary_json(&fixture("summary_left_out.csv"));
    let latency = &rows["Queue.Latency"];
    assert_eq!(latency["wall_p10"], 0.0);
    assert_eq!(latency["wall_p90"], 0.013);
    assert_eq!(latency["stable"], true);
    assert_eq!(latency["cycles"], 0);
    assert_eq!(rows["Queue.Throughput"]["wall_p10"], 0.042);

    let rows = summary_json(&fixture("summary_gpu_rows.csv"));
    assert_eq!(rows["Saxpy.Cpu"]["wall_median"], 0.045);
    assert_eq!(rows["Saxpy.Gpu"]["wall_median"], 0.015);

    let rows = summary_json(&fixture("sample_bench_old_format.csv"));
    assert_eq!(rows["Queue.Latency"]["stable"], true);
    assert_eq!(rows.len(), 2);
}

/// A test name that is empty or only whitespace, and what the loader says.
const BLANK_NAMES: [(&str, &str); 2] = [
    ("compare_name_empty.csv", "the row has no test name"),
    (
        "compare_name_blank.csv",
        "test name \"   \" is only whitespace",
    ),
];

/// @test A row with an empty or whitespace-only test name fails summary, as
/// text and as JSON, naming the file and line.
#[test]
fn summary_blank_test_name_is_an_error() {
    let modes: [&[&str]; 2] = [&[], &["--json"]];
    for (file, problem) in BLANK_NAMES {
        let csv = fixture(file);
        let expected = format!("{csv}, line 3: {problem}");
        for mode in modes {
            let mut args = vec!["summary", csv.as_str()];
            args.extend_from_slice(mode);
            let (code, out, err) = run(&args);
            assert_eq!(code, 1, "{args:?}: {err}");
            assert!(out.is_empty(), "{args:?} printed a summary: {out}");
            assert!(err.contains(&expected), "{args:?}: {err}");
        }
    }
}

/// @test A test name with spaces around it is kept as written in summary.
#[test]
fn summary_keeps_a_padded_test_name() {
    let rows = summary_json(&fixture("compare_name_padded.csv"));
    assert!(rows.contains_key(" Queue.Latency "), "{rows:?}");
    assert!(!rows.contains_key("Queue.Latency"), "{rows:?}");
}

/// @test A CSV without a test column is refused by summary and compare.
#[test]
fn missing_test_column_is_an_error() {
    let csv = fixture("compare_no_test_column.csv");
    let good = fixture("sample_bench.csv");
    let (csv, good) = (csv.as_str(), good.as_str());
    for args in [
        vec!["summary", csv],
        vec!["summary", csv, "--json"],
        vec!["compare", good, csv],
        vec!["compare", csv, good, "--json", "--fail-on-regression"],
    ] {
        let (code, out, err) = run(&args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(out.is_empty(), "{args:?}: {out}");
        assert!(
            err.contains(&format!("missing required column 'test' in {csv}")),
            "{args:?}: {err}"
        );
    }
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

/// @test A finite change near the largest double is labelled against a huge
/// threshold, and stays neutral under a larger one.
#[test]
fn compare_huge_change_is_labelled() {
    let base = fixture("compare_huge_baseline.csv");
    let cand = fixture("compare_huge_candidate.csv");
    // 1 -> 1e306 is +1e308%.
    assert_eq!(
        labels(&base, &cand, "8e307"),
        pairs(&[("Scale.Huge", "REGRESSION")])
    );
    let (code, _, err) = run(&[
        "compare",
        &base,
        &cand,
        "--threshold",
        "8e307",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 1, "{err}");

    assert_eq!(
        labels(&base, &cand, "1.5e308"),
        pairs(&[("Scale.Huge", "neutral")])
    );
    let (code, _, err) = run(&[
        "compare",
        &base,
        &cand,
        "--threshold",
        "1.5e308",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 0, "{err}");
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

/// @test A rise too large to express as a percentage is refused in every mode.
#[test]
fn compare_unrepresentable_change_is_an_error() {
    let small = fixture("compare_extreme_small.csv");
    let large = fixture("compare_extreme_large.csv");
    for mode in COMPARE_MODES {
        let mut args = vec!["compare", small.as_str(), large.as_str()];
        args.extend_from_slice(mode);
        let (code, out, err) = run(&args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(out.is_empty(), "{args:?} printed a comparison: {out}");
        assert!(
            err.contains(
                "wallMedian for test 'Scale.Extreme' goes from 1e-300 in the baseline \
                 to 1e300 in the candidate: the percentage change is too large to represent"
            ),
            "{args:?}: {err}"
        );
    }

    // The same pair the other way round is a fall of exactly 100%.
    let (code, out, err) = run(&["compare", &large, &small, "--json"]);
    assert_eq!(code, 0, "{err}");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert_eq!(parsed["results"][0]["delta_pct"], -100.0, "{out}");
    assert_eq!(
        parsed["results"][0]["classification"], "IMPROVEMENT",
        "{out}"
    );
}

/// @test A blank test name fails a comparison on either side or both, advisory
/// or gated, in every format.
#[test]
fn compare_blank_test_name_is_an_error() {
    let good = fixture("sample_bench.csv");
    for (file, problem) in BLANK_NAMES {
        let bad = fixture(file);
        let expected = format!("{bad}, line 3: {problem}");
        for (base, cand) in [(&good, &bad), (&bad, &good), (&bad, &bad)] {
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

/// @test A padded test name is its own identity: it matches itself, and against
/// the plain name it is one missing test and one new one.
#[test]
fn compare_keeps_a_padded_test_name() {
    let padded = fixture("compare_name_padded.csv");
    let plain = fixture("sample_bench.csv");

    let (code, out, err) = run(&[
        "compare",
        &padded,
        &padded,
        "--json",
        "--fail-on-regression",
    ]);
    assert_eq!(code, 0, "{err}");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    let names: Vec<&str> = parsed["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|r| r["test"].as_str().expect("test"))
        .collect();
    assert!(names.contains(&" Queue.Latency "), "{out}");

    let (code, out, _) = run(&["compare", &plain, &padded, "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert_eq!(
        parsed["baseline_only"],
        serde_json::json!(["Queue.Latency"])
    );
    assert_eq!(
        parsed["candidate_only"],
        serde_json::json!([" Queue.Latency "])
    );
    let (code, _, err) = run(&["compare", &plain, &padded, "--fail-on-regression"]);
    assert_eq!(
        code, 1,
        "the plain name is missing from the candidate: {err}"
    );
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

/// @test Without a binary, validate on this host prints the presence-only
/// header and exits 0 whatever it finds.
#[test]
fn validate_runs_successfully() {
    let (code, out, _) = run(&["validate"]);
    assert_eq!(code, 0);
    assert!(out.contains("=== bench validate: profiling tools and settings on this host ==="));
    assert!(out.contains("Presence only"), "{out}");
    assert!(!out.contains("[FAIL]"), "{out}");
}

/// @test Without a binary, --json prints one array of rows, none failing.
#[test]
fn validate_json_output() {
    let (code, out, _) = run(&["validate", "--json"]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    let rows = parsed.as_array().expect("an array");
    assert_eq!(rows.len(), 14);
    for row in rows {
        assert!(row["status"] == "ok" || row["status"] == "warn", "{row}");
    }
}

/// Run `bench validate <args>` with PATH set to @p path; returns the exit
/// status, stdout and stderr.
fn run_validate_with_path(path: &Path, args: &[&str]) -> (i32, String, String) {
    let out = output_of(
        Command::new(bin())
            .arg("validate")
            .args(args)
            .env("PATH", path),
    );
    (
        out.status.code().unwrap_or(255),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The row of `bench validate --json` output labelled @p label.
fn validate_row<'a>(rows: &'a serde_json::Value, label: &str) -> &'a serde_json::Value {
    rows.as_array()
        .expect("an array")
        .iter()
        .find(|r| r["label"] == label)
        .unwrap_or_else(|| panic!("no row {label}: {rows}"))
}

/// @test Without a binary validate reports tool-presence facts: a found
/// tool is OK with its path and the version its --version prints, a file
/// without an execute bit and an absent tool are WARN, perf that does not
/// run is WARN, the gperftools row names the analyzer PATH gives (pprof when
/// there is no google-pprof), Nsight Compute has its own row, and no row is
/// FAIL. The header says access and modes are the doctor's.
#[test]
fn validate_without_binary_reports_facts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tools = dir.path().join("tools");
    std::fs::create_dir_all(&tools).expect("mkdir");
    write_executable(&tools.join("valgrind"), "#!/bin/sh\necho valgrind-3.99.0\n");
    write_executable(
        &tools.join("ncu"),
        "#!/bin/sh\necho 'NVIDIA (R) Nsight Compute Command Line Profiler'\necho 'Copyright (c) 2018-2099'\necho 'Version 2099.1.0.0 (build 1)'\n",
    );
    write_executable(
        &tools.join("perf"),
        "#!/bin/sh\necho 'WARNING: perf not found for kernel 9.9' >&2\nexit 2\n",
    );
    write_executable(&tools.join("pprof"), "#!/bin/sh\nexit 2\n");
    std::fs::write(tools.join("heaptrack"), "#!/bin/sh\necho heaptrack 9.9\n").expect("write");

    let (code, out, err) = run_validate_with_path(&tools, &[]);
    assert_eq!(code, 0, "{err}");
    assert!(
        out.contains("Presence only: whether each profiler can run here, and in which"),
        "{out}"
    );
    assert!(!out.contains("[FAIL]"), "{out}");

    let (code, out, err) = run_validate_with_path(&tools, &["--json"]);
    assert_eq!(code, 0, "{err}");
    let rows: serde_json::Value = serde_json::from_str(&out).expect("one JSON array");
    for row in rows.as_array().expect("an array") {
        assert!(row["status"] == "ok" || row["status"] == "warn", "{row}");
    }
    let t = tools.display();
    let valgrind = validate_row(&rows, "valgrind (callgrind/massif/memcheck/helgrind)");
    assert_eq!(valgrind["status"], "ok");
    assert_eq!(
        valgrind["detail"],
        format!(
            "installed at {t}/valgrind (valgrind-3.99.0); access and modes: bench doctor <binary>"
        )
    );
    let ncu = validate_row(&rows, "Nsight Compute");
    assert_eq!(ncu["status"], "ok");
    assert!(
        ncu["detail"]
            .as_str()
            .unwrap()
            .contains("(Version 2099.1.0.0 (build 1))"),
        "{ncu}"
    );
    let nsys = validate_row(&rows, "Nsight Systems");
    assert_eq!(nsys["status"], "warn");
    assert!(
        nsys["detail"]
            .as_str()
            .unwrap()
            .starts_with("nsys not found on PATH"),
        "{nsys}"
    );
    let heaptrack = validate_row(&rows, "heaptrack");
    assert_eq!(heaptrack["status"], "warn");
    assert!(
        heaptrack["detail"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{t}/heaptrack is not an executable file")),
        "{heaptrack}"
    );
    let perf = validate_row(&rows, "perf");
    assert_eq!(perf["status"], "warn");
    assert!(
        perf["detail"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{t}/perf does not run")),
        "{perf}"
    );
    let gperf = validate_row(&rows, "gperftools");
    assert_eq!(gperf["status"], "ok");
    assert!(
        gperf["detail"].as_str().unwrap().starts_with(&format!(
            "--profile gperf --profile-analyze runs {t}/pprof;"
        )),
        "{gperf}"
    );
    assert_eq!(validate_row(&rows, "compute-sanitizer")["status"], "warn");

    // With both analyzers, google-pprof is the one the analysis runs, as the
    // benchmark chooses it, wherever PATH puts pprof.
    let later = dir.path().join("later");
    std::fs::create_dir_all(&later).expect("mkdir");
    write_executable(
        &later.join("google-pprof"),
        "#!/bin/sh\necho 'pprof (part of gperftools 9.9)'\n",
    );
    let path = std::env::join_paths([&tools, &later]).expect("join");
    let (code, out, err) = run_validate_with_path(Path::new(&path), &["--json"]);
    assert_eq!(code, 0, "{err}");
    let rows: serde_json::Value = serde_json::from_str(&out).expect("one JSON array");
    assert_eq!(
        validate_row(&rows, "gperftools")["detail"],
        format!(
            "--profile gperf --profile-analyze runs {}/google-pprof (pprof (part of gperftools \
             9.9)); collection needs libprofiler in the binary: bench doctor <binary>",
            later.display()
        )
    );
}

/// @test With a binary validate shows the host facts and the binary's own
/// default-mode rows at advisory severity: a failing row is WARN "not usable
/// here:" with the doctor's message and hint, and the exit is 0.
#[test]
fn validate_with_binary_is_advisory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let b = bench.to_string_lossy().into_owned();

    let (code, out, err) = run(&["validate", &b]);
    assert_eq!(code, 0, "{err}");
    assert!(
        out.contains(&format!("=== bench validate: {b} on this host ===")),
        "{out}"
    );
    assert!(
        out.contains(
            "offcpu                         not usable here: missing: bpftrace not found on PATH\n"
        ),
        "{out}"
    );
    assert!(
        out.contains(&format!("\n  {:<37} apt install bpftrace\n", "")),
        "{out}"
    );
    assert!(!out.contains("[FAIL]"), "{out}");

    let (code, out, err) = run(&["validate", &b, "--json"]);
    assert_eq!(code, 0, "{err}");
    let rows: serde_json::Value = serde_json::from_str(&out).expect("one JSON array");
    let labels: Vec<&str> = rows
        .as_array()
        .expect("an array")
        .iter()
        .map(|r| r["label"].as_str().unwrap())
        .collect();
    assert_eq!(
        labels,
        [
            "ASLR",
            "FlameGraph",
            "perf_event_paranoid",
            "perf",
            "offcpu"
        ]
    );
    assert_eq!(
        validate_row(&rows, "perf"),
        &serde_json::json!({"label": "perf", "status": "ok", "detail": "perf stat counted"})
    );
    assert_eq!(
        validate_row(&rows, "offcpu"),
        &serde_json::json!({"label": "offcpu", "status": "warn",
            "detail": "not usable here: missing: bpftrace not found on PATH",
            "hint": "apt install bpftrace"})
    );
    let argv = std::fs::read_to_string(dir.path().join("argv.log")).unwrap_or_default();
    assert_eq!(argv, "--profile-check-json\n--profile-check-json\n");
}

/// @test A binary named without a directory is the file in the working
/// directory, for bench validate and bench doctor alike, not a program looked
/// up on PATH.
#[test]
fn validate_and_doctor_start_a_bare_file_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    doctor_stand_in(dir.path(), DOCTOR_DOC);
    for args in [
        vec!["validate", "doctor_bench", "--json"],
        vec!["doctor", "doctor_bench", "--json"],
        vec!["doctor", "doctor_bench"],
    ] {
        let out = output_of(Command::new(bin()).args(&args).current_dir(dir.path()));
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{args:?}: {err}");
    }
    let argv = std::fs::read_to_string(dir.path().join("argv.log")).unwrap_or_default();
    assert_eq!(
        argv,
        "--profile-check-json\n--profile-check-json\n--profile-check\n"
    );
}

/// @test A binary that is missing, does not start, prints no JSON document
/// (with how it ended and its last stderr line), or prints one without
/// usable backend rows is an operational error: exit 1 with the cause on
/// stderr and nothing on stdout, with or without --json.
#[test]
fn validate_operational_errors_exit_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("missing");
    let plain = dir.path().join("plain");
    std::fs::write(&plain, "#!/bin/sh\n").expect("write");
    let unloadable = dir.path().join("unloadable");
    write_executable(
        &unloadable,
        "#!/bin/sh\necho 'error while loading shared libraries: libbench.so' >&2\nexit 127\n",
    );
    let mut cases = vec![
        (missing, "binary not found".to_string()),
        (plain.clone(), format!("{} did not start", plain.display())),
        (
            unloadable.clone(),
            format!(
                "{} --profile-check-json printed no valid doctor document (exit status 127; \
                 stderr: error while loading shared libraries: libbench.so)",
                unloadable.display()
            ),
        ),
    ];
    for (name, doc, cause) in [
        (
            "garbage",
            "not json\n",
            "printed no valid doctor document (exit status 0)",
        ),
        (
            "no_rows",
            "{\"binary\": {}}\n",
            "printed no usable doctor document: it has no \"backends\" array",
        ),
        (
            "bad_status",
            "{\"backends\": [{\"name\": \"perf\", \"status\": \"bogus\"}]}\n",
            "backend row 'perf' has status \"bogus\"",
        ),
    ] {
        let sub = dir.path().join(name);
        std::fs::create_dir_all(&sub).expect("mkdir");
        cases.push((doctor_stand_in(&sub, doc), cause.to_string()));
    }
    for (binary, cause) in cases {
        let b = binary.to_string_lossy().into_owned();
        for json in [false, true] {
            let mut args = vec!["validate", b.as_str()];
            if json {
                args.push("--json");
            }
            let (code, out, err) = run(&args);
            assert_eq!(code, 1, "{args:?}: {err}");
            assert!(err.contains(&cause), "{args:?}: {err}");
            assert!(out.is_empty(), "{args:?}: {out}");
        }
    }
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
    let out = output_of(
        Command::new(bin())
            .args(args)
            .env("PATH", &empty_path)
            .current_dir(cwd),
    );
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

/// @test A wrapper that PATH finds without an execute bit is named with its
/// path and the request before anything starts: no raw permission error, no
/// artifact directory.
#[test]
fn run_names_a_non_executable_wrapper() {
    let target = bin().to_string_lossy().into_owned();
    let dir = tempfile::tempdir().expect("tempdir");
    let tools = dir.path().join("tools");
    std::fs::create_dir_all(&tools).expect("mkdir");
    let valgrind = tools.join("valgrind");
    std::fs::write(&valgrind, "#!/bin/sh\nexit 0\n").expect("write");
    let out = output_of(
        Command::new(bin())
            .args(["run", &target, "--profile", "massif"])
            .env("PATH", &tools)
            .current_dir(dir.path()),
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(
        err.contains(&format!(
            "'valgrind' at {} is not executable; --profile massif runs the benchmark under it",
            valgrind.display()
        )),
        "{err}"
    );
    assert!(!err.contains("Permission denied"), "{err}");
    assert!(!dir.path().join("bench-out").exists());
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

/* ----------------------------- Run: Analyze ----------------------------- */

/// @test bench run --analyze refuses, before printing it, a summary bench summary
/// refuses; the run's own output stays on stdout whether or not the summary follows.
#[test]
fn run_analyze_refuses_an_unreadable_measurement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let written = dir.path().join("results.csv");
    let written_arg = written.to_string_lossy().into_owned();
    for (i, (source, code)) in [("sample_bench.csv", 0), ("compare_cv_malformed.csv", 1)]
        .into_iter()
        .enumerate()
    {
        // A stand-in benchmark that prints a line and copies a fixture to
        // where --csv points.
        let fake = dir.path().join(format!("fake_bench_{i}"));
        let script = format!(
            "#!/bin/sh\necho 'stand-in benchmark ran'\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = --csv ]; then cp '{}' \"$2\"; fi\n  shift\ndone\n",
            fixture(source)
        );
        write_executable(&fake, &script);

        let out = output_of(
            Command::new(bin())
                .args([
                    "run",
                    &fake.to_string_lossy(),
                    "--csv",
                    &written_arg,
                    "--analyze",
                ])
                .current_dir(dir.path()),
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(code), "{source}: {stderr}");
        // The run's own output is not held back when the analysis is refused.
        assert!(stdout.contains("Running: "), "{source}: {stdout}");
        assert!(
            stdout.contains("stand-in benchmark ran"),
            "{source}: {stdout}"
        );
        if code == 0 {
            assert!(stdout.contains("--- Post-run analysis ---"), "{stdout}");
            assert!(stdout.contains("Queue.Latency"), "{stdout}");
        } else {
            assert!(!stdout.contains("Post-run analysis"), "{stdout}");
            assert!(
                stderr.contains(&format!(
                    "{written_arg}, line 3: test 'Queue.Latency' has wallCV 'garbage', which is not a number"
                )),
                "{stderr}"
            );
        }
    }
}

/* ----------------------------- Run: Wrap Folder ----------------------------- */

/// @test A wrapped run tells the benchmark which folder holds the wrap's output.
#[test]
fn run_wrapped_exports_wrap_folder_to_child() {
    // A stand-in for valgrind that records what the runner exported and
    // exits cleanly; built from shell builtins because PATH holds only it.
    let dir = tempfile::tempdir().expect("tempdir");
    let tools = dir.path().join("tools");
    std::fs::create_dir_all(&tools).expect("create tools dir");
    let record = dir.path().join("exported.txt");
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n%s\\n' \"$VERNIER_EXTERNAL_WRAP\" \"$VERNIER_EXTERNAL_WRAP_DIR\" > '{}'\n\
         for a in \"$@\"; do case \"$a\" in --massif-out-file=*) echo heap > \"${{a#*=}}\" ;; esac; done\n",
        record.display()
    );
    let fake = tools.join("valgrind");
    write_executable(&fake, &script);

    let target = bin();
    let stem = target.file_stem().unwrap().to_string_lossy().into_owned();
    let out = output_of(
        Command::new(bin())
            .args(["run", &target.to_string_lossy(), "--profile", "massif"])
            .env("PATH", &tools)
            .current_dir(dir.path()),
    );
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

/* ----------------------------- Run: Benchmark Exit ----------------------------- */

/// @test bench run names how the benchmark ended -- its status, its signal, or,
/// for status 4, the failed profile request -- and exits 1 itself each time.
#[test]
fn run_reports_the_benchmark_exit_status() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (i, (body, code, expected)) in [
        ("exit 0", 0, ""),
        ("exit 1", 1, "Error: the benchmark exited with status 1\n"),
        (
            "exit 4",
            1,
            "Error: the requested profile failed (the benchmark's report above says why); \
             the benchmark exited with status 4\n",
        ),
        (
            "kill -9 $$",
            1,
            "Error: the benchmark was ended by signal 9\n",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        // A stand-in benchmark that prints a line and ends as the case says.
        let fake = dir.path().join(format!("fake_bench_{i}"));
        let script = format!("#!/bin/sh\necho 'stand-in benchmark ran'\n{body}\n");
        write_executable(&fake, &script);

        let out = output_of(
            Command::new(bin())
                .args(["run", &fake.to_string_lossy()])
                .current_dir(dir.path()),
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(code), "{body}: {stderr}");
        assert!(
            stdout.contains("stand-in benchmark ran"),
            "{body}: {stdout}{stderr}"
        );
        assert!(stderr.ends_with(expected), "{body}: {stderr}");
        assert!(!stderr.contains("parse error"), "{body}: {stderr}");
    }
}

/* ----------------------------- Run: Routes ----------------------------- */

/// A private PATH holding recording stand-ins for wrapper programs, and a
/// stand-in benchmark. Every stand-in appends one line to `log` -- its name,
/// the wrap variable it was given, and its arguments -- and exits 0; a
/// wrapper does not start the benchmark, whose argv is what is under test.
/// A wrapper writes the output file its arguments name, as the tool would
/// (`FAKE_WRITE=none` writes nothing, `FAKE_WRITE=empty` an empty file), and
/// `nsys stats` prints a summary line (`FAKE_NSYS_STATS=fail` fails instead),
/// and `callgrind_annotate` one line naming its profile (`FAKE_ANNOTATE=fail`
/// fails instead). `FAKE_NSYS_STATS=leave`: each `nsys stats` also leaves a
/// process holding its output, its pid added to `nsys_left.pids`;
/// `FAKE_NSYS_STATS=wait`: the first one waits on a process it started (pids
/// in `nsys_stats.pid` and `nsys_started.pid`). valgrind reads its output option's value as valgrind does
/// (valgrind 3.18.1, recorded runs): `%%` is one `%`, `%p` its process id,
/// and any other `%` is refused with status 1. compute-sanitizer is
/// `SANITIZER_STAND_IN`.
struct RouteRig {
    dir: tempfile::TempDir,
    log: std::path::PathBuf,
    bench: std::path::PathBuf,
}

/// A compute-sanitizer stand-in (bash, for its pattern replacement) that
/// writes the report `FAKE_SANITIZER` names, in the real tool's words
/// (compute-sanitizer 2025.4.0, recorded runs; racecheck's warning lines take
/// the form of its recorded error lines), and ends as the tool does: errors
/// found end with its `--error-exitcode` value (0 without one, the tool's
/// default), otherwise with the benchmark's own status. It reads its
/// `--log-file` value as the tool does: `%%` is one `%`, and any other `%`
/// is refused, with status 255 and no report.
const SANITIZER_STAND_IN: &str = r#"#!/bin/bash
echo "compute-sanitizer wrap=$VERNIER_EXTERNAL_WRAP $*" >> '{log}'
log=""
ec=0
prev=""
for a in "$@"; do
  [ "$prev" = --log-file ] && log="$a"
  [ "$prev" = --error-exitcode ] && ec="$a"
  prev="$a"
done
if [[ "${log//\%\%/}" == *%* ]]; then
  echo "fake compute-sanitizer: a '%' macro in the log path $log" >&2
  exit 255
fi
log="${log//\%\%/%}"
report() { printf '========= %s\n' 'COMPUTE-SANITIZER' "$@" > "$log"; }
case "${FAKE_SANITIZER:-clean}" in
  clean) report 'ERROR SUMMARY: 0 errors'; exit 0 ;;
  findings) report 'Invalid __global__ write of size 4 bytes' 'ERROR SUMMARY: 1 error'; exit "$ec" ;;
  app-fail) report 'Target application returned an error' 'ERROR SUMMARY: 0 errors'; exit 1 ;;
  app-fail-findings)
    report 'Invalid __global__ write of size 4 bytes' 'Target application returned an error' \
      'ERROR SUMMARY: 3 errors'
    [ "$ec" = 0 ] && exit 1
    exit "$ec" ;;
  collision) report 'Target application returned an error' 'ERROR SUMMARY: 0 errors'; exit 5 ;;
  abnormal)
    report "Error: process didn't terminate successfully" 'Target application returned an error' \
      'ERROR SUMMARY: 0 errors'
    exit 9 ;;
  race-clean) report 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'; exit 0 ;;
  race-findings)
    report 'Error: Race reported between Write access at k(int *, int)+0x70' \
      '    and Write access at k(int *, int)+0x70 [63 hazards]' '' \
      'RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)'
    exit "$ec" ;;
  race-warnings)
    report 'Warning: Race reported between Read access at k(int *, int)+0x60' \
      '    and Write access at k(int *, int)+0x70 [2 hazards]' '' \
      'RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)'
    exit 0 ;;
  race-warnings-findings)
    report 'Error: Race reported between Write access at k(int *, int)+0x70' \
      '    and Write access at k(int *, int)+0x70 [63 hazards]' '' \
      'Warning: Race reported between Read access at k(int *, int)+0x60' \
      '    and Write access at k(int *, int)+0x70 [2 hazards]' '' \
      'RACECHECK SUMMARY: 2 hazards displayed (1 error, 1 warning)'
    exit "$ec" ;;
  startup) report 'Error: Target application terminated before first instrumented API call'; exit 255 ;;
  truncated) report 'Invalid __global__ write of size 4 bytes'; exit "${FAKE_SANITIZER_EXIT:-5}" ;;
  no-report) exit 5 ;;
esac
"#;

fn route_rig(programs: &[&str]) -> RouteRig {
    let dir = tempfile::tempdir().expect("tempdir");
    let tools = dir.path().join("tools");
    std::fs::create_dir_all(&tools).expect("create tools dir");
    let log = dir.path().join("argv.log");
    let write = |path: &std::path::Path, name: &str| {
        let script = format!(
            r#"#!/bin/sh
echo "{name} wrap=$VERNIER_EXTERNAL_WRAP $*" >> '{log}'
if [ "{name}" = nsys ] && [ "$1" = stats ]; then
  if [ "$FAKE_NSYS_STATS" = fail ]; then echo "fake nsys: cannot export the report" >&2; exit 1; fi
  if [ "$FAKE_NSYS_STATS" = leave ]; then /bin/sleep 60 & echo $! >> nsys_left.pids; fi
  if [ "$FAKE_NSYS_STATS" = wait ]; then
    /bin/sleep 60 &
    echo $! > nsys_started.pid
    echo $$ > nsys_stats.pid
    wait
  fi
  echo "fake summary"
  exit 0
fi
if [ "{name}" = callgrind_annotate ]; then
  if [ "$FAKE_ANNOTATE" = fail ]; then echo "fake annotate: cannot read the profile" >&2; exit 1; fi
  echo "fake annotation of $2"
  exit 0
fi
[ "$FAKE_WRITE" = none ] && exit 0
out=""
prev=""
for a in "$@"; do
  case "$a" in
    --callgrind-out-file=*|--massif-out-file=*|--log-file=*) out="${{a#*=}}" ;;
  esac
  case "$prev" in
    --log-file) out="$a" ;;
    -o) case "{name}" in nsys) out="$a.nsys-rep" ;; ncu) out="$a.ncu-rep" ;; heaptrack) out="$a.gz" ;; esac ;;
  esac
  prev="$a"
done
# valgrind, nsys and ncu read '%' in their output option as the tools do:
# '%%' is one '%' and '%p' the process id, except that ncu refuses a macro
# in the folder part; nsys that cannot create its report says so and exits 0.
case "{name}" in valgrind | nsys | ncu) [ -n "$out" ] && macros=yes ;; esac
if [ "$macros" = yes ]; then
  rest="$out"
  out=""
  while [ -n "$rest" ]; do
    case "$rest" in
      *%*)
        out="$out${{rest%%\%*}}"
        rest="${{rest#*%}}"
        case "{name}:$rest" in
          "{name}:%"*) out="$out%"; rest="${{rest#%}}" ;;
          ncu:*) echo "==ERROR== Macro '%${{rest%"${{rest#?}}"}}' can only be used in the file name and not in the file path." >&2; exit 1 ;;
          "{name}:p"*) out="$out$$"; rest="${{rest#p}}" ;;
          valgrind:*) echo "==$$== Expected 'p' or 'q' or '%' after '%'" >&2; exit 1 ;;
          *) echo "{name}: this stand-in reads only %p and %% in its output option" >&2; exit 1 ;;
        esac ;;
      *) out="$out$rest"; rest="" ;;
    esac
  done
  if [ ! -d "${{out%/*}}" ]; then
    case "{name}" in
      nsys) echo "Failed to create '$out': No such file or directory." >&2; exit 0 ;;
      ncu) echo "==ERROR== Unable to write to file $out." >&2; exit 1 ;;
    esac
  fi
fi
# heaptrack replaces %h and %p in -o with the host name and its process id,
# with no escape, and creates the folders the result names, as its script does.
if [ "{name}" = heaptrack ] && [ -n "$out" ]; then
  rest="$out"
  out=""
  while :; do
    case "$rest" in
      *%[hp]*)
        pre="${{rest%%\%[hp]*}}"
        rest="${{rest#"$pre"}}"
        case "$rest" in
          %h*) out="$out${{pre}}stand-in-host"; rest="${{rest#%h}}" ;;
          *) out="$out$pre$$"; rest="${{rest#%p}}" ;;
        esac ;;
      *) out="$out$rest"; break ;;
    esac
  done
  /bin/mkdir -p "${{out%/*}}"
fi
if [ -n "$out" ]; then
  if [ "$FAKE_WRITE" = empty ]; then : > "$out"; else echo "fake {name} output" > "$out"; fi
fi
"#,
            log = log.display()
        );
        write_executable(path, &script);
    };
    for program in programs {
        if *program == "compute-sanitizer" {
            let script = SANITIZER_STAND_IN.replace("{log}", &log.display().to_string());
            write_executable(&tools.join(program), &script);
        } else {
            write(&tools.join(program), program);
        }
    }
    let bench = dir.path().join("fake_bench");
    write(&bench, "bench");
    RouteRig { dir, log, bench }
}

/// Run `bench run <rig's benchmark> <args>` with PATH set to the rig's
/// stand-ins; returns the exit status, stderr and the stand-ins' log.
fn run_rig(rig: &RouteRig, args: &[&str]) -> (i32, String, String) {
    let (code, _, err, log) = run_rig_env(rig, args, &[]);
    (code, err, log)
}

/// `run_rig` with extra environment for the stand-ins; also returns stdout.
fn run_rig_env(
    rig: &RouteRig,
    args: &[&str],
    env: &[(&str, &str)],
) -> (i32, String, String, String) {
    let mut command = Command::new(bin());
    command
        .arg("run")
        .arg(&rig.bench)
        .args(args)
        .env("PATH", rig.dir.path().join("tools"))
        .current_dir(rig.dir.path());
    for (k, v) in env {
        command.env(k, v);
    }
    let out = output_of(&mut command);
    (
        out.status.code().unwrap_or(255),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        std::fs::read_to_string(&rig.log).unwrap_or_default(),
    )
}

/// @test --profile nsys runs under nsys exactly as --profile nsight does, and
/// the benchmark is told nsight, the canonical name.
#[test]
fn run_nsys_alias_wraps_like_nsight() {
    for spelling in ["nsys", "nsight"] {
        let rig = route_rig(&["nsys"]);
        let (code, err, log) = run_rig(&rig, &["--profile", spelling]);
        assert_eq!(code, 0, "{spelling}: {err}");
        let mut lines = log.lines();
        assert_eq!(
            lines.next().unwrap_or(""),
            format!(
                "nsys wrap=nsight profile -o bench-out/fake_bench.nsight/profile -t cuda,nvtx \
                 --force-overwrite true {} --profile nsight",
                rig.bench.display()
            ),
            "{spelling}"
        );
        // Then the four summaries, from the report of this run.
        let stats: Vec<&str> = lines.collect();
        assert_eq!(stats.len(), 4, "{spelling}: {log}");
        assert!(
            stats
                .iter()
                .all(|l| l.starts_with("nsys wrap= stats --force-export=true")
                    && l.ends_with(" bench-out/fake_bench.nsight/profile.nsys-rep")),
            "{spelling}: {log}"
        );
    }
}

/// @test An explicit --profile-test-timeout 0 reaches the benchmark as 0 (the
/// watchdog off); an omitted one is not forwarded, so the binary decides.
#[test]
fn run_forwards_an_explicit_zero_timeout() {
    let rig = route_rig(&[]);
    let (code, err, _) = run_rig(&rig, &["--profile", "perf", "--profile-test-timeout", "0"]);
    assert_eq!(code, 0, "{err}");
    let (code, err, log) = run_rig(&rig, &["--profile", "perf"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        "bench wrap= --profile perf --profile-test-timeout 0\nbench wrap= --profile perf\n"
    );
}

/// @test massif's modes reach valgrind.
#[test]
fn run_massif_modes_reach_valgrind() {
    for (mode, flag) in [("pages", "--pages-as-heap=yes"), ("stacks", "--stacks=yes")] {
        let rig = route_rig(&["valgrind"]);
        let (code, err, log) = run_rig(&rig, &["--profile", "massif", "--profile-args", mode]);
        assert_eq!(code, 0, "{mode}: {err}");
        assert_eq!(
            log,
            format!(
                "valgrind wrap=massif --tool=massif {flag} \
                 --massif-out-file=bench-out/fake_bench.massif/massif.out {} --profile massif \
                 --profile-args {mode}\n",
                rig.bench.display()
            )
        );
    }
}

/// @test memcheck's track-origins reaches valgrind; the leak check stays full.
#[test]
fn run_memcheck_track_origins() {
    let rig = route_rig(&["valgrind"]);
    let (code, err, log) = run_rig(
        &rig,
        &["--profile", "memcheck", "--profile-args", "track-origins"],
    );
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        format!(
            "valgrind wrap=memcheck --tool=memcheck --leak-check=full --error-exitcode=0 \
             --track-origins=yes --log-file=bench-out/fake_bench.memcheck/memcheck.log {} \
             --profile memcheck --profile-args track-origins\n",
            rig.bench.display()
        )
    );
}

/// @test helgrind's drd mode runs valgrind's DRD tool, into helgrind's folder.
#[test]
fn run_helgrind_drd() {
    let rig = route_rig(&["valgrind"]);
    let (code, err, log) = run_rig(&rig, &["--profile", "helgrind", "--profile-args", "drd"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        format!(
            "valgrind wrap=helgrind --tool=drd --log-file=bench-out/fake_bench.helgrind/helgrind.log \
             {} --profile helgrind --profile-args drd\n",
            rig.bench.display()
        )
    );
}

/// @test A mode given after `--` is the request's: the wrap runs it, and the
/// benchmark receives the request once, so the tool and the benchmark agree.
#[test]
fn run_forwarded_mode_selects_the_wrap() {
    let rig = route_rig(&["valgrind"]);
    let (code, err, log) = run_rig(
        &rig,
        &["--profile", "helgrind", "--", "--profile-args", "drd"],
    );
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        format!(
            "valgrind wrap=helgrind --tool=drd --log-file=bench-out/fake_bench.helgrind/helgrind.log \
             {} --profile helgrind --profile-args drd\n",
            rig.bench.display()
        )
    );
}

/// @test A whole request after `--`, profile, mode and output folder, is
/// routed as if bench run had been given it, and the benchmark's other
/// arguments follow it unchanged.
#[test]
fn run_forwarded_request_is_routed() {
    let rig = route_rig(&["valgrind"]);
    let (code, err, log) = run_rig(
        &rig,
        &[
            "--",
            "--profile",
            "massif",
            "--profile-args",
            "pages",
            "--artifact-root",
            "out",
            "--repeats",
            "2",
        ],
    );
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        format!(
            "valgrind wrap=massif --tool=massif --pages-as-heap=yes \
             --massif-out-file=out/fake_bench.massif/massif.out {} --profile massif \
             --profile-args pages --profile-output-dir out --repeats 2\n",
            rig.bench.display()
        )
    );
}

/// @test A routed field given to bench run and again after `--` with another
/// value, or twice after `--`, is refused before anything starts or is
/// created.
#[test]
fn run_refuses_a_forwarded_field_that_conflicts() {
    for (args, expected) in [
        (
            &[
                "--profile",
                "massif",
                "--profile-args",
                "pages",
                "--",
                "--profile-args",
                "stacks",
            ][..],
            "--profile-args is given more than once with different values ('pages' to bench run, \
             'stacks' after --); give it once",
        ),
        (
            &["--profile", "massif", "--", "--profile", "memcheck"][..],
            "--profile is given more than once with different values ('massif' to bench run, \
             'memcheck' after --); give it once",
        ),
    ] {
        let rig = route_rig(&["valgrind"]);
        let (code, err, log) = run_rig(&rig, args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(
            err.contains(&format!("Error: invalid arguments: {expected}")),
            "{args:?}: {err}"
        );
        assert_eq!(log, "", "{args:?}: something was started");
        assert!(
            !rig.dir.path().join("bench-out").exists(),
            "{args:?}: a folder was created"
        );
    }
}

/// @test compute-sanitizer's tools reach it.
#[test]
fn run_compute_sanitizer_tools() {
    for tool in ["racecheck", "synccheck", "initcheck"] {
        let rig = route_rig(&["compute-sanitizer"]);
        let (code, err, log) = run_rig(
            &rig,
            &["--profile", "compute-sanitizer", "--profile-args", tool],
        );
        assert_eq!(code, 0, "{tool}: {err}");
        assert!(
            log.starts_with(&format!(
                "compute-sanitizer wrap=compute-sanitizer --tool={tool} --error-exitcode 5 \
                 --log-file bench-out/fake_bench.compute-sanitizer/sanitizer.log {}",
                rig.bench.display()
            )),
            "{tool}: {log}"
        );
    }
}

/// @test nsight's compute mode runs under ncu, into nsight's folder, and nsys
/// is not started.
#[test]
fn run_nsight_compute_mode_uses_ncu() {
    let rig = route_rig(&["nsys", "ncu"]);
    let (code, err, log) = run_rig(&rig, &["--profile", "nsight", "--profile-args", "compute"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        log,
        format!(
            "ncu wrap=nsight -o bench-out/fake_bench.nsight/kernel_profile -f --target-processes \
             all {} --profile nsight --profile-args compute\n",
            rig.bench.display()
        )
    );
}

/// @test A mode bench run does not wrap, a word the tool does not take and a
/// combination the tool cannot run are refused before anything starts or is
/// created.
#[test]
fn run_refuses_replay_and_unknown_modes() {
    for (profile, args, expected) in [
        (
            "nsight",
            "replay",
            "bench run does not wrap a kernel replay",
        ),
        ("ncu", "replay", "bench run does not wrap a kernel replay"),
        (
            "massif",
            "heap",
            "'heap' is not a mode of massif; its modes are pages, stacks",
        ),
        (
            "massif",
            "pages,stacks",
            "valgrind's massif cannot combine pages",
        ),
        (
            "callgrind",
            "instr",
            "'instr' is not a mode of callgrind, which takes none",
        ),
        (
            "compute-sanitizer",
            "memcheck racecheck",
            "compute-sanitizer runs one tool at a time",
        ),
    ] {
        let rig = route_rig(&["valgrind", "nsys", "ncu", "compute-sanitizer"]);
        let (code, err, log) = run_rig(&rig, &["--profile", profile, "--profile-args", args]);
        assert_eq!(code, 1, "{profile} {args}: {err}");
        assert!(
            err.contains(&format!(
                "Error: invalid arguments: --profile {profile} --profile-args '{args}': {expected}"
            )),
            "{profile} {args}: {err}"
        );
        assert_eq!(log, "", "{profile} {args}: something was started");
        assert!(
            !rig.dir.path().join("bench-out").exists(),
            "{profile} {args}: a folder was created"
        );
    }
}

/// @test --profile-args may start with a hyphen, and reaches the benchmark
/// as one argument.
#[test]
fn run_profile_args_may_start_with_a_hyphen() {
    // After a space, and attached with '='.
    for args in [
        &["--profile", "perf", "--profile-args", "-e cycles"][..],
        &["--profile", "perf", "--profile-args=-e cycles"][..],
    ] {
        let rig = route_rig(&[]);
        let (code, err, log) = run_rig(&rig, args);
        assert_eq!(code, 0, "{args:?}: {err}");
        // The stand-in benchmark logs its own arguments only.
        assert_eq!(
            log, "bench wrap= --profile perf --profile-args -e cycles\n",
            "{args:?}"
        );
    }
}

/* ----------------------------- Run: Completion ----------------------------- */

/// @test An output folder that cannot be created stops the run before
/// anything starts, naming the folder.
#[test]
fn run_refuses_an_output_folder_it_cannot_create() {
    let rig = route_rig(&["valgrind"]);
    let blocker = rig.dir.path().join("not-a-folder");
    std::fs::write(&blocker, "a file where the output root should be").expect("write blocker");
    let root = blocker.join("out");
    let (code, err, log) = run_rig(
        &rig,
        &[
            "--profile",
            "massif",
            "--profile-output-dir",
            &root.to_string_lossy(),
        ],
    );
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains(&format!(
            "Error: I/O error: cannot create the output folder {}",
            root.join("fake_bench.massif").display()
        )),
        "{err}"
    );
    assert_eq!(log, "", "something was started: {log}");
}

/// @test A wrapped run that exits 0 without its output, or with it empty,
/// fails by naming the file; with it, the run says what was written.
#[test]
fn run_wrapped_output_is_checked_after_exit() {
    let out = "bench-out/fake_bench.massif/massif.out";
    for (write, code, expected) in [
        (
            "none",
            1,
            format!("Error: --profile massif failed: completion: {out} was not written"),
        ),
        (
            "empty",
            1,
            format!("Error: --profile massif failed: completion: {out} is empty"),
        ),
        ("full", 0, format!("[bench] massif wrote {out} (")),
    ] {
        let rig = route_rig(&["valgrind"]);
        let (rc, stdout, err, log) =
            run_rig_env(&rig, &["--profile", "massif"], &[("FAKE_WRITE", write)]);
        assert_eq!(rc, code, "{write}: {err}");
        assert!(
            stdout.contains(&expected) || err.contains(&expected),
            "{write}: {stdout}{err}"
        );
        assert!(
            log.starts_with("valgrind "),
            "{write}: the wrap did not run: {log}"
        );
    }
}

/// @test heaptrack's output is found under the suffix its build chose.
#[test]
fn run_heaptrack_output_is_found_by_suffix() {
    let rig = route_rig(&["heaptrack"]);
    let (rc, stdout, err, _) = run_rig_env(&rig, &["--profile", "heaptrack"], &[]);
    assert_eq!(rc, 0, "{err}");
    assert!(
        stdout.contains("[bench] heaptrack wrote bench-out/fake_bench.heaptrack/run.gz ("),
        "{stdout}"
    );
}

/// @test A previous run's output cannot stand for this run's: the wrap's own
/// files are removed from its folder before the run, and nothing else is.
#[test]
fn run_wrapped_stale_output_fails() {
    let rig = route_rig(&["valgrind"]);
    let folder = rig.dir.path().join("bench-out/fake_bench.massif");
    std::fs::create_dir_all(&folder).expect("create the wrap folder");
    std::fs::write(folder.join("massif.out"), "a previous run's profile").expect("stale");
    std::fs::write(folder.join("massif.out.old"), "kept").expect("neighbour");
    std::fs::write(folder.join("notes.txt"), "kept").expect("neighbour");
    std::fs::write(rig.dir.path().join("bench-out/massif.out"), "kept").expect("outside");
    let (rc, stdout, err, _) =
        run_rig_env(&rig, &["--profile", "massif"], &[("FAKE_WRITE", "none")]);
    assert_eq!(rc, 1, "{err}");
    assert!(
        stdout
            .contains("[bench] removed bench-out/fake_bench.massif/massif.out from a previous run"),
        "{stdout}"
    );
    assert!(
        err.contains("completion: bench-out/fake_bench.massif/massif.out was not written"),
        "{err}"
    );
    assert!(
        !folder.join("massif.out").exists(),
        "the stale file stands for this run"
    );
    for kept in [
        folder.join("massif.out.old"),
        folder.join("notes.txt"),
        rig.dir.path().join("bench-out/massif.out"),
    ] {
        assert!(kept.exists(), "{} was removed", kept.display());
    }
}

/// @test A failing nsys stats fails the run as an analysis failure naming
/// the summary, and keeps the report.
#[test]
fn run_nsight_stats_failure_is_reported() {
    let rig = route_rig(&["nsys"]);
    let (rc, _, err, _) = run_rig_env(
        &rig,
        &["--profile", "nsight"],
        &[("FAKE_NSYS_STATS", "fail")],
    );
    assert_eq!(rc, 1, "{err}");
    assert!(
        err.contains(
            "Error: --profile nsight failed: analysis: nsys stats --report cuda_gpu_kern_sum \
             exited with status 1: fake nsys: cannot export the report; the report is kept at \
             bench-out/fake_bench.nsight/profile.nsys-rep"
        ),
        "{err}"
    );
    assert!(rig
        .dir
        .path()
        .join("bench-out/fake_bench.nsight/profile.nsys-rep")
        .is_file());
}

/// @test An `nsys stats` that leaves a process holding its output and exits:
/// bench run returns at once instead of waiting on it, ends each one itself
/// (this test only looks at them), says so, and writes every summary.
#[test]
fn run_nsight_stats_leaving_a_process() {
    let rig = route_rig(&["nsys"]);
    let run = run_rig_bounded(
        &rig,
        &["--profile", "nsight"],
        &[("FAKE_NSYS_STATS", "leave")],
        Sigint::Default,
        std::time::Duration::from_secs(30),
        |_| {},
    );
    let pids = std::fs::read_to_string(rig.dir.path().join("nsys_left.pids")).unwrap_or_default();
    let running: Vec<&str> = pids
        .lines()
        .filter(|pid| pid.parse().is_ok_and(still_running))
        .collect();
    assert!(
        run.returned && run.elapsed < std::time::Duration::from_secs(10),
        "bench run waited on what nsys stats left: returned {} after {:?}",
        run.returned,
        run.elapsed
    );
    assert_eq!(pids.lines().count(), 4, "{pids}");
    assert!(running.is_empty(), "still running: {running:?}");
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    for report in [
        "cuda_gpu_kern_sum",
        "cuda_api_sum",
        "cuda_gpu_mem_size_sum",
        "cuda_gpu_mem_time_sum",
    ] {
        assert!(
            run.stderr.contains(&format!(
                "[bench] nsys stats --report {report} exited and left 1 process of its own \
                 running; bench run ended it"
            )),
            "{}",
            run.stderr
        );
        assert_eq!(
            std::fs::read_to_string(
                rig.dir
                    .path()
                    .join(format!("bench-out/fake_bench.nsight/{report}.txt"))
            )
            .unwrap_or_default(),
            "fake summary\n",
            "{report}"
        );
    }
}

/// @test bench run interrupted while `nsys stats` runs (SIGINT sent to bench
/// run alone, at its default action): nsys stats and the process it started
/// are ended, and bench run ends by that signal.
#[test]
fn run_nsight_stats_interrupted() {
    let rig = route_rig(&["nsys"]);
    let stats = rig.dir.path().join("nsys_stats.pid");
    let run = run_rig_bounded(
        &rig,
        &["--profile", "nsight"],
        &[("FAKE_NSYS_STATS", "wait")],
        Sigint::Default,
        std::time::Duration::from_secs(30),
        |bench| {
            pid_written(&stats);
            let out =
                output_of(Command::new("/bin/sh").args(["-c", &format!("kill -INT {bench}")]));
            assert!(out.status.success(), "kill -INT {bench} failed");
        },
    );
    let stats = pid_written(&stats);
    let started = pid_written(&rig.dir.path().join("nsys_started.pid"));
    let (stats_running, started_running) = (still_running(stats), still_running(started));
    assert!(
        run.returned && run.elapsed < std::time::Duration::from_secs(10),
        "bench run did not end on SIGINT: returned {} after {:?}",
        run.returned,
        run.elapsed
    );
    assert_eq!(run.signal, Some(2), "{}", run.stderr);
    assert!(
        !stats_running && !started_running,
        "still running: nsys stats {stats} {stats_running}, its process {started} \
         {started_running}"
    );
}

/* ----------------------------- Run: Callgrind Analysis ----------------------------- */

/// @test --profile-analyze, given to bench run or forwarded after `--`,
/// annotates the callgrind profile after valgrind has written it, and the
/// benchmark is asked for the analysis once.
#[test]
fn run_callgrind_analyze_after_exit() {
    for forwarded in [false, true] {
        let rig = route_rig(&["valgrind", "callgrind_annotate"]);
        let args: &[&str] = if forwarded {
            &["--profile", "callgrind", "--", "--profile-analyze"]
        } else {
            &["--profile", "callgrind", "--profile-analyze"]
        };
        let (rc, stdout, err, log) = run_rig_env(&rig, args, &[]);
        assert_eq!(rc, 0, "{args:?}: {err}");
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2, "{args:?}: {log}");
        assert!(
            lines[0].starts_with("valgrind wrap=callgrind --tool=callgrind")
                && lines[0].ends_with("--profile callgrind --profile-analyze"),
            "{args:?}: the benchmark is asked for the analysis once: {log}"
        );
        assert_eq!(
            lines[1],
            "callgrind_annotate wrap= --auto=yes \
             bench-out/fake_bench.callgrind/callgrind.out",
            "{args:?}: annotated after the run: {log}"
        );
        assert!(
            stdout.contains(
                "--- callgrind_annotate bench-out/fake_bench.callgrind/callgrind.out \
                 (first 40 lines) ---\n\nfake annotation of \
                 bench-out/fake_bench.callgrind/callgrind.out\n"
            ),
            "{args:?}: {stdout}"
        );
    }
}

/// @test A failing or missing callgrind_annotate fails the run as an
/// analysis failure, and the profile is kept.
#[test]
fn run_callgrind_analyze_failure() {
    let profile = "bench-out/fake_bench.callgrind/callgrind.out";
    let rig = route_rig(&["valgrind", "callgrind_annotate"]);
    let (rc, _, err, _) = run_rig_env(
        &rig,
        &["--profile", "callgrind", "--profile-analyze"],
        &[("FAKE_ANNOTATE", "fail")],
    );
    assert_eq!(rc, 1, "{err}");
    assert!(
        err.contains(&format!(
            "Error: --profile callgrind failed: analysis: {} exited with status 1: fake annotate: \
             cannot read the profile; the profile is kept at {profile}",
            rig.dir.path().join("tools/callgrind_annotate").display()
        )),
        "{err}"
    );
    assert!(rig.dir.path().join(profile).is_file());

    // No profile: the run fails at completion, and nothing is annotated.
    let rig = route_rig(&["valgrind", "callgrind_annotate"]);
    let (rc, _, err, log) = run_rig_env(
        &rig,
        &["--profile", "callgrind", "--profile-analyze"],
        &[("FAKE_WRITE", "none")],
    );
    assert_eq!(rc, 1, "{err}");
    assert!(
        err.contains(&format!("completion: {profile} was not written")),
        "{err}"
    );
    assert!(
        !log.contains("callgrind_annotate"),
        "annotated a missing profile: {log}"
    );

    let rig = route_rig(&["valgrind"]);
    let (rc, _, err, _) = run_rig_env(&rig, &["--profile", "callgrind", "--profile-analyze"], &[]);
    assert_eq!(rc, 1, "{err}");
    assert!(
        err.contains(&format!(
            "analysis: callgrind_annotate is not on PATH (it ships with valgrind); the profile \
             is kept at {profile}"
        )),
        "{err}"
    );
    assert!(rig.dir.path().join(profile).is_file());
}

/// How a bounded `bench run` ended.
struct BoundedRun {
    /// False when bench run had not returned within the bound and was killed.
    returned: bool,
    elapsed: std::time::Duration,
    code: Option<i32>,
    signal: Option<i32>,
    stdout: String,
    stderr: String,
}

/// The SIGINT action a bounded run's `bench run` starts with. A child keeps
/// an ignored action across exec, so without this the test would run bench
/// with whatever this test process inherited: a background job of a
/// non-interactive shell starts with SIGINT ignored.
#[derive(Clone, Copy)]
enum Sigint {
    /// The default action: the signal ends the process.
    Default,
    /// Ignored, as such a background job starts.
    Ignored,
}

/// Give SIGINT the action @p sigint in the process @p command starts.
fn start_with_sigint(command: &mut Command, sigint: Sigint) {
    use std::os::unix::process::CommandExt;

    let handler = match sigint {
        Sigint::Default => libc::SIG_DFL,
        Sigint::Ignored => libc::SIG_IGN,
    };
    // SAFETY: the closure runs in the child between fork and exec and calls
    // only sigaction, which is async-signal-safe; a zeroed sigaction is a
    // valid value of the type, given the default or the ignore handler.
    unsafe {
        command.pre_exec(move || {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler;
            if libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
}

/// `bench run <rig's benchmark> <args>` with PATH set to the rig's stand-ins,
/// @p env added and SIGINT's action @p sigint, given @p bound to return; past
/// it, bench run is killed and the result says so. @p during runs once bench
/// run has started, with its process id.
fn run_rig_bounded(
    rig: &RouteRig,
    args: &[&str],
    env: &[(&str, &str)],
    sigint: Sigint,
    bound: std::time::Duration,
    during: impl FnOnce(u32),
) -> BoundedRun {
    use std::os::unix::process::ExitStatusExt;

    let out_path = rig.dir.path().join("bench.stdout");
    let err_path = rig.dir.path().join("bench.stderr");
    let mut command = Command::new(bin());
    command
        .arg("run")
        .arg(&rig.bench)
        .args(args)
        .env("PATH", rig.dir.path().join("tools"))
        .current_dir(rig.dir.path())
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&out_path).expect("bench stdout file"))
        .stderr(std::fs::File::create(&err_path).expect("bench stderr file"));
    for (k, v) in env {
        command.env(k, v);
    }
    start_with_sigint(&mut command, sigint);
    let started = std::time::Instant::now();
    let mut child = {
        let _gate = START_GATE.read().unwrap_or_else(|e| e.into_inner());
        command.spawn().expect("spawn bench")
    };
    during(child.id());
    let mut returned = true;
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait for bench") {
            break status;
        }
        if started.elapsed() >= bound {
            returned = false;
            let _ = child.kill();
            break child.wait().expect("reap bench");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    BoundedRun {
        returned,
        elapsed: started.elapsed(),
        code: status.code(),
        signal: status.signal(),
        stdout: std::fs::read_to_string(&out_path).unwrap_or_default(),
        stderr: std::fs::read_to_string(&err_path).unwrap_or_default(),
    }
}

/// Whether process @p pid still runs: listed in /proc and neither a zombie
/// nor dead. Read only; this file never signals a process a stand-in started.
fn still_running(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    let state = stat
        .rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("");
    !matches!(state, "" | "Z" | "X")
}

/// The process id a stand-in wrote to @p path, waiting up to 10 s for it.
fn pid_written(path: &Path) -> u32 {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(pid) = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| text.trim().parse().ok())
        {
            return pid;
        }
        assert!(
            std::time::Instant::now() < until,
            "{} was not written",
            path.display()
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// @test A callgrind_annotate that starts a process holding its output and
/// exits: bench run returns at once instead of waiting on that process, ends
/// it itself (this test only looks at it), prints the annotation and says
/// what it ended.
#[test]
fn run_callgrind_analyze_annotator_leaves_a_process() {
    let rig = route_rig(&["valgrind", "callgrind_annotate"]);
    let left = rig.dir.path().join("left.pid");
    write_executable(
        &rig.dir.path().join("tools/callgrind_annotate"),
        &format!(
            "#!/bin/sh\necho \"fake annotation of $2\"\n/bin/sleep 60 &\necho $! > '{}'\nexit 0\n",
            left.display()
        ),
    );
    let run = run_rig_bounded(
        &rig,
        &["--profile", "callgrind", "--profile-analyze"],
        &[],
        Sigint::Default,
        std::time::Duration::from_secs(20),
        |_| {},
    );
    let left = pid_written(&left);
    let left_running = still_running(left);
    assert!(
        run.returned && run.elapsed < std::time::Duration::from_secs(10),
        "bench run waited on the annotator's process: returned {} after {:?}",
        run.returned,
        run.elapsed
    );
    assert!(
        !left_running,
        "the annotator's process {left} still runs after bench run returned"
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(
        run.stdout
            .contains("fake annotation of bench-out/fake_bench.callgrind/callgrind.out"),
        "{}",
        run.stdout
    );
    assert!(
        run.stderr.contains(&format!(
            "[bench] {} exited and left 1 process of its own running; bench run ended it",
            rig.dir.path().join("tools/callgrind_annotate").display()
        )),
        "{}",
        run.stderr
    );
}

/// @test bench run interrupted while callgrind_annotate runs (SIGINT sent
/// to bench run alone, which starts with SIGINT's default action whatever
/// this test inherited): the annotator and the process it started are ended
/// by bench run, which then ends by that signal.
#[test]
fn run_callgrind_analyze_interrupted() {
    let rig = route_rig(&["valgrind", "callgrind_annotate"]);
    let annotator = rig.dir.path().join("annotator.pid");
    let started = rig.dir.path().join("started.pid");
    write_executable(
        &rig.dir.path().join("tools/callgrind_annotate"),
        &format!(
            "#!/bin/sh\n/bin/sleep 60 &\necho $! > '{}'\necho $$ > '{}'\nwait\n",
            started.display(),
            annotator.display()
        ),
    );
    let run = run_rig_bounded(
        &rig,
        &["--profile", "callgrind", "--profile-analyze"],
        &[],
        Sigint::Default,
        std::time::Duration::from_secs(20),
        |bench| {
            pid_written(&annotator);
            let out =
                output_of(Command::new("/bin/sh").args(["-c", &format!("kill -INT {bench}")]));
            assert!(out.status.success(), "kill -INT {bench} failed");
        },
    );
    let (annotator, started) = (pid_written(&annotator), pid_written(&started));
    let (annotator_running, started_running) = (still_running(annotator), still_running(started));
    assert!(
        run.returned && run.elapsed < std::time::Duration::from_secs(10),
        "bench run did not end on SIGINT: returned {} after {:?}",
        run.returned,
        run.elapsed
    );
    assert_eq!(
        run.signal,
        Some(2),
        "bench run did not end by SIGINT: {}",
        run.stderr
    );
    assert!(
        !annotator_running && !started_running,
        "still running after bench run ended: annotator {annotator} {annotator_running}, \
         its process {started} {started_running}"
    );
}

/// @test bench run started with SIGINT ignored, as a background job of a
/// non-interactive shell starts, leaves it ignored: SIGINT sent while
/// callgrind_annotate runs does not end the run, and the annotation ends by
/// itself, well within its bound, and is printed.
#[test]
fn run_callgrind_analyze_inherited_ignore_stays() {
    let rig = route_rig(&["valgrind", "callgrind_annotate"]);
    let annotator = rig.dir.path().join("annotator.pid");
    write_executable(
        &rig.dir.path().join("tools/callgrind_annotate"),
        &format!(
            "#!/bin/sh\necho $$ > '{}'\n/bin/sleep 2\necho \"fake annotation of $2\"\nexit 0\n",
            annotator.display()
        ),
    );
    let run = run_rig_bounded(
        &rig,
        &["--profile", "callgrind", "--profile-analyze"],
        &[],
        Sigint::Ignored,
        std::time::Duration::from_secs(20),
        |bench| {
            pid_written(&annotator);
            let out =
                output_of(Command::new("/bin/sh").args(["-c", &format!("kill -INT {bench}")]));
            assert!(out.status.success(), "kill -INT {bench} failed");
        },
    );
    let annotator = pid_written(&annotator);
    let annotator_running = still_running(annotator);
    assert!(
        run.returned && run.elapsed < std::time::Duration::from_secs(10),
        "the annotation did not end within its time: returned {} after {:?}",
        run.returned,
        run.elapsed
    );
    assert_eq!(run.signal, None, "an ignored SIGINT ended bench run");
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(
        run.stdout
            .contains("fake annotation of bench-out/fake_bench.callgrind/callgrind.out"),
        "{}",
        run.stdout
    );
    assert!(!annotator_running, "the annotator {annotator} still runs");
}

/* ----------------------------- Run: The Benchmark's Own Path ----------------------------- */

/// A route rig with `mybench` in its working directory, a stand-in benchmark
/// that logs "local <args>"; with @p impostor, another `mybench` first on
/// PATH logs "impostor <args>". The local one is executable unless
/// @p local_executable is false.
fn bare_name_rig(programs: &[&str], impostor: bool, local_executable: bool) -> RouteRig {
    let rig = route_rig(programs);
    let local = rig.dir.path().join("mybench");
    write_executable(
        &local,
        &format!("#!/bin/sh\necho \"local $*\" >> '{}'\n", rig.log.display()),
    );
    if !local_executable {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&local, std::fs::Permissions::from_mode(0o644)).expect("chmod");
    }
    if impostor {
        write_executable(
            &rig.dir.path().join("tools/mybench"),
            &format!(
                "#!/bin/sh\necho \"impostor $*\" >> '{}'\n",
                rig.log.display()
            ),
        );
    }
    rig
}

/// `bench <command> mybench <args>` in the rig's folder, PATH its stand-ins:
/// the exit status, stdout, stderr and the stand-ins' log.
fn run_bare_name(rig: &RouteRig, command: &str, args: &[&str]) -> (i32, String, String, String) {
    let out = output_of(
        Command::new(bin())
            .arg(command)
            .arg("mybench")
            .args(args)
            .env("PATH", rig.dir.path().join("tools"))
            .current_dir(rig.dir.path()),
    );
    (
        out.status.code().unwrap_or(255),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        std::fs::read_to_string(&rig.log).unwrap_or_default(),
    )
}

/// @test A benchmark named without a directory is the working directory's,
/// as `bench doctor` reads it: `bench run mybench` starts `./mybench` when
/// nothing of that name is on PATH, and when another `mybench` is.
#[test]
fn run_bare_name_is_the_working_directorys() {
    for impostor in [false, true] {
        let rig = bare_name_rig(&[], impostor, true);
        let (code, stdout, err, log) = run_bare_name(&rig, "run", &["--cycles", "1"]);
        assert_eq!(code, 0, "impostor {impostor}: {err}");
        assert!(
            stdout.contains("Running: ./mybench --cycles 1"),
            "impostor {impostor}: {stdout}"
        );
        assert_eq!(log, "local --cycles 1\n", "impostor {impostor}");
    }
}

/// @test The taskset and wrapped routes and `bench profile-all` hand the
/// tools `./mybench`, which no tool looks up on PATH, with an impostor there.
#[test]
fn run_bare_name_reaches_every_route() {
    let rig = bare_name_rig(&["taskset", "valgrind"], true, true);
    let (code, _, err, log) = run_bare_name(&rig, "run", &["--taskset", "0"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        log.starts_with("taskset wrap= -c 0 ./mybench"),
        "taskset: {log}"
    );
    let rig = bare_name_rig(&["taskset", "valgrind"], true, true);
    let (code, _, err, log) = run_bare_name(&rig, "run", &["--profile", "massif"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        log.starts_with("valgrind wrap=massif --tool=massif") && log.contains(" ./mybench "),
        "wrapped: {log}"
    );
    let rig = bare_name_rig(&["taskset", "valgrind"], true, true);
    let out = rig.dir.path().join("out");
    let (code, _, err, log) = run_bare_name(
        &rig,
        "profile-all",
        &[
            "--profilers",
            "massif",
            "--out",
            out.to_str().expect("UTF-8"),
        ],
    );
    assert_eq!(code, 0, "{err}");
    assert!(
        log.starts_with("valgrind wrap=massif --tool=massif") && log.contains(" ./mybench "),
        "profile-all: {log}"
    );
    assert!(!log.contains("impostor"), "{log}");
}

/// @test A working-directory benchmark without an execute bit is refused
/// before anything starts, naming it; the `mybench` on PATH is not run in
/// its place.
#[test]
fn run_bare_name_not_executable_is_refused() {
    let rig = bare_name_rig(&[], true, false);
    let (code, _, err, log) = run_bare_name(&rig, "run", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains("invalid arguments: ./mybench is not executable"),
        "{err}"
    );
    assert_eq!(log, "", "something ran: {log}");
    let (code, _, err, log) = run_bare_name(&rig, "profile-all", &["--profilers", "massif"]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains("invalid arguments: ./mybench is not executable"),
        "{err}"
    );
    assert_eq!(log, "", "something ran: {log}");
}

/* ----------------------------- Profile-all ----------------------------- */

/// A route rig whose stand-in benchmark logs its argv and ends with the status
/// `FAKE_EXIT_<tool>` names for the `--profile <tool>` it is given, else
/// `FAKE_EXIT`, else 0. The rig's valgrind stand-in writes callgrind's output
/// (`FAKE_WRITE=none`: nothing).
fn profile_all_rig() -> RouteRig {
    let rig = route_rig(&["valgrind"]);
    let script = format!(
        r#"#!/bin/sh
echo "bench $*" >> '{log}'
tool=""
prev=""
for a in "$@"; do
  [ "$prev" = --profile ] && tool="$a"
  prev="$a"
done
eval "code=\${{FAKE_EXIT_$tool:-\${{FAKE_EXIT:-0}}}}"
exit "$code"
"#,
        log = rig.log.display()
    );
    write_executable(&rig.bench, &script);
    rig
}

/// Run `bench profile-all` on the rig's benchmark with its stand-ins on PATH,
/// into `<rig>/out` (or `out`, when given); returns the exit status, stderr
/// and the stand-ins' log.
fn run_profile_all(
    rig: &RouteRig,
    profilers: &str,
    out: Option<&Path>,
    env: &[(&str, &str)],
) -> (i32, String, String) {
    let default_out = rig.dir.path().join("out");
    let out = out.unwrap_or(&default_out);
    let mut command = Command::new(bin());
    command
        .args(["profile-all"])
        .arg(&rig.bench)
        .args(["--profilers", profilers, "--out"])
        .arg(out)
        .env("PATH", rig.dir.path().join("tools"))
        .current_dir(rig.dir.path());
    for (k, v) in env {
        command.env(k, v);
    }
    let output = output_of(&mut command);
    (
        output.status.code().unwrap_or(255),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        std::fs::read_to_string(&rig.log).unwrap_or_default(),
    )
}

/// The lines of profile-all's summary, one per run, as printed.
fn profile_all_summary(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .skip_while(|l| *l != "=== bench profile-all: summary ===")
        .skip(1)
        .take_while(|l| l.starts_with("  "))
        .map(str::to_string)
        .collect()
}

/// @test profile-all runs every profiler and, when each completes, exits 0
/// after one summary line per run naming its folder.
#[test]
fn profile_all_all_succeed() {
    let rig = profile_all_rig();
    let (code, err, log) = run_profile_all(&rig, "gperf,perf,callgrind", None, &[]);
    assert_eq!(code, 0, "{err}");
    let out = rig.dir.path().join("out");
    assert_eq!(
        profile_all_summary(&err),
        [
            format!("  gperf      completed  {}", out.join("gperf").display()),
            format!("  perf       completed  {}", out.join("perf").display()),
            format!(
                "  callgrind  completed  {}",
                out.join("callgrind").display()
            ),
        ],
        "{err}"
    );
    assert!(!err.contains("Error:"), "{err}");
    assert!(
        log.contains("--profile gperf") && log.contains("--profile perf"),
        "{log}"
    );
    assert!(
        log.contains("valgrind wrap=callgrind --tool=callgrind"),
        "{log}"
    );
}

/// @test A profiler that fails does not stop the others; the summary names it
/// with its reason, and profile-all exits 1 naming how many of how many failed.
#[test]
fn profile_all_mixed() {
    let rig = profile_all_rig();
    let (code, err, log) = run_profile_all(
        &rig,
        "gperf,perf,callgrind",
        None,
        &[("FAKE_EXIT_perf", "4")],
    );
    assert_eq!(code, 1, "{err}");
    let out = rig.dir.path().join("out");
    assert_eq!(
        profile_all_summary(&err),
        [
            format!("  gperf      completed  {}", out.join("gperf").display()),
            format!(
                "  perf       failed     {} -- the requested profile failed (the benchmark's \
                 report above says why); the benchmark exited with status 4",
                out.join("perf").display()
            ),
            format!(
                "  callgrind  completed  {}",
                out.join("callgrind").display()
            ),
        ],
        "{err}"
    );
    assert!(
        err.ends_with("Error: 1 of 3 profile runs failed: perf\n"),
        "{err}"
    );
    assert!(
        log.contains("valgrind wrap=callgrind"),
        "callgrind ran after perf failed: {log}"
    );
}

/// @test When every profiler fails, each is still run and reported, and
/// profile-all exits 1.
#[test]
fn profile_all_all_fail() {
    let rig = profile_all_rig();
    let (code, err, log) = run_profile_all(
        &rig,
        "gperf,perf,callgrind",
        None,
        &[("FAKE_EXIT", "1"), ("FAKE_WRITE", "none")],
    );
    assert_eq!(code, 1, "{err}");
    let out = rig.dir.path().join("out");
    let callgrind = out.join("callgrind");
    let profile = callgrind.join("fake_bench.callgrind").join("callgrind.out");
    assert_eq!(
        profile_all_summary(&err),
        [
            format!(
                "  gperf      failed     {} -- the benchmark exited with status 1",
                out.join("gperf").display()
            ),
            format!(
                "  perf       failed     {} -- the benchmark exited with status 1",
                out.join("perf").display()
            ),
            format!(
                "  callgrind  failed     {} -- --profile callgrind failed: completion: {} was \
                 not written",
                callgrind.display(),
                profile.display()
            ),
        ],
        "{err}"
    );
    assert!(
        err.ends_with("Error: 3 of 3 profile runs failed: gperf, perf, callgrind\n"),
        "{err}"
    );
    assert_eq!(log.matches("--profile gperf").count(), 1, "{log}");
    assert_eq!(log.matches("--profile perf").count(), 1, "{log}");
}

/// @test A folder profile-all cannot create fails that run, not the others'
/// turn: every profiler is reported.
#[test]
fn profile_all_reports_a_folder_it_cannot_create() {
    let rig = profile_all_rig();
    let blocker = rig.dir.path().join("not-a-folder");
    std::fs::write(&blocker, "a regular file").expect("write the blocker");
    let root = blocker.join("out");
    let (code, err, log) = run_profile_all(&rig, "gperf,perf", Some(&root), &[]);
    assert_eq!(code, 1, "{err}");
    let summary = profile_all_summary(&err);
    assert_eq!(summary.len(), 2, "{err}");
    for (line, tool) in summary.iter().zip(["gperf", "perf"]) {
        assert!(
            line.contains(&format!(
                "failed     {} -- I/O error: cannot create the output folder {}",
                root.join(tool).display(),
                root.join(tool).display()
            )),
            "{line}"
        );
    }
    assert!(
        log.is_empty(),
        "nothing may start without its folder: {log}"
    );
    assert!(
        err.ends_with("Error: 2 of 2 profile runs failed: gperf, perf\n"),
        "{err}"
    );
}

/* ----------------------------- Run: Compute Sanitizer ----------------------------- */

/// The report path of a compute-sanitizer run of the rig's benchmark.
const SANITIZER_REPORT: &str = "bench-out/fake_bench.compute-sanitizer/sanitizer.log";

/// Run `bench run --profile compute-sanitizer` on a rig whose stand-in writes
/// the report `scenario` names; returns the exit status, stdout, stderr and
/// the stand-ins' log.
fn run_sanitizer(scenario: &str, extra: &[(&str, &str)]) -> (i32, String, String, String) {
    let rig = route_rig(&["compute-sanitizer"]);
    let mut env = vec![("FAKE_SANITIZER", scenario)];
    env.extend_from_slice(extra);
    run_rig_env(&rig, &["--profile", "compute-sanitizer"], &env)
}

/// @test A clean application with no findings passes: the tool's report counts
/// no errors and the run prints where it is.
#[test]
fn run_compute_sanitizer_clean() {
    let (code, stdout, err, log) = run_sanitizer("clean", &[]);
    assert_eq!(code, 0, "{err}");
    assert!(
        stdout.contains(&format!(
            "[bench] compute-sanitizer wrote {SANITIZER_REPORT} ("
        )),
        "{stdout}"
    );
    assert!(log.contains("--error-exitcode 5"), "{log}");
}

/// @test A successful application with a reported invalid access fails the
/// run, the count taken from the report's summary, and names the report.
#[test]
fn run_compute_sanitizer_findings() {
    let (code, _, err, _) = run_sanitizer("findings", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: compute-sanitizer reported 1 error in the benchmark; the report is \
             {SANITIZER_REPORT} (the tool exited with status 5)\n"
        )),
        "{err}"
    );
}

/// @test racecheck summarizes in words of its own: a clean racecheck run
/// passes, and one with a reported hazard fails with the summary's error
/// count, as the other tools' runs do.
#[test]
fn run_compute_sanitizer_racecheck() {
    let run = |scenario: &str| {
        let rig = route_rig(&["compute-sanitizer"]);
        run_rig_env(
            &rig,
            &[
                "--profile",
                "compute-sanitizer",
                "--profile-args",
                "racecheck",
            ],
            &[("FAKE_SANITIZER", scenario)],
        )
    };
    let (code, stdout, err, log) = run("race-clean");
    assert_eq!(code, 0, "{err}");
    assert!(
        stdout.contains(&format!(
            "[bench] compute-sanitizer wrote {SANITIZER_REPORT} ("
        )),
        "{stdout}"
    );
    assert!(log.contains("--tool=racecheck --error-exitcode 5"), "{log}");
    let (code, _, err, _) = run("race-findings");
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: compute-sanitizer reported 1 error in the benchmark; the report is \
             {SANITIZER_REPORT} (the tool exited with status 5)\n"
        )),
        "{err}"
    );
}

/// @test racecheck's warnings are not errors: a run whose summary counts
/// warnings and no errors (the tool ending with the benchmark's status) passes;
/// with one error beside them, the tool ends with the reserved status and the
/// run fails with the summary's error count alone.
#[test]
fn run_compute_sanitizer_racecheck_warnings() {
    let run = |scenario: &str| {
        let rig = route_rig(&["compute-sanitizer"]);
        run_rig_env(
            &rig,
            &[
                "--profile",
                "compute-sanitizer",
                "--profile-args",
                "racecheck",
            ],
            &[("FAKE_SANITIZER", scenario)],
        )
    };
    let (code, stdout, err, _) = run("race-warnings");
    assert_eq!(code, 0, "{err}");
    assert!(
        stdout.contains(&format!(
            "[bench] compute-sanitizer wrote {SANITIZER_REPORT} ("
        )),
        "{stdout}"
    );
    let (code, _, err, _) = run("race-warnings-findings");
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: compute-sanitizer reported 1 error in the benchmark; the report is \
             {SANITIZER_REPORT} (the tool exited with status 5)\n"
        )),
        "{err}"
    );
}

/// @test A failing application keeps its own status when the tool finds
/// nothing; when the tool also finds errors, it returns its own status in
/// place of the application's, and the run says both.
#[test]
fn run_compute_sanitizer_application_failure() {
    let (code, _, err, _) = run_sanitizer("app-fail", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with("Error: the benchmark exited with status 1\n"),
        "{err}"
    );
    let (code, _, err, _) = run_sanitizer("app-fail-findings", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: compute-sanitizer reported 3 errors in the benchmark; the report is \
             {SANITIZER_REPORT} (the tool exited with status 5); the report also says the \
             benchmark returned an error, whose status the tool does not pass on\n"
        )),
        "{err}"
    );
}

/// @test A tool that fails before the benchmark ran to its end is a
/// collection failure in the tool's own words, not the benchmark's status.
#[test]
fn run_compute_sanitizer_startup_failure() {
    let (code, _, err, _) = run_sanitizer("startup", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: --profile compute-sanitizer failed: collection: compute-sanitizer \
             reported its own error, \"Error: Target application terminated before first \
             instrumented API call\" (the tool exited with status 255); the report is \
             {SANITIZER_REPORT}\n"
        )),
        "{err}"
    );
}

/// @test An absent or truncated report is named by its path and nothing is
/// counted, whatever the tool's status.
#[test]
fn run_compute_sanitizer_report_missing() {
    let (code, _, err, _) = run_sanitizer("no-report", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: --profile compute-sanitizer failed: completion: compute-sanitizer exited \
             with status 5 and wrote no report at {SANITIZER_REPORT}: nothing can be counted\n"
        )),
        "{err}"
    );
    for status in ["5", "0"] {
        let (code, _, err, _) = run_sanitizer("truncated", &[("FAKE_SANITIZER_EXIT", status)]);
        assert_eq!(code, 1, "exit {status}: {err}");
        assert!(
            err.ends_with(&format!(
                "Error: --profile compute-sanitizer failed: completion: compute-sanitizer \
                 exited with status {status}, and its report {SANITIZER_REPORT} holds no \
                 error summary: it is incomplete, and nothing in it is counted\n"
            )),
            "exit {status}: {err}"
        );
        assert!(!err.contains("reported"), "exit {status}: {err}");
    }
}

/// @test An application that returns the reserved status itself is reported
/// by its own status, never as a finding.
#[test]
fn run_compute_sanitizer_status_collision() {
    let (code, stdout, err, _) = run_sanitizer("collision", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with("Error: the benchmark exited with status 5\n"),
        "{err}"
    );
    assert!(
        stdout.contains(&format!(
            "[bench] compute-sanitizer's report {SANITIZER_REPORT} counts no errors: status 5 \
             is the benchmark's own"
        )),
        "{stdout}"
    );
    assert!(!err.contains("reported"), "{err}");
}

/// @test A benchmark that did not end normally under the tool is reported in
/// the report's words, not as a plain status.
#[test]
fn run_compute_sanitizer_abnormal_end() {
    let (code, _, err, _) = run_sanitizer("abnormal", &[]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.ends_with(&format!(
            "Error: the benchmark did not end normally under compute-sanitizer, whose report \
             says \"Error: process didn't terminate successfully\" (the tool exited with status \
             9); the report is {SANITIZER_REPORT}\n"
        )),
        "{err}"
    );
}

/// @test An output folder holding '%' reaches the tool with each '%' doubled,
/// and the report lands in the folder as named.
#[test]
fn run_compute_sanitizer_log_path_with_percent() {
    let rig = route_rig(&["compute-sanitizer"]);
    let (code, stdout, err, log) = run_rig_env(
        &rig,
        &[
            "--profile",
            "compute-sanitizer",
            "--profile-output-dir",
            "out%p",
        ],
        &[],
    );
    assert_eq!(code, 0, "{err}");
    assert!(
        log.contains("--log-file out%%p/fake_bench.compute-sanitizer/sanitizer.log"),
        "{log}"
    );
    let report = rig
        .dir
        .path()
        .join("out%p/fake_bench.compute-sanitizer/sanitizer.log");
    assert!(report.is_file(), "{stdout}{err}");
    assert!(
        stdout.contains(
            "[bench] compute-sanitizer wrote out%p/fake_bench.compute-sanitizer/sanitizer.log ("
        ),
        "{stdout}"
    );
}

/// @test An output folder holding '%' reaches valgrind with each '%' doubled
/// in its output option, for each valgrind route, and the output lands in
/// the folder as named.
#[test]
fn run_valgrind_output_folder_with_percent() {
    for (tool, option, file) in [
        ("callgrind", "--callgrind-out-file=", "callgrind.out"),
        ("massif", "--massif-out-file=", "massif.out"),
        ("memcheck", "--log-file=", "memcheck.log"),
        ("helgrind", "--log-file=", "helgrind.log"),
    ] {
        for (folder, spelled) in [("out%p", "out%%p"), ("a%x", "a%%x")] {
            let rig = route_rig(&["valgrind"]);
            let (code, stdout, err, log) = run_rig_env(
                &rig,
                &["--profile", tool, "--profile-output-dir", folder],
                &[],
            );
            assert_eq!(code, 0, "{tool} {folder}: {err}");
            assert!(
                log.contains(&format!("{option}{spelled}/fake_bench.{tool}/{file} ")),
                "{tool} {folder}: {log}"
            );
            let output = format!("{folder}/fake_bench.{tool}/{file}");
            assert!(
                rig.dir.path().join(&output).is_file(),
                "{tool} {folder}: {stdout}{err}"
            );
            assert!(
                stdout.contains(&format!("[bench] {tool} wrote {output} (")),
                "{tool} {folder}: {stdout}"
            );
        }
    }
}

/// @test An output folder holding '%' reaches nsys and ncu with each '%'
/// doubled in -o, for nsight's nsys route (also spelled nsys), its compute
/// mode and the ncu route, and the report lands in the folder as named.
#[test]
fn run_nsight_output_folder_with_percent() {
    for (profile, mode, tool, file) in [
        ("nsight", None, "nsight", "profile.nsys-rep"),
        ("nsys", None, "nsight", "profile.nsys-rep"),
        (
            "nsight",
            Some("compute"),
            "nsight",
            "kernel_profile.ncu-rep",
        ),
        ("ncu", None, "ncu", "kernel_profile.ncu-rep"),
    ] {
        for (folder, spelled) in [("out%p", "out%%p"), ("a%%b", "a%%%%b")] {
            let rig = route_rig(&["nsys", "ncu"]);
            let mut args = vec!["--profile", profile, "--profile-output-dir", folder];
            if let Some(mode) = mode {
                args.extend(["--profile-args", mode]);
            }
            let what = format!("{profile} {mode:?} {folder}");
            let (code, stdout, err, log) = run_rig_env(&rig, &args, &[]);
            assert_eq!(code, 0, "{what}: {err}");
            let stem = file.split('.').next().unwrap_or_default();
            assert!(
                log.contains(&format!(" -o {spelled}/fake_bench.{tool}/{stem} ")),
                "{what}: {log}"
            );
            let output = format!("{folder}/fake_bench.{tool}/{file}");
            assert!(
                rig.dir.path().join(&output).is_file(),
                "{what}: {stdout}{err}"
            );
            assert!(
                stdout.contains(&format!("[bench] {tool} wrote {output} (")),
                "{what}: {stdout}"
            );
        }
    }
}

/// @test heaptrack replaces %h and %p in -o and has no escape for them: an
/// output folder holding either is refused before anything starts or is
/// created, naming the folder, and one holding any other '%' reaches
/// heaptrack as it is and gets the trace.
#[test]
fn run_heaptrack_output_folder_with_percent() {
    for (folder, held) in [("out%h", "%h"), ("out%p", "%p")] {
        let rig = route_rig(&["heaptrack"]);
        let (code, err, log) = run_rig(
            &rig,
            &["--profile", "heaptrack", "--profile-output-dir", folder],
        );
        assert_eq!(code, 1, "{folder}: {err}");
        assert!(
            err.contains(&format!(
                "Error: invalid arguments: --profile heaptrack: heaptrack replaces %h (the host \
                 name) and %p (its process id) in its -o value and has no escape for them, so \
                 it cannot write into {folder}/fake_bench.heaptrack, which holds {held}; choose \
                 a --profile-output-dir without %h or %p"
            )),
            "{folder}: {err}"
        );
        assert_eq!(log, "", "{folder}: nothing may start");
        let created: Vec<_> = std::fs::read_dir(rig.dir.path())
            .expect("the rig's folder")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "tools" && n != "fake_bench")
            .collect();
        assert!(
            created.is_empty(),
            "{folder}: nothing may be created: {created:?}"
        );
    }
    let rig = route_rig(&["heaptrack"]);
    let folder = "a%%b%x";
    let (code, stdout, err, log) = run_rig_env(
        &rig,
        &["--profile", "heaptrack", "--profile-output-dir", folder],
        &[],
    );
    assert_eq!(code, 0, "{err}");
    assert!(
        log.contains(&format!(" -o {folder}/fake_bench.heaptrack/run ")),
        "{log}"
    );
    let output = format!("{folder}/fake_bench.heaptrack/run.gz");
    assert!(rig.dir.path().join(&output).is_file(), "{stdout}{err}");
    assert!(
        stdout.contains(&format!("[bench] heaptrack wrote {output} (")),
        "{stdout}"
    );
}

/* ----------------------------- Doctor ----------------------------- */

/// A doctor document with one backend that is ready and one that is not.
const DOCTOR_DOC: &str = r#"{"binary": {"frameInfo": "ok"}, "backendScope": "default-mode", "backends": [{"name": "perf", "status": "ok", "message": "perf stat counted", "hint": ""}, {"name": "offcpu", "status": "fail", "message": "missing: bpftrace not found on PATH", "hint": "apt install bpftrace"}]}
"#;

/// A stand-in benchmark whose doctor prints @p doc for --profile-check-json
/// and a line of text for --profile-check, and appends its arguments to
/// `argv.log` beside it; returns its path.
fn doctor_stand_in(dir: &Path, doc: &str) -> PathBuf {
    let doc_file = dir.join("doc.json");
    std::fs::write(&doc_file, doc).expect("write the document");
    let bench = dir.join("doctor_bench");
    write_executable(
        &bench,
        &format!(
            "#!/bin/sh\necho \"$*\" >> '{}'\ncase \"$*\" in\n  *--profile-check-json*) cat '{}' ;;\n  *--profile-check*) echo 'text doctor' ;;\nesac\n",
            dir.join("argv.log").display(),
            doc_file.display()
        ),
    );
    bench
}

/// Run `bench doctor <args>`; returns the exit status, stdout and stderr.
fn run_doctor(args: &[&str]) -> (i32, String, String) {
    let mut all = vec!["doctor"];
    all.extend_from_slice(args);
    run(&all)
}

/// @test With --json and --require, stdout is the binary's document and
/// nothing else, whether the requirements are met (exit 0) or not (exit 1);
/// the verdict goes to stderr.
#[test]
fn doctor_json_require_is_one_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let b = bench.to_string_lossy().into_owned();
    for (require, code, verdict) in [
        ("perf", 0, "[require] perf: OK\n"),
        (
            "perf,offcpu",
            1,
            "[require] offcpu: NOT READY (missing: bpftrace not found on PATH)\n",
        ),
    ] {
        let (rc, out, err) = run_doctor(&[&b, "--json", "--require", require]);
        assert_eq!(rc, code, "--require {require}: {err}");
        assert_eq!(
            out, DOCTOR_DOC,
            "--require {require}: stdout is the document"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("stdout parses as one JSON document");
        assert_eq!(parsed["backends"][0]["name"], "perf");
        assert!(err.contains(verdict), "--require {require}: {err}");
    }
}

/// @test --json alone prints the document and exits with the binary's status.
#[test]
fn doctor_json_prints_the_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let (rc, out, err) = run_doctor(&[&bench.to_string_lossy(), "--json"]);
    assert_eq!(rc, 0, "{err}");
    assert_eq!(out, DOCTOR_DOC);
}

/// @test A binary whose doctor prints no valid document is an error on
/// stderr, and nothing reaches stdout.
#[test]
fn doctor_unparseable_document_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), "[doctor] not a document\n");
    let b = bench.to_string_lossy().into_owned();
    for args in [
        vec![b.as_str(), "--json"],
        vec![b.as_str(), "--require", "perf"],
    ] {
        let (rc, out, err) = run_doctor(&args);
        assert_eq!(rc, 1, "{args:?}: {err}");
        assert_eq!(out, "", "{args:?}: stdout");
        assert!(
            err.contains("--profile-check-json printed no valid doctor document"),
            "{args:?}: {err}"
        );
    }
}

/// @test Without --json, the --require verdict is the output, on stdout.
#[test]
fn doctor_require_without_json_prints_the_verdict() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let (rc, out, _) = run_doctor(&[&bench.to_string_lossy(), "--require", "offcpu"]);
    assert_eq!(rc, 1);
    assert!(
        out.starts_with("[require] offcpu: NOT READY"),
        "the verdict on stdout: {out}"
    );
    assert!(!out.contains("\"backends\""), "{out}");
}

/// @test bench doctor passes a profile request to the binary as bench run
/// does: the canonical --profile, its mode (which may start with '-'), the
/// analysis, and the arguments after --.
#[test]
fn doctor_forwards_the_request() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let b = bench.to_string_lossy().into_owned();
    let (rc, _, err) = run_doctor(&[
        &b,
        "--profile",
        "nsys",
        "--profile-args",
        "-e cycles",
        "--profile-analyze",
        "--",
        "--gtest_filter=A.B",
    ]);
    assert_eq!(rc, 0, "{err}");
    let (rc, out, err) = run_doctor(&[&b, "--json", "--profile", "massif", "--", "--x"]);
    assert_eq!(rc, 0, "{err}");
    assert_eq!(out, DOCTOR_DOC);
    let argv = std::fs::read_to_string(dir.path().join("argv.log")).unwrap_or_default();
    assert_eq!(
        argv,
        "--profile-check --profile nsight --profile-args -e cycles --profile-analyze \
         --gtest_filter=A.B\n--profile-check-json --profile massif --x\n"
    );
}

/// @test --require judges the requested backend by the request's own row:
/// a default mode that is ready does not meet a requested mode that is not.
/// Other backends keep their default-mode rows.
#[test]
fn doctor_require_uses_the_selected_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let doc = DOCTOR_DOC.trim_end().trim_end_matches('}').to_string()
        + r#", "selected": {"name": "perf", "profileArgs": "record", "status": "fail", "message": "denied: perf record", "hint": "lower paranoid"}}"#
        + "\n";
    let bench = doctor_stand_in(dir.path(), &doc);
    let b = bench.to_string_lossy().into_owned();
    let (rc, out, _) = run_doctor(&[&b, "--require", "perf"]);
    assert_eq!(rc, 0, "without --profile, perf's default-mode row: {out}");
    let (rc, out, _) = run_doctor(&[
        &b,
        "--profile",
        "perf",
        "--profile-args",
        "record",
        "--require",
        "perf",
    ]);
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.contains(
            "[require] perf (--profile perf --profile-args 'record'): NOT READY (denied: perf record)"
        ),
        "{out}"
    );
    let (rc, out, _) = run_doctor(&[&b, "--profile", "perf", "--require", "offcpu"]);
    assert_eq!(rc, 1, "{out}");
    assert!(out.contains("[require] offcpu: NOT READY"), "{out}");
}

/// @test A binary that reports no selected row cannot meet a requirement on
/// the requested backend, and says why.
#[test]
fn doctor_require_old_binary_is_unmet() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bench = doctor_stand_in(dir.path(), DOCTOR_DOC);
    let (rc, out, _) = run_doctor(&[
        &bench.to_string_lossy(),
        "--profile",
        "perf",
        "--require",
        "perf",
    ]);
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.contains(
            "[require] perf: this binary does not report selected requests; rebuild it \
             against this vernier, or drop --profile"
        ),
        "{out}"
    );
}

/// @test The doctor checks the request bench run would make of the same
/// arguments: a profile given after `--` selects the row it judges, and one
/// that conflicts with --profile is refused before the binary runs, with
/// nothing on stdout.
#[test]
fn doctor_checks_the_request_bench_run_makes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let doc = DOCTOR_DOC.trim_end().trim_end_matches('}').to_string()
        + r#", "selected": {"name": "gperf", "profileArgs": "", "status": "ok", "message": "gperftools profiles cpu", "hint": ""}}"#
        + "\n";
    let bench = doctor_stand_in(dir.path(), &doc);
    let b = bench.to_string_lossy().into_owned();
    let (rc, out, err) = run_doctor(&[&b, "--require", "gperf", "--", "--profile", "gperf"]);
    assert_eq!(rc, 0, "{out}{err}");
    assert_eq!(out, "[require] gperf (--profile gperf): OK\n");
    let (rc, out, err) = run_doctor(&[
        &b,
        "--profile",
        "perf",
        "--require",
        "perf",
        "--json",
        "--",
        "--profile",
        "gperf",
    ]);
    assert_eq!(rc, 1, "{err}");
    assert_eq!(out, "", "nothing on stdout");
    assert!(
        err.contains(
            "Error: invalid arguments: --profile is given more than once with different values \
             ('perf' to bench doctor, 'gperf' after --); give it once"
        ),
        "{err}"
    );
    let argv = std::fs::read_to_string(dir.path().join("argv.log")).unwrap_or_default();
    assert_eq!(
        argv, "--profile-check-json --profile gperf\n",
        "the refused request reached the binary"
    );
}
