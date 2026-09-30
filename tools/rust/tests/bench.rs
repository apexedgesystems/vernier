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
/// `nsys stats` prints a summary line (`FAKE_NSYS_STATS=fail` fails instead).
struct RouteRig {
    dir: tempfile::TempDir,
    log: std::path::PathBuf,
    bench: std::path::PathBuf,
}

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
  echo "fake summary"
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
if [ -n "$out" ]; then
  if [ "$FAKE_WRITE" = empty ]; then : > "$out"; else echo "fake {name} output" > "$out"; fi
fi
"#,
            log = log.display()
        );
        write_executable(path, &script);
    };
    for program in programs {
        write(&tools.join(program), program);
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
                "compute-sanitizer wrap=compute-sanitizer --tool={tool} --log-file \
                 bench-out/fake_bench.compute-sanitizer/sanitizer.log {}",
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
