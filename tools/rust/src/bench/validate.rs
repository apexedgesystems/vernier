//! `bench validate`: an advisory inventory of the host's profiling tools.
//!
//! Without a binary it reports facts and makes no readiness claim: which
//! profiling tools PATH finds, each with its version where `--version` gives
//! one, the msr device RAPL reads, ASLR, the FlameGraph scripts and
//! `kernel.perf_event_paranoid`. Whether a profiler can run here, and in which
//! modes, is what `bench doctor <binary>` checks: each backend decides it in
//! the benchmark binary. With a binary, the tool rows give way to the
//! binary's own default-mode doctor rows at advisory severity: `ok` stays OK,
//! `warn` stays WARN, and `fail` becomes WARN "not usable here: <message>",
//! the doctor's remedy kept. No row fails the command (exit 0); a lane gates
//! on `bench doctor <binary> --require <backends>`. A binary that is missing,
//! does not start or prints no usable doctor document is an error (exit 1).

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::{lookup_in_path, CheckStatus, Error, InPath};

/* ----------------------------- Row ----------------------------- */

/// One row of `bench validate`: a fact about the host, or one of a binary's
/// doctor rows at advisory severity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub label: String,
    /// OK or WARN: validate reports no FAIL row.
    pub status: CheckStatus,
    pub detail: String,
    /// The doctor's remedy on a binary's row; empty on a host fact.
    pub hint: String,
}

impl Row {
    fn fact(label: &str, status: CheckStatus, detail: impl Into<String>) -> Self {
        Row {
            label: label.to_string(),
            status,
            detail: detail.into(),
            hint: String::new(),
        }
    }
}

/* ----------------------------- API ----------------------------- */

/// What a found tool's row leaves to the doctor.
const NOT_CHECKED: &str = "access and modes: bench doctor <binary>";

/// The rows without a binary: tool-presence facts and host facts.
pub fn run_checks() -> Vec<Row> {
    vec![
        check_perf(),
        check_perf_paranoid(),
        check_gperftools(),
        tool_row(
            "valgrind (callgrind/massif/memcheck/helgrind)",
            "valgrind",
            "install valgrind for callgrind, massif, memcheck and helgrind",
        ),
        tool_row(
            "heaptrack",
            "heaptrack",
            "install heaptrack for low-overhead heap profiling",
        ),
        tool_row(
            "jemalloc",
            "jeprof",
            "install jemalloc (built with prof) for allocation sampling",
        ),
        check_rapl(),
        check_aslr(),
        check_flamegraph(),
        tool_row(
            "bpftrace",
            "bpftrace",
            "install bpftrace for kernel-level tracing",
        ),
        tool_row(
            "Nsight Systems",
            "nsys",
            "install the CUDA toolkit for GPU timelines",
        ),
        tool_row(
            "Nsight Compute",
            "ncu",
            "install the CUDA toolkit for GPU kernel metrics",
        ),
        tool_row(
            "compute-sanitizer",
            "compute-sanitizer",
            "it ships with the CUDA toolkit, for GPU memcheck and racecheck",
        ),
        tool_row("rocprof", "rocprof", "install ROCm for AMD GPU profiling"),
    ]
}

/// The rows for @p binary: the host facts, then the binary's default-mode
/// doctor rows at advisory severity (its `--profile-check-json` document).
pub fn binary_rows(binary: &Path) -> Result<Vec<Row>, Error> {
    let doc = super::workflow::read_doctor_document(binary, &[], &[])?;
    let backends = advisory_rows(&doc.value).map_err(|why| {
        Error::Parse(format!(
            "{} --profile-check-json printed no usable doctor document: {why}",
            binary.display()
        ))
    })?;
    let mut rows = vec![check_aslr(), check_flamegraph(), check_perf_paranoid()];
    rows.extend(backends);
    Ok(rows)
}

/// A doctor document's backend rows at advisory severity: `ok` stays OK,
/// `warn` stays WARN, `fail` becomes WARN "not usable here: <message>"; the
/// hint is kept. The reason, when the document has no backend rows or a row
/// has no name or another status.
pub fn advisory_rows(doc: &serde_json::Value) -> Result<Vec<Row>, String> {
    let rows = doc["backends"]
        .as_array()
        .ok_or_else(|| "it has no \"backends\" array".to_string())?;
    rows.iter().map(advisory_row).collect()
}

fn advisory_row(row: &serde_json::Value) -> Result<Row, String> {
    let name = row["name"]
        .as_str()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("a backend row has no name: {row}"))?;
    let message = row["message"].as_str().unwrap_or_default();
    let (status, detail) = match row["status"].as_str() {
        Some("ok") => (CheckStatus::Ok, message.to_string()),
        Some("warn") => (CheckStatus::Warn, message.to_string()),
        Some("fail") => (CheckStatus::Warn, format!("not usable here: {message}")),
        _ => {
            return Err(format!(
                "backend row '{name}' has status {}, not ok, warn or fail",
                row["status"]
            ))
        }
    };
    Ok(Row {
        label: name.to_string(),
        status,
        detail,
        hint: row["hint"].as_str().unwrap_or_default().to_string(),
    })
}

/// Print the rows under a header that says what they are and what they
/// leave out: without a binary, presence only; with one, the binary's rows
/// at advisory severity. A row's hint follows it on a line of its own.
pub fn print_results(rows: &[Row], binary: Option<&Path>) {
    println!();
    match binary {
        None => {
            println!("=== bench validate: profiling tools and settings on this host ===");
            println!();
            println!("  Presence only: whether each profiler can run here, and in which");
            println!("  modes, is what bench doctor <binary> checks.");
        }
        Some(bin) => {
            println!("=== bench validate: {} on this host ===", bin.display());
            println!();
            println!("  The binary's rows for each profiler's default mode, advisory: a");
            println!("  profiler it cannot use here shows as WARN \"not usable here\".");
        }
    }
    println!();

    let (mut ok, mut warn, mut fail) = (0, 0, 0);
    for r in rows {
        let (tag, color) = match r.status {
            CheckStatus::Ok => {
                ok += 1;
                ("OK", "\x1b[92m")
            }
            CheckStatus::Warn => {
                warn += 1;
                ("WARN", "\x1b[93m")
            }
            CheckStatus::Fail => {
                fail += 1;
                ("FAIL", "\x1b[91m")
            }
        };
        println!("  {color}[{tag:>4}]\x1b[0m {:<30} {}", r.label, r.detail);
        if !r.hint.is_empty() {
            println!("  {:<37} {}", "", r.hint);
        }
    }

    println!();
    println!("  ---");
    if fail > 0 {
        println!("  {ok} OK, {warn} WARN, {fail} FAIL");
    } else {
        println!("  {ok} OK, {warn} WARN");
    }
    println!();
    println!("  Advisory: this command does not fail on these rows. To fail a lane on");
    println!("  the profilers it needs: bench doctor <binary> --require <backends>.");
    println!();
}

/* ----------------------------- Tools ----------------------------- */

/// How long one tool's `--version` may take.
const VERSION_TIMEOUT: Duration = Duration::from_secs(5);

/// How `<tool> --version` ended.
enum VersionRun {
    /// It exited: whether with status 0, and its stdout followed by its
    /// stderr.
    Exited { success: bool, output: String },
    /// It did not start, or did not finish in time and was stopped.
    Failed,
}

/// Run `<path> --version`, bounded by `VERSION_TIMEOUT`.
fn run_version(path: &Path) -> VersionRun {
    let Ok(mut child) = Command::new(path)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return VersionRun::Failed;
    };
    // Each pipe is read on its own thread, so neither can fill. A process
    // the tool leaves behind can hold a pipe open, so the reads are awaited
    // with a bound instead of joined.
    let (tx, rx) = mpsc::channel();
    let pipes: [(usize, Box<dyn Read + Send>); 2] = [
        (0, Box::new(child.stdout.take().expect("piped stdout"))),
        (1, Box::new(child.stderr.take().expect("piped stderr"))),
    ];
    for (index, mut pipe) in pipes {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            let _ = tx.send((index, String::from_utf8_lossy(&bytes).into_owned()));
        });
    }
    drop(tx);
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < VERSION_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(10));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return VersionRun::Failed;
            }
        }
    };
    let mut texts = [String::new(), String::new()];
    for _ in 0..2 {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok((index, text)) => texts[index] = text,
            Err(_) => break,
        }
    }
    VersionRun::Exited {
        success: status.success(),
        output: texts.concat(),
    }
}

/// The first line of @p output that holds a version number (a digit, a dot
/// and a digit), trimmed, at most 80 characters.
fn version_line(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|line| {
            line.as_bytes()
                .windows(3)
                .any(|w| w[0].is_ascii_digit() && w[1] == b'.' && w[2].is_ascii_digit())
        })
        .map(|line| line.chars().take(80).collect())
}

/// The version `<path> --version` reports, when it exits 0 and prints one.
fn version_of(path: &Path) -> Option<String> {
    match run_version(path) {
        VersionRun::Exited {
            success: true,
            output,
        } => version_line(&output),
        _ => None,
    }
}

/// "installed at <path> (<version>)", and what the row does not check.
fn installed(path: &Path, version: Option<&str>) -> String {
    match version {
        Some(v) => format!("installed at {} ({v}); {NOT_CHECKED}", path.display()),
        None => format!("installed at {}; {NOT_CHECKED}", path.display()),
    }
}

/// A PATH entry named like @p program that is not an executable file.
fn not_executable(path: &Path, program: &str) -> String {
    format!(
        "{} is not an executable file; fix its permissions, or put a working {program} \
         first on PATH",
        path.display()
    )
}

/// A tool-presence fact: OK with the executable PATH finds and its version,
/// where `--version` gives one; WARN when PATH finds no executable file of
/// that name, with @p advice when it finds nothing at all.
fn tool_row(label: &str, program: &str, advice: &str) -> Row {
    match lookup_in_path(program) {
        InPath::Executable(path) => Row::fact(
            label,
            CheckStatus::Ok,
            installed(&path, version_of(&path).as_deref()),
        ),
        InPath::NotExecutable(path) => {
            Row::fact(label, CheckStatus::Warn, not_executable(&path, program))
        }
        InPath::Absent => Row::fact(
            label,
            CheckStatus::Warn,
            format!("{program} not found on PATH; {advice}"),
        ),
    }
}

/* ----------------------------- Individual Checks ----------------------------- */

fn check_perf() -> Row {
    const LABEL: &str = "perf";
    match lookup_in_path("perf") {
        // Presence is not enough: distro perf is a kernel-version shim that
        // can sit on PATH yet refuse to run when the installed linux-tools
        // does not match the running kernel (the classic in-container
        // failure).
        InPath::Executable(path) => match run_version(&path) {
            VersionRun::Exited {
                success: true,
                output,
            } => Row::fact(
                LABEL,
                CheckStatus::Ok,
                installed(&path, version_line(&output).as_deref()),
            ),
            _ => Row::fact(
                LABEL,
                CheckStatus::Warn,
                format!(
                    "{} does not run (perf --version fails); install linux-tools for the \
                     running kernel (linux-tools-$(uname -r))",
                    path.display()
                ),
            ),
        },
        InPath::NotExecutable(path) => {
            Row::fact(LABEL, CheckStatus::Warn, not_executable(&path, "perf"))
        }
        InPath::Absent => Row::fact(
            LABEL,
            CheckStatus::Warn,
            "perf not found on PATH; install linux-tools-common",
        ),
    }
}

/// The value of `kernel.perf_event_paranoid` and what the kernel lets a user
/// without CAP_PERFMON do at it: a fact; what perf can do is the doctor's.
fn check_perf_paranoid() -> Row {
    const LABEL: &str = "perf_event_paranoid";
    let Ok(text) = std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid") else {
        return Row::fact(
            LABEL,
            CheckStatus::Warn,
            "cannot read /proc/sys/kernel/perf_event_paranoid",
        );
    };
    let Ok(value) = text.trim().parse::<i32>() else {
        return Row::fact(
            LABEL,
            CheckStatus::Warn,
            format!(
                "/proc/sys/kernel/perf_event_paranoid holds '{}', not a number",
                text.trim()
            ),
        );
    };
    let (status, meaning) = paranoid_meaning(value);
    Row::fact(
        LABEL,
        status,
        format!("value={value}: {meaning}; what perf can do here: bench doctor <binary>"),
    )
}

/// What the kernel lets a user without CAP_PERFMON do at @p value, with the
/// row's status: OK up to 1, WARN above (kernel events need privileges).
fn paranoid_meaning(value: i32) -> (CheckStatus, &'static str) {
    match value {
        i32::MIN..=-1 => (CheckStatus::Ok, "(almost) all events are open to all users"),
        0 | 1 => (
            CheckStatus::Ok,
            "unprivileged users may profile their own processes, kernel included",
        ),
        2 => (
            CheckStatus::Warn,
            "the kernel's default: unprivileged users may count only user-space events",
        ),
        _ => (
            CheckStatus::Warn,
            "above 2, which upstream kernels treat as 2; Debian's and Ubuntu's kernels \
             refuse perf to unprivileged users at such values",
        ),
    }
}

/// The analyzer `--profile gperf --profile-analyze` runs: the first of
/// google-pprof and pprof that PATH finds executable, as the benchmark
/// chooses it. Collection needs no analyzer, so a missing one is a warning.
fn check_gperftools() -> Row {
    const LABEL: &str = "gperftools";
    let mut skipped = None;
    for program in ["google-pprof", "pprof"] {
        match lookup_in_path(program) {
            InPath::Executable(path) => {
                let analyzer = match version_of(&path) {
                    Some(v) => format!("{} ({v})", path.display()),
                    None => path.display().to_string(),
                };
                return Row::fact(
                    LABEL,
                    CheckStatus::Ok,
                    format!(
                        "--profile gperf --profile-analyze runs {analyzer}; collection needs \
                         libprofiler in the binary: bench doctor <binary>"
                    ),
                );
            }
            InPath::NotExecutable(path) if skipped.is_none() => skipped = Some(path),
            _ => {}
        }
    }
    let detail = match skipped {
        Some(path) => format!(
            "no executable google-pprof or pprof on PATH ({} is not an executable file); \
             only --profile-analyze needs one",
            path.display()
        ),
        None => "neither google-pprof nor pprof is on PATH; only --profile-analyze needs one: \
                 install google-pprof (gperftools) or Go's pprof"
            .to_string(),
    };
    Row::fact(LABEL, CheckStatus::Warn, detail)
}

/// The msr device RAPL reads: its presence is a fact, not energy access.
fn check_rapl() -> Row {
    if Path::new("/dev/cpu/0/msr").exists() {
        Row::fact(
            "rapl",
            CheckStatus::Ok,
            "/dev/cpu/0/msr present; energy access (an Intel CPU, root or CAP_SYS_RAWIO): \
             bench doctor <binary>",
        )
    } else {
        Row::fact(
            "rapl",
            CheckStatus::Warn,
            "/dev/cpu/0/msr absent; on an Intel host 'sudo modprobe msr' creates it",
        )
    }
}

fn check_aslr() -> Row {
    const LABEL: &str = "ASLR";
    match std::fs::read_to_string("/proc/sys/kernel/randomize_va_space") {
        Ok(contents) => {
            let val: i32 = contents.trim().parse().unwrap_or(-1);
            if val == 0 {
                Row::fact(LABEL, CheckStatus::Ok, "disabled (randomize_va_space=0)")
            } else {
                Row::fact(
                    LABEL,
                    CheckStatus::Warn,
                    format!(
                        "enabled (value={val}); use 'setarch $(uname -m) -R' for consistent \
                         profiles"
                    ),
                )
            }
        }
        Err(_) => Row::fact(
            LABEL,
            CheckStatus::Warn,
            "cannot read /proc/sys/kernel/randomize_va_space",
        ),
    }
}

fn check_flamegraph() -> Row {
    const LABEL: &str = "FlameGraph";
    // $FLAMEGRAPH_DIR first, then the common clones.
    let mut dirs = Vec::new();
    if let Ok(val) = std::env::var("FLAMEGRAPH_DIR") {
        dirs.push(std::path::PathBuf::from(val));
    }
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::PathBuf::from(&home).join("FlameGraph"));
    }
    dirs.push(std::path::PathBuf::from("/usr/local/FlameGraph"));
    dirs.push(std::path::PathBuf::from("/opt/FlameGraph"));

    for dir in &dirs {
        if dir.join("flamegraph.pl").is_file() {
            return Row::fact(
                LABEL,
                CheckStatus::Ok,
                format!("found at {}", dir.display()),
            );
        }
    }
    if super::find_in_path("flamegraph.pl").is_some() {
        return Row::fact(LABEL, CheckStatus::Ok, "flamegraph.pl found in PATH");
    }
    Row::fact(
        LABEL,
        CheckStatus::Warn,
        "not found; set $FLAMEGRAPH_DIR or clone to ~/FlameGraph",
    )
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test Without a binary validate reports every tool row, the separate
    /// Nsight Compute row among them, and no FAIL row whatever this host has.
    #[test]
    fn inventory_has_no_fail_row() {
        let rows = run_checks();
        assert_eq!(rows.len(), 14);
        for r in &rows {
            assert!(!r.label.is_empty());
            assert!(!r.detail.is_empty(), "{}", r.label);
            assert_ne!(r.status, CheckStatus::Fail, "{}: {}", r.label, r.detail);
        }
        assert!(rows.iter().any(|r| r.label == "Nsight Systems"));
        assert!(rows.iter().any(|r| r.label == "Nsight Compute"));
    }

    /// @test A doctor row keeps its words at advisory severity: ok stays
    /// OK, warn stays WARN, fail becomes WARN "not usable here:", and the
    /// hint is kept.
    #[test]
    fn advisory_rows_map_fail_to_not_usable_here() {
        let doc = serde_json::json!({"backends": [
            {"name": "perf", "status": "ok", "message": "perf stat counted", "hint": ""},
            {"name": "massif", "status": "warn", "message": "unchecked", "hint": "look"},
            {"name": "offcpu", "status": "fail", "message": "missing: bpftrace", "hint": "apt install bpftrace"},
        ]});
        let rows = advisory_rows(&doc).expect("well formed");
        let row = |label: &str, status, detail: &str, hint: &str| Row {
            label: label.to_string(),
            status,
            detail: detail.to_string(),
            hint: hint.to_string(),
        };
        assert_eq!(
            rows,
            [
                row("perf", CheckStatus::Ok, "perf stat counted", ""),
                row("massif", CheckStatus::Warn, "unchecked", "look"),
                row(
                    "offcpu",
                    CheckStatus::Warn,
                    "not usable here: missing: bpftrace",
                    "apt install bpftrace"
                ),
            ]
        );
    }

    /// @test A document without backend rows, a row without a name and a
    /// row with another status are malformed, each with its reason.
    #[test]
    fn advisory_rows_refuse_a_malformed_document() {
        for (doc, why) in [
            (serde_json::json!({"binary": {}}), "no \"backends\" array"),
            (
                serde_json::json!({"backends": "perf"}),
                "no \"backends\" array",
            ),
            (
                serde_json::json!({"backends": [{"status": "ok"}]}),
                "has no name",
            ),
            (
                serde_json::json!({"backends": [{"name": "perf", "status": "bogus"}]}),
                "backend row 'perf' has status \"bogus\"",
            ),
        ] {
            let err = advisory_rows(&doc).expect_err("malformed");
            assert!(err.contains(why), "{doc}: {err}");
        }
    }

    /// @test The version is the first line with a version number, from the
    /// output the real tools print.
    #[test]
    fn version_line_finds_the_version() {
        assert_eq!(
            version_line("perf version 6.8.12\n").as_deref(),
            Some("perf version 6.8.12")
        );
        assert_eq!(
            version_line("NVIDIA (R) Nsight Compute Command Line Profiler\nCopyright (c) 2018-2024 NVIDIA Corporation\nVersion 2024.3.2.0 (build 34861637) (public-release)\n")
                .as_deref(),
            Some("Version 2024.3.2.0 (build 34861637) (public-release)")
        );
        assert_eq!(
            version_line("pprof (part of gperftools 2.0)\n\nCopyright 1998-2007 Google Inc.\n")
                .as_deref(),
            Some("pprof (part of gperftools 2.0)")
        );
        assert_eq!(version_line("usage: tool [options]\n"), None);
        assert_eq!(version_line(&"9.9 ".repeat(40)).map(|v| v.len()), Some(80));
    }

    /// @test perf_event_paranoid: OK up to 1, WARN above, each value named
    /// by what the kernel allows at it.
    #[test]
    fn paranoid_meaning_by_value() {
        for (value, status, words) in [
            (-1, CheckStatus::Ok, "all events"),
            (1, CheckStatus::Ok, "kernel included"),
            (2, CheckStatus::Warn, "only user-space events"),
            (4, CheckStatus::Warn, "refuse perf to unprivileged users"),
        ] {
            let (got, meaning) = paranoid_meaning(value);
            assert_eq!(got, status, "{value}");
            assert!(meaning.contains(words), "{value}: {meaning}");
        }
    }
}
