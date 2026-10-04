//! Benchmark analysis: CSV loading, statistics, comparison, and reporting.
//!
//! This module provides the non-plotting analysis functionality for the Vernier
//! benchmarking framework, backing the `bench` subcommands: summary, compare,
//! validate, run, doctor, profile-all, profile-summarize, init,
//! config-validate, gpu-env, gpu-topo, gpu-monitor, gpu-lock, flamegraph.

use std::{
    fmt,
    path::{Path, PathBuf},
};

/* ----------------------------- Error ----------------------------- */

/// How a benchmark process ended when it did not succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkExit {
    /// Exited with this nonzero status other than 4 (a test failed, a flag
    /// was refused, the watchdog fired, ...).
    Status(i32),
    /// Exited with status 4: the tests passed and the requested profile failed.
    ProfileFailed,
    /// Ended by this signal.
    Signal(i32),
}

impl fmt::Display for BenchmarkExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkExit::Status(code) => write!(f, "the benchmark exited with status {code}"),
            BenchmarkExit::ProfileFailed => write!(
                f,
                "the requested profile failed (the benchmark's report above says why); \
                 the benchmark exited with status 4"
            ),
            BenchmarkExit::Signal(sig) => write!(f, "the benchmark was ended by signal {sig}"),
        }
    }
}

/// A requested profile that failed after its benchmark ran: its tool failed
/// on its own (collection), its output is missing, incomplete or not this
/// run's (completion), or its analysis failed (analysis).
#[derive(Debug)]
pub struct ProfileFailure {
    /// The request as the command line states it, e.g. `--profile massif`.
    pub request: String,
    /// "collection", "completion" or "analysis".
    pub stage: &'static str,
    pub message: String,
}

impl fmt::Display for ProfileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed: {}: {}",
            self.request, self.stage, self.message
        )
    }
}

/// A compute-sanitizer run whose report shows what failed: errors the tool
/// found in the benchmark, or a benchmark that did not end normally under it.
/// A count comes only from the report's summary.
#[derive(Debug)]
pub enum SanitizerFailure {
    /// The report's summary counts errors in the benchmark.
    Findings {
        errors: u64,
        report: PathBuf,
        /// How the tool ended, e.g. "exited with status 5".
        status: String,
        /// The report says the benchmark itself returned an error; the tool
        /// returns its own status in place of the benchmark's.
        benchmark_failed: bool,
    },
    /// The report counts no errors and says the benchmark did not end normally.
    AbnormalEnd {
        line: String,
        report: PathBuf,
        status: String,
    },
}

impl fmt::Display for SanitizerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SanitizerFailure::Findings {
                errors,
                report,
                status,
                benchmark_failed,
            } => {
                write!(
                    f,
                    "compute-sanitizer reported {errors} error{} in the benchmark; the \
                     report is {} (the tool {status})",
                    if *errors == 1 { "" } else { "s" },
                    report.display()
                )?;
                if *benchmark_failed {
                    write!(
                        f,
                        "; the report also says the benchmark returned an error, whose \
                         status the tool does not pass on"
                    )?;
                }
                Ok(())
            }
            SanitizerFailure::AbnormalEnd {
                line,
                report,
                status,
            } => write!(
                f,
                "the benchmark did not end normally under compute-sanitizer, whose report \
                 says \"{line}\" (the tool {status}); the report is {}",
                report.display()
            ),
        }
    }
}

/// Unified error type for the bench module.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Csv(csv::Error),
    Parse(String),
    InvalidArgs(String),
    ToolNotFound(String),
    /// The benchmark ran and did not succeed; how it ended.
    Benchmark(BenchmarkExit),
    /// The benchmark succeeded and its requested profile did not.
    Profile(ProfileFailure),
    /// A compute-sanitizer run whose report shows errors in the benchmark, or
    /// a benchmark that did not end normally under the tool.
    Sanitizer(SanitizerFailure),
    /// `bench profile-all`: these of `total` profile runs failed.
    ProfileAll {
        failed: Vec<String>,
        total: usize,
    },
    /// Two runs cannot be compared; the cause names itself.
    Compare(compare::CompareError),
    /// `bench compare --fail-on-regression` found a labelled regression or a
    /// baseline test the candidate does not run.
    Gate {
        regressions: usize,
        missing: usize,
    },
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Csv(e) => write!(f, "CSV error: {e}"),
            Error::Parse(s) => write!(f, "parse error: {s}"),
            Error::InvalidArgs(s) => write!(f, "invalid arguments: {s}"),
            Error::ToolNotFound(s) => write!(f, "tool not found: {s}"),
            Error::Benchmark(end) => write!(f, "{end}"),
            Error::Profile(failure) => write!(f, "{failure}"),
            Error::Sanitizer(failure) => write!(f, "{failure}"),
            Error::ProfileAll { failed, total } => write!(
                f,
                "{} of {total} profile runs failed: {}",
                failed.len(),
                failed.join(", ")
            ),
            Error::Compare(e) => write!(f, "{e}"),
            Error::Gate {
                regressions,
                missing,
            } => write!(
                f,
                "--fail-on-regression: {regressions} regression(s), \
                 {missing} baseline test(s) missing from the candidate"
            ),
        }
    }
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<csv::Error> for Error {
    #[inline]
    fn from(e: csv::Error) -> Self {
        Error::Csv(e)
    }
}

impl From<compare::CompareError> for Error {
    #[inline]
    fn from(e: compare::CompareError) -> Self {
        Error::Compare(e)
    }
}

/* ----------------------------- Classification ----------------------------- */

/// Regression/improvement classification for a single test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    Regression,
    Improvement,
    Neutral,
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Classification::Regression => "REGRESSION",
            Classification::Improvement => "IMPROVEMENT",
            Classification::Neutral => "neutral",
        })
    }
}

/* ----------------------------- Sort Column ----------------------------- */

/// Column to sort summary output by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortColumn {
    #[default]
    Name,
    Median,
    Cv,
    Throughput,
}

impl std::str::FromStr for SortColumn {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "name" => Ok(SortColumn::Name),
            "median" => Ok(SortColumn::Median),
            "cv" => Ok(SortColumn::Cv),
            "throughput" => Ok(SortColumn::Throughput),
            other => Err(Error::InvalidArgs(format!(
                "unknown sort column '{other}'. Expected: name, median, cv, throughput"
            ))),
        }
    }
}

/* ----------------------------- Validate Result ----------------------------- */

/// Result of a single environment validation check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub label: String,
    pub status: CheckStatus,
    pub detail: String,
}

/// Status of a validation check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

/* ----------------------------- Tool Search ----------------------------- */

/// Where PATH resolves a program name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InPath {
    /// The first regular file of that name with an execute bit set.
    Executable(PathBuf),
    /// No executable one; the first entry of that name, which is not an
    /// executable file (no execute bit, or not a regular file).
    NotExecutable(PathBuf),
    /// Nothing of that name.
    Absent,
}

/// Look `name` up on PATH by the rule the benchmark's own resolver uses: the
/// first regular file with an execute bit set wins; failing that, the first
/// entry of that name is reported as not executable. An unset PATH searches
/// `/bin:/usr/bin`, as `execvp` does; an empty entry is the current
/// directory.
pub fn lookup_in_path(name: &str) -> InPath {
    let search = std::env::var_os("PATH").unwrap_or_else(|| "/bin:/usr/bin".into());
    lookup_in(&search, name)
}

/// @p binary as the path that starts it: a bare file name becomes
/// `./<name>`, the working directory's file, which is where `bench doctor`,
/// `bench validate` and `bench run` find it; started as it is, a bare name
/// would be looked up on PATH (by the process start, taskset and every
/// wrapping tool alike).
pub(crate) fn launch_path(binary: &Path) -> PathBuf {
    if binary
        .parent()
        .is_some_and(|dir| dir.as_os_str().is_empty())
    {
        Path::new(".").join(binary)
    } else {
        binary.to_path_buf()
    }
}

/// Whether @p meta describes a file PATH would run: a regular file with an
/// execute bit set (on Unix; elsewhere any regular file).
pub(crate) fn is_executable(meta: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.is_file() && meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        meta.is_file()
    }
}

/// `lookup_in_path` over the PATH value `search`.
fn lookup_in(search: &std::ffi::OsStr, name: &str) -> InPath {
    let mut not_executable = None;
    for dir in std::env::split_paths(search) {
        let dir = if dir.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            dir
        };
        let candidate = dir.join(name);
        let Ok(meta) = std::fs::metadata(&candidate) else {
            continue;
        };
        if is_executable(&meta) {
            return InPath::Executable(candidate);
        }
        if not_executable.is_none() {
            not_executable = Some(candidate);
        }
    }
    not_executable.map_or(InPath::Absent, InPath::NotExecutable)
}

/// Search PATH for an executable named `name` (`lookup_in_path`): a file
/// without an execute bit is skipped, as the shell skips it.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    match lookup_in_path(name) {
        InPath::Executable(path) => Some(path),
        InPath::NotExecutable(_) | InPath::Absent => None,
    }
}

/* ----------------------------- Modules ----------------------------- */

pub mod compare;
pub mod config;
pub mod csv_loader;
pub mod flamegraph;
pub mod gpu_env;
pub mod gpu_lock;
pub mod gpu_monitor;
pub mod gpu_topo;
pub mod report;
pub mod runner;
pub mod stats;
pub mod validate;
pub mod workflow;

/* ----------------------------- Re-exports ----------------------------- */

pub use compare::{compare_runs, has_regressions, CompareError, CompareResult, Comparison};
pub use csv_loader::{load_csv, load_csv_strict, BenchRow, Need};
pub use flamegraph::generate_flamegraph;
pub use report::{print_comparison_table, print_summary_table, to_json, to_markdown};
pub use runner::run_benchmark;
pub use stats::{median, percentile};
pub use validate::run_checks;

/* ----------------------------- Tests ----------------------------- */

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn file(dir: &std::path::Path, name: &str, mode: u32) -> PathBuf {
        std::fs::create_dir_all(dir).expect("mkdir");
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).expect("chmod");
        path
    }

    fn search(dirs: &[&std::path::Path]) -> std::ffi::OsString {
        std::env::join_paths(dirs).expect("join")
    }

    /// @test The first executable file wins; a file without an execute bit
    /// before it is skipped, as the shell skips it.
    #[test]
    fn lookup_skips_a_file_without_an_execute_bit() {
        let root = tempfile::tempdir().expect("tempdir");
        let (a, b) = (root.path().join("a"), root.path().join("b"));
        file(&a, "tool", 0o644);
        let tool = file(&b, "tool", 0o755);
        assert_eq!(
            lookup_in(&search(&[&a, &b]), "tool"),
            InPath::Executable(tool)
        );
    }

    /// @test Without an executable one, the first entry of that name is
    /// reported as not executable: a file without an execute bit, or a
    /// directory.
    #[test]
    fn lookup_names_what_is_not_executable() {
        let root = tempfile::tempdir().expect("tempdir");
        let (a, b) = (root.path().join("a"), root.path().join("b"));
        let plain = file(&a, "tool", 0o644);
        file(&b, "tool", 0o600);
        assert_eq!(
            lookup_in(&search(&[&a, &b]), "tool"),
            InPath::NotExecutable(plain)
        );
        std::fs::create_dir_all(b.join("dir")).expect("mkdir");
        assert_eq!(
            lookup_in(&search(&[&b]), "dir"),
            InPath::NotExecutable(b.join("dir"))
        );
        assert_eq!(lookup_in(&search(&[&a, &b]), "other"), InPath::Absent);
    }
}
