//! Benchmark analysis: CSV loading, statistics, comparison, and reporting.
//!
//! This module provides the non-plotting analysis functionality for the Vernier
//! benchmarking framework. Subcommands: summary, compare, validate, run,
//! flamegraph.

use std::{fmt, path::PathBuf};

/* ----------------------------- Error ----------------------------- */

/// Unified error type for the bench module.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Csv(csv::Error),
    Parse(String),
    InvalidArgs(String),
    ToolNotFound(String),
    Regression(usize),
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
            Error::Regression(n) => write!(f, "{n} regression(s) detected"),
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

/// Search for an executable in PATH.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(name);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

/* ----------------------------- Modules ----------------------------- */

pub mod compare;
pub mod csv_loader;
pub mod flamegraph;
pub mod report;
pub mod runner;
pub mod stats;
pub mod validate;

/* ----------------------------- Re-exports ----------------------------- */

pub use compare::{compare_runs, has_regressions, CompareResult};
pub use csv_loader::{load_csv, BenchRow};
pub use flamegraph::generate_flamegraph;
pub use report::{print_comparison_table, print_summary_table, to_json, to_markdown};
pub use runner::run_benchmark;
pub use stats::{mann_whitney_u, median, percentile};
pub use validate::run_checks;
