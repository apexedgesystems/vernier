//! Nsight report extraction: saved Nsight Systems and Nsight Compute reports
//! read into one CSV.
//!
//! An `.nsys-rep` is exported once, with `nsys export --type sqlite`, into a
//! private temporary directory, and the summaries in `NSYS_SUMMARIES` are read
//! from that export with `nsys stats --format csv`. Nothing is written beside
//! the report: an export `bench run` left there is neither used nor changed,
//! and `nsys stats` can refuse a report whose export it judges older than the
//! report. An `.ncu-rep` is read with `ncu --import <report> --csv
//! --print-summary per-kernel`. A directory stands for every report under it.
//!
//! The CSV is its own format, not a benchmark CSV: one row per row of each
//! nsys summary (per kernel name, per CUDA call, per kind of copy) and one row
//! per ncu launch shape, section and metric; `source`, `report`, `kernel`,
//! `instances`, `time_total_ns`, `time_avg_ns` and `time_pct` first, then every
//! other column the tools print, sorted by name. It has no `test` column, so
//! the benchmark CSV tools refuse it, and no row is attributed to a test.
//!
//! Each input that cannot be read is an error line naming it and its cause; a
//! summary without data is a warning. The rows of the inputs that were read
//! are written all the same. Running a tool (`exec`) is kept apart from
//! reading what it printed (`parse`); `extract` runs the workflow and writes
//! the CSV.

use std::ffi::OsStr;
use std::path::Path;

/* ----------------------------- Constants ----------------------------- */

/// The `bench` command that reads Nsight reports, and the prefix of its
/// message lines.
pub const COMMAND: &str = "nsight-parse";

/// The nsys summaries Vernier reads from a report: `bench run --profile
/// nsight` writes them as text beside the report it made, and
/// `bench nsight-parse` reads them into its CSV.
pub const NSYS_SUMMARIES: [&str; 4] = [
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_gpu_mem_time_sum",
];

/// How long one `nsys` or `ncu` command may run when no `--timeout` is given:
/// exporting a large report is slow.
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// The command's detailed help.
pub const LONG_HELP: &str = "\
Read saved Nsight Systems (.nsys-rep) and Nsight Compute (.ncu-rep) reports
into one CSV. A directory stands for every report under it.

An .nsys-rep is exported once, into a private temporary directory, and the
summaries cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_mem_size_sum and
cuda_gpu_mem_time_sum are read from that export with `nsys stats --format csv`.
Nothing is written beside a report. An .ncu-rep is read with
`ncu --import <report> --csv --print-summary per-kernel`.

The CSV is not a benchmark CSV: one row per row of each nsys summary and per ncu
launch shape, section and metric, with source, report, kernel, instances,
time_total_ns, time_avg_ns and time_pct first, then the tools' own columns.

Exit status: 0 when every requested input was read; 1 when any was not (a tool
failed, is missing, ran past --timeout or left a process holding its output, an
input is not a report or is empty, a directory holds none), with each failure
named on stderr and the rows that were read written all the same. A summary
with no data is only a warning.";

/* ----------------------------- Modules ----------------------------- */

mod exec;
mod extract;
mod parse;

/* ----------------------------- Re-exports ----------------------------- */

pub use exec::Interrupt;
pub use extract::{run, RunEnd};

/* ----------------------------- API ----------------------------- */

/// The arguments that make `ncu` print @p report's per-kernel summary as
/// CSV: how this command reads an `.ncu-rep`, and how `bench run` reads back
/// the report of an Nsight Compute run that names metrics.
pub(crate) fn ncu_summary_args(report: &Path) -> [&OsStr; 5] {
    [
        OsStr::new("--import"),
        report.as_os_str(),
        OsStr::new("--csv"),
        OsStr::new("--print-summary"),
        OsStr::new("per-kernel"),
    ]
}

/// Of the metric names @p asked, those the per-kernel summary @p summary
/// holds no row for, in the order asked, each once. @p summary and
/// @p stderr are what ncu printed for `ncu_summary_args`. A row's
/// `Metric Name` is the metric's own name when ncu's command line named it,
/// as `bench run` names a replay's metrics and those of `--metrics`, and a
/// summary with no row holds none of them. Refused, in words, when ncu
/// printed no summary or one that cannot be read.
pub(crate) fn metrics_not_in_summary<'a>(
    summary: &str,
    stderr: &str,
    asked: &'a [String],
) -> Result<Vec<&'a str>, String> {
    let held = parse::ncu_metric_names(summary)
        .map_err(|malformed| format!("its summary cannot be read: {malformed}"))?
        .ok_or_else(|| {
            format!(
                "ncu printed no per-kernel summary ({})",
                parse::last_line(&[stderr, summary])
            )
        })?;
    let mut missing: Vec<&str> = Vec::new();
    for name in asked {
        if !held.contains(name) && !missing.contains(&name.as_str()) {
            missing.push(name);
        }
    }
    Ok(missing)
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nsight");

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    /// @test Of the names asked for, those the summary holds no row for, in
    /// the order asked and each once: on the rig, the DRAM throughput of a
    /// --metrics list; a summary with no row holds none of them; no summary,
    /// or one that cannot be read, is refused, saying why.
    #[test]
    fn metrics_not_in_a_summary() {
        let rig = std::fs::read_to_string(format!("{FIXTURES}/ncu_import_one_metric_missing.out"))
            .expect("fixture");
        let dram = "dram__throughput.avg.pct_of_peak_sustained_elapsed";
        let asked = names(&["sm__throughput.avg.pct_of_peak_sustained_elapsed", dram]);
        assert_eq!(metrics_not_in_summary(&rig, "", &asked), Ok(vec![dram]));
        let summary = "\"Kernel Name\",\"Metric Name\",\"Average\"\n\"k\",\"b.sum\",\"1\"\n";
        let asked = names(&["a.sum", "b.sum", "c.sum", "a.sum"]);
        assert_eq!(
            metrics_not_in_summary(summary, "", &asked),
            Ok(vec!["a.sum", "c.sum"])
        );
        let header = summary.lines().next().expect("a header");
        assert_eq!(
            metrics_not_in_summary(header, "", &asked),
            Ok(vec!["a.sum", "b.sum", "c.sum"])
        );
        assert_eq!(
            metrics_not_in_summary("", "", &asked),
            Err("ncu printed no per-kernel summary (no output)".to_string())
        );
        assert_eq!(
            metrics_not_in_summary("", "==WARNING== a warning\n", &asked),
            Err("ncu printed no per-kernel summary (==WARNING== a warning)".to_string())
        );
        assert_eq!(
            metrics_not_in_summary("A,Metric Name,B\nx,y,z,w\n", "", &asked),
            Err(
                "its summary cannot be read: line 2 of its output has 4 fields for 3 columns"
                    .to_string()
            )
        );
    }
}
