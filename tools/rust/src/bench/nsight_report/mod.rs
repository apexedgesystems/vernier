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
