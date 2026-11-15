//! bench: Benchmark analysis CLI tool.
//!
//! Single binary for benchmark analysis, comparison, validation, execution,
//! and flamegraph generation.
//!
//! Usage:
//!   bench summary <results.csv>                      # Pretty-print one CSV
//!   bench compare <baseline.csv> <candidate.csv>     # Colored median-change diff
//!   bench validate [<binary>]                        # Profiling tools here (advisory)
//!   bench gpu-env                                    # GPU environment readiness
//!   bench gpu-lock lock [--freq MHz] [-- cmd...]     # Lock GPU clocks for benchmarking
//!   bench gpu-lock reset                             # Reset GPU clocks to default
//!   bench gpu-monitor snapshot [-o file.json]        # Capture GPU state snapshot
//!   bench gpu-monitor diff <before> <after>          # Diff two GPU snapshots
//!   bench run <binary> [-- extra_args...]            # Execute benchmark binary
//!   bench flamegraph <perf.data>                     # Generate SVG flamegraph
//!
//! The comparison contract:
//!
//! `bench compare` labels every test both runs report by the percentage
//! change of its reported median against `--threshold` (default 5%). The two
//! CV values travel with each row as context. The command is advisory and
//! exits 0; `--fail-on-regression` is the gate and exits 1 when a test is
//! labelled REGRESSION or the candidate does not run a baseline test.
//! Candidate-only tests are reported as new and do not fail on their own.
//! Unusable input exits 1 with the cause on stderr and no comparison on
//! stdout, with or without the gate flag: an unusable `--threshold`, a
//! `wallMedian` or `wallCV` that is missing, empty or not a finite number, a
//! median of zero or less, a negative CV, a blank or duplicate test identity,
//! no test in both runs, or a percentage change too large to represent. Test
//! names are matched exactly as written.
//!
//! The summary contract:
//!
//! `bench summary`, and the summary `bench run --analyze` prints after a run,
//! show every row's wallMedian, wallCV and callsPerSecond, and its wallP10,
//! wallP90, stable, cvThreshold, cycles and repeats where the row gives them.
//! A row whose test name is empty or only whitespace, a required value that
//! is missing, or any shown value that is not a finite number of its kind
//! exits 1 with the cause on stderr and no summary output; `bench summary`
//! leaves stdout empty, while `run --analyze` preserves run output already
//! emitted. An empty timing cell in a row whose every timing cell is empty is
//! named as a profiling capture's: the benchmark leaves a row so under Nsight
//! Compute, which times nothing, so `run --analyze` is refused before a run
//! under ncu starts.

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use clap::{Parser, Subcommand};
use vernier_rust_tools::bench::{self, Error, SortColumn};
use vernier_rust_tools::lossy_eprintln;

/* ----------------------------- CLI ----------------------------- */

#[derive(Parser, Debug)]
#[command(
    name = "bench",
    about = "Benchmark analysis, comparison, and execution",
    version
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pretty-print a benchmark results CSV
    Summary {
        /// Path to results CSV file
        csv: PathBuf,

        /// Sort by: name, median, cv, throughput
        #[arg(long, default_value = "name")]
        sort: String,

        /// Output as JSON instead of table
        #[arg(long)]
        json: bool,
    },

    /// Compare two benchmark runs by the change in each reported median
    Compare {
        /// Baseline results CSV
        baseline: PathBuf,

        /// Candidate results CSV
        candidate: PathBuf,

        /// Median change, in percent, beyond which a test is labelled
        #[arg(long, default_value = "5.0")]
        threshold: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Output as markdown table
        #[arg(long)]
        markdown: bool,

        /// Exit 1 on a labelled regression or a missing baseline test (CI gate)
        #[arg(long)]
        fail_on_regression: bool,
    },

    /// Report the profiling tools and settings this host has, or a binary's
    /// profilers at advisory severity; never fails on them (bench doctor
    /// --require is the gate)
    Validate {
        /// A benchmark binary whose default-mode doctor rows to report
        binary: Option<PathBuf>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute a benchmark binary with optional CPU pinning and profiling
    Run {
        /// Benchmark binary: a full path, or a short name that auto-resolves
        /// under build/*/bin/{ptests,tests,examples}, or where .bench.yaml's
        /// bin_roots and bin_subdirs (or VERNIER_BENCH_BIN_ROOTS and
        /// VERNIER_BENCH_BIN_SUBDIRS) say, the first root listed with a match
        /// deciding. The resolver tries the name as given, then with `_PTEST`
        /// / `_pTest` suffixes, then with a `BenchDemo_` prefix.
        binary: String,

        /// CSV output path (passed to binary as --csv)
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Quick mode (passed to binary as --quick); .bench.yaml's cycles,
        /// repeats and target_time are then not used
        #[arg(long)]
        quick: bool,

        /// Number of cycles (passed to binary as --cycles)
        #[arg(long)]
        cycles: Option<u32>,
        /// Auto-size cycles so one repeat spans ~this wall time (e.g. 500us, 100ms, 2s; bare number = ms). Passed to the binary as --target-time
        #[arg(long)]
        target_time: Option<String>,

        /// Number of repeats (passed to binary as --repeats)
        #[arg(long)]
        repeats: Option<u32>,

        /// Profiling tool, passed to the binary as --profile by its canonical
        /// name (nsys is nsight)
        #[arg(long)]
        profile: Option<String>,

        /// The profile's mode, passed to the binary as --profile-args. For a
        /// profile bench run wraps it also selects the wrap (massif: pages or
        /// stacks; memcheck: leak-full, track-origins; helgrind: drd;
        /// compute-sanitizer: memcheck, racecheck, synccheck or initcheck;
        /// nsight: compute; ncu, and nsight's compute route: replay, or
        /// --set NAME, --section NAME and --metrics NAMES), and any other word
        /// is refused. May start with '-', e.g. "-e cycles" for perf
        #[arg(long, allow_hyphen_values = true)]
        profile_args: Option<String>,

        /// gperf's CPU sampling rate, a whole number of Hz from 1 to 4000: the
        /// binary is started with CPUPROFILE_FREQUENCY set to it, which
        /// gperftools reads only as the process starts, and is passed
        /// --profile-frequency; refused with any other profile
        #[arg(long, allow_hyphen_values = true)]
        profile_frequency: Option<String>,

        /// Per-test profiler watchdog timeout in seconds, passed to the binary
        /// as --profile-test-timeout: 0 turns the watchdog off; omitted, the
        /// binary uses 300 under --profile
        #[arg(long)]
        profile_test_timeout: Option<u32>,

        /// Where wrap-externally backends (massif, memcheck, callgrind,
        /// heaptrack, compute-sanitizer) write their artifacts, from the
        /// working directory. Defaults to .bench.yaml's profile_output_dir
        /// (from the file's folder), else `bench-out/`. Forwarded to the
        /// binary as --profile-output-dir so the C++ harness's per-test
        /// subdirs land in the same root.
        #[arg(long)]
        profile_output_dir: Option<PathBuf>,

        /// Pin to CPUs (e.g., "0,1,3")
        #[arg(long)]
        taskset: Option<String>,

        /// Run the requested profile's analysis (passed to the binary as
        /// --profile-analyze; for callgrind, bench run annotates the profile
        /// after valgrind has written it, and a failed annotation fails the run)
        #[arg(long)]
        profile_analyze: bool,

        /// Print the summary of the run's CSV after it ends (requires --csv;
        /// refused for a run under Nsight Compute, which has no timing)
        #[arg(long)]
        analyze: bool,

        /// Extra arguments passed to the benchmark binary
        #[arg(last = true)]
        extra_args: Vec<String>,
    },

    /// Lock or reset GPU clocks for reproducible benchmarks
    GpuLock {
        #[command(subcommand)]
        action: GpuLockAction,
    },

    /// Capture and diff GPU state snapshots
    GpuMonitor {
        #[command(subcommand)]
        action: GpuMonitorAction,
    },

    /// Check GPU environment readiness for benchmarking
    GpuEnv {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show GPU / CPU affinity topology and recommended CPU pinning
    GpuTopo {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Run --profile-check (binary readiness + per-backend env doctor)
    Doctor {
        /// Path to a benchmark binary (any vernier ptest works)
        binary: PathBuf,
        /// Emit one JSON document (binary checks + backend rows) instead of text
        #[arg(long)]
        json: bool,
        /// Comma-separated backends that must report OK (exit nonzero otherwise),
        /// e.g. --require offcpu,heaptrack. Implies parsing the JSON form. The
        /// backend of --profile is judged by that request's own row
        #[arg(long, value_delimiter = ',')]
        require: Vec<String>,
        /// A profile request to check, passed to the binary as bench run
        /// passes it (--profile by its canonical name; nsys is nsight)
        #[arg(long)]
        profile: Option<String>,
        /// The request's mode, passed to the binary as --profile-args
        #[arg(long, allow_hyphen_values = true)]
        profile_args: Option<String>,
        /// gperf's CPU sampling rate, as bench run applies it: the binary
        /// starts with CPUPROFILE_FREQUENCY set and is passed
        /// --profile-frequency
        #[arg(long, allow_hyphen_values = true)]
        profile_frequency: Option<String>,
        /// The request's analysis, passed to the binary as --profile-analyze
        #[arg(long)]
        profile_analyze: bool,
        /// Further arguments for the binary, after --, as bench run passes them
        #[arg(last = true)]
        extra_args: Vec<String>,
    },

    /// Run a benchmark binary under each profiler in sequence
    ProfileAll {
        /// Benchmark binary (full path or short name, resolved as bench run
        /// resolves it)
        binary: String,

        /// Profilers to run, comma-separated; defaults to gperf,perf,callgrind
        #[arg(long)]
        profilers: Option<String>,

        /// Artifact root; per-profiler subdirs land here. Default:
        /// .bench.yaml's profile_output_dir (from the file's folder), else
        /// bench-out/ (--profile-output-dir is accepted as an alias, matching
        /// `bench run`)
        #[arg(long, visible_alias = "profile-output-dir")]
        out: Option<PathBuf>,

        /// --gtest_filter to pass through; without it, .bench.yaml's
        /// gtest_filter is passed, as bench run passes it
        #[arg(long)]
        filter: Option<String>,

        /// --cycles override
        #[arg(long)]
        cycles: Option<u32>,

        /// --repeats override
        #[arg(long)]
        repeats: Option<u32>,

        /// --quick mode; .bench.yaml's cycles and repeats are then not used
        #[arg(long)]
        quick: bool,
    },

    /// Walk an artifact directory and tabulate per-tool output
    ProfileSummarize {
        /// Directory containing per-tool subdirectories (e.g. bench-out/)
        dir: PathBuf,
    },

    /// Read saved Nsight reports into one CSV (not a benchmark CSV)
    #[command(name = bench::nsight_report::COMMAND, long_about = bench::nsight_report::LONG_HELP)]
    NsightReport {
        /// .nsys-rep and .ncu-rep reports, or directories searched for both
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// The CSV to write; written even when no report could be read
        #[arg(long)]
        csv: PathBuf,

        /// Seconds one nsys or ncu command may run before it and every process
        /// in its process group are stopped
        #[arg(
            long,
            default_value_t = bench::nsight_report::DEFAULT_TIMEOUT_SECS,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        timeout: u64,
    },

    /// Scaffold a .bench.yaml at the project root with sensible defaults
    Init {
        /// Path to write (default: .bench.yaml in the current directory)
        #[arg(long, default_value = ".bench.yaml")]
        path: PathBuf,

        /// Overwrite an existing file
        #[arg(long)]
        force: bool,
    },

    /// Check a .bench.yaml: each problem run and profile-all stop on (then
    /// exit 1), and warnings (unknown keys, low repeats)
    ConfigValidate {
        /// Path to the config file (default: walk up from CWD for .bench.yaml)
        path: Option<PathBuf>,
    },

    /// Generate an SVG flamegraph from perf.data
    Flamegraph {
        /// Input perf.data file
        input: PathBuf,

        /// Output SVG path
        #[arg(long, default_value = "flamegraph.svg")]
        output: PathBuf,

        /// Baseline perf.data for differential flamegraph
        #[arg(long)]
        baseline: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum GpuLockAction {
    /// Lock GPU clocks to a fixed frequency (default: max clock)
    Lock {
        /// GPU device index
        #[arg(long, default_value = "0")]
        device: u32,

        /// Target frequency in MHz (default: max reported clock)
        #[arg(long)]
        freq: Option<u32>,

        /// Command to run while the clocks are locked; they are reset once it
        /// has ended, also when SIGINT, SIGTERM or SIGHUP interrupts the run.
        /// bench exits with the command's status (128 + N for its signal N),
        /// or 125 when the lock, the command's start or the reset fails
        #[arg(last = true)]
        command: Vec<String>,
    },

    /// Reset GPU clocks to driver-managed default
    Reset {
        /// GPU device index
        #[arg(long, default_value = "0")]
        device: u32,
    },
}

#[derive(Subcommand, Debug)]
enum GpuMonitorAction {
    /// Capture current GPU state as JSON
    Snapshot {
        /// Write snapshot to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare two GPU state snapshots and highlight changes
    Diff {
        /// Before-run snapshot JSON
        before: PathBuf,

        /// After-run snapshot JSON
        after: PathBuf,

        /// Output as JSON instead of table
        #[arg(long)]
        json: bool,
    },
}

/* ----------------------------- Main ----------------------------- */

fn main() -> ExitCode {
    let args = Args::parse();

    // Before any command starts a child: each stays waitable until the
    // command reaps it, whatever SIGCHLD action this process inherited.
    if let Err(e) = bench::runner::keep_children_waitable() {
        lossy_eprintln!("Error: SIGCHLD could not be given its default action: {e}");
        return ExitCode::FAILURE;
    }

    match run(args) {
        Ok(status) => ExitCode::from(status),
        Err(e) => {
            lossy_eprintln!("Error: {e}");
            ExitCode::from(e.exit_status())
        }
    }
}

/// The project's `.bench.yaml`, the first found from the working directory
/// up, and the defaults it holds (none without one). A file that cannot be
/// read or used is the error, before the command starts anything.
fn project_defaults() -> Result<(Option<PathBuf>, bench::config::Config), Error> {
    let cwd = std::env::current_dir().map_err(Error::Io)?;
    let file = bench::config::find(&cwd);
    let config = match file.as_deref() {
        Some(path) => bench::config::load(path)?,
        None => bench::config::Config::default(),
    };
    Ok((file, config))
}

/// The file's `profile_output_dir`, a relative one read from the folder of
/// @p file, the `.bench.yaml` @p file_cfg was read from.
fn file_output_dir(file: Option<&Path>, file_cfg: &bench::config::Config) -> Option<PathBuf> {
    let dir = file_cfg.profile_output_dir.as_deref()?;
    Some(bench::config::from_folder_of(file?, dir))
}

/// The run sizes a `.bench.yaml` gives: cycles, repeats and target_time.
#[derive(Default)]
struct Sizes {
    cycles: Option<u32>,
    repeats: Option<u32>,
    target_time: Option<String>,
}

/// The sizes @p file_cfg gives a run; none when the command line gave
/// `--quick` (@p quick), which takes precedence over the file's sizes, the
/// benchmark then sizing the run as `--quick` says (beside any size given on
/// the command line itself).
fn file_sizes(file_cfg: &bench::config::Config, quick: bool) -> Sizes {
    if quick {
        return Sizes::default();
    }
    Sizes {
        cycles: file_cfg.cycles,
        repeats: file_cfg.repeats,
        target_time: file_cfg.target_time.clone(),
    }
}

/// Run the command @p args name; the status bench exits with, 0 but for a
/// command gpu-lock wrapped, whose status is passed through.
fn run(args: Args) -> Result<u8, Error> {
    match args.command {
        Command::Summary { csv, sort, json } => {
            let sort_col: SortColumn = sort.parse()?;
            // A value the summary could not show truthfully is refused before
            // anything is printed.
            let rows = bench::load_csv_strict(&csv, &bench::report::SUMMARY_COLUMNS)?;

            if json {
                println!("{}", bench::report::summary_to_json(&rows));
            } else {
                bench::print_summary_table(&rows, sort_col);
            }
        }

        Command::Compare {
            baseline,
            candidate,
            threshold,
            json,
            markdown,
            fail_on_regression,
        } => {
            let columns = bench::compare::measured_columns();
            let base_rows = bench::load_csv_strict(&baseline, &columns)?;
            let cand_rows = bench::load_csv_strict(&candidate, &columns)?;
            // Unusable input is reported instead of a comparison, so nothing
            // reaches stdout that a gate could read as a pass.
            let comparison = bench::compare_runs(&base_rows, &cand_rows, threshold)?;

            if json {
                println!("{}", bench::to_json(&comparison));
            } else if markdown {
                print!("{}", bench::to_markdown(&comparison));
            } else {
                bench::print_comparison_table(&comparison);
            }

            if fail_on_regression {
                let regressions = comparison.regression_count();
                let missing = comparison.baseline_only.len();
                if regressions > 0 || missing > 0 {
                    return Err(Error::Gate {
                        regressions,
                        missing,
                    });
                }
            }
        }

        Command::Validate { binary, json } => {
            let rows = match binary.as_deref() {
                Some(binary) => bench::validate::binary_rows(binary)?,
                None => bench::validate::run_checks(),
            };

            if json {
                println!("{}", bench::report::validate_to_json(&rows));
            } else {
                bench::validate::print_results(&rows, binary.as_deref());
            }
        }

        Command::Run {
            binary,
            csv,
            quick,
            cycles,
            target_time,
            repeats,
            profile,
            profile_args,
            profile_frequency,
            profile_test_timeout,
            profile_output_dir,
            taskset,
            profile_analyze,
            analyze,
            extra_args,
        } => {
            // Pull defaults from .bench.yaml if one exists; unset CLI flags
            // win from the file. CLI takes precedence when both are set, and
            // a --quick given here over the file's sizes.
            let (file, file_cfg) = project_defaults()?;
            // Accept either a full path or a short name, looked for where
            // the environment, else the file, else the default says.
            let search = bench::workflow::BinSearch::new(&file_cfg, file.as_deref());
            let resolved = bench::workflow::resolve_binary(&binary, &search)?;
            let sizes = file_sizes(&file_cfg, quick);
            let cfg = bench::runner::RunConfig {
                binary: resolved,
                csv: csv.clone(),
                quick,
                cycles: cycles.or(sizes.cycles),
                target_time: target_time.or(sizes.target_time),
                repeats: repeats.or(sizes.repeats),
                profile,
                profile_args,
                profile_frequency,
                profile_test_timeout,
                profile_output_dir,
                // Taken when no folder is given, here or after `--`.
                default_profile_output_dir: file_output_dir(file.as_deref(), &file_cfg),
                profile_analyze,
                taskset,
                // Given unless a filter follows `--` (runner::configured_filter).
                gtest_filter: file_cfg.gtest_filter,
                extra_args,
            };

            // The summary is of the run's timing, which a run under Nsight
            // Compute does not have: refused before the run starts.
            if analyze {
                bench::runner::refuse_analysis_of_capture(&cfg)?;
            }
            let csv_path = bench::run_benchmark(&cfg)?;

            if analyze {
                if let Some(ref path) = csv_path {
                    let rows = bench::load_csv_strict(path, &bench::report::SUMMARY_COLUMNS)?;
                    println!();
                    println!("--- Post-run analysis ---");
                    bench::print_summary_table(&rows, SortColumn::Name);
                } else {
                    lossy_eprintln!("Warning: --analyze requires --csv to produce output");
                }
            }
        }

        Command::GpuLock { action } => match action {
            GpuLockAction::Lock {
                device,
                freq,
                command,
            } => {
                let lock_cfg = bench::gpu_lock::LockConfig {
                    device,
                    freq,
                    tool: bench::gpu_lock::Tool::default(),
                };

                if command.is_empty() {
                    // Standalone lock mode
                    bench::gpu_lock::lock_clocks(&lock_cfg)?;
                } else {
                    // Wrapper mode: lock -> run -> reset, then the command's
                    // status. Interrupted, it ends the process by the signal
                    // once the clocks are reset.
                    let wrap_cfg = bench::gpu_lock::WrapConfig {
                        lock: lock_cfg,
                        command,
                    };
                    return bench::gpu_lock::wrap_command(&wrap_cfg);
                }
            }
            GpuLockAction::Reset { device } => {
                bench::gpu_lock::reset_clocks(device, &bench::gpu_lock::Tool::default())?;
            }
        },

        Command::GpuMonitor { action } => match action {
            GpuMonitorAction::Snapshot { output } => {
                let snapshot = bench::gpu_monitor::take_snapshot()?;
                let json = serde_json::to_string_pretty(&snapshot)
                    .map_err(|e| Error::Parse(e.to_string()))?;

                if let Some(path) = output {
                    bench::gpu_monitor::save_snapshot(&snapshot, &path)?;
                    lossy_eprintln!("Snapshot saved to {}", path.display());
                } else {
                    println!("{json}");
                }
            }
            GpuMonitorAction::Diff {
                before,
                after,
                json,
            } => {
                let before_snap = bench::gpu_monitor::load_snapshot(&before)?;
                let after_snap = bench::gpu_monitor::load_snapshot(&after)?;
                let diffs = bench::gpu_monitor::diff_snapshots(&before_snap, &after_snap);

                if json {
                    let json_str = serde_json::to_string_pretty(&diffs)
                        .map_err(|e| Error::Parse(e.to_string()))?;
                    println!("{json_str}");
                } else {
                    bench::gpu_monitor::print_diff(&before_snap, &after_snap, &diffs);
                }
            }
        },

        Command::GpuEnv { json } => {
            let results = bench::gpu_env::run_checks();

            if json {
                println!("{}", bench::report::gpu_env_to_json(&results));
            } else {
                bench::gpu_env::print_results(&results);
            }
        }

        Command::GpuTopo { json } => {
            let report = bench::gpu_topo::discover();
            if json {
                bench::gpu_topo::print_results_json(&report);
            } else {
                bench::gpu_topo::print_results(&report);
            }
        }

        Command::Doctor {
            binary,
            json,
            require,
            profile,
            profile_args,
            profile_frequency,
            profile_analyze,
            extra_args,
        } => {
            let request = bench::runner::RunConfig {
                binary: binary.clone(),
                profile,
                profile_args,
                profile_frequency,
                profile_analyze,
                extra_args,
                ..Default::default()
            };
            let rc = bench::workflow::doctor(&binary, &request, json, &require)?;
            std::process::exit(rc);
        }

        Command::ProfileAll {
            binary,
            profilers,
            out,
            filter,
            cycles,
            repeats,
            quick,
        } => {
            // .bench.yaml fills in any unset flags (its sizes not beside a
            // --quick given here), and says where a short name is looked for,
            // as for bench run.
            let (file, file_cfg) = project_defaults()?;
            let search = bench::workflow::BinSearch::new(&file_cfg, file.as_deref());
            let resolved = bench::workflow::resolve_binary(&binary, &search)?;
            let sizes = file_sizes(&file_cfg, quick);
            let parsed_profilers = profilers
                .map(|s| {
                    s.split(',')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            bench::workflow::profile_all(&bench::workflow::ProfileAllConfig {
                binary: resolved,
                profilers: parsed_profilers,
                artifact_root: out.or_else(|| file_output_dir(file.as_deref(), &file_cfg)),
                gtest_filter: filter.or(file_cfg.gtest_filter.clone()),
                cycles: cycles.or(sizes.cycles),
                repeats: repeats.or(sizes.repeats),
                quick,
            })?;
        }

        Command::ProfileSummarize { dir } => {
            let report = bench::workflow::profile_summarize(&dir)?;
            bench::workflow::print_summary(&report);
        }

        Command::NsightReport {
            inputs,
            csv,
            timeout,
        } => match bench::nsight_report::run(&inputs, &csv, Duration::from_secs(timeout))? {
            bench::nsight_report::RunEnd::AllRead => {}
            // Each unread input is already named on stderr.
            bench::nsight_report::RunEnd::SomeUnread => std::process::exit(1),
            bench::nsight_report::RunEnd::Interrupted(signal) => signal.end_process(),
        },

        Command::Init { path, force } => {
            bench::config::write_template(&path, force)?;
            println!("[bench] wrote scaffold to {}", path.display());
        }

        Command::ConfigValidate { path } => {
            let resolved = match path {
                Some(p) => p,
                None => {
                    let cwd = std::env::current_dir().map_err(Error::Io)?;
                    bench::config::find(&cwd).ok_or_else(|| {
                        Error::InvalidArgs(
                            "no .bench.yaml found in this directory or any parent.".into(),
                        )
                    })?
                }
            };
            let report = bench::config::validate(&resolved)?;
            bench::config::print_validation(&report);
            // Ends as run and profile-all would on this file.
            report.verdict()?;
        }

        Command::Flamegraph {
            input,
            output,
            baseline,
        } => {
            let cfg = bench::flamegraph::FlameGraphConfig {
                input,
                output,
                baseline,
            };
            bench::generate_flamegraph(&cfg)?;
        }
    }

    Ok(0)
}
