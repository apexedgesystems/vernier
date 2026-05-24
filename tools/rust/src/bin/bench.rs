//! bench: Benchmark analysis CLI tool.
//!
//! Single binary for benchmark analysis, comparison, validation, execution,
//! and flamegraph generation.
//!
//! Usage:
//!   bench summary <results.csv>                      # Pretty-print one CSV
//!   bench compare <baseline.csv> <candidate.csv>     # Colored regression diff
//!   bench validate                                   # CPU environment readiness
//!   bench gpu-env                                    # GPU environment readiness
//!   bench gpu-lock lock [--freq MHz] [-- cmd...]     # Lock GPU clocks for benchmarking
//!   bench gpu-lock reset                             # Reset GPU clocks to default
//!   bench gpu-monitor snapshot [-o file.json]        # Capture GPU state snapshot
//!   bench gpu-monitor diff <before> <after>          # Diff two GPU snapshots
//!   bench run <binary> [-- extra_args...]            # Execute benchmark binary
//!   bench flamegraph <perf.data>                     # Generate SVG flamegraph

use std::{path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use vernier_rust_tools::bench::{self, Error, SortColumn};

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

    /// Compare two benchmark runs and detect regressions
    Compare {
        /// Baseline results CSV
        baseline: PathBuf,

        /// Candidate results CSV
        candidate: PathBuf,

        /// Regression threshold percentage
        #[arg(long, default_value = "5.0")]
        threshold: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Output as markdown table
        #[arg(long)]
        markdown: bool,

        /// Exit with code 1 if any regression detected (CI mode)
        #[arg(long)]
        fail_on_regression: bool,
    },

    /// Check environment readiness for profiling
    Validate {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute a benchmark binary with optional CPU pinning and profiling
    Run {
        /// Path to benchmark binary
        binary: PathBuf,

        /// CSV output path (passed to binary as --csv)
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Quick mode (passed to binary as --quick)
        #[arg(long)]
        quick: bool,

        /// Number of cycles (passed to binary as --cycles)
        #[arg(long)]
        cycles: Option<u32>,

        /// Number of repeats (passed to binary as --repeats)
        #[arg(long)]
        repeats: Option<u32>,

        /// Profiling tool (passed to binary as --profile)
        #[arg(long)]
        profile: Option<String>,

        /// Pin to CPUs (e.g., "0,1,3")
        #[arg(long)]
        taskset: Option<String>,

        /// Auto-run comparison after execution (requires --csv)
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

        /// Command to run while clocks are locked (clocks reset on exit)
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

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Error::Regression(n)) => {
            eprintln!("Error: {n} regression(s) detected");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), Error> {
    match args.command {
        Command::Summary { csv, sort, json } => {
            let sort_col: SortColumn = sort.parse()?;
            let rows = bench::load_csv(&csv)?;

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
            let base_rows = bench::load_csv(&baseline)?;
            let cand_rows = bench::load_csv(&candidate)?;
            let results = bench::compare_runs(&base_rows, &cand_rows, threshold);

            if json {
                println!("{}", bench::to_json(&results));
            } else if markdown {
                print!("{}", bench::to_markdown(&results));
            } else {
                bench::print_comparison_table(&results);
            }

            if fail_on_regression && bench::has_regressions(&results) {
                let count = results
                    .iter()
                    .filter(|r| r.classification == bench::Classification::Regression)
                    .count();
                return Err(Error::Regression(count));
            }
        }

        Command::Validate { json } => {
            let results = bench::validate::run_checks();

            if json {
                println!("{}", bench::report::validate_to_json(&results));
            } else {
                bench::validate::print_results(&results);
            }
        }

        Command::Run {
            binary,
            csv,
            quick,
            cycles,
            repeats,
            profile,
            taskset,
            analyze,
            extra_args,
        } => {
            let cfg = bench::runner::RunConfig {
                binary,
                csv: csv.clone(),
                quick,
                cycles,
                repeats,
                profile,
                taskset,
                extra_args,
            };

            let csv_path = bench::run_benchmark(&cfg)?;

            if analyze {
                if let Some(ref path) = csv_path {
                    println!();
                    println!("--- Post-run analysis ---");
                    let rows = bench::load_csv(path)?;
                    bench::print_summary_table(&rows, SortColumn::Name);
                } else {
                    eprintln!("Warning: --analyze requires --csv to produce output");
                }
            }
        }

        Command::GpuLock { action } => match action {
            GpuLockAction::Lock {
                device,
                freq,
                command,
            } => {
                let lock_cfg = bench::gpu_lock::LockConfig { device, freq };

                if command.is_empty() {
                    // Standalone lock mode
                    bench::gpu_lock::lock_clocks(&lock_cfg)?;
                } else {
                    // Wrapper mode: lock → run → reset
                    let wrap_cfg = bench::gpu_lock::WrapConfig {
                        lock: lock_cfg,
                        command,
                    };
                    let exit_code = bench::gpu_lock::wrap_command(&wrap_cfg)?;
                    if exit_code != 0 {
                        return Err(Error::InvalidArgs(format!(
                            "wrapped command exited with code {exit_code}"
                        )));
                    }
                }
            }
            GpuLockAction::Reset { device } => {
                bench::gpu_lock::reset_clocks(device)?;
            }
        },

        Command::GpuMonitor { action } => match action {
            GpuMonitorAction::Snapshot { output } => {
                let snapshot = bench::gpu_monitor::take_snapshot()?;
                let json = serde_json::to_string_pretty(&snapshot)
                    .map_err(|e| Error::Parse(e.to_string()))?;

                if let Some(path) = output {
                    bench::gpu_monitor::save_snapshot(&snapshot, &path)?;
                    eprintln!("Snapshot saved to {}", path.display());
                } else {
                    println!("{json}");
                }
            }
            GpuMonitorAction::Diff { before, after, json } => {
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

    Ok(())
}
