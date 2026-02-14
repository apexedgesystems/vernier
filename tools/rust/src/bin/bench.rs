//! bench: Benchmark analysis CLI tool.
//!
//! Single binary for benchmark analysis, comparison, validation, execution,
//! and flamegraph generation.
//!
//! Usage:
//!   bench summary <results.csv>                      # Pretty-print one CSV
//!   bench compare <baseline.csv> <candidate.csv>     # Colored regression diff
//!   bench validate                                   # Environment readiness
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
