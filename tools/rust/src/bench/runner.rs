//! Benchmark binary execution with optional CPU pinning and profiling.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::Error;

/* ----------------------------- RunConfig ----------------------------- */

/// Configuration for executing a benchmark binary.
#[derive(Debug, Clone, Default)]
pub struct RunConfig {
    pub binary: PathBuf,
    pub csv: Option<PathBuf>,
    pub quick: bool,
    pub cycles: Option<u32>,
    pub repeats: Option<u32>,
    pub profile: Option<String>,
    pub taskset: Option<String>,
    pub extra_args: Vec<String>,
}

/* ----------------------------- API ----------------------------- */

/// Execute a benchmark binary with the given configuration.
///
/// Streams stdout/stderr to the terminal. Returns the CSV path (if set)
/// for use in post-run analysis.
pub fn run_benchmark(cfg: &RunConfig) -> Result<Option<PathBuf>, Error> {
    if !cfg.binary.is_file() {
        return Err(Error::InvalidArgs(format!(
            "binary not found: {}",
            cfg.binary.display()
        )));
    }

    let mut args: Vec<String> = Vec::new();

    // Build the command arguments for the benchmark binary
    if let Some(ref csv) = cfg.csv {
        args.push("--csv".to_string());
        args.push(csv.display().to_string());
    }
    if cfg.quick {
        args.push("--quick".to_string());
    }
    if let Some(cycles) = cfg.cycles {
        args.push("--cycles".to_string());
        args.push(cycles.to_string());
    }
    if let Some(repeats) = cfg.repeats {
        args.push("--repeats".to_string());
        args.push(repeats.to_string());
    }
    if let Some(ref profile) = cfg.profile {
        args.push("--profile".to_string());
        args.push(profile.clone());
    }
    args.extend(cfg.extra_args.iter().cloned());

    // Build the outer command (with optional taskset)
    let mut cmd = if let Some(ref cpuset) = cfg.taskset {
        let mut c = Command::new("taskset");
        c.arg("-c").arg(cpuset).arg(&cfg.binary);
        c
    } else {
        Command::new(&cfg.binary)
    };

    cmd.args(&args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    println!(
        "Running: {}",
        format_command(&cfg.binary, &cfg.taskset, &args)
    );

    let status = cmd.status()?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(Error::Parse(format!("benchmark exited with code {code}")));
    }

    Ok(cfg.csv.clone())
}

/* ----------------------------- Helpers ----------------------------- */

fn format_command(binary: &Path, taskset: &Option<String>, args: &[String]) -> String {
    let mut parts = Vec::new();
    if let Some(ref cpuset) = taskset {
        parts.push(format!("taskset -c {cpuset}"));
    }
    parts.push(binary.display().to_string());
    parts.extend(args.iter().cloned());
    parts.join(" ")
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test Missing binary returns InvalidArgs error.
    #[test]
    fn missing_binary_errors() {
        let cfg = RunConfig {
            binary: PathBuf::from("/nonexistent/binary"),
            ..Default::default()
        };
        let result = run_benchmark(&cfg);
        assert!(result.is_err());
    }

    /// @test format_command produces readable output.
    #[test]
    fn format_command_basic() {
        let s = format_command(
            Path::new("./my_test"),
            &None,
            &["--csv".to_string(), "out.csv".to_string()],
        );
        assert_eq!(s, "./my_test --csv out.csv");
    }

    /// @test format_command with taskset.
    #[test]
    fn format_command_taskset() {
        let s = format_command(
            Path::new("./my_test"),
            &Some("0,1".to_string()),
            &["--quick".to_string()],
        );
        assert_eq!(s, "taskset -c 0,1 ./my_test --quick");
    }
}
