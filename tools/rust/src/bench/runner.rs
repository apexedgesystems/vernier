//! Benchmark binary execution with optional CPU pinning and profiling.
//!
//! When `--profile <X>` names a wrap-externally backend (valgrind tools,
//! heaptrack, compute-sanitizer), the runner transparently invokes the
//! correct wrap command so a single `bench run --profile massif <bin>`
//! produces real heap-profile artifacts without the caller copy/pasting
//! the printed wrap instruction.

use std::fs;
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
    pub profile_output_dir: Option<PathBuf>,
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
    if let Some(ref dir) = cfg.profile_output_dir {
        args.push("--profile-output-dir".to_string());
        args.push(dir.display().to_string());
    }
    args.extend(cfg.extra_args.iter().cloned());

    // If the requested profile is a wrap-externally backend we know how to
    // wrap, build the wrap command (e.g. `valgrind --tool=massif ...`) and
    // run the benchmark binary under it. Otherwise execute the binary
    // directly. taskset, if requested, layers on the outside of either.
    let wrap = cfg
        .profile
        .as_deref()
        .and_then(|t| wrap_command_for(t, &cfg.binary, cfg.profile_output_dir.as_deref()));

    let mut cmd = match (&cfg.taskset, &wrap) {
        (Some(cpuset), Some((prog, prefix))) => {
            let mut c = Command::new("taskset");
            c.arg("-c").arg(cpuset).arg(prog).args(prefix);
            c
        }
        (Some(cpuset), None) => {
            let mut c = Command::new("taskset");
            c.arg("-c").arg(cpuset).arg(&cfg.binary);
            c
        }
        (None, Some((prog, prefix))) => {
            let mut c = Command::new(prog);
            c.args(prefix);
            c
        }
        (None, None) => Command::new(&cfg.binary),
    };

    cmd.args(&args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    println!(
        "Running: {}",
        format_command(&cfg.binary, &cfg.taskset, &wrap, &args)
    );

    let status = cmd.status()?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(Error::Parse(format!("benchmark exited with code {code}")));
    }

    Ok(cfg.csv.clone())
}

/* ----------------------------- Wrap-externally backends ----------------------------- */

/// Wrap command (program + prefix args ending in the binary path) for a
/// backend that must be invoked externally. Returns `None` for backends the
/// runner does not argv-wrap: the in-process ones (perf, gperf, rapl,
/// bpftrace, offcpu) that the binary drives from its own `--profile` flag,
/// plus jemalloc/nsight/rocprof whose wraps aren't reducible to argv.
///
/// Per-binary artifact dir convention: `<output-dir-or-bench-out>/<binary-stem>.<tool>/`.
/// The wrap tool writes its raw output there; the C++ harness then layers
/// per-test subdirs alongside as needed.
fn wrap_command_for(
    tool: &str,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Option<(String, Vec<String>)> {
    // Return early for in-process backends so we don't materialize a
    // per-tool artifact directory the C++ harness will never use --
    // perf/gperf/rapl/bpftrace/offcpu manage their own per-test dirs and
    // jemalloc/nsight/rocprof need wraps that aren't reducible to argv.
    if !matches!(
        tool,
        "callgrind" | "massif" | "memcheck" | "helgrind" | "heaptrack" | "compute-sanitizer"
    ) {
        return None;
    }

    let bin = binary.to_str()?.to_string();
    let stem = binary.file_stem()?.to_str()?.to_string();
    let root = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("bench-out"));
    let per_dir = root.join(format!("{}.{}", stem, tool));
    fs::create_dir_all(&per_dir).ok()?;
    let dir = per_dir.display().to_string();

    match tool {
        "callgrind" => Some((
            "valgrind".into(),
            // `--instr-atstart=no` would require `callgrind_control` to
            // toggle instrumentation around the measured region, which
            // can't cross PID namespaces (i.e. fails in Docker). Letting
            // callgrind instrument the whole run is slower but works
            // everywhere and matches what the binary's `Docker fallback`
            // hint already prints.
            vec![
                "--tool=callgrind".into(),
                format!("--callgrind-out-file={dir}/callgrind.out"),
                bin,
            ],
        )),
        "massif" => Some((
            "valgrind".into(),
            vec![
                "--tool=massif".into(),
                format!("--massif-out-file={dir}/massif.out"),
                bin,
            ],
        )),
        "memcheck" => Some((
            "valgrind".into(),
            vec![
                "--tool=memcheck".into(),
                "--leak-check=full".into(),
                "--error-exitcode=0".into(),
                format!("--log-file={dir}/memcheck.log"),
                bin,
            ],
        )),
        "helgrind" => Some((
            "valgrind".into(),
            vec![
                "--tool=helgrind".into(),
                format!("--log-file={dir}/helgrind.log"),
                bin,
            ],
        )),
        "heaptrack" => Some((
            "heaptrack".into(),
            vec!["-o".into(), format!("{dir}/run"), bin],
        )),
        "compute-sanitizer" => Some((
            "compute-sanitizer".into(),
            vec![
                "--tool=memcheck".into(),
                "--log-file".into(),
                format!("{dir}/sanitizer.log"),
                bin,
            ],
        )),
        _ => unreachable!("matches! filter above kept only wrap-externally tools"),
    }
}

/* ----------------------------- Helpers ----------------------------- */

fn format_command(
    binary: &Path,
    taskset: &Option<String>,
    wrap: &Option<(String, Vec<String>)>,
    args: &[String],
) -> String {
    let mut parts = Vec::new();
    if let Some(ref cpuset) = taskset {
        parts.push(format!("taskset -c {cpuset}"));
    }
    match wrap {
        Some((prog, prefix)) => {
            parts.push(prog.clone());
            parts.extend(prefix.iter().cloned());
        }
        None => parts.push(binary.display().to_string()),
    }
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
            &None,
            &["--quick".to_string()],
        );
        assert_eq!(s, "taskset -c 0,1 ./my_test --quick");
    }

    /// @test format_command with a wrap-externally backend.
    #[test]
    fn format_command_wrapped() {
        let wrap = Some((
            "valgrind".to_string(),
            vec![
                "--tool=massif".to_string(),
                "--massif-out-file=out/foo.massif/massif.out".to_string(),
                "./my_test".to_string(),
            ],
        ));
        let s = format_command(
            Path::new("./my_test"),
            &None,
            &wrap,
            &["--profile".to_string(), "massif".to_string()],
        );
        assert_eq!(
            s,
            "valgrind --tool=massif --massif-out-file=out/foo.massif/massif.out ./my_test --profile massif"
        );
    }

    /// @test wrap_command_for returns None for in-process backends.
    #[test]
    fn wrap_command_for_in_process_is_none() {
        for tool in [
            "perf", "gperf", "rapl", "bpftrace", "offcpu", "jemalloc", "nsight", "rocprof",
        ] {
            let r = wrap_command_for(tool, Path::new("./bin"), None);
            assert!(r.is_none(), "tool {tool} should be None");
        }
    }
}
