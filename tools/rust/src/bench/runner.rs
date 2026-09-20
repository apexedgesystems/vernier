//! Benchmark binary execution with optional CPU pinning and profiling.
//!
//! When `--profile <X>` names a wrap-externally backend (valgrind tools,
//! heaptrack, compute-sanitizer, nsys), the runner transparently invokes
//! the correct wrap command so a single `bench run --profile massif <bin>`
//! produces real heap-profile artifacts without the caller copy/pasting
//! the printed wrap instruction. Wrapped children get
//! `VERNIER_EXTERNAL_WRAP=<tool>` so in-process backends stay passive
//! instead of re-attaching or printing manual-wrap hints; for nsight the
//! runner also extracts the canonical `nsys stats` reports after the run
//! (the .nsys-rep only exists once the wrapped process exits).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::{find_in_path, Error};

/* ----------------------------- RunConfig ----------------------------- */

/// Configuration for executing a benchmark binary.
#[derive(Debug, Clone, Default)]
pub struct RunConfig {
    pub binary: PathBuf,
    pub csv: Option<PathBuf>,
    pub quick: bool,
    pub cycles: Option<u32>,
    /// Auto-size cycles to a wall-time window (e.g. "100ms"); forwarded verbatim
    pub target_time: Option<String>,
    pub repeats: Option<u32>,
    pub profile: Option<String>,
    pub profile_args: Option<String>,
    pub profile_test_timeout: Option<u32>,
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
    if let Some(target_time) = &cfg.target_time {
        args.push("--target-time".to_string());
        args.push(target_time.clone());
    }
    if let Some(repeats) = cfg.repeats {
        args.push("--repeats".to_string());
        args.push(repeats.to_string());
    }
    if let Some(ref profile) = cfg.profile {
        args.push("--profile".to_string());
        args.push(profile.clone());
    }
    if let Some(ref pa) = cfg.profile_args {
        args.push("--profile-args".to_string());
        args.push(pa.clone());
    }
    if let Some(t) = cfg.profile_test_timeout {
        args.push("--profile-test-timeout".to_string());
        args.push(t.to_string());
    }
    if let Some(ref dir) = cfg.profile_output_dir {
        args.push("--profile-output-dir".to_string());
        args.push(dir.display().to_string());
    }
    args.extend(cfg.extra_args.iter().cloned());

    // Every program the runner itself spawns must resolve before anything
    // is created or started: a spawn failure only says "No such file or
    // directory", without saying which file.
    require_launch_programs(cfg.taskset.is_some(), cfg.profile.as_deref(), find_in_path)?;

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

    // Env-shaped wraps (jemalloc) inject via the environment instead of argv.
    let env_wrap = cfg
        .profile
        .as_deref()
        .and_then(|t| env_wrap_for(t, &cfg.binary, cfg.profile_output_dir.as_deref()));
    if let Some(ref pairs) = env_wrap {
        for (k, v) in pairs {
            cmd.env(k, v);
        }
    }

    // Tell the child which tool already wraps it, and which folder that wrap
    // writes into. Its in-process backend then stays passive (no re-attach,
    // no manual-wrap hint), creates no per-test folder of its own, and reports
    // this folder as the artifact location, which is what the CSV records.
    // See profiler_env::externalWrapTool() / externalWrapDir() on the C++ side.
    if wrap.is_some() || env_wrap.is_some() {
        if let Some(ref tool) = cfg.profile {
            for (k, v) in wrap_child_env(tool, &cfg.binary, cfg.profile_output_dir.as_deref()) {
                cmd.env(k, v);
            }
        }
    }

    let env_prefix = env_wrap
        .as_ref()
        .map(|pairs| {
            pairs
                .iter()
                .map(|(k, v)| format!("{k}={v} "))
                .collect::<String>()
        })
        .unwrap_or_default();
    println!(
        "Running: {}{}",
        env_prefix,
        format_command(&cfg.binary, &cfg.taskset, &wrap, &args)
    );

    let status = cmd.status()?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(Error::Parse(format!("benchmark exited with code {code}")));
    }

    // The wrapped nsys session writes its .nsys-rep at child exit, so the
    // report extraction the C++ backend does for attach-mode runs has to
    // happen runner-side for wrapped runs.
    if wrap.is_some() && cfg.profile.as_deref() == Some("nsight") {
        extract_nsys_stats(&wrap_artifact_dir(
            "nsight",
            &cfg.binary,
            cfg.profile_output_dir.as_deref(),
        ));
    }

    Ok(cfg.csv.clone())
}

/* ----------------------------- Wrap-externally backends ----------------------------- */

/// Per-binary artifact dir for a wrapped run:
/// `<output-dir-or-bench-out>/<binary-stem>.<tool>/`. The wrap tool writes
/// its raw output there; the C++ harness then layers per-test subdirs
/// alongside as needed.
fn wrap_artifact_dir(tool: &str, binary: &Path, output_dir: Option<&Path>) -> PathBuf {
    let stem = binary
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("bench");
    output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("bench-out"))
        .join(format!("{stem}.{tool}"))
}

/// Environment a wrapped child gets: the wrapping tool's name and the folder
/// the wrap writes into (the same `wrap_artifact_dir` the wrap command uses).
fn wrap_child_env(tool: &str, binary: &Path, output_dir: Option<&Path>) -> [(String, String); 2] {
    [
        ("VERNIER_EXTERNAL_WRAP".to_string(), tool.to_string()),
        (
            "VERNIER_EXTERNAL_WRAP_DIR".to_string(),
            wrap_artifact_dir(tool, binary, output_dir)
                .display()
                .to_string(),
        ),
    ]
}

/// Program that wraps the benchmark binary for a wrap-externally backend,
/// `None` for every other backend.
fn wrap_program(tool: &str) -> Option<&'static str> {
    match tool {
        "callgrind" | "massif" | "memcheck" | "helgrind" => Some("valgrind"),
        "heaptrack" => Some("heaptrack"),
        "compute-sanitizer" => Some("compute-sanitizer"),
        "nsight" => Some("nsys"),
        "ncu" => Some("ncu"),
        _ => None,
    }
}

/// Check that the programs a run is launched through resolve: `taskset` when
/// pinning, and the wrapper of a wrap-externally profile. `lookup` is the
/// PATH search (injected so tests need not edit the process environment).
fn require_launch_programs(
    pinned: bool,
    profile: Option<&str>,
    lookup: impl Fn(&str) -> Option<PathBuf>,
) -> Result<(), Error> {
    if pinned && lookup("taskset").is_none() {
        return Err(Error::ToolNotFound(
            "'taskset' is not on PATH; --taskset runs the benchmark under it. \
             Install taskset, or drop --taskset"
                .to_string(),
        ));
    }
    if let Some(tool) = profile {
        if let Some(program) = wrap_program(tool) {
            if lookup(program).is_none() {
                return Err(Error::ToolNotFound(format!(
                    "'{program}' is not on PATH; --profile {tool} runs the benchmark under it. \
                     Install {program}, or run `bench doctor` to see which profilers this \
                     machine can use"
                )));
            }
        }
    }
    Ok(())
}

/// Wrap command (program + prefix args ending in the binary path) for a
/// backend that must be invoked externally. Returns `None` for backends the
/// runner does not argv-wrap: the in-process ones (perf, gperf, rapl,
/// bpftrace, offcpu) that the binary drives from its own `--profile` flag,
/// plus jemalloc (env-shaped wrap: LD_PRELOAD + MALLOC_CONF) and rocprof
/// (injects its own tracer libs).
fn wrap_command_for(
    tool: &str,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Option<(String, Vec<String>)> {
    // Return early for in-process backends so we don't materialize a
    // per-tool artifact directory the C++ harness will never use --
    // perf/gperf/rapl/bpftrace/offcpu manage their own per-test dirs and
    // jemalloc/rocprof need wraps that aren't reducible to argv.
    let program = wrap_program(tool)?.to_string();

    let bin = binary.to_str()?.to_string();
    let per_dir = wrap_artifact_dir(tool, binary, output_dir);
    fs::create_dir_all(&per_dir).ok()?;
    let dir = per_dir.display().to_string();

    let args = match tool {
        "callgrind" => {
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
            ]
        }
        "massif" => {
            vec![
                "--tool=massif".into(),
                format!("--massif-out-file={dir}/massif.out"),
                bin,
            ]
        }
        "memcheck" => {
            vec![
                "--tool=memcheck".into(),
                "--leak-check=full".into(),
                "--error-exitcode=0".into(),
                format!("--log-file={dir}/memcheck.log"),
                bin,
            ]
        }
        "helgrind" => {
            vec![
                "--tool=helgrind".into(),
                format!("--log-file={dir}/helgrind.log"),
                bin,
            ]
        }
        "heaptrack" => {
            vec!["-o".into(), format!("{dir}/run"), bin]
        }
        "compute-sanitizer" => {
            vec![
                "--tool=memcheck".into(),
                "--log-file".into(),
                format!("{dir}/sanitizer.log"),
                bin,
            ]
        }
        // Mirrors the wrap command the C++ backend prints as its Docker
        // fallback hint; the child's backend stays passive via
        // VERNIER_EXTERNAL_WRAP and the runner extracts stats post-run.
        "nsight" => {
            vec![
                "profile".into(),
                "-o".into(),
                format!("{dir}/profile"),
                "-t".into(),
                "cuda,nvtx".into(),
                // nsys refuses to overwrite an existing report and the rerun
                // then silently reprocesses the stale capture; force it.
                "--force-overwrite".into(),
                "true".into(),
                bin,
            ]
        }
        // First-class Nsight Compute: same external-wrap pattern; the
        // replay pass stays a --profile-args opt-in inside the binary.
        "ncu" => {
            vec![
                "-o".into(),
                format!("{dir}/kernel_profile"),
                "-f".into(),
                "--target-processes".into(),
                "all".into(),
                bin,
            ]
        }
        _ => unreachable!("wrap_program() admits only the tools matched above"),
    };
    Some((program, args))
}

/// Env pairs for jemalloc's LD_PRELOAD wrap, pointing prof dumps at @p dir.
/// prof_final:true guarantees the exit-time dump the backend's docs promise.
/// The preload is by soname: ld.so searches the same dirs for LD_PRELOAD as
/// for any other resolution, so no path lookup is needed or wanted.
fn jemalloc_env(dir: &Path) -> Vec<(String, String)> {
    vec![
        ("LD_PRELOAD".to_string(), "libjemalloc.so.2".to_string()),
        (
            "MALLOC_CONF".to_string(),
            format!(
                "prof:true,prof_final:true,prof_prefix:{}/jeprof",
                dir.display()
            ),
        ),
    ]
}

/// Wrap for backends that inject via environment rather than argv.
/// jemalloc: LD_PRELOAD the library and enable profiling with a final dump
/// into the per-binary artifact dir. Returns None when the tool is not
/// env-shaped or the loader cannot resolve the library (the binary's own
/// hint then explains the manual setup).
fn env_wrap_for(
    tool: &str,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Option<Vec<(String, String)>> {
    if tool != "jemalloc" {
        return None;
    }
    if !jemalloc_preloadable() {
        return None;
    }
    let dir = wrap_artifact_dir(tool, binary, output_dir);
    fs::create_dir_all(&dir).ok()?;
    Some(jemalloc_env(&dir))
}

/// The loader is the only honest oracle for "will LD_PRELOAD work": preload
/// by soname against /bin/true and let ld.so answer. Stderr stays silent on
/// success; "cannot be preloaded" means the wrapped binary would not get the
/// library either (ld.so warns and continues, and the C++ side would then
/// see LD_PRELOAD set and believe it is profiled -- this gate prevents that).
fn jemalloc_preloadable() -> bool {
    Command::new("/bin/true")
        .env("LD_PRELOAD", "libjemalloc.so.2")
        .output()
        .map(|o| !String::from_utf8_lossy(&o.stderr).contains("cannot be preloaded"))
        .unwrap_or(false)
}

/// The four canonical nsys stats reports, matching what the C++ backend
/// extracts for attach-mode runs (ProfilerNsight.cu).
const NSYS_STATS_REPORTS: [&str; 4] = [
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_gpu_mem_time_sum",
];

/// Extract the canonical `nsys stats` reports beside a wrapped run's
/// .nsys-rep. Best-effort: a missing report file (nsys produced nothing)
/// or a failing nsys invocation prints a notice rather than erroring the
/// run, matching the attach-mode behavior.
fn extract_nsys_stats(dir: &Path) {
    let rep = dir.join("profile.nsys-rep");
    if !rep.is_file() {
        eprintln!(
            "[nsight] no report at {} (nsys produced nothing for this run)",
            rep.display()
        );
        return;
    }
    for report in NSYS_STATS_REPORTS {
        let out_path = dir.join(format!("{report}.txt"));
        let Ok(out_file) = fs::File::create(&out_path) else {
            continue;
        };
        let _ = Command::new("nsys")
            .args(["stats", "--force-export=true", "--report", report])
            .arg(&rep)
            .stdout(Stdio::from(out_file))
            .stderr(Stdio::null())
            .status();
    }
    println!(
        "[nsight] auto-extracted nsys stats reports into {}",
        dir.display()
    );
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

    /// @test Each wrap-externally profile maps to the program that wraps it.
    #[test]
    fn wrap_program_names_the_wrapper() {
        for (tool, program) in [
            ("callgrind", "valgrind"),
            ("massif", "valgrind"),
            ("memcheck", "valgrind"),
            ("helgrind", "valgrind"),
            ("heaptrack", "heaptrack"),
            ("compute-sanitizer", "compute-sanitizer"),
            ("nsight", "nsys"),
            ("ncu", "ncu"),
        ] {
            assert_eq!(wrap_program(tool), Some(program), "tool {tool}");
        }
        for tool in [
            "perf", "gperf", "rapl", "bpftrace", "offcpu", "jemalloc", "rocprof",
        ] {
            assert_eq!(wrap_program(tool), None, "tool {tool}");
        }
    }

    /// @test A missing wrapper is reported by program and profile name.
    #[test]
    fn require_launch_programs_names_missing_wrapper() {
        let err = require_launch_programs(false, Some("callgrind"), |_| None)
            .expect_err("valgrind does not resolve");
        assert!(matches!(err, Error::ToolNotFound(_)), "got {err:?}");
        let text = err.to_string();
        assert!(text.contains("'valgrind'"), "{text}");
        assert!(text.contains("--profile callgrind"), "{text}");
    }

    /// @test A missing taskset is reported when pinning is requested, and only then.
    #[test]
    fn require_launch_programs_names_missing_taskset() {
        let err =
            require_launch_programs(true, None, |_| None).expect_err("taskset does not resolve");
        assert!(err.to_string().contains("'taskset'"), "{err}");
        assert!(require_launch_programs(false, None, |_| None).is_ok());
    }

    /// @test Only the programs a run needs are looked up.
    #[test]
    fn require_launch_programs_looks_up_only_what_it_needs() {
        let only = |wanted: &'static str| {
            move |name: &str| (name == wanted).then(|| PathBuf::from("/usr/bin").join(name))
        };
        assert!(require_launch_programs(false, Some("massif"), only("valgrind")).is_ok());
        assert!(require_launch_programs(true, None, only("taskset")).is_ok());
        assert!(require_launch_programs(true, Some("massif"), only("taskset")).is_err());
        // In-process profiles are driven by the binary itself: nothing to resolve.
        assert!(require_launch_programs(false, Some("perf"), |_| None).is_ok());
    }

    /// @test The child is told the same folder the wrap command writes into.
    #[test]
    fn wrap_child_env_names_the_wrap_folder() {
        let root = std::env::temp_dir().join("vernier_runner_utst_wrap_env");
        for tool in [
            "callgrind",
            "massif",
            "heaptrack",
            "nsight",
            "ncu",
            "jemalloc",
        ] {
            let env = wrap_child_env(tool, Path::new("./my_test"), Some(&root));
            let expected = root.join(format!("my_test.{tool}"));
            assert_eq!(
                env[0],
                ("VERNIER_EXTERNAL_WRAP".to_string(), tool.to_string())
            );
            assert_eq!(
                env[1],
                (
                    "VERNIER_EXTERNAL_WRAP_DIR".to_string(),
                    expected.display().to_string()
                )
            );
            if let Some((_, args)) = wrap_command_for(tool, Path::new("./my_test"), Some(&root)) {
                assert!(
                    args.iter()
                        .any(|a| a.contains(&expected.display().to_string())),
                    "{tool}: wrap command does not write into {}: {args:?}",
                    expected.display()
                );
            }
        }
        let default_root = wrap_child_env("ncu", Path::new("./my_test"), None);
        assert_eq!(default_root[1].1, "bench-out/my_test.ncu");
    }

    /// @test wrap_command_for returns None for in-process backends.
    #[test]
    fn wrap_command_for_in_process_is_none() {
        for tool in [
            "perf", "gperf", "rapl", "bpftrace", "offcpu", "jemalloc", "rocprof",
        ] {
            let r = wrap_command_for(tool, Path::new("./bin"), None);
            assert!(r.is_none(), "tool {tool} should be None");
        }
    }

    /// @test nsight wraps with nsys profile into the per-binary artifact dir.
    #[test]
    fn wrap_command_for_nsight_uses_nsys_profile() {
        let root = std::env::temp_dir().join("vernier_runner_utst_nsight");
        let (prog, args) = wrap_command_for("nsight", Path::new("./my_test"), Some(&root))
            .expect("nsight should wrap externally");
        let dir = root.join("my_test.nsight");
        assert_eq!(prog, "nsys");
        assert_eq!(
            args,
            vec![
                "profile".to_string(),
                "-o".to_string(),
                format!("{}/profile", dir.display()),
                "-t".to_string(),
                "cuda,nvtx".to_string(),
                "--force-overwrite".to_string(),
                "true".to_string(),
                "./my_test".to_string(),
            ]
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// @test ncu wraps with the kernel-profile output and all target processes.
    #[test]
    fn wrap_command_for_ncu_uses_ncu() {
        let root = std::env::temp_dir().join("vernier_runner_utst_ncu");
        let (prog, args) = wrap_command_for("ncu", Path::new("./my_test"), Some(&root))
            .expect("ncu should wrap externally");
        let dir = root.join("my_test.ncu");
        assert_eq!(prog, "ncu");
        assert_eq!(
            args,
            vec![
                "-o".to_string(),
                format!("{}/kernel_profile", dir.display()),
                "-f".to_string(),
                "--target-processes".to_string(),
                "all".to_string(),
                "./my_test".to_string(),
            ]
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// @test wrap_artifact_dir follows the <root>/<stem>.<tool> convention.
    #[test]
    fn wrap_artifact_dir_convention() {
        let d = wrap_artifact_dir("nsight", Path::new("build/bin/ptests/Foo_PTEST"), None);
        assert_eq!(d, PathBuf::from("bench-out/Foo_PTEST.nsight"));
        let d = wrap_artifact_dir("massif", Path::new("./t"), Some(Path::new("out")));
        assert_eq!(d, PathBuf::from("out/t.massif"));
    }

    /// @test jemalloc env wrap composes LD_PRELOAD + MALLOC_CONF with a
    /// final dump into the artifact dir. The preload is by soname: ld.so
    /// owns the search, so no path appears anywhere.
    #[test]
    fn jemalloc_env_composition() {
        let pairs = jemalloc_env(Path::new("out/t.jemalloc"));
        assert_eq!(
            pairs[0],
            ("LD_PRELOAD".to_string(), "libjemalloc.so.2".to_string())
        );
        assert_eq!(
            pairs[1],
            (
                "MALLOC_CONF".to_string(),
                "prof:true,prof_final:true,prof_prefix:out/t.jemalloc/jeprof".to_string()
            )
        );
    }

    /// @test env_wrap_for is None for tools that are not env-shaped.
    #[test]
    fn env_wrap_for_non_jemalloc_is_none() {
        for tool in ["perf", "massif", "nsight", "ncu", "rocprof", "heaptrack"] {
            assert!(env_wrap_for(tool, Path::new("./b"), None).is_none());
        }
    }
}
