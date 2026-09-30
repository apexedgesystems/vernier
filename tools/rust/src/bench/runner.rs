//! Benchmark binary execution with optional CPU pinning and profiling.
//!
//! A request is read by its canonical name (`nsys` is `nsight`), which is also
//! the `--profile` the benchmark receives. When it names a wrap-externally
//! backend (valgrind tools, heaptrack, compute-sanitizer, nsys, ncu), the
//! runner starts the benchmark under the route that request selects: the
//! tool, and the mode its `--profile-args` words name, or it refuses a word
//! the tool does not take before anything is created or started. Wrapped
//! children get `VERNIER_EXTERNAL_WRAP=<tool>` so in-process backends stay
//! passive instead of re-attaching or printing manual-wrap hints; for
//! nsight's nsys route the runner also extracts the canonical `nsys stats`
//! reports after the run (the .nsys-rep only exists once the wrapped process
//! exits), and for compute-sanitizer it reads the tool's report to tell the
//! errors the tool found from the benchmark's own status.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use super::workflow::canonical_backend;
use super::{
    find_in_path, lookup_in_path, BenchmarkExit, Error, InPath, ProfileFailure, SanitizerFailure,
};

/// Exit status of a benchmark whose tests passed and whose requested profile
/// failed: libbench's `BENCH_PROFILE_FAILED_EXIT_CODE` (ProfilerRegistry.hpp).
pub const PROFILE_FAILED_EXIT_CODE: i32 = 4;

/// Exit status the compute-sanitizer route asks the tool to return when it
/// reports errors in the benchmark (`--error-exitcode`). Distinct from every
/// status libbench ends a benchmark with (1 to 4); the report, not this
/// status, says whether and how many errors were found.
pub const TOOL_FINDINGS_EXIT_CODE: i32 = 5;

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
    /// `--profile-analyze`: the requested profile's analysis (forwarded to
    /// the binary; for callgrind, run by the runner after the exit).
    pub profile_analyze: bool,
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
    let tool = cfg.profile.as_deref().map(canonical_backend);
    args.extend(profile_request_args(cfg));
    // --profile-analyze given to bench run, or forwarded after `--`: one
    // request either way.
    let analyze = cfg.profile_analyze || cfg.extra_args.iter().any(|a| a == "--profile-analyze");
    args.extend(cfg.extra_args.iter().cloned());

    // The route a wrapped profile runs under (e.g. `valgrind --tool=massif
    // ...`), decided before anything is created or started: a mode the tool
    // does not take is refused here.
    let route = match tool {
        Some(tool) => route_for(
            tool,
            cfg.profile_args.as_deref(),
            &cfg.binary,
            cfg.profile_output_dir.as_deref(),
        )?,
        None => None,
    };

    // Every program the runner itself spawns must resolve before anything
    // is created or started: a spawn failure only says "No such file or
    // directory", without saying which file.
    require_launch_programs(
        cfg.taskset.is_some(),
        route.as_ref().map(|r| {
            (
                r.program,
                request_text(&r.tool, cfg.profile_args.as_deref()),
            )
        }),
        lookup_in_path,
    )?;

    // The route's folder exists and holds none of the files this run's wrap
    // will write, so what is there after the run is this run's. The
    // benchmark runs under its route when it has one, directly otherwise;
    // taskset, if requested, layers on the outside of either.
    if let Some(ref r) = route {
        prepare_folder(&r.dir, &r.stale_names())?;
    }
    let wrap = route
        .as_ref()
        .map(|r| (r.program.to_string(), r.args.clone()));

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
    let env_wrap = match tool {
        Some(t) => env_wrap_for(t, &cfg.binary, cfg.profile_output_dir.as_deref())?,
        None => None,
    };
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
        if let Some(tool) = tool {
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

    // compute-sanitizer's status mixes its own verdict with the benchmark's:
    // its report decides which one this is.
    if let Some(ref r) = route {
        if r.program == "compute-sanitizer" {
            check_sanitizer_run(
                &r.dir.join("sanitizer.log"),
                status,
                &request_text(&r.tool, cfg.profile_args.as_deref()),
            )?;
        }
    }
    if let Some(end) = benchmark_exit(status) {
        return Err(Error::Benchmark(end));
    }

    // A wrap writes its output when the process it started exits: this run's
    // output is checked here, after that exit, and only then analysed.
    if let Some(ref r) = route {
        let request = request_text(&r.tool, cfg.profile_args.as_deref());
        for alternatives in r.outputs {
            require_output(&r.tool, &request, &r.dir, alternatives)?;
        }
        if r.program == "nsys" {
            extract_nsys_stats(&r.dir, &request)?;
        }
        if analyze && r.tool == "callgrind" {
            annotate_callgrind(&r.dir.join("callgrind.out"), &request)?;
        }
    }
    if env_wrap.is_some() {
        if let Some(tool) = tool {
            require_jemalloc_dumps(
                &wrap_artifact_dir(tool, &cfg.binary, cfg.profile_output_dir.as_deref()),
                &request_text(tool, cfg.profile_args.as_deref()),
            )?;
        }
    }

    Ok(cfg.csv.clone())
}

/// The profile request as the benchmark reads it: `--profile` by its
/// canonical name, `--profile-args`, `--profile-test-timeout`,
/// `--profile-output-dir`, and `--profile-analyze` unless it is among the
/// arguments forwarded after `--` (which the caller appends after these).
/// `bench run` and `bench doctor` spell a request with this one function.
pub fn profile_request_args(cfg: &RunConfig) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(tool) = cfg.profile.as_deref().map(canonical_backend) {
        args.push("--profile".to_string());
        args.push(tool.to_string());
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
    if cfg.profile_analyze && !cfg.extra_args.iter().any(|a| a == "--profile-analyze") {
        args.push("--profile-analyze".to_string());
    }
    args
}

/// Create a wrap's folder and remove from it the previous run's copies of the
/// files the wrap writes (@p names, exact names), so that no stale output can
/// stand for this run's. Nothing else in the folder or outside it is touched.
fn prepare_folder(dir: &Path, names: &[&str]) -> Result<(), Error> {
    fs::create_dir_all(dir).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("cannot create the output folder {}: {e}", dir.display()),
        ))
    })?;
    for name in names {
        remove_previous(&dir.join(name))?;
    }
    Ok(())
}

/// Remove one file a previous run left, saying so; a file that is not there
/// is fine.
fn remove_previous(path: &Path) -> Result<(), Error> {
    match fs::remove_file(path) {
        Ok(()) => {
            println!("[bench] removed {} from a previous run", path.display());
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::Io(std::io::Error::new(
            e.kind(),
            format!(
                "cannot remove {} left by a previous run: {e}",
                path.display()
            ),
        ))),
    }
}

/// The failure of a requested profile after its benchmark ran.
fn profile_failure(request: &str, stage: &'static str, message: String) -> Error {
    Error::Profile(ProfileFailure {
        request: request.to_string(),
        stage,
        message,
    })
}

/// Require one of @p alternatives (file names in @p dir) to exist and hold
/// something, and print which, with its size.
fn require_output(
    tool: &str,
    request: &str,
    dir: &Path,
    alternatives: &[&str],
) -> Result<(), Error> {
    let found = alternatives
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file());
    let Some(path) = found else {
        let names = alternatives
            .iter()
            .map(|name| dir.join(name).display().to_string())
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(profile_failure(
            request,
            "completion",
            format!("{names} was not written"),
        ));
    };
    let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if size == 0 {
        return Err(profile_failure(
            request,
            "completion",
            format!("{} is empty", path.display()),
        ));
    }
    println!("[bench] {tool} wrote {} ({size} bytes)", path.display());
    Ok(())
}

/// jemalloc's dumps, `jeprof.<pid>.<n>.<kind>.heap` under the prefix the
/// environment wrap sets: the names carry the process id, so they are matched
/// by that prefix and suffix inside the wrap's own folder.
fn jemalloc_dumps(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut dumps: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("jeprof.") && n.ends_with(".heap"))
        })
        .collect();
    dumps.sort();
    dumps
}

/// Require a non-empty jemalloc dump from this run, and print each.
fn require_jemalloc_dumps(dir: &Path, request: &str) -> Result<(), Error> {
    let dumps: Vec<(PathBuf, u64)> = jemalloc_dumps(dir)
        .into_iter()
        .map(|p| {
            let size = fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            (p, size)
        })
        .filter(|(_, size)| *size > 0)
        .collect();
    if dumps.is_empty() {
        return Err(profile_failure(
            request,
            "completion",
            format!(
                "no jemalloc dump (jeprof.*.heap) was written in {}",
                dir.display()
            ),
        ));
    }
    for (dump, size) in dumps {
        println!("[bench] jemalloc wrote {} ({size} bytes)", dump.display());
    }
    Ok(())
}

/// How a helper program ended, for messages.
fn describe_status(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exited with status {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("was ended by signal {signal}");
        }
    }
    "ended without a status".to_string()
}

/// How a finished benchmark ended, or `None` when it succeeded.
///
/// For a wrapped run this is the wrap's status; valgrind (memcheck with
/// `--error-exitcode=0`), heaptrack and taskset pass the benchmark's own
/// status through.
fn benchmark_exit(status: ExitStatus) -> Option<BenchmarkExit> {
    if status.success() {
        return None;
    }
    if let Some(code) = status.code() {
        return Some(if code == PROFILE_FAILED_EXIT_CODE {
            BenchmarkExit::ProfileFailed
        } else {
            BenchmarkExit::Status(code)
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return Some(BenchmarkExit::Signal(signal));
        }
    }
    Some(BenchmarkExit::Status(-1))
}

/* ----------------------------- Wrap-externally backends ----------------------------- */

/// Per-binary artifact dir for a wrapped run:
/// `<output-dir-or-bench-out>/<binary-stem>.<tool>/`. The wrap tool writes
/// its output there; the wrapped benchmark creates no per-test folders and
/// reports this one as its artifact location (see `wrap_child_env`).
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

/// A request as the command line states it, for messages.
fn request_text(tool: &str, profile_args: Option<&str>) -> String {
    match profile_args {
        Some(a) if !a.is_empty() => format!("--profile {tool} --profile-args '{a}'"),
        _ => format!("--profile {tool}"),
    }
}

/// Check that the programs a run is launched through resolve to executable
/// files: `taskset` when pinning, and the program of a wrapped profile's route
/// with the request it serves. `lookup` is the PATH search (injected so tests
/// need not edit the process environment).
fn require_launch_programs(
    pinned: bool,
    wrapper: Option<(&str, String)>,
    lookup: impl Fn(&str) -> InPath,
) -> Result<(), Error> {
    if pinned {
        match lookup("taskset") {
            InPath::Executable(_) => {}
            InPath::NotExecutable(path) => {
                return Err(Error::ToolNotFound(format!(
                    "'taskset' at {} is not executable; --taskset runs the benchmark under \
                     it. Make it executable, put a working taskset first on PATH, or drop \
                     --taskset",
                    path.display()
                )))
            }
            InPath::Absent => {
                return Err(Error::ToolNotFound(
                    "'taskset' is not on PATH; --taskset runs the benchmark under it. \
                     Install taskset, or drop --taskset"
                        .to_string(),
                ))
            }
        }
    }
    if let Some((program, request)) = wrapper {
        match lookup(program) {
            InPath::Executable(_) => {}
            InPath::NotExecutable(path) => {
                return Err(Error::ToolNotFound(format!(
                    "'{program}' at {} is not executable; {request} runs the benchmark under \
                     it. Make it executable, or put a working {program} first on PATH",
                    path.display()
                )))
            }
            InPath::Absent => {
                return Err(Error::ToolNotFound(format!(
                    "'{program}' is not on PATH; {request} runs the benchmark under it. \
                     Install {program}, or run `bench doctor` to see which profilers this \
                     machine can use"
                )))
            }
        }
    }
    Ok(())
}

/* ----------------------------- Routes ----------------------------- */

/// How `bench run` starts a wrap-externally profile.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Route {
    /// Canonical backend name: the child's VERNIER_EXTERNAL_WRAP and the
    /// suffix of the folder.
    tool: String,
    /// The program started in place of the benchmark.
    program: &'static str,
    /// The program's arguments, ending with the benchmark binary.
    args: Vec<String>,
    /// The folder the wrap writes into.
    dir: PathBuf,
    /// The files the wrap writes into the folder: each entry is one required
    /// output, satisfied by any one of its names.
    outputs: &'static [&'static [&'static str]],
    /// Files the runner derives from the output after the run.
    derived: &'static [&'static str],
}

impl Route {
    /// Every name a run of this route writes into its folder: removed from
    /// the folder before the run.
    fn stale_names(&self) -> Vec<&'static str> {
        self.outputs
            .iter()
            .flat_map(|alternatives| alternatives.iter().copied())
            .chain(self.derived.iter().copied())
            .collect()
    }
}

/// The summaries extracted from an nsys report, one `<report>.txt` each, and
/// the SQLite export `nsys stats` writes beside the report.
const NSYS_DERIVED: [&str; 5] = [
    "profile.sqlite",
    "cuda_gpu_kern_sum.txt",
    "cuda_api_sum.txt",
    "cuda_gpu_mem_size_sum.txt",
    "cuda_gpu_mem_time_sum.txt",
];

/// The words of a `--profile-args` value, split on whitespace and commas:
/// for a wrapped profile, the modes it selects.
fn mode_words(profile_args: Option<&str>) -> Vec<&str> {
    profile_args
        .unwrap_or("")
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty())
        .collect()
}

/// The modes a wrap-externally backend takes, as its header advertises them;
/// `None` for every profile `bench run` does not wrap in argv.
fn wrapped_modes(tool: &str) -> Option<&'static [&'static str]> {
    Some(match tool {
        "callgrind" | "heaptrack" | "ncu" => &[],
        "massif" => &["pages", "stacks"],
        "memcheck" => &["leak-full", "track-origins"],
        "helgrind" => &["drd"],
        "compute-sanitizer" => &["memcheck", "racecheck", "synccheck", "initcheck"],
        "nsight" => &["compute", "ncu"],
        _ => return None,
    })
}

/// The route of a request for canonical backend @p tool with its
/// `--profile-args`, or `None` for the profiles the benchmark runs itself
/// (perf, gperf, rapl, bpftrace, offcpu), jemalloc's environment wrap and
/// rocprof. A word the tool does not take, a combination it cannot run and a
/// mode `bench run` does not wrap are refused, naming what is accepted.
/// Creates nothing.
fn route_for(
    tool: &str,
    profile_args: Option<&str>,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Result<Option<Route>, Error> {
    let Some(accepted) = wrapped_modes(tool) else {
        return Ok(None);
    };
    let request = request_text(tool, profile_args);
    let refuse = |why: String| Err(Error::InvalidArgs(format!("{request}: {why}")));
    let words = mode_words(profile_args);
    if (tool == "nsight" || tool == "ncu") && words.contains(&"replay") {
        return refuse(
            "bench run does not wrap a kernel replay; run the benchmark directly with \
             these flags, and it prints the ncu command that replays its kernels"
                .to_string(),
        );
    }
    if let Some(word) = words.iter().find(|w| !accepted.contains(w)) {
        return refuse(if accepted.is_empty() {
            format!("'{word}' is not a mode of {tool}, which takes none")
        } else {
            format!(
                "'{word}' is not a mode of {tool}; its modes are {}",
                accepted.join(", ")
            )
        });
    }
    let has = |mode: &str| words.contains(&mode);
    let bin = binary
        .to_str()
        .ok_or_else(|| {
            Error::InvalidArgs(format!(
                "{request}: the binary path {} is not valid UTF-8",
                binary.display()
            ))
        })?
        .to_string();
    let dir = wrap_artifact_dir(tool, binary, output_dir);
    let d = dir.display().to_string();
    let ncu = |d: &str| -> Vec<String> {
        vec![
            "-o".into(),
            format!("{d}/kernel_profile"),
            "-f".into(),
            "--target-processes".into(),
            "all".into(),
        ]
    };
    let (program, mut args, outputs): Routed = match tool {
        // The whole process is recorded: under the runner's wrap the
        // benchmark's callgrind backend leaves the recording alone.
        "callgrind" => (
            "valgrind",
            vec![
                "--tool=callgrind".into(),
                format!("--callgrind-out-file={d}/callgrind.out"),
            ],
            &[&["callgrind.out"]],
        ),
        "massif" => {
            if has("pages") && has("stacks") {
                return refuse(
                    "valgrind's massif cannot combine pages (--pages-as-heap=yes) with \
                     stacks (--stacks=yes); choose one"
                        .to_string(),
                );
            }
            let mut a = vec!["--tool=massif".to_string()];
            if has("pages") {
                a.push("--pages-as-heap=yes".into());
            }
            if has("stacks") {
                a.push("--stacks=yes".into());
            }
            a.push(format!("--massif-out-file={d}/massif.out"));
            ("valgrind", a, &[&["massif.out"]])
        }
        "memcheck" => {
            let mut a = vec![
                "--tool=memcheck".to_string(),
                "--leak-check=full".into(),
                "--error-exitcode=0".into(),
            ];
            if has("track-origins") {
                a.push("--track-origins=yes".into());
            }
            a.push(format!("--log-file={d}/memcheck.log"));
            ("valgrind", a, &[&["memcheck.log"]])
        }
        "helgrind" => (
            "valgrind",
            vec![
                if has("drd") {
                    "--tool=drd"
                } else {
                    "--tool=helgrind"
                }
                .into(),
                format!("--log-file={d}/helgrind.log"),
            ],
            &[&["helgrind.log"]],
        ),
        // heaptrack appends the suffix of the compression it was built with.
        "heaptrack" => (
            "heaptrack",
            vec!["-o".into(), format!("{d}/run")],
            &[&["run.zst", "run.gz"]],
        ),
        "compute-sanitizer" => {
            if words.len() > 1 {
                return refuse(format!(
                    "compute-sanitizer runs one tool at a time; choose one of {}",
                    accepted.join(", ")
                ));
            }
            let sanitizer = words.first().copied().unwrap_or("memcheck");
            (
                "compute-sanitizer",
                vec![
                    format!("--tool={sanitizer}"),
                    "--error-exitcode".into(),
                    TOOL_FINDINGS_EXIT_CODE.to_string(),
                    "--log-file".into(),
                    format!("{}/sanitizer.log", escape_percent(&d)),
                ],
                &[&["sanitizer.log"]],
            )
        }
        // nsight's compute mode is ncu's route, into nsight's folder.
        "nsight" if has("compute") || has("ncu") => {
            ("ncu", ncu(&d), &[&["kernel_profile.ncu-rep"]])
        }
        // nsys records the whole process; the benchmark's backend stays
        // passive (VERNIER_EXTERNAL_WRAP) and the summaries are extracted
        // after the run. nsys refuses to overwrite a report, so a rerun
        // forces it rather than reprocess the previous capture.
        "nsight" => (
            "nsys",
            vec![
                "profile".into(),
                "-o".into(),
                format!("{d}/profile"),
                "-t".into(),
                "cuda,nvtx".into(),
                "--force-overwrite".into(),
                "true".into(),
            ],
            &[&["profile.nsys-rep"]],
        ),
        "ncu" => ("ncu", ncu(&d), &[&["kernel_profile.ncu-rep"]]),
        _ => unreachable!("wrapped_modes() admits only the tools matched above"),
    };
    args.push(bin);
    Ok(Some(Route {
        tool: tool.to_string(),
        program,
        args,
        dir,
        outputs,
        derived: if program == "nsys" {
            &NSYS_DERIVED
        } else {
            &[]
        },
    }))
}

/// A route's program, arguments and required outputs, as `route_for` builds them.
type Routed = (
    &'static str,
    Vec<String>,
    &'static [&'static [&'static str]],
);

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
/// into the per-binary artifact dir, whose previous dumps are removed first.
/// `Ok(None)` when the tool is not env-shaped or the loader cannot resolve
/// the library (the binary's own hint then explains the manual setup); an
/// error when the folder cannot be prepared.
fn env_wrap_for(
    tool: &str,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Result<Option<Vec<(String, String)>>, Error> {
    if tool != "jemalloc" || !jemalloc_preloadable() {
        return Ok(None);
    }
    let dir = wrap_artifact_dir(tool, binary, output_dir);
    prepare_folder(&dir, &[])?;
    for dump in jemalloc_dumps(&dir) {
        remove_previous(&dump)?;
    }
    Ok(Some(jemalloc_env(&dir)))
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

/// How long `callgrind_annotate` may take on one profile.
const ANNOTATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// How many lines of the annotation are printed.
const ANNOTATE_LINES: usize = 40;

/// Annotate a callgrind profile valgrind has finished writing (checked
/// before this runs): `callgrind_annotate --auto=yes <profile>`, bounded,
/// its first lines printed. A missing, failing or overrunning annotator is
/// an analysis failure; the profile is kept.
fn annotate_callgrind(profile: &Path, request: &str) -> Result<(), Error> {
    let kept = format!("; the profile is kept at {}", profile.display());
    let Some(annotator) = find_in_path("callgrind_annotate") else {
        return Err(profile_failure(
            request,
            "analysis",
            format!("callgrind_annotate is not on PATH (it ships with valgrind){kept}"),
        ));
    };
    let mut child = Command::new(&annotator)
        .arg("--auto=yes")
        .arg(profile)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            profile_failure(
                request,
                "analysis",
                format!("{} could not be started: {e}{kept}", annotator.display()),
            )
        })?;
    // Read both pipes while waiting, so a long annotation cannot fill one.
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let out_reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut stdout, &mut text);
        text
    });
    let err_reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
        text
    });
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if started.elapsed() >= ANNOTATE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let out = out_reader.join().unwrap_or_default();
    let err = err_reader.join().unwrap_or_default();
    let Some(status) = status else {
        return Err(profile_failure(
            request,
            "analysis",
            format!(
                "{} did not finish within {} s and was stopped{kept}",
                annotator.display(),
                ANNOTATE_TIMEOUT.as_secs()
            ),
        ));
    };
    if !status.success() {
        let tail = err
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("");
        return Err(profile_failure(
            request,
            "analysis",
            format!(
                "{} {}{}{}{kept}",
                annotator.display(),
                describe_status(status),
                if tail.is_empty() { "" } else { ": " },
                tail
            ),
        ));
    }
    println!(
        "\n--- callgrind_annotate {} (first {ANNOTATE_LINES} lines) ---\n",
        profile.display()
    );
    for line in out.lines().take(ANNOTATE_LINES) {
        println!("{line}");
    }
    println!();
    Ok(())
}

/// The four summaries extracted from a wrapped nsight run's report.
const NSYS_STATS_REPORTS: [&str; 4] = [
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_gpu_mem_time_sum",
];

/// Extract the four `nsys stats` summaries beside the report of a wrapped
/// nsight run, one `<report>.txt` each. The report exists (checked before
/// this runs). An `nsys stats` that fails is an analysis failure naming the
/// summary, its status and the end of its error output; the report is kept.
fn extract_nsys_stats(dir: &Path, request: &str) -> Result<(), Error> {
    let rep = dir.join("profile.nsys-rep");
    for report in NSYS_STATS_REPORTS {
        let out_path = dir.join(format!("{report}.txt"));
        let out_file = fs::File::create(&out_path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot write {}: {e}", out_path.display()),
            ))
        })?;
        let out = Command::new("nsys")
            .args(["stats", "--force-export=true", "--report", report])
            .arg(&rep)
            .stdin(Stdio::null())
            .stdout(Stdio::from(out_file))
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot start nsys stats: {e}"),
                ))
            })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail = stderr
                .lines()
                .rev()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("");
            return Err(profile_failure(
                request,
                "analysis",
                format!(
                    "nsys stats --report {report} {}{}{}; the report is kept at {}",
                    describe_status(out.status),
                    if tail.is_empty() { "" } else { ": " },
                    tail,
                    rep.display()
                ),
            ));
        }
    }
    println!(
        "[nsight] wrote the nsys stats summaries into {}",
        dir.display()
    );
    Ok(())
}

/// compute-sanitizer's `--log-file` value for @p path: the tool expands `%p`,
/// `%q{VAR}` and `%%` in it and refuses any other `%`, so each `%` of the
/// path is doubled.
fn escape_percent(path: &str) -> String {
    path.replace('%', "%%")
}

/// The error count of a compute-sanitizer report: its last `ERROR SUMMARY: N
/// error(s)` line, or `None` when it has none (a report cut short, or a tool
/// that stopped before summarizing).
fn sanitizer_error_count(report: &str) -> Option<u64> {
    report.lines().rev().find_map(|line| {
        let rest = line.split_once("ERROR SUMMARY:")?.1.trim();
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let count = rest[..digits].parse().ok()?;
        rest[digits..]
            .trim_start()
            .starts_with("error")
            .then_some(count)
    })
}

/// The first line of @p report holding @p text, without the tool's
/// `=========` prefix.
fn report_line<'a>(report: &'a str, text: &str) -> Option<&'a str> {
    report
        .lines()
        .find(|line| line.contains(text))
        .map(|line| line.trim_start_matches('=').trim())
}

/// Decide a finished compute-sanitizer run from this run's report and the
/// tool's exit status. The route passes `--error-exitcode 5`, and the tool
/// then returns 5 whenever it reports errors, whatever the benchmark itself
/// returned; without errors it returns the benchmark's own status. So the
/// report's summary decides: errors counted there are findings; no errors
/// leave the status to the benchmark (a benchmark returning 5 itself is not a
/// finding), unless the report says the benchmark did not end normally. A
/// report without its summary is the tool's own failure when it holds the
/// tool's `Error:` line (collection), and otherwise incomplete (completion);
/// no report at all is a completion failure naming its path. Nothing is
/// counted without a summary. `Ok` hands the status on as the benchmark's.
fn check_sanitizer_run(report: &Path, status: ExitStatus, request: &str) -> Result<(), Error> {
    let how = describe_status(status);
    let Ok(text) = fs::read_to_string(report) else {
        return Err(profile_failure(
            request,
            "completion",
            format!(
                "compute-sanitizer {how} and wrote no report at {}: nothing can be counted",
                report.display()
            ),
        ));
    };
    match sanitizer_error_count(&text) {
        Some(0) => {
            if status.success() {
                return Ok(());
            }
            if let Some(line) = report_line(&text, "process didn't terminate successfully") {
                return Err(Error::Sanitizer(SanitizerFailure::AbnormalEnd {
                    line: line.to_string(),
                    report: report.to_path_buf(),
                    status: how,
                }));
            }
            if status.code() == Some(TOOL_FINDINGS_EXIT_CODE) {
                println!(
                    "[bench] compute-sanitizer's report {} counts no errors: status {} is \
                     the benchmark's own",
                    report.display(),
                    TOOL_FINDINGS_EXIT_CODE
                );
            }
            Ok(())
        }
        Some(errors) => Err(Error::Sanitizer(SanitizerFailure::Findings {
            errors,
            report: report.to_path_buf(),
            status: how,
            benchmark_failed: report_line(&text, "Target application returned an error").is_some(),
        })),
        None => Err(match report_line(&text, "Error:") {
            Some(line) => profile_failure(
                request,
                "collection",
                format!(
                    "compute-sanitizer reported its own error, \"{line}\" (the tool {how}); \
                     the report is {}",
                    report.display()
                ),
            ),
            None => profile_failure(
                request,
                "completion",
                format!(
                    "compute-sanitizer {how}, and its report {} holds no error summary: it \
                     is incomplete, and nothing in it is counted",
                    report.display()
                ),
            ),
        }),
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

    /// @test Each way a benchmark can end maps to one report; success to none.
    #[test]
    #[cfg(unix)]
    fn benchmark_exit_names_status_signal_and_profile_failure() {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(benchmark_exit(ExitStatus::from_raw(0)), None);
        assert_eq!(
            benchmark_exit(ExitStatus::from_raw(1 << 8)),
            Some(BenchmarkExit::Status(1))
        );
        assert_eq!(
            benchmark_exit(ExitStatus::from_raw(PROFILE_FAILED_EXIT_CODE << 8)),
            Some(BenchmarkExit::ProfileFailed)
        );
        assert_eq!(
            benchmark_exit(ExitStatus::from_raw(9)),
            Some(BenchmarkExit::Signal(9))
        );
        assert_eq!(PROFILE_FAILED_EXIT_CODE, 4);
    }

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

    /// @test Each wrap-externally profile maps to the program that wraps it;
    /// the profiles the benchmark runs itself have no route.
    #[test]
    fn route_program_names_the_wrapper() {
        let bin = Path::new("./my_test");
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
            let route = route_for(tool, None, bin, None)
                .expect("the default mode is accepted")
                .expect("a wrapped tool has a route");
            assert_eq!(route.program, program, "tool {tool}");
            assert_eq!(route.tool, tool);
        }
        for tool in [
            "perf", "gperf", "rapl", "bpftrace", "offcpu", "jemalloc", "rocprof",
        ] {
            // Whatever the mode text: it is the benchmark's to read.
            let route = route_for(tool, Some("record -g"), bin, None).expect("not refused");
            assert!(route.is_none(), "tool {tool}");
        }
    }

    /// @test A missing wrapper is reported by program and request.
    #[test]
    fn require_launch_programs_names_missing_wrapper() {
        let err = require_launch_programs(
            false,
            Some(("ncu", request_text("nsight", Some("compute")))),
            |_| InPath::Absent,
        )
        .expect_err("ncu does not resolve");
        assert!(matches!(err, Error::ToolNotFound(_)), "got {err:?}");
        let text = err.to_string();
        assert!(text.contains("'ncu'"), "{text}");
        assert!(
            text.contains("--profile nsight --profile-args 'compute'"),
            "{text}"
        );
    }

    /// @test A missing taskset is reported when pinning is requested, and only then.
    #[test]
    fn require_launch_programs_names_missing_taskset() {
        let err = require_launch_programs(true, None, |_| InPath::Absent)
            .expect_err("taskset does not resolve");
        assert!(err.to_string().contains("'taskset'"), "{err}");
        assert!(require_launch_programs(false, None, |_| InPath::Absent).is_ok());
    }

    /// @test A wrapper or taskset that PATH finds without an execute bit is
    /// named with its path, before anything is spawned.
    #[test]
    fn require_launch_programs_names_a_non_executable_program() {
        let plain = |name: &str| InPath::NotExecutable(PathBuf::from("/opt/x").join(name));
        let err = require_launch_programs(
            false,
            Some(("valgrind", request_text("massif", Some("pages")))),
            plain,
        )
        .expect_err("valgrind is not executable");
        let text = err.to_string();
        assert!(
            text.contains("'valgrind' at /opt/x/valgrind is not executable"),
            "{text}"
        );
        assert!(
            text.contains("--profile massif --profile-args 'pages'"),
            "{text}"
        );
        let err = require_launch_programs(true, None, plain).expect_err("taskset");
        assert!(
            err.to_string()
                .contains("'taskset' at /opt/x/taskset is not executable"),
            "{err}"
        );
    }

    /// @test Only the programs a run needs are looked up.
    #[test]
    fn require_launch_programs_looks_up_only_what_it_needs() {
        let only = |wanted: &'static str| {
            move |name: &str| {
                if name == wanted {
                    InPath::Executable(PathBuf::from("/usr/bin").join(name))
                } else {
                    InPath::Absent
                }
            }
        };
        let massif = || Some(("valgrind", request_text("massif", None)));
        assert!(require_launch_programs(false, massif(), only("valgrind")).is_ok());
        assert!(require_launch_programs(true, None, only("taskset")).is_ok());
        assert!(require_launch_programs(true, massif(), only("taskset")).is_err());
        // In-process profiles are driven by the binary itself: nothing to resolve.
        assert!(require_launch_programs(false, None, |_| InPath::Absent).is_ok());
    }

    /// @test The child is told the same folder the route writes into.
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
            if let Some(route) =
                route_for(tool, None, Path::new("./my_test"), Some(&root)).expect("accepted")
            {
                assert_eq!(route.dir, expected, "{tool}");
                assert!(
                    route
                        .args
                        .iter()
                        .any(|a| a.contains(&expected.display().to_string())),
                    "{tool}: route does not write into {}: {:?}",
                    expected.display(),
                    route.args
                );
            }
        }
        let default_root = wrap_child_env("ncu", Path::new("./my_test"), None);
        assert_eq!(default_root[1].1, "bench-out/my_test.ncu");
    }

    /// @test A route decides without creating its folder.
    #[test]
    fn route_for_creates_nothing() {
        let root = std::env::temp_dir().join("vernier_runner_utst_route_creates_nothing");
        let _ = fs::remove_dir_all(&root);
        let route = route_for("massif", Some("pages"), Path::new("./my_test"), Some(&root))
            .expect("accepted")
            .expect("massif has a route");
        assert!(!route.dir.exists(), "{} was created", route.dir.display());
        assert!(!root.exists());
    }

    /// One row of the shared table, split on tabs, with `-` read as empty.
    fn table_rows(kind: &str) -> Vec<Vec<String>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/profile_routes.tsv");
        let text = fs::read_to_string(&path).expect("the shared route table");
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                l.split('\t')
                    .map(|f| {
                        if f == "-" {
                            String::new()
                        } else {
                            f.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|fields| fields[0] == kind)
            .collect()
    }

    /// @test Every alias, route, refusal and the profile-failed status match
    /// the table the C++ tests read too.
    #[test]
    fn routes_match_the_shared_table() {
        for row in table_rows("alias") {
            assert_eq!(canonical_backend(&row[1]), row[2], "alias {}", row[1]);
        }
        let root = Path::new("out");
        let bin = Path::new("./my_test");
        let routes = table_rows("route");
        assert!(routes.len() >= 20, "the table lost its routes");
        for row in routes {
            let tool = canonical_backend(&row[1]);
            let args = (!row[2].is_empty()).then_some(row[2].as_str());
            let dir = format!("out/my_test.{tool}");
            let expected: Vec<String> = row[4]
                .split(' ')
                .map(|a| a.replace("<dir>", &dir).replace("<bin>", "./my_test"))
                .collect();
            let route = route_for(tool, args, bin, Some(root))
                .unwrap_or_else(|e| panic!("{row:?} refused: {e}"))
                .unwrap_or_else(|| panic!("{row:?} has no route"));
            assert_eq!(route.program, row[3], "{row:?}");
            assert_eq!(route.args, expected, "{row:?}");
            assert_eq!(route.dir, PathBuf::from(&dir), "{row:?}");
        }
        for row in table_rows("unwrapped") {
            let route = route_for(canonical_backend(&row[1]), Some("-e x"), bin, None);
            assert!(matches!(route, Ok(None)), "{row:?}: {route:?}");
        }
        for row in table_rows("refuse") {
            let tool = canonical_backend(&row[1]);
            let err = route_for(tool, Some(&row[2]), bin, Some(root))
                .expect_err(&format!("{row:?} was accepted"));
            assert!(matches!(err, Error::InvalidArgs(_)), "{row:?}: {err:?}");
            let text = err.to_string();
            assert!(text.contains(&row[3]), "{row:?}: {text}");
            assert!(
                text.contains(&request_text(tool, Some(&row[2]))),
                "{row:?}: {text}"
            );
        }
        for row in table_rows("escape") {
            let tool = canonical_backend(&row[1]);
            let route = route_for(tool, None, bin, Some(Path::new(&row[2])))
                .unwrap_or_else(|e| panic!("{row:?} refused: {e}"))
                .unwrap_or_else(|| panic!("{row:?} has no route"));
            let at = route.args.iter().position(|a| a == "--log-file");
            let value = at.and_then(|i| route.args.get(i + 1));
            assert_eq!(value, Some(&row[3]), "{row:?}: {:?}", route.args);
            assert_eq!(
                route.dir,
                Path::new(&row[2]).join(format!("my_test.{tool}")),
                "{row:?}: the folder keeps its own name"
            );
        }
        let exits = table_rows("exit");
        assert_eq!(exits.len(), 2);
        assert_eq!(exits[0][1], "profile-failed");
        assert_eq!(exits[0][2], PROFILE_FAILED_EXIT_CODE.to_string());
        assert_eq!(exits[1][1], "tool-findings");
        assert_eq!(exits[1][2], TOOL_FINDINGS_EXIT_CODE.to_string());
    }

    /// @test A report's error count comes from its last summary line, and a
    /// report without one has no count.
    #[test]
    fn sanitizer_error_count_reads_the_summary() {
        let head = "========= COMPUTE-SANITIZER\n";
        for (text, count) in [
            ("========= ERROR SUMMARY: 0 errors\n", Some(0)),
            ("========= ERROR SUMMARY: 1 error\n", Some(1)),
            ("========= ERROR SUMMARY: 259 errors\n", Some(259)),
            ("========= Invalid __global__ write of size 4 bytes\n", None),
            ("========= ERROR SUMMARY: many errors\n", None),
            ("========= ERROR SUMMARY: 3 warnings\n", None),
            ("", None),
        ] {
            assert_eq!(
                sanitizer_error_count(&format!("{head}{text}")),
                count,
                "{text:?}"
            );
        }
    }

    /// @test compute-sanitizer's --log-file value doubles every '%' and
    /// nothing else.
    #[test]
    fn escape_percent_doubles_each_percent() {
        assert_eq!(escape_percent("out/a.b"), "out/a.b");
        assert_eq!(escape_percent("out%p"), "out%%p");
        assert_eq!(escape_percent("%%q{X}%"), "%%%%q{X}%%");
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

    /// @test jemalloc's dumps are found by the wrap's prefix and suffix only,
    /// and a run needs a non-empty one.
    #[test]
    fn jemalloc_dumps_match_the_wrap_prefix() {
        let dir = tempfile_dir("vernier_runner_utst_jemalloc_dumps");
        for (name, text) in [
            ("jeprof.41.0.f.heap", "heap"),
            ("jeprof.41.1.i0.heap", ""),
            ("jeprof.txt", "not a dump"),
            ("other.heap", "not the wrap's"),
        ] {
            fs::write(dir.join(name), text).expect("write");
        }
        let names: Vec<String> = jemalloc_dumps(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["jeprof.41.0.f.heap", "jeprof.41.1.i0.heap"]);
        assert!(require_jemalloc_dumps(&dir, "--profile jemalloc").is_ok());
        fs::remove_file(dir.join("jeprof.41.0.f.heap")).expect("remove");
        let err = require_jemalloc_dumps(&dir, "--profile jemalloc")
            .expect_err("only an empty dump is left");
        assert!(
            err.to_string().starts_with(
                "--profile jemalloc failed: completion: no jemalloc dump (jeprof.*.heap) was \
                 written in"
            ),
            "{err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A fresh, empty directory under the system's temporary directory.
    fn tempfile_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create the test directory");
        dir
    }

    /// @test env_wrap_for is None for tools that are not env-shaped.
    #[test]
    fn env_wrap_for_non_jemalloc_is_none() {
        for tool in ["perf", "massif", "nsight", "ncu", "rocprof", "heaptrack"] {
            assert!(matches!(
                env_wrap_for(tool, Path::new("./b"), None),
                Ok(None)
            ));
        }
    }
}
