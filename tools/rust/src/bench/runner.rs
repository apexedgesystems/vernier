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
    // One request decides the wrap, the benchmark's arguments and the folder.
    let cfg = &effective_request(cfg, "bench run")?;

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

/// The fields of a profile request that `bench run` routes on, as the
/// benchmark spells them after `--`: each takes the word after it, and
/// `--artifact-root` is the benchmark's other name for the output folder.
const ROUTED_FIELDS: [&str; 4] = [
    "--profile",
    "--profile-args",
    "--profile-output-dir",
    "--artifact-root",
];

/// The one profile request @p cfg makes, for `bench run` and `bench doctor`
/// (@p command, for messages) alike: its `--profile`, `--profile-args` and
/// `--profile-output-dir`, each also taken from the arguments after `--`,
/// where the benchmark would read it, and those arguments without them. The
/// wrap, the folder, the benchmark's arguments and the doctor's row all come
/// from this request, and the benchmark receives each field once. A field
/// given twice with different values (to the command and after `--`, or
/// twice after `--`) is refused, as is one after `--` without its value,
/// before anything is created or started; the same value twice is one
/// request (`nsys` is `nsight`). Every other argument after `--`,
/// `--profile-analyze` among them, stays as it is, in its place.
pub fn effective_request(cfg: &RunConfig, command: &str) -> Result<RunConfig, Error> {
    let mut forwarded: Vec<(&str, &str)> = Vec::new();
    let mut rest = Vec::with_capacity(cfg.extra_args.len());
    let mut args = cfg.extra_args.iter();
    while let Some(arg) = args.next() {
        let Some(field) = ROUTED_FIELDS.iter().find(|f| **f == arg) else {
            rest.push(arg.clone());
            continue;
        };
        let Some(value) = args.next() else {
            return Err(Error::InvalidArgs(format!("{field} after -- has no value")));
        };
        let field = if *field == "--artifact-root" {
            "--profile-output-dir"
        } else {
            field
        };
        forwarded.push((field, value));
    }
    // The value of @p field: the command's own and every one after `--`
    // (each marked true), which must agree by @p same.
    let one = |field: &str,
               own: Option<String>,
               same: &dyn Fn(&str, &str) -> bool|
     -> Result<Option<String>, Error> {
        let mut values: Vec<(String, bool)> = own.into_iter().map(|v| (v, false)).collect();
        values.extend(
            forwarded
                .iter()
                .filter(|(f, _)| *f == field)
                .map(|(_, v)| (v.to_string(), true)),
        );
        if values.iter().any(|(v, _)| !same(v, &values[0].0)) {
            let each = values
                .iter()
                .map(|(v, after)| {
                    if *after {
                        format!("'{v}' after --")
                    } else {
                        format!("'{v}' to {command}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::InvalidArgs(format!(
                "{field} is given more than once with different values ({each}); give it once"
            )));
        }
        Ok(values.into_iter().next().map(|(v, _)| v))
    };
    let exact = |a: &str, b: &str| a == b;
    let mut out = cfg.clone();
    out.profile = one("--profile", cfg.profile.clone(), &|a, b| {
        canonical_backend(a) == canonical_backend(b)
    })?;
    out.profile_args = one("--profile-args", cfg.profile_args.clone(), &exact)?;
    out.profile_output_dir = one(
        "--profile-output-dir",
        cfg.profile_output_dir
            .as_ref()
            .map(|d| d.display().to_string()),
        &exact,
    )?
    .map(PathBuf::from);
    out.extra_args = rest;
    Ok(out)
}

/// The profile request as the benchmark reads it: `--profile` by its
/// canonical name, `--profile-args`, `--profile-test-timeout`,
/// `--profile-output-dir`, and `--profile-analyze` unless it is among the
/// arguments forwarded after `--` (which the caller appends after these).
/// `bench run` and `bench doctor` spell a request with this one function,
/// for the request `effective_request` made.
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
/// rocprof. A word the tool does not take, a combination it cannot run, a
/// mode `bench run` does not wrap and an output folder the tool would
/// rewrite (heaptrack's %h and %p) are refused, naming what is accepted.
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
    // The folder as the tools' output options read it (valgrind's,
    // compute-sanitizer's, and the -o of nsys and ncu): each '%' doubled.
    let escaped = escape_percent(&d);
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
                format!("--callgrind-out-file={escaped}/callgrind.out"),
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
            a.push(format!("--massif-out-file={escaped}/massif.out"));
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
            a.push(format!("--log-file={escaped}/memcheck.log"));
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
                format!("--log-file={escaped}/helgrind.log"),
            ],
            &[&["helgrind.log"]],
        ),
        // heaptrack appends the suffix of the compression it was built with.
        // It replaces %h and %p in its -o value with the host name and its
        // process id and has no escape for them, so a folder holding either
        // is refused; any other '%' reaches it as it is.
        "heaptrack" => {
            let held: Vec<&str> = ["%h", "%p"].into_iter().filter(|m| d.contains(m)).collect();
            if !held.is_empty() {
                return refuse(format!(
                    "heaptrack replaces %h (the host name) and %p (its process id) in its -o \
                     value and has no escape for them, so it cannot write into {d}, which holds \
                     {}; choose a --profile-output-dir without %h or %p",
                    held.join(" and ")
                ));
            }
            (
                "heaptrack",
                vec!["-o".into(), format!("{d}/run")],
                &[&["run.zst", "run.gz"]],
            )
        }
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
                    format!("{escaped}/sanitizer.log"),
                ],
                &[&["sanitizer.log"]],
            )
        }
        // nsight's compute mode is ncu's route, into nsight's folder.
        "nsight" if has("compute") || has("ncu") => {
            ("ncu", ncu(&escaped), &[&["kernel_profile.ncu-rep"]])
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
                format!("{escaped}/profile"),
                "-t".into(),
                "cuda,nvtx".into(),
                "--force-overwrite".into(),
                "true".into(),
            ],
            &[&["profile.nsys-rep"]],
        ),
        "ncu" => ("ncu", ncu(&escaped), &[&["kernel_profile.ncu-rep"]]),
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
/// an analysis failure; the profile is kept. The annotator is an owned run
/// (`owned_run`): what it leaves running in its process group is ended before
/// the annotation returns, and SIGINT, SIGTERM or SIGHUP received meanwhile
/// end the group before bench run ends by the same signal.
fn annotate_callgrind(profile: &Path, request: &str) -> Result<(), Error> {
    let kept = format!("; the profile is kept at {}", profile.display());
    let Some(annotator) = find_in_path("callgrind_annotate") else {
        return Err(profile_failure(
            request,
            "analysis",
            format!("callgrind_annotate is not on PATH (it ships with valgrind){kept}"),
        ));
    };
    let mut command = Command::new(&annotator);
    command.arg("--auto=yes").arg(profile);
    let name = annotator.display().to_string();
    let (status, out, err) = match run_owned(&mut command, &name, ANNOTATE_TIMEOUT) {
        Ok(OwnedEnd::Exited(status, out, err)) => (status, out, err),
        Ok(OwnedEnd::TimedOut) => {
            return Err(profile_failure(
                request,
                "analysis",
                format!(
                    "{name} did not finish within {} s and was stopped{kept}",
                    ANNOTATE_TIMEOUT.as_secs()
                ),
            ));
        }
        Err(OwnedError::Start(e)) => {
            return Err(profile_failure(
                request,
                "analysis",
                format!("{name} could not be started: {e}{kept}"),
            ));
        }
        Err(OwnedError::Wait(e)) => return Err(Error::Io(e)),
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

/// How an owned run of a helper program ended, for its caller.
enum OwnedEnd {
    /// It exited: its status, stdout and stderr.
    Exited(ExitStatus, String, String),
    /// Its bound passed first; its group was ended.
    TimedOut,
}

/// Why an owned run gave no ending.
enum OwnedError {
    /// The program could not be started.
    Start(std::io::Error),
    /// It could not be waited for.
    Wait(std::io::Error),
}

/// Run @p command as an owned run (`owned_run`) bounded by @p bound. What it
/// leaves running in its process group is ended, and reported under
/// @p name; a SIGINT, SIGTERM or SIGHUP that bench run receives while it
/// runs ends the group, and then bench run by that signal. A signal whose
/// action was not the default when bench run started is left alone.
fn run_owned(
    command: &mut Command,
    name: &str,
    bound: std::time::Duration,
) -> Result<OwnedEnd, OwnedError> {
    let watch = owned_run::Watch::start();
    let child = owned_run::spawn(command).map_err(OwnedError::Start)?;
    let run = owned_run::finish(child, bound, watch.flag());
    drop(watch);
    let run = run.map_err(OwnedError::Wait)?;
    if run.left_running > 0 {
        eprintln!(
            "[bench] {name} exited and left {} of its own running; bench run ended {}",
            if run.left_running == 1 {
                "1 process".to_string()
            } else {
                format!("{} processes", run.left_running)
            },
            if run.left_running == 1 { "it" } else { "them" }
        );
    }
    match run.ending {
        owned_run::Ending::Exited(status) => Ok(OwnedEnd::Exited(status, run.stdout, run.stderr)),
        owned_run::Ending::TimedOut => Ok(OwnedEnd::TimedOut),
        owned_run::Ending::Interrupted(signal) => owned_run::end_by(signal),
    }
}

/// A helper program bench run owns until it returns. The program runs in a
/// process group of its own, made when it is spawned, so the processes it
/// starts stay in that group after it exits. When the program exits, outruns
/// its bound or bench run is interrupted, the whole group is ended, the run
/// waits (up to `GONE_WAIT`) until no process of the group runs, and the
/// program's output is read to its end; a pipe still held open after that (by
/// a process that left the group, which the run does not own) is read only as
/// far as it has been written, never waited on.
///
/// The invariants the `unsafe` calls below rely on:
/// - The group's id is the program's process id, reserved while the program
///   is unreaped: the group is signalled only before `Child::wait` reaps the
///   program, and the program's exit is seen with `WNOWAIT`, which does not
///   reap it.
/// - The signal handler only stores the signal's number in an atomic.
/// - Only the read ends of the program's own pipes are made non-blocking.
mod owned_run {
    use std::io::{self, Read};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    /// How often the run looks at the program, the signals and the bound.
    const POLL: Duration = Duration::from_millis(20);

    /// How long the run waits for an ended group's processes to be gone.
    const GONE_WAIT: Duration = Duration::from_secs(2);

    /// How the program's run ended.
    pub(super) enum Ending {
        /// The program exited by itself.
        Exited(ExitStatus),
        /// The bound passed first; the group was ended.
        TimedOut,
        /// bench run received this signal; the group was ended.
        Interrupted(i32),
    }

    /// A finished run: how it ended, the program's output, and how many
    /// processes of its group still ran when it exited (ended by the run).
    pub(super) struct Finished {
        pub(super) ending: Ending,
        pub(super) stdout: String,
        pub(super) stderr: String,
        pub(super) left_running: usize,
    }

    /// Start @p command with stdin closed and both outputs piped, in a
    /// process group of its own.
    pub(super) fn spawn(command: &mut Command) -> io::Result<Child> {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        command.spawn()
    }

    /// Wait for @p child (from `spawn`) until it exits, @p bound passes or
    /// @p interrupted holds a signal's number; then end its group, wait for
    /// the group to be gone and collect the output.
    pub(super) fn finish(
        mut child: Child,
        bound: Duration,
        interrupted: &AtomicI32,
    ) -> io::Result<Finished> {
        let stop = Arc::new(AtomicBool::new(false));
        let stdout = read_to_end(child.stdout.take(), &stop);
        let stderr = read_to_end(child.stderr.take(), &stop);
        let started = Instant::now();
        let ending = loop {
            if exited(&mut child) {
                break None;
            }
            let signal = interrupted.load(Ordering::SeqCst);
            if signal != 0 {
                break Some(Ending::Interrupted(signal));
            }
            if started.elapsed() >= bound {
                break Some(Ending::TimedOut);
            }
            std::thread::sleep(POLL);
        };
        let group = child.id();
        let left_running = if ending.is_none() {
            running_in_group(group).len()
        } else {
            0
        };
        end_group(&mut child);
        let status = child.wait();
        let until = Instant::now() + GONE_WAIT;
        while !running_in_group(group).is_empty() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(10));
        }
        stop.store(true, Ordering::SeqCst);
        let stdout = stdout.join().unwrap_or_default();
        let stderr = stderr.join().unwrap_or_default();
        let status = status?;
        // A signal received while the group was being ended still counts.
        let signal = interrupted.load(Ordering::SeqCst);
        let ending = match ending {
            Some(ending) => ending,
            None if signal != 0 => Ending::Interrupted(signal),
            None => Ending::Exited(status),
        };
        Ok(Finished {
            ending,
            stdout,
            stderr,
            left_running,
        })
    }

    /// Read @p pipe to its end on a thread of its own. Once @p stop is set,
    /// the reader also ends when nothing is left to read: a process outside
    /// the group may hold the pipe open.
    fn read_to_end<P>(pipe: Option<P>, stop: &Arc<AtomicBool>) -> JoinHandle<String>
    where
        P: Read + Send + 'static + PipeEnd,
    {
        let stop = Arc::clone(stop);
        std::thread::spawn(move || {
            let Some(mut pipe) = pipe else {
                return String::new();
            };
            let polled = pipe.set_nonblocking();
            let mut bytes = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => bytes.extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) if polled && e.kind() == io::ErrorKind::WouldBlock => {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// A pipe's read end that can stop blocking.
    trait PipeEnd {
        /// Make reads return `WouldBlock` instead of waiting; false when the
        /// end stays blocking.
        fn set_nonblocking(&self) -> bool;
    }

    #[cfg(unix)]
    impl<T: std::os::fd::AsRawFd> PipeEnd for T {
        fn set_nonblocking(&self) -> bool {
            let fd = self.as_raw_fd();
            // SAFETY: fcntl reads and sets the status flags of `fd`, the read
            // end of a pipe this run owns; only O_NONBLOCK is added.
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFL);
                flags >= 0 && libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) == 0
            }
        }
    }

    #[cfg(not(unix))]
    impl<T> PipeEnd for T {
        fn set_nonblocking(&self) -> bool {
            false
        }
    }

    /// Whether @p child has exited, without reaping it.
    #[cfg(unix)]
    fn exited(child: &mut Child) -> bool {
        let pid: libc::id_t = child.id();
        // SAFETY: a zeroed siginfo_t is a valid value of the type; waitid
        // only writes into `info`, owned here, and with WNOWAIT leaves the
        // child for `Child::wait` to reap.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if rc != 0 {
            // EINTR: look again; anything else: let `Child::wait` report it.
            return io::Error::last_os_error().kind() != io::ErrorKind::Interrupted;
        }
        // SAFETY: waitid filled `info`; with WNOHANG its process id stays 0
        // while the child runs.
        unsafe { info.si_pid() != 0 }
    }

    /// Whether @p child has exited (reaped here; `Child::wait` returns the
    /// status it kept).
    #[cfg(not(unix))]
    fn exited(child: &mut Child) -> bool {
        child.try_wait().map_or(true, |status| status.is_some())
    }

    /// End every process of @p child's group, the child included if it
    /// still runs. The child is unreaped here, so the group's id is its own.
    #[cfg(unix)]
    fn end_group(child: &mut Child) {
        if let Ok(group) = libc::pid_t::try_from(child.id()) {
            // SAFETY: killpg only sends a signal, to the group made at spawn,
            // whose id the unreaped child keeps.
            unsafe { libc::killpg(group, libc::SIGKILL) };
        }
    }

    #[cfg(not(unix))]
    fn end_group(child: &mut Child) {
        let _ = child.kill();
    }

    /// The processes of group @p group that still run (zombies excluded),
    /// from /proc; empty where /proc does not list processes.
    fn running_in_group(group: u32) -> Vec<u32> {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
                let stat = std::fs::read_to_string(entry.path().join("stat")).ok()?;
                // After the command name's closing parenthesis: state, parent, group.
                let mut fields = stat.rsplit_once(')')?.1.split_whitespace();
                let state = fields.next()?;
                let pgrp: u32 = fields.nth(1)?.parse().ok()?;
                (pgrp == group && state != "Z" && state != "X").then_some(pid)
            })
            .collect()
    }

    /// The signal a `Watch` recorded; 0 for none.
    static PENDING: AtomicI32 = AtomicI32::new(0);

    #[cfg(unix)]
    extern "C" fn record(signal: libc::c_int) {
        PENDING.store(signal, Ordering::SeqCst);
    }

    /// While it lives, SIGINT, SIGTERM and SIGHUP are recorded for the run to
    /// end its group: the group is not the terminal's foreground group, so a
    /// Ctrl-C reaches bench run alone. A signal whose action is not the
    /// default (ignored, as under nohup, or handled) is left as it is.
    /// Dropping it restores each action it replaced.
    pub(super) struct Watch {
        #[cfg(unix)]
        replaced: Vec<(libc::c_int, libc::sigaction)>,
    }

    impl Watch {
        pub(super) fn start() -> Self {
            PENDING.store(0, Ordering::SeqCst);
            Self {
                #[cfg(unix)]
                replaced: record_signals(),
            }
        }

        /// The flag the recorded signal's number is stored in.
        pub(super) fn flag(&self) -> &'static AtomicI32 {
            &PENDING
        }
    }

    impl Drop for Watch {
        fn drop(&mut self) {
            #[cfg(unix)]
            restore_signals(&self.replaced);
        }
    }

    /// Record SIGINT, SIGTERM and SIGHUP in `PENDING` where their action is
    /// the default; returns each replaced action.
    #[cfg(unix)]
    fn record_signals() -> Vec<(libc::c_int, libc::sigaction)> {
        let mut replaced = Vec::new();
        for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            // SAFETY: zeroed sigaction values are valid values of the type;
            // sigaction only reads `action` and writes `old`, both owned here;
            // the handler installed only stores to an atomic.
            unsafe {
                let mut old: libc::sigaction = std::mem::zeroed();
                if libc::sigaction(signal, std::ptr::null(), &mut old) != 0
                    || old.sa_sigaction != libc::SIG_DFL
                {
                    continue;
                }
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = record as extern "C" fn(libc::c_int) as libc::sighandler_t;
                libc::sigemptyset(&mut action.sa_mask);
                action.sa_flags = libc::SA_RESTART;
                if libc::sigaction(signal, &action, std::ptr::null_mut()) == 0 {
                    replaced.push((signal, old));
                }
            }
        }
        replaced
    }

    /// Put back the actions `record_signals` replaced.
    #[cfg(unix)]
    fn restore_signals(replaced: &[(libc::c_int, libc::sigaction)]) {
        for (signal, old) in replaced {
            // SAFETY: restores the action read for this signal before.
            unsafe { libc::sigaction(*signal, old, std::ptr::null_mut()) };
        }
    }

    /// End bench run by @p signal, after the `Watch` that recorded it is
    /// dropped (its default action restored), so whoever sent it sees the
    /// run ended by it.
    pub(super) fn end_by(signal: i32) -> ! {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        #[cfg(unix)]
        // SAFETY: raise only sends @p signal to this process.
        unsafe {
            libc::raise(signal);
        }
        std::process::exit(128 + signal)
    }
}

/// How long one `nsys stats` summary may take.
const NSYS_STATS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Extract the four `nsys stats` summaries beside the report of a wrapped
/// nsight run, one `<report>.txt` each. The report exists (checked before
/// this runs). An `nsys stats` that fails, or still runs after
/// `NSYS_STATS_TIMEOUT`, is an analysis failure naming the summary, its
/// status and the end of its error output; the report is kept. Each summary
/// is an owned run (`run_owned`), as the callgrind annotation is.
fn extract_nsys_stats(dir: &Path, request: &str) -> Result<(), Error> {
    let rep = dir.join("profile.nsys-rep");
    for report in super::nsight_report::NSYS_SUMMARIES {
        let out_path = dir.join(format!("{report}.txt"));
        let cannot_write = |e: std::io::Error| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot write {}: {e}", out_path.display()),
            ))
        };
        fs::File::create(&out_path).map_err(cannot_write)?;
        let mut command = Command::new("nsys");
        command
            .args(["stats", "--force-export=true", "--report", report])
            .arg(&rep);
        let name = format!("nsys stats --report {report}");
        let (status, out, err) = match run_owned(&mut command, &name, NSYS_STATS_TIMEOUT) {
            Ok(OwnedEnd::Exited(status, out, err)) => (status, out, err),
            Ok(OwnedEnd::TimedOut) => {
                return Err(profile_failure(
                    request,
                    "analysis",
                    format!(
                        "{name} did not finish within {} s and was stopped; the report is \
                         kept at {}",
                        NSYS_STATS_TIMEOUT.as_secs(),
                        rep.display()
                    ),
                ));
            }
            Err(OwnedError::Start(e)) => {
                return Err(Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot start nsys stats: {e}"),
                )));
            }
            Err(OwnedError::Wait(e)) => return Err(Error::Io(e)),
        };
        fs::write(&out_path, out).map_err(cannot_write)?;
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
                    "{name} {}{}{}; the report is kept at {}",
                    describe_status(status),
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

/// @p path as an output option of valgrind (`--log-file=`,
/// `--massif-out-file=`, `--callgrind-out-file=`), compute-sanitizer
/// (`--log-file`), nsys or ncu (`-o`) must spell it: each tool reads a `%`
/// in the value as the start of a macro (`%p`, `%q{VAR}`, ...) and `%%` as
/// one `%`, so each `%` of the path is doubled.
fn escape_percent(path: &str) -> String {
    path.replace('%', "%%")
}

/// The error count of a compute-sanitizer report, from its last summary line:
/// `ERROR SUMMARY: N error(s)` and nothing after it (memcheck, synccheck,
/// initcheck), or racecheck's `RACECHECK SUMMARY: H hazard(s) displayed (E
/// error(s), W warning(s))`, whose errors count and whose warnings do not.
/// The line the print limit adds after the total, `ERROR SUMMARY: N errors
/// were not printed...`, counts what was left out and is not a total. `None`
/// when the report has no summary in that form (a report cut short, or a
/// tool that stopped before summarizing): nothing is counted from a line read
/// any other way.
fn sanitizer_error_count(report: &str) -> Option<u64> {
    report.lines().rev().find_map(|line| {
        if let Some((_, rest)) = line.split_once("RACECHECK SUMMARY:") {
            return racecheck_errors(rest);
        }
        whole_count(line.split_once("ERROR SUMMARY:")?.1, "error")
    })
}

/// N of @p text when it is "N <noun>" or "N <noun>s" and nothing else
/// (spaces around it aside).
fn whole_count(text: &str, noun: &str) -> Option<u64> {
    let count = leading_count(text, noun)?;
    let rest = text
        .trim_start()
        .trim_start_matches(|c: char| c.is_ascii_digit());
    let rest = rest.trim_start().strip_prefix(noun)?;
    matches!(rest.trim_end(), "" | "s").then_some(count)
}

/// N of "N <noun>" or "N <noun>s" at the start of @p text (after spaces).
fn leading_count(text: &str, noun: &str) -> Option<u64> {
    let text = text.trim_start();
    let digits = text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let count = text[..digits].parse().ok()?;
    text[digits..]
        .trim_start()
        .starts_with(noun)
        .then_some(count)
}

/// E of racecheck's summary after its label, "H hazard(s) displayed (E
/// error(s), W warning(s))"; `None` for any other form.
fn racecheck_errors(summary: &str) -> Option<u64> {
    let (hazards, rest) = summary.split_once('(')?;
    leading_count(hazards, "hazard")?;
    let (counts, _) = rest.split_once(')')?;
    let mut parts = counts.split(',');
    let errors = leading_count(parts.next()?, "error")?;
    leading_count(parts.next()?, "warning")?;
    parts.next().is_none().then_some(errors)
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

    /// A request with @p profile and @p args given to the command and
    /// @p extra after `--`.
    fn request(profile: Option<&str>, args: Option<&str>, extra: &[&str]) -> RunConfig {
        RunConfig {
            binary: PathBuf::from("./my_test"),
            profile: profile.map(str::to_string),
            profile_args: args.map(str::to_string),
            extra_args: extra.iter().map(|a| a.to_string()).collect(),
            ..Default::default()
        }
    }

    /// @test A routed field given only after `--` is the request's: the mode,
    /// the profile itself and the output folder (by either of its names) are
    /// taken in and removed from the arguments after `--`, whose other words,
    /// --profile-analyze among them, stay in their order.
    #[test]
    fn effective_request_takes_in_the_fields_after_the_separator() {
        let r = effective_request(
            &request(
                Some("helgrind"),
                None,
                &[
                    "--gtest_filter=A.*",
                    "--profile-args",
                    "drd",
                    "--profile-analyze",
                ],
            ),
            "bench run",
        )
        .expect("one request");
        assert_eq!(r.profile.as_deref(), Some("helgrind"));
        assert_eq!(r.profile_args.as_deref(), Some("drd"));
        assert_eq!(r.extra_args, ["--gtest_filter=A.*", "--profile-analyze"]);
        let r = effective_request(
            &request(
                None,
                None,
                &[
                    "--profile",
                    "massif",
                    "--artifact-root",
                    "out",
                    "--profile-test-timeout",
                    "0",
                ],
            ),
            "bench run",
        )
        .expect("one request");
        assert_eq!(r.profile.as_deref(), Some("massif"));
        assert_eq!(r.profile_output_dir, Some(PathBuf::from("out")));
        assert_eq!(r.extra_args, ["--profile-test-timeout", "0"]);
        let plain = request(Some("perf"), Some("-e cycles"), &["--cycles", "5"]);
        let r = effective_request(&plain, "bench run").expect("one request");
        assert_eq!(
            (r.profile, r.profile_args, r.extra_args),
            (plain.profile, plain.profile_args, plain.extra_args)
        );
    }

    /// @test A routed field given twice with different values is refused,
    /// naming both and where each was given; the same value twice, nsys for
    /// nsight included, is one request; a field after `--` without its value
    /// is refused.
    #[test]
    fn effective_request_refuses_a_field_given_twice() {
        for (cfg, expected) in [
            (
                request(Some("massif"), Some("pages"), &["--profile-args", "stacks"]),
                "--profile-args is given more than once with different values ('pages' to \
                 bench run, 'stacks' after --); give it once",
            ),
            (
                request(Some("perf"), None, &["--profile", "gperf"]),
                "--profile is given more than once with different values ('perf' to bench \
                 run, 'gperf' after --); give it once",
            ),
            (
                request(
                    None,
                    None,
                    &["--profile-output-dir", "a", "--artifact-root", "b"],
                ),
                "--profile-output-dir is given more than once with different values ('a' \
                 after --, 'b' after --); give it once",
            ),
            (
                request(Some("massif"), None, &["--profile-args"]),
                "--profile-args after -- has no value",
            ),
        ] {
            let err = effective_request(&cfg, "bench run").expect_err("refused");
            assert!(matches!(err, Error::InvalidArgs(_)), "{err:?}");
            assert_eq!(err.to_string(), format!("invalid arguments: {expected}"));
        }
        let r = effective_request(
            &request(
                Some("nsys"),
                Some("compute"),
                &["--profile", "nsight", "--profile-args", "compute"],
            ),
            "bench run",
        )
        .expect("the same request twice");
        assert_eq!(r.profile.as_deref(), Some("nsys"));
        assert_eq!(r.profile_args.as_deref(), Some("compute"));
        assert!(r.extra_args.is_empty());
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
        let escapes = table_rows("escape");
        assert!(escapes.len() >= 10, "the table lost its escape rows");
        for row in escapes {
            let tool = canonical_backend(&row[1]);
            let args = (!row[2].is_empty()).then_some(row[2].as_str());
            let route = route_for(tool, args, bin, Some(Path::new(&row[3])))
                .unwrap_or_else(|e| panic!("{row:?} refused: {e}"))
                .unwrap_or_else(|| panic!("{row:?} has no route"));
            // The option's value: the next argument, or after its '='.
            let option = &row[4];
            let joined = format!("{option}=");
            let value = route.args.iter().enumerate().find_map(|(i, a)| {
                if a == option {
                    route.args.get(i + 1).cloned()
                } else {
                    a.strip_prefix(&joined).map(str::to_string)
                }
            });
            assert_eq!(value.as_ref(), Some(&row[5]), "{row:?}: {:?}", route.args);
            assert_eq!(
                route.dir,
                Path::new(&row[3]).join(format!("my_test.{tool}")),
                "{row:?}: the folder keeps its own name"
            );
        }
        let folder_refusals = table_rows("refuse-folder");
        assert!(
            folder_refusals.len() >= 2,
            "the table lost its folder refusals"
        );
        for row in folder_refusals {
            let tool = canonical_backend(&row[1]);
            let err = route_for(tool, None, bin, Some(Path::new(&row[2])))
                .expect_err(&format!("{row:?} was accepted"));
            assert!(matches!(err, Error::InvalidArgs(_)), "{row:?}: {err:?}");
            let text = err.to_string();
            assert!(text.contains(&row[3]), "{row:?}: {text}");
            assert!(
                text.contains(&format!("{}/my_test.{tool}", row[2])),
                "{row:?}: the refusal names the folder: {text}"
            );
        }
        let exits = table_rows("exit");
        assert_eq!(exits.len(), 2);
        assert_eq!(exits[0][1], "profile-failed");
        assert_eq!(exits[0][2], PROFILE_FAILED_EXIT_CODE.to_string());
        assert_eq!(exits[1][1], "tool-findings");
        assert_eq!(exits[1][2], TOOL_FINDINGS_EXIT_CODE.to_string());
    }

    /// @test A report's error count comes from its last summary line, in
    /// memcheck's, synccheck's and initcheck's form or in racecheck's (whose
    /// warnings do not count), the print limit's line after it not counted,
    /// and a report without one, or with a summary in any other form, has no
    /// count.
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
            // The print limit's line after the total (the report case of the
            // Compute Sanitizer walkthrough's check), and that line alone.
            (
                "========= ERROR SUMMARY: 259 errors\n========= ERROR SUMMARY: 159 errors \
                 were not printed. Use --print-limit option to adjust the number of printed \
                 errors\n",
                Some(259),
            ),
            (
                "========= ERROR SUMMARY: 159 errors were not printed. Use --print-limit \
                 option to adjust the number of printed errors\n",
                None,
            ),
            ("========= ERROR SUMMARY: 3 errors (and more)\n", None),
            ("========= ERROR SUMMARY: 3 errorsx\n", None),
            ("========= ERROR SUMMARY: 2 errors   \n", Some(2)),
            (
                "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
                Some(0),
            ),
            (
                "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n",
                Some(1),
            ),
            (
                "========= RACECHECK SUMMARY: 3 hazards displayed (0 errors, 3 warnings)\n",
                Some(0),
            ),
            ("========= RACECHECK SUMMARY: 2 hazards displayed\n", None),
            (
                "========= RACECHECK SUMMARY: 2 hazards displayed (2 errors)\n",
                None,
            ),
            (
                "========= RACECHECK SUMMARY: some hazards displayed (2 errors, 0 warnings)\n",
                None,
            ),
            (
                "========= RACECHECK SUMMARY: 2 hazards displayed (2 warnings, 0 errors)\n",
                None,
            ),
            ("", None),
        ] {
            assert_eq!(
                sanitizer_error_count(&format!("{head}{text}")),
                count,
                "{text:?}"
            );
        }
    }

    /// @test An output option's value doubles every '%' and nothing else.
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

    /* ----------------------------- Owned runs ----------------------------- */

    /// Whether process @p pid still runs: listed in /proc, not a zombie.
    /// Read only: these tests signal only the process a test says it owns.
    #[cfg(unix)]
    fn still_runs(pid: u32) -> bool {
        fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| {
                let state = stat.rsplit_once(')')?.1.split_whitespace().next()?;
                Some(state != "Z" && state != "X")
            })
            .unwrap_or(false)
    }

    /// The process id a script wrote to @p path.
    #[cfg(unix)]
    fn written_pid(path: &Path) -> u32 {
        fs::read_to_string(path)
            .expect("the script wrote its process id")
            .trim()
            .parse()
            .expect("a process id")
    }

    /// `sh -c <script>` as an owned run with @p bound and @p interrupted;
    /// returns the run and how long it took.
    #[cfg(unix)]
    fn owned_sh(
        script: &str,
        bound: std::time::Duration,
        interrupted: &std::sync::atomic::AtomicI32,
    ) -> (owned_run::Finished, std::time::Duration) {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", script]);
        let started = std::time::Instant::now();
        let child = owned_run::spawn(&mut command).expect("start sh");
        let run = owned_run::finish(child, bound, interrupted).expect("finish the run");
        (run, started.elapsed())
    }

    /// @test A program that starts a process holding its output and exits:
    /// the run returns at once with the program's status and output, and the
    /// process left behind is ended by the run (the test only looks at it).
    #[test]
    #[cfg(unix)]
    fn owned_run_ends_what_the_program_leaves() {
        let dir = tempfile_dir("owned_run_leaves");
        let left = dir.join("left.pid");
        let (run, took) = owned_sh(
            &format!(
                "echo out; echo err >&2; /bin/sleep 60 & echo $! > '{}'; exit 3",
                left.display()
            ),
            std::time::Duration::from_secs(30),
            &std::sync::atomic::AtomicI32::new(0),
        );
        let left = written_pid(&left);
        let left_runs = still_runs(left);
        assert!(
            took < std::time::Duration::from_secs(5),
            "the run waited on the process left behind: {took:?}"
        );
        assert!(!left_runs, "the process left behind, {left}, still runs");
        assert_eq!(run.left_running, 1);
        assert!(
            matches!(run.ending, owned_run::Ending::Exited(status) if status.code() == Some(3))
        );
        assert_eq!(run.stdout, "out\n");
        assert_eq!(run.stderr, "err\n");
        let _ = fs::remove_dir_all(&dir);
    }

    /// @test A program still running at the bound: the run ends it and the
    /// process it started, and says the bound passed.
    #[test]
    #[cfg(unix)]
    fn owned_run_bound_ends_the_group() {
        let dir = tempfile_dir("owned_run_bound");
        let (started, program) = (dir.join("started.pid"), dir.join("program.pid"));
        let (run, took) = owned_sh(
            &format!(
                "/bin/sleep 60 & echo $! > '{}'; echo $$ > '{}'; wait",
                started.display(),
                program.display()
            ),
            std::time::Duration::from_secs(1),
            &std::sync::atomic::AtomicI32::new(0),
        );
        let (started, program) = (written_pid(&started), written_pid(&program));
        let (started_runs, program_runs) = (still_runs(started), still_runs(program));
        assert!(matches!(run.ending, owned_run::Ending::TimedOut));
        assert!(
            took >= std::time::Duration::from_secs(1) && took < std::time::Duration::from_secs(5),
            "{took:?}"
        );
        assert!(
            !started_runs && !program_runs,
            "still running: the program {program} {program_runs}, its process {started} \
             {started_runs}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// @test A signal recorded while the program runs: the run ends the
    /// group at once and reports the signal.
    #[test]
    #[cfg(unix)]
    fn owned_run_interrupted_ends_the_group() {
        let dir = tempfile_dir("owned_run_interrupted");
        let (started, program) = (dir.join("started.pid"), dir.join("program.pid"));
        let flag = std::sync::atomic::AtomicI32::new(0);
        let (run, took) = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(300));
                flag.store(2, std::sync::atomic::Ordering::SeqCst);
            });
            owned_sh(
                &format!(
                    "/bin/sleep 60 & echo $! > '{}'; echo $$ > '{}'; wait",
                    started.display(),
                    program.display()
                ),
                std::time::Duration::from_secs(30),
                &flag,
            )
        });
        let (started, program) = (written_pid(&started), written_pid(&program));
        let (started_runs, program_runs) = (still_runs(started), still_runs(program));
        assert!(matches!(run.ending, owned_run::Ending::Interrupted(2)));
        assert!(took < std::time::Duration::from_secs(5), "{took:?}");
        assert!(
            !started_runs && !program_runs,
            "still running: the program {program} {program_runs}, its process {started} \
             {started_runs}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// @test A process that leaves the program's group (setsid) is not
    /// owned: the run neither ends it nor waits on the output it holds open,
    /// and keeps what the program wrote.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_does_not_wait_on_a_process_outside_its_group() {
        let dir = tempfile_dir("owned_run_outside");
        let outside = dir.join("outside.pid");
        let (run, took) = owned_sh(
            &format!(
                "echo out; /usr/bin/setsid /bin/sleep 60 & echo $! > '{}'; exit 0",
                outside.display()
            ),
            std::time::Duration::from_secs(30),
            &std::sync::atomic::AtomicI32::new(0),
        );
        let outside = written_pid(&outside);
        let outside_runs = still_runs(outside);
        // This test started that process, through the script, and ends it,
        // with SIGKILL: the process keeps an ignored SIGTERM it inherited.
        let _ = Command::new("/bin/sh")
            .args(["-c", &format!("kill -KILL {outside}")])
            .status();
        assert!(
            took < std::time::Duration::from_secs(5),
            "the run waited on a pipe held outside its group: {took:?}"
        );
        assert!(outside_runs, "a process outside the group was ended");
        assert_eq!(run.left_running, 0);
        assert_eq!(run.stdout, "out\n");
        let _ = fs::remove_dir_all(&dir);
    }

    /// @test Output larger than a pipe holds is read whole from both pipes.
    #[test]
    #[cfg(unix)]
    fn owned_run_reads_all_output() {
        let (run, _) = owned_sh(
            "i=0; while [ $i -lt 5000 ]; do \
             echo 0123456789012345678901234567890123456789; echo e >&2; i=$((i+1)); done",
            std::time::Duration::from_secs(30),
            &std::sync::atomic::AtomicI32::new(0),
        );
        assert!(matches!(run.ending, owned_run::Ending::Exited(status) if status.success()));
        assert_eq!(run.stdout.len(), 5000 * 41);
        assert_eq!(run.stderr.len(), 5000 * 2);
        assert_eq!(run.left_running, 0);
    }
}
