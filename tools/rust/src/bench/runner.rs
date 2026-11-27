//! Benchmark binary execution with optional CPU pinning and profiling.
//!
//! A request is read by its canonical name (`nsys` is `nsight`), which is also
//! the `--profile` the benchmark receives. When it names a wrap-externally
//! backend (valgrind tools, heaptrack, compute-sanitizer, nsys, ncu), the
//! runner starts the benchmark under the route that request selects: the
//! tool, and the mode its `--profile-args` words name (for Nsight Compute,
//! also the set, sections and metrics of Vernier's supported subset of ncu's
//! options), or it refuses a word the tool does not take before anything is
//! created or started. Wrapped
//! children get `VERNIER_EXTERNAL_WRAP=<tool>` so in-process backends stay
//! passive instead of re-attaching or printing manual-wrap hints; for
//! nsight's nsys route the runner also extracts the canonical `nsys stats`
//! reports after the run (the .nsys-rep only exists once the wrapped process
//! exits), for an ncu route that names metrics it reads the report back and
//! names each metric the report holds nothing for (ncu itself says nothing of
//! a metric the GPU does not report), and for compute-sanitizer it reads the
//! tool's report to tell the errors the tool found from the benchmark's own
//! status. Before a heaptrack run it asks heaptrack whether it passes a
//! program's status through (`heaptrack_drops_status`); when it does not,
//! heaptrack's script runs under bash, whose trap reports the status of the
//! program the script starts (`ShellStatus`), once three runs show that it
//! does, and the run takes the benchmark's status from that report. jemalloc
//! is wrapped through the environment instead (`env_wrap_for`): the library
//! preloaded and its profiling options set, refused before anything starts
//! when the loader cannot preload it, and its exit dump checked after the
//! run.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use super::workflow::canonical_backend;
use super::{
    find_in_path, lookup_in_path, BenchmarkExit, Error, InPath, PrivateDir, ProfileFailure,
    SanitizerFailure,
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
    /// `--profile-frequency`: gperf's CPU sampling rate as given, checked by
    /// `startup_env` against the one grammar both languages share.
    pub profile_frequency: Option<String>,
    pub profile_test_timeout: Option<u32>,
    pub profile_output_dir: Option<PathBuf>,
    /// The project's default output folder (`.bench.yaml`'s
    /// `profile_output_dir`): taken when neither the command nor the
    /// arguments after `--` give one (`effective_request`).
    pub default_profile_output_dir: Option<PathBuf>,
    /// `--profile-analyze`: the requested profile's analysis (forwarded to
    /// the binary; for callgrind, run by the runner after the exit).
    pub profile_analyze: bool,
    pub taskset: Option<String>,
    /// The project's default test filter (`.bench.yaml`'s `gtest_filter`):
    /// given to the benchmark as `--gtest_filter=<it>`, before the arguments
    /// after `--`, unless one of those starts with `--gtest_filter=`.
    pub gtest_filter: Option<String>,
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
    // What the benchmark reads only as it starts (gperf's sampling rate):
    // decided before anything is created or started, never a wrap.
    let startup = startup_env(cfg)?;

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
    args.extend(configured_filter(cfg));
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

    // heaptrack's exit status stands for the benchmark's (`benchmark_exit`)
    // unless this heaptrack drops it: then heaptrack's script runs under
    // bash, whose trap reports the status of the program the script starts
    // (`ShellStatus`), and the run takes the benchmark's status from there.
    let shell_status = match route {
        Some(ref r) if r.program == "heaptrack" && heaptrack_drops_status()? => {
            let shell = ShellStatus::confirmed()?;
            println!(
                "[bench] heaptrack does not pass the benchmark's exit status through \
                 ({HEAPTRACK_PROBE} exited 0): it runs under bash, whose trap takes the \
                 benchmark's status from the shell that waits on it (checked: {})",
                checked_statuses()
            );
            Some(shell)
        }
        _ => None,
    };

    // The route's folder exists and holds none of the files this run's wrap
    // will write, so what is there after the run is this run's. The
    // benchmark runs under its route when it has one, directly otherwise;
    // taskset, if requested, layers on the outside of either.
    if let Some(ref r) = route {
        prepare_folder(&r.dir, &r.stale_names())?;
    }
    let wrap = route.as_ref().map(|r| match shell_status {
        Some(ref shell) => shell.wrap(&r.args),
        None => (r.program.to_string(), r.args.clone()),
    });

    let launch = launch_command(&cfg.binary, &cfg.taskset, &wrap);
    let mut cmd = Command::new(&launch[0]);
    cmd.args(&launch[1..])
        .args(&args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    // Env-shaped wraps (jemalloc) inject via the environment instead of argv.
    let env_wrap = match tool {
        Some(t) => env_wrap_for(t, &cfg.binary, cfg.profile_output_dir.as_deref())?,
        None => None,
    };
    // The benchmark as a process's memory map names it, for the completion
    // check of an environment wrap's dump.
    let binary_path = fs::canonicalize(&cfg.binary).unwrap_or_else(|_| cfg.binary.clone());
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
    let shell_env = shell_status
        .as_ref()
        .map(|shell| shell.env_for(&shell.report()))
        .unwrap_or_default();
    for (k, v) in shell_env.iter().chain(&startup) {
        cmd.env(k, v);
    }

    let env_prefix = shell_env
        .iter()
        .chain(env_wrap.iter().flatten())
        .chain(startup.iter())
        .map(|(k, v)| format!("{k}={v} "))
        .collect::<String>();
    println!("Running: {}{}", env_prefix, format_command(&launch, &args));

    // The process started here is the benchmark (taskset execs it): jemalloc
    // names its exit dump after this process id.
    let mut child = cmd.spawn()?;
    let pid = child.id();
    let status = child.wait()?;

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
    match (&shell_status, &route) {
        (Some(shell), Some(r)) => shell.benchmark_end(
            status,
            &request_text(&r.tool, cfg.profile_args.as_deref()),
            &kept_output(&r.tool, &r.dir, r.outputs),
        )?,
        _ => {
            if let Some(end) = benchmark_exit(status) {
                return Err(Error::Benchmark(end));
            }
        }
    }

    // A wrap writes its output when the process it started exits: this run's
    // output is checked here, after that exit, and only then analysed.
    if let Some(ref r) = route {
        let request = request_text(&r.tool, cfg.profile_args.as_deref());
        for alternatives in r.outputs {
            let output = require_output(&r.tool, &request, &r.dir, alternatives)?;
            if !r.metrics.is_empty() {
                name_missing_metrics(&output, &r.metrics);
            }
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
            let dump = require_jemalloc_dump(
                &wrap_artifact_dir(tool, &cfg.binary, cfg.profile_output_dir.as_deref()),
                &request_text(tool, cfg.profile_args.as_deref()),
                pid,
                &binary_path,
            )?;
            for line in jeprof_hint(&cfg.binary, &dump, lookup_in_path) {
                println!("{line}");
            }
        }
    }

    Ok(cfg.csv.clone())
}

/// Refuse `bench run --analyze` for a run that is a profiling capture under
/// Nsight Compute (ncu), before anything is created or started: the
/// benchmark leaves the timing cells empty under ncu, so there is no timing
/// to summarise. The route is the one `run_benchmark` takes for
/// the request (`effective_request`, then `route_for`); a request it refuses
/// is left for it to refuse.
pub fn refuse_analysis_of_capture(cfg: &RunConfig) -> Result<(), Error> {
    let Ok(cfg) = effective_request(cfg, "bench run") else {
        return Ok(());
    };
    let Some(tool) = cfg.profile.as_deref().map(canonical_backend) else {
        return Ok(());
    };
    let route = route_for(
        tool,
        cfg.profile_args.as_deref(),
        &cfg.binary,
        cfg.profile_output_dir.as_deref(),
    );
    match route {
        Ok(Some(route)) if route.program == "ncu" => Err(Error::InvalidArgs(format!(
            "--analyze summarises the run's timing, and {} runs the benchmark under Nsight \
             Compute (ncu), a profiling capture that leaves the timing cells empty; drop \
             --analyze, or time the benchmark in a run without ncu",
            request_text(tool, cfg.profile_args.as_deref())
        ))),
        _ => Ok(()),
    }
}

/// The fields of a profile request that `bench run` routes on, as the
/// benchmark spells them after `--`: each takes the word after it, and
/// `--artifact-root` is the benchmark's other name for the output folder.
const ROUTED_FIELDS: [&str; 5] = [
    "--profile",
    "--profile-args",
    "--profile-frequency",
    "--profile-output-dir",
    "--artifact-root",
];

/// The one profile request @p cfg makes, for `bench run` and `bench doctor`
/// (@p command, for messages) alike: its `--profile`, `--profile-args`,
/// `--profile-frequency` and `--profile-output-dir`, each also taken from
/// the arguments after `--`, where the benchmark would read it, and those
/// arguments without them. The wrap, the folder, the start-up setting, the
/// benchmark's arguments and the doctor's row all come from this request,
/// and the benchmark receives each field once. A field given twice with
/// different values (to the command and after `--`, or twice after `--`) is
/// refused, as is one after `--` without its value, and a rate written
/// `--profile-frequency=N` after `--`, which the benchmark does not read,
/// before anything is created or started; the same value twice is one
/// request (`nsys` is `nsight`, `0250` is 250). The project's folder
/// (`default_profile_output_dir`) is taken when neither the command nor the
/// arguments after `--` give one. Every other argument after `--`,
/// `--profile-analyze` among them, stays as it is, in its place.
pub fn effective_request(cfg: &RunConfig, command: &str) -> Result<RunConfig, Error> {
    let mut forwarded: Vec<(&str, &str)> = Vec::new();
    let mut rest = Vec::with_capacity(cfg.extra_args.len());
    let mut args = cfg.extra_args.iter();
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--profile-frequency=") {
            return Err(Error::InvalidArgs(format!(
                "{arg} after --: give it as --profile-frequency {value}, which the benchmark \
                 reads"
            )));
        }
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
    // Two spellings of one rate are one request; text that is no rate is
    // compared as written and refused by startup_env.
    out.profile_frequency = one(
        "--profile-frequency",
        cfg.profile_frequency.clone(),
        &|a, b| {
            a == b
                || parse_profile_frequency(a)
                    .is_some_and(|hz| Some(hz) == parse_profile_frequency(b))
        },
    )?;
    // The project's folder is a default: one given to the command, or
    // after `--`, takes precedence over it, and only those can conflict.
    out.profile_output_dir = one(
        "--profile-output-dir",
        cfg.profile_output_dir
            .as_ref()
            .map(|d| d.display().to_string()),
        &exact,
    )?
    .map(PathBuf::from)
    .or_else(|| cfg.default_profile_output_dir.clone());
    out.extra_args = rest;
    Ok(out)
}

/// The profile request as the benchmark reads it: `--profile` by its
/// canonical name, `--profile-args`, `--profile-frequency` by its canonical
/// value, `--profile-test-timeout`, `--profile-output-dir`, and
/// `--profile-analyze` unless it is among the arguments forwarded after `--`
/// (which the caller appends after these). `bench run` and `bench doctor`
/// spell a request with this one function, for the request
/// `effective_request` made.
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
    if let Some(ref hz) = cfg.profile_frequency {
        args.push("--profile-frequency".to_string());
        args.push(parse_profile_frequency(hz).map_or_else(|| hz.clone(), |v| v.to_string()));
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

/// The one spelling of an argument GoogleTest reads its test filter from,
/// up to the filter.
const GTEST_FILTER_FLAG: &str = "--gtest_filter=";

/// What the benchmark is given for the project's default test filter
/// (`RunConfig::gtest_filter`): `--gtest_filter=<it>`, unless an argument
/// after `--` starts with `--gtest_filter=`, which is then the only filter
/// the benchmark gets; nothing without a default. A filter the benchmark is
/// given takes precedence over `GTEST_FILTER` in its environment, which
/// GoogleTest reads only when no flag gives one.
fn configured_filter(cfg: &RunConfig) -> Option<String> {
    let filter = cfg.gtest_filter.as_deref()?;
    let given = cfg
        .extra_args
        .iter()
        .any(|arg| arg.starts_with(GTEST_FILTER_FLAG));
    (!given).then(|| format!("{GTEST_FILTER_FLAG}{filter}"))
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
/// something, print which, with its size, and return its path.
fn require_output(
    tool: &str,
    request: &str,
    dir: &Path,
    alternatives: &[&str],
) -> Result<PathBuf, Error> {
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
    Ok(path)
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

/// The process id and kind a jemalloc dump's name carries,
/// `jeprof.<pid>.<n>.<kind>.heap`: `f` for the dump written at exit.
fn jemalloc_dump_owner(dump: &Path) -> Option<(u32, String)> {
    let name = dump.file_name()?.to_str()?;
    let middle = name.strip_prefix("jeprof.")?.strip_suffix(".heap")?;
    let mut parts = middle.split('.');
    let pid = parts.next()?.parse().ok()?;
    let _sequence = parts.next()?;
    Some((pid, parts.next()?.to_string()))
}

/// True when a heap profile's text has its totals line,
/// `t*: <a>: <b> [<c>: <d>]`, four integers. Only its presence is checked:
/// jemalloc writes these totals in an encoding of its own for `jeprof`
/// (with its default `prof_unbias` they are neither the samples taken nor
/// their bytes), and `jeprof` estimates the bytes from each stack's record,
/// not from this line.
fn has_totals_line(text: &str) -> bool {
    let Some(line) = text.lines().find_map(|l| l.trim().strip_prefix("t*:")) else {
        return false;
    };
    let numbers: Result<Vec<u64>, _> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter(|n| !n.is_empty())
        .map(str::parse::<u64>)
        .collect();
    numbers.is_ok_and(|n| n.len() == 4)
}

/// True when the `MAPPED_LIBRARIES:` section of a heap profile's text maps
/// @p binary: a line whose path, from its first '/', is the binary's.
fn dump_maps(text: &str, binary: &Path) -> bool {
    text.lines()
        .skip_while(|l| !l.starts_with("MAPPED_LIBRARIES:"))
        .filter_map(|l| l.find('/').map(|at| &l[at..]))
        .any(|path| Path::new(path.trim_end()) == binary)
}

/// The dump jemalloc writes when the benchmark exits,
/// `jeprof.<pid>.<n>.f.heap` for the process `bench run` started (@p pid),
/// checked after that exit: present, not empty, a heap profile (`heap_v2/`)
/// mapping @p binary (the benchmark, as a memory map names it), with its
/// totals line (all zero for a run that sampled nothing, which is a valid
/// run's). Every other dump in the folder, of a process the benchmark started
/// under the same preload for one, is named and never taken for the
/// benchmark's. Prints the dump and its size, and returns it.
fn require_jemalloc_dump(
    dir: &Path,
    request: &str,
    pid: u32,
    binary: &Path,
) -> Result<PathBuf, Error> {
    let fail = |message: String| Err(profile_failure(request, "completion", message));
    let (own, others): (Vec<PathBuf>, Vec<PathBuf>) =
        jemalloc_dumps(dir).into_iter().partition(|dump| {
            jemalloc_dump_owner(dump).is_some_and(|(owner, kind)| owner == pid && kind == "f")
        });
    for other in &others {
        println!(
            "[bench] jemalloc: {} is not the benchmark's exit dump (the benchmark is process \
             {pid}); it is not this run's profile",
            other.display()
        );
    }
    let Some(dump) = own.into_iter().next() else {
        let held = if others.is_empty() {
            String::new()
        } else {
            format!(
                "; the folder holds only other dumps: {}",
                others
                    .iter()
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        return fail(format!(
            "jemalloc wrote no exit dump of the benchmark (process {pid}), \
             jeprof.{pid}.<n>.f.heap, in {}{held}",
            dir.display()
        ));
    };
    let bytes = fs::read(&dump).map_err(|e| {
        profile_failure(
            request,
            "completion",
            format!("{} cannot be read: {e}", dump.display()),
        )
    })?;
    if bytes.is_empty() {
        return fail(format!("{} is empty", dump.display()));
    }
    let text = String::from_utf8_lossy(&bytes);
    if !text.starts_with("heap_v2/") {
        let head: String = text.lines().next().unwrap_or("").chars().take(40).collect();
        return fail(format!(
            "{} is not a heap profile: it starts '{head}'",
            dump.display()
        ));
    }
    if !dump_maps(&text, binary) {
        return fail(format!(
            "{} does not map the benchmark, {}: it is not this run's profile",
            dump.display(),
            binary.display()
        ));
    }
    if !has_totals_line(&text) {
        return fail(format!(
            "{} has no totals line (t*:); it is not a whole heap profile",
            dump.display()
        ));
    }
    println!(
        "[bench] jemalloc wrote {} ({} bytes) when the benchmark exited",
        dump.display(),
        bytes.len()
    );
    Ok(dump)
}

/// After a jemalloc capture, what bench run prints: the command that reads
/// its dump, and, when `jeprof` is not on PATH (@p lookup), that only reading
/// needs it.
fn jeprof_hint(binary: &Path, dump: &Path, lookup: impl Fn(&str) -> InPath) -> Vec<String> {
    let mut lines = vec![format!(
        "[bench] Read it: jeprof --text --alloc_space {} {}",
        binary.display(),
        dump.display()
    )];
    if !matches!(lookup("jeprof"), InPath::Executable(_)) {
        lines.push(
            "[bench] jeprof is not on PATH: reading the dump needs it (Debian and Ubuntu ship \
             it in libjemalloc-dev); collecting it did not"
                .to_string(),
        );
    }
    lines
}

/// How a program ended, for messages ("exited with status 3", "was ended by
/// signal 9"): a helper's here, the wrapped command's in `gpu_lock`.
pub(crate) fn describe_status(status: ExitStatus) -> String {
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
/// For a wrapped run this is the wrap's status: valgrind (memcheck with
/// `--error-exitcode=0`), taskset and a heaptrack that passes its program's
/// status through (`heaptrack_drops_status`) pass the benchmark's own status
/// through. Under a heaptrack that drops it, the status the shell that waited
/// on the benchmark reported decides instead (`ShellStatus::benchmark_end`).
fn benchmark_exit(status: ExitStatus) -> Option<BenchmarkExit> {
    if status.success() {
        return None;
    }
    if let Some(code) = status.code() {
        return exit_of_code(code);
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

/// How a benchmark that exited with @p code ended, or `None` for 0.
fn exit_of_code(code: i32) -> Option<BenchmarkExit> {
    match code {
        0 => None,
        PROFILE_FAILED_EXIT_CODE => Some(BenchmarkExit::ProfileFailed),
        _ => Some(BenchmarkExit::Status(code)),
    }
}

/* ----------------------------- heaptrack's shell ----------------------------- */

/// The variable that names the file the trap writes the status into:
/// libbench's `heaptrack_status::STATUS_VARIABLE` (HeaptrackStatusTrap.hpp).
const HEAPTRACK_STATUS_VARIABLE: &str = "VERNIER_HEAPTRACK_STATUS";

/// What bash reads as it starts (BASH_ENV) when heaptrack's script runs under
/// it: the trap keeps the path `HEAPTRACK_STATUS_VARIABLE` names, removes that
/// variable and BASH_ENV from what the script starts, and, after the command
/// that starts the program (`"$client" "$@"` in heaptrack's script), writes
/// that program's exit status, as the shell that waited on it read it, into
/// the file. libbench's doctor gives bash the same text
/// (HeaptrackStatusTrap.hpp); both are held to
/// tests/fixtures/heaptrack_status.bash.
const HEAPTRACK_STATUS_TRAP: &str = r#"# bench run and bench doctor give bash this file as BASH_ENV when they run
# heaptrack's script under it: after the command that starts the program
# ("$client" "$@"), the trap writes that program's exit status, as the shell
# that waited on it read it, into the file VERNIER_HEAPTRACK_STATUS names.
__vernier_status_file=$VERNIER_HEAPTRACK_STATUS
unset BASH_ENV VERNIER_HEAPTRACK_STATUS
__vernier_status_next=0
__vernier_status_trap() {
  local status=$?
  if [ "$__vernier_status_next" = 1 ]; then
    printf '%s\n' "$status" >"$__vernier_status_file"
    __vernier_status_next=0
  fi
  # The command as heaptrack's script writes it, not expanded.
  # shellcheck disable=SC2016
  case $BASH_COMMAND in
  *'"$client" "$@"'*) __vernier_status_next=1 ;;
  esac
}
trap __vernier_status_trap DEBUG
"#;

/// One run the shell's report is checked with: the program heaptrack's script
/// starts, its arguments, the status the report must hold, and the program as
/// messages name it.
struct ShellCheck {
    program: &'static str,
    args: &'static [&'static str],
    status: i32,
    name: &'static str,
}

/// What fails, what passes, and what a signal ends (128 and SIGKILL's 9, as
/// a shell reports it).
const SHELL_CHECKS: [ShellCheck; 3] = [
    ShellCheck {
        program: "/bin/false",
        args: &[],
        status: 1,
        name: "/bin/false",
    },
    ShellCheck {
        program: "/bin/true",
        args: &[],
        status: 0,
        name: "/bin/true",
    },
    ShellCheck {
        program: "/bin/sh",
        args: &["-c", "kill -KILL $$"],
        status: 137,
        name: "a program ended by SIGKILL",
    },
];

/// The checks as messages name them: "/bin/false 1, /bin/true 0, ...".
fn checked_statuses() -> String {
    SHELL_CHECKS
        .iter()
        .map(|check| format!("{} {}", check.name, check.status))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Where the shell that runs heaptrack's script under bash reports the exit
/// status of the program the script starts, for a heaptrack that does not
/// pass that status through (`heaptrack_drops_status`): a private folder of
/// this run's holds the trap (`HEAPTRACK_STATUS_TRAP`) and the files the trap
/// writes, and goes with it.
struct ShellStatus {
    /// The private folder.
    folder: PrivateDir,
    /// heaptrack's script, as PATH finds it: bash runs it by this path.
    heaptrack: PathBuf,
}

impl ShellStatus {
    /// The report, set up and confirmed on this host: heaptrack's script runs
    /// each of `SHELL_CHECKS` under bash with the trap, as owned runs, and
    /// each report must hold the status the check names. Without bash on
    /// PATH, or with a report that does not, heaptrack is refused before
    /// anything is created or started, naming what the check saw and the
    /// remedy; a check that cannot start or does not end within
    /// `HEAPTRACK_PROBE_BOUND` is an error.
    fn confirmed() -> Result<Self, Error> {
        let refusal = |what: String, remedy: &str| {
            Error::ToolNotFound(format!(
                "heaptrack exits 0 for a program that fails ({HEAPTRACK_PROBE} exited 0), and \
                 {what}. {remedy}; bench doctor <binary> --profile heaptrack checks it"
            ))
        };
        let passes = "a heaptrack that passes the exit status of the program it runs through, \
                      as heaptrack 1.5.0 does";
        if find_in_path("bash").is_none() {
            return Err(refusal(
                "bash, under which bench run takes the benchmark's status from the shell that \
                 waits on it, is not on PATH"
                    .to_string(),
                &format!("Install bash, or use {passes}"),
            ));
        }
        let Some(heaptrack) = find_in_path("heaptrack") else {
            return Err(Error::ToolNotFound("heaptrack is not on PATH".to_string()));
        };
        let private = |e: std::io::Error| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("a private folder for the status heaptrack's shell reports: {e}"),
            ))
        };
        let folder = PrivateDir::create("heaptrack-status").map_err(private)?;
        fs::write(folder.path().join("trap.bash"), HEAPTRACK_STATUS_TRAP).map_err(private)?;
        let shell = ShellStatus { folder, heaptrack };
        for (n, check) in SHELL_CHECKS.iter().enumerate() {
            let report = shell.folder.path().join(format!("check-{n}"));
            let mut command = Command::new("bash");
            command
                .arg(&shell.heaptrack)
                .arg("-o")
                .arg(shell.folder.path().join(format!("probe-{n}")))
                .arg(check.program)
                .args(check.args)
                .envs(shell.env_for(&report));
            let what = format!(
                "bash {} -o <private folder>/probe {}",
                shell.heaptrack.display(),
                std::iter::once(check.program)
                    .chain(check.args.iter().copied())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            match run_owned(&mut command, &what, HEAPTRACK_PROBE_BOUND) {
                Ok(OwnedEnd::Exited(..)) => {}
                Ok(OwnedEnd::TimedOut) => {
                    return Err(Error::ToolNotFound(format!(
                        "{what} did not end within {} s, so whether the shell that runs \
                         heaptrack reports the benchmark's exit status is not known",
                        HEAPTRACK_PROBE_BOUND.as_secs()
                    )))
                }
                Err(OwnedError::Start(e) | OwnedError::Wait(e)) => {
                    return Err(Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("{what}: {e}"),
                    )))
                }
            }
            let reported = read_reported(&report);
            if reported != Some(check.status) {
                return Err(refusal(
                    format!(
                        "the shell that runs it under bash did not report the status of the \
                         program it started ({}: {}, not {}), so whether a benchmark passed \
                         under it cannot be known",
                        check.name,
                        reported.map_or("no status".to_string(), |s| s.to_string()),
                        check.status
                    ),
                    &format!(
                        "Use {passes}, or run the benchmark under heaptrack by hand and read \
                         whether it passed in its own output"
                    ),
                ));
            }
        }
        Ok(shell)
    }

    /// The program and arguments that run heaptrack's script under bash with
    /// @p args, heaptrack's own.
    fn wrap(&self, args: &[String]) -> (String, Vec<String>) {
        let mut words = vec![self.heaptrack.display().to_string()];
        words.extend(args.iter().cloned());
        ("bash".to_string(), words)
    }

    /// The environment that has bash read the trap as it starts, and the trap
    /// write the status into @p report.
    fn env_for(&self, report: &Path) -> Vec<(String, String)> {
        vec![
            (
                "BASH_ENV".to_string(),
                self.folder.path().join("trap.bash").display().to_string(),
            ),
            (
                HEAPTRACK_STATUS_VARIABLE.to_string(),
                report.display().to_string(),
            ),
        ]
    }

    /// Where the trap reports the benchmark's status.
    fn report(&self) -> PathBuf {
        self.folder.path().join("status")
    }

    /// How the benchmark ended, from the status the shell that waited on it
    /// reported, for a run whose wrap ended as @p wrap: that status, a failure
    /// of the wrap itself after a benchmark that exited 0 (a failed collection
    /// of @p request), or, with no report, a run whose status is not known,
    /// naming how the wrap ended and what of its output is kept (@p kept).
    fn benchmark_end(&self, wrap: ExitStatus, request: &str, kept: &str) -> Result<(), Error> {
        match read_reported(&self.report()) {
            Some(code) => {
                if let Some(end) = exit_of_code(code) {
                    return Err(Error::Benchmark(end));
                }
                if !wrap.success() {
                    return Err(profile_failure(
                        request,
                        "collection",
                        format!(
                            "heaptrack {} after the benchmark exited with status 0; {kept}",
                            describe_status(wrap)
                        ),
                    ));
                }
                Ok(())
            }
            None => Err(Error::BenchmarkStatusUnknown(format!(
                "the shell that ran the benchmark under heaptrack reported no status for it, so \
                 whether it passed is not known. heaptrack, which does not pass the benchmark's \
                 exit status through, {}; {kept}",
                describe_status(wrap)
            ))),
        }
    }
}

/// The most of a report read: a status and a newline.
const REPORT_LIMIT: u64 = 8;

/// The status the report at @p path holds (`parse_reported`), or None for a
/// report that is missing or unreadable.
fn read_reported(path: &Path) -> Option<i32> {
    use std::io::Read;

    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|file| file.take(REPORT_LIMIT).read_to_end(&mut bytes))
        .ok()?;
    parse_reported(&String::from_utf8_lossy(&bytes))
}

/// The status a report's @p text holds, as the trap writes it: from 0 to 255
/// in decimal, without a leading zero, and a newline; None for anything else.
fn parse_reported(text: &str) -> Option<i32> {
    let digits = text.strip_suffix('\n')?;
    let decimal = (1..=3).contains(&digits.len())
        && digits.bytes().all(|b| b.is_ascii_digit())
        && (digits.len() == 1 || !digits.starts_with('0'));
    decimal
        .then(|| digits.parse::<i32>().ok())
        .flatten()
        .filter(|code| (0..=255).contains(code))
}

/// What a failed run leaves of a wrap's output in @p dir, for its message:
/// the first of the route's outputs (@p outputs) that is there, with its size.
fn kept_output(tool: &str, dir: &Path, outputs: &[&[&str]]) -> String {
    for name in outputs.iter().flat_map(|alternatives| alternatives.iter()) {
        let path = dir.join(name);
        if let Ok(meta) = fs::metadata(&path) {
            if meta.is_file() {
                return format!(
                    "{tool}'s trace is kept: {} ({} bytes)",
                    path.display(),
                    meta.len()
                );
            }
        }
    }
    format!("{tool} wrote no trace in {}", dir.display())
}

/* ----------------------------- Start-up settings ----------------------------- */

/// The environment variable gperftools reads its CPU sampling rate from,
/// once, as the process starts (the benchmark's `SAMPLING_RATE_VARIABLE`).
const SAMPLING_RATE_VARIABLE: &str = "CPUPROFILE_FREQUENCY";

/// The most samples a second gperftools takes; it clamps a higher rate.
const MAX_SAMPLING_HZ: u32 = 4000;

/// The one profile that takes a sampling rate.
const SAMPLING_RATE_PROFILE: &str = "gperf";

/// `--profile` names that select no profiler: the benchmark runs unprofiled.
const NO_PROFILER: [&str; 1] = ["cupti"];

/// A sampling rate as the benchmark reads `--profile-frequency` and
/// CPUPROFILE_FREQUENCY: ASCII digits only, a whole number of Hz from 1 to
/// MAX_SAMPLING_HZ, leading zeros allowed; `None` for anything else, which
/// gperftools would read as its default rate or its maximum. The shared
/// table `tests/fixtures/profile_frequency.tsv` holds the cases both
/// languages are tested against.
pub(crate) fn parse_profile_frequency(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Digits only: the one way left to fail is a number too large.
    let hz: u64 = text.parse().ok()?;
    (1..=u64::from(MAX_SAMPLING_HZ))
        .contains(&hz)
        .then_some(hz as u32)
}

/// What the benchmark of request @p cfg must find in its environment as it
/// starts, because it is read only then: for a sampling rate with `--profile
/// gperf`, CPUPROFILE_FREQUENCY set to the rate's canonical value; nothing
/// when no rate is asked for, so an inherited value reaches the benchmark
/// untouched. Never a wrap: no wrap variables, no folder, no CPUPROFILE. A
/// rate that is not a whole number from 1 to 4000, one with no profiler to
/// take it and one for a profile that takes none are refused, before
/// anything is created or started.
pub fn startup_env(cfg: &RunConfig) -> Result<Vec<(String, String)>, Error> {
    let Some(ref text) = cfg.profile_frequency else {
        return Ok(Vec::new());
    };
    let Some(hz) = parse_profile_frequency(text) else {
        return Err(Error::InvalidArgs(format!(
            "--profile-frequency: '{text}' is not a whole number of Hz from 1 to \
             {MAX_SAMPLING_HZ} (gperftools samples at most {MAX_SAMPLING_HZ} times a second)"
        )));
    };
    match cfg.profile.as_deref().map(canonical_backend) {
        Some(tool) if tool == SAMPLING_RATE_PROFILE => {
            Ok(vec![(SAMPLING_RATE_VARIABLE.to_string(), hz.to_string())])
        }
        Some(tool) if !NO_PROFILER.contains(&tool) => Err(Error::InvalidArgs(format!(
            "--profile-frequency {hz} sets gperftools' CPU sampling rate; --profile {tool} \
             does not take it; drop --profile-frequency, or profile with --profile gperf"
        ))),
        _ => Err(Error::InvalidArgs(format!(
            "--profile-frequency {hz} cannot apply: no profiler is selected; add --profile \
             gperf, or drop --profile-frequency"
        ))),
    }
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

/// A request as the command line states it, for messages: `bench run`'s and
/// `bench doctor`'s (an empty mode is no mode).
pub(crate) fn request_text(tool: &str, profile_args: Option<&str>) -> String {
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
    /// The metric names ncu is given (`NcuRequest::asked_metrics`), which the
    /// output is read back for after the run (`name_missing_metrics`); none
    /// for every other route.
    metrics: Vec<String>,
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

/// The scheduler option every valgrind route passes after `--tool`:
/// valgrind's fair scheduler. valgrind runs one thread at a time, and its
/// default lets a thread that computes keep the others from running, the
/// benchmark watchdog's thread among them. `try` falls back to the default
/// where the platform lacks it; `yes` would refuse to start there. The C++
/// side's is `valgrind_tool::FAIR_SCHEDULER`; the shared route table pins
/// the two together.
const VALGRIND_FAIR_SCHEDULER: &str = "--fair-sched=try";

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
/// `None` for every profile `bench run` does not wrap in argv. ncu and
/// nsight read their words by Nsight Compute's grammar (`ncu_request`).
fn wrapped_modes(tool: &str) -> Option<&'static [&'static str]> {
    Some(match tool {
        "callgrind" | "heaptrack" => &[],
        "massif" => &["pages", "stacks"],
        "memcheck" => &["leak-full", "track-origins"],
        "helgrind" => &["drd"],
        "compute-sanitizer" => &["memcheck", "racecheck", "synccheck", "initcheck"],
        "ncu" => &["replay"],
        "nsight" => &["compute", "ncu", "replay"],
        _ => return None,
    })
}

/// The metrics a kernel replay collects (`--profile-args replay`), as ncu's
/// `--metrics`: the throughputs, achieved occupancy, warp execution and
/// branch efficiency, registers per thread and the instruction mix (the ALU,
/// FMA and load/store pipes), which the GPU guide says a replay shows. The
/// C++ backend prints the same list (`REPLAY_METRICS`); the shared table's
/// replay routes pin both.
const REPLAY_METRICS: [&str; 11] = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__thread_inst_executed_per_inst_executed.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "launch__registers_per_thread",
    "smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
];

/// What a request for Nsight Compute collects: `--profile ncu`, or
/// `--profile nsight` with `compute`, `ncu`, `replay` or a selection.
#[derive(Debug, Default, PartialEq, Eq)]
struct NcuRequest {
    /// Nsight Compute is asked for: always under ncu; under nsight by a mode
    /// word or a selection.
    compute: bool,
    /// `replay`: the replay metric list, written to `kernel_replay`.
    replay: bool,
    /// `--set NAME`.
    set: Option<String>,
    /// Each `--section NAME`, in order.
    sections: Vec<String>,
    /// Every name of every `--metrics`, in order.
    metrics: Vec<String>,
}

impl NcuRequest {
    /// The options ncu is given for it, after bench run's own: the set, each
    /// section, then one `--metrics` with every name it asks for
    /// (`asked_metrics`).
    fn options(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(set) = &self.set {
            out.extend(["--set".to_string(), set.clone()]);
        }
        for section in &self.sections {
            out.extend(["--section".to_string(), section.clone()]);
        }
        let metrics = self.asked_metrics();
        if !metrics.is_empty() {
            out.extend(["--metrics".to_string(), metrics.join(",")]);
        }
        out
    }

    /// The metric names ncu is asked for: for `replay`, the replay metric
    /// list (which takes no set or section); otherwise every name of every
    /// `--metrics`, in order.
    fn asked_metrics(&self) -> Vec<String> {
        if self.replay {
            REPLAY_METRICS.iter().map(|name| name.to_string()).collect()
        } else {
            self.metrics.clone()
        }
    }
}

/// An ncu option `bench run` passes on: what its value names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NcuOption {
    Set,
    Section,
    Metrics,
}

impl NcuOption {
    /// The option @p word spells, `--set` or `--set=VALUE` (and the same for
    /// the others), with the value given after '='.
    fn of(word: &str) -> Option<(Self, Option<&str>)> {
        let (name, value) = match word.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (word, None),
        };
        let option = match name {
            "--set" => NcuOption::Set,
            "--section" => NcuOption::Section,
            "--metrics" => NcuOption::Metrics,
            _ => return None,
        };
        Some((option, value))
    }

    /// The refusal of the option given without its value.
    fn takes(self) -> &'static str {
        match self {
            NcuOption::Set => "--set takes one section set's name",
            NcuOption::Section => "--section takes one section's identifier",
            NcuOption::Metrics => "--metrics takes one or more metric names",
        }
    }
}

/// Read @p words, the `--profile-args` of a request for @p tool ("ncu" or
/// "nsight"), as Vernier's supported subset of Nsight Compute's options:
/// `replay`; `--set NAME`, once; `--section NAME` and `--metrics NAME`, each
/// as often as needed, a metric list running to the next option or mode
/// word; each option also as `--option=VALUE`; and under nsight the words
/// `compute` and `ncu`. NAME is letters, digits and '_'; a metric name may
/// also hold '.'. A mode word is never a value. A selection asks for Nsight
/// Compute, and replay takes none (it collects its own list). The first
/// problem is the refusal; the C++ parser (`parseNsightRequest`) gives the
/// same text, which the shared table's refuse rows hold both to.
fn ncu_request(tool: &str, words: &[&str]) -> Result<NcuRequest, String> {
    let nsight = tool == "nsight";
    let mut request = NcuRequest {
        compute: !nsight,
        ..NcuRequest::default()
    };
    // The option whose value comes next, and whether a metric list is open.
    let mut awaiting: Option<NcuOption> = None;
    let mut listing = false;
    for &word in words {
        let mode = word == "replay" || (nsight && (word == "compute" || word == "ncu"));
        if mode || word.starts_with('-') {
            if let Some(option) = awaiting {
                return Err(option.takes().to_string());
            }
            listing = false;
        }
        if mode {
            request.compute = true;
            request.replay |= word == "replay";
        } else if word.starts_with('-') {
            let Some((option, value)) = NcuOption::of(word) else {
                return Err(format!(
                    "'{word}' is not an option bench run passes to ncu; it passes --set, \
                     --section and --metrics (the report, its overwrite and \
                     --target-processes are bench run's own)"
                ));
            };
            request.compute = true;
            match value {
                None => awaiting = Some(option),
                Some(value) => {
                    take_ncu_value(&mut request, option, value)?;
                    listing = option == NcuOption::Metrics;
                }
            }
        } else if let Some(option) = awaiting.take() {
            take_ncu_value(&mut request, option, word)?;
            listing = option == NcuOption::Metrics;
        } else if listing {
            take_ncu_value(&mut request, NcuOption::Metrics, word)?;
        } else {
            let takes = if nsight {
                "compute, ncu, replay, --set NAME, --section NAME and --metrics NAMES"
            } else {
                "replay, --set NAME, --section NAME and --metrics NAMES"
            };
            return Err(format!(
                "'{word}' is not a mode of {tool}; it takes {takes}"
            ));
        }
    }
    if let Some(option) = awaiting {
        return Err(option.takes().to_string());
    }
    let selects =
        request.set.is_some() || !request.sections.is_empty() || !request.metrics.is_empty();
    if request.replay && selects {
        return Err(
            "replay collects its own metric list; give replay alone, or name what to collect \
             without it"
                .to_string(),
        );
    }
    Ok(request)
}

/// Add @p value, given to @p option, to @p request, or the refusal of a value
/// outside its form or of a second set.
fn take_ncu_value(request: &mut NcuRequest, option: NcuOption, value: &str) -> Result<(), String> {
    let name = |extra: char| {
        !value.is_empty()
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == extra)
    };
    let no_pattern = "bench run passes ncu no regular expression or metric group";
    match option {
        NcuOption::Set => {
            if value.is_empty() {
                return Err(option.takes().to_string());
            }
            if !name('_') {
                return Err(format!(
                    "'{value}' is not a section set's name (letters, digits and '_')"
                ));
            }
            if let Some(first) = &request.set {
                return Err(format!(
                    "--set is given twice ('{first}', '{value}'); ncu takes one set"
                ));
            }
            request.set = Some(value.to_string());
        }
        NcuOption::Section => {
            if value.is_empty() {
                return Err(option.takes().to_string());
            }
            if !name('_') {
                return Err(format!(
                    "'{value}' is not a section's identifier (letters, digits and '_'); \
                     {no_pattern}"
                ));
            }
            request.sections.push(value.to_string());
        }
        NcuOption::Metrics => {
            if value.is_empty() {
                return Err(option.takes().to_string());
            }
            if !name('.') {
                return Err(format!(
                    "'{value}' is not a metric name (letters, digits, '_' and '.'); {no_pattern}"
                ));
            }
            request.metrics.push(value.to_string());
        }
    }
    Ok(())
}

/// The route of a request for canonical backend @p tool with its
/// `--profile-args`, or `None` for the profiles the benchmark runs itself
/// (perf, gperf, rapl, bpftrace, offcpu), jemalloc's environment wrap and
/// rocprof. A word the tool does not take, a combination it cannot run, an
/// ncu option outside the supported subset (`ncu_request`) and an output
/// folder the tool would rewrite (heaptrack's %h and %p) or misread
/// (jemalloc's ',') are refused, naming what is accepted. Creates nothing.
fn route_for(
    tool: &str,
    profile_args: Option<&str>,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Result<Option<Route>, Error> {
    if tool == "jemalloc" {
        refuse_for_jemalloc(profile_args, binary, output_dir)?;
        return Ok(None);
    }
    let Some(accepted) = wrapped_modes(tool) else {
        return Ok(None);
    };
    let request = request_text(tool, profile_args);
    let refuse = |why: String| Err(Error::InvalidArgs(format!("{request}: {why}")));
    let words = mode_words(profile_args);
    let compute = if tool == "nsight" || tool == "ncu" {
        match ncu_request(tool, &words) {
            Ok(compute) => Some(compute),
            Err(why) => return refuse(why),
        }
    } else {
        None
    };
    if let Some(word) = words
        .iter()
        .find(|w| compute.is_none() && !accepted.contains(w))
    {
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
    // The folder as valgrind's output options and the -o of nsys and ncu
    // read it: each '%' doubled. compute-sanitizer's log escapes the folder
    // made absolute, and heaptrack's -o takes it unescaped (both below).
    let escaped = escape_percent(&d);
    // ncu's report, its overwrite and every process of the run, then what the
    // request selects (`NcuRequest::options`).
    let ncu = |request: &NcuRequest| -> Routed {
        let report = if request.replay {
            "kernel_replay"
        } else {
            "kernel_profile"
        };
        let mut a = vec![
            "-o".to_string(),
            format!("{escaped}/{report}"),
            "-f".into(),
            "--target-processes".into(),
            "all".into(),
        ];
        a.extend(request.options());
        let outputs: &'static [&'static [&'static str]] = if request.replay {
            &[&["kernel_replay.ncu-rep"]]
        } else {
            &[&["kernel_profile.ncu-rep"]]
        };
        ("ncu", a, outputs)
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
            // compute-sanitizer joins a relative log name to its working
            // directory before it reads '%' macros, so a '%' in that
            // directory's path would make it refuse the log: the folder is
            // named whole, then escaped.
            let whole = std::path::absolute(&dir).unwrap_or_else(|_| dir.clone());
            let log = escape_percent(&whole.display().to_string());
            (
                "compute-sanitizer",
                vec![
                    format!("--tool={sanitizer}"),
                    "--error-exitcode".into(),
                    TOOL_FINDINGS_EXIT_CODE.to_string(),
                    "--log-file".into(),
                    format!("{log}/sanitizer.log"),
                ],
                &[&["sanitizer.log"]],
            )
        }
        // nsight's compute modes and selections are ncu's route, into
        // nsight's folder.
        "nsight" if compute.as_ref().is_some_and(|c| c.compute) => {
            ncu(compute.as_ref().expect("nsight's request is read"))
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
        "ncu" => ncu(compute.as_ref().expect("ncu's request is read")),
        _ => unreachable!("wrapped_modes() admits only the tools matched above"),
    };
    if program == "valgrind" {
        args.insert(1, VALGRIND_FAIR_SCHEDULER.to_string());
    }
    args.push(bin);
    let metrics = match &compute {
        Some(request) if program == "ncu" => request.asked_metrics(),
        _ => Vec::new(),
    };
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
        metrics,
    }))
}

/// What jemalloc's environment wrap refuses before anything is created or
/// started: any `--profile-args` word (jemalloc takes no mode), and a folder
/// holding ',', since the folder reaches jemalloc inside MALLOC_CONF, which
/// ends an option at ',' and has no escape (the rest would be lost, or read
/// as options of its own).
fn refuse_for_jemalloc(
    profile_args: Option<&str>,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Result<(), Error> {
    let request = request_text("jemalloc", profile_args);
    if let Some(word) = mode_words(profile_args).first() {
        return Err(Error::InvalidArgs(format!(
            "{request}: '{word}' is not a mode of jemalloc, which takes none"
        )));
    }
    let dir = wrap_artifact_dir("jemalloc", binary, output_dir);
    let d = dir.display().to_string();
    if d.contains(',') {
        return Err(Error::InvalidArgs(format!(
            "{request}: jemalloc reads its dump folder inside MALLOC_CONF, which ends an \
             option at ',' and has no escape, so it cannot write into {d}, which holds ','; \
             choose a --profile-output-dir (and a binary name) without ','"
        )));
    }
    Ok(())
}

/// A route's program, arguments and required outputs, as `route_for` builds them.
type Routed = (
    &'static str,
    Vec<String>,
    &'static [&'static [&'static str]],
);

/// What jemalloc's environment wrap preloads: its soname, which ld.so
/// resolves through LD_LIBRARY_PATH first, as for any other library.
const JEMALLOC_PRELOAD: &str = "libjemalloc.so.2";

/// Env pairs for jemalloc's LD_PRELOAD wrap, pointing its dumps at @p dir:
/// profiling on (`prof`), a dump written when the process exits
/// (`prof_final`), holding the bytes allocated over the whole run
/// (`prof_accum`, which `jeprof --alloc_space` reads). The shared route table
/// pins the pairs, which the benchmark's jemalloc backend probes with too.
fn jemalloc_env(dir: &Path) -> Vec<(String, String)> {
    vec![
        ("LD_PRELOAD".to_string(), JEMALLOC_PRELOAD.to_string()),
        (
            "MALLOC_CONF".to_string(),
            format!(
                "prof:true,prof_final:true,prof_accum:true,prof_prefix:{}/jeprof",
                dir.display()
            ),
        ),
    ]
}

/// Wrap for backends that inject via environment rather than argv.
/// jemalloc: LD_PRELOAD the library and enable profiling with a dump at exit
/// into the per-binary artifact dir, whose previous dumps are removed first.
/// A library the loader cannot preload is refused before anything is created
/// or started: the benchmark would run without it, the loader only warning.
/// Whether the library profiles is the benchmark's backend's decision, made
/// when its profiler is created. `Ok(None)` when the tool is not env-shaped;
/// an error when the folder cannot be prepared.
fn env_wrap_for(
    tool: &str,
    binary: &Path,
    output_dir: Option<&Path>,
) -> Result<Option<Vec<(String, String)>>, Error> {
    if tool != "jemalloc" {
        return Ok(None);
    }
    if let Some(refusal) = jemalloc_preload_refusal()? {
        return Err(Error::ToolNotFound(format!(
            "the loader cannot preload {JEMALLOC_PRELOAD} (\"{refusal}\"), and --profile \
             jemalloc runs the benchmark with it preloaded. Use a jemalloc built with profiling \
             (configure --enable-prof; Debian's and Ubuntu's libjemalloc2 have it on amd64 and \
             i386, not on arm64), with its library folder first on LD_LIBRARY_PATH when it is \
             not the system's; bench doctor <binary> --profile jemalloc checks it"
        )));
    }
    let dir = wrap_artifact_dir(tool, binary, output_dir);
    prepare_folder(&dir, &[])?;
    for dump in jemalloc_dumps(&dir) {
        remove_previous(&dump)?;
    }
    Ok(Some(jemalloc_env(&dir)))
}

/// How long the loader's check of jemalloc's preload may take.
const PRELOAD_CHECK_BOUND: std::time::Duration = std::time::Duration::from_secs(10);

/// The loader is the only honest oracle for "will LD_PRELOAD work": it
/// preloads jemalloc by soname into /bin/true, an owned run, and its
/// "cannot be preloaded" line, if it prints one, is returned (the wrapped
/// benchmark would not get the library either: ld.so warns and continues).
/// None when the library loads.
fn jemalloc_preload_refusal() -> Result<Option<String>, Error> {
    let mut command = Command::new("/bin/true");
    command
        .env("LD_PRELOAD", JEMALLOC_PRELOAD)
        .env_remove("MALLOC_CONF");
    let what = format!("/bin/true with {JEMALLOC_PRELOAD} preloaded");
    match run_owned(&mut command, &what, PRELOAD_CHECK_BOUND) {
        Ok(OwnedEnd::Exited(_, _, stderr)) => Ok(stderr
            .lines()
            .find(|l| l.contains("cannot be preloaded"))
            .map(|l| l.trim().to_string())),
        Ok(OwnedEnd::TimedOut) => Err(Error::ToolNotFound(format!(
            "{what} did not end within {} s, so whether the loader preloads it is not known",
            PRELOAD_CHECK_BOUND.as_secs()
        ))),
        Err(OwnedError::Start(e) | OwnedError::Wait(e)) => Err(Error::Io(std::io::Error::new(
            e.kind(),
            format!("{what}: {e}"),
        ))),
    }
}

/// How long heaptrack's status probe may take.
const HEAPTRACK_PROBE_BOUND: std::time::Duration = std::time::Duration::from_secs(30);

/// heaptrack's status probe, as messages name it.
const HEAPTRACK_PROBE: &str = "heaptrack -o <private folder>/probe /bin/false";

/// Whether the heaptrack on PATH ends with a status of its own instead of the
/// program's it runs: heaptrack 1.3's script exits with its trace
/// compressor's status, 1.5 passes the program's through. The probe, an
/// owned run, is `heaptrack -o <private folder>/probe /bin/false`, which
/// exits 0 only under a heaptrack that drops the status; the folder goes with
/// the probe. A probe that cannot start, or does not end within
/// `HEAPTRACK_PROBE_BOUND`, is an error: where the benchmark's status would
/// come from is not known.
fn heaptrack_drops_status() -> Result<bool, Error> {
    let folder = PrivateDir::create("heaptrack-probe").map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("a private folder for heaptrack's status probe: {e}"),
        ))
    })?;
    let mut command = Command::new("heaptrack");
    command
        .arg("-o")
        .arg(folder.path().join("probe"))
        .arg("/bin/false");
    let end = run_owned(&mut command, HEAPTRACK_PROBE, HEAPTRACK_PROBE_BOUND);
    drop(folder);
    match end {
        Ok(OwnedEnd::Exited(status, _, _)) => Ok(status.success()),
        Ok(OwnedEnd::TimedOut) => Err(Error::ToolNotFound(format!(
            "{HEAPTRACK_PROBE} did not end within {} s, so whether heaptrack passes the \
             benchmark's exit status through is not known",
            HEAPTRACK_PROBE_BOUND.as_secs()
        ))),
        Err(OwnedError::Start(e) | OwnedError::Wait(e)) => Err(Error::Io(std::io::Error::new(
            e.kind(),
            format!("{HEAPTRACK_PROBE}: {e}"),
        ))),
    }
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
        lossy_eprintln!(
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

/// Give SIGCHLD its default action, without SA_NOCLDWAIT, for the rest of the
/// process, so that every child a bench command starts stays waitable until
/// the command reaps it: ignored (kept across exec) or with SA_NOCLDWAIT, it
/// would have the system reap each child as it exits, and the command could
/// not read how the child ended (its wait fails with ECHILD), nor keep the
/// process id an owned run's group is signalled by. The children then start
/// with the default action too. The bench binary sets it before any command
/// runs; owned runs (`owned_run::spawn`) set it again before each start.
#[cfg(unix)]
pub fn keep_children_waitable() -> std::io::Result<()> {
    // SAFETY: a zeroed sigaction is a valid value of the type; sigaction only
    // reads `action`, owned here; the default action installs no handler.
    let rc = unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut())
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// No SIGCHLD where there are no Unix signals: nothing to set.
#[cfg(not(unix))]
pub fn keep_children_waitable() -> std::io::Result<()> {
    Ok(())
}

/// The owned run (`super::owned_run`), where `run_owned` and `validate` reach it.
pub(super) use super::owned_run;

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

/// How long reading back an Nsight Compute report may take.
const METRIC_CHECK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Name each metric of @p asked that @p report, the Nsight Compute report
/// this run wrote, holds nothing for, one line each; nothing when it holds
/// them all. ncu says nothing of such a metric, whether the GPU does not
/// report it or ncu does not know the name, so the report is read back with
/// the per-kernel summary `bench nsight-parse` reads, as an owned run
/// (`run_owned`) bounded by `METRIC_CHECK_TIMEOUT`, after the benchmark has
/// exited. A report that cannot be read back is one line saying that the
/// metrics were not checked, and why. Either way the run ends as it would
/// have, and the report is left as it is.
fn name_missing_metrics(report: &Path, asked: &[String]) {
    let mut command = Command::new("ncu");
    command.args(super::nsight_report::ncu_summary_args(report));
    let checked = match run_owned(&mut command, "ncu --import", METRIC_CHECK_TIMEOUT) {
        Ok(OwnedEnd::Exited(status, out, err)) if status.success() => {
            super::nsight_report::metrics_not_in_summary(&out, &err, asked)
        }
        Ok(OwnedEnd::Exited(status, out, err)) => {
            let tail = [&err, &out]
                .iter()
                .find_map(|text| text.lines().rev().map(str::trim).find(|l| !l.is_empty()))
                .unwrap_or("");
            Err(format!(
                "ncu --import {}{}{}",
                describe_status(status),
                if tail.is_empty() { "" } else { ": " },
                tail
            ))
        }
        Ok(OwnedEnd::TimedOut) => Err(format!(
            "ncu --import did not finish within {} s and was stopped",
            METRIC_CHECK_TIMEOUT.as_secs()
        )),
        Err(OwnedError::Start(e)) => Err(format!("ncu could not be started: {e}")),
        Err(OwnedError::Wait(e)) => Err(format!("ncu could not be waited for: {e}")),
    };
    match checked {
        Ok(missing) => {
            for metric in missing {
                println!(
                    "[bench] the report holds nothing for {metric}: ncu reported nothing for it \
                     on this GPU"
                );
            }
        }
        Err(why) => println!("[bench] the report was not checked for the metrics asked for: {why}"),
    }
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

/// How racecheck begins each race it reports as an error. Its report holds
/// these lines before its summary, so they are findings, not the tool's own
/// error.
const RACE_FINDING: &str = "Error: Race reported between";

/// The tool's own error in @p report, without its `=========` prefix: the
/// first line holding `Error:` that is not one of racecheck's findings.
fn sanitizer_own_error(report: &str) -> Option<&str> {
    report
        .lines()
        .map(|line| line.trim_start_matches('=').trim())
        .find(|line| line.contains("Error:") && !line.starts_with(RACE_FINDING))
}

/// Decide a finished compute-sanitizer run from this run's report and the
/// tool's exit status. The route passes `--error-exitcode 5`, and the tool
/// then returns 5 whenever it reports errors, whatever the benchmark itself
/// returned; without errors it returns the benchmark's own status. So the
/// report's summary decides: errors counted there are findings; no errors
/// leave the status to the benchmark (a benchmark returning 5 itself is not a
/// finding), unless the report says the benchmark did not end normally. A
/// report without its summary is the tool's own failure when it holds the
/// tool's `Error:` line (collection; racecheck's findings, which also begin
/// `Error:`, are not that line), and otherwise incomplete (completion); no
/// report at all is a completion failure naming its path. Nothing is counted
/// without a summary. `Ok` hands the status on as the benchmark's.
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
        None => Err(match sanitizer_own_error(&text) {
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

/// The program a run starts and its arguments before the benchmark's own,
/// each element one argument: `taskset -c <cpuset>` when pinning, then the
/// route's wrapper and its arguments (which end at the benchmark), or the
/// benchmark itself, its path as the OS has it.
fn launch_command(
    binary: &Path,
    taskset: &Option<String>,
    wrap: &Option<(String, Vec<String>)>,
) -> Vec<OsString> {
    let mut launch: Vec<OsString> = match taskset {
        Some(cpuset) => vec!["taskset".into(), "-c".into(), cpuset.into()],
        None => Vec::new(),
    };
    match wrap {
        Some((prog, prefix)) => {
            launch.extend(std::iter::once(prog).chain(prefix).map(OsString::from));
        }
        None => launch.push(binary.as_os_str().to_owned()),
    }
    launch
}

/// The `Running:` line's command: @p launch (`launch_command`), then @p args.
fn format_command(launch: &[OsString], args: &[String]) -> String {
    let mut parts: Vec<String> = launch
        .iter()
        .map(|part| part.to_string_lossy().into_owned())
        .collect();
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

    /// @test The trap bench run gives bash is the one the shared fixture
    /// holds, which libbench's doctor gives it too.
    #[test]
    fn heaptrack_status_trap_is_the_shared_one() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/heaptrack_status.bash");
        let shared = fs::read_to_string(&path).expect("the shared fixture");
        assert_eq!(HEAPTRACK_STATUS_TRAP, shared);
    }

    /// Run @p script under bash with the trap, its report going to a file of
    /// @p dir: the report as written (None when there is none) and stdout.
    fn under_trap(dir: &Path, script: &str, args: &[&str]) -> (Option<String>, String) {
        let trap = dir.join("trap.bash");
        let report = dir.join("status");
        let file = dir.join("script.sh");
        fs::write(&trap, HEAPTRACK_STATUS_TRAP).expect("the trap");
        fs::write(&file, script).expect("the script");
        let _ = fs::remove_file(&report);
        let out = Command::new("bash")
            .arg(&file)
            .args(args)
            .env("BASH_ENV", &trap)
            .env(HEAPTRACK_STATUS_VARIABLE, &report)
            .output()
            .expect("bash runs");
        (
            fs::read_to_string(&report).ok(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    }

    /// @test Under bash with the trap, a script that starts its program as
    /// heaptrack's does (`"$client" "$@"`) has the status the shell read for
    /// it reported: a failure's, a success's, 128 and the signal for a signal's
    /// end; the program sees neither the trap's variable nor BASH_ENV. A
    /// program started by another command is not reported.
    #[test]
    fn heaptrack_status_trap_reports_the_status_of_the_program_the_script_starts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = "client=\"$1\"\nshift\n\"$client\" \"$@\"\nwait\n";
        for (args, report) in [
            (&["/bin/false"][..], "1\n"),
            (&["/bin/true"][..], "0\n"),
            (&["/bin/sh", "-c", "exit 3"][..], "3\n"),
            (&["/bin/sh", "-c", "kill -KILL $$"][..], "137\n"),
            (&["/bin/sh", "-c", "kill -TERM $$"][..], "143\n"),
        ] {
            assert_eq!(
                under_trap(dir.path(), script, args).0.as_deref(),
                Some(report),
                "{args:?}"
            );
        }
        let (report, stdout) = under_trap(
            dir.path(),
            script,
            &[
                "/bin/sh",
                "-c",
                "echo \"${BASH_ENV-unset} ${VERNIER_HEAPTRACK_STATUS-unset}\"",
            ],
        );
        assert_eq!(report.as_deref(), Some("0\n"));
        assert_eq!(stdout, "unset unset\n");
        let other = "program=\"$1\"\nshift\n\"$program\" \"$@\"\nwait\n";
        assert_eq!(under_trap(dir.path(), other, &["/bin/false"]).0, None);
    }

    /// @test A report holds a status as the trap writes it, from 0 to 255 and
    /// a newline; anything else is no report.
    #[test]
    fn shell_status_report_is_a_status_and_a_newline() {
        for (text, code) in [
            ("0\n", 0),
            ("1\n", 1),
            ("4\n", 4),
            ("137\n", 137),
            ("255\n", 255),
        ] {
            assert_eq!(parse_reported(text), Some(code), "{text:?}");
        }
        for text in [
            "", "\n", "0", "1", "x\n", "-1\n", "256\n", "1000\n", "01\n", "00\n", " 1\n", "1 \n",
            "1\n\n", "1\n0\n", "+1\n",
        ] {
            assert_eq!(parse_reported(text), None, "{text:?}");
        }
    }

    /// @test A reported status ends the run as the same exit status would.
    #[test]
    fn reported_status_ends_the_run_as_the_status_would() {
        use std::os::unix::process::ExitStatusExt;
        for code in [0, 1, 2, 3, 4, 127, 134, 137, 255] {
            assert_eq!(
                exit_of_code(code),
                benchmark_exit(ExitStatus::from_raw(code << 8)),
                "{code}"
            );
        }
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
            &launch_command(Path::new("./my_test"), &None, &None),
            &["--csv".to_string(), "out.csv".to_string()],
        );
        assert_eq!(s, "./my_test --csv out.csv");
    }

    /// @test format_command with taskset.
    #[test]
    fn format_command_taskset() {
        let s = format_command(
            &launch_command(Path::new("./my_test"), &Some("0,1".to_string()), &None),
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
            &launch_command(Path::new("./my_test"), &None, &wrap),
            &["--profile".to_string(), "massif".to_string()],
        );
        assert_eq!(
            s,
            "valgrind --tool=massif --massif-out-file=out/foo.massif/massif.out ./my_test --profile massif"
        );
        let s = format_command(
            &launch_command(Path::new("./my_test"), &Some("0,1".to_string()), &wrap),
            &["--profile".to_string(), "massif".to_string()],
        );
        assert_eq!(
            s,
            "taskset -c 0,1 valgrind --tool=massif --massif-out-file=out/foo.massif/massif.out \
             ./my_test --profile massif"
        );
    }

    /// @test The launch keeps each argument whole, a path with a space in it
    /// too, in the order taskset, wrapper, benchmark.
    #[test]
    fn launch_command_keeps_each_argument_whole() {
        let binary = Path::new("./my bench");
        let wrap = Some((
            "valgrind".to_string(),
            vec![
                "--log-file=out/a b/log".to_string(),
                "./my bench".to_string(),
            ],
        ));
        let words = |launch: Vec<OsString>| -> Vec<String> {
            launch
                .into_iter()
                .map(|part| part.into_string().expect("UTF-8"))
                .collect()
        };
        assert_eq!(words(launch_command(binary, &None, &None)), ["./my bench"]);
        assert_eq!(
            words(launch_command(binary, &Some("0,1".to_string()), &None)),
            ["taskset", "-c", "0,1", "./my bench"]
        );
        assert_eq!(
            words(launch_command(binary, &None, &wrap)),
            ["valgrind", "--log-file=out/a b/log", "./my bench"]
        );
        assert_eq!(
            words(launch_command(binary, &Some("0,1".to_string()), &wrap)),
            [
                "taskset",
                "-c",
                "0,1",
                "valgrind",
                "--log-file=out/a b/log",
                "./my bench"
            ]
        );
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

    /// @test The project's output folder is a default: taken when neither
    /// the command nor the arguments after `--` give one, and set aside by
    /// either; the command's own and one after `--` still conflict, each named
    /// where it was given.
    #[test]
    fn effective_request_takes_the_projects_output_folder_last() {
        let with = |own: Option<&str>, extra: &[&str]| RunConfig {
            profile_output_dir: own.map(PathBuf::from),
            default_profile_output_dir: Some(PathBuf::from("from-yaml")),
            ..request(Some("perf"), None, extra)
        };
        let folder = |cfg: RunConfig| {
            effective_request(&cfg, "bench run")
                .expect("one request")
                .profile_output_dir
        };
        assert_eq!(folder(with(None, &[])), Some(PathBuf::from("from-yaml")));
        assert_eq!(
            folder(with(None, &["--artifact-root", "other"])),
            Some(PathBuf::from("other"))
        );
        assert_eq!(folder(with(Some("own"), &[])), Some(PathBuf::from("own")));
        let err = effective_request(
            &with(Some("own"), &["--profile-output-dir", "other"]),
            "bench run",
        )
        .expect_err("two folders");
        assert_eq!(
            err.to_string(),
            "invalid arguments: --profile-output-dir is given more than once with different \
             values ('own' to bench run, 'other' after --); give it once"
        );
    }

    /// A request for @p profile with the rate @p hz given to the command.
    fn rate_request(profile: Option<&str>, hz: &str) -> RunConfig {
        RunConfig {
            profile_frequency: Some(hz.to_string()),
            ..request(profile, None, &[])
        }
    }

    /// @test The rates the shared table accepts parse to its values, and
    /// every other input, which gperftools would read as another rate, is
    /// none; the variable and the maximum are the table's.
    #[test]
    fn parse_profile_frequency_follows_the_shared_table() {
        let rows = fixture_rows("profile_frequency.tsv", "rate");
        assert!(rows.len() >= 20, "the shared table was not read");
        for row in rows {
            let parsed = parse_profile_frequency(&row[1]);
            if row[2] == "ok" {
                assert_eq!(
                    parsed,
                    Some(row[3].parse().expect("a value")),
                    "'{}'",
                    row[1]
                );
            } else {
                assert_eq!(parsed, None, "'{}' is accepted", row[1]);
            }
        }
        let variable = fixture_rows("profile_frequency.tsv", "variable");
        assert_eq!(variable.len(), 1);
        assert_eq!(SAMPLING_RATE_VARIABLE, variable[0][1]);
        let maximum = fixture_rows("profile_frequency.tsv", "maximum");
        assert_eq!(maximum.len(), 1);
        assert_eq!(MAX_SAMPLING_HZ.to_string(), maximum[0][1]);
    }

    /// @test A rate for gperf is one start-up setting, the variable with the
    /// rate's canonical value, and nothing else: no wrap metadata, no
    /// folder, no CPUPROFILE. Without a rate there is none, whatever the
    /// profile.
    #[test]
    fn startup_env_sets_only_the_sampling_rate() {
        for (hz, canonical) in [
            ("250", "250"),
            ("0250", "250"),
            ("4000", "4000"),
            ("1", "1"),
        ] {
            let env = startup_env(&rate_request(Some("gperf"), hz)).expect("a rate for gperf");
            assert_eq!(
                env,
                [(SAMPLING_RATE_VARIABLE.to_string(), canonical.to_string())],
                "'{hz}'"
            );
        }
        for profile in [
            None,
            Some("gperf"),
            Some("perf"),
            Some("massif"),
            Some("cupti"),
        ] {
            assert!(
                startup_env(&request(profile, None, &[]))
                    .expect("no rate")
                    .is_empty(),
                "{profile:?}"
            );
        }
    }

    /// @test A rate is refused before anything starts when the shared table
    /// refuses it, when no profiler takes it (no --profile, or a name that
    /// selects none) and for every profile the table says does not take it;
    /// it is accepted for the one that does.
    #[test]
    fn startup_env_refuses_a_rate_that_cannot_apply() {
        for row in fixture_rows("profile_frequency.tsv", "rate") {
            if row[2] == "refused" {
                let err = startup_env(&rate_request(Some("gperf"), &row[1]))
                    .expect_err("an invalid rate");
                assert_eq!(
                    err.to_string(),
                    format!(
                        "invalid arguments: --profile-frequency: '{}' is not a whole number of \
                         Hz from 1 to 4000 (gperftools samples at most 4000 times a second)",
                        row[1]
                    )
                );
            }
        }
        let none = "invalid arguments: --profile-frequency 250 cannot apply: no profiler is \
                    selected; add --profile gperf, or drop --profile-frequency";
        let err = startup_env(&rate_request(None, "250")).expect_err("no profiler");
        assert_eq!(err.to_string(), none);
        let names = fixture_rows("profile_frequency.tsv", "noprofiler");
        assert_eq!(names.len(), 1, "the shared table was not read");
        for row in names {
            let err = startup_env(&rate_request(Some(&row[1]), "250")).expect_err("none");
            assert_eq!(err.to_string(), none, "{}", row[1]);
        }
        let backends = fixture_rows("profile_frequency.tsv", "backend");
        assert!(backends.len() >= 16, "the shared table lost its backends");
        for row in backends {
            let result = startup_env(&rate_request(Some(&row[1]), "250"));
            if row[2] == "take" {
                assert!(result.is_ok(), "{}: {result:?}", row[1]);
            } else {
                let err = result.expect_err("a profile that takes no rate");
                assert_eq!(
                    err.to_string(),
                    format!(
                        "invalid arguments: --profile-frequency 250 sets gperftools' CPU \
                         sampling rate; --profile {} does not take it; drop \
                         --profile-frequency, or profile with --profile gperf",
                        canonical_backend(&row[1])
                    ),
                    "{}",
                    row[1]
                );
            }
        }
    }

    /// @test Each precedence row's request, as bench run prepares it: the
    /// child gets the request's canonical rate, and without a request bench
    /// run adds nothing, so the inherited value (or its absence) reaches the
    /// child untouched.
    #[test]
    fn startup_env_follows_the_precedence_rows() {
        let rows = fixture_rows("profile_frequency.tsv", "precedence");
        assert!(rows.len() >= 8, "the shared table was not read");
        for row in rows {
            let (requested, inherited, child) = (&row[1], &row[2], &row[3]);
            let cfg = if requested.is_empty() {
                request(Some("gperf"), None, &[])
            } else {
                rate_request(Some("gperf"), requested)
            };
            let env = startup_env(&cfg).expect("a request the table expects to start");
            let prepared = env
                .iter()
                .find(|(k, _)| k == SAMPLING_RATE_VARIABLE)
                .map(|(_, v)| v.as_str());
            // What the child sees: the runner's value over the inherited one.
            let seen = prepared.or((!inherited.is_empty()).then_some(inherited.as_str()));
            assert_eq!(
                seen.unwrap_or(""),
                child.as_str(),
                "request '{requested}', inherited '{inherited}'"
            );
        }
    }

    /// @test A rate after `--` is the request's: taken in and removed from
    /// the arguments after `--`; the same rate given to the command and after
    /// `--` is one request, by value ("0250" is 250), different rates are
    /// refused, and so is the "--profile-frequency=N" spelling, which the
    /// benchmark does not read.
    #[test]
    fn effective_request_takes_in_a_forwarded_rate() {
        let r = effective_request(
            &request(
                Some("gperf"),
                None,
                &[
                    "--gtest_filter=A.*",
                    "--profile-frequency",
                    "1000",
                    "--cycles",
                    "5",
                ],
            ),
            "bench run",
        )
        .expect("one request");
        assert_eq!(r.profile_frequency.as_deref(), Some("1000"));
        assert_eq!(r.extra_args, ["--gtest_filter=A.*", "--cycles", "5"]);
        let same = RunConfig {
            extra_args: vec!["--profile-frequency".into(), "0250".into()],
            ..rate_request(Some("gperf"), "250")
        };
        let r = effective_request(&same, "bench run").expect("the same rate twice");
        assert_eq!(r.profile_frequency.as_deref(), Some("250"));
        assert!(r.extra_args.is_empty());
        let differ = RunConfig {
            extra_args: vec!["--profile-frequency".into(), "1000".into()],
            ..rate_request(Some("gperf"), "250")
        };
        let err = effective_request(&differ, "bench run").expect_err("two rates");
        assert_eq!(
            err.to_string(),
            "invalid arguments: --profile-frequency is given more than once with different \
             values ('250' to bench run, '1000' after --); give it once"
        );
        let err = effective_request(
            &request(Some("gperf"), None, &["--profile-frequency=1000"]),
            "bench run",
        )
        .expect_err("the = spelling");
        assert_eq!(
            err.to_string(),
            "invalid arguments: --profile-frequency=1000 after --: give it as \
             --profile-frequency 1000, which the benchmark reads"
        );
    }

    /// @test The project's default filter is given when no argument after
    /// `--` starts with `--gtest_filter=`, and not when one does, which is
    /// then the benchmark's only filter (an empty one too); arguments that
    /// only look like a filter flag, which GoogleTest does not read as one,
    /// do not count; without a default nothing is given.
    #[test]
    fn configured_filter_gives_way_to_the_command_lines() {
        let cfg = |filter: Option<&str>, extra: &[&str]| RunConfig {
            gtest_filter: filter.map(str::to_string),
            ..request(None, None, extra)
        };
        let configured = Some("--gtest_filter=Join.*".to_string());
        assert_eq!(configured_filter(&cfg(Some("Join.*"), &[])), configured);
        assert_eq!(
            configured_filter(&cfg(Some("Join.*"), &["--cycles", "5"])),
            configured
        );
        for given in ["--gtest_filter=Other.*", "--gtest_filter="] {
            assert_eq!(
                configured_filter(&cfg(Some("Join.*"), &["--cycles", "5", given])),
                None,
                "{given}"
            );
        }
        for similar in [
            "--gtest_filter",
            "-gtest_filter=Other.*",
            "--gtest_filters=Other.*",
            "gtest_filter=Other.*",
        ] {
            assert_eq!(
                configured_filter(&cfg(Some("Join.*"), &[similar])),
                configured,
                "{similar}"
            );
        }
        assert_eq!(configured_filter(&cfg(None, &[])), None);
        assert_eq!(
            configured_filter(&cfg(None, &["--gtest_filter=Other.*"])),
            None
        );
    }

    /// @test The benchmark is given the rate once, by its canonical value,
    /// after the profile's mode.
    #[test]
    fn profile_request_args_names_the_rate_once() {
        let cfg = RunConfig {
            profile_args: Some("cpu".into()),
            ..rate_request(Some("gperf"), "0250")
        };
        assert_eq!(
            profile_request_args(&cfg),
            [
                "--profile",
                "gperf",
                "--profile-args",
                "cpu",
                "--profile-frequency",
                "250"
            ]
        );
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

    /// @test Nsight Compute's grammar: plain nsight is Systems; a selection
    /// or a mode word asks for Compute; a metric list runs on through
    /// further words, to the next option or mode word, and its '=' form
    /// takes a comma list; anything after a closed list is a mode word or
    /// nothing.
    #[test]
    fn ncu_request_reads_the_supported_subset() {
        let read = |tool: &str, args: &str| ncu_request(tool, &mode_words(Some(args)));
        assert_eq!(read("nsight", ""), Ok(NcuRequest::default()));
        let listed = read("nsight", "--metrics=a.sum,b.avg c ncu --section X").expect("read");
        assert!(listed.compute && !listed.replay);
        assert_eq!(listed.metrics, ["a.sum", "b.avg", "c"]);
        assert_eq!(listed.sections, ["X"]);
        assert_eq!(
            listed.options(),
            ["--section", "X", "--metrics", "a.sum,b.avg,c"]
        );
        let ended = read("nsight", "--metrics a replay");
        assert_eq!(
            ended.expect_err("replay ends the list and conflicts with it"),
            "replay collects its own metric list; give replay alone, or name what to collect \
             without it"
        );
        assert_eq!(
            read("ncu", "--section A b").expect_err("b is no value"),
            "'b' is not a mode of ncu; it takes replay, --set NAME, --section NAME and --metrics \
             NAMES"
        );
        let replay = read("ncu", "replay").expect("read");
        assert!(replay.compute && replay.replay);
        assert_eq!(replay.options()[1], REPLAY_METRICS.join(","));
    }

    /// @test --analyze is refused for a run under Nsight Compute, whichever
    /// way the request reaches ncu (after `--` too), and taken for a run
    /// under any other route or none; a request bench run refuses for itself
    /// is left to it.
    #[test]
    fn analysis_of_an_ncu_capture_is_refused() {
        for (profile, args, extra, text) in [
            (Some("ncu"), None, &[][..], "--profile ncu"),
            (
                Some("ncu"),
                Some("replay"),
                &[],
                "--profile ncu --profile-args 'replay'",
            ),
            (
                Some("nsight"),
                Some("compute"),
                &[],
                "--profile nsight --profile-args 'compute'",
            ),
            (
                Some("nsys"),
                Some("--section Occupancy"),
                &[],
                "--profile nsight --profile-args '--section Occupancy'",
            ),
            (
                None,
                None,
                &["--profile", "ncu", "--profile-args", "--set full"],
                "--profile ncu --profile-args '--set full'",
            ),
        ] {
            assert_eq!(
                refuse_analysis_of_capture(&request(profile, args, extra))
                    .expect_err("refused")
                    .to_string(),
                format!(
                    "invalid arguments: --analyze summarises the run's timing, and {text} runs \
                     the benchmark under Nsight Compute (ncu), a profiling capture that leaves \
                     the timing cells empty; drop --analyze, or time the benchmark in a run \
                     without ncu"
                ),
                "{profile:?} {args:?} {extra:?}"
            );
        }
        for (profile, args, extra) in [
            (None, None, &[][..]),
            (Some("nsight"), None, &[]),
            (Some("nsys"), None, &[]),
            (Some("perf"), None, &[]),
            (Some("massif"), Some("pages"), &[]),
            (Some("ncu"), Some("--kernel-name k"), &[]),
            (Some("ncu"), None, &["--profile", "perf"]),
        ] {
            assert!(
                refuse_analysis_of_capture(&request(profile, args, extra)).is_ok(),
                "{profile:?} {args:?} {extra:?}"
            );
        }
    }

    /// One row of the shared route table, split on tabs, with `-` read as empty.
    fn table_rows(kind: &str) -> Vec<Vec<String>> {
        fixture_rows("profile_routes.tsv", kind)
    }

    /// The rows of @p kind in the shared table @p file, split on tabs, with
    /// `-` read as empty.
    fn fixture_rows(file: &str, kind: &str) -> Vec<Vec<String>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(file);
        let text = fs::read_to_string(&path).expect("the shared table");
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
        // The working directory as the table's <cwd> spells it.
        let cwd = escape_percent(
            &std::env::current_dir()
                .expect("the working directory")
                .display()
                .to_string(),
        );
        let routes = table_rows("route");
        assert!(routes.len() >= 20, "the table lost its routes");
        for row in routes {
            let tool = canonical_backend(&row[1]);
            let args = (!row[2].is_empty()).then_some(row[2].as_str());
            let dir = format!("out/my_test.{tool}");
            let whole = format!("{cwd}/{dir}");
            let expected: Vec<String> = row[4]
                .split(' ')
                .map(|a| {
                    a.replace("<whole-dir>", &whole)
                        .replace("<dir>", &dir)
                        .replace("<bin>", "./my_test")
                })
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
            let expected = row[5].replace("<cwd>", &cwd);
            assert_eq!(value.as_ref(), Some(&expected), "{row:?}: {:?}", route.args);
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

    /// A report compute-sanitizer 2025.4.0 wrote, kept in
    /// tests/fixtures/compute_sanitizer (its README says how).
    fn recorded_sanitizer_report(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/compute_sanitizer")
            .join(name)
    }

    /// @test The verdict reads the real tool's reports as the tool summarized
    /// them: the unguarded memcheck run's 4 errors and the race's 2 fail the
    /// run as findings of that count, with the status the tool ended with
    /// (the route's --error-exitcode, 5), and each clean report, the
    /// same-value race included, counts none and passes the run, handing the
    /// benchmark's status 0 on.
    #[test]
    #[cfg(unix)]
    fn sanitizer_verdict_reads_the_real_reports() {
        use std::os::unix::process::ExitStatusExt;
        for (file, errors, status) in [
            ("memcheck_unguarded.txt", 4, 5),
            ("memcheck_clean.txt", 0, 0),
            ("racecheck_clean.txt", 0, 0),
            ("synccheck_clean.txt", 0, 0),
            ("initcheck_clean.txt", 0, 0),
            ("racecheck_race.txt", 2, 5),
            ("racecheck_same_value.txt", 0, 0),
        ] {
            let path = recorded_sanitizer_report(file);
            let text = fs::read_to_string(&path).expect("the recorded report");
            assert_eq!(sanitizer_error_count(&text), Some(errors), "{file}");
            let verdict = check_sanitizer_run(
                &path,
                ExitStatus::from_raw(status << 8),
                "compute-sanitizer",
            );
            match (errors, verdict) {
                (0, Ok(())) => {}
                (0, other) => panic!("{file}: a clean report must pass the run, got {other:?}"),
                (
                    expected,
                    Err(Error::Sanitizer(SanitizerFailure::Findings { errors: found, .. })),
                ) => assert_eq!(found, expected, "{file}"),
                (expected, other) => {
                    panic!("{file}: {expected} errors must fail the run as findings, got {other:?}")
                }
            }
        }
    }

    /// @test A real racecheck report cut before its summary, as when the tool
    /// is ended before it summarizes, is incomplete: its race findings begin
    /// `Error:` but are not the tool's own error, and nothing is counted.
    #[test]
    #[cfg(unix)]
    fn a_racecheck_report_cut_before_its_summary_is_incomplete() {
        use std::os::unix::process::ExitStatusExt;
        let whole = fs::read_to_string(recorded_sanitizer_report("racecheck_race.txt"))
            .expect("the recorded report");
        let cut: String = whole
            .lines()
            .take_while(|line| !line.contains("RACECHECK SUMMARY"))
            .map(|line| format!("{line}\n"))
            .collect();
        assert!(
            cut.contains("Error: Race reported between"),
            "the cut report keeps its findings"
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sanitizer.log");
        fs::write(&path, &cut).expect("write the cut report");
        let err = check_sanitizer_run(&path, ExitStatus::from_raw(9), "compute-sanitizer")
            .expect_err("a report without its summary passed");
        assert_eq!(
            err.to_string(),
            format!(
                "compute-sanitizer failed: completion: compute-sanitizer was ended by signal 9, \
                 and its report {} holds no error summary: it is incomplete, and nothing in it \
                 is counted",
                path.display()
            )
        );
    }

    /// @test The tool's own error is its `Error:` line, the first that is not
    /// one of racecheck's findings, without the tool's prefix.
    #[test]
    fn sanitizer_own_error_skips_race_findings() {
        let head = "========= COMPUTE-SANITIZER\n";
        let race =
            "========= Error: Race reported between Write access at k(int *)+0x80 in k.cu:15\n";
        let own =
            "========= Error: Target application terminated before first instrumented API call\n";
        assert_eq!(sanitizer_own_error(&format!("{head}{race}")), None);
        assert_eq!(
            sanitizer_own_error(&format!("{head}{race}{own}")),
            Some("Error: Target application terminated before first instrumented API call")
        );
        assert_eq!(
            sanitizer_own_error(&format!("{head}{own}")),
            Some("Error: Target application terminated before first instrumented API call")
        );
        assert_eq!(sanitizer_own_error(head), None);
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

    /// @test jemalloc's environment wrap is the shared table's `env` rows, in
    /// order: the preload by soname (ld.so owns the search, so no path
    /// appears), profiling on, a dump at exit, the bytes allocated over the
    /// run, into the folder. The benchmark's backend probes with the same rows.
    #[test]
    fn jemalloc_env_matches_the_shared_table() {
        let rows: Vec<(String, String)> = table_rows("env")
            .into_iter()
            .filter(|row| row[1] == "jemalloc")
            .map(|row| (row[2].clone(), row[3].replace("<dir>", "out/t.jemalloc")))
            .collect();
        assert_eq!(rows.len(), 2, "the table lost jemalloc's environment rows");
        assert_eq!(jemalloc_env(Path::new("out/t.jemalloc")), rows);
        assert_eq!(
            rows[1].1,
            "prof:true,prof_final:true,prof_accum:true,prof_prefix:out/t.jemalloc/jeprof"
        );
    }

    /// @test jemalloc's dumps are found by the wrap's prefix and suffix, and
    /// their names give the process and the kind.
    #[test]
    fn jemalloc_dumps_match_the_wrap_prefix() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, text) in [
            ("jeprof.41.0.f.heap", "heap"),
            ("jeprof.41.1.i0.heap", ""),
            ("jeprof.txt", "not a dump"),
            ("other.heap", "not the wrap's"),
        ] {
            fs::write(dir.path().join(name), text).expect("write");
        }
        let names: Vec<String> = jemalloc_dumps(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["jeprof.41.0.f.heap", "jeprof.41.1.i0.heap"]);
        assert_eq!(
            jemalloc_dump_owner(Path::new("d/jeprof.41.0.f.heap")),
            Some((41, "f".to_string()))
        );
        assert_eq!(
            jemalloc_dump_owner(Path::new("jeprof.7.3.i0.heap")),
            Some((7, "i0".to_string()))
        );
        assert_eq!(jemalloc_dump_owner(Path::new("jeprof.x.0.f.heap")), None);
        assert_eq!(jemalloc_dump_owner(Path::new("jeprof.41.heap")), None);
    }

    /// @test A heap profile has its totals line when its `t*:` line holds
    /// four integers, zeros included; a text without one, or with a part of
    /// one, has none.
    #[test]
    fn totals_line_holds_four_integers() {
        assert!(has_totals_line(
            "heap_v2/524288\n  t*: 1: 82338 [2287: 14782155]\n  t0: 1: 82338 [2287: 14782155]\n"
        ));
        assert!(has_totals_line("heap_v2/524288\n  t*: 0: 0 [0: 0]\n"));
        assert!(!has_totals_line("heap_v2/524288\n"));
        assert!(!has_totals_line("  t*: 1: 2\n"));
        assert!(!has_totals_line(
            "  t*: 1: 82338 [99999999999999999999: 1]\n"
        ));
    }

    /// A heap profile of process @p binary as jemalloc 5.3.0 writes one
    /// (header, totals, a stack, the map), with @p count where its totals
    /// and stack record hold the run's encoded count: 0 for a run that
    /// sampled nothing.
    fn heap_profile(binary: &Path, count: u64) -> String {
        format!(
            "heap_v2/524288\n  t*: 1: 82338 [{count}: {}]\n  t0: 1: 82338 [{count}: {}]\n\
             @ 0x7f0000001000 0x55d0c1a01000\n  t*: 1: 82338 [{count}: {}]\n\n\
             MAPPED_LIBRARIES:\n\
             55d0c1a00000-55d0c1a2f000 r-xp 00000000 fd:01 1048601 {}\n\
             7f0000000000-7f00000a0000 r-xp 00000000 fd:01 900001 /usr/lib/libjemalloc.so.2\n",
            count * 6000,
            count * 6000,
            count * 6000,
            binary.display()
        )
    }

    /// @test The completion check takes the benchmark's own exit dump and
    /// nothing else: another process's dump is named and never accepted, an
    /// empty dump, a text that is not a heap profile and a profile of another
    /// program fail at completion, and a run that sampled nothing passes.
    #[test]
    fn jemalloc_completion_takes_the_benchmarks_own_dump() {
        let binary = Path::new("/opt/bench/bin/MyTest");
        let check = |files: &[(&str, String)]| {
            let dir = tempfile::tempdir().expect("tempdir");
            for (name, text) in files {
                fs::write(dir.path().join(name), text).expect("write");
            }
            let result = require_jemalloc_dump(dir.path(), "--profile jemalloc", 4711, binary);
            (dir, result)
        };
        let own = heap_profile(binary, 2287);
        let (dir, ok) = check(&[("jeprof.4711.0.f.heap", own.clone())]);
        assert_eq!(
            ok.expect("the benchmark's dump"),
            dir.path().join("jeprof.4711.0.f.heap")
        );

        let (dir, both) = check(&[
            ("jeprof.4711.0.f.heap", own.clone()),
            (
                "jeprof.4713.0.f.heap",
                heap_profile(Path::new("/usr/bin/git"), 3),
            ),
        ]);
        assert_eq!(
            both.expect("the benchmark's dump"),
            dir.path().join("jeprof.4711.0.f.heap")
        );

        let (_dir, zero) = check(&[("jeprof.4711.0.f.heap", heap_profile(binary, 0))]);
        assert!(
            zero.is_ok(),
            "a run that sampled nothing is a valid run: {zero:?}"
        );

        let failures = [
            (
                vec![(
                    "jeprof.4713.0.f.heap",
                    heap_profile(Path::new("/usr/bin/git"), 3),
                )],
                "jemalloc wrote no exit dump of the benchmark (process 4711), \
                 jeprof.4711.<n>.f.heap, in ",
            ),
            (
                vec![],
                "jemalloc wrote no exit dump of the benchmark (process 4711)",
            ),
            (
                vec![("jeprof.4711.0.i0.heap", own.clone())],
                "the folder holds only other dumps: ",
            ),
            (
                vec![("jeprof.4711.0.f.heap", String::new())],
                "jeprof.4711.0.f.heap is empty",
            ),
            (
                vec![("jeprof.4711.0.f.heap", "garbage\n".to_string())],
                "jeprof.4711.0.f.heap is not a heap profile: it starts 'garbage'",
            ),
            (
                vec![(
                    "jeprof.4711.0.f.heap",
                    heap_profile(Path::new("/usr/bin/git"), 3),
                )],
                "jeprof.4711.0.f.heap does not map the benchmark, /opt/bench/bin/MyTest: it \
                 is not this run's profile",
            ),
            (
                vec![("jeprof.4711.0.f.heap", own.replace("t*:", "t?:"))],
                "has no totals line (t*:)",
            ),
        ];
        for (files, words) in failures {
            let (_dir, result) = check(&files);
            let err = result.expect_err(words);
            let text = err.to_string();
            assert!(
                text.starts_with("--profile jemalloc failed: completion: "),
                "{words}: {text}"
            );
            assert!(text.contains(words), "{words}: {text}");
        }
    }

    /// @test After a capture bench run prints the jeprof command, and says
    /// that only reading needs jeprof when it is not on PATH.
    #[test]
    fn jeprof_hint_names_the_command_and_a_missing_analyzer() {
        let binary = Path::new("./build/MyTest");
        let dump = Path::new("bench-out/MyTest.jemalloc/jeprof.4711.0.f.heap");
        let found = jeprof_hint(binary, dump, |_| {
            InPath::Executable(PathBuf::from("/usr/bin/jeprof"))
        });
        assert_eq!(
            found,
            [
                "[bench] Read it: jeprof --text --alloc_space ./build/MyTest \
              bench-out/MyTest.jemalloc/jeprof.4711.0.f.heap"
            ]
        );
        let missing = jeprof_hint(binary, dump, |_| InPath::Absent);
        assert_eq!(missing.len(), 2);
        assert_eq!(missing[0], found[0]);
        assert!(
            missing[1].starts_with("[bench] jeprof is not on PATH: reading the dump needs it"),
            "{}",
            missing[1]
        );
    }

    /// @test jemalloc takes no mode and cannot write into a folder holding
    /// ',', from the output folder or the binary's name: both are refused
    /// before anything is created, naming the folder.
    #[test]
    fn jemalloc_refusals_create_nothing() {
        let root = tempfile::tempdir().expect("tempdir");
        let out = root.path().join("out");
        let refused = route_for("jemalloc", Some("pages"), Path::new("./b"), Some(&out))
            .expect_err("a mode word");
        assert_eq!(
            refused.to_string(),
            "invalid arguments: --profile jemalloc --profile-args 'pages': 'pages' is not a \
             mode of jemalloc, which takes none"
        );
        let named = route_for("jemalloc", None, Path::new("./a,b"), Some(&out))
            .expect_err("a binary name with ','");
        assert!(
            named.to_string().contains(&format!(
                "cannot write into {}/a,b.jemalloc, which holds ','",
                out.display()
            )),
            "{named}"
        );
        assert!(matches!(
            route_for("jemalloc", None, Path::new("./b"), Some(&out)),
            Ok(None)
        ));
        assert!(!out.exists(), "a refusal or a decision created the folder");
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
