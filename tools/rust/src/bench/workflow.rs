//! Workflow subcommands that orchestrate the bench binary across multiple
//! profilers / artifacts.
//!
//! Each command shells out to the user's compiled bench binary; nothing here
//! parses CSV or .nsys-rep directly. The Rust tool stays a thin orchestrator
//! so the C++ harness remains the source of truth for profiler dispatch,
//! env validation, and artifact layout.
//!
//! Subcommands:
//!   doctor              run --profile-check against a binary
//!   profile-all         iterate a set of profilers, one run each
//!   profile-summarize   walk an artifact directory and tabulate
//!   resolve-binary      shared helper for the Run subcommand
//!
//! `resolve-binary` is the small helper that lets `bench run SLIP` find
//! `build/native-linux-debug/bin/ptests/SLIP_PTEST` automatically. It also
//! powers the doctor and profile-all entry points.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::Error;

/* ----------------------------- Binary resolution ----------------------------- */

/// Possible candidate names tried when the user supplies a short name.
fn candidate_names(name: &str) -> Vec<String> {
    let n = name.to_string();
    if n.contains('/') || n.ends_with("_PTEST") || n.ends_with("_PTest") {
        return vec![n];
    }
    vec![
        n.clone(),
        format!("{}_PTEST", n),
        format!("{}_pTest", n),
        format!("BenchDemo_{}", n),
    ]
}

/// Search conventional ptest/test/example directories for a binary that
/// matches `name`. The default search roots are `build/*` (every subdir of
/// `build/` -- CMake default convention); override by setting the env var
/// `VERNIER_BENCH_BIN_ROOTS` to a colon-separated list of roots and
/// `VERNIER_BENCH_BIN_SUBDIRS` to a colon-separated list of subdirs.
///
/// Example for a project that builds into `out/<preset>/` with `tests/` only:
///   VERNIER_BENCH_BIN_ROOTS=out/* VERNIER_BENCH_BIN_SUBDIRS=tests bench run Foo
pub fn resolve_binary(name: &str) -> Result<PathBuf, Error> {
    // Exact path first: respect any explicit override. A bare name found in
    // the working directory is that file (`launch_path`), which must be
    // executable: started as it is, a bare name would run whatever PATH holds.
    let direct = PathBuf::from(name);
    if direct.is_file() {
        let path = super::launch_path(&direct);
        if !direct.metadata().is_ok_and(|m| super::is_executable(&m)) {
            return Err(Error::InvalidArgs(format!(
                "{} is not executable",
                path.display()
            )));
        }
        return Ok(path);
    }

    let candidates = candidate_names(name);

    // Collect search roots. Env override expands shell-style "build/*" via
    // a simple readdir; otherwise fall back to the CMake default.
    let roots_spec =
        std::env::var("VERNIER_BENCH_BIN_ROOTS").unwrap_or_else(|_| "build/*".to_string());
    let mut roots: Vec<PathBuf> = Vec::new();
    for entry in roots_spec.split(':') {
        if let Some(prefix) = entry.strip_suffix("/*") {
            if let Ok(read) = fs::read_dir(prefix) {
                for e in read.flatten() {
                    roots.push(e.path());
                }
            }
        } else {
            let p = PathBuf::from(entry);
            if p.is_dir() {
                roots.push(p);
            }
        }
    }

    let subs_spec = std::env::var("VERNIER_BENCH_BIN_SUBDIRS")
        .unwrap_or_else(|_| "bin/ptests:bin/tests:bin/examples".to_string());
    let subs: Vec<&str> = subs_spec.split(':').collect();

    let mut found: Vec<PathBuf> = Vec::new();
    for root in &roots {
        for sub in &subs {
            for cand in &candidates {
                let p = root.join(sub).join(cand);
                if p.is_file() {
                    found.push(p);
                }
            }
        }
    }
    found.sort();
    found.dedup();
    if found.is_empty() {
        return Err(Error::InvalidArgs(format!(
            "no binary matching '{}' under {} (override with \
             VERNIER_BENCH_BIN_ROOTS / VERNIER_BENCH_BIN_SUBDIRS).",
            name, roots_spec
        )));
    }
    if found.len() > 1 {
        eprintln!(
            "[bench] '{}' matched {} binaries; picking '{}'. Pass a full path to disambiguate.",
            name,
            found.len(),
            found[0].display()
        );
    }
    Ok(found.remove(0))
}

/* ----------------------------- doctor ----------------------------- */

/// Map user-typed backend names to registered ones, mirroring the C++
/// registry's canonicalName (the tool is called nsys everywhere outside
/// the registry). `tools/rust/tests/fixtures/profile_routes.tsv` holds the
/// aliases both sides are tested against.
pub(crate) fn canonical_backend(name: &str) -> &str {
    match name {
        "nsys" => "nsight",
        other => other,
    }
}

/// Run a binary's `--profile-check` (binary readiness + backend doctor),
/// with @p request's profile flags spelled as `bench run` passes them
/// (`runner::effective_request`, `runner::profile_request_args`, then its
/// other arguments after `--`), and
/// @p env added to the binary's environment. Text mode streams the human
/// report. `json` prints the binary's JSON document, once it parses, and
/// nothing else on stdout (fleet capability records). `require` reads that
/// document and exits 1 unless every named backend reports OK -- warn is not
/// good enough for a profile lane; the requested backend is judged by the
/// document's `selected` row, the others by their default-mode rows. Its
/// verdict lines go to stderr with `json`, to stdout without it. A document
/// that does not parse is an error.
pub fn doctor(
    binary: &Path,
    request: &super::runner::RunConfig,
    env: &[(String, String)],
    json: bool,
    require: &[String],
) -> Result<i32, Error> {
    let bin = require_binary(binary)?;
    // The request bench run would make of the same arguments: its fields
    // given after -- are taken in, and a conflict between them is refused.
    let request = &super::runner::effective_request(request, "bench doctor")?;
    let mut request_args = super::runner::profile_request_args(request);
    request_args.extend(request.extra_args.iter().cloned());
    let selected = request.profile.as_deref().map(|profile| SelectedRequest {
        backend: canonical_backend(profile),
        profile_args: request.profile_args.as_deref().unwrap_or(""),
    });
    if json || !require.is_empty() {
        let doc = read_doctor_document(&bin, &request_args, env)?;
        if json {
            print!("{}", doc.text);
        }
        if require.is_empty() {
            return Ok(doc.status.unwrap_or(1));
        }
        let verdict = evaluate_required_backends(&doc.value, require, selected);
        for line in &verdict.lines {
            if json {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
        }
        return Ok(verdict.status);
    }
    let status = Command::new(&bin)
        .arg("--profile-check")
        .args(&request_args)
        .envs(env.iter().map(|(k, v)| (k, v)))
        .status()
        .map_err(|e| did_not_start(&bin, e))?;
    Ok(status.code().unwrap_or(1))
}

/// A spawn failure, naming the binary that did not start.
fn did_not_start(bin: &Path, e: std::io::Error) -> Error {
    Error::Io(std::io::Error::new(
        e.kind(),
        format!("{} did not start: {e}", bin.display()),
    ))
}

/// A binary's `--profile-check-json` output: the text, the parsed document
/// and the binary's exit status (None when a signal ended it).
pub(crate) struct DoctorDocument {
    pub text: String,
    pub value: serde_json::Value,
    pub status: Option<i32>,
}

/// @p binary as a path to an existing file, or the error that names it. A
/// bare file name becomes `./<name>` (`launch_path`): it was found in the
/// working directory, and a bare name would be looked up on PATH when it is
/// started.
fn require_binary(binary: &Path) -> Result<PathBuf, Error> {
    if !binary.is_file() {
        return Err(Error::InvalidArgs(format!(
            "binary not found: {}",
            binary.display()
        )));
    }
    Ok(super::launch_path(binary))
}

/// Run `<binary> --profile-check-json <args>` with @p env added to its
/// environment, and parse what it prints: the doctor's reader, shared by
/// `bench doctor` and `bench validate <binary>`. A binary that is not a file
/// or does not start, and output that is not a JSON document, are errors.
pub(crate) fn read_doctor_document(
    binary: &Path,
    args: &[String],
    env: &[(String, String)],
) -> Result<DoctorDocument, Error> {
    let bin = require_binary(binary)?;
    let out = Command::new(&bin)
        .arg("--profile-check-json")
        .args(args)
        .envs(env.iter().map(|(k, v)| (k, v)))
        .output()
        .map_err(|e| did_not_start(&bin, e))?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let value = serde_json::from_str(&text).map_err(|e| {
        // How it ended and its last word, for a binary that could not load
        // or run its doctor.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let last = stderr.lines().rev().map(str::trim).find(|l| !l.is_empty());
        let ended = match out.status.code() {
            Some(code) => format!("exit status {code}"),
            None => "ended by a signal".to_string(),
        };
        let how = match last {
            Some(line) => format!("{ended}; stderr: {line}"),
            None => ended,
        };
        Error::Parse(format!(
            "{} --profile-check-json printed no valid doctor document ({how}): {e}",
            bin.display()
        ))
    })?;
    Ok(DoctorDocument {
        text,
        value,
        status: out.status.code(),
    })
}

/// The --require verdict: its lines, one per requirement and a summary when
/// one is unmet, and the exit status (0 when every one is met, 1 otherwise).
struct RequireVerdict {
    lines: Vec<String>,
    status: i32,
}

/// The request the doctor was asked about: its backend by the canonical
/// name, and its mode ("" for none).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SelectedRequest<'a> {
    backend: &'a str,
    profile_args: &'a str,
}

/// A request as the command line states it, for messages.
fn request_text(backend: &str, profile_args: &str) -> String {
    if profile_args.is_empty() {
        format!("--profile {backend}")
    } else {
        format!("--profile {backend} --profile-args '{profile_args}'")
    }
}

/// The --require verdict: every named backend must exist and report "ok".
/// @p selected is the request the doctor was asked about: its backend is
/// judged by the document's `selected` row, which must be that request's,
/// by the backend's canonical name and the mode; a row for another request,
/// a row without that identity, and no row (a binary older than the
/// selected row) leave the requirement unmet. Every other backend is judged
/// by its default-mode row.
fn evaluate_required_backends(
    doc: &serde_json::Value,
    require: &[String],
    selected: Option<SelectedRequest<'_>>,
) -> RequireVerdict {
    let rows = doc["backends"].as_array().cloned().unwrap_or_default();
    let mut lines = Vec::new();
    let mut failed = 0;
    for raw in require {
        let want = canonical_backend(raw.trim());
        let (row, label) = match selected.filter(|s| s.backend == want) {
            Some(request) => {
                let label = format!(
                    "{want} ({})",
                    request_text(request.backend, request.profile_args)
                );
                let Some(row) = doc.get("selected").filter(|row| row.is_object()) else {
                    failed += 1;
                    lines.push(format!(
                        "[require] {want}: this binary does not report selected requests; \
                         rebuild it against this vernier, or drop --profile"
                    ));
                    continue;
                };
                // The request the binary's row answers, by the same names:
                // a row for another one does not answer this one.
                let answered = match (row["name"].as_str(), row["profileArgs"].as_str()) {
                    (Some(name), Some(args)) if !name.is_empty() => {
                        let name = canonical_backend(name);
                        (name != request.backend || args != request.profile_args)
                            .then(|| request_text(name, args))
                    }
                    _ => Some("no named request".to_string()),
                };
                if let Some(other) = answered {
                    failed += 1;
                    lines.push(format!(
                        "[require] {label}: NOT READY (the binary's selected row answers \
                         {other}, not this request)"
                    ));
                    continue;
                }
                (Some(row.clone()), label)
            }
            None => (
                rows.iter().find(|r| r["name"] == want).cloned(),
                want.to_string(),
            ),
        };
        match row {
            Some(r) if r["status"] == "ok" => {
                lines.push(format!("[require] {label}: OK"));
            }
            Some(r) => {
                failed += 1;
                let msg = r["message"].as_str().unwrap_or("");
                let hint = r["hint"].as_str().unwrap_or("");
                lines.push(format!("[require] {label}: NOT READY ({msg})"));
                if !hint.is_empty() {
                    lines.push(format!("          {hint}"));
                }
            }
            None => {
                failed += 1;
                lines.push(format!("[require] {want}: no such backend in this binary"));
            }
        }
    }
    if failed > 0 {
        lines.push(format!("[require] {failed} requirement(s) unmet"));
    }
    RequireVerdict {
        lines,
        status: i32::from(failed > 0),
    }
}

#[cfg(test)]
mod doctor_tests {
    use super::*;

    fn doc() -> serde_json::Value {
        serde_json::json!({"backends": [
            {"name": "offcpu", "status": "ok", "message": "", "hint": ""},
            {"name": "nsight", "status": "ok", "message": "", "hint": ""},
            {"name": "perf", "status": "warn", "message": "paranoid=2", "hint": "lower it"},
        ]})
    }

    /// @test A backend that reports ok meets its requirement.
    #[test]
    fn require_ok_passes() {
        let verdict = evaluate_required_backends(&doc(), &["offcpu".into()], None);
        assert_eq!(verdict.status, 0);
        assert_eq!(verdict.lines, ["[require] offcpu: OK"]);
    }

    /// @test A warning does not meet a requirement: the verdict names it and its remedy.
    #[test]
    fn require_warn_fails() {
        let verdict = evaluate_required_backends(&doc(), &["perf".into()], None);
        assert_eq!(verdict.status, 1);
        assert_eq!(
            verdict.lines,
            [
                "[require] perf: NOT READY (paranoid=2)",
                "          lower it",
                "[require] 1 requirement(s) unmet",
            ]
        );
    }

    /// @test A backend the binary does not have does not meet a requirement.
    #[test]
    fn require_missing_fails() {
        assert_eq!(
            evaluate_required_backends(&doc(), &["rocprof".into()], None).status,
            1
        );
    }

    /// @test The requested backend is judged by its selected row, the others
    /// by their default-mode rows.
    #[test]
    fn require_selected_row_judges_the_request() {
        let mut doc = doc();
        doc["selected"] = serde_json::json!({"name": "offcpu", "profileArgs": "x",
            "status": "fail", "message": "configuration: 'x'", "hint": "drop it"});
        let verdict = evaluate_required_backends(
            &doc,
            &["offcpu".into(), "nsight".into()],
            Some(SelectedRequest {
                backend: "offcpu",
                profile_args: "x",
            }),
        );
        assert_eq!(verdict.status, 1);
        assert_eq!(
            verdict.lines[0],
            "[require] offcpu (--profile offcpu --profile-args 'x'): NOT READY (configuration: 'x')"
        );
        assert!(verdict.lines.contains(&"[require] nsight: OK".to_string()));
    }

    /// The default rows with a selected row of @p name, @p args and @p status.
    fn doc_with_selected(
        name: serde_json::Value,
        args: serde_json::Value,
        status: &str,
    ) -> serde_json::Value {
        let mut doc = doc();
        doc["selected"] = serde_json::json!({"name": name, "profileArgs": args,
            "status": status, "message": "checked", "hint": ""});
        doc
    }

    /// @test A selected row that answers another request, by its backend or
    /// its mode, or names none, does not meet the requested backend's
    /// requirement, whatever its status; the line says which request it
    /// answers.
    #[test]
    fn require_selected_row_of_another_request_is_unmet() {
        let perf = SelectedRequest {
            backend: "perf",
            profile_args: "",
        };
        for (name, args, answers) in [
            (
                serde_json::json!("gperf"),
                serde_json::json!(""),
                "--profile gperf",
            ),
            (
                serde_json::json!("perf"),
                serde_json::json!("record"),
                "--profile perf --profile-args 'record'",
            ),
            (
                serde_json::json!(""),
                serde_json::json!(""),
                "no named request",
            ),
            (
                serde_json::Value::Null,
                serde_json::json!(""),
                "no named request",
            ),
            (
                serde_json::json!("perf"),
                serde_json::Value::Null,
                "no named request",
            ),
        ] {
            let doc = doc_with_selected(name.clone(), args.clone(), "ok");
            let verdict = evaluate_required_backends(&doc, &["perf".into()], Some(perf));
            assert_eq!(verdict.status, 1, "{name} {args}");
            assert_eq!(
                verdict.lines,
                [
                    format!(
                        "[require] perf (--profile perf): NOT READY (the binary's selected row \
                         answers {answers}, not this request)"
                    ),
                    "[require] 1 requirement(s) unmet".to_string(),
                ],
                "{name} {args}"
            );
        }
    }

    /// @test The control: a selected row that answers the request meets the
    /// requirement when ok, by the canonical name (a row naming nsys answers
    /// a request for nsight); the other backends keep their default rows.
    #[test]
    fn require_selected_row_of_the_request_is_judged() {
        let doc = doc_with_selected(serde_json::json!("perf"), serde_json::json!(""), "ok");
        let perf = SelectedRequest {
            backend: "perf",
            profile_args: "",
        };
        let verdict =
            evaluate_required_backends(&doc, &["perf".into(), "offcpu".into()], Some(perf));
        assert_eq!(verdict.status, 0);
        assert_eq!(
            verdict.lines,
            [
                "[require] perf (--profile perf): OK",
                "[require] offcpu: OK"
            ]
        );
        let doc = doc_with_selected(
            serde_json::json!("nsys"),
            serde_json::json!("compute"),
            "ok",
        );
        let nsight = SelectedRequest {
            backend: "nsight",
            profile_args: "compute",
        };
        let verdict = evaluate_required_backends(&doc, &["nsys".into()], Some(nsight));
        assert_eq!(
            verdict.lines,
            ["[require] nsight (--profile nsight --profile-args 'compute'): OK"]
        );
    }

    /// @test nsys is required as nsight, its registered name.
    #[test]
    fn require_nsys_alias_resolves() {
        assert_eq!(
            evaluate_required_backends(&doc(), &["nsys".into()], None).status,
            0
        );
    }
}

/* ----------------------------- profile-all ----------------------------- */

/// Default profiler ladder when the user doesn't supply one: three CPU
/// sampling and instruction tools. rocprof and nsight are GPU-specific and a
/// CPU binary cannot serve them, so they are excluded by default.
const DEFAULT_PROFILERS: &[&str] = &["gperf", "perf", "callgrind"];

pub struct ProfileAllConfig {
    pub binary: PathBuf,
    pub profilers: Vec<String>,
    pub artifact_root: Option<PathBuf>,
    pub gtest_filter: Option<String>,
    pub cycles: Option<u32>,
    pub repeats: Option<u32>,
    pub quick: bool,
}

/// One profiler's run in `bench profile-all`: the tool, its folder, and why
/// it failed (`None` when it completed).
struct ProfileAllRun {
    tool: String,
    folder: PathBuf,
    failure: Option<String>,
}

/// Run the binary under each profiler in sequence, each into its own folder
/// `<artifact_root>/<profiler>/`, through `run_benchmark`, so a wrapped
/// profile (callgrind, massif, memcheck, helgrind, heaptrack,
/// compute-sanitizer, nsight, ncu) is started and checked exactly as `bench
/// run --profile <X>` does. Every profiler runs whatever the others did; the
/// run ends with one summary line per profiler (completed or failed, its
/// folder, and the reason) and fails when any of them failed: every profiler
/// in the list, given or default, is required.
pub fn profile_all(cfg: &ProfileAllConfig) -> Result<(), Error> {
    if !cfg.binary.is_file() {
        return Err(Error::InvalidArgs(format!(
            "binary not found: {}",
            cfg.binary.display()
        )));
    }
    let profilers: Vec<&str> = if cfg.profilers.is_empty() {
        DEFAULT_PROFILERS.to_vec()
    } else {
        cfg.profilers.iter().map(|s| s.as_str()).collect()
    };
    let root = cfg
        .artifact_root
        .clone()
        .unwrap_or_else(|| PathBuf::from("bench-out"));
    let mut runs: Vec<ProfileAllRun> = Vec::new();
    for tool in &profilers {
        let out_dir = root.join(tool);
        eprintln!(
            "\n=== bench profile-all: tool={} -> {} ===",
            tool,
            out_dir.display()
        );

        let run_cfg = super::runner::RunConfig {
            binary: cfg.binary.clone(),
            csv: None,
            quick: cfg.quick,
            target_time: None,
            cycles: cfg.cycles,
            repeats: cfg.repeats,
            profile: Some(tool.to_string()),
            profile_args: None,
            profile_test_timeout: None,
            profile_output_dir: Some(out_dir.clone()),
            profile_analyze: false,
            taskset: None,
            extra_args: cfg
                .gtest_filter
                .as_ref()
                .map(|f| vec![format!("--gtest_filter={}", f)])
                .unwrap_or_default(),
        };
        let result = fs::create_dir_all(&out_dir)
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot create the output folder {}: {e}", out_dir.display()),
                ))
            })
            .and_then(|()| super::runner::run_benchmark(&run_cfg).map(|_| ()));
        let failure = result.err().map(|e| e.to_string());
        if let Some(ref why) = failure {
            eprintln!("[bench] --profile {tool} failed: {why}");
        }
        runs.push(ProfileAllRun {
            tool: tool.to_string(),
            folder: out_dir,
            failure,
        });
    }
    print_profile_all_summary(&runs);
    let failed: Vec<String> = runs
        .iter()
        .filter(|r| r.failure.is_some())
        .map(|r| r.tool.clone())
        .collect();
    if failed.is_empty() {
        Ok(())
    } else {
        Err(Error::ProfileAll {
            failed,
            total: runs.len(),
        })
    }
}

/// One line per run: the profiler, `completed` or `failed`, its folder, and
/// for a failure the reason.
fn print_profile_all_summary(runs: &[ProfileAllRun]) {
    let width = runs.iter().map(|r| r.tool.len()).max().unwrap_or(0);
    eprintln!("\n=== bench profile-all: summary ===");
    for run in runs {
        match &run.failure {
            None => eprintln!(
                "  {:<width$}  completed  {}",
                run.tool,
                run.folder.display()
            ),
            Some(why) => eprintln!(
                "  {:<width$}  failed     {} -- {why}",
                run.tool,
                run.folder.display()
            ),
        }
    }
}

/* ----------------------------- profile-summarize ----------------------------- */

#[derive(Debug)]
pub struct SummarizedTool {
    pub name: String,
    pub artifact_count: usize,
    pub bytes_total: u64,
    pub sample_artifact: Option<PathBuf>,
}

/// Walk an artifact directory produced by profile-all (or per-test runs) and
/// report what each profiler produced. This is intentionally a coarse summary
/// -- the per-tool analyzers (callgrind_annotate, pprof, nsys stats,
/// bench nsight-parse) remain the source of truth for the actual numbers.
pub fn profile_summarize(root: &Path) -> Result<Vec<SummarizedTool>, Error> {
    if !root.is_dir() {
        return Err(Error::InvalidArgs(format!(
            "not a directory: {}",
            root.display()
        )));
    }
    let mut out: Vec<SummarizedTool> = Vec::new();
    let entries = fs::read_dir(root).map_err(Error::Io)?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut count = 0_usize;
        let mut bytes = 0_u64;
        let mut sample: Option<PathBuf> = None;
        for sub in walk_files(&path) {
            count += 1;
            if let Ok(meta) = fs::metadata(&sub) {
                bytes += meta.len();
            }
            if sample.is_none() {
                sample = Some(sub);
            }
        }
        out.push(SummarizedTool {
            name,
            artifact_count: count,
            bytes_total: bytes,
            sample_artifact: sample,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(entries) = fs::read_dir(&p) else {
            continue;
        };
        for entry in entries.flatten() {
            let q = entry.path();
            if q.is_dir() {
                stack.push(q);
            } else if q.is_file() {
                out.push(q);
            }
        }
    }
    out
}

/// Render a `profile-summarize` report to stdout.
pub fn print_summary(report: &[SummarizedTool]) {
    println!();
    println!("=== Profile artifact summary ===");
    println!();
    if report.is_empty() {
        println!("  (no per-tool subdirectories found)");
        return;
    }
    println!(
        "  {:<22} {:>10} {:>14}   Sample artifact",
        "Tool subdir", "Files", "Total bytes"
    );
    println!("  {}", "-".repeat(80));
    for r in report {
        let sample = r
            .sample_artifact
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "  {:<22} {:>10} {:>14}   {}",
            truncate(&r.name, 22),
            r.artifact_count,
            r.bytes_total,
            sample
        );
    }
    println!();
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars()
            .take(n.saturating_sub(1))
            .chain(std::iter::once('+'))
            .collect()
    }
}
