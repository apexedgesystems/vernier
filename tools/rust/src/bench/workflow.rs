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
    // Exact path first: respect any explicit override.
    let direct = PathBuf::from(name);
    if direct.is_file() {
        return Ok(direct);
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

/// Run a binary's `--profile-check` (binary readiness + backend doctor) and
/// stream the output. When no binary is given, surface a friendly message
/// directing the user at the conventional ptest pattern.
pub fn doctor(binary: Option<&Path>) -> Result<i32, Error> {
    let bin = match binary {
        Some(p) => p.to_path_buf(),
        None => {
            return Err(Error::InvalidArgs(
                "no binary given. Pass a ptest binary path, e.g. \
             `bench doctor build/native-linux-debug/bin/ptests/BenchDemo_01_BasicWorkflow`."
                    .into(),
            ))
        }
    };
    if !bin.is_file() {
        return Err(Error::InvalidArgs(format!(
            "binary not found: {}",
            bin.display()
        )));
    }
    let status = Command::new(&bin)
        .arg("--profile-check")
        .status()
        .map_err(Error::Io)?;
    Ok(status.code().unwrap_or(1))
}

/* ----------------------------- profile-all ----------------------------- */

/// Default profiler ladder when the user doesn't supply one.
/// Three CPU sampling/instruction tools; rocprof / nsight are GPU-specific
/// and would silently no-op on a CPU binary, so they're excluded by default.
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

/// Iterate over each profiler, invoking the binary in sequence. Each run gets
/// its own artifact subdirectory: `<artifact_root>/<profiler>/`.
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
    for tool in &profilers {
        let out_dir = cfg
            .artifact_root
            .clone()
            .unwrap_or_else(|| PathBuf::from("bench-out"))
            .join(tool);
        fs::create_dir_all(&out_dir).map_err(Error::Io)?;

        let mut cmd = Command::new(&cfg.binary);
        cmd.arg("--profile").arg(tool);
        cmd.arg("--profile-output-dir").arg(&out_dir);
        if let Some(f) = &cfg.gtest_filter {
            cmd.arg(format!("--gtest_filter={}", f));
        }
        if cfg.quick {
            cmd.arg("--quick");
        }
        if let Some(c) = cfg.cycles {
            cmd.arg("--cycles").arg(c.to_string());
        }
        if let Some(r) = cfg.repeats {
            cmd.arg("--repeats").arg(r.to_string());
        }

        eprintln!(
            "\n=== bench profile-all: tool={} -> {} ===",
            tool,
            out_dir.display()
        );
        let status = cmd.status().map_err(Error::Io)?;
        if !status.success() {
            eprintln!(
                "[bench] --profile {} exited with code {}; continuing with next profiler.",
                tool,
                status.code().unwrap_or(-1)
            );
        }
    }
    Ok(())
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
/// nsight-parse) remain the source of truth for the actual numbers.
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
