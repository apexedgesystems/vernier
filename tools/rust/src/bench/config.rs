//! `.bench.yaml` reader, writer, and validator.
//!
//! Downstream consumers that FetchContent vernier end up duplicating the
//! same CLI invocation pattern across dozens of libraries:
//!
//!   ./MyTest --cycles 1000 --repeats 10 --gtest_filter=-*Large* --csv ...
//!
//! `.bench.yaml` at the project root captures those defaults so per-binary
//! invocations don't have to. The runner reads the file at startup and
//! applies any unset CLI flags from it. The bench binary itself does NOT
//! read this file -- it stays purely flag-driven so it works the same way
//! whether or not the runner is involved.
//!
//! Schema (minimal, intentional -- expand only as concrete need arises):
//!
//!   # .bench.yaml at project root
//!   cycles: 10000
//!   repeats: 10
//!   profile_output_dir: bench-out
//!   gtest_filter: "-*Large*:-*PayloadScaling*"
//!   bin_roots: ["build/*"]               # for resolve_binary
//!   bin_subdirs: ["bin/ptests", "bin/tests"]
//!
//! Anything not listed here is parsed as a top-level string-keyed map and
//! preserved in `Config::extras` so a project can layer custom fields
//! without us having to add knobs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::Error;

/* ----------------------------- Config ----------------------------- */

/// Parsed `.bench.yaml`: the per-project defaults `bench run` and
/// `bench profile-all` fall back to when the corresponding CLI flag is
/// not supplied.
///
/// Named fields are the recognized keys; everything else round-trips
/// through `extras` so projects can layer custom fields without forcing
/// us to grow the schema.
#[derive(Debug, Default, Clone)]
pub struct Config {
    pub cycles: Option<u32>,
    pub repeats: Option<u32>,
    pub profile_output_dir: Option<PathBuf>,
    pub gtest_filter: Option<String>,
    pub bin_roots: Vec<String>,
    pub bin_subdirs: Vec<String>,
    /// Anything in the YAML not consumed by the named fields above.
    pub extras: BTreeMap<String, String>,
}

/* ----------------------------- IO ----------------------------- */

/// Default scaffold written by `bench init`. Kept small on purpose; users
/// add keys as they need them.
pub const DEFAULT_TEMPLATE: &str = "\
# .bench.yaml -- vernier project defaults.
#
# Anything set here is the default for `bench run` / `bench profile-all`
# when the user hasn't passed the corresponding CLI flag explicitly.
# Remove a key to let the harness's own defaults win.

# Iterations per repeat (--cycles N)
cycles: 10000

# Measurement repeats (--repeats N)
repeats: 10

# Where profile artifacts land (--profile-output-dir / --artifact-root)
profile_output_dir: bench-out

# Default --gtest_filter; leading minus excludes patterns.
# Useful for skipping tests that are inherently noisy or blocking.
# gtest_filter: \"-*Large*:-*PayloadScaling*\"

# Where the binary lookup walks when you pass `bench run Foo` (short name).
# Override only when the build layout differs from CMake's default.
# bin_roots:
#   - build/*
# bin_subdirs:
#   - bin/ptests
#   - bin/tests
#   - bin/examples
";

/// Walk up from `start` looking for `.bench.yaml`. Returns the path of the
/// first match, or `None` if the search reaches `/`.
pub fn find(start: &Path) -> Option<PathBuf> {
    let mut here: PathBuf = start.to_path_buf();
    loop {
        let candidate = here.join(".bench.yaml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !here.pop() {
            return None;
        }
    }
}

/// Parse a `.bench.yaml` file. The parser is intentionally minimal -- handles
/// `key: value`, `key: ["a", "b"]`, and `# comment` lines. Avoids pulling in
/// a YAML crate so this tool keeps its build tree small.
pub fn load(path: &Path) -> Result<Config, Error> {
    let text = fs::read_to_string(path).map_err(Error::Io)?;
    parse(&text, path)
}

/// Write the default template to `path`. Refuses to overwrite an existing
/// file unless `overwrite == true`.
pub fn write_template(path: &Path, overwrite: bool) -> Result<(), Error> {
    if path.exists() && !overwrite {
        return Err(Error::InvalidArgs(format!(
            "{} already exists (pass --force to overwrite)",
            path.display()
        )));
    }
    fs::write(path, DEFAULT_TEMPLATE).map_err(Error::Io)?;
    Ok(())
}

/* ----------------------------- Parsing ----------------------------- */

fn parse(text: &str, path: &Path) -> Result<Config, Error> {
    let mut cfg = Config::default();
    let mut in_list_for: Option<String> = None;

    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            in_list_for = None;
            continue;
        }
        // List item continuation ("  - value")
        if let Some(stripped) = line.trim_start().strip_prefix("- ") {
            let item = strip_quotes(stripped.trim());
            match in_list_for.as_deref() {
                Some("bin_roots") => cfg.bin_roots.push(item.to_string()),
                Some("bin_subdirs") => cfg.bin_subdirs.push(item.to_string()),
                _ => {
                    return Err(Error::Parse(format!(
                        "{}:{} list item with no preceding key",
                        path.display(),
                        lineno + 1
                    )));
                }
            }
            continue;
        }
        in_list_for = None;

        // key: value (or key:)
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim();

        if val.is_empty() {
            // Start of a block list -- "key:" followed by indented "- ..." lines
            in_list_for = Some(key.to_string());
            continue;
        }

        // Inline list: key: ["a", "b"]
        if val.starts_with('[') && val.ends_with(']') {
            let items = parse_inline_list(&val[1..val.len() - 1]);
            match key {
                "bin_roots" => cfg.bin_roots = items,
                "bin_subdirs" => cfg.bin_subdirs = items,
                _ => {
                    cfg.extras.insert(key.to_string(), val.to_string());
                }
            }
            continue;
        }

        let v = strip_quotes(val).to_string();
        match key {
            "cycles" => cfg.cycles = v.parse().ok(),
            "repeats" => cfg.repeats = v.parse().ok(),
            "profile_output_dir" => cfg.profile_output_dir = Some(PathBuf::from(v)),
            "gtest_filter" => cfg.gtest_filter = Some(v),
            _ => {
                cfg.extras.insert(key.to_string(), v);
            }
        }
    }

    Ok(cfg)
}

fn strip_comment(line: &str) -> &str {
    line.split_once('#').map(|(a, _)| a).unwrap_or(line)
}

fn strip_quotes(s: &str) -> &str {
    let t = s.trim();
    if (t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')) {
        &t[1..t.len() - 1]
    } else {
        t
    }
}

fn parse_inline_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| strip_quotes(p.trim()).to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/* ----------------------------- Validation ----------------------------- */

#[derive(Debug)]
pub struct ValidationReport {
    pub path: PathBuf,
    pub findings: Vec<Finding>,
}

#[derive(Debug)]
pub struct Finding {
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Warning,
    Error,
}

/// Sanity-check a loaded Config. Catches typos / impossible values; does not
/// shell out to anything.
pub fn validate(path: &Path) -> Result<ValidationReport, Error> {
    let cfg = load(path)?;
    let mut findings = Vec::new();

    if let Some(c) = cfg.cycles {
        if c == 0 {
            findings.push(Finding {
                severity: Severity::Error,
                message: "cycles must be > 0".into(),
            });
        }
    }
    if let Some(r) = cfg.repeats {
        if r == 0 {
            findings.push(Finding {
                severity: Severity::Error,
                message: "repeats must be > 0".into(),
            });
        } else if r < 5 {
            findings.push(Finding {
                severity: Severity::Warning,
                message: format!("repeats={} is low; consider >= 10 for stable medians", r),
            });
        }
    }
    if let Some(ref dir) = cfg.profile_output_dir {
        if dir.is_absolute() {
            findings.push(Finding {
                severity: Severity::Warning,
                message: "profile_output_dir is absolute; consider a path under the project root \
                          so artifacts stay with the source tree"
                    .into(),
            });
        }
    }
    if cfg.bin_roots.is_empty() && cfg.bin_subdirs.is_empty() {
        findings.push(Finding {
            severity: Severity::Ok,
            message: "no custom binary search paths -- defaults (build/*, bin/{ptests,tests,examples}) will apply"
                .into(),
        });
    }
    if !cfg.extras.is_empty() {
        let keys: Vec<&String> = cfg.extras.keys().collect();
        findings.push(Finding {
            severity: Severity::Warning,
            message: format!(
                "unrecognized keys preserved as extras: {:?} (typo? or custom field?)",
                keys
            ),
        });
    }

    Ok(ValidationReport {
        path: path.to_path_buf(),
        findings,
    })
}

/// Render a validation report.
pub fn print_validation(report: &ValidationReport) {
    println!();
    println!("=== {} ===", report.path.display());
    let mut fails = 0;
    for f in &report.findings {
        let tag = match f.severity {
            Severity::Ok => "[OK]  ",
            Severity::Warning => "[WARN]",
            Severity::Error => {
                fails += 1;
                "[FAIL]"
            }
        };
        println!("  {} {}", tag, f.message);
    }
    if report.findings.is_empty() {
        println!("  (nothing to flag; config looks reasonable)");
    }
    println!();
    if fails > 0 {
        println!("  {} fail(s).", fails);
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test Minimal key/value pairs parse into typed fields.
    #[test]
    fn parse_minimal() {
        let text = "cycles: 5000\nrepeats: 20\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.cycles, Some(5000));
        assert_eq!(cfg.repeats, Some(20));
    }

    /// @test Inline `[a, b]` list syntax populates the matching field.
    #[test]
    fn parse_inline_list() {
        let text = "bin_roots: [\"build/*\", \"out/*\"]\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.bin_roots, vec!["build/*", "out/*"]);
    }

    /// @test Block list syntax (`key:` followed by `- item`) populates the matching field.
    #[test]
    fn parse_block_list() {
        let text = "bin_subdirs:\n  - bin/ptests\n  - bin/tests\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.bin_subdirs, vec!["bin/ptests", "bin/tests"]);
    }

    /// @test Surrounding quotes are stripped from scalar values.
    #[test]
    fn parse_quoted_strings_strip() {
        let text = "gtest_filter: \"-*Large*\"\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.gtest_filter.as_deref(), Some("-*Large*"));
    }

    /// @test Unknown keys flow through to `Config::extras` unchanged.
    #[test]
    fn parse_preserves_extras() {
        let text = "future_knob: 42\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.extras.get("future_knob"), Some(&"42".to_string()));
    }

    /// @test Full-line and trailing `#` comments are stripped.
    #[test]
    fn comments_ignored() {
        let text = "# comment\ncycles: 100 # trailing\n";
        let cfg = parse(text, Path::new("test.yaml")).unwrap();
        assert_eq!(cfg.cycles, Some(100));
    }
}
