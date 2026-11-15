//! `.bench.yaml` reader, writer, and validator.
//!
//! Downstream consumers that FetchContent vernier end up duplicating the
//! same CLI invocation pattern across dozens of libraries:
//!
//!   ./MyTest --cycles 1000 --repeats 10 --gtest_filter=-*Large* --csv ...
//!
//! `.bench.yaml` at the project root captures those defaults so per-binary
//! invocations don't have to. The runner reads the file at startup and
//! applies any unset CLI flags from it; `gtest_filter` is given to the
//! benchmark unless a filter follows `--` (`runner::RunConfig::gtest_filter`),
//! and a `--quick` on the command line keeps the file's cycles, repeats and
//! target_time out of the run.
//! The bench binary itself does NOT read this file -- it stays purely
//! flag-driven so it works the same way whether or not the runner is
//! involved.
//!
//! Schema (minimal, intentional -- expand only as concrete need arises):
//!
//!   # .bench.yaml at project root
//!   cycles: 10000
//!   repeats: 10
//!   target_time: 100ms                   # bench run's --target-time
//!   profile_output_dir: bench-out
//!   gtest_filter: "-*Large*:*PayloadScaling*"
//!   bin_roots: ["build/*"]               # workflow::BinSearch; a relative
//!                                        # root is relative to this file's folder
//!   bin_subdirs: ["bin/ptests", "bin/tests"]  # below each root
//!
//! Any other top-level key is kept, as written, in `Config::extras`, so a
//! project can layer custom fields without us having to add knobs; nothing
//! here reads them, and `bench config-validate` warns about them (a misspelt
//! key shows there). The rest is read strictly (`load`): a file that cannot
//! be read, a line that is not a top-level `key: value`, a `- item` below a
//! `key:` line or a comment, a quote left open, a key given twice, or a value
//! of the wrong kind for a key above is refused with each problem's line, and
//! `bench run` and `bench profile-all` stop on it before they start anything.

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
    /// Auto-size cycles to a wall-time window (e.g. "100ms"): `bench run`'s
    /// `--target-time`, passed on as written; the benchmark reads the value.
    pub target_time: Option<String>,
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
# when the user hasn't passed the corresponding CLI flag explicitly; a
# --quick on the command line takes precedence over cycles, repeats and
# target_time.
# Remove a key to let the harness's own defaults win.

# Iterations per repeat (--cycles N)
cycles: 10000

# Measurement repeats (--repeats N)
repeats: 10

# Size cycles so one repeat spans about this wall time (--target-time; e.g.
# 500us, 100ms, 2s; a bare number is ms). Read by `bench run` only.
# target_time: 100ms

# Where profile artifacts land (--profile-output-dir / --artifact-root)
profile_output_dir: bench-out

# Default --gtest_filter, given unless a --gtest_filter= follows `--`.
# Everything after its first minus excludes: skips noisy or blocking tests.
# gtest_filter: \"-*Large*:*PayloadScaling*\"

# Where the binary lookup walks when you pass `bench run Foo` (short name).
# Override only when the build layout differs from CMake's default. A
# relative root is relative to this file's folder; VERNIER_BENCH_BIN_ROOTS
# and VERNIER_BENCH_BIN_SUBDIRS take precedence.
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

/// Read and parse a `.bench.yaml` file. The parser is intentionally
/// minimal -- it reads top-level `key: value`, `key: ["a", "b"]`, `key:`
/// followed by `- item` lines, and `# comment` lines -- and avoids pulling in
/// a YAML crate so this tool keeps its build tree small. A file that cannot be
/// read, or that holds a problem (`parse`), is refused whole, each problem
/// named by line, key and what was expected: `bench run` and
/// `bench profile-all` stop on it before anything starts, and
/// `bench config-validate` reports the same problems (`validate`).
pub fn load(path: &Path) -> Result<Config, Error> {
    let (config, problems) = parse(&read(path)?);
    refuse(path, &problems)?;
    Ok(config)
}

/// The text of the file at @p path, or why it cannot be read.
fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("{} cannot be read: {e}", path.display())))
}

/// The refusal of the file at @p path for @p problems, the one error `load`
/// and `ValidationReport::verdict` give; `Ok` when there are none.
fn refuse(path: &Path, problems: &[String]) -> Result<(), Error> {
    if problems.is_empty() {
        return Ok(());
    }
    Err(Error::Config(format!(
        "{} cannot be used: {}",
        path.display(),
        problems.join("; ")
    )))
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

/// What a key takes. The keys are the named fields of `Config`; any other
/// top-level key is a project's own field, kept in `extras`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A whole number greater than 0: `cycles`, `repeats`.
    Count,
    /// A duration of at least 1us, as the benchmark's `--target-time` reads
    /// it: `target_time`.
    Duration,
    /// A path: `profile_output_dir`.
    Path,
    /// A single value: `gtest_filter`.
    Text,
    /// A list of values: `bin_roots`, `bin_subdirs`.
    List,
}

impl Kind {
    /// The kind of @p key, or `None` for a key `Config` does not name.
    fn of(key: &str) -> Option<Self> {
        match key {
            "cycles" | "repeats" => Some(Kind::Count),
            "target_time" => Some(Kind::Duration),
            "profile_output_dir" => Some(Kind::Path),
            "gtest_filter" => Some(Kind::Text),
            "bin_roots" | "bin_subdirs" => Some(Kind::List),
            _ => None,
        }
    }

    /// What a value of this kind looks like, for a problem's message.
    fn expected(self) -> &'static str {
        match self {
            Kind::Count => "a whole number greater than 0",
            Kind::Duration => {
                "a duration of at least 1us (a number followed by us, ms or s; a bare number \
                 is milliseconds)"
            }
            Kind::Path => "a path",
            Kind::Text => "a single value",
            Kind::List => "a list ([\"a\", \"b\"], or one '- item' line each)",
        }
    }
}

/// A key's value as written.
enum Value {
    /// `key: value`.
    One(String),
    /// `key: [a, b]`, or `key:` and its `- item` lines.
    List(Vec<String>),
    /// `key:` with no `- item` line below it: no value, as YAML reads it.
    Unset,
}

/// The configuration a parse builds, and the problems it finds on the way.
#[derive(Default)]
struct Reader {
    config: Config,
    problems: Vec<String>,
    /// Each key read so far, and the line it was first given on.
    seen: BTreeMap<String, usize>,
    /// The open `key:` line with no value: its key, its line, and the
    /// `- item` lines read below it so far.
    block: Option<(String, usize, Vec<String>)>,
}

/// Parse the text of a `.bench.yaml`: the configuration, and every problem
/// that keeps it from being used, in line order, each "line N: <key>:
/// expected <what>, got <what>" or what is wrong with the line itself. A
/// top-level key `Config` does not name is no problem: it is a project's own
/// field, kept in `extras` (`validate` warns about it).
fn parse(text: &str) -> (Config, Vec<String>) {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut reader = Reader::default();
    for (index, raw) in text.lines().enumerate() {
        reader.line(index + 1, strip_comment(raw));
    }
    reader.end_block();
    (reader.config, reader.problems)
}

impl Reader {
    /// Read @p line, line @p number of the file, its comment removed.
    fn line(&mut self, number: usize, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() || (trimmed == "---" && self.seen.is_empty() && self.block.is_none())
        {
            return;
        }
        if trimmed == "-" || trimmed.starts_with("- ") {
            let Some((key, _, items)) = self.block.as_mut() else {
                self.problems.push(format!(
                    "line {number}: expected a 'key:' line above this '- item', got none"
                ));
                return;
            };
            match unquote(&trimmed[1..]) {
                Ok("") => {}
                Ok(item) => items.push(item.to_string()),
                Err(got) => {
                    let problem =
                        format!("line {number}: {key}: expected a value that {CLOSED}, got {got}");
                    self.problems.push(problem);
                }
            }
            return;
        }
        self.end_block();
        if line.starts_with(char::is_whitespace) {
            self.problems.push(format!(
                "line {number}: expected 'key: value' at the start of the line or a '- item', \
                 got the indented '{trimmed}'"
            ));
            return;
        }
        let Some((key, val)) = line.split_once(':') else {
            self.problems.push(format!(
                "line {number}: expected 'key: value', got '{trimmed}'"
            ));
            return;
        };
        let key = match unquote(key) {
            Ok("") => {
                self.problems.push(format!(
                    "line {number}: expected 'key: value', got no key before ':'"
                ));
                return;
            }
            Ok(key) => key,
            Err(got) => {
                self.problems.push(format!(
                    "line {number}: expected a key that {CLOSED}, got {got}"
                ));
                return;
            }
        };
        let val = val.trim();
        if val.is_empty() {
            self.block = Some((key.to_string(), number, Vec::new()));
            return;
        }
        let value = match val.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
            Some(inner) => inline_items(inner).map(Value::List),
            None => unquote(val).map(|text| Value::One(text.to_string())),
        };
        match value {
            Ok(value) => self.assign(key, number, value),
            Err(got) => self.problems.push(format!(
                "line {number}: {key}: expected a value that {CLOSED}, got {got}"
            )),
        }
    }

    /// Assign the open `key:` line's key, if any: its items, or no value
    /// without one.
    fn end_block(&mut self) {
        if let Some((key, at, items)) = self.block.take() {
            let value = if items.is_empty() {
                Value::Unset
            } else {
                Value::List(items)
            };
            self.assign(&key, at, value);
        }
    }

    /// Set @p key, given on line @p number, to @p value, or note the problem
    /// that keeps it from being set.
    fn assign(&mut self, key: &str, number: usize, value: Value) {
        if let Some(first) = self.seen.get(key) {
            self.problems.push(format!(
                "line {number}: {key}: expected each key once, got it again (first on line {first})"
            ));
            return;
        }
        self.seen.insert(key.to_string(), number);
        let Some(kind) = Kind::of(key) else {
            let text = match value {
                Value::One(text) => text,
                Value::List(items) => format!("[{}]", items.join(", ")),
                Value::Unset => String::new(),
            };
            self.config.extras.insert(key.to_string(), text);
            return;
        };
        let got = match value {
            Value::Unset => return,
            Value::List(items) if kind == Kind::List => {
                match key {
                    "bin_roots" => self.config.bin_roots = items,
                    _ => self.config.bin_subdirs = items,
                }
                return;
            }
            Value::List(_) => "a list".to_string(),
            Value::One(text) => match self.set_one(key, kind, text) {
                Ok(()) => return,
                Err(got) => got,
            },
        };
        self.problems.push(format!(
            "line {number}: {key}: expected {}, got {got}",
            kind.expected()
        ));
    }

    /// Set @p key, of @p kind, to the single value @p text; what was given,
    /// for the problem's message, when it is not a value of that kind.
    fn set_one(&mut self, key: &str, kind: Kind, text: String) -> Result<(), String> {
        if text.is_empty() && kind != Kind::Text {
            return Err("an empty value".to_string());
        }
        let config = &mut self.config;
        match kind {
            Kind::Count => match text.parse::<u32>() {
                Ok(count) if count > 0 && key == "cycles" => config.cycles = Some(count),
                Ok(count) if count > 0 => config.repeats = Some(count),
                _ => return Err(format!("'{text}'")),
            },
            Kind::Duration if duration_us(&text).is_some() => config.target_time = Some(text),
            Kind::Path => config.profile_output_dir = Some(PathBuf::from(text)),
            Kind::Text => config.gtest_filter = Some(text),
            Kind::Duration | Kind::List => return Err(format!("'{text}'")),
        }
        Ok(())
    }
}

/// @p text as the benchmark's `--target-time` reads it (`parseDurationUs` in
/// `PerfConfig.hpp`): a number followed by `us`, `ms` or `s`, or a bare
/// number of milliseconds, that comes to at least one whole microsecond.
/// Those whole microseconds, or `None` when the benchmark would not take it.
fn duration_us(text: &str) -> Option<u64> {
    let (number, scale) = if let Some(number) = text.strip_suffix("us") {
        (number, 1.0)
    } else if let Some(number) = text.strip_suffix("ms") {
        (number, 1_000.0)
    } else if let Some(number) = text.strip_suffix('s') {
        (number, 1_000_000.0)
    } else {
        (text, 1_000.0)
    };
    let us = number.parse::<f64>().ok()? * scale;
    (us.is_finite() && us >= 1.0).then_some(us as u64)
}

/// What a quoted key or value must do, for a problem's message.
const CLOSED: &str = "closes the quote it opens";

/// @p line without its comment: from a `#` that starts it or follows a space
/// or tab, as YAML reads one (the parser does not look into quotes).
fn strip_comment(line: &str) -> &str {
    let mut after_blank = true;
    for (at, c) in line.char_indices() {
        if c == '#' && after_blank {
            return &line[..at];
        }
        after_blank = c == ' ' || c == '\t';
    }
    line
}

/// @p s, trimmed, without the quotes around it; what was given, for the
/// problem's message, when it opens a quote it does not close.
fn unquote(s: &str) -> Result<&str, String> {
    let t = s.trim();
    match t.chars().next() {
        Some(quote @ ('"' | '\'')) if t.len() < 2 || !t.ends_with(quote) => Err(format!("'{t}'")),
        Some('"' | '\'') => Ok(&t[1..t.len() - 1]),
        _ => Ok(t),
    }
}

/// The items of an inline list, @p inner its text between `[` and `]`; an
/// empty item is dropped.
fn inline_items(inner: &str) -> Result<Vec<String>, String> {
    let mut items = Vec::new();
    for item in inner.split(',') {
        match unquote(item)? {
            "" => {}
            item => items.push(item.to_string()),
        }
    }
    Ok(items)
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

/// Check the file at @p path: each problem `load` would refuse it for, as an
/// error, then the warnings about what it holds. A file that cannot be read
/// is the same error `load` gives. Does not shell out to anything.
pub fn validate(path: &Path) -> Result<ValidationReport, Error> {
    let (cfg, problems) = parse(&read(path)?);
    let mut findings: Vec<Finding> = problems
        .into_iter()
        .map(|message| Finding {
            severity: Severity::Error,
            message,
        })
        .collect();

    if let Some(r) = cfg.repeats {
        if r < 5 {
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

impl ValidationReport {
    /// How `bench config-validate` ends: the error `load` gives this file
    /// when the report holds an error, else `Ok`.
    pub fn verdict(&self) -> Result<(), Error> {
        let problems: Vec<String> = self
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .map(|f| f.message.clone())
            .collect();
        refuse(&self.path, &problems)
    }
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

    /// The configuration @p text holds, which must hold no problem.
    fn read_ok(text: &str) -> Config {
        let (config, problems) = parse(text);
        assert!(problems.is_empty(), "{problems:?}");
        config
    }

    /// The problems @p text holds.
    fn problems(text: &str) -> Vec<String> {
        parse(text).1
    }

    /// @test Minimal key/value pairs parse into typed fields.
    #[test]
    fn parse_minimal() {
        let cfg = read_ok("cycles: 5000\nrepeats: 20\n");
        assert_eq!(cfg.cycles, Some(5000));
        assert_eq!(cfg.repeats, Some(20));
    }

    /// @test Inline `[a, b]` list syntax populates the matching field.
    #[test]
    fn parse_inline_list() {
        let cfg = read_ok("bin_roots: [\"build/*\", \"out/*\"]\n");
        assert_eq!(cfg.bin_roots, vec!["build/*", "out/*"]);
    }

    /// @test Block list syntax (`key:` followed by `- item`) populates the
    /// matching field, items at the key's indentation or below it, a blank
    /// line or a comment among them.
    #[test]
    fn parse_block_list() {
        let cfg = read_ok("bin_subdirs:\n  - bin/ptests\n\n  # more\n  - bin/tests\n");
        assert_eq!(cfg.bin_subdirs, vec!["bin/ptests", "bin/tests"]);
        let cfg = read_ok("bin_roots:\n- rel\n- 'dbg'\ncycles: 3\n");
        assert_eq!(cfg.bin_roots, vec!["rel", "dbg"]);
        assert_eq!(cfg.cycles, Some(3));
    }

    /// @test target_time is a key of its own, kept as written for the
    /// benchmark to read, and not an extra.
    #[test]
    fn parse_target_time() {
        let cfg = read_ok("target_time: 250ms\ncycles: 5\n");
        assert_eq!(cfg.target_time.as_deref(), Some("250ms"));
        assert_eq!(cfg.cycles, Some(5));
        assert!(cfg.extras.is_empty(), "{:?}", cfg.extras);
    }

    /// @test Surrounding quotes are stripped from keys and scalar values.
    #[test]
    fn parse_quoted_strings_strip() {
        let cfg = read_ok("gtest_filter: \"-*Large*\"\n'repeats': 7\n");
        assert_eq!(cfg.gtest_filter.as_deref(), Some("-*Large*"));
        assert_eq!(cfg.repeats, Some(7));
    }

    /// @test A key a project adds is kept in `Config::extras` whatever it
    /// holds, and is no problem.
    #[test]
    fn parse_preserves_extras() {
        let cfg = read_ok("future_knob: 42\nowners: [a, b]\nlanes:\n  - fast\n  - slow\nnotes:\n");
        assert_eq!(cfg.extras.get("future_knob"), Some(&"42".to_string()));
        assert_eq!(cfg.extras.get("owners"), Some(&"[a, b]".to_string()));
        assert_eq!(cfg.extras.get("lanes"), Some(&"[fast, slow]".to_string()));
        assert_eq!(cfg.extras.get("notes"), Some(&String::new()));
    }

    /// @test Full-line and trailing `#` comments are stripped; a leading
    /// document marker and a byte order mark are no problem.
    #[test]
    fn comments_ignored() {
        let cfg = read_ok("# comment\ncycles: 100 # trailing\n");
        assert_eq!(cfg.cycles, Some(100));
        let cfg = read_ok("\u{feff}---\n# project\nrepeats: 4\n");
        assert_eq!(cfg.repeats, Some(4));
    }

    /// @test The scaffold `bench init` writes holds no problem, nor does it
    /// with each of its commented keys given.
    #[test]
    fn the_scaffold_holds_no_problem() {
        let cfg = read_ok(DEFAULT_TEMPLATE);
        assert_eq!((cfg.cycles, cfg.repeats), (Some(10000), Some(10)));
        assert_eq!(cfg.profile_output_dir, Some(PathBuf::from("bench-out")));
        let given = DEFAULT_TEMPLATE
            .lines()
            .map(|line| match line.strip_prefix("# ") {
                Some(rest) if rest.starts_with("  - ") => rest,
                Some(rest)
                    if rest
                        .split_once(':')
                        .is_some_and(|(k, _)| Kind::of(k).is_some()) =>
                {
                    rest
                }
                _ => line,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let cfg = read_ok(&given);
        assert_eq!(cfg.target_time.as_deref(), Some("100ms"));
        assert_eq!(
            cfg.gtest_filter.as_deref(),
            Some("-*Large*:*PayloadScaling*")
        );
        assert_eq!(cfg.bin_roots, vec!["build/*"]);
        assert_eq!(
            cfg.bin_subdirs,
            vec!["bin/ptests", "bin/tests", "bin/examples"]
        );
        assert!(cfg.extras.is_empty(), "{:?}", cfg.extras);
    }

    /// @test A named key with no value, `key:` and no item below it, is
    /// unset, as YAML reads it.
    #[test]
    fn a_named_key_with_no_value_is_unset() {
        let cfg = read_ok("cycles:\nbin_roots:\ngtest_filter:\n");
        assert_eq!(cfg.cycles, None);
        assert!(cfg.bin_roots.is_empty());
        assert_eq!(cfg.gtest_filter, None);
        assert!(cfg.extras.is_empty(), "{:?}", cfg.extras);
    }

    /// @test A value of the wrong kind for a named key is a problem that
    /// names its line, the key, what the key takes and what was given.
    #[test]
    fn a_value_of_the_wrong_kind_is_a_problem_naming_line_key_and_expectation() {
        let count = "a whole number greater than 0";
        let duration = "a duration of at least 1us (a number followed by us, ms or s; a bare \
                        number is milliseconds)";
        let list = "a list ([\"a\", \"b\"], or one '- item' line each)";
        let cases = [
            (
                "cycles: many",
                format!("cycles: expected {count}, got 'many'"),
            ),
            ("cycles: 0", format!("cycles: expected {count}, got '0'")),
            (
                "repeats: -3",
                format!("repeats: expected {count}, got '-3'"),
            ),
            (
                "repeats: 2.5",
                format!("repeats: expected {count}, got '2.5'"),
            ),
            (
                "cycles: [1, 2]",
                format!("cycles: expected {count}, got a list"),
            ),
            (
                "cycles: []",
                format!("cycles: expected {count}, got a list"),
            ),
            (
                "cycles: \"\"",
                format!("cycles: expected {count}, got an empty value"),
            ),
            (
                "target_time: fast",
                format!("target_time: expected {duration}, got 'fast'"),
            ),
            (
                "target_time: 0ms",
                format!("target_time: expected {duration}, got '0ms'"),
            ),
            (
                "target_time: 0.5us",
                format!("target_time: expected {duration}, got '0.5us'"),
            ),
            (
                "target_time: 5 sec",
                format!("target_time: expected {duration}, got '5 sec'"),
            ),
            (
                "profile_output_dir: ''",
                "profile_output_dir: expected a path, got an empty value".to_string(),
            ),
            (
                "profile_output_dir: [a]",
                "profile_output_dir: expected a path, got a list".to_string(),
            ),
            (
                "gtest_filter: [A.*, B.*]",
                "gtest_filter: expected a single value, got a list".to_string(),
            ),
            (
                "bin_roots: build/*",
                format!("bin_roots: expected {list}, got 'build/*'"),
            ),
            (
                "bin_subdirs: bin",
                format!("bin_subdirs: expected {list}, got 'bin'"),
            ),
        ];
        for (line, expected) in cases {
            let found = problems(&format!("# project\n{line}\n"));
            assert_eq!(found, vec![format!("line 2: {expected}")], "for {line:?}");
        }
        let found = problems("cycles:\n  - 1\n  - 2\n");
        assert_eq!(
            found,
            vec![format!("line 1: cycles: expected {count}, got a list")]
        );
    }

    /// @test A line the parser cannot place is a problem, not skipped, and
    /// every problem in the file is listed, in line order.
    #[test]
    fn a_line_the_parser_cannot_read_is_a_problem() {
        let found = problems(
            "cycles 100\n- build\nci:\n  owner: team\n: 3\nrepeats: 9\nrepeats: 10\n---\n",
        );
        assert_eq!(
            found,
            vec![
                "line 1: expected 'key: value', got 'cycles 100'".to_string(),
                "line 2: expected a 'key:' line above this '- item', got none".to_string(),
                "line 4: expected 'key: value' at the start of the line or a '- item', got the \
                 indented 'owner: team'"
                    .to_string(),
                "line 5: expected 'key: value', got no key before ':'".to_string(),
                "line 7: repeats: expected each key once, got it again (first on line 6)"
                    .to_string(),
                "line 8: expected 'key: value', got '---'".to_string(),
            ]
        );
        // The first of a key given twice is the one read.
        assert_eq!(parse("repeats: 9\nrepeats: 10\n").0.repeats, Some(9));
    }

    /// @test A `#` starts a comment at the start of a line or after a blank,
    /// as YAML reads one, and a quote opened around a key, a value or an item
    /// must be closed: otherwise the line is a problem, not read with the
    /// quote in it (nor a panic for a lone quote).
    #[test]
    fn comments_follow_a_blank_and_quotes_close() {
        let cfg = read_ok("profile_output_dir: out#1 # where\ngtest_filter: \"A#B\" # c\n");
        assert_eq!(cfg.profile_output_dir, Some(PathBuf::from("out#1")));
        assert_eq!(cfg.gtest_filter.as_deref(), Some("A#B"));
        let closes = "closes the quote it opens";
        let cases = [
            (
                "gtest_filter: \"-*Large*\n",
                format!("line 1: gtest_filter: expected a value that {closes}, got '\"-*Large*'"),
            ),
            (
                "gtest_filter: \"\n",
                format!("line 1: gtest_filter: expected a value that {closes}, got '\"'"),
            ),
            (
                "bin_roots:\n  - 'build/*\n",
                format!("line 2: bin_roots: expected a value that {closes}, got ''build/*'"),
            ),
            (
                "bin_roots: [\"a\", 'b]\n",
                format!("line 1: bin_roots: expected a value that {closes}, got ''b'"),
            ),
            (
                "\"cycles: 5\n",
                format!("line 1: expected a key that {closes}, got '\"cycles'"),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(problems(text), vec![expected], "for {text:?}");
        }
    }

    /// @test A duration is read as the benchmark's `--target-time` reads it.
    #[test]
    fn durations_are_read_as_the_benchmark_reads_them() {
        let read = [
            ("500us", 500),
            ("100ms", 100_000),
            ("2s", 2_000_000),
            ("1.5s", 1_500_000),
            ("250", 250_000),
            ("1e3us", 1_000),
            ("1.9us", 1),
        ];
        for (text, us) in read {
            assert_eq!(duration_us(text), Some(us), "for {text:?}");
        }
        let refused = [
            "0", "0ms", "0.5us", "-1s", "us", "ms", "s", "5 ms", "5MS", "fast", "inf", "nan",
        ];
        for text in refused {
            assert_eq!(duration_us(text), None, "for {text:?}");
        }
    }

    /// @test `validate` reports as errors exactly the problems `load`
    /// refuses the file for, and its verdict is the error `load` gives; the
    /// warnings alone leave the file usable.
    #[test]
    fn validate_reports_and_refuses_what_load_refuses() {
        let dir = super::super::tempfile_dir("config_validate_verdict");
        let path = dir.join(".bench.yaml");
        fs::write(
            &path,
            "cycles: many\nrepeats: 2\nbin_roots: build\nmine: 1\n",
        )
        .unwrap();

        let report = validate(&path).expect("a readable file is reported on");
        let errors: Vec<&str> = report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .map(|f| f.message.as_str())
            .collect();
        assert_eq!(errors, problems(&fs::read_to_string(&path).unwrap()));
        assert_eq!(errors.len(), 2, "{errors:?}");
        let warnings = report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .count();
        assert_eq!(warnings, 2, "{:?}", report.findings);
        let refused = load(&path)
            .expect_err("the file holds problems")
            .to_string();
        assert_eq!(
            report
                .verdict()
                .expect_err("the report holds errors")
                .to_string(),
            refused
        );
        assert!(
            refused.starts_with(&format!(
                "{} cannot be used: line 1: cycles:",
                path.display()
            )),
            "{refused}"
        );

        fs::write(&path, "repeats: 2\nmine: 1\n").unwrap();
        let report = validate(&path).expect("a readable file is reported on");
        assert!(report.verdict().is_ok(), "{:?}", report.findings);
        assert_eq!(
            load(&path).expect("warnings alone do not refuse").repeats,
            Some(2)
        );

        fs::write(&path, b"cycles: \xff\n").unwrap();
        let unread = load(&path).expect_err("not text").to_string();
        assert!(
            unread.starts_with(&format!("{} cannot be read: ", path.display())),
            "{unread}"
        );
        assert_eq!(validate(&path).expect_err("not text").to_string(), unread);
        fs::remove_dir_all(&dir).unwrap();
    }
}
