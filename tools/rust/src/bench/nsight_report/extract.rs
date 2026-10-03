//! The extraction workflow: the inputs to reports, each report to rows, the
//! rows to one CSV, and the command around them.
//!
//! Inputs are taken in the order given; a directory stands for every
//! `.nsys-rep` under it, then every `.ncu-rep`, each sorted by path. Hidden
//! entries are searched; a symbolic link to a directory is not followed. Paths
//! are shown and handed to the tools as given, without `.` components or a
//! trailing `/`. All the inputs are resolved before any tool runs, so an
//! output path that is one of the reports is refused before anything is read.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::exec::{run_tool, Interrupt, Interrupts, ToolRun};
use super::parse::{self, Row};
use super::{COMMAND, NSYS_SUMMARIES};
use crate::bench::Error;

/* ----------------------------- Constants ----------------------------- */

/// The columns every CSV starts with, in this order; the others follow,
/// sorted by name.
const LEADING_COLUMNS: [&str; 7] = [
    "source",
    "report",
    "kernel",
    "instances",
    "time_total_ns",
    "time_avg_ns",
    "time_pct",
];

/* ----------------------------- Types ----------------------------- */

/// How a run of the command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunEnd {
    /// Every requested input was read.
    AllRead,
    /// An input was not read; its error is printed, and the CSV holds the rows
    /// of the others.
    SomeUnread,
    /// A signal stopped the run before the CSV was written; nothing was.
    Interrupted(Interrupt),
}

/// Which tool reads a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Nsight Systems, `.nsys-rep`.
    Systems,
    /// Nsight Compute, `.ncu-rep`.
    Compute,
}

/// What one input stands for, in the order its lines are reported: problems
/// found while resolving it, then its reports.
#[derive(Debug, Default)]
struct Input {
    problems: Vec<String>,
    reports: Vec<(PathBuf, Kind)>,
}

/// What the reports yielded.
#[derive(Debug, Default)]
struct Extraction {
    rows: Vec<Row>,
    warnings: Vec<String>,
    errors: Vec<String>,
}

/// A directory of this process's own, for one report's export, removed when
/// the export has been read.
struct PrivateDir {
    path: Option<PathBuf>,
}

/* ----------------------------- Helpers ----------------------------- */

/// @p path as it is shown and handed to the tools: without `.` components or
/// a trailing `/`, and `.` when nothing else is left.
fn normalized(path: &Path) -> PathBuf {
    let clean: PathBuf = path
        .components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();
    if clean.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        clean
    }
}

/// The tool that reads @p path, by its name's ending.
fn kind_of(path: &Path) -> Option<Kind> {
    match path.extension()?.to_str()? {
        "nsys-rep" => Some(Kind::Systems),
        "ncu-rep" => Some(Kind::Compute),
        _ => None,
    }
}

/// Collect the reports under @p dir into @p systems and @p compute.
fn walk(
    dir: &Path,
    systems: &mut Vec<PathBuf>,
    compute: &mut Vec<PathBuf>,
    problems: &mut Vec<String>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            problems.push(format!("cannot read directory {}: {e}", dir.display()));
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk(&path, systems, compute, problems);
            continue;
        }
        // A link named like a report counts unless it leads to a directory;
        // one that leads nowhere is left for the tool to report.
        if file_type.is_symlink() && fs::metadata(&path).is_ok_and(|m| m.is_dir()) {
            continue;
        }
        match kind_of(&path) {
            Some(Kind::Systems) => systems.push(path),
            Some(Kind::Compute) => compute.push(path),
            None => {}
        }
    }
}

/// What @p given stands for.
fn resolve(given: &Path) -> Input {
    let path = normalized(given);
    let mut input = Input::default();
    match fs::metadata(&path) {
        Ok(meta) if meta.is_dir() => {
            let (mut systems, mut compute) = (Vec::new(), Vec::new());
            walk(&path, &mut systems, &mut compute, &mut input.problems);
            if systems.is_empty() && compute.is_empty() {
                input.problems.push(format!(
                    "no .nsys-rep or .ncu-rep file under {}",
                    path.display()
                ));
            }
            systems.sort();
            compute.sort();
            input
                .reports
                .extend(systems.into_iter().map(|p| (p, Kind::Systems)));
            input
                .reports
                .extend(compute.into_iter().map(|p| (p, Kind::Compute)));
        }
        Ok(meta) => match kind_of(&path) {
            Some(kind) if meta.is_file() => input.reports.push((path, kind)),
            _ => input.problems.push(format!(
                "not an .nsys-rep, an .ncu-rep or a directory: {}",
                path.display()
            )),
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => input
            .problems
            .push(format!("no such file or directory: {}", path.display())),
        Err(e) => input
            .problems
            .push(format!("cannot read {}: {e}", path.display())),
    }
    input
}

/// Why a tool command did not give its output, in words.
fn cause(tool: &str, run: &ToolRun, timeout: Duration) -> String {
    let secs = timeout.as_secs();
    match run {
        ToolRun::Finished {
            status,
            stdout,
            stderr,
        } => {
            let detail = parse::last_line(&[stderr, stdout]);
            match status.code() {
                Some(code) => format!("exit status {code}: {detail}"),
                None => format!("ended by signal {}: {detail}", signal_of(status)),
            }
        }
        ToolRun::NotFound => format!("{tool} not found on PATH"),
        ToolRun::NotStarted(e) => format!("{tool} could not be started: {e}"),
        ToolRun::WaitFailed(e) => format!("{tool} could not be waited for and was stopped: {e}"),
        ToolRun::TimedOut => format!("{tool} did not finish within {secs} s and was stopped"),
        ToolRun::OutputHeldOpen => {
            format!("{tool} ended, but a process it started kept its output open past {secs} s")
        }
        ToolRun::Interrupted(signal) => format!("stopped by {signal}"),
    }
}

#[cfg(unix)]
fn signal_of(status: &std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status.signal().unwrap_or(0)
}

#[cfg(not(unix))]
fn signal_of(_status: &std::process::ExitStatus) -> i32 {
    0
}

impl PrivateDir {
    /// A new directory under the temporary directory, readable only by this
    /// user; an existing one is never reused.
    fn create() -> io::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir();
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("vernier-{COMMAND}-{}-{n}", std::process::id()));
            match create_private(&path) {
                Ok(()) => return Ok(Self { path: Some(path) }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists && n < 1000 => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn path(&self) -> &Path {
        self.path.as_deref().unwrap_or(Path::new(""))
    }

    /// Remove the directory and what it holds.
    fn remove(mut self) -> io::Result<()> {
        match self.path.take() {
            Some(path) => fs::remove_dir_all(path),
            None => Ok(()),
        }
    }
}

impl Drop for PrivateDir {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

#[cfg(unix)]
fn create_private(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

/// Export one `.nsys-rep` into a private directory and read every summary of
/// `NSYS_SUMMARIES` from that export. The report's own directory is not
/// written.
fn read_systems(
    report: &Path,
    timeout: Duration,
    interrupts: &Interrupts<'_>,
    out: &mut Extraction,
) -> Result<(), Interrupt> {
    let dir = match PrivateDir::create() {
        Ok(dir) => dir,
        Err(e) => {
            out.errors.push(format!(
                "cannot create a private directory for the export of {}: {e}",
                report.display()
            ));
            return Ok(());
        }
    };
    let stem = report.file_stem().unwrap_or(OsStr::new("report"));
    let mut name = stem.to_os_string();
    name.push(".sqlite");
    let export = dir.path().join(name);
    let read = read_export(report, &export, timeout, interrupts, out);
    let removed_from = dir.path().to_path_buf();
    if let Err(e) = dir.remove() {
        out.warnings
            .push(format!("could not remove {}: {e}", removed_from.display()));
    }
    read
}

fn read_export(
    report: &Path,
    export: &Path,
    timeout: Duration,
    interrupts: &Interrupts<'_>,
    out: &mut Extraction,
) -> Result<(), Interrupt> {
    let args = [
        OsStr::new("export"),
        OsStr::new("--type"),
        OsStr::new("sqlite"),
        OsStr::new("--force-overwrite"),
        OsStr::new("true"),
        OsStr::new("-o"),
        export.as_os_str(),
        report.as_os_str(),
    ];
    match run_tool("nsys", &args, timeout, interrupts) {
        ToolRun::Finished {
            status,
            stdout,
            stderr,
        } if status.success() => {
            if !export.is_file() {
                out.errors.push(format!(
                    "nsys export failed for {}: no export was written ({})",
                    report.display(),
                    parse::last_line(&[&stdout, &stderr])
                ));
                return Ok(());
            }
        }
        ToolRun::Interrupted(signal) => return Err(signal),
        failed => {
            out.errors.push(format!(
                "nsys export failed for {}: {}",
                report.display(),
                cause("nsys", &failed, timeout)
            ));
            return Ok(());
        }
    }
    for summary in NSYS_SUMMARIES {
        let args = [
            OsStr::new("stats"),
            OsStr::new("--report"),
            OsStr::new(summary),
            OsStr::new("--format"),
            OsStr::new("csv"),
            export.as_os_str(),
        ];
        match run_tool("nsys", &args, timeout, interrupts) {
            ToolRun::Finished {
                status,
                stdout,
                stderr,
            } if status.success() => match parse::nsys_rows(summary, &stdout) {
                Ok(rows) => {
                    if rows.is_empty() {
                        out.warnings.push(format!(
                            "{summary} has no rows for {}: {}",
                            report.display(),
                            parse::last_line(&[&stdout, &stderr])
                        ));
                    }
                    out.rows.extend(rows);
                }
                Err(malformed) => out.errors.push(format!(
                    "nsys stats --report {summary} failed for {}: {malformed}",
                    report.display()
                )),
            },
            ToolRun::Interrupted(signal) => return Err(signal),
            failed => out.errors.push(format!(
                "nsys stats --report {summary} failed for {}: {}",
                report.display(),
                cause("nsys", &failed, timeout)
            )),
        }
    }
    Ok(())
}

/// Read one `.ncu-rep` through `ncu --import`.
fn read_compute(
    report: &Path,
    timeout: Duration,
    interrupts: &Interrupts<'_>,
    out: &mut Extraction,
) -> Result<(), Interrupt> {
    let args = [
        OsStr::new("--import"),
        report.as_os_str(),
        OsStr::new("--csv"),
        OsStr::new("--print-summary"),
        OsStr::new("per-kernel"),
    ];
    match run_tool("ncu", &args, timeout, interrupts) {
        ToolRun::Finished {
            status,
            stdout,
            stderr,
        } if status.success() => match parse::ncu_rows(&stdout) {
            Ok(rows) => {
                if rows.is_empty() {
                    out.warnings.push(format!(
                        "no kernel in {}: {}",
                        report.display(),
                        parse::last_line(&[&stdout, &stderr])
                    ));
                }
                out.rows.extend(rows);
            }
            Err(malformed) => out.errors.push(format!(
                "ncu --import failed for {}: {malformed}",
                report.display()
            )),
        },
        ToolRun::Interrupted(signal) => return Err(signal),
        failed => out.errors.push(format!(
            "ncu --import failed for {}: {}",
            report.display(),
            cause("ncu", &failed, timeout)
        )),
    }
    Ok(())
}

/// Read every report of @p inputs in order, reporting each input's problems
/// where they occur.
fn extract(
    inputs: &[Input],
    timeout: Duration,
    interrupts: &Interrupts<'_>,
) -> Result<Extraction, Interrupt> {
    let mut out = Extraction::default();
    for input in inputs {
        out.errors.extend(input.problems.iter().cloned());
        for (report, kind) in &input.reports {
            if let Some(signal) = interrupts.pending() {
                return Err(signal);
            }
            // An empty file is no report; no tool is asked to read it.
            if fs::metadata(report).is_ok_and(|m| m.is_file() && m.len() == 0) {
                out.errors.push(format!("{} is empty", report.display()));
                continue;
            }
            match kind {
                Kind::Systems => read_systems(report, timeout, interrupts, &mut out)?,
                Kind::Compute => read_compute(report, timeout, interrupts, &mut out)?,
            }
        }
    }
    Ok(out)
}

/// The CSV's columns for @p rows: the leading ones, then every other column a
/// row has, sorted by name.
fn columns(rows: &[Row]) -> Vec<&str> {
    let others: BTreeSet<&str> = rows
        .iter()
        .flat_map(Row::columns)
        .filter(|column| !LEADING_COLUMNS.contains(column))
        .collect();
    LEADING_COLUMNS.iter().copied().chain(others).collect()
}

/// Write @p rows to @p path: the columns of `columns`, CRLF line ends, quotes
/// only around a field that holds a comma, a quote or a line break. Without
/// rows the file is created empty.
fn write_csv(rows: &[Row], path: &Path) -> io::Result<()> {
    let file = fs::File::create(path)?;
    if rows.is_empty() {
        return Ok(());
    }
    let columns = columns(rows);
    let to_io = |e: csv::Error| match e.into_kind() {
        csv::ErrorKind::Io(e) => e,
        other => io::Error::other(format!("{other:?}")),
    };
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(io::BufWriter::new(file));
    writer.write_record(&columns).map_err(to_io)?;
    for row in rows {
        writer
            .write_record(columns.iter().map(|c| row.get(c).unwrap_or("")))
            .map_err(to_io)?;
    }
    writer.flush()
}

/* ----------------------------- API ----------------------------- */

/// The command: read @p inputs, each tool command bounded by @p timeout, and
/// write their rows to @p csv. Warnings and errors go to stderr, one line
/// each, then the count of rows written to stdout. While it runs, SIGINT and
/// SIGTERM stop it: the tool is stopped, its private directory removed, and
/// nothing written.
pub fn run(inputs: &[PathBuf], csv: &Path, timeout: Duration) -> Result<RunEnd, Error> {
    let interrupts = Interrupts::watch_signals();
    let resolved: Vec<Input> = inputs.iter().map(|input| resolve(input)).collect();
    if let Ok(target) = fs::canonicalize(csv) {
        let reports = resolved.iter().flat_map(|input| &input.reports);
        for (report, _) in reports {
            if fs::canonicalize(report).is_ok_and(|r| r == target) {
                return Err(Error::InvalidArgs(format!(
                    "--csv {} is one of the reports to read; name another file",
                    normalized(csv).display()
                )));
            }
        }
    }
    let extraction = match extract(&resolved, timeout, &interrupts) {
        Ok(extraction) => extraction,
        Err(signal) => return Ok(stopped(signal)),
    };
    if let Some(signal) = interrupts.pending() {
        return Ok(stopped(signal));
    }
    let written = write_csv(&extraction.rows, csv);
    for warning in &extraction.warnings {
        eprintln!("[{COMMAND}] warning: {warning}");
    }
    for error in &extraction.errors {
        eprintln!("[{COMMAND}] error: {error}");
    }
    let shown = normalized(csv);
    written.map_err(|e| {
        Error::Io(io::Error::new(
            e.kind(),
            format!("cannot write {}: {e}", shown.display()),
        ))
    })?;
    println!(
        "[{COMMAND}] wrote {} rows to {}",
        extraction.rows.len(),
        shown.display()
    );
    let _ = io::stdout().flush();
    Ok(if extraction.errors.is_empty() {
        RunEnd::AllRead
    } else {
        RunEnd::SomeUnread
    })
}

fn stopped(signal: Interrupt) -> RunEnd {
    eprintln!("[{COMMAND}] stopped by {signal}; nothing was written");
    RunEnd::Interrupted(signal)
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: &[(&str, &str)]) -> Row {
        let mut row = Row::default();
        for (column, value) in cells {
            row.set(column, *value);
        }
        row
    }

    /// @test The leading columns come first, in order, then the others sorted.
    #[test]
    fn known_columns_first() {
        let rows = [
            row(&[("source", "nsys"), ("zeta", "1"), ("kernel", "k")]),
            row(&[("Med (ns)", "2"), ("alpha", "3"), ("time_pct", "4")]),
        ];
        assert_eq!(
            columns(&rows),
            [
                "source",
                "report",
                "kernel",
                "instances",
                "time_total_ns",
                "time_avg_ns",
                "time_pct",
                "Med (ns)",
                "alpha",
                "zeta"
            ]
        );
    }

    /// @test Without rows the CSV is created empty, replacing what was there.
    #[test]
    fn no_rows_writes_an_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        fs::write(&path, "old").unwrap();
        write_csv(&[], &path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"");
    }

    /// @test Lines end in CRLF and only fields with a comma, a quote or a line
    /// break are quoted; a missing cell is empty.
    #[test]
    fn csv_line_ends_and_quoting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.csv");
        let rows = [
            row(&[
                ("source", "ncu"),
                ("kernel", "f(int, float)"),
                ("b", "say \"x\""),
            ]),
            row(&[("source", "nsys"), ("a", "two\nlines"), ("b", " spaced ")]),
        ];
        write_csv(&rows, &path).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "source,report,kernel,instances,time_total_ns,time_avg_ns,time_pct,a,b\r\n\
             ncu,,\"f(int, float)\",,,,,,\"say \"\"x\"\"\"\r\n\
             nsys,,,,,,,\"two\nlines\", spaced \r\n"
        );
    }

    /// @test Paths are shown without `.` components or a trailing `/`.
    #[test]
    fn paths_are_normalized() {
        assert_eq!(
            normalized(Path::new("bench-out/X.nsight/")),
            Path::new("bench-out/X.nsight")
        );
        assert_eq!(normalized(Path::new("./a//b/./c/")), Path::new("a/b/c"));
        assert_eq!(normalized(Path::new(".")), Path::new("."));
        assert_eq!(normalized(Path::new("../a")), Path::new("../a"));
        assert_eq!(normalized(Path::new("/x/./y")), Path::new("/x/y"));
    }

    /// @test A directory stands for its .nsys-rep reports, then its .ncu-rep
    /// ones, each sorted by path; hidden entries count, a link to a directory
    /// is not followed, a directory named like a report is searched, not read,
    /// and the suffix must match exactly.
    #[cfg(unix)]
    #[test]
    fn directory_walk() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        for sub in ["a", "a-b", ".hidden", "folder.nsys-rep", "elsewhere/deep"] {
            fs::create_dir_all(root.join(sub)).unwrap();
        }
        for file in [
            "a/x.nsys-rep",
            "a-b/y.nsys-rep",
            ".hidden/h.nsys-rep",
            "folder.nsys-rep/inner.ncu-rep",
            "z.ncu-rep",
            "top.nsys-rep",
            "UPPER.NSYS-REP",
            "notes.txt",
            "elsewhere/deep/s.nsys-rep",
        ] {
            fs::write(root.join(file), "x").unwrap();
        }
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("link")).unwrap();
        std::os::unix::fs::symlink(root.join("top.nsys-rep"), root.join("flink.nsys-rep")).unwrap();
        // The link's target is reached once directly; through the link it is
        // not followed.
        let input = resolve(&root);
        assert!(input.problems.is_empty(), "{:?}", input.problems);
        let found: Vec<String> = input
            .reports
            .iter()
            .map(|(p, _)| p.strip_prefix(&root).unwrap().display().to_string())
            .collect();
        assert_eq!(
            found,
            [
                ".hidden/h.nsys-rep",
                "a/x.nsys-rep",
                "a-b/y.nsys-rep",
                "elsewhere/deep/s.nsys-rep",
                "flink.nsys-rep",
                "top.nsys-rep",
                "folder.nsys-rep/inner.ncu-rep",
                "z.ncu-rep",
            ]
        );
    }

    /// @test An input that does not exist, one that is not a report, and a
    /// directory without reports are each a problem of its own.
    #[test]
    fn input_problems() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.nsys-rep");
        let text = dir.path().join("notes.txt");
        fs::write(&text, "x").unwrap();
        let empty = dir.path().join("empty");
        fs::create_dir(&empty).unwrap();
        assert_eq!(
            resolve(&missing).problems,
            [format!("no such file or directory: {}", missing.display())]
        );
        assert_eq!(
            resolve(&text).problems,
            [format!(
                "not an .nsys-rep, an .ncu-rep or a directory: {}",
                text.display()
            )]
        );
        assert_eq!(
            resolve(&empty).problems,
            [format!(
                "no .nsys-rep or .ncu-rep file under {}",
                empty.display()
            )]
        );
    }

    /// @test A directory that cannot be read is reported, not skipped.
    #[cfg(unix)]
    #[test]
    fn unreadable_directory_is_reported() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        fs::create_dir(&locked).unwrap();
        fs::write(dir.path().join("r.nsys-rep"), "x").unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
        let readable = fs::read_dir(&locked).is_ok();
        let input = resolve(dir.path());
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o700)).unwrap();
        // A user who can read it anyway (root) has nothing to report.
        if readable {
            return;
        }
        assert_eq!(input.reports.len(), 1);
        assert_eq!(input.problems.len(), 1, "{:?}", input.problems);
        assert!(
            input.problems[0].starts_with(&format!("cannot read directory {}: ", locked.display()))
        );
    }

    /// @test The private directory is this user's alone and goes when removed.
    #[cfg(unix)]
    #[test]
    fn private_dir_is_private_and_removed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = PrivateDir::create().unwrap();
        let path = dir.path().to_path_buf();
        fs::write(path.join("profile.sqlite"), "x").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
        let other = PrivateDir::create().unwrap();
        assert_ne!(other.path(), path);
        dir.remove().unwrap();
        assert!(!path.exists());
        let other_path = other.path().to_path_buf();
        drop(other);
        assert!(!other_path.exists());
    }
}
