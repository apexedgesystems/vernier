//! `bench nsight-parse parse`: the reports it reads, the CSV it writes and the
//! failures it reports, against fake `nsys` and `ncu` that replay what the real
//! tools printed on the reference rig (tests/fixtures/nsight/README.md) and log
//! every call.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime};

use vernier_rust_tools::bench::nsight_report::{ACTION, COMMAND, NSYS_SUMMARIES};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nsight");

/// nsys as the reference rig's printed it. It refuses `stats` on a report that
/// has the runner's export beside it, as nsys 2025.3.2 did; fails the export
/// of a report named "broken"; exits 0 without an export for one named
/// "unwritten"; ends its export by SIGKILL for one named "killed"; fails
/// `cuda_api_sum` for one named "failing_summary" and prints a record longer
/// than its header there for one named "long_record"; and for one named
/// "slow" starts a long `sleep` under itself and waits for it, as a launcher
/// waits for the tool it started.
const FAKE_NSYS: &str = r#"#!/bin/sh
echo "nsys $*" >> "$NSIGHT_FAKE_LOG"
cmd=$1
shift
if [ "$cmd" = export ]; then
  out=""
  rep=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -o) out=$2; shift 2 ;;
      --type|--force-overwrite) shift 2 ;;
      *) rep=$1; shift ;;
    esac
  done
  case "$rep" in *broken*) echo "fake nsys: cannot read $rep" >&2; exit 1 ;; esac
  case "$rep" in *unwritten*) echo "fake nsys: exported nothing"; exit 0 ;; esac
  case "$rep" in *killed*) kill -s KILL $$ ;; esac
  case "$rep" in *slow*) "$SLEEP" 30 & echo $! > "$NSIGHT_FAKE_PIDS"; wait; exit 0 ;; esac
  : > "$out"
  exit 0
fi
if [ "$cmd" = stats ]; then
  report=""
  input=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --report) report=$2; shift 2 ;;
      --format) shift 2 ;;
      --force-export=*) shift ;;
      *) input=$1; shift ;;
    esac
  done
  case "$input" in
    *.nsys-rep)
      if [ -f "${input%.nsys-rep}.sqlite" ]; then
        "$CAT" "$NSIGHT_FIXTURES/nsys_stats_beside_runner_export.err" >&2
        exit 1
      fi ;;
  esac
  case "$input:$report" in
    *no_kernels*:cuda_gpu_kern_sum) "$CAT" "$NSIGHT_FIXTURES/nsys_stats_no_kernels.out"; exit 0 ;;
    *failing_summary*:cuda_api_sum) echo "fake nsys: cuda_api_sum failed" >&2; exit 2 ;;
    *long_record*:cuda_api_sum) printf 'Time (%%),Total Time (ns),Name\n1.0,2,a,EXTRA\n'; exit 0 ;;
  esac
  "$CAT" "$NSIGHT_FIXTURES/nsys_stats_$report.out"
  exit 0
fi
exit 2
"#;

/// ncu as the reference rig's printed it: it fails without `--import` and on a
/// report named "damaged"; for one named "slow" it starts a long `sleep` under
/// itself and waits, as the launcher on PATH runs the real binary.
const FAKE_NCU: &str = r#"#!/bin/sh
echo "ncu $*" >> "$NSIGHT_FAKE_LOG"
rep=""
while [ $# -gt 0 ]; do
  case "$1" in
    --import) rep=$2; shift 2 ;;
    *) shift ;;
  esac
done
if [ -z "$rep" ]; then
  "$CAT" "$NSIGHT_FIXTURES/ncu_without_import.out"
  exit 1
fi
case "$rep" in *damaged*) "$CAT" "$NSIGHT_FIXTURES/ncu_import_damaged.out"; exit 1 ;; esac
case "$rep" in *slow*) "$SLEEP" 30 & echo $! > "$NSIGHT_FAKE_PIDS"; wait; exit 0 ;; esac
"$CAT" "$NSIGHT_FIXTURES/ncu_import_per_kernel.out"
exit 0
"#;

/* ----------------------------- Helpers ----------------------------- */

/// What one run of `bench` printed and how it ended.
struct Run {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

/// A scratch area for one test: the temporary directory `bench` is given in
/// `tmp/`, the working directory `work/`, and the fakes' call log and pid file.
struct Case {
    dir: tempfile::TempDir,
}

/// The directory holding the fake `nsys` and `ncu`, written once for the whole
/// test process before any test starts a process. A script written while
/// another thread forks can be held open for writing in that child until it
/// execs, and executing it then fails with "Text file busy".
fn fakes() -> &'static Path {
    static FAKES: OnceLock<tempfile::TempDir> = OnceLock::new();
    FAKES
        .get_or_init(|| {
            let dir = tempfile::Builder::new()
                .prefix("nsight-fakes-")
                .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
                .expect("tempdir for the fakes");
            for (name, script) in [("nsys", FAKE_NSYS), ("ncu", FAKE_NCU)] {
                let path = dir.path().join(name);
                fs::write(&path, script).expect("write fake");
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
            }
            dir
        })
        .path()
}

/// The absolute path of @p program on this test's PATH: the fakes run with a
/// PATH that holds only themselves.
fn which(program: &str) -> PathBuf {
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(program))
                .find(|path| path.is_file())
        })
        .unwrap_or_else(|| PathBuf::from("/bin").join(program))
}

fn fixture(name: &str) -> Vec<u8> {
    fs::read(Path::new(FIXTURES).join(name)).expect("fixture")
}

impl Case {
    fn new() -> Self {
        fakes();
        let dir = tempfile::tempdir().expect("tempdir");
        for sub in ["tmp", "work", "empty-bin"] {
            fs::create_dir(dir.path().join(sub)).expect("mkdir");
        }
        Self { dir }
    }

    fn path(&self, sub: &str) -> PathBuf {
        self.dir.path().join(sub)
    }

    /// A report file under `work/`: the fakes never read its contents, but an
    /// empty file is no report.
    fn report(&self, rel: &str) -> PathBuf {
        let path = self.path("work").join(rel);
        fs::create_dir_all(path.parent().unwrap()).expect("mkdir");
        fs::write(&path, "fake report").expect("write report");
        path
    }

    /// `bench` with @p args, run from `work/` with only the fakes on PATH.
    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_bench"));
        cmd.args(args)
            .current_dir(self.path("work"))
            .env("PATH", fakes())
            .env("TMPDIR", self.path("tmp"))
            .env("NSIGHT_FAKE_LOG", self.path("calls.log"))
            .env("NSIGHT_FAKE_PIDS", self.path("tool.pid"))
            .env("NSIGHT_FIXTURES", FIXTURES)
            .env("CAT", which("cat"))
            .env("SLEEP", which("sleep"));
        cmd
    }

    fn run(&self, args: &[&str]) -> Run {
        let out = self.command(args).output().expect("run bench");
        Run {
            status: out.status,
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }

    /// `bench nsight-parse parse <args>`.
    fn parse(&self, args: &[&str]) -> Run {
        let mut all = vec![COMMAND, ACTION];
        all.extend_from_slice(args);
        self.run(&all)
    }

    fn calls(&self) -> Vec<String> {
        fs::read_to_string(self.path("calls.log"))
            .map(|text| text.lines().map(str::to_string).collect())
            .unwrap_or_default()
    }

    /// What is left in the temporary directory `bench` was given.
    fn leftovers(&self) -> Vec<String> {
        fs::read_dir(self.path("tmp"))
            .expect("tmp")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    }

    fn csv(&self, name: &str) -> Vec<u8> {
        fs::read(self.path("work").join(name)).expect("the CSV")
    }

    /// The pid of the `sleep` a slow fake started, once it has started.
    fn tool_pid(&self) -> u32 {
        let until = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(pid) = fs::read_to_string(self.path("tool.pid"))
                .ok()
                .and_then(|text| text.trim().parse().ok())
            {
                return pid;
            }
            assert!(
                Instant::now() < until,
                "the slow fake never started its tool"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Whether @p pid has ended: gone, or a zombie no one has reaped yet.
fn ended(pid: u32) -> bool {
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        let state = fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| {
                let after = &stat[stat.rfind(')')? + 1..];
                after.split_whitespace().next().map(str::to_string)
            });
        match state.as_deref() {
            None | Some("Z") | Some("X") => return true,
            _ if Instant::now() >= until => return false,
            _ => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

/// Send @p signal (a name for `kill -s`) to the process @p pid alone.
fn send_signal(pid: u32, signal: &str) {
    let status = Command::new("/bin/sh")
        .arg("-c")
        .arg(format!("kill -s {signal} {pid}"))
        .status()
        .expect("run kill");
    assert!(status.success(), "kill -s {signal} {pid}");
}

/// End a process a failed test left running, by its pid.
fn end(pid: u32) {
    let _ = Command::new("/bin/sh")
        .arg("-c")
        .arg(format!("kill -s KILL {pid}"))
        .status();
}

/// The export path an `nsys export` call wrote to.
fn export_path(call: &str) -> PathBuf {
    let words: Vec<&str> = call.split_whitespace().collect();
    let at = words.iter().position(|w| *w == "-o").expect("-o");
    PathBuf::from(words[at + 1])
}

/* ----------------------------- Nsight Systems ----------------------------- */

/// @test One export into a private directory, then the four summaries read
/// from it: 13 rows, the CSV byte for byte the reference rig's
/// (fixtures/nsight/README.md), and the private directory gone afterwards.
#[test]
fn nsys_report_yields_every_summary_row() {
    let case = Case::new();
    case.report("run/profile.nsys-rep");

    let run = case.parse(&["run/profile.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert_eq!(run.stderr, "");
    assert_eq!(
        run.stdout,
        format!("[{COMMAND}] wrote 13 rows to out.csv\n")
    );
    assert_eq!(case.csv("out.csv"), fixture("expected_nsys_summaries.csv"));
    let calls = case.calls();
    assert_eq!(calls.len(), 5, "{calls:?}");
    let export = export_path(&calls[0]);
    assert_eq!(
        calls[0],
        format!(
            "nsys export --type sqlite --force-overwrite true -o {} run/profile.nsys-rep",
            export.display()
        )
    );
    assert!(
        export.starts_with(case.path("tmp")),
        "the export must go to a private directory, not beside the report: {}",
        export.display()
    );
    assert_eq!(export.file_name().unwrap(), "profile.sqlite");
    for (call, summary) in calls[1..].iter().zip(NSYS_SUMMARIES) {
        assert_eq!(
            *call,
            format!(
                "nsys stats --report {summary} --format csv {}",
                export.display()
            )
        );
    }
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
}

/// @test The layout bench run leaves: the export beside the report is neither
/// used nor changed, and the report itself is not touched.
#[test]
fn report_beside_the_runners_export() {
    let case = Case::new();
    let report = case.report("bench-out/Bin.nsight/profile.nsys-rep");
    let runner_export = report.with_extension("sqlite");
    fs::write(&runner_export, "the runner's export").unwrap();
    let mtime = |p: &Path| fs::metadata(p).unwrap().modified().unwrap();
    let (export_time, report_time): (SystemTime, SystemTime) =
        (mtime(&runner_export), mtime(&report));
    // The stand-in refuses the report itself here, as nsys 2025.3.2 did.
    let refused = Command::new(fakes().join("nsys"))
        .args(["stats", "--report", "cuda_gpu_kern_sum", "--format", "csv"])
        .arg(&report)
        .env("NSIGHT_FAKE_LOG", case.path("refused.log"))
        .env("NSIGHT_FIXTURES", FIXTURES)
        .env("CAT", which("cat"))
        .output()
        .expect("run the fake");
    assert_eq!(refused.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("older than input file"));

    let run = case.parse(&["bench-out/Bin.nsight/", "--csv", "nsys.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert_eq!(case.csv("nsys.csv"), fixture("expected_nsys_summaries.csv"));
    assert_eq!(
        fs::read_to_string(&runner_export).unwrap(),
        "the runner's export"
    );
    assert_eq!(mtime(&runner_export), export_time);
    assert_eq!(fs::read_to_string(&report).unwrap(), "fake report");
    assert_eq!(mtime(&report), report_time);
    assert!(case
        .calls()
        .iter()
        .all(|call| !call.contains("bench-out/Bin.nsight/profile.sqlite")));
}

/// @test A report with no kernel: that summary has no rows, which is a
/// warning, not a failure.
#[test]
fn summary_without_data_is_a_warning() {
    let case = Case::new();
    case.report("no_kernels.nsys-rep");

    let run = case.parse(&["no_kernels.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    let lines: Vec<&str> = run.stderr.lines().collect();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with(&format!(
        "[{COMMAND}] warning: cuda_gpu_kern_sum has no rows for no_kernels.nsys-rep: SKIPPED: "
    )));
    assert!(lines[0].ends_with("does not contain CUDA kernel data."));
    assert_eq!(
        run.stdout,
        format!("[{COMMAND}] wrote 12 rows to out.csv\n")
    );
}

/// @test A failing summary is an error naming it; the other summaries' rows
/// are written.
#[test]
fn failed_summary_keeps_the_others() {
    let case = Case::new();
    case.report("failing_summary.nsys-rep");

    let run = case.parse(&["failing_summary.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: nsys stats --report cuda_api_sum failed for \
             failing_summary.nsys-rep: exit status 2: fake nsys: cuda_api_sum failed\n"
        )
    );
    assert_eq!(run.stdout, format!("[{COMMAND}] wrote 5 rows to out.csv\n"));
}

/// @test An export that fails is an error with the tool's own message; the
/// CSV is still written (empty), and the run exits 1.
#[test]
fn failed_export_is_an_error() {
    let case = Case::new();
    case.report("broken.nsys-rep");

    let run = case.parse(&["broken.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: nsys export failed for broken.nsys-rep: exit status 1: \
             fake nsys: cannot read broken.nsys-rep\n"
        )
    );
    assert_eq!(run.stdout, format!("[{COMMAND}] wrote 0 rows to out.csv\n"));
    assert_eq!(case.csv("out.csv"), b"");
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
}

/// @test An export that succeeds without writing the file is an error, not an
/// empty read.
#[test]
fn export_without_a_file_is_an_error() {
    let case = Case::new();
    case.report("unwritten.nsys-rep");

    let run = case.parse(&["unwritten.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: nsys export failed for unwritten.nsys-rep: no export was \
             written (fake nsys: exported nothing)\n"
        )
    );
    assert_eq!(case.calls().len(), 1);
}

/// @test A tool ended by a signal is reported with that signal.
#[test]
fn tool_ended_by_a_signal_is_named() {
    let case = Case::new();
    case.report("killed.nsys-rep");

    let run = case.parse(&["killed.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: nsys export failed for killed.nsys-rep: ended by signal 9: \
             no output\n"
        )
    );
}

/// @test A summary record longer than its header fails that summary, naming
/// the line, instead of adding a column the tool did not name.
#[test]
fn long_record_fails_its_summary() {
    let case = Case::new();
    case.report("long_record.nsys-rep");

    let run = case.parse(&["long_record.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: nsys stats --report cuda_api_sum failed for \
             long_record.nsys-rep: line 2 of its output has 4 fields for 3 columns\n"
        )
    );
    assert_eq!(run.stdout, format!("[{COMMAND}] wrote 5 rows to out.csv\n"));
}

/* ----------------------------- Nsight Compute ----------------------------- */

/// @test ncu reads the saved report through --import: one row per launch
/// shape, section and metric, byte for byte the reference rig's CSV.
#[test]
fn ncu_report_is_imported() {
    let case = Case::new();
    case.report("Bin.ncu/kernel_profile.ncu-rep");

    let run = case.parse(&["Bin.ncu/kernel_profile.ncu-rep", "--csv", "ncu.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert_eq!(
        run.stdout,
        format!("[{COMMAND}] wrote 88 rows to ncu.csv\n")
    );
    assert_eq!(case.csv("ncu.csv"), fixture("expected_ncu_metrics.csv"));
    assert_eq!(
        case.calls(),
        ["ncu --import Bin.ncu/kernel_profile.ncu-rep --csv --print-summary per-kernel"]
    );
}

/// @test Without ncu on PATH the request fails, naming the missing tool; it
/// does not come back empty and successful.
#[test]
fn missing_tool_is_an_error() {
    let case = Case::new();
    case.report("kernel_profile.ncu-rep");

    let run = case.run_with_path(
        &[
            COMMAND,
            ACTION,
            "kernel_profile.ncu-rep",
            "--csv",
            "out.csv",
        ],
        &case.path("empty-bin"),
    );

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: ncu --import failed for kernel_profile.ncu-rep: \
             ncu not found on PATH\n"
        )
    );
}

/// @test A tool on PATH that cannot be executed is reported as such, apart
/// from a missing one.
#[test]
fn tool_that_cannot_start_is_named() {
    let case = Case::new();
    case.report("kernel_profile.ncu-rep");
    let not_executable = case.path("plain-bin");
    fs::create_dir(&not_executable).unwrap();
    fs::write(not_executable.join("ncu"), "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(
        not_executable.join("ncu"),
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();

    let run = case.run_with_path(
        &[
            COMMAND,
            ACTION,
            "kernel_profile.ncu-rep",
            "--csv",
            "out.csv",
        ],
        &not_executable,
    );

    assert_eq!(run.status.code(), Some(1));
    assert!(
        run.stderr.starts_with(&format!(
            "[{COMMAND}] error: ncu --import failed for kernel_profile.ncu-rep: \
             ncu could not be started: "
        )),
        "{}",
        run.stderr
    );
}

/* ----------------------------- Inputs ----------------------------- */

/// @test One report read and one refused: the rows are written, the refusal is
/// named with ncu's message, and the run fails.
#[test]
fn mixed_directory_keeps_good_rows_and_fails() {
    let case = Case::new();
    case.report("mixed/good.nsys-rep");
    case.report("mixed/damaged.ncu-rep");

    let run = case.parse(&["mixed/", "--csv", "mixed.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        case.csv("mixed.csv"),
        fixture("expected_nsys_summaries.csv")
    );
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: ncu --import failed for mixed/damaged.ncu-rep: exit status 1: \
             ==ERROR== An unexpected incompatibility with this Nsight Compute version occurred. \
             Try opening the file with the same tool version it was created with.\n"
        )
    );
    assert_eq!(
        run.stdout,
        format!("[{COMMAND}] wrote 13 rows to mixed.csv\n")
    );
}

/// @test A directory stands for its .nsys-rep reports and then its .ncu-rep
/// ones: 101 rows, byte for byte the reference rig's CSV.
#[test]
fn directory_with_both_report_types() {
    let case = Case::new();
    case.report("bench-out/Bin.ncu/kernel_profile.ncu-rep");
    case.report("bench-out/Bin.nsight/profile.nsys-rep");

    let run = case.parse(&["bench-out", "--csv", "all.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert_eq!(case.csv("all.csv"), fixture("expected_all_reports.csv"));
    let tools: Vec<String> = case
        .calls()
        .iter()
        .map(|c| c.split_whitespace().next().unwrap().to_string())
        .collect();
    assert_eq!(tools, ["nsys", "nsys", "nsys", "nsys", "nsys", "ncu"]);
}

/// @test A directory with no report is a failed request, and says where it
/// looked; the CSV is still written.
#[test]
fn directory_without_reports_is_an_error() {
    let case = Case::new();
    fs::create_dir(case.path("work").join("empty")).unwrap();

    let run = case.parse(&["empty/", "--csv", "x.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!("[{COMMAND}] error: no .nsys-rep or .ncu-rep file under empty\n")
    );
    assert_eq!(case.csv("x.csv"), b"");
}

/// @test An input that is not a report, one that does not exist and an empty
/// report are each an error of its own, and no tool is asked to read them.
#[test]
fn invalid_inputs_are_errors() {
    let case = Case::new();
    fs::write(case.path("work").join("notes.txt"), "not a report").unwrap();
    fs::write(case.path("work").join("empty.ncu-rep"), "").unwrap();

    let run = case.parse(&[
        "notes.txt",
        "missing.nsys-rep",
        "empty.ncu-rep",
        "--csv",
        "x.csv",
    ]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: not an .nsys-rep, an .ncu-rep or a directory: notes.txt\n\
             [{COMMAND}] error: no such file or directory: missing.nsys-rep\n\
             [{COMMAND}] error: empty.ncu-rep is empty\n"
        )
    );
    assert!(case.calls().is_empty(), "{:?}", case.calls());
}

/// @test An output path that is one of the reports is refused before any tool
/// runs, and the report is left as it was.
#[test]
fn output_that_is_a_report_is_refused() {
    let case = Case::new();
    let report = case.report("run/profile.nsys-rep");

    let run = case.parse(&["run", "--csv", "run/profile.nsys-rep"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        "Error: invalid arguments: --csv run/profile.nsys-rep is one of the reports to read; \
         name another file\n"
    );
    assert!(case.calls().is_empty(), "{:?}", case.calls());
    assert_eq!(fs::read_to_string(report).unwrap(), "fake report");
}

/// @test A CSV that cannot be written still leaves every input's line printed
/// first, then the write's own error, and the run fails.
#[test]
fn unwritable_csv_is_an_error() {
    let case = Case::new();
    case.report("mixed/good.nsys-rep");
    case.report("mixed/damaged.ncu-rep");

    let run = case.parse(&["mixed", "--csv", "no-such-dir/out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    let lines: Vec<&str> = run.stderr.lines().collect();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with(&format!(
        "[{COMMAND}] error: ncu --import failed for mixed/damaged.ncu-rep"
    )));
    assert!(
        lines[1].starts_with("Error: I/O error: cannot write no-such-dir/out.csv: "),
        "{}",
        lines[1]
    );
    assert_eq!(run.stdout, "");
}

/// @test Usage errors exit 2: no input, no --csv, an empty argument, a timeout
/// of 0.
#[test]
fn usage_errors_exit_two() {
    let case = Case::new();
    case.report("r.nsys-rep");
    for args in [
        vec!["--csv", "x.csv"],
        vec!["r.nsys-rep"],
        vec!["", "--csv", "x.csv"],
        vec!["r.nsys-rep", "--csv", ""],
        vec!["r.nsys-rep", "--csv", "x.csv", "--timeout", "0"],
    ] {
        let run = case.parse(&args);
        assert_eq!(run.status.code(), Some(2), "{args:?}: {}", run.stderr);
    }
    assert!(case.calls().is_empty());
}

/* ----------------------------- Bounded runs ----------------------------- */

/// @test A tool that runs past --timeout is stopped with the process it
/// started under itself, well before that process would have ended, and the
/// run says so.
#[test]
fn timeout_stops_the_tool_and_what_it_started() {
    let case = Case::new();
    case.report("slow.ncu-rep");
    let started = Instant::now();

    let run = case.parse(&["slow.ncu-rep", "--csv", "out.csv", "--timeout", "1"]);

    let tool = case.tool_pid();
    let stopped = ended(tool);
    if !stopped {
        end(tool);
    }
    assert!(stopped, "the tool's own process {tool} still runs");
    assert!(started.elapsed() < Duration::from_secs(20));
    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: ncu --import failed for slow.ncu-rep: ncu did not finish \
             within 1 s and was stopped\n"
        )
    );
}

/// @test An export that runs past --timeout is stopped the same way, and its
/// private directory is removed.
#[test]
fn timeout_removes_the_private_export() {
    let case = Case::new();
    case.report("slow.nsys-rep");

    let run = case.parse(&["slow.nsys-rep", "--csv", "out.csv", "--timeout", "1"]);

    let tool = case.tool_pid();
    let stopped = ended(tool);
    if !stopped {
        end(tool);
    }
    assert!(stopped, "the tool's own process {tool} still runs");
    assert_eq!(run.status.code(), Some(1));
    assert!(run
        .stderr
        .contains("nsys export failed for slow.nsys-rep: nsys did not finish within 1 s"));
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
}

/// Start a run on a slow export, send it @p signal once its tool runs, and
/// check that the tool, the private export and the CSV are all gone and that
/// the run ended by that signal.
fn interrupted_run_cleans_up(signal: &str, number: i32, name: &str) {
    let case = Case::new();
    case.report("slow.nsys-rep");
    let mut child = case
        .command(&[COMMAND, ACTION, "slow.nsys-rep", "--csv", "out.csv"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bench");
    let tool = case.tool_pid();

    send_signal(child.id(), signal);

    let until = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait") {
            break status;
        }
        if Instant::now() >= until {
            let _ = child.kill();
            end(tool);
            panic!("bench did not end after {signal}");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let out = child.wait_with_output().expect("output");
    let stopped = ended(tool);
    if !stopped {
        end(tool);
    }
    assert_eq!(status.signal(), Some(number), "{status:?}");
    assert!(stopped, "the tool's own process {tool} still runs");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!("[{COMMAND}] stopped by {name}; nothing was written\n")
    );
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
    assert!(!case.path("work").join("out.csv").exists());
}

/// @test SIGINT stops the tool and what it started, removes the private
/// export, writes nothing, and ends the run by SIGINT.
#[test]
fn sigint_cleans_up() {
    interrupted_run_cleans_up("INT", 2, "SIGINT");
}

/// @test SIGTERM does the same and ends the run by SIGTERM.
#[test]
fn sigterm_cleans_up() {
    interrupted_run_cleans_up("TERM", 15, "SIGTERM");
}

/* ----------------------------- Not a benchmark CSV ----------------------------- */

/// @test The CSV is not a benchmark CSV: bench summary and bench compare
/// refuse it rather than read unlike metrics.
#[test]
fn benchmark_tools_refuse_the_csv() {
    let case = Case::new();
    case.report("profile.nsys-rep");
    let run = case.parse(&["profile.nsys-rep", "--csv", "nsys.csv"]);
    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);

    let summary = case.run(&["summary", "nsys.csv"]);
    let compare = case.run(&["compare", "nsys.csv", "nsys.csv"]);

    for refused in [summary, compare] {
        assert_eq!(refused.status.code(), Some(1));
        assert!(
            refused.stderr.contains("missing required column 'test'"),
            "{}",
            refused.stderr
        );
        assert_eq!(refused.stdout, "");
    }
}

/// @test profile-summarize stays an inventory of files and bytes: it runs no
/// tool on the reports it lists.
#[test]
fn profile_summarize_reads_no_report() {
    let case = Case::new();
    case.report("bench-out/Bin.nsight/profile.nsys-rep");
    fs::write(
        case.path("work")
            .join("bench-out/Bin.nsight/profile.sqlite"),
        "abc",
    )
    .unwrap();
    case.report("bench-out/Bin.ncu/kernel_profile.ncu-rep");

    let run = case.run(&["profile-summarize", "bench-out"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert!(case.calls().is_empty(), "{:?}", case.calls());
    let rows: Vec<Vec<&str>> = run
        .stdout
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|w| w.first().is_some_and(|n| n.starts_with("Bin.")))
        .collect();
    assert_eq!(rows.len(), 2, "{}", run.stdout);
    assert_eq!(rows[0][..3], ["Bin.ncu", "1", "11"]);
    assert_eq!(rows[1][..3], ["Bin.nsight", "2", "14"]);
}

/// @test The help names the inputs, --csv, --timeout and the exit status.
#[test]
fn help_describes_the_command() {
    let case = Case::new();
    let run = case.parse(&["--help"]);
    assert_eq!(run.status.code(), Some(0));
    for needle in [
        "--csv <CSV>",
        "--timeout <TIMEOUT>",
        "[default: 600]",
        "Exit status: 0",
    ] {
        assert!(run.stdout.contains(needle), "{needle}: {}", run.stdout);
    }
}

impl Case {
    /// `bench` with @p args and @p path as the whole PATH.
    fn run_with_path(&self, args: &[&str], path: &Path) -> Run {
        let out = self
            .command(args)
            .env("PATH", path)
            .output()
            .expect("run bench");
        Run {
            status: out.status,
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }
}
