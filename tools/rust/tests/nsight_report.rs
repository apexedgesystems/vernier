//! `bench nsight-parse`: the reports it reads, the CSV it writes and the
//! failures it reports, against fake `nsys` and `ncu` that replay what the real
//! tools printed on the reference rig (tests/fixtures/nsight/README.md) and log
//! every call.

use std::fs;
use std::io::{PipeReader, PipeWriter, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime};

use vernier_rust_tools::bench::nsight_report::{COMMAND, NSYS_SUMMARIES};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nsight");

/// nsys as the reference rig's printed it. It refuses `stats` on a report that
/// has the runner's export beside it, as nsys 2025.3.2 did; fails the export
/// of a report named "broken"; exits 0 without an export for one named
/// "unwritten"; ends its export by SIGKILL for one named "killed"; fails
/// `cuda_api_sum` for one named "failing_summary" and prints a record longer
/// than its header there for one named "long_record"; and for one named
/// "slow" starts a process under itself and waits for it, as a launcher waits
/// for the tool it started. That process runs until its process group is
/// killed and leaves by itself once the test's folder is gone; the fake
/// creates `started` once it runs.
const FAKE_NSYS: &str = r#"#!/bin/sh
echo "nsys $*" >> "$NSIGHT_FAKE_LOG"
lasting() {
  while [ -d "$NSIGHT_CASE" ]; do "$SLEEP" 0.05; done
}
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
  case "$rep" in *slow*) lasting & : > "$NSIGHT_FAKE_STARTED"; wait; exit 0 ;; esac
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
/// report named "damaged"; for one named "slow" it starts a process under
/// itself and waits, as the launcher on PATH runs the real binary. For one
/// named "leftover" it starts a process that keeps its outputs and exits at
/// once; for one named "detached" it starts one with its outputs elsewhere
/// and then prints the recorded import. Each such process runs until its
/// process group is killed and leaves by itself once the test's folder is
/// gone; the fake creates `started` once it runs. A "leftover" fake holds the
/// test's FIFO `launcher` open, when the test has made one, and the process it
/// leaves does not, so that FIFO tells when the fake itself has ended. A
/// process started with redirections runs in a subshell: dash keeps a copy of
/// each redirected descriptor in a function call it puts in the background.
const FAKE_NCU: &str = r#"#!/bin/sh
echo "ncu $*" >> "$NSIGHT_FAKE_LOG"
lasting() {
  while [ -d "$NSIGHT_CASE" ]; do "$SLEEP" 0.05; done
}
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
case "$rep" in *slow*) lasting & : > "$NSIGHT_FAKE_STARTED"; wait; exit 0 ;; esac
case "$rep" in *leftover*)
  [ -p "$NSIGHT_LAUNCHER_FIFO" ] && exec 9<>"$NSIGHT_LAUNCHER_FIFO"
  ( lasting ) 9>&- & : > "$NSIGHT_FAKE_STARTED"; exit 0 ;;
esac
case "$rep" in *detached*) ( lasting ) > /dev/null 2>&1 & : > "$NSIGHT_FAKE_STARTED" ;; esac
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
/// `tmp/`, the working directory `work/`, and the fakes' call log and
/// markers.
struct Case {
    dir: tempfile::TempDir,
}

/// The action one signal has when a test starts `bench`, set by the test
/// whatever this test process inherited: a run keeps a signal that is
/// ignored when it starts ignored, and a test run can start with signals
/// ignored (a background job of a shell without job control, `nohup`).
#[derive(Debug, Clone, Copy)]
enum Start {
    /// The signal, by its name for `kill -s`, at its default action.
    Default(&'static str),
    /// The signal ignored.
    Ignoring(&'static str),
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
        self.command_from(Command::new(env!("CARGO_BIN_EXE_bench")), args)
    }

    /// The same, started by GNU env (coreutils 8.31 or later) with one
    /// signal's action set as @p start says.
    fn command_with(&self, start: Start, args: &[&str]) -> Command {
        let option = match start {
            Start::Default(signal) => format!("--default-signal={signal}"),
            Start::Ignoring(signal) => format!("--ignore-signal={signal}"),
        };
        let mut env = Command::new(which("env"));
        env.arg(option).arg(env!("CARGO_BIN_EXE_bench"));
        self.command_from(env, args)
    }

    /// @p cmd with @p args, in this case's directories and environment.
    fn command_from(&self, mut cmd: Command, args: &[&str]) -> Command {
        cmd.args(args)
            .current_dir(self.path("work"))
            .env("PATH", fakes())
            .env("TMPDIR", self.path("tmp"))
            .env("NSIGHT_FAKE_LOG", self.path("calls.log"))
            .env("NSIGHT_CASE", self.dir.path())
            .env("NSIGHT_FAKE_STARTED", self.path("started"))
            .env("NSIGHT_LAUNCHER_FIFO", self.path("launcher"))
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

    /// `bench nsight-parse <args>`.
    fn parse(&self, args: &[&str]) -> Run {
        let mut all = vec![COMMAND];
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

    /// Run @p command (bench, from `command` or `command_with`) to its end as
    /// this test's child holding the write end of a pipe (`Owned`): what it
    /// printed and how it ended, and the pipe's read end, which reads its end
    /// once every process bench started has ended.
    fn run_held(&self, mut command: Command) -> (Run, PipeReader) {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let (held, holder) = std::io::pipe().expect("a pipe");
        let mut owned = Owned::start(command, holder);
        let stdout = read_in_background(owned.bench.stdout.take());
        let stderr = read_in_background(owned.bench.stderr.take());
        let status = wait_for_exit(&mut owned.bench, Duration::from_secs(60));
        let run = Run {
            status,
            stdout: stdout.join().expect("read bench's stdout"),
            stderr: stderr.join().expect("read bench's stderr"),
        };
        (run, held)
    }

    /// `bench nsight-parse <args>`, as `run_held` runs it.
    fn parse_held(&self, args: &[&str]) -> (Run, PipeReader) {
        let mut all = vec![COMMAND];
        all.extend_from_slice(args);
        self.run_held(self.command(&all))
    }

    /// Make the FIFO @p name in this case's folder and open it for reading,
    /// without waiting: a fake that opens it too holds it while it runs
    /// (`all_ended` reads its end once no process holds it).
    fn fifo(&self, name: &str) -> fs::File {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;

        let path = self.path(name);
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).expect("a path");
        // SAFETY: mkfifo(3) only reads the path, a NUL-terminated string that
        // lives through the call.
        let made = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(made, 0, "mkfifo: {}", std::io::Error::last_os_error());
        fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)
            .expect("open the FIFO")
    }
}

/// Read @p pipe to its end on a thread of its own; nothing for none.
fn read_in_background<R: Read + Send + 'static>(
    pipe: Option<R>,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_string(&mut text);
        }
        text
    })
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

/// Wait at most @p within for @p child to exit.
fn wait_for_exit(child: &mut std::process::Child, within: Duration) -> ExitStatus {
    let until = Instant::now() + within;
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return status;
        }
        if Instant::now() >= until {
            let _ = child.kill();
            panic!("bench did not end within {within:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `bench`, started by a test as its own child, holding the write end of a
/// pipe whose read end the test keeps: bench and every process it starts
/// inherit that end, so the pipe reads its end once all of them have ended
/// (`all_ended`), whatever their process ids. Dropped, it ends bench by its
/// handle, which signals nothing once bench has been reaped; the processes
/// bench started are left to end by themselves (the slow fake's process
/// leaves once its case's folder is gone). Nothing is signalled by a process
/// id read from a file.
struct Owned {
    bench: std::process::Child,
}

impl Owned {
    /// Start @p command with @p holder, a pipe's write end, open in its
    /// process without close-on-exec; this test's copies of it are closed
    /// once the process has started. The process gets it as a descriptor
    /// numbered 10 or more, which no redirection of the fakes (`sh` takes
    /// 0 to 9) replaces in a process bench starts.
    fn start(mut command: Command, holder: PipeWriter) -> Self {
        use std::os::fd::{FromRawFd, OwnedFd};

        // SAFETY: fcntl(2) F_DUPFD_CLOEXEC only duplicates the descriptor
        // `holder` owns, onto the lowest free number from 10.
        let raised = unsafe { libc::fcntl(holder.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 10) };
        assert!(
            raised >= 10,
            "F_DUPFD_CLOEXEC: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: fcntl has just returned `raised`, which nothing else owns.
        let raised = unsafe { OwnedFd::from_raw_fd(raised) };
        drop(holder);
        let fd = raised.as_raw_fd();
        // SAFETY: the closure runs in the child between fork and exec and
        // calls only fcntl(2), which is async-signal-safe; `fd` is open in the
        // child, as `raised` lives until the spawn has returned.
        unsafe {
            command.pre_exec(move || {
                if libc::fcntl(fd, libc::F_SETFD, 0) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
        let bench = command.spawn().expect("spawn bench");
        drop(raised);
        Self { bench }
    }

    /// Wait at most 20 s for the fake to start its process (it then creates
    /// `started`); a run that ends first fails the test.
    fn wait_for_start(&mut self, case: &Case) {
        let until = Instant::now() + Duration::from_secs(20);
        while !case.path("started").exists() {
            if let Some(status) = self.bench.try_wait().expect("wait") {
                panic!("bench ended before its tool started: {status:?}");
            }
            assert!(Instant::now() < until, "the fake never started");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// What bench wrote to its stderr, once it has ended, when it was piped.
    fn stderr(&mut self) -> String {
        let mut text = String::new();
        if let Some(mut stderr) = self.bench.stderr.take() {
            stderr
                .read_to_string(&mut text)
                .expect("read bench's stderr");
        }
        text
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        // By its handle: a bench already reaped is not signalled again, and
        // one not yet reaped still has its own process id.
        let _ = self.bench.kill();
        let _ = self.bench.wait();
    }
}

/// Whether every process holding @p held's write end has ended within
/// @p within: the pipe then reads its end.
fn all_ended(held: &mut (impl Read + AsRawFd), within: Duration) -> bool {
    let fd = held.as_raw_fd();
    // SAFETY: fcntl(2) only reads and sets the flags of a descriptor that
    // @p held owns and keeps open through the calls.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        assert!(flags >= 0, "F_GETFL: {}", std::io::Error::last_os_error());
        let set = libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        assert!(set >= 0, "F_SETFL: {}", std::io::Error::last_os_error());
    }
    let until = Instant::now() + within;
    let mut chunk = [0u8; 64];
    loop {
        match held.read(&mut chunk) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= until {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("read the pipe: {e}"),
        }
    }
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

/// @test With a stderr no one reads, the warning and error lines are lost
/// and nothing else changes: the CSV is written, stdout says so, and the
/// run exits 0 after a warning and 1 after an error.
#[test]
fn lines_no_one_reads_change_nothing_else() {
    for (report, rows, status) in [
        ("no_kernels.nsys-rep", 12, 0),
        ("failing_summary.nsys-rep", 5, 1),
    ] {
        let case = Case::new();
        case.report(report);
        let (reader, writer) = std::io::pipe().expect("a pipe");
        drop(reader);
        let out = case
            .command(&[COMMAND, report, "--csv", "out.csv"])
            .stdin(Stdio::null())
            .stderr(writer)
            .output()
            .expect("run bench");
        assert_eq!(
            out.status.code(),
            Some(status),
            "{report}: {:?}",
            out.status
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("[{COMMAND}] wrote {rows} rows to out.csv\n"),
            "{report}"
        );
        assert!(!case.csv("out.csv").is_empty(), "{report}: no CSV");
    }
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
        &[COMMAND, "kernel_profile.ncu-rep", "--csv", "out.csv"],
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
        &[COMMAND, "kernel_profile.ncu-rep", "--csv", "out.csv"],
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

/// @test An output path that is a symbolic link to one of the reports is that
/// report, and is refused before any tool runs; the report is left as it was.
#[test]
fn output_symlinked_to_a_report_is_refused() {
    let case = Case::new();
    let report = case.report("run/profile.nsys-rep");
    std::os::unix::fs::symlink(&report, case.path("work").join("out.csv")).unwrap();

    let run = case.parse(&["run", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        "Error: invalid arguments: --csv out.csv is one of the reports to read; \
         name another file\n"
    );
    assert!(case.calls().is_empty(), "{:?}", case.calls());
    assert_eq!(fs::read_to_string(report).unwrap(), "fake report");
}

/// @test An output path that is a hard link to one of the reports is that
/// report, and is refused before any tool runs, whether the report is named
/// or found in a directory; the report is left as it was.
#[test]
fn output_hard_linked_to_a_report_is_refused() {
    for input in ["run/profile.ncu-rep", "run"] {
        let case = Case::new();
        let report = case.report("run/profile.ncu-rep");
        fs::hard_link(&report, case.path("work").join("out.csv")).unwrap();

        let run = case.parse(&[input, "--csv", "out.csv"]);

        assert_eq!(run.status.code(), Some(1), "{input}: {}", run.stderr);
        assert_eq!(
            run.stderr,
            "Error: invalid arguments: --csv out.csv is one of the reports to read; \
             name another file\n",
            "{input}"
        );
        assert!(case.calls().is_empty(), "{input}: {:?}", case.calls());
        assert_eq!(
            fs::read_to_string(&report).unwrap(),
            "fake report",
            "{input}"
        );
    }
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

/// @test A line with an action word after the command reads that word as an
/// input: it is an error of its own, the reports are read all the same, and
/// the run exits 1.
#[test]
fn a_word_after_the_command_is_an_input() {
    let case = Case::new();
    case.report("run/profile.nsys-rep");

    let run = case.parse(&["parse", "run/profile.nsys-rep", "--csv", "out.csv"]);

    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!("[{COMMAND}] error: no such file or directory: parse\n")
    );
    assert_eq!(
        run.stdout,
        format!("[{COMMAND}] wrote 13 rows to out.csv\n")
    );
    assert_eq!(case.csv("out.csv"), fixture("expected_nsys_summaries.csv"));
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
/// started under itself, which would otherwise have run as long as the test,
/// and the run says so.
#[test]
fn timeout_stops_the_tool_and_what_it_started() {
    let case = Case::new();
    case.report("slow.ncu-rep");
    let started = Instant::now();

    let (run, mut held) = case.parse_held(&["slow.ncu-rep", "--csv", "out.csv", "--timeout", "1"]);

    assert!(
        case.path("started").exists(),
        "the tool never started its process"
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the tool, or the process it started, still runs"
    );
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

    let (run, mut held) = case.parse_held(&["slow.nsys-rep", "--csv", "out.csv", "--timeout", "1"]);

    assert!(
        case.path("started").exists(),
        "the tool never started its process"
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the tool, or the process it started, still runs"
    );
    assert_eq!(run.status.code(), Some(1));
    assert!(run
        .stderr
        .contains("nsys export failed for slow.nsys-rep: nsys did not finish within 1 s"));
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
}

/// Start a run on @p report in @p case with @p signal at its default action,
/// send it that signal once its tool runs, and check that the tool and what
/// it started, the private export and the CSV are all gone and that the run
/// ended by that signal. With @p name, stderr is the line naming the signal;
/// without, stderr is a pipe no one reads (its read end closed). With
/// @p path, bench's PATH is that folder instead of the fakes'.
///
/// bench is this test's child (`Owned`): it is signalled before it is reaped
/// and, if it outlasts the bound, ended by its handle. Whether the tool and
/// what it started have ended is read from the pipe they inherit from bench,
/// not from a process id; the slow fake's process leaves by itself once the
/// case's folder is gone, so a run that fails leaves nothing behind.
fn interrupt(
    case: &Case,
    report: &str,
    signal: &'static str,
    number: i32,
    name: Option<&str>,
    path: Option<&Path>,
) {
    let stderr = if name.is_some() {
        Stdio::piped()
    } else {
        let (reader, writer) = std::io::pipe().expect("a pipe");
        drop(reader);
        writer.into()
    };
    let mut command = case.command_with(
        Start::Default(signal),
        &[COMMAND, report, "--csv", "out.csv"],
    );
    command.stdout(Stdio::null()).stderr(stderr);
    if let Some(path) = path {
        command.env("PATH", path);
    }
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let mut run = Owned::start(command, holder);
    run.wait_for_start(case);

    send_signal(run.bench.id(), signal);

    let status = wait_for_exit(&mut run.bench, Duration::from_secs(30));
    let stderr = run.stderr();
    let stopped = all_ended(&mut held, Duration::from_secs(5));
    assert_eq!(status.signal(), Some(number), "{status:?}");
    assert!(stopped, "the tool, or a process it started, still runs");
    if let Some(name) = name {
        assert_eq!(
            stderr,
            format!("[{COMMAND}] stopped by {name}; nothing was written\n")
        );
    }
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
    assert!(!case.path("work").join("out.csv").exists());
}

/// `interrupt` on a slow export, with the fakes on PATH.
fn interrupted_run_cleans_up(signal: &'static str, number: i32, name: Option<&str>) {
    let case = Case::new();
    case.report("slow.nsys-rep");
    interrupt(&case, "slow.nsys-rep", signal, number, name, None);
}

/// @test SIGINT stops the tool and what it started, removes the private
/// export, writes nothing, and ends the run by SIGINT.
#[test]
fn sigint_cleans_up() {
    interrupted_run_cleans_up("INT", 2, Some("SIGINT"));
}

/// @test SIGTERM does the same and ends the run by SIGTERM.
#[test]
fn sigterm_cleans_up() {
    interrupted_run_cleans_up("TERM", 15, Some("SIGTERM"));
}

/// @test SIGHUP, which a terminal's hangup sends to bench and not to the
/// tool's own process group, does the same and ends the run by SIGHUP.
#[test]
fn sighup_cleans_up() {
    interrupted_run_cleans_up("HUP", 1, Some("SIGHUP"));
}

/// @test With a stderr no one reads, the line naming the signal is lost and
/// nothing else changes: SIGINT still stops the tool, removes the private
/// export, writes nothing, and ends the run by SIGINT.
#[test]
fn sigint_with_a_stderr_no_one_reads_cleans_up() {
    interrupted_run_cleans_up("INT", 2, None);
}

/// @test No process is signalled by a process id read from a file. tool.pid
/// names a live process of this test's own (a sentinel, kept unreaped, so
/// that its id stays its own); an interruption run whose tool is not on PATH,
/// or ends at once, then leaves the sentinel running, whether the run's
/// checks pass or fail.
#[test]
fn a_recorded_process_id_is_never_signalled() {
    for (report, unavailable) in [("slow.nsys-rep", true), ("unwritten.nsys-rep", false)] {
        let mut sentinel = Command::new(which("sleep"))
            .arg("60")
            .stdin(Stdio::null())
            .spawn()
            .expect("start the sentinel");
        let case = Case::new();
        case.report(report);
        fs::write(case.path("tool.pid"), sentinel.id().to_string()).expect("a stale tool.pid");
        let empty = case.path("empty-bin");
        let path = unavailable.then_some(empty.as_path());
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            interrupt(&case, report, "INT", 2, Some("SIGINT"), path);
        }));
        let state = sentinel.try_wait().expect("the sentinel's state");
        let _ = sentinel.kill();
        let _ = sentinel.wait();
        assert_eq!(
            state, None,
            "{report}: the process tool.pid names was signalled"
        );
    }
}

/// @test A run that fails while the slow fake runs leaves nothing behind.
/// bench, ended by its handle as a failed check ends it, does not stop its
/// tool's process group: the fake and what it started keep running, and
/// leave by themselves once the case's folder is gone.
#[test]
fn a_failed_run_leaves_nothing_behind() {
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let case = Case::new();
    case.report("slow.nsys-rep");
    let mut command = case.command_with(
        Start::Default("INT"),
        &[COMMAND, "slow.nsys-rep", "--csv", "out.csv"],
    );
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut run = Owned::start(command, holder);
    run.wait_for_start(&case);
    drop(run);
    assert!(
        !all_ended(&mut held, Duration::from_millis(300)),
        "the fake ended with bench"
    );
    drop(case);
    assert!(
        all_ended(&mut held, Duration::from_secs(20)),
        "the fake outlived its case's folder"
    );
}

/// Start a run on a slow export with @p signal ignored, send it that signal
/// once its tool runs, and check that the run kept ignoring it: --timeout
/// still stops the tool and what it started, the private export is removed,
/// and the run fails by its deadline, not by the signal.
fn ignored_signal_stays_ignored(signal: &'static str) {
    let case = Case::new();
    case.report("slow.nsys-rep");
    let mut command = case.command_with(
        Start::Ignoring(signal),
        &[
            COMMAND,
            "slow.nsys-rep",
            "--csv",
            "out.csv",
            "--timeout",
            "2",
        ],
    );
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let mut run = Owned::start(command, holder);
    run.wait_for_start(&case);

    send_signal(run.bench.id(), signal);

    let status = wait_for_exit(&mut run.bench, Duration::from_secs(30));
    let stderr = run.stderr();
    assert_eq!(status.code(), Some(1), "{status:?}");
    assert_eq!(
        stderr,
        format!(
            "[{COMMAND}] error: nsys export failed for slow.nsys-rep: nsys did not finish \
             within 2 s and was stopped\n"
        )
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the tool, or the process it started, still runs"
    );
    assert!(case.leftovers().is_empty(), "{:?}", case.leftovers());
}

/// @test A run started with SIGHUP ignored, as nohup starts one, keeps
/// ignoring it: the tool runs on until --timeout stops it.
#[test]
fn ignored_hangup_stays_ignored() {
    ignored_signal_stays_ignored("HUP");
}

/// @test A run started with SIGINT ignored, as a shell without job control
/// starts a background command, keeps ignoring it: the tool runs on until
/// --timeout stops it with what it started.
#[test]
fn ignored_interrupt_stays_ignored() {
    ignored_signal_stays_ignored("INT");
}

/// @test A tool that ends while a process it started keeps its output open is
/// a failed read 2 s later, however far off --timeout, and that process is
/// stopped before the run returns.
#[test]
fn leftover_holding_the_output_is_stopped() {
    let case = Case::new();
    case.report("leftover.ncu-rep");
    let started = Instant::now();

    let (run, mut held) =
        case.parse_held(&["leftover.ncu-rep", "--csv", "out.csv", "--timeout", "30"]);

    let took = started.elapsed();
    assert_eq!(run.status.code(), Some(1));
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: ncu --import failed for leftover.ncu-rep: ncu ended, but a \
             process it started still held its output 2 s later and was stopped\n"
        )
    );
    assert!(took < Duration::from_secs(10), "{took:?}");
    assert!(
        case.path("started").exists(),
        "the tool never started its process"
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the process the tool left still runs"
    );
}

/// @test A signal that arrives after the tool ended, while a process it
/// started still holds its output, ends the run at once: that process is
/// stopped, nothing is written, and the run ends by the signal.
#[test]
fn signal_while_a_leftover_holds_the_output_ends_at_once() {
    let case = Case::new();
    case.report("leftover.ncu-rep");
    // Held by the tool alone: the process it leaves starts without it.
    let mut tool = case.fifo("launcher");
    let mut command = case.command_with(
        Start::Default("TERM"),
        &[
            COMMAND,
            "leftover.ncu-rep",
            "--csv",
            "out.csv",
            "--timeout",
            "30",
        ],
    );
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let mut run = Owned::start(command, holder);
    run.wait_for_start(&case);
    assert!(
        all_ended(&mut tool, Duration::from_secs(5)),
        "the tool never ended"
    );
    // The run is now waiting for the output the leftover holds.
    std::thread::sleep(Duration::from_millis(200));

    let signalled = Instant::now();
    send_signal(run.bench.id(), "TERM");

    let status = wait_for_exit(&mut run.bench, Duration::from_secs(40));
    let took = signalled.elapsed();
    let stderr = run.stderr();
    assert_eq!(status.signal(), Some(15), "{status:?}");
    assert!(took < Duration::from_secs(1), "{took:?} after SIGTERM");
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the process the tool left still runs"
    );
    assert_eq!(
        stderr,
        format!("[{COMMAND}] stopped by SIGTERM; nothing was written\n")
    );
    assert!(!case.path("work").join("out.csv").exists());
}

/// @test A tool that leaves a process running with its output elsewhere still
/// gives its rows, and that process is stopped before the run returns.
#[test]
fn process_left_by_a_finished_tool_is_stopped() {
    let case = Case::new();
    case.report("detached.ncu-rep");

    let (run, mut held) = case.parse_held(&["detached.ncu-rep", "--csv", "ncu.csv"]);

    assert_eq!(run.status.code(), Some(0), "{}", run.stderr);
    assert_eq!(case.csv("ncu.csv"), fixture("expected_ncu_metrics.csv"));
    assert!(
        case.path("started").exists(),
        "the tool never started its process"
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the process the tool left still runs"
    );
}

/// @test A run started with SIGCHLD ignored, which would have the kernel reap
/// the tool unseen, still sees the tool end and stops what it left.
#[test]
fn ignored_child_signal_does_not_hide_the_tools_end() {
    let case = Case::new();
    case.report("leftover.ncu-rep");
    let (run, mut held) = case.run_held(case.command_with(
        Start::Ignoring("CHLD"),
        &[
            COMMAND,
            "leftover.ncu-rep",
            "--csv",
            "out.csv",
            "--timeout",
            "1",
        ],
    ));

    assert_eq!(run.status.code(), Some(1), "{}", run.stderr);
    assert_eq!(
        run.stderr,
        format!(
            "[{COMMAND}] error: ncu --import failed for leftover.ncu-rep: ncu ended, but a \
             process it started still held its output 2 s later and was stopped\n"
        )
    );
    assert!(
        case.path("started").exists(),
        "the tool never started its process"
    );
    assert!(
        all_ended(&mut held, Duration::from_secs(5)),
        "the process the tool left still runs"
    );
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

/// @test The help shows the one-word command with its inputs, --csv and
/// --timeout, and the exit status.
#[test]
fn help_describes_the_command() {
    let case = Case::new();
    let run = case.parse(&["--help"]);
    assert_eq!(run.status.code(), Some(0));
    for needle in [
        "Usage: bench nsight-parse [OPTIONS] --csv <CSV> <INPUTS>...",
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
