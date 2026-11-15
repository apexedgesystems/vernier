//! `bench gpu-lock`: how each ending of a run leaves the GPU clocks, against
//! a stand-in nvidia-smi that logs each call and touches no GPU.
//!
//! Each wrapped run starts as the leader of a process group of its own: the
//! command it runs joins that group and this test process stays outside it,
//! so a signal sent to the group reaches bench and the command, as a
//! terminal's Ctrl-C does, and nothing else. bench starts with SIGINT,
//! SIGTERM and SIGHUP set explicitly, to their default action or, for the
//! runs that stand for a parent ignoring them, to ignored, so nothing depends
//! on what this test process inherited. Every wait is for something a
//! process wrote (a process id, a log line, a line of bench's own), bounded
//! only to fail a stuck run; the stand-in and the commands wait for files the
//! test creates, never for a delay. A run's cleanup signals only bench's
//! process group, while bench, its leader, is not yet reaped, and reaps bench
//! after; no process is signalled by a number read from a file.

use std::fs;
use std::io::{ErrorKind, PipeReader, PipeWriter, Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use vernier_rust_tools::bench::gpu_lock::{self, LockConfig, Tool};

/* ----------------------------- Fixtures ----------------------------- */

/// nvidia-smi as gpu-lock calls it. Each rig links it into its own folder as
/// `nvidia-smi` and writes `stand-in.env` beside the link: the rig's folder,
/// where to log, and whether to hold the lock or the reset until a file
/// exists (`lock_gate`, `reset_gate`), to hang in a call (`hang`; it then
/// opens the rig's `hang.fifo`, holds it until it ends and, once it holds
/// it, creates `hang.held`) or to refuse one (`fail`). A held or hung call
/// ends, refused, once the rig's folder is gone, and a hung one also once
/// `hang.end` exists: it runs in a process group of its own, which a run's
/// cleanup does not signal, so a test that fails while it is held leaves
/// nothing behind.
const STAND_IN: &str = r#"#!/bin/sh
. "$(dirname "$0")/stand-in.env"
case " $* " in
  *" -lgc "*) call=lock ;;
  *" -rgc "*) call=reset ;;
  *) call=other ;;
esac
echo "$*" >> "$log"
case " $* " in
  *" --query-gpu=persistence_mode "*) echo Enabled; exit 0 ;;
  *" --query-gpu=clocks.max.graphics "*) echo 2100; exit 0 ;;
esac
gate=""
[ "$call" = lock ] && gate=$lock_gate
[ "$call" = reset ] && gate=$reset_gate
if [ -n "$gate" ]; then
  echo "$call waiting" >> "$log"
  while [ ! -e "$gate" ]; do
    [ -d "$rig" ] || exit 1
    /bin/sleep 0.01
  done
fi
if [ "$call" = "$hang" ]; then
  exec 9<>"$rig/hang.fifo"
  : > "$rig/hang.held"
  while [ -d "$rig" ] && [ ! -e "$rig/hang.end" ]; do /bin/sleep 0.01; done
  exit 1
fi
if [ "$call" = "$fail" ]; then
  echo "Insufficient Permissions" >&2
  exit 4
fi
[ "$call" = other ] || echo "$call done" >> "$log"
exit 0
"#;

/// A command that logs its start beside the stand-in's calls, writes its
/// process id, records each SIGINT, SIGTERM and SIGHUP it receives (a line
/// each in signals.log), waits for the release file, logs its end and exits.
const WAITING: &str = r#"
trap 'echo INT >> "$RIG/signals.log"' INT
trap 'echo TERM >> "$RIG/signals.log"' TERM
trap 'echo HUP >> "$RIG/signals.log"' HUP
echo "command started" >> "$RIG/nvidia-smi.log"
echo $$ > "$RIG/command.pid"
while [ ! -e "$RIG/release" ]; do /bin/sleep 0.01; done
echo "command ended" >> "$RIG/nvidia-smi.log"
"#;

/// The longest any wait of these tests lasts before the test fails.
const BOUND: Duration = Duration::from_secs(20);

/// The bound the library tests give the stand-in.
const SHORT_BOUND: Duration = Duration::from_millis(200);

/// The lock call, the reset call and their marks, as the stand-in logs them.
const LOCK: &str = "-i 0 -lgc 1000,1000";
const RESET: &str = "-i 0 -rgc";

/// The stand-in, written once, before any test of this file starts a
/// process: nothing is written and run at the same time.
fn stand_in() -> &'static Path {
    static STAND_IN_DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
    STAND_IN_DIR
        .get_or_init(|| {
            let dir = tempfile::Builder::new()
                .prefix("gpu-lock-stand-in-")
                .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
                .expect("tempdir for the stand-in");
            let path = dir.path().join("nvidia-smi");
            fs::write(&path, STAND_IN).expect("write the stand-in");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
            dir
        })
        .path()
}

/// One run's folder: the stand-in's link and settings, its log, and the
/// command's files.
struct Rig {
    dir: tempfile::TempDir,
}

impl Rig {
    /// A folder with the stand-in linked in, logging to nvidia-smi.log.
    fn new() -> Self {
        let target = stand_in().join("nvidia-smi");
        let dir = tempfile::tempdir().expect("tempdir");
        symlink(target, dir.path().join("nvidia-smi")).expect("link the stand-in");
        let rig = Self { dir };
        let settings = format!(
            "rig='{}'\nlog='{}'\nlock_gate=\nreset_gate=\nhang=\nfail=\n",
            rig.dir.path().display(),
            rig.path("nvidia-smi.log").display()
        );
        fs::write(rig.path("stand-in.env"), settings).expect("the stand-in's settings");
        rig
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// Set the stand-in's @p key to @p value; a later setting wins.
    fn set(&self, key: &str, value: &str) {
        let mut settings = fs::read_to_string(self.path("stand-in.env")).expect("settings");
        settings.push_str(&format!("{key}='{value}'\n"));
        fs::write(self.path("stand-in.env"), settings).expect("the stand-in's settings");
    }

    /// Hold the stand-in's @p call ("lock" or "reset") until `open(call)`.
    fn gate(&self, call: &str) {
        let gate = self.path(&format!("{call}.gate"));
        self.set(&format!("{call}_gate"), &gate.display().to_string());
    }

    fn open(&self, call: &str) {
        fs::write(self.path(&format!("{call}.gate")), "").expect("open the gate");
    }

    /// Have the stand-in hang in @p call ("lock" or "reset"). It then holds
    /// `hang.fifo` open until it ends, and says so (`hang.held`) once it holds
    /// it, so `hang_ended` tells whether it has without its process id.
    fn hang(&self, call: &str) {
        use std::os::unix::ffi::OsStrExt;

        let fifo = std::ffi::CString::new(self.path("hang.fifo").as_os_str().as_bytes())
            .expect("a path without NUL");
        // SAFETY: mkfifo(3) only reads the path, a NUL-terminated string
        // that lives through the call.
        let made = unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) };
        assert_eq!(made, 0, "mkfifo: {}", std::io::Error::last_os_error());
        self.set("hang", call);
    }

    /// Whether the hung call has ended, within `BOUND` (`hang_ended_within`).
    fn hang_ended(&self) -> bool {
        self.hang_ended_within(BOUND)
    }

    /// Whether the hung call has ended, within @p within. Its end is read
    /// from `hang.fifo`, which reads its end once no process holds it open,
    /// but also before the stand-in has opened it: so the stand-in's word
    /// that it holds the FIFO (`hang.held`, created after the open) comes
    /// first, and only an end read after it counts. A stand-in that never
    /// says so within @p within has not been seen to end.
    fn hang_ended_within(&self, within: Duration) -> bool {
        use std::os::unix::fs::OpenOptionsExt;

        let until = Instant::now() + within;
        while !self.path("hang.held").exists() {
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut fifo = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(self.path("hang.fifo"))
            .expect("open hang.fifo");
        let mut chunk = [0u8; 64];
        loop {
            match fifo.read(&mut chunk) {
                Ok(0) => return true,
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    if Instant::now() >= until {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("read hang.fifo: {e}"),
            }
        }
    }

    /// Let the waiting command end.
    fn release(&self) {
        fs::write(self.path("release"), "").expect("release the command");
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.path(name)).unwrap_or_default()
    }

    /// The stand-in's lock and reset calls, their marks and the command's
    /// start and end, in the order they happened (the two queries left out).
    fn events(&self) -> Vec<String> {
        self.read("nvidia-smi.log")
            .lines()
            .filter(|line| {
                line.contains("-lgc")
                    || line.contains("-rgc")
                    || line.ends_with(" waiting")
                    || line.ends_with(" done")
                    || line.starts_with("command ")
            })
            .map(str::to_string)
            .collect()
    }

    /// The signals the waiting command received, in order.
    fn signals(&self) -> Vec<String> {
        self.read("signals.log")
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn stderr(&self) -> String {
        self.read("bench.stderr")
    }

    /// The process id a process of the run wrote to @p name, once written.
    fn pid(&self, name: &str) -> u32 {
        let mut pid = None;
        wait_until(&format!("{name} to be written"), || {
            pid = self.read(name).trim().parse().ok();
            pid.is_some()
        });
        pid.unwrap_or_default()
    }
}

/// How a test's bench starts its three watched signals.
#[derive(Clone, Copy)]
enum Parent {
    /// At their default action.
    Default,
    /// Ignored, as under nohup or in a background job of a shell without job
    /// control.
    Ignoring,
}

/// A started `bench`, the leader of its own process group. Its end is seen
/// without reaping it (`exited`): bench is reaped only when the run is
/// dropped, once its whole group has been sent SIGKILL. Until then its
/// process id, which is also its group's, cannot pass to another process,
/// so a signal sent to bench or to its group, by a test or by the cleanup,
/// reaches bench and the processes of its group (the command and what the
/// command starts) and nothing else. No process is signalled by a number
/// read from a file.
struct Run {
    bench: Child,
    status: Option<ExitStatus>,
}

impl Run {
    /// `bench <args>` in @p rig, with the stand-in first on PATH and the
    /// watched signals as @p parent says; its stderr to bench.stderr, or to
    /// @p stderr.
    fn start(rig: &Rig, args: &[&str], parent: Parent, stderr: Option<Stdio>) -> Self {
        Self::start_with(rig, args, parent, stderr, None)
    }

    /// As `start`, with @p held, a pipe's write end, as descriptor 3 of
    /// bench, inherited by every process bench starts and by what they
    /// start: the pipe's read end sees its end once all of them have ended.
    fn start_with(
        rig: &Rig,
        args: &[&str],
        parent: Parent,
        stderr: Option<Stdio>,
        held: Option<PipeWriter>,
    ) -> Self {
        let out = fs::File::create(rig.path("bench.stdout")).expect("bench's stdout");
        let err = match stderr {
            Some(stderr) => stderr,
            None => fs::File::create(rig.path("bench.stderr"))
                .expect("bench's stderr")
                .into(),
        };
        let handler = match parent {
            Parent::Default => libc::SIG_DFL,
            Parent::Ignoring => libc::SIG_IGN,
        };
        let mut command = Command::new(env!("CARGO_BIN_EXE_bench"));
        command
            .args(args)
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", rig.dir.path().display()),
            )
            .env("RIG", rig.dir.path())
            .current_dir(rig.dir.path())
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .process_group(0);
        let held_fd = held.as_ref().map(AsRawFd::as_raw_fd);
        // SAFETY: the closure runs in the child between fork and exec and
        // calls only sigaction, fcntl and dup2, which are async-signal-safe;
        // a zeroed sigaction given the default or the ignore handler is a
        // valid value of the type; `held_fd` is open in the child, as @p held
        // is alive until the spawn returns.
        unsafe {
            command.pre_exec(move || {
                for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = handler;
                    if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                // Descriptor 3, without close-on-exec, so that it is inherited.
                let held = match held_fd {
                    None => 0,
                    Some(3) => libc::fcntl(3, libc::F_SETFD, 0),
                    Some(fd) => libc::dup2(fd, 3),
                };
                if held < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let bench = command.spawn().expect("start bench");
        // This test's own copy of the held end: from here only bench and
        // what it starts hold it.
        drop(held);
        Self {
            bench,
            status: None,
        }
    }

    /// bench's process id, which is also its group's: neither can pass to
    /// another process before the run is dropped, which alone reaps bench.
    fn pid(&self) -> libc::pid_t {
        libc::pid_t::try_from(self.bench.id()).expect("a process id")
    }

    /// How bench ended, once it has, seen with waitid(2) and WNOWAIT, which
    /// leaves it unreaped.
    fn exited(&mut self) -> Option<ExitStatus> {
        while self.status.is_none() {
            // SAFETY: a zeroed siginfo_t is a valid value of the type; waitid
            // only writes into `info`, owned here, and with WNOWAIT leaves
            // bench for the drop to reap.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let rc = unsafe {
                libc::waitid(
                    libc::P_PID,
                    self.bench.id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if rc != 0 {
                let e = std::io::Error::last_os_error();
                assert_eq!(e.kind(), ErrorKind::Interrupted, "waitid on bench: {e}");
                continue;
            }
            // SAFETY: waitid filled `info`; with WNOHANG its process id stays
            // 0 while bench runs, and its status is the exit code or the
            // signal, as si_code says.
            let (pid, code, value) = unsafe { (info.si_pid(), info.si_code, info.si_status()) };
            if pid == 0 {
                break;
            }
            // The wait status ExitStatus is built from: an exit code in the
            // second byte, a signal in the first (0x80 for a core dump).
            let raw = match code {
                libc::CLD_EXITED => (value & 0xff) << 8,
                libc::CLD_DUMPED => value | 0x80,
                _ => value,
            };
            self.status = Some(ExitStatus::from_raw(raw));
        }
        self.status
    }

    /// Send @p signal to bench's whole group, as a terminal does.
    fn signal_group(&self, signal: libc::c_int) {
        // SAFETY: killpg only sends a signal, to the group bench leads; bench
        // is unreaped, so the group's id is still its own.
        assert_eq!(unsafe { libc::killpg(self.pid(), signal) }, 0, "killpg");
    }

    /// Send @p signal to bench alone, as kill <pid> does.
    fn signal_bench(&self, signal: libc::c_int) {
        // SAFETY: kill only sends a signal, to bench, which is unreaped.
        assert_eq!(unsafe { libc::kill(self.pid(), signal) }, 0, "kill");
    }

    /// Wait for bench to end, and return how it ended.
    fn end(&mut self) -> ExitStatus {
        wait_until("bench to end", || self.exited().is_some());
        self.status.expect("bench ended")
    }

    /// Wait until bench's stderr in @p rig holds @p text @p times; a bench
    /// that ends first fails the test at once, with how it ended.
    fn wait_for_line(&mut self, rig: &Rig, text: &str, times: usize) {
        wait_until(&format!("bench to say '{text}' {times} time(s)"), || {
            if rig.stderr().matches(text).count() >= times {
                return true;
            }
            if let Some(status) = self.exited() {
                panic!(
                    "bench ended ({status}) before it said '{text}' {times} time(s):\n{}",
                    rig.stderr()
                );
            }
            false
        });
    }
}

impl Drop for Run {
    /// End what is left of bench's group (bench too, if it still runs, and a
    /// command that outlived it), then reap bench.
    fn drop(&mut self) {
        // SAFETY: killpg only sends a signal, to the group bench leads; bench
        // is reaped only below, so the group's id is still its own.
        unsafe { libc::killpg(self.pid(), libc::SIGKILL) };
        let _ = self.bench.wait();
    }
}

/// Wait until @p ready holds, looking every 5 ms; past BOUND the test fails,
/// naming @p what it waited for.
fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
    let until = Instant::now() + BOUND;
    while !ready() {
        assert!(Instant::now() < until, "waited {BOUND:?} for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `gpu-lock lock --freq 1000 -- <command>`.
fn wrap<'a>(command: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec!["gpu-lock", "lock", "--freq", "1000", "--"];
    args.extend_from_slice(command);
    args
}

/// Wait until the waiting command has recorded exactly @p signals.
fn wait_for_signals(rig: &Rig, signals: &[&str]) {
    wait_until(&format!("the command to receive {signals:?}"), || {
        rig.signals() == signals
    });
}

/// The events of a run whose command started and ended before the reset.
fn command_then_reset() -> Vec<String> {
    [
        LOCK,
        "lock done",
        "command started",
        "command ended",
        RESET,
        "reset done",
    ]
    .map(str::to_string)
    .to_vec()
}

/// Line bench prints for a recorded SIGINT or SIGHUP, which it leaves to the
/// terminal, and for a SIGTERM, which it passes on.
const NOT_PASSED_ON: &str = ": the clocks are reset once the command (process";
const PASSED_ON: &str = "[bench] gpu-lock: SIGTERM: passed to the command (process";

/// Make @p fd's reads or writes return instead of waiting (@p on), or wait
/// again.
fn set_nonblocking(fd: RawFd, on: bool) {
    // SAFETY: fcntl reads and sets the status flags of `fd`, a pipe end this
    // test owns; only O_NONBLOCK changes.
    let set = unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        let flags = if on {
            flags | libc::O_NONBLOCK
        } else {
            flags & !libc::O_NONBLOCK
        };
        flags >= 0 && libc::fcntl(fd, libc::F_SETFL, flags) == 0
    };
    assert!(set, "fcntl: {}", std::io::Error::last_os_error());
}

/// Read @p reader until no process holds its pipe's write end any more, and
/// return what it held; past BOUND the test fails, naming @p what was to
/// close it.
fn read_to_eof(reader: &mut PipeReader, what: &str) -> Vec<u8> {
    set_nonblocking(reader.as_raw_fd(), true);
    let until = Instant::now() + BOUND;
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => return bytes,
            Ok(n) => bytes.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                assert!(Instant::now() < until, "waited {BOUND:?} for {what}");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("read the pipe: {e}"),
        }
    }
}

/* ----------------------------- Endings ----------------------------- */

/// @test A command that exits 0: the clocks are locked, then reset, and bench
/// exits 0.
#[test]
fn the_clocks_are_reset_after_the_command_exits() {
    let rig = Rig::new();
    let mut run = Run::start(&rig, &wrap(&["/bin/true"]), Parent::Default, None);
    let status = run.end();
    assert_eq!(status.code(), Some(0), "{}", rig.stderr());
    assert_eq!(rig.events(), [LOCK, "lock done", RESET, "reset done"]);
    let stderr = rig.stderr();
    assert!(
        stderr.contains("GPU 0: clocks locked to 1000 MHz")
            && stderr.contains("GPU 0: clocks reset to default"),
        "{stderr}"
    );
}

/// @test A command that fails, by its status or by a signal of its own: the
/// clocks are reset, and bench exits with the command's status, 128 + N for
/// its signal N, as a shell reports it, saying how the command ended; 125, the
/// status bench keeps for its own failures, passes through too.
#[test]
fn a_failed_command_passes_its_status_through_and_the_clocks_reset() {
    let usr1 = libc::SIGUSR1;
    for (script, code, said) in [
        (
            "exit 3",
            3,
            "[bench] gpu-lock: the wrapped command exited with status 3; bench exits 3".to_string(),
        ),
        (
            "exit 125",
            125,
            "[bench] gpu-lock: the wrapped command exited with status 125; bench exits 125"
                .to_string(),
        ),
        (
            "kill -USR1 $$",
            128 + usr1,
            format!(
                "[bench] gpu-lock: the wrapped command was ended by signal {usr1}; bench \
                 exits {}",
                128 + usr1
            ),
        ),
    ] {
        let rig = Rig::new();
        let mut run = Run::start(
            &rig,
            &wrap(&["/bin/sh", "-c", script]),
            Parent::Default,
            None,
        );
        let status = run.end();
        assert_eq!(status.code(), Some(code), "{script}: {}", rig.stderr());
        assert!(rig.stderr().contains(&said), "{script}: {}", rig.stderr());
        assert!(
            !rig.stderr().contains("Error"),
            "{script}: {}",
            rig.stderr()
        );
        assert_eq!(
            rig.events(),
            [LOCK, "lock done", RESET, "reset done"],
            "{script}"
        );
    }
}

/// @test A command that cannot start: the clocks, locked first, are reset,
/// and bench exits 125, its own failure, naming the command.
#[test]
fn a_command_that_cannot_start_is_named_and_the_clocks_reset() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/nonexistent/command"]),
        Parent::Default,
        None,
    );
    let status = run.end();
    assert_eq!(status.code(), Some(125), "{}", rig.stderr());
    assert!(
        rig.stderr()
            .contains("Error: failed to start '/nonexistent/command'"),
        "{}",
        rig.stderr()
    );
    assert_eq!(rig.events(), [LOCK, "lock done", RESET, "reset done"]);
}

/// @test Without --freq the clocks are locked to the maximum nvidia-smi
/// reports.
#[test]
fn without_freq_the_clocks_lock_to_the_maximum() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &["gpu-lock", "lock", "--", "/bin/true"],
        Parent::Default,
        None,
    );
    assert_eq!(run.end().code(), Some(0), "{}", rig.stderr());
    assert_eq!(
        rig.events(),
        ["-i 0 -lgc 2100,2100", "lock done", RESET, "reset done"]
    );
}

/// @test A reset nvidia-smi refuses fails the run with 125, whatever the
/// command's status, which the error names when it is not 0, and says how to
/// reset the clocks by hand.
#[test]
fn a_refused_reset_fails_the_run() {
    let refused = "the clocks of GPU 0 were not reset: Insufficient Permissions; reset them \
                   with `bench gpu-lock reset --device 0`, or as root with \
                   `nvidia-smi -i 0 -rgc`";
    for (script, said) in [
        ("exit 0", format!("Error: {refused}")),
        (
            "exit 3",
            format!("Error: the wrapped command exited with status 3; {refused}"),
        ),
    ] {
        let rig = Rig::new();
        rig.set("fail", "reset");
        let mut run = Run::start(
            &rig,
            &wrap(&["/bin/sh", "-c", script]),
            Parent::Default,
            None,
        );
        assert_eq!(run.end().code(), Some(125), "{script}: {}", rig.stderr());
        assert!(rig.stderr().contains(&said), "{script}: {}", rig.stderr());
        assert_eq!(rig.events(), [LOCK, "lock done", RESET], "{script}");
    }
}

/// @test A lock nvidia-smi refuses ends bench with 125, its own failure,
/// naming why: no command starts, and nothing is reset.
#[test]
fn a_refused_lock_ends_bench_with_125_and_starts_nothing() {
    let rig = Rig::new();
    rig.set("fail", "lock");
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    assert_eq!(run.end().code(), Some(125), "{}", rig.stderr());
    assert!(
        rig.stderr()
            .contains("Error: failed to lock clocks on GPU 0: Insufficient Permissions"),
        "{}",
        rig.stderr()
    );
    assert_eq!(rig.events(), [LOCK]);
}

/// @test A stderr no one reads (a pipe whose read end is closed, as a
/// terminal that hung up refuses writes) stops neither the run nor the reset.
#[test]
fn a_stderr_no_one_reads_does_not_stop_the_reset() {
    let rig = Rig::new();
    let (reader, writer) = std::io::pipe().expect("a pipe");
    drop(reader);
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/true"]),
        Parent::Default,
        Some(writer.into()),
    );
    assert_eq!(run.end().code(), Some(0));
    assert_eq!(rig.events(), [LOCK, "lock done", RESET, "reset done"]);
}

/// @test With a stderr no one reads, a run that fails still ends with the
/// status the README gives it, the command's own when it failed (3) and 125
/// when the reset was refused: the line bench cannot write is lost, and the
/// clocks are reset first.
#[test]
fn a_failed_run_with_a_stderr_no_one_reads_keeps_its_status() {
    for (script, refuse_reset, code, events) in [
        (
            "exit 3",
            false,
            3,
            &[LOCK, "lock done", RESET, "reset done"][..],
        ),
        ("exit 0", true, 125, &[LOCK, "lock done", RESET][..]),
    ] {
        let rig = Rig::new();
        if refuse_reset {
            rig.set("fail", "reset");
        }
        let (reader, writer) = std::io::pipe().expect("a pipe");
        drop(reader);
        let mut run = Run::start(
            &rig,
            &wrap(&["/bin/sh", "-c", script]),
            Parent::Default,
            Some(writer.into()),
        );
        assert_eq!(
            run.end().code(),
            Some(code),
            "{script}, reset refused: {refuse_reset}"
        );
        assert_eq!(rig.events(), events, "{script}");
    }
}

/* ----------------------------- Signals ----------------------------- */

/// @test A SIGINT sent to the group, as a terminal's Ctrl-C: the command
/// receives it once, from the group (bench does not pass it on again); the
/// clocks stay locked while the command runs, are reset after it ends, and
/// bench ends by SIGINT.
#[test]
fn sigint_to_the_group_resets_once_the_command_ends() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_group(libc::SIGINT);
    run.wait_for_line(&rig, &format!("[bench] gpu-lock: SIGINT{NOT_PASSED_ON}"), 1);
    wait_for_signals(&rig, &["INT"]);
    assert!(
        !rig.events().iter().any(|e| e == RESET),
        "reset while the command ran: {:?}",
        rig.events()
    );
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGINT), "{}", rig.stderr());
    assert_eq!(rig.signals(), ["INT"], "the command received SIGINT twice");
    assert_eq!(rig.events(), command_then_reset());
}

/// @test A SIGHUP sent to the group, as a terminal that hangs up: as for
/// SIGINT, the command receives it once, and bench ends by SIGHUP once the
/// clocks are reset.
#[test]
fn sighup_to_the_group_resets_once_the_command_ends() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_group(libc::SIGHUP);
    run.wait_for_line(&rig, &format!("[bench] gpu-lock: SIGHUP{NOT_PASSED_ON}"), 1);
    wait_for_signals(&rig, &["HUP"]);
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGHUP), "{}", rig.stderr());
    assert_eq!(rig.signals(), ["HUP"]);
    assert_eq!(rig.events(), command_then_reset());
}

/// @test A SIGTERM sent to bench alone is passed on to the command, once;
/// the clocks stay locked until the command ends, and bench ends by SIGTERM.
#[test]
fn sigterm_to_bench_is_passed_to_the_command() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_bench(libc::SIGTERM);
    run.wait_for_line(&rig, PASSED_ON, 1);
    wait_for_signals(&rig, &["TERM"]);
    assert!(!rig.events().iter().any(|e| e == RESET));
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
    assert_eq!(rig.signals(), ["TERM"]);
    assert_eq!(rig.events(), command_then_reset());
}

/// @test A SIGINT sent to bench alone is not passed on, a terminal's signal
/// being the terminal's to send: bench waits for the command, resets the
/// clocks once it ends, and ends by SIGINT.
#[test]
fn sigint_to_bench_alone_is_not_passed_on() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_bench(libc::SIGINT);
    run.wait_for_line(&rig, &format!("[bench] gpu-lock: SIGINT{NOT_PASSED_ON}"), 1);
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGINT), "{}", rig.stderr());
    assert!(rig.signals().is_empty(), "passed on: {:?}", rig.signals());
    assert_eq!(rig.events(), command_then_reset());
}

/// @test Two different signals in a known order, SIGTERM to bench and then,
/// once bench has recorded it, SIGINT to the group: bench ends by the first,
/// SIGTERM, not by the later SIGINT.
#[test]
fn two_signals_end_bench_by_the_first() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_bench(libc::SIGTERM);
    run.wait_for_line(&rig, PASSED_ON, 1);
    wait_for_signals(&rig, &["TERM"]);
    run.signal_group(libc::SIGINT);
    run.wait_for_line(&rig, &format!("[bench] gpu-lock: SIGINT{NOT_PASSED_ON}"), 1);
    wait_for_signals(&rig, &["TERM", "INT"]);
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
    assert_eq!(rig.events(), command_then_reset());
}

/// The signals blocked by bench's thread named @p name, from its status in
/// /proc (bench is unreaped, so its process id is still its own).
fn blocked_in_thread(run: &Run, name: &str) -> u64 {
    let tasks = format!("/proc/{}/task", run.pid());
    for task in fs::read_dir(&tasks).expect("bench's threads") {
        let task = task.expect("a thread").path();
        if fs::read_to_string(task.join("comm"))
            .unwrap_or_default()
            .trim()
            != name
        {
            continue;
        }
        let status = fs::read_to_string(task.join("status")).expect("the thread's status");
        let mask = status
            .lines()
            .find_map(|line| line.strip_prefix("SigBlk:"))
            .expect("SigBlk in the thread's status");
        return u64::from_str_radix(mask.trim(), 16).expect("a signal mask");
    }
    panic!("bench has no thread named '{name}'");
}

/// @test The thread that writes bench's messages blocks SIGINT, SIGTERM and
/// SIGHUP: a signal sent to bench reaches the thread that records it and
/// ends bench by the first, never one where its default action would end
/// bench by a later one.
#[test]
fn the_message_thread_never_takes_a_watched_signal() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    let watched = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP]
        .iter()
        .fold(0u64, |mask, &signal| mask | 1 << (signal - 1));
    let blocked = blocked_in_thread(&run, "gpu-lock stderr");
    rig.release();
    assert_eq!(run.end().code(), Some(0), "{}", rig.stderr());
    assert_eq!(blocked & watched, watched, "blocked {blocked:#x}");
}

/// @test A second SIGTERM is passed on as the first was, and bench still
/// waits: the clocks are reset once, after the command ends.
#[test]
fn a_second_sigterm_is_passed_on_and_the_reset_still_waits() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_bench(libc::SIGTERM);
    run.wait_for_line(&rig, PASSED_ON, 1);
    wait_for_signals(&rig, &["TERM"]);
    run.signal_bench(libc::SIGTERM);
    run.wait_for_line(&rig, PASSED_ON, 2);
    wait_for_signals(&rig, &["TERM", "TERM"]);
    assert!(!rig.events().iter().any(|e| e == RESET));
    rig.release();
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
    assert_eq!(rig.events(), command_then_reset());
}

/// @test Started by a parent that ignores SIGINT, SIGTERM and SIGHUP, bench
/// leaves them ignored, for itself and for the command: the three sent to the
/// group change nothing, and the run ends as the command does.
#[test]
fn signals_a_parent_ignores_change_nothing() {
    let rig = Rig::new();
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Ignoring,
        None,
    );
    rig.pid("command.pid");
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        run.signal_group(signal);
    }
    rig.release();
    let status = run.end();
    assert_eq!(status.code(), Some(0), "{}", rig.stderr());
    assert!(rig.signals().is_empty(), "{:?}", rig.signals());
    assert!(
        !rig.stderr().contains("[bench] gpu-lock:"),
        "{}",
        rig.stderr()
    );
    assert_eq!(rig.events(), command_then_reset());
}

/// @test A SIGINT that arrives while the clocks are being locked lets the
/// lock finish (nvidia-smi runs in a group of its own, which the signal does
/// not reach), starts no command, resets the clocks, and ends bench by it.
#[test]
fn a_signal_during_the_lock_starts_no_command() {
    let rig = Rig::new();
    rig.gate("lock");
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    wait_until("the lock to wait", || {
        rig.events().iter().any(|e| e == "lock waiting")
    });
    run.signal_group(libc::SIGINT);
    rig.open("lock");
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGINT), "{}", rig.stderr());
    assert_eq!(
        rig.events(),
        [LOCK, "lock waiting", "lock done", RESET, "reset done"]
    );
    assert!(!rig.path("command.pid").exists(), "the command was started");
    assert!(
        rig.stderr().contains(
            "[bench] gpu-lock: SIGINT: arrived while the clocks were being locked; the command \
             is not started"
        ),
        "{}",
        rig.stderr()
    );
}

/// @test A signal delivered while the reset runs neither stops the reset
/// (nvidia-smi runs in a group of its own) nor replaces the first signal:
/// SIGTERM first, then SIGINT to the group while the stand-in holds the
/// reset; the reset is called and completes once, and bench ends by SIGTERM.
#[test]
fn a_signal_during_the_reset_keeps_the_first() {
    let rig = Rig::new();
    rig.gate("reset");
    let mut run = Run::start(
        &rig,
        &wrap(&["/bin/sh", "-c", WAITING]),
        Parent::Default,
        None,
    );
    rig.pid("command.pid");
    run.signal_bench(libc::SIGTERM);
    run.wait_for_line(&rig, PASSED_ON, 1);
    wait_for_signals(&rig, &["TERM"]);
    rig.release();
    wait_until("the reset to wait", || {
        rig.events().iter().any(|e| e == "reset waiting")
    });
    run.signal_group(libc::SIGINT);
    rig.open("reset");
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
    assert_eq!(
        rig.events(),
        [
            LOCK,
            "lock done",
            "command started",
            "command ended",
            RESET,
            "reset waiting",
            "reset done"
        ]
    );
    assert_eq!(
        rig.stderr()
            .matches("GPU 0: clocks reset to default")
            .count(),
        1,
        "{}",
        rig.stderr()
    );
}

/// @test A signal delivered while the reset runs, with none before it, is
/// not lost: the reset completes, and then bench ends by that signal.
#[test]
fn a_signal_during_the_reset_is_not_lost() {
    let rig = Rig::new();
    rig.gate("reset");
    let mut run = Run::start(&rig, &wrap(&["/bin/true"]), Parent::Default, None);
    wait_until("the reset to wait", || {
        rig.events().iter().any(|e| e == "reset waiting")
    });
    run.signal_group(libc::SIGINT);
    rig.open("reset");
    let status = run.end();
    assert_eq!(status.signal(), Some(libc::SIGINT), "{}", rig.stderr());
    assert_eq!(
        rig.events(),
        [LOCK, "lock done", RESET, "reset waiting", "reset done"]
    );
    assert_eq!(
        rig.stderr()
            .matches("GPU 0: clocks reset to default")
            .count(),
        1,
        "{}",
        rig.stderr()
    );
}

/* ----------------------------- A full stderr ----------------------------- */

/// A pipe filled to the brim whose read end stays open and is not read
/// while bench runs: a stderr that takes nothing more.
struct FullPipe {
    reader: PipeReader,
    filled: usize,
}

impl FullPipe {
    /// The pipe, and its write end for bench's stderr, blocking again as a
    /// terminal or a pipe a parent hands down is.
    fn new() -> (Self, PipeWriter) {
        let (reader, mut writer) = std::io::pipe().expect("a pipe");
        set_nonblocking(writer.as_raw_fd(), true);
        let chunk = [b'X'; 4096];
        let mut filled = 0;
        loop {
            match writer.write(&chunk) {
                Ok(n) => filled += n,
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => panic!("fill the pipe: {e}"),
            }
        }
        set_nonblocking(writer.as_raw_fd(), false);
        (Self { reader, filled }, writer)
    }

    /// Once bench and every process it started have ended: whether the pipe
    /// holds what filled it and nothing else, no message having got in.
    fn holds_only_its_filling(mut self) -> bool {
        let held = read_to_eof(&mut self.reader, "bench and its command to end");
        held.len() == self.filled && held.iter().all(|&byte| byte == b'X')
    }
}

/// A first line for a command's script: record the status flags of the
/// command's stderr (octal, as /proc shows them) in stderr.flags.
const RECORD_STDERR_FLAGS: &str =
    "sed -n 's/^flags:[[:space:]]*//p' /proc/$$/fdinfo/2 > \"$RIG/stderr.flags\"\n";

/// Whether the stderr the command in @p rig inherited is blocking, as the
/// command recorded it (`RECORD_STDERR_FLAGS`).
fn command_stderr_blocks(rig: &Rig) -> bool {
    let flags = rig.read("stderr.flags");
    let flags = u32::from_str_radix(flags.trim(), 8).expect("the command's stderr flags");
    flags & libc::O_NONBLOCK as u32 == 0
}

/// For each of @p runs, whose command is `WAITING`: once the command has
/// started, send bench SIGTERM; once the command has received it, passed on
/// by bench, check that the clocks are still locked; then release the
/// command and wait for bench to end. The runs go in step, so they take as
/// long as one.
fn sigterm_then_release(runs: &mut [(&Rig, &mut Run)]) -> Vec<ExitStatus> {
    for (rig, _) in runs.iter() {
        rig.pid("command.pid");
    }
    for (_, run) in runs.iter() {
        run.signal_bench(libc::SIGTERM);
    }
    for (rig, _) in runs.iter() {
        wait_for_signals(rig, &["TERM"]);
        assert!(
            !rig.events().iter().any(|e| e == RESET),
            "reset while the command ran: {:?}",
            rig.events()
        );
    }
    for (rig, _) in runs.iter() {
        rig.release();
    }
    runs.iter_mut().map(|(_, run)| run.end()).collect()
}

/// @test A stderr that takes nothing (a full pipe, its read end open and
/// never read) holds up nothing: bench starts the command, passes a SIGTERM
/// on, resets the clocks once the command ends and ends by SIGTERM, exactly
/// as beside it with a stderr that takes its messages (the control), and
/// none of its messages gets into the pipe. The command inherits stderr as
/// it was handed to bench, blocking.
#[test]
fn a_full_stderr_holds_up_neither_the_reset_nor_the_end_by_the_signal() {
    let (control, full) = (Rig::new(), Rig::new());
    let (pipe, stderr) = FullPipe::new();
    let script = format!("{RECORD_STDERR_FLAGS}{WAITING}");
    let waiting = wrap(&["/bin/sh", "-c", &script]);
    let mut control_run = Run::start(&control, &waiting, Parent::Default, None);
    let mut full_run = Run::start(&full, &waiting, Parent::Default, Some(stderr.into()));
    let ended = sigterm_then_release(&mut [(&control, &mut control_run), (&full, &mut full_run)]);
    for (rig, status) in [(&control, ended[0]), (&full, ended[1])] {
        assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
        assert_eq!(rig.signals(), ["TERM"]);
        assert_eq!(rig.events(), command_then_reset());
        assert!(
            command_stderr_blocks(rig),
            "the command's stderr is non-blocking"
        );
    }
    let said = control.stderr();
    assert!(
        said.contains("GPU 0: clocks locked to 1000 MHz")
            && said.contains(PASSED_ON)
            && said.contains("GPU 0: clocks reset to default"),
        "{said}"
    );
    assert!(
        pipe.holds_only_its_filling(),
        "a message got into the full pipe"
    );
}

/// @test A full stderr holds up nothing either where bench reports an error
/// before it ends by the signal: with a reset nvidia-smi refuses and a
/// SIGTERM, bench asks for the reset once the command has ended and ends by
/// SIGTERM, as beside it with a stderr that takes the error (the control),
/// which names the manual reset.
#[test]
fn a_full_stderr_holds_up_nothing_when_the_reset_is_refused() {
    let (control, full) = (Rig::new(), Rig::new());
    for rig in [&control, &full] {
        rig.set("fail", "reset");
    }
    let (pipe, stderr) = FullPipe::new();
    let waiting = wrap(&["/bin/sh", "-c", WAITING]);
    let mut control_run = Run::start(&control, &waiting, Parent::Default, None);
    let mut full_run = Run::start(&full, &waiting, Parent::Default, Some(stderr.into()));
    let ended = sigterm_then_release(&mut [(&control, &mut control_run), (&full, &mut full_run)]);
    for (rig, status) in [(&control, ended[0]), (&full, ended[1])] {
        assert_eq!(status.signal(), Some(libc::SIGTERM), "{}", rig.stderr());
        assert_eq!(
            rig.events(),
            [LOCK, "lock done", "command started", "command ended", RESET]
        );
    }
    assert!(
        control.stderr().contains(
            "Error: the clocks of GPU 0 were not reset: Insufficient Permissions; reset them \
             with `bench gpu-lock reset --device 0`, or as root with `nvidia-smi -i 0 -rgc`"
        ),
        "{}",
        control.stderr()
    );
    assert!(
        pipe.holds_only_its_filling(),
        "a message got into the full pipe"
    );
}

/* ----------------------------- Cleanup ----------------------------- */

/// A process this test starts and owns, outside bench's group: its process
/// id stands for the number a stale command.pid holds once the command it
/// named was reaped and the number passed to another process. No cleanup of
/// a run may signal it. Asked with SIGUSR1, it answers by exiting 0; dropped
/// unasked, it is ended through its own handle.
struct Sentinel(Child);

impl Sentinel {
    /// The sentinel, once its answer is set up (it touches @p ready then).
    fn start(ready: &Path) -> Self {
        let child = Command::new("/bin/sh")
            .args([
                "-c",
                "trap 'exit 0' USR1; : > \"$0\"; while :; do /bin/sleep 0.01; done",
            ])
            .arg(ready)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the sentinel");
        wait_until("the sentinel to be ready", || ready.exists());
        Self(child)
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    /// Whether no one else signalled it: asked with SIGUSR1, it exits 0,
    /// where a SIGKILL sent to it earlier has ended it already, by SIGKILL,
    /// which the kernel acts on before any later signal. Either way it ends
    /// at once, and is reaped here.
    fn untouched(&mut self) -> bool {
        let pid = libc::pid_t::try_from(self.0.id()).expect("a process id");
        // SAFETY: kill only sends a signal, to this test's own child, which
        // is unreaped (it is reaped only below, once it has ended).
        unsafe { libc::kill(pid, libc::SIGUSR1) };
        let mut status = None;
        wait_until("the sentinel to end", || {
            status = self.0.try_wait().expect("wait for the sentinel");
            status.is_some()
        });
        status.and_then(|status| status.code()) == Some(0)
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// @test A run's cleanup ends what bench left in its group, and nothing
/// else: the command leaves a process in bench's group and exits, bench
/// resets the clocks and exits 0, and command.pid then names a process this
/// test owns outside the group, as a stale file does once the command was
/// reaped and its number reused. Dropping the run ends the process left
/// behind (no process holds bench's descriptor 3 any more) and leaves the
/// test's own process running.
#[test]
fn cleanup_ends_what_bench_left_in_its_group_and_nothing_else() {
    let rig = Rig::new();
    let mut sentinel = Sentinel::start(&rig.path("sentinel.ready"));
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let mut run = Run::start_with(
        &rig,
        &wrap(&["/bin/sh", "-c", "/bin/sleep 60 & exit 0"]),
        Parent::Default,
        None,
        Some(holder),
    );
    assert_eq!(run.end().code(), Some(0), "{}", rig.stderr());
    assert_eq!(rig.events(), [LOCK, "lock done", RESET, "reset done"]);
    fs::write(rig.path("command.pid"), sentinel.pid().to_string()).expect("a stale command.pid");
    drop(run);
    assert!(
        sentinel.untouched(),
        "the cleanup signalled a process outside bench's group"
    );
    read_to_eof(&mut held, "the process bench left in its group to end");
}

/// @test A run dropped while bench and its command run, as when a test
/// fails partway: the cleanup ends both, and nothing else, though
/// command.pid names a process this test owns outside bench's group.
#[test]
fn cleanup_after_a_failure_ends_bench_and_its_command_and_nothing_else() {
    let place = tempfile::tempdir().expect("tempdir");
    let mut sentinel = Sentinel::start(&place.path().join("sentinel.ready"));
    let stale = sentinel.pid().to_string();
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let rig = Rig::new();
        let _run = Run::start_with(
            &rig,
            &wrap(&["/bin/sh", "-c", WAITING]),
            Parent::Default,
            None,
            Some(holder),
        );
        rig.pid("command.pid");
        fs::write(rig.path("command.pid"), stale).expect("a stale command.pid");
        panic!("a test that fails while bench and its command run");
    }));
    assert!(failed.is_err());
    assert!(
        sentinel.untouched(),
        "the cleanup signalled a process outside bench's group"
    );
    read_to_eof(&mut held, "bench and its command to end");
}

/// @test A run dropped while nvidia-smi holds the lock, as when a test fails
/// there: nvidia-smi, in a process group of its own that the cleanup does not
/// signal, ends by itself once the rig's folder is gone, so nothing bench
/// started is left.
#[test]
fn cleanup_after_a_failure_leaves_no_held_nvidia_smi() {
    let (mut held, holder) = std::io::pipe().expect("a pipe");
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let rig = Rig::new();
        rig.gate("lock");
        let _run = Run::start_with(
            &rig,
            &wrap(&["/bin/true"]),
            Parent::Default,
            None,
            Some(holder),
        );
        wait_until("the lock to wait", || {
            rig.events().iter().any(|e| e == "lock waiting")
        });
        panic!("a test that fails while nvidia-smi holds the lock");
    }));
    assert!(failed.is_err());
    read_to_eof(&mut held, "bench and the held nvidia-smi to end");
}

/* ----------------------------- Bounds ----------------------------- */

/// nvidia-smi linked into @p rig, with the short bound.
fn short_tool(rig: &Rig) -> Tool {
    Tool {
        program: rig.path("nvidia-smi"),
        bound: SHORT_BOUND,
    }
}

/// @test A lock still running at its bound is ended with its process group,
/// and the clocks are reset, since it may have taken effect.
#[test]
fn a_lock_past_its_bound_is_ended_and_reset() {
    let rig = Rig::new();
    rig.hang("lock");
    let cfg = LockConfig {
        device: 0,
        freq: Some(1000),
        tool: short_tool(&rig),
    };
    let err = gpu_lock::lock_clocks(&cfg).expect_err("a lock past its bound");
    assert_eq!(
        err.to_string(),
        "nvidia-smi -lgc did not finish within 0.2 s and was ended; the clocks were reset, \
         since it may have taken effect"
    );
    assert_eq!(rig.events(), [LOCK, RESET, "reset done"]);
    assert!(
        rig.hang_ended(),
        "the hung lock still runs, or never said it holds hang.fifo"
    );
}

/// @test A reset still running at its bound is ended with its process group,
/// and the error says how to reset the clocks by hand.
#[test]
fn a_reset_past_its_bound_says_how_to_reset_by_hand() {
    let rig = Rig::new();
    rig.hang("reset");
    let err = gpu_lock::reset_clocks(0, &short_tool(&rig)).expect_err("a reset past its bound");
    assert_eq!(
        err.to_string(),
        "the clocks of GPU 0 were not reset: nvidia-smi -rgc did not finish within 0.2 s and \
         was ended; reset them with `bench gpu-lock reset --device 0`, or as root with \
         `nvidia-smi -i 0 -rgc`"
    );
    assert!(
        rig.hang_ended(),
        "the hung reset still runs, or never said it holds hang.fifo"
    );
}

/// @test `hang_ended` does not take a stand-in that has not yet opened
/// `hang.fifo` for one that has ended. Started here, held at its gate before
/// the open and never stopped, it is not seen to end within 300 ms; once it
/// holds the FIFO it is still not, while it runs; once it ends by itself
/// (`hang.end`), it is. Should the test fail first, the stand-in ends once
/// the rig's folder is gone.
#[test]
fn hang_ended_waits_for_the_stand_in_to_hold_the_fifo() {
    let rig = Rig::new();
    rig.hang("lock");
    rig.gate("lock");
    let mut stand_in = Command::new(rig.path("nvidia-smi"))
        .args(["-i", "0", "-lgc", "1000,1000"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start the stand-in");
    wait_until("the stand-in to wait at its gate", || {
        rig.events().iter().any(|e| e == "lock waiting")
    });
    assert!(
        !rig.hang_ended_within(Duration::from_millis(300)),
        "a stand-in yet to open hang.fifo was taken for one that ended"
    );
    rig.open("lock");
    wait_until("the stand-in to hold hang.fifo", || {
        rig.path("hang.held").exists()
    });
    assert!(
        !rig.hang_ended_within(Duration::from_millis(100)),
        "a stand-in holding hang.fifo was taken for one that ended"
    );
    fs::write(rig.path("hang.end"), "").expect("end the stand-in");
    assert!(rig.hang_ended(), "the stand-in did not end");
    let status = stand_in.wait().expect("reap the stand-in");
    assert_eq!(status.code(), Some(1), "{status:?}");
}
