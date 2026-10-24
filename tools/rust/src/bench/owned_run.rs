//! A helper program a bench command owns until it returns: bench run's
//! annotator and `nsys stats` (`run_owned`), and bench validate's version
//! probes (`validate::run_version`), each with its own caller policy. The
//! program runs in a process group of its own, made when it is spawned, so
//! the processes it starts stay in that group after it exits. When the
//! program exits, outruns its bound or the command is interrupted, the whole
//! group is ended, the run waits (up to `GONE_WAIT`) until no process of the
//! group runs, and the program's output is read to its end; a pipe still held
//! open after that (by a process that left the group, which the run does not
//! own) is read only as far as it has been written, never waited on.
//!
//! The invariants the `unsafe` calls below rely on:
//! - The group's id is the program's process id, reserved while the program
//!   is unreaped: the group is signalled only before `Child::wait` reaps the
//!   program, and the program's exit is seen with `WNOWAIT`, which does not
//!   reap it.
//! - Nothing else reaps the program: before it is spawned, SIGCHLD gets its
//!   default action without SA_NOCLDWAIT, for the rest of the process (as the
//!   Nsight executor's runs do), so an ignored SIGCHLD inherited across exec
//!   or SA_NOCLDWAIT cannot have the system reap it as it exits. A program
//!   whose end cannot be observed (another wait in this process took it) is
//!   not the run's any more: its group is not signalled.
//! - The signal handler only stores the signal's number in an atomic.
//! - Only the read ends of the program's own pipes are made non-blocking.

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
pub(crate) enum Ending {
    /// The program exited by itself.
    Exited(ExitStatus),
    /// The bound passed first; the group was ended.
    TimedOut,
    /// The command received this signal; the group was ended.
    Interrupted(i32),
}

/// A finished run: how it ended, the program's output, and how many
/// processes of its group still ran when it exited (ended by the run).
pub(crate) struct Finished {
    pub(crate) ending: Ending,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) left_running: usize,
}

/// Start @p command with stdin closed and both outputs piped, in a
/// process group of its own, once this process keeps its children
/// waitable (`super::runner::keep_children_waitable`); nothing starts where it
/// cannot.
pub(crate) fn spawn(command: &mut Command) -> io::Result<Child> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    super::runner::keep_children_waitable()?;
    command.spawn()
}

/// Wait for @p child (from `spawn`) until it exits, @p bound passes or
/// @p interrupted holds a signal's number; then end its group, wait for
/// the group to be gone and collect the output. A program whose end
/// cannot be observed is left with its group (`unowned`).
pub(crate) fn finish(
    mut child: Child,
    bound: Duration,
    interrupted: &AtomicI32,
) -> io::Result<Finished> {
    let stop = Arc::new(AtomicBool::new(false));
    let stdout = read_to_end(child.stdout.take(), &stop);
    let stderr = read_to_end(child.stderr.take(), &stop);
    let started = Instant::now();
    let ending = loop {
        match look(&mut child) {
            Seen::Running => {}
            Seen::Exited => break None,
            Seen::Lost(e) => return unowned(e, stdout, stderr, &stop, interrupted),
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
    // Looked at again just before the group is signalled.
    if let Seen::Lost(e) = look(&mut child) {
        return unowned(e, stdout, stderr, &stop, interrupted);
    }
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

/// The run of a program whose end this process could not observe (@p e):
/// another wait took it, so its process id, and the group's, may have
/// passed to another process, and nothing is signalled. The output read
/// so far is kept; a signal the command received still ends it; otherwise
/// the error says what happened.
fn unowned(
    e: io::Error,
    stdout: JoinHandle<String>,
    stderr: JoinHandle<String>,
    stop: &AtomicBool,
    interrupted: &AtomicI32,
) -> io::Result<Finished> {
    stop.store(true, Ordering::SeqCst);
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    match interrupted.load(Ordering::SeqCst) {
        0 => Err(io::Error::new(
            e.kind(),
            format!(
                "the program's end could not be observed ({e}): another wait in this \
                 process took it, so its process group was left as it is"
            ),
        )),
        signal => Ok(Finished {
            ending: Ending::Interrupted(signal),
            stdout,
            stderr,
            left_running: 0,
        }),
    }
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

/// How the program stands.
enum Seen {
    Running,
    /// Exited, not yet reaped: its process id, the group's id, is still
    /// its own.
    Exited,
    /// Its end could not be observed (another wait in this process took
    /// it): the group's id may have passed on.
    Lost(io::Error),
}

/// How @p child stands, without reaping it.
#[cfg(unix)]
fn look(child: &mut Child) -> Seen {
    let pid: libc::id_t = child.id();
    loop {
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
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Seen::Lost(e);
        }
        // SAFETY: waitid filled `info`; with WNOHANG its process id stays
        // 0 while the child runs.
        return if unsafe { info.si_pid() } != 0 {
            Seen::Exited
        } else {
            Seen::Running
        };
    }
}

/// How @p child stands (reaped here once it has exited; `Child::wait`
/// returns the status it kept).
#[cfg(not(unix))]
fn look(child: &mut Child) -> Seen {
    match child.try_wait() {
        Ok(Some(_)) => Seen::Exited,
        Ok(None) => Seen::Running,
        Err(e) => Seen::Lost(e),
    }
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
/// Ctrl-C reaches the bench command alone. A signal whose action is not the
/// default (ignored, as under nohup, or handled) is left as it is.
/// Dropping it restores each action it replaced.
pub(crate) struct Watch {
    #[cfg(unix)]
    replaced: Vec<(libc::c_int, libc::sigaction)>,
}

impl Watch {
    pub(crate) fn start() -> Self {
        PENDING.store(0, Ordering::SeqCst);
        Self {
            #[cfg(unix)]
            replaced: record_signals(),
        }
    }

    /// The flag the recorded signal's number is stored in.
    pub(crate) fn flag(&self) -> &'static AtomicI32 {
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

/// End the bench command by @p signal, after the `Watch` that recorded it
/// is dropped (its default action restored), so whoever sent it sees the
/// command ended by it.
pub(crate) fn end_by(signal: i32) -> ! {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{owned_run, tempfile_dir};
    use std::fs;
    use std::path::{Path, PathBuf};

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

    /// Shell that waits, 5 s at the most, until the process `$!` leads a
    /// session of its own, as setsid makes it: only then has it left the
    /// group the script runs in.
    #[cfg(target_os = "linux")]
    const UNTIL_ITS_OWN_SESSION: &str = "i=0; while read -r s < /proc/$!/stat && set -- $s && \
        [ \"$6\" != \"$!\" ] && [ $i -lt 500 ]; do sleep 0.01; i=$((i + 1)); done;";

    /// @test A process that leaves the program's group (setsid) is not
    /// owned: the run neither ends it nor waits on the output it holds open,
    /// and keeps what the program wrote. The program exits only once that
    /// process has left its group, which a background start may take longer
    /// to do than the run takes to see the program exit.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_does_not_wait_on_a_process_outside_its_group() {
        let dir = tempfile_dir("owned_run_outside");
        let outside = dir.join("outside.pid");
        let (run, took) = owned_sh(
            &format!(
                "echo out; /usr/bin/setsid /bin/sleep 60 & echo $! > '{}'; {} exit 0",
                outside.display(),
                UNTIL_ITS_OWN_SESSION
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

    /// The SIGCHLD setting `owned_run_in_a_process_of_its_own` runs under,
    /// and the file it writes what it saw to.
    #[cfg(target_os = "linux")]
    const POLICY_VARIABLE: &str = "VERNIER_OWNED_RUN_POLICY";
    #[cfg(target_os = "linux")]
    const REPORT_VARIABLE: &str = "VERNIER_OWNED_RUN_REPORT";

    /// How SIGCHLD stands in this process: "default", "ignored" or
    /// "handled", and "with SA_NOCLDWAIT" when that flag is set.
    #[cfg(target_os = "linux")]
    fn sigchld_setting() -> String {
        // SAFETY: a zeroed sigaction is a valid value of the type; sigaction
        // only writes the current action into `now`, owned here.
        let now = unsafe {
            let mut now: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGCHLD, std::ptr::null(), &mut now);
            now
        };
        let action = match now.sa_sigaction {
            libc::SIG_DFL => "default",
            libc::SIG_IGN => "ignored",
            _ => "handled",
        };
        if now.sa_flags & libc::SA_NOCLDWAIT != 0 {
            format!("{action} with SA_NOCLDWAIT")
        } else {
            action.to_string()
        }
    }

    /// Not a test of its own: the body of
    /// `owned_run_keeps_its_program_whatever_reaps_children`, which runs this
    /// test binary for it in a process of its own; without that test's
    /// variables it does nothing. "ignored" is inherited across exec;
    /// "nocldwait" is set here, since the flag does not survive exec.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_in_a_process_of_its_own() {
        let Ok(policy) = std::env::var(POLICY_VARIABLE) else {
            return;
        };
        let report = PathBuf::from(std::env::var(REPORT_VARIABLE).expect("the report's path"));
        if policy == "nocldwait" {
            // SAFETY: a zeroed sigaction is a valid value of the type;
            // sigaction only reads `action`, owned here; the default action
            // with SA_NOCLDWAIT installs no handler.
            let rc = unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = libc::SIG_DFL;
                action.sa_flags = libc::SA_NOCLDWAIT;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut())
            };
            assert_eq!(rc, 0, "set SA_NOCLDWAIT");
        }
        let before = sigchld_setting();
        let dir = tempfile_dir(&format!("owned_run_policy_{policy}"));
        let left = dir.join("left.pid");
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            &format!(
                "echo out; /bin/sleep 60 & echo $! > '{}'; exit 3",
                left.display()
            ),
        ]);
        let run = owned_run::spawn(&mut command).and_then(|child| {
            owned_run::finish(
                child,
                std::time::Duration::from_secs(30),
                &std::sync::atomic::AtomicI32::new(0),
            )
        });
        let left = written_pid(&left);
        let left_runs = still_runs(left);
        if left_runs {
            // This test started that process, through the script, and ends it.
            let _ = Command::new("/bin/sh")
                .args(["-c", &format!("kill -KILL {left}")])
                .status();
        }
        let line = match run {
            Ok(run) => {
                let ending = match run.ending {
                    owned_run::Ending::Exited(status) => format!("exited {:?}", status.code()),
                    owned_run::Ending::TimedOut => "timed out".to_string(),
                    owned_run::Ending::Interrupted(signal) => format!("interrupted by {signal}"),
                };
                format!(
                    "before: {before}; {ending}; stdout {:?}; left running {}; the process left \
                     still runs: {left_runs}; after: {}",
                    run.stdout,
                    run.left_running,
                    sigchld_setting()
                )
            }
            Err(e) => format!("before: {before}; the run failed: {e}"),
        };
        fs::write(&report, line).expect("write the report");
        let _ = fs::remove_dir_all(&dir);
    }

    /// @test An owned run keeps its program and its group its own whatever
    /// this process inherited or set for SIGCHLD: at its default action,
    /// ignored (inherited across exec, as by a process started with it
    /// ignored) and with SA_NOCLDWAIT, under which the system would reap the
    /// program as it exits. Each in a process of its own (this test binary,
    /// running `owned_run_in_a_process_of_its_own`): the run reports the
    /// program's own status and output, ends the process the program left in
    /// its group, and leaves SIGCHLD at its default action.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_keeps_its_program_whatever_reaps_children() {
        use std::os::unix::process::CommandExt;

        let dir = tempfile_dir("owned_run_policies");
        let name = format!(
            "{}::owned_run_in_a_process_of_its_own",
            module_path!().split_once("::").map_or("", |(_, path)| path)
        );
        let mut seen = Vec::new();
        for (policy, before) in [
            ("default", "default"),
            ("ignored", "ignored"),
            ("nocldwait", "default with SA_NOCLDWAIT"),
        ] {
            let report = dir.join(format!("{policy}.txt"));
            let mut command = Command::new(std::env::current_exe().expect("this test binary"));
            command
                .args(["--exact", &name, "--nocapture", "--test-threads=1"])
                .env(POLICY_VARIABLE, policy)
                .env(REPORT_VARIABLE, &report);
            let handler = if policy == "ignored" {
                libc::SIG_IGN
            } else {
                libc::SIG_DFL
            };
            // SAFETY: the closure runs in the child between fork and exec and
            // calls only sigaction, which is async-signal-safe; a zeroed
            // sigaction is a valid value of the type, given the default or the
            // ignore handler.
            unsafe {
                command.pre_exec(move || {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = handler;
                    if libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut()) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let out = command.output().expect("run this test binary");
            let line = fs::read_to_string(&report).unwrap_or_default();
            let expected = format!(
                "before: {before}; exited Some(3); stdout \"out\\n\"; left running 1; the process \
                 left still runs: false; after: default"
            );
            if !out.status.success() || line != expected {
                seen.push(format!(
                    "{policy}: {line}\n{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
        }
        let _ = fs::remove_dir_all(&dir);
        assert!(seen.is_empty(), "{}", seen.join("\n"));
    }

    /// @test A program another wait in this process reaped (against the
    /// ownership of children the runner keeps) is not the run's any more: its
    /// group's id may have passed on, so the run signals nothing and says that
    /// it could not observe the end. Here the program leaves a process in its
    /// group, which keeps the group's id from passing to another group: a
    /// signal to the group would end that process, and it still runs after
    /// the run (the test then ends it).
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_does_not_signal_a_group_it_no_longer_owns() {
        let dir = tempfile_dir("owned_run_reaped_elsewhere");
        let left = dir.join("left.pid");
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            &format!(
                "/bin/sleep 60 > /dev/null 2>&1 & echo $! > '{}'; exit 0",
                left.display()
            ),
        ]);
        let child = owned_run::spawn(&mut command).expect("start sh");
        let pid = libc::pid_t::try_from(child.id()).expect("a process id");
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| Some(stat.rsplit_once(')')?.1.trim_start().starts_with('Z')))
            .unwrap_or(false)
            && std::time::Instant::now() < until
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let mut status = 0;
        // SAFETY: waitpid only writes the status into `status`, owned here;
        // it reaps this test's own child, the run's program, which has exited.
        let reaped = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(reaped, pid, "reap the program here");
        let run = owned_run::finish(
            child,
            std::time::Duration::from_secs(30),
            &std::sync::atomic::AtomicI32::new(0),
        );
        let left = written_pid(&left);
        let left_runs = still_runs(left);
        // This test started that process, through the script, and ends it.
        let _ = Command::new("/bin/sh")
            .args(["-c", &format!("kill -KILL {left}")])
            .status();
        let _ = fs::remove_dir_all(&dir);
        assert!(
            left_runs,
            "the run signalled the group of a program another wait had reaped"
        );
        let Err(error) = run else {
            panic!("the run reported an end it could not observe");
        };
        assert!(
            error.to_string().contains("could not be observed"),
            "{error}"
        );
    }
}
