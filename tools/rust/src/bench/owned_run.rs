//! A helper program a bench command owns until it returns: bench run's
//! annotator and `nsys stats` (`run_owned`), bench validate's version
//! probes (`validate::run_version`) and bench gpu-lock's nvidia-smi calls
//! (`gpu_lock`), each with its own caller policy. The
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
//! - The signal handler only stores to lock-free atomics: the latest signal,
//!   the first one (a compare-and-swap, so a later signal cannot replace it)
//!   and a bit for each signal that arrived.
//! - A stopped watch (`Watch::stop`) blocks SIGINT, SIGTERM and SIGHUP in its
//!   thread before it puts their actions back, and `end_by` raises the first
//!   signal while it is blocked and then unblocks it alone, so that no later
//!   one can end the process before it; a thread started under
//!   `with_watched_blocked` never takes one of them.
//! - A program started under `end_if_already_signalled` reads the record it
//!   inherited, the watch's at the moment the child was created, in the child
//!   before exec, with only async-signal-safe calls.
//! - Only the read ends of the program's own pipes are made non-blocking.

use std::io::{self, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
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

/// The signal a `Watch` recorded last; 0 for none.
static PENDING: AtomicI32 = AtomicI32::new(0);

/// The first signal a `Watch` recorded; 0 for none. Set once, by a
/// compare-and-swap, so a later signal cannot replace it.
static FIRST: AtomicI32 = AtomicI32::new(0);

/// One bit per signal number a `Watch` recorded since its last look
/// (`Watch::arrivals`). Arrivals are not counted: one signal arriving twice
/// between two looks is one, as the kernel keeps one pending instance of it.
static ARRIVED: AtomicU32 = AtomicU32::new(0);

/// The signals a `Watch` records.
#[cfg(unix)]
const WATCHED: [libc::c_int; 3] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP];

#[cfg(unix)]
extern "C" fn record(signal: libc::c_int) {
    // The first is settled before the arrival can be seen.
    let _ = FIRST.compare_exchange(0, signal, Ordering::SeqCst, Ordering::SeqCst);
    if (0..32).contains(&signal) {
        ARRIVED.fetch_or(1 << signal, Ordering::SeqCst);
    }
    PENDING.store(signal, Ordering::SeqCst);
}

/// While it lives, SIGINT, SIGTERM and SIGHUP are recorded for the bench
/// command to act on: an owned run ends its group, which is not the
/// terminal's foreground group, so a Ctrl-C reaches the bench command alone;
/// bench gpu-lock resets the clocks once its command, which stays in the
/// terminal's group, has ended. A signal whose action is not the default
/// (ignored, as under nohup, or handled) is left as it is. Dropping it
/// restores each action it replaced; stopping it (`stop`) does too, and
/// keeps the first signal recorded ahead of any that follows.
pub(crate) struct Watch {
    #[cfg(unix)]
    replaced: Vec<(libc::c_int, libc::sigaction)>,
}

impl Watch {
    /// Forget what an earlier watch recorded, and record SIGINT, SIGTERM and
    /// SIGHUP where their action is the default.
    pub(crate) fn start() -> Self {
        PENDING.store(0, Ordering::SeqCst);
        FIRST.store(0, Ordering::SeqCst);
        ARRIVED.store(0, Ordering::SeqCst);
        Self {
            #[cfg(unix)]
            replaced: record_signals(),
        }
    }

    /// The flag the latest recorded signal's number is stored in.
    pub(crate) fn flag(&self) -> &'static AtomicI32 {
        &PENDING
    }

    /// The first signal recorded since the watch started, if one was.
    pub(crate) fn first(&self) -> Option<i32> {
        match FIRST.load(Ordering::SeqCst) {
            0 => None,
            signal => Some(signal),
        }
    }

    /// The signals recorded since the last look, or since the start, lowest
    /// number first; a signal that arrived more than once in between is
    /// reported once.
    pub(crate) fn arrivals(&self) -> Vec<i32> {
        let bits = ARRIVED.swap(0, Ordering::SeqCst);
        (0..32).filter(|signal| bits & (1 << signal) != 0).collect()
    }

    /// Stop recording (`stop`); then, if a signal was recorded, call
    /// @p before with it and end the process by it (`end_by`). A signal that
    /// arrives once the actions are back, @p before running included, is
    /// held blocked meanwhile, so none can end the process before the first.
    /// With none recorded, returns once any signal held meanwhile has taken
    /// its own action.
    pub(crate) fn end_by_first(self, before: impl FnOnce(i32)) {
        let stopped = self.stop();
        if let Some(signal) = stopped.first() {
            before(signal);
            end_by(signal);
        }
    }

    /// Stop recording: block SIGINT, SIGTERM and SIGHUP in this thread, put
    /// back each action the watch replaced, then read the first signal
    /// recorded. A signal that arrives before the block is recorded; one that
    /// arrives after is held, blocked, and takes its own action once the
    /// returned `Stopped` is dropped, so none is lost; or never, if `end_by`
    /// ends the process by the first signal before, which then no later one
    /// can preempt.
    fn stop(mut self) -> Stopped {
        #[cfg(unix)]
        let blocked = Blocked::watched();
        #[cfg(unix)]
        restore_signals(&std::mem::take(&mut self.replaced));
        Stopped {
            first: match FIRST.load(Ordering::SeqCst) {
                0 => None,
                signal => Some(signal),
            },
            #[cfg(unix)]
            _blocked: blocked,
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        #[cfg(unix)]
        restore_signals(&self.replaced);
    }
}

/// A stopped `Watch` (`Watch::stop`): the actions it replaced are back, and
/// SIGINT, SIGTERM and SIGHUP stay blocked in the thread that stopped it
/// until it is dropped.
struct Stopped {
    first: Option<i32>,
    #[cfg(unix)]
    _blocked: Blocked,
}

impl Stopped {
    /// The first signal the watch recorded, if one was.
    fn first(&self) -> Option<i32> {
        self.first
    }
}

/// While it lives, SIGINT, SIGTERM and SIGHUP are blocked in the thread
/// that made it; dropping it puts that thread's signal mask back.
#[cfg(unix)]
struct Blocked {
    before: libc::sigset_t,
}

#[cfg(unix)]
impl Blocked {
    fn watched() -> Self {
        // SAFETY: zeroed sigset_t values are valid values of the type, and
        // sigemptyset initialises `watched`; pthread_sigmask only reads
        // `watched` and writes `before`, both owned here.
        unsafe {
            let mut watched: libc::sigset_t = std::mem::zeroed();
            let mut before: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut watched);
            for signal in WATCHED {
                libc::sigaddset(&mut watched, signal);
            }
            libc::pthread_sigmask(libc::SIG_BLOCK, &watched, &mut before);
            Self { before }
        }
    }
}

#[cfg(unix)]
impl Drop for Blocked {
    fn drop(&mut self) {
        // SAFETY: pthread_sigmask only reads `before`, this thread's mask
        // from before the block.
        unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &self.before, std::ptr::null_mut()) };
    }
}

/// Run @p start with SIGINT, SIGTERM and SIGHUP blocked in this thread, then
/// put this thread's mask back. A thread @p start starts keeps them blocked
/// for good, so it never takes one: they reach the thread that acts on them,
/// where a `Stopped` can hold them.
pub(crate) fn with_watched_blocked<T>(start: impl FnOnce() -> T) -> T {
    #[cfg(unix)]
    let _blocked = Blocked::watched();
    start()
}

/// Have @p command, started while a `Watch` records, end by a watched signal
/// recorded before it was created: in the child, before exec, it puts back
/// the default action of each watched signal it does not ignore and, when
/// the record it inherited holds a first signal, raises that signal on
/// itself. Its record is the watch's at the moment the child was created.
/// On Linux a signal sent to the process group is either handled before
/// that moment (a fork that meets a pending signal is restarted after the
/// handler has run) or delivered to the child as well, so a Ctrl-C that
/// reaches bench as its command starts ends the command once, either way.
#[cfg(unix)]
pub(crate) fn end_if_already_signalled(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: the closure runs in the child between fork and exec and calls
    // only sigaction, raise and _exit, which are async-signal-safe, and loads
    // an atomic; zeroed sigaction values are valid values of the type.
    unsafe {
        command.pre_exec(|| {
            for signal in WATCHED {
                let mut now: libc::sigaction = std::mem::zeroed();
                if libc::sigaction(signal, std::ptr::null(), &mut now) == 0
                    && now.sa_sigaction != libc::SIG_IGN
                {
                    let mut default: libc::sigaction = std::mem::zeroed();
                    default.sa_sigaction = libc::SIG_DFL;
                    libc::sigaction(signal, &default, std::ptr::null_mut());
                }
            }
            let first = FIRST.load(Ordering::SeqCst);
            if first != 0 {
                libc::raise(first);
                // raise returns only when the signal is blocked: end with
                // the status a shell gives a program the signal ended.
                libc::_exit(128 + first);
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
pub(crate) fn end_if_already_signalled(_command: &mut Command) {}

/// Record SIGINT, SIGTERM and SIGHUP (`record`) where their action is the
/// default; returns each replaced action.
#[cfg(unix)]
fn record_signals() -> Vec<(libc::c_int, libc::sigaction)> {
    let mut replaced = Vec::new();
    for signal in WATCHED {
        // SAFETY: zeroed sigaction values are valid values of the type;
        // sigaction only reads `action` and writes `old`, both owned here;
        // the handler installed only stores to lock-free atomics.
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
/// is dropped or stopped (its default action restored), so whoever sent it
/// sees the command ended by it. Under a `Stopped`, which blocks it, @p
/// signal is raised and then unblocked alone: it ends the process as it is
/// unblocked, and the other signals held there stay blocked, so none of them
/// can end it first.
pub(crate) fn end_by(signal: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    // SAFETY: raise only sends @p signal to this thread; a zeroed sigset_t is
    // a valid value of the type, initialised by sigemptyset; pthread_sigmask
    // only reads `only`, which holds @p signal alone, owned here.
    unsafe {
        libc::raise(signal);
        let mut only: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut only);
        libc::sigaddset(&mut only, signal);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &only, std::ptr::null_mut());
    }
    std::process::exit(128 + signal)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::bench::HeldFifo;
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

    /// Shell that starts, in a session of its own (setsid), a process that
    /// holds the script's output and the FIFO @p fifo open and runs until
    /// @p dir is gone; it waits, 5 s at the most, until that process has
    /// created @p ready, which it does once it has left the script's group
    /// and opened the FIFO.
    #[cfg(target_os = "linux")]
    fn outside_the_group(fifo: &Path, ready: &Path, dir: &Path) -> String {
        format!(
            "/usr/bin/setsid /bin/sh -c 'exec 9<>\"$0\"; : > \"$1\"; \
             while [ -d \"$2\" ]; do /bin/sleep 0.05; done' '{}' '{}' '{}' & \
             i=0; while [ ! -e '{}' ] && [ $i -lt 500 ]; do sleep 0.01; i=$((i + 1)); done;",
            fifo.display(),
            ready.display(),
            dir.display(),
            ready.display()
        )
    }

    /// @test A process that leaves the program's group (setsid) is not
    /// owned: the run neither ends it nor waits on the output it holds open,
    /// and keeps what the program wrote. The program exits only once that
    /// process has left its group, which a background start may take longer
    /// to do than the run takes to see the program exit. That process holds
    /// the FIFO `outside` and leaves once the test's folder is gone.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_does_not_wait_on_a_process_outside_its_group() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fifo = dir.path().join("outside");
        let mut outside = HeldFifo::new(&fifo);
        let (run, took) = owned_sh(
            &format!(
                "echo out; {} exit 0",
                outside_the_group(&fifo, &dir.path().join("ready"), dir.path())
            ),
            std::time::Duration::from_secs(30),
            &std::sync::atomic::AtomicI32::new(0),
        );
        let outside_runs = outside.held();
        drop(dir);
        let outside_left = outside.ends_within(std::time::Duration::from_secs(5));
        assert!(
            took < std::time::Duration::from_secs(5),
            "the run waited on a pipe held outside its group: {took:?}"
        );
        assert!(outside_runs, "a process outside the group was ended");
        assert!(
            outside_left,
            "the process outside the group outlived its folder"
        );
        assert_eq!(run.left_running, 0);
        assert_eq!(run.stdout, "out\n");
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
        // The process the program leaves holds it open while it runs, and
        // creates `ready` once it does; the program exits only then (5 s at
        // the most), so the FIFO's end, read after the run, is that process's
        // end and not a start still to come.
        let mut left = HeldFifo::new(&dir.join("left"));
        let ready = dir.join("ready");
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            &format!(
                "echo out; /bin/sh -c 'exec 9<>\"$0\"; : > \"$1\"; exec /bin/sleep 60' '{}' '{}' & \
                 i=0; while [ ! -e '{}' ] && [ $i -lt 500 ]; do sleep 0.01; i=$((i + 1)); done; \
                 exit 3",
                dir.join("left").display(),
                ready.display(),
                ready.display()
            ),
        ]);
        let run = owned_run::spawn(&mut command).and_then(|child| {
            owned_run::finish(
                child,
                std::time::Duration::from_secs(30),
                &std::sync::atomic::AtomicI32::new(0),
            )
        });
        let left_held = ready.exists();
        let left_runs = left.held();
        let line = match run {
            Ok(run) => {
                let ending = match run.ending {
                    owned_run::Ending::Exited(status) => format!("exited {:?}", status.code()),
                    owned_run::Ending::TimedOut => "timed out".to_string(),
                    owned_run::Ending::Interrupted(signal) => format!("interrupted by {signal}"),
                };
                format!(
                    "before: {before}; {ending}; stdout {:?}; left running {}; the process left \
                     held its FIFO: {left_held}; still runs: {left_runs}; after: {}",
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
                 left held its FIFO: true; still runs: false; after: default"
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
    /// the run. That process holds the FIFO `left` and leaves once the test's
    /// folder is gone.
    #[test]
    #[cfg(target_os = "linux")]
    fn owned_run_does_not_signal_a_group_it_no_longer_owns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fifo = dir.path().join("left");
        let mut left = HeldFifo::new(&fifo);
        let ready = dir.path().join("ready");
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            &format!(
                "/bin/sh -c 'exec 9<>\"$0\"; : > \"$1\"; \
                 while [ -d \"$2\" ]; do /bin/sleep 0.05; done' '{}' '{}' '{}' \
                 > /dev/null 2>&1 & \
                 i=0; while [ ! -e '{}' ] && [ $i -lt 500 ]; do sleep 0.01; i=$((i + 1)); done; \
                 exit 0",
                fifo.display(),
                ready.display(),
                dir.path().display(),
                ready.display()
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
        let left_runs = left.held();
        drop(dir);
        let left_ended = left.ends_within(std::time::Duration::from_secs(5));
        assert!(
            left_runs,
            "the run signalled the group of a program another wait had reaped"
        );
        assert!(
            left_ended,
            "the process the program left outlived the test's folder"
        );
        let Err(error) = run else {
            panic!("the run reported an end it could not observe");
        };
        assert!(
            error.to_string().contains("could not be observed"),
            "{error}"
        );
    }

    /* ----------------------------- Watch ----------------------------- */

    /// The file `watch_keeps_the_first_signal`'s process of its own writes
    /// what its watch recorded to.
    #[cfg(target_os = "linux")]
    const WATCH_REPORT_VARIABLE: &str = "VERNIER_WATCH_REPORT";

    /// Whether SIGINT, SIGTERM and SIGHUP all have their default action.
    #[cfg(target_os = "linux")]
    fn watched_signals_at_default() -> bool {
        [libc::SIGINT, libc::SIGTERM, libc::SIGHUP]
            .iter()
            .all(|&signal| {
                // SAFETY: a zeroed sigaction is a valid value of the type;
                // with a null new action, sigaction only writes the current
                // one into `now`, owned here.
                unsafe {
                    let mut now: libc::sigaction = std::mem::zeroed();
                    libc::sigaction(signal, std::ptr::null(), &mut now) == 0
                        && now.sa_sigaction == libc::SIG_DFL
                }
            })
    }

    /// Not a test of its own: the body of `watch_keeps_the_first_signal`,
    /// which runs this test binary for it in a process of its own, since a
    /// signal's action is the process's; without that test's variable it
    /// does nothing. raise() runs the handler before it returns, so the
    /// signals are recorded in the order they are raised.
    #[test]
    #[cfg(target_os = "linux")]
    fn watch_in_a_process_of_its_own() {
        let Ok(report) = std::env::var(WATCH_REPORT_VARIABLE) else {
            return;
        };
        let raise = |signal: libc::c_int| {
            // SAFETY: raise only sends @p signal to this thread, whose action
            // for it is the watch's handler, which only stores to atomics.
            unsafe { libc::raise(signal) };
        };
        let watch = Watch::start();
        raise(libc::SIGTERM);
        raise(libc::SIGINT);
        raise(libc::SIGTERM);
        let arrived = watch.arrivals();
        raise(libc::SIGHUP);
        let then = watch.arrivals();
        let latest = watch.flag().load(Ordering::SeqCst);
        let first = watch.first();
        let stopped = watch.stop().first();
        let line = format!(
            "first {first:?}; arrived {arrived:?}; then {then:?}; latest {latest}; stopped \
             {stopped:?}; defaults back {}",
            watched_signals_at_default()
        );
        fs::write(report, line).expect("write the report");
    }

    /// @test A watch keeps the first signal it records, whatever arrives
    /// after it (it also stores the latest); reports each signal that arrived
    /// since its last look once, however often it arrived; and on stopping
    /// puts the default actions back and still returns the first. In a
    /// process of its own (this test binary, running
    /// `watch_in_a_process_of_its_own`), with SIGINT, SIGTERM and SIGHUP at
    /// their default actions, raised in the order SIGTERM, SIGINT, SIGTERM,
    /// SIGHUP.
    #[test]
    #[cfg(target_os = "linux")]
    fn watch_keeps_the_first_signal() {
        use std::os::unix::process::CommandExt;

        let dir = tempfile_dir("watch_first");
        let report = dir.join("report.txt");
        let name = format!(
            "{}::watch_in_a_process_of_its_own",
            module_path!().split_once("::").map_or("", |(_, path)| path)
        );
        let mut command = Command::new(std::env::current_exe().expect("this test binary"));
        command
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(WATCH_REPORT_VARIABLE, &report);
        // SAFETY: the closure runs in the child between fork and exec and
        // calls only sigaction, which is async-signal-safe; a zeroed sigaction
        // given the default handler is a valid value of the type.
        unsafe {
            command.pre_exec(|| {
                for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = libc::SIG_DFL;
                    if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let out = command.output().expect("run this test binary");
        let line = fs::read_to_string(&report).unwrap_or_default();
        let _ = fs::remove_dir_all(&dir);
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            line,
            format!(
                "first Some({term}); arrived [{int}, {term}]; then [{hup}]; latest {hup}; \
                 stopped Some({term}); defaults back true",
                term = libc::SIGTERM,
                int = libc::SIGINT,
                hup = libc::SIGHUP
            )
        );
    }

    /// The case `watch_stop_in_a_process_of_its_own` runs.
    #[cfg(target_os = "linux")]
    const STOP_CASE_VARIABLE: &str = "VERNIER_WATCH_STOP_CASE";

    /// What this thread does with SIGINT, SIGTERM and SIGHUP: each one's
    /// action, and those it blocks.
    #[cfg(target_os = "linux")]
    fn watched_signals_state() -> String {
        let actions: Vec<&str> = WATCHED
            .iter()
            .map(|&signal| {
                // SAFETY: a zeroed sigaction is a valid value of the type;
                // with a null new action, sigaction only writes the current
                // one into `now`, owned here.
                let now = unsafe {
                    let mut now: libc::sigaction = std::mem::zeroed();
                    libc::sigaction(signal, std::ptr::null(), &mut now);
                    now
                };
                match now.sa_sigaction {
                    libc::SIG_DFL => "default",
                    libc::SIG_IGN => "ignored",
                    _ => "handled",
                }
            })
            .collect();
        // SAFETY: a zeroed sigset_t is a valid value of the type; with a null
        // set, pthread_sigmask only writes this thread's mask into `mask`,
        // owned here, which sigismember only reads.
        let blocked: Vec<i32> = unsafe {
            let mut mask: libc::sigset_t = std::mem::zeroed();
            libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), &mut mask);
            WATCHED
                .iter()
                .copied()
                .filter(|&signal| libc::sigismember(&mask, signal) == 1)
                .collect()
        };
        format!("actions {actions:?}; blocked {blocked:?}")
    }

    /// Not a test of its own: the body of
    /// `watch_stop_ends_by_the_first_signal_whatever_follows`, which runs this
    /// test binary for it in a process of its own, once for each case;
    /// without that test's variables it does nothing. raise() sends a signal
    /// to this thread alone, so one this thread blocks waits here, whatever
    /// the test harness's other threads do.
    #[test]
    #[cfg(target_os = "linux")]
    fn watch_stop_in_a_process_of_its_own() {
        let Ok(case) = std::env::var(STOP_CASE_VARIABLE) else {
            return;
        };
        let report =
            PathBuf::from(std::env::var(WATCH_REPORT_VARIABLE).expect("the report's path"));
        let raise = |signal: libc::c_int| {
            // SAFETY: raise only sends @p signal to this thread.
            unsafe { libc::raise(signal) };
        };
        let watch = Watch::start();
        if case == "first" {
            // SIGTERM recorded; then, where bench gpu-lock says what went
            // wrong (`end_by_first`'s @p before), the actions being back and
            // the first not raised again yet, a second signal whose default
            // action would end the process by it at once.
            raise(libc::SIGTERM);
            watch.end_by_first(|first| {
                let line = format!("first {first}; {}", watched_signals_state());
                fs::write(&report, line).expect("write the report");
                raise(libc::SIGINT);
            });
            return;
        }
        let stopped = watch.stop();
        let line = format!("first {:?}; {}", stopped.first(), watched_signals_state());
        fs::write(&report, &line).expect("write the report");
        if case == "none" {
            // None recorded: a signal arriving now is held, and takes its own
            // action once the stopped watch is dropped.
            raise(libc::SIGHUP);
            drop(stopped);
        } else {
            // Nothing arrives: dropping the stopped watch puts the mask back.
            drop(stopped);
            fs::write(&report, format!("{line}; then {}", watched_signals_state()))
                .expect("write the report");
        }
    }

    /// @test Stopping a watch keeps the first signal it recorded ahead of
    /// any that follows, through putting the actions back and ending by it,
    /// and puts everything back when none was recorded. Each case runs in a
    /// process of its own (this test binary, running
    /// `watch_stop_in_a_process_of_its_own`), started with no signal blocked,
    /// SIGTERM and SIGHUP at their default actions, and SIGINT at its default
    /// or ignored:
    /// - first: SIGTERM recorded; a SIGINT raised in `end_by_first`'s
    ///   @p before (where bench gpu-lock says what went wrong), the actions
    ///   being back and SIGINT's default action ending the process at once,
    ///   waits blocked, and the process ends by SIGTERM;
    /// - none: nothing recorded; a SIGHUP raised once the actions are back
    ///   is not lost: the process ends by it once the stopped watch is
    ///   dropped;
    /// - clean: nothing raised, SIGINT ignored by the parent: while stopped
    ///   the three are blocked, the actions are back and SIGINT is still
    ///   ignored; once dropped, nothing is blocked.
    #[test]
    #[cfg(target_os = "linux")]
    fn watch_stop_ends_by_the_first_signal_whatever_follows() {
        use std::os::unix::process::{CommandExt, ExitStatusExt};

        let dir = tempfile_dir("watch_stop");
        let name = format!(
            "{}::watch_stop_in_a_process_of_its_own",
            module_path!().split_once("::").map_or("", |(_, path)| path)
        );
        let defaults = "actions [\"default\", \"default\", \"default\"]";
        let int_ignored = "actions [\"ignored\", \"default\", \"default\"]";
        let all = format!(
            "blocked [{}, {}, {}]",
            libc::SIGINT,
            libc::SIGTERM,
            libc::SIGHUP
        );
        let cases = [
            (
                "first",
                libc::SIG_DFL,
                Some(libc::SIGTERM),
                format!("first {}; {defaults}; {all}", libc::SIGTERM),
            ),
            (
                "none",
                libc::SIG_DFL,
                Some(libc::SIGHUP),
                format!("first None; {defaults}; {all}"),
            ),
            (
                "clean",
                libc::SIG_IGN,
                None,
                format!("first None; {int_ignored}; {all}; then {int_ignored}; blocked []"),
            ),
        ];
        let mut seen = Vec::new();
        for (case, int_action, ended_by, expected) in cases {
            let report = dir.join(format!("{case}.txt"));
            let mut command = Command::new(std::env::current_exe().expect("this test binary"));
            command
                .args(["--exact", &name, "--nocapture", "--test-threads=1"])
                .env(STOP_CASE_VARIABLE, case)
                .env(WATCH_REPORT_VARIABLE, &report);
            // SAFETY: the closure runs in the child between fork and exec and
            // calls only sigaction and pthread_sigmask, which are
            // async-signal-safe; zeroed sigaction and sigset_t values are
            // valid values of the types, given the default or the ignore
            // handler, and initialised by sigemptyset.
            unsafe {
                command.pre_exec(move || {
                    let mut none: libc::sigset_t = std::mem::zeroed();
                    libc::sigemptyset(&mut none);
                    libc::pthread_sigmask(libc::SIG_SETMASK, &none, std::ptr::null_mut());
                    for (signal, handler) in [
                        (libc::SIGINT, int_action),
                        (libc::SIGTERM, libc::SIG_DFL),
                        (libc::SIGHUP, libc::SIG_DFL),
                    ] {
                        let mut action: libc::sigaction = std::mem::zeroed();
                        action.sa_sigaction = handler;
                        if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                    }
                    Ok(())
                });
            }
            let out = command.output().expect("run this test binary");
            let line = fs::read_to_string(&report).unwrap_or_default();
            let ended_as_expected = match ended_by {
                Some(signal) => out.status.signal() == Some(signal),
                None => out.status.success(),
            };
            if !ended_as_expected || line != expected {
                seen.push(format!(
                    "{case}: ended {:?}, expected by {ended_by:?}; reported {line:?}, expected \
                     {expected:?}\n{}{}",
                    out.status,
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
        }
        let _ = fs::remove_dir_all(&dir);
        assert!(seen.is_empty(), "{}", seen.join("\n"));
    }
}
