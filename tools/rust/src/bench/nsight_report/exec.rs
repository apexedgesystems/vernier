//! Running one Nsight tool command with a deadline, in a process group of its
//! own.
//!
//! The tool starts as the leader of a new process group, its input closed and
//! both outputs captured. Every process it starts belongs to that group unless
//! it leaves it for a session or group of its own: the `ncu` on PATH is a
//! shell launcher whose real binary runs as a separate process under it, and a
//! process the tool starts can outlive it and keep its outputs open. The
//! calling thread reads both outputs as they arrive and checks, at least every
//! 10 ms, for the tool's end, its deadline and an interrupt. Once the tool has
//! ended, its outputs get 2 s more to close, whatever the deadline. However the
//! run ends, the whole group is then killed with one signal, the
//! tool is reaped and both outputs are closed: no process of the group outlives
//! the run, and nothing is left reading its outputs.
//!
//! The group's ID is the tool's process ID, which no other process can take
//! while the tool exists, even once it has ended and until it is reaped. The
//! tool's end is therefore observed without reaping it, and it is reaped only
//! after the group's signal.
//!
//! The tool's group is not the terminal's foreground group, so the terminal's
//! signals reach `bench` and not the tool. While the extraction runs, SIGHUP,
//! SIGINT and SIGTERM only record which signal arrived, unless the process
//! started with that signal ignored; the extraction then stops the tool's
//! group, removes its private directories and ends the process by the same
//! signal (`Interrupt`). SIGQUIT keeps its default action and SIGKILL cannot be
//! caught: either ends `bench` at once without stopping the tool's group.
//!
//! Signal actions belong to the process, not to one run: the command is one
//! operation per process, not an executor that runs in several threads of one
//! process can share.
//!
//! Every operating-system call this needs beyond the standard library is in
//! this file, behind `run_tool`, `Interrupts` and `Interrupt`. Elsewhere than
//! on Unix, the tool alone is stopped and its outputs are read on threads.

use std::ffi::OsStr;
use std::fmt;
use std::io::{self, ErrorKind, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

/* ----------------------------- Constants ----------------------------- */

/// The longest wait between two checks of a running tool for its end, its
/// deadline and an interrupt; output and signals end a wait sooner.
const POLL: Duration = Duration::from_millis(10);

/// How long an ended tool's outputs may stay open before the processes that
/// hold them are stopped.
pub(crate) const DRAIN: Duration = Duration::from_secs(2);

/// How long a tool whose group has been killed is waited for before it is
/// left unreaped.
const REAP: Duration = Duration::from_secs(2);

/// The signal recorded since `Interrupts::watch_signals`; 0 while none has
/// arrived.
static SIGNALLED: AtomicI32 = AtomicI32::new(0);

/* ----------------------------- Types ----------------------------- */

/// How one tool command ended.
#[derive(Debug)]
pub(crate) enum ToolRun {
    /// It exited, or was ended by a signal it did not get from here, and its
    /// outputs closed, with what it printed.
    Finished {
        status: ExitStatus,
        stdout: String,
        stderr: String,
    },
    /// No program of that name is on PATH.
    NotFound,
    /// It could not be started for another reason.
    NotStarted(io::Error),
    /// Its end could not be observed. Its group was not signalled: once its
    /// end is unknown, its ID may belong to another process.
    WaitFailed(io::Error),
    /// It ran past its deadline; its group was killed.
    TimedOut,
    /// It ended, but a process it started still held its outputs `DRAIN`
    /// later; its group was killed.
    OutputHeldOpen,
    /// An interrupt arrived; the tool's group, if it had started, was killed.
    Interrupted(Interrupt),
}

/// The signal that interrupted a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interrupt(i32);

/// Where a run looks for an interrupt: the signals the command watches, or a
/// flag the caller owns.
pub(crate) struct Interrupts<'a> {
    flag: &'a AtomicI32,
}

/// Why the watch over a running tool ended.
#[derive(Debug)]
enum Stop {
    /// It ended and both its outputs closed.
    Ended,
    /// Its deadline passed while it ran.
    Deadline,
    /// It ended, and its outputs were still open `DRAIN` later.
    HeldOpen,
    /// An interrupt arrived.
    Interrupted(Interrupt),
    /// Its end could not be observed.
    WaitFailed(io::Error),
}

/// What a watched signal does.
#[derive(Debug, Clone, Copy)]
enum Action {
    /// Record its number for the run to stop itself.
    Record,
    /// Its default action.
    Default,
}

/// Both outputs of a running tool, read on the calling thread as they arrive.
#[cfg(unix)]
struct Outputs {
    /// stdout and stderr, each until it closes.
    pipes: [Option<std::fs::File>; 2],
    /// What each has given so far.
    bytes: [Vec<u8>; 2],
}

/// Both outputs of a running tool, each read to its end on a thread of its
/// own.
#[cfg(not(unix))]
struct Outputs {
    received: std::sync::mpsc::Receiver<(usize, Vec<u8>)>,
    /// What each gave, once it has closed.
    bytes: [Option<Vec<u8>>; 2],
}

/* ----------------------------- Operating system ----------------------------- */

// The calls below are the only ones in this crate that need `unsafe`; each is
// wrapped once here, with the reason it is sound.

/// Start @p command's process as the leader of a new process group, whose ID
/// is then its process ID.
#[cfg(unix)]
fn own_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
fn own_group(_command: &mut Command) {}

/// Whether @p child has ended. It is not reaped: it stays waitable, and its
/// process ID taken, until `reap`.
#[cfg(unix)]
fn has_ended(child: &mut Child) -> io::Result<bool> {
    let pid = libc::id_t::from(child.id());
    // SAFETY: an all-zero `siginfo_t` is a valid value of that plain C struct,
    // and waitid(2) writes only into `info`, a local that lives through the
    // call. WNOHANG returns at once; WNOWAIT leaves the child waitable, so the
    // call reaps nothing and `child` still does.
    let (rc, info) = unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        let rc = libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        );
        (rc, info)
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    // No ended child to report leaves the signal number 0.
    Ok(info.si_signo == libc::SIGCHLD)
}

#[cfg(not(unix))]
fn has_ended(child: &mut Child) -> io::Result<bool> {
    child.try_wait().map(|status| status.is_some())
}

/// Kill every process in the group @p child leads. @p child must not have
/// been reaped: only then is its process ID, the group's ID, still its own.
#[cfg(unix)]
fn kill_group(child: &mut Child) {
    let Ok(group) = libc::pid_t::try_from(child.id()) else {
        return;
    };
    // A group ID of 1 or less would mean every process (-1) or this process's
    // own group (0); a child's never is.
    if group <= 1 {
        return;
    }
    // SAFETY: kill(2) only reads its two integer arguments; it touches no
    // memory of this process. The target is the group `own_group` made for
    // @p child: its ID is @p child's process ID, which no other process or
    // group can take while @p child is unreaped (the caller's guarantee).
    // Members that have already ended are not affected.
    unsafe {
        libc::kill(-group, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_group(child: &mut Child) {
    let _ = child.kill();
}

/// Wait at most @p up_to for one of @p fds to be readable or closed, or for a
/// signal to arrive; each entry's `revents` then says which. With no entries,
/// only waits.
#[cfg(unix)]
fn wait_readable(fds: &mut [libc::pollfd], up_to: Duration) {
    let ms = libc::c_int::try_from(up_to.as_millis()).unwrap_or(libc::c_int::MAX);
    // At most two entries: stdout and stderr.
    let count = fds.len() as libc::nfds_t;
    // SAFETY: poll(2) reads `count` entries from the pointer and writes only
    // their `revents`; the slice holds exactly `count` initialized entries and
    // is borrowed mutably for the whole call. Each entry's descriptor is an
    // output pipe that `Outputs` owns and keeps open across the call.
    let rc = unsafe { libc::poll(fds.as_mut_ptr(), count, ms) };
    if rc < 0 {
        // A signal (checked by the caller next) or no resources: nothing is
        // reported readable, and the caller checks again.
        let interrupted = io::Error::last_os_error().kind() == ErrorKind::Interrupted;
        for fd in fds.iter_mut() {
            fd.revents = 0;
        }
        if !interrupted {
            std::thread::sleep(up_to);
        }
    }
}

/// Record the signal's number in `SIGNALLED`: only a store to a lock-free
/// atomic, which is safe in a signal handler.
#[cfg(unix)]
extern "C" fn record_signal(signal: libc::c_int) {
    SIGNALLED.store(signal, Ordering::SeqCst);
}

/// Whether @p signal is ignored now.
#[cfg(unix)]
fn is_ignored(signal: libc::c_int) -> bool {
    // SAFETY: an all-zero `sigaction` is a valid value of that plain C struct.
    // With a null new action, sigaction(2) changes nothing: it only writes the
    // current action into `current`, a local that lives through the call.
    let (rc, current) = unsafe {
        let mut current: libc::sigaction = std::mem::zeroed();
        let rc = libc::sigaction(signal, std::ptr::null(), &mut current);
        (rc, current)
    };
    rc == 0 && current.sa_sigaction == libc::SIG_IGN
}

#[cfg(not(unix))]
fn is_ignored(_signal: i32) -> bool {
    false
}

/// Give @p signal @p action, for the whole process.
#[cfg(unix)]
fn set_signal_action(signal: libc::c_int, action: Action) {
    let handler = match action {
        Action::Record => record_signal as extern "C" fn(libc::c_int) as libc::sighandler_t,
        Action::Default => libc::SIG_DFL,
    };
    // SAFETY: signal(2) only reads its two integer arguments. What it
    // installs is the default action or `record_signal`, an `extern "C"`
    // function that lives as long as the process and only stores to an
    // atomic. The action is the process's, in every thread, until it is set
    // again: the module's documentation states that boundary.
    unsafe {
        libc::signal(signal, handler);
    }
}

#[cfg(not(unix))]
fn set_signal_action(_signal: i32, _action: Action) {}

/// Send @p signal to this process.
#[cfg(unix)]
fn raise_signal(signal: libc::c_int) {
    // SAFETY: raise(3) only reads its integer argument and signals the calling
    // thread; its action is the process's own.
    unsafe {
        libc::raise(signal);
    }
}

#[cfg(not(unix))]
fn raise_signal(_signal: i32) {}

/// Every child stays waitable until it is reaped here: a SIGCHLD inherited as
/// ignored would have the kernel reap each child itself, and with it give up
/// the process ID a tool's group signal is sent to.
#[cfg(unix)]
fn keep_children_waitable() {
    set_signal_action(libc::SIGCHLD, Action::Default);
}

#[cfg(not(unix))]
fn keep_children_waitable() {}

/// The signals the command watches: a terminal's hangup and interrupt, and
/// the request to terminate.
#[cfg(unix)]
const WATCHED_SIGNALS: [i32; 3] = [libc::SIGHUP, libc::SIGINT, libc::SIGTERM];

#[cfg(not(unix))]
const WATCHED_SIGNALS: [i32; 0] = [];

/* ----------------------------- Helpers ----------------------------- */

#[cfg(unix)]
impl Outputs {
    /// Take @p child's stdout and stderr.
    fn take(child: &mut Child) -> Self {
        use std::os::fd::OwnedFd;
        let stdout = child
            .stdout
            .take()
            .map(|pipe| std::fs::File::from(OwnedFd::from(pipe)));
        let stderr = child
            .stderr
            .take()
            .map(|pipe| std::fs::File::from(OwnedFd::from(pipe)));
        Self {
            pipes: [stdout, stderr],
            bytes: [Vec::new(), Vec::new()],
        }
    }

    /// Whether both outputs have closed.
    fn closed(&self) -> bool {
        self.pipes.iter().all(Option::is_none)
    }

    /// Wait at most @p up_to for output, and read what has arrived; an output
    /// that has closed is let go.
    fn read_for(&mut self, up_to: Duration) {
        use std::os::fd::AsRawFd;
        let open: Vec<usize> = (0..2).filter(|&i| self.pipes[i].is_some()).collect();
        let mut fds: Vec<libc::pollfd> = open
            .iter()
            .filter_map(|&i| self.pipes[i].as_ref())
            .map(|pipe| libc::pollfd {
                fd: pipe.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        wait_readable(&mut fds, up_to);
        for (fd, &i) in fds.iter().zip(&open) {
            if fd.revents == 0 {
                continue;
            }
            let Some(pipe) = self.pipes[i].as_mut() else {
                continue;
            };
            let mut chunk = [0u8; 64 * 1024];
            match pipe.read(&mut chunk) {
                Ok(0) => self.pipes[i] = None,
                Ok(n) => self.bytes[i].extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => self.pipes[i] = None,
            }
        }
    }

    /// What the outputs gave; both are closed.
    fn into_texts(self) -> (String, String) {
        let [stdout, stderr] = self.bytes;
        (
            String::from_utf8_lossy(&stdout).into_owned(),
            String::from_utf8_lossy(&stderr).into_owned(),
        )
    }
}

#[cfg(not(unix))]
impl Outputs {
    /// Start reading @p child's stdout and stderr.
    fn take(child: &mut Child) -> Self {
        let (sender, received) = std::sync::mpsc::channel();
        let mut bytes = [Some(Vec::new()), Some(Vec::new())];
        let stdout = child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>);
        let stderr = child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>);
        for (index, pipe) in [stdout, stderr].into_iter().enumerate() {
            let Some(mut pipe) = pipe else { continue };
            bytes[index] = None;
            let sender = sender.clone();
            std::thread::spawn(move || {
                let mut read = Vec::new();
                let _ = pipe.read_to_end(&mut read);
                let _ = sender.send((index, read));
            });
        }
        Self { received, bytes }
    }

    /// Whether both outputs have closed.
    fn closed(&self) -> bool {
        self.bytes.iter().all(Option::is_some)
    }

    /// Wait at most @p up_to for an output to close.
    fn read_for(&mut self, up_to: Duration) {
        if self.closed() {
            std::thread::sleep(up_to);
        } else if let Ok((index, read)) = self.received.recv_timeout(up_to) {
            self.bytes[index] = Some(read);
        }
    }

    /// What the outputs gave.
    fn into_texts(self) -> (String, String) {
        let [stdout, stderr] = self.bytes;
        let text = |bytes: Option<Vec<u8>>| {
            String::from_utf8_lossy(&bytes.unwrap_or_default()).into_owned()
        };
        (text(stdout), text(stderr))
    }
}

/// Read @p child's @p outputs until it has ended and they have closed, its
/// deadline passes while it runs, they stay open `DRAIN` after its end, or an
/// interrupt arrives.
fn watch(
    child: &mut Child,
    outputs: &mut Outputs,
    deadline: Instant,
    interrupts: &Interrupts<'_>,
) -> Stop {
    let mut ended_at = None;
    loop {
        if let Some(interrupt) = interrupts.pending() {
            return Stop::Interrupted(interrupt);
        }
        if ended_at.is_none() {
            match has_ended(child) {
                Ok(true) => ended_at = Some(Instant::now()),
                Ok(false) => {}
                Err(e) => return Stop::WaitFailed(e),
            }
        }
        let now = Instant::now();
        match ended_at {
            Some(_) if outputs.closed() => return Stop::Ended,
            // An ended tool's last output may still be in flight; outputs a
            // process it started keeps open past that are let go.
            Some(at) if now >= at + DRAIN => return Stop::HeldOpen,
            None if now >= deadline => return Stop::Deadline,
            _ => outputs.read_for(POLL),
        }
    }
}

/// Reap @p child, waiting at most `REAP` for it to end; `None` if it has not.
fn reap(child: &mut Child) -> io::Result<Option<ExitStatus>> {
    let until = Instant::now() + REAP;
    loop {
        match child.try_wait() {
            Ok(None) if Instant::now() < until => std::thread::sleep(POLL),
            other => return other,
        }
    }
}

/* ----------------------------- API ----------------------------- */

impl Interrupt {
    /// The signal's number.
    pub fn number(self) -> i32 {
        self.0
    }

    /// End this process the way the signal would have ended it: its default
    /// action is restored and the signal raised again, so a shell sees the
    /// usual status.
    pub fn end_process(self) -> ! {
        set_signal_action(self.0, Action::Default);
        raise_signal(self.0);
        // Reached only where the signal's default action does not end the
        // process: the shell convention for an end by signal.
        std::process::exit(128 + self.0)
    }
}

impl fmt::Display for Interrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            1 => write!(f, "SIGHUP"),
            2 => write!(f, "SIGINT"),
            15 => write!(f, "SIGTERM"),
            other => write!(f, "signal {other}"),
        }
    }
}

impl Interrupts<'static> {
    /// From now on, for the rest of the process: SIGHUP, SIGINT and SIGTERM
    /// are recorded for the run to stop itself, instead of ending the process
    /// at once, except one that is ignored now, which stays ignored (as
    /// `nohup` and a shell without job control start commands); and SIGCHLD
    /// has its default action, so every tool stays waitable until it is
    /// reaped here.
    pub(crate) fn watch_signals() -> Self {
        keep_children_waitable();
        for signal in WATCHED_SIGNALS {
            if !is_ignored(signal) {
                set_signal_action(signal, Action::Record);
            }
        }
        Self { flag: &SIGNALLED }
    }
}

impl<'a> Interrupts<'a> {
    /// A flag the caller owns: any value but 0 is the number of the signal
    /// that arrived.
    #[cfg(test)]
    pub(crate) fn from_flag(flag: &'a AtomicI32) -> Self {
        Self { flag }
    }

    /// The signal that has arrived, if one has.
    pub(crate) fn pending(&self) -> Option<Interrupt> {
        match self.flag.load(Ordering::SeqCst) {
            0 => None,
            signal => Some(Interrupt(signal)),
        }
    }
}

/// Run @p program with @p args, its input closed and its outputs captured,
/// for at most @p timeout, in a process group of its own. Nothing starts when
/// an interrupt has already arrived. However the run ends, every process of
/// the group is killed and the outputs are closed before this returns.
pub(crate) fn run_tool(
    program: &str,
    args: &[&OsStr],
    timeout: Duration,
    interrupts: &Interrupts<'_>,
) -> ToolRun {
    if let Some(interrupt) = interrupts.pending() {
        return ToolRun::Interrupted(interrupt);
    }
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    own_group(&mut command);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) if e.kind() == ErrorKind::NotFound => return ToolRun::NotFound,
        Err(e) => return ToolRun::NotStarted(e),
    };
    let mut outputs = Outputs::take(&mut child);
    let stop = watch(
        &mut child,
        &mut outputs,
        Instant::now() + timeout,
        interrupts,
    );
    // `watch` has not reaped the tool, unless its end could not be observed.
    if !matches!(stop, Stop::WaitFailed(_)) {
        kill_group(&mut child);
    }
    let reaped = reap(&mut child);
    let (stdout, stderr) = outputs.into_texts();
    match stop {
        Stop::Ended => match reaped {
            Ok(Some(status)) => ToolRun::Finished {
                status,
                stdout,
                stderr,
            },
            Ok(None) => ToolRun::WaitFailed(ErrorKind::TimedOut.into()),
            Err(e) => ToolRun::WaitFailed(e),
        },
        Stop::Deadline => ToolRun::TimedOut,
        Stop::HeldOpen => ToolRun::OutputHeldOpen,
        Stop::Interrupted(interrupt) => ToolRun::Interrupted(interrupt),
        Stop::WaitFailed(e) => ToolRun::WaitFailed(e),
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::path::{Path, PathBuf};

    fn sh(script: &str) -> Vec<&OsStr> {
        vec![OsStr::new("-c"), OsStr::new(script)]
    }

    fn no_interrupt() -> AtomicI32 {
        AtomicI32::new(0)
    }

    /// The pid a test script wrote to @p file, waiting for it to appear.
    fn pid_in(file: &Path) -> u32 {
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(text) = std::fs::read_to_string(file) {
                if let Ok(pid) = text.trim().parse() {
                    return pid;
                }
            }
            assert!(
                Instant::now() < until,
                "{} was never written",
                file.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The state letter of @p pid in /proc, or `None` once it is gone.
    fn state(pid: u32) -> Option<String> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after = &stat[stat.rfind(')')? + 1..];
        after.split_whitespace().next().map(str::to_string)
    }

    /// Whether @p pid has ended: gone, or a zombie no one has reaped yet.
    fn ended(pid: u32) -> bool {
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            match state(pid).as_deref() {
                None | Some("Z") | Some("X") => return true,
                _ if Instant::now() >= until => return false,
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    /// Whether @p file appears within @p within.
    fn appears(file: &Path, within: Duration) -> bool {
        let until = Instant::now() + within;
        while !file.exists() {
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }

    /// The absolute path of @p program on PATH.
    fn which(program: &str) -> PathBuf {
        std::env::var_os("PATH")
            .and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|dir| dir.join(program))
                    .find(|path| path.is_file())
            })
            .unwrap_or_else(|| panic!("{program} is not on PATH"))
    }

    /// Processes a test started, killed when the test ends, after its
    /// assertions, if they still run then. It ends nothing a test checks.
    struct Guard(Vec<u32>);

    impl Drop for Guard {
        fn drop(&mut self) {
            for &pid in &self.0 {
                if !matches!(state(pid).as_deref(), None | Some("Z") | Some("X")) {
                    let _ = std::process::Command::new("/bin/sh")
                        .arg("-c")
                        .arg(format!("kill -s KILL {pid}"))
                        .status();
                }
            }
        }
    }

    /// @test A program that is not on PATH is reported as such.
    #[test]
    fn missing_program_is_not_found() {
        let flag = no_interrupt();
        let run = run_tool(
            "vernier-no-such-nsight-tool",
            &[],
            Duration::from_secs(5),
            &Interrupts::from_flag(&flag),
        );
        assert!(matches!(run, ToolRun::NotFound), "{run:?}");
    }

    /// @test A program that exits keeps its status and both outputs apart.
    #[test]
    fn exit_status_and_outputs() {
        let flag = no_interrupt();
        let run = run_tool(
            "/bin/sh",
            &sh("echo out; echo err >&2; exit 3"),
            Duration::from_secs(10),
            &Interrupts::from_flag(&flag),
        );
        let ToolRun::Finished {
            status,
            stdout,
            stderr,
        } = run
        else {
            panic!("{run:?}");
        };
        assert_eq!(status.code(), Some(3));
        assert_eq!(stdout, "out\n");
        assert_eq!(stderr, "err\n");
    }

    /// @test Output larger than a pipe's buffer is read whole while the
    /// program runs.
    #[test]
    fn large_output_is_read_whole() {
        let flag = no_interrupt();
        let script = "i=0; while [ $i -lt 20000 ]; do \
                      echo 0123456789012345678901234567890123456789; i=$((i+1)); done";
        let run = run_tool(
            "/bin/sh",
            &sh(script),
            Duration::from_secs(60),
            &Interrupts::from_flag(&flag),
        );
        let ToolRun::Finished { status, stdout, .. } = run else {
            panic!("{run:?}");
        };
        assert!(status.success());
        assert_eq!(stdout.len(), 20000 * 41);
    }

    /// @test A program ended by a signal of its own is reported with that
    /// signal, not as a timeout.
    #[test]
    fn signal_end_is_reported() {
        let flag = no_interrupt();
        let run = run_tool(
            "/bin/sh",
            &sh("kill -s KILL $$"),
            Duration::from_secs(10),
            &Interrupts::from_flag(&flag),
        );
        let ToolRun::Finished { status, .. } = run else {
            panic!("{run:?}");
        };
        assert_eq!(status.signal(), Some(9));
    }

    /// @test On the deadline a launcher and the process it started under
    /// itself are both stopped, well before the process would have ended.
    #[test]
    fn deadline_stops_the_launcher_and_its_child() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("child.pid");
        let script = format!("/bin/sleep 30 & echo $! > '{}'; wait", pid_file.display());
        let flag = no_interrupt();
        let started = Instant::now();
        let run = run_tool(
            "/bin/sh",
            &sh(&script),
            Duration::from_millis(500),
            &Interrupts::from_flag(&flag),
        );
        let returned = started.elapsed();
        let child = pid_in(&pid_file);
        let _guard = Guard(vec![child]);
        assert!(matches!(run, ToolRun::TimedOut), "{run:?}");
        assert!(returned < Duration::from_secs(10), "{returned:?}");
        assert!(ended(child), "the launcher's child {child} still runs");
    }

    /// @test An interrupt stops a running program and the process under it,
    /// and is reported as the interrupt.
    #[test]
    fn interrupt_stops_the_launcher_and_its_child() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("child.pid");
        let script = format!("/bin/sleep 30 & echo $! > '{}'; wait", pid_file.display());
        let flag = no_interrupt();
        let run = std::thread::scope(|scope| {
            scope.spawn(|| {
                pid_in(&pid_file);
                flag.store(15, Ordering::SeqCst);
            });
            run_tool(
                "/bin/sh",
                &sh(&script),
                Duration::from_secs(30),
                &Interrupts::from_flag(&flag),
            )
        });
        let child = pid_in(&pid_file);
        let _guard = Guard(vec![child]);
        assert!(
            matches!(run, ToolRun::Interrupted(Interrupt(15))),
            "{run:?}"
        );
        assert!(ended(child), "the launcher's child {child} still runs");
    }

    /// @test Nothing starts once an interrupt has arrived.
    #[test]
    fn nothing_starts_after_an_interrupt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let marker = dir.path().join("ran");
        let flag = AtomicI32::new(2);
        let run = run_tool(
            "/bin/sh",
            &sh(&format!("touch '{}'", marker.display())),
            Duration::from_secs(10),
            &Interrupts::from_flag(&flag),
        );
        assert!(matches!(run, ToolRun::Interrupted(Interrupt(2))), "{run:?}");
        assert!(!marker.exists());
    }

    /// @test A program that ends while a process it started still holds its
    /// outputs is reported as such 2 s later, however far off its deadline,
    /// and that process is stopped with it.
    #[test]
    fn output_held_open_by_a_leftover_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("leftover.pid");
        let script = format!("/bin/sleep 30 & echo $! > '{}'", pid_file.display());
        let flag = no_interrupt();
        let started = Instant::now();
        let run = run_tool(
            "/bin/sh",
            &sh(&script),
            Duration::from_secs(30),
            &Interrupts::from_flag(&flag),
        );
        let returned = started.elapsed();
        let leftover = pid_in(&pid_file);
        let _guard = Guard(vec![leftover]);
        assert!(matches!(run, ToolRun::OutputHeldOpen), "{run:?}");
        assert!(returned < Duration::from_secs(5), "{returned:?}");
        assert!(
            ended(leftover),
            "the leftover process {leftover} still runs"
        );
    }

    /// @test A program that ends at once, leaving a process it started
    /// running with its outputs elsewhere, has finished; that process is
    /// stopped before the run returns.
    #[test]
    fn ended_launcher_leaves_no_process_behind() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("leftover.pid");
        let script = format!(
            "/bin/sleep 30 > /dev/null 2>&1 & echo $! > '{}'; echo done",
            pid_file.display()
        );
        let flag = no_interrupt();
        let run = run_tool(
            "/bin/sh",
            &sh(&script),
            Duration::from_secs(30),
            &Interrupts::from_flag(&flag),
        );
        let leftover = pid_in(&pid_file);
        let _guard = Guard(vec![leftover]);
        let ToolRun::Finished { status, stdout, .. } = run else {
            panic!("{run:?}");
        };
        assert!(status.success());
        assert_eq!(stdout, "done\n");
        assert!(
            ended(leftover),
            "the leftover process {leftover} still runs"
        );
    }

    /// @test An interrupt that arrives after the program ended, while a
    /// process it started still holds its outputs, ends the run at once, and
    /// that process is stopped.
    #[test]
    fn interrupt_while_outputs_are_held_ends_the_run_at_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let launcher_file = dir.path().join("launcher.pid");
        let pid_file = dir.path().join("leftover.pid");
        let script = format!(
            "echo $$ > '{}'; /bin/sleep 30 & echo $! > '{}'",
            launcher_file.display(),
            pid_file.display()
        );
        let flag = no_interrupt();
        let (run, signalled, returned) = std::thread::scope(|scope| {
            let signaller = scope.spawn(|| {
                let launcher = pid_in(&launcher_file);
                pid_in(&pid_file);
                assert!(ended(launcher), "the launcher {launcher} never ended");
                // The run is now waiting for the outputs the leftover holds.
                std::thread::sleep(Duration::from_millis(200));
                let at = Instant::now();
                flag.store(15, Ordering::SeqCst);
                at
            });
            let run = run_tool(
                "/bin/sh",
                &sh(&script),
                Duration::from_secs(30),
                &Interrupts::from_flag(&flag),
            );
            let returned = Instant::now();
            (run, signaller.join().expect("signaller"), returned)
        });
        let leftover = pid_in(&pid_file);
        let _guard = Guard(vec![leftover]);
        assert!(
            matches!(run, ToolRun::Interrupted(Interrupt(15))),
            "{run:?}"
        );
        let took = returned.saturating_duration_since(signalled);
        assert!(
            took < Duration::from_secs(1),
            "{took:?} after the interrupt"
        );
        assert!(
            ended(leftover),
            "the leftover process {leftover} still runs"
        );
    }

    /// @test Outputs held by a process that left the program's group are
    /// let go when the run ends: the run returns at its bound, nothing is
    /// left reading them, and that process's next write fails.
    #[test]
    fn outputs_held_outside_the_group_are_let_go() {
        let dir = tempfile::tempdir().expect("tempdir");
        let holder_file = dir.path().join("holder.pid");
        let released = dir.path().join("released");
        let script = format!(
            "'{}' /bin/sh -c 'trap \"\" PIPE; echo $$ > \"$0\"; \
             while :; do echo tick || {{ : > \"$1\"; exit 0; }}; sleep 0.05; done' \
             '{}' '{}' &",
            which("setsid").display(),
            holder_file.display(),
            released.display()
        );
        let flag = no_interrupt();
        let started = Instant::now();
        let run = run_tool(
            "/bin/sh",
            &sh(&script),
            Duration::from_secs(30),
            &Interrupts::from_flag(&flag),
        );
        let returned = started.elapsed();
        let holder = pid_in(&holder_file);
        let _guard = Guard(vec![holder]);
        let let_go = appears(&released, Duration::from_secs(5));
        assert!(matches!(run, ToolRun::OutputHeldOpen), "{run:?}");
        assert!(returned < Duration::from_secs(5), "{returned:?}");
        assert!(
            let_go,
            "the holder {holder} could still write: its outputs were still being read"
        );
    }

    /// @test An interrupt names SIGHUP, SIGINT and SIGTERM, and any other
    /// signal by number.
    #[test]
    fn interrupt_names() {
        assert_eq!(Interrupt(1).to_string(), "SIGHUP");
        assert_eq!(Interrupt(2).to_string(), "SIGINT");
        assert_eq!(Interrupt(15).to_string(), "SIGTERM");
        assert_eq!(Interrupt(3).to_string(), "signal 3");
        assert_eq!(Interrupt(15).number(), 15);
    }
}
