//! Running one Nsight tool command with a deadline.
//!
//! A command runs with its input closed and both outputs read on threads of
//! their own, so neither pipe can fill and stall it, while the calling thread
//! polls for its end, its deadline and an interrupt. On the deadline or an
//! interrupt the tool and every process under it are stopped: the `ncu` on
//! PATH is a shell launcher whose real binary runs as a separate process under
//! it, so stopping the launcher alone would leave the tool running and holding
//! its outputs. The processes under the tool are found through `/proc`, each
//! frozen as it is found so that none can start another, then all are killed.
//! The tool stays in the caller's process group, so Ctrl-C reaches it as it
//! reaches any foreground command.
//!
//! While the extraction runs, SIGINT and SIGTERM only record which signal
//! arrived; the extraction then stops its tool, removes its private
//! directories and ends the process by the same signal (`Interrupt`).
//!
//! Every operating-system call this needs beyond the standard library is in
//! this file, behind `run_tool`, `Interrupts` and `Interrupt`.

use std::ffi::OsStr;
use std::fmt;
use std::io::{ErrorKind, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/* ----------------------------- Constants ----------------------------- */

/// How often a running tool is checked for its end, its deadline and an
/// interrupt.
const POLL: Duration = Duration::from_millis(10);

/// How long a tool's outputs may stay open once it has ended or been stopped.
const DRAIN: Duration = Duration::from_secs(2);

/// The signal recorded since `Interrupts::watch_signals`; 0 while none has
/// arrived.
static SIGNALLED: AtomicI32 = AtomicI32::new(0);

/* ----------------------------- Types ----------------------------- */

/// How one tool command ended.
#[derive(Debug)]
pub(crate) enum ToolRun {
    /// It exited, or was ended by a signal it did not get from here, with what
    /// it printed.
    Finished {
        status: ExitStatus,
        stdout: String,
        stderr: String,
    },
    /// No program of that name is on PATH.
    NotFound,
    /// It could not be started for another reason.
    NotStarted(std::io::Error),
    /// Its end could not be waited for; it was stopped with every process
    /// under it.
    WaitFailed(std::io::Error),
    /// It ran past its deadline and was stopped with every process under it.
    TimedOut,
    /// It ended, but a process it started kept its outputs open past the
    /// deadline.
    OutputHeldOpen,
    /// An interrupt arrived; the tool, if it was running, was stopped with
    /// every process under it.
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

/// What a signal sent from here does to a process.
#[derive(Debug, Clone, Copy)]
enum Signal {
    /// Freeze it: a frozen process cannot start another.
    Stop,
    /// End it.
    Kill,
}

/// Both outputs of a running tool, each read to its end on a thread of its
/// own.
struct Outputs {
    received: mpsc::Receiver<(usize, Vec<u8>)>,
}

/* ----------------------------- Operating system ----------------------------- */

// The three calls below are the only ones in this crate that need `unsafe`.
// Each takes plain integers, touches no memory of this process, and is wrapped
// once here.

/// Send @p signal to process @p pid. A process that has already ended is not
/// an error: the call then does nothing.
#[cfg(unix)]
fn signal_process(pid: u32, signal: Signal) {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return;
    };
    let number = match signal {
        Signal::Stop => libc::SIGSTOP,
        Signal::Kill => libc::SIGKILL,
    };
    // SAFETY: kill(2) only reads its two integer arguments. The pid is one
    // this process started or found under it in /proc; if it has ended, the
    // call fails with ESRCH and changes nothing.
    unsafe {
        libc::kill(pid, number);
    }
}

#[cfg(not(unix))]
fn signal_process(_pid: u32, _signal: Signal) {}

/// Record the signal's number in `SIGNALLED`; only an atomic store, which is
/// safe in a signal handler.
#[cfg(unix)]
extern "C" fn record_signal(signal: libc::c_int) {
    SIGNALLED.store(signal, Ordering::SeqCst);
}

/// Make @p signal call `record_signal`, or, with @p record false, take its
/// default action again.
#[cfg(unix)]
fn set_signal_action(signal: i32, record: bool) {
    let action = if record {
        record_signal as extern "C" fn(libc::c_int) as libc::sighandler_t
    } else {
        libc::SIG_DFL
    };
    // SAFETY: signal(2) installs either the default action or
    // `record_signal`, an `extern "C"` function that lives as long as the
    // process and does nothing but an atomic store.
    unsafe {
        libc::signal(signal, action);
    }
}

#[cfg(not(unix))]
fn set_signal_action(_signal: i32, _record: bool) {}

/// Send @p signal to this process.
#[cfg(unix)]
fn raise_signal(signal: i32) {
    // SAFETY: raise(3) only reads its integer argument.
    unsafe {
        libc::raise(signal);
    }
}

#[cfg(not(unix))]
fn raise_signal(_signal: i32) {}

/// The signals the command watches.
#[cfg(unix)]
const WATCHED_SIGNALS: [i32; 2] = [libc::SIGINT, libc::SIGTERM];

#[cfg(not(unix))]
const WATCHED_SIGNALS: [i32; 0] = [];

/* ----------------------------- Helpers ----------------------------- */

/// The processes under @p root, read from `/proc` (each process's parent is
/// the fourth field of its `stat`, after the command name in parentheses).
#[cfg(target_os = "linux")]
fn processes_under(root: u32) -> Vec<u32> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let links: Vec<(u32, u32)> = entries
        .flatten()
        .filter_map(|entry| {
            let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
            let stat = std::fs::read_to_string(entry.path().join("stat")).ok()?;
            let after_name = &stat[stat.rfind(')')? + 1..];
            let parent: u32 = after_name.split_whitespace().nth(1)?.parse().ok()?;
            Some((pid, parent))
        })
        .collect();
    let mut found = Vec::new();
    let mut parents = vec![root];
    while let Some(parent) = parents.pop() {
        for &(pid, _) in links.iter().filter(|&&(_, p)| p == parent) {
            if pid != root && !found.contains(&pid) {
                found.push(pid);
                parents.push(pid);
            }
        }
    }
    found
}

#[cfg(not(target_os = "linux"))]
fn processes_under(_root: u32) -> Vec<u32> {
    Vec::new()
}

/// Stop the tool and every process under it: each is frozen as it is found,
/// the search repeated until it finds no new one, then all are killed and the
/// tool is reaped.
fn stop_tree(child: &mut Child) {
    let root = child.id();
    signal_process(root, Signal::Stop);
    let mut frozen = vec![root];
    // Frozen processes start no others, so each round can only find processes
    // that were still running in the round before; the bound is a backstop.
    for _ in 0..100 {
        let new: Vec<u32> = processes_under(root)
            .into_iter()
            .filter(|pid| !frozen.contains(pid))
            .collect();
        if new.is_empty() {
            break;
        }
        for pid in new {
            signal_process(pid, Signal::Stop);
            frozen.push(pid);
        }
    }
    for &pid in &frozen {
        signal_process(pid, Signal::Kill);
    }
    let _ = child.kill();
    let _ = child.wait();
}

impl Outputs {
    /// Start reading @p child's stdout and stderr.
    fn read(child: &mut Child) -> Self {
        let (sender, received) = mpsc::channel();
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
            let sender = sender.clone();
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = pipe.read_to_end(&mut bytes);
                let _ = sender.send((index, bytes));
            });
        }
        Self { received }
    }

    /// Both outputs once both have closed, or `None` if one is still open at
    /// @p until.
    fn collect(self, until: Instant) -> Option<(String, String)> {
        let mut texts: [Option<String>; 2] = [None, None];
        while texts.iter().any(Option::is_none) {
            let left = until.saturating_duration_since(Instant::now());
            let (index, bytes) = self.received.recv_timeout(left).ok()?;
            texts[index] = Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        let [stdout, stderr] = texts;
        Some((stdout.unwrap_or_default(), stderr.unwrap_or_default()))
    }

    /// Give the outputs of a stopped tool a moment to close, so their threads
    /// end with it.
    fn discard(self) {
        let until = Instant::now() + DRAIN;
        for _ in 0..2 {
            let left = until.saturating_duration_since(Instant::now());
            if self.received.recv_timeout(left).is_err() {
                break;
            }
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
        set_signal_action(self.0, false);
        raise_signal(self.0);
        // Reached only where the signal's default action does not end the
        // process: the shell convention for an end by signal.
        std::process::exit(128 + self.0)
    }
}

impl fmt::Display for Interrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            2 => write!(f, "SIGINT"),
            15 => write!(f, "SIGTERM"),
            other => write!(f, "signal {other}"),
        }
    }
}

impl Interrupts<'static> {
    /// From now on, SIGINT and SIGTERM are recorded for the run to stop
    /// itself, instead of ending the process at once.
    pub(crate) fn watch_signals() -> Self {
        for signal in WATCHED_SIGNALS {
            set_signal_action(signal, true);
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
/// for at most @p timeout. On the deadline or an interrupt the program and
/// every process under it are stopped. Nothing starts when an interrupt has
/// already arrived.
pub(crate) fn run_tool(
    program: &str,
    args: &[&OsStr],
    timeout: Duration,
    interrupts: &Interrupts<'_>,
) -> ToolRun {
    if let Some(interrupt) = interrupts.pending() {
        return ToolRun::Interrupted(interrupt);
    }
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) if e.kind() == ErrorKind::NotFound => return ToolRun::NotFound,
        Err(e) => return ToolRun::NotStarted(e),
    };
    let outputs = Outputs::read(&mut child);
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(interrupt) = interrupts.pending() {
            stop_tree(&mut child);
            outputs.discard();
            return ToolRun::Interrupted(interrupt);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                // A tool that ended just before its deadline still gets a
                // moment for its outputs to close.
                let until = deadline.max(Instant::now() + DRAIN);
                return match outputs.collect(until) {
                    Some((stdout, stderr)) => ToolRun::Finished {
                        status,
                        stdout,
                        stderr,
                    },
                    None => ToolRun::OutputHeldOpen,
                };
            }
            Ok(None) => {}
            Err(e) => {
                stop_tree(&mut child);
                outputs.discard();
                return ToolRun::WaitFailed(e);
            }
        }
        if Instant::now() >= deadline {
            stop_tree(&mut child);
            outputs.discard();
            return ToolRun::TimedOut;
        }
        std::thread::sleep(POLL);
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::path::Path;

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

    /// Whether @p pid has ended: gone, or a zombie no one has reaped yet.
    fn ended(pid: u32) -> bool {
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            let state = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .and_then(|s| {
                    let after = &s[s.rfind(')')? + 1..];
                    after.split_whitespace().next().map(str::to_string)
                });
            match state.as_deref() {
                None | Some("Z") | Some("X") => return true,
                _ if Instant::now() >= until => return false,
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    /// End a process a test started and left running, by its pid.
    fn end(pid: u32) {
        signal_process(pid, Signal::Kill);
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
        let child = pid_in(&pid_file);
        assert!(matches!(run, ToolRun::TimedOut), "{run:?}");
        assert!(started.elapsed() < Duration::from_secs(10));
        let stopped = ended(child);
        if !stopped {
            end(child);
        }
        assert!(stopped, "the launcher's child {child} still runs");
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
        assert!(
            matches!(run, ToolRun::Interrupted(Interrupt(15))),
            "{run:?}"
        );
        let stopped = ended(child);
        if !stopped {
            end(child);
        }
        assert!(stopped, "the launcher's child {child} still runs");
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
    /// outputs is reported as such once the deadline passes.
    #[test]
    fn output_held_open_by_a_leftover_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("leftover.pid");
        let script = format!("/bin/sleep 30 & echo $! > '{}'", pid_file.display());
        let flag = no_interrupt();
        let run = run_tool(
            "/bin/sh",
            &sh(&script),
            Duration::from_millis(300),
            &Interrupts::from_flag(&flag),
        );
        end(pid_in(&pid_file));
        assert!(matches!(run, ToolRun::OutputHeldOpen), "{run:?}");
    }

    /// @test An interrupt names SIGINT and SIGTERM, and any other signal by
    /// number.
    #[test]
    fn interrupt_names() {
        assert_eq!(Interrupt(2).to_string(), "SIGINT");
        assert_eq!(Interrupt(15).to_string(), "SIGTERM");
        assert_eq!(Interrupt(1).to_string(), "signal 1");
        assert_eq!(Interrupt(15).number(), 15);
    }
}
