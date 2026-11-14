//! GPU clock management: lock and reset GPU clocks for reproducible
//! benchmarks, through `nvidia-smi -lgc` and `nvidia-smi -rgc`. Three modes:
//!
//!   1. `bench gpu-lock lock`              -- lock the clocks and exit
//!   2. `bench gpu-lock reset`             -- reset them and exit
//!   3. `bench gpu-lock lock -- <command>` -- lock, run the command, reset
//!
//! In the wrapper mode the command runs in bench's own process group, with
//! bench's terminal, as it would without the wrapper: a terminal's Ctrl-C,
//! Ctrl-Z and hangup reach the command and bench together. While the command
//! runs, bench records SIGINT, SIGTERM and SIGHUP (`owned_run::Watch`; a
//! signal ignored when bench started stays ignored, for bench and for the
//! command) and passes a SIGTERM on to the command, since no terminal sends
//! one; SIGTERMs that arrive between two looks, a tenth of a second apart,
//! are passed on once. It waits for the command to end, however it ends and
//! however long it takes, so the clocks stay locked exactly as long as the
//! command runs; then it resets them once and ends by the first signal it
//! recorded, which no later signal replaces. A command that does not end
//! keeps bench waiting: the user ends it (kill -KILL), and the reset follows.
//! SIGKILL and SIGQUIT end bench itself at once, without the reset:
//! `bench gpu-lock reset` resets the clocks by hand.
//!
//! Every nvidia-smi call is an owned run (`owned_run`): it runs in a process
//! group of its own, which no terminal signal reaches, so a lock or a reset
//! is never stopped halfway, and a call still running at its bound is ended.
//! Messages go to stderr without failing: a terminal that hung up, or a pipe
//! nobody reads, cannot stop a reset.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::AtomicI32;
use std::time::Duration;

use super::{owned_run, Error};

/* ----------------------------- Constants ----------------------------- */

/// How long one nvidia-smi call may run before it, and every process of its
/// group, is ended.
const TOOL_BOUND: Duration = Duration::from_secs(30);

/// How often the wrapper looks at its command and at the signals recorded:
/// it runs beside the measured command, so it wakes rarely.
const WAIT_POLL: Duration = Duration::from_millis(100);

/* ----------------------------- Types ----------------------------- */

/// nvidia-smi as gpu-lock runs it.
#[derive(Debug, Clone)]
pub struct Tool {
    /// The program; a bare name is looked up on PATH.
    pub program: PathBuf,

    /// How long one call may run before it and its process group are ended.
    pub bound: Duration,
}

impl Default for Tool {
    /// `nvidia-smi` on PATH, with a bound of 30 s.
    fn default() -> Self {
        Self {
            program: PathBuf::from("nvidia-smi"),
            bound: TOOL_BOUND,
        }
    }
}

/// Configuration for a gpu-lock operation.
#[derive(Debug, Clone)]
pub struct LockConfig {
    /// GPU device index (nvidia-smi -i).
    pub device: u32,

    /// Target frequency in MHz. If None, locks to max reported clock.
    pub freq: Option<u32>,

    /// The nvidia-smi that locks and resets the clocks.
    pub tool: Tool,
}

/// Configuration for the wrapper mode (lock --> run --> reset).
#[derive(Debug, Clone)]
pub struct WrapConfig {
    pub lock: LockConfig,

    /// Command and arguments to execute while clocks are locked.
    pub command: Vec<String>,
}

/// How one nvidia-smi call ended.
enum Call {
    /// It exited 0, with this output.
    Done(String),
    /// It exited otherwise: why, in its own words where it gave some.
    Failed(String),
    /// It was still running at its bound and was ended.
    TimedOut,
}

/// Resets the clocks if it is dropped while armed: on a panic between the
/// lock and the wrapper's own reset.
struct ResetOnUnwind<'a> {
    lock: &'a LockConfig,
    armed: bool,
}

/* ----------------------------- Helpers ----------------------------- */

/// Write @p line to stderr. A write that fails, to a terminal that hung up
/// or a pipe nobody reads, is let go: no message may stop a reset or the end
/// by a signal.
fn say(line: &str) {
    let _ = writeln!(std::io::stderr(), "{line}");
}

/// @p bound as messages state it: "30 s", "0.2 s".
fn seconds(bound: Duration) -> String {
    format!("{} s", bound.as_secs_f64())
}

/// What a failed call says: its stderr, else its stdout, else how it ended.
fn failure(stdout: &str, stderr: &str, status: ExitStatus) -> String {
    [stderr, stdout]
        .iter()
        .map(|text| text.trim())
        .find(|text| !text.is_empty())
        .map_or_else(|| super::runner::describe_status(status), str::to_string)
}

/// Run @p tool with @p args as an owned run: in a process group of its own,
/// ended at the tool's bound. A tool that cannot start is an error naming it.
fn call(tool: &Tool, args: &[&str]) -> Result<Call, Error> {
    let mut command = Command::new(&tool.program);
    command.args(args);
    let child = owned_run::spawn(&mut command).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::ToolNotFound(format!(
                "'{}' is not on PATH; gpu-lock locks and resets the clocks with it",
                tool.program.display()
            ))
        } else {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("{} could not be started: {e}", tool.program.display()),
            ))
        }
    })?;
    // A flag no signal sets: a signal bench records is acted on once the call
    // has finished, so that no lock or reset stops halfway.
    let unset = AtomicI32::new(0);
    let run = owned_run::finish(child, tool.bound, &unset).map_err(Error::Io)?;
    Ok(match run.ending {
        owned_run::Ending::Exited(status) if status.success() => Call::Done(run.stdout),
        owned_run::Ending::Exited(status) => {
            Call::Failed(failure(&run.stdout, &run.stderr, status))
        }
        // The flag is never set, so the call cannot end interrupted.
        owned_run::Ending::TimedOut | owned_run::Ending::Interrupted(_) => Call::TimedOut,
    })
}

/// Query the max graphics clock for a device via nvidia-smi.
fn query_max_graphics_clock(device: u32, tool: &Tool) -> Result<u32, Error> {
    let device_str = device.to_string();
    let args = [
        "-i",
        &device_str,
        "--query-gpu=clocks.max.graphics",
        "--format=csv,noheader,nounits",
    ];
    match call(tool, &args)? {
        Call::Done(text) => text.trim().parse().map_err(|_| {
            Error::GpuLock(format!(
                "unexpected max clock value '{}' for GPU {device}",
                text.trim()
            ))
        }),
        Call::Failed(why) => Err(Error::GpuLock(format!(
            "cannot query the max clock of GPU {device}: {why}"
        ))),
        Call::TimedOut => Err(Error::GpuLock(format!(
            "nvidia-smi did not report the max clock of GPU {device} within {} and was ended",
            seconds(tool.bound)
        ))),
    }
}

/// Enable persistence mode if it's not already on.
fn ensure_persistence_mode(device: u32, tool: &Tool) -> Result<(), Error> {
    let device_str = device.to_string();
    let query = [
        "-i",
        &device_str,
        "--query-gpu=persistence_mode",
        "--format=csv,noheader",
    ];
    if let Call::Done(mode) = call(tool, &query)? {
        if mode.trim().eq_ignore_ascii_case("enabled") {
            return Ok(());
        }
    }
    match call(tool, &["-i", &device_str, "-pm", "1"])? {
        Call::Done(_) => say(&format!("GPU {device}: enabled persistence mode")),
        // Not fatal: clock locking might still work without persistence mode.
        Call::Failed(_) | Call::TimedOut => say(&format!(
            "GPU {device}: warning: could not enable persistence mode (may need root)"
        )),
    }
    Ok(())
}

/// The name of @p signal, for messages.
fn signal_name(signal: i32) -> String {
    #[cfg(unix)]
    {
        match signal {
            libc::SIGHUP => return "SIGHUP".to_string(),
            libc::SIGINT => return "SIGINT".to_string(),
            libc::SIGTERM => return "SIGTERM".to_string(),
            _ => {}
        }
    }
    format!("signal {signal}")
}

/// Pass @p signal on to @p child when no terminal sends it to the command
/// itself: a SIGTERM. True when it was passed on.
#[cfg(unix)]
fn pass_on(child: &Child, signal: i32) -> bool {
    if signal != libc::SIGTERM {
        return false;
    }
    let Ok(pid) = libc::pid_t::try_from(child.id()) else {
        return false;
    };
    // SAFETY: kill(2) only sends a signal. @p child is the command this
    // wrapper started and has not reaped, so its process id cannot have
    // passed to another process.
    unsafe { libc::kill(pid, libc::SIGTERM) == 0 }
}

#[cfg(not(unix))]
fn pass_on(_child: &Child, _signal: i32) -> bool {
    false
}

impl Drop for ResetOnUnwind<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Err(e) = reset_clocks(self.lock.device, &self.lock.tool) {
                say(&format!("Error: {e}"));
            }
        }
    }
}

/// Start @p program with @p args in bench's own process group, and wait for
/// it to end, however long it takes, passing on to it each SIGTERM @p watch
/// records meanwhile. Returns how it ended, or why it could not start.
fn run_command(
    program: &str,
    args: &[String],
    watch: &owned_run::Watch,
) -> Result<ExitStatus, Error> {
    let mut child = Command::new(program)
        .args(args)
        .spawn()
        .map_err(|e| Error::GpuLock(format!("failed to start '{program}': {e}")))?;
    let pid = child.id();
    loop {
        // The command has not been reaped yet: its process id is still its
        // own when a signal is passed on.
        for signal in watch.arrivals() {
            let name = signal_name(signal);
            let what = if pass_on(&child, signal) {
                format!("passed to the command (process {pid}); the clocks are reset once it ends")
            } else {
                format!("the clocks are reset once the command (process {pid}) ends")
            };
            say(&format!(
                "[bench] gpu-lock: {name}: {what}; if it does not end, end it with kill -KILL {pid}"
            ));
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => std::thread::sleep(WAIT_POLL),
            Err(e) => return Err(Error::Io(e)),
        }
    }
}

/* ----------------------------- API ----------------------------- */

/// Lock GPU clocks to a fixed frequency. A lock nvidia-smi refuses leaves
/// nothing to reset; one still running at the tool's bound may have taken
/// effect before it was ended, so the clocks are then reset.
pub fn lock_clocks(cfg: &LockConfig) -> Result<(), Error> {
    let freq = match cfg.freq {
        Some(f) => f,
        None => query_max_graphics_clock(cfg.device, &cfg.tool)?,
    };

    // Ensure persistence mode is on (required for clock locking)
    ensure_persistence_mode(cfg.device, &cfg.tool)?;

    let device_str = cfg.device.to_string();
    let pair = format!("{freq},{freq}");
    match call(&cfg.tool, &["-i", &device_str, "-lgc", &pair])? {
        Call::Done(_) => {
            say(&format!("GPU {}: clocks locked to {freq} MHz", cfg.device));
            Ok(())
        }
        Call::Failed(why) => Err(Error::GpuLock(format!(
            "failed to lock clocks on GPU {}: {why}",
            cfg.device
        ))),
        Call::TimedOut => {
            let reset = match reset_clocks(cfg.device, &cfg.tool) {
                Ok(()) => "the clocks were reset, since it may have taken effect".to_string(),
                Err(e) => e.to_string(),
            };
            Err(Error::GpuLock(format!(
                "nvidia-smi -lgc did not finish within {} and was ended; {reset}",
                seconds(cfg.tool.bound)
            )))
        }
    }
}

/// Reset GPU clocks to default (driver-managed) state. A reset that does not
/// happen names the commands that reset the clocks by hand.
pub fn reset_clocks(device: u32, tool: &Tool) -> Result<(), Error> {
    let device_str = device.to_string();
    let why = match call(tool, &["-i", &device_str, "-rgc"]) {
        Ok(Call::Done(_)) => {
            say(&format!("GPU {device}: clocks reset to default"));
            return Ok(());
        }
        Ok(Call::Failed(why)) => why,
        Ok(Call::TimedOut) => format!(
            "nvidia-smi -rgc did not finish within {} and was ended",
            seconds(tool.bound)
        ),
        Err(e) => e.to_string(),
    };
    Err(Error::GpuLock(format!(
        "the clocks of GPU {device} were not reset: {why}; reset them with \
         `bench gpu-lock reset --device {device}`, or as root with \
         `nvidia-smi -i {device} -rgc`"
    )))
}

/// Lock the clocks, run the command, and reset the clocks once the command
/// has ended, however it ends and however long it takes.
///
/// SIGINT, SIGTERM and SIGHUP are recorded from before the lock until after
/// the reset (each one whose action is the default; one ignored stays
/// ignored). A SIGTERM is passed on to the command; SIGINT and SIGHUP reach
/// it from the terminal. No signal stops the wait or the reset: the clocks
/// are reset after the command ends, and bench then ends by the first signal
/// recorded, which a later one cannot replace; this function does not return
/// then. A signal that arrives while the clocks are being locked lets the
/// lock finish, starts no command and resets the clocks.
///
/// Otherwise `Ok` when the command exited 0 and the clocks were reset, and
/// an error naming each problem: the lock refused, the command that did not
/// start or did not succeed, the reset that did not happen.
pub fn wrap_command(cfg: &WrapConfig) -> Result<(), Error> {
    let Some((program, args)) = cfg.command.split_first() else {
        return Err(Error::InvalidArgs(
            "no command provided after '--'".to_string(),
        ));
    };

    // Recording from before the lock: a signal that arrives while the clocks
    // are being locked is acted on as soon as the lock is in place.
    let watch = owned_run::Watch::start();

    let outcome = match lock_clocks(&cfg.lock) {
        Err(e) => Err(e),
        Ok(()) => {
            let mut guard = ResetOnUnwind {
                lock: &cfg.lock,
                armed: true,
            };
            let ran = if watch.first().is_some() {
                for signal in watch.arrivals() {
                    say(&format!(
                        "[bench] gpu-lock: {}: arrived while the clocks were being locked; the \
                         command is not started",
                        signal_name(signal)
                    ));
                }
                None
            } else {
                Some(run_command(program, args, &watch))
            };
            guard.armed = false;
            let reset = reset_clocks(cfg.lock.device, &cfg.lock.tool);
            let mut problems = Vec::new();
            match ran {
                Some(Ok(status)) if !status.success() => problems.push(format!(
                    "the wrapped command {}",
                    super::runner::describe_status(status)
                )),
                Some(Err(e)) => problems.push(e.to_string()),
                _ => {}
            }
            if let Err(e) = reset {
                problems.push(e.to_string());
            }
            if problems.is_empty() {
                Ok(())
            } else {
                Err(Error::GpuLock(problems.join("; ")))
            }
        }
    };

    // Recording stops only now, after the reset: a signal that arrived during
    // it is recorded too, and the first one recorded is the one bench ends by.
    if let Some(signal) = watch.stop() {
        if let Err(e) = &outcome {
            say(&format!("Error: {e}"));
        }
        owned_run::end_by(signal);
    }
    outcome
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// A lock request for GPU 0 at @p freq, with the default nvidia-smi.
    fn lock(freq: Option<u32>) -> LockConfig {
        LockConfig {
            device: 0,
            freq,
            tool: Tool::default(),
        }
    }

    /// @test wrap_command rejects empty command list, before it locks.
    #[test]
    fn wrap_empty_command_errors() {
        let cfg = WrapConfig {
            lock: lock(Some(1000)),
            command: vec![],
        };
        let result = wrap_command(&cfg);
        assert!(result.is_err());
    }

    /// @test LockConfig with explicit freq stores it.
    #[test]
    fn lock_config_explicit_freq() {
        let cfg = LockConfig {
            device: 2,
            ..lock(Some(1500))
        };
        assert_eq!(cfg.device, 2);
        assert_eq!(cfg.freq, Some(1500));
    }

    /// @test LockConfig with None freq means "use max".
    #[test]
    fn lock_config_auto_freq() {
        let cfg = lock(None);
        assert!(cfg.freq.is_none());
    }

    /// @test The default tool is nvidia-smi on PATH, bounded at 30 s.
    #[test]
    fn default_tool_is_nvidia_smi_bounded() {
        let tool = Tool::default();
        assert_eq!(tool.program, PathBuf::from("nvidia-smi"));
        assert_eq!(tool.bound, Duration::from_secs(30));
    }

    /// @test A bound reads in seconds, whole or not.
    #[test]
    fn seconds_states_the_bound() {
        assert_eq!(seconds(Duration::from_secs(30)), "30 s");
        assert_eq!(seconds(Duration::from_millis(200)), "0.2 s");
    }

    /// @test A failed call is told by its stderr, else its stdout, else by
    /// how it ended.
    #[test]
    #[cfg(unix)]
    fn failure_prefers_stderr_then_stdout_then_status() {
        use std::os::unix::process::ExitStatusExt;
        let status = ExitStatus::from_raw(4 << 8);
        assert_eq!(failure("out\n", "err\n", status), "err");
        assert_eq!(failure("out\n", "  \n", status), "out");
        assert_eq!(failure("", "", status), "exited with status 4");
    }

    /// @test The three recorded signals are named; any other by its number.
    #[test]
    #[cfg(unix)]
    fn signal_names() {
        assert_eq!(signal_name(libc::SIGHUP), "SIGHUP");
        assert_eq!(signal_name(libc::SIGINT), "SIGINT");
        assert_eq!(signal_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(
            signal_name(libc::SIGUSR1),
            format!("signal {}", libc::SIGUSR1)
        );
    }
}
