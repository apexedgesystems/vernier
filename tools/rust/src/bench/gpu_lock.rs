//! GPU clock management: lock/reset GPU clocks for reproducible benchmarks.
//!
//! Wraps `nvidia-smi -lgc` / `-rgc` to eliminate clock frequency variance
//! during benchmark runs. Supports three modes:
//!
//!   1. `bench gpu-lock lock`   -- lock clocks and exit
//!   2. `bench gpu-lock reset`  -- reset clocks and exit
//!   3. `bench gpu-lock lock -- <command>` -- lock, run command, reset on exit
//!
//! The wrapper mode resets clocks via an RAII drop guard, so clocks are
//! restored even if the wrapped command fails or the runner panics.

use std::process::Command;

use super::Error;

/* ----------------------------- Public Types ----------------------------- */

/// Configuration for a gpu-lock operation.
#[derive(Debug)]
pub struct LockConfig {
    /// GPU device index (nvidia-smi -i).
    pub device: u32,

    /// Target frequency in MHz. If None, locks to max reported clock.
    pub freq: Option<u32>,
}

/// Configuration for the wrapper mode (lock --> run --> reset).
#[derive(Debug)]
pub struct WrapConfig {
    pub lock: LockConfig,

    /// Command and arguments to execute while clocks are locked.
    pub command: Vec<String>,
}

/* ----------------------------- API ----------------------------- */

/// Lock GPU clocks to a fixed frequency.
pub fn lock_clocks(cfg: &LockConfig) -> Result<(), Error> {
    let freq = match cfg.freq {
        Some(f) => f,
        None => query_max_graphics_clock(cfg.device)?,
    };

    // Ensure persistence mode is on (required for clock locking)
    ensure_persistence_mode(cfg.device)?;

    let freq_str = freq.to_string();
    let device_str = cfg.device.to_string();

    let output = Command::new("nvidia-smi")
        .args(["-i", &device_str, "-lgc", &format!("{freq_str},{freq_str}")])
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if !stderr.is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        return Err(Error::InvalidArgs(format!(
            "failed to lock clocks on GPU {}: {msg}",
            cfg.device
        )));
    }

    eprintln!("GPU {}: clocks locked to {freq} MHz", cfg.device);
    Ok(())
}

/// Reset GPU clocks to default (driver-managed) state.
pub fn reset_clocks(device: u32) -> Result<(), Error> {
    let device_str = device.to_string();

    let output = Command::new("nvidia-smi")
        .args(["-i", &device_str, "-rgc"])
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if !stderr.is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        return Err(Error::InvalidArgs(format!(
            "failed to reset clocks on GPU {device}: {msg}"
        )));
    }

    eprintln!("GPU {device}: clocks reset to default");
    Ok(())
}

/// Lock clocks, run a child command, then reset clocks on exit (even on failure).
pub fn wrap_command(cfg: &WrapConfig) -> Result<i32, Error> {
    if cfg.command.is_empty() {
        return Err(Error::InvalidArgs(
            "no command provided after '--'".to_string(),
        ));
    }

    // Lock clocks
    lock_clocks(&cfg.lock)?;

    // Guard ensures clocks are reset even if the command fails or we panic
    let _guard = ClockResetGuard {
        device: cfg.lock.device,
    };

    // Run the child command
    let program = &cfg.command[0];
    let args = &cfg.command[1..];

    let status = Command::new(program)
        .args(args)
        .status()
        .map_err(|e| Error::InvalidArgs(format!("failed to execute '{}': {e}", program)))?;

    // Guard will reset clocks on drop
    Ok(status.code().unwrap_or(255))
}

/* ----------------------------- Helpers ----------------------------- */

/// Query the max graphics clock for a device via nvidia-smi.
fn query_max_graphics_clock(device: u32) -> Result<u32, Error> {
    let device_str = device.to_string();
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            &device_str,
            "--query-gpu=clocks.max.graphics",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        return Err(Error::InvalidArgs(format!(
            "cannot query max clock for GPU {device}"
        )));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let freq: u32 = text.trim().parse().map_err(|_| {
        Error::InvalidArgs(format!(
            "unexpected max clock value '{}' for GPU {device}",
            text.trim()
        ))
    })?;

    Ok(freq)
}

/// Enable persistence mode if it's not already on.
fn ensure_persistence_mode(device: u32) -> Result<(), Error> {
    let device_str = device.to_string();

    // Check current state
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            &device_str,
            "--query-gpu=persistence_mode",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(Error::Io)?;

    if output.status.success() {
        let mode = String::from_utf8_lossy(&output.stdout);
        if mode.trim().eq_ignore_ascii_case("enabled") {
            return Ok(());
        }
    }

    // Try to enable it
    let enable = Command::new("nvidia-smi")
        .args(["-i", &device_str, "-pm", "1"])
        .output()
        .map_err(Error::Io)?;

    if enable.status.success() {
        eprintln!("GPU {device}: enabled persistence mode");
        Ok(())
    } else {
        // Non-fatal: clock locking might still work without persistence mode
        eprintln!("GPU {device}: warning: could not enable persistence mode (may need root)");
        Ok(())
    }
}

/// RAII guard that resets clocks on drop.
struct ClockResetGuard {
    device: u32,
}

impl Drop for ClockResetGuard {
    fn drop(&mut self) {
        if let Err(e) = reset_clocks(self.device) {
            eprintln!("warning: failed to reset GPU clocks: {e}");
        }
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test query_max_graphics_clock doesn't panic on any system.
    #[test]
    fn query_max_clock_graceful() {
        // On machines without GPU, returns an error; with GPU, returns a value
        let result = query_max_graphics_clock(0);
        match result {
            Ok(freq) => assert!(freq > 0, "max clock should be positive"),
            Err(_) => {} // Expected on non-GPU machines
        }
    }

    /// @test wrap_command rejects empty command list.
    #[test]
    fn wrap_empty_command_errors() {
        let cfg = WrapConfig {
            lock: LockConfig {
                device: 0,
                freq: Some(1000),
            },
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
            freq: Some(1500),
        };
        assert_eq!(cfg.device, 2);
        assert_eq!(cfg.freq, Some(1500));
    }

    /// @test LockConfig with None freq means "use max".
    #[test]
    fn lock_config_auto_freq() {
        let cfg = LockConfig {
            device: 0,
            freq: None,
        };
        assert!(cfg.freq.is_none());
    }

    /// @test ClockResetGuard doesn't panic on drop (even without GPU).
    #[test]
    fn reset_guard_drop_graceful() {
        // Just construct and drop -- should not panic even without GPU
        let _guard = ClockResetGuard { device: 99 };
        // Guard drops here
    }
}
