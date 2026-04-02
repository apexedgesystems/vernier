//! GPU state monitoring: structured snapshots and diff detection.
//!
//! Captures GPU state (temperature, power, clocks, memory, throttle reasons)
//! as structured JSON. Designed to bracket benchmark runs so environmental
//! drift can be detected.
//!
//! Modes:
//!   - `bench gpu-monitor snapshot`              --> JSON to stdout
//!   - `bench gpu-monitor snapshot -o before.json` --> JSON to file
//!   - `bench gpu-monitor diff before.json after.json` --> highlights changes

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::Error;

/* ----------------------------- Types ----------------------------- */

/// Full GPU state snapshot (per-device).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSnapshot {
    /// ISO 8601 timestamp when the snapshot was taken.
    pub timestamp: String,

    /// Per-device state.
    pub devices: Vec<DeviceState>,
}

/// State of a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    pub index: u32,
    pub name: String,

    // Thermal
    pub temperature_c: f64,

    // Power
    pub power_draw_w: f64,
    pub power_limit_w: f64,

    // Clocks (MHz)
    pub clock_graphics_mhz: f64,
    pub clock_mem_mhz: f64,
    pub clock_max_graphics_mhz: f64,

    // Memory (MiB)
    pub memory_used_mib: f64,
    pub memory_total_mib: f64,

    // Utilization (%)
    pub gpu_utilization_pct: f64,
    pub memory_utilization_pct: f64,

    // Throttle
    pub throttle_reasons: String,

    // Performance state (P0-P12)
    pub pstate: String,
}

/// A single field difference between two snapshots.
#[derive(Debug, Clone, Serialize)]
pub struct FieldDiff {
    pub device: u32,
    pub field: String,
    pub before: String,
    pub after: String,
    pub severity: DiffSeverity,
}

/// How concerning is this diff?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffSeverity {
    Info,
    Warning,
}

/* ----------------------------- API ----------------------------- */

/// Capture a GPU state snapshot right now.
pub fn take_snapshot() -> Result<GpuSnapshot, Error> {
    let query = [
        "index",
        "name",
        "temperature.gpu",
        "power.draw",
        "power.default_limit",
        "clocks.current.graphics",
        "clocks.current.memory",
        "clocks.max.graphics",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "gpu_operation_mode.current",
        "pstate",
    ]
    .join(",");

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu", &query, "--format=csv,noheader,nounits"])
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        return Err(Error::ToolNotFound(
            "nvidia-smi failed; is an NVIDIA GPU present?".into(),
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 14 {
            continue;
        }

        devices.push(DeviceState {
            index: fields[0].parse().unwrap_or(0),
            name: fields[1].to_string(),
            temperature_c: fields[2].parse().unwrap_or(0.0),
            power_draw_w: fields[3].parse().unwrap_or(0.0),
            power_limit_w: fields[4].parse().unwrap_or(0.0),
            clock_graphics_mhz: fields[5].parse().unwrap_or(0.0),
            clock_mem_mhz: fields[6].parse().unwrap_or(0.0),
            clock_max_graphics_mhz: fields[7].parse().unwrap_or(0.0),
            memory_used_mib: fields[8].parse().unwrap_or(0.0),
            memory_total_mib: fields[9].parse().unwrap_or(0.0),
            gpu_utilization_pct: fields[10].parse().unwrap_or(0.0),
            memory_utilization_pct: fields[11].parse().unwrap_or(0.0),
            throttle_reasons: fields[12].to_string(),
            pstate: fields[13].to_string(),
        });
    }

    // Also query throttle reasons separately (comma-separated fields conflict
    // with the CSV format, so we use a dedicated query)
    let throttle_output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks_event_reasons.active",
            "--format=csv,noheader",
        ])
        .output()
        .ok();

    if let Some(out) = throttle_output {
        if out.status.success() {
            let throttle_text = String::from_utf8_lossy(&out.stdout);
            for (i, line) in throttle_text.lines().enumerate() {
                if let Some(dev) = devices.get_mut(i) {
                    dev.throttle_reasons = line.trim().to_string();
                }
            }
        }
    }

    let timestamp = chrono_now();

    Ok(GpuSnapshot { timestamp, devices })
}

/// Load a snapshot from a JSON file.
pub fn load_snapshot(path: &Path) -> Result<GpuSnapshot, Error> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|e| Error::Parse(format!("invalid snapshot JSON: {e}")))
}

/// Save a snapshot to a JSON file.
pub fn save_snapshot(snapshot: &GpuSnapshot, path: &Path) -> Result<(), Error> {
    let json = serde_json::to_string_pretty(snapshot).map_err(|e| Error::Parse(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Compare two snapshots and return per-field diffs.
pub fn diff_snapshots(before: &GpuSnapshot, after: &GpuSnapshot) -> Vec<FieldDiff> {
    let mut diffs = Vec::new();

    for b_dev in &before.devices {
        let Some(a_dev) = after.devices.iter().find(|d| d.index == b_dev.index) else {
            diffs.push(FieldDiff {
                device: b_dev.index,
                field: "device".into(),
                before: b_dev.name.clone(),
                after: "(missing)".into(),
                severity: DiffSeverity::Warning,
            });
            continue;
        };

        // Temperature drift
        diff_float(
            &mut diffs,
            b_dev.index,
            "temperature_c",
            b_dev.temperature_c,
            a_dev.temperature_c,
            5.0, // > 5C change is a warning
        );

        // Power draw change
        diff_float(
            &mut diffs,
            b_dev.index,
            "power_draw_w",
            b_dev.power_draw_w,
            a_dev.power_draw_w,
            10.0, // > 10W change is a warning
        );

        // Clock speed changes
        diff_float(
            &mut diffs,
            b_dev.index,
            "clock_graphics_mhz",
            b_dev.clock_graphics_mhz,
            a_dev.clock_graphics_mhz,
            50.0, // > 50 MHz is a warning
        );

        diff_float(
            &mut diffs,
            b_dev.index,
            "clock_mem_mhz",
            b_dev.clock_mem_mhz,
            a_dev.clock_mem_mhz,
            50.0,
        );

        // Memory usage change
        diff_float(
            &mut diffs,
            b_dev.index,
            "memory_used_mib",
            b_dev.memory_used_mib,
            a_dev.memory_used_mib,
            100.0, // > 100 MiB is a warning
        );

        // Utilization change
        diff_float(
            &mut diffs,
            b_dev.index,
            "gpu_utilization_pct",
            b_dev.gpu_utilization_pct,
            a_dev.gpu_utilization_pct,
            20.0,
        );

        // P-state change
        if b_dev.pstate != a_dev.pstate {
            diffs.push(FieldDiff {
                device: b_dev.index,
                field: "pstate".into(),
                before: b_dev.pstate.clone(),
                after: a_dev.pstate.clone(),
                severity: DiffSeverity::Warning,
            });
        }

        // Throttle reasons change
        if b_dev.throttle_reasons != a_dev.throttle_reasons {
            diffs.push(FieldDiff {
                device: b_dev.index,
                field: "throttle_reasons".into(),
                before: b_dev.throttle_reasons.clone(),
                after: a_dev.throttle_reasons.clone(),
                severity: DiffSeverity::Warning,
            });
        }
    }

    // Check for new devices in after
    for a_dev in &after.devices {
        if !before.devices.iter().any(|d| d.index == a_dev.index) {
            diffs.push(FieldDiff {
                device: a_dev.index,
                field: "device".into(),
                before: "(missing)".into(),
                after: a_dev.name.clone(),
                severity: DiffSeverity::Warning,
            });
        }
    }

    diffs
}

/// Print diffs to the terminal with colored output.
pub fn print_diff(before: &GpuSnapshot, after: &GpuSnapshot, diffs: &[FieldDiff]) {
    println!();
    println!("=== GPU State Diff ===");
    println!();
    println!("  Before: {}", before.timestamp);
    println!("  After:  {}", after.timestamp);
    println!();

    if diffs.is_empty() {
        println!("  No significant changes detected.");
        println!();
        return;
    }

    for d in diffs {
        let color = match d.severity {
            DiffSeverity::Info => "\x1b[94m",    // blue
            DiffSeverity::Warning => "\x1b[93m", // yellow
        };
        let tag = match d.severity {
            DiffSeverity::Info => "INFO",
            DiffSeverity::Warning => "WARN",
        };

        println!(
            "  {color}[{tag:>4}]\x1b[0m GPU {} {:<24} {} -> {}",
            d.device, d.field, d.before, d.after
        );
    }

    let warn_count = diffs
        .iter()
        .filter(|d| d.severity == DiffSeverity::Warning)
        .count();
    let info_count = diffs.len() - warn_count;

    println!();
    println!(
        "  {} change(s): {} warning(s), {} info",
        diffs.len(),
        warn_count,
        info_count
    );

    if warn_count > 0 {
        println!();
        println!("  Environment changed during benchmark run. Results may not be comparable.");
    }
    println!();
}

/* ----------------------------- Helpers ----------------------------- */

/// Compare two float fields and add a diff entry if they differ significantly.
fn diff_float(
    diffs: &mut Vec<FieldDiff>,
    device: u32,
    field: &str,
    before: f64,
    after: f64,
    warn_threshold: f64,
) {
    let delta = (after - before).abs();
    if delta < 0.01 {
        return; // No meaningful change
    }

    let severity = if delta >= warn_threshold {
        DiffSeverity::Warning
    } else {
        DiffSeverity::Info
    };

    diffs.push(FieldDiff {
        device,
        field: field.to_string(),
        before: format!("{before:.1}"),
        after: format!("{after:.1}"),
        severity,
    });
}

/// Simple ISO 8601 timestamp without pulling in the chrono crate.
fn chrono_now() -> String {
    let output = Command::new("date")
        .args(["+%Y-%m-%dT%H:%M:%S%z"])
        .output()
        .ok();

    match output {
        Some(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device(index: u32, temp: f64, clock: f64, pstate: &str) -> DeviceState {
        DeviceState {
            index,
            name: format!("GPU {index}"),
            temperature_c: temp,
            power_draw_w: 50.0,
            power_limit_w: 300.0,
            clock_graphics_mhz: clock,
            clock_mem_mhz: 5000.0,
            clock_max_graphics_mhz: 2100.0,
            memory_used_mib: 1024.0,
            memory_total_mib: 8192.0,
            gpu_utilization_pct: 50.0,
            memory_utilization_pct: 25.0,
            throttle_reasons: "None".into(),
            pstate: pstate.into(),
        }
    }

    fn make_snapshot(devices: Vec<DeviceState>) -> GpuSnapshot {
        GpuSnapshot {
            timestamp: "2026-03-21T12:00:00-0500".into(),
            devices,
        }
    }

    /// @test Identical snapshots produce no diffs.
    #[test]
    fn identical_no_diffs() {
        let snap = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P0")]);
        let diffs = diff_snapshots(&snap, &snap);
        assert!(diffs.is_empty());
    }

    /// @test Temperature drift above threshold triggers warning.
    #[test]
    fn temp_drift_warning() {
        let before = make_snapshot(vec![make_device(0, 40.0, 1800.0, "P0")]);
        let after = make_snapshot(vec![make_device(0, 50.0, 1800.0, "P0")]);
        let diffs = diff_snapshots(&before, &after);
        let temp_diff = diffs.iter().find(|d| d.field == "temperature_c").unwrap();
        assert_eq!(temp_diff.severity, DiffSeverity::Warning);
    }

    /// @test Small temperature change is info, not warning.
    #[test]
    fn temp_small_change_info() {
        let before = make_snapshot(vec![make_device(0, 40.0, 1800.0, "P0")]);
        let after = make_snapshot(vec![make_device(0, 43.0, 1800.0, "P0")]);
        let diffs = diff_snapshots(&before, &after);
        let temp_diff = diffs.iter().find(|d| d.field == "temperature_c").unwrap();
        assert_eq!(temp_diff.severity, DiffSeverity::Info);
    }

    /// @test Clock speed change triggers a diff.
    #[test]
    fn clock_change_detected() {
        let before = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P0")]);
        let after = make_snapshot(vec![make_device(0, 45.0, 1500.0, "P0")]);
        let diffs = diff_snapshots(&before, &after);
        assert!(diffs.iter().any(|d| d.field == "clock_graphics_mhz"));
    }

    /// @test P-state change triggers warning.
    #[test]
    fn pstate_change_warning() {
        let before = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P0")]);
        let after = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P2")]);
        let diffs = diff_snapshots(&before, &after);
        let ps_diff = diffs.iter().find(|d| d.field == "pstate").unwrap();
        assert_eq!(ps_diff.severity, DiffSeverity::Warning);
        assert_eq!(ps_diff.before, "P0");
        assert_eq!(ps_diff.after, "P2");
    }

    /// @test Missing device in after snapshot triggers warning.
    #[test]
    fn missing_device_warning() {
        let before = make_snapshot(vec![
            make_device(0, 45.0, 1800.0, "P0"),
            make_device(1, 42.0, 1700.0, "P0"),
        ]);
        let after = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P0")]);
        let diffs = diff_snapshots(&before, &after);
        assert!(diffs.iter().any(|d| d.field == "device" && d.device == 1));
    }

    /// @test Snapshot serializes to valid JSON and round-trips.
    #[test]
    fn snapshot_json_roundtrip() {
        let snap = make_snapshot(vec![make_device(0, 45.0, 1800.0, "P0")]);
        let json = serde_json::to_string_pretty(&snap).unwrap();
        let parsed: GpuSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.devices.len(), 1);
        assert_eq!(parsed.devices[0].index, 0);
        assert!((parsed.devices[0].temperature_c - 45.0).abs() < 0.01);
    }

    /// @test diff_float ignores negligible changes.
    #[test]
    fn diff_float_ignores_tiny() {
        let mut diffs = Vec::new();
        diff_float(&mut diffs, 0, "test", 100.0, 100.005, 5.0);
        assert!(diffs.is_empty());
    }

    /// @test take_snapshot doesn't panic regardless of GPU presence.
    #[test]
    fn take_snapshot_graceful() {
        let _ = take_snapshot();
    }
}
