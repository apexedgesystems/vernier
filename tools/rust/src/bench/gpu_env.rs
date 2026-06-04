//! GPU environment validation: device discovery, driver/toolkit checks,
//! clock state, and profiler availability.
//!
//! All checks shell out to `nvidia-smi` and toolkit CLIs rather than linking
//! against CUDA libraries, so the binary works on any machine -- checks simply
//! report "not found" when the tools are absent.

use std::process::Command;

use super::{find_in_path, CheckResult, CheckStatus};

/* ----------------------------- API ----------------------------- */

/// Run all GPU environment checks. Returns a list of results.
pub fn run_checks() -> Vec<CheckResult> {
    let mut results = Vec::new();

    results.push(check_nvidia_smi());
    results.push(check_nvidia_driver());
    results.push(check_cuda_toolkit());
    results.push(check_nsys_trace_compat());
    results.extend(check_gpu_devices());
    results.push(check_persistence_mode());
    results.push(check_clock_state());
    results.push(check_ecc_mode());
    results.push(check_power_state());
    results.push(check_thermal_state());
    results.push(check_nsight_systems());
    results.push(check_nsight_compute());
    results.extend(check_p2p_access());

    results
}

/// Print GPU validation results with colored output.
pub fn print_results(results: &[CheckResult]) {
    println!();
    println!("=== GPU Environment Check ===");
    println!();

    let mut pass_count = 0;
    let mut warn_count = 0;
    let mut fail_count = 0;

    for r in results {
        let (tag, color) = match r.status {
            CheckStatus::Ok => {
                pass_count += 1;
                ("OK", "\x1b[92m")
            }
            CheckStatus::Warn => {
                warn_count += 1;
                ("WARN", "\x1b[93m")
            }
            CheckStatus::Fail => {
                fail_count += 1;
                ("FAIL", "\x1b[91m")
            }
        };
        println!("  {color}[{tag:>4}]\x1b[0m {:<30} {}", r.label, r.detail);
    }

    println!();
    println!("  ---");
    println!(
        "  {} passed, {} warnings, {} failures",
        pass_count, warn_count, fail_count
    );

    if fail_count > 0 {
        println!();
        println!("  GPU environment has issues that will prevent GPU benchmarking.");
    } else if warn_count > 0 {
        println!();
        println!("  GPU environment is mostly ready. Address warnings for reproducible results.");
    } else {
        println!();
        println!("  GPU environment is ready for benchmarking.");
    }
    println!();
}

/// Returns true if any check failed.
pub fn has_failures(results: &[CheckResult]) -> bool {
    results.iter().any(|r| r.status == CheckStatus::Fail)
}

/* ----------------------------- Helpers ----------------------------- */

/// Run nvidia-smi with a query and return trimmed stdout, or None on failure.
fn smi_query(query: &str) -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu", query, "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Run nvidia-smi with a query and return per-GPU lines.
fn smi_query_lines(query: &str) -> Vec<String> {
    smi_query(query)
        .map(|s| s.lines().map(|l| l.trim().to_string()).collect())
        .unwrap_or_default()
}

/// Run a command and capture its stdout.
fn cmd_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/* ----------------------------- Individual Checks ----------------------------- */

fn check_nvidia_smi() -> CheckResult {
    match find_in_path("nvidia-smi") {
        Some(path) => {
            // Verify it actually runs
            match Command::new("nvidia-smi").arg("--version").output() {
                Ok(output) if output.status.success() => CheckResult {
                    label: "nvidia-smi".into(),
                    status: CheckStatus::Ok,
                    detail: format!("found at {}", path.display()),
                },
                _ => CheckResult {
                    label: "nvidia-smi".into(),
                    status: CheckStatus::Fail,
                    detail: format!("found at {} but failed to execute", path.display()),
                },
            }
        }
        None => CheckResult {
            label: "nvidia-smi".into(),
            status: CheckStatus::Fail,
            detail: "not found in PATH; install NVIDIA driver".into(),
        },
    }
}

fn check_nvidia_driver() -> CheckResult {
    match smi_query("driver_version") {
        Some(version) => CheckResult {
            label: "NVIDIA driver".into(),
            status: CheckStatus::Ok,
            detail: format!("version {version}"),
        },
        None => CheckResult {
            label: "NVIDIA driver".into(),
            status: CheckStatus::Fail,
            detail: "cannot query driver version; is nvidia-smi working?".into(),
        },
    }
}

fn check_cuda_toolkit() -> CheckResult {
    // Check nvcc first (CUDA toolkit compiler)
    if let Some(out) = cmd_stdout("nvcc", &["--version"]) {
        // Parse version from output like "Cuda compilation tools, release 12.8, V12.8.89"
        if let Some(line) = out.lines().find(|l| l.contains("release")) {
            return CheckResult {
                label: "CUDA toolkit".into(),
                status: CheckStatus::Ok,
                detail: line.trim().to_string(),
            };
        }
        return CheckResult {
            label: "CUDA toolkit".into(),
            status: CheckStatus::Ok,
            detail: "nvcc found".into(),
        };
    }

    // Fall back to nvidia-smi banner output which contains "CUDA Version: X.Y"
    if let Some(banner) = cmd_stdout("nvidia-smi", &[]) {
        if let Some(cuda_ver) = banner.lines().find_map(|l| l.split("CUDA Version:").nth(1)) {
            let version = cuda_ver
                .trim()
                .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.');
            return CheckResult {
                label: "CUDA toolkit".into(),
                status: CheckStatus::Warn,
                detail: format!(
                    "nvcc not found; driver supports CUDA {version}. Install CUDA toolkit for compilation"
                ),
            };
        }
    }

    CheckResult {
        label: "CUDA toolkit".into(),
        status: CheckStatus::Fail,
        detail: "neither nvcc nor CUDA-capable driver found".into(),
    }
}

fn check_gpu_devices() -> Vec<CheckResult> {
    let names = smi_query_lines("name");
    let memories = smi_query_lines("memory.total");
    let computes = smi_query_lines("compute_cap");

    if names.is_empty() {
        return vec![CheckResult {
            label: "GPU devices".into(),
            status: CheckStatus::Fail,
            detail: "no GPU devices detected".into(),
        }];
    }

    let mut results = Vec::new();
    results.push(CheckResult {
        label: "GPU devices".into(),
        status: CheckStatus::Ok,
        detail: format!("{} device(s) detected", names.len()),
    });

    for (i, name) in names.iter().enumerate() {
        let mem = memories
            .get(i)
            .map(|m| format!("{m} MiB"))
            .unwrap_or_default();
        let cc = computes
            .get(i)
            .map(|c| format!("SM {c}"))
            .unwrap_or_default();

        let parts: Vec<&str> = [name.as_str(), mem.as_str(), cc.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();

        results.push(CheckResult {
            label: format!("  GPU {i}"),
            status: CheckStatus::Ok,
            detail: parts.join(", "),
        });
    }

    results
}

fn check_persistence_mode() -> CheckResult {
    match smi_query("persistence_mode") {
        Some(mode) => {
            let lines: Vec<&str> = mode.lines().collect();
            let all_enabled = lines
                .iter()
                .all(|l| l.trim().eq_ignore_ascii_case("enabled"));

            if all_enabled {
                CheckResult {
                    label: "persistence mode".into(),
                    status: CheckStatus::Ok,
                    detail: "enabled (GPU stays initialized between runs)".into(),
                }
            } else {
                CheckResult {
                    label: "persistence mode".into(),
                    status: CheckStatus::Warn,
                    detail: "disabled; run 'sudo nvidia-smi -pm 1' to avoid cold-start overhead"
                        .into(),
                }
            }
        }
        None => CheckResult {
            label: "persistence mode".into(),
            status: CheckStatus::Warn,
            detail: "cannot query persistence mode".into(),
        },
    }
}

fn check_clock_state() -> CheckResult {
    let graphics = smi_query_lines("clocks.current.graphics");
    let max_graphics = smi_query_lines("clocks.max.graphics");

    if graphics.is_empty() || max_graphics.is_empty() {
        return CheckResult {
            label: "GPU clocks".into(),
            status: CheckStatus::Warn,
            detail: "cannot query clock speeds".into(),
        };
    }

    // Check first GPU (primary benchmark target)
    let current: f64 = graphics[0].parse().unwrap_or(0.0);
    let max: f64 = max_graphics[0].parse().unwrap_or(0.0);

    if max > 0.0 && current > 0.0 {
        let ratio = current / max;
        if ratio > 0.95 {
            CheckResult {
                label: "GPU clocks".into(),
                status: CheckStatus::Ok,
                detail: format!("{current:.0}/{max:.0} MHz (at max)"),
            }
        } else if ratio > 0.5 {
            CheckResult {
                label: "GPU clocks".into(),
                status: CheckStatus::Warn,
                detail: format!(
                    "{current:.0}/{max:.0} MHz; clocks not locked. \
                     Lock with 'sudo nvidia-smi -lgc {max:.0},{max:.0}' for stable benchmarks"
                ),
            }
        } else {
            CheckResult {
                label: "GPU clocks".into(),
                status: CheckStatus::Warn,
                detail: format!(
                    "{current:.0}/{max:.0} MHz (idle). \
                     Clocks will boost under load but may cause variance"
                ),
            }
        }
    } else {
        CheckResult {
            label: "GPU clocks".into(),
            status: CheckStatus::Warn,
            detail: format!("current={current:.0} MHz, max={max:.0} MHz"),
        }
    }
}

fn check_ecc_mode() -> CheckResult {
    match smi_query("ecc.mode.current") {
        Some(mode) => {
            let first = mode.lines().next().unwrap_or("").trim();
            if first.eq_ignore_ascii_case("enabled") {
                CheckResult {
                    label: "ECC memory".into(),
                    status: CheckStatus::Warn,
                    detail: "enabled (reduces memory bandwidth ~6%; disable for max throughput benchmarks)".into(),
                }
            } else if first.eq_ignore_ascii_case("disabled") {
                CheckResult {
                    label: "ECC memory".into(),
                    status: CheckStatus::Ok,
                    detail: "disabled (max memory bandwidth available)".into(),
                }
            } else if first.contains("[N/A]") || first.eq_ignore_ascii_case("n/a") {
                CheckResult {
                    label: "ECC memory".into(),
                    status: CheckStatus::Ok,
                    detail: "not applicable (consumer GPU)".into(),
                }
            } else {
                CheckResult {
                    label: "ECC memory".into(),
                    status: CheckStatus::Ok,
                    detail: format!("mode: {first}"),
                }
            }
        }
        None => CheckResult {
            label: "ECC memory".into(),
            status: CheckStatus::Ok,
            detail: "not supported on this GPU".into(),
        },
    }
}

fn check_power_state() -> CheckResult {
    let draw = smi_query_lines("power.draw");

    if draw.is_empty() {
        return CheckResult {
            label: "power state".into(),
            status: CheckStatus::Warn,
            detail: "cannot query power draw".into(),
        };
    }

    let current: f64 = draw[0].parse().unwrap_or(0.0);

    // Try power.limit first, fall back to power.default_limit, then power.max_limit
    let limit = smi_query_lines("power.limit");
    let default_limit = smi_query_lines("power.default_limit");
    let max_limit = smi_query_lines("power.max_limit");

    let max: f64 = limit
        .first()
        .and_then(|s| s.parse().ok())
        .or_else(|| default_limit.first().and_then(|s| s.parse().ok()))
        .or_else(|| max_limit.first().and_then(|s| s.parse().ok()))
        .unwrap_or(0.0);

    if max > 0.0 && current > 0.0 {
        let headroom = ((max - current) / max) * 100.0;
        if headroom < 5.0 {
            CheckResult {
                label: "power state".into(),
                status: CheckStatus::Warn,
                detail: format!(
                    "{current:.1}/{max:.1} W (near limit; may throttle under sustained load)"
                ),
            }
        } else {
            CheckResult {
                label: "power state".into(),
                status: CheckStatus::Ok,
                detail: format!("{current:.1}/{max:.1} W ({headroom:.0}% headroom)"),
            }
        }
    } else {
        CheckResult {
            label: "power state".into(),
            status: CheckStatus::Warn,
            detail: format!("draw={current:.1} W, limit={max:.1} W"),
        }
    }
}

fn check_thermal_state() -> CheckResult {
    let temps = smi_query_lines("temperature.gpu");
    let tlimit = smi_query_lines("temperature.gpu.tlimit");

    if temps.is_empty() {
        return CheckResult {
            label: "thermal state".into(),
            status: CheckStatus::Warn,
            detail: "cannot query GPU temperature".into(),
        };
    }

    let current: f64 = temps[0].parse().unwrap_or(0.0);

    // temperature.gpu.tlimit is the delta (degrees of headroom to throttle point),
    // NOT the absolute threshold. If available, margin = tlimit value directly.
    // Otherwise fall back to a conservative 83C default threshold.
    let margin: f64 = tlimit
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(83.0 - current);

    let threshold = current + margin;

    if margin > 20.0 {
        CheckResult {
            label: "thermal state".into(),
            status: CheckStatus::Ok,
            detail: format!("{current:.0}C (throttle at {threshold:.0}C, {margin:.0}C headroom)"),
        }
    } else if margin > 5.0 {
        CheckResult {
            label: "thermal state".into(),
            status: CheckStatus::Warn,
            detail: format!(
                "{current:.0}C (throttle at {threshold:.0}C, only {margin:.0}C headroom; \
                 sustained benchmarks may throttle)"
            ),
        }
    } else {
        CheckResult {
            label: "thermal state".into(),
            status: CheckStatus::Warn,
            detail: format!(
                "{current:.0}C (throttle at {threshold:.0}C; GPU is hot, \
                 results will be unreliable)"
            ),
        }
    }
}

fn check_nsight_systems() -> CheckResult {
    if let Some(out) = cmd_stdout("nsys", &["--version"]) {
        let version = out
            .lines()
            .find(|l| l.contains("version"))
            .unwrap_or(&out)
            .trim();
        CheckResult {
            label: "Nsight Systems".into(),
            status: CheckStatus::Ok,
            detail: version.to_string(),
        }
    } else if find_in_path("nsys").is_some() {
        CheckResult {
            label: "Nsight Systems".into(),
            status: CheckStatus::Ok,
            detail: "nsys found in PATH".into(),
        }
    } else {
        CheckResult {
            label: "Nsight Systems".into(),
            status: CheckStatus::Warn,
            detail: "nsys not found; install CUDA toolkit for GPU timeline profiling".into(),
        }
    }
}

fn check_nsight_compute() -> CheckResult {
    if let Some(out) = cmd_stdout("ncu", &["--version"]) {
        let version = out
            .lines()
            .find(|l| l.contains("version") || l.contains("Version"))
            .unwrap_or(&out)
            .trim();
        CheckResult {
            label: "Nsight Compute".into(),
            status: CheckStatus::Ok,
            detail: version.to_string(),
        }
    } else if find_in_path("ncu").is_some() {
        CheckResult {
            label: "Nsight Compute".into(),
            status: CheckStatus::Ok,
            detail: "ncu found in PATH".into(),
        }
    } else {
        CheckResult {
            label: "Nsight Compute".into(),
            status: CheckStatus::Warn,
            detail: "ncu not found; install CUDA toolkit for kernel-level profiling".into(),
        }
    }
}

fn check_p2p_access() -> Vec<CheckResult> {
    let names = smi_query_lines("name");

    // P2P only relevant for multi-GPU
    if names.len() < 2 {
        return vec![];
    }

    // Use nvidia-smi topo to check P2P connectivity
    match Command::new("nvidia-smi").arg("topo").arg("-m").output() {
        Ok(output) if output.status.success() => {
            let topo = String::from_utf8_lossy(&output.stdout);
            let has_nvlink = topo.contains("NV");
            let has_pix = topo.contains("PIX") || topo.contains("PXB") || topo.contains("PHB");

            let detail = if has_nvlink {
                "NVLink detected; high-bandwidth P2P available".to_string()
            } else if has_pix {
                "PCIe P2P available; no NVLink".to_string()
            } else {
                "P2P topology queried; check 'nvidia-smi topo -m' for details".to_string()
            };

            vec![CheckResult {
                label: "P2P topology".into(),
                status: CheckStatus::Ok,
                detail,
            }]
        }
        _ => vec![CheckResult {
            label: "P2P topology".into(),
            status: CheckStatus::Warn,
            detail: "cannot query GPU topology".into(),
        }],
    }
}

/// Parse a CUDA `major.minor` from strings like "release 13.1, V13.1.115",
/// "CUDA Version: 13.0", or "13.1" (the patch component, if any, is ignored).
fn parse_cuda_version(text: &str) -> Option<(u32, u32)> {
    for token in text.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
        let mut parts = token.split('.');
        if let (Some(major), Some(minor)) = (parts.next(), parts.next()) {
            if let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                return Some((major, minor));
            }
        }
    }
    None
}

/// Compare the local CUDA toolkit (nvcc) against the CUDA version the driver
/// provides (nvidia-smi). nsys can only trace a binary whose toolkit CUDA is
/// not newer than the driver's, so a toolkit ahead of the driver is the common
/// cause of a silent empty nsys trace. ncu and in-process CUPTI are unaffected.
fn check_nsys_trace_compat() -> CheckResult {
    let label = "nsys trace compatibility".to_string();

    let toolkit = cmd_stdout("nvcc", &["--version"])
        .and_then(|out| {
            out.lines()
                .find(|l| l.contains("release"))
                .map(str::to_string)
        })
        .and_then(|line| parse_cuda_version(&line));

    let driver = cmd_stdout("nvidia-smi", &[])
        .and_then(|banner| {
            banner
                .lines()
                .find_map(|l| l.split("CUDA Version:").nth(1).map(str::to_string))
        })
        .and_then(|s| parse_cuda_version(&s));

    match (toolkit, driver) {
        (Some(t), Some(d)) if t > d => CheckResult {
            label,
            status: CheckStatus::Warn,
            detail: format!(
                "CUDA toolkit {}.{} is ahead of the driver's CUDA {}.{}; nsys cannot trace CUDA \
                 activity (build with a toolkit <= the driver's CUDA, or upgrade the driver). ncu \
                 and in-process CUPTI are unaffected.",
                t.0, t.1, d.0, d.1
            ),
        },
        (Some(t), Some(d)) => CheckResult {
            label,
            status: CheckStatus::Ok,
            detail: format!(
                "toolkit {}.{} <= driver CUDA {}.{}; nsys can trace",
                t.0, t.1, d.0, d.1
            ),
        },
        (None, _) => CheckResult {
            label,
            status: CheckStatus::Ok,
            detail: "no local CUDA toolkit (nvcc); toolkit/driver skew check not applicable".into(),
        },
        (Some(_), None) => CheckResult {
            label,
            status: CheckStatus::Warn,
            detail: "cannot read the driver's CUDA version from nvidia-smi to compare".into(),
        },
    }
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test run_checks returns non-empty results.
    #[test]
    fn checks_return_results() {
        let results = run_checks();
        // Should always have at least the tool/driver/toolkit checks
        assert!(results.len() >= 3, "got {} results", results.len());
    }

    /// @test CUDA version strings parse to (major, minor), patch ignored.
    #[test]
    fn parse_cuda_version_forms() {
        assert_eq!(
            parse_cuda_version("Cuda compilation tools, release 13.1, V13.1.115"),
            Some((13, 1))
        );
        assert_eq!(parse_cuda_version("CUDA Version: 13.0"), Some((13, 0)));
        assert_eq!(parse_cuda_version("13.2"), Some((13, 2)));
        assert_eq!(parse_cuda_version("no version here"), None);
    }

    /// @test The toolkit-vs-driver ordering used by the skew check.
    #[test]
    fn toolkit_ahead_of_driver_ordering() {
        assert!((13u32, 1u32) > (13u32, 0u32)); // 13.1 toolkit ahead of 13.0 driver
        assert!(!((13u32, 0u32) > (13u32, 1u32))); // matched / behind is fine
        assert!((14u32, 0u32) > (13u32, 9u32)); // major dominates minor
    }

    /// @test Each check has a non-empty label and detail.
    #[test]
    fn all_checks_have_labels() {
        let results = run_checks();
        for r in &results {
            assert!(!r.label.is_empty(), "empty label found");
            assert!(!r.detail.is_empty(), "empty detail for '{}'", r.label);
        }
    }

    /// @test smi_query returns None when nvidia-smi is absent (graceful).
    #[test]
    fn smi_query_graceful_on_missing() {
        // This test verifies we don't panic; result depends on environment
        let _ = smi_query("name");
    }

    /// @test cmd_stdout returns None for nonexistent commands.
    #[test]
    fn cmd_stdout_missing_program() {
        assert!(cmd_stdout("__nonexistent_program_vernier__", &["--version"]).is_none());
    }

    /// @test check_nvidia_smi produces a valid CheckResult regardless of GPU presence.
    #[test]
    fn nvidia_smi_check_valid() {
        let result = check_nvidia_smi();
        assert!(!result.label.is_empty());
        assert!(!result.detail.is_empty());
        // On machines without GPU, should be Fail; with GPU, should be Ok
        assert!(matches!(result.status, CheckStatus::Ok | CheckStatus::Fail));
    }

    /// @test has_failures detects failures correctly.
    #[test]
    fn has_failures_logic() {
        let ok_only = vec![CheckResult {
            label: "test".into(),
            status: CheckStatus::Ok,
            detail: "ok".into(),
        }];
        assert!(!has_failures(&ok_only));

        let with_fail = vec![
            CheckResult {
                label: "a".into(),
                status: CheckStatus::Ok,
                detail: "ok".into(),
            },
            CheckResult {
                label: "b".into(),
                status: CheckStatus::Fail,
                detail: "bad".into(),
            },
        ];
        assert!(has_failures(&with_fail));
    }

    /// @test P2P check returns empty vec for environments with <2 GPUs.
    #[test]
    fn p2p_skipped_single_gpu() {
        // We can't control GPU count, but verify it doesn't panic
        let results = check_p2p_access();
        // Either empty (0-1 GPUs) or has a result (multi-GPU)
        for r in &results {
            assert!(!r.label.is_empty());
        }
    }
}
