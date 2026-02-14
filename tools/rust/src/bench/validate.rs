//! Environment validation: check for profiling tools, permissions, and settings.
//!
//! Mirrors the C++ `--profile-check` functionality in PerfConfig.hpp but adds
//! checks for FlameGraph scripts, nsight, and bpftrace.

use std::path::PathBuf;

use super::{find_in_path, CheckResult, CheckStatus};

/* ----------------------------- API ----------------------------- */

/// Run all environment validation checks. Returns a list of results.
pub fn run_checks() -> Vec<CheckResult> {
    vec![
        check_perf(),
        check_perf_paranoid(),
        check_gperftools(),
        check_aslr(),
        check_flamegraph(),
        check_nsight(),
        check_bpftrace(),
    ]
}

/// Print validation results in the same format as C++ --profile-check.
pub fn print_results(results: &[CheckResult]) {
    println!();
    println!("=== Profile Readiness Check ===");
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
        println!("  Install missing tools for full profiling support.");
    } else if warn_count > 0 {
        println!();
        println!("  Environment is mostly ready. Address warnings for best results.");
    } else {
        println!();
        println!("  Environment is ready for profiling.");
    }
    println!();
}

/// Returns true if any check failed.
pub fn has_failures(results: &[CheckResult]) -> bool {
    results.iter().any(|r| r.status == CheckStatus::Fail)
}

/* ----------------------------- Individual Checks ----------------------------- */

fn check_perf() -> CheckResult {
    match find_in_path("perf") {
        Some(path) => CheckResult {
            label: "perf".to_string(),
            status: CheckStatus::Ok,
            detail: format!("found at {}", path.display()),
        },
        None => CheckResult {
            label: "perf".to_string(),
            status: CheckStatus::Fail,
            detail: "not found in PATH; install linux-tools-common".to_string(),
        },
    }
}

fn check_perf_paranoid() -> CheckResult {
    match std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid") {
        Ok(contents) => {
            let val: i32 = contents.trim().parse().unwrap_or(4);
            if val <= 1 {
                CheckResult {
                    label: "perf_event_paranoid".to_string(),
                    status: CheckStatus::Ok,
                    detail: format!("value={val} (user-space profiling allowed)"),
                }
            } else {
                CheckResult {
                    label: "perf_event_paranoid".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!(
                        "value={val}; run 'sudo sysctl kernel.perf_event_paranoid=1' for user profiling"
                    ),
                }
            }
        }
        Err(_) => CheckResult {
            label: "perf_event_paranoid".to_string(),
            status: CheckStatus::Warn,
            detail: "cannot read /proc/sys/kernel/perf_event_paranoid".to_string(),
        },
    }
}

fn check_gperftools() -> CheckResult {
    // Try google-pprof first, then pprof
    if let Some(path) = find_in_path("google-pprof") {
        return CheckResult {
            label: "gperftools".to_string(),
            status: CheckStatus::Ok,
            detail: format!("google-pprof found at {}", path.display()),
        };
    }
    if let Some(path) = find_in_path("pprof") {
        return CheckResult {
            label: "gperftools".to_string(),
            status: CheckStatus::Ok,
            detail: format!("pprof found at {}", path.display()),
        };
    }
    CheckResult {
        label: "gperftools".to_string(),
        status: CheckStatus::Fail,
        detail: "neither google-pprof nor pprof found; install gperftools".to_string(),
    }
}

fn check_aslr() -> CheckResult {
    match std::fs::read_to_string("/proc/sys/kernel/randomize_va_space") {
        Ok(contents) => {
            let val: i32 = contents.trim().parse().unwrap_or(-1);
            if val == 0 {
                CheckResult {
                    label: "ASLR".to_string(),
                    status: CheckStatus::Ok,
                    detail: "disabled (randomize_va_space=0)".to_string(),
                }
            } else {
                CheckResult {
                    label: "ASLR".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!(
                        "enabled (value={val}); use 'setarch $(uname -m) -R' for consistent profiles"
                    ),
                }
            }
        }
        Err(_) => CheckResult {
            label: "ASLR".to_string(),
            status: CheckStatus::Warn,
            detail: "cannot read /proc/sys/kernel/randomize_va_space".to_string(),
        },
    }
}

fn check_flamegraph() -> CheckResult {
    // Check $FLAMEGRAPH_DIR first, then common paths
    let search_dirs: Vec<PathBuf> = {
        let mut dirs = Vec::new();
        if let Ok(val) = std::env::var("FLAMEGRAPH_DIR") {
            dirs.push(PathBuf::from(val));
        }
        if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(&home).join("FlameGraph"));
        }
        dirs.push(PathBuf::from("/usr/local/FlameGraph"));
        dirs.push(PathBuf::from("/opt/FlameGraph"));
        dirs
    };

    for dir in &search_dirs {
        let script = dir.join("flamegraph.pl");
        if script.is_file() {
            return CheckResult {
                label: "FlameGraph".to_string(),
                status: CheckStatus::Ok,
                detail: format!("found at {}", dir.display()),
            };
        }
    }

    // Also check PATH for flamegraph.pl
    if find_in_path("flamegraph.pl").is_some() {
        return CheckResult {
            label: "FlameGraph".to_string(),
            status: CheckStatus::Ok,
            detail: "flamegraph.pl found in PATH".to_string(),
        };
    }

    CheckResult {
        label: "FlameGraph".to_string(),
        status: CheckStatus::Warn,
        detail: "not found; set $FLAMEGRAPH_DIR or clone to ~/FlameGraph".to_string(),
    }
}

fn check_nsight() -> CheckResult {
    if let Some(path) = find_in_path("nsys") {
        return CheckResult {
            label: "Nsight Systems".to_string(),
            status: CheckStatus::Ok,
            detail: format!("nsys found at {}", path.display()),
        };
    }
    CheckResult {
        label: "Nsight Systems".to_string(),
        status: CheckStatus::Warn,
        detail: "nsys not found; install CUDA toolkit for GPU profiling".to_string(),
    }
}

fn check_bpftrace() -> CheckResult {
    match find_in_path("bpftrace") {
        Some(path) => CheckResult {
            label: "bpftrace".to_string(),
            status: CheckStatus::Ok,
            detail: format!("found at {}", path.display()),
        },
        None => CheckResult {
            label: "bpftrace".to_string(),
            status: CheckStatus::Warn,
            detail: "not found; install bpftrace for kernel-level tracing".to_string(),
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
        assert!(results.len() >= 7);
    }

    /// @test Each check has a non-empty label.
    #[test]
    fn all_checks_have_labels() {
        let results = run_checks();
        for r in &results {
            assert!(!r.label.is_empty());
            assert!(!r.detail.is_empty());
        }
    }
}
