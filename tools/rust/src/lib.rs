//! Vernier Rust Tools Library
//!
//! This library hosts the code of the vernier-rust-tools executable, `bench`:
//! benchmark CSV analysis, comparison and validation; profile runs
//! (`run`, `profile-all`, `profile-summarize`, `nsight-parse`); the doctor;
//! the GPU commands (`gpu-env`, `gpu-lock`, `gpu-monitor`, `gpu-topo`);
//! `init`, `config-validate` and `flamegraph`. `bench --help` lists them.
//!
//! Shared code lives under `vernier_rust_tools::<module>`.

/// Write a line to stderr as `eprintln!` does, and let a write that fails
/// go: a line stderr refuses (a pipe whose reader is gone, a terminal that
/// hung up) is lost, where `eprintln!` would panic and end bench with
/// status 101. Whether a line can be written never changes what a command
/// does or the status it ends with.
#[macro_export]
macro_rules! lossy_eprintln {
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let _ = ::std::writeln!(::std::io::stderr(), $($arg)*);
    }};
}

pub mod bench;
