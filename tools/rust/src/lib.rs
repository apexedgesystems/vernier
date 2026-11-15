//! Vernier Rust Tools Library
//!
//! This library hosts the code of the vernier-rust-tools executable, `bench`:
//! benchmark CSV analysis, comparison and validation; profile runs
//! (`run`, `profile-all`, `profile-summarize`, `nsight-parse`); the doctor;
//! the GPU commands (`gpu-env`, `gpu-lock`, `gpu-monitor`, `gpu-topo`);
//! `init`, `config-validate` and `flamegraph`. `bench --help` lists them.
//!
//! Shared code lives under `vernier_rust_tools::<module>`.

pub mod bench;
