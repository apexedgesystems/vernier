# bpftrace helper scripts (optional)

These scripts are **optional** companions for the gtest-based perf suite. They provide
kernel and syscall visibility during a perf run. Use them locally while investigating
performance, not by default in CI.

## Requirements

- Linux with eBPF support and `bpftrace` installed
- Sufficient privileges (root or `sudo -n` available)

## PID filtering

Scripts contain the placeholder `{{PID}}`. The C++ `BpfRunner` replaces it with the
current test process PID and writes a temporary script before execution. This confines
tracing to the test process to reduce noise.

## Scripts

- `write_latency.bt`: histogram of `write()` latency (us) for the target PID
- `fsync_latency.bt`: histogram of `fsync()`/`fdatasync()` latency (us) for the PID

Run manually (example):

```bash
sudo bpftrace -q vernier/bpf/write_latency.bt | cat
```

(Replace `{{PID}}` with a number first if running manually.)
