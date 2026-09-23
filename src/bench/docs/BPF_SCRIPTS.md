# bpftrace helper scripts (optional)

These scripts are **optional** companions for the gtest-based perf suite. They provide
kernel and syscall visibility during a perf run. Use them locally while investigating
performance, not by default in CI.

## Requirements

- Linux with eBPF support and `bpftrace` installed
- Privileges to attach the scripts' tracepoints. `bpftrace` runs as the current
  user unless you opt in, which works as root or with `CAP_BPF` and
  `CAP_PERFMON`. `BENCH_SUDO=1` runs it, and the `kill` that stops it, through
  `sudo -n`; the sudoers grant must allow `bpftrace` with the run's script
  arguments and `kill` with `-2`, `-15` and `-9`. `PERF_BPF_SUDO` is a
  deprecated alias honoured by the `bpftrace` backend only: `BENCH_SUDO` wins
  when both are set, and an invalid value of either is a configuration error.
- `--profile-check --profile bpftrace --bpf <scripts>` attaches each selected
  script through that route for a second and stops it, and reports what failed;
  a run makes the same decision before its first case.

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
