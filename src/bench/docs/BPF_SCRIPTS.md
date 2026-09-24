# bpftrace helper scripts (optional)

These scripts are **optional** companions for the gtest-based perf suite. They provide
kernel and syscall visibility during a perf run. Use them locally while investigating
performance, not by default in CI.

## Requirements

- Linux with eBPF support and `bpftrace` installed
- Sufficient privileges (root or `sudo -n` available)

## PID filtering

Scripts contain the placeholder `{{PID}}`. The bpftrace backend (`--profile bpftrace`)
replaces it with the current test process PID and writes a temporary script before
execution. This confines tracing to the test process to reduce noise.

## Scripts

The scripts are in `src/bench/bpf/`:

- `write_latency.bt`: histogram of `write()` latency (us) for the target PID
- `fsync_latency.bt`: histogram of `fsync()` latency (us) for the target PID
- `wakeup_latency.bt`: histogram of scheduler wakeup latency (us), from a thread
  being woken to its getting a CPU, for wakeups of the target PID's main thread
  and wakeups made by the target process
- `cpu_migrations.bt`: CPU migrations of the target PID's main thread, counted by
  destination CPU

`write_latency.bt` and `fsync_latency.bt` use the `syscalls` tracepoints, which some
vendor kernels leave out; `wakeup_latency.bt` and `cpu_migrations.bt` use only
`sched` tracepoints.

Run manually (example): replace `{{PID}}` with the PID to trace (1234 here), then run
the copy. The histogram prints when bpftrace exits (Ctrl-C).

```bash
sed 's/{{PID}}/1234/' src/bench/bpf/write_latency.bt > /tmp/write_latency.bt
sudo bpftrace -q /tmp/write_latency.bt
```

The bpftrace in the project's dev image (0.20.2) matches the PID as the host sees it:
inside a container with its own PID namespace, the PID the container reports is a
different number and nothing matches, so run such a container with `--pid=host`.
