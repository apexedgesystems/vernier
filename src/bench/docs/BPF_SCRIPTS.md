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
- `--profile-check --profile bpftrace --bpf <scripts>` runs a copy of each
  selected script, with a 5 s self-exit added, through that route for a second
  and stops it, and reports what failed; a run makes the same decision before
  its first case. The copy lives in a private temporary directory, while the
  run's own copy lives in its capture folder, so the check cannot try the run's
  exact command: with `BENCH_SUDO=1`, a grant that refuses the check's copy is
  reported as `unverified` rather than denied, and the run's start shows
  whether the grant allows the run's command. A copy that cannot be stopped
  (the grant refuses `kill`) ends by its self-exit, and the check waits for it
  rather than leave it running. A ready check lists what it did not check.

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
