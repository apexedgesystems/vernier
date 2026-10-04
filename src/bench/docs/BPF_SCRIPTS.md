# bpftrace helper scripts (optional)

These scripts are **optional** companions for the gtest-based perf suite. They provide
kernel and syscall visibility during a perf run. Use them locally while investigating
performance, not by default in CI.

## Requirements

- Linux with eBPF support and `bpftrace` installed
- Root, to attach the scripts' tracepoints: bpftrace refuses every other
  effective user, whatever its capabilities ("bpftrace currently only supports
  running as the root user."; checked in the source of 0.14.0, 0.20.2 and
  0.23.2). The backend runs `bpftrace` as the current user, which works when
  the benchmark runs as root. Otherwise `BENCH_SUDO=1` runs it, and the `kill`
  that stops it, through `sudo -n`; the sudoers grant must allow `bpftrace`
  with the run's script arguments and `kill` with `-2`, `-15` and `-9`.
  `PERF_BPF_SUDO` is a deprecated alias honoured by the `bpftrace` backend
  only: a valid `BENCH_SUDO` wins when both are set, and an invalid alias is
  then ignored with a warning; an invalid `BENCH_SUDO`, or an invalid alias on
  its own, is a configuration error.
- `--profile-check --profile bpftrace --bpf <scripts>` runs a copy of each
  selected script, with the capture window (below, bound to no thread, so it
  never arms), a line it prints once bpftrace has attached it and a 5 s
  self-exit added, through that route until the line shows, a second at
  least, then stops it and reports what failed; a run makes the same decision
  before its first case. How the copy ended decides: one that ends by itself
  before the stop is refused, one that ends with an error, before the stop or
  after it, is reported as a copy that fails at its start, and one still
  starting 5 s after its start (on a busy machine) is stopped and reported as
  `unverified`. The copy lives in a private
  temporary directory, while the run's own copy lives in its capture folder, so
  the check cannot try the run's exact command: with `BENCH_SUDO=1`, a grant
  that refuses the check's copy is reported as `unverified` rather than denied,
  and the run's start shows whether the grant allows the run's command. A copy that cannot be stopped
  (the grant refuses `kill`) ends by its self-exit, and the check waits for it
  rather than leave it running. A ready check lists what it did not check.

## Selecting scripts

`--bpf` takes a comma-separated list. A name without a `/` is a script in the
scripts directory (`<name>.bt` there); a name with a `/` is a path to a script
file, absolute or from the working directory, not from the scripts directory:
`--bpf sub/x` names `./sub/x.bt`. Either form may leave out the `.bt` suffix,
so `--bpf write_latency`, `--bpf write_latency.bt`, `--bpf ./my_script.bt` and
`--bpf /path/to/my_script` all work. Without `--bpf`, the backend runs
`write_latency` and `fsync_latency`.

## PID filtering

Scripts contain the placeholder `{{PID}}`. The bpftrace backend (`--profile bpftrace`)
replaces it with the current test process PID and writes a temporary script before
execution. This confines tracing to the test process to reduce noise.

## What a run writes

Each test's capture folder (`<Suite.Case>.bpf/` under the working directory, or
under `--profile-output-dir`) holds three files per script, named after the
script file without its `.bt`, wherever the script came from: `<name>.tmp.bt`,
the copy bpftrace ran, with the PID filled in and the capture window appended;
`<name>.out.text` (or `<name>.out.json` with `PERF_BPF_FMT=json`), what bpftrace
printed: the capture window's two lines, then the maps at its exit; and
`<name>.err.txt`, its error output. The files stay whatever the capture's
outcome. A request refused before the run (no privileges, a script not found,
a probe the kernel lacks) starts no tracer and writes no capture folder. Two
selected scripts whose file names match would overwrite each other's files, so
such a request is refused before anything runs.

## The capture window

A run measures only while its tracers show that they see the benchmark. Each
run's copy of a script ends with one program the backend appends; for a
benchmark whose PID is 4242, whose arm thread is 4250 and whose measured
repeats run on its main thread:

```text
// Added by Vernier: the capture window (BPF_SCRIPTS.md). The measured
// repeats run between its arm line and its stop line.
tracepoint:sched:sched_switch {
  if (pid == 4242 && (args->prev_state == 1 || args->prev_state == 2)) {
    if (tid == 4250 && comm == "vernier-arm") {
      if (@vernier_window == 0) {
        @vernier_window = 1;
        printf("bpftrace armed %d %d\n", pid, tid);
      }
    } else if (tid == 4242 && comm == "vernier-stop") {
      if (@vernier_window == 1) {
        @vernier_window = 2;
        printf("bpftrace disarmed %d %d\n", pid, tid);
        clear(@vernier_window);
      }
    }
  }
}
```

Before it launches the tracers, the backend starts a thread named `vernier-arm`
that sleeps 5 ms at a time; the program names that thread and the thread that
runs the test by their ids. Once every selected script's tracer has started,
the thread that runs the test takes the name `vernier-wait` and waits, up to
5 s, until each tracer has printed `bpftrace armed <pid> <tid>` with the
benchmark's PID and the arm thread's id; the measured repeats start then.
After them it takes the name `vernier-stop` and waits, up to 3 s, for each
tracer's `bpftrace disarmed <pid> <tid>` with its own id; the tracers are then
stopped with SIGINT. bpftrace runs with `-B none`, so each line reaches the
report as it is printed. bpftrace clears a map only when its own process
handles the `clear()`, a moment after the program ran, so the disarm also sets
`@vernier_window` to 2 at once: a later switch-out does not print the line
again. Both lines stay in the report, before what bpftrace prints at its exit.

bpftrace prints what its programs emit only once it has attached every probe
of the script: 0.14.0, 0.20.2 and 0.23.2 read their programs' output after
both of their attach passes. The arm line therefore shows that the script's
probes are attached and that the tracer sees the benchmark's threads under the
ids the backend knows. It does not confine the script's own probes to the
measured repeats: they record from their attach until the tracer stops, which
includes the waits before and after the measured repeats.

The capture is reported as failed, with a `[bpftrace]` line, when a tracer ends
before its arm or its stop line, prints neither within its bound, prints either
with other ids or out of order (a stop line before the arm, or before the stop
was asked for), cannot be stopped, or leaves no output, an empty one or one
without the two lines, and when the measured repeats end on another thread
than the one they started on. A report that holds nothing but the two lines is
a complete capture whose script printed no data of its own: the run says so
with a `[bpftrace]` line, and the capture counts as a caveat, not as a zero.

The thread names `vernier-arm`, `vernier-wait` and `vernier-stop` are reserved
for the backend's own threads: a benchmark thread must not take them. A script
must not use a map whose name starts with `@vernier_`, nor print
`bpftrace armed` or `bpftrace disarmed`: the check refuses such a script
before anything runs. A script with an iterator probe (`iter:`) cannot take
the program, since bpftrace runs an iterator probe only as a script's single
probe; the check reports it unsupported, and such a script runs by hand. The
program needs the `sched:sched_switch` tracepoint; where bpftrace refuses it,
the check says so.

## Scripts

The scripts are in `src/bench/bpf/`. `--profile bpftrace` finds them there from
any working directory: the library records the directory's absolute path when
it is built. `--bpf-scripts DIR` selects another directory; it sets
`PERF_BPF_SCRIPTS` in the benchmark's environment, over a value the benchmark
inherited, and the variable alone does the same (an empty value selects the
bundled directory). A library installed after its source tree was removed
reports the path it looked in; point `--bpf-scripts` at a copy of the scripts.

- `write_latency.bt`: histogram of `write()` latency (us) for the target PID
- `fsync_latency.bt`: histogram of `fsync()` latency (us) for the target PID
- `wakeup_latency.bt`: histogram of wakeup latency (us), from a wake request to
  the woken thread being switched in, for the wakeups the target process makes
  (one of its threads handing a lock to another, say) and wakeups of its main
  thread. The interval includes the wakeup's trip to the woken thread's CPU and
  any wait for that CPU: under a lock it is not the time a thread waited for the
  lock.
- `cpu_migrations.bt`: CPU migrations of the target PID's main thread, counted by
  destination CPU

`write_latency.bt` and `fsync_latency.bt` use the `syscalls` tracepoints, which some
vendor kernels leave out; `wakeup_latency.bt` and `cpu_migrations.bt` use only
`sched` tracepoints.

Each script ends itself when the traced process's main thread exits
(`sched_process_exit` filtered on `tid == {{PID}}`), so a trace of a threaded test
lasts through its workers' exits; the backend stops it with SIGINT once the
measured repeats finish. A tracer that ends before then, by its own `exit()`, is
reported, and the capture counts as incomplete; one that ends before the check
stops its copy, once the copy has attached and a second after its start at the
earliest, is refused before the run, as a script that ends itself too soon to
be checked.

Run manually: replace every `{{PID}}` with the process to trace, then run the
copy. For a benchmark already running (PID 1234 here), the trace starts once
bpftrace has attached and ends at Ctrl-C or when the benchmark's main thread
exits; the maps print then.

```bash
sed 's/{{PID}}/1234/g' src/bench/bpf/write_latency.bt > /tmp/write_latency.bt
sudo bpftrace -q /tmp/write_latency.bt
```

Or let bpftrace start the benchmark with `-c`, and use its `cpid`, the PID of
the command it starts: the trace then covers the whole process, warm-up and
test framework included, and ends with it; the benchmark runs as root, as
bpftrace does.

```bash
sed 's/{{PID}}/cpid/g' src/bench/bpf/write_latency.bt > /tmp/write_latency.bt
sudo bpftrace -q -c './build/bin/ptests/<Binary> --gtest_filter=<Suite.Case>' /tmp/write_latency.bt
```

Neither has the capture window: nothing marks the measured repeats, so the maps
hold everything the probes saw while bpftrace ran.

## PID namespaces

The scripts are supported where the benchmark runs in the host's PID namespace:
natively, or in a container started with `--pid=host`. There every id a script
compares is the same number. In a PID namespace of its own (a container without
`--pid=host`, or `unshare --pid`), `{{PID}}` is the benchmark's PID as that
namespace numbers it, and no bundled script sees the benchmark as it should:

- bpftrace 0.20 to 0.22 (0.20.2 is the project's dev image) numbers `pid` and
  `tid` as the host does, so neither matches `{{PID}}`.
- bpftrace 0.23.0 to 0.24.1 (0.23.2 is the reference rig's) numbers them in its
  own namespace, but swapped: `pid` is the thread's id and `tid` the process's.
  `pid == {{PID}}` then matches the main thread alone, so `write_latency.bt` and
  `fsync_latency.bt` miss every other thread's calls, and `tid == {{PID}}`
  matches every thread, so each script ends at the first thread's exit.
- A tracepoint's own fields (`args->pid`, `args->next_pid`) always hold the
  host's numbers, so `cpu_migrations.bt` records nothing there, and
  `wakeup_latency.bt` keeps only the wakeups the main thread makes.

With either of those versions the capture window's arm line never comes there,
so a run in such a namespace reports its capture as failed and names the
namespace. Run such a benchmark on the host, or in a container started with
`--pid=host`.
