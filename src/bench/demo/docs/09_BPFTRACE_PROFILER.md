# Demo 09: bpftrace Syscall Overhead Analysis

## Overview

Demonstrates using bpftrace to identify syscall overhead from inefficient I/O
patterns. Shows how per-byte write() calls create massive kernel context-switch
overhead and how batching eliminates it. This is a common anti-pattern in
embedded telemetry, logging, and serial communication code.

## Prerequisites

```bash
make compose-debug
make tools-rust
# Requires root/sudo for bpftrace
# bpftrace is pre-installed in the dev-cuda container
```

## Step 1: Baseline Measurement

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_09_BpftraceProfiler --quick \
    --csv /tmp/bpf_demo.csv
  ./bin/tools/rust/bench summary /tmp/bpf_demo.csv
'
```

Expected output:

```
BpftraceProfiler.ManySmallWrites     ~800 us/call    ~1.3K calls/s
BpftraceProfiler.SingleBatchedWrite    ~1 us/call    ~1.0M calls/s
```

The difference is extreme because each "small writes" iteration makes 1024
individual write() syscalls while the batched version makes exactly 1.

## Step 2: Profile with bpftrace

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  sudo ./bin/ptests/BenchDemo_09_BpftraceProfiler --profile bpftrace \
    --bpf syslat --gtest_filter="*ManySmallWrites*" \
    --cycles 500 --repeats 1
'
```

The `syslat` script attaches to the write() syscall and produces a latency
histogram showing how much time the process spends in kernel transitions.

## Step 3: Diagnose

### bpftrace syscall latency output (ManySmallWrites)

```
@syscall_lat_ns[write]:
[512, 1K)            128 |@@                                |
[1K, 2K)            4096 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[2K, 4K)             820 |@@@@@@                            |
[4K, 8K)              12 |                                  |
```

The histogram shows that most write() calls take 1-2 us each. With 1024
calls per iteration, that is 1-2 ms per iteration -- nearly all of which is
kernel context-switch overhead, not actual data movement.

### bpftrace syscall latency output (SingleBatchedWrite)

```
@syscall_lat_ns[write]:
[512, 1K)            500 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1K, 2K)               3 |                                  |
```

One write() call per iteration. The single 1 KB write takes the same ~1 us
as each 1-byte write. The kernel processes the buffer in one pass.

## Step 4: Compare

```bash
docker compose run --rm -T dev-cuda bash -c '
  cd build/native-linux-debug
  ./bin/ptests/BenchDemo_09_BpftraceProfiler --quick \
    --gtest_filter="*ManySmallWrites*" --csv /tmp/bpf_slow.csv
  ./bin/ptests/BenchDemo_09_BpftraceProfiler --quick \
    --gtest_filter="*SingleBatchedWrite*" --csv /tmp/bpf_fast.csv
  ./bin/tools/rust/bench compare /tmp/bpf_slow.csv /tmp/bpf_fast.csv
'
```

Expected output:

```
   Test                                  Baseline       Candidate      Change
-- ----                                  --------       ---------      ------
   BpftraceProfiler.ManySmallWrites       ~800 us         ~1 us       ~800x faster
   BpftraceProfiler.SingleBatchedWrite      --             --          --
```

## Why This Happens

| Metric            | Many Small Writes         | Batched Write |
| ----------------- | ------------------------- | ------------- |
| Payload           | 1024 bytes                | 1024 bytes    |
| Syscalls          | 1024                      | 1             |
| Context switches  | 2048 (user->kernel->user) | 2             |
| Per-call overhead | ~1-2 us                   | ~1-2 us       |
| Total overhead    | ~1-2 ms                   | ~1-2 us       |
| Actual data work  | ~0 (to /dev/null)         | ~0            |

Each write() syscall requires:

1. Save user-mode registers
2. Switch to kernel mode
3. Validate file descriptor
4. Copy from user buffer (even 1 byte)
5. Return to user mode
6. Restore registers

The data transfer itself is trivial. The overhead is entirely in the kernel
transition, which is a fixed cost per syscall regardless of payload size.

## Real-World Applications

- **Serial ports**: Writing one byte at a time to UART wastes 90%+ CPU
- **Logging**: Formatting + flushing each log line individually
- **Telemetry**: Sending one sensor reading per packet
- **File I/O**: Writing struct fields one at a time instead of the whole struct

## Key Takeaways

- bpftrace attaches to live kernel tracepoints (requires root)
- Syscall overhead is a fixed cost (~1-2 us per call on modern x86)
- Batching I/O operations is the single highest-leverage optimization for I/O
- 1024 one-byte writes cost ~1000x more than one 1024-byte write
- The `syslat` script produces latency histograms per syscall type
- Always buffer writes and flush periodically, not per-byte or per-field

## Further Reading

- `docs/CPU_GUIDE.md` -- bpftrace profiling section
- Demo 02 (perf) -- Use hardware counters to measure CPU overhead per syscall
