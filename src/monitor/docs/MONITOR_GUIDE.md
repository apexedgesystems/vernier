# vernier::monitor -- Runtime Performance Monitor

**Namespace:** `vernier::monitor`
**Platform:** Linux-only
**C++ Standard:** C++20 (C++23 used when available)

Lightweight, lock-free instrumentation library for measuring runtime
behavior of long-running applications. Complements the `vernier::bench`
harness, which targets controlled benchmark loops; `vernier::monitor`
targets observability in real runs you can't repeat.

## At a glance

| Capability            | API                                                        |
| --------------------- | ---------------------------------------------------------- |
| Scoped timer          | `VERNIER_MONITOR_SCOPE(monitor, "name", tag)`              |
| Point-in-time gauge   | `VERNIER_MONITOR_GAUGE(monitor, "name", tag, value)`       |
| Counter increment     | `VERNIER_MONITOR_INCREMENT(monitor, "name", tag[, delta])` |
| Threshold alert       | `monitor.setThreshold("name", tag.id, thresholdUs)`        |
| End-of-run summary    | `monitor.stop()` (also auto-called by the destructor)      |
| Zero-overhead disable | `cfg.enabled = false` or `VERNIER_MONITOR=0`               |

The hot path is lock-free and bounded: scope entry/exit cost is a clock
read plus a single MPMC queue write (~100-200ns). A dedicated I/O
thread drains the queue and feeds the configured sinks.

## Construction

```cpp
#include "src/monitor/inc/Monitor.hpp"
#include "src/monitor/inc/MonitorConfig.hpp"

vernier::monitor::MonitorConfig cfg;
cfg.queueCapacity = 8192;
cfg.sinks         = vernier::monitor::SINK_CONSOLE | vernier::monitor::SINK_FILE;
cfg.filePath      = "/tmp/run.vmon";
cfg.consoleLevel  = vernier::monitor::AlertLevel::WARNING;

vernier::monitor::Monitor monitor(cfg);
const vernier::monitor::MonitorTag decoder{"decoder", 1};
```

## Zero-code-change enablement via env vars

A binary instrumented with the `VERNIER_MONITOR_*` macros stays silent
until the operator sets the relevant env var. Build a config from the
environment instead of hard-coding it:

```cpp
auto cfg = vernier::monitor::configFromEnv();
vernier::monitor::Monitor monitor(cfg);
```

Recognized env vars:

| Var                                  | Effect                                               |
| ------------------------------------ | ---------------------------------------------------- |
| `VERNIER_MONITOR=1`                  | enable; default behavior is enabled-when-constructed |
| `VERNIER_MONITOR_DISABLE=1`          | hard-disable (overrides `VERNIER_MONITOR`)           |
| `VERNIER_MONITOR_FILE=/tmp/run.vmon` | enable file sink with this path                      |
| `VERNIER_MONITOR_CONSOLE=WARNING`    | console sink at this min level (or `off`)            |
| `VERNIER_MONITOR_QUEUE=8192`         | ring-buffer capacity (rounded up to pow2)            |

Same code, different deployments:

```bash
# Production: quiet
./my_app

# Investigation: warnings to console, all samples to file
VERNIER_MONITOR_FILE=/tmp/issue.vmon \
VERNIER_MONITOR_CONSOLE=WARNING \
./my_app
```

## Instrumentation patterns

### Scoped timer (RAII)

```cpp
void processFrame() {
  VERNIER_MONITOR_SCOPE(monitor, "process_frame", decoder);
  // ... work ...
}  // duration recorded on scope exit
```

### Multi-phase scope

```cpp
void processFrame() {
  {
    VERNIER_MONITOR_SCOPE(monitor, "decode", decoder);
    decodeStream();
  }
  {
    VERNIER_MONITOR_SCOPE(monitor, "render", decoder);
    renderFrame();
  }
}
```

### Counters and gauges

```cpp
// Monotonic counter (e.g. frames produced)
VERNIER_MONITOR_INCREMENT(monitor, "frames", decoder);

// Counter with custom delta
VERNIER_MONITOR_INCREMENT(monitor, "bytes_written", io, n);

// Point-in-time gauge (e.g. current queue depth)
VERNIER_MONITOR_GAUGE(monitor, "queue_depth", decoder, queue.size());
```

### Threshold alerts

```cpp
// Warn (and flag in summary) if "decode" ever exceeds 5 ms.
// setThreshold takes the numeric tag id, not the full MonitorTag.
monitor.setThreshold("decode", decoder.id, 5000);
```

When a scope exceeds its threshold, the sample is flagged as
`THRESHOLD_BREACH` and -- if `consoleLevel <= WARNING` -- a line is
emitted immediately so an operator can react in real time.

## Output

The end-of-run summary table groups samples by tag + scope:

```
vernier::monitor summary
--------------------------------------------------------------------------
 Tag           Scope              Calls   Median     P99      Max     Breaches
 decoder/1     process_frame      10421   1.23 ms   4.87 ms  12.1 ms   3
 decoder/1     decode              5210   0.61 ms   2.10 ms   8.4 ms   1
 decoder/1     render              5210   0.55 ms   1.91 ms   3.4 ms   0
 io/2          bytes_written (c)  10421   -         -         9043712  -
--------------------------------------------------------------------------
 Total samples: 31263 | Dropped: 0 | Wall time: 62.3 s
```

The file sink writes a tab-delimited record per sample, suitable for
post-run analysis with awk / pandas / `bench` Python tools.

## When to reach for monitor vs. bench

- **vernier::bench** -- you control the iteration count; you want a
  statistical distribution of N repeats. Best for A/B comparisons and
  CI regression gates.
- **vernier::monitor** -- you can't replay; you want to know what
  actually happened in this one run. Best for production deployments
  and intermittent issues.

The two compose: a `bench` test can install a monitor to record its
internal phases for the end-of-run summary.

## See also

- [`src/monitor/examples/`](../examples/) -- end-to-end usage examples
- [`src/bench/docs/CPU_GUIDE.md`](../../bench/docs/CPU_GUIDE.md) -- the
  controlled-benchmark counterpart
