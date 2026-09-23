# vernier::monitor -- Runtime Performance Monitor

**Namespace:** `vernier::monitor`
**Platform:** Linux-only
**C++ Standard:** C++20 (C++23 used when available)

Lightweight, lock-free instrumentation library for measuring runtime
behavior of long-running applications. Complements the `vernier::bench`
harness, which targets controlled benchmark loops; `vernier::monitor`
targets observability in real runs you can't repeat.

## At a glance

| Capability          | API                                                        |
| ------------------- | ---------------------------------------------------------- |
| Scoped timer        | `VERNIER_MONITOR_SCOPE(monitor, "name", tag)`              |
| Point-in-time gauge | `VERNIER_MONITOR_GAUGE(monitor, "name", tag, value)`       |
| Counter increment   | `VERNIER_MONITOR_INCREMENT(monitor, "name", tag[, delta])` |
| Threshold alert     | `monitor.setThreshold("name", tag.id, thresholdUs)`        |
| End-of-run summary  | `monitor.stop()` (also auto-called by the destructor)      |
| Disable             | `cfg.enabled = false`, `VERNIER_MONITOR=0`, `setEnabled()` |

Recording a sample takes no lock and no allocation: the scope guard reads
a steady clock on entry and exit and copies the scope name into a
fixed-size record, and the record is published into one slot of a bounded
MPMC ring buffer. A dedicated I/O thread drains that queue into the
configured sinks and into the in-memory summary. Disabling stops a sample
at the enabled check inside the recording call -- the queue is still
allocated at construction, the scope guard still reads the clock and
copies the name, and the arguments you pass to the macros are still
evaluated -- so "disabled" means "records nothing", not "costs nothing".
This guide quotes no per-sample time: the repository measures none.

## Lifecycle

The documented sequence is: configure, set thresholds, `start()`,
instrument, let the producers finish, `stop()`.

```cpp
vernier::monitor::Monitor monitor(cfg);
monitor.setThreshold("decode", decoder.id, 5000);  // before start()
monitor.start();                                   // sinks + I/O thread
// ... instrumented work on any number of threads ...
monitor.stop();                                    // drains, then reports
```

| Call                              | What it does                                                                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Construction                      | Allocates the queue. No thread, no sink, no output file.                                                                                    |
| `start()`                         | Freezes the thresholds, opens the configured sinks, starts the I/O thread. Idempotent.                                                      |
| `start()` while disabled          | Nothing: no thread, no sink, no output file, no summary at `stop()`, and `isRunning()` stays false.                                         |
| `setEnabled(true)` then `start()` | Starts a monitor that was disabled at its first `start()`.                                                                                  |
| `setEnabled(false)` while running | Later recording calls return at the enabled check. It is a switch, not a barrier: samples already queued are kept and reported.             |
| `stop()`                          | Writes every sample queued before it to the sinks and the summary, joins the I/O thread, then reports. Idempotent; the destructor calls it. |

`stop()` prints the summary table to stderr only when the console sink is
configured. Output selection does not affect measurement: with
`SINK_NONE`, or with the console sink off, samples are still collected
and `monitor.summary()` still carries the whole table.

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

## Configuration from the environment

`configFromEnv()` builds the same `MonitorConfig` from environment
variables, so one instrumented binary covers several deployments without
a rebuild:

```cpp
auto cfg = vernier::monitor::configFromEnv();
vernier::monitor::Monitor monitor(cfg);
```

The defaults are the struct's own: enabled, console sink at `INFO`, no
file, 4096-slot queue. Set the variables to change them.

| Var                                  | Effect                                                |
| ------------------------------------ | ----------------------------------------------------- |
| `VERNIER_MONITOR=0`                  | a false value disables; unset leaves it enabled       |
| `VERNIER_MONITOR_DISABLE=1`          | disable, whatever `VERNIER_MONITOR` says              |
| `VERNIER_MONITOR_FILE=/tmp/run.vmon` | add the file sink with this path                      |
| `VERNIER_MONITOR_CONSOLE=WARNING`    | console minimum level; `off` removes the console sink |
| `VERNIER_MONITOR_QUEUE=8192`         | ring-buffer capacity (rounded up to a power of two)   |

`VERNIER_MONITOR` and `VERNIER_MONITOR_DISABLE` treat `0`, `false`, `off`,
`no` and an empty value as false and any other value as true. The
comparison is exact, so `FALSE` and `Off` count as true.

Same code, different deployments:

```bash
# Default: per-sample lines and the summary table on stderr
./my_app

# Silent run: no console output, no summary, samples collected in memory
VERNIER_MONITOR_CONSOLE=off ./my_app

# Investigation: breaches and the summary on the console, all in the file
VERNIER_MONITOR_FILE=/tmp/issue.vmon \
VERNIER_MONITOR_CONSOLE=WARNING \
./my_app

# Records nothing at all
VERNIER_MONITOR_DISABLE=1 ./my_app
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
monitor.setThreshold("decode", decoder.id, 5000);  // before start()
```

`start()` freezes the thresholds into the table the recording path reads,
so a threshold set afterwards applies only to a later `start()`.

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

`stop()` writes it to stderr when the console sink is configured, after
the queue has been drained: `Total samples` counts every sample the
monitor accepted, `Dropped` counts what the ring buffer had no room for,
and the `Calls` column adds up to the difference. When the console sink
is off, the same table is still available through `monitor.summary()`.

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
