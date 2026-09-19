# Reference Rigs

**Location:** `src/bench/docs/rigs/`
**Platform:** Linux (aarch64 boards)

A reference rig is a named machine, set up a stated way, on which a demo
walkthrough's commands were run and its output captured. Every walkthrough
names its rig and links here instead of repeating the setup.

---

## Table of Contents

1. [Why Rigs](#1-why-rigs)
2. [The Rigs](#2-the-rigs)
3. [What Reproduces and What Does Not](#3-what-reproduces-and-what-does-not)
4. [Verify Your Rig](#4-verify-your-rig)
5. [Using Another Machine](#5-using-another-machine)

---

## 1. Why Rigs

A benchmark number means nothing without the machine, the build type and
the conditions it came from. "About 5x faster" cannot be checked. "4.0x on
a Raspberry Pi 4, Release build, governor pinned, one pinned core, captured
with Vernier 1.0.4" can.

Each rig document states the hardware, the operating system and tool
versions, the one-time setup, the build command, and the conditions a
measurement runs under. If you own the same board and follow the rig
document, the walkthrough's commands should give you nearly the same
output.

## 2. The Rigs

| Rig                                  | Document                           | Used by                                                         |
| ------------------------------------ | ---------------------------------- | --------------------------------------------------------------- |
| Raspberry Pi 4 Model B               | [RIG_PI4.md](RIG_PI4.md)           | CPU walkthroughs                                                |
| NVIDIA Jetson AGX Thor Developer Kit | [RIG_THOR_AGX.md](RIG_THOR_AGX.md) | GPU walkthroughs, and CPU walkthroughs that need a GPU timeline |

Three walkthroughs fall outside the two rigs, and each says so at its top:
RAPL needs an Intel CPU; jemalloc needs a jemalloc built with profiling,
which neither rig's distribution provides; rocprof needs an AMD GPU.

## 3. What Reproduces and What Does Not

On the same rig, expect:

- **Ratios** between the slow and fast variants to match closely.
- **The profiler's finding** to match: the same hot function, the same
  allocation count, the same kernel launch shape.
- **Absolute times** to land within a few percent.

On a different machine, expect the ratios and the findings to hold in
direction and rough size, and the absolute times to differ. Each
walkthrough states which of its numbers depend on the rig, for example
host-to-device transfer times on a board where the CPU and GPU share
memory.

## 4. Verify Your Rig

Each rig document ends with a `bench doctor` command and its expected
output. The doctor probes every profiler backend on the machine in front
of you and prints what is missing and the command that fixes it.

Treat a matching doctor output as a prerequisite, not a guarantee. The
doctor checks backend-specific prerequisites, such as tool availability
and permissions; it does not check that the tool can see what a
walkthrough needs it to see. The known case
is heaptrack in a build that links tcmalloc: the doctor reports heaptrack
`[OK]`, and heaptrack misses C++ allocations. Each rig document lists such
limits under its rig-specific behavior.

## 5. Using Another Machine

Nothing in Vernier requires a rig. To run a walkthrough elsewhere:

1. Build Release (the walkthroughs assume it; optimization-dependent
   effects do not appear in a Debug build).
2. Run `bench doctor <test-binary>` and fix what the walkthrough's
   profiler needs.
3. Hold the machine still: pin the test to one core with `taskset`, fix
   the CPU frequency if the platform lets you, and keep other work off the
   machine.
4. Read the result's CV before its median. A comparison only means
   something when the difference is far larger than the run-to-run spread.
