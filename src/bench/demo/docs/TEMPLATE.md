# Demo NN: <Tool or Technique>

<!--
Template for a demo walkthrough. Copy it, fill every section from a real
run on the reference rig, and delete these comments. Rules:

- Every command was run as written, on the rig named below, from a
  Release build. Every output block is pasted from that run, trimmed only
  where marked with "...".
- Never state a number you did not capture. Round in prose, not in the
  pasted output.
- State which numbers depend on the rig.
- The test must assert the effect it demonstrates, so the walkthrough
  fails loudly when it stops being true, and something must run it: on
  the rig before every release, and on the rig's CI lane where one exists.
- Give `bench run` the same explicit path Step 1 runs
  (`./build/bin/ptests/<Binary>`). A bare binary name is resolved under
  `build/*/bin/ptests`, which the rig layout (`-B build`) does not match.
-->

**Reference rig:** [<rig name>](../../docs/rigs/RIG_<NAME>.md)
**Build:** Release
**Example:** `<shared example>` (see [Shared Workloads](../README.md#5-shared-workloads))
**Captured:** <YYYY-MM-DD>, Vernier <version>

## Overview

Two or three sentences: the question this tool answers, and what the
example will show.

## What is <tool>?

What it is, how it works in one sentence, its overhead, and what it is
not for. Then:

**In Vernier:** the flag, what gets written, and where.

**Needs:** installed packages and privileges. Link the rig's setup
section instead of repeating it.

## The Example

The function under test, in full if it is short, and the performance test
that drives it. Say what the slow and fast variants differ in.

## Step 1: Measure

```bash
taskset -c <core> ./build/bin/ptests/<Binary> --repeats 10 --csv out.csv
```

Captured output:

```
<result lines, pasted>
```

What to read: the median, then the CV. State the ratio between the
variants.

## Step 2: Profile

```bash
bench run ./build/bin/ptests/<Binary> --profile <tool> -- --gtest_filter='<Test>'
```

Where the report lands, and the command that reads it. Captured output:

```
<the tool's own report, trimmed to the lines that matter>
```

## Step 3: Read the Report

Point at the lines that carry the finding. Name the columns. Say what the
reader should conclude and what they should not.

## Step 4: Confirm the Fix

Run the same commands on the fast variant. Show the same report. The
point of the walkthrough is that the tool's reading changes the way the
timing did.

## What Should Reproduce

| Reading                  | On this rig | Elsewhere                                                         |
| ------------------------ | ----------- | ----------------------------------------------------------------- |
| slow / fast ratio        | <n>x        | same direction; size depends on <cache, allocator, memory system> |
| <the profiler's finding> | <value>     | should match                                                      |
| absolute times           | <values>    | will differ                                                       |

## If It Does Not Match

The two or three failures a reader is most likely to hit on this tool,
each with the doctor line that reveals it and the fix.

## Check Against the Reference

```bash
bench compare <reference csv for this walkthrough> out.csv
```

## See Also

Related walkthroughs and the guide section for this backend.
