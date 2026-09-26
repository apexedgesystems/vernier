# CI/CD Integration Guide

Fail a pull request that makes a benchmark slower: build the change and the
commit it is based on in the same job, run both on the same machine, and
compare the two runs with `bench compare --fail-on-regression`. One script does
the work; each CI system's file checks out the history, gives the script the
base commit, and keeps its report whether the gate passes or fails.

---

## Table of Contents

- [Overview](#overview)
- [The Gate Script](#the-gate-script)
- [GitHub Actions](#github-actions)
- [GitLab CI](#gitlab-ci)
- [Azure Pipelines](#azure-pipelines)
- [Jenkins](#jenkins)
- [Local Pre-Push Hook](#local-pre-push-hook)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### How It Works

```
pull request or merge request
  |
  +--> check out with full history (the base commit must be in the clone)
  |
  +--> ci/bench-gate.sh
  |      reset:   this run's files in bench-report/ ("did not finish")
  |      build:   this checkout, with vernier's tools, and the base
  |               commit in a separate worktree (a failed build fails)
  |      run:     the benchmark of each build, one after the other
  |                                     --> baseline.csv, candidate.csv
  |      report:  bench compare         --> report.md (result first),
  |                                         report.json
  |      status:  bench compare --fail-on-regression (1 on a regression)
  |
  +--> keep bench-report/ (job summary, artifacts), pass or fail
```

The baseline is built and run in the same job as the candidate, so both runs
come from the same machine, one right after the other once both builds are
done; a baseline CSV saved by an earlier job may come from another machine.

### What the Job Needs

- A project laid out as in the guides: its `CMakeLists.txt` fetches vernier
  (for example with `FetchContent`) and defines the benchmark,
  `MyComponent_PTEST` here, at its top level, so it builds to `build/`
- CMake 3.24 or newer, a C++20 compiler, git, and a Rust toolchain: the script
  builds vernier's `bench` CLI (`-DVERNIER_BUILD_TOOLS=ON`) to compare the runs.
  Ubuntu 24.04's packaged `cargo` (1.75) cannot read the CLI's `Cargo.lock`; a
  toolchain from rustup can
- The base commit in the clone: `actions/checkout` fetches one commit unless
  told otherwise, new GitLab projects fetch 20, and new Azure pipelines may
  fetch one, so each example below asks for the full history

---

## The Gate Script

**ci/bench-gate.sh:**

```bash
#!/usr/bin/env bash
# Build and run a benchmark at the base commit and at this checkout, compare
# the two runs, and exit 1 when bench compare reports a regression.
#
#   BENCH_BASE       commit to compare against (required unless bootstrapping)
#   BENCH_TARGET     benchmark executable (default: MyComponent_PTEST)
#   BENCH_THRESHOLD  regression threshold in percent (default: 5)
#   BENCH_ARGS       extra benchmark arguments for both runs, e.g. "--quick"
#   BENCH_OUT        report directory (default: bench-report)
#   BENCH_BOOTSTRAP  1 for the change that adds the benchmark: run only the
#                    candidate, compare nothing, and pass
set -euo pipefail

target=${BENCH_TARGET:-MyComponent_PTEST}
threshold=${BENCH_THRESHOLD:-5}
read -r -a args <<<"${BENCH_ARGS:-}"
mkdir -p "${BENCH_OUT:-bench-report}"
out=$(cd "${BENCH_OUT:-bench-report}" && pwd)

# This run's report only: remove what an earlier run wrote, and nothing else
rm -f "$out"/{report.md,report.json,candidate.csv,baseline.csv,baseline-build.log}
echo "## Benchmark gate: did not finish" >"$out/report.md"
stage="setup" work="" reported=""
finish() {
  local status=$?
  if [ -n "$work" ]; then
    git worktree remove --force "$work/src" >/dev/null 2>&1 || true
    rm -rf "$work"
  fi
  if [ -z "$reported" ] && [ "$status" -ne 0 ]; then
    printf '## Benchmark gate: failed in %s (exit %d)\n\n%s\n' "$stage" "$status" \
      "Nothing was compared; the job log has the error." >"$out/report.md"
  elif [ -z "$reported" ]; then # a signal (the job's timeout, a cancel) ended the run
    echo "## Benchmark gate: did not finish (stopped in $stage)" >"$out/report.md"
  fi
}
trap finish EXIT

if [ "${BENCH_BOOTSTRAP:-}" != 1 ]; then
  stage="checking BENCH_BASE"
  base_rev=$(git rev-parse --verify --quiet "${BENCH_BASE:-}^{commit}") || {
    echo "BENCH_BASE='${BENCH_BASE:-}' is not a commit in this clone" >&2
    exit 2
  }
fi

# Candidate build: this checkout, with vernier's CLI tools for the comparison
stage="the candidate build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVERNIER_BUILD_TOOLS=ON
cmake --build build --parallel "$(nproc)"
bench=build/bin/tools/rust/bench

if [ "${BENCH_BOOTSTRAP:-}" = 1 ]; then
  stage="the candidate run"
  "build/$target" "${args[@]}" --csv "$out/candidate.csv"
  printf '## Benchmark gate: no comparison (BENCH_BOOTSTRAP=1)\n\n%s\n' \
    "Only the candidate ran, as asked; candidate.csv holds its results." >"$out/report.md"
  reported=1
  cat "$out/report.md"
  exit 0
fi

# Baseline build: the base commit in a worktree outside the checkout
stage="the baseline build"
work=$(mktemp -d)
git worktree add --quiet --detach "$work/src" "$base_rev"
built=0
{
  cmake -S "$work/src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release &&
    cmake --build "$work/build" --parallel "$(nproc)" --target "$target"
} >"$out/baseline-build.log" 2>&1 || built=$?
if [ "$built" -ne 0 ]; then
  stage="the candidate run"
  "build/$target" "${args[@]}" --csv "$out/candidate.csv"
  printf '## Benchmark gate: failed, the baseline did not build (exit %d)\n\n%s\n' "$built" \
    "$target did not build at $base_rev (see baseline-build.log), so nothing was compared.
If this change adds the benchmark, run its gate with BENCH_BOOTSTRAP=1." >"$out/report.md"
  reported=1
  cat "$out/report.md"
  exit "$built"
fi

# Both runs after both builds, one after the other
stage="the baseline run"
"$work/build/$target" "${args[@]}" --csv "$out/baseline.csv"
stage="the candidate run"
"build/$target" "${args[@]}" --csv "$out/candidate.csv"

# The report first; the gate's status is bench compare's
stage="the comparison"
status=0
table=$("$bench" compare "$out/baseline.csv" "$out/candidate.csv" \
  --threshold "$threshold" --markdown --fail-on-regression 2>&1) || status=$?
"$bench" compare "$out/baseline.csv" "$out/candidate.csv" --threshold "$threshold" \
  --json >"$out/report.json" 2>/dev/null || rm -f "$out/report.json"
verdict=passed
[ "$status" -eq 0 ] || verdict="failed (bench compare exit $status)"
printf '## Benchmark gate: %s\n\nBaseline %s, threshold %s%%.\n\n%s\n' \
  "$verdict" "$base_rev" "$threshold" "$table" >"$out/report.md"
reported=1
cat "$out/report.md"
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
```

Run it from the repository's top directory. Locally, against your main branch:

```bash
BENCH_BASE=origin/main bash ci/bench-gate.sh
```

**Its report.** The script owns five files in `BENCH_OUT`: `report.md`,
`report.json`, `candidate.csv`, `baseline.csv` and `baseline-build.log`. It
deletes them when it starts and leaves anything else in the directory alone,
so nothing an earlier run wrote can pass for this run's result. The first line
of `report.md` says how this run ended: passed, failed (and in which step), no
comparison, or "did not finish" when a signal such as the job's timeout
stopped it. After a comparison, `report.md` holds the
`bench compare --markdown` table and `report.json` its JSON. Which tests are
compared, and how a test that is in only one of the two files is treated, is
`bench compare`'s behavior; see `compare` in
[tools/README.md](../../../tools/README.md).

**Exit status:** 0 when the runs were compared and
`bench compare --fail-on-regression` passed, or when `BENCH_BOOTSTRAP=1` asked
for no comparison. Otherwise it is nonzero, and `report.md` says why: 1 when
`bench compare` fails (a regression, among the reasons its README lists), 2
when `BENCH_BASE` is not set or not a commit in the clone, and the failing
command's own status when a build or a benchmark run fails, the baseline's
included. A run that a signal stops ends nonzero as well.

**A baseline that does not build** fails the gate. The candidate still runs,
so `candidate.csv` shows its results, and `baseline-build.log` has the
baseline's build output. A missing dependency, a failed download and a
benchmark the base commit does not have yet all look the same to the script,
so it passes none of them on its own. For the change that adds the benchmark,
set `BENCH_BOOTSTRAP=1`: the script builds and runs only the candidate, writes
a `report.md` that says nothing was compared, and exits 0. It compares nothing
while it is set, so set it for that change's run only, not in the workflow for
good.

---

## GitHub Actions

**.github/workflows/benchmarks.yml:**

```yaml
name: Benchmarks

on:
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  benchmark-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0 # full history, so the base commit is in the clone

      - name: Benchmark gate
        env:
          BENCH_BASE: ${{ github.event.pull_request.base.sha }}
          BENCH_TARGET: MyComponent_PTEST
        run: bash ci/bench-gate.sh

      - name: Job summary
        if: always()
        run: |
          if [ -f bench-report/report.md ]; then
            cat bench-report/report.md >> "$GITHUB_STEP_SUMMARY"
          fi

      - name: Upload the report
        if: always()
        uses: actions/upload-artifact@v7
        with:
          name: benchmark-report
          path: bench-report/
```

- `ubuntu-latest` (Ubuntu 24.04) comes with CMake, GCC, git and Rust, so the
  job installs nothing.
- The report is written to the run's job summary and kept as the
  `benchmark-report` artifact, when the gate passes and when it fails. The
  workflow only reads the repository: a pull request from a fork gets a
  read-only token, so a step that comments on the pull request would fail
  there.
- For a GPU benchmark, run the same job on a self-hosted runner that has the
  NVIDIA driver and the CUDA toolkit (for example `runs-on: [self-hosted, gpu]`,
  with a label you gave the runner), and set `BENCH_TARGET` to the GPU
  benchmark. The gate compares each test's `wallMedian`, for GPU rows as for
  CPU rows, not `kernelTimeUs`. A kernel-only test's `wallMedian` is its
  kernel time, so the two move together. For a test that copies its data in
  and out, `wallMedian` is the whole round trip: a slower kernel can be a
  small part of its change, and the copies' own movement from run to run can
  outweigh it. Walkthrough 10's
  [GPU columns](../demo/docs/10_GPU_BASIC_WORKFLOW.md#what-the-gpu-harness-measures)
  say what each one measures.

---

## GitLab CI

**.gitlab-ci.yml:**

```yaml
benchmark-gate:
  image: ubuntu:24.04
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  variables:
    GIT_DEPTH: "0" # full history, so the base commit is in the clone
    BENCH_TARGET: MyComponent_PTEST
  before_script:
    - apt-get update
    - apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl git
    - curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    - export PATH="$HOME/.cargo/bin:$PATH"
  script:
    - BENCH_BASE=$CI_MERGE_REQUEST_DIFF_BASE_SHA bash ci/bench-gate.sh
  artifacts:
    when: always
    paths:
      - bench-report/
    expire_in: 30 days
```

- The job runs in merge request pipelines, where
  `CI_MERGE_REQUEST_DIFF_BASE_SHA` is the base commit of the merge request's
  diff.
- Rust comes from rustup, since Ubuntu's `cargo` cannot read the CLI's
  `Cargo.lock`.
- `when: always` keeps `bench-report/` when the gate fails; GitLab keeps no
  artifacts from a job that timed out.

---

## Azure Pipelines

**azure-pipelines.yml:**

```yaml
trigger: none

pr:
  branches:
    include:
      - main

pool:
  vmImage: ubuntu-latest

steps:
  - checkout: self
    fetchDepth: "0" # full history, so the base commit is in the clone

  - script: bash ci/bench-gate.sh
    displayName: Benchmark gate
    env:
      BENCH_BASE: HEAD^1 # a pull request build checks out a merge commit
      BENCH_TARGET: MyComponent_PTEST

  - publish: bench-report
    artifact: benchmark-report
    condition: always()
```

- A pull request build checks out a merge commit of the source and target
  branches. GitHub's merge commits have the target branch as their first
  parent, `HEAD^1`; on another host, `git log -1 --format=%P` in a pull
  request build shows the order of the parents.
- `ubuntu-latest` is the same Ubuntu 24.04 image as on GitHub Actions, with
  CMake, GCC, git and Rust.
- The `pr` trigger applies to repositories on GitHub and Bitbucket Cloud. For a
  repository in Azure Repos, pull request builds come from a build validation
  branch policy on the target branch instead.

---

## Jenkins

**Jenkinsfile** (a multibranch pipeline; the agent needs the tools listed under
[What the Job Needs](#what-the-job-needs)):

```groovy
pipeline {
  agent any
  options {
    timeout(time: 60, unit: 'MINUTES')
  }
  stages {
    stage('Benchmark gate') {
      when { changeRequest() }
      environment {
        BENCH_TARGET = 'MyComponent_PTEST'
      }
      steps {
        sh '''
          git fetch --no-tags origin "+refs/heads/$CHANGE_TARGET:refs/remotes/origin/$CHANGE_TARGET"
          BENCH_BASE=$(git merge-base HEAD "origin/$CHANGE_TARGET") bash ci/bench-gate.sh
        '''
      }
    }
  }
  post {
    always {
      archiveArtifacts artifacts: 'bench-report/**', allowEmptyArchive: true
    }
  }
}
```

- The stage runs for change requests (pull and merge requests), for which the
  branch source sets `CHANGE_TARGET` to the branch the change would merge into.
  It fetches that branch itself, so `git fetch` must work from a shell step on
  the agent.
- `archiveArtifacts` in `post { always { ... } }` keeps `bench-report/` when the
  gate fails.

---

## Local Pre-Push Hook

Run the gate before a push, against the branch's upstream:

**.git/hooks/pre-push:**

```bash
#!/usr/bin/env bash
# Run the benchmark gate against the upstream branch before a push.
set -euo pipefail
upstream=$(git rev-parse --verify --quiet '@{upstream}') || {
  echo "pre-push: no upstream branch; benchmark gate skipped"
  exit 0
}
BENCH_BASE=$(git merge-base HEAD "$upstream") bash ci/bench-gate.sh
```

```bash
chmod +x .git/hooks/pre-push
```

Git runs the hook from the top of the working tree and cancels the push when
it exits nonzero. It gates the checked-out branch against the commit it shares
with its upstream, builds in `build/` as the guides do, and writes
`bench-report/`, which you may want in `.gitignore`. `git push --no-verify`
skips it.

---

## Best Practices

### 1. Gate on a Machine That Runs Nothing Else

The gate compares two runs made one after the other, so what matters is that
the machine's speed does not change between them. A self-hosted runner that runs
one job at a time is the place for it. On shared hosted runners, expect runs to
move more, and choose the threshold from that movement.

### 2. Threshold Selection

`BENCH_THRESHOLD` is passed to `bench compare --threshold` (percent, default 5).
Choose it on the machine that runs the gate, from runs of unchanged code:
`BENCH_BASE=HEAD bash ci/bench-gate.sh` compares the checkout with itself, so
every change it reports is noise. Run it a few times and set the threshold
above the largest movement you see. The CSVs' `wallCV` column is the spread of
one run's repeats (standard deviation over mean); it does not show how far a
test moves from one run to the next, which is what the gate compares.

### 3. Shorter or Longer Runs

`BENCH_ARGS` is passed to both runs. `--quick` shortens them; `--repeats N` or
`--target-time` (for example `--target-time 100ms`) gives each test more or
longer repeats; `--gtest_filter` selects the tests that gate.

### 4. Keep Reports

Each example keeps `bench-report/` as an artifact. To keep it longer or shorter
than the provider's default, set `retention-days` in the upload step's `with:`
on GitHub Actions or change `expire_in` on GitLab.

---

## Troubleshooting

### A Test Is Reported Slower Without a Change

1. Measure how far it moves with no change: run
   `BENCH_BASE=HEAD bash ci/bench-gate.sh` a few times (see
   [Threshold Selection](#2-threshold-selection)).
2. Give it more repeats: `BENCH_ARGS="--repeats 30"`.
3. Run the gate on a dedicated machine, or raise `BENCH_THRESHOLD` above the
   movement you measured.

### The Report Says "the baseline did not build"

The benchmark did not configure or build at the base commit, and the gate
failed. `baseline-build.log` in the report shows why. A missing dependency or
a failed download needs fixing, and the gate rerun. When the change adds or
renames the benchmark, the base commit cannot have it: run that change's gate
with `BENCH_BOOTSTRAP=1` (see [The Gate Script](#the-gate-script)).

### The Report Says "did not finish"

A signal stopped the script before it could write a result, for example the
job's timeout or a cancel; the report names the step it was in, unless the
signal was one that allows no cleanup (`SIGKILL`). Nothing in `bench-report/`
comes from an earlier run.

### "BENCH_BASE=... is not a commit in this clone"

The clone is shallow, or `BENCH_BASE` names something else. Fetch the full
history (`fetch-depth: 0`, `GIT_DEPTH: "0"`, `fetchDepth: "0"` in the examples).

### "error: failed to parse lock file"

The `cargo` that builds vernier's CLI is too old for its `Cargo.lock`, as
Ubuntu 24.04's packaged 1.75 is. Install Rust with rustup, as the GitLab
example does.

### The Job Runs Too Long

Run fewer or shorter tests: `BENCH_ARGS="--quick"`, or a `--gtest_filter` in
`BENCH_ARGS`. The gate builds two trees; the job's timeout (`timeout-minutes`
on GitHub Actions) must cover both builds and both runs.

---

## See Also

- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[Docker Setup](DOCKER_SETUP.md)** - Running benchmarks in containers
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking best practices
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking best practices
- **[Main README](../../../README.md)** - Framework overview
