# CI/CD Integration Guide

Integrate performance benchmarking into your continuous integration pipelines. Catch performance regressions before they reach production with automated benchmark validation.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [GitHub Actions](#github-actions)
- [GitLab CI](#gitlab-ci)
- [Jenkins](#jenkins)
- [Azure Pipelines](#azure-pipelines)
- [Local Pre-Commit Hooks](#local-pre-commit-hooks)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What This Enables

**Automatic regression detection** - Block PRs with performance regressions
**Historical tracking** - Store benchmark results as CI artifacts
**PR comments** - Post benchmark comparisons directly on pull requests
**Baseline management** - Track performance across branches
**Fail-fast** - Catch regressions before code review
**Statistical validation** - Distinguish real regressions from noise

### How It Works

```
+---------------+
|  PR Created   |
+-------+-------+
        |
        +--> Build & test baseline (main branch)
        |    +--> baseline.csv
        |
        +--> Build & test candidate (PR branch)
        |    +--> candidate.csv
        |
        +--> Compare with bench compare
        |    +--> Statistical analysis
        |    +--> Threshold checking (default: 5%)
        |    +--> Exit code: 0=pass, 1=fail
        |
        +--> Generate reports
             +--> Markdown (PR comment)
             +--> JSON (artifact storage)
```

### Prerequisites

- Benchmarks built with this framework
- Rust bench tool (`make tools-rust`)
- Optional: Python bench-plot for visualizations (`make tools-py`)

---

## Quick Start

**Minimal integration (works with any CI system):**

```bash
#!/bin/bash
# ci_benchmark.sh

set -e

# Install Python dependencies
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)

# Build benchmarks
cmake -B build -S . && cmake --build build

# Run baseline (from main branch)
git checkout main
cmake --build build
./build/bin/ptests/MyComponent_PTEST --csv baseline.csv
git checkout -

# Rebuild with PR changes
cmake --build build

# Run candidate benchmarks
./build/bin/ptests/MyComponent_PTEST --csv candidate.csv

# Compare with auto-fail on regression
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold 5 \
--fail-on-regression \
--output-md pr_comment.md \
--output-json results.json

# Exit code 0 = pass, 1 = regression detected
```

---

## GitHub Actions

### Basic Workflow

**.github/workflows/performance.yml:**

```yaml
name: Performance Benchmarks

on:
pull_request:
branches: [main]

jobs:
benchmark:
runs-on: ubuntu-latest
timeout-minutes: 30

steps:
- name: Checkout code
uses: actions/checkout@v4
with:
fetch-depth: 0 # Need full history

- name: Install dependencies
run: |
sudo apt-get update
sudo apt-get install -y cmake g++ libgtest-dev
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)

- name: Build benchmarks
run: |
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

- name: Run candidate benchmarks
run: |
./build/bin/ptests/MyComponent_PTEST --csv candidate.csv

- name: Generate baseline
run: |
git checkout ${{ github.event.pull_request.base.sha }}
cmake --build build
./build/bin/ptests/MyComponent_PTEST --csv baseline.csv
git checkout -

- name: Compare and detect regressions
id: compare
run: |
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold 5 \
--fail-on-regression \
--output-md pr_comment.md \
--output-json regression_report.json

- name: Post PR comment
if: always()
uses: actions/github-script@v7
with:
script: |
const fs = require('fs');
const comment = fs.readFileSync('pr_comment.md', 'utf8');
github.rest.issues.createComment({
issue_number: context.issue.number,
owner: context.repo.owner,
repo: context.repo.repo,
body: comment
});

- name: Upload artifacts
if: always()
uses: actions/upload-artifact@v4
with:
name: benchmark-results
path: |
baseline.csv
candidate.csv
regression_report.json
pr_comment.md
```

### Advanced Workflow with Caching

```yaml
name: Performance CI (Advanced)

on:
pull_request:
branches: [main]
paths:
- 'src/**'
- 'benchmarks/**'

jobs:
benchmark:
runs-on: ubuntu-latest
timeout-minutes: 30

steps:
- name: Checkout with history
uses: actions/checkout@v4
with:
fetch-depth: 0

- name: Cache baseline results
id: cache-baseline
uses: actions/cache@v4
with:
path: baseline.csv
key: benchmark-baseline-${{ github.event.pull_request.base.sha }}

- name: Setup Python
uses: actions/setup-python@v5
with:
python-version: '3.10'
cache: 'pip'

- name: Install Python dependencies
run: make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)

- name: Install build dependencies
run: |
sudo apt-get update
sudo apt-get install -y cmake g++ libgtest-dev taskset

- name: Build candidate
run: |
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

- name: Run candidate benchmarks (with CPU pinning)
run: |
taskset -c 2-9 ./build/bin/ptests/MyComponent_PTEST \
--cycles 10000 \
--repeats 20 \
--csv candidate.csv

- name: Generate baseline (if not cached)
if: steps.cache-baseline.outputs.cache-hit != 'true'
run: |
git checkout ${{ github.event.pull_request.base.sha }}
cmake --build build
taskset -c 2-9 ./build/bin/ptests/MyComponent_PTEST \
--cycles 10000 \
--repeats 20 \
--csv baseline.csv
git checkout -

- name: Regression detection
id: regression
run: |
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold 5 \
--fail-on-regression \
--output-md pr_comment.md \
--output-json results.json
continue-on-error: true

- name: Generate visualizations
if: always()
run: |
bench compare \
baseline.csv candidate.csv \
--output comparison/

- name: Post detailed PR comment
if: always()
uses: actions/github-script@v7
with:
script: |
const fs = require('fs');
let comment = fs.readFileSync('pr_comment.md', 'utf8');

// Add link to artifacts
comment += '\n\n---\n';
comment += `\n [View detailed comparison](https://github.com/${{github.repository}}/actions/runs/${{github.run_id}})`;

github.rest.issues.createComment({
issue_number: context.issue.number,
owner: context.repo.owner,
repo: context.repo.repo,
body: comment
});

- name: Upload all results
if: always()
uses: actions/upload-artifact@v4
with:
name: benchmark-results
path: |
baseline.csv
candidate.csv
results.json
pr_comment.md
comparison/

- name: Fail if regressions detected
if: steps.regression.outcome == 'failure'
run: exit 1
```

### GPU Benchmarks in GitHub Actions

```yaml
name: GPU Performance

on:
pull_request:
branches: [main]

jobs:
gpu-benchmark:
runs-on: [self-hosted, gpu] # Self-hosted runner with NVIDIA GPU

steps:
- name: Checkout code
uses: actions/checkout@v4
with:
fetch-depth: 0

- name: Verify GPU access
run: nvidia-smi

- name: Install dependencies
run: |
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)

- name: Build GPU benchmarks
run: |
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target MyKernel_GPU_PTEST

- name: Run GPU benchmarks
run: |
./build/bin/ptests/MyKernel_GPU_PTEST --csv candidate.csv

- name: Generate baseline
run: |
git checkout ${{ github.event.pull_request.base.sha }}
cmake --build build --target MyKernel_GPU_PTEST
./build/bin/ptests/MyKernel_GPU_PTEST --csv baseline.csv
git checkout -

- name: Compare GPU performance
run: |
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold 5 \
--fail-on-regression \
--output-md gpu_report.md

# Upload and comment steps...
```

---

## GitLab CI

### Basic Pipeline

**.gitlab-ci.yml:**

```yaml
stages:
- build
- benchmark
- report

variables:
REGRESSION_THRESHOLD: "5"

build:
stage: build
script:
- cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
- cmake --build build -j$(nproc)
artifacts:
paths:
- build/
expire_in: 1 hour

benchmark:
stage: benchmark
dependencies:
- build
script:
# Install Python deps
- make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)

# Run candidate benchmarks
- ./build/bin/ptests/MyComponent_PTEST --csv candidate.csv

# Generate baseline
- git fetch origin main
- git checkout origin/main
- cmake --build build
- ./build/bin/ptests/MyComponent_PTEST --csv baseline.csv
- git checkout $CI_COMMIT_SHA

# Compare with auto-fail
- bench compare
--baseline baseline.csv
--candidate candidate.csv
--threshold $REGRESSION_THRESHOLD
--fail-on-regression
--output-json regression_report.json
--output-md pr_comment.md

artifacts:
when: always
paths:
- regression_report.json
- candidate.csv
- baseline.csv
- pr_comment.md
expire_in: 30 days
reports:
junit: regression_report.json

report:
stage: report
dependencies:
- benchmark
script:
- cat pr_comment.md
when: always
```

### Merge Request Comments

```yaml
benchmark:
stage: benchmark
only:
- merge_requests
script:
# ... benchmark steps ...

# Post comment to MR
- |
curl --request POST \
--header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
--data-urlencode "body=$(cat pr_comment.md)" \
"$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
```

---

## Jenkins

### Declarative Pipeline

**Jenkinsfile:**

```groovy
pipeline {
agent any

parameters {
string(name: 'THRESHOLD', defaultValue: '5.0', description: 'Regression threshold %')
}

environment {
BASELINE_BRANCH = 'main'
}

stages {
stage('Build') {
steps {
sh '''
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
'''
}
}

stage('Candidate Benchmark') {
steps {
sh './build/bin/ptests/MyComponent_PTEST --csv candidate.csv'
}
}

stage('Baseline Benchmark') {
steps {
sh '''
git stash
git checkout ${BASELINE_BRANCH}
cmake --build build
./build/bin/ptests/MyComponent_PTEST --csv baseline.csv
git checkout -
git stash pop || true
'''
}
}

stage('Compare') {
steps {
sh '''
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold ${THRESHOLD} \
--fail-on-regression \
--output-json regression_report.json \
--output-md report.md
'''
}
}
}

post {
always {
archiveArtifacts artifacts: '*.csv,*.json,*.md', fingerprint: true

script {
def report = readFile('report.md')
if (env.CHANGE_ID) {
// Post to pull request if available
pullRequest.comment(report)
}
}
}

success {
echo ' No performance regressions detected'
}

failure {
echo 'ERROR: Performance regressions detected!'
}
}
}
```

---

## Azure Pipelines

**azure-pipelines.yml:**

```yaml
trigger:
- main

pr:
- main

pool:
vmImage: 'ubuntu-latest'

steps:
- checkout: self
fetchDepth: 0

- task: UsePythonVersion@0
inputs:
versionSpec: '3.10'

- script: |
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)
displayName: 'Install Python dependencies'

- script: |
sudo apt-get update
sudo apt-get install -y cmake g++ libgtest-dev
displayName: 'Install build dependencies'

- script: |
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
displayName: 'Build benchmarks'

- script: |
./build/bin/ptests/MyComponent_PTEST --csv candidate.csv
displayName: 'Run candidate benchmarks'

- script: |
git checkout $(System.PullRequest.TargetBranch)
cmake --build build
./build/bin/ptests/MyComponent_PTEST --csv baseline.csv
git checkout -
displayName: 'Generate baseline'
condition: eq(variables['Build.Reason'], 'PullRequest')

- script: |
bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold 5 \
--fail-on-regression \
--output-json results.json \
--output-md pr_comment.md
displayName: 'Regression detection'

- task: PublishBuildArtifacts@1
inputs:
pathToPublish: '$(Build.SourcesDirectory)'
artifactName: 'benchmark-results'
condition: always()
```

---

## Local Pre-Commit Hooks

Catch regressions before pushing to remote:

**.git/hooks/pre-push:**

```bash
#!/bin/bash
# Pre-push hook for performance regression detection

set -e

echo " Running performance checks..."

# Build
cmake --build build -j$(nproc)

# Run benchmarks
./build/bin/ptests/MyComponent_PTEST --quick --csv candidate.csv

# Compare with cached baseline (if exists)
if [ -f ".baseline.csv" ]; then
bench compare \
.baseline.csv candidate.csv \
--threshold 10 \
--fail-on-regression

if [ $? -ne 0 ]; then
echo "ERROR: Performance regression detected!"
echo " Review changes or update baseline with:"
echo " cp candidate.csv .baseline.csv"
read -p "Push anyway? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
exit 1
fi
fi
else
echo "No baseline found. Creating one..."
cp candidate.csv .baseline.csv
fi

echo " Performance checks passed"
```

Make executable:

```bash
chmod +x .git/hooks/pre-push
```

---

## Best Practices

### 1. Baseline Management

**Strategy: Branch-based baselines**

```yaml
# Cache baselines per base branch
- name: Cache baseline
uses: actions/cache@v4
with:
path: baseline.csv
key: benchmark-${{ github.event.pull_request.base.sha }}
```

**Strategy: Periodic baseline updates**

```yaml
# Nightly job to update main baseline
name: Update Baseline
on:
schedule:
- cron: '0 2 * * *' # 2 AM daily

jobs:
update-baseline:
runs-on: ubuntu-latest
steps:
- # ... build and run benchmarks ...
- name: Upload new baseline
uses: actions/upload-artifact@v4
with:
name: main-baseline
path: baseline.csv
```

### 2. Threshold Selection

**Conservative (catch most regressions):**

```bash
--threshold 3 # Flag changes >3%
```

**Balanced (default):**

```bash
--threshold 5 # Flag changes >5%
```

**Lenient (noisy environments):**

```bash
--threshold 10 # Flag changes >10%
```

**Per-test thresholds (advanced):**

```python
# In Python wrapper script
thresholds = {
"MyComponent.CriticalPath": 3,
"MyComponent.SlowTest": 10,
"MyComponent.NoiseTest": 15
}
```

### 3. CPU Pinning in CI

Reduce variance with taskset:

```yaml
- name: Run benchmarks with pinning
run: |
taskset -c 2-9 ./build/bin/ptests/MyComponent_PTEST \
--cycles 10000 --repeats 20 --csv results.csv
```

### 4. Quick vs Full Benchmarks

**Pull Request: Quick checks**

```yaml
on:
pull_request:
steps:
  - run: ./test --quick --csv results.csv
```

**Nightly: Full characterization**

```yaml
on:
schedule:
  - cron: "0 2 * * *"
steps:
  - run: ./test --cycles 50000 --repeats 50 --csv results.csv
```

### 5. Artifact Retention

```yaml
- name: Upload artifacts
uses: actions/upload-artifact@v4
with:
name: benchmark-results-${{ github.run_id }}
path: |
baseline.csv
candidate.csv
regression_report.json
retention-days: 90 # Keep for 3 months
```

---

## Troubleshooting

### False Positives (Noise Flagged as Regression)

**Cause:** Measurements too noisy for chosen threshold.

**Solutions:**

1. **Check CV% in baseline:**

```bash
grep "wallCV" baseline.csv
# If >0.10 (10%), measurements are inherently noisy
```

2. **Increase threshold:**

```bash
--threshold 10 # More lenient
```

3. **Add CPU pinning:**

```bash
taskset -c 2-9 ./test --csv results.csv
```

4. **Increase samples:**

```bash
./test --repeats 30 --csv results.csv
```

### CI Timeouts

**Cause:** Benchmarks take too long.

**Solutions:**

1. **Use --quick mode for PRs:**

```bash
./test --quick --csv results.csv
```

2. **Reduce cycles/repeats:**

```bash
./test --cycles 5000 --repeats 10
```

3. **Filter tests:**

```bash
./test --gtest_filter="Critical*" --csv results.csv
```

4. **Split into separate jobs:**

```yaml
jobs:
quick-check: # Fast, on every PR
run: ./test --quick

full-benchmark: # Slow, nightly only
if: github.event_name == 'schedule'
run: ./test --cycles 50000
```

### Inconsistent Results Across CI Runs

**Cause:** Shared CI runners, background processes.

**Solutions:**

1. **Use self-hosted runners** for consistent hardware

2. **Relax thresholds** for shared runners:

```bash
--threshold 10 # More tolerant
```

3. **Run multiple times and average:**

```bash
for i in {1..3}; do
./test --csv run_$i.csv
done
# Merge results
```

### Missing Dependencies in CI

**Python packages:**

```yaml
- name: Install Python deps
run: make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts)
```

**Build tools:**

```yaml
- name: Install build deps
run: |
sudo apt-get update
sudo apt-get install -y cmake g++ libgtest-dev
```

---

## Example: Production-Ready Workflow

**Complete workflow with all best practices:**

```yaml
name: Performance CI (Production)

on:
pull_request:
branches: [main, develop]
paths:
- 'src/**'
- 'include/**'
- 'benchmarks/**'
schedule:
- cron: '0 2 * * *' # Nightly full benchmark

jobs:
benchmark:
runs-on: ubuntu-latest
timeout-minutes: 45

strategy:
matrix:
mode: [quick, full]
exclude:
- mode: full
# Only run full on schedule
${{ github.event_name != 'schedule' }}

steps:
- name: Checkout
uses: actions/checkout@v4
with:
fetch-depth: 0

- name: Cache baseline
uses: actions/cache@v4
with:
path: baseline.csv
key: benchmark-${{ github.event.pull_request.base.sha || 'main' }}

- name: Setup
run: |
sudo apt-get update
sudo apt-get install -y cmake g++ libgtest-dev taskset
make tools-rust  # bench compare, bench summary
make tools-py    # bench-plot (optional, for charts) plotly

- name: Build
run: |
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

- name: Run benchmarks
run: |
FLAGS=""
if [ "${{ matrix.mode }}" = "quick" ]; then
FLAGS="--quick"
else
FLAGS="--cycles 20000 --repeats 30"
fi

taskset -c 2-9 ./build/bin/ptests/MyComponent_PTEST \
$FLAGS --csv candidate.csv

- name: Generate baseline
if: steps.cache-baseline.outputs.cache-hit != 'true'
run: |
git checkout ${{ github.event.pull_request.base.sha || 'main' }}
cmake --build build
taskset -c 2-9 ./build/bin/ptests/MyComponent_PTEST \
$FLAGS --csv baseline.csv
git checkout -

- name: Regression detection
id: regression
run: |
THRESHOLD=5
if [ "${{ matrix.mode }}" = "quick" ]; then
THRESHOLD=10
fi

bench compare \
--baseline baseline.csv \
--candidate candidate.csv \
--threshold $THRESHOLD \
--fail-on-regression \
--output-md pr_comment.md \
--output-json results.json
continue-on-error: true

- name: Generate visualizations
if: always()
run: |
bench compare \
baseline.csv candidate.csv \
--output comparison/

bench-plot dashboard \
candidate.csv \
--output dashboard.html

- name: Post PR comment
if: github.event_name == 'pull_request' && always()
uses: actions/github-script@v7
with:
script: |
const fs = require('fs');
const comment = fs.readFileSync('pr_comment.md', 'utf8');
github.rest.issues.createComment({
issue_number: context.issue.number,
owner: context.repo.owner,
repo: context.repo.repo,
body: comment
});

- name: Upload artifacts
if: always()
uses: actions/upload-artifact@v4
with:
name: benchmark-results-${{ matrix.mode }}
path: |
baseline.csv
candidate.csv
results.json
pr_comment.md
comparison/
dashboard.html
retention-days: 90

- name: Fail on regression
if: steps.regression.outcome == 'failure'
run: exit 1
```

---

## See Also

- **CLI Tools:** `tools/README.md` - CLI tools reference
- **[Docker Setup](DOCKER_SETUP.md)** - Running benchmarks in containers
- **[CPU Guide](CPU_GUIDE.md)** - CPU benchmarking best practices
- **[GPU Guide](GPU_GUIDE.md)** - GPU benchmarking best practices
- **[Main README](../../../README.md)** - Framework overview
