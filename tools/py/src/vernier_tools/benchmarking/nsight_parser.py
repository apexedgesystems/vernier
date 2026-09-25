#!/usr/bin/env python3
"""
nsight_parser.py -- Read Nsight Systems / Nsight Compute reports into one CSV.

Runs `nsys stats` (the four CUDA summaries) and `ncu --csv --print-summary
per-kernel` on each report and writes what they print as one CSV of its own.
That CSV is not a benchmark CSV: bench summary, bench compare and bench-plot
need test, wallMedian, wallCV and callsPerSecond columns and refuse it. Read
it with a CSV tool.

Inputs:
    *.nsys-rep      Nsight Systems report
    *.ncu-rep       Nsight Compute report (not read in this release; below)
    a directory     every report under it

Usage:
    nsight-parse parse run.nsys-rep --csv summaries.csv
    nsight-parse parse <dir>/       --csv combined.csv

Output (nsys): one row per row of each summary -- per kernel name, per CUDA
call name, per kind of copy -- not one per launch. Columns:
    source              "nsys"
    report              the summary (cuda_gpu_kern_sum, cuda_api_sum, ...)
    kernel              its Name; empty for the copy summaries (Operation)
    instances           Instances or Num Calls; empty for the copy summaries
    time_total_ns       Total Time (ns)
    time_avg_ns         Avg (ns)
    time_pct            Time (%)
    ...                 every other column the summaries print, under its own
                        name, in alphabetical order

Limits in this release (each failed command is a warning on stderr; the rows
that were read are written, and the exit status is 0):
    nsys stats runs without --force-export=true, so a report that
    `bench run --profile nsight` summarized, with its SQLite export beside it,
    can be refused ("older than input file") and yield no rows; a copy of the
    report in a directory of its own parses.
    ncu runs without --import, so it takes an .ncu-rep for a program to launch
    and yields no rows. Meanwhile: ncu --import run.ncu-rep --csv
    --print-summary per-kernel.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# =============================== Public API ===================================

NSYS_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_gpu_mem_time_sum",
)

NCU_DEFAULT_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
)


@dataclass
class ParseResult:
    """Tidy in-memory rows from one or more Nsight report files."""

    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def write_csv(self, path: Path) -> None:
        """Write rows to `path` as CSV with canonical column order.

        Always creates the file. Known columns (source, report, kernel,
        instances, time_*) come first; any per-tool metric columns are
        appended in alphabetical order so downstream tooling sees a
        stable header.
        """
        if not self.rows:
            path.write_text("")  # still produce the file
            return
        # Stable column order: known keys first, then any extras.
        preferred = [
            "source",
            "report",
            "kernel",
            "instances",
            "time_total_ns",
            "time_avg_ns",
            "time_pct",
        ]
        extras = sorted({k for r in self.rows for k in r.keys()} - set(preferred))
        cols = preferred + extras
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k, "") for k in cols})


# =============================== Entry Point =================================


def parse_paths(paths: Iterable[Path]) -> ParseResult:
    """Parse one or more Nsight report files (or directories of them)."""
    result = ParseResult()
    files = list(_iter_inputs(paths))
    if not files:
        result.warnings.append("no .nsys-rep / .ncu-rep files found in inputs")
        return result
    for f in files:
        if f.suffix == ".nsys-rep":
            _parse_nsys(f, result)
        elif f.suffix == ".ncu-rep":
            _parse_ncu(f, result)
    return result


def _iter_inputs(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(p.glob("**/*.nsys-rep"))
            yield from sorted(p.glob("**/*.ncu-rep"))
        elif p.suffix in {".nsys-rep", ".ncu-rep"} and p.is_file():
            yield p


# =============================== Nsight Systems ==============================


def _parse_nsys(path: Path, result: ParseResult) -> None:
    """Run nsys stats on each canonical report and append the rows."""
    for report in NSYS_REPORTS:
        text = _run(["nsys", "stats", "--report", report, "--format", "csv", str(path)])
        if text is None:
            result.warnings.append(f"nsys stats --report {report} failed for {path.name}")
            continue
        for row in _iter_csv_after_header(text):
            row.setdefault("kernel", row.pop("Name", row.pop("Range", "")))
            row.setdefault("instances", row.pop("Instances", row.pop("Num Calls", "")))
            row.setdefault("time_total_ns", row.pop("Total Time (ns)", ""))
            row.setdefault("time_avg_ns", row.pop("Avg (ns)", ""))
            row.setdefault("time_pct", row.pop("Time (%)", ""))
            result.rows.append({"source": "nsys", "report": report, **row})


# =============================== Nsight Compute ==============================


def _parse_ncu(path: Path, result: ParseResult) -> None:
    """Run ncu --csv summary; one row per kernel, columns are metrics."""
    text = _run(["ncu", "--csv", "--print-summary", "per-kernel", str(path)])
    if text is None:
        result.warnings.append(f"ncu --csv failed for {path.name}")
        return
    for row in _iter_csv_after_header(text):
        result.rows.append(
            {
                "source": "ncu",
                "report": "per_kernel",
                "kernel": row.pop("Kernel Name", row.pop("Kernel", "")),
                **{_clean_key(k): v for k, v in row.items()},
            }
        )


def _clean_key(s: str) -> str:
    # ncu column names contain spaces / parens; normalize to snake-ish.
    return (
        s.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")
    )


# =============================== Subprocess Helpers ==========================


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _iter_csv_after_header(text: str) -> Iterable[dict]:
    """nsys / ncu prefix their CSV output with a multi-line banner; locate the
    first row that looks like a CSV header (multiple comma-separated cells)
    and parse from there.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.count(",") >= 2 and not line.lstrip().startswith("#"):
            start = i
            break
    if start is None:
        return
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    yield from reader


# =============================== Main CLI ====================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nsight-parse", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="parse Nsight report file(s) into CSV")
    p_parse.add_argument("inputs", nargs="+", type=Path, help="file(s) or directory")
    p_parse.add_argument("--csv", type=Path, required=True, help="output CSV path")

    args = parser.parse_args(argv)

    if args.cmd == "parse":
        result = parse_paths(args.inputs)
        result.write_csv(args.csv)
        for w in result.warnings:
            print(f"[nsight-parse] {w}", file=sys.stderr)
        print(f"[nsight-parse] wrote {len(result.rows)} rows to {args.csv}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
