#!/usr/bin/env python3
"""
nsight_parser.py -- Read Nsight Systems / Nsight Compute reports into one CSV.

An .nsys-rep is exported once, to a private temporary SQLite file, and its
four CUDA summaries are read from that export with `nsys stats --format
csv`; an export `bench run` left beside the report is neither used nor
changed. An .ncu-rep is imported with `ncu --import <report> --csv
--print-summary per-kernel`. What the tools print is written as one CSV of
its own. That CSV is not a benchmark CSV: bench summary, bench compare and
bench-plot need test, wallMedian, wallCV and callsPerSecond columns and
refuse it. Read it with a CSV tool.

Inputs:
    *.nsys-rep      Nsight Systems report
    *.ncu-rep       Nsight Compute report
    a directory     every report under it

Usage:
    nsight-parse parse run.nsys-rep --csv summaries.csv
    nsight-parse parse bench-out/   --csv combined.csv

Output, nsys: one row per row of each summary -- per kernel name, per CUDA
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

Output, ncu: one row per launch shape, section and metric: source "ncu",
report "per_kernel", kernel (Kernel Name), then ncu's own columns in snake
case (block_size, grid_size, invocations, section_name, metric_name,
metric_unit, minimum, maximum, average, ...).

Exit status: 0 when every requested report was read; 1 when any was not (a
tool failed or is missing, an input is not a report, a directory holds
none). Each failure is an error line on stderr, and the rows of the reports
that were read are written all the same. A summary with no data, such as the
kernel summary of a report with no kernel, is a warning.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import tempfile
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
    errors: list[str] = field(default_factory=list)

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
    """Parse one or more Nsight report files (or directories of them).

    Every input that cannot be read, and every tool command that fails, is
    an entry in ``errors``; the rows of the reports that were read are kept.
    """
    result = ParseResult()
    for f in _iter_inputs(paths, result):
        if f.suffix == ".nsys-rep":
            _parse_nsys(f, result)
        else:
            _parse_ncu(f, result)
    return result


def _iter_inputs(paths: Iterable[Path], result: ParseResult) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            found = sorted(p.glob("**/*.nsys-rep")) + sorted(p.glob("**/*.ncu-rep"))
            if not found:
                result.errors.append(f"no .nsys-rep or .ncu-rep file under {p}")
            yield from found
        elif p.suffix in {".nsys-rep", ".ncu-rep"} and p.is_file():
            yield p
        else:
            result.errors.append(f"not an .nsys-rep, an .ncu-rep or a directory: {p}")


# =============================== Nsight Systems ==============================


def _parse_nsys(path: Path, result: ParseResult) -> None:
    """Export the report once to a fresh SQLite file, then read each summary from it.

    The export goes to a private temporary directory, never beside the
    report: an export `bench run` already left there can be refused by
    `nsys stats` ("older than input file"), and the report's folder is not
    ours to change.
    """
    with tempfile.TemporaryDirectory(prefix="nsight-parse-") as tmp:
        export = Path(tmp) / (path.stem + ".sqlite")
        run = _run(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--force-overwrite",
                "true",
                "-o",
                str(export),
                str(path),
            ]
        )
        if not run.ok or not export.is_file():
            result.errors.append(f"nsys export failed for {path}: {run.detail}")
            return
        for report in NSYS_REPORTS:
            run = _run(["nsys", "stats", "--report", report, "--format", "csv", str(export)])
            if not run.ok:
                result.errors.append(
                    f"nsys stats --report {report} failed for {path}: {run.detail}"
                )
                continue
            rows = list(_iter_csv_after_header(run.stdout))
            if not rows:
                result.warnings.append(f"{report} has no rows for {path}: {run.detail}")
            for row in rows:
                row.setdefault("kernel", row.pop("Name", row.pop("Range", "")))
                row.setdefault("instances", row.pop("Instances", row.pop("Num Calls", "")))
                row.setdefault("time_total_ns", row.pop("Total Time (ns)", ""))
                row.setdefault("time_avg_ns", row.pop("Avg (ns)", ""))
                row.setdefault("time_pct", row.pop("Time (%)", ""))
                result.rows.append({"source": "nsys", "report": report, **row})


# =============================== Nsight Compute ==============================


def _parse_ncu(path: Path, result: ParseResult) -> None:
    """Import the saved report: one row per launch shape, section and metric."""
    run = _run(["ncu", "--import", str(path), "--csv", "--print-summary", "per-kernel"])
    if not run.ok:
        result.errors.append(f"ncu --import failed for {path}: {run.detail}")
        return
    rows = list(_iter_csv_after_header(run.stdout))
    if not rows:
        result.warnings.append(f"no kernel in {path}: {run.detail}")
    for row in rows:
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

#: Seconds one nsys or ncu command may take; exporting a large report is slow.
TOOL_TIMEOUT_S = 600


@dataclass
class _Run:
    """What one tool command did: whether it succeeded, its output, and a one-line detail."""

    ok: bool
    stdout: str = ""
    detail: str = ""  # why it failed, or the last line it printed


def _last_line(*texts: str) -> str:
    for text in texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "no output"


def _run(cmd: list[str]) -> _Run:
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S, check=False
        )
    except FileNotFoundError:
        return _Run(ok=False, detail=f"{cmd[0]} not found on PATH")
    except subprocess.TimeoutExpired:
        return _Run(ok=False, detail=f"{cmd[0]} did not finish within {TOOL_TIMEOUT_S} s")
    if done.returncode != 0:
        return _Run(
            ok=False,
            detail=f"exit status {done.returncode}: {_last_line(done.stderr, done.stdout)}",
        )
    return _Run(ok=True, stdout=done.stdout, detail=_last_line(done.stdout, done.stderr))


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
            print(f"[nsight-parse] warning: {w}", file=sys.stderr)
        for e in result.errors:
            print(f"[nsight-parse] error: {e}", file=sys.stderr)
        print(f"[nsight-parse] wrote {len(result.rows)} rows to {args.csv}")
        return 1 if result.errors else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
