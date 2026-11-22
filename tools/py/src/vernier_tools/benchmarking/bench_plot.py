#!/usr/bin/env python3
"""
bench_plot.py - Benchmark visualization tool (plotting only)

Generates publication-quality plots and interactive dashboards from benchmark
CSV files produced by the Vernier benchmarking framework.

For CSV analysis, comparison, regression detection, validation, and execution,
use the Rust `bench` tool instead:
    bench summary results.csv
    bench compare baseline.csv candidate.csv
    bench validate

Usage:
    bench-plot plot results.csv --output plots/
    bench-plot dashboard results.csv --output dashboard.html
    bench-plot report results.csv --output report/
    bench-plot scaling 1kb.csv 64kb.csv 1mb.csv --output scaling.html

A file is read as `bench summary` reads it, and refused for the same reasons:
the command stops with one line naming the file and the reason, exit status
1, for a file that cannot be read or is not UTF-8, lacks a required column,
has no row, or has a row whose test name is empty or only whitespace (naming
the line the row starts on). A test name is text, whatever it looks like. A
row whose plotted timing cells do not all hold finite numbers is left out of
every plot, and stderr names it, saying when its every timing cell is empty,
as in a run under Nsight Compute; a file with no such row left is refused,
naming the file. A file without a column a command plots is refused, naming
the command and the column, and so is a test name longer than LONGEST_LABEL
characters, for a plot matplotlib draws; a '$' in a name is drawn as written.

Dependencies:
    pip install pandas matplotlib seaborn
    Optional: pip install plotly (for interactive dashboards; without it,
    dashboard falls back to the report's plots and scaling's .html output to a
    .png beside it)
"""

import argparse
import csv
import io
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Optional: plotly for interactive dashboards
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# Data Loading
# =============================================================================


# The columns besides the timing that the plots read as numbers. Text in one of
# them reads as no number, as pandas's default missing markers did; an infinity
# stays one.
PLOTTED_OTHER_NUMBERS = ("msgBytes", "kernelTimeUs", "transferTimeUs", "speedupVsCpu", "occupancy")

# The columns bench-plot shows or copies as text: the test names label the
# plots, and the report copies the first row's timestamp, git hash and host
# name into metadata.json. Each is read as written, whatever it looks like:
# `007` stays `007`, not 7, and `1234e56` stays `1234e56`, not 1.234e+59.
TEXT_COLUMNS = ("test", "timestamp", "gitHash", "hostname")

# What Rust's str::trim takes for whitespace (Unicode's White_Space property):
# `bench summary` refuses a test name made only of these characters.
WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


def unusable_test_name(name: str):
    """Why a test name, as written, cannot name a row, in `bench summary`'s
    words, or None when it can: an empty name, or one only of whitespace,
    which is shown as Rust's Debug form shows it (`"\\t"`, `"\\u{a0}"`). A
    name with spaces around it can name a row, as a different test from the
    same name without them."""
    if name == "":
        return "the row has no test name"
    if all(c in WHITE_SPACE for c in name):
        shown = "".join(
            {" ": " ", "\t": "\\t", "\n": "\\n", "\r": "\\r"}.get(c, f"\\u{{{ord(c):x}}}")
            for c in name
        )
        return f'test name "{shown}" is only whitespace'
    return None


# The columns every file must have, in the order `bench summary` checks them:
# the first one a file lacks is the one named.
REQUIRED_COLUMNS = ("test", "wallMedian", "wallCV", "callsPerSecond")


def refuse(path, why, line=None):
    """Stop the command with one line naming @p path, and @p line when given,
    and @p why: exit status 1."""
    where = f"{path}, line {line}" if line is not None else f"{path}"
    raise SystemExit(f"bench-plot: {where}: {why}")


def line_at(data: bytes, offset: int) -> int:
    """The line of @p data that byte @p offset is on, counting from 1, as an
    editor counts lines: each CRLF, lone CR and lone LF before it ends one."""
    before = data[:offset]
    return before.count(b"\n") + before.count(b"\r") - before.count(b"\r\n") + 1


def read_text(path: Path) -> str:
    """The text of the file at @p path, its bytes as UTF-8 without a byte
    order mark. A file that cannot be read stops the command with the
    system's reason, and one that is not UTF-8 with the line of its first
    byte that is not, as `bench summary` stops."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as error:
        refuse(path, error.strerror)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        refuse(path, "not valid UTF-8", line_at(data, error.start))
    return text[1:] if text.startswith("\ufeff") else text


def absent_character(text: str) -> str:
    """A character @p text does not hold, from Unicode's private use area."""
    return next(chr(c) for c in range(0xE000, 0x110000) if chr(c) not in text)


def records_of(text: str):
    """The header of @p text and each record after it with the line it
    starts on, as `bench summary`'s reader takes them: a quoted field runs to
    its closing quote or to the end of the file, a blank line is no record
    but counts as a line, as does each line of a record that spans several,
    the header is the first record, after any blank line, and a NUL is a
    character like any other. The fields of both hold `stand_in` for each
    NUL, which the reader of Python before 3.11 refuses; the third value is
    that character, one the text does not hold."""
    stand_in = absent_character(text)
    # A field may be as long as the file.
    csv.field_size_limit(max(csv.field_size_limit(), len(text) + 1))
    reader = csv.reader(io.StringIO(text.replace("\x00", stand_in), newline=""))
    header = next((record for record in reader if record), [])
    records = []
    while True:
        start = reader.line_num + 1
        record = next(reader, None)
        if record is None:
            return header, records, stand_in
        if record:
            records.append((start, record))


def load_csv(path: Path) -> pd.DataFrame:
    """Load and validate a benchmark CSV file, read as `bench summary` reads
    it, refusing it as `bench summary` refuses it.

    A file that cannot be read or is not UTF-8 (`read_text`), lacks a column
    of `REQUIRED_COLUMNS`, has a row whose test name is empty or only
    whitespace (naming the line the row starts on), or has no row, stops the
    command with one line naming the file and the reason. A column the header
    names twice is read from its last place; a row with more fields than the
    header has the rest left unread, and one with fewer, the columns it lacks
    empty. Only an empty field is missing: text that pandas takes for a
    missing value by default (`NA`, `nan`, `NULL`, `None`, ...) stays text, so
    a test of that name keeps it and a timing cell holding it is damaged, not
    empty. A NUL is kept in a name and in the text of `TEXT_COLUMNS`, and a
    number's cell holding one holds no number. The columns of `TEXT_COLUMNS`
    are read as text, those of `PLOTTED_OTHER_NUMBERS` as numbers, and
    `OTHER_NUMBERS_OF_THEIR_KIND` is added for `untimed_rows`.
    """
    header, records, stand_in = records_of(read_text(path))
    for column in REQUIRED_COLUMNS:
        if column not in header:
            refuse(path, f"missing required column '{column}'")
    place = {name: index for index, name in enumerate(header)}
    test = place["test"]
    for line, fields in records:
        why = unusable_test_name(fields[test] if test < len(fields) else "")
        if why is not None:
            refuse(path, why.replace(stand_in, "\x00"), line)
    if not records:
        refuse(path, "no data rows")

    # The records again as CSV text that pandas reads, each column once, the
    # stand-in kept: pandas ends a field at a NUL, which would read the name
    # `A<NUL>B` as `A` and the cell `10.0<NUL>x` as the number 10.0. Each line
    # ends in CRLF: Python before 3.12 quotes a field holding a CR only when
    # the line end holds one, and pandas ends a record at an unquoted CR.
    kept = sorted(place.values())
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow([header[i] for i in kept])
    for _, fields in records:
        writer.writerow([fields[i] for i in kept if i < len(fields)])
    text = out.getvalue()
    df = pd.read_csv(
        io.StringIO(text),
        keep_default_na=False,
        na_values=[""],
        dtype=dict.fromkeys(TEXT_COLUMNS, str),
    )
    for column in PLOTTED_OTHER_NUMBERS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df[OTHER_NUMBERS_OF_THEIR_KIND] = other_numbers_of_their_kind(text, df)
    # The column names and the text bench-plot shows hold their NULs again; in
    # a number's cell the stand-in stays, so that cell holds no number.
    if stand_in in text:
        df.columns = [name.replace(stand_in, "\x00") for name in df.columns]
        for column in TEXT_COLUMNS:
            if column in df.columns:
                df[column] = df[column].str.replace(stand_in, "\x00", regex=False)
    return df


def load_csvs_from_list(paths: List[Path]) -> pd.DataFrame:
    """Load multiple CSVs and concatenate."""
    frames = []
    for p in paths:
        df = load_csv(p)
        df["_source"] = p.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Rows Without Timing
# =============================================================================

# A row's timing cells. The benchmark leaves every one of them empty in a row
# it did not time, as it does under Nsight Compute (ncu), whose instrumentation
# adds to what the benchmark would measure.
TIMING_COLUMNS = (
    "wallMedian",
    "wallP10",
    "wallP90",
    "wallP99",
    "wallP999",
    "wallMin",
    "wallMax",
    "wallMean",
    "wallStddev",
    "wallCV",
    "callsPerSecond",
    "stable",
)

# The timing cells the plots and the report's summary read.
PLOTTED_TIMING = (
    "wallMedian",
    "wallP10",
    "wallP90",
    "wallMin",
    "wallMax",
    "wallCV",
    "callsPerSecond",
)

# A row's other numbers, which the benchmark writes in every row it times or
# leaves untimed, and the kind `bench summary` reads each as: the counts as
# whole numbers from 0 to 4294967295 (u32), cvThreshold as a finite number
# (f64), each parsed as written, untrimmed.
COUNTS = ("cycles", "repeats", "warmup", "threads", "msgBytes")
OTHER_NUMBERS = COUNTS + ("cvThreshold",)
LARGEST_COUNT = 4294967295
LARGEST_COUNT_DIGITS = len(str(LARGEST_COUNT))
# What Rust's str::parse takes for a u32 and for an f64 (its literal grammar).
WHOLE_NUMBER = re.compile(r"\+?[0-9]+")
FLOAT_LITERAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")

# The column load_csv adds, true where every other number of the row holds a
# value of its kind; timed_rows leaves it out of the rows it returns.
OTHER_NUMBERS_OF_THEIR_KIND = "_otherNumbersOfTheirKind"


def of_its_kind(column: str, text) -> bool:
    """Whether @p text, a cell of the other number @p column as written, holds
    a value of the kind `bench summary` reads that column as. An empty field
    holds none. A count's digits are read whatever their number: leading zeros
    do not count, and more significant digits than the largest count has make
    it too large, so int() is never asked for a longer number than that."""
    if not isinstance(text, str):
        return False
    if column in COUNTS:
        if WHOLE_NUMBER.fullmatch(text) is None:
            return False
        digits = text.lstrip("+").lstrip("0")
        return len(digits) <= LARGEST_COUNT_DIGITS and int(digits or "0") <= LARGEST_COUNT
    return FLOAT_LITERAL.fullmatch(text) is not None and math.isfinite(float(text))


def other_numbers_of_their_kind(text: str, df: pd.DataFrame) -> pd.Series:
    """For each row of @p df, read from @p text, whether every other number
    the file has holds a value of its kind (`of_its_kind`). The cells are read
    again as text: pandas's own reading takes `1e3`, `1.0`, ` 5` and `+5` alike
    for numbers, where `bench summary` takes only some of them."""
    present = [c for c in OTHER_NUMBERS if c in df.columns]
    if not present:
        return pd.Series(True, index=df.index)
    written = pd.read_csv(
        io.StringIO(text), keep_default_na=False, na_values=[""], dtype=str, usecols=present
    )
    return written.apply(lambda cells: cells.map(lambda text: of_its_kind(cells.name, text))).all(
        axis=1
    )


# What may explain a row of the untimed shape. The file cannot say what ran, so
# this says what the shape is like, not that the row came from such a run.
UNTIMED_REASON = (
    "every timing cell is empty, as in a run under Nsight Compute (ncu), "
    "which the benchmark does not time"
)


def numbers(df: pd.DataFrame, columns) -> pd.DataFrame:
    """@p df's @p columns (those it has) as finite numbers: a cell that holds
    none, or holds an infinity, is NaN."""
    present = [c for c in columns if c in df.columns]
    values = df[present].apply(pd.to_numeric, errors="coerce")
    return values.mask(values.abs() == float("inf"))


def untimed_rows(df: pd.DataFrame) -> pd.Series:
    """Whether each row of @p df, as load_csv gives it, has the shape the
    benchmark gives a row it did not time: every timing cell the file has is
    empty, and every other number it has holds a value of its kind."""
    timing = [c for c in TIMING_COLUMNS if c in df.columns]
    if not timing:
        return pd.Series(False, index=df.index)
    empty = df[timing].isna().all(axis=1)
    return empty & df[OTHER_NUMBERS_OF_THEIR_KIND]


def timed_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """The rows of @p df whose plotted timing cells all hold finite numbers,
    which every plot draws, those cells as numbers.

    Every other row is left out, and stderr names it: a row of the untimed
    shape (`untimed_rows`) with what may explain it, any other with an empty
    or damaged timing cell. When no row is left, the command stops with a
    message naming @p source, as `bench summary` refuses such a row.
    """
    timing = numbers(df, PLOTTED_TIMING)
    unplottable = timing.isna().any(axis=1)
    untimed = untimed_rows(df) & unplottable
    damaged = unplottable & ~untimed
    if unplottable.all():
        reason = (
            f"{UNTIMED_REASON}; if the run was under ncu, plot a run without it"
            if untimed.all()
            else "every row has an empty or damaged timing cell"
        )
        raise SystemExit(f"bench-plot: {source}: no row has timing: {reason}")
    for rows, why in (
        (untimed, f"rows without timing ({UNTIMED_REASON})"),
        (damaged, "rows with an empty or damaged timing cell"),
    ):
        if rows.any():
            names = ", ".join(df.loc[rows, "test"].astype(str))
            print(
                f"bench-plot: {source}: left out {int(rows.sum())} {why}: {names}",
                file=sys.stderr,
            )
    kept = df[~unplottable].drop(columns=OTHER_NUMBERS_OF_THEIR_KIND)
    kept[timing.columns] = timing[~unplottable]
    return kept


# =============================================================================
# Plot Style
# =============================================================================


def setup_plot_style():
    """Configure matplotlib style for publication-quality plots."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    # A test name is drawn as written: a '$' in it starts no mathtext, which a
    # name such as `$$` would make matplotlib refuse to draw.
    plt.rcParams["text.parse_math"] = False


# =============================================================================
# CPU Plots
# =============================================================================


def plot_latency_distribution(df: pd.DataFrame, output_path: Path):
    """Box plot showing latency distribution (p10/median/p90)."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    tests = df["test"].tolist()
    positions = list(range(len(tests)))

    box_data = []
    for _, row in df.iterrows():
        box_data.append(
            [
                row["wallMin"],
                row["wallP10"],
                row["wallMedian"],
                row["wallP90"],
                row["wallMax"],
            ]
        )

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tests, rotation=45, ha="right")
    ax.set_ylabel("Latency (us/call)")
    ax.set_title("Latency Distribution (p10/median/p90)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved latency distribution: {output_path}")


def plot_throughput_bars(df: pd.DataFrame, output_path: Path):
    """Bar chart of throughput (calls per second)."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    df_sorted = df.sort_values("callsPerSecond", ascending=True)

    bars = ax.barh(
        df_sorted["test"],
        df_sorted["callsPerSecond"],
        color="seagreen",
        alpha=0.8,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Throughput (calls/second)")
    ax.set_title("Throughput Comparison")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved throughput bars: {output_path}")


def plot_cv_analysis(df: pd.DataFrame, output_path: Path):
    """Plot coefficient of variation (jitter analysis)."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    df_plot = df.copy()
    df_plot["cv_pct"] = df_plot["wallCV"] * 100
    df_plot = df_plot.sort_values("cv_pct", ascending=True)

    bars = ax.barh(df_plot["test"], df_plot["cv_pct"], color="coral", alpha=0.8)

    for i, bar in enumerate(bars):
        if df_plot.iloc[i]["cv_pct"] > 10:
            bar.set_color("darkred")

    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_title("Measurement Stability (Jitter Analysis)")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved jitter analysis: {output_path}")


def plot_comparison_waterfall(comparison_df: pd.DataFrame, output_path: Path):
    """Waterfall chart showing performance delta between baseline and candidate."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    df_sorted = comparison_df.sort_values("delta_pct", ascending=True)

    colors = []
    for _, row in df_sorted.iterrows():
        if row.get("is_regression", False):
            colors.append("crimson")
        elif row.get("is_improvement", False):
            colors.append("seagreen")
        else:
            colors.append("steelblue")

    ax.barh(df_sorted["test"], df_sorted["delta_pct"], color=colors, alpha=0.8)

    ax.set_xlabel("Performance Change (%)")
    ax.set_title("Performance Comparison (negative = faster)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved comparison waterfall: {output_path}")


# =============================================================================
# GPU Plots
# =============================================================================


def plot_gpu_breakdown(df: pd.DataFrame, output_path: Path):
    """GPU kernel vs transfer time breakdown."""
    setup_plot_style()
    gpu_df = df[df["gpuModel"].notna()].copy()
    if gpu_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    tests = gpu_df["test"].tolist()
    kernel_times = gpu_df["kernelTimeUs"].tolist()
    transfer_times = gpu_df.get("transferTimeUs", pd.Series([0] * len(gpu_df))).tolist()

    x = range(len(tests))
    ax.barh(
        [i + 0.2 for i in x],
        kernel_times,
        height=0.4,
        label="Kernel",
        color="steelblue",
        alpha=0.8,
    )
    ax.barh(
        [i - 0.2 for i in x],
        transfer_times,
        height=0.4,
        label="Transfer",
        color="coral",
        alpha=0.8,
    )

    ax.set_yticks(list(x))
    ax.set_yticklabels(tests)
    ax.set_xlabel("Time (us)")
    ax.set_title("GPU Time Breakdown")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved GPU breakdown: {output_path}")


def plot_gpu_speedup(df: pd.DataFrame, output_path: Path):
    """GPU speedup vs CPU scatter plot."""
    setup_plot_style()
    gpu_df = df[df["speedupVsCpu"].notna()].copy()
    if gpu_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(gpu_df["test"], gpu_df["speedupVsCpu"], color="seagreen", alpha=0.8)
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="1x (CPU baseline)")
    ax.set_xlabel("Speedup vs CPU")
    ax.set_title("GPU Speedup over CPU")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved GPU speedup: {output_path}")


def plot_gpu_occupancy(df: pd.DataFrame, output_path: Path):
    """GPU occupancy percentage bar chart."""
    setup_plot_style()
    gpu_df = df[df["occupancy"].notna()].copy()
    if gpu_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    occ_pct = gpu_df["occupancy"] * 100
    bars = ax.barh(gpu_df["test"], occ_pct, color="steelblue", alpha=0.8)

    for i, bar in enumerate(bars):
        if occ_pct.iloc[i] < 50:
            bar.set_color("coral")

    ax.set_xlabel("Occupancy (%)")
    ax.set_title("GPU Kernel Occupancy")
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved GPU occupancy: {output_path}")


# =============================================================================
# Scaling Analysis
# =============================================================================


# The size a test name ends with: a number, then K, M or G for KiB, MiB or GiB
# and an optional B, in any case. The number starts where its run of digits
# does, so a long run is read once, not once from each of its digits.
PAYLOAD_SIZE = re.compile(r"(?<!\d)(\d+)([KMG]?)B?$", re.IGNORECASE)
# The largest payload a name's size is read as: the largest byte count a 64-bit
# size holds.
LARGEST_PAYLOAD = 2**64 - 1
LARGEST_PAYLOAD_DIGITS = len(str(LARGEST_PAYLOAD))


def extract_payload_size(test_name, msg_bytes=0):
    """The payload size of a row, for the scaling plots: the size its test
    name ends with (`PAYLOAD_SIZE`), else @p msg_bytes. A name that is not
    text holds no size (load_csv gives every name as text and refuses a file
    with an empty one; another caller may pass anything), and neither does
    one whose size would be larger than LARGEST_PAYLOAD bytes: that number is
    not a byte count. Leading zeros do not count, so int() is never asked for
    more digits than LARGEST_PAYLOAD has, and no name makes this raise."""
    if not isinstance(test_name, str):
        return msg_bytes
    match = PAYLOAD_SIZE.search(test_name)
    if match is None:
        return msg_bytes
    digits = match.group(1)
    first = next((i for i, c in enumerate(digits) if unicodedata.decimal(c)), len(digits))
    if len(digits) - first > LARGEST_PAYLOAD_DIGITS:
        return msg_bytes
    size = int(digits[first:] or "0")
    suffix = match.group(2).upper()
    if suffix == "K":
        size *= 1024
    elif suffix == "M":
        size *= 1024 * 1024
    elif suffix == "G":
        size *= 1024 * 1024 * 1024
    return size if size <= LARGEST_PAYLOAD else msg_bytes


def plot_scaling_analysis(df: pd.DataFrame, output_path: Path, interactive: bool = False):
    """Generate scaling analysis plots."""
    setup_plot_style()

    df = df.copy()
    df["payloadBytes"] = df.apply(
        lambda row: extract_payload_size(row["test"], row.get("msgBytes", 0)), axis=1
    )

    if interactive and HAS_PLOTLY:
        _plot_scaling_plotly(df, output_path)
    else:
        _plot_scaling_static(df, output_path)


def _plot_scaling_static(df: pd.DataFrame, output_path: Path):
    """Static matplotlib scaling plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Throughput vs payload size
    ax = axes[0, 0]
    for test in df["test"].unique():
        test_df = df[df["test"] == test].sort_values("payloadBytes")
        if len(test_df) > 1:
            ax.plot(test_df["payloadBytes"], test_df["callsPerSecond"], "o-", label=test)
    ax.set_xlabel("Payload Size (bytes)")
    ax.set_ylabel("Throughput (calls/sec)")
    ax.set_title("Throughput Scaling")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Latency vs payload size
    ax = axes[0, 1]
    for test in df["test"].unique():
        test_df = df[df["test"] == test].sort_values("payloadBytes")
        if len(test_df) > 1:
            ax.plot(test_df["payloadBytes"], test_df["wallMedian"], "o-", label=test)
    ax.set_xlabel("Payload Size (bytes)")
    ax.set_ylabel("Latency (us/call)")
    ax.set_title("Latency Scaling")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # CV vs payload size
    ax = axes[1, 0]
    for test in df["test"].unique():
        test_df = df[df["test"] == test].sort_values("payloadBytes")
        if len(test_df) > 1:
            ax.plot(test_df["payloadBytes"], test_df["wallCV"] * 100, "o-", label=test)
    ax.set_xlabel("Payload Size (bytes)")
    ax.set_ylabel("CV (%)")
    ax.set_title("Stability vs Payload Size")
    ax.set_xscale("log")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis("off")
    summary_data = []
    for test in sorted(df["test"].unique()):
        test_df = df[df["test"] == test]
        summary_data.append(
            [
                test[:30],
                f"{test_df['wallMedian'].median():.3f}",
                f"{test_df['callsPerSecond'].median():.0f}",
                f"{test_df['wallCV'].median() * 100:.1f}%",
            ]
        )
    if summary_data:
        table = ax.table(
            cellText=summary_data,
            colLabels=["Test", "Median (us)", "Calls/sec", "CV"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved scaling analysis: {output_path}")


def _plot_scaling_plotly(df: pd.DataFrame, output_path: Path):
    """Interactive plotly scaling dashboard."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Throughput Scaling",
            "Latency Scaling",
            "Stability vs Payload",
            "Summary",
        ],
    )

    for test in df["test"].unique():
        test_df = df[df["test"] == test].sort_values("payloadBytes")
        if len(test_df) > 1:
            fig.add_trace(
                go.Scatter(
                    x=test_df["payloadBytes"],
                    y=test_df["callsPerSecond"],
                    mode="lines+markers",
                    name=test,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=test_df["payloadBytes"],
                    y=test_df["wallMedian"],
                    mode="lines+markers",
                    name=test,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=test_df["payloadBytes"],
                    y=test_df["wallCV"] * 100,
                    mode="lines+markers",
                    name=test,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_xaxes(type="log")
    fig.update_layout(height=800, title_text="Payload Scaling Analysis")

    fig.write_html(str(output_path))
    print(f"  Saved interactive scaling dashboard: {output_path}")


# =============================================================================
# Interactive Dashboard
# =============================================================================


def create_interactive_dashboard(df: pd.DataFrame, output_path: Path):
    """Generate interactive HTML dashboard with Plotly."""
    if not HAS_PLOTLY:
        print("  plotly not installed. Install with: pip install plotly")
        print("  Falling back to static report.")
        generate_report(df, output_path.parent)
        return

    has_gpu = has_gpu_rows(df)

    if has_gpu:
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[
                "Latency Distribution",
                "Throughput",
                "Jitter (CV%)",
                "GPU Breakdown",
                "GPU Speedup",
                "GPU Occupancy",
                "Transfer Overhead",
                "Memory Bandwidth",
            ],
        )
    else:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Latency Distribution",
                "Throughput",
                "Jitter (CV%)",
                "P10/P90 Range",
                "Calls/Second Distribution",
                "Summary Stats",
            ],
        )

    # Latency box-like trace
    fig.add_trace(
        go.Bar(
            x=df["test"],
            y=df["wallMedian"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (df["wallP90"] - df["wallMedian"]).tolist(),
                "arrayminus": (df["wallMedian"] - df["wallP10"]).tolist(),
            },
            name="Median (p10-p90)",
            marker_color="steelblue",
        ),
        row=1,
        col=1,
    )

    # Throughput
    df_sorted = df.sort_values("callsPerSecond", ascending=False)
    fig.add_trace(
        go.Bar(
            x=df_sorted["test"],
            y=df_sorted["callsPerSecond"],
            name="Calls/sec",
            marker_color="seagreen",
        ),
        row=1,
        col=2,
    )

    # CV%
    fig.add_trace(
        go.Bar(
            x=df["test"],
            y=df["wallCV"] * 100,
            name="CV%",
            marker_color=["darkred" if cv > 10 else "coral" for cv in df["wallCV"] * 100],
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800 if not has_gpu else 1200,
        title_text="Benchmark Analysis Dashboard",
        showlegend=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"  Saved interactive dashboard: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive analysis report with plots and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_distribution(df, output_dir / "latency_distribution.png")
    plot_throughput_bars(df, output_dir / "throughput.png")
    plot_cv_analysis(df, output_dir / "jitter_analysis.png")

    # GPU plots if data present
    if has_gpu_rows(df):
        plot_gpu_breakdown(df, output_dir / "gpu_breakdown.png")
        plot_gpu_speedup(df, output_dir / "gpu_speedup.png")
        plot_gpu_occupancy(df, output_dir / "gpu_occupancy.png")

    # Summary stats CSV. Python before 3.11 writes a NUL, which a test name may
    # hold, only when an escape character is set; one no name holds is never
    # written. Python before 3.12 quotes a name holding a CR only when the line
    # end holds one: such a file's lines end in CRLF, every other file's as
    # before.
    summary = df[["test", "wallMedian", "wallP10", "wallP90", "wallCV", "callsPerSecond"]].copy()
    names = "".join(summary["test"])
    escape = absent_character(names) if "\x00" in names else None
    summary.to_csv(
        output_dir / "summary_stats.csv",
        index=False,
        escapechar=escape,
        lineterminator="\r\n" if "\r" in names else None,
    )

    # Metadata JSON
    metadata = {
        "tests": len(df),
        "columns": list(df.columns),
    }
    # The first row's text as written (TEXT_COLUMNS); an empty cell is null,
    # where NaN would not be JSON.
    for column in ("timestamp", "gitHash", "hostname"):
        if column in df.columns:
            value = df[column].iloc[0]
            metadata[column] = value if isinstance(value, str) else None

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  Report generated: {output_dir}")


# =============================================================================
# Command Handlers
# =============================================================================


# The columns the plots read beyond REQUIRED_COLUMNS: the latency
# distribution's (the report's summary reads two of them), and, for rows with a
# GPU model, the GPU plots'.
LATENCY_COLUMNS = ("wallMin", "wallP10", "wallP90", "wallMax")
GPU_COLUMNS = ("kernelTimeUs", "speedupVsCpu", "occupancy")


def has_gpu_rows(df: pd.DataFrame) -> bool:
    """Whether a row of @p df names a GPU model, which draws the GPU plots."""
    return "gpuModel" in df.columns and df["gpuModel"].notna().any()


def needs(df: pd.DataFrame, path, command: str, columns, what: str):
    """Stop @p command, naming the first of @p columns @p df lacks and @p
    what reads it, when it lacks one: a file `bench summary` reads can lack a
    column only a plot reads."""
    for column in columns:
        if column not in df.columns:
            refuse(path, f"{command} needs the {column} column, for {what}")


# The longest test name a matplotlib plot draws: a much longer label makes a
# figure wider or taller than the 2^16 pixels matplotlib can write.
LONGEST_LABEL = 1000


def drawable_names(df: pd.DataFrame, source, command: str):
    """Stop @p command when a test name of @p df, the rows it draws with
    matplotlib from @p source, is longer than LONGEST_LABEL characters,
    naming the start of the first such name and its length."""
    lengths = df["test"].str.len()
    if (lengths > LONGEST_LABEL).any():
        name = df["test"][lengths > LONGEST_LABEL].iloc[0]
        refuse(
            source,
            f"{command} draws a test name of at most {LONGEST_LABEL} characters, and the one "
            f"starting '{name[:40]}' has {len(name)}",
        )


def rows_to_draw(args, command: str, matplotlib_plots: bool = True) -> pd.DataFrame:
    """The timed rows of the file @p args names (`timed_rows`), after the
    columns @p command draws are found: the latency distribution's, the GPU
    plots' when a timed row names a GPU model, and test names short enough to
    draw, when it draws @p matplotlib_plots; otherwise (the plotly dashboard)
    the latency bars' p10 and p90."""
    path = Path(args.csv)
    df = load_csv(path)
    columns = LATENCY_COLUMNS if matplotlib_plots else ("wallP10", "wallP90")
    needs(df, path, command, columns, "the latency plot")
    df = timed_rows(df, args.csv)
    if matplotlib_plots:
        if has_gpu_rows(df):
            needs(df, path, command, GPU_COLUMNS, "the GPU plots of its rows with a gpuModel")
        drawable_names(df, path, command)
    return df


def cmd_plot(args):
    """Generate standalone plots from CSV."""
    df = rows_to_draw(args, "plot")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_distribution(df, output_dir / "latency_distribution.png")
    plot_throughput_bars(df, output_dir / "throughput.png")
    plot_cv_analysis(df, output_dir / "jitter_analysis.png")

    # GPU plots if data present
    if has_gpu_rows(df):
        plot_gpu_breakdown(df, output_dir / "gpu_breakdown.png")
        plot_gpu_speedup(df, output_dir / "gpu_speedup.png")
        plot_gpu_occupancy(df, output_dir / "gpu_occupancy.png")


def cmd_dashboard(args):
    """Generate interactive HTML dashboard."""
    # Without plotly the dashboard is the report's plots.
    df = rows_to_draw(args, "dashboard", matplotlib_plots=not HAS_PLOTLY)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_interactive_dashboard(df, output_path)


def cmd_report(args):
    """Generate full analysis report."""
    df = rows_to_draw(args, "report")
    generate_report(df, Path(args.output))


def cmd_scaling(args):
    """Generate payload size scaling analysis."""
    csv_paths = [Path(p) for p in args.csvs]
    source = ", ".join(args.csvs)
    df_combined = timed_rows(load_csvs_from_list(csv_paths), source)
    output_path = Path(args.output)
    interactive = output_path.suffix == ".html" and HAS_PLOTLY
    if not interactive:
        drawable_names(df_combined, source, "scaling")
    if output_path.suffix == ".html" and not HAS_PLOTLY:
        # matplotlib writes no HTML: the static plot goes beside the page.
        output_path = output_path.with_suffix(".png")
        print("  plotly not installed. Install with: pip install plotly")
        print(f"  Falling back to a static plot: {output_path}")
    plot_scaling_analysis(df_combined, output_path, interactive=interactive)


# =============================================================================
# Main CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark plotting and visualization (charts only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "For analysis, comparison, and validation use the Rust 'bench' tool:\n"
            "  bench summary results.csv\n"
            "  bench compare baseline.csv candidate.csv\n"
            "  bench validate\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots from CSV")
    plot_parser.add_argument("csv", help="Path to benchmark CSV file")
    plot_parser.add_argument("--output", default="plots/", help="Output directory")

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Generate interactive HTML dashboard"
    )
    dashboard_parser.add_argument("csv", help="Path to benchmark CSV file")
    dashboard_parser.add_argument("--output", default="dashboard.html", help="Output HTML file")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive analysis report")
    report_parser.add_argument("csv", help="Path to benchmark CSV file")
    report_parser.add_argument("--output", default="report/", help="Output directory")

    # Scaling command
    scaling_parser = subparsers.add_parser(
        "scaling", help="Payload size scaling analysis across multiple CSVs"
    )
    scaling_parser.add_argument("csvs", nargs="+", help="CSV files with different payload sizes")
    scaling_parser.add_argument(
        "--output",
        default="scaling_dashboard.html",
        help="Output file (HTML for interactive, PNG for static)",
    )

    args = parser.parse_args()

    commands = {
        "plot": cmd_plot,
        "dashboard": cmd_dashboard,
        "report": cmd_report,
        "scaling": cmd_scaling,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
