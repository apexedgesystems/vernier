"""Tests for bench_plot data loading and parsing functions."""

import io
import json
import os
import sys
import textwrap
import time
from csv import reader as csv_records

import pandas as pd
import pytest
from vernier_tools.benchmarking import bench_plot
from vernier_tools.benchmarking.bench_plot import (
    extract_payload_size,
    load_csv,
    load_csvs_from_list,
)

from .malformed_csv import CORPUS, write

# =============================================================================
# Fixtures
# =============================================================================


# Long enough that Python's int() refuses the digits (4300 at most).
LONG = 5000

MINIMAL_CSV = textwrap.dedent(
    """\
    test,wallMedian,wallCV,callsPerSecond,wallMin,wallP10,wallP90,wallMax
    Noop,0.001,0.05,1000000,0.0005,0.0008,0.0015,0.002
    Add1K,0.5,0.03,2000,0.4,0.45,0.55,0.6
"""
)


@pytest.fixture()
def csv_file(tmp_path):
    """Write minimal valid CSV and return its path."""
    p = tmp_path / "results.csv"
    p.write_text(MINIMAL_CSV)
    return p


@pytest.fixture()
def csv_file_b(tmp_path):
    """Second CSV for multi-file tests."""
    p = tmp_path / "results_b.csv"
    p.write_text(
        textwrap.dedent(
            """\
            test,wallMedian,wallCV,callsPerSecond,wallMin,wallP10,wallP90,wallMax
            Copy4K,1.2,0.08,833,1.0,1.05,1.35,1.5
        """
        )
    )
    return p


# =============================================================================
# load_csv
# =============================================================================


def test_load_csv_valid(csv_file):
    df = load_csv(csv_file)
    assert len(df) == 2
    assert list(df["test"]) == ["Noop", "Add1K"]


def test_load_csv_file_not_found(tmp_path):
    p = tmp_path / "nonexistent.csv"
    with pytest.raises(SystemExit) as stopped:
        load_csv(p)
    assert str(stopped.value) == f"bench-plot: {p}: No such file or directory"


def test_load_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("test,wallMedian\nFoo,1.0\n")
    with pytest.raises(SystemExit) as stopped:
        load_csv(p)
    assert str(stopped.value) == f"bench-plot: {p}: missing required column 'wallCV'"


def test_load_csv_empty_data(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("test,wallMedian,wallCV,callsPerSecond\n")
    with pytest.raises(SystemExit) as stopped:
        load_csv(p)
    assert str(stopped.value) == f"bench-plot: {p}: no data rows"


# =============================================================================
# load_csvs_from_list
# =============================================================================


def test_load_csvs_concatenates(csv_file, csv_file_b):
    df = load_csvs_from_list([csv_file, csv_file_b])
    assert len(df) == 3
    assert "_source" in df.columns
    assert set(df["_source"]) == {"results", "results_b"}


def test_load_csvs_single_file(csv_file):
    df = load_csvs_from_list([csv_file])
    assert len(df) == 2
    assert df["_source"].iloc[0] == "results"


# =============================================================================
# extract_payload_size
# =============================================================================


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Send1KB", 1024),
        ("Send64KB", 64 * 1024),
        ("Send1MB", 1024 * 1024),
        ("Send2GB", 2 * 1024 * 1024 * 1024),
        ("Send256B", 256),
        ("Send1024", 1024),
        ("NoMatch", 0),
    ],
)
def test_extract_payload_size_from_name(name, expected):
    assert extract_payload_size(name) == expected


def test_extract_payload_size_fallback_to_msg_bytes():
    assert extract_payload_size("NoMatch", msg_bytes=4096) == 4096


def test_extract_payload_size_case_insensitive():
    assert extract_payload_size("send1kb") == 1024
    assert extract_payload_size("SEND1KB") == 1024


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Send" + "9" * LONG, 64),
        ("Send" + "9" * LONG + "KB", 64),
        ("Send" + "9" * 400, 64),
        ("Send" + "0" * LONG + "5", 5),
        ("Send" + "0" * LONG + "1K", 1024),
        ("Send" + "\u0660" * LONG + "\u0665", 5),
        ("Send18446744073709551615", 2**64 - 1),
        ("Send18446744073709551616", 64),
        ("Send17179869183G", (2**34 - 1) * 2**30),
        ("Send17179869184G", 64),
    ],
    ids=[
        "nines",
        "nines-kb",
        "400-nines",
        "zeros-then-5",
        "zeros-then-1k",
        "arabic-indic-zeros-then-5",
        "largest",
        "one-past-largest",
        "largest-whole-g",
        "one-g-past-largest",
    ],
)
def test_extract_payload_size_of_a_number_too_large_for_a_byte_count(name, expected):
    """A name whose size would be larger than the largest byte count a 64-bit
    size holds has none, msgBytes standing in, however many digits it has;
    leading zeros, in any script, do not count."""
    assert extract_payload_size(name, msg_bytes=64) == expected


@pytest.mark.parametrize(
    "name", [float("nan"), None, 1024, 1024.0], ids=["nan", "none", "int", "float"]
)
def test_extract_payload_size_of_a_name_that_is_not_text(name):
    """An empty test cell reads as NaN, and names that all read as numbers as
    numbers: such a name holds no size."""
    assert extract_payload_size(name, msg_bytes=64) == 64


def test_extract_payload_size_reads_a_long_run_of_digits_once():
    """A name holding a run of 20000 digits that is not its size is read in
    well under a second."""
    started = time.perf_counter()
    assert extract_payload_size("Send" + "1" * 20000 + "X", msg_bytes=64) == 64
    assert time.perf_counter() - started < 1.0


# =============================================================================
# Rows without timing (a run under Nsight Compute)
# =============================================================================

# The columns a benchmark CSV starts with, and two rows a run under Nsight
# Compute wrote: every timing cell empty, the configuration and cvThreshold
# kept (a case timed through the CPU harness, then a GPU kernel).
CAPTURE_HEADER = (
    "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,"
    "wallMedian,wallP10,wallP90,wallP99,wallP999,wallMin,wallMax,wallMean,wallStddev,"
    "wallCV,callsPerSecond,stable,cvThreshold,gpuModel,kernelTimeUs,occupancy"
)
UNTIMED_ROWS = [
    "Saxpy.EndToEnd,2,2,1,1,64,0,0,INFO,,,,,,,,,,,,,0.1,,,",
    "Saxpy.Kernel,2,2,1,1,64,0,0,INFO,,,,,,,,,,,,,0.1,Generic GPU,,1.0",
]
TIMED_ROW = (
    "Saxpy.Plain,2,2,1,1,64,0,0,INFO,3082.75,3082.15,3083.35,3083.49,3083.5,3082,3083.5,"
    "3082.75,0.75,0.0002,324.4,1,0.1,,,"
)


def write_csv(path, rows):
    """Write @p rows under the capture's header to @p path and return it."""
    path.write_text("\n".join([CAPTURE_HEADER] + rows) + "\n")
    return path


@pytest.fixture()
def recorded_plots(monkeypatch):
    """Each plot function, recording the tests it was given instead of drawing."""
    drawn = {}

    def recorder(name):
        def record(df, output_path, *args, **kwargs):
            drawn.setdefault(name, []).append(list(df["test"]))

        return record

    for name in (
        "plot_latency_distribution",
        "plot_throughput_bars",
        "plot_cv_analysis",
        "plot_gpu_breakdown",
        "plot_gpu_speedup",
        "plot_gpu_occupancy",
        "plot_scaling_analysis",
    ):
        monkeypatch.setattr(bench_plot, name, recorder(name))
    # The dashboard draws with plotly when it is installed; without it, as in
    # the tools' environment, it falls back to the report's plots.
    monkeypatch.setattr(bench_plot, "HAS_PLOTLY", False)
    return drawn


def run_main(monkeypatch, argv):
    """bench-plot's main() with @p argv."""
    monkeypatch.setattr(sys, "argv", ["bench-plot"] + argv)
    bench_plot.main()


@pytest.mark.parametrize("command", ["plot", "report", "dashboard"])
def test_a_capture_is_refused_naming_the_file(tmp_path, monkeypatch, recorded_plots, command):
    """A file whose every row is untimed stops the command, naming the file and why."""
    csv = write_csv(tmp_path / "capture.csv", UNTIMED_ROWS)
    output = tmp_path / "out" / ("d.html" if command == "dashboard" else "")
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, [command, str(csv), "--output", str(output)])
    assert str(stopped.value) == (
        f"bench-plot: {csv}: no row has timing: every timing cell is empty, as in a run under "
        "Nsight Compute (ncu), which the benchmark does not time; if the run was under ncu, "
        "plot a run without it"
    )
    assert recorded_plots == {}
    assert not (tmp_path / "out" / "summary_stats.csv").exists()


def test_scaling_refuses_files_without_timing(tmp_path, monkeypatch, recorded_plots):
    """Scaling stops when none of its files has a timed row, naming them."""
    first = write_csv(tmp_path / "a.csv", UNTIMED_ROWS[:1])
    second = write_csv(tmp_path / "b.csv", UNTIMED_ROWS[1:])
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, ["scaling", str(first), str(second)])
    assert str(stopped.value).startswith(f"bench-plot: {first}, {second}: no row has timing: ")
    assert recorded_plots == {}


@pytest.mark.parametrize("command", ["plot", "report", "dashboard"])
def test_untimed_rows_are_left_out_and_named(
    tmp_path, monkeypatch, capsys, recorded_plots, command
):
    """In a file mixing timed and untimed rows, every plot gets the timed rows
    only, and stderr names the rows left out."""
    csv = write_csv(tmp_path / "mixed.csv", [TIMED_ROW] + UNTIMED_ROWS)
    output = tmp_path / "out" / ("d.html" if command == "dashboard" else "")
    run_main(monkeypatch, [command, str(csv), "--output", str(output)])
    assert recorded_plots["plot_latency_distribution"] == [["Saxpy.Plain"]]
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
    assert recorded_plots["plot_cv_analysis"] == [["Saxpy.Plain"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 2 rows without timing (every timing cell is empty, as in "
        "a run under Nsight Compute (ncu), which the benchmark does not time): Saxpy.EndToEnd, "
        "Saxpy.Kernel\n"
    )
    if command != "plot":
        summary = (tmp_path / "out" / "summary_stats.csv").read_text().splitlines()
        assert [line.split(",")[0] for line in summary] == ["test", "Saxpy.Plain"]


def test_rows_that_cannot_be_plotted_are_left_out_without_the_ncu_note(
    tmp_path, monkeypatch, capsys, recorded_plots
):
    """A row with some timing cells empty, rows whose timing is all empty but
    whose cycles cell is damaged or whose cvThreshold is infinite, and rows
    whose median is text or infinite are left out too, and named without the
    Nsight Compute note: their shape is not the one the benchmark writes for
    a row it did not time."""
    partly = TIMED_ROW.replace("Saxpy.Plain", "Saxpy.Partly").replace(",0.0002,", ",,")
    damaged = UNTIMED_ROWS[0].replace("Saxpy.EndToEnd,2,", "Saxpy.Damaged,x,")
    unbounded = (
        UNTIMED_ROWS[0].replace("Saxpy.EndToEnd", "Saxpy.Unbounded").replace(",0.1,", ",inf,")
    )
    malformed = TIMED_ROW.replace("Saxpy.Plain", "Saxpy.Malformed").replace(
        ",3082.75,3082.15,", ",garbage,3082.15,"
    )
    infinite = TIMED_ROW.replace("Saxpy.Plain", "Saxpy.Infinite").replace(
        ",3082.75,3082.15,", ",inf,3082.15,"
    )
    rows = [TIMED_ROW, partly, damaged, unbounded, malformed, infinite]
    csv = write_csv(tmp_path / "rows.csv", rows)
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 5 rows with an empty or damaged timing cell: "
        "Saxpy.Partly, Saxpy.Damaged, Saxpy.Unbounded, Saxpy.Malformed, Saxpy.Infinite\n"
    )


def test_a_file_of_unplottable_rows_is_refused_without_the_ncu_note(
    tmp_path, monkeypatch, recorded_plots
):
    """A file whose rows all lack timing, some not in the untimed shape, is
    refused with the plain reason."""
    partly = TIMED_ROW.replace(",0.0002,", ",,")
    csv = write_csv(tmp_path / "rows.csv", [partly, UNTIMED_ROWS[0]])
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert str(stopped.value) == (
        f"bench-plot: {csv}: no row has timing: every row has an empty or damaged timing cell"
    )
    assert recorded_plots == {}


# The texts pandas reads as a missing value unless told otherwise.
MISSING_MARKERS = ("n/a", "nan", "NULL", "NA", "None")


def every_timing_cell(test, text):
    """A row whose twelve timing cells each hold @p text, its other numbers sound."""
    return f"{test},2,2,1,1,64,0,0,INFO," + ",".join([text] * 12) + ",0.1,,,"


def test_text_in_every_timing_cell_is_damage_not_emptiness(
    tmp_path, monkeypatch, capsys, recorded_plots
):
    """A row whose timing cells all hold text that pandas would take for a
    missing value is left out as damaged, without the Nsight Compute note,
    while a row whose cells are empty fields keeps it."""
    texts = [every_timing_cell(f"Saxpy.Text{i}", text) for i, text in enumerate(MISSING_MARKERS)]
    csv = write_csv(tmp_path / "rows.csv", [TIMED_ROW, UNTIMED_ROWS[0]] + texts)
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 1 rows without timing (every timing cell is empty, as in "
        "a run under Nsight Compute (ncu), which the benchmark does not time): Saxpy.EndToEnd\n"
        f"bench-plot: {csv}: left out 5 rows with an empty or damaged timing cell: "
        "Saxpy.Text0, Saxpy.Text1, Saxpy.Text2, Saxpy.Text3, Saxpy.Text4\n"
    )


@pytest.mark.parametrize("text", MISSING_MARKERS)
def test_a_file_of_text_timing_is_refused_with_the_plain_reason(
    tmp_path, monkeypatch, recorded_plots, text
):
    """A file whose only row holds text in every timing cell is refused as
    damaged, not as a run under Nsight Compute."""
    csv = write_csv(tmp_path / "rows.csv", [every_timing_cell("Saxpy.Text", text)])
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert str(stopped.value) == (
        f"bench-plot: {csv}: no row has timing: every row has an empty or damaged timing cell"
    )
    assert recorded_plots == {}


def test_a_test_named_as_a_missing_marker_keeps_its_name(
    tmp_path, monkeypatch, capsys, recorded_plots
):
    """Tests named NA and None are plotted by those names, and an untimed test
    named null is named so, as bench summary names them."""
    rows = [
        TIMED_ROW,
        TIMED_ROW.replace("Saxpy.Plain", "NA"),
        TIMED_ROW.replace("Saxpy.Plain", "None"),
        UNTIMED_ROWS[0].replace("Saxpy.EndToEnd", "null"),
    ]
    csv = write_csv(tmp_path / "names.csv", rows)
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain", "NA", "None"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 1 rows without timing (every timing cell is empty, as in "
        "a run under Nsight Compute (ncu), which the benchmark does not time): null\n"
    )


def test_the_other_plotted_numbers_read_text_as_no_number(tmp_path):
    """msgBytes and the GPU columns hold numbers after loading: an empty field
    or any text in one is no number, and a number stays one, an infinity
    included, as pandas's default reading gave them."""
    rows = [
        TIMED_ROW.replace(",64,", ",n/a,").replace(",0.1,,,", ",0.1,RTX,NULL,0.5"),
        TIMED_ROW.replace(",0.1,,,", ",0.1,RTX,12.5,garbage"),
        TIMED_ROW.replace(",0.1,,,", ",0.1,RTX,,inf"),
    ]
    df = load_csv(write_csv(tmp_path / "gpu.csv", rows))
    assert df["msgBytes"].isna().tolist() == [True, False, False]
    assert df["kernelTimeUs"].isna().tolist() == [True, False, True]
    assert df["kernelTimeUs"].iloc[1] == 12.5
    assert df["occupancy"].isna().tolist() == [False, True, False]
    assert df["occupancy"].iloc[0] == 0.5
    assert df["occupancy"].iloc[2] == float("inf")
    assert df["gpuModel"].tolist() == ["RTX", "RTX", "RTX"]


CAPTURE_COLUMNS = CAPTURE_HEADER.split(",")


def untimed_row(test, **cells):
    """UNTIMED_ROWS[0] (every timing cell empty) named @p test, with @p cells
    set by column name, as written."""
    values = UNTIMED_ROWS[0].split(",")
    values[0] = test
    for column, text in cells.items():
        values[CAPTURE_COLUMNS.index(column)] = text
    return ",".join(values)


# Other numbers that `bench summary` does not read as values of their kind:
# a count is a whole number from 0 to 4294967295 written without spaces, and
# cvThreshold a finite number written without spaces.
DAMAGED_OTHER_NUMBERS = [
    ("cycles", "-1"),
    ("cycles", "1.5"),
    ("cycles", "4294967296"),
    ("warmup", "1e3"),
    ("threads", " 5"),
    ("msgBytes", "1.0"),
    ("cvThreshold", " 5"),
    ("cvThreshold", "1e400"),
]


def test_an_untimed_row_needs_other_numbers_of_their_kind(
    tmp_path, monkeypatch, capsys, recorded_plots
):
    """A row of empty timing whose counts are whole numbers in range, zero
    and the largest included, written with a plus sign or leading zeros, and
    whose cvThreshold is any finite number, keeps the Nsight Compute note; a
    count that is negative, a fraction, too large, in exponent form, spaced or
    written with a point, and a spaced or infinite cvThreshold, do not."""
    good = [
        untimed_row("Saxpy.Zero", cycles="0"),
        untimed_row("Saxpy.Largest", msgBytes="4294967295"),
        untimed_row("Saxpy.Plus", repeats="+5"),
        untimed_row("Saxpy.Zeros", threads="007"),
        untimed_row("Saxpy.NegativeThreshold", cvThreshold="-1"),
        untimed_row("Saxpy.ExponentThreshold", cvThreshold="1e3"),
    ]
    damaged = [
        untimed_row(f"Saxpy.Damaged{i}", **{column: text})
        for i, (column, text) in enumerate(DAMAGED_OTHER_NUMBERS)
    ]
    csv = write_csv(tmp_path / "rows.csv", [TIMED_ROW] + good + damaged)
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 6 rows without timing (every timing cell is empty, as in "
        "a run under Nsight Compute (ncu), which the benchmark does not time): Saxpy.Zero, "
        "Saxpy.Largest, Saxpy.Plus, Saxpy.Zeros, Saxpy.NegativeThreshold, "
        "Saxpy.ExponentThreshold\n"
        f"bench-plot: {csv}: left out 8 rows with an empty or damaged timing cell: "
        + ", ".join(f"Saxpy.Damaged{i}" for i in range(len(DAMAGED_OTHER_NUMBERS)))
        + "\n"
    )


@pytest.mark.parametrize("column,text", DAMAGED_OTHER_NUMBERS)
def test_a_file_of_damaged_other_numbers_is_refused_with_the_plain_reason(
    tmp_path, monkeypatch, recorded_plots, column, text
):
    """A file whose only row has empty timing and one other number not of its
    kind is refused as damaged, not as a run under Nsight Compute."""
    csv = write_csv(tmp_path / "rows.csv", [untimed_row("Saxpy.Damaged", **{column: text})])
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert str(stopped.value) == (
        f"bench-plot: {csv}: no row has timing: every row has an empty or damaged timing cell"
    )
    assert recorded_plots == {}


def test_other_numbers_of_thousands_of_digits_are_read_as_bench_summary_reads_them(
    tmp_path, monkeypatch, capsys, recorded_plots
):
    """A count of 5000 zeros then 5, or then the largest count, and a
    cvThreshold of 1e- then 5000 nines (zero) keep the Nsight Compute note; a
    count of 5000 nines or of 5000 zeros then one past the largest is too
    large, and a cvThreshold of 1e then 5000 nines is not finite, so those
    rows are damaged. None of them makes the check raise."""
    nines, zeros = "9" * LONG, "0" * LONG
    noted = [
        untimed_row("Saxpy.ZerosThenFive", cycles=zeros + "5"),
        untimed_row("Saxpy.ZerosThenLargest", msgBytes=zeros + "4294967295"),
        untimed_row("Saxpy.TinyThreshold", cvThreshold="1e-" + nines),
    ]
    damaged = [
        untimed_row("Saxpy.Nines", cycles=nines),
        untimed_row("Saxpy.ZerosThenTooLarge", repeats=zeros + "4294967296"),
        untimed_row("Saxpy.HugeThreshold", cvThreshold="1e" + nines),
    ]
    csv = write_csv(tmp_path / "rows.csv", [TIMED_ROW] + noted + damaged)
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
    assert capsys.readouterr().err == (
        f"bench-plot: {csv}: left out 3 rows without timing (every timing cell is empty, as in "
        "a run under Nsight Compute (ncu), which the benchmark does not time): "
        "Saxpy.ZerosThenFive, Saxpy.ZerosThenLargest, Saxpy.TinyThreshold\n"
        f"bench-plot: {csv}: left out 3 rows with an empty or damaged timing cell: "
        "Saxpy.Nines, Saxpy.ZerosThenTooLarge, Saxpy.HugeThreshold\n"
    )


@pytest.mark.parametrize(
    "count,reason",
    [
        ("9" * LONG, "every row has an empty or damaged timing cell"),
        (
            "0" * LONG + "5",
            "every timing cell is empty, as in a run under Nsight Compute (ncu), which the "
            "benchmark does not time; if the run was under ncu, plot a run without it",
        ),
    ],
    ids=["nines", "zeros-then-five"],
)
def test_a_file_of_one_long_count_is_refused_as_bench_summary_reads_it(
    tmp_path, monkeypatch, recorded_plots, count, reason
):
    """A file whose only row has empty timing and a count of thousands of
    digits is refused, with the plain reason for one too large and the Nsight
    Compute reason for one whose leading zeros hide a 5."""
    csv = write_csv(tmp_path / "rows.csv", [untimed_row("Saxpy.Long", cycles=count)])
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert str(stopped.value) == f"bench-plot: {csv}: no row has timing: {reason}"
    assert recorded_plots == {}


def test_scaling_places_a_row_whose_name_is_no_size_by_its_msg_bytes(tmp_path, monkeypatch):
    """bench-plot scaling places a row whose name ends in thousands of digits
    by its msgBytes, and a row whose name ends in a size by that size, in its
    plotly page, which draws a name of any length."""
    placed = []
    monkeypatch.setattr(bench_plot, "HAS_PLOTLY", True)
    monkeypatch.setattr(
        bench_plot, "_plot_scaling_plotly", lambda df, path: placed.append(list(df["payloadBytes"]))
    )
    first = write_csv(tmp_path / "a.csv", [TIMED_ROW.replace("Saxpy.Plain", "Send" + "9" * LONG)])
    second = write_csv(tmp_path / "b.csv", [TIMED_ROW.replace("Saxpy.Plain", "Send1KB")])
    run_main(
        monkeypatch, ["scaling", str(first), str(second), "--output", str(tmp_path / "s.html")]
    )
    assert placed == [[64, 1024]]


# =============================================================================
# Test names and the other text bench-plot shows, read as written
# =============================================================================


def named(*names):
    """TIMED_ROW once for each of @p names, as written into the test cell."""
    return [TIMED_ROW.replace("Saxpy.Plain", name, 1) for name in names]


@pytest.fixture()
def quick_savefig(monkeypatch):
    """matplotlib's own savefig, which checks the file's format, at a low
    resolution, so a test that draws a real plot stays fast."""
    save = bench_plot.plt.savefig
    monkeypatch.setattr(
        bench_plot.plt, "savefig", lambda path, **kwargs: save(path, **{**kwargs, "dpi": 10})
    )


@pytest.mark.parametrize("command", ["plot", "report", "dashboard"])
def test_test_names_that_look_like_numbers_stay_text(
    tmp_path, monkeypatch, recorded_plots, command
):
    """Names that all look like numbers label the plots as written, as text,
    as bench summary names them: 007 is not 7 and 1e3 is not 1000."""
    csv = write_csv(tmp_path / "numbers.csv", named("64", "007", "1e3"))
    output = tmp_path / "out" / ("d.html" if command == "dashboard" else "")
    run_main(monkeypatch, [command, str(csv), "--output", str(output)])
    assert recorded_plots["plot_throughput_bars"] == [["64", "007", "1e3"]]


def test_scaling_plots_names_that_look_like_numbers_by_those_names(
    tmp_path, monkeypatch, quick_savefig
):
    """bench-plot scaling draws files whose names are all numbers, each test
    by its name as written."""
    placed = []
    draw = bench_plot._plot_scaling_static

    def drawn(df, path):
        placed.append(sorted(df["test"].unique()))
        draw(df, path)

    monkeypatch.setattr(bench_plot, "_plot_scaling_static", drawn)
    # Each test in two files, so each is drawn as a line.
    first = write_csv(tmp_path / "a.csv", named("64", "128", "256"))
    second = write_csv(tmp_path / "b.csv", named("64", "128", "256"))
    output = tmp_path / "s.png"
    run_main(monkeypatch, ["scaling", str(first), str(second), "--output", str(output)])
    assert placed == [["128", "256", "64"]]
    assert output.stat().st_size > 0


def test_names_with_spaces_or_unusual_characters_are_kept_as_written(
    tmp_path, monkeypatch, recorded_plots
):
    """A name with spaces around it is its own name, as bench summary keeps
    it, and a name of a character Rust does not call whitespace (the file
    separator, which Python's isspace() does) names its row."""
    csv = write_csv(tmp_path / "names.csv", named(" Saxpy.Padded ", "\x1c", "Saxpy.Plain"))
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [[" Saxpy.Padded ", "\x1c", "Saxpy.Plain"]]


@pytest.mark.parametrize(
    "name,why",
    [("", "the row has no test name"), ("   ", 'test name "   " is only whitespace')],
    ids=["empty", "spaces"],
)
@pytest.mark.parametrize("command", ["plot", "report", "dashboard", "scaling"])
def test_a_row_without_a_test_name_stops_the_command_naming_the_line(
    tmp_path, monkeypatch, command, name, why
):
    """A file with a row whose test name is empty or only whitespace is
    refused, in bench summary's words, naming the file and the line, before
    anything is drawn, as bench summary and bench compare refuse it."""
    csv = write_csv(tmp_path / "rows.csv", named("Saxpy.Plain", name, "Saxpy.Other"))
    output = tmp_path / "out" / {"dashboard": "d.html", "scaling": "s.png"}.get(command, "")
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, [command, str(csv), "--output", str(output)])
    assert str(stopped.value) == f"bench-plot: {csv}, line 3: {why}"
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_each_unusable_test_name_is_refused_as_bench_summary_words_it(tmp_path, monkeypatch):
    """A quoted empty name, a tab, a no-break space (Unicode whitespace, shown
    as Rust shows it) and a name that is only a newline are refused; the line
    named is the one the row starts on, counting each line of a row that spans
    two and a blank line, which is no row."""
    two_lines = TIMED_ROW.replace("Saxpy.Plain", '"Saxpy.Two\nLines"', 1)
    for rows, why, line in [
        (named("Saxpy.Plain", '""'), "the row has no test name", 3),
        (named("Saxpy.Plain", "\t"), 'test name "\\t" is only whitespace', 3),
        (named("Saxpy.Plain", "\xa0"), 'test name "\\u{a0}" is only whitespace', 3),
        (named("Saxpy.Plain", '"\n"'), 'test name "\\n" is only whitespace', 3),
        ([two_lines] + named("Saxpy.Plain", ""), "the row has no test name", 5),
        (named("Saxpy.Plain") + [""] + named(""), "the row has no test name", 4),
    ]:
        csv = write_csv(tmp_path / "rows.csv", rows)
        with pytest.raises(SystemExit) as stopped:
            run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
        assert str(stopped.value) == f"bench-plot: {csv}, line {line}: {why}"


def test_the_report_copies_text_columns_as_written(tmp_path, monkeypatch, recorded_plots):
    """The report's metadata.json holds the first row's timestamp, git hash
    and host name as written, whatever they look like, and null for an empty
    one, which NaN would not be in JSON."""
    header = "test,wallMedian,wallCV,callsPerSecond,wallMin,wallP10,wallP90,wallMax"
    for cells, held in [
        (
            "20261115,0123456,101",
            {"timestamp": "20261115", "gitHash": "0123456", "hostname": "101"},
        ),
        (",1234e56,", {"timestamp": None, "gitHash": "1234e56", "hostname": None}),
    ]:
        csv = tmp_path / "meta.csv"
        csv.write_text(f"{header},timestamp,gitHash,hostname\nSend1KB,1,0.1,10,1,1,1,1,{cells}\n")
        out = tmp_path / "report"
        run_main(monkeypatch, ["report", str(csv), "--output", str(out)])
        metadata = json.loads((out / "metadata.json").read_text())
        assert {k: metadata[k] for k in held} == held


def test_scaling_without_plotly_writes_its_static_plot_beside_the_page(
    tmp_path, monkeypatch, capsys, quick_savefig
):
    """Without plotly, bench-plot scaling given an .html output draws its
    static plot as a .png of the same name and says so, as dashboard falls
    back to the report's plots."""
    monkeypatch.setattr(bench_plot, "HAS_PLOTLY", False)
    first = write_csv(tmp_path / "a.csv", named("Send.A", "Send.B"))
    second = write_csv(tmp_path / "b.csv", named("Send.A", "Send.B"))
    run_main(
        monkeypatch, ["scaling", str(first), str(second), "--output", str(tmp_path / "s.html")]
    )
    assert (tmp_path / "s.png").stat().st_size > 0
    assert not (tmp_path / "s.html").exists()
    assert f"Falling back to a static plot: {tmp_path / 's.png'}" in capsys.readouterr().out


# =============================================================================
# The malformed corpus (malformed_csv.py)
# =============================================================================

# Each command, drawing with matplotlib or with plotly, and the output it is
# given.
COMMANDS = [
    ("plot", False, ""),
    ("report", False, ""),
    ("dashboard", False, "d.html"),
    ("dashboard", True, "d.html"),
    ("scaling", False, "s.png"),
    ("scaling", True, "s.html"),
]


def message(path, reason):
    """bench-plot's one line for @p reason about the file at @p path."""
    return f"bench-plot: {path}{', ' if reason.startswith('line ') else ': '}{reason}"


@pytest.mark.parametrize(
    "command,plotly,output",
    COMMANDS,
    ids=[f"{command}-{'plotly' if plotly else 'matplotlib'}" for command, plotly, _ in COMMANDS],
)
@pytest.mark.parametrize("case", CORPUS, ids=[case.name for case in CORPUS])
def test_every_command_draws_a_malformed_file_or_refuses_it_in_one_line(
    tmp_path, monkeypatch, capsys, recorded_plots, case, command, plotly, output
):
    """Every command draws each file of the corpus or stops, before drawing,
    with one line naming the file and the reason, and never raises: the
    reason `bench summary` refuses the file with, or the column or the name
    the command cannot draw. A row whose timing `bench summary` refuses is
    left out, and stderr names it."""
    path = write(case, str(tmp_path))
    reason = case.refusal(command, plotly)
    if reason == "Permission denied" and os.access(path, os.R_OK):
        reason = None  # as root, who reads a file of mode 000
    monkeypatch.setattr(bench_plot, "HAS_PLOTLY", plotly)
    if plotly:
        monkeypatch.setattr(
            bench_plot,
            "create_interactive_dashboard",
            lambda df, output_path: recorded_plots.setdefault("dashboard", []).append(
                list(df["test"])
            ),
        )
    argv = [command, path, "--output", str(tmp_path / "out" / output)]
    if reason is None:
        run_main(monkeypatch, argv)
        assert recorded_plots
    else:
        with pytest.raises(SystemExit) as stopped:
            run_main(monkeypatch, argv)
        assert stopped.value.code == message(path, reason)
        assert recorded_plots == {}
    left_out = case.left_out and (
        f"bench-plot: {path}: left out 1 rows with an empty or damaged timing cell: "
        f"{case.left_out}\n"
    )
    assert capsys.readouterr().err == (left_out or "")


@pytest.mark.parametrize("command", ["report", "scaling"])
def test_test_names_holding_a_dollar_are_drawn_as_written(
    tmp_path, monkeypatch, quick_savefig, command
):
    """Names such as `$$` and `$\\frac{$` label the report's plots, and
    scaling's lines, as written: a '$' starts no mathtext, which would make
    matplotlib refuse to draw them."""
    path = write(next(case for case in CORPUS if case.name == "dollar_name"), str(tmp_path))
    # Scaling draws a test as a line, with its name, when two files hold it.
    files = [path, path] if command == "scaling" else [path]
    output = tmp_path / ("s.png" if command == "scaling" else "report")
    run_main(monkeypatch, [command, *files, "--output", str(output)])
    drawn = [output] if command == "scaling" else sorted(output.glob("*.png"))
    assert len(drawn) == (1 if command == "scaling" else 3)
    assert all(png.stat().st_size > 0 for png in drawn)


def test_a_test_name_is_drawn_up_to_the_longest_label():
    """A name of LONGEST_LABEL characters is drawn, and a longer one stops
    the command, naming its start and its length."""
    longest = "x" * bench_plot.LONGEST_LABEL
    bench_plot.drawable_names(pd.DataFrame({"test": ["Short", longest]}), "f.csv", "plot")
    with pytest.raises(SystemExit) as stopped:
        bench_plot.drawable_names(pd.DataFrame({"test": [longest + "y"]}), "f.csv", "plot")
    assert stopped.value.code == (
        "bench-plot: f.csv: plot draws a test name of at most 1000 characters, and the one "
        f"starting '{longest[:40]}' has 1001"
    )


def test_a_column_named_twice_is_read_from_its_last_place(tmp_path):
    """A column the header names twice is read from its last place, once, as
    bench summary reads it: the later wallMedian, and the later test name."""
    cases = {case.name: case for case in CORPUS}
    df = load_csv(write(cases["duplicate_wallMedian"], str(tmp_path)))
    assert df["wallMedian"].tolist() == [20.0, 20.0]
    assert list(df.columns).count("wallMedian") == 1 and "wallMedian.1" not in df.columns
    df = load_csv(write(cases["duplicate_test"], str(tmp_path)))
    assert df["test"].tolist() == ["Corpus.Other", "Corpus.Other"]


def test_a_nul_is_kept_as_written_and_in_a_number_is_no_number(tmp_path):
    """A NUL in a test name and in the host name the report copies is kept,
    where pandas would end the field at it, and a timing cell holding one
    holds no number, where pandas would read `10.0<NUL>x` as 10.0."""
    csv = tmp_path / "nul.csv"
    damaged = TIMED_ROW.replace(",3082.75,", ",3082.75\x00x,", 1)
    csv.write_text(
        f"{CAPTURE_HEADER},hostname\n"
        + TIMED_ROW.replace("Saxpy.Plain", "Saxpy.\x00A")
        + ",ho\x00st\n"
        + damaged.replace("Saxpy.Plain", "Saxpy.\x00B")
        + ",host\n"
    )
    df = load_csv(csv)
    assert df["test"].tolist() == ["Saxpy.\x00A", "Saxpy.\x00B"]
    assert df["hostname"].tolist() == ["ho\x00st", "host"]
    assert bench_plot.timed_rows(df, str(csv))["test"].tolist() == ["Saxpy.\x00A"]


# =============================================================================
# Quoted names, through both of bench-plot's CSV writes
# =============================================================================

# Names valid CSV holds in a quoted field: a CR, an LF and a CRLF, a doubled
# quote, a comma, a NUL, and spaces at both edges. bench summary reads each as
# written from the same bytes (tests/bench.rs).
QUOTED_NAMES = [
    ("cr", "Two\rLines"),
    ("lf", "Two\nLines"),
    ("crlf", "Two\r\nLines"),
    ("quote", 'say "x"'),
    ("comma", "a,b"),
    ("nul", "a\x00b"),
    ("edge_spaces", " Padded "),
]


def quoted(name):
    """@p name as a quoted CSV field, its quotes doubled."""
    return '"' + name.replace('"', '""') + '"'


def quoted_name_file(path, name):
    """A file, written byte for byte, of two timed rows: @p name, quoted, with
    a median of 10, then Other, with 20; return @p path."""
    path.write_bytes(
        (
            "test,wallMedian,wallCV,callsPerSecond,wallMin,wallP10,wallP90,wallMax\r\n"
            + quoted(name)
            + ",10,0.1,100000,8,9,11,12\r\nOther,20,0.1,50000,18,19,21,22\r\n"
        ).encode("utf-8")
    )
    return path


def read_back(data):
    """The records Python's csv module reads from the bytes @p data
    (newline=""), a NUL read through a stand-in, which the module of Python
    before 3.11 needs."""
    text = data.decode("utf-8")
    stand_in = bench_plot.absent_character(text)
    records = csv_records(io.StringIO(text.replace("\x00", stand_in), newline=""))
    return [[field.replace(stand_in, "\x00") for field in record] for record in records]


def written_as_csv_needs(data, name):
    """Whether the bytes @p data hold @p name's row with the name as CSV must
    write it: quoted when it holds a CR, an LF, a quote or a comma, as it is
    when it holds only a NUL or spaces, which need no quotes."""
    field = quoted(name) if any(c in name for c in '\r\n",') else name
    return ("\n" + field + ",10").encode("utf-8") in data


@pytest.mark.parametrize(
    "name", [name for _, name in QUOTED_NAMES], ids=[case for case, _ in QUOTED_NAMES]
)
def test_a_quoted_name_is_read_as_written_with_its_own_timing(tmp_path, name):
    """A valid file whose quoted name holds a line end, a quote, a comma, a
    NUL or edge spaces is read as its two rows, that name and Other, each
    with its own median, and the rows the plots draw keep both."""
    csv = quoted_name_file(tmp_path / "names.csv", name)
    df = load_csv(csv)
    assert df["test"].tolist() == [name, "Other"]
    assert df["wallMedian"].tolist() == [10.0, 20.0]
    kept = bench_plot.timed_rows(df, str(csv))
    assert kept["test"].tolist() == [name, "Other"]
    assert kept["wallMedian"].tolist() == [10.0, 20.0]


@pytest.mark.parametrize(
    "name", [name for _, name in QUOTED_NAMES], ids=[case for case, _ in QUOTED_NAMES]
)
def test_the_report_of_a_quoted_name_reads_back_as_written(
    tmp_path, monkeypatch, recorded_plots, name
):
    """The report's summary_stats.csv of that file, read back by Python's csv
    module, holds the header and the same two rows, and its bytes quote the
    name when CSV must."""
    csv = quoted_name_file(tmp_path / "names.csv", name)
    output = tmp_path / "report"
    run_main(monkeypatch, ["report", str(csv), "--output", str(output)])
    data = (output / "summary_stats.csv").read_bytes()
    assert [record[0] for record in read_back(data)] == ["test", name, "Other"]
    assert written_as_csv_needs(data, name), data


@pytest.mark.parametrize(
    "name", [name for _, name in QUOTED_NAMES], ids=[case for case, _ in QUOTED_NAMES]
)
def test_the_report_writer_keeps_a_quoted_name_in_its_record(tmp_path, recorded_plots, name):
    """summary_stats.csv written from rows that hold that name reads back as
    those rows, however the file they came from was read."""
    rows = pd.DataFrame(
        {
            "test": [name, "Other"],
            "wallMedian": [10.0, 20.0],
            "wallP10": [9.0, 19.0],
            "wallP90": [11.0, 21.0],
            "wallCV": [0.1, 0.1],
            "callsPerSecond": [100000.0, 50000.0],
        }
    )
    bench_plot.generate_report(rows, tmp_path / "report")
    data = (tmp_path / "report" / "summary_stats.csv").read_bytes()
    assert [record[0] for record in read_back(data)] == ["test", name, "Other"]
    assert written_as_csv_needs(data, name), data
