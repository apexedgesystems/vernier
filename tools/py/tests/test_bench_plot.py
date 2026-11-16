"""Tests for bench_plot data loading and parsing functions."""

import sys
import textwrap

import pytest
from vernier_tools.benchmarking import bench_plot
from vernier_tools.benchmarking.bench_plot import (
    extract_payload_size,
    load_csv,
    load_csvs_from_list,
)

# =============================================================================
# Fixtures
# =============================================================================


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
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nonexistent.csv")


def test_load_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("test,wallMedian\nFoo,1.0\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(p)


def test_load_csv_empty_data(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("test,wallMedian,wallCV,callsPerSecond\n")
    with pytest.raises(ValueError, match="No data rows"):
        load_csv(p)


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
