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
        f"bench-plot: {csv}: no row has timing: every timing cell is empty, as the benchmark "
        "leaves a row it did not time: a run under Nsight Compute (ncu) is a profiling "
        "capture; plot a run without ncu"
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
        f"bench-plot: {csv}: left out 2 rows without timing (every timing cell is empty, as "
        "the benchmark leaves a row it did not time: a run under Nsight Compute (ncu) is a "
        "profiling capture): Saxpy.EndToEnd, Saxpy.Kernel\n"
    )
    if command != "plot":
        summary = (tmp_path / "out" / "summary_stats.csv").read_text().splitlines()
        assert [line.split(",")[0] for line in summary] == ["test", "Saxpy.Plain"]


def test_a_partly_empty_row_is_not_taken_for_untimed(tmp_path, monkeypatch, recorded_plots):
    """Only a row whose every timing cell is empty is left out."""
    partly = TIMED_ROW.replace(",0.0002,", ",,")
    csv = write_csv(tmp_path / "partly.csv", [partly])
    run_main(monkeypatch, ["plot", str(csv), "--output", str(tmp_path / "out")])
    assert recorded_plots["plot_throughput_bars"] == [["Saxpy.Plain"]]
