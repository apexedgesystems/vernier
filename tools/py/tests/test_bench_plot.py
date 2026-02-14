"""Tests for bench_plot data loading and parsing functions."""

import textwrap

import pytest
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
