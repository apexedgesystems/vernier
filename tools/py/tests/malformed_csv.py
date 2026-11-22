"""Malformed benchmark CSV files, written as bytes by hand (no CSV writer),
for bench-plot's tests. Each case says what is wrong with its file, makes the
file, and says what `bench summary` and each bench-plot command make of it.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

# The benchmark's CSV header, and a timed CPU row of it after its test name.
HEADER = (
    "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,wallMedian,"
    "wallP10,wallP90,wallP99,wallP999,wallMin,wallMax,wallMean,wallStddev,wallCV,"
    "callsPerSecond,stable,cvThreshold,timestamp,gitHash,hostname,platform,gpuModel,"
    "computeCapability,kernelTimeUs,transferTimeUs,h2dBytes,d2hBytes,speedupVsCpu,"
    "memBandwidthGBs,occupancy,smClockMHz,throttling,powerDrawW,powerLimitW,temperatureC,"
    "temperatureDeltaC,cuptiKernelLaunches,cuptiRegistersMedian,cuptiRegistersMax,"
    "cuptiStaticSmemBytes,cuptiDynamicSmemBytes,deviceId,deviceCount,multiGpuEfficiency,"
    "p2pBandwidthGBs,umPageFaults,umH2DMigrations,umD2HMigrations,umMigrationTimeUs,"
    "umThrashing"
).split(",")
ROW = (
    "2,2,1,1,64,0,0,INFO,10.0,9.0,11.0,12.0,12.5,8.5,13.0,10.1,0.5,0.05,100000,1,0.1,"
    "2026-11-15T00:00:00Z,unknown,host,x86_64" + "," * 29
).split(",")
NAMES = ("Corpus.First_64", "Corpus.Second_128")
# A GPU row's own cells: a model, a kernel time, a speedup and an occupancy.
GPU_CELLS = {"gpuModel": "RTX", "kernelTimeUs": "4.0", "speedupVsCpu": "2.5", "occupancy": "0.75"}


@dataclass(frozen=True)
class Case:
    """One file of the corpus and what each reader makes of it.

    `summary` is the reason `bench summary` refuses the file with, which every
    bench-plot command refuses it with as it reads it, unless `left_out` names
    the row `bench summary` refuses for its timing: bench-plot leaves that row
    out and names it, and draws the other. `plot`, `plotly` and `scaling` are
    the reasons a command refuses a file it has read with: plot, report and
    the dashboard drawing with matplotlib; the dashboard drawing with plotly;
    scaling drawing with matplotlib. `{command}` stands for the command.
    None: the file is read, or the command draws it.
    """

    name: str
    what: str
    make: Callable[[str], None]
    summary: Optional[str] = None
    plot: Optional[str] = None
    plotly: Optional[str] = None
    scaling: Optional[str] = None
    left_out: Optional[str] = None

    def refusal(self, command: str, plotly: bool) -> Optional[str]:
        """The reason bench-plot's @p command refuses the file with, drawing
        with plotly when @p plotly, or None when it draws the file."""
        if self.summary is not None and self.left_out is None:
            return self.summary
        if command == "scaling":
            reason = None if plotly else self.scaling
        elif command == "dashboard" and plotly:
            reason = self.plotly
        else:
            reason = self.plot
        return None if reason is None else reason.format(command=command)


def refused(name, what, make, reason):
    """A case that `bench summary`, and so every bench-plot command, refuses
    with @p reason."""
    return Case(name, what, make, summary=reason)


def rows(header=HEADER, cells=None, names=NAMES):
    """Each name's row under @p header: ROW's cells, with @p cells set by
    column name, and the columns @p header lacks left out."""
    table = []
    for name in names:
        full = dict(zip(HEADER, [name] + ROW))
        full.update(cells or {})
        table.append([full.get(column, "") for column in header])
    return table


def text(header, table, end="\n"):
    """@p header and @p table as CSV text, every field as it is, each line
    ended by @p end."""
    return "".join(",".join(fields) + end for fields in [header] + table)


def writes(data):
    """A maker that writes @p data (bytes, or text as UTF-8) to the path."""

    def make(path):
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    return make


def without(column, cells=None):
    """The file with @p column taken out of the header and every row."""
    header = [c for c in HEADER if c != column]
    return text(header, rows(header, cells))


def directory(path):
    os.mkdir(path)


def unreadable(path):
    writes(GOOD)(path)
    os.chmod(path, 0)


def symlink_loop(path):
    os.symlink(os.path.basename(path), path)


def nothing(path):
    """No file at all."""


GOOD = text(HEADER, rows())
FIRST, SECOND = (",".join(r) for r in rows())
HEAD = ",".join(HEADER)
ROW_REST = FIRST.split(",", 1)[1]
LATENCY = "{command} needs the %s column, for the latency plot"
GPU = "{command} needs the %s column, for the GPU plots of its rows with a gpuModel"
LONG_NAME = "Corpus." + "x" * 1_000_000
LONG = (
    "{command} draws a test name of at most 1000 characters, and the one starting "
    f"'{LONG_NAME[:40]}' has {len(LONG_NAME)}"
)
# The columns bench-plot adds to the table it reads a file into.
INTERNAL_NAMES = ["_source", "_otherNumbersOfTheirKind", "payloadBytes", "cv_pct"]

CORPUS = [
    Case("good", "a well-formed file", writes(GOOD)),
    Case(
        "gpu_rows",
        "GPU rows, with every column a GPU plot reads",
        writes(text(HEADER, rows(cells=GPU_CELLS))),
    ),
    refused("empty", "no bytes at all", writes(b""), "missing required column 'test'"),
    refused(
        "header_only", "the header and its line end, no row", writes(HEAD + "\n"), "no data rows"
    ),
    refused(
        "header_no_line_end", "the header without a line end, no row", writes(HEAD), "no data rows"
    ),
    *(
        refused(f"no_{c}", f"no {c} column", writes(without(c)), f"missing required column '{c}'")
        for c in ("test", "wallMedian", "wallCV", "callsPerSecond")
    ),
    *(
        Case(
            f"no_{c}",
            f"no {c} column, which the latency plot reads",
            writes(without(c)),
            plot=LATENCY % c,
        )
        for c in ("wallMin", "wallMax")
    ),
    *(
        Case(
            f"no_{c}",
            f"no {c} column, which the latency plot and its plotly bars read",
            writes(without(c)),
            plot=LATENCY % c,
            plotly=LATENCY % c,
        )
        for c in ("wallP10", "wallP90")
    ),
    *(
        Case(f"gpu_no_{c}", f"GPU rows without {c}", writes(without(c, GPU_CELLS)), plot=GPU % c)
        for c in ("kernelTimeUs", "speedupVsCpu", "occupancy")
    ),
    Case(
        "no_msgBytes",
        "no msgBytes column, which scaling reads when present",
        writes(without("msgBytes")),
    ),
    Case(
        "internal_column_names",
        "columns named as those bench-plot adds to its own tables",
        writes(text(HEADER + INTERNAL_NAMES, [r + ["a", "b", "c", "d"] for r in rows()])),
    ),
    Case(
        "duplicate_wallMedian",
        "a second wallMedian column after the last",
        writes(text(HEADER + ["wallMedian"], [r + ["20.0"] for r in rows()])),
    ),
    Case(
        "duplicate_test",
        "a second test column after the last",
        writes(text(HEADER + ["test"], [r + ["Corpus.Other"] for r in rows()])),
    ),
    Case(
        "short_row",
        "a row that ends after wallMedian",
        writes(HEAD + "\n" + FIRST + "\n" + ",".join(rows()[1][:10]) + "\n"),
        summary=(
            "line 3: test 'Corpus.Second_128' has no wallCV value: "
            "the row ends after 10 of 55 columns"
        ),
        left_out=NAMES[1],
    ),
    Case(
        "long_row",
        "a row with three fields more than the header",
        writes(HEAD + "\n" + FIRST + "\n" + SECOND + ",x,y,z\n"),
    ),
    Case(
        "open_quote",
        "a quoted field opened in the last row and never closed",
        writes(HEAD + "\n" + FIRST + "\n" + SECOND.replace("host", '"host', 1) + "\n"),
    ),
    Case(
        "bom", "a UTF-8 byte order mark before the header", writes(b"\xef\xbb\xbf" + GOOD.encode())
    ),
    refused(
        "bom_only",
        "only a UTF-8 byte order mark",
        writes(b"\xef\xbb\xbf"),
        "missing required column 'test'",
    ),
    refused(
        "not_utf8",
        "a test name holding the bytes FF FE, which are not UTF-8",
        writes(GOOD.replace(NAMES[0], "Corpus.\udcff\udcfe").encode("utf-8", "surrogateescape")),
        "line 2: not valid UTF-8",
    ),
    refused(
        "latin1",
        "a test name in Latin-1 (an e with an acute accent as one byte)",
        writes(GOOD.replace(NAMES[0], "Corpus.Caf\xe9").encode("latin-1")),
        "line 2: not valid UTF-8",
    ),
    refused(
        "utf16",
        "the whole file in UTF-16, with its byte order mark",
        writes(GOOD.encode("utf-16")),
        "line 1: not valid UTF-8",
    ),
    Case(
        "nul",
        "a NUL byte inside a test name",
        writes(GOOD.replace(NAMES[0], "Corpus.\x00First_64")),
    ),
    Case(
        "nul_in_number",
        "a NUL byte inside a wallMedian cell, after its number",
        writes(GOOD.replace(",10.0,", ",10.0\x00x,", 1)),
        summary="line 2: test 'Corpus.First_64' has wallMedian '10.0\x00x', which is not a number",
        left_out=NAMES[0],
    ),
    refused(
        "nul_in_header",
        "a NUL byte inside the header's wallMedian, after the name",
        writes(GOOD.replace(",wallMedian,", ",wallMedian\x00x,", 1)),
        "missing required column 'wallMedian'",
    ),
    refused("directory", "a directory where the file should be", directory, "Is a directory"),
    refused("unreadable", "a file that cannot be read (mode 000)", unreadable, "Permission denied"),
    refused(
        "symlink_loop",
        "a symbolic link to itself",
        symlink_loop,
        "Too many levels of symbolic links",
    ),
    refused("missing_file", "no file at the path", nothing, "No such file or directory"),
    Case(
        "long_line",
        "a test name of a million characters, one very long line",
        writes(GOOD.replace(NAMES[0], LONG_NAME)),
        plot=LONG,
        scaling=LONG,
    ),
    Case(
        "long_name",
        "a test name of 1000 characters",
        writes(GOOD.replace(NAMES[0], "Corpus." + "x" * 993)),
    ),
    Case(
        "long_field",
        "a host name of a million characters, which no plot draws",
        writes(GOOD.replace(",host,", "," + "h" * 1_000_000 + ",", 1)),
    ),
    Case(
        "dollar_name",
        "test names holding '$', as matplotlib's mathtext would read them",
        writes(GOOD.replace(NAMES[0], "Corpus.$$").replace(NAMES[1], r"$\frac{$")),
    ),
    Case(
        "crlf_header",
        "only the header's line end is CRLF",
        writes(HEAD + "\r\n" + FIRST + "\n" + SECOND + "\n"),
    ),
    Case(
        "cr_header",
        "only the header's line end is a lone CR",
        writes(HEAD + "\r" + FIRST + "\n" + SECOND + "\n"),
    ),
    Case("all_cr", "every line ended by a lone CR", writes(text(HEADER, rows(), end="\r"))),
    Case("last_row_no_line_end", "the last row without a line end", writes(GOOD[:-1])),
    Case("blank_lines_first", "two blank lines before the header", writes("\n\r\n" + GOOD)),
    Case(
        "blank_lines",
        "blank lines between the header and the rows and between rows",
        writes(HEAD + "\n\n" + FIRST + "\n\n\n" + SECOND + "\n"),
    ),
    refused(
        "spaces_line_first",
        "a line of spaces before the header, which is then the header",
        writes("   \n" + GOOD),
        "missing required column 'test'",
    ),
    refused(
        "tab_separated",
        "fields separated by tabs, so one column",
        writes("\t".join(HEADER) + "\n" + FIRST.replace(",", "\t") + "\n"),
        "missing required column 'test'",
    ),
    refused(
        "spaced_test_header",
        "the test column named ' test', with a space",
        writes(" " + HEAD + "\n" + FIRST + "\n"),
        "missing required column 'test'",
    ),
    Case(
        "quoted_header",
        "every name of the header quoted",
        writes(",".join(f'"{c}"' for c in HEADER) + "\n" + FIRST + "\n" + SECOND + "\n"),
    ),
    Case(
        "empty_header_name",
        "an empty name between two columns of the header",
        writes(HEAD.replace(",console,", ",,", 1) + "\n" + FIRST + "\n" + SECOND + "\n"),
    ),
    Case(
        "quoted_line_end_name",
        "a quoted test name holding a line end",
        writes(HEAD + '\n"Corpus.Two\nLines",' + ROW_REST + "\n" + SECOND + "\n"),
    ),
    refused(
        "empty_name",
        "a row whose test name is empty",
        writes(HEAD + "\n" + FIRST + "\n," + ROW_REST + "\n"),
        "line 3: the row has no test name",
    ),
]


def write(case, directory_path):
    """Make @p case's file in @p directory_path; return its path."""
    path = os.path.join(directory_path, case.name + ".csv")
    case.make(path)
    return path
