//! Reading what the Nsight tools print into CSV rows.
//!
//! `nsys stats --format csv` and `ncu --csv` print a banner before their CSV,
//! so the records start at the first line that looks like a header: two or
//! more commas, and no `#` first. A record shorter than its header leaves its
//! last columns empty, a header name given twice keeps the later value, and
//! blank lines are skipped.
//! A record longer than its header is refused rather than given made-up column
//! names. Lines end at `\n`, `\r` and `\r\n` and at the other line boundaries
//! Unicode names; the carriage returns matter: `nsys export` draws its progress
//! bar with them.

use std::fmt;

/* ----------------------------- Types ----------------------------- */

/// One CSV row: column names with their values, in the order first set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Row {
    cells: Vec<(String, String)>,
}

/// Tool output that cannot be read as rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Malformed {
    /// The line of the tool's output, counting from 1.
    line: u64,
    fields: usize,
    columns: usize,
}

/* ----------------------------- Helpers ----------------------------- */

impl Row {
    /// The value of @p column, if the row has one.
    pub(crate) fn get(&self, column: &str) -> Option<&str> {
        self.cells
            .iter()
            .find(|(name, _)| name == column)
            .map(|(_, value)| value.as_str())
    }

    /// The row's column names, in order.
    pub(crate) fn columns(&self) -> impl Iterator<Item = &str> {
        self.cells.iter().map(|(name, _)| name.as_str())
    }

    /// Give @p column the value @p value, in its place if the row has it.
    pub(crate) fn set(&mut self, column: &str, value: impl Into<String>) {
        let value = value.into();
        match self.cells.iter_mut().find(|(name, _)| name == column) {
            Some(cell) => cell.1 = value,
            None => self.cells.push((column.to_string(), value)),
        }
    }

    /// Remove @p column and return its value.
    fn take(&mut self, column: &str) -> Option<String> {
        let index = self.cells.iter().position(|(name, _)| name == column)?;
        Some(self.cells.remove(index).1)
    }
}

impl fmt::Display for Malformed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {} of its output has {} fields for {} columns",
            self.line, self.fields, self.columns
        )
    }
}

/// @p text split into lines: at `\n`, `\r`, `\r\n` and the other line
/// boundaries Unicode names (vertical tab, form feed, the file, group and
/// record separators, next line, line separator, paragraph separator).
fn lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if !matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            continue;
        }
        out.push(&text[start..at]);
        start = at + c.len_utf8();
        if c == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
            chars.next();
            start += 1;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The records of a tool's CSV output, each a row keyed by the header.
fn records(text: &str) -> Result<Vec<Row>, Malformed> {
    let all = lines(text);
    let Some(start) = all
        .iter()
        .position(|line| line.matches(',').count() >= 2 && !line.trim_start().starts_with('#'))
    else {
        return Ok(Vec::new());
    };
    let csv_text = all[start..].join("\n");
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(csv_text.as_bytes());
    let mut records = reader.records();
    let header: Vec<String> = match records.next() {
        Some(Ok(record)) => record.iter().map(str::to_string).collect(),
        _ => return Ok(Vec::new()),
    };
    let mut rows = Vec::new();
    for record in records {
        // The text is valid UTF-8 and the reader is flexible, so a record
        // cannot fail to read.
        let Ok(record) = record else { continue };
        if record.len() > header.len() {
            let line = record.position().map_or(0, |p| p.line()) + start as u64;
            return Err(Malformed {
                line,
                fields: record.len(),
                columns: header.len(),
            });
        }
        let mut row = Row::default();
        for (index, name) in header.iter().enumerate() {
            row.set(name, record.get(index).unwrap_or(""));
        }
        rows.push(row);
    }
    Ok(rows)
}

/* ----------------------------- API ----------------------------- */

/// The rows of one nsys summary: `Name` (or `Range`) as `kernel`, `Instances`
/// (or `Num Calls`) as `instances`, the three times under their own names, and
/// every other column as the summary prints it. A column the summary itself
/// names `kernel`, `instances` or one of the times keeps its value; `source`
/// and `report` are this reader's.
pub(crate) fn nsys_rows(summary: &str, text: &str) -> Result<Vec<Row>, Malformed> {
    Ok(records(text)?
        .into_iter()
        .map(|mut record| {
            let range = record.take("Range");
            let name = record.take("Name");
            let calls = record.take("Num Calls");
            let instances = record.take("Instances");
            let named = [
                ("kernel", name.or(range)),
                ("instances", instances.or(calls)),
                ("time_total_ns", record.take("Total Time (ns)")),
                ("time_avg_ns", record.take("Avg (ns)")),
                ("time_pct", record.take("Time (%)")),
            ];
            let mut row = Row::default();
            row.set("source", "nsys");
            row.set("report", summary);
            for (column, value) in record.cells {
                if column != "source" && column != "report" {
                    row.set(&column, value);
                }
            }
            for (column, value) in named {
                if row.get(column).is_none() {
                    row.set(column, value.unwrap_or_default());
                }
            }
            row
        })
        .collect())
}

/// The rows of `ncu --print-summary per-kernel`: `Kernel Name` (or `Kernel`)
/// as `kernel`, and every other column under its cleaned name.
pub(crate) fn ncu_rows(text: &str) -> Result<Vec<Row>, Malformed> {
    Ok(records(text)?
        .into_iter()
        .map(|mut record| {
            let short = record.take("Kernel");
            let kernel = record.take("Kernel Name").or(short).unwrap_or_default();
            let mut row = Row::default();
            row.set("source", "ncu");
            row.set("report", "per_kernel");
            row.set("kernel", kernel);
            for (column, value) in record.cells {
                let column = clean_key(&column);
                if column != "source" && column != "report" {
                    row.set(&column, value);
                }
            }
            row
        })
        .collect())
}

/// An ncu column name as the CSV writes it: trimmed, lower case, spaces as
/// underscores, without parentheses, `/` as `_per_`.
pub(crate) fn clean_key(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .replace(' ', "_")
        .replace(['(', ')'], "")
        .replace('/', "_per_")
}

/// The last line with text in the first of @p texts that has one, trimmed; or
/// `no output`.
pub(crate) fn last_line(texts: &[&str]) -> String {
    texts
        .iter()
        .find_map(|text| {
            lines(text)
                .into_iter()
                .map(str::trim)
                .rfind(|line| !line.is_empty())
        })
        .unwrap_or("no output")
        .to_string()
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nsight");

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!("{FIXTURES}/{name}")).expect("fixture")
    }

    fn cells(row: &Row) -> Vec<(&str, &str)> {
        row.cells
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect()
    }

    /// @test The records start after the banner nsys and ncu print.
    #[test]
    fn header_after_banner() {
        let text = "*** Performance counter data ***\nGenerated by nsys 2025.1\n\n\
                    Time (%),Total Time (ns),Instances,Avg (ns),Name\n\
                    40.1,200000,10,20000,kernel_a\n35.6,180000,10,18000,kernel_b\n";
        let rows = records(text).expect("records");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("Name"), Some("kernel_a"));
        assert_eq!(rows[0].get("Time (%)"), Some("40.1"));
        assert_eq!(rows[1].get("Instances"), Some("10"));
    }

    /// @test Text with no header line yields no rows.
    #[test]
    fn no_header_no_rows() {
        assert!(records("no comma-separated content here")
            .unwrap()
            .is_empty());
        assert!(records("").unwrap().is_empty());
        assert!(records("# a,b,c comment\n").unwrap().is_empty());
    }

    /// @test A short record leaves its last columns empty, blank lines are
    /// skipped, quoted commas stay in their field, and a repeated header name
    /// keeps the later value in the first one's place.
    #[test]
    fn record_shapes() {
        let text = "a,b,c,b\n1,2\n\n\"x, y\",3,4,5\n";
        let rows = records(text).expect("records");
        assert_eq!(rows.len(), 2);
        assert_eq!(cells(&rows[0]), [("a", "1"), ("b", ""), ("c", "")]);
        assert_eq!(cells(&rows[1]), [("a", "x, y"), ("b", "5"), ("c", "4")]);
    }

    /// @test A record longer than its header is refused, naming its line.
    #[test]
    fn long_record_is_refused() {
        let text = "Processing...\nTime (%),Total Time (ns),Name\n1.0,2,a,EXTRA\n";
        let err = records(text).expect_err("longer than its header");
        assert_eq!(
            err,
            Malformed {
                line: 3,
                fields: 4,
                columns: 3
            }
        );
        assert_eq!(
            err.to_string(),
            "line 3 of its output has 4 fields for 3 columns"
        );
    }

    /// @test Lines end at `\r` as well as `\n`, so the last line of a progress
    /// bar drawn with carriage returns is its final state.
    #[test]
    fn last_line_of_progress_output() {
        let progress = "\rProcessing 10 events: [1%  ]\rProcessing 10 events: [100%]\n";
        assert_eq!(last_line(&[progress]), "Processing 10 events: [100%]");
        assert_eq!(last_line(&["", " \n  \n", "second\nlast  \n\n"]), "last");
        assert_eq!(last_line(&["", "\n"]), "no output");
        assert_eq!(lines("a\r\nb\rc\nd\u{2028}e"), ["a", "b", "c", "d", "e"]);
        assert_eq!(lines("a\n\n"), ["a", ""]);
    }

    /// @test An nsys summary maps Name, Instances and the times to the
    /// CSV's own columns and keeps its other columns as printed.
    #[test]
    fn nsys_kernel_summary() {
        let text = fixture("nsys_stats_cuda_gpu_kern_sum.out");
        let rows = nsys_rows("cuda_gpu_kern_sum", &text).expect("rows");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.get("source"), Some("nsys"));
        assert_eq!(row.get("report"), Some("cuda_gpu_kern_sum"));
        assert!(row
            .get("kernel")
            .unwrap()
            .starts_with("vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *"));
        assert_eq!(row.get("instances"), Some("61"));
        assert_eq!(row.get("time_total_ns"), Some("178481856"));
        assert_eq!(row.get("time_avg_ns"), Some("2925932.1"));
        assert_eq!(row.get("time_pct"), Some("100.0"));
        assert_eq!(row.get("Med (ns)"), Some("2890848.0"));
        for gone in [
            "Name",
            "Instances",
            "Total Time (ns)",
            "Avg (ns)",
            "Time (%)",
        ] {
            assert_eq!(row.get(gone), None, "{gone}");
        }
    }

    /// @test The API summary's Num Calls is the instance count; a copy
    /// summary has no kernel or instances and keeps Operation and Count.
    #[test]
    fn nsys_call_and_copy_summaries() {
        let api = nsys_rows("cuda_api_sum", &fixture("nsys_stats_cuda_api_sum.out")).unwrap();
        assert_eq!(api.len(), 8);
        assert_eq!(api[0].get("kernel"), Some("cudaMemcpy"));
        assert_eq!(api[0].get("instances"), Some("183"));
        let sizes = nsys_rows(
            "cuda_gpu_mem_size_sum",
            &fixture("nsys_stats_cuda_gpu_mem_size_sum.out"),
        )
        .unwrap();
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0].get("kernel"), Some(""));
        assert_eq!(sizes[0].get("instances"), Some(""));
        assert_eq!(
            sizes[0].get("Operation"),
            Some("[CUDA memcpy Host-to-Device]")
        );
        assert_eq!(sizes[1].get("Count"), Some("61"));
    }

    /// @test Name wins over Range and Instances over Num Calls; a summary
    /// column named like a mapped one keeps its value; source and report are
    /// the reader's.
    #[test]
    fn nsys_mapping_precedence() {
        let text = "Range,Name,Num Calls,Instances,source,kernel\nr,n,5,7,tool,k\n";
        let rows = nsys_rows("nvtx_sum", text).unwrap();
        let row = &rows[0];
        assert_eq!(row.get("kernel"), Some("k"));
        assert_eq!(row.get("instances"), Some("7"));
        assert_eq!(row.get("source"), Some("nsys"));
        assert_eq!(row.get("report"), Some("nvtx_sum"));
        assert_eq!(row.get("Range"), None);
        assert_eq!(row.get("Num Calls"), None);
        let ranges = nsys_rows("nvtx_sum", "Time (%),Total Time (ns),Range\n1,2,r\n").unwrap();
        assert_eq!(ranges[0].get("kernel"), Some("r"));
    }

    /// @test A summary with no data, as nsys prints it for a report without
    /// kernels, yields no rows.
    #[test]
    fn nsys_summary_without_data() {
        let text = fixture("nsys_stats_no_kernels.out");
        assert!(nsys_rows("cuda_gpu_kern_sum", &text).unwrap().is_empty());
        assert!(last_line(&[&text]).ends_with("does not contain CUDA kernel data."));
    }

    /// @test ncu's per-kernel summary: one row per launch shape, section and
    /// metric, Kernel Name as kernel, the other names cleaned.
    #[test]
    fn ncu_per_kernel_summary() {
        let rows = ncu_rows(&fixture("ncu_import_per_kernel.out")).unwrap();
        assert_eq!(rows.len(), 88);
        let first = &rows[0];
        assert_eq!(first.get("source"), Some("ncu"));
        assert_eq!(first.get("report"), Some("per_kernel"));
        assert!(first
            .get("kernel")
            .unwrap()
            .starts_with("unnamed>::saxpyKernel"));
        assert_eq!(first.get("block_size"), Some("(256, 1, 1)"));
        assert_eq!(first.get("metric_name"), Some("SM Frequency"));
        assert_eq!(first.get("maximum"), Some("1,330.98"));
        assert_eq!(first.get("Kernel Name"), None);
        let occupancy: Vec<_> = rows
            .iter()
            .filter(|r| r.get("metric_name") == Some("Theoretical Occupancy"))
            .map(|r| (r.get("block_size").unwrap(), r.get("minimum").unwrap()))
            .collect();
        assert_eq!(
            occupancy,
            [("(256, 1, 1)", "100.00"), ("(1, 1, 1)", "50.00")]
        );
    }

    /// @test Kernel Name wins over Kernel; source and report are the reader's.
    #[test]
    fn ncu_kernel_column_precedence() {
        let rows = ncu_rows("Kernel,Kernel Name,Block Size\nshort,long,\"(1, 1, 1)\"\n").unwrap();
        assert_eq!(rows[0].get("kernel"), Some("long"));
        assert_eq!(rows[0].get("kernel_name"), None);
        let rows =
            ncu_rows("Kernel,Grid Size,Block Size\nonly,\"(2, 1, 1)\",\"(1, 1, 1)\"\n").unwrap();
        assert_eq!(rows[0].get("kernel"), Some("only"));
        let rows = ncu_rows("Source,Report,Kernel Name\ntool,r,k\n").unwrap();
        assert_eq!(rows[0].get("source"), Some("ncu"));
        assert_eq!(rows[0].get("report"), Some("per_kernel"));
    }

    /// @test ncu's column names are written trimmed, in lower case, with
    /// underscores for spaces, without parentheses and with `_per_` for `/`.
    #[test]
    fn cleans_ncu_column_names() {
        assert_eq!(clean_key("SM Throughput (% peak)"), "sm_throughput_%_peak");
        assert_eq!(clean_key("DRAM Bytes/sec"), "dram_bytes_per_sec");
        assert_eq!(clean_key("  Metric Unit "), "metric_unit");
    }

    /// @test A row keeps the place of a column it sets again.
    #[test]
    fn row_set_replaces_in_place() {
        let mut row = Row::default();
        row.set("a", "1");
        row.set("b", "2");
        row.set("a", "3");
        assert_eq!(cells(&row), [("a", "3"), ("b", "2")]);
        assert_eq!(row.columns().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(row.take("a"), Some("3".to_string()));
        assert_eq!(row.take("a"), None);
    }
}
