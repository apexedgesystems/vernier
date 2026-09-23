//! CSV loading and validation for benchmark results.
//!
//! Handles all CSV variants produced by the benchmarking framework:
//! - Old-format (no stable/cvThreshold columns, has metadata columns)
//! - New-format (with stable/cvThreshold)
//! - GPU-extended (extra GPU columns; CPU rows may be shorter than header)
//!
//! Uses manual field extraction (not serde Deserialize) so that rows shorter
//! than the header are handled gracefully -- missing trailing columns get defaults.
//!
//! A caller that computes from or presents a numeric column states how it
//! needs the column (`Need`) and loads with `load_csv_strict`: a row that
//! falls short is an error naming the file, line, test, column and value,
//! rather than a default or a NaN that looks like a measurement.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use super::Error;

/* ----------------------------- BenchRow ----------------------------- */

/// One row from a benchmark results CSV.
///
/// Fields match the core columns emitted by PerfCsv.hpp. Optional columns
/// (stable, cvThreshold, GPU columns) default to sensible values when absent.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub test: String,
    pub cycles: u32,
    pub repeats: u32,
    pub warmup: u32,
    pub threads: u32,
    pub msg_bytes: u32,
    pub wall_median: f64,
    pub wall_p10: f64,
    pub wall_p90: f64,
    pub wall_p99: f64,
    pub wall_p999: f64,
    pub wall_min: f64,
    pub wall_max: f64,
    pub wall_mean: f64,
    pub wall_stddev: f64,
    pub wall_cv: f64,
    pub calls_per_second: f64,
    pub stable: u8,
    pub cv_threshold: f64,
}

/* ----------------------------- Need ----------------------------- */

/// How a caller needs one numeric column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// Every row holds a finite number in the column; the header must have it.
    Required,
    /// A row may leave the column out: the header lacks it, the row ends
    /// before it, or the field is empty. A value a row does give must be a
    /// finite number of the column's kind.
    IfPresent,
}

/* ----------------------------- Helpers ----------------------------- */

/// Build a column-name-to-index map from the CSV header.
fn header_map(headers: &csv::StringRecord) -> HashMap<String, usize> {
    headers
        .iter()
        .enumerate()
        .map(|(i, name)| (name.to_string(), i))
        .collect()
}

/// Get a string field from a record by header name. Returns "" if missing.
fn get_str<'a>(
    record: &'a csv::StringRecord,
    hmap: &HashMap<String, usize>,
    name: &str,
) -> &'a str {
    hmap.get(name).and_then(|&i| record.get(i)).unwrap_or("")
}

/// A kind of number a column holds.
trait Number: FromStr + Copy {
    /// What a message calls a value of this kind.
    const KIND: &'static str;

    /// Whether a parsed value can stand for a measurement: NaN and the
    /// infinities parse as `f64` but cannot.
    fn usable(self) -> bool {
        true
    }
}

impl Number for f64 {
    const KIND: &'static str = "a number";

    fn usable(self) -> bool {
        self.is_finite()
    }
}

impl Number for u32 {
    const KIND: &'static str = "a whole number from 0 to 4294967295";
}

impl Number for u8 {
    const KIND: &'static str = "a whole number from 0 to 255";
}

/// What one row holds in one numeric column.
#[derive(Debug, Clone, Copy)]
enum Found<'a> {
    /// A finite number of the column's kind.
    Value,
    /// The header has no such column.
    NoColumn,
    /// The row ends before the column, after this many fields.
    CutOff(usize),
    /// The field is empty.
    Empty,
    /// Text that is not a number of the column's kind, and that kind.
    NotANumber(&'a str, &'static str),
    /// Text that parses to NaN or an infinity.
    NotFinite(&'a str),
}

/// One record's numeric fields, read by column name. Each read notes what the
/// row held, so a caller's needs can be checked once the row is parsed.
struct Fields<'a> {
    record: &'a csv::StringRecord,
    hmap: &'a HashMap<String, usize>,
    found: Vec<(&'static str, Found<'a>)>,
}

impl<'a> Fields<'a> {
    /// The column read as `T`: `default` where the row holds no number of
    /// that kind, and a NaN or infinity as read.
    fn number<T: Number>(&mut self, name: &'static str, default: T) -> T {
        let record: &'a csv::StringRecord = self.record;
        let (found, value) = match self.hmap.get(name) {
            None => (Found::NoColumn, default),
            Some(&i) => match record.get(i) {
                None => (Found::CutOff(record.len()), default),
                Some("") => (Found::Empty, default),
                Some(text) => match text.parse::<T>() {
                    Ok(value) if value.usable() => (Found::Value, value),
                    Ok(value) => (Found::NotFinite(text), value),
                    Err(_) => (Found::NotANumber(text, T::KIND), default),
                },
            },
        };
        self.found.push((name, found));
        value
    }
}

/// What falls short when a row holds `found` in `column` and a caller needs
/// it as `need`, or None when it does not.
fn unmet(column: &str, found: Found<'_>, need: Need, header_len: usize) -> Option<String> {
    match (found, need) {
        (Found::Value, _) => None,
        (Found::NoColumn | Found::CutOff(_) | Found::Empty, Need::IfPresent) => None,
        (Found::NoColumn, Need::Required) => Some(format!("has no {column} column")),
        (Found::CutOff(fields), Need::Required) => Some(format!(
            "has no {column} value: the row ends after {fields} of {header_len} columns"
        )),
        (Found::Empty, Need::Required) => {
            Some(format!("has no {column} value: the field is empty"))
        }
        (Found::NotANumber(text, kind), _) => {
            Some(format!("has {column} '{text}', which is not {kind}"))
        }
        (Found::NotFinite(text), _) => Some(format!(
            "has {column} '{text}', which is not a finite number"
        )),
    }
}

/// Where a record is, for a message: the file and, when known, the line.
fn location(path: &Path, record: &csv::StringRecord) -> String {
    match record.position() {
        Some(pos) => format!("{}, line {}", path.display(), pos.line()),
        None => path.display().to_string(),
    }
}

/// Parse a single CSV record into a BenchRow, with what each numeric column
/// held.
fn parse_row<'a>(
    record: &'a csv::StringRecord,
    hmap: &'a HashMap<String, usize>,
) -> (BenchRow, Vec<(&'static str, Found<'a>)>) {
    let mut fields = Fields {
        record,
        hmap,
        found: Vec::new(),
    };
    let row = BenchRow {
        test: get_str(record, hmap, "test").to_string(),
        cycles: fields.number("cycles", 0),
        repeats: fields.number("repeats", 0),
        warmup: fields.number("warmup", 0),
        threads: fields.number("threads", 0),
        msg_bytes: fields.number("msgBytes", 0),
        wall_median: fields.number("wallMedian", 0.0),
        wall_p10: fields.number("wallP10", 0.0),
        wall_p90: fields.number("wallP90", 0.0),
        // Absent in pre-v1.0.4 CSVs, which read them as 0.0.
        wall_p99: fields.number("wallP99", 0.0),
        wall_p999: fields.number("wallP999", 0.0),
        wall_min: fields.number("wallMin", 0.0),
        wall_max: fields.number("wallMax", 0.0),
        wall_mean: fields.number("wallMean", 0.0),
        wall_stddev: fields.number("wallStddev", 0.0),
        wall_cv: fields.number("wallCV", 0.0),
        calls_per_second: fields.number("callsPerSecond", 0.0),
        stable: fields.number("stable", 1),
        cv_threshold: fields.number("cvThreshold", 0.10),
    };
    (row, fields.found)
}

/* ----------------------------- API ----------------------------- */

/// Load benchmark rows from a CSV file.
///
/// Uses flexible mode and manual field extraction to handle all CSV variants:
/// old/new format, with/without GPU columns, short rows (CPU rows with
/// GPU-extended headers). A numeric field that is missing, empty or not a
/// number gets its default and a NaN or infinity is kept as read;
/// `load_csv_strict` refuses them in the columns a caller names.
pub fn load_csv(path: &Path) -> Result<Vec<BenchRow>, Error> {
    load_csv_strict(path, &[])
}

/// Load benchmark rows, checking every row against the caller's `needs`.
///
/// A caller that computes from or presents a numeric column cannot tell the
/// default `load_csv` gives an unreadable field, or a NaN, from a measured
/// value, so it names the column here with how it needs it. A row that falls
/// short is an error naming the file, the line, the test, the column and
/// what the field held. Every other column is read as `load_csv` reads it.
pub fn load_csv_strict(path: &Path, needs: &[(&str, Need)]) -> Result<Vec<BenchRow>, Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)?;

    let headers = rdr.headers()?.clone();
    let hmap = header_map(&headers);

    // Validate required headers are present
    let required = ["test", "wallMedian", "wallCV", "callsPerSecond"];
    let needed = needs
        .iter()
        .filter(|(_, need)| *need == Need::Required)
        .map(|(column, _)| column);
    for &col in required.iter().chain(needed) {
        if !hmap.contains_key(col) {
            return Err(Error::Parse(format!(
                "missing required column '{}' in {}",
                col,
                path.display()
            )));
        }
    }

    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let (row, found) = parse_row(&record, &hmap);
        for &(column, need) in needs {
            let held = found
                .iter()
                .find(|(name, _)| *name == column)
                .map(|&(_, held)| held)
                .ok_or_else(|| {
                    Error::InvalidArgs(format!("no numeric column '{column}' is read into a row"))
                })?;
            if let Some(problem) = unmet(column, held, need, headers.len()) {
                return Err(Error::Parse(format!(
                    "{}: test '{}' {problem}",
                    location(path, &record),
                    row.test
                )));
            }
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Err(Error::Parse(format!("no data rows in {}", path.display())));
    }

    Ok(rows)
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// @test Parses new-format CSV with stable and cvThreshold columns.
    #[test]
    fn parse_new_format() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,wallMedian,wallP10,wallP90,wallMin,wallMax,wallMean,wallStddev,wallCV,callsPerSecond,stable,cvThreshold").unwrap();
        writeln!(tmp, "Foo.Bar,1000,10,1,1,64,0,0,INFO,0.05,0.04,0.06,0.03,0.07,0.05,0.01,0.2,20000000,1,0.10").unwrap();
        tmp.flush().unwrap();

        let rows = load_csv(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].test, "Foo.Bar");
        assert_eq!(rows[0].stable, 1);
        assert!((rows[0].cv_threshold - 0.10).abs() < 1e-9);
    }

    /// @test Parses old-format CSV without stable/cvThreshold; defaults apply.
    #[test]
    fn parse_old_format() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,wallMedian,wallP10,wallP90,wallMin,wallMax,wallMean,wallStddev,wallCV,callsPerSecond,timestamp,gitHash,hostname,platform").unwrap();
        writeln!(tmp, "Baz.Qux,5000,5,2,1,64,0,0,INFO,0.10,0.09,0.11,0.08,0.12,0.10,0.01,0.1,10000000,2025-12-22T07:13:11Z,abc123,host,x86_64").unwrap();
        tmp.flush().unwrap();

        let rows = load_csv(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].test, "Baz.Qux");
        // Defaults for old format
        assert_eq!(rows[0].stable, 1);
        assert!((rows[0].cv_threshold - 0.10).abs() < 1e-9);
    }

    /// @test Parses GPU-extended CSV where CPU rows are shorter than header.
    #[test]
    fn parse_gpu_header_short_rows() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        // GPU-extended header with 25 columns
        writeln!(tmp, "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,wallMedian,wallP10,wallP90,wallMin,wallMax,wallMean,wallStddev,wallCV,callsPerSecond,timestamp,gitHash,hostname,platform,gpuModel,computeCapability,kernelTimeUs").unwrap();
        // CPU row only has 22 columns (ends at platform)
        writeln!(tmp, "Foo.Cpu,1000,10,1,1,64,0,0,INFO,0.05,0.04,0.06,0.03,0.07,0.05,0.01,0.2,20000000,2025-01-01,abc,host,x86_64").unwrap();
        // GPU row has all 25 columns
        writeln!(tmp, "Foo.Gpu,1000,10,1,1,64,0,0,INFO,0.03,0.02,0.04,0.01,0.05,0.03,0.005,0.15,33333333,2025-01-01,abc,host,x86_64,RTX 5000,8.9,11.7").unwrap();
        tmp.flush().unwrap();

        let rows = load_csv(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].test, "Foo.Cpu");
        assert_eq!(rows[1].test, "Foo.Gpu");
        // Both should parse correctly
        assert!((rows[0].wall_median - 0.05).abs() < 1e-9);
        assert!((rows[1].wall_median - 0.03).abs() < 1e-9);
    }

    /// @test Empty CSV returns an error.
    #[test]
    fn empty_csv_errors() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "test,cycles,repeats,warmup,threads,msgBytes,console,nonBlocking,minLevel,wallMedian,wallP10,wallP90,wallMin,wallMax,wallMean,wallStddev,wallCV,callsPerSecond").unwrap();
        tmp.flush().unwrap();

        let result = load_csv(tmp.path());
        assert!(result.is_err());
    }

    /// @test Missing required column returns an error.
    #[test]
    fn missing_column_errors() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "test,cycles").unwrap();
        writeln!(tmp, "Foo,1000").unwrap();
        tmp.flush().unwrap();

        let result = load_csv(tmp.path());
        assert!(result.is_err());
    }

    /// Write `rows` under the four columns every results CSV carries.
    fn minimal_csv(rows: &[&str]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "test,wallMedian,wallCV,callsPerSecond").unwrap();
        for row in rows {
            writeln!(tmp, "{row}").unwrap();
        }
        tmp.flush().unwrap();
        tmp
    }

    const MEASURED: [(&str, Need); 2] =
        [("wallMedian", Need::Required), ("wallCV", Need::Required)];

    /// @test A strict column that is not a number names the file, line, test, column and text.
    #[test]
    fn strict_column_not_a_number_errors() {
        let tmp = minimal_csv(&["A,1,0.1,1", "B,1,garbage,1"]);
        let err = load_csv_strict(tmp.path(), &MEASURED).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "parse error: {}, line 3: test 'B' has wallCV 'garbage', which is not a number",
                tmp.path().display()
            )
        );

        let tmp = minimal_csv(&["A,n/a,0.1,1"]);
        let err = load_csv_strict(tmp.path(), &MEASURED).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "parse error: {}, line 2: test 'A' has wallMedian 'n/a', which is not a number",
                tmp.path().display()
            )
        );
    }

    /// @test An empty strict column is an error, not a zero.
    #[test]
    fn strict_column_empty_errors() {
        let tmp = minimal_csv(&["A,1,,1"]);
        let err = load_csv_strict(tmp.path(), &MEASURED).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "parse error: {}, line 2: test 'A' has no wallCV value: the field is empty",
                tmp.path().display()
            )
        );
    }

    /// @test A row that ends before a strict column is an error, not a zero.
    #[test]
    fn strict_column_cut_off_by_a_short_row_errors() {
        let tmp = minimal_csv(&["A,1"]);
        let err = load_csv_strict(tmp.path(), &MEASURED).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "parse error: {}, line 2: test 'A' has no wallCV value: the row ends after 2 of 4 columns",
                tmp.path().display()
            )
        );
    }

    /// @test A strict column holding 0 loads as the value 0.
    #[test]
    fn strict_column_zero_is_a_value() {
        let tmp = minimal_csv(&["A,1,0,1"]);
        let rows = load_csv_strict(tmp.path(), &MEASURED).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].wall_cv, 0.0);
        assert_eq!(rows[0].wall_median, 1.0);
    }

    /// @test Columns not named strict keep their defaults, as do short GPU-style rows.
    #[test]
    fn strict_columns_leave_the_others_lenient() {
        let tmp = minimal_csv(&["A,1,0.1,garbage"]);
        let rows = load_csv_strict(tmp.path(), &MEASURED).unwrap();
        assert_eq!(rows[0].calls_per_second, 0.0);

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            "test,wallMedian,wallCV,callsPerSecond,gpuModel,kernelTimeUs"
        )
        .unwrap();
        writeln!(tmp, "Foo.Cpu,0.05,0.2,20000000").unwrap();
        writeln!(tmp, "Foo.Gpu,0.03,0.15,33333333,Generic GPU,11.7").unwrap();
        tmp.flush().unwrap();
        let rows = load_csv_strict(tmp.path(), &MEASURED).unwrap();
        assert_eq!(rows.len(), 2);
    }

    /// @test The lenient loader still gives an unreadable field its default.
    #[test]
    fn lenient_load_defaults_unreadable_fields() {
        let tmp = minimal_csv(&["A,1,garbage,1", "B,1"]);
        let rows = load_csv(tmp.path()).unwrap();
        assert_eq!(rows[0].wall_cv, 0.0);
        assert_eq!(rows[1].wall_cv, 0.0);
    }

    /// Write `rows` under `header`.
    fn csv_with(header: &str, rows: &[&str]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "{header}").unwrap();
        for row in rows {
            writeln!(tmp, "{row}").unwrap();
        }
        tmp.flush().unwrap();
        tmp
    }

    /// @test A required column holding NaN or an infinity is an error, not a measurement.
    #[test]
    fn strict_column_not_finite_errors() {
        for (row, column, text) in [
            ("A,nan,0.1,1", "wallMedian", "nan"),
            ("A,1,-inf,1", "wallCV", "-inf"),
        ] {
            let tmp = minimal_csv(&[row]);
            let err = load_csv_strict(tmp.path(), &MEASURED).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "parse error: {}, line 2: test 'A' has {column} '{text}', which is not a finite number",
                    tmp.path().display()
                )
            );
        }
    }

    /// @test An IfPresent column may be absent from the header, empty, or cut off.
    #[test]
    fn if_present_column_may_be_left_out() {
        let needs = [
            ("wallP10", Need::IfPresent),
            ("wallP90", Need::IfPresent),
            ("stable", Need::IfPresent),
            ("cycles", Need::IfPresent),
        ];
        // No wallP90 or cycles column; A leaves wallP10 empty; B stops before stable.
        let tmp = csv_with(
            "test,wallMedian,wallCV,callsPerSecond,wallP10,stable",
            &["A,1,0.1,1,,0", "B,1,0.1,1,0.9"],
        );
        let rows = load_csv_strict(tmp.path(), &needs).unwrap();
        assert_eq!(rows[0].wall_p10, 0.0);
        assert_eq!(rows[0].stable, 0);
        assert_eq!(rows[1].wall_p10, 0.9);
        assert_eq!(rows[1].stable, 1);
        assert_eq!(rows[1].wall_p90, 0.0);
        assert_eq!(rows[1].cycles, 0);
    }

    /// @test An IfPresent column must hold a finite number of its kind where a row gives one.
    #[test]
    fn if_present_column_must_be_usable_where_given() {
        for (column, text, problem) in [
            ("wallP10", "garbage", "which is not a number"),
            ("wallP90", "NaN", "which is not a finite number"),
            (
                "cycles",
                "1.5",
                "which is not a whole number from 0 to 4294967295",
            ),
            ("stable", "yes", "which is not a whole number from 0 to 255"),
        ] {
            let tmp = csv_with(
                &format!("test,wallMedian,wallCV,callsPerSecond,{column}"),
                &[&format!("A,1,0.1,1,{text}")],
            );
            let err = load_csv_strict(tmp.path(), &[(column, Need::IfPresent)]).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "parse error: {}, line 2: test 'A' has {column} '{text}', {problem}",
                    tmp.path().display()
                )
            );
        }
    }

    /// @test The lenient loader keeps a NaN as read and defaults text it cannot parse.
    #[test]
    fn lenient_load_keeps_what_it_reads() {
        let tmp = minimal_csv(&["A,nan,inf,garbage"]);
        let rows = load_csv(tmp.path()).unwrap();
        assert!(rows[0].wall_median.is_nan());
        assert_eq!(rows[0].wall_cv, f64::INFINITY);
        assert_eq!(rows[0].calls_per_second, 0.0);
    }

    /// @test A need for a column no row field is read from is an error, not a silent pass.
    #[test]
    fn need_for_an_unread_column_errors() {
        let tmp = minimal_csv(&["A,1,0.1,1"]);
        let err = load_csv_strict(tmp.path(), &[("kernelTimeUs", Need::IfPresent)]).unwrap_err();
        assert!(
            err.to_string().contains("no numeric column 'kernelTimeUs'"),
            "{err}"
        );
    }

    /// @test A strict column absent from the header is a missing column.
    #[test]
    fn strict_column_absent_from_header_errors() {
        let tmp = minimal_csv(&["A,1,0.1,1"]);
        let err = load_csv_strict(tmp.path(), &[("wallP90", Need::Required)]).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required column 'wallP90'"),
            "{err}"
        );
    }
}
