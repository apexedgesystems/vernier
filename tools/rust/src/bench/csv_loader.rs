//! CSV loading and validation for benchmark results.
//!
//! Handles all CSV variants produced by the benchmarking framework:
//! - Old-format (no stable/cvThreshold columns, has metadata columns)
//! - New-format (with stable/cvThreshold)
//! - GPU-extended (43 columns, CPU rows may be shorter than header)
//!
//! Uses manual field extraction (not serde Deserialize) so that rows shorter
//! than the header are handled gracefully — missing trailing columns get defaults.

use std::collections::HashMap;
use std::path::Path;

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
    pub wall_min: f64,
    pub wall_max: f64,
    pub wall_mean: f64,
    pub wall_stddev: f64,
    pub wall_cv: f64,
    pub calls_per_second: f64,
    pub stable: u8,
    pub cv_threshold: f64,
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

/// Get a float field, defaulting to 0.0 if missing or empty.
fn get_f64(record: &csv::StringRecord, hmap: &HashMap<String, usize>, name: &str) -> f64 {
    let s = get_str(record, hmap, name);
    if s.is_empty() {
        0.0
    } else {
        s.parse().unwrap_or(0.0)
    }
}

/// Get a u32 field, defaulting to 0 if missing or empty.
fn get_u32(record: &csv::StringRecord, hmap: &HashMap<String, usize>, name: &str) -> u32 {
    let s = get_str(record, hmap, name);
    if s.is_empty() {
        0
    } else {
        s.parse().unwrap_or(0)
    }
}

/// Get a u8 field, defaulting to the provided value if missing or empty.
fn get_u8_or(
    record: &csv::StringRecord,
    hmap: &HashMap<String, usize>,
    name: &str,
    default: u8,
) -> u8 {
    let s = get_str(record, hmap, name);
    if s.is_empty() {
        default
    } else {
        s.parse().unwrap_or(default)
    }
}

/// Get a f64 field, defaulting to the provided value if missing or empty.
fn get_f64_or(
    record: &csv::StringRecord,
    hmap: &HashMap<String, usize>,
    name: &str,
    default: f64,
) -> f64 {
    let s = get_str(record, hmap, name);
    if s.is_empty() {
        default
    } else {
        s.parse().unwrap_or(default)
    }
}

/// Parse a single CSV record into a BenchRow.
fn parse_row(record: &csv::StringRecord, hmap: &HashMap<String, usize>) -> BenchRow {
    BenchRow {
        test: get_str(record, hmap, "test").to_string(),
        cycles: get_u32(record, hmap, "cycles"),
        repeats: get_u32(record, hmap, "repeats"),
        warmup: get_u32(record, hmap, "warmup"),
        threads: get_u32(record, hmap, "threads"),
        msg_bytes: get_u32(record, hmap, "msgBytes"),
        wall_median: get_f64(record, hmap, "wallMedian"),
        wall_p10: get_f64(record, hmap, "wallP10"),
        wall_p90: get_f64(record, hmap, "wallP90"),
        wall_min: get_f64(record, hmap, "wallMin"),
        wall_max: get_f64(record, hmap, "wallMax"),
        wall_mean: get_f64(record, hmap, "wallMean"),
        wall_stddev: get_f64(record, hmap, "wallStddev"),
        wall_cv: get_f64(record, hmap, "wallCV"),
        calls_per_second: get_f64(record, hmap, "callsPerSecond"),
        stable: get_u8_or(record, hmap, "stable", 1),
        cv_threshold: get_f64_or(record, hmap, "cvThreshold", 0.10),
    }
}

/* ----------------------------- API ----------------------------- */

/// Load benchmark rows from a CSV file.
///
/// Uses flexible mode and manual field extraction to handle all CSV variants:
/// old/new format, with/without GPU columns, short rows (CPU rows with
/// GPU-extended headers).
pub fn load_csv(path: &Path) -> Result<Vec<BenchRow>, Error> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)?;

    let headers = rdr.headers()?.clone();
    let hmap = header_map(&headers);

    // Validate required headers are present
    let required = ["test", "wallMedian", "wallCV", "callsPerSecond"];
    for &col in &required {
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
        rows.push(parse_row(&record, &hmap));
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
}
