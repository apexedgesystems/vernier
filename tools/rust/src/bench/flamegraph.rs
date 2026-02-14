//! FlameGraph generation: orchestrate perf script + stackcollapse + flamegraph.pl.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::{find_in_path, Error};

/* ----------------------------- FlameGraphConfig ----------------------------- */

/// Configuration for flamegraph generation.
#[derive(Debug, Clone)]
pub struct FlameGraphConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub baseline: Option<PathBuf>,
}

/* ----------------------------- API ----------------------------- */

/// Generate an SVG flamegraph from a perf.data file.
///
/// Requires FlameGraph scripts (flamegraph.pl, stackcollapse-perf.pl) to be
/// available via `$FLAMEGRAPH_DIR`, `~/FlameGraph`, or PATH.
pub fn generate_flamegraph(cfg: &FlameGraphConfig) -> Result<PathBuf, Error> {
    let fg_dir = find_flamegraph_dir()?;

    if !cfg.input.is_file() {
        return Err(Error::InvalidArgs(format!(
            "input file not found: {}",
            cfg.input.display()
        )));
    }

    if let Some(ref baseline) = cfg.baseline {
        generate_differential(cfg, baseline, &fg_dir)
    } else {
        generate_standard(cfg, &fg_dir)
    }
}

/* ----------------------------- Standard FlameGraph ----------------------------- */

fn generate_standard(cfg: &FlameGraphConfig, fg_dir: &Path) -> Result<PathBuf, Error> {
    let stackcollapse = fg_dir.join("stackcollapse-perf.pl");
    let flamegraph_pl = fg_dir.join("flamegraph.pl");

    // perf script -i <input> | stackcollapse-perf.pl | flamegraph.pl > output.svg
    let perf_script = Command::new("perf")
        .args(["script", "-i"])
        .arg(&cfg.input)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("perf script: {e}")))?;

    let collapse = Command::new("perl")
        .arg(&stackcollapse)
        .stdin(perf_script.stdout.unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("stackcollapse-perf.pl: {e}")))?;

    let output_file = std::fs::File::create(&cfg.output)?;

    let flamegraph = Command::new("perl")
        .arg(&flamegraph_pl)
        .stdin(collapse.stdout.unwrap())
        .stdout(output_file)
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("flamegraph.pl: {e}")))?;

    let status = flamegraph.wait_with_output()?;
    if !status.status.success() {
        return Err(Error::Parse("flamegraph.pl failed".to_string()));
    }

    println!("Flamegraph written to: {}", cfg.output.display());
    Ok(cfg.output.clone())
}

/* ----------------------------- Differential FlameGraph ----------------------------- */

fn generate_differential(
    cfg: &FlameGraphConfig,
    baseline: &Path,
    fg_dir: &Path,
) -> Result<PathBuf, Error> {
    let stackcollapse = fg_dir.join("stackcollapse-perf.pl");
    let difffolded = fg_dir.join("difffolded.pl");
    let flamegraph_pl = fg_dir.join("flamegraph.pl");

    if !baseline.is_file() {
        return Err(Error::InvalidArgs(format!(
            "baseline file not found: {}",
            baseline.display()
        )));
    }

    // Collapse baseline
    let base_folded = collapse_perf_data(baseline, &stackcollapse)?;

    // Collapse candidate
    let cand_folded = collapse_perf_data(&cfg.input, &stackcollapse)?;

    // Write folded data to temp files for difffolded.pl
    let tmp_dir = std::env::temp_dir().join(format!("bench_flamegraph_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;
    let base_path = tmp_dir.join("baseline.folded");
    let cand_path = tmp_dir.join("candidate.folded");
    std::fs::write(&base_path, &base_folded)?;
    std::fs::write(&cand_path, &cand_folded)?;

    // difffolded.pl baseline.folded candidate.folded | flamegraph.pl > output.svg
    let diff = Command::new("perl")
        .arg(&difffolded)
        .arg(&base_path)
        .arg(&cand_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("difffolded.pl: {e}")))?;

    let output_file = std::fs::File::create(&cfg.output)?;

    let flamegraph = Command::new("perl")
        .arg(&flamegraph_pl)
        .stdin(diff.stdout.unwrap())
        .stdout(output_file)
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("flamegraph.pl: {e}")))?;

    let status = flamegraph.wait_with_output()?;

    // Cleanup temp files
    let _ = std::fs::remove_dir_all(&tmp_dir);

    if !status.status.success() {
        return Err(Error::Parse("flamegraph.pl failed".to_string()));
    }

    println!(
        "Differential flamegraph written to: {}",
        cfg.output.display()
    );
    Ok(cfg.output.clone())
}

/// Run `perf script | stackcollapse-perf.pl` and return the folded output.
fn collapse_perf_data(perf_data: &Path, stackcollapse: &Path) -> Result<Vec<u8>, Error> {
    let perf_script = Command::new("perf")
        .args(["script", "-i"])
        .arg(perf_data)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::ToolNotFound(format!("perf script: {e}")))?;

    let output = Command::new("perl")
        .arg(stackcollapse)
        .stdin(perf_script.stdout.unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| Error::ToolNotFound(format!("stackcollapse-perf.pl: {e}")))?;

    if !output.status.success() {
        return Err(Error::Parse(format!(
            "stackcollapse-perf.pl failed for {}",
            perf_data.display()
        )));
    }

    Ok(output.stdout)
}

/* ----------------------------- FlameGraph Directory ----------------------------- */

fn find_flamegraph_dir() -> Result<PathBuf, Error> {
    // Check $FLAMEGRAPH_DIR
    if let Ok(val) = std::env::var("FLAMEGRAPH_DIR") {
        let dir = PathBuf::from(&val);
        if dir.join("flamegraph.pl").is_file() {
            return Ok(dir);
        }
    }

    // Check common paths
    let candidates = [
        dirs_from_home("FlameGraph"),
        Some(PathBuf::from("/usr/local/FlameGraph")),
        Some(PathBuf::from("/opt/FlameGraph")),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.join("flamegraph.pl").is_file() {
            return Ok(candidate);
        }
    }

    // Check if flamegraph.pl is in PATH (use its parent dir)
    if let Some(path) = find_in_path("flamegraph.pl") {
        if let Some(parent) = path.parent() {
            return Ok(parent.to_path_buf());
        }
    }

    Err(Error::ToolNotFound(
        "FlameGraph scripts not found; set $FLAMEGRAPH_DIR or clone to ~/FlameGraph".to_string(),
    ))
}

fn dirs_from_home(subdir: &str) -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(subdir))
}

/* ----------------------------- Tests ----------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /// @test Missing input file returns error.
    #[test]
    fn missing_input_errors() {
        let cfg = FlameGraphConfig {
            input: PathBuf::from("/nonexistent/perf.data"),
            output: PathBuf::from("/tmp/flamegraph.svg"),
            baseline: None,
        };
        let result = generate_flamegraph(&cfg);
        assert!(result.is_err());
    }
}
