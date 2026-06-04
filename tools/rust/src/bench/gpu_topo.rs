//! GPU/CPU affinity mapper.
//!
//! `nvidia-smi topo -m` reports the PCIe/NVLink interconnect between every
//! pair of GPUs and the kernel-reported CPU affinity for each GPU. Each PCIe
//! device also exposes its NUMA node at `/sys/bus/pci/devices/<bdf>/numa_node`,
//! and `/sys/devices/system/node/nodeN/cpulist` enumerates the CPUs in that
//! node. Cross-referencing the three produces a precise "for benchmarks on
//! GPU i, pin host threads to cores X-Y" recommendation.
//!
//! All discovery shells out to nvidia-smi + reads /sys; no CUDA link.
//! When the inputs are unavailable (no GPU, /sys masked), a single Warn
//! entry is emitted explaining what was missing.
//!
//! Output:
//!  - `print_results` -- ANSI-colored table for terminals
//!  - `print_results_json` -- machine-readable for CI

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde::Serialize;

/* ----------------------------- Types ----------------------------- */

/// Per-GPU topology and CPU-affinity recommendation row.
///
/// Captures everything `bench gpu-topo` needs to recommend a CPU pin
/// list for a given device: PCI / NUMA placement, the affinity mask
/// reported by `nvidia-smi topo -m`, and peer-to-peer links to other
/// GPUs on the host.
#[derive(Debug, Clone, Serialize)]
pub struct GpuTopology {
    pub gpu_index: u32,
    pub gpu_name: String,
    pub pci_bus_id: String,
    pub numa_node: Option<i32>,
    pub cpu_affinity_raw: Option<String>, // "0-15,128-143" style from nvidia-smi topo -m
    pub recommended_cpu_list: Option<String>, // /sys/devices/system/node/nodeN/cpulist
    pub peers: Vec<PeerLink>,
}

/// One entry in `GpuTopology::peers` -- the link to another GPU on the host.
#[derive(Debug, Clone, Serialize)]
pub struct PeerLink {
    pub peer_index: u32,
    pub link_type: String, // "NV1", "PIX", "SYS", etc. -- nvidia-smi notation
}

/// Whole-host topology report returned by `discover()`.
///
/// `warnings` accumulates non-fatal diagnostics (missing nvidia-smi,
/// PCI sysfs paths that didn't resolve) so callers can surface them
/// without crashing.
#[derive(Debug, Clone, Serialize)]
pub struct TopologyReport {
    pub gpus: Vec<GpuTopology>,
    pub warnings: Vec<String>,
}

/* ----------------------------- API ----------------------------- */

/// Discover GPU topology and CPU affinity recommendations.
pub fn discover() -> TopologyReport {
    let mut warnings = Vec::new();

    let bus_ids = match nvidia_smi_bus_ids() {
        Some(ids) if !ids.is_empty() => ids,
        Some(_) => {
            warnings.push("nvidia-smi reports zero GPUs; recommendations limited.".to_string());
            Vec::new()
        }
        None => {
            warnings.push("nvidia-smi unavailable or failed; recommendations limited.".to_string());
            return TopologyReport {
                gpus: Vec::new(),
                warnings,
            };
        }
    };

    let topo_matrix = nvidia_smi_topo_matrix();
    if topo_matrix.is_none() {
        warnings.push("nvidia-smi topo -m failed; peer-link annotations omitted.".to_string());
    }

    let gpus = bus_ids
        .iter()
        .enumerate()
        .map(|(i, bus)| {
            let idx = i as u32;
            let numa = read_numa_node(bus);
            let cpu_list = numa.and_then(read_cpu_list);
            let (affinity, peers) = topo_matrix
                .as_ref()
                .map(|m| (m.cpu_affinity_for(idx), m.peers_for(idx)))
                .unwrap_or((None, Vec::new()));
            GpuTopology {
                gpu_index: idx,
                gpu_name: nvidia_smi_name(idx).unwrap_or_else(|| "?".into()),
                pci_bus_id: bus.clone(),
                numa_node: numa,
                cpu_affinity_raw: affinity,
                recommended_cpu_list: cpu_list,
                peers,
            }
        })
        .collect();

    TopologyReport { gpus, warnings }
}

/// Render the report as a human-readable table.
pub fn print_results(report: &TopologyReport) {
    println!();
    println!("=== GPU / CPU Affinity Topology ===");
    println!();

    if report.gpus.is_empty() {
        for w in &report.warnings {
            println!("  \x1b[93m[WARN]\x1b[0m {w}");
        }
        return;
    }

    println!(
        "  {:<4} {:<28} {:<14} {:<6} {:<18} Recommended CPUs",
        "GPU", "Name", "Bus ID", "NUMA", "Affinity (nv-smi)"
    );
    println!("  {}", "-".repeat(98));
    for g in &report.gpus {
        let numa = g
            .numa_node
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let aff = g.cpu_affinity_raw.clone().unwrap_or_else(|| "-".into());
        let rec = g.recommended_cpu_list.clone().unwrap_or_else(|| "-".into());
        println!(
            "  {:<4} {:<28} {:<14} {:<6} {:<18} {}",
            g.gpu_index,
            truncate(&g.gpu_name, 28),
            truncate(&g.pci_bus_id, 14),
            numa,
            truncate(&aff, 18),
            rec,
        );
    }

    let any_peers = report.gpus.iter().any(|g| !g.peers.is_empty());
    if any_peers {
        println!();
        println!("  Peer links:");
        for g in &report.gpus {
            if g.peers.is_empty() {
                continue;
            }
            let summary = g
                .peers
                .iter()
                .map(|p| format!("GPU{}={}", p.peer_index, p.link_type))
                .collect::<Vec<_>>()
                .join("  ");
            println!("    GPU{}: {}", g.gpu_index, summary);
        }
    }

    if !report.warnings.is_empty() {
        println!();
        for w in &report.warnings {
            println!("  \x1b[93m[WARN]\x1b[0m {w}");
        }
    }
    println!();
    println!(
        "  Tip: pin worker threads to a NUMA-affine CPU range when running\n        \
         per-GPU workloads, e.g. `taskset -c <recommended-cpus> ./your_test`."
    );
    println!();
}

/// Machine-readable JSON output.
pub fn print_results_json(report: &TopologyReport) {
    match serde_json::to_string_pretty(report) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("(failed to serialize topology: {e})"),
    }
}

/* ----------------------------- Discovery helpers ----------------------------- */

fn nvidia_smi_bus_ids() -> Option<Vec<String>> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    )
}

fn nvidia_smi_name(idx: u32) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            &idx.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Parse the GPUs * GPUs matrix from `nvidia-smi topo -m`. Each row also has
/// a "CPU Affinity" column (and on newer drivers, "NUMA Affinity").
struct TopoMatrix {
    gpus: usize,
    rows: Vec<Vec<String>>, // peer link table cells
    cpu_affinity: Vec<String>,
}

impl TopoMatrix {
    fn cpu_affinity_for(&self, gpu: u32) -> Option<String> {
        self.cpu_affinity.get(gpu as usize).cloned()
    }
    fn peers_for(&self, gpu: u32) -> Vec<PeerLink> {
        let i = gpu as usize;
        if i >= self.gpus {
            return Vec::new();
        }
        self.rows[i]
            .iter()
            .enumerate()
            .filter_map(|(j, cell)| {
                if j == i || cell == "X" || cell.is_empty() {
                    None
                } else {
                    Some(PeerLink {
                        peer_index: j as u32,
                        link_type: cell.clone(),
                    })
                }
            })
            .collect()
    }
}

fn nvidia_smi_topo_matrix() -> Option<TopoMatrix> {
    let out = Command::new("nvidia-smi")
        .args(["topo", "-m"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    parse_topo_matrix(&text)
}

fn parse_topo_matrix(text: &str) -> Option<TopoMatrix> {
    // Format (example):
    //         GPU0    GPU1    CPU Affinity    NUMA Affinity
    //   GPU0  X       NV1     0-15            0
    //   GPU1  NV1     X       16-31           1
    //   ...
    // Lines beyond the GPU rows describe NICs etc.; we ignore them.
    let mut lines = text.lines();
    let header = lines.find(|l| l.contains("GPU0"))?;
    let columns: Vec<&str> = header.split_whitespace().collect();
    let gpu_count = columns.iter().take_while(|c| c.starts_with("GPU")).count();
    if gpu_count == 0 {
        return None;
    }
    let aff_col = columns.iter().position(|c| *c == "CPU")?;
    // CPU column is "CPU Affinity"; absolute index is 1 (label) + aff_col
    let mut rows = Vec::with_capacity(gpu_count);
    let mut cpu_affinity = Vec::with_capacity(gpu_count);
    for _ in 0..gpu_count {
        let line = loop {
            match lines.next() {
                Some(l) if l.trim_start().starts_with("GPU") => break l,
                Some(_) => continue,
                None => return None,
            }
        };
        let cells: Vec<&str> = line.split_whitespace().collect();
        if cells.len() < 1 + gpu_count + 1 {
            return None;
        }
        let peer_cells: Vec<String> = cells[1..1 + gpu_count]
            .iter()
            .map(|s| s.to_string())
            .collect();
        rows.push(peer_cells);
        // "CPU Affinity" cell sits at index 1 + gpu_count (the header gpu_count GPUs + name col)
        let aff_cell_idx = 1 + gpu_count;
        cpu_affinity.push(
            cells
                .get(aff_cell_idx)
                .map(|s| s.to_string())
                .unwrap_or_default(),
        );
    }
    let _ = aff_col; // header position retained for future use; we index by structure above
    Some(TopoMatrix {
        gpus: gpu_count,
        rows,
        cpu_affinity,
    })
}

fn read_numa_node(bus_id: &str) -> Option<i32> {
    // PCI bus id from nvidia-smi looks like "00000000:01:00.0"; normalize to "0000:01:00.0".
    let normalized = normalize_bus_id(bus_id)?;
    let path: PathBuf = format!("/sys/bus/pci/devices/{}/numa_node", normalized).into();
    let raw = fs::read_to_string(&path).ok()?;
    raw.trim().parse::<i32>().ok()
}

fn normalize_bus_id(bus_id: &str) -> Option<String> {
    // nvidia-smi: "00000000:01:00.0"; sysfs uses "0000:01:00.0" or full domain "00000000:01:00.0".
    // Try both forms in priority order.
    let stripped = bus_id.trim_start_matches('0').trim_start_matches(':');
    if PathBuf::from(format!("/sys/bus/pci/devices/{}", bus_id)).exists() {
        return Some(bus_id.to_string());
    }
    let short = format!("0000:{}", stripped.trim_start_matches(':'));
    if PathBuf::from(format!("/sys/bus/pci/devices/{}", short)).exists() {
        return Some(short);
    }
    Some(bus_id.to_string())
}

fn read_cpu_list(numa: i32) -> Option<String> {
    if numa < 0 {
        return None;
    }
    let path: PathBuf = format!("/sys/devices/system/node/node{}/cpulist", numa).into();
    fs::read_to_string(&path).ok().map(|s| s.trim().to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('+');
        out
    }
}
