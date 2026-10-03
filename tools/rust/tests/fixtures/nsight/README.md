# Recorded Nsight tool output

What `nsys` and `ncu` printed on the Jetson AGX Thor reference rig (Nsight
Systems 2025.3.2, Nsight Compute 2025.3.1) for demo 02's reports, the ones
walkthrough 11 reads. `tests/nsight_report.rs` replays these through fake
`nsys` and `ncu` executables, and the reader's unit tests parse them, so
`bench nsight-parse parse` is tested against real output on any machine.
Trailing blanks are removed; nothing else is edited.

| File                                  | Command                                                                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nsys_stats_<report>.out`             | `nsys stats --report <report> --format csv <export>.sqlite`, for the four CUDA summaries of `bench run --profile nsight` on G0, after `nsys export --type sqlite` |
| `nsys_stats_no_kernels.out`           | the same for `cuda_gpu_kern_sum`, on a report that holds no kernel (exit 0)                                                                                       |
| `nsys_stats_beside_runner_export.err` | `nsys stats --report cuda_gpu_kern_sum --format csv profile.nsys-rep` with the export `bench run` left beside the report: the refusal (stderr, exit 1)            |
| `ncu_import_per_kernel.out`           | `ncu --import kernel_profile.ncu-rep --csv --print-summary per-kernel` for `bench run --profile ncu` on the two kernel tests                                      |
| `ncu_without_import.out`              | `ncu --csv --print-summary per-kernel kernel_profile.ncu-rep`, without `--import` (stdout, exit 1)                                                                |
| `ncu_import_damaged.out`              | what `ncu --import` printed for a truncated copy of that report (exit 1)                                                                                          |

The `expected_*.csv` files are the CSVs the Python `nsight-parse` wrote on the
same rig, with the real tools, from the reports the outputs above were recorded
from: `expected_nsys_summaries.csv` for the Nsight Systems report (also what a
folder with that report and a truncated Nsight Compute report yields),
`expected_ncu_metrics.csv` for the Nsight Compute report, and
`expected_all_reports.csv` for a folder with both. The command must write them
byte for byte, CRLF line ends included, which `.gitattributes` keeps.
