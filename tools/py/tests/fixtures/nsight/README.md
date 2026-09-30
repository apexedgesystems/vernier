# Recorded Nsight tool output

What `nsys` and `ncu` printed on the Jetson AGX Thor reference rig (Nsight
Systems 2025.3.2, Nsight Compute 2025.3.1) for demo 02's reports, the ones
walkthrough 11 reads. `test_nsight_parser.py` replays these through fake
`nsys` and `ncu` executables, so the parser is tested against real output on
any machine. Trailing blanks are removed; nothing else is edited.

| File                                  | Command                                                                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nsys_stats_<report>.out`             | `nsys stats --report <report> --format csv <export>.sqlite`, for the four CUDA summaries of `bench run --profile nsight` on G0, after `nsys export --type sqlite` |
| `nsys_stats_no_kernels.out`           | the same for `cuda_gpu_kern_sum`, on a report that holds no kernel (exit 0)                                                                                       |
| `nsys_stats_beside_runner_export.err` | `nsys stats --report cuda_gpu_kern_sum --format csv profile.nsys-rep` with the export `bench run` left beside the report: the refusal (stderr, exit 1)            |
| `ncu_import_per_kernel.out`           | `ncu --import kernel_profile.ncu-rep --csv --print-summary per-kernel` for `bench run --profile ncu` on the two kernel tests                                      |
| `ncu_without_import.out`              | `ncu --csv --print-summary per-kernel kernel_profile.ncu-rep`, without `--import` (stdout, exit 1)                                                                |
| `ncu_import_damaged.out`              | what `ncu --import` printed for a truncated copy of that report (exit 1), as the parser reported it                                                               |
