# Recorded compute-sanitizer reports

What compute-sanitizer 2025.4.0 (build 36782660) wrote as `sanitizer.log`
for `bench run --profile compute-sanitizer`, in the project's CUDA
development image (CUDA 13.1) on an NVIDIA RTX 5000 Ada laptop GPU, with a
Release build. `runner.rs`'s unit tests read them, so `bench run`'s verdict
is tested against the real tool's reports on any machine. Trailing blanks
are removed; nothing else is edited.

| File                       | Command (after `bench run`)                                                                                                                   | The tool ended with |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `memcheck_unguarded.txt`   | `BenchDemo_Gpu_04_ComputeSanitizerProfiler --profile compute-sanitizer -- --gtest_filter=ComputeSanitizer.SaxpyUnguarded`                     | status 5            |
| `memcheck_clean.txt`       | `BenchDemo_Gpu_04_ComputeSanitizerProfiler --cycles 1 --repeats 1 --profile compute-sanitizer -- --gtest_filter=ComputeSanitizer.SaxpyKernel` | status 0            |
| `racecheck_clean.txt`      | the same, with `--profile-args racecheck`                                                                                                     | status 0            |
| `synccheck_clean.txt`      | the same, with `--profile-args synccheck`                                                                                                     | status 0            |
| `initcheck_clean.txt`      | the same, with `--profile-args initcheck`                                                                                                     | status 0            |
| `racecheck_race.txt`       | `racy --profile compute-sanitizer --profile-args racecheck`                                                                                   | status 5            |
| `racecheck_same_value.txt` | the same program, given `s` so that only its same-value kernel runs                                                                           | status 0            |

`racy` is a small program, not in the tree: its kernel `racy` has thread 0
store a `__shared__` int (line 11 of its source) and then every thread add
its index to it with no barrier between (line 15), one block of 256 threads;
its kernel `sameValue` has every thread store 7 to a `__shared__` int before
a barrier. `bench run` exited 1 for `memcheck_unguarded.txt` and
`racecheck_race.txt`, reporting 4 and 2 errors, and 0 for the others.
