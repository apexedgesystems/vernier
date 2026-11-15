# Third-Party Source

Files from other projects that libbench's own sources include. Each project
has a folder of its own holding its files exactly as that project released
them: never edited, never reformatted (`.pre-commit-config.yaml` excludes
every `third_party/<project>/` folder from its hooks). They are private to
libbench: not installed, and not reachable from any installed header.

## valgrind

| Item    | Value                                                                                    |
| ------- | ---------------------------------------------------------------------------------------- |
| Version | 3.28.0                                                                                   |
| Source  | `valgrind-3.28.0.tar.bz2` from <https://sourceware.org/pub/valgrind/>, signature checked |
| Files   | `valgrind.h`, `callgrind.h`, `helgrind.h` (`include/`, `callgrind/`, `helgrind/`)        |
| License | BSD-style, stated at the top of each file, applying to that file alone                   |
| Used by | `src/bench/src/CallgrindRequests.cpp`, `src/bench/src/HelgrindRequests.cpp`              |

valgrind's client requests: with them libbench switches callgrind's recording
around each measured window and keeps `contentionRun`'s start gate out of
helgrind's reports, in every build, whether or not valgrind is installed where
it is built. None of the rest of valgrind, which is under the GPL, is copied.
The headers declare their request numbers an interface valgrind does not
change; the five libbench uses are the same in 3.18.1, 3.22.0, 3.24.0 and
3.28.0, and so is the code they compile to on x86-64 and AArch64, so a
libbench built with these headers drives any of those releases. 3.25 is the
first release whose `valgrind.h` supports RISC-V; with an older one a RISC-V
build would have no requests. Defining `NVALGRIND` compiles the requests out.

To update, replace the three files with those of a newer release tarball,
unmodified, and change the version above.
