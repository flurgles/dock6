# DOCK6.13.1 — CLAUDE.md

## Overview
DOCK is a molecular docking program written in C++ (with some C and Fortran utilities).
Identifies binding geometries and interactions of small molecules (ligands) to macromolecular targets (receptors).

## Project Structure
```
.
├── bin/                     # Built executables (created at install time)
├── docs/                    # User manual (HTML)
├── install/                 # Build system root
│   ├── configure            # Config selection script
│   ├── config.h             # Active configuration (generated, then edited)
│   ├── Makefile             # Top-level build targets
│   ├── rules.h              # Shared make rules/suffixes
│   ├── homebrew             # macOS + Homebrew GCC config template
│   ├── gnu                  # Generic GCC/Linux config template
│   └── test/                # Test suite (per-feature directories)
├── src/                     # All source code
│   ├── dock/                # Main dock6 executable (C++)
│   │   ├── nab/             # NAB library (C, used by Amber scoring)
│   │   ├── gzstream/        # gzstream (C++ zlib wrapper)
│   │   └── tnt/             # Template Numerical Toolkit headers
│   ├── accessories/         # showbox, showsphere, sphgen, sphere_selector (F77/C++)
│   ├── antechamber/         # antechamber, atomtype, bondtype, etc. (C/C++)
│   ├── docktools/           # chemgrids, solvgrids, convgrids (F77/C)
│   ├── gbsa_grids/          # nchemgrid_GB, nchemgrid_SA (F77)
│   ├── grid/                # grid utility (mixed C/F77)
│   ├── mopac6/              # MOPAC6 semi-empirical (F77)
│   ├── rdkit/               # RDKit integration (optional)
│   ├── resp/                # RESP charge derivation (F77)
│   └── tleap/               # teLeap (C, uses yacc/bison parser)
├── parameters/              # Force field parameter files
├── template_pipeline/       # Pipeline templates
├── tutorials/               # Tutorial inputs
├── CLAUDE.md                # This file
└── project.log              # Build log
```

## Build System

### Commands (from `install/`)
```bash
./configure homebrew          # Select homebrew config → creates config.h
make all                      # Build dock6 + all utilities
make dock                     # Build only dock6
make utils                    # Build only utilities
make test                     # Run test suite
make clean                    # Remove object files
make -i distclean             # Full clean (remove config.h too)
```

### Configuration
The build is controlled by `install/config.h` (created by `./configure`, manually edited afterward). Key variables:

| Variable | Purpose |
|---|---|
| `CC`, `CXX`, `FC` | C, C++, Fortran compilers |
| `CFLAGS`, `CXXFLAGS`, `FFLAGS` | Compiler flags |
| `LOAD` | Linker command |
| `LIBS` | Linker flags |
| `LEX` | Lex/flex command |
| `YACC` | Yacc/bison command |
| `OCFLAGS` | High-optimization flags (NAB library) |
| `DOCK_SUFFIX` | Executable suffix (e.g., `.rdkit`) |
| `SFX` | Suffix variant (some Makefiles) |

### Compilation Rules (from `install/rules.h`)
- `.cpp.o`: `$(CXX) -c $(CXXFLAGS) -o $@ $(DOCKBUILDFLAGS) $<`
- `.c.o`: `$(CC) -c $(CPPFLAGS) $(CFLAGS) -o $@ $(DOCKBUILDFLAGS) $<`
- `.f.o`: `$(FC) -c $(FFLAGS) -o $@ $(DOCKBUILDFLAGS) $<`
- `.F.o`: `$(FC) -c $(FPPFLAGS) $(FFLAGS) -o $@ $(DOCKBUILDFLAGS) $<`

### Build order (from `src/Makefile`)
1. `dock/` — main dock6 executable
2. `accessories/`, `gbsa_grids/`, `grid/`, `tleap/`, `antechamber/`, `mopac6/src/`, `resp/`, `docktools/` — utilities

### macOS-Specific Issues
- GCC must target the correct SDK (`MacOSX26.sdk`, not `MacOSX14.sdk`)
- Use Homebrew GCC 15 (`gcc-15`, `g++-15`, `gfortran-15`) on modern macOS
- `-D_DARWIN_C_SOURCE -D_XOPEN_SOURCE=600` required for `strdup` and POSIX functions in C99/C11 mode
- `-Wl,-ld_classic` needed for linker compatibility (deprecated but still works)
- `flex` with `-D_DARWIN_C_SOURCE -D_XOPEN_SOURCE=600` for NAB lex file generation
- tleap requires `yacc` (uses bison on macOS)

## Key Source Files
- `src/dock/dock.cpp` — main() entry point
- `src/dock/version.h` — version string `DOCK v6.13`
- `install/config.h` — active build configuration
- `install/rules.h` — shared make rules

## Test Suite
Located in `install/test/` with per-feature directories. Run with `make test` from `install/`.

## Important Notes
- **Never edit config templates** (`install/gnu`, `install/homebrew`); edit `install/config.h` instead
- The main `dock6` executable links NO Fortran code; Fortran is only in separate utilities
- NAB library provides Amber scoring support
- gzstream provides gzipped file I/O

## Context Management (Token Control)
Use context-mode (`ctx_execute`, `ctx_execute_file`, `ctx_search`, `ctx_stats`) for ALL large-output operations. Bash only for whitelist ops (file mutations, git writes, navigation, echo). pi-dcp auto-dedupes and purges errors; call `compress(startId, endId, topic, summary)` after closed work-streams. Check usage: `ctx_stats`. See `context-mode` and `pi-dcp` skills for details.

### Aggressive Token Control Protocol
1. **Default to `ctx_execute`** for ALL data processing (grep, awk, CSV, JSON, logs, multi-file scans). Only raw bytes enter sandbox; only your `console.log`/`print` summary enters context.
2. **`ctx_execute_file` for large file reads** — process in sandbox, emit only derived answer.
3. **`ctx_batch_execute` + `queries`** for multi-command + query patterns — fetch+query in one round trip.
4. **Compress every ~10 tool calls** or when `ctx_stats` shows >80KB entered. Use `compress(startId, endId, topic, summary)` to replace tool spans with lossless summary.
5. **`ctx_index` + `ctx_search`** for docs/repos you'll query repeatedly — never re-read raw.
6. **NEVER** `bash` + `grep | head -100` or `cat large.log` — use `ctx_execute` with processing code.
7. **Bash whitelist only**: `edit`, `write`, `git add/commit/push`, `cd`, `ls`, `mkdir`, `echo`, `mv`, `cp`, `rm`.
8. **Monitor**: `ctx_stats` every 5-10 turns; if "entered context" > 100KB, compress immediately.
