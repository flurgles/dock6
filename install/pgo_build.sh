#!/bin/bash
# PGO (Profile-Guided Optimization) build for DOCK6 — dock6 binary only.
#
# Three-phase process:
#   1. Instrument dock6   — builds dock6 with -fprofile-generate; utilities
#                           built normally (Fortran Makefiles don't all pass
#                           DOCKBUILDFLAGS through to link lines)
#   2. Training run        — runs test suite to collect profile data
#   3. Optimize dock6      — rebuilds dock6 using collected profile data
#
# Usage:
#   ./pgo_build.sh              # Full PGO build
#   ./pgo_build.sh clean        # Remove PGO artifacts
#
# Profile data directory (preserved after build):
#   /tmp/dock6_pgo_gcda/

set -euo pipefail

PGO_DIR="/tmp/dock6_pgo_gcda"
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$INSTALL_DIR/.." && pwd)"

GEN_FLAGS="-fprofile-generate=$PGO_DIR"
USE_FLAGS="-fprofile-use=$PGO_DIR -fprofile-correction"

clean_pgo() {
    echo "=== Cleaning PGO artifacts ==="
    rm -rf "$PGO_DIR"
    find "$PROJECT_DIR/src/dock" -name '*.gcda' -delete
    find "$PROJECT_DIR/src/dock" -name '*.gcno' -delete
    find "$PROJECT_DIR/src/dock" -name '*.o' -delete
    echo "Done."
    exit 0
}

if [ $# -ge 1 ] && [ "$1" = "clean" ]; then
    clean_pgo
fi

echo "============================================"
echo "  DOCK6 PGO Build (dock6 only)"
echo "============================================"
echo " Profile data: $PGO_DIR"
echo " Project:      $PROJECT_DIR"
echo "============================================"
echo ""

# ---- Phase 1: Instrumentation build (dock6 only) ----
echo ">>> Phase 1/3: Build dock6 with instrumentation"
echo "    Flags: $GEN_FLAGS"
echo ""

cd "$INSTALL_DIR"
mkdir -p "$PGO_DIR"

# Full clean, then build dock6 with PGO and utilities normally
make clean 2>/dev/null || true
LOG1="$INSTALL_DIR/pgo_build_phase1.log"

# Build dock6 with instrumentation
echo "    Building dock6 (PGO instrumented)..."
if ! make DOCKBUILDFLAGS="$GEN_FLAGS" dock 2>&1 | tee -a "$LOG1"; then
    echo "Phase 1 (dock6) failed. Check $LOG1 for details."
    exit 1
fi

# Build utilities normally (no PGO flags — several Fortran Makefiles don't
# include DOCKBUILDFLAGS in their link rules, causing unresolved gcov symbols)
echo "    Building utilities (normal)..."
if ! make utils 2>&1 | tee -a "$LOG1"; then
    echo "Phase 1 (utils) failed. Check $LOG1 for details."
    exit 1
fi

echo "Phase 1 completed successfully."
echo ""

# ---- Phase 2: Training run ----
echo ">>> Phase 2/3: Training with test suite"
echo ""

LOG2="$INSTALL_DIR/pgo_build_phase2.log"
if make test 2>&1 | tee "$LOG2"; then
    echo "Phase 2 completed successfully."
else
    echo "Phase 2 finished (some tests may have failed — PGO data is still usable)."
fi

echo ""

# ---- Phase 3: Optimization build (dock6 only) ----
echo ">>> Phase 3/3: Rebuild dock6 with profile guidance"
echo "    Flags: $USE_FLAGS"
echo ""

cd "$PROJECT_DIR"
# Delete only dock6 .o files, keeping .gcno files for -fprofile-use
find src/dock -name '*.o' -delete
rm -f bin/dock6

cd "$INSTALL_DIR"
LOG3="$INSTALL_DIR/pgo_build_phase3.log"
if ! make DOCKBUILDFLAGS="$USE_FLAGS" dock 2>&1 | tee "$LOG3"; then
    echo "Phase 3 failed. Check $LOG3 for details."
    exit 1
fi

echo ""
echo "============================================"
echo "  PGO build complete!"
echo "============================================"
echo ""
echo "Profile data preserved in: $PGO_DIR"
echo "Clean:   $0 clean"
echo ""

# Verify
if [ -x "$PROJECT_DIR/bin/dock6" ]; then
    echo "dock6 binary:"
    ls -la "$PROJECT_DIR/bin/dock6"
else
    echo "WARNING: dock6 binary not found!"
    exit 1
fi
