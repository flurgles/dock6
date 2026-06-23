# GPU Acceleration for dock6 — Developer Log

> **Purpose**: This is a chronological development log for GPU-accelerated flex docking
> in DOCK6.13.  It covers the full journey: plans, roadblocks, failed approaches,
> root causes, and what ultimately works.  Read it to understand *why* things are the
> way they are, not just *what* was done.
>
> Status tags: ✅ **working** | 🟡 **partial** | ❌ **failed/retracted** | 🔴 **broken**

---

## Phase 1: Infrastructure (Done — Jun 22)

### Plan
Create the GPU API layer: a header declaring `dock_gpu_*` functions, a stub returning
"not available" for CPU builds, and a Metal backend for Apple Silicon.

### Files created
| File | Purpose |
|------|---------|
| `score_dock_gpu.h` | Public API declarations |
| `score_dock_gpu_stub.c` | CPU stub (returns 0/false) |
| `score_dock_gpu_metal.h` | C struct shared with Obj-C++ backend |
| `score_dock_gpu_metal.mm` | Metal implementation |

### Makefile integration
`GPU_BACKEND=auto` → `-DUSE_METAL` on Darwin with `clang++` for `.mm` files.
Stub linked on other platforms.

### Verification
- CPU build (`GPU_BACKEND=cpu`): all tests pass, bit-identical output
- Metal build (`GPU_BACKEND=metal`): singlepoint test works, grid scores match CPU

### Roadblocks
- **LTO stripping**: `-flto` in `CFLAGS`/`CXXFLAGS` stripped all Obj-C++ symbols and
  framework links at link time.  Binary ran but Metal code was dead (stub path).

  **Solution**: Removed `-flto` from `install/config.h` entirely.  Trade-off: slightly
  less optimized CPU code.  Benefit: Metal actually works.

---

## Phase 2: Growth Pre-Pruning (Attempted & Retracted — Jun 22)

### Plan
Score un-minimized partial conformers on GPU immediately after growth, before CPU
minimization.  Keep only top-225 conformers per anchor.  Expected speedup: prune
early, minimize fewer.

### What was done
- Added GPU scoring call in `conf_gen_ag.cpp::grow_periphery()` after seed generation
- Submitted all newly-grown conformers as one GPU batch
- Filtered to top-225 based on GPU scores
- Code guarded by `#ifdef USE_METAL`

### What went wrong 🟡
The conformers were scored **before minimization**, and the un-minimized
scores have ZERO correlation with post-minimization rankings.  Pruning "bad"
pre-minimization poses often discarded the very poses that would minimize to the
best scores.  Result: **no speedup, wrong poses pruned**.

### Diagnosis
- Grid score + IE on un-minimized geometry is dominated by severe steric clashes
  (atoms overlapping from random torsion angles)
- A pose with a single bad torsion will score terribly but may minimize to the
  global minimum
- The `inside_grid` sentinel caused by grids not covering the full placement volume
  made this even worse — many valid poses were `-FLT_MAX`

### Decision
**❌ Retracted.** Pruning on un-minimized scores is fundamentally wrong.  No amount
of tuning threshold parameters can fix this.  All pre-pruning code removed from
`conf_gen_ag.cpp`.

### Lesson
> GPU scoring must act on poses that are already at a valid minimum, not on
> un-minimized intermediates.  The right target is accelerating the scoring function
> *inside* the minimizer.

---

## Phase 3A: Speculative Batch Evaluation in Simplex (Jun 22–23)

### The insight that changed the approach
CPU profiling showed:
- `compute_ligand_internal_energy`: ~17% of CPU time
- `compute_score` (grid): ~15% of CPU time
- Simplex control logic: ~10% of CPU time

Grid + IE together = **~32% of CPU time**.  The Nelder-Mead simplex evaluates
one vertex at a time — but we can batch multiple candidate vertices into a
single GPU launch.

### Key observation
Each simplex iteration evaluates **4 candidate points** (reflect/expand/
outside-contract/inside-contract), one at a time.  The CPU evaluates one, then
decides whether to evaluate the next.  This is sequential, but the 4 candidates
are **independent** — we can evaluate all 4 speculatively in one GPU launch,
then let the CPU choose which to keep.

### Batch points in simplex
| Point | Batch size | When |
|-------|-----------|------|
| Initial N+1 vertices | N+1 | Start of minimization |
| 4 candidates per iteration | 4 | Each main-loop iteration |
| Shrink | N-1 | When all 4 candidates fail |

### GPU kernel design
One thread per pose.  Each thread:
1. Grid score: loops over atoms, trilinear interpolation on 3 grids
2. Internal energy: loops over non-bonded pair list
3. Returns `grid_score + ie_score` (or `-FLT_MAX` sentinel if outside grid)

### Code structure changes
- **`score_dock_gpu.h/c`**: Added `dock_gpu_set_ligand_ie()` and `dock_gpu_batch_score_with_ie()`
- **`score_dock_gpu_metal.mm`**: Two kernel sources — grid-only (`g_pso`) and grid+IE (`g_pso_ie`)
- **`minimizer.cpp`**: New `gpu_batch_eval_scores()` — the batch pipeline
- **`simplex.cpp`**: Restructured for 4-way speculative batch + shrink batch

---

## Roadblock 🔴: GPU Hang in `batch_score_with_ie_kernel`

### Symptom
The grid+IE kernel hangs consistently after 4-5 successful dispatches during
flex docking.  The grid-only kernel works indefinitely (verified through 20+
calls in the same run).  The IE kernel succeeds for the first 4-5 batches
(initial N+1 vertices, first 2-3 speculative iterations), then the GPU stalls
at `[cmdbuf waitUntilCompleted]` and never returns.

### Initial clues
- Hang is **deterministic** — always at call 4 or 5, never immediately
- Metal validation layer: enabled but reports NO errors
- Command buffer status: completed (all correct) for the first 4-5 calls,
  then `waitUntilCompleted` never returns
- `endEncoding` may be the blocking point — the GPU starts executing but
  stalls mid-kernel

### Attempted fixes that did NOT work

#### 1. Persistent buffers (Jun 22)
**Problem**: Replaced per-call `newBufferWithBytes()` allocations for constant
parameters (GridParams, num_atoms, IEParams, num_nb_pairs) with persistent buffers
allocated once in `dock_gpu_init()`.

**Why it didn't work**: The issue isn't allocation — it's execution.  The GPU
completes 4-5 successful dispatches with persistent buffers before stalling.
The first successful dispatches prove the buffer setup is correct.

#### 2. Threadgroup size reduction (Jun 23)
**Problem**: Changed threadgroup size from `MIN(256, maxTotalThreadsPerThreadgroup)`
to `MTLSizeMake(1, 1, 1)` — one thread per threadgroup.

**Why it didn't work**: The IE kernel uses `thread_position_in_grid`, not
threadgroup-local addressing.  1 thread per group just means more threadgroups.
The standalone test uses the same pattern and works fine.  Not the root cause.

#### 3. `@autoreleasepool` wrapping (Jun 23)
**Problem**: Added `@autoreleasepool {}` in both minimizer.cpp (invalid — .cpp
file) and inside the batch function.

**Why it didn't work / fixed**: The `@autoreleasepool` in minimizer.cpp caused a
compile error (Objetive-C keyword in C++ file).  The inner pool in the `.mm` file
didn't change behavior — the Metal objects are already managed correctly.

#### 4. Eliminated debug logging (Jun 23)
**Problem**: `fprintf(stderr)` before/after Metal calls was adding latency.

**Why it didn't work**: The logging is ~lines of text, not the bottleneck.
Removing it didn't change behavior.

#### 5. Buffer zeroing before memcpy (Jun 23)
**Problem**: Zero `g_buf_xyz` before copying new data.

**Why it didn't work**: Uninitialized buffer reads shouldn't cause a hang
(shader would just compute garbage).  The grid-only kernel works without
zeroing and the IE kernel uses the same buffer.

#### 6. Proper IEParams struct packing (Jun 23)
**Problem**: `write_iep` used split `memcpy` (floats then ints), which could
mismatch Metal shader struct padding.

**Why it didn't work**: The `__attribute__((packed))` struct write didn't
change the data layout — both old and new code write an identical 16-byte
pattern.  Not the root cause.

### Current hypothesis and next debugging steps

**Working theory**: The IE kernel hangs at `[enc endEncoding]` or
`[cmdbuf waitUntilCompleted]` after ~4 successful dispatches.  The grid-only
kernel using the same Metal infrastructure works indefinitely.  The only
difference between the IE and grid-only dispatches is:
1. **4 additional buffer bindings** (indices 10-13): `ie_vdwA`, `nb_int`,
   `IEParams`, `num_nb_pairs`
2. **The actual shader workload**: IE kernel has `inside_grid` check + IE loop

**Hypothesis A — Buffer binding**: The `g_buf_nb_int` buffer
(`GPU_MAX_NB_PAIRS * 2 = 32768 * 2 = 65536` ints = 256 KB) combined with
`g_buf_xyz` (~25 MB for 4096 poses × 512 atoms) exceeds some internal Metal
resource limit for a single dispatch.  The accumulation over multiple dispatches
triggers a GC or paging stall.

**Hypothesis B — Shader non-termination**: The IE loop `for (int p = 0; p <
num_nb_pairs; p++)` could read garbage `nb_int` entries beyond the initialized
region.  If `nb_int[p*2]` or `nb_int[p*2+1]` are huge negative values, the
coordinate read `xyz[base + a1*3]` reads out of bounds.  Metal buffers are
bound-checked — an out-of-bounds read returns 0, not a fault.  But an
out-of-bounds on the **output** buffer could theoretically stall.

**Hypothesis C — Metal shader compiler bug**: The second shader compilation
(`lib_ie`) may produce `g_pso_ie` with an incorrect internal state that
manifests after several dispatches.  Could test by inlining the IE code into
the grid-only kernel's shader source and using `g_pso` for both.

### Next debugging steps
1. **Add Metal command buffer error checking** — ✅ done in initial code
2. **Test with 1 thread per threadgroup** — ✅ done, no change
3. **Combine both kernels into one library** (single `newLibraryWithSource`)
4. **Test with tiny xyz buffer** (match standalone test buffer size)
5. **Remove `inside_grid` call** from IE kernel — simplify to bare minimum
6. **Add `print()` to Metal shader** — diagnostic output from GPU
7. **Validate buffer sizes** — ensure output buffer (`g_buf_scores`) isn't
   overflowed by num_atoms + num_nb_pairs debug write (was investigated but
   code was rejected due to exact-text match failure)

---

## What Works ✅

| Component | Status | Note |
|-----------|--------|------|
| GPU init with Metal on Apple Silicon | ✅ | Device detected, grid loaded, IE data uploaded |
| Grid-only batch scoring kernel | ✅ | Works for 20+ dispatches, used as baseline |
| Combined grid+IE kernel (standalone) | ✅ | 100 iterations in isolated test program |
| Dual-path `gpu_batch_eval_scores()` | 🟡 | CPU fallback path works; GPU path hangs |
| Simplex restructuring (4-way batch) | 🟡 | Code compiles, CPU path verified correct |
| Shrink batch | 🟡 | Code compiles |
| Ligand param upload from conf_gen_ag.cpp | ✅ | vdwA/vdwB/charges + ie_vdwA + nb_int |
| `failure_exit` cleanup lambda | ✅ | 6 copies → 1 |
| All `make test` with `GPU_BACKEND=cpu` | ✅ | All pass |
| All `make test` with `GPU_BACKEND=metal` | ✅ | All pass (tests don't exercise flex minimizer) |

---

## What Still Needs to Be Done

### Critical: Fix IE kernel hang 🔴
The top priority.  Grid-only kernel works, IE kernel doesn't.  Standalone test
works, dock6 integration doesn't.  Need to identify the root cause and fix it.

### When hang is fixed: DT100 benchmarking
1. **1A28** (1 torsion, ~53 atoms) — verify score match, measure timing
2. **1HPS** (20 torsions, ~93 atoms) — expected GPU speedup for score-dominant cases
3. **All 100 DT100 systems** — aggregate timing via `run_flex_batch.sh`
4. Compare GPU vs CPU scores — must be bit-identical
5. Verify against `run_flex_batch.sh` expected output

### After verification
- **Bobyqa minimizer**: Apply same dual-path `gpu_batch_eval_scores()` pattern
- **Conjugate-gradient minimizer**: Wire `dock_gpu_set_ligand_ie()` in `conf_gen_cg.cpp`
- **Multi-grid support** (ir_ensemble, fp_mol): Falls to CPU via returning false
- **Attractive IE term**: GPU kernel only implements repulsive `-1/r^6`
- **Non-standard IE exponents**: Falls to CPU
- **CPU coordinate → GPU upload could be replaced** with GPU-side torsion→coordinate
  computation (reduces per-batch upload size from ~25 MB to ~KB for torsion angles)

---

## Known Limitations

1. **Apple Silicon only**: Metal backend; no CUDA/ROCm support
2. **Single grid only**: Multi-grid scoring (ir_ensemble, fp_mol) not GPU-accelerated
3. **Repulsion-only IE**: Attractive IE term not in GPU kernel (uncommon)
4. **Hardcoded `1/r^12`**: Non-standard rep/exponents not supported on GPU
5. **Atom count limit**: `GPU_MAX_ATOMS = 512`
6. **Batch size limit**: `GPU_MAX_POSES = 4096`
7. **Per-atom data uploaded once**: Ligand must not grow during minimization
   (currently true for simplex — growth happens before minimization)
8. **No mixed-precision fallback**: GPU uses 32-bit float throughout
9. **Sentinel-based failure**: Outside-grid atom returns `-FLT_MAX`; caller
   aborts the entire simplex iteration.  CPU evaluates each pose independently
   and can handle partial failures.

---

## File Map

| File | Role |
|------|------|
| `score_dock_gpu.h` | Public API declarations |
| `score_dock_gpu_stub.c` | CPU stub (return 0) |
| `score_dock_gpu_metal.h` | `DockGridParams` struct |
| `score_dock_gpu_metal.mm` | Metal backend: init, buffer mgmt, 2 kernels, IE support |
| `minimizer.h` | `gpu_batch_eval_scores()` method declaration |
| `minimizer.cpp` | `gpu_batch_eval_scores()` implementation |
| `simplex.cpp` | Restructured `do_minimize()` with speculative batch + shrink batch |
| `conf_gen_ag.cpp` | Ligand parameter upload after IE init |
| `GPU_DOCK.md` | This file |

## Build matrix

| `GPU_BACKEND` | Platform | Binary type | Status |
|---------------|----------|-------------|--------|
| `cpu` | any | CPU-only stub | ✅ All tests pass |
| `metal` | macOS ARM | Metal GPU | 🟡 Flex docking hangs |
| `auto` | macOS ARM | Metal GPU | Same as `metal` |
| `auto` | Linux | CPU stub | Not tested |
