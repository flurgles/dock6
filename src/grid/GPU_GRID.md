# GPU-Accelerated Grid Generation — Architecture Guide

## Overview

The GPU grid generation system offloads the expensive energy grid computation
(attractive VDW, repulsive VDW, electrostatic) to the GPU while keeping the
CPU as an always-available fallback.

```
score_grid.c                     ← modified: added GPU dispatch
  → score_grid_gpu.h             ← NEW: abstraction API (forward decls only)
      ├── score_grid_gpu_stub.c  ← NEW: CPU fallback (no GPU → returns 0)
      ├── score_grid_gpu_metal.h ← NEW: Metal internal header (GridParams struct)
      ├── score_grid_gpu_metal.mm← NEW: Metal implementation (Obj-C++ + embedded shader)
      └── verify_gpu_grids.sh    ← NEW: batch validation against CPU grids
```

**Key design principle:** Any GPU backend (Metal, Vulkan, CUDA) implements
the same 6 functions declared in `score_grid_gpu.h`. The caller in
`score_grid.c` tries `gpu_grid_init()` — if it returns 1, the GPU does all
the work; if 0, the existing CPU `for` loops run. The CPU stub is always
available as a safety net.

---

## File-by-File Walkthrough

### 1. `score_grid_gpu.h` — The GPU Abstraction API

**Purpose:** Declares the 6-function lifecycle that every GPU backend must
implement. Only uses **forward declarations** (no full struct definitions).

```c
/* Forward declarations — the DOCK headers lack include guards so we
   cannot safely include them here.  We only pass pointers, so forward
   declarations suffice. */
typedef struct score_energy_struct   SCORE_ENERGY;
typedef struct score_grid_struct     SCORE_GRID;
typedef struct score_bump_struct     SCORE_BUMP;
typedef struct score_contact_struct  SCORE_CHEMICAL;
typedef struct score_chemical_struct SCORE_CHEMICAL;
typedef struct molecule_struct       MOLECULE;
typedef struct label_struct          LABEL;
```

**Why forward declarations?** The DOCK headers (`score.h`, `mol.h`, etc.)
lack `#ifndef` include guards. If we included them in this header, and
`score_grid.c` already included `score.h`, we'd get double-definition
errors. Since we only pass **pointers** through this API, the compiler
doesn't need the full struct layout — just knowing the type names exist
suffices.

**`extern "C"` linkage** — critical detail:

```c
#ifdef __cplusplus
extern "C" {
#endif

int  gpu_grid_init(SCORE_ENERGY *energy, ...);
void gpu_grid_upload(SCORE_ENERGY *energy, ...);
void gpu_grid_compute(float soft_delta);
void gpu_grid_download(SCORE_ENERGY *energy, ...);
void gpu_grid_cleanup(void);
int  gpu_grid_is_active(void);

#ifdef __cplusplus
}
#endif
```

`score_grid.c` is compiled as **C** (unmangled symbol `_gpu_grid_init`).
The Metal `.mm` file is compiled as **Objective‑C++** which normally
mangles symbols to `__Z13gpu_grid_init...`. `extern "C"` forces C linkage
so both sides agree on the symbol name. Without it, the linker cannot
resolve the call.

**The 6-function lifecycle:**
```
init → upload → compute → download → cleanup   (GPU path)
              ↓
init returns 0 → fall through to CPU loops      (fallback path)
```
If `init()` returns 0, the caller falls through to the CPU path
(no cleanup needed).

---

### 2. `score_grid_gpu_stub.c` — The Safety Net

**Purpose:** Stub implementation for systems without GPU support. Every
function is a no-op; `init()` returns 0.

```c
int gpu_grid_init(SCORE_ENERGY *energy, ...)
{
    (void)energy;  /* suppress unused-parameter warnings */
    return 0;      /* "GPU not available, use CPU" */
}
```

**Build-time linking:** This file is always on disk but only linked when
no GPU backend is active. The Makefile controls this:

```makefile
GPU_STUB = score_grid_gpu_stub.o     # default: link the stub
ifeq ($(GPU_BACKEND),metal)
  GPU_STUB =                          # Metal active → exclude stub
endif
```

**Why exclude the stub?** With `extern "C"`, both the stub and the Metal
backend define the same 6 symbols — which would cause duplicate symbol
errors at link time.

---

### 3. `score_grid_gpu_metal.h` — The Internal Metal Header

**Purpose:** Defines the `GridParams` struct shared between the C++ host
code (`.mm`) and the Metal shader. Not included by `score_grid.c`.

```c
typedef struct {
    float origin_x, origin_y, origin_z;  // 12 bytes (3 × 4)
    int    span_x, span_y, span_z;       // 12 bytes
    float spacing;
    float distance;        // interaction cutoff in Angstroms
    float dist_sq_min;
    float rep_exponent;
    float att_exponent;
    int   distance_dielectric;
    float dielectric_factor;
    float soft_delta;
    int   grid_size;       // span_x * span_y * span_z
                           // also reused to transport num_atoms to shader
} GridParams;
```

**⚠️ The `float3` alignment trap (very important):** Metal's `float3` is
16-byte aligned (like `float4`) — it has 4 bytes of padding after 3 floats.
But C packs `origin_x, origin_y, origin_z` consecutively (12 bytes, no
padding). If the Metal shader used `float3 origin`, every field after
`origin` would be shifted by 4 bytes relative to the C struct. The grid
kernel would read garbage bounds and produce all-zero output.

**Always use individual `float` members** when matching C structs, never
`float3` or `packed_float3`.

---

### 4. `score_grid_gpu_metal.mm` — The Metal Implementation

This is the core of the GPU backend. It's an Objective‑C++ file with an
embedded Metal compute shader, Metal device management, buffer allocation,
and the full init/upload/compute/download/cleanup lifecycle.

---

#### 4a. Kernel Strategy: Tiled Gather (NOT Scatter)

The kernel uses a **tiled gather** pattern, replacing the original
scatter+atomic design. Key differences:

| Aspect | Scatter (v1, discarded) | Tiled Gather (v2, current) |
|--------|------------------------|---------------------------|
| Thread mapping | 1 thread per atom | 1 thread per **grid point** |
| Thread count | 4K (atom count) | **1.35M** (grid points) |
| Output writes | atomic fetch_add | **plain stores** (no atomics) |
| Atom data access | each thread reads global mem | **shared mem batch** |
| Performance | ~58s | **~0.64s** (~90× faster) |
| Scaling with grid size | poor (atom-bound) | **excellent** |

**Why more threads is faster on GPU:** CPU threads are expensive (OS
scheduler, stack, context switch). GPU threads are **free** — the hardware
manages thousands of lightweight threads, switching between them on every
cycle to hide memory latency. Apple M1 supports ~8192 concurrent threads.
Going from 4K → 1.35M threads **increases occupancy** and lets the GPU
hide memory access latency better.

**Why no atomics:** In the scatter pattern, multiple atoms update the same
grid point → atomic operations serialize through the memory controller.
In the tiled gather pattern, **each grid point is owned by exactly one
thread**. The thread accumulates contributions from all atoms in registers,
then writes once to its unique output location. No coordination needed.

---

#### 4b. The Embedded Shader (`shader_src`)

```objc
static const char* shader_src = \
"#include <metal_stdlib>\n"
"using namespace metal;\n"
...
```

The Metal compute shader is stored as a C string literal inside the `.mm`
file. At runtime, `newLibraryWithSource:` compiles it into a Metal library.
This avoids needing a separate `.metal` file and a dedicated build step.

**The `GridParams` struct in the shader must mirror the C layout exactly:**

```metal
struct GridParams {
    float origin_x, origin_y, origin_z;
    int span_x, span_y, span_z;
    float spacing;
    float distance;
    float dist_sq_min;
    float rep_exponent;
    float att_exponent;
    int distance_dielectric;
    float dielectric_factor;
    float soft_delta;
    int grid_size;       /* reused as num_atoms during compute */
};
```

**Integer exponentiation (matching the CPU's POWER macro):**

```metal
static float int_pow(float base, int exp)
{
    float result = 1.0;
    float run = base;
    while (exp) {
        if (exp & 1) result *= run;
        run *= run;
        exp >>= 1;
    }
    return result;
}
```

This is exponentiation-by-squaring — cheap and exact for small integer
exponents (6 and 9 for our soft-core LJ). No `pow()` function call needed.

**The tiled gather kernel — one thread per grid point:**

```metal
kernel void grid_energy_kernel(
    device const float*    atom_pos     [[buffer(0)]],
    ...
    constant GridParams&   p            [[buffer(7)]],
    uint3                  gid          [[thread_position_in_grid]],
    uint3                  tpt          [[thread_position_in_threadgroup]])
{
    /* 1. Each thread computes its unique grid point position */
    float3 gpos;
    gpos.x = (float)gid.x * p.spacing + p.origin_x;
    gpos.y = (float)gid.y * p.spacing + p.origin_y;
    gpos.z = (float)gid.z * p.spacing + p.origin_z;

    /* 2. Per-thread accumulators (registers — no atomics) */
    float my_avdw = 0.0;
    float my_bvdw = 0.0;
    float my_es   = 0.0;

    /* 3. Cooperative loading: each threadgroup thread loads one atom
          into shared memory.  Batch loop processes all atoms. */
    threadgroup float3  sh_pos[TILE_VOL];
    threadgroup float   sh_vdwA[TILE_VOL];
    // ...

    for (batch_start = 0; batch_start < num_atoms; batch_start += TILE_VOL) {
        /* Each thread loads one atom into shared memory */
        int atom_id = batch_start + tid;
        sh_pos[tid] = atom_pos[atom_id];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* All 512 threads iterate over the shared batch */
        for (int a = 0; a < batch_end; a++) {
            float3 apos = sh_pos[a];
            float dx = gpos.x - apos.x;
            /* ... compute VDW and ES contributions ... */
            my_avdw += sh_vdwA[a] * rep_power;
            my_bvdw += sh_vdwB[a] * att_power;
            my_es   += sh_charge[a] * es_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* 4. Write once — no atomics needed, each thread owns a unique idx */
    avdw[idx] = my_avdw;
    bvdw[idx] = my_bvdw;
    es[idx]   = my_es;
}
```

**Why `threadgroup_barrier`?** Each thread in the threadgroup loads one
atom into shared memory. All threads must finish loading before any thread
starts reading. `threadgroup_barrier(mem_flags::mem_threadgroup)` ensures
all shared-memory writes are visible to all threads in the threadgroup.

**Shared memory sizing:** 512 threads × (12 bytes float3 + 12 bytes three floats)
= ~12 KB per batch. Apple Silicon has 32 KB shared memory per threadgroup —
well within budget. Larger proteins (100K atoms) still fit: they just run
more batches through the same shared memory.

**Soft-core LJ support:**

```metal
if (p.soft_delta > 0.0) {
    float sd = sqrt(dist_sq + p.soft_delta);
    rep_dist_inv = 1.0 / sd;
} else {
    rep_dist_inv = dist_inv;
}
```

---

#### 4c. Static State

```objc
static id<MTLDevice>               g_device    = nil;
static id<MTLCommandQueue>         g_cmdq      = nil;
static id<MTLComputePipelineState> g_pso       = nil;
static id<MTLBuffer> g_buf_atom_pos    = nil;
static id<MTLBuffer> g_buf_atom_vdwA   = nil;
static id<MTLBuffer> g_buf_atom_vdwB   = nil;
static id<MTLBuffer> g_buf_atom_charge = nil;
static id<MTLBuffer> g_buf_avdw        = nil;
static id<MTLBuffer> g_buf_bvdw        = nil;
static id<MTLBuffer> g_buf_es          = nil;
static GridParams g_params;
static int g_num_atoms   = 0;
static int g_grid_size   = 0;
static int g_initialized = 0;
static int g_uploaded    = 0;
```

All state is **file-scope static**. Acceptable because the `grid` binary
generates one grid per invocation (no reentrancy needed).

Two status flags track lifecycle state:
- `g_initialized` — device + pipeline state created
- `g_uploaded` — buffers allocated and populated

---

#### 4d. `gpu_grid_init()` — Device Setup

```objc
@autoreleasepool {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) { return 0; }

    g_cmdq = [g_device newCommandQueue];
    id<MTLLibrary> lib = [g_device newLibraryWithSource:
                           [NSString stringWithUTF8String:shader_src]
                                                 options:nil error:&err];
    if (!lib) { /* shader compile error */ return 0; }

    id<MTLFunction> func = [lib newFunctionWithName:@"grid_energy_kernel"];
    g_pso = [g_device newComputePipelineStateWithFunction:func error:&err];
    if (!g_pso) { /* pipeline error */ return 0; }

    /* Cache grid parameters from the DOCK data structures */
    g_params.origin_x          = grid->origin[0];
    g_params.span_x            = grid->span[0];
    g_params.spacing           = grid->spacing;
    g_params.distance          = grid->distance;
    g_params.rep_exponent      = (float)energy->repulsive_exponent;
    g_params.att_exponent      = (float)energy->attractive_exponent;
    g_params.distance_dielectric = energy->distance_dielectric;
    g_params.dielectric_factor = energy->dielectric_factor;
    g_params.grid_size         = grid->span[0] * grid->span[1] * grid->span[2];

    g_initialized = 1;
    return 1;
}
```

Steps:
1. `MTLCreateSystemDefaultDevice()` — get the GPU handle (Apple M1/M2/M3).
   Returns `nil` on systems without Metal support.
2. `newCommandQueue()` — creates a serial queue for dispatching work.
3. `newLibraryWithSource:` — **runtime compilation** of the embedded shader.
4. `newComputePipelineStateWithFunction:` — creates the PSO (optimized
   kernel ready to run). Validates the kernel against the device.
5. Cache all grid parameters into the `GridParams` struct.

Everything is wrapped in `@autoreleasepool { }` — Objective-C objects
created inside are released when the pool drains.

---

#### 4e. `gpu_grid_upload()` — Data to the GPU

```objc
g_buf_atom_pos = alloc_buffer(sizeof(float) * 3 * g_num_atoms, "atom_pos");
float* pos_ptr = (float*)[g_buf_atom_pos contents];
for (int i = 0; i < g_num_atoms; i++) {
    pos_ptr[i*3 + 0] = receptor->coord[i][0];
    pos_ptr[i*3 + 1] = receptor->coord[i][1];
    pos_ptr[i*3 + 2] = receptor->coord[i][2];
}
```

On Apple Silicon's **unified memory**, `[contents]` returns a CPU-accessible
pointer to the shared buffer. We write into it, and the GPU sees the same
data — **zero-copy, no PCIe transfer**.

**Key optimization:** VDW A/B values are pre-resolved per atom (removing
the type-lookup indirection that the CPU path does inside the inner loop):

```objc
for (int i = 0; i < g_num_atoms; i++) {
    int vid = receptor->atom[i].vdw_id;
    vdwA_ptr[i] = energy->vdwA[vid];  // resolved once, not per pair
    vdwB_ptr[i] = energy->vdwB[vid];
}
```

---

#### 4f. `gpu_grid_compute()` — Kernel Launch

```objc
g_params.soft_delta = soft_delta;
g_params.grid_size  = g_num_atoms;  // transport atom count to shader

id<MTLCommandBuffer>  cmdbuf  = [g_cmdq commandBuffer];
id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

/* Bind all 8 buffers (0-6: data, 7: params) */
[enc setBuffer:g_buf_atom_pos    offset:0 atIndex:0];
[enc setBuffer:g_buf_atom_vdwA   offset:0 atIndex:1];
// ... bind all 8 buffers ...

/* Write params into a temp buffer */
id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&g_params ...];
[enc setBuffer:params_buf offset:0 atIndex:7];

/* Dispatch one thread per grid point, organized in 8x8x8 tiles */
MTLSize threadsPerGrid  = MTLSizeMake(g_params.span_x,
                                      g_params.span_y,
                                      g_params.span_z);
MTLSize threadgroupSize = MTLSizeMake(TILE_W, TILE_H, TILE_D);
[enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadgroupSize];
```

The command buffer is the GPU work container:
1. **`dispatchThreads:`** — tells Metal to run `span_x × span_y × span_z`
   threads, organized into 8×8×8 threadgroups. Metal handles threadgroup
   scheduling across the GPU cores.
2. **`waitUntilCompleted`** — blocks the CPU until the GPU finishes.

**Tile dimensions (8×8×8 = 512 threads):** Powers of two for memory
coalescing. 512 threads per threadgroup is Apple Silicon's sweet spot —
occupancy on the M1 8-core GPU is nearly 100%.

---

#### 4g. `gpu_grid_download()` — Results Back to CPU

```objc
memcpy(energy->avdw, [g_buf_avdw contents], sizeof(float) * g_grid_size);
memcpy(energy->bvdw, [g_buf_bvdw contents], sizeof(float) * g_grid_size);
memcpy(energy->es,   [g_buf_es   contents], sizeof(float) * g_grid_size);
```

On unified memory, this is just a `memcpy` from one RAM address to another
— no real "download." The GPU already wrote directly into the shared buffer
via the tiled gather kernel's plain stores. We copy into the DOCK energy
struct so the rest of the grid-writing code (`write_grids()`) can access
the computed values.

---

#### 4h. `gpu_grid_cleanup()` — Free Resources

```objc
g_buf_atom_pos    = nil;  // ARC releases the MTLBuffer
g_pso             = nil;
g_cmdq            = nil;
g_device          = nil;
g_initialized     = 0;
g_uploaded        = 0;
```

Setting `id<>` pointers to `nil` triggers Automatic Reference Counting
(ARC) to release the underlying Metal objects.

---

### 5. The Dispatch Point in `score_grid.c`

```c
#ifdef USE_METAL
  if (gpu_grid_init(energy, receptor, grid, bump, contact, chemical, label))
  {
    gpu_grid_upload(energy, receptor, grid, bump, contact, chemical, label);
    gpu_grid_compute(soft_delta);
    gpu_grid_download(energy, grid, bump, contact, chemical);
    gpu_grid_cleanup();
    fprintf(global.outfile, "GPU grid computation finished");
    return;    // ← skip the CPU loop below
  }
#endif

  // CPU fallback: existing for-loop over all atoms
  for (atomi = 0; atomi < receptor->total.atoms; atomi++) { ... }
```

- `#ifdef USE_METAL` — the GPU code compiles to nothing on non-Metal builds.
- If `gpu_grid_init()` returns 1 → full GPU lifecycle → `return`.
- If `gpu_grid_init()` returns 0 → fall through to CPU loop as if GPU
  code doesn't exist.

---

### 6. The Makefile

**Platform detection:**

```makefile
UNAME_S := $(shell uname -s)
GPU_BACKEND ?= auto

ifeq ($(GPU_BACKEND),auto)
  ifeq ($(UNAME_S),Darwin)
    GPU_BACKEND = metal
  else
    GPU_BACKEND = cpu
  endif
endif
```

Auto-selects Metal on macOS, CPU everywhere else. Override with:
```bash
make GPU_BACKEND=cpu       # force CPU fallback
make GPU_BACKEND=metal     # force Metal (e.g., for testing on macOS)
```

**Backend objects:**

```makefile
GPU_STUB = score_grid_gpu_stub.o
ifeq ($(GPU_BACKEND),metal)
  GPU_OBJS += score_grid_gpu_metal.o
  GPU_LIBS += -framework Metal -framework Foundation -lc++
  DOCKBUILDFLAGS += -DUSE_METAL
  GPU_STUB =       # excluded to avoid duplicate symbols
endif
```

`-lc++` is needed because the `.mm` file (Objective‑C++) generates C++
exception handling code (`___gxx_personality_v0`), and the Fortran linker
(`$(FC)`) doesn't pull in `libc++` automatically.

**`.mm` compilation rule:**

```makefile
.SUFFIXES: .mm

.mm.o:
	$(CXX) -c $(CXXFLAGS) $(DOCKBUILDFLAGS) -fobjc-arc -o $@ $<
```

Without `.SUFFIXES: .mm`, GNU make wouldn't recognize `.mm` as a valid
suffix for the `.mm.o:` suffix rule. `-fobjc-arc` enables Automatic
Reference Counting.

---

## Bugs Encountered (And Why)

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| **Undefined `_gpu_grid_init`** | Link error | C++ name mangling in `.mm` produces `__Z13gpu_grid_init...`, C code expects `_gpu_grid_init` | `extern "C"` in the header |
| **`___gxx_personality_v0` missing** | Link error | Fortran linker doesn't link `libc++` | `-lc++` in `GPU_LIBS` |
| **GPU grid all zeros** | Score = 0.0 | `float3 origin` has 16-byte alignment (4 bytes padding); C struct has no padding → every field after `origin` shifted by 4 bytes | Individual `float` members instead of `float3` |
| **GPU VDW energy 25% of correct** | Wrong scores | `device const float3* atom_pos` strides by 16 bytes per element; C data is 12 bytes per atom (3 consecutive floats) | `device const float*` with manual `ai = 3 * atom_id` |
| **Shader compile error: TILE_VOL unknown** | Build fails | C macros not available in shader string | Define constants in-shader: `constant int TILE_W = 8;` |
| **grid binary exits with code 70** | No output grid files | Absolute `-i` path crashes DOCK 4.0.1 grid parser | Always use relative paths from the working directory |

---

## Performance Results (1A28 benchmark, 4138 atoms, 1.35M grid points)

| Path | Real time | CPU time | vs CPU |
|------|-----------|----------|--------|
| CPU (GCC, sequential) | ~66s | 65.8s | — |
| GPU Scatter+Atomic (v1) | ~58s | 0.02s | -12% |
| **GPU Tiled Gather (v2)** | **~0.71s** | **0.02s** | **~93× faster** |

The tiled gather kernel eliminates the atomic bottleneck and achieves
near-peak memory bandwidth on the M1 GPU. The speedup scales with grid
size — larger boxes with more grid points benefit even more.

### Validation (649 systems, SB2025 dataset)

Automated batch validation against existing CPU grids using
`verify_gpu_grids.sh`:

```
=== Results: 649 passed, 0 failed (out of 649) ===
```

Maximum score difference across all systems: 0.000031 kcal/mol
(all within floating-point accumulation tolerance).

---

## Adding a New GPU Backend (e.g., Vulkan or CUDA)

1. **Create** `score_grid_gpu_vulkan.c` implementing the 6 functions.
2. **Add Makefile branch:**
   ```makefile
   ifeq ($(GPU_BACKEND),vulkan)
     GPU_OBJS += score_grid_gpu_vulkan.o
     GPU_LIBS += -lvulkan
     DOCKBUILDFLAGS += -DUSE_VULKAN
     GPU_STUB =
   endif
   ```
3. **Update `score_grid.c` dispatch** with another `#ifdef`.
4. The `GridParams` struct and tiled gather kernel pattern remain the same —
   only the GPU API calls change.

The tiled gather kernel maps naturally to any GPU API:
- **CUDA:** `__shared__ float sh_vdwA[512];` + `__syncthreads()`
- **Vulkan:** `shared float sh_vdwA[512];` + `barrier()`
- **Metal:** `threadgroup float sh_vdwA[512];` + `threadgroup_barrier()`

The cooperative batch loading pattern is universal.

---

## Footguns for Future GPU Work

1. **`float3` alignment** — Always use individual `float` members in structs
   shared between C and the GPU. `float3` has 16-byte alignment in Metal,
   12-byte alignment in CUDA — inconsistent across platforms.

2. **Shader compilation errors** — The embedded shader string is compiled
   at runtime. Error messages print to stderr but don't halt the program
   (falls through to CPU). Always check stderr during development.

3. **Buffer sizing** — `gpu_grid_upload()` allocates buffers for the exact
   atom count. If atom count changes between upload calls (reentrancy),
   buffers would be the wrong size. Not an issue for the single-use `grid`
   binary but important if reused for scoring in a loop.

4. **`dispatchThreads:threadsPerThreadgroup:` vs
   `dispatchThreadgroups:`** — The former takes total thread count and
   threadgroup size; Metal divides internally to compute the number of
   threadgroups. The latter takes threadgroup count directly. Using the
   wrong one results in too many or too few threads.
