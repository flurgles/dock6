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
      └── score_grid_gpu_metal.mm← NEW: Metal implementation (Obj-C++ + embedded shader)
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
typedef struct score_contact_struct  SCORE_CONTACT;
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

int gpu_grid_init(SCORE_ENERGY *energy, ...);
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
init → upload → compute → download → cleanup
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

#### 4a. The Embedded Shader (`shader_src`)

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
    float origin_x, origin_y, origin_z;  // individual floats, NOT float3
    int span_x, span_y, span_z;
    float spacing;
    float distance;
    float dist_sq_min;
    float rep_exponent;
    float att_exponent;
    int distance_dielectric;
    float dielectric_factor;
    float soft_delta;
    int grid_size;
};
```

**The kernel function — `[[buffer(N)]]` attribute:**

```metal
kernel void grid_energy_kernel(
    device const float*  atom_pos    [[buffer(0)]],
    device const int*    vdw_id      [[buffer(1)]],
    device const float*  charge      [[buffer(2)]],
    device const float*  vdwA        [[buffer(3)]],
    device const float*  vdwB        [[buffer(4)]],
    device float*        avdw        [[buffer(5)]],
    device float*        bvdw        [[buffer(6)]],
    device float*        es          [[buffer(7)]],
    constant GridParams& p           [[buffer(8)]],
    uint atom_id [[thread_position_in_grid]])
```

- Each `[[buffer(N)]]` corresponds to the buffer index set by
  `[enc setBuffer:... atIndex:N]` in the host code.
- `[[thread_position_in_grid]]` gives each GPU thread its unique index.
  Here, one thread per receptor atom.
- `device const float* atom_pos` — we use `float*` (not `float3*`) to
  avoid Metal's 16-byte `float3` stride. Each atom's position is 3
  consecutive floats at `atom_pos[3*id + 0..2]`, exactly matching the
  C layout of `receptor->coord[i][0..2]`.

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

**The scatter pattern — one atom → many grid points:**

```metal
// 1. Atom's grid-coordinate
float3 ocrd = float3(pos.x - p.origin_x, pos.y - p.origin_y, pos.z - p.origin_z);
int3 g = int3(round(ocrd / p.spacing));

// 2. How many grid points in each direction within cutoff
int grid_cutoff = (int)(p.distance / p.spacing + 1.0f);

// 3. Quick reject if atom is far outside grid bounds
if (g.x < -grid_cutoff || g.x >= p.span_x + grid_cutoff) return;

// 4. Bounding box of grid points within cutoff of this atom
int i0 = max(0, g.x - grid_cutoff);
int i1 = min(p.span_x, g.x + grid_cutoff + 1);
// ... same for j, k

// 5. Triple-nested loop over bounding box
for (int i = i0; i < i1; i++)
  for (int j = j0; j < j1; j++)
    for (int k = k0; k < k1; k++) {
      float dist_sq = dxy2 + dz*dz;
      // soft-core effective distance
      float sd = sqrt(dist_sq + p.soft_delta);
      float rep_dist_inv = 1.0f / sd;
      float att_dist_inv = 1.0f / sqrt(dist_sq);

      float rep_power = int_pow(rep_dist_inv, rep_exp);
      float att_power = int_pow(dist_inv, att_exp);

      atomic_fetch_add_explicit(
          (device atomic_float*)&avdw[idx],
          aA * rep_power, memory_order_relaxed);

      atomic_fetch_add_explicit(
          (device atomic_float*)&bvdw[idx],
          aB * att_power, memory_order_relaxed);
      // ... electrostatic
    }
```

**Why `atomic_fetch_add_explicit`?** Multiple GPU threads (atoms) can
contribute to the same grid point simultaneously. Without atomics, a
thread's write could be overwritten by another thread before the addition
completes — a classic data race. `memory_order_relaxed` is safe here
because grid accumulation has no ordering constraints; we just need the
final arithmetic sum.

---

#### 4b. Static State

```objc
static id<MTLDevice>               g_device    = nil;
static id<MTLCommandQueue>         g_cmdq      = nil;
static id<MTLComputePipelineState> g_pso       = nil;
static id<MTLBuffer> g_buf_atom_pos = nil;
// ... 7 more MTLBuffer pointers
static GridParams g_params;
static int g_initialized = 0;
static int g_uploaded    = 0;
```

All state is **file-scope static**. Acceptable because the `grid` binary
generates one grid per invocation (no reentrancy needed).

Two status flags track lifecycle state:
- `g_initialized` — device + pipeline state created
- `g_uploaded` — buffers allocated and populated

---

#### 4c. `alloc_buffer()` — Helper

```objc
static id<MTLBuffer> alloc_buffer(NSUInteger size, const char* label)
{
    id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (buf)
        buf.label = [NSString stringWithUTF8String:label];
    return buf;
}
```

`MTLResourceStorageModeShared` is the key to **zero-copy** on Apple Silicon.
The CPU and GPU share the same physical memory — no PCIe transfer needed.
The `label` property helps with Metal debugging tools (Xcode GPU capture).

---

#### 4d. `gpu_grid_init()` — Device Setup

```objc
@autoreleasepool {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "GPU-METAL: Metal GPU not available — CPU fallback\n");
        return 0;
    }

    g_cmdq = [g_device newCommandQueue];
    // ...

    /* Compile embedded shader at runtime */
    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:
                           [NSString stringWithUTF8String:shader_src]
                                                 options:nil error:&err];
    if (!lib) { /* ... fallback ... */ return 0; }

    id<MTLFunction> func = [lib newFunctionWithName:@"grid_energy_kernel"];
    g_pso = [g_device newComputePipelineStateWithFunction:func error:&err];
    if (!g_pso) { /* ... fallback ... */ return 0; }

    /* Cache grid parameters */
    g_params.origin_x = grid->origin[0];
    g_params.rep_exponent = (float)energy->repulsive_exponent;
    // ... etc

    g_initialized = 1;
    return 1;
}
```

Steps:
1. `MTLCreateSystemDefaultDevice()` — get the GPU handle (Apple M1/M2/M3).
   Returns `nil` on systems without Metal support.
2. `newCommandQueue()` — creates a serial queue for dispatching work.
3. `newLibraryWithSource:` — **runtime compilation** of the embedded shader.
   The shader string is compiled into a Metal library at runtime. Slightly
   slower on first call (~0.5s) but avoids a separate `.metal` build step.
4. `newComputePipelineStateWithFunction:` — creates the PSO (optimized
   kernel ready to run). Validates the kernel against the device.
5. Cache all grid parameters (origin, spacing, exponents, etc.) into the
   `GridParams` struct that will be sent to the GPU.

Everything is wrapped in `@autoreleasepool { }` — Objective-C objects
created inside are released when the pool drains.

---

#### 4e. `gpu_grid_upload()` — Data to the GPU

```objc
g_buf_atom_pos = alloc_buffer(sizeof(float) * 3 * GPU_MAX_ATOMS, "atom_pos");
float* pos_ptr = (float*)[g_buf_atom_pos contents];
for (int i = 0; i < g_num_atoms; i++) {
    pos_ptr[i*3 + 0] = receptor->coord[i][0];
    pos_ptr[i*3 + 1] = receptor->coord[i][1];
    pos_ptr[i*3 + 2] = receptor->coord[i][2];
}
```

On Apple Silicon's **unified memory**, `[contents]` returns a CPU-accessible
pointer to the shared buffer. We write into it, and the GPU sees the same
data — **zero-copy, no PCIe transfer**. On Intel Macs (discrete GPU), this
would need a blit encoder to copy to private GPU memory, but the API is
the same; Metal handles the difference transparently.

All output grid buffers (`avdw`, `bvdw`, `es`) are zeroed with `memset`.
`GPU_MAX_ATOMS` is set to 65536 as a generous upper bound; actual atom
count is typically 4000–10000 for a receptor.

---

#### 4f. `gpu_grid_compute()` — Kernel Launch

```objc
g_params.soft_delta = soft_delta;

id<MTLCommandBuffer>  cmdbuf  = [g_cmdq commandBuffer];
cmdbuf.label = @"GridCompute";

id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
enc.label = @"GridEnergyKernel";

[enc setComputePipelineState:g_pso];
[enc setBuffer:g_buf_atom_pos offset:0 atIndex:0];  // → [[buffer(0)]]
[enc setBuffer:g_buf_vdw_id   offset:0 atIndex:1];  // → [[buffer(1)]]
// ... bind all 9 buffers

/* Write params into a temp buffer */
id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&g_params
                                                  length:sizeof(GridParams)
                                                 options:MTLResourceStorageModeShared];
[enc setBuffer:params_buf offset:0 atIndex:8];

/* Dispatch one thread per atom */
MTLSize threadsPerGrid  = MTLSizeMake(g_num_atoms, 1, 1);
MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
[enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadgroupSize];
[enc endEncoding];

[cmdbuf commit];
[cmdbuf waitUntilCompleted];
```

The command buffer is the GPU work container:
1. **Encoder records all commands** (set pipeline, bind buffers, dispatch).
2. **Buffer bindings** — each `[[buffer(N)]]` in the shader gets its data.
3. **`dispatchThreads:`** — tells Metal to run `g_num_atoms` threads (one
   per receptor atom). `threadsPerThreadgroup:256` is Apple Silicon's
   sweet-spot threadgroup size.
4. **`commit`** submits the work. **`waitUntilCompleted`** blocks the CPU
   until the GPU finishes — essential before reading results.

---

#### 4g. `gpu_grid_download()` — Results Back to CPU

```objc
memcpy(energy->avdw, [g_buf_avdw contents], sizeof(float) * g_grid_size);
memcpy(energy->bvdw, [g_buf_bvdw contents], sizeof(float) * g_grid_size);
memcpy(energy->es,   [g_buf_es   contents], sizeof(float) * g_grid_size);
```

On unified memory, this is just a `memcpy` from one RAM address to another
— no real "download." The GPU already wrote directly into the shared buffer
via atomic operations. We copy into the DOCK energy struct so the rest of
the grid-writing code (`write_grids()`) can access the computed values.

---

#### 4h. `gpu_grid_cleanup()` — Free Resources

```objc
g_buf_atom_pos = nil;  // ARC releases the MTLBuffer
g_pso = nil;
g_cmdq = nil;
g_device = nil;
g_initialized = 0;
g_uploaded = 0;
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
| **GPU grid wrong values** | VDW energy 25% of CPU | `device const float3* atom_pos` strides by 16 bytes per element; C data is 12 bytes per atom (3 consecutive floats) | `device const float*` with manual `ai = 3 * atom_id` |

Each is a **silent ABI mismatch** — no compilation warning, but garbage at
runtime. They only show up when comparing the final scores.

---

## Adding a New GPU Backend (e.g., Vulkan)

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
4. The `GridParams` struct and scatter kernel pattern remain the same —
   only the GPU API calls change (Vulkan: `vkCreateDevice`, `vkCmdDispatch`,
   etc.; CUDA: `cudaMalloc`, `cudaLaunchKernel`, etc.).

---

## Performance Results (1A28 benchmark, 4138 atoms, 1.35M grid points)

| Path | Real time | CPU time |
|---|---|---|
| GPU (Metal, Apple M1) | **~58s** | 0.02s |
| CPU (GCC, sequential) | ~66s | 65.8s |
| **Speedup** | **~12%** | — |

The 12% gain is modest because the scatter pattern (one thread per atom,
each atomically accumulating to shared grid arrays) creates contention
on the GPU's memory system. The real value of this infrastructure is as
a **scaffold toward GPU-accelerated docking**, where per-conformer
evaluation against pre-computed grids would benefit far more from GPU
parallelism.
