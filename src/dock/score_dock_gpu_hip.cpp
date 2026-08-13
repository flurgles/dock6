/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
  HIP GPU backend for dock6 scoring.

  Implements the dock_gpu_* C API (score_dock_gpu.h) in portable HIP
  so the same source compiles for AMD ROCm (via hipcc) and NVIDIA
  (HIP_PLATFORM=nvcc -> nvcc).  Backend selection mirrors the Vulkan
  and Metal backends; see src/dock/Makefile GPU_BACKEND=hip and
  install/gnu.hip template.

  Grid interpolation replicates Base_Grid::interpolate() exactly
  (base_grid.cpp:212) — manual trilinear over 8 neighbors with the
  out-of-bounds nearest-neighbor fallback — so HIP scores are
  bit-matchable against the CPU path for validation.
 */

#include "score_dock_gpu.h"
#include "score_dock_gpu_hip.h"

#include <hip/hip_runtime.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>

static void gpu_stat(const char *name, int poses, int atoms, long long start_ms)
{
    long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    fprintf(stderr, "[GPUSTAT] %s poses=%d atoms=%d t=%lldms\n",
            name, poses, atoms, now - start_ms);
}

#define GPU_MAX_POSES       4096
#define GPU_MAX_ATOMS       512
#define GPU_MAX_NB_PAIRS    32768
#define HIP_SCORE_THREADS   64

/* Multi-ligand virtual-screen dispatch: one shared grid, per-ligand
   parameter LUT slots.  A "slot" is one ligand's per-atom data; poses in
   a dispatch reference their ligand's slot via pose_lig[]. */
#define GPU_MAX_LUT_LIGANDS 128
#define GPU_LUT_MAX_PAIRS   16384

#define CHECK_HIP(expr) do {                                               \
    hipError_t ce = (expr);                                                \
    if (ce != hipSuccess) {                                                \
        fprintf(stderr, "HIP: %s failed: %s\n", #expr, hipGetErrorString(ce)); \
    }                                                                      \
} while (0)

/* Per-dispatch kernel parameters (passed by value to kernels). */
struct HipKernelParams {
    float  origin_x, origin_y, origin_z;
    int    span_x, span_y, span_z;
    float  spacing;
    int    num_atoms;
    float  ie_soft_delta;
    float  ie_cutoff_sq;
    int    num_nb_pairs;
    int    num_poses;
};

/* ================================================================== */
/*  Device kernels                                                     */
/* ================================================================== */

/* Trilinear interpolation via hardware 3D texture sampling, matching the
   Vulkan backend exactly (score_dock_gpu_vulkan.cpp:100-113).
   Coordinates are in grid units: rx=(x-origin)/spacing.  Normalized
   [0,1] texel coordinates add a half-texel offset so texel centers fall
   exactly on integer grid nodes, reproducing the CPU 8-neighbor trilinear
   weights.  Clamp-to-edge addressing gives the same out-of-bounds
   nearest-neighbor result as the CPU / former manual fallback. */
__device__ __forceinline__ float hip_sample_grid(hipTextureObject_t tex,
                                                 float x, float y, float z,
                                                 const HipKernelParams &p)
{
    const float rx = (x - p.origin_x) / p.spacing;
    const float ry = (y - p.origin_y) / p.spacing;
    const float rz = (z - p.origin_z) / p.spacing;

    const float u = (rx + 0.5f) / (float)p.span_x;
    const float v = (ry + 0.5f) / (float)p.span_y;
    const float w = (rz + 0.5f) / (float)p.span_z;

    return tex3D<float>(tex, u, v, w);
}

/* Grid-only batch kernel: one thread per pose. */
__global__ void hip_grid_kernel(const float *xyz,
                                const float *vdwA, const float *vdwB,
                                const float *charges,
                                hipTextureObject_t avdw, hipTextureObject_t bvdw,
                                hipTextureObject_t es,
                                const int *active_flags,
                                float *out_scores,
                                HipKernelParams p)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p.num_poses) return;

    const int stride = tid * p.num_atoms * 3;
    float score = 0.0f;
    for (int a = 0; a < p.num_atoms; a++) {
        if (active_flags[a] == 0) continue;
        const float x = xyz[stride + a*3];
        const float y = xyz[stride + a*3 + 1];
        const float z = xyz[stride + a*3 + 2];
        const float vdw  = hip_sample_grid(avdw, x, y, z, p);
        const float bwdv = hip_sample_grid(bvdw, x, y, z, p);
        const float esv  = hip_sample_grid(es,   x, y, z, p);
        score += vdwA[a]*vdw - vdwB[a]*bwdv + charges[a]*esv;
    }
    out_scores[tid] = score;
}

/* Persistent grid + internal-energy kernel.
   Each workgroup claims a pose via an atomic work-counter, splits the
   atom loop across threads, block-reduces, writes one score. */
__global__ void hip_ie_kernel(const float *xyz,
                              const float *vdwA, const float *vdwB,
                              const float *charges,
                              hipTextureObject_t avdw, hipTextureObject_t bvdw,
                              hipTextureObject_t es,
                              const int *active_flags,
                              const float *ie_vdwA,
                              const int *pair_starts, const int *pair_indices,
                              unsigned int *pose_counter,
                              float *out_scores,
                              HipKernelParams p)
{
    __shared__ float s_partial[HIP_SCORE_THREADS];
    __shared__ unsigned int s_candidate;
    const int tid = (int)threadIdx.x;
    const int tg_size = (int)blockDim.x;

    if (tid == 0) {
        s_candidate = atomicAdd(pose_counter, 1u);
    }
    __syncthreads();

    while ((int)s_candidate < p.num_poses) {
        const int pose = (int)s_candidate;
        const int stride = pose * p.num_atoms * 3;
        float total = 0.0f;

        for (int a = tid; a < p.num_atoms; a += tg_size) {
            if (active_flags[a] == 0) continue;
            const float x = xyz[stride + a*3];
            const float y = xyz[stride + a*3 + 1];
            const float z = xyz[stride + a*3 + 2];
            const float vdw  = hip_sample_grid(avdw, x, y, z, p);
            const float bwdv = hip_sample_grid(bvdw, x, y, z, p);
            const float esv  = hip_sample_grid(es,   x, y, z, p);
            total += vdwA[a]*vdw - vdwB[a]*bwdv + charges[a]*esv;

            const int start = pair_starts[a];
            const int end   = pair_starts[a + 1];
            for (int i = start; i < end; i++) {
                const int a2 = pair_indices[i];
                if (a2 < 0 || a2 >= p.num_atoms) continue;
                if (active_flags[a2] == 0) continue;
                const float dx = xyz[stride + a*3]     - xyz[stride + a2*3];
                const float dy = xyz[stride + a*3 + 1] - xyz[stride + a2*3 + 1];
                const float dz = xyz[stride + a*3 + 2] - xyz[stride + a2*3 + 2];
                const float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < p.ie_cutoff_sq) {
                    const float r2eff = r2 + p.ie_soft_delta;
                    const float denom = r2eff*r2eff*r2eff;
                    total += (ie_vdwA[a]*ie_vdwA[a2]) / (denom*denom);
                }
            }
        }

        s_partial[tid] = total;
        __syncthreads();

        if (tid == 0) {
            float final_score = 0.0f;
            for (int i = 0; i < tg_size; i++) final_score += s_partial[i];
            out_scores[pose] = final_score;
            s_candidate = atomicAdd(pose_counter, 1u);
        }
        __syncthreads();
    }
}

/* Persistent grid + internal-energy kernel with per-ligand parameter LUT
   (virtual-screen dispatch).  Identical to hip_ie_kernel except that
   atom parameters, IE pair tables, and active flags are indexed per pose
   by the pose's ligand slot: lig = pose_lig[pose]. */
__global__ void hip_ie_vs_kernel(const float *xyz,
                                 const float *lut_vdwA, const float *lut_vdwB,
                                 const float *lut_charges,
                                 hipTextureObject_t avdw, hipTextureObject_t bvdw,
                                 hipTextureObject_t es,
                                 const int *lut_active_flags,
                                 const float *lut_ie_vdwA,
                                 const int *lut_pair_starts,
                                 const int *lut_pair_indices,
                                 const int *pose_lig,
                                 unsigned int *pose_counter,
                                 float *out_scores,
                                 HipKernelParams p)
{
    __shared__ float s_partial[HIP_SCORE_THREADS];
    __shared__ unsigned int s_candidate;
    const int tid = (int)threadIdx.x;
    const int tg_size = (int)blockDim.x;

    if (tid == 0) {
        s_candidate = atomicAdd(pose_counter, 1u);
    }
    __syncthreads();

    while ((int)s_candidate < p.num_poses) {
        const int pose = (int)s_candidate;
        const int lig  = pose_lig[pose];
        const int stride = pose * p.num_atoms * 3;
        const int lstride = lig * GPU_MAX_ATOMS;
        float total = 0.0f;

        for (int a = tid; a < p.num_atoms; a += tg_size) {
            if (lut_active_flags[lstride + a] == 0) continue;
            const float x = xyz[stride + a*3];
            const float y = xyz[stride + a*3 + 1];
            const float z = xyz[stride + a*3 + 2];
            const float vdw  = hip_sample_grid(avdw, x, y, z, p);
            const float bwdv = hip_sample_grid(bvdw, x, y, z, p);
            const float esv  = hip_sample_grid(es,   x, y, z, p);
            total += lut_vdwA[lstride + a]*vdw
                   - lut_vdwB[lstride + a]*bwdv
                   + lut_charges[lstride + a]*esv;

            const int start = lut_pair_starts[lstride + a];
            const int end   = lut_pair_starts[lstride + a + 1];
            for (int i = start; i < end; i++) {
                const int a2 = lut_pair_indices[lig * GPU_LUT_MAX_PAIRS + i];
                if (a2 < 0 || a2 >= p.num_atoms) continue;
                if (lut_active_flags[lstride + a2] == 0) continue;
                const float dx = xyz[stride + a*3]     - xyz[stride + a2*3];
                const float dy = xyz[stride + a*3 + 1] - xyz[stride + a2*3 + 1];
                const float dz = xyz[stride + a*3 + 2] - xyz[stride + a2*3 + 2];
                const float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < p.ie_cutoff_sq) {
                    const float r2eff = r2 + p.ie_soft_delta;
                    const float denom = r2eff*r2eff*r2eff;
                    total += (lut_ie_vdwA[lstride + a]*lut_ie_vdwA[lstride + a2]) / (denom*denom);
                }
            }
        }

        s_partial[tid] = total;
        __syncthreads();

        if (tid == 0) {
            float final_score = 0.0f;
            for (int i = 0; i < tg_size; i++) final_score += s_partial[i];
            out_scores[pose] = final_score;
            s_candidate = atomicAdd(pose_counter, 1u);
        }
        __syncthreads();
    }
}

/* Grid-component-only virtual-screen kernel.  Identical to hip_ie_vs_kernel
   minus the internal-energy pair loop — the growth prune pass needs the
   grid score and internal energy as separate numbers (dock_gpu_batch_score_vs
   returns their sum).  Per-pose ligand rows come from the LUT via pose_lig. */
__global__ void hip_grid_vs_kernel(const float *xyz,
                                   const float *lut_vdwA, const float *lut_vdwB,
                                   const float *lut_charges,
                                   hipTextureObject_t avdw, hipTextureObject_t bvdw,
                                   hipTextureObject_t es,
                                   const int *lut_active_flags,
                                   const int *pose_lig,
                                   unsigned int *pose_counter,
                                   float *out_scores,
                                   HipKernelParams p)
{
    __shared__ float s_partial[HIP_SCORE_THREADS];
    __shared__ unsigned int s_candidate;
    const int tid = (int)threadIdx.x;
    const int tg_size = (int)blockDim.x;

    if (tid == 0) {
        s_candidate = atomicAdd(pose_counter, 1u);
    }
    __syncthreads();

    while ((int)s_candidate < p.num_poses) {
        const int pose = (int)s_candidate;
        const int lig  = pose_lig[pose];
        const int stride = pose * p.num_atoms * 3;
        const int lstride = lig * GPU_MAX_ATOMS;
        float total = 0.0f;

        for (int a = tid; a < p.num_atoms; a += tg_size) {
            if (lut_active_flags[lstride + a] == 0) continue;
            const float x = xyz[stride + a*3];
            const float y = xyz[stride + a*3 + 1];
            const float z = xyz[stride + a*3 + 2];
            const float vdw  = hip_sample_grid(avdw, x, y, z, p);
            const float bwdv = hip_sample_grid(bvdw, x, y, z, p);
            const float esv  = hip_sample_grid(es,   x, y, z, p);
            total += lut_vdwA[lstride + a]*vdw
                   - lut_vdwB[lstride + a]*bwdv
                   + lut_charges[lstride + a]*esv;
        }

        s_partial[tid] = total;
        __syncthreads();

        if (tid == 0) {
            float final_score = 0.0f;
            for (int i = 0; i < tg_size; i++) final_score += s_partial[i];
            out_scores[pose] = final_score;
            s_candidate = atomicAdd(pose_counter, 1u);
        }
        __syncthreads();
    }
}

/* ================================================================== */
/*  Static state                                                       */
/* ================================================================== */

static int   g_initialized = 0;
static int   g_num_atoms    = 0;
static int   g_num_nb_pairs = 0;
static int   g_num_lut_ligands = 0;
static float g_ie_soft_delta = 0.0f;
static float g_ie_cutoff_sq  = 1e10f;
static int   g_compute_units = 0;
static char  g_device_name[128];

static float *d_vdwA = NULL, *d_vdwB = NULL, *d_charges = NULL;
static float *d_ie_vdwA = NULL;
static int   *d_pair_starts = NULL, *d_pair_indices = NULL;
static int   *d_active_flags = NULL;
static float *d_xyz = NULL, *d_scores = NULL;
static unsigned int *d_pose_counter = NULL;

/* Multi-ligand VS LUT state. */
static float *d_lut_vdwA = NULL, *d_lut_vdwB = NULL, *d_lut_charges = NULL;
static float *d_lut_ie_vdwA = NULL;
static int   *d_lut_active_flags = NULL;
static int   *d_lut_pair_starts = NULL, *d_lut_pair_indices = NULL;
static int   *d_pose_lig = NULL;

/* Grid as hardware 3D textures (trilinear filtered), matching the Vulkan
   and Metal backends. */
static hipArray *d_avdw = NULL, *d_bvdw = NULL, *d_es = NULL;
static hipTextureObject_t h_avdw = 0, h_bvdw = 0, h_es = 0;

static HipKernelParams g_params;

/* ================================================================== */
/*  Async pipelining state (host/GPU load balancing)                   */
/* ================================================================== */

/* One background stream for all async batch scoring.  All enqueues are
   stream-ordered, so the shared device buffers (d_xyz, d_pose_lig,
   d_pose_counter, d_scores) are never touched by two kernels at once. */
#define LBAL_RING     4

static hipStream_t g_stream = 0;
static float *g_pin_xyz[LBAL_RING] = {0};
static int   *g_pin_lig[LBAL_RING] = {0};
static float *g_pin_score[LBAL_RING] = {0};
static int    g_pin_busy[LBAL_RING] = {0};

/* Pending async batches: results are copied into the caller's buffers at
   dock_gpu_batch_score_sync() time. */
#define LBAL_MAX_PENDING 64
struct PendingBatch {
    const float *xyz;
    const int   *pose_lig;
    float       *out;
    int         poses;
    int         atoms;
    int         ring;
    int         grid_only;
    long long   t0;
};
static PendingBatch g_pending[LBAL_MAX_PENDING];
static int g_npending = 0;

/* Governor EMAs (smoothed host-prep vs GPU-busy timings). */
static double g_host_ms_ema = 1.0;
static double g_gpu_ms_ema = 1.0;

static long long lbal_now_ms(void)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ================================================================== */
/*  HIP utility helpers                                                */
/* ================================================================== */

static int hip_init_device(void)
{
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev <= 0) {
        fprintf(stderr, "HIP: no HIP-capable device found — CPU fallback\n");
        return 0;
    }
    if (hipSetDevice(0) != hipSuccess) return 0;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        snprintf(g_device_name, sizeof(g_device_name), "%s", props.name);
        g_compute_units = props.multiProcessorCount;
    } else {
        snprintf(g_device_name, sizeof(g_device_name), "hip-device-0");
        g_compute_units = 8;
    }
    if (g_compute_units <= 0) g_compute_units = 8;
    fprintf(stderr, "GPU-HIP: device: %s\n", g_device_name);
    fflush(stderr);
    return 1;
}

/* Upload a grid into a 3D texture with hardware trilinear filtering and
   clamp-to-edge (matching Vulkan/Metal backend semantics). */
static int hip_upload_grid_texture(hipArray **arr_out, hipTextureObject_t *tex_out,
                                   const float *host, int sx, int sy, int sz)
{
    hipChannelFormatDesc cd = hipCreateChannelDesc<float>();
    hipArray *a = NULL;
    hipExtent ext = make_hipExtent((size_t)sx, (size_t)sy, (size_t)sz);
    if (hipMalloc3DArray(&a, &cd, ext, 0) != hipSuccess) {
        fprintf(stderr, "HIP: hipMalloc3DArray(%dx%dx%d) failed\n", sx, sy, sz);
        return 0;
    }
    hipMemcpy3DParms mp;
    memset(&mp, 0, sizeof(mp));
    mp.srcPtr = make_hipPitchedPtr((void*)host, (size_t)sx * sizeof(float), (size_t)sx, (size_t)sy);
    mp.dstArray = a;
    mp.extent = ext;
    mp.kind = hipMemcpyHostToDevice;
    if (hipMemcpy3D(&mp) != hipSuccess) {
        fprintf(stderr, "HIP: hipMemcpy3D failed\n");
        hipFree(a);
        return 0;
    }

    hipResourceDesc res;
    memset(&res, 0, sizeof(res));
    res.resType = hipResourceTypeArray;
    res.res.array.array = a;

    hipTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 1;
    td.filterMode = hipFilterModeLinear;
    td.addressMode[0] = hipAddressModeClamp;
    td.addressMode[1] = hipAddressModeClamp;
    td.addressMode[2] = hipAddressModeClamp;

    hipTextureObject_t to = 0;
    if (hipCreateTextureObject(&to, &res, &td, NULL) != hipSuccess) {
        fprintf(stderr, "HIP: hipCreateTextureObject failed\n");
        hipFree((hipArray*)a);
        return 0;
    }
    *arr_out = (hipArray*)a;
    *tex_out = to;
    return 1;
}

/* ================================================================== */
/*  GPU abstraction API                                                */
/* ================================================================== */

extern "C" {

int dock_gpu_init(const float *avdw, const float *bvdw, const float *es,
                  int span_x, int span_y, int span_z,
                  float origin_x, float origin_y, float origin_z,
                  float spacing)
{
    if (g_initialized) return 1;

    /* gfx1033 (Steam Deck / Van Gogh) intermittently deadlocks the SDMA
       engine on host->device copies. route copies through the compute engine
       unless the user explicitly opted into the SDMA-batch path. */
    if (!getenv("HSA_ENABLE_SDMA"))
        setenv("HSA_ENABLE_SDMA", "0", 1);

    if (!hip_init_device()) return 0;

    HipKernelParams kp;
    kp.origin_x = origin_x; kp.origin_y = origin_y; kp.origin_z = origin_z;
    kp.span_x = span_x; kp.span_y = span_y; kp.span_z = span_z;
    kp.spacing = spacing;
    kp.num_atoms = 0; kp.num_poses = 0;
    kp.ie_soft_delta = 0.0f; kp.ie_cutoff_sq = 1e10f;
    kp.num_nb_pairs = 0;
    g_params = kp;

    if (!hip_upload_grid_texture(&d_avdw, &h_avdw, avdw, span_x, span_y, span_z) ||
        !hip_upload_grid_texture(&d_bvdw, &h_bvdw, bvdw, span_x, span_y, span_z) ||
        !hip_upload_grid_texture(&d_es,   &h_es,   es,   span_x, span_y, span_z)) {
        fprintf(stderr, "HIP: grid texture upload failed — CPU fallback\n");
        return 0;
    }

    CHECK_HIP(hipMalloc(&d_vdwA,     sizeof(float) * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_vdwB,     sizeof(float) * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_charges,  sizeof(float) * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_ie_vdwA,  sizeof(float) * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_active_flags, sizeof(int) * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_pair_starts,  sizeof(int) * (GPU_MAX_ATOMS + 1)));
    CHECK_HIP(hipMalloc(&d_pair_indices, sizeof(int) * GPU_MAX_NB_PAIRS));
    CHECK_HIP(hipMalloc(&d_xyz,  sizeof(float) * (size_t)GPU_MAX_POSES * GPU_MAX_ATOMS * 3));
    CHECK_HIP(hipMalloc(&d_scores, sizeof(float) * GPU_MAX_POSES));
    CHECK_HIP(hipMalloc(&d_pose_counter, sizeof(unsigned int)));

    /* Multi-ligand VS LUT: per-ligand param tables + per-ligand pair tables */
    const size_t vs_lig_bytes = sizeof(float) * (size_t)GPU_MAX_LUT_LIGANDS * GPU_MAX_ATOMS;
    CHECK_HIP(hipMalloc(&d_lut_vdwA,        vs_lig_bytes));
    CHECK_HIP(hipMalloc(&d_lut_vdwB,        vs_lig_bytes));
    CHECK_HIP(hipMalloc(&d_lut_charges,     vs_lig_bytes));
    CHECK_HIP(hipMalloc(&d_lut_ie_vdwA,     vs_lig_bytes));
    CHECK_HIP(hipMalloc(&d_lut_active_flags, sizeof(int) * (size_t)GPU_MAX_LUT_LIGANDS * GPU_MAX_ATOMS));
    CHECK_HIP(hipMalloc(&d_lut_pair_starts,  sizeof(int) * (size_t)GPU_MAX_LUT_LIGANDS * (GPU_MAX_ATOMS + 1)));
    CHECK_HIP(hipMalloc(&d_lut_pair_indices, sizeof(int) * (size_t)GPU_MAX_LUT_LIGANDS * GPU_LUT_MAX_PAIRS));
    CHECK_HIP(hipMalloc(&d_pose_lig,        sizeof(int) * GPU_MAX_POSES));

    /* Async pipeline: one background stream + pinned staging ring. */
    CHECK_HIP(hipStreamCreate(&g_stream));
    g_npending = 0;
    for (int i = 0; i < LBAL_RING; i++) {
        CHECK_HIP(hipHostMalloc(&g_pin_xyz[i],
                                sizeof(float) * (size_t)GPU_MAX_POSES * GPU_MAX_ATOMS * 3,
                                hipHostMallocDefault));
        CHECK_HIP(hipHostMalloc(&g_pin_lig[i],
                                sizeof(int) * (size_t)GPU_MAX_POSES,
                                hipHostMallocDefault));
        CHECK_HIP(hipHostMalloc(&g_pin_score[i],
                                sizeof(float) * (size_t)GPU_MAX_POSES,
                                hipHostMallocDefault));
        g_pin_busy[i] = 0;
    }
    g_host_ms_ema = 1.0;
    g_gpu_ms_ema = 1.0;

    g_initialized = 1;
    g_num_atoms = 0;
    g_num_nb_pairs = 0;
    return 1;
}

int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                        const float *charges, int num_atoms)
{
    if (!g_initialized) return 0;
    if (num_atoms <= 0 || num_atoms > GPU_MAX_ATOMS) {
        fprintf(stderr, "HIP: invalid num_atoms %d (max %d)\n", num_atoms, GPU_MAX_ATOMS);
        return 0;
    }
    const size_t bytes = sizeof(float) * (size_t)num_atoms;
    CHECK_HIP(hipMemcpy(d_vdwA, vdwA, bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_vdwB, vdwB, bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_charges, charges, bytes, hipMemcpyHostToDevice));

    std::vector<int> af(num_atoms, 1);
    CHECK_HIP(hipMemcpy(d_active_flags, af.data(), sizeof(int) * (size_t)num_atoms, hipMemcpyHostToDevice));

    g_num_atoms = num_atoms;
    return 1;
}

int dock_gpu_set_ligand_ie(const float *ie_vdwA, const float *ie_vdwB,
                           const int *nb_int_pairs, int num_nb_pairs,
                           float ie_soft_delta, float ie_cutoff_sq)
{
    (void)ie_vdwB;
    if (!g_initialized) return 0;
    if (num_nb_pairs > GPU_MAX_NB_PAIRS) {
        fprintf(stderr, "HIP: num_nb_pairs %d exceeds max %d\n", num_nb_pairs, GPU_MAX_NB_PAIRS);
        return 0;
    }

    const size_t ie_bytes = sizeof(float) * (size_t)g_num_atoms;
    CHECK_HIP(hipMemcpy(d_ie_vdwA, ie_vdwA, ie_bytes, hipMemcpyHostToDevice));

    int na = g_num_atoms;
    if (na <= 0) return 0;

    std::vector<int> counts(na, 0);
    for (int p = 0; p < num_nb_pairs; p++) {
        int a1 = nb_int_pairs[p*2];
        if (a1 >= 0 && a1 < na) counts[a1]++;
    }
    std::vector<int> starts(na + 1, 0);
    std::vector<int> offsets(na, 0);
    int total = 0;
    for (int a = 0; a < na; a++) {
        starts[a] = total;
        offsets[a] = total;
        total += counts[a];
    }
    starts[na] = total;

    std::vector<int> pair_indices(total, -1);
    for (int p = 0; p < num_nb_pairs; p++) {
        int a1 = nb_int_pairs[p*2];
        int a2 = nb_int_pairs[p*2+1];
        if (a1 >= 0 && a1 < na) pair_indices[offsets[a1]++] = a2;
    }

    CHECK_HIP(hipMemcpy(d_pair_starts, starts.data(), sizeof(int) * (size_t)(na + 1), hipMemcpyHostToDevice));
    if (total > 0)
        CHECK_HIP(hipMemcpy(d_pair_indices, pair_indices.data(), sizeof(int) * (size_t)total, hipMemcpyHostToDevice));

    g_num_nb_pairs = num_nb_pairs;
    g_ie_soft_delta = ie_soft_delta;
    g_ie_cutoff_sq = ie_cutoff_sq;
    return 1;
}

int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         const int *active_flags, float *out_scores)
{
    if (!g_initialized) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    const size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    CHECK_HIP(hipMemcpy(d_xyz, xyz, xyz_bytes, hipMemcpyHostToDevice));

    if (active_flags) {
        CHECK_HIP(hipMemcpy(d_active_flags, active_flags,
                            sizeof(int) * (size_t)num_atoms,
                            hipMemcpyHostToDevice));
    } else {
        std::vector<int> af(num_atoms, 1);
        CHECK_HIP(hipMemcpy(d_active_flags, af.data(),
                            sizeof(int) * (size_t)num_atoms,
                            hipMemcpyHostToDevice));
    }

    HipKernelParams kp = g_params;
    kp.num_atoms = num_atoms;
    kp.num_poses = num_poses;
    kp.ie_soft_delta = 0.0f;
    kp.ie_cutoff_sq = 1e10f;
    kp.num_nb_pairs = 0;

    const int threads = HIP_SCORE_THREADS;
    const int blocks = (num_poses + threads - 1) / threads;
    const long long t0b = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    hipLaunchKernelGGL(hip_grid_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_xyz, d_vdwA, d_vdwB, d_charges,
                       h_avdw, h_bvdw, h_es, d_active_flags,
                       d_scores, kp);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(out_scores, d_scores, sizeof(float) * (size_t)num_poses, hipMemcpyDeviceToHost));
    gpu_stat("grid", num_poses, num_atoms, t0b);
    return 1;
}

int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                 const int *active_flags, float *out_scores)
{
    return dock_gpu_batch_score_with_ie_persistent(xyz, num_poses, num_atoms, active_flags, out_scores);
}

int dock_gpu_batch_score_with_ie_persistent(const float *xyz, int num_poses, int num_atoms,
                                            const int *active_flags, float *out_scores)
{
    if (!g_initialized || g_num_nb_pairs == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    const size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    CHECK_HIP(hipMemcpy(d_xyz, xyz, xyz_bytes, hipMemcpyHostToDevice));
    if (active_flags) {
        CHECK_HIP(hipMemcpy(d_active_flags, active_flags, sizeof(int) * (size_t)num_atoms, hipMemcpyHostToDevice));
    } else {
        std::vector<int> af(num_atoms, 1);
        CHECK_HIP(hipMemcpy(d_active_flags, af.data(), sizeof(int) * (size_t)num_atoms, hipMemcpyHostToDevice));
    }

    unsigned int zero = 0;
    CHECK_HIP(hipMemcpy(d_pose_counter, &zero, sizeof(zero), hipMemcpyHostToDevice));

    HipKernelParams kp = g_params;
    kp.num_atoms = num_atoms;
    kp.num_poses = num_poses;
    kp.ie_soft_delta = g_ie_soft_delta;
    kp.ie_cutoff_sq = g_ie_cutoff_sq;
    kp.num_nb_pairs = g_num_nb_pairs;

    const int threads = HIP_SCORE_THREADS;
    const long long t0b = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    /* Persistent kernel: each block is an independent workgroup that claims
       poses via an atomic work counter.  Launch as many workgroups as the
       recommended batch size (mirrors the Vulkan persistent dispatch). */
    int blocks = dock_gpu_recommended_batch_size();
    if (blocks < 1) blocks = 1;

    hipLaunchKernelGGL(hip_ie_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_xyz, d_vdwA, d_vdwB, d_charges,
                       h_avdw, h_bvdw, h_es,
                       d_active_flags, d_ie_vdwA,
                       d_pair_starts, d_pair_indices,
                       d_pose_counter, d_scores, kp);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(out_scores, d_scores, sizeof(float) * (size_t)num_poses, hipMemcpyDeviceToHost));
    gpu_stat("ie", num_poses, num_atoms, t0b);
    return 1;
}

int dock_gpu_vs_register_ligand(int lig_idx,
                                const float *vdwA, const float *vdwB,
                                const float *charges, const int *active_flags,
                                const float *ie_vdwA,
                                const int *nb_int_pairs, int num_nb_pairs,
                                int num_atoms,
                                float ie_soft_delta, float ie_cutoff_sq)
{
    if (!g_initialized) return 0;
    if (lig_idx < 0 || lig_idx >= GPU_MAX_LUT_LIGANDS) return 0;
    if (num_atoms <= 0 || num_atoms > GPU_MAX_ATOMS) return 0;
    if (num_nb_pairs > GPU_LUT_MAX_PAIRS) return 0;

    const size_t row_bytes = sizeof(float) * (size_t)num_atoms;
    const size_t row_off = (size_t)lig_idx * GPU_MAX_ATOMS;
    const int    lig_off = lig_idx * GPU_MAX_ATOMS;
    CHECK_HIP(hipMemcpy(d_lut_vdwA + row_off, vdwA, row_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_lut_vdwB + row_off, vdwB, row_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_lut_charges + row_off, charges, row_bytes, hipMemcpyHostToDevice));
    if (ie_vdwA) {
        CHECK_HIP(hipMemcpy(d_lut_ie_vdwA + row_off, ie_vdwA, row_bytes, hipMemcpyHostToDevice));
    }
    CHECK_HIP(hipMemcpy(d_lut_active_flags + lig_off, active_flags, sizeof(int) * (size_t)num_atoms, hipMemcpyHostToDevice));

    /* IE soft-core delta and cutoff are per-run constants shared by every
       ligand; store them so dock_gpu_batch_score_vs() applies the same
       parameters the sequential path passes through set_ligand_ie(). */
    g_ie_soft_delta = ie_soft_delta;
    g_ie_cutoff_sq  = ie_cutoff_sq;

    /* Zero the row tail beyond num_atoms.  The VS stride is the maximal
       atom count across the window, so shorter ligands would otherwise
       score stale (previous row owner / first-use garbage) flags and
       per-atom parameters in every pose's unused tail. */
    const size_t tail_cnt = (size_t)(GPU_MAX_ATOMS - num_atoms);
    if (tail_cnt > 0) {
        CHECK_HIP(hipMemset(d_lut_active_flags + lig_off + num_atoms, 0,
                            sizeof(int) * tail_cnt));
        const size_t tail_bytes = sizeof(float) * tail_cnt;
        CHECK_HIP(hipMemset(d_lut_vdwA + row_off + num_atoms, 0, tail_bytes));
        CHECK_HIP(hipMemset(d_lut_vdwB + row_off + num_atoms, 0, tail_bytes));
        CHECK_HIP(hipMemset(d_lut_charges + row_off + num_atoms, 0, tail_bytes));
        if (ie_vdwA) {
            CHECK_HIP(hipMemset(d_lut_ie_vdwA + row_off + num_atoms, 0, tail_bytes));
        }
        CHECK_HIP(hipMemset(d_lut_pair_starts + lig_off + num_atoms + 1, 0,
                            sizeof(int) * (size_t)(GPU_MAX_ATOMS + 1 - num_atoms)));
    }

    /* CS-per-atom pair index so the kernel can iterate pairs with
       pair_starts[lig][a].  Same layout as dock_gpu_set_ligand_ie(). */
    std::vector<int> counts(num_atoms, 0);
    for (int p = 0; p < num_nb_pairs; p++) {
        int a1 = nb_int_pairs[p*2];
        if (a1 >= 0 && a1 < num_atoms) counts[a1]++;
    }
    std::vector<int> starts(num_atoms + 1, 0);
    std::vector<int> offsets(num_atoms, 0);
    int total = 0;
    for (int a = 0; a < num_atoms; a++) {
        starts[a] = total;
        offsets[a] = total;
        total += counts[a];
    }
    starts[num_atoms] = total;

    std::vector<int> indices(GPU_LUT_MAX_PAIRS, -1);
    int cap = total < GPU_LUT_MAX_PAIRS ? total : GPU_LUT_MAX_PAIRS;
    for (int p = 0; p < num_nb_pairs && cap > 0; p++) {
        int a1 = nb_int_pairs[p*2];
        int a2 = nb_int_pairs[p*2+1];
        if (a1 >= 0 && a1 < num_atoms && offsets[a1] < cap) indices[offsets[a1]++] = a2;
    }

    CHECK_HIP(hipMemcpy(d_lut_pair_starts + lig_off, starts.data(),
                        sizeof(int) * (size_t)(num_atoms + 1), hipMemcpyHostToDevice));
    size_t pair_off = (size_t)lig_idx * GPU_LUT_MAX_PAIRS;
    CHECK_HIP(hipMemcpy(d_lut_pair_indices + pair_off, indices.data(),
                        sizeof(int) * (size_t)GPU_LUT_MAX_PAIRS, hipMemcpyHostToDevice));

    if (lig_idx + 1 > g_num_lut_ligands) g_num_lut_ligands = lig_idx + 1;
    return 1;
}

int dock_gpu_vs_max_ligands(void)
{
    return GPU_MAX_LUT_LIGANDS;
}

int dock_gpu_batch_score_vs(const float *xyz, int num_poses, int num_atoms,
                            const int *pose_lig, float *out_scores)
{
    if (!g_initialized || g_num_lut_ligands == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    long long t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    const size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    CHECK_HIP(hipMemcpy(d_xyz, xyz, xyz_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_pose_lig, pose_lig, sizeof(int) * (size_t)num_poses, hipMemcpyHostToDevice));

    unsigned int zero = 0;
    CHECK_HIP(hipMemcpy(d_pose_counter, &zero, sizeof(zero), hipMemcpyHostToDevice));

    HipKernelParams kp = g_params;
    kp.num_atoms = num_atoms;
    kp.num_poses = num_poses;
    kp.ie_soft_delta = g_ie_soft_delta;
    kp.ie_cutoff_sq = g_ie_cutoff_sq;
    kp.num_nb_pairs = g_num_nb_pairs;

    int blocks = dock_gpu_recommended_batch_size();
    if (blocks < 1) blocks = 1;

    hipLaunchKernelGGL(hip_ie_vs_kernel, dim3(blocks), dim3(HIP_SCORE_THREADS), 0, 0,
                       d_xyz, d_lut_vdwA, d_lut_vdwB, d_lut_charges,
                       h_avdw, h_bvdw, h_es,
                       d_lut_active_flags, d_lut_ie_vdwA,
                       d_lut_pair_starts, d_lut_pair_indices,
                       d_pose_lig, d_pose_counter, d_scores, kp);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(out_scores, d_scores, sizeof(float) * (size_t)num_poses, hipMemcpyDeviceToHost));
    gpu_stat("ie_vs", num_poses, num_atoms, t0);
    return 1;
}

int dock_gpu_batch_score_vs_grid(const float *xyz, int num_poses,
                                 int num_atoms, const int *pose_lig,
                                 float *out_scores)
{
    if (!g_initialized || g_num_lut_ligands == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    long long t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    const size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    CHECK_HIP(hipMemcpy(d_xyz, xyz, xyz_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_pose_lig, pose_lig, sizeof(int) * (size_t)num_poses, hipMemcpyHostToDevice));

    unsigned int zero = 0;
    CHECK_HIP(hipMemcpy(d_pose_counter, &zero, sizeof(zero), hipMemcpyHostToDevice));

    HipKernelParams kp = g_params;
    kp.num_atoms = num_atoms;
    kp.num_poses = num_poses;
    kp.ie_soft_delta = 0.0f;
    kp.ie_cutoff_sq = 1e10f;
    kp.num_nb_pairs = 0;

    int blocks = dock_gpu_recommended_batch_size();
    if (blocks < 1) blocks = 1;

    hipLaunchKernelGGL(hip_grid_vs_kernel, dim3(blocks), dim3(HIP_SCORE_THREADS), 0, 0,
                       d_xyz, d_lut_vdwA, d_lut_vdwB, d_lut_charges,
                       h_avdw, h_bvdw, h_es,
                       d_lut_active_flags,
                       d_pose_lig, d_pose_counter, d_scores, kp);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(out_scores, d_scores, sizeof(float) * (size_t)num_poses, hipMemcpyDeviceToHost));
    gpu_stat("grid_vs", num_poses, num_atoms, t0);
    return 1;
}

/* ================================================================== */
/*  Async VS batch scoring (host/GPU load balancing)                   */
/* ================================================================== */

int dock_gpu_batch_score_vs_enqueue(const float *xyz, int num_poses,
                                    int num_atoms, const int *pose_lig,
                                    float *out_scores, int grid_only)
{
    if (!g_initialized || g_num_lut_ligands == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;
    if (g_npending >= LBAL_MAX_PENDING) return 0;

    long long t0 = lbal_now_ms();

    /* Hand the batch to the GPU via a pinned staging slot so the async
       copies never touch pageable host memory (which would silently turn
       hipMemcpyAsync into a blocking copy). */
    int ring = -1;
    for (int i = 0; i < LBAL_RING; i++) {
        if (!g_pin_busy[i]) { ring = i; break; }
    }
    if (ring < 0) {
        /* Ring exhausted (rare: a step enqueued a huge pose volume).
           Drain the pipeline first — keeps ordering and correctness. */
        dock_gpu_batch_score_sync();
        ring = 0;
    }
    g_pin_busy[ring] = 1;

    const size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    memcpy(g_pin_xyz[ring], xyz, xyz_bytes);
    memcpy(g_pin_lig[ring], pose_lig, sizeof(int) * (size_t)num_poses);

    CHECK_HIP(hipMemcpyAsync(d_xyz, g_pin_xyz[ring], xyz_bytes,
                             hipMemcpyHostToDevice, g_stream));
    CHECK_HIP(hipMemcpyAsync(d_pose_lig, g_pin_lig[ring],
                             sizeof(int) * (size_t)num_poses,
                             hipMemcpyHostToDevice, g_stream));

    static const unsigned int zero_async = 0;
    CHECK_HIP(hipMemcpyAsync(d_pose_counter, &zero_async, sizeof(zero_async),
                             hipMemcpyHostToDevice, g_stream));

    HipKernelParams kp = g_params;
    kp.num_atoms = num_atoms;
    kp.num_poses = num_poses;
    kp.ie_soft_delta = grid_only ? 0.0f : g_ie_soft_delta;
    kp.ie_cutoff_sq = grid_only ? 1e10f : g_ie_cutoff_sq;
    kp.num_nb_pairs = grid_only ? 0 : g_num_nb_pairs;

    int blocks = dock_gpu_recommended_batch_size();
    if (blocks < 1) blocks = 1;

    if (grid_only) {
        hipLaunchKernelGGL(hip_grid_vs_kernel, dim3(blocks), dim3(HIP_SCORE_THREADS),
                           0, g_stream,
                           d_xyz, d_lut_vdwA, d_lut_vdwB, d_lut_charges,
                           h_avdw, h_bvdw, h_es,
                           d_lut_active_flags,
                           d_pose_lig, d_pose_counter, d_scores, kp);
    } else {
        hipLaunchKernelGGL(hip_ie_vs_kernel, dim3(blocks), dim3(HIP_SCORE_THREADS),
                           0, g_stream,
                           d_xyz, d_lut_vdwA, d_lut_vdwB, d_lut_charges,
                           h_avdw, h_bvdw, h_es,
                           d_lut_active_flags, d_lut_ie_vdwA,
                           d_lut_pair_starts, d_lut_pair_indices,
                           d_pose_lig, d_pose_counter, d_scores, kp);
    }
    CHECK_HIP(hipGetLastError());

    CHECK_HIP(hipMemcpyAsync(g_pin_score[ring], d_scores,
                             sizeof(float) * (size_t)num_poses,
                             hipMemcpyDeviceToHost, g_stream));

    /* Record the pending batch; its scores land in out_scores at sync. */
    PendingBatch &pb = g_pending[g_npending++];
    pb.xyz = xyz;
    pb.pose_lig = pose_lig;
    pb.out = out_scores;
    pb.poses = num_poses;
    pb.atoms = num_atoms;
    pb.ring = ring;
    pb.grid_only = grid_only;
    pb.t0 = t0;

    /* Governor input: host-side cost of preparing one batch. */
    double dur = (double)(lbal_now_ms() - t0);
    if (dur < 0.1) dur = 0.1;
    g_host_ms_ema = 0.85 * g_host_ms_ema + 0.15 * dur;
    return 1;
}

int dock_gpu_batch_score_sync(void)
{
    if (!g_initialized) return 0;
    if (g_npending == 0) return 0;

    long long t_sync = lbal_now_ms();
    CHECK_HIP(hipStreamSynchronize(g_stream));

    for (int i = 0; i < g_npending; i++) {
        PendingBatch &pb = g_pending[i];
        memcpy(pb.out, g_pin_score[pb.ring],
               sizeof(float) * (size_t)pb.poses);
        gpu_stat(pb.grid_only ? "grid_vs" : "ie_vs", pb.poses, pb.atoms, pb.t0);
        double dur = (double)(lbal_now_ms() - pb.t0);
        if (dur < 0.1) dur = 0.1;
        g_gpu_ms_ema = 0.85 * g_gpu_ms_ema + 0.15 * dur;
        g_pin_busy[pb.ring] = 0;
    }
    g_npending = 0;

    double dur = (double)(lbal_now_ms() - t_sync);
    if (dur < 0.1) dur = 0.1;
    g_gpu_ms_ema = 0.85 * g_gpu_ms_ema + 0.15 * dur;
    return 1;
}

void dock_gpu_lbal_stats(long long *host_ms, long long *gpu_ms)
{
    *host_ms = (long long)(g_host_ms_ema + 0.5);
    *gpu_ms = (long long)(g_gpu_ms_ema + 0.5);
}

int dock_gpu_grid_bounds(float *minx, float *miny, float *minz,
                         float *maxx, float *maxy, float *maxz)
{
    if (!g_initialized) return 0;
    /* Mirror Base_Grid::is_inside_grid_box(): valid points must be
       strictly between origin+spacing and origin+(span-2)*spacing so
       trilinear interpolation never touches the clamped texture edge. */
    *minx = g_params.origin_x + g_params.spacing;
    *miny = g_params.origin_y + g_params.spacing;
    *minz = g_params.origin_z + g_params.spacing;
    *maxx = g_params.origin_x + (float)(g_params.span_x - 2) * g_params.spacing;
    *maxy = g_params.origin_y + (float)(g_params.span_y - 2) * g_params.spacing;
    *maxz = g_params.origin_z + (float)(g_params.span_z - 2) * g_params.spacing;
    return 1;
}

void dock_gpu_cleanup(void)
{
    if (h_avdw)          { hipDestroyTextureObject(h_avdw); h_avdw = 0; }
    if (h_bvdw)          { hipDestroyTextureObject(h_bvdw); h_bvdw = 0; }
    if (h_es)            { hipDestroyTextureObject(h_es);   h_es = 0; }
    if (d_avdw)          { hipFree(d_avdw);         d_avdw = NULL; }
    if (d_bvdw)          { hipFree(d_bvdw);         d_bvdw = NULL; }
    if (d_es)            { hipFree(d_es);           d_es = NULL; }
    if (d_vdwA)         { hipFree(d_vdwA);         d_vdwA = NULL; }
    if (d_vdwB)         { hipFree(d_vdwB);         d_vdwB = NULL; }
    if (d_charges)      { hipFree(d_charges);      d_charges = NULL; }
    if (d_ie_vdwA)      { hipFree(d_ie_vdwA);      d_ie_vdwA = NULL; }
    if (d_pair_starts)  { hipFree(d_pair_starts);  d_pair_starts = NULL; }
    if (d_pair_indices) { hipFree(d_pair_indices); d_pair_indices = NULL; }
    if (d_active_flags) { hipFree(d_active_flags); d_active_flags = NULL; }
    if (d_xyz)          { hipFree(d_xyz);          d_xyz = NULL; }
    if (d_scores)       { hipFree(d_scores);       d_scores = NULL; }
    if (d_pose_counter) { hipFree(d_pose_counter); d_pose_counter = NULL; }
    if (d_lut_vdwA)        { hipFree(d_lut_vdwA);        d_lut_vdwA = NULL; }
    if (d_lut_vdwB)        { hipFree(d_lut_vdwB);        d_lut_vdwB = NULL; }
    if (d_lut_charges)     { hipFree(d_lut_charges);     d_lut_charges = NULL; }
    if (d_lut_ie_vdwA)     { hipFree(d_lut_ie_vdwA);     d_lut_ie_vdwA = NULL; }
    if (d_lut_active_flags){ hipFree(d_lut_active_flags);d_lut_active_flags = NULL; }
    if (d_lut_pair_starts) { hipFree(d_lut_pair_starts); d_lut_pair_starts = NULL; }
    if (d_lut_pair_indices){ hipFree(d_lut_pair_indices);d_lut_pair_indices = NULL; }
    if (d_pose_lig)        { hipFree(d_pose_lig);        d_pose_lig = NULL; }

    for (int i = 0; i < LBAL_RING; i++) {
        if (g_pin_xyz[i])   { hipHostFree(g_pin_xyz[i]);   g_pin_xyz[i] = NULL; }
        if (g_pin_lig[i])   { hipHostFree(g_pin_lig[i]);   g_pin_lig[i] = NULL; }
        if (g_pin_score[i]) { hipHostFree(g_pin_score[i]); g_pin_score[i] = NULL; }
        g_pin_busy[i] = 0;
    }
    if (g_stream) { hipStreamDestroy(g_stream); g_stream = 0; }
    g_npending = 0;

    g_initialized = 0;
    g_num_atoms = 0;
    g_num_nb_pairs = 0;
    g_num_lut_ligands = 0;
    g_ie_soft_delta = 0.0f;
    g_ie_cutoff_sq = 1e10f;
    memset(&g_params, 0, sizeof(g_params));
}

int dock_gpu_is_active(void)
{
    return g_initialized;
}

int dock_gpu_recommended_batch_size(void)
{
    if (!g_initialized) return 128;
    int size = g_compute_units * 16;
    int cap = GPU_MAX_POSES / 2;
    if (size > cap) size = cap;
    if (size < 32) size = 32;
    return size;
}

void dock_gpu_monitor(int layer, int segment, int total_segments)
{
    (void)layer; (void)segment; (void)total_segments;
    if (!g_initialized) return;
    fprintf(stderr, "GPU-HIP: backend=hip device_ready=1 device=%s "
            "compute_units=%d\n", g_device_name, g_compute_units);
}

} /* extern "C" */