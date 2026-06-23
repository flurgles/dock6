/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Metal GPU backend for dock6 batch scoring.

Implements the GPU abstraction API defined in score_dock_gpu.h using
Apple Metal on macOS (Apple Silicon).

Kernel strategy: 1 thread per pose.  Each thread loops over all atoms
in its pose, performing trilinear interpolation on the precomputed
VDW and ES grids, and accumulates per-atom contributions.  For the
combined kernel, each thread also accumulates internal energy over
the non-bonded pair list.  No atomics needed (each thread owns its
pose score).  Grid data is stored in GPU device memory and reused
across batches.

Two kernels:
  batch_score_kernel        — grid score only (legacy)
  batch_score_with_ie_kernel — grid score + internal energy
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "score_dock_gpu.h"
#include "score_dock_gpu_metal.h"

/* Max poses per batch */
#define GPU_MAX_POSES       4096
#define GPU_MAX_ATOMS       512
#define GPU_MAX_NB_PAIRS    32768  /* Max non-bonded pairs per ligand */


/* ================================================================== */
/*  Embedded Metal Shader Source — Both kernels in one library         */
/* ================================================================== */

static const char* shader_src = \
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct GridParams {\n"
"    float origin_x, origin_y, origin_z;\n"
"    int   span_x, span_y, span_z;\n"
"    float spacing;\n"
"    int   grid_size;\n"
"};\n"
"\n"
"struct IEParams {\n"
"    float soft_delta;\n"
"    float cutoff_sq;\n"
"    int   num_pairs;\n"
"    int   _pad;\n"
"};\n"
"\n"
"/* Forward declaration for trilinear (defined below) */\n"
"static float trilinear(device const float *grid,\n"
"                       constant GridParams &p,\n"
"                       float x, float y, float z);\n"
"\n"
"kernel void batch_score_kernel(\n"
"    device const float*    xyz          [[buffer(0)]],\n"
"    device const float*    grid_avdw    [[buffer(1)]],\n"
"    device const float*    grid_bvdw    [[buffer(2)]],\n"
"    device const float*    grid_es      [[buffer(3)]],\n"
"    device const float*    vdwA         [[buffer(4)]],\n"
"    device const float*    vdwB         [[buffer(5)]],\n"
"    device const float*    charges      [[buffer(6)]],\n"
"    constant GridParams&   gp           [[buffer(7)]],\n"
"    constant int&          num_atoms    [[buffer(8)]],\n"
"    device float*          out_scores   [[buffer(9)]],\n"
"    uint                   tid          [[thread_position_in_grid]])\n"
"{\n"
"    int atoms_per_pose = num_atoms;\n"
"    int stride = 3 * atoms_per_pose;\n"
"    int base = tid * stride;\n"
"\n"
"    float score = 0.0;\n"
"\n"
"    for (int a = 0; a < atoms_per_pose; a++) {\n"
"        int o3 = base + a * 3;\n"
"        float x = xyz[o3];\n"
"        float y = xyz[o3 + 1];\n"
"        float z = xyz[o3 + 2];\n"
"\n"
"        float vdw = trilinear(grid_avdw, gp, x, y, z);\n"
"        float bvdw = trilinear(grid_bvdw, gp, x, y, z);\n"
"        float es = trilinear(grid_es, gp, x, y, z);\n"
"\n"
"        score += vdwA[a] * vdw - vdwB[a] * bvdw + charges[a] * es;\n"
"    }\n"
"\n"
"    out_scores[tid] = score;\n"
"}\n"
"static bool inside_grid(constant GridParams &p, float x, float y, float z)\n"
"{\n"
"    float gx = (x - p.origin_x) / p.spacing;\n"
"    float gy = (y - p.origin_y) / p.spacing;\n"
"    float gz = (z - p.origin_z) / p.spacing;\n"
"    return (gx >= 1.0 && gx <= (float)(p.span_x - 2) &&\n"
"            gy >= 1.0 && gy <= (float)(p.span_y - 2) &&\n"
"            gz >= 1.0 && gz <= (float)(p.span_z - 2));\n"
"}\n"
"\n"
"/* Trilinear interpolation */\n"
"static float trilinear(device const float *grid,\n"
"                       constant GridParams &p,\n"
"                       float x, float y, float z)\n"
"{\n"
"    float gx = (x - p.origin_x) / p.spacing;\n"
"    float gy = (y - p.origin_y) / p.spacing;\n"
"    float gz = (z - p.origin_z) / p.spacing;\n"
"\n"
"    /* Bounds-check is done before calling trilinear in the kernel */\n"
"    /* Fallback: clamp and return 0 for safety */\n"
"    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||\n"
"        gx >= (float)(p.span_x - 1) ||\n"
"        gy >= (float)(p.span_y - 1) ||\n"
"        gz >= (float)(p.span_z - 1))\n"
"        return 0.0;\n"
"\n"
"    int ix = (int)gx;\n"
"    int iy = (int)gy;\n"
"    int iz = (int)gz;\n"
"    float fx = gx - (float)ix;\n"
"    float fy = gy - (float)iy;\n"
"    float fz = gz - (float)iz;\n"
"    float fx1 = 1.0 - fx;\n"
"    float fy1 = 1.0 - fy;\n"
"    float fz1 = 1.0 - fz;\n"
"\n"
"    int sx = p.span_x;\n"
"    int sy = p.span_y;\n"
"\n"
"    int i000 = iz * sx * sy + iy * sx + ix;\n"
"    int i001 = i000 + sx * sy;\n"
"    int i010 = i000 + sx;\n"
"    int i011 = i010 + sx * sy;\n"
"    int i100 = i000 + 1;\n"
"    int i101 = i100 + sx * sy;\n"
"    int i110 = i100 + sx;\n"
"    int i111 = i110 + sx * sy;\n"
"\n"
"    float c000 = grid[i000], c001 = grid[i001];\n"
"    float c010 = grid[i010], c011 = grid[i011];\n"
"    float c100 = grid[i100], c101 = grid[i101];\n"
"    float c110 = grid[i110], c111 = grid[i111];\n"
"\n"
"    float c00 = c000 * fx1 + c100 * fx;\n"
"    float c01 = c001 * fx1 + c101 * fx;\n"
"    float c10 = c010 * fx1 + c110 * fx;\n"
"    float c11 = c011 * fx1 + c111 * fx;\n"
"\n"
"    float c0 = c00 * fy1 + c10 * fy;\n"
"    float c1 = c01 * fy1 + c11 * fy;\n"
"\n"
"    return c0 * fz1 + c1 * fz;\n"
"}\n"
"\n"
"kernel void batch_score_with_ie_kernel(\n"
"    device const float*    xyz            [[buffer(0)]],\n"
"    device const float*    grid_avdw      [[buffer(1)]],\n"
"    device const float*    grid_bvdw      [[buffer(2)]],\n"
"    device const float*    grid_es        [[buffer(3)]],\n"
"    device const float*    vdwA           [[buffer(4)]],\n"
"    device const float*    vdwB           [[buffer(5)]],\n"
"    device const float*    charges        [[buffer(6)]],\n"
"    constant GridParams&   gp             [[buffer(7)]],\n"
"    constant int&          num_atoms      [[buffer(8)]],\n"
"    device float*          out_scores     [[buffer(9)]],\n"
"    device const float*    ie_vdwA        [[buffer(10)]],\n"
"    device const int*      nb_int         [[buffer(11)]],\n"
"    constant IEParams&     iep            [[buffer(12)]],\n"
"    constant int&          num_nb_pairs   [[buffer(13)]],\n"
"    uint                   tid            [[thread_position_in_grid]])\n"
"{\n"
"    int atoms_per_pose = num_atoms;\n"
"    int stride = 3 * atoms_per_pose;\n"
"    int base = tid * stride;\n"
"\n"
"    float grid_score = 0.0;\n"
"    bool has_outside_atom = false;\n"
"\n"
"    /* ---- Grid score ---- */\n"
"    for (int a = 0; a < atoms_per_pose; a++) {\n"
"        int o3 = base + a * 3;\n"
"        float x = xyz[o3];\n"
"        float y = xyz[o3 + 1];\n"
"        float z = xyz[o3 + 2];\n"
"\n"
"        if (!inside_grid(gp, x, y, z)) {\n"
"            has_outside_atom = true;\n"
"        }\n"
"\n"
"        float vdw = trilinear(grid_avdw, gp, x, y, z);\n"
"        float bvdw = trilinear(grid_bvdw, gp, x, y, z);\n"
"        float es = trilinear(grid_es, gp, x, y, z);\n"
"\n"
"        grid_score += vdwA[a] * vdw - vdwB[a] * bvdw + charges[a] * es;\n"
"    }\n"
"\n"
"    /* ---- Internal energy ---- */\n"
"    float ie_score = 0.0;\n"
"    if (num_nb_pairs > 0) {\n"
"        for (int p = 0; p < num_nb_pairs; p++) {\n"
"            int a1 = nb_int[p * 2];\n"
"            int a2 = nb_int[p * 2 + 1];\n"
"\n"
"            int o1 = base + a1 * 3;\n"
"            int o2 = base + a2 * 3;\n"
"            float dx = xyz[o1] - xyz[o2];\n"
"            float dy = xyz[o1 + 1] - xyz[o2 + 1];\n"
"            float dz = xyz[o1 + 2] - xyz[o2 + 2];\n"
"            float r2 = dx*dx + dy*dy + dz*dz;\n"
"\n"
"            if (r2 < iep.cutoff_sq) {\n"
"                float r2eff = r2 + iep.soft_delta;\n"
"                float denom = r2eff * r2eff * r2eff;  /* r^3 */\n"
"                ie_score += (ie_vdwA[a1] * ie_vdwA[a2]) / (denom * denom);  /* r^6 */\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    /* If any atom is outside the grid, return sentinel (matches CPU -MIN_FLOAT) */\n"
"    if (has_outside_atom) {\n"
"        out_scores[tid] = -3.40282347e+38;  /* -FLT_MAX */\n"
"    } else {\n"
"        out_scores[tid] = grid_score + ie_score;\n"
"    }\n"
"}\n";


/* ================================================================== */
/*  Static state                                                       */
/* ================================================================== */

static id<MTLDevice>               g_device     = nil;
static id<MTLCommandQueue>         g_cmdq       = nil;
static id<MTLComputePipelineState> g_pso        = nil;  /* grid-only kernel */
static id<MTLComputePipelineState> g_pso_ie    = nil;  /* grid+IE kernel */

/* GPU buffers (shared memory) */
static id<MTLBuffer> g_buf_grid_avdw  = nil;
static id<MTLBuffer> g_buf_grid_bvdw  = nil;
static id<MTLBuffer> g_buf_grid_es    = nil;
static id<MTLBuffer> g_buf_vdwA       = nil;
static id<MTLBuffer> g_buf_vdwB       = nil;
static id<MTLBuffer> g_buf_charges    = nil;
static id<MTLBuffer> g_buf_ie_vdwA    = nil;
static id<MTLBuffer> g_buf_nb_int     = nil;
static id<MTLBuffer> g_buf_xyz        = nil;
static id<MTLBuffer> g_buf_scores     = nil;
/* Persistent constant buffers (allocated once, reused per dispatch) */
static id<MTLBuffer> g_buf_params     = nil;  /* DockGridParams */
static id<MTLBuffer> g_buf_natoms     = nil;  /* int num_atoms */
static id<MTLBuffer> g_buf_iep        = nil;  /* IEParams */
static id<MTLBuffer> g_buf_nnp        = nil;  /* int num_nb_pairs */

/* Cached grid params */
static DockGridParams g_params;
static int  g_initialized = 0;   /* device + pipeline created */
static int  g_active      = 0;   /* buffers allocated + ready */
static int  g_num_atoms   = 0;   /* current ligand atom count */
static int  g_num_nb_pairs = 0;  /* current IE pair count */
static float g_ie_soft_delta = 0.0;  /* internal energy soft-core delta */
static float g_ie_cutoff_sq  = 1e10; /* internal energy distance cutoff^2 */


/* ================================================================== */
/*  Helper                                                              */
/* ================================================================== */

/* Helpers: write small constants to the persistent buffers */
static void write_natoms(int n) {
    memcpy([g_buf_natoms contents], &n, sizeof(int));
}
static void write_iep(float sd, float csq, int np) {
    /* Packed struct matches Metal IEParams */
    struct __attribute__((packed)) {
        float sd, csq;
        int np, pad;
    } iep = {sd, csq, np, 0};
    memcpy([g_buf_iep contents], &iep, sizeof(iep));
}
static void write_nnp(int n) {
    memcpy([g_buf_nnp contents], &n, sizeof(int));
}

static id<MTLBuffer> alloc_buffer(NSUInteger size, const char* label)
{
    id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (buf)
        buf.label = [NSString stringWithUTF8String:label];
    else
        NSLog(@"GPU-DOCK: failed to allocate buffer '%s' (%lu bytes)",
              label, (unsigned long)size);
    return buf;
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
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "GPU-DOCK: Metal GPU not available — CPU fallback\n");
            return 0;
        }
        fprintf(stderr, "GPU-DOCK: Metal device: %s\n",
                [g_device.name UTF8String]);

        g_cmdq = [g_device newCommandQueue];
        if (!g_cmdq) {
            fprintf(stderr, "GPU-DOCK: failed to create command queue — CPU fallback\n");
            dock_gpu_cleanup();
            return 0;
        }

        /* Compile grid-only shader */
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:
                               [NSString stringWithUTF8String:shader_src]
                                                     options:nil
                                                       error:&err];
        if (!lib) {
            fprintf(stderr, "GPU-DOCK: shader compilation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            dock_gpu_cleanup();
            return 0;
        }

        id<MTLFunction> func = [lib newFunctionWithName:@"batch_score_kernel"];
        if (!func) {
            fprintf(stderr, "GPU-DOCK: kernel 'batch_score_kernel' not found\n");
            dock_gpu_cleanup();
            return 0;
        }

        g_pso = [g_device newComputePipelineStateWithFunction:func error:&err];
        if (!g_pso) {
            fprintf(stderr, "GPU-DOCK: pipeline state creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            dock_gpu_cleanup();
            return 0;
        }

        id<MTLFunction> func_ie = [lib newFunctionWithName:@"batch_score_with_ie_kernel"];
        if (!func_ie) {
            fprintf(stderr, "GPU-DOCK: kernel 'batch_score_with_ie_kernel' not found\n");
            dock_gpu_cleanup();
            return 0;
        }

        g_pso_ie = [g_device newComputePipelineStateWithFunction:func_ie error:&err];
        if (!g_pso_ie) {
            fprintf(stderr, "GPU-DOCK: IE pipeline state creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            dock_gpu_cleanup();
            return 0;
        }

        /* Cache grid parameters */
        g_params.origin_x = origin_x;
        g_params.origin_y = origin_y;
        g_params.origin_z = origin_z;
        g_params.span_x   = span_x;
        g_params.span_y   = span_y;
        g_params.span_z   = span_z;
        g_params.spacing  = spacing;
        g_params.grid_size = span_x * span_y * span_z;

        /* Upload grid data to GPU */
        size_t grid_bytes = sizeof(float) * (size_t)g_params.grid_size;
        g_buf_grid_avdw = alloc_buffer(grid_bytes, "grid_avdw");
        g_buf_grid_bvdw = alloc_buffer(grid_bytes, "grid_bvdw");
        g_buf_grid_es   = alloc_buffer(grid_bytes, "grid_es");
        if (!g_buf_grid_avdw || !g_buf_grid_bvdw || !g_buf_grid_es) {
            fprintf(stderr, "GPU-DOCK: grid buffer allocation failed\n");
            dock_gpu_cleanup();
            return 0;
        }
        memcpy([g_buf_grid_avdw contents], avdw, grid_bytes);
        memcpy([g_buf_grid_bvdw contents], bvdw, grid_bytes);
        memcpy([g_buf_grid_es   contents], es,   grid_bytes);

        /* Allocate ligand parameter buffers (populated by dock_gpu_set_ligand later) */
        g_buf_vdwA    = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "vdwA");
        g_buf_vdwB    = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "vdwB");
        g_buf_charges = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "charges");
        g_buf_ie_vdwA = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "ie_vdwA");
        g_buf_nb_int  = alloc_buffer(sizeof(int) * GPU_MAX_NB_PAIRS * 2, "nb_int");

        /* Persistent constants buffers */
        g_buf_params = alloc_buffer(sizeof(DockGridParams), "params");
        g_buf_natoms = alloc_buffer(sizeof(int), "natoms");
        g_buf_iep    = alloc_buffer(32, "iep"); /* holds IEParams struct */
        g_buf_nnp    = alloc_buffer(sizeof(int), "nnp");

        /* Allocate per-batch buffers */
        g_buf_xyz    = alloc_buffer(sizeof(float) * 3 * GPU_MAX_ATOMS * GPU_MAX_POSES, "xyz");
        g_buf_scores = alloc_buffer(sizeof(float) * GPU_MAX_POSES, "scores");

        if (!g_buf_vdwA || !g_buf_vdwB || !g_buf_charges ||
            !g_buf_ie_vdwA || !g_buf_nb_int ||
            !g_buf_xyz || !g_buf_scores ||
            !g_buf_params || !g_buf_natoms || !g_buf_iep || !g_buf_nnp) {
            fprintf(stderr, "GPU-DOCK: per-batch buffer allocation failed\n");
            dock_gpu_cleanup();
            return 0;
        }

        /* Initialize persistent constants */
        memcpy([g_buf_params contents], &g_params, sizeof(DockGridParams));

        g_initialized = 1;
        g_active      = 0;
        g_num_nb_pairs = 0;

        NSLog(@"GPU-DOCK: Metal GPU ready. Grid: %dx%dx%d = %d pts",
              span_x, span_y, span_z, g_params.grid_size);
        return 1;
    }
}


int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                        const float *charges, int num_atoms)
{
    if (!g_initialized) return 0;

    if (num_atoms <= 0 || num_atoms > GPU_MAX_ATOMS) {
        fprintf(stderr, "GPU-DOCK: invalid num_atoms %d (max %d)\n",
                num_atoms, GPU_MAX_ATOMS);
        return 0;
    }

    @autoreleasepool {
        size_t bytes = sizeof(float) * (size_t)num_atoms;
        memcpy([g_buf_vdwA contents], vdwA, bytes);
        memcpy([g_buf_vdwB contents], vdwB, bytes);
        memcpy([g_buf_charges contents], charges, bytes);
        g_num_atoms = num_atoms;
        g_active = 1;
        return 1;
    }
}


int dock_gpu_set_ligand_ie(const float *ie_vdwA, const float *ie_vdwB,
                            const int *nb_int_pairs, int num_nb_pairs,
                            float ie_soft_delta, float ie_cutoff_sq)
{
    (void)ie_vdwB;  /* IE uses only vdwA (repulsion-only) for now */

    if (!g_initialized) return 0;

    if (num_nb_pairs > GPU_MAX_NB_PAIRS) {
        fprintf(stderr, "GPU-DOCK: num_nb_pairs %d exceeds max %d\n",
                num_nb_pairs, GPU_MAX_NB_PAIRS);
        return 0;
    }

    @autoreleasepool {
        size_t ie_bytes = sizeof(float) * (size_t)g_num_atoms;
        memcpy([g_buf_ie_vdwA contents], ie_vdwA, ie_bytes);

        size_t nb_bytes = sizeof(int) * (size_t)num_nb_pairs * 2;
        memcpy([g_buf_nb_int contents], nb_int_pairs, nb_bytes);

        g_num_nb_pairs   = num_nb_pairs;
        g_ie_soft_delta  = ie_soft_delta;
        g_ie_cutoff_sq   = ie_cutoff_sq;

        NSLog(@"GPU-DOCK: IE data set: %d pairs, soft_delta=%.2f, cutoff_sq=%.2f",
              num_nb_pairs, ie_soft_delta, ie_cutoff_sq);
        return 1;
    }
}


int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         float *out_scores)
{
    if (!g_initialized || !g_device) return 0;

    if (num_poses > GPU_MAX_POSES) {
        fprintf(stderr, "GPU-DOCK: batch size %d exceeds max %d\n",
                num_poses, GPU_MAX_POSES);
        return 0;
    }
    if (num_atoms > GPU_MAX_ATOMS) {
        fprintf(stderr, "GPU-DOCK: atoms/pose %d exceeds max %d\n",
                num_atoms, GPU_MAX_ATOMS);
        return 0;
    }

    @autoreleasepool {
        size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
        memcpy([g_buf_xyz contents], xyz, xyz_bytes);

        id<MTLCommandBuffer>  cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"DockBatchScore";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"BatchScoreKernel";

        [enc setComputePipelineState:g_pso];

        [enc setBuffer:g_buf_xyz        offset:0 atIndex:0];
        [enc setBuffer:g_buf_grid_avdw  offset:0 atIndex:1];
        [enc setBuffer:g_buf_grid_bvdw  offset:0 atIndex:2];
        [enc setBuffer:g_buf_grid_es    offset:0 atIndex:3];
        [enc setBuffer:g_buf_vdwA       offset:0 atIndex:4];
        [enc setBuffer:g_buf_vdwB       offset:0 atIndex:5];
        [enc setBuffer:g_buf_charges    offset:0 atIndex:6];

        [enc setBuffer:g_buf_params offset:0 atIndex:7];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms offset:0 atIndex:8];

        [enc setBuffer:g_buf_scores offset:0 atIndex:9];

        MTLSize threadsPerGrid  = MTLSizeMake(num_poses, 1, 1);
        NSUInteger tg = MIN(g_pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);

        [enc dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.error) {
            NSLog(@"GPU-DOCK: command buffer error: %@", cmdbuf.error);
            return 0;
        }

        memcpy(out_scores, [g_buf_scores contents], sizeof(float) * (size_t)num_poses);

        return 1;
    }
}


int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                  float *out_scores)
{
    if (!g_initialized || !g_device) { return 0; }
    if (g_num_nb_pairs == 0) {
        return 0;
    }

    if (num_poses > GPU_MAX_POSES) {
        fprintf(stderr, "GPU-DOCK: batch size %d exceeds max %d\n",
                num_poses, GPU_MAX_POSES);
        return 0;
    }
    if (num_atoms > GPU_MAX_ATOMS) {
        fprintf(stderr, "GPU-DOCK: atoms/pose %d exceeds max %d\n",
                num_atoms, GPU_MAX_ATOMS);
        return 0;
    }
    @autoreleasepool {

        size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
        memcpy([g_buf_xyz contents], xyz, xyz_bytes);

        id<MTLCommandBuffer>  cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"DockBatchScoreIE";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"BatchScoreIEKernel";

        [enc setComputePipelineState:g_pso_ie];

        /* Bind buffers — must match shader buffer layout */
        [enc setBuffer:g_buf_xyz        offset:0 atIndex:0];
        [enc setBuffer:g_buf_grid_avdw  offset:0 atIndex:1];
        [enc setBuffer:g_buf_grid_bvdw  offset:0 atIndex:2];
        [enc setBuffer:g_buf_grid_es    offset:0 atIndex:3];
        [enc setBuffer:g_buf_vdwA       offset:0 atIndex:4];
        [enc setBuffer:g_buf_vdwB       offset:0 atIndex:5];
        [enc setBuffer:g_buf_charges    offset:0 atIndex:6];
        [enc setBuffer:g_buf_params     offset:0 atIndex:7];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms    offset:0 atIndex:8];
        [enc setBuffer:g_buf_scores    offset:0 atIndex:9];
        [enc setBuffer:g_buf_ie_vdwA   offset:0 atIndex:10];
        [enc setBuffer:g_buf_nb_int    offset:0 atIndex:11];
        write_iep(g_ie_soft_delta, g_ie_cutoff_sq, g_num_nb_pairs);
        [enc setBuffer:g_buf_iep       offset:0 atIndex:12];
        write_nnp(g_num_nb_pairs);
        [enc setBuffer:g_buf_nnp       offset:0 atIndex:13];

        MTLSize threadsPerGrid  = MTLSizeMake(num_poses, 1, 1);
        NSUInteger tg = MIN(g_pso_ie.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);

        [enc dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.error) {
            NSLog(@"GPU-DOCK: command buffer error (IE): %@", cmdbuf.error);
            return 0;
        }
    }

    /* Readback outside the autoreleasepool to ensure buffer lifetime */
    memcpy(out_scores, [g_buf_scores contents], sizeof(float) * (size_t)num_poses);
    return 1;
}


void dock_gpu_cleanup(void)
{
    @autoreleasepool {
        g_buf_grid_avdw  = nil;
        g_buf_grid_bvdw  = nil;
        g_buf_grid_es    = nil;
        g_buf_vdwA       = nil;
        g_buf_vdwB       = nil;
        g_buf_charges    = nil;
        g_buf_ie_vdwA    = nil;
        g_buf_nb_int     = nil;
        g_buf_xyz        = nil;
        g_buf_scores     = nil;
        g_pso            = nil;
        g_pso_ie         = nil;
        g_cmdq           = nil;
        g_device         = nil;
        g_initialized    = 0;
        g_active         = 0;
        g_num_atoms      = 0;
        g_num_nb_pairs   = 0;
        g_ie_soft_delta  = 0.0;
        g_ie_cutoff_sq   = 1e10;
        memset(&g_params, 0, sizeof(g_params));
    }
}


int dock_gpu_is_active(void)
{
    return g_initialized && (g_device != nil);
}

} /* extern "C" */
