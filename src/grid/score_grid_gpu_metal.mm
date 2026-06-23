/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Metal GPU backend for grid generation — Tiled Gather Kernel.

Implements the GPU abstraction API defined in score_grid_gpu.h using
Apple Metal on macOS (Apple Silicon).

Kernel strategy: Tiled gather — 1 GPU thread per grid point, organized
into 8x8x8 tiles processed by threadgroups.  Atom data is loaded into
shared memory in batches, so all 512 threads in a threadgroup reuse the
same atom data without redundant global-memory reads.  Each thread
accumulates contributions to its unique grid point in registers (no
atomic operations needed), then writes the result coalesced.

This replaces the earlier scatter+atomic approach.  Key benefits:
  - Zero atomic operations (each thread owns its grid point)
  - Atom data cached in threadgroup shared memory (12 KB per batch)
  - High thread occupancy for any GPU runtime (span_x * span_y * span_z)
  - Scales gracefully with box size and grid spacing
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* Full struct definitions — needed because we access struct members like
   energy->repulsive_exponent, receptor->coord[i], label->vdw.total, etc.
   The public header score_grid_gpu.h only uses forward declarations to avoid
   including unguarded headers. */
#include "define.h"
#include "mol.h"
#include "label.h"
#include "score.h"
#include "grid.h"
#include "score_grid_gpu.h"
#include "score_grid_gpu_metal.h"

/* Tile dimensions — powers of two for efficient memory coalescing */
#define TILE_W  8
#define TILE_H  8
#define TILE_D  8
#define TILE_VOL (TILE_W * TILE_H * TILE_D)  /* 512 threads per tile */

/* Batch size for loading atoms into threadgroup shared memory.
   Must match TILE_VOL so each thread loads one atom per batch. */
/* Atom batch size matches tile volume for 1:1 thread-to-atom loading */


/* ================================================================== */
/*  Embedded Metal Shader Source — Tiled Gather Kernel                 */
/* ================================================================== */

static const char* shader_src = \
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant int TILE_W = 8;\n"
"constant int TILE_H = 8;\n"
"constant int TILE_D = 8;\n"
"\n"
"struct GridParams {\n"
"    float origin_x, origin_y, origin_z;\n"
"    int span_x, span_y, span_z;\n"
"    float spacing;\n"
"    float distance;\n"
"    float dist_sq_min;\n"
"    float rep_exponent;\n"
"    float att_exponent;\n"
"    int distance_dielectric;\n"
"    float dielectric_factor;\n"
"    float soft_delta;\n"
"    int grid_size;\n"
"};\n"
"\n"
"/* Integer exponentiation by squaring, matching CPU POWER() macro */\n"
"static float int_pow(float base, int exp)\n"
"{\n"
"    float result = 1.0;\n"
"    float run = base;\n"
"    while (exp) {\n"
"        if (exp & 1) result *= run;\n"
"        run *= run;\n"
"        exp >>= 1;\n"
"    }\n"
"    return result;\n"
"}\n"
"\n"
"kernel void grid_energy_kernel(\n"
"    device const float*    atom_pos     [[buffer(0)]],\n"
"    device const float*    atom_vdwA    [[buffer(1)]],\n"
"    device const float*    atom_vdwB    [[buffer(2)]],\n"
"    device const float*    atom_charge  [[buffer(3)]],\n"
"    device float*          avdw         [[buffer(4)]],\n"
"    device float*          bvdw         [[buffer(5)]],\n"
"    device float*          es           [[buffer(6)]],\n"
"    constant GridParams&   p            [[buffer(7)]],\n"
"    uint3                  gid          [[thread_position_in_grid]],\n"
"    uint3                  tpt          [[thread_position_in_threadgroup]])\n"
"{\n"
"    /* Clamp to grid bounds (handles edge tiles) */\n"
"    if (gid.x >= (uint)p.span_x ||\n"
"        gid.y >= (uint)p.span_y ||\n"
"        gid.z >= (uint)p.span_z) return;\n"
"\n"
"    /* Linear index for this grid point */\n"
"    int idx = p.span_x * p.span_y * (int)gid.z\n"
"            + p.span_x * (int)gid.y\n"
"            + (int)gid.x;\n"
"\n"
"    /* Position of this grid point in Angstroms */\n"
"    float3 gpos;\n"
"    gpos.x = (float)gid.x * p.spacing + p.origin_x;\n"
"    gpos.y = (float)gid.y * p.spacing + p.origin_y;\n"
"    gpos.z = (float)gid.z * p.spacing + p.origin_z;\n"
"\n"
"    /* Per-thread accumulators (registers — no atomics needed) */\n"
"    float my_avdw = 0.0;\n"
"    float my_bvdw = 0.0;\n"
"    float my_es   = 0.0;\n"
"\n"
"    /* Linear thread ID within threadgroup for cooperative loading */\n"
"    int tid = (int)(tpt.x + TILE_W * (tpt.y + TILE_H * tpt.z));\n"
"\n"
"    /* Shared memory buffers for one batch of atom data */\n"
"    threadgroup float3  sh_pos[TILE_W * TILE_H * TILE_D];\n"
"    threadgroup float   sh_vdwA[TILE_W * TILE_H * TILE_D];\n"
"    threadgroup float   sh_vdwB[TILE_W * TILE_H * TILE_D];\n"
"    threadgroup float   sh_charge[TILE_W * TILE_H * TILE_D];\n"
"\n"
"    /* Number of atoms to process */\n"
"    /* NOTE: In a real kernel we'd pass num_atoms as a parameter.  We use\n"
"       a large sentinel (max_uint32) and the batch loop runs until the host\n"
"       signals completion.  For now we iterate over ALL atoms per tile.\n"
"       Future: pass p.num_atoms and only load that many batches. */\n"
"    /* --- Process atoms in batches --- */\n"
"    int num_atoms = p.grid_size;  /* reuse field as atom count for now */\n"
"    for (int batch_start = 0; batch_start < num_atoms; batch_start += TILE_W * TILE_H * TILE_D) {\n"
"\n"
"        /* Cooperative load: each thread loads one atom into shared memory */\n"
"        int atom_id = batch_start + tid;\n"
"        if (atom_id < num_atoms) {\n"
"            int ai3 = 3 * atom_id;\n"
"            sh_pos[tid].x = atom_pos[ai3];\n"
"            sh_pos[tid].y = atom_pos[ai3 + 1];\n"
"            sh_pos[tid].z = atom_pos[ai3 + 2];\n"
"            sh_vdwA[tid]    = atom_vdwA[atom_id];\n"
"            sh_vdwB[tid]    = atom_vdwB[atom_id];\n"
"            sh_charge[tid]  = atom_charge[atom_id];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"        int batch_end = min(TILE_W * TILE_H * TILE_D, num_atoms - batch_start);\n"
"        for (int a = 0; a < batch_end; a++) {\n"
"            float3 apos = sh_pos[a];\n"
"            float dx = gpos.x - apos.x;\n"
"            float dy = gpos.y - apos.y;\n"
"            float dz = gpos.z - apos.z;\n"
"            float dist_sq = dx * dx + dy * dy + dz * dz;\n"
"\n"
"            if (dist_sq < p.dist_sq_min)\n"
"                dist_sq = p.dist_sq_min;\n"
"\n"
"            float dist = sqrt(dist_sq);\n"
"\n"
"            if (dist <= p.distance) {\n"
"                float dist_inv = 1.0 / dist;\n"
"\n"
"                /* Soft-core repulsive distance */\n"
"                float rep_dist_inv;\n"
"                if (p.soft_delta > 0.0) {\n"
"                    float sd = sqrt(dist_sq + p.soft_delta);\n"
"                    rep_dist_inv = 1.0 / sd;\n"
"                } else {\n"
"                    rep_dist_inv = dist_inv;\n"
"                }\n"
"\n"
"                int rep_exp = (int)(p.rep_exponent + 0.5);\n"
"                int att_exp = (int)(p.att_exponent + 0.5);\n"
"                float rep_power = int_pow(rep_dist_inv, rep_exp);\n"
"                float att_power = int_pow(dist_inv, att_exp);\n"
"\n"
"                my_avdw += sh_vdwA[a] * rep_power;\n"
"                my_bvdw += sh_vdwB[a] * att_power;\n"
"\n"
"                /* Electrostatic */\n"
"                float es_val;\n"
"                if (p.distance_dielectric)\n"
"                    es_val = p.dielectric_factor * sh_charge[a]\n"
"                           * (dist_inv * dist_inv);\n"
"                else\n"
"                    es_val = p.dielectric_factor * sh_charge[a] * dist_inv;\n"
"                my_es += es_val;\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"\n"
"    /* Write results — no atomics needed, each thread owns a unique index */\n"
"    avdw[idx] = my_avdw;\n"
"    bvdw[idx] = my_bvdw;\n"
"    es[idx]   = my_es;\n"
"}\n";


/* ================================================================== */
/*  Static state (single-instance, non-reentrant — fine for grid)     */
/* ================================================================== */

static id<MTLDevice>               g_device    = nil;
static id<MTLCommandQueue>         g_cmdq      = nil;
static id<MTLComputePipelineState> g_pso       = nil;

/* GPU buffers (shared memory — CPU and GPU see the same data) */
static id<MTLBuffer> g_buf_atom_pos    = nil;
static id<MTLBuffer> g_buf_atom_vdwA   = nil;
static id<MTLBuffer> g_buf_atom_vdwB   = nil;
static id<MTLBuffer> g_buf_atom_charge = nil;
static id<MTLBuffer> g_buf_avdw        = nil;
static id<MTLBuffer> g_buf_bvdw        = nil;
static id<MTLBuffer> g_buf_es          = nil;

/* Cached parameters */
static GridParams g_params;           /* filled during init, used by compute */
static int  g_num_atoms   = 0;
static int  g_grid_size   = 0;
static int  g_initialized = 0;        /* device + pipeline created */
static int  g_uploaded    = 0;        /* buffers allocated + populated */


/* ================================================================== */
/*  Helper: allocate an MTLBuffer with shared storage                  */
/* ================================================================== */

static id<MTLBuffer> alloc_buffer(NSUInteger size, const char* label)
{
    id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (buf)
        buf.label = [NSString stringWithUTF8String:label];
    else
        NSLog(@"Metal: failed to allocate buffer '%s' (%lu bytes)",
              label, (unsigned long)size);
    return buf;
}


/* ================================================================== */
/*  GPU abstraction API                                                */
/* ================================================================== */

int gpu_grid_init(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                  SCORE_BUMP *bump, SCORE_CONTACT *contact,
                  SCORE_CHEMICAL *chemical, LABEL *label)
{
    (void)energy; (void)receptor;
    (void)bump; (void)contact; (void)chemical;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "GPU-METAL: Metal GPU not available — CPU fallback\n");
            return 0;
        }
        fprintf(stderr, "GPU-METAL: Metal device: %s\n",
                [g_device.name UTF8String]);

        g_cmdq = [g_device newCommandQueue];
        if (!g_cmdq) {
            fprintf(stderr, "GPU-METAL: failed to create command queue — CPU fallback\n");
            gpu_grid_cleanup();
            return 0;
        }

        /* Compile embedded shader at runtime */
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:
                               [NSString stringWithUTF8String:shader_src]
                                                     options:nil
                                                       error:&err];
        if (!lib) {
            fprintf(stderr, "GPU-METAL: shader compilation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            gpu_grid_cleanup();
            return 0;
        }

        id<MTLFunction> func = [lib newFunctionWithName:@"grid_energy_kernel"];
        if (!func) {
            fprintf(stderr, "GPU-METAL: kernel 'grid_energy_kernel' not found\n");
            gpu_grid_cleanup();
            return 0;
        }

        g_pso = [g_device newComputePipelineStateWithFunction:func error:&err];
        if (!g_pso) {
            fprintf(stderr, "GPU-METAL: pipeline state creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            gpu_grid_cleanup();
            return 0;
        }

        /* Cache grid parameters for the compute kernel.
           We also reuse the grid_size field as a num_atoms transport for the
           shader, since the grid params are the only constant buffer sent to
           the kernel.  This avoids adding a separate num_atoms parameter. */
        g_params.origin_x          = grid->origin[0];
        g_params.origin_y          = grid->origin[1];
        g_params.origin_z          = grid->origin[2];
        g_params.span_x            = grid->span[0];
        g_params.span_y            = grid->span[1];
        g_params.span_z            = grid->span[2];
        g_params.spacing           = grid->spacing;
        g_params.distance          = grid->distance;
        g_params.dist_sq_min       = SQR(DISTANCE_MIN);
        g_params.rep_exponent      = (float)energy->repulsive_exponent;
        g_params.att_exponent      = (float)energy->attractive_exponent;
        g_params.distance_dielectric = energy->distance_dielectric;
        g_params.dielectric_factor = energy->dielectric_factor;
        g_params.soft_delta        = 0.0f;  /* set at compute time */
        g_params.grid_size         = grid->span[0] * grid->span[1] * grid->span[2];

        g_grid_size = g_params.grid_size;

        g_initialized = 1;
        g_uploaded    = 0;

        NSLog(@"Metal GPU ready: grid %dx%dx%d = %d pts, tile %dx%dx%d",
              g_params.span_x, g_params.span_y, g_params.span_z,
              g_grid_size, TILE_W, TILE_H, TILE_D);
        return 1;
    }
}


void gpu_grid_upload(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                     SCORE_BUMP *bump, SCORE_CONTACT *contact,
                     SCORE_CHEMICAL *chemical, LABEL *label)
{
    (void)bump; (void)contact; (void)chemical;

    if (!g_initialized) return;

    @autoreleasepool {
        g_num_atoms = receptor->total.atoms;
        if (g_num_atoms <= 0) {
            NSLog(@"Metal: atom count %d invalid — CPU fallback", g_num_atoms);
            gpu_grid_cleanup();
            return;
        }

        /* Update grid params from the SCORE_GRID (may differ from init if
           the grid struct was reconstructed between init and upload) */
        g_params.origin_x = grid->origin[0];
        g_params.origin_y = grid->origin[1];
        g_params.origin_z = grid->origin[2];
        g_params.span_x   = grid->span[0];
        g_params.span_y   = grid->span[1];
        g_params.span_z   = grid->span[2];
        g_params.spacing  = grid->spacing;
        g_params.distance = grid->distance;
        g_params.grid_size = grid->span[0] * grid->span[1] * grid->span[2];
        g_grid_size = g_params.grid_size;

        /* Allocate shared-memory buffers — exact sizes, no sentinel */
        g_buf_atom_pos    = alloc_buffer(sizeof(float) * 3 * g_num_atoms, "atom_pos");
        g_buf_atom_vdwA   = alloc_buffer(sizeof(float) * g_num_atoms,    "atom_vdwA");
        g_buf_atom_vdwB   = alloc_buffer(sizeof(float) * g_num_atoms,    "atom_vdwB");
        g_buf_atom_charge = alloc_buffer(sizeof(float) * g_num_atoms,    "atom_charge");
        g_buf_avdw        = alloc_buffer(sizeof(float) * g_grid_size,    "avdw");
        g_buf_bvdw        = alloc_buffer(sizeof(float) * g_grid_size,    "bvdw");
        g_buf_es          = alloc_buffer(sizeof(float) * g_grid_size,    "es");

        if (!g_buf_atom_pos || !g_buf_atom_vdwA || !g_buf_atom_vdwB ||
            !g_buf_atom_charge ||
            !g_buf_avdw   || !g_buf_bvdw   || !g_buf_es) {
            NSLog(@"Metal: buffer allocation failed — CPU fallback");
            gpu_grid_cleanup();
            return;
        }

        /* --- Populate buffers --- */

        /* Atom positions: 3 consecutive floats per atom */
        float* pos_ptr = (float*)[g_buf_atom_pos contents];
        for (int i = 0; i < g_num_atoms; i++) {
            pos_ptr[i*3 + 0] = receptor->coord[i][0];
            pos_ptr[i*3 + 1] = receptor->coord[i][1];
            pos_ptr[i*3 + 2] = receptor->coord[i][2];
        }

        /* Pre-resolve VDW A/B per atom (remove type-lookup indirection) */
        float* vdwA_ptr = (float*)[g_buf_atom_vdwA contents];
        float* vdwB_ptr = (float*)[g_buf_atom_vdwB contents];
        float* chg_ptr  = (float*)[g_buf_atom_charge contents];
        for (int i = 0; i < g_num_atoms; i++) {
            int vid = receptor->atom[i].vdw_id;
            vdwA_ptr[i] = energy->vdwA[vid];
            vdwB_ptr[i] = energy->vdwB[vid];
            chg_ptr[i]  = receptor->atom[i].charge;
        }

        /* Zero output grids */
        memset([g_buf_avdw contents], 0, sizeof(float) * g_grid_size);
        memset([g_buf_bvdw contents], 0, sizeof(float) * g_grid_size);
        memset([g_buf_es   contents], 0, sizeof(float) * g_grid_size);

        g_uploaded = 1;
        NSLog(@"Metal: uploaded %d atoms, grid %d pts (%dx%dx%d)",
              g_num_atoms, g_grid_size,
              g_params.span_x, g_params.span_y, g_params.span_z);
    }
}


void gpu_grid_compute(float soft_delta)
{
    if (!g_initialized || !g_uploaded) return;

    @autoreleasepool {
        g_params.soft_delta = soft_delta;
        /* Reuse grid_size field to pass atom count to the shader.
           This is safe because grid_size is only used to size global-memory
           buffers, which are allocated in upload() before compute() runs.
           The shader uses grid_size as num_atoms for the batch loop. */
        g_params.grid_size = g_num_atoms;

        id<MTLCommandBuffer>  cmdbuf  = [g_cmdq commandBuffer];
        cmdbuf.label = @"GridCompute";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"GridEnergyKernel";

        [enc setComputePipelineState:g_pso];

        /* Bind all buffers */
        [enc setBuffer:g_buf_atom_pos    offset:0 atIndex:0];
        [enc setBuffer:g_buf_atom_vdwA   offset:0 atIndex:1];
        [enc setBuffer:g_buf_atom_vdwB   offset:0 atIndex:2];
        [enc setBuffer:g_buf_atom_charge offset:0 atIndex:3];
        [enc setBuffer:g_buf_avdw        offset:0 atIndex:4];
        [enc setBuffer:g_buf_bvdw        offset:0 atIndex:5];
        [enc setBuffer:g_buf_es          offset:0 atIndex:6];

        /* Write GridParams into a temporary buffer for the constant argument */
        id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&g_params
                                                          length:sizeof(GridParams)
                                                         options:MTLResourceStorageModeShared];
        [enc setBuffer:params_buf offset:0 atIndex:7];

        /* Dispatch one thread per grid point, organized in 8x8x8 tiles */
        MTLSize threadsPerGrid  = MTLSizeMake(g_params.span_x,
                                              g_params.span_y,
                                              g_params.span_z);
        MTLSize threadgroupSize = MTLSizeMake(TILE_W, TILE_H, TILE_D);

        [enc dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        /* Commit and wait for completion */
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        NSLog(@"Metal: GPU compute finished (%d atoms, %d grid pts, tiles %dx%dx%d)",
              g_num_atoms, g_grid_size,
              (g_params.span_x + TILE_W - 1) / TILE_W,
              (g_params.span_y + TILE_H - 1) / TILE_H,
              (g_params.span_z + TILE_D - 1) / TILE_D);
    }
}


void gpu_grid_download(SCORE_ENERGY *energy, SCORE_GRID *grid,
                       SCORE_BUMP *bump, SCORE_CONTACT *contact,
                       SCORE_CHEMICAL *chemical)
{
    (void)grid; (void)bump; (void)contact; (void)chemical;

    if (!g_initialized || !g_uploaded) return;

    @autoreleasepool {
        /* Shared memory — CPU can read buffers directly after GPU completes */
        memcpy(energy->avdw, [g_buf_avdw contents], sizeof(float) * g_grid_size);
        memcpy(energy->bvdw, [g_buf_bvdw contents], sizeof(float) * g_grid_size);
        memcpy(energy->es,   [g_buf_es   contents], sizeof(float) * g_grid_size);
        NSLog(@"Metal: downloaded %d floats per grid (avdw, bvdw, es)", g_grid_size);
    }
}


void gpu_grid_cleanup(void)
{
    @autoreleasepool {
        g_buf_atom_pos    = nil;
        g_buf_atom_vdwA   = nil;
        g_buf_atom_vdwB   = nil;
        g_buf_atom_charge = nil;
        g_buf_avdw        = nil;
        g_buf_bvdw        = nil;
        g_buf_es          = nil;
        g_pso             = nil;
        g_cmdq            = nil;
        g_device          = nil;
        g_num_atoms       = 0;
        g_grid_size       = 0;
        g_initialized     = 0;
        g_uploaded        = 0;
        memset(&g_params, 0, sizeof(g_params));
    }
}


int gpu_grid_is_active(void)
{
    return g_initialized && g_uploaded && (g_device != nil);
}
