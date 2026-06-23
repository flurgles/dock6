/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Metal GPU backend for grid generation.

Implements the GPU abstraction API defined in score_grid_gpu.h using
Apple Metal on macOS (Apple Silicon).  The Metal compute kernel is
embedded as a C string and compiled at runtime via newLibraryWithSource:
to avoid adding a .metal compilation step to the Makefile.

Architecture:
    one GPU thread per receptor atom (scatter pattern)
    each thread iterates its bounding box of grid points
    accumulates contributions via atomic_fetch_add_explicit
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
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

/* ================================================================== */
/*  Embedded Metal Shader Source                                      */
/* ================================================================== */

static const char* shader_src = \
"#include <metal_stdlib>\n"
"using namespace metal;\n"
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
"    device const float*    atom_pos    [[buffer(0)]],\n"
"    device const int*      vdw_id      [[buffer(1)]],\n"
"    device const float*    charge      [[buffer(2)]],\n"
"    device const float*    vdwA        [[buffer(3)]],\n"
"    device const float*    vdwB        [[buffer(4)]],\n"
"    device float*          avdw        [[buffer(5)]],\n"
"    device float*          bvdw        [[buffer(6)]],\n"
"    device float*          es          [[buffer(7)]],\n"
"    constant GridParams&   p           [[buffer(8)]],\n"
"    uint                   atom_id     [[thread_position_in_grid]])\n"
"{\n"
"    int ai = 3 * atom_id;\n"
"    float3 pos = float3(atom_pos[ai], atom_pos[ai+1], atom_pos[ai+2]);\n"
"    int vid = vdw_id[atom_id];\n"
"\n"
"    /* Skip zero-well-depth atoms */\n"
"    if (vdwA[vid] == 0.0f) return;\n"
"\n"
"    /* Grid-coordinate of this atom */\n"
"    float3 ocrd = float3(pos.x - p.origin_x, pos.y - p.origin_y, pos.z - p.origin_z);\n"
"    int3 g = int3(round(ocrd / p.spacing));\n"
"\n"
"    int grid_cutoff = (int)(p.distance / p.spacing + 1.0f);\n"
"\n"
"    /* Quick reject if atom is far outside grid bounds */\n"
"    if (g.x < -grid_cutoff || g.x >= p.span_x + grid_cutoff) return;\n"
"    if (g.y < -grid_cutoff || g.y >= p.span_y + grid_cutoff) return;\n"
"    if (g.z < -grid_cutoff || g.z >= p.span_z + grid_cutoff) return;\n"
"\n"
"    /* Bounding box of grid points within cutoff of this atom */\n"
"    int i0 = max(0, g.x - grid_cutoff);\n"
"    int i1 = min(p.span_x, g.x + grid_cutoff + 1);\n"
"    int j0 = max(0, g.y - grid_cutoff);\n"
"    int j1 = min(p.span_y, g.y + grid_cutoff + 1);\n"
"    int k0 = max(0, g.z - grid_cutoff);\n"
"    int k1 = min(p.span_z, g.z + grid_cutoff + 1);\n"
"\n"
"    float aA = vdwA[vid];\n"
"    float aB = vdwB[vid];\n"
"    float q = charge[atom_id];\n"
"    int rep_exp = (int)(p.rep_exponent + 0.5f);\n"
"    int att_exp = (int)(p.att_exponent + 0.5f);\n"
"\n"
"    for (int i = i0; i < i1; i++) {\n"
"        float gx = float(i) * p.spacing + p.origin_x;\n"
"        float dx = gx - pos.x;\n"
"        float dx2 = dx * dx;\n"
"\n"
"        for (int j = j0; j < j1; j++) {\n"
"            float gy = float(j) * p.spacing + p.origin_y;\n"
"            float dy = gy - pos.y;\n"
"            float dy2 = dy * dy;\n"
"            float dxy2 = dx2 + dy2;\n"
"\n"
"            for (int k = k0; k < k1; k++) {\n"
"                float gz = float(k) * p.spacing + p.origin_z;\n"
"                float dz = gz - pos.z;\n"
"                float dist_sq = dxy2 + dz * dz;\n"
"\n"
"                if (dist_sq < p.dist_sq_min)\n"
"                    dist_sq = p.dist_sq_min;\n"
"\n"
"                float dist = sqrt(dist_sq);\n"
"\n"
"                if (dist <= p.distance) {\n"
"                    int idx = p.span_x * p.span_y * k + p.span_x * j + i;\n"
"                    float dist_inv = 1.0f / dist;\n"
"\n"
"                    /* Soft-core repulsive distance */\n"
"                    float rep_dist_inv;\n"
"                    if (p.soft_delta > 0.0f) {\n"
"                        float sd = sqrt(dist_sq + p.soft_delta);\n"
"                        rep_dist_inv = 1.0f / sd;\n"
"                    } else {\n"
"                        rep_dist_inv = dist_inv;\n"
"                    }\n"
"\n"
"                    float rep_power = int_pow(rep_dist_inv, rep_exp);\n"
"                    float att_power = int_pow(dist_inv, att_exp);\n"
"\n"
"                    atomic_fetch_add_explicit(\n"
"                        (device atomic_float*)&avdw[idx],\n"
"                        aA * rep_power, memory_order_relaxed);\n"
"\n"
"                    atomic_fetch_add_explicit(\n"
"                        (device atomic_float*)&bvdw[idx],\n"
"                        aB * att_power, memory_order_relaxed);\n"
"\n"
"                    /* Electrostatic */\n"
"                    float es_val;\n"
"                    if (p.distance_dielectric)\n"
"                        es_val = p.dielectric_factor * q * (dist_inv * dist_inv);\n"
"                    else\n"
"                        es_val = p.dielectric_factor * q * dist_inv;\n"
"\n"
"                    atomic_fetch_add_explicit(\n"
"                        (device atomic_float*)&es[idx],\n"
"                        es_val, memory_order_relaxed);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";


/* ================================================================== */
/*  Static state (single-instance, non-reentrant — fine for grid)     */
/* ================================================================== */

static id<MTLDevice>               g_device    = nil;
static id<MTLCommandQueue>         g_cmdq      = nil;
static id<MTLComputePipelineState> g_pso       = nil;

/* GPU buffers (shared memory — CPU and GPU see the same data) */
static id<MTLBuffer> g_buf_atom_pos = nil;
static id<MTLBuffer> g_buf_vdw_id   = nil;
static id<MTLBuffer> g_buf_charge   = nil;
static id<MTLBuffer> g_buf_vdwA     = nil;
static id<MTLBuffer> g_buf_vdwB     = nil;
static id<MTLBuffer> g_buf_avdw     = nil;
static id<MTLBuffer> g_buf_bvdw     = nil;
static id<MTLBuffer> g_buf_es       = nil;

/* Cached parameters */
static GridParams g_params;           /* filled during init, used by compute */
static int  g_num_atoms   = 0;
static int  g_grid_size   = 0;
static int  g_vdw_total   = 0;
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

        /* Cache grid parameters for the compute kernel */
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
        g_vdw_total = label->vdw.total;

        g_initialized = 1;
        g_uploaded    = 0;

        NSLog(@"Metal GPU ready: %d atom slots, grid %dx%dx%d = %d pts",
              GPU_MAX_ATOMS,
              g_params.span_x, g_params.span_y, g_params.span_z,
              g_grid_size);
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
        if (g_num_atoms <= 0 || g_num_atoms > GPU_MAX_ATOMS) {
            NSLog(@"Metal: atom count %d out of range [1,%d] — CPU fallback",
                  g_num_atoms, GPU_MAX_ATOMS);
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

        /* Allocate shared-memory buffers */
        g_buf_atom_pos = alloc_buffer(sizeof(float) * 3 * GPU_MAX_ATOMS, "atom_pos");
        g_buf_vdw_id   = alloc_buffer(sizeof(int)   * GPU_MAX_ATOMS,    "vdw_id");
        g_buf_charge   = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS,    "charge");
        g_buf_vdwA     = alloc_buffer(sizeof(float) * g_vdw_total,      "vdwA");
        g_buf_vdwB     = alloc_buffer(sizeof(float) * g_vdw_total,      "vdwB");
        g_buf_avdw     = alloc_buffer(sizeof(float) * g_grid_size,      "avdw");
        g_buf_bvdw     = alloc_buffer(sizeof(float) * g_grid_size,      "bvdw");
        g_buf_es       = alloc_buffer(sizeof(float) * g_grid_size,      "es");

        if (!g_buf_atom_pos || !g_buf_vdw_id || !g_buf_charge ||
            !g_buf_vdwA   || !g_buf_vdwB   ||
            !g_buf_avdw   || !g_buf_bvdw   || !g_buf_es) {
            NSLog(@"Metal: buffer allocation failed — CPU fallback");
            gpu_grid_cleanup();
            return;
        }

        /* --- Populate buffers --- */

        /* Atom positions: float3 = 3 consecutive floats per atom */
        float* pos_ptr = (float*)[g_buf_atom_pos contents];
        for (int i = 0; i < g_num_atoms; i++) {
            pos_ptr[i*3 + 0] = receptor->coord[i][0];
            pos_ptr[i*3 + 1] = receptor->coord[i][1];
            pos_ptr[i*3 + 2] = receptor->coord[i][2];
        }

        /* VDW IDs and charges */
        int*   vdw_ptr = (int*)[g_buf_vdw_id contents];
        float* chg_ptr = (float*)[g_buf_charge contents];
        for (int i = 0; i < g_num_atoms; i++) {
            vdw_ptr[i] = receptor->atom[i].vdw_id;
            chg_ptr[i] = receptor->atom[i].charge;
        }

        /* VDW parameters */
        memcpy([g_buf_vdwA contents], energy->vdwA, sizeof(float) * g_vdw_total);
        memcpy([g_buf_vdwB contents], energy->vdwB, sizeof(float) * g_vdw_total);

        /* Zero output grids */
        memset([g_buf_avdw contents], 0, sizeof(float) * g_grid_size);
        memset([g_buf_bvdw contents], 0, sizeof(float) * g_grid_size);
        memset([g_buf_es   contents], 0, sizeof(float) * g_grid_size);

        g_uploaded = 1;
        NSLog(@"Metal: uploaded %d atoms, %d vdw types, grid %d pts",
              g_num_atoms, g_vdw_total, g_grid_size);
    }
}


void gpu_grid_compute(float soft_delta)
{
    if (!g_initialized || !g_uploaded) return;

    @autoreleasepool {
        g_params.soft_delta = soft_delta;

        id<MTLCommandBuffer>  cmdbuf  = [g_cmdq commandBuffer];
        cmdbuf.label = @"GridCompute";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"GridEnergyKernel";

        [enc setComputePipelineState:g_pso];

        /* Bind all buffers */
        [enc setBuffer:g_buf_atom_pos offset:0 atIndex:0];
        [enc setBuffer:g_buf_vdw_id   offset:0 atIndex:1];
        [enc setBuffer:g_buf_charge   offset:0 atIndex:2];
        [enc setBuffer:g_buf_vdwA     offset:0 atIndex:3];
        [enc setBuffer:g_buf_vdwB     offset:0 atIndex:4];
        [enc setBuffer:g_buf_avdw     offset:0 atIndex:5];
        [enc setBuffer:g_buf_bvdw     offset:0 atIndex:6];
        [enc setBuffer:g_buf_es       offset:0 atIndex:7];

        /* Write GridParams into a temporary buffer for the constant argument */
        id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&g_params
                                                          length:sizeof(GridParams)
                                                         options:MTLResourceStorageModeShared];
        [enc setBuffer:params_buf offset:0 atIndex:8];

        /* Dispatch one thread per receptor atom */
        MTLSize threadsPerGrid  = MTLSizeMake(g_num_atoms, 1, 1);
        MTLSize maxThreadgroup  = MTLSizeMake([g_pso maxTotalThreadsPerThreadgroup], 1, 1);
        /* Use a reasonable threadgroup size (Apple Silicon sweet spot) */
        NSUInteger tg_size = 256;
        if (tg_size > maxThreadgroup.width) tg_size = maxThreadgroup.width;
        MTLSize threadgroupSize = MTLSizeMake(tg_size, 1, 1);

        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        /* Commit and wait for completion */
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        NSLog(@"Metal: GPU compute finished (%d atoms, %d grid pts)",
              g_num_atoms, g_grid_size);
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
        g_buf_atom_pos = nil;
        g_buf_vdw_id   = nil;
        g_buf_charge   = nil;
        g_buf_vdwA     = nil;
        g_buf_vdwB     = nil;
        g_buf_avdw     = nil;
        g_buf_bvdw     = nil;
        g_buf_es       = nil;
        g_pso          = nil;
        g_cmdq         = nil;
        g_device       = nil;
        g_num_atoms    = 0;
        g_grid_size    = 0;
        g_vdw_total    = 0;
        g_initialized  = 0;
        g_uploaded     = 0;
        memset(&g_params, 0, sizeof(g_params));
    }
}


int gpu_grid_is_active(void)
{
    return g_initialized && g_uploaded && (g_device != nil);
}
