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
#import <IOKit/IOKitLib.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include "score_dock_gpu.h"
#include "score_dock_gpu_metal.h"

/* Max poses per batch */
#define GPU_MAX_POSES       4096
#define GPU_MAX_ATOMS       512
#define GPU_MAX_NB_PAIRS    32768  /* Max non-bonded pairs per ligand */
#define GPU_MAX_TORSIONS   50    /* Max rotatable bonds */
#define GPU_DOF_MAX        56    /* 6 + max torsions */
#define BATCH_MAX         32    /* Max conformers per batch dispatch */
#define MAX_NVERTS        (GPU_DOF_MAX + 1)  /* Max simplex vertices */


/* ================================================================== */
/*  Embedded Metal Shader Source — Both kernels in one library         */
/* ================================================================== */
/*  Simplex shader (unified: batch + IE + simplex in one raw string)   */
/* ================================================================== */

static const char* shader_src_all = R"shader(
#include <metal_stdlib>
using namespace metal;

constexpr sampler grid_sampler(filter::linear, address::clamp_to_edge);

#define PI 3.14159265358979323846f
#define MAX_ATOMS 512
#define MAX_TORSIONS 50
/* NOTE: DOF_MAX = 56 (=6+50).  Ligands with >50 rotatable bonds will have
 * their DOF silently truncated to 56 on GPU (the CPU path handles them
 * correctly).  Add a runtime check or bump MAX_TORSIONS if this becomes
 * a limitation for your target systems. */
#define DOF_MAX (6 + MAX_TORSIONS)
#define MAX_SIMPLEX_THREADS 256
#define GPU_MAX_ATOMS MAX_ATOMS
#define MAX_NVERTS (DOF_MAX + 1)

struct GridParams {
    float origin_x, origin_y, origin_z;
    int   span_x, span_y, span_z;
    float spacing;
    int   grid_size;
};

struct IEParams {
    float soft_delta;
    float cutoff_sq;
    int   num_pairs;
    int   _pad;
};

/* Forward declarations (used by batch kernels before full definitions) */
static float trilinear(texture3d<float> grid, constant GridParams &p,
                        float x, float y, float z);
static bool inside_grid(constant GridParams &p, float x, float y, float z);

/* ============ Grid-only batch kernel ============ */
kernel void batch_score_kernel(
    device const float*    xyz          [[buffer(0)]],
    texture3d<float>       grid_avdw    [[texture(0)]],
    texture3d<float>       grid_bvdw    [[texture(1)]],
    texture3d<float>       grid_es      [[texture(2)]],
    device const float*    vdwA         [[buffer(1)]],
    device const float*    vdwB         [[buffer(2)]],
    device const float*    charges      [[buffer(3)]],
    constant GridParams&   gp           [[buffer(4)]],
    constant int&          num_atoms    [[buffer(5)]],
    device float*          out_scores   [[buffer(6)]],
    device const int*     active_flags [[buffer(7)]],
    uint                   tid          [[thread_position_in_grid]])
{
    int atoms_per_pose = num_atoms;
    int stride = 3 * atoms_per_pose;
    int base = tid * stride;
    float score = 0.0;
    for (int a = 0; a < atoms_per_pose; a++) {
        int o3 = base + a * 3;
        float x = xyz[o3], y = xyz[o3+1], z = xyz[o3+2];
        float vdw = trilinear(grid_avdw, gp, x, y, z);
        float bvdw = trilinear(grid_bvdw, gp, x, y, z);
        float es = trilinear(grid_es, gp, x, y, z);
        if (active_flags[a])
            score += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
    }
    out_scores[tid] = score;
}

/* ============ Grid + IE batch kernel ============ */
kernel void batch_score_with_ie_kernel(
    device const float*    xyz            [[buffer(0)]],
    texture3d<float>       grid_avdw      [[texture(0)]],
    texture3d<float>       grid_bvdw      [[texture(1)]],
    texture3d<float>       grid_es        [[texture(2)]],
    device const float*    vdwA           [[buffer(1)]],
    device const float*    vdwB           [[buffer(2)]],
    device const float*    charges        [[buffer(3)]],
    constant GridParams&   gp             [[buffer(4)]],
    constant int&          num_atoms      [[buffer(5)]],
    device float*          out_scores     [[buffer(6)]],
    device const float*    ie_vdwA        [[buffer(7)]],
    device const int*      nb_int         [[buffer(8)]],
    constant IEParams&     iep            [[buffer(9)]],
    constant int&          num_nb_pairs   [[buffer(10)]],
    device const int*     active_flags   [[buffer(11)]],
    uint                   tid            [[thread_position_in_grid]])
{
    int atoms_per_pose = num_atoms;
    int stride = 3 * atoms_per_pose;
    int base = tid * stride;
    float grid_score = 0.0;
    for (int a = 0; a < atoms_per_pose; a++) {
        int o3 = base + a * 3;
        float x = xyz[o3], y = xyz[o3+1], z = xyz[o3+2];
        if (active_flags[a] && !inside_grid(gp, x, y, z)) {
            out_scores[tid] = -3.40282347e+38;
            return;
        }
        if (active_flags[a]) {
            float vdw = trilinear(grid_avdw, gp, x, y, z);
            float bvdw = trilinear(grid_bvdw, gp, x, y, z);
            float es = trilinear(grid_es, gp, x, y, z);
            grid_score += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
        }
    }
    float ie_score = 0.0;
    if (num_nb_pairs > 0) {
        for (int p = 0; p < num_nb_pairs; p++) {
            int a1 = nb_int[p*2], a2 = nb_int[p*2+1];
            /* C5: NB pairs pre-filtered to active-only on CPU */
            int o1 = base + a1*3, o2 = base + a2*3;
            float dx = xyz[o1]-xyz[o2], dy = xyz[o1+1]-xyz[o2+1], dz = xyz[o1+2]-xyz[o2+2];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < iep.cutoff_sq) {
                float r2eff = r2 + iep.soft_delta;
                float denom = r2eff*r2eff*r2eff;
                ie_score += (ie_vdwA[a1]*ie_vdwA[a2]) / (denom*denom);
            }
        }
    }
    out_scores[tid] = grid_score + ie_score;
}

/* ============ Atom-parallel batch scoring kernel ============
 * Each threadgroup handles one pose. Threads are atom-parallel:
 * each thread scores 1-2 atoms. Grid contribution is skipped if
 * atom is OOB; IE always runs via per-atom pair lists. Single-phase,
 * no sentinel, one barrier (cross-SIMD reduction). Threadgroup size = 64.
 *
 * Buffer layout (differs from batch_score_with_ie_kernel):
 *   buf  0: xyz [N][num_atoms][3]
 *   tex 0-2: grid_avdw, grid_bvdw, grid_es
 *   buf  1: vdwA
 *   buf  2: vdwB
 *   buf  3: charges
 *   buf  4: GridParams (constant)
 *   buf  5: num_atoms (constant)
 *   buf  6: active_flags
 *   buf  7: ie_vdwA
 *   buf  8: pair_starts  (per-atom pair list prefix-sum)
 *   buf  9: pair_indices (per-atom pair list flattened indices)
 *   buf 10: total_pairs (constant, for bounds check)
 *   buf 11: ie_soft_delta (constant)
 *   buf 12: ie_cutoff_sq (constant)
 *   buf 13: out_scores [N]
 */
kernel void score_batch_kernel_atom_parallel(
    device const float*    xyz            [[buffer(0)]],
    texture3d<float>       grid_avdw      [[texture(0)]],
    texture3d<float>       grid_bvdw      [[texture(1)]],
    texture3d<float>       grid_es        [[texture(2)]],
    device const float*    vdwA           [[buffer(1)]],
    device const float*    vdwB           [[buffer(2)]],
    device const float*    charges        [[buffer(3)]],
    constant GridParams&   gp             [[buffer(4)]],
    constant int&          num_atoms      [[buffer(5)]],
    device const int*      active_flags   [[buffer(6)]],
    device const float*    ie_vdwA        [[buffer(7)]],
    device const int*      pair_starts    [[buffer(8)]],
    device const int*      pair_indices   [[buffer(9)]],
    constant int&          total_pairs    [[buffer(10)]],
    constant float&        ie_soft_delta  [[buffer(11)]],
    constant float&        ie_cutoff_sq   [[buffer(12)]],
    device float*          out_scores     [[buffer(13)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]])
{
    int stride = candidate * num_atoms * 3;
    uint simd_idx = tid / 32;
    uint num_simd = (tg_size + 31) / 32;

    float total = 0.0;
    for (uint a = tid; a < num_atoms; a += tg_size) {
        if (!active_flags[a]) continue;

        float x = xyz[stride + a*3];
        float y = xyz[stride + a*3 + 1];
        float z = xyz[stride + a*3 + 2];

        // Grid score — skip if this atom is outside the grid
        if (inside_grid(gp, x, y, z)) {
            float vdw  = trilinear(grid_avdw, gp, x, y, z);
            float bvdw = trilinear(grid_bvdw, gp, x, y, z);
            float es   = trilinear(grid_es, gp, x, y, z);
            total += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
        }

        // IE score — always runs, doesn't depend on grid position
        int start = pair_starts[a];
        int end   = pair_starts[a + 1];
        for (int i = start; i < end; i++) {
            int a2 = pair_indices[i];
            float dx = xyz[stride + a*3]   - xyz[stride + a2*3];
            float dy = xyz[stride + a*3+1] - xyz[stride + a2*3+1];
            float dz = xyz[stride + a*3+2] - xyz[stride + a2*3+2];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < ie_cutoff_sq) {
                float r2eff = r2 + ie_soft_delta;
                float denom = r2eff*r2eff*r2eff;
                total += (ie_vdwA[a]*ie_vdwA[a2]) / (denom*denom);
            }
        }
    }

    // Cross-SIMD reduction
    total = simd_sum(total);
    threadgroup float tg_partial[8];
    if (simd_is_first()) {
        tg_partial[simd_idx] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tid == 0 final serial sum (NOT simd_sum — uninitialized lanes)
    if (tid == 0) {
        float final = 0.0;
        for (uint i = 0; i < num_simd; i++) {
            final += tg_partial[i];
        }
        out_scores[candidate] = final;
    }
}

struct MolData {
    float ref_xyz[MAX_ATOMS][3];
    int   active_flags[MAX_ATOMS];
    int   num_atoms;
    int   num_active_atoms;
    int   num_torsions;
    float com_x, com_y, com_z;   /* precomputed center of mass (cached from CPU) */
    float trans_step;
    float rot_step;
    float tors_step;
    float torsion_scale_factors[MAX_TORSIONS];
    int   torsion_a1[MAX_TORSIONS];
    int   torsion_a2[MAX_TORSIONS];
    int   torsion_a3[MAX_TORSIONS];
    int   torsion_a4[MAX_TORSIONS];
    int   child_idx[MAX_TORSIONS][MAX_ATOMS];
    int   child_cnt[MAX_TORSIONS];
};

/* NOTE: 1-voxel safety margin (gx > 1.0) is more conservative than
 * CPU scoring, which may handle boundary atoms differently.  This
 * can cause GPU minimizations to abort (via inside-grid sentinel)
 * where CPU would not, potentially contributing to score divergence
 * on deep growth trees (observed 1J4H Δ=1.19 vs typical ~0.1).
 * If score differences exceed expectations, remove the margin and
 * match CPU semantics exactly. */
static bool inside_grid(constant GridParams &p, float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    return (gx > 1.0 && gx < (float)(p.span_x - 1) &&
            gy > 1.0 && gy < (float)(p.span_y - 1) &&
            gz > 1.0 && gz < (float)(p.span_z - 1));
}

static float trilinear(texture3d<float> grid, constant GridParams &p,
                        float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
        gx >= (float)(p.span_x - 1) ||
        gy >= (float)(p.span_y - 1) ||
        gz >= (float)(p.span_z - 1))
        return 0.0;
    float3 norm = float3(gx / (float)(p.span_x - 1),
                          gy / (float)(p.span_y - 1),
                          gz / (float)(p.span_z - 1));
    return grid.sample(grid_sampler, norm).x;
}

static void quat_to_rmat(thread float3x3 &m, float3 qin) {
    float qn, qn2;
    float q[3] = {qin.x, qin.y, qin.z};
    float q2[3];
    float sum2 = 0.0;
    for (int i = 0; i < 3; i++) {
        if (q[i] > 1.0)      q[i] = fmod(q[i] + 1.0, 2.0) - 1.0;
        else if (q[i] < -1.0) q[i] = fmod(q[i] - 1.0, 2.0) + 1.0;
        q2[i] = q[i] * q[i];
        sum2 += q2[i];
    }
    if (sum2 < 1.0) { qn2 = 1.0 - sum2; qn = sqrt(qn2); }
    else if (sum2 == 1.0) { qn = qn2 = 0.0; }
    else {
        for (int i = 0; i < 3; i++) q2[i] /= sum2;
        float sum = sqrt(sum2);
        for (int i = 0; i < 3; i++) q[i] /= sum;
        qn = qn2 = 0.0;
    }
    m[0][0] = qn2 + q2[0] - q2[1] - q2[2];
    float t1 = q[0]*q[1], t2 = qn*q[2];
    m[0][1] = 2.0*(t1 + t2);
    t1 = q[0]*q[2]; t2 = qn*q[1];
    m[0][2] = 2.0*(t1 - t2);
    t1 = q[0]*q[1]; t2 = qn*q[2];
    m[1][0] = 2.0*(t1 - t2);
    m[1][1] = qn2 - q2[0] + q2[1] - q2[2];
    t1 = q[1]*q[2]; t2 = qn*q[0];
    m[1][2] = 2.0*(t1 + t2);
    t1 = q[0]*q[2]; t2 = qn*q[1];
    m[2][0] = 2.0*(t1 + t2);
    t1 = q[1]*q[2]; t2 = qn*q[0];
    m[2][1] = 2.0*(t1 - t2);
    m[2][2] = qn2 - q2[0] - q2[1] + q2[2];
}

/* compute_com was removed in favor of precomputing the center of mass
 * once on the CPU and storing it in MolData.com_{x,y,z}.  This avoids
 * recomputing the same COM (from constant ref_xyz + active_flags) on
 * every dof_to_xyz call — thousands of times per conformation. */
static void rigid_transform(device float *xyz, int num_atoms,
                             float3 com, float3 trans, thread float3x3 &rmat) {
    for (int i = 0; i < num_atoms; i++) {
        float tx = xyz[i*3+0] - com.x;
        float ty = xyz[i*3+1] - com.y;
        float tz = xyz[i*3+2] - com.z;
        float nx = rmat[0][0]*tx + rmat[1][0]*ty + rmat[2][0]*tz + com.x + trans.x;
        float ny = rmat[0][1]*tx + rmat[1][1]*ty + rmat[2][1]*tz + com.y + trans.y;
        float nz = rmat[0][2]*tx + rmat[1][2]*ty + rmat[2][2]*tz + com.z + trans.z;
        xyz[i*3+0] = nx; xyz[i*3+1] = ny; xyz[i*3+2] = nz;
    }
}

static float dihedral_degrees(device const float *xyz, int a1, int a2, int a3, int a4) {
    float3 v1 = {xyz[a1*3], xyz[a1*3+1], xyz[a1*3+2]};
    float3 v2 = {xyz[a2*3], xyz[a2*3+1], xyz[a2*3+2]};
    float3 v3 = {xyz[a3*3], xyz[a3*3+1], xyz[a3*3+2]};
    float3 v4 = {xyz[a4*3], xyz[a4*3+1], xyz[a4*3+2]};
    float3 b1 = {v1.x-v2.x, v1.y-v2.y, v1.z-v2.z};
    float3 b2 = {v2.x-v3.x, v2.y-v3.y, v2.z-v3.z};
    float3 b3 = {v3.x-v4.x, v3.y-v4.y, v3.z-v4.z};
    float3 c1 = {b1.y*b2.z - b1.z*b2.y, -(b1.x*b2.z - b1.z*b2.x), b1.x*b2.y - b1.y*b2.x};
    float3 c2 = {b2.y*b3.z - b2.z*b3.y, -(b2.x*b3.z - b2.z*b3.x), b2.x*b3.y - b2.y*b3.x};
    float3 c3 = {c1.y*c2.z - c1.z*c2.y, -(c1.x*c2.z - c1.z*c2.x), c1.x*c2.y - c1.y*c2.x};
    float c1len = sqrt(c1.x*c1.x + c1.y*c1.y + c1.z*c1.z);
    float c2len = sqrt(c2.x*c2.x + c2.y*c2.y + c2.z*c2.z);
    if (c1len*c2len < 0.001) return 0.0;
    float dot = (c1.x*c2.x + c1.y*c2.y + c1.z*c2.z) / (c1len*c2len);
    if (dot < -0.999999) dot = -0.999999;
    if (dot > 0.999999) dot = 0.999999;
    float angle = acos(dot) * 180.0 / PI;
    float b2c3 = b2.x*c3.x + b2.y*c3.y + b2.z*c3.z;
    if (b2c3 > 0.0) angle = -angle;
    return angle;
}

static void apply_torsion(device float *xyz, int a1, int a2, int a3, int a4,
                           device const int *children, int child_cnt, float new_angle_rad) {
    float cur_deg = dihedral_degrees(xyz, a1, a2, a3, a4);
    float cur_rad = cur_deg * PI / 180.0;
    float rotang = new_angle_rad - cur_rad;
    if (fabs(rotang) < 1e-10) return;
    float ax = xyz[a3*3] - xyz[a2*3];
    float ay = xyz[a3*3+1] - xyz[a2*3+1];
    float az = xyz[a3*3+2] - xyz[a2*3+2];
    float mag = sqrt(ax*ax + ay*ay + az*az);
    if (mag < 1e-10) return;
    ax /= mag; ay /= mag; az /= mag;
    float cx = xyz[a2*3], cy = xyz[a2*3+1], cz = xyz[a2*3+2];
    float sn = sin(rotang), cs = cos(rotang), omc = 1.0 - cs;
    for (int i = 0; i < child_cnt; i++) {
        int j = children[i];
        float px = xyz[j*3]-cx, py = xyz[j*3+1]-cy, pz = xyz[j*3+2]-cz;
        float kxv_x = ay*pz - az*py, kxv_y = az*px - ax*pz, kxv_z = ax*py - ay*px;
        float kdotv = ax*px + ay*py + az*pz;
        xyz[j*3]   = cx + px*cs + kxv_x*sn + ax*kdotv*omc;
        xyz[j*3+1] = cy + py*cs + kxv_y*sn + ay*kdotv*omc;
        xyz[j*3+2] = cz + pz*cs + kxv_z*sn + az*kdotv*omc;
    }
}

static void dof_to_xyz(threadgroup const float *dof, device const MolData &mol,
                        device float *xyz) {
    int na = mol.num_atoms;
    for (int i = 0; i < na; i++) {
        xyz[i*3] = mol.ref_xyz[i][0];
        xyz[i*3+1] = mol.ref_xyz[i][1];
        xyz[i*3+2] = mol.ref_xyz[i][2];
    }
    float3 trans = {dof[0] * mol.trans_step,
                    dof[1] * mol.trans_step,
                    dof[2] * mol.trans_step};
    float3 quat  = {dof[3] * mol.rot_step,
                    dof[4] * mol.rot_step,
                    dof[5] * mol.rot_step};
    float3x3 rmat;
    quat_to_rmat(rmat, quat);
    float3 com = {mol.com_x, mol.com_y, mol.com_z};
    rigid_transform(xyz, na, com, trans, rmat);
    for (int t = 0; t < mol.num_torsions; t++) {
        float delta_deg = dof[6 + t] * mol.tors_step;
        if (fabs(delta_deg) < 1e-10) continue;
        float cur_deg = dihedral_degrees(xyz, mol.torsion_a1[t], mol.torsion_a2[t],
                                          mol.torsion_a3[t], mol.torsion_a4[t]);
        float new_rad = (PI/180.0)*(cur_deg + delta_deg);
        apply_torsion(xyz, mol.torsion_a1[t], mol.torsion_a2[t], mol.torsion_a3[t],
                       mol.torsion_a4[t], mol.child_idx[t], mol.child_cnt[t], new_rad);
    }
}

/* Grid scoring only (called by thread 0 — reads from xyz device buffer) */
static float score_grid(
    device const float *xyz, int num_atoms,
    texture3d<float> grid_avdw, texture3d<float> grid_bvdw,
    texture3d<float> grid_es,
    device const float *vdwA, device const float *vdwB,
    device const float *charges,
    constant GridParams &gp,
    device const int *active_flags) {
    float grid_score = 0.0;
    for (int a = 0; a < num_atoms; a++) {
        if (!active_flags[a]) continue;
        float x = xyz[a*3], y = xyz[a*3+1], z = xyz[a*3+2];
        if (!inside_grid(gp, x, y, z)) return -3.40282347e+38;
        float vdw = trilinear(grid_avdw, gp, x, y, z);
        float bvdw = trilinear(grid_bvdw, gp, x, y, z);
        float es = trilinear(grid_es, gp, x, y, z);
        grid_score += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
    }
    return grid_score;
}

/* Parallel IE scoring — called by ALL threads. Each thread processes
 * (nnp / tg_size) pairs via strided loop, then SIMD-group reduction via
 * simd_sum + one write per SIMD group. This eliminates TGM bank conflicts
 * (256 threads → 8 SIMD groups → 8 writes instead of 256).
 */
static float ie_score_parallel(
    device const float *xyz, int nnp,
    device const float *ie_vdwA, device const int *nb_int,
    constant IEParams &iep,
    /* C5: NB pairs pre-filtered to active-only on CPU — no active_flags check needed */
    uint tid, int tg_size,
    threadgroup float *tg_sums,
    int sg_size) {
    float partial = 0.0;
    if (nnp > 0) {
        for (int p = tid; p < nnp; p += tg_size) {
            int a1 = nb_int[p*2], a2 = nb_int[p*2+1];
            /* C5: NB pairs pre-filtered to active-only on CPU */
            float dx = xyz[a1*3]-xyz[a2*3];
            float dy = xyz[a1*3+1]-xyz[a2*3+1];
            float dz = xyz[a1*3+2]-xyz[a2*3+2];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < iep.cutoff_sq) {
                float r2eff = r2 + iep.soft_delta;
                float denom = r2eff*r2eff*r2eff;
                partial += (ie_vdwA[a1]*ie_vdwA[a2]) / (denom*denom);
            }
        }
    }
    /* SIMD-group reduction — 1 write per SIMD group instead of 1 per thread */
    float simd_val = simd_sum(partial);
    if (simd_is_first()) {
        tg_sums[tid / sg_size] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0;
        int num_sg = tg_size / sg_size;
        for (int s = 0; s < num_sg; s++) total += tg_sums[s];
        return total;
    }
    return 0.0;
}

/* Single-thread simplex kernel. Each dispatch handles all iterations
 * internally (avoids CPU-GPU sync per iteration).
 * Uses 3D texture samplers for hardware-accelerated trilinear interpolation
 * on the precomputed VDW and ES grids.
 */
)shader";



/* ================================================================== */
/*  Static state                                                       */
/* ================================================================== */

static id<MTLDevice>               g_device     = nil;
static id<MTLCommandQueue>         g_cmdq       = nil;
static id<MTLComputePipelineState> g_pso        = nil;  /* grid-only kernel */
static id<MTLComputePipelineState> g_pso_ie    = nil;  /* grid+IE kernel */
static id<MTLComputePipelineState> g_pso_atom_parallel = nil; /* atom-parallel scoring kernel */

/* GPU buffers (shared memory) */
static id<MTLBuffer> g_buf_grid_avdw  = nil;
static id<MTLBuffer> g_buf_grid_bvdw  = nil;
static id<MTLBuffer> g_buf_grid_es    = nil;
/* Grid data also stored as 3D textures for hardware trilinear filtering */
static id<MTLTexture> g_tex_grid_avdw  = nil;
static id<MTLTexture> g_tex_grid_bvdw  = nil;
static id<MTLTexture> g_tex_grid_es    = nil;
static id<MTLBuffer> g_buf_vdwA       = nil;
static id<MTLBuffer> g_buf_vdwB       = nil;
static id<MTLBuffer> g_buf_charges    = nil;
static id<MTLBuffer> g_buf_ie_vdwA      = nil;
static id<MTLBuffer> g_buf_nb_int       = nil;
static id<MTLBuffer> g_buf_pair_starts  = nil;  /* int pair_starts[num_atoms + 1] */
static id<MTLBuffer> g_buf_pair_indices = nil;  /* int pair_indices[total_pairs] */
static id<MTLBuffer> g_buf_xyz        = nil;
static id<MTLBuffer> g_buf_scores     = nil;
/* Persistent constant buffers (allocated once, reused per dispatch) */
static id<MTLBuffer> g_buf_active_flags = nil;  /* int active_flags[num_atoms] */
static id<MTLBuffer> g_buf_params     = nil;  /* DockGridParams */
static id<MTLBuffer> g_buf_natoms     = nil;  /* int num_atoms */
static id<MTLBuffer> g_buf_iep        = nil;  /* IEParams */
static id<MTLBuffer> g_buf_nnp        = nil;  /* int num_nb_pairs */
static id<MTLBuffer> g_buf_tg_header  = nil;  /* 2 ints: tg_size, num_simd_groups */

/* Cache for redundant-ligand skip (B2 optimization) */
static int  g_set_ligand_num_atoms   = -1;
static int  g_set_ie_num_nb_pairs    = -1;

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

/* ---- Profiling helpers (Metal-only, for dev use) ---- */
static uint64_t prof_tic(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC);
}
static void prof_toc(const char *label, uint64_t t0) {
#if 0
    uint64_t dt = clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0;
    NSLog(@"GPU-DOCK: [PROF] %s = %llu us", label, dt / 1000);
#endif
}
/* Dispatch counters */
static int  prof_dispatch_count  = 0;
static int  prof_total_conformers = 0;

/* Rolling GPU wait-time buffer for dock_gpu_monitor() */
#define GPU_WAIT_BUF_SIZE 20
static uint64_t g_gpu_wait_us[GPU_WAIT_BUF_SIZE] = {0};
static int g_gpu_wait_idx  = 0;   /* next slot index (mod buffer size) */
static int g_gpu_wait_total = 0;  /* total measurements taken */
/* ----------------------------------------------------- */

static id<MTLBuffer> alloc_buffer(NSUInteger size, const char* label)
{
    /* C6: Use Managed (not Shared) on discrete-GPU Macs for read-only buffers.
     * Apple Silicon unified memory: choose Shared (hardware coherency is free).
     * Discrete GPUs: Managed avoids CPU↔GPU coherence traffic on every commit. */
    MTLResourceOptions storage = MTLResourceStorageModeShared;
    if (![g_device hasUnifiedMemory]) {
        storage = MTLResourceStorageModeManaged;
    }
    id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                              options:storage];
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
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        if (@available(macOS 13.0, *)) {
            opts.optimizationLevel = MTLLibraryOptimizationLevelDefault;
        }
        id<MTLLibrary> lib = [g_device newLibraryWithSource:
                               [NSString stringWithUTF8String:shader_src_all]
                                                     options:opts
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

        /* Compile atom-parallel batch scoring kernel */
        id<MTLFunction> func_ap = [lib newFunctionWithName:@"score_batch_kernel_atom_parallel"];
        if (!func_ap) {
            fprintf(stderr, "GPU-DOCK: kernel 'score_batch_kernel_atom_parallel' not found\n");
            dock_gpu_cleanup();
            return 0;
        }
        g_pso_atom_parallel = [g_device newComputePipelineStateWithFunction:func_ap error:&err];
        if (!g_pso_atom_parallel) {
            fprintf(stderr, "GPU-DOCK: atom-parallel PSO creation failed: %s\n",
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

        /* Create 3D textures for hardware-accelerated trilinear interpolation */
        {
            MTLTextureDescriptor *tdesc = [[MTLTextureDescriptor alloc] init];
            tdesc.textureType = MTLTextureType3D;
            tdesc.pixelFormat = MTLPixelFormatR32Float;
            tdesc.width  = (NSUInteger)span_x;
            tdesc.height = (NSUInteger)span_y;
            tdesc.depth  = (NSUInteger)span_z;
            tdesc.mipmapLevelCount = 1;
            tdesc.usage = MTLTextureUsageShaderRead;
            /* C6: Shared storage — Private would be ideal but replaceRegion requires
             * Shared on some Metal configurations.  alloc_buffer (below) uses Managed
             * on discrete GPUs where it matters more. */
            tdesc.storageMode = MTLStorageModeShared;

            g_tex_grid_avdw = [g_device newTextureWithDescriptor:tdesc];
            g_tex_grid_bvdw = [g_device newTextureWithDescriptor:tdesc];
            g_tex_grid_es   = [g_device newTextureWithDescriptor:tdesc];

            if (!g_tex_grid_avdw || !g_tex_grid_bvdw || !g_tex_grid_es) {
                fprintf(stderr, "GPU-DOCK: texture allocation failed\n");
                dock_gpu_cleanup();
                return 0;
            }

            MTLRegion region = MTLRegionMake3D(0, 0, 0, (NSUInteger)span_x,
                                                (NSUInteger)span_y,
                                                (NSUInteger)span_z);
            NSUInteger bpr = (NSUInteger)span_x * sizeof(float);
            NSUInteger bpi = (NSUInteger)span_x * (NSUInteger)span_y * sizeof(float);

            [g_tex_grid_avdw replaceRegion:region mipmapLevel:0 slice:0 withBytes:avdw
                                bytesPerRow:bpr bytesPerImage:bpi];
            [g_tex_grid_bvdw replaceRegion:region mipmapLevel:0 slice:0 withBytes:bvdw
                                bytesPerRow:bpr bytesPerImage:bpi];
            [g_tex_grid_es   replaceRegion:region mipmapLevel:0 slice:0 withBytes:es
                                bytesPerRow:bpr bytesPerImage:bpi];
        }

        /* Allocate ligand parameter buffers (populated by dock_gpu_set_ligand later) */
        g_buf_vdwA    = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "vdwA");
        g_buf_vdwB    = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "vdwB");
        g_buf_charges = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "charges");
        g_buf_ie_vdwA = alloc_buffer(sizeof(float) * GPU_MAX_ATOMS, "ie_vdwA");
        g_buf_nb_int  = alloc_buffer(sizeof(int) * GPU_MAX_NB_PAIRS * 2, "nb_int");
        g_buf_pair_starts  = alloc_buffer(sizeof(int) * (GPU_MAX_ATOMS + 1), "pair_starts");
        g_buf_pair_indices = alloc_buffer(sizeof(int) * GPU_MAX_NB_PAIRS, "pair_indices");

        /* Persistent constants buffers */
        g_buf_params = alloc_buffer(sizeof(DockGridParams), "params");
        g_buf_natoms = alloc_buffer(sizeof(int), "natoms");
        g_buf_iep    = alloc_buffer(32, "iep"); /* holds IEParams struct */
        g_buf_nnp    = alloc_buffer(sizeof(int), "nnp");
        g_buf_active_flags = alloc_buffer(sizeof(int) * GPU_MAX_ATOMS, "active_flags");
        g_buf_tg_header   = alloc_buffer(sizeof(int) * 2, "tg_header");

        /* Allocate per-batch buffers */
        g_buf_xyz    = alloc_buffer(sizeof(float) * 3 * GPU_MAX_ATOMS * GPU_MAX_POSES, "xyz");
        g_buf_scores = alloc_buffer(sizeof(float) * GPU_MAX_POSES, "scores");

        if (!g_buf_vdwA || !g_buf_vdwB || !g_buf_charges ||
            !g_buf_ie_vdwA || !g_buf_nb_int ||
            !g_buf_pair_starts || !g_buf_pair_indices ||
            !g_buf_xyz || !g_buf_scores ||
            !g_buf_params || !g_buf_natoms || !g_buf_iep || !g_buf_nnp ||
            !g_buf_active_flags || !g_buf_tg_header) {
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

    /* R3: No skip guard here.  The caller (simplex.cpp) already gates this
     * call behind a 64-bit content signature, so we only get called when the
     * ligand actually changed.  The old g_set_ligand_num_atoms guard silently
     * skipped re-uploads for different ligands with the same atom count,
     * producing wrong scores in multi-ligand workflows. */
    @autoreleasepool {
        size_t bytes = sizeof(float) * (size_t)num_atoms;
        memcpy([g_buf_vdwA contents], vdwA, bytes);
        memcpy([g_buf_vdwB contents], vdwB, bytes);
        memcpy([g_buf_charges contents], charges, bytes);
        g_num_atoms = num_atoms;
        g_set_ligand_num_atoms = num_atoms;
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

    /* R3: No skip guard here.  The caller (simplex.cpp) gates this call
     * behind the same content signature that covers vdwA/vdwB/charges, so
     * we only get called when the ligand actually changed.  The old
     * g_set_ie_num_nb_pairs guard could skip a needed re-upload when two
     * different ligands happened to have the same NB-pair count. */
    @autoreleasepool {
        size_t ie_bytes = sizeof(float) * (size_t)g_num_atoms;
        memcpy([g_buf_ie_vdwA contents], ie_vdwA, ie_bytes);

        size_t nb_bytes = sizeof(int) * (size_t)num_nb_pairs * 2;
        memcpy([g_buf_nb_int contents], nb_int_pairs, nb_bytes);

        /* Build per-atom pair lists: pair_starts[num_atoms+1] + pair_indices[total_pairs]
         * from the flat nb_int[] array. pair_starts[a] = start offset in pair_indices
         * for atom a's pairs. pair_starts[a+1] - pair_starts[a] = count.
         * This eliminates the O(num_atoms × num_pairs) stride loop in IE scoring. */
        {
            int na = g_num_atoms;
            int *counts = (int *)calloc(na, sizeof(int));
            int *starts = (int *)[g_buf_pair_starts contents];

            /* Phase 1: count pairs per atom */
            for (int p = 0; p < num_nb_pairs; p++) {
                int a1 = nb_int_pairs[p*2];
                if (a1 >= 0 && a1 < na) counts[a1]++;
            }

            /* Phase 2: prefix sum into starts + save offsets for insertion */
            int *offsets = (int *)malloc(na * sizeof(int));
            int total = 0;
            for (int a = 0; a < na; a++) {
                starts[a] = total;
                offsets[a] = total;
                total += counts[a];
            }
            starts[na] = total;  /* sentinel end */
            free(counts);

            /* Phase 3: fill pair_indices using saved offsets */
            int *pair_indices = (int *)[g_buf_pair_indices contents];
            for (int p = 0; p < num_nb_pairs; p++) {
                int a1 = nb_int_pairs[p*2];
                int a2 = nb_int_pairs[p*2+1];
                if (a1 >= 0 && a1 < na) {
                    pair_indices[offsets[a1]++] = a2;
                }
            }
            free(offsets);

            NSLog(@"GPU-DOCK: per-atom pair lists built: %d total pairs for %d atoms",
                  total, na);
        }

        g_num_nb_pairs   = num_nb_pairs;
        g_set_ie_num_nb_pairs = num_nb_pairs;
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
        /* Grid data via 3D textures for hardware trilinear filtering */
        [enc setTexture:g_tex_grid_avdw  atIndex:0];
        [enc setTexture:g_tex_grid_bvdw  atIndex:1];
        [enc setTexture:g_tex_grid_es    atIndex:2];
        [enc setBuffer:g_buf_vdwA       offset:0 atIndex:1];
        [enc setBuffer:g_buf_vdwB       offset:0 atIndex:2];
        [enc setBuffer:g_buf_charges    offset:0 atIndex:3];

        [enc setBuffer:g_buf_params offset:0 atIndex:4];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms offset:0 atIndex:5];

        [enc setBuffer:g_buf_scores offset:0 atIndex:6];
        [enc setBuffer:g_buf_active_flags offset:0 atIndex:7];

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
                                  const int *active_flags, float *out_scores)
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

        /* Upload active_flags if provided */
        if (active_flags) {
            memcpy([g_buf_active_flags contents], active_flags, sizeof(int) * (size_t)num_atoms);
        }

        id<MTLCommandBuffer>  cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"DockBatchScoreAP";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"BatchScoreAPKernel";

        [enc setComputePipelineState:g_pso_atom_parallel];

        /* Bind buffers — must match score_batch_kernel_atom_parallel layout */
        [enc setBuffer:g_buf_xyz         offset:0 atIndex:0];
        /* Grid textures (hardware trilinear filtering) */
        [enc setTexture:g_tex_grid_avdw   atIndex:0];
        [enc setTexture:g_tex_grid_bvdw   atIndex:1];
        [enc setTexture:g_tex_grid_es     atIndex:2];
        [enc setBuffer:g_buf_vdwA        offset:0 atIndex:1];
        [enc setBuffer:g_buf_vdwB        offset:0 atIndex:2];
        [enc setBuffer:g_buf_charges     offset:0 atIndex:3];
        [enc setBuffer:g_buf_params      offset:0 atIndex:4];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms      offset:0 atIndex:5];
        [enc setBuffer:g_buf_active_flags offset:0 atIndex:6];
        [enc setBuffer:g_buf_ie_vdwA     offset:0 atIndex:7];
        [enc setBuffer:g_buf_pair_starts offset:0 atIndex:8];
        [enc setBuffer:g_buf_pair_indices offset:0 atIndex:9];
        /* IE constants via inline setBytes (no buffer allocation needed) */
        [enc setBytes:&g_num_nb_pairs  length:sizeof(int)   atIndex:10];
        [enc setBytes:&g_ie_soft_delta length:sizeof(float)  atIndex:11];
        [enc setBytes:&g_ie_cutoff_sq  length:sizeof(float)  atIndex:12];
        [enc setBuffer:g_buf_scores      offset:0 atIndex:13];

        /* Atom-parallel dispatch: 1 threadgroup per pose, 64 threads each */
        NSUInteger tg = 64;
        MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);
        MTLSize threadsPerGrid  = MTLSizeMake((NSUInteger)num_poses * tg, 1, 1);

        [enc dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.error) {
            NSLog(@"GPU-DOCK: command buffer error (AP): %@", cmdbuf.error);
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
        NSLog(@"GPU-DOCK: [PROF] total dispatches=%d conformers=%d", 
              prof_dispatch_count, prof_total_conformers);
        if (prof_dispatch_count > 0)
            NSLog(@"GPU-DOCK: [PROF] avg batch size=%.1f",
                  (float)prof_total_conformers / (float)prof_dispatch_count);
        g_buf_grid_avdw  = nil;
        g_buf_grid_bvdw  = nil;
        g_buf_grid_es    = nil;
        g_tex_grid_avdw  = nil;
        g_tex_grid_bvdw  = nil;
        g_tex_grid_es    = nil;
        g_buf_vdwA       = nil;
        g_buf_vdwB       = nil;
        g_buf_charges    = nil;
        g_buf_ie_vdwA      = nil;
        g_buf_nb_int       = nil;
        g_buf_pair_starts  = nil;
        g_buf_pair_indices = nil;
        g_buf_active_flags = nil;
        g_buf_xyz        = nil;
        g_buf_scores     = nil;
        g_buf_tg_header  = nil;
        g_pso               = nil;
        g_pso_ie            = nil;
        g_pso_atom_parallel = nil;
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





/* ================================================================== */
/*  dock_gpu_recommended_batch_size — Query GPU core count via IOKit  */
/* ================================================================== */

/* Query the exact number of GPU compute cores from IOKit using the
   Metal device's registryID.  This avoids hardcoding chip names and
   works on all current/future Apple Silicon GPUs. */
static int dock_gpu_get_core_count(void)
{
    if (!g_device) return 0;

    int core_count = 0;
    io_service_t gpu_service = IOServiceGetMatchingService(
        kIOMainPortDefault,
        IORegistryEntryIDMatching(g_device.registryID));

    if (gpu_service) {
        CFNumberRef numberRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            gpu_service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (numberRef) {
            CFNumberGetValue(numberRef, kCFNumberSInt32Type, &core_count);
            CFRelease(numberRef);
        }
        IOObjectRelease(gpu_service);
    }

    return core_count;
}


int dock_gpu_recommended_batch_size(void)
{
    if (!g_device) return 32;  /* safe default for unknown or inactive */

    int cores = dock_gpu_get_core_count();
    if (cores <= 0) return 32;  /* IOKit query failed — fallback */

    /* Heuristic: each CU can sustain ~4 threadgroups for good occupancy.
       Each threadgroup handles one conformer.  Cap at GPU_MAX_POSES/4
       to stay within the per-batch xyz buffer allocation. */
    int size = cores * 4;
    int cap  = GPU_MAX_POSES / 4;
    if (size > cap) size = cap;

    return size;
}


/* ================================================================== */
/*  dock_gpu_monitor — Report thermal state + GPU dispatch time trend */
/* ================================================================== */

void dock_gpu_monitor(int layer, int segment, int total_segments)
{
    if (!g_initialized) return;

    /* Thermal state (Apple API, no special permissions) */
    static const char *thermal_label[] = {
        "Nominal", "Fair", "Serious", "Critical"
    };
    NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
    const char *ts_str = (ts >= 0 && ts <= 3) ? thermal_label[ts] : "?";

    /* Rolling average over GPU_WAIT_BUF_SIZE measurements */
    int n = (g_gpu_wait_total < GPU_WAIT_BUF_SIZE) ? g_gpu_wait_total : GPU_WAIT_BUF_SIZE;
    double avg_us = 0.0;
    if (n > 0) {
        int base = g_gpu_wait_idx;
        for (int k = 0; k < n; k++) {
            int idx = (base - 1 - k + GPU_WAIT_BUF_SIZE) % GPU_WAIT_BUF_SIZE;
            avg_us += (double)g_gpu_wait_us[idx];
        }
        avg_us /= (double)n;
    }

    /* Print to stdout (captured in dock output file with -v) */
    std::cout << "GPU: thermal=" << ts_str
              << " disp=" << g_gpu_wait_total
              << " avg_wait_ms=" << (avg_us / 1000.0) << std::endl;
}


} /* extern "C" */
