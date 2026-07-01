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
#define GPU_MAX_TORSIONS   50    /* Max rotatable bonds */
#define GPU_DOF_MAX        56    /* 6 + max torsions */


/* ================================================================== */
/*  Embedded Metal Shader Source — Both kernels in one library         */
/* ================================================================== */

static const char* shader_src = \
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constexpr sampler grid_sampler(filter::linear, address::clamp_to_edge);\n"
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
"/* Trilinear interpolation via hardware 3D texture sampling */\n"
"static float trilinear(texture3d<float> grid,\n"
"                       constant GridParams &p,\n"
"                       float x, float y, float z);\n"
"\n"
"kernel void batch_score_kernel(\n"
"    device const float*    xyz          [[buffer(0)]],\n"
"    texture3d<float>       grid_avdw    [[texture(0)]],\n"
"    texture3d<float>       grid_bvdw    [[texture(1)]],\n"
"    texture3d<float>       grid_es      [[texture(2)]],\n"
"    device const float*    vdwA         [[buffer(1)]],\n"
"    device const float*    vdwB         [[buffer(2)]],\n"
"    device const float*    charges      [[buffer(3)]],\n"
"    constant GridParams&   gp           [[buffer(4)]],\n"
"    constant int&          num_atoms    [[buffer(5)]],\n"
"    device float*          out_scores   [[buffer(6)]],\n"
"    device const int*     active_flags [[buffer(7)]],\n"
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
"        if (active_flags[a]) {\n"
"            score += vdwA[a] * vdw - vdwB[a] * bvdw + charges[a] * es;\n"
"        }\n"
"    }\n"
"\n"
"    out_scores[tid] = score;\n"
"}\n"
"static bool inside_grid(constant GridParams &p, float x, float y, float z)\n"
"{\n"
"    float gx = (x - p.origin_x) / p.spacing;\n"
"    float gy = (y - p.origin_y) / p.spacing;\n"
"    float gz = (z - p.origin_z) / p.spacing;\n"
"    return (gx > 1.0 && gx < (float)(p.span_x - 1) &&\n"
"            gy > 1.0 && gy < (float)(p.span_y - 1) &&\n"
"            gz > 1.0 && gz < (float)(p.span_z - 1));\n"
"}\n"
"\n"
"/* Hardware trilinear via 3D texture sampler */\n"
"static float trilinear(texture3d<float> grid,\n"
"                       constant GridParams &p,\n"
"                       float x, float y, float z)\n"
"{\n"
"    float gx = (x - p.origin_x) / p.spacing;\n"
"    float gy = (y - p.origin_y) / p.spacing;\n"
"    float gz = (z - p.origin_z) / p.spacing;\n"
"\n"
"    /* Bounds-check (keep for safety; inside_grid catches most) */\n"
"    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||\n"
"        gx >= (float)(p.span_x - 1) ||\n"
"        gy >= (float)(p.span_y - 1) ||\n"
"        gz >= (float)(p.span_z - 1))\n"
"        return 0.0;\n"
"\n"
"    float3 norm = float3(gx / (float)(p.span_x - 1),\n"
"                          gy / (float)(p.span_y - 1),\n"
"                          gz / (float)(p.span_z - 1));\n"
"    return grid.sample(grid_sampler, norm).x;\n"
"}\n"
"\n"
"kernel void batch_score_with_ie_kernel(\n"
"    device const float*    xyz            [[buffer(0)]],\n"
"    texture3d<float>       grid_avdw      [[texture(0)]],\n"
"    texture3d<float>       grid_bvdw      [[texture(1)]],\n"
"    texture3d<float>       grid_es        [[texture(2)]],\n"
"    device const float*    vdwA           [[buffer(1)]],\n"
"    device const float*    vdwB           [[buffer(2)]],\n"
"    device const float*    charges        [[buffer(3)]],\n"
"    constant GridParams&   gp             [[buffer(4)]],\n"
"    constant int&          num_atoms      [[buffer(5)]],\n"
"    device float*          out_scores     [[buffer(6)]],\n"
"    device const float*    ie_vdwA        [[buffer(7)]],\n"
"    device const int*      nb_int         [[buffer(8)]],\n"
"    constant IEParams&     iep            [[buffer(9)]],\n"
"    constant int&          num_nb_pairs   [[buffer(10)]],\n"
"    device const int*     active_flags   [[buffer(11)]],\n"
"    uint                   tid            [[thread_position_in_grid]])\n"
"{\n"
"    int atoms_per_pose = num_atoms;\n"
"    int stride = 3 * atoms_per_pose;\n"
"    int base = tid * stride;\n"
"\n"
"    float grid_score = 0.0;\n"
"    bool has_outside_atom = false;\n"
"\n"
"    /* ---- Grid score (short-circuit on outside-grid) ---- */\n"
"    for (int a = 0; a < atoms_per_pose; a++) {\n"
"        int o3 = base + a * 3;\n"
"        float x = xyz[o3];\n"
"        float y = xyz[o3 + 1];\n"
"        float z = xyz[o3 + 2];\n"
"\n"
"        if (active_flags[a] && !inside_grid(gp, x, y, z)) {\n"
"            out_scores[tid] = -3.40282347e+38;  /* -FLT_MAX */\n"
"            return;\n"
"        }\n"
"\n"
"        if (active_flags[a]) {\n"
"            float vdw = trilinear(grid_avdw, gp, x, y, z);\n"
"            float bvdw = trilinear(grid_bvdw, gp, x, y, z);\n"
"            float es = trilinear(grid_es, gp, x, y, z);\n"
"\n"
"            grid_score += vdwA[a] * vdw - vdwB[a] * bvdw + charges[a] * es;\n"
"        }\n"
"    }\n"
"\n"
"    /* ---- Internal energy ---- */\n"
"    float ie_score = 0.0;\n"
"    if (num_nb_pairs > 0) {\n"
"        for (int p = 0; p < num_nb_pairs; p++) {\n"
"            int a1 = nb_int[p * 2];\n"
"            int a2 = nb_int[p * 2 + 1];\n"
"            if (!active_flags[a1] || !active_flags[a2]) continue;\n"
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
"    out_scores[tid] = grid_score + ie_score;\n"
"}\n"
;


/* ================================================================== */
/*  Simplex shader source (separate compilation)                       */
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
kernel void simplex_iteration_kernel(
    device float*    vertex_dof    [[buffer(0)]],
    device float*    scores        [[buffer(1)]],
    device int*      state         [[buffer(2)]],
    device const MolData *mol      [[buffer(3)]],
    texture3d<float>       grid_avdw     [[texture(0)]],
    texture3d<float>       grid_bvdw     [[texture(1)]],
    texture3d<float>       grid_es       [[texture(2)]],
    device const float*    vdwA          [[buffer(4)]],
    device const float*    vdwB          [[buffer(5)]],
    device const float*    charges       [[buffer(6)]],
    constant GridParams &  gp            [[buffer(7)]],
    constant int&          num_atoms_buf [[buffer(8)]],
    device float*          xyz_dev       [[buffer(9)]],
    device const float*    ie_vdwA       [[buffer(10)]],
    device const int*      nb_int        [[buffer(11)]],
    constant IEParams&     iep           [[buffer(12)]],
    constant int&          num_nb_pairs  [[buffer(13)]],
    constant int&          tg_size       [[buffer(14)]],
    constant int&          sg_size_buf   [[buffer(15)]],
    uint tid                       [[thread_position_in_grid]])
{
    int n = state[2];       /* DOF size */
    int nverts = n + 1;     /* number of vertices */
    float alpha = 1.0, beta = 0.5, gamma = 2.0;
    int na = mol->num_atoms;
    int nnp = num_nb_pairs;
    int dof_max = (n < DOF_MAX) ? n : DOF_MAX;

    /* Threadgroup memory for cross-thread communication.
     * All threads must participate in barriers; thread 0 writes these,
     * barriers sync visibility, then all threads read to decide flow. */
    threadgroup float tg_centroid[DOF_MAX];
    threadgroup float tg_pr[DOF_MAX];
    threadgroup float tg_prr[DOF_MAX];
    threadgroup float tg_grid_refl;
    float grid_score_second;  /* per-thread local; only thread 0 uses it */
    threadgroup int tg_ilo, tg_ihi, tg_inhi;
    threadgroup int tg_widx;
    threadgroup int tg_action;   /* 0=accept, 1=expand, 2=contract, 3=shrink */
    threadgroup int tg_repl_flag;
    /* Sized at MAX_SIMPLEX_THREADS to support any SIMD width >= 1.
     * On Apple Sillcon (SIMD=32): 256/32=8 entries used.
     * On Intel/AMD with SIMD=8: 256/8=32 entries used — still safe.
     * Previously was /16 which overflowed when SIMD width < 16. */
    threadgroup float tg_ie_sums[MAX_SIMPLEX_THREADS];

    for (int iter = 0; iter < state[3]; iter++) {
        if (state[0]) return;  /* ALL threads — safe exit */
        if (tid == 0) state[1] = iter + 1;

        /* ===== Phase 0: Find best/worst, centroid, reflect DOF ===== */
        if (tid == 0) {
            tg_ilo = 0; tg_ihi = 1; tg_inhi = 0;
            if (scores[0] > scores[1]) { tg_ihi = 0; tg_inhi = 1; }
            else { tg_ihi = 1; tg_inhi = 0; }
            for (int i = 0; i < nverts; i++) {
                if (scores[i] < scores[tg_ilo]) tg_ilo = i;
                if (scores[i] > scores[tg_ihi]) { tg_inhi = tg_ihi; tg_ihi = i; }
                else if (i != tg_ihi && scores[i] > scores[tg_inhi]) tg_inhi = i;
            }
            for (int j = 0; j < dof_max; j++) tg_centroid[j] = 0.0;
            for (int i = 0; i < nverts; i++) {
                if (i != tg_ihi) {
                    for (int j = 0; j < dof_max; j++)
                        tg_centroid[j] += vertex_dof[i * dof_max + j];
                }
            }
            for (int j = 0; j < dof_max; j++) tg_centroid[j] /= (float)n;
            tg_widx = tg_ihi * dof_max;
            for (int j = 0; j < dof_max; j++)
                tg_pr[j] = tg_centroid[j] + alpha * (tg_centroid[j] - vertex_dof[tg_widx + j]);
        }

        /* ===== Scoring point 1: Reflect (grid + IE) ===== */
        if (tid == 0) {
            dof_to_xyz(tg_pr, *mol, xyz_dev);
            tg_grid_refl = score_grid(xyz_dev, na, grid_avdw, grid_bvdw, grid_es,
                                       vdwA, vdwB, charges, gp,
                                       mol->active_flags);
            if (tg_grid_refl < -1e30) state[5] = -1;
        }
        threadgroup_barrier(mem_flags::mem_device);

        float ie_refl = ie_score_parallel(xyz_dev, nnp, ie_vdwA, nb_int, iep,
                                           tid, tg_size, tg_ie_sums, sg_size_buf);

        /* ===== Decision tree — set up next action ===== */
        if (tid == 0) {
            if (state[5] < 0) {
                state[0] = 1;
                tg_action = 0;
            } else {
                float total_refl = tg_grid_refl + ie_refl;

                if (total_refl <= scores[tg_ilo]) {
                    /* --- Expand --- */
                    for (int j = 0; j < dof_max; j++)
                        tg_prr[j] = tg_centroid[j] + gamma * (tg_pr[j] - tg_centroid[j]);
                    dof_to_xyz(tg_prr, *mol, xyz_dev);
                    grid_score_second = score_grid(xyz_dev, na, grid_avdw, grid_bvdw, grid_es,
                                                 vdwA, vdwB, charges, gp,
                                                 mol->active_flags);
                    if (grid_score_second < -1e30) {
                        state[5] = -1; state[0] = 1; tg_action = 0;
                    } else {
                        tg_action = 1;
                    }
                } else if (total_refl >= scores[tg_inhi]) {
                    /* --- Contract path --- */
                    tg_repl_flag = (total_refl < scores[tg_ihi]) ? 1 : 0;
                    if (tg_repl_flag) {
                        for (int j = 0; j < dof_max; j++)
                            tg_prr[j] = tg_centroid[j] + beta * (tg_pr[j] - tg_centroid[j]);
                    } else {
                        for (int j = 0; j < dof_max; j++)
                            tg_prr[j] = tg_centroid[j] + beta * (vertex_dof[tg_widx + j] - tg_centroid[j]);
                    }
                    dof_to_xyz(tg_prr, *mol, xyz_dev);
                    grid_score_second = score_grid(xyz_dev, na, grid_avdw, grid_bvdw, grid_es,
                                                 vdwA, vdwB, charges, gp,
                                                 mol->active_flags);
                    if (grid_score_second < -1e30) {
                        state[5] = -1; state[0] = 1; tg_action = 0;
                    } else {
                        tg_action = 2;
                    }
                } else {
                    /* --- Accept reflect --- */
                    for (int j = 0; j < dof_max; j++) vertex_dof[tg_widx + j] = tg_pr[j];
                    scores[tg_ihi] = total_refl;
                    tg_action = 0;
                }
            }
        }

        /* ===== Scoring point 2 (if expand/contract) ===== */
        if (tg_action == 1 || tg_action == 2) {
            threadgroup_barrier(mem_flags::mem_device);
            float ie_second = ie_score_parallel(xyz_dev, nnp, ie_vdwA, nb_int, iep,
                                                  tid, tg_size, tg_ie_sums, sg_size_buf);

            if (tid == 0 && state[5] >= 0) {
                float total_second = grid_score_second + ie_second;
                float total_refl = tg_grid_refl + ie_refl;

                if (tg_action == 1) {
                    /* --- Expand: pick best --- */
                    if (total_second < total_refl) {
                        for (int j = 0; j < dof_max; j++) vertex_dof[tg_widx + j] = tg_prr[j];
                        scores[tg_ihi] = total_second;
                    } else {
                        for (int j = 0; j < dof_max; j++) vertex_dof[tg_widx + j] = tg_pr[j];
                        scores[tg_ihi] = total_refl;
                    }
                } else {
                    /* --- Contract: compare with current worst --- */
                    float compare_base = tg_repl_flag ? total_refl : scores[tg_ihi];

                    if (total_second < compare_base) {
                        for (int j = 0; j < dof_max; j++) vertex_dof[tg_widx + j] = tg_prr[j];
                        scores[tg_ihi] = total_second;
                        tg_repl_flag = 1;
                    } else if (tg_repl_flag) {
                        /* Reflect is better — use it */
                        for (int j = 0; j < dof_max; j++) vertex_dof[tg_widx + j] = tg_pr[j];
                        scores[tg_ihi] = total_refl;
                        tg_repl_flag = 1;
                    }

                    if (!tg_repl_flag) {
                        /* --- Shrink --- */
                        tg_action = 3;
                    } else {
                        tg_action = 0;  /* done */
                    }
                }
            }
        }

        /* ===== Shrink loop (if contract failed) ===== */
        if (tg_action == 3) {
            for (int si = 0; si < nverts; si++) {
                if (si == tg_ilo) continue;  /* skip best vertex — unchanged */
                if (tid == 0) {
                    int wi = si * dof_max;
                    for (int j = 0; j < dof_max; j++) {
                        float old = vertex_dof[wi + j];
                        vertex_dof[wi + j] = 0.5 * (old + vertex_dof[tg_ilo * dof_max + j]);
                    }
                    for (int j = 0; j < dof_max; j++) tg_pr[j] = vertex_dof[wi + j];
                    dof_to_xyz(tg_pr, *mol, xyz_dev);
                    grid_score_second = score_grid(xyz_dev, na, grid_avdw, grid_bvdw, grid_es,
                                                 vdwA, vdwB, charges, gp,
                                                 mol->active_flags);
                    if (grid_score_second < -1e30) { state[5] = -1; state[0] = 1; }
                }
                threadgroup_barrier(mem_flags::mem_device);
                float ie_s = ie_score_parallel(xyz_dev, nnp, ie_vdwA, nb_int, iep,
                                                tid, tg_size, tg_ie_sums, sg_size_buf);
                if (tid == 0 && state[5] >= 0) {
                    scores[si] = grid_score_second + ie_s;
                }
            }
            tg_action = 0;  /* shrink complete */
        }

        /* ===== Convergence check ===== */
        if (tid == 0 && state[5] >= 0) {
            int ilo = 0, ihi = 0;
            for (int i = 0; i < nverts; i++) {
                if (scores[i] < scores[ilo]) ilo = i;
                if (scores[i] > scores[ihi]) ihi = i;
            }
            float diff = fabs(scores[ihi] - scores[ilo]);
            float converge_tol = as_type<float>(state[4]);
            if (converge_tol < 1e-10) converge_tol = 0.001;
            state[0] = (diff < converge_tol) ? 1 : 0;
        }
    }
}
)shader";



/* ================================================================== */
/*  Static state                                                       */
/* ================================================================== */

static id<MTLDevice>               g_device     = nil;
static id<MTLCommandQueue>         g_cmdq       = nil;
static id<MTLComputePipelineState> g_pso        = nil;  /* grid-only kernel */
static id<MTLComputePipelineState> g_pso_ie    = nil;  /* grid+IE kernel */
static id<MTLComputePipelineState> g_pso_simplex = nil; /* simplex kernel */
#define MAX_SIMPLEX_THREADS_C 256
static int g_simplex_threads         = 128;  /* threads per dispatch for parallel IE */
static int g_simplex_simd_width      = 32;   /* SIMD group width (device-dependent) */

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
static id<MTLBuffer> g_buf_ie_vdwA    = nil;
static id<MTLBuffer> g_buf_nb_int     = nil;
static id<MTLBuffer> g_buf_xyz        = nil;
static id<MTLBuffer> g_buf_scores     = nil;
/* Persistent constant buffers (allocated once, reused per dispatch) */
static id<MTLBuffer> g_buf_active_flags = nil;  /* int active_flags[num_atoms] */
static id<MTLBuffer> g_buf_params     = nil;  /* DockGridParams */
static id<MTLBuffer> g_buf_natoms     = nil;  /* int num_atoms */
static id<MTLBuffer> g_buf_iep        = nil;  /* IEParams */
static id<MTLBuffer> g_buf_nnp        = nil;  /* int num_nb_pairs */
static id<MTLBuffer> g_buf_tg_header  = nil;  /* 2 ints: tg_size, num_simd_groups */

/* Simplex-specific buffers */
static id<MTLBuffer> g_buf_vertex_dof = nil;
static id<MTLBuffer> g_buf_simplex_state = nil;
static id<MTLBuffer> g_buf_mol_data    = nil;

/* Cached simplex params */
static int  g_simplex_ready = 0;

/* Cache for redundant-ligand skip (B2 optimization) */
static int  g_set_ligand_num_atoms   = -1;
static int  g_set_ie_num_nb_pairs    = -1;
static int  g_buf_simplex_allocated = 0;

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

        /* Get simplex kernel function from the same library (all kernels unified) */
        id<MTLFunction> func_simplex = [lib newFunctionWithName:@"simplex_iteration_kernel"];
        if (!func_simplex) {
            fprintf(stderr, "GPU-DOCK: kernel 'simplex_iteration_kernel' not found\n");
            dock_gpu_cleanup();
            return 0;
        }
        g_pso_simplex = [g_device newComputePipelineStateWithFunction:func_simplex error:&err];
        if (!g_pso_simplex) {
            fprintf(stderr, "GPU-DOCK: simplex PSO creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            dock_gpu_cleanup();
            return 0;
        }

        /* Query max threadgroup size for parallel IE reduction */
        {
            NSUInteger max_tg = [g_pso_simplex maxTotalThreadsPerThreadgroup];
            NSUInteger sw = [g_pso_simplex threadExecutionWidth];
            g_simplex_threads = (int)MIN(max_tg, (NSUInteger)MAX_SIMPLEX_THREADS_C);
            /* Round down to multiple of simd width */
            g_simplex_threads = (g_simplex_threads / (int)sw) * (int)sw;
            g_simplex_simd_width = (int)sw;
            fprintf(stderr, "GPU-DOCK: simplex threads=%d (max_tg=%lu, simd=%lu)\n",
                    g_simplex_threads, (unsigned long)max_tg, (unsigned long)sw);
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

    /* Skip upload if same-size ligand (common case: same molecule,
       multiple anchors).  If num_atoms differs, re-upload is needed. */
    if (g_set_ligand_num_atoms == num_atoms && g_active) {
        return 1;
    }

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

    /* Skip upload if same number of NB pairs (common case: same molecule,
       multiple anchors).  NB pairs don't change between anchors. */
    if (g_set_ie_num_nb_pairs == num_nb_pairs && g_num_nb_pairs > 0) {
        return 1;
    }

    @autoreleasepool {
        size_t ie_bytes = sizeof(float) * (size_t)g_num_atoms;
        memcpy([g_buf_ie_vdwA contents], ie_vdwA, ie_bytes);

        size_t nb_bytes = sizeof(int) * (size_t)num_nb_pairs * 2;
        memcpy([g_buf_nb_int contents], nb_int_pairs, nb_bytes);

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

        id<MTLCommandBuffer>  cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"DockBatchScoreIE";

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"BatchScoreIEKernel";

        [enc setComputePipelineState:g_pso_ie];

        /* Bind buffers — must match shader buffer layout */
        [enc setBuffer:g_buf_xyz        offset:0 atIndex:0];
        /* Grid data via 3D textures for hardware trilinear filtering */
        [enc setTexture:g_tex_grid_avdw  atIndex:0];
        [enc setTexture:g_tex_grid_bvdw  atIndex:1];
        [enc setTexture:g_tex_grid_es    atIndex:2];
        [enc setBuffer:g_buf_vdwA       offset:0 atIndex:1];
        [enc setBuffer:g_buf_vdwB       offset:0 atIndex:2];
        [enc setBuffer:g_buf_charges    offset:0 atIndex:3];
        [enc setBuffer:g_buf_params     offset:0 atIndex:4];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms    offset:0 atIndex:5];
        [enc setBuffer:g_buf_scores    offset:0 atIndex:6];
        [enc setBuffer:g_buf_ie_vdwA   offset:0 atIndex:7];
        [enc setBuffer:g_buf_nb_int    offset:0 atIndex:8];
        write_iep(g_ie_soft_delta, g_ie_cutoff_sq, g_num_nb_pairs);
        [enc setBuffer:g_buf_iep       offset:0 atIndex:9];
        write_nnp(g_num_nb_pairs);
        [enc setBuffer:g_buf_nnp       offset:0 atIndex:10];
        if (active_flags) {
            memcpy([g_buf_active_flags contents], active_flags, sizeof(int) * (size_t)num_atoms);
        }
        [enc setBuffer:g_buf_active_flags offset:0 atIndex:11];

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
        g_tex_grid_avdw  = nil;
        g_tex_grid_bvdw  = nil;
        g_tex_grid_es    = nil;
        g_buf_vdwA       = nil;
        g_buf_vdwB       = nil;
        g_buf_charges    = nil;
        g_buf_ie_vdwA    = nil;
        g_buf_nb_int     = nil;
        g_buf_active_flags = nil;
        g_buf_xyz        = nil;
        g_buf_scores     = nil;
        g_buf_vertex_dof = nil;
        g_buf_simplex_state = nil;
        g_buf_mol_data   = nil;
        g_buf_tg_header  = nil;
        g_pso            = nil;
        g_pso_ie         = nil;
        g_pso_simplex    = nil;
        g_cmdq           = nil;
        g_device         = nil;
        g_initialized    = 0;
        g_active         = 0;
        g_simplex_ready  = 0;
        g_buf_simplex_allocated = 0;
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
/*  GPU-side Simplex Minimization                                       */
/* ================================================================== */

void dock_gpu_simplex_init(void)
{
    /* Simplex PSO is compiled in dock_gpu_init().
       Buffers are allocated on first use in minimize().
       Nothing to do here unless we want lazy init. */
    if (!g_initialized) return;
    g_simplex_ready = 1;
}


int dock_gpu_simplex_minimize(const float *ref_xyz,
                               const int *active_flags,
                               int num_atoms, int num_active_atoms,
                               int num_torsions,
                               const int *torsion_a1,
                               const int *torsion_a2,
                               const int *torsion_a3,
                               const int *torsion_a4,
                               const int *child_idx_flat,
                               const int *child_starts,
                               const int *child_counts,
                               const int *torsion_scale_factors,
                               float *dof, float *scores,
                               int dof_size, int nverts,
                               int max_iterations,
                               float score_converge,
                               float trans_step_size,
                               float rot_step_size,
                               float tors_step_size)
{
    if (!g_initialized || !g_pso_simplex) return 0;

    @autoreleasepool {
        int i, j;

        /* ---- Allocate simplex buffers on first use ---- */
        int dof_pad = dof_size;
        if (dof_pad < 6) dof_pad = 6;
        if (dof_pad > GPU_DOF_MAX) dof_pad = GPU_DOF_MAX;

        if (!g_buf_simplex_allocated) {
            size_t buf_size = (size_t)nverts * (size_t)dof_pad * sizeof(float);
            g_buf_vertex_dof = alloc_buffer(buf_size, "simplex_vertex_dof");

            size_t state_size = (size_t)nverts * sizeof(float) + 6 * sizeof(int);
            g_buf_simplex_state = alloc_buffer(state_size, "simplex_state");

            /* MolData buffer: fixed size struct */
            size_t mol_size = sizeof(float) * GPU_MAX_ATOMS * 3
                            + sizeof(int) * GPU_MAX_ATOMS
                            + 3 * sizeof(int)
                            + 3 * sizeof(float)
                            + sizeof(float) * GPU_MAX_TORSIONS
                            + sizeof(int) * GPU_MAX_TORSIONS * 4
                            + sizeof(int) * GPU_MAX_TORSIONS * GPU_MAX_ATOMS
                            + sizeof(int) * GPU_MAX_TORSIONS;
            g_buf_mol_data = alloc_buffer(mol_size, "simplex_mol_data");

            if (!g_buf_vertex_dof || !g_buf_simplex_state || !g_buf_mol_data) {
                fprintf(stderr, "GPU-DOCK: simplex buffer allocation failed\n");
                return 0;
            }
            g_buf_simplex_allocated = 1;
        }

        /* ---- Upload molecule reference data to MolData buffer ---- */
        int dof_max = (dof_size < GPU_DOF_MAX) ? dof_size : GPU_DOF_MAX;
        if (dof_max < 6) dof_max = 6;

        /* Build MolData from raw arrays */
        struct MolData {
            float ref_xyz[512][3];
            int   active_flags[512];
            int   num_atoms;
            int   num_active_atoms;
            int   num_torsions;
            float com_x, com_y, com_z;   /* precomputed center of mass */
            float trans_step;
            float rot_step;
            float tors_step;
            float torsion_scale_factors[50];
            int   torsion_a1[50];
            int   torsion_a2[50];
            int   torsion_a3[50];
            int   torsion_a4[50];
            int   child_idx[50][512];
            int   child_cnt[50];
        };

        struct MolData md;
        memset(&md, 0, sizeof(md));

        for (i = 0; i < num_atoms && i < 512; i++) {
            md.ref_xyz[i][0] = ref_xyz[i*3];
            md.ref_xyz[i][1] = ref_xyz[i*3+1];
            md.ref_xyz[i][2] = ref_xyz[i*3+2];
            md.active_flags[i] = active_flags[i];
        }
        md.num_atoms = num_atoms;
        md.num_active_atoms = num_active_atoms;
        md.num_torsions = num_torsions;

        /* Precompute center of mass from ref_xyz + active_flags
         * (constant for a given active set — avoid recomputing in kernel). */
        {
            double comx = 0.0, comy = 0.0, comz = 0.0;
            int cnt = 0;
            for (int ai = 0; ai < num_atoms && ai < 512; ai++) {
                if (active_flags[ai]) {
                    comx += ref_xyz[ai*3];
                    comy += ref_xyz[ai*3+1];
                    comz += ref_xyz[ai*3+2];
                    cnt++;
                }
            }
            if (cnt > 0) { comx /= cnt; comy /= cnt; comz /= cnt; }
            md.com_x = (float)comx;
            md.com_y = (float)comy;
            md.com_z = (float)comz;
        }

        md.trans_step = trans_step_size;
        md.rot_step   = rot_step_size;
        md.tors_step  = tors_step_size;

        for (i = 0; i < num_torsions && i < 50; i++) {
            md.torsion_scale_factors[i] = (float)torsion_scale_factors[i];
            md.torsion_a1[i] = torsion_a1[i];
            md.torsion_a2[i] = torsion_a2[i];
            md.torsion_a3[i] = torsion_a3[i];
            md.torsion_a4[i] = torsion_a4[i];
            md.child_cnt[i] = child_counts[i];
            int start = child_starts[i];
            int cnt = child_counts[i];
            for (j = 0; j < cnt && j < 512; j++) {
                md.child_idx[i][j] = child_idx_flat[start + j];
            }
        }

        memcpy(g_buf_mol_data.contents, &md, sizeof(md));

        /* ---- Upload initial vertex DOF and scores ---- */
        /* Pack DOF vectors (each row padded to dof_pad) */
        float *vdst = (float *)g_buf_vertex_dof.contents;
        for (i = 0; i < nverts; i++) {
            for (j = 0; j < dof_pad; j++) {
                if (j < dof_size)
                    vdst[i * dof_pad + j] = dof[i * dof_size + j];
                else
                    vdst[i * dof_pad + j] = 0.0f;
            }
        }

        /* Scores go into the first nverts floats of g_buf_simplex_state */
        float *sdst = (float *)g_buf_simplex_state.contents;
        for (i = 0; i < nverts; i++) sdst[i] = scores[i];


        /* State = {converged, iter, n, max_iter} */
        int *statep = (int *)g_buf_simplex_state.contents;
        statep += nverts;
        statep[0] = 0;
        statep[1] = 0;
        statep[2] = dof_pad;
        statep[3] = max_iterations;
        /* Pack score_converge as IEEE float bits into int slot */
        {
            union { float f; int i; } u;
            u.f = score_converge;
            statep[4] = u.i;
        statep[5] = 0;
        }

        /* ---- Encode all iterations in one command buffer ---- */
        id<MTLCommandBuffer> cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"SimplexFullRun";

        /* Single dispatch — kernel loops internally over all iterations */
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        enc.label = @"SimplexAllIters";

        [enc setComputePipelineState:g_pso_simplex];

        /* Bind buffers + textures */
        [enc setBuffer:g_buf_vertex_dof    offset:0 atIndex:0];
        /* scores + state share a buffer: scores[0..nverts-1], state[nverts..nverts+5] */
        [enc setBuffer:g_buf_simplex_state  offset:0 atIndex:1];
        [enc setBuffer:g_buf_simplex_state  offset:(nverts * sizeof(float)) atIndex:2];
        [enc setBuffer:g_buf_mol_data       offset:0 atIndex:3];
        /* Grid data via 3D textures for hardware trilinear filtering */
        [enc setTexture:g_tex_grid_avdw     atIndex:0];
        [enc setTexture:g_tex_grid_bvdw     atIndex:1];
        [enc setTexture:g_tex_grid_es       atIndex:2];
        [enc setBuffer:g_buf_vdwA           offset:0 atIndex:4];
        [enc setBuffer:g_buf_vdwB           offset:0 atIndex:5];
        [enc setBuffer:g_buf_charges        offset:0 atIndex:6];
        [enc setBuffer:g_buf_params         offset:0 atIndex:7];
        write_natoms(num_atoms);
        [enc setBuffer:g_buf_natoms         offset:0 atIndex:8];
        [enc setBuffer:g_buf_xyz            offset:0 atIndex:9];  /* xyz device buffer */
        [enc setBuffer:g_buf_ie_vdwA        offset:0 atIndex:10];
        [enc setBuffer:g_buf_nb_int         offset:0 atIndex:11];
        write_iep(g_ie_soft_delta, g_ie_cutoff_sq, g_num_nb_pairs);
        [enc setBuffer:g_buf_iep            offset:0 atIndex:12];
        write_nnp(g_num_nb_pairs);
        [enc setBuffer:g_buf_nnp            offset:0 atIndex:13];

        /* Write tg_header: [tg_size, simd_group_size] */
        {
            int *hdr = (int *)[g_buf_tg_header contents];
            hdr[0] = g_simplex_threads;
            hdr[1] = g_simplex_simd_width;
        }
        [enc setBuffer:g_buf_tg_header      offset:0 atIndex:14];
        [enc setBuffer:g_buf_tg_header      offset:sizeof(int) atIndex:15];

        MTLSize gridSize = MTLSizeMake(g_simplex_threads, 1, 1);
        MTLSize tgSize   = MTLSizeMake(g_simplex_threads, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.error) {
            NSLog(@"GPU-DOCK: simplex command buffer error: %@", cmdbuf.error);
            return 0;
        }

        /* Check for outside-grid sentinel (-1) */
        statep = (int *)g_buf_simplex_state.contents;
        statep += nverts;
        if (statep[5] == -1) {
            return 0;
        }

        /* ---- Read back results ---- */
        float *final_scores = (float *)g_buf_simplex_state.contents;
        int ilo = 0;
        for (i = 1; i < nverts; i++) {
            if (final_scores[i] < final_scores[ilo]) ilo = i;
        }

        float *final_vdof = (float *)g_buf_vertex_dof.contents;
        for (i = 0; i < nverts; i++) {
            for (j = 0; j < dof_size; j++) {
                dof[i * dof_size + j] = final_vdof[i * dof_pad + j];
            }
            scores[i] = final_scores[i];
        }



        /* Success */
        return 1;
    }
}


void dock_gpu_simplex_cleanup(void)
{
    @autoreleasepool {
        g_buf_vertex_dof    = nil;
        g_buf_simplex_state = nil;
        g_buf_mol_data      = nil;
        g_simplex_ready        = 0;
        g_buf_simplex_allocated = 0;
    }
}


} /* extern "C" */
