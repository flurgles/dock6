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
"}\n"
;


/* ================================================================== */
/*  Simplex shader source (separate compilation)                       */
/* ================================================================== */

static const char* shader_src_simplex = R"shader(
#include <metal_stdlib>
using namespace metal;

#define PI 3.14159265358979323846f
#define MAX_ATOMS 512
#define MAX_TORSIONS 50

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

struct MolData {
    float ref_xyz[MAX_ATOMS][3];
    int   active_flags[MAX_ATOMS];
    int   num_atoms;
    int   num_active_atoms;
    int   num_torsions;
    int   torsion_a1[MAX_TORSIONS];
    int   torsion_a2[MAX_TORSIONS];
    int   torsion_a3[MAX_TORSIONS];
    int   torsion_a4[MAX_TORSIONS];
    int   child_idx[MAX_TORSIONS][MAX_ATOMS];
    int   child_cnt[MAX_TORSIONS];
};

static bool inside_grid(constant GridParams &p, float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    return (gx >= 1.0 && gx <= (float)(p.span_x - 2) &&
            gy >= 1.0 && gy <= (float)(p.span_y - 2) &&
            gz >= 1.0 && gz <= (float)(p.span_z - 2));
}

static float trilinear(device const float *grid, constant GridParams &p,
                        float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
        gx >= (float)(p.span_x - 1) ||
        gy >= (float)(p.span_y - 1) ||
        gz >= (float)(p.span_z - 1))
        return 0.0;
    int ix = (int)gx, iy = (int)gy, iz = (int)gz;
    float fx = gx - (float)ix, fy = gy - (float)iy, fz = gz - (float)iz;
    float fx1 = 1.0 - fx, fy1 = 1.0 - fy, fz1 = 1.0 - fz;
    int sx = p.span_x, sy = p.span_y;
    int i000 = iz * sx * sy + iy * sx + ix;
    int i001 = i000 + sx * sy, i010 = i000 + sx, i011 = i010 + sx * sy;
    int i100 = i000 + 1, i101 = i100 + sx * sy;
    int i110 = i100 + sx, i111 = i110 + sx * sy;
    float c000 = grid[i000], c001 = grid[i001];
    float c010 = grid[i010], c011 = grid[i011];
    float c100 = grid[i100], c101 = grid[i101];
    float c110 = grid[i110], c111 = grid[i111];
    float c00 = c000 * fx1 + c100 * fx;
    float c01 = c001 * fx1 + c101 * fx;
    float c10 = c010 * fx1 + c110 * fx;
    float c11 = c011 * fx1 + c111 * fx;
    float c0 = c00 * fy1 + c10 * fy;
    float c1 = c01 * fy1 + c11 * fy;
    return c0 * fz1 + c1 * fz;
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

static float3 compute_com(device const MolData &mol) {
    float3 com = {0,0,0};
    int cnt = 0;
    for (int i = 0; i < mol.num_atoms; i++) {
        if (mol.active_flags[i]) {
            com.x += mol.ref_xyz[i][0];
            com.y += mol.ref_xyz[i][1];
            com.z += mol.ref_xyz[i][2];
            cnt++;
        }
    }
    if (cnt > 0) { com.x /= cnt; com.y /= cnt; com.z /= cnt; }
    return com;
}

static void rigid_transform(thread float *xyz, int num_atoms,
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

static float dihedral_degrees(thread const float *xyz, int a1, int a2, int a3, int a4) {
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

static void apply_torsion(thread float *xyz, int a1, int a2, int a3, int a4,
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

static void dof_to_xyz(thread const float *dof, device const MolData &mol,
                        thread float *xyz) {
    int na = mol.num_atoms;
    for (int i = 0; i < na; i++) {
        xyz[i*3] = mol.ref_xyz[i][0];
        xyz[i*3+1] = mol.ref_xyz[i][1];
        xyz[i*3+2] = mol.ref_xyz[i][2];
    }
    float3 trans = {dof[0], dof[1], dof[2]};
    float3 quat = {dof[3], dof[4], dof[5]};
    float3x3 rmat;
    quat_to_rmat(rmat, quat);
    float3 com = compute_com(mol);
    rigid_transform(xyz, na, com, trans, rmat);
    for (int t = 0; t < mol.num_torsions; t++) {
        float delta_deg = dof[6 + t];
        if (fabs(delta_deg) < 1e-10) continue;
        float cur_deg = dihedral_degrees(xyz, mol.torsion_a1[t], mol.torsion_a2[t],
                                          mol.torsion_a3[t], mol.torsion_a4[t]);
        float new_rad = (PI/180.0)*(cur_deg + delta_deg);
        apply_torsion(xyz, mol.torsion_a1[t], mol.torsion_a2[t], mol.torsion_a3[t],
                       mol.torsion_a4[t], mol.child_idx[t], mol.child_cnt[t], new_rad);
    }
}

static float score_xyz(thread const float *xyz, int num_atoms,
                        device const float *gavdw, device const float *gbvdw,
                        device const float *ges,
                        device const float *vdwA, device const float *vdwB,
                        device const float *charges,
                        constant GridParams &gp,
                        device const float *ie_vdwA, device const int *nb_int,
                        constant IEParams &iep, int nnp) {
    float grid_score = 0.0;
    for (int a = 0; a < num_atoms; a++) {
        float x = xyz[a*3], y = xyz[a*3+1], z = xyz[a*3+2];
        if (!inside_grid(gp, x, y, z)) return -3.40282347e+38;
        float vdw = trilinear(gavdw, gp, x, y, z);
        float bvdw = trilinear(gbvdw, gp, x, y, z);
        float es = trilinear(ges, gp, x, y, z);
        grid_score += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
    }
    float ie_score = 0.0;
    if (nnp > 0) {
        for (int p = 0; p < nnp; p++) {
            int a1 = nb_int[p*2], a2 = nb_int[p*2+1];
            float dx = xyz[a1*3]-xyz[a2*3];
            float dy = xyz[a1*3+1]-xyz[a2*3+1];
            float dz = xyz[a1*3+2]-xyz[a2*3+2];
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < iep.cutoff_sq) {
                float r2eff = r2 + iep.soft_delta;
                float denom = r2eff*r2eff*r2eff;
                ie_score += (ie_vdwA[a1]*ie_vdwA[a2]) / (denom*denom);
            }
        }
    }
    return grid_score + ie_score;
}

kernel void simplex_iteration_kernel(
    device float*    vertex_dof    [[buffer(0)]],
    device float*    scores        [[buffer(1)]],
    device int*      state         [[buffer(2)]],
    device const MolData *mol      [[buffer(3)]],
    device const float* gavdw      [[buffer(4)]],
    device const float* gbvdw      [[buffer(5)]],
    device const float* ges        [[buffer(6)]],
    device const float* vdwA       [[buffer(7)]],
    device const float* vdwB       [[buffer(8)]],
    device const float* charges    [[buffer(9)]],
    constant GridParams &gp        [[buffer(10)]],
    constant int& num_atoms_buf    [[buffer(11)]],
    device float* out_scores       [[buffer(12)]],
    device const float* ie_vdwA    [[buffer(13)]],
    device const int* nb_int       [[buffer(14)]],
    constant IEParams& iep         [[buffer(15)]],
    constant int& num_nb_pairs     [[buffer(16)]],
    uint tid                       [[thread_position_in_grid]])
{
    if (tid > 0) return;

    int converged = state[0];
    if (converged) return;
    int iter = state[1];
    int n = state[2];       /* DOF size */
    int nverts = n + 1;     /* number of vertices */
    float alpha = 1.0, beta = 0.5, gamma = 2.0;
    int i, j;

    state[1] = iter + 1;

    /* 1. Find best (ilo), worst (ihi), second-worst (inhi) */
    int ilo = 0, ihi = 1, inhi = 0;
    if (scores[0] > scores[1]) { ihi = 0; inhi = 1; }
    else { ihi = 1; inhi = 0; }
    for (i = 0; i < nverts; i++) {
        if (scores[i] < scores[ilo]) ilo = i;
        if (scores[i] > scores[ihi]) { inhi = ihi; ihi = i; }
        else if (i != ihi && scores[i] > scores[inhi]) inhi = i;
    }

    /* 2. Compute centroid */
    float centroid[56];  /* DOF_MAX */
    int dof_max = (n < 56) ? n : 56;
    for (j = 0; j < dof_max; j++) centroid[j] = 0.0;
    for (i = 0; i < nverts; i++) {
        if (i != ihi) {
            for (j = 0; j < dof_max; j++)
                centroid[j] += vertex_dof[i * dof_max + j];
        }
    }
    for (j = 0; j < dof_max; j++) centroid[j] /= (float)n;

    /* 3. Reflect: pr = centroid + alpha*(centroid - worst) */
    float pr[56];
    int widx = ihi * dof_max;
    for (j = 0; j < dof_max; j++)
        pr[j] = centroid[j] + alpha * (centroid[j] - vertex_dof[widx + j]);

    /* 4. Score reflect */
    float xyz[MAX_ATOMS * 3];
    int na = mol->num_atoms;
    dof_to_xyz(pr, *mol, xyz);
    float score_refl = score_xyz(xyz, na, gavdw, gbvdw, ges,
                                  vdwA, vdwB, charges, gp,
                                  ie_vdwA, nb_int, iep, num_nb_pairs);
    if (score_refl < -1e30) { state[0] = -1; return; }

    /* 5. Decision tree */
    if (score_refl <= scores[ilo]) {
        /* Expand */
        float prr[56];
        for (j = 0; j < dof_max; j++)
            prr[j] = centroid[j] + gamma * (pr[j] - centroid[j]);
        dof_to_xyz(prr, *mol, xyz);
        float score_exp = score_xyz(xyz, na, gavdw, gbvdw, ges,
                                     vdwA, vdwB, charges, gp,
                                     ie_vdwA, nb_int, iep, num_nb_pairs);
        if (score_exp < -1e30) { state[0] = -1; return; }

        if (score_exp < score_refl) {
            for (j = 0; j < dof_max; j++) vertex_dof[widx + j] = prr[j];
            scores[ihi] = score_exp;
        } else {
            for (j = 0; j < dof_max; j++) vertex_dof[widx + j] = pr[j];
            scores[ihi] = score_refl;
        }
    } else if (score_refl >= scores[inhi]) {
        /* Contract or shrink */
        int replace_flag = 0;

        if (score_refl < scores[ihi]) {
            for (j = 0; j < dof_max; j++) vertex_dof[widx + j] = pr[j];
            scores[ihi] = score_refl;
            replace_flag = 1;
        }

        float prr[56];
        if (replace_flag) {
            for (j = 0; j < dof_max; j++)
                prr[j] = centroid[j] + beta * (pr[j] - centroid[j]);
        } else {
            for (j = 0; j < dof_max; j++)
                prr[j] = centroid[j] + beta * (vertex_dof[widx + j] - centroid[j]);
        }

        dof_to_xyz(prr, *mol, xyz);
        float score_con = score_xyz(xyz, na, gavdw, gbvdw, ges,
                                     vdwA, vdwB, charges, gp,
                                     ie_vdwA, nb_int, iep, num_nb_pairs);
        if (score_con < -1e30) { state[0] = -1; return; }

        if (score_con < scores[ihi]) {
            for (j = 0; j < dof_max; j++) vertex_dof[widx + j] = prr[j];
            scores[ihi] = score_con;
            replace_flag = 1;
        }

        if (!replace_flag) {
            /* Shrink */
            for (i = 0; i < nverts; i++) {
                if (i != ilo) {
                    for (j = 0; j < dof_max; j++) {
                        float old = vertex_dof[i * dof_max + j];
                        vertex_dof[i * dof_max + j] = 0.5 * (old + vertex_dof[ilo * dof_max + j]);
                    }
                    float local_dof[64];
                    for (j = 0; j < dof_max; j++)
                        local_dof[j] = vertex_dof[i * dof_max + j];
                    dof_to_xyz(local_dof, *mol, xyz);
                    float s = score_xyz(xyz, na, gavdw, gbvdw, ges,
                                         vdwA, vdwB, charges, gp,
                                         ie_vdwA, nb_int, iep, num_nb_pairs);
                    if (s < -1e30) { state[0] = -1; return; }
                    scores[i] = s;
                }
            }
        }
    } else {
        /* Accept reflect */
        for (j = 0; j < dof_max; j++) vertex_dof[widx + j] = pr[j];
        scores[ihi] = score_refl;
    }

    /* 6. Check convergence */
    ilo = 0; ihi = 0;
    for (i = 0; i < nverts; i++) {
        if (scores[i] < scores[ilo]) ilo = i;
        if (scores[i] > scores[ihi]) ihi = i;
    }
    float diff = fabs(scores[ihi] - scores[ilo]);
    state[0] = (diff < 0.001 || iter >= state[3]) ? 1 : 0;
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

/* Simplex-specific buffers */
static id<MTLBuffer> g_buf_vertex_dof = nil;
static id<MTLBuffer> g_buf_simplex_state = nil;
static id<MTLBuffer> g_buf_mol_data    = nil;

/* Cached simplex params */
static int  g_simplex_ready = 0;

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

        /* Compile simplex iteration shader (separate library) */
        id<MTLLibrary> lib_simplex = [g_device newLibraryWithSource:
                                        [NSString stringWithUTF8String:shader_src_simplex]
                                                              options:nil
                                                                error:&err];
        if (!lib_simplex) {
            fprintf(stderr, "GPU-DOCK: simplex shader compilation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            dock_gpu_cleanup();
            return 0;
        }
        id<MTLFunction> func_simplex = [lib_simplex newFunctionWithName:@"simplex_iteration_kernel"];
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
        g_buf_vertex_dof = nil;
        g_buf_simplex_state = nil;
        g_buf_mol_data   = nil;
        g_pso            = nil;
        g_pso_ie         = nil;
        g_pso_simplex    = nil;
        g_cmdq           = nil;
        g_device         = nil;
        g_initialized    = 0;
        g_active         = 0;
        g_simplex_ready  = 0;
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
                               float *dof, float *scores,
                               int dof_size, int nverts,
                               int max_iterations,
                               float score_converge)
{
    if (!g_initialized || !g_pso_simplex) return 0;

    @autoreleasepool {
        int i, j;

        /* ---- Allocate simplex buffers on first use ---- */
        static int buf_allocated = 0;
        int dof_pad = dof_size;
        if (dof_pad < 6) dof_pad = 6;
        if (dof_pad > GPU_DOF_MAX) dof_pad = GPU_DOF_MAX;

        if (!buf_allocated) {
            size_t buf_size = (size_t)nverts * (size_t)dof_pad * sizeof(float);
            g_buf_vertex_dof = alloc_buffer(buf_size, "simplex_vertex_dof");

            size_t state_size = (size_t)nverts * sizeof(float) + 4 * sizeof(int);
            g_buf_simplex_state = alloc_buffer(state_size, "simplex_state");

            /* MolData buffer: fixed size struct */
            size_t mol_size = sizeof(float) * GPU_MAX_ATOMS * 3
                            + sizeof(int) * GPU_MAX_ATOMS
                            + 3 * sizeof(int)
                            + sizeof(int) * GPU_MAX_TORSIONS * 4
                            + sizeof(int) * GPU_MAX_TORSIONS * GPU_MAX_ATOMS
                            + sizeof(int) * GPU_MAX_TORSIONS;
            g_buf_mol_data = alloc_buffer(mol_size, "simplex_mol_data");

            if (!g_buf_vertex_dof || !g_buf_simplex_state || !g_buf_mol_data) {
                fprintf(stderr, "GPU-DOCK: simplex buffer allocation failed\n");
                return 0;
            }
            buf_allocated = 1;
        }

        /* ---- Upload molecule reference data to MolData buffer ---- */
        int dof_max = (dof_size < 56) ? dof_size : 56;
        if (dof_max < 6) dof_max = 6;

        /* Build MolData from raw arrays */
        struct MolData {
            float ref_xyz[512][3];
            int   active_flags[512];
            int   num_atoms;
            int   num_active_atoms;
            int   num_torsions;
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

        for (i = 0; i < num_torsions && i < 50; i++) {
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

        /* ---- Encode all iterations in one command buffer ---- */
        id<MTLCommandBuffer> cmdbuf = [g_cmdq commandBuffer];
        cmdbuf.label = @"SimplexFullRun";

        for (int iter = 0; iter < max_iterations; iter++) {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            enc.label = [NSString stringWithFormat:@"SimplexIter%d", iter];

            [enc setComputePipelineState:g_pso_simplex];

            /* Bind all 17 buffers */
            [enc setBuffer:g_buf_vertex_dof    offset:0 atIndex:0];
            /* scores + state share a buffer: scores[0..nverts-1], state[nverts..nverts+3] */
            [enc setBuffer:g_buf_simplex_state  offset:0 atIndex:1];
            [enc setBuffer:g_buf_simplex_state  offset:(nverts * sizeof(float)) atIndex:2];
            [enc setBuffer:g_buf_mol_data       offset:0 atIndex:3];
            [enc setBuffer:g_buf_grid_avdw      offset:0 atIndex:4];
            [enc setBuffer:g_buf_grid_bvdw      offset:0 atIndex:5];
            [enc setBuffer:g_buf_grid_es        offset:0 atIndex:6];
            [enc setBuffer:g_buf_vdwA           offset:0 atIndex:7];
            [enc setBuffer:g_buf_vdwB           offset:0 atIndex:8];
            [enc setBuffer:g_buf_charges        offset:0 atIndex:9];
            [enc setBuffer:g_buf_params         offset:0 atIndex:10];
            write_natoms(num_atoms);
            [enc setBuffer:g_buf_natoms         offset:0 atIndex:11];
            [enc setBuffer:g_buf_xyz            offset:0 atIndex:12];  /* out_scores (unused) */
            [enc setBuffer:g_buf_ie_vdwA        offset:0 atIndex:13];
            [enc setBuffer:g_buf_nb_int         offset:0 atIndex:14];
            write_iep(g_ie_soft_delta, g_ie_cutoff_sq, g_num_nb_pairs);
            [enc setBuffer:g_buf_iep            offset:0 atIndex:15];
            write_nnp(g_num_nb_pairs);
            [enc setBuffer:g_buf_nnp            offset:0 atIndex:16];

            MTLSize one = MTLSizeMake(1, 1, 1);
            [enc dispatchThreads:one threadsPerThreadgroup:one];
            [enc endEncoding];
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        if (cmdbuf.error) {
            NSLog(@"GPU-DOCK: simplex command buffer error: %@", cmdbuf.error);
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
        g_simplex_ready     = 0;
    }
}


} /* extern "C" */
