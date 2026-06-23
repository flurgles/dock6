/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
GPU abstraction API for dock6 scoring acceleration.

Provides a clean interface for batch-pose scoring on GPU.
Each GPU backend (Metal, Vulkan, CUDA) implements these functions.
The CPU fallback stub returns 0 from init(), causing the
existing CPU code path to run.

Usage in dock scoring code (e.g., conf_gen_ag.cpp):
    #include "score_dock_gpu.h"

    if (dock_gpu_is_active()) {
        // Pack N pose coordinates into flat array
        float *xyz = pack_poses(poses, num_poses);
        float *scores = new float[num_poses];
        dock_gpu_batch_score(xyz, num_poses, scores);
        // Use scores for pruning
        ...
    }
    // fall through to CPU path if GPU not active
*/

#ifndef SCORE_DOCK_GPU_H
#define SCORE_DOCK_GPU_H

/* ------------------------------------------------------------------ */
/*  GPU abstraction API                                               */
/* ------------------------------------------------------------------ */

/* All functions use C linkage so they can be called from C++ code
   even when the backend implementation is compiled as Objective-C++. */
#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU device and upload grid data.
   avdw, bvdw, es: flat arrays of length span_x * span_y * span_z.
   Returns 1 on success, 0 if GPU unavailable (caller falls back to CPU). */
int dock_gpu_init(const float *avdw, const float *bvdw, const float *es,
                  int span_x, int span_y, int span_z,
                  float origin_x, float origin_y, float origin_z,
                  float spacing);

/* Score N poses in one GPU batch launch (grid score only).
   xyz: flat array of N * num_atoms * 3 floats, laid out as:
        pose[0].atom[0].x, .y, .z,  pose[0].atom[1].x, .y, .z, ...
        pose[1].atom[0].x, .y, .z, ...
   num_poses: number of poses to score (≥ 1)
   num_atoms: number of atoms per pose
   out_scores: output array of num_poses floats (caller-allocated)
   Returns 1 on success, 0 on error. */
int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         float *out_scores);

/* Upload per-atom internal energy parameters (constant per ligand).
   Called after dock_gpu_set_ligand() and dock_gpu_init().
   ie_vdwA/B: per-atom LJ parameters, arrays of length num_atoms.
   nb_int_pairs: flat array of num_nb_pairs * 2 ints (a1, a2, a1, a2, ...).
   ie_soft_delta: soft-core delta for IE (0.0 = none).
   ie_cutoff_sq: distance-squared cutoff for IE pairs.
   Returns 1 on success, 0 on error. */
int dock_gpu_set_ligand_ie(const float *ie_vdwA, const float *ie_vdwB,
                            const int *nb_int_pairs, int num_nb_pairs,
                            float ie_soft_delta, float ie_cutoff_sq);

/* Score N poses in one GPU batch launch (grid + internal energy).
   Same layout as dock_gpu_batch_score.
   Requires dock_gpu_set_ligand_ie() to have been called.
   Returns 1 on success, 0 on error. */
int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                  float *out_scores);

/* Upload per-atom scoring parameters (constant per ligand).
   vdwA, vdwB, charges: arrays of length num_atoms.
   These are the pre-resolved values used in the scoring kernel:
       atom_score = vdwA[a] * avdw(x) - vdwB[a] * bvdw(x) + charges[a] * es(x)
   Must be called after dock_gpu_init() and before any batch_score call.
   Returns 1 on success, 0 on error. */
int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                        const float *charges, int num_atoms);

/* Release all GPU resources. */
void dock_gpu_cleanup(void);

/* Returns 1 if GPU is active and ready for batch scoring. */
int dock_gpu_is_active(void);

#ifdef __cplusplus
}
#endif

#endif /* SCORE_DOCK_GPU_H */
