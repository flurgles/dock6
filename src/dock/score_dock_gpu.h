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
#include <stddef.h>

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
                                  const int *active_flags, float *out_scores);

/* Same as dock_gpu_batch_score_with_ie but uses persistent threadgroups
   via atomic work-counter for better GPU core utilization across varying
   batch sizes.  API identical — takes xyz, returns scores.  Backend
   dispatches a fixed number of threadgroups that dynamically claim work. */
int dock_gpu_batch_score_with_ie_persistent(const float *xyz, int num_poses,
                                              int num_atoms,
                                              const int *active_flags,
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
int dock_gpu_is_integrated(void);
int dock_gpu_mem_info(size_t *freeB, size_t *totalB);

/* Recommended batch size for the conformer pool.
   Queries the GPU's compute-unit count via IOKit and returns
   (cores * 4) as the heuristic batch size.  Falls back to 32
   on unknown devices or when GPU is inactive.
   The conformer pool uses this as its default capacity. */
int dock_gpu_recommended_batch_size(void);

/* Per-dispatch pose ceiling enforced by the backend. Callers must size
   their own dispatch buffers to at most this many poses; exceeding it is
   a hard error, not a silent truncation. */
int dock_gpu_max_poses(void);


/* Scrollable channel serializer for growth layers (see conf_gen_ag.cpp).
   Never called for the CPU path; only used on GPU when the whole growth
   loop is offloaded.  Returns 1 on success, 0 if unavailable. */

/* ------------------------------------------------------------------ */
/*  GPU anchor growth                                                  */
/* ------------------------------------------------------------------ */

/* Grow ALL orients of one anchor in a SINGLE GPU dispatch.
 * The anchor shader (anchor_grow.comp) runs one workgroup per orient:
 * per-layer torsion drive, simplex minimization, pruning and RMSD
 * clustering all run on the GPU; the CPU thread only submits and waits.
 *
 * orient_xyz: [N*na*3] one full-molecule coordinate copy per orient
 *             (the anchor as positioned in that orientation, all atoms).
 * ref_xyz:    [na*3] ligand template — the anchor positions for the
 *             minimization (na = total ligand atoms).
 * tors_a1234: [num_tors*4] rotatable-bond atoms (a1,a2,a3,a4) in the
 *             layer/segment order used by grow_periphery.
 * tors_mask:  [num_tors*na] 1 where atom `a` is driven by torsion `t`.
 *
 * Growth parameters mirror the CPU flex.growth options:
 *   simplex_max_iter  (500 anchor / 250 grow)
 *   score_converge    (0.1)
 *   trans_step, rot_step, tors_step
 *   det_seed, pruning_score_cutoff (100), cluster_rmsd (2.0)
 *
 * out_xyz:    [max_out*na*3] survivor poses.
 * out_scores: [max_out] their grid+IE scores.
 * out_count:  number of survivors (≤ max_out), poses are in slot order.
 *
 * Returns 1 on success, 0 on failure (caller MUST fall back to CPU growth).
 * Requires prior dock_gpu_set_ligand() + dock_gpu_set_ligand_ie(). */
int dock_gpu_grow_anchor(const float *orient_xyz, const float *ref_xyz,
                         int N, int na, int num_tors, int num_layers, int max_out,
                         const int *tors_a1234, const int *tors_mask,
                         const float *tors_angles, const int *actset_rows,
                         int simplex_max_iter, float score_converge,
                         float trans_step, float rot_step, float tors_step,
                         int det_seed, float pruning_score_cutoff, float cluster_rmsd,
                         float *out_xyz, float *out_scores, int *out_count);


/* ------------------------------------------------------------------ */
/*  GPU thermal/performance monitor (per-segment, -v only)              */
/* ------------------------------------------------------------------ */

/* Report thermal state and GPU dispatch time rolling average to stdout.
   Caller should guard with -v verbose flag.  Thread-safe (no state). */
void dock_gpu_monitor(int layer, int segment, int total_segments);



#ifdef __cplusplus
}
#endif

#endif /* SCORE_DOCK_GPU_H */
