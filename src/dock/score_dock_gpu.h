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
/*  GPU pool sizing constants                                          */
/* ------------------------------------------------------------------ */

/* Safe per-dispatch pose cap.  The device xyz buffer is sized
   GPU_MAX_POSES (4096) x GPU_MAX_ATOMS (512) x 3 floats; keeping
   dispatches under this cap guarantees they are never rejected, no
   matter how large the packed pool batch is. */
#define GPU_MAX_BATCH_POSES 3968

/* ConformerPool slot budget for the anchor and growth pools.  With 512
   slots and worst-case 7 (anchor) / ~20 (growth) candidates per slot,
   one step packs up to ~3-4 chunks of GPU_MAX_BATCH_POSES poses. */
#define GPU_POOL_BATCH_MAX 512

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
   active_flags: per-atom mask (array of num_atoms ints; nonzero = active).
        Inactive atoms are skipped, matching the CPU grid scoring.
        May be NULL, in which case all atoms are treated as active.
   out_scores: output array of num_poses floats (caller-allocated)
   Returns 1 on success, 0 on error. */
int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         const int *active_flags, float *out_scores);

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

/* Recommended batch size for the conformer pool.
   Queries the GPU's compute-unit count via IOKit and returns
   (cores * 4) as the heuristic batch size.  Falls back to 32
   on unknown devices or when GPU is inactive.
   The conformer pool uses this as its default capacity. */
int dock_gpu_recommended_batch_size(void);


/* ------------------------------------------------------------------ */
/*  GPU thermal/performance monitor (per-segment, -v only)              */
/* ------------------------------------------------------------------ */

/* Report thermal state and GPU dispatch time rolling average to stdout.
   Caller should guard with -v verbose flag.  Thread-safe (no state). */
void dock_gpu_monitor(int layer, int segment, int total_segments);



/* ------------------------------------------------------------------ */
/*  Multi-ligand virtual-screen scoring                                 */
/* ------------------------------------------------------------------ */

/* Register a ligand's per-atom parameters in the VS ligand table.
   lig_idx: 0 .. dock_gpu_vs_max_ligands()-1
   vdwA, vdwB, charges, active_flags, ie_vdwA: length num_atoms
   nb_int_pairs: flat array of num_nb_pairs*2 ints (a1, a2, ...)
   ie_soft_delta / ie_cutoff_sq: internal-energy soft-core delta and
   distance-squared cutoff for THIS ligand's pair scoring (mirrors
   dock_gpu_set_ligand_ie).  Stored as the current IE parameters used
   by dock_gpu_batch_score_vs().
   The row tail beyond num_atoms is zeroed so poses of shorter ligands
   are never scored against stale parameters left by a previous row owner.
   Returns 1 on success, 0 on error. */
int dock_gpu_vs_register_ligand(int lig_idx,
                                const float *vdwA, const float *vdwB,
                                const float *charges, const int *active_flags,
                                const float *ie_vdwA,
                                const int *nb_int_pairs, int num_nb_pairs,
                                int num_atoms,
                                float ie_soft_delta, float ie_cutoff_sq);

/* Max ligand-table rows the backend accepts. */
int dock_gpu_vs_max_ligands(void);

/* Score N poses against the shared grid, each pose belonging to one
   registered ligand.  xyz is flat N * num_atoms * 3 (num_atoms = padded
   max atom count; inactive atoms must have active_flags[a]==0 in their
   ligand row).  pose_lig: length num_poses, values < table size.
   Returns 1 on success, 0 on error. */
int dock_gpu_batch_score_vs(const float *xyz, int num_poses, int num_atoms,
                            const int *pose_lig, float *out_scores);

/* Report the grid box volume that poses must occupy to be scorable:
   [minx,maxx]x[miny,maxy]x[minz,maxz] in Angstroms, mirroring
   Base_Grid::is_inside_grid_box() (points within one voxel of the box
   edge are excluded, since trilinear interpolation needs the full
   neighborhood).  Callers use this to mark out-of-bounds poses the
   kernel cannot reject (clamp-to-edge sampling).  Returns 1 on
   success, 0 if the backend cannot provide the box. */
int dock_gpu_grid_bounds(float *minx, float *miny, float *minz,
                         float *maxx, float *maxy, float *maxz);



#ifdef __cplusplus
}
#endif

#endif /* SCORE_DOCK_GPU_H */
