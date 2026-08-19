/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
CPU fallback stub — all functions are no-ops.
dock_gpu_init() returns 0, so callers fall through to the CPU path.
*/

#include "score_dock_gpu.h"

int dock_gpu_init(const float *avdw, const float *bvdw, const float *es,
                  int span_x, int span_y, int span_z,
                  float origin_x, float origin_y, float origin_z,
                  float spacing)
{
    (void)avdw; (void)bvdw; (void)es;
    (void)span_x; (void)span_y; (void)span_z;
    (void)origin_x; (void)origin_y; (void)origin_z;
    (void)spacing;
    return 1;  /* pretend GPU exists so the windowed scheduler runs */
}

int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         const int *active_flags, float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms;
    (void)active_flags; (void)out_scores;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                         const float *charges, int num_atoms)
{
    (void)vdwA; (void)vdwB; (void)charges; (void)num_atoms;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_set_scales(float vdw_scale, float es_scale)
{
    (void)vdw_scale; (void)es_scale;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_set_ligand_ie(const float *ie_vdwA, const float *ie_vdwB,
                            const int *nb_int_pairs, int num_nb_pairs,
                            float ie_soft_delta, float ie_cutoff_sq)
{
    (void)ie_vdwA; (void)ie_vdwB; (void)nb_int_pairs;
    (void)num_nb_pairs; (void)ie_soft_delta; (void)ie_cutoff_sq;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                  const int *active_flags, float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)active_flags; (void)out_scores;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_batch_score_with_ie_persistent(const float *xyz, int num_poses,
                                              int num_atoms,
                                              const int *active_flags,
                                              float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)active_flags; (void)out_scores;
    return 0;  /* not implemented on CPU */
}

void dock_gpu_cleanup(void)
{
}

int dock_gpu_is_active(void)
{
    return 0;
}

int dock_gpu_recommended_batch_size(void)
{
    return 32;  /* safe default when GPU unavailable */
}


void dock_gpu_monitor(int layer, int segment, int total_segments)
{
    (void)layer; (void)segment; (void)total_segments;
    /* no-op: GPU not available */
}

int dock_gpu_vs_register_ligand(int lig_idx,
                                const float *vdwA, const float *vdwB,
                                const float *charges, const int *active_flags,
                                const float *ie_vdwA,
                                const int *nb_int_pairs, int num_nb_pairs,
                                int num_atoms,
                                float ie_soft_delta, float ie_cutoff_sq)
{
    (void)lig_idx; (void)vdwA; (void)vdwB; (void)charges;
    (void)active_flags; (void)ie_vdwA; (void)nb_int_pairs;
    (void)num_nb_pairs; (void)num_atoms;
    return 0;
}

int dock_gpu_vs_update_active_flags(int lig_idx, const int *active_flags,
                                    int num_atoms)
{
    (void)lig_idx; (void)active_flags; (void)num_atoms;
    return 0;
}

int dock_gpu_vs_max_ligands(void)
{
    return 0;
}

int dock_gpu_batch_score_vs(const float *xyz, int num_poses, int num_atoms,
                            const int *pose_lig, float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores;
    return 0;
}

int dock_gpu_batch_score_vs_grid(const float *xyz, int num_poses,
                                 int num_atoms, const int *pose_lig,
                                 float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores;
    return 0;
}

int dock_gpu_batch_score_vs_enqueue(const float *xyz, int num_poses,
                                    int num_atoms, const int *pose_lig,
                                    float *out_scores, int grid_only)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores; (void)grid_only;
    return 0;
}

int dock_gpu_batch_score_sync(void)
{
    return 0;
}

int dock_gpu_batch_score_vs_enqueue2(const float *xyz, int num_poses,
                                     int num_atoms, const int *pose_lig,
                                     float *out_scores, int grid_only)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores; (void)grid_only;
    return 0;
}

int dock_gpu_batch_score_sync2(void)
{
    return 0;
}

int dock_gpu_npends2(void)
{
    return 0;
}

void dock_gpu_lbal_stats(long long *host_ms, long long *gpu_ms)
{
    *host_ms = 0;
    *gpu_ms = 0;
}

int dock_gpu_grid_bounds(float *minx, float *miny, float *minz,
                         float *maxx, float *maxy, float *maxz)
{
    (void)minx; (void)miny; (void)minz; (void)maxx; (void)maxy; (void)maxz;
    return 0;
}
