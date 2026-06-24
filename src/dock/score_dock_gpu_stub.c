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
    return 0;  /* GPU not available, use CPU fallback */
}

int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)out_scores;
    return 0;  /* not implemented on CPU */
}

int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                        const float *charges, int num_atoms)
{
    (void)vdwA; (void)vdwB; (void)charges; (void)num_atoms;
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

void dock_gpu_cleanup(void)
{
}

int dock_gpu_is_active(void)
{
    return 0;
}


void dock_gpu_simplex_init(void)
{
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
    (void)ref_xyz; (void)active_flags;
    (void)num_atoms; (void)num_active_atoms;
    (void)num_torsions;
    (void)torsion_a1; (void)torsion_a2;
    (void)torsion_a3; (void)torsion_a4;
    (void)child_idx_flat; (void)child_starts; (void)child_counts;
    (void)torsion_scale_factors;
    (void)dof; (void)scores;
    (void)dof_size; (void)nverts;
    (void)max_iterations; (void)score_converge;
    (void)trans_step_size; (void)rot_step_size; (void)tors_step_size;
    return 0;  /* not implemented on CPU */
}


void dock_gpu_simplex_cleanup(void)
{
}
