#include <iostream>
#include <vector>
#include <string.h>
#include "dockmol.h"
#include "score_dock_gpu.h"
#include "minimizer.h"
using namespace std;


/* ================================================================= */
/*  GPU Batch Scoring — combined grid + internal energy for N vertices */
/* ================================================================= */

bool
Minimizer::gpu_batch_eval_scores(Base_Score & score,
                                   DOCKMol & ref_mol,
                                   DOCKMol & tmp_mol,
                                   const std::vector<FLOATVec> & vertices,
                                   float trans_step_size,
                                   float rot_step_size,
                                   float tors_step_size,
                                   float Econstraint,
                                   float * scores)
{
    if (!dock_gpu_is_active()) return false;

    int n = (int)vertices.size();
    if (n == 0) return true;

    int na = ref_mol.num_atoms;
    float *xyz = new float[n * na * 3];
    int *active_flags = new int[na];
    for (int a = 0; a < na; a++) {
        active_flags[a] = ref_mol.atom_active_flags[a] ? 1 : 0;
    }

    /* For each vertex: apply vector_to_dockmol, extract coordinates */
    for (int vi = 0; vi < n; vi++) {
        copy_crds(tmp_mol, ref_mol);
        FLOATVec new_vec;
        scale_vector(new_vec, const_cast<FLOATVec&>(vertices[vi]),
                     trans_step_size, rot_step_size, tors_step_size);
        vector_to_dockmol(tmp_mol, new_vec);

        int base = vi * na * 3;
        for (int a = 0; a < na; a++) {
            xyz[base + a*3]     = tmp_mol.x[a];
            xyz[base + a*3 + 1] = tmp_mol.y[a];
            xyz[base + a*3 + 2] = tmp_mol.z[a];
        }
    }

    /* Launch GPU batch with internal energy */
    int ok = dock_gpu_batch_score_with_ie_persistent(xyz, n, na, active_flags, scores);
    delete[] xyz;
    delete[] active_flags;

    if (!ok) {
        /* No IE data loaded — fall back to CPU */
        return false;
    }

    /* Add Econstraint and check for sentinel values */
    for (int vi = 0; vi < n; vi++) {
        if (scores[vi] < -1e30) {
            /* Outside grid sentinel — abort like CPU eval_score() failure */
            return false;
        }
        scores[vi] += Econstraint;
    }

    return true;
}
