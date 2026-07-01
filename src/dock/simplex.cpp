#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "master_score.h"
#include "minimizer.h"
#include "simplex.h"
#include "conf_gen_ag.h"
#include "score_dock_gpu.h"

using namespace std;


#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "master_score.h"
#include "minimizer.h"
#include "simplex.h"
#include "conf_gen_ag.h"
#include "score_dock_gpu.h"

using namespace std;

void
Simplex_Minimizer::input_parameters(Parameter_Reader & parm,
                                    bool flexible_ligand, bool genetic_algorithm, bool denovo_design, Master_Score & score)
{

    advanced_min_params = false;
    use_min_rigid_anchor = false;
    use_min_flex_growth = false;
    use_min_flex_growth_ramp = false;
    final_min = false;
    secondary_min_pose = false;

    minimize_ligand = false;
    random_seed = 0;

    // restrained minimization parameters
    restrained_min = false;
    coefficient_restraint = 0.0;


    if (score.primary_min) {

        cout << "\nSimplex Minimization Parameters" << endl;
        cout <<
            "------------------------------------------------------------------------------------------"
            << endl;

        minimize_ligand =
            (parm.query_param("minimize_ligand", "yes", "yes no") ==
             "yes") ? true : false;

        if (minimize_ligand) {

            // if anchor and grow is flagged, check which portions of molecule
            // to minimize
            // PAK added logic and queries so user can choose if they want the simplex ramp
            if (flexible_ligand) {
                use_min_rigid_anchor = (parm.query_param("minimize_anchor", "yes", "yes no") == "yes") ? true : false;
                use_min_flex_growth = (parm.  query_param("minimize_flexible_growth", "yes", "yes no") == "yes") ? true : false;
		        advanced_min_params = (parm.  query_param("use_advanced_simplex_parameters", "no", "yes no") == "yes") ? true : false;
                // LEP - ga and dn not ready for simplex ramp
                if (!genetic_algorithm && !denovo_design){
                    use_min_flex_growth_ramp = (parm.  query_param("minimize_flexible_growth_ramp", "yes", "yes no") == "yes") ? true : false;
                }

            }
            // if docking is rigid or if the same parameters will be used for
            // all levels of primary minimization
            if (!advanced_min_params) {
                if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                    const char simplex_max_it[] = "simplex_max_iterations";
                    max_iterations = atoi(parm.query_param(simplex_max_it, "1000").c_str());
                    if (max_iterations < 0) { // this is so that we can run only tors_premin
                        cout << "ERROR:  Parameter \"" << simplex_max_it
                            << "\" must be an integer greater than or equal to zero."
                            << endl
                            << "Program will terminate."
                            << endl;
                        exit(0);
                    }
                   
                    // ask for # of iterations of the torsion premin 
                    torsion_iterations = atoi(parm.query_param("simplex_tors_premin_iterations", "0").  c_str());
                    if (torsion_iterations < 0) {
                        cout << "ERROR:  simplex_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                        exit(0);
                    }

                }
                max_cycles =
                    atoi(parm.query_param("simplex_max_cycles", "1").c_str());
                if (max_cycles <= 0) {
                    cout <<
                        "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                /// PAK
                if (!use_min_flex_growth_ramp){
                    score_converge =
                        atof(parm.query_param("simplex_score_converge", "0.1").
                             c_str());
                    if (score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                }
                if (use_min_flex_growth_ramp){
                    score_converge =
                           atof(parm.query_param("simplex_score_converge", "0.1").
                                c_str());
                       if (score_converge <= 0.0) {
                           cout <<
                               "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                               << endl;
                           exit(0);
                       }
                   initial_score_converge = atof(parm.query_param("simplex_initial_score_coverge","5").c_str());
                       if (initial_score_converge <= score_converge) {
                           cout <<
                               "ERROR:  Parameter must be larger than score converge value.  Program will terminate."
                              << endl;
                           exit(0);
                       }
                 }
                cycle_converge =
                    atof(parm.query_param("simplex_cycle_converge", "1.0").
                         c_str());
                if (cycle_converge <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                trans_step_size =
                    atof(parm.query_param("simplex_trans_step", "1.0").c_str());
                if (trans_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                rot_step_size =
                    atof(parm.query_param("simplex_rot_step", "0.1").c_str());
                if (rot_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                tors_step_size =
                    atof(parm.query_param("simplex_tors_step", "10.0").c_str());
                if (tors_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
            }
            // parameters for anchor minimization
            if (use_min_rigid_anchor) {
                anchor_min_max_iterations = atoi(parm.  query_param("simplex_anchor_max_iterations", "500").c_str());
                if (anchor_min_max_iterations <= 0) {
                    cout <<
                        "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }

                if (advanced_min_params) {
                    anchor_min_max_cycles =
                        atoi(parm.query_param("simplex_anchor_max_cycles", "1").
                             c_str());
                    if (anchor_min_max_cycles <= 0) {
                        cout <<
                            "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    anchor_min_score_converge =
                        atof(parm.
                             query_param("simplex_anchor_score_converge",
                                         "0.1").c_str());
                    if (anchor_min_score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    anchor_min_cycle_converge =
                        atof(parm.
                             query_param("simplex_anchor_cycle_converge",
                                         "1.0").c_str());
                    if (anchor_min_cycle_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    anchor_min_trans_step_size =
                        atof(parm.
                             query_param("simplex_anchor_trans_step",
                                         "1.0").c_str());
                    if (anchor_min_trans_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    anchor_min_rot_step_size =
                        atof(parm.query_param("simplex_anchor_rot_step", "0.1").
                             c_str());
                    if (anchor_min_rot_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    anchor_min_tors_step_size =
                        atof(parm.
                             query_param("simplex_anchor_tors_step",
                                         "10.0").c_str());
                    if (anchor_min_tors_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                } else {
                    anchor_min_max_cycles = max_cycles;
                    anchor_min_score_converge = score_converge;
                    anchor_min_cycle_converge = cycle_converge;
                    anchor_min_trans_step_size = trans_step_size;
                    anchor_min_rot_step_size = rot_step_size;
                    anchor_min_tors_step_size = tors_step_size;
                }
            }
            // parameters for flexible minimization
            if (use_min_flex_growth) {

                // If ramp is on simplex_grow_max_iterations should be 250, otherwise 500
                if (use_min_flex_growth_ramp) {
                   flex_min_max_iterations = atoi(parm.query_param("simplex_grow_max_iterations", "250").  c_str());
                }
                else {
                   flex_min_max_iterations = atoi(parm.query_param("simplex_grow_max_iterations", "500").  c_str());
                }

                if (flex_min_max_iterations < 0) {
                    cout <<
                        "ERROR:  simplex_grow_max_iterations cannot be negative.  Program will terminate."
                        << endl;
                    exit(0);
                }

                flex_min_torsion_iterations = atoi(parm.query_param("simplex_grow_tors_premin_iterations", "0").  c_str());
                if (flex_min_torsion_iterations < 0) {
                    cout << "ERROR:  simplex_grow_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                    exit(0);
                }

                if (advanced_min_params) {
                    flex_min_max_cycles =
                        atoi(parm.query_param("simplex_grow_max_cycles", "1").
                             c_str());
                    if (flex_min_max_cycles <= 0) {
                        cout <<
                            "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_score_converge =
                        atof(parm.
                             query_param("simplex_grow_score_converge",
                                         "0.1").c_str());
                    if (flex_min_score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_cycle_converge =
                        atof(parm.
                             query_param("simplex_grow_cycle_converge",
                                         "1.0").c_str());
                    if (flex_min_cycle_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_trans_step_size =
                        atof(parm.query_param("simplex_grow_trans_step", "1.0").
                             c_str());
                    if (flex_min_trans_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_rot_step_size =
                        atof(parm.query_param("simplex_grow_rot_step", "0.1").
                             c_str());
                    if (flex_min_rot_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_tors_step_size =
                        atof(parm.query_param("simplex_grow_tors_step", "10.0").
                             c_str());
                    if (flex_min_tors_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                } else {
                    flex_min_max_cycles = max_cycles;
                    flex_min_score_converge = score_converge;
                    flex_min_cycle_converge = cycle_converge;
                    flex_min_trans_step_size = trans_step_size;
                    flex_min_rot_step_size = rot_step_size;
                    flex_min_tors_step_size = tors_step_size;
                }
            }

        }
        // final_min has been superseded by the option to perform anchor and grow 
        // docking with internal energy at every level of growth
        // option to perform one more round of minimization 
        // TEB put final min back, 2023
        //
        final_min = (parm.query_param("simplex_final_min", "no", "yes no") == "yes") ? true : false;
        if (final_min) {
            final_min_rep_radius_scale = atof(parm.query_param("simplex_final_min_rep_rad_scale", "1").c_str());
            if (final_min_rep_radius_scale <= 0.0) {
                cout <<
                    "ERROR:  Parameter must be a float greater than zero. Program will terminate."
                    << endl;
                exit(0);
            }
            final_min_max_iterations =
                atoi(parm.query_param("simplex_final_max_iterations", "500").
                     c_str());
            if (final_min_max_iterations <= 0) {
                cout <<
                    "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(0);
            }
            //if (advanced_min_params) {
                final_min_max_cycles =
                    atoi(parm.query_param("simplex_final_max_cycles", "1").
                         c_str());
                if (final_min_max_cycles <= 0) {
                    cout <<
                        "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                final_min_score_converge =
                    atof(parm.
                         query_param("simplex_final_score_converge",
                                     "0.1").c_str());
                if (final_min_score_converge <= 0.0) {
                    cout <<
                        "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                final_min_cycle_converge =
                    atof(parm.
                         query_param("simplex_final_cycle_converge",
                                     "1.0").c_str());
                if (final_min_cycle_converge <= 0.0) {
                    cout <<
                        "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                final_min_trans_step_size =
                    atof(parm.
                         query_param("simplex_final_trans_step",
                                     "1.0").c_str());
                if (final_min_trans_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                final_min_rot_step_size =
                    atof(parm.query_param("simplex_final_rot_step", "0.1").
                         c_str());
                if (final_min_rot_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
                final_min_tors_step_size =
                    atof(parm.
                         query_param("simplex_final_tors_step",
                                     "10.0").c_str());
                if (final_min_tors_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(0);
                }
            //} else {
            //    final_min_max_cycles = max_cycles;
            //    final_min_score_converge = score_converge;
            //    final_min_cycle_converge = cycle_converge;
            //    final_min_trans_step_size = trans_step_size;
            //    final_min_rot_step_size = rot_step_size;
            //    final_min_tors_step_size = tors_step_size;
            //}
        }
      
        if (final_min or minimize_ligand) {  

           random_seed =
               atoi(parm.query_param("simplex_random_seed", "0").c_str());
           //trent balius 2009/08/27
           restrained_min = (parm.query_param("simplex_restraint_min", "no", "yes no") == "yes") ? true : false;
           if (restrained_min) {
               coefficient_restraint =
                     atof(parm.query_param("simplex_coefficient_restraint", "10.0").c_str());
           }
        }
    }
}
void
Simplex_Minimizer::initialize()
{
    //cout << "Initializing simplex" << endl;
    srand(random_seed);

}


/* ================================================================== */
/*  C1 Batch Queue Infrastructure                                      */
/* ================================================================== */

struct GpuBatchSlot {
    DOCKMol  *mol;              /* pointer to conformer's molecule (stable) */
    float    *p_flat;           /* flat copy of p[][] [nv * size] */
    float    *y_flat;           /* flat copy of y[] [nv] */
    float    *ref_xyz;          /* reference coordinates [na*3] */
    int       na;               /* number of atoms */
    int       nv, size;         /* simplex dimensions */
    float     trans_step, rot_step, tors_step;
    /* Pointers for cleanup (preserved from do_minimize allocation) */
    float   **p;
    float    *y;
    float    *pr;
    float    *prr;
    float    *pbar;
    float    *old_vertex;
};

/* Shared batch parameters — set once per batch (by the first queued call).
 * These must be identical for all slots in the batch (same molecule,
 * same growth layer, same max_iterations/score_converge). */
struct GpuBatchParams {
    std::vector<float>  ref_xyz;
    std::vector<int>    active_flags;
    int    num_atoms;
    int    num_active_atoms;
    int    num_torsions;
    std::vector<int>    ta1, ta2, ta3, ta4, torsion_scale_factors;
    std::vector<int>    child_idx, child_starts, child_counts;
    int    max_iterations;
    float  score_converge;
};

static std::vector<GpuBatchSlot> s_batch_queue;
static GpuBatchParams *s_batch_params = nullptr;
static bool s_batch_enabled = false;


void enable_gpu_batch_mode(bool enabled)
{
    s_batch_enabled = enabled;
    if (!enabled && !s_batch_queue.empty()) {
        /* Cleanup without mol update — needed if e.g. an error occurred mid-batch */
        fprintf(stderr, "GPU-DOCK: enable_gpu_batch_mode(false) with %zu queued slots — discarding\n",
                s_batch_queue.size());
        for (auto &slot : s_batch_queue) {
            delete[] slot.p_flat; delete[] slot.y_flat; delete[] slot.ref_xyz;
            for (int i = 0; i < slot.nv; i++) { delete[] slot.p[i]; }
            delete[] slot.p; delete[] slot.y;
            delete[] slot.pr; delete[] slot.prr; delete[] slot.pbar;
            delete[] slot.old_vertex;
        }
        delete s_batch_params;
        s_batch_params = nullptr;
        s_batch_queue.clear();
    }
}


int flush_gpu_batch(Minimizer &min)
{
    if (s_batch_queue.empty() || !s_batch_params) return 0;

    int N = (int)s_batch_queue.size();

    GpuBatchSlot &first = s_batch_queue[0];
    int nv = first.nv;
    int size = first.size;

    /* Build flat batch arrays */
    std::vector<float> dof_flat((size_t)N * (size_t)nv * (size_t)size);
    std::vector<float> score_flat((size_t)N * (size_t)nv);
    std::vector<int>   state_flat((size_t)N * 6, 0);

    for (int b = 0; b < N; b++) {
        GpuBatchSlot &slot = s_batch_queue[b];
        int dof_off = b * nv * size;
        int scr_off = b * nv;
        memcpy(&dof_flat[dof_off], slot.p_flat, (size_t)nv * (size_t)size * sizeof(float));
        memcpy(&score_flat[scr_off], slot.y_flat, (size_t)nv * sizeof(float));
    }

    /* Use shared parameters from the first batch */
    GpuBatchParams &bp = *s_batch_params;

    /* Dispatch all N conformers in one command buffer */
    bool ok = dock_gpu_simplex_minimize_batch(
        N,
        bp.ref_xyz.data(), bp.active_flags.data(),
        bp.num_atoms, bp.num_active_atoms, bp.num_torsions,
        bp.ta1.data(), bp.ta2.data(), bp.ta3.data(), bp.ta4.data(),
        bp.child_idx.data(), bp.child_starts.data(), bp.child_counts.data(),
        bp.torsion_scale_factors.data(),
        dof_flat.data(), score_flat.data(), state_flat.data(),
        N,
        size, nv,
        bp.max_iterations, bp.score_converge,
        first.trans_step, first.rot_step, first.tors_step);

    if (!ok) {
        fprintf(stderr, "GPU-DOCK: batch simplex minimize failed for %d conformers\n", N);
    }

    /* ---- Unpack results and update molecules ---- */
    for (int b = 0; b < N; b++) {
        GpuBatchSlot &slot = s_batch_queue[b];

        if (!ok || state_flat[b * 6 + 5] == -1) {
            slot.y_flat[0] = 1e30f;
        }

        /* Find best vertex (ilo) */
        int ilo = 0;
        for (int i = 1; i < nv; i++)
            if (score_flat[b * nv + i] < score_flat[b * nv + ilo]) ilo = i;

        /* Update mol with best solution:
         *   Restore reference coords from saved ref_xyz, then apply DOF
         *   using the minimizer's scale_vector + vector_to_dockmol. */
        DOCKMol *m = slot.mol;
        for (int ai = 0; ai < slot.na; ai++) {
            m->x[ai] = slot.ref_xyz[ai*3];
            m->y[ai] = slot.ref_xyz[ai*3+1];
            m->z[ai] = slot.ref_xyz[ai*3+2];
        }

        /* Scale DOF values by step sizes (inline of scale_vector) */
        FLOATVec new_vec;
        for (int i = 0; i < 3 && i < size; i++)
            new_vec.push_back(slot.p_flat[ilo * size + i] * slot.trans_step);
        for (int i = 3; i < 6 && i < size; i++)
            new_vec.push_back(slot.p_flat[ilo * size + i] * slot.rot_step);
        for (int i = 6; i < size; i++)
            new_vec.push_back(slot.p_flat[ilo * size + i] * slot.tors_step);
        min.vector_to_dockmol(*m, new_vec);

        /* ---- Cleanup this slot's allocations ---- */
        delete[] slot.p_flat;
        delete[] slot.y_flat;
        delete[] slot.ref_xyz;
        for (int i = 0; i < slot.nv; i++) { delete[] slot.p[i]; }
        delete[] slot.p;
        delete[] slot.y;
        delete[] slot.pr;
        delete[] slot.prr;
        delete[] slot.pbar;
        delete[] slot.old_vertex;
    }

    delete s_batch_params;
    s_batch_params = nullptr;
    s_batch_queue.clear();

    return N;
}


float
Simplex_Minimizer::do_minimize(Base_Score & score, DOCKMol & mol,
                                    FLOATVec & vertex, int max_iterations,
                                    float score_converge, float trans_step_size,
                                    float rot_step_size, float tors_step_size)
{
    // This is the function the does the work!! not a wrapper function.


    int             iteration;
    int             size;

    float           delta = 0.0;

    // old variables
    int             i,
                    j,
                    x;
    int             ihi,
                    inhi;
    int             ilo = 0;
    float         **p;
    float          *pr;
    float          *prr;
    float          *pbar;
    float          *y;

    float           ypr = 0;
    float           yprr = 0;
    float           alpha = 1.0;        /* range: 0=no extrap, 1=unit step
                                         * extrap, higher OK */
    float           beta = 0.5; /* range: 0=no contraction, 1=full contraction */
    float           optimum;
    int             replace_flag;       /* flag for whether bad vertex replaced 
                                         */
    // end old vars

    FLOATVec        new_vec;
    DOCKMol         tmp_mol,
                    ref_mol,
                    min_mol;
    DOCKMol         rmsd_ref; // this is used to restrain min to starting position.

    double          Econstraint;

    /* GPU batch flag set once per call */
    bool            use_gpu = dock_gpu_is_active();

    float          *old_vertex;

    /* Cleanup lambda to avoid duplicating the 25-line failure exit pattern */
    auto failure_exit = [&](float opt) -> float {
        for (x = 0; x < size; x++) vertex[x] = p[ilo][x];
        if (restrained_min)
            Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, ref_mol);
        else
            Econstraint = 0.0;
        optimum = opt;
        copy_crds(min_mol, ref_mol);
        scale_vector(new_vec, vertex, trans_step_size, rot_step_size, tors_step_size);
        vector_to_dockmol(min_mol, new_vec);
        copy_crds(mol, min_mol);
        for (x = 0; x < size + 1; x++) { delete[]p[x]; p[x] = NULL; }
        delete[]p; p = NULL;
        delete[]y; y = NULL;
        delete[]pr; pr = NULL;
        delete[]prr; prr = NULL;
        delete[]pbar; pbar = NULL;
        delete[]old_vertex; old_vertex = NULL;
        delta -= optimum;
        return optimum;
    };
    float           temp1,
                    temp2;

    float           diff;
    int             step_count;
    int             fail_count;

    // move to main dock loop
    //srand(random_seed); // reset the seed so that molecule order in the file does not mater. 

    size = vertex.size();

    // allocate arrays
    old_vertex = new float[size];
    memset(old_vertex, '\0', sizeof(float) * size);

    p = new float  *[size + 1];
    memset(p, '\0', sizeof(float *) * (size + 1));
    for (i = 0; i < size + 1; i++) {
        p[i] = new float[size];
        memset(p[i], '\0', sizeof(float) * size);
    }
    y = new float[size + 1];
    memset(y, '\0', sizeof(float) * (size + 1));
    pr = new float[size];
    memset(pr, '\0', sizeof(float) * size);
    prr = new float[size];
    memset(prr, '\0', sizeof(float) * size);
    pbar = new float[size];
    memset(pbar, '\0', sizeof(float) * size);

    // End allocation

    // copy molecules
    copy_molecule(ref_mol, mol);
    copy_molecule(min_mol, mol);
    copy_molecule(tmp_mol, mol);
    copy_molecule(rmsd_ref, mol);

    iteration = 0;

    do {

        // initialize all the simplex points
        if (iteration == 0) {

/***
            if((use_mc_premin)&&(!mc_premin_override)) {

                // Monte Carlo Preminimizer
                // Used to generate initial simplex points

                for(i=0;i<size;i++)
                    old_vertex[i] = vertex[i];

                temp2 = eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size, rot_step_size, tors_step_size);
                int count_v = 0;
                fail_count = 0;
                step_count = 0;
                bool accept_flag = false;

                do {

                    temp1 = eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size, rot_step_size, tors_step_size);
                    diff = temp1-temp2;
                    step_count++;

                    if((diff < 0)||( exp(-diff) > r_value )||(step_count > max_steps)) {
                        for(i=0;i<size;i++)
                            old_vertex[i] = vertex[i];

                        temp2 = temp1;

                        if((fail_count > fail_threshold)||(step_count > max_steps))
                            accept_flag = true;
                        fail_count = 0;

                        if(accept_flag) {
                            for(i=0;i<size;i++)
                                p[count_v][i] = old_vertex[i];

                            count_v++;
                        }

                    } else
                        fail_count++;

                    for(i=0;i<size;i++)
                        vertex[i] = old_vertex[i] + 0.5*(((float)rand()/(float)RAND_MAX) - 0.5); // 1.0 rather than 2.0 works better

                } while( count_v < (size+1));

            } else {

                // generate random initial simplex points

                for(i=0;i<size;i++)
                    p[0][i] = vertex[i];

                for(i=1;i<size+1;i++) {
                    for(j=0;j<size;j++) {
                        p[i][j] = p[0][j] + 2.0*(((float)rand()/(float)RAND_MAX) - 0.5);
                    }
                }

            }
***/

            // generate random initial simplex points
            for (i = 0; i < size; i++) {
                p[0][i] = vertex[i];
                //p[0][i] = 0.0;
            }

            for (i = 1; i < size + 1; i++) {
                for (j = 0; j < size; j++) {
                    p[i][j] =
                        //p[0][j] + 2.0 * (((float) rand() / (float) RAND_MAX) -
                        vertex[j] + 2.0 * (((float) rand() / (float) RAND_MAX) -
                                         0.5);
                }
            }

            // score initial simplex points — GPU batch (or CPU fallback)
            Econstraint = 0.0;
            if (use_gpu) {
                /* Collect all N+1 initial vertices into a batch */
                std::vector<FLOATVec> init_verts;
                for (i = 0; i < size + 1; i++) {
                    FLOATVec v(size);
                    for (j = 0; j < size; j++) v[j] = p[i][j];
                    init_verts.push_back(v);
                }
                if (restrained_min)
                    Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, ref_mol);
                float* init_scores = new float[size + 1];
                bool gpu_ok = gpu_batch_eval_scores(score, ref_mol, tmp_mol,
                                                      init_verts, trans_step_size,
                                                      rot_step_size, tors_step_size,
                                                      (float)Econstraint, init_scores);
                if (gpu_ok) {
                    for (i = 0; i < size + 1; i++) y[i] = init_scores[i];
                    delete[] init_scores;
                } else {
                    /* GPU batch scoring failed (e.g. IE data not uploaded
                       for rigid path) — fall back to CPU scoring */
                    delete[] init_scores;
                    for (i = 0; i < size + 1; i++) {
                        for (j = 0; j < size; j++) vertex[j] = p[i][j];
                        if (eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size,
                             rot_step_size, tors_step_size)) {
                             if (restrained_min)
                                 Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                             else
                                 Econstraint = 0.0;
                             y[i] = tmp_mol.current_score + tmp_mol.internal_energy + Econstraint;
                        } else {
                             return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                        }
                    }
                }
            } else {
                /* CPU fallback: score one by one */
                for (i = 0; i < size + 1; i++) {
                    for (j = 0; j < size; j++) vertex[j] = p[i][j];
                    if (eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size,
                         rot_step_size, tors_step_size)) {
                         if (restrained_min)
                             Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                         else
                             Econstraint = 0.0;
                         y[i] = tmp_mol.current_score + tmp_mol.internal_energy + Econstraint;
                    } else {
                         return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                    }
                }
            }
            delta = y[0];

            /* ---- GPU-side simplex: run all iterations on GPU ---- */
            /* B2: upload ligand atom data + IE params once (static guard).
               B3: reuse scratch buffers across calls (vector capacity,
               no per-call new[]/delete[]). */
            if (use_gpu) {
                /* ---- B3: static-reuse scratch buffers ---- */
                static std::vector<float> s_ref_xyz;
                static std::vector<int>   s_active;
                static std::vector<int>   s_ta1, s_ta2, s_ta3, s_ta4, s_tbn;
                static std::vector<int>   s_child_starts, s_child_counts, s_child_idx;
                static std::vector<float> s_dof, s_score;
                /* B2: upload-once guard */
                static bool s_ligand_uploaded = false;
                /* ---- */

                int na = ref_mol.num_atoms;
                int nv = size + 1;
                int nt = (int)torsions.size();

                /* Grow re-usable ref/active buffers if needed */
                s_ref_xyz.resize((size_t)na * 3);
                s_active.resize(na);
                for (i = 0; i < na; i++) {
                    s_ref_xyz[i*3]   = ref_mol.x[i];
                    s_ref_xyz[i*3+1] = ref_mol.y[i];
                    s_ref_xyz[i*3+2] = ref_mol.z[i];
                    s_active[i] = ref_mol.atom_active_flags[i] ? 1 : 0;
                }

                /* Torsion definitions */
                s_ta1.resize(nt);  s_ta2.resize(nt);
                s_ta3.resize(nt);  s_ta4.resize(nt);  s_tbn.resize(nt);
                for (i = 0; i < nt; i++) {
                    s_ta1[i] = torsions[i].atom1;
                    s_ta2[i] = torsions[i].atom2;
                    s_ta3[i] = torsions[i].atom3;
                    s_ta4[i] = torsions[i].atom4;
                    s_tbn[i] = torsions[i].bond_num;
                }

                /* Child atom lists for each torsion */
                bool has_dir_torsions = false;
                for (i = 0; i < nt; i++) {
                    if (bond_vectors[s_tbn[i]] != -1) { has_dir_torsions = true; break; }
                }

                bool gpu_simplex_ok = false;
                if (!has_dir_torsions) {
                    s_child_starts.resize(nt);
                    s_child_counts.resize(nt);
                    int total_children = 0;
                    for (i = 0; i < nt; i++) {
                        int bond_idx = ref_mol.get_bond(s_ta2[i], s_ta3[i]);
                        int cl_idx = (s_ta2[i] < s_ta3[i]) ? 2 * bond_idx : 2 * bond_idx + 1;
                        s_child_counts[i] = (int)ref_mol.atom_child_list[cl_idx].size();
                        total_children += s_child_counts[i];
                    }
                    s_child_idx.resize(total_children > 0 ? total_children : 1);
                    int off = 0;
                    for (i = 0; i < nt; i++) {
                        s_child_starts[i] = off;
                        int bond_idx = ref_mol.get_bond(s_ta2[i], s_ta3[i]);
                        int cl_idx = (s_ta2[i] < s_ta3[i]) ? 2 * bond_idx : 2 * bond_idx + 1;
                        for (j = 0; j < s_child_counts[i]; j++)
                            s_child_idx[off++] = ref_mol.atom_child_list[cl_idx][j];
                    }

                    /* Pack p[][] and y[] into flat vector buffers */
                    s_dof.resize((size_t)nv * (size_t)size);
                    for (i = 0; i < nv; i++)
                        for (j = 0; j < size; j++)
                            s_dof[i * size + j] = p[i][j];
                    s_score.resize(nv);
                    for (i = 0; i < nv; i++) s_score[i] = y[i];

                    /* Upload ligand atom parameters to GPU — ONCE (B2) */
                    if (!s_ligand_uploaded) {
                        std::vector<float> gpu_vdwA(na), gpu_vdwB(na), gpu_chg(na);
                        for (int ai = 0; ai < na; ai++) {
                            int type = ref_mol.amber_at_id[ai];
                            gpu_vdwA[ai] = score.vdwA[type];
                            gpu_vdwB[ai] = score.vdwB[type];
                            gpu_chg[ai]  = ref_mol.charges[ai];
                        }
                        dock_gpu_set_ligand(gpu_vdwA.data(), gpu_vdwB.data(), gpu_chg.data(), na);
                        if (!skip_internal_energy && score.use_internal_energy && (int)score.nb_int.size() > 0) {
                            std::vector<float> gpu_ie_vdwA(na);
                            for (int ai = 0; ai < na; ai++)
                                gpu_ie_vdwA[ai] = score.ie_vdwA[ai];
                            /* C5: Pre-filter nb_int to active-only pairs, removing the branch in the
                             * GPU kernel's IE hot loop.  Inactive atoms never contribute to IE, so
                             * skipping them here reduces wasted SIMD lanes and removes a branch. */
                            int np = (int)score.nb_int.size();
                            std::vector<int> gpu_nb_flat;
                            int filtered_count = 0;
                            for (int pi = 0; pi < np; pi++) {
                                int a1 = score.nb_int[pi].first;
                                int a2 = score.nb_int[pi].second;
                                if (s_active[a1] && s_active[a2]) {
                                    gpu_nb_flat.push_back(a1);
                                    gpu_nb_flat.push_back(a2);
                                    filtered_count++;
                                }
                            }
                            dock_gpu_set_ligand_ie(gpu_ie_vdwA.data(), NULL, gpu_nb_flat.data(), filtered_count,
                                                   score.ie_soft_delta, score.ie_vdw_cutoff_sq);
                        }
                        dock_gpu_simplex_init();
                        s_ligand_uploaded = true;
                    }

                    if (s_batch_enabled) {
                        /* ---- C1: Queue for later GPU batch dispatch ---- */
                        /* Save shared params on first call */
                        if (!s_batch_params) {
                            s_batch_params = new GpuBatchParams;
                            s_batch_params->ref_xyz = s_ref_xyz;
                            s_batch_params->active_flags = s_active;
                            s_batch_params->num_atoms = na;
                            s_batch_params->num_active_atoms = ref_mol.num_active_atoms;
                            s_batch_params->num_torsions = nt;
                            s_batch_params->ta1 = s_ta1;
                            s_batch_params->ta2 = s_ta2;
                            s_batch_params->ta3 = s_ta3;
                            s_batch_params->ta4 = s_ta4;
                            s_batch_params->torsion_scale_factors = torsion_scale_factors;
                            s_batch_params->child_idx = s_child_idx;
                            s_batch_params->child_starts = s_child_starts;
                            s_batch_params->child_counts = s_child_counts;
                            s_batch_params->max_iterations = max_iterations;
                            s_batch_params->score_converge = score_converge;
                        }

                        /* Save this slot */
                        GpuBatchSlot slot;
                        slot.mol = &mol;
                        slot.p_flat = new float[(size_t)nv * (size_t)size];
                        memcpy(slot.p_flat, s_dof.data(), (size_t)nv * (size_t)size * sizeof(float));
                        slot.y_flat = new float[(size_t)nv];
                        memcpy(slot.y_flat, s_score.data(), (size_t)nv * sizeof(float));
                        slot.ref_xyz = new float[(size_t)na * 3];
                        memcpy(slot.ref_xyz, s_ref_xyz.data(), (size_t)na * 3 * sizeof(float));
                        slot.na = na;
                        slot.nv = nv; slot.size = size;
                        slot.trans_step = trans_step_size;
                        slot.rot_step   = rot_step_size;
                        slot.tors_step  = tors_step_size;
                        slot.p = p; slot.y = y;
                        slot.pr = pr; slot.prr = prr; slot.pbar = pbar;
                        slot.old_vertex = old_vertex;

                        s_batch_queue.push_back(slot);

                        return 0.0f;  /* Early return — no cleanup, no mol update */
                    }

                    /* ---- C1: Immediate GPU dispatch (non-batch path) ---- */
                    float eff_trans = trans_step_size / (float)(current_cycle + 1);
                    float eff_rot   = rot_step_size   / (current_cycle + 1);
                    float eff_tors  = tors_step_size  / (current_cycle + 1);

                    gpu_simplex_ok = dock_gpu_simplex_minimize(
                        s_ref_xyz.data(), s_active.data(), na,
                        ref_mol.num_active_atoms, nt,
                        s_ta1.data(), s_ta2.data(), s_ta3.data(), s_ta4.data(),
                        s_child_idx.data(), s_child_starts.data(), s_child_counts.data(),
                        torsion_scale_factors.data(),
                        s_dof.data(), s_score.data(), size, nv,
                        max_iterations, score_converge,
                        eff_trans, eff_rot, eff_tors);

                    if (gpu_simplex_ok) {
                        for (i = 0; i < nv; i++) {
                            for (j = 0; j < size; j++)
                                p[i][j] = s_dof[i * size + j];
                            y[i] = s_score[i];
                        }
                        ilo = 0;
                        for (i = 1; i < nv; i++) if (y[i] < y[ilo]) ilo = i;
                        ihi = 0;
                        for (i = 1; i < nv; i++) if (y[i] > y[ihi]) ihi = i;
                        iteration = max_iterations + 1;
                    }
                }

                if (!gpu_simplex_ok) {
                    use_gpu = false;
                } else {
                    goto end_of_simplex;
                }
            }

            /* ---- CPU path: fallback (original code below) ---- */
        } else {

            // Begin a new iteration
            for (i = 0; i < size; i++) pbar[i] = 0.0;
            // compute vector ave. of all points except the highest
            for (i = 0; i < size + 1; i++) {
                if (i != ihi) {
                    for (j = 0; j < size; j++) pbar[j] += p[i][j];
                }
            }
            // extrapolate by a factor alpha through the face
            for (i = 0; i < size; i++) {
                pbar[i] /= (float)size;
                pr[i] = (1.0 + alpha) * pbar[i] - alpha * p[ihi][i];
            }

            /* ---- Pre-compute all candidate vertex vectors for GPU speculative batch ---- */
            /* Candidates:
               0: reflected point (pr)  — always needed
               1: expanded point (prr_exp = (1+alpha)*pr - alpha*pbar)
               2: contracted-with-pr (prr_cA = beta*pr + (1-beta)*pbar)
               3: contracted-with-orig (prr_cB = beta*p[ihi] + (1-beta)*pbar)
            */
            {
            int use_speculative = use_gpu;
            float batch_scores[4];
            FLOATVec prr_exp(size), prr_cA(size), prr_cB(size);
            for (i = 0; i < size; i++) {
                prr_exp[i] = (1.0 + alpha) * pr[i] - alpha * pbar[i];
                prr_cA[i]  = beta * pr[i] + (1.0 - beta) * pbar[i];
                prr_cB[i]  = beta * p[ihi][i] + (1.0 - beta) * pbar[i];
            }

            if (use_speculative) {
                /* GPU: batch-score all 4 candidates */
                std::vector<FLOATVec> spec_verts;
                FLOATVec pr_v;
                pr_v.assign(pr, pr + size);
                spec_verts.push_back(pr_v);       /* idx 0: reflected */
                spec_verts.push_back(prr_exp);    /* idx 1: expanded */
                spec_verts.push_back(prr_cA);     /* idx 2: contracted-with-pr */
                spec_verts.push_back(prr_cB);     /* idx 3: contracted-with-orig */

                Econstraint = 0.0;
                if (restrained_min)
                    Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, ref_mol);

                bool gpu_ok = gpu_batch_eval_scores(score, ref_mol, tmp_mol,
                                                      spec_verts, trans_step_size,
                                                      rot_step_size, tors_step_size,
                                                      (float)Econstraint, batch_scores);
                if (!gpu_ok) {
                    /* GPU failed — fall back to CPU sequential scoring */
                    use_speculative = 0;
                }
            }

            if (!use_speculative) {
                /* CPU fallback: score reflected point */
                for (i = 0; i < size; i++) vertex[i] = pr[i];
                if (eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size,
                     rot_step_size, tors_step_size)) {
                     if (restrained_min)
                         Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                     else
                         Econstraint = 0.0;
                     batch_scores[0] = tmp_mol.current_score + tmp_mol.internal_energy + (float)Econstraint;
                } else {
                     return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                }
            }

            ypr = batch_scores[0];  /* reflected score */

            if (ypr <= y[ilo]) {
                /* ---- Expansion path ---- */
                float yprr_exp;
                if (!use_speculative) {
                    /* CPU fallback: score expanded point */
                    for (i = 0; i < size; i++) vertex[i] = prr_exp[i];
                    if (eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size,
                         rot_step_size, tors_step_size)) {
                         if (restrained_min)
                             Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                         else
                             Econstraint = 0.0;
                         yprr_exp = tmp_mol.current_score + tmp_mol.internal_energy + (float)Econstraint;
                    } else {
                         return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                    }
                } else {
                    yprr_exp = batch_scores[1];
                }

                if (yprr_exp < y[ilo]) {
                    for (i = 0; i < size; i++) p[ihi][i] = prr_exp[i];
                    y[ihi] = yprr_exp;
                } else {
                    for (i = 0; i < size; i++) p[ihi][i] = pr[i];
                    y[ihi] = ypr;
                }

            } else if (ypr >= y[inhi]) {
                /* ---- Contraction / Shrink path ---- */
                replace_flag = false;

                if (ypr < y[ihi]) {
                    for (i = 0; i < size; i++) p[ihi][i] = pr[i];
                    y[ihi] = ypr;
                    replace_flag = true;
                }

                float yprr_contract;
                if (!use_speculative) {
                    /* CPU fallback: score contraction */
                    for (i = 0; i < size; i++) vertex[i] = prr[i] = beta * p[ihi][i] + (1.0 - beta) * pbar[i];
                    if (eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size,
                         rot_step_size, tors_step_size)) {
                         if (restrained_min)
                             Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                         else
                             Econstraint = 0.0;
                         yprr_contract = tmp_mol.current_score + tmp_mol.internal_energy + (float)Econstraint;
                    } else {
                         return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                    }
                } else {
                    /* Pick the right contraction variant based on whether ypr < y[ihi] */
                    const FLOATVec& contr_vec = (ypr < y[ihi]) ? prr_cA : prr_cB;
                    yprr_contract = (ypr < y[ihi]) ? batch_scores[2] : batch_scores[3];
                    /* Copy into prr for compatibility with downstream code */
                    for (i = 0; i < size; i++) prr[i] = contr_vec[i];
                }

                if (yprr_contract < y[ihi]) {
                    for (i = 0; i < size; i++) p[ihi][i] = prr[i];
                    y[ihi] = yprr_contract;
                    replace_flag = true;
                }

                if (replace_flag == false) {
                    /* SHRINK — can't eliminate high point */
                    /* Collect shrink vertices and batch-score them */
                    std::vector<FLOATVec> shrink_verts;
                    for (i = 0; i < size + 1; i++) {
                        if (i != ilo) {
                            FLOATVec sv(size);
                            for (j = 0; j < size; j++) {
                                sv[j] = p[i][j] = 0.5 * (p[i][j] + p[ilo][j]);
                            }
                            shrink_verts.push_back(sv);
                        }
                    }

                    bool shrink_ok = false;
                    if (use_gpu) {
                        Econstraint = 0.0;
                        if (restrained_min)
                            Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, ref_mol);
                        float *shrink_scores = new float[shrink_verts.size()];
                        shrink_ok = gpu_batch_eval_scores(score, ref_mol, tmp_mol,
                                                            shrink_verts, trans_step_size,
                                                            rot_step_size, tors_step_size,
                                                            (float)Econstraint, shrink_scores);
                        if (shrink_ok) {
                            int si = 0;
                            for (i = 0; i < size + 1; i++) {
                                if (i != ilo) y[i] = shrink_scores[si++];
                            }
                        }
                        delete[] shrink_scores;
                    }

                    if (!shrink_ok) {
                        /* CPU fallback: score shrink vertices one by one */
                        for (i = 0; i < size + 1; i++) {
                            if (i != ilo) {
                                for (j = 0; j < size; j++) vertex[j] = p[i][j];
                                if (eval_score(score, ref_mol, tmp_mol, vertex,
                                     trans_step_size, rot_step_size, tors_step_size)) {
                                    if (restrained_min)
                                        Econstraint = coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                                    else
                                        Econstraint = 0.0;
                                    y[i] = tmp_mol.current_score + tmp_mol.internal_energy + (float)Econstraint;
                                } else {
                                    return failure_exit(ref_mol.current_score + ref_mol.internal_energy + Econstraint);
                                }
                            }
                        }
                    }
                }
            } else {
                /* ---- Middling: accept reflected point ---- */
                for (i = 0; i < size; i++) p[ihi][i] = pr[i];
                y[ihi] = ypr;
            }
            }  // end speculative-scope block
        }

        // ID Best & Worst vertices in current simplex

        if (y[0] > y[1]) {
            ihi = 0;
            inhi = 1;
        } else {
            ihi = 1;
            inhi = 0;
        }

        // loop over simplex points
        for (i = 0; i < size + 1; i++) {

            if (y[i] < y[ilo]){
                ilo = i;
            }
            if (y[i] > y[ihi]) {
                inhi = ihi;
                ihi = i;
            } else if (y[i] > y[inhi]) {
                if (i != ihi)
                    inhi = i;
            }
        }

        
/**
        // print out simplex trajectory
        for(i=0;i<size;i++)
            vertex[i] = prr[i] = p[ilo][i];

        copy_molecule(min_mol, ref_mol);
        scale_vector(new_vec, vertex, trans_step_size, rot_step_size, tors_step_size);
        vector_to_dockmol(min_mol, new_vec);
        //Write_Mol2(min_mol, cout);

        // DTM - 11-12-08 output simplex scores!
        score.compute_score(min_mol);
        // end print out of trajectory
**/

    } while ((iteration++ < max_iterations)
             && (fabs(y[ihi] - y[ilo]) > score_converge));

end_of_simplex:
    // store best scoring vertex
    for (i = 0; i < size; i++){
        vertex[i] = p[ilo][i];
    }
    optimum = y[ilo];

/***
    // Monte Carlo Post-minimizer
    if(use_mc_postmin) {

        for(i=0;i<size;i++)
            old_vertex[i] = vertex[i];

        temp2 = eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size, rot_step_size, tors_step_size);
        fail_count = 0;
        step_count = 0;

        float ave_score = 0.0;
        int ave_count = 0;

        do {

            temp1 = eval_score(score, ref_mol, tmp_mol, vertex, trans_step_size, rot_step_size, tors_step_size);
            diff = temp1-temp2;
            step_count++;

            if((diff < 0)||( exp(-diff) > r_value )) {
                for(i=0;i<size;i++)
                    old_vertex[i] = vertex[i];

                ave_score += temp1;
                ave_count++;

                temp2 = temp1;

            } else
                fail_count++;

            for(i=0;i<size;i++)
                vertex[i] = old_vertex[i] + 0.5*(((float)rand()/(float)RAND_MAX) - 0.5);

        } while((step_count < max_steps)&&(fail_count < fail_threshold));

        for(i=0;i<size;i++)
                vertex[i] = old_vertex[i];

        optimum = temp2;
    }
***/

    // copy best mol to min_mol, and generate min structure
    copy_crds(min_mol, ref_mol);
    scale_vector(new_vec, vertex, trans_step_size, rot_step_size,
                         tors_step_size);
    vector_to_dockmol(min_mol, new_vec);
    copy_crds(mol, min_mol);

    // free arrays
    for (i = 0; i < size + 1; i++) {
        delete[]p[i];
        p[i] = NULL;
    }

    delete[]p;
    p = NULL;

    delete[]y;
    y = NULL;

    delete[]pr;
    pr = NULL;

    delete[]prr;
    prr = NULL;

    delete[]pbar;
    pbar = NULL;

    delete[]old_vertex;
    old_vertex = NULL;

    delta -= optimum;
    //cout << "Optimum= " << optimum << endl;
    return optimum;

}
