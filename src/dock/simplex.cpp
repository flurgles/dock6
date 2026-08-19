#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "master_score.h"
#include "minimizer.h"
#include "simplex.h"
#include "conf_gen_ag.h"
#include "score_dock_gpu.h"

using namespace std;

/* input_parameters moved to simplex_input.cpp */

void
Simplex_Minimizer::initialize()
{
    //cout << "Initializing simplex" << endl;
    srand(random_seed);

}
/* enable_gpu_batch_mode / flush_gpu_batch removed in P2 — ConformerPool manages GPU work */


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
    int             ihi = 0;   // index of highest (worst)  score in simplex  (Nelder & Mead, Numerical Recipes)
    int             inhi = 0;  // index of 2nd-highest (2nd-worst) score     (Numerical Recipes)
    int             ilo = 0;   // index of lowest  (best)   score in simplex  (Nelder & Mead, Numerical Recipes)
    float         **p;
    float          *pr;
    float          *prr;
    float          *pbar;
    float          *y;

    float           ypr = 0;
    float           yprr = 0;
    // Classical Nelder-Mead coefficients (Lagarias et al. 1998).  In the
    // non-adaptive case alpha (reflection) doubles as the expansion step
    // size via (1+alpha)*pr - alpha*pbar, giving an effective gamma=2.0.
    // The adaptive case (Gao & Han 2012) uses a separate gamma.
    float           alpha = 1.0;        /* reflection (always 1.0 in Nelder-Mead) */
    float           gamma = 2.0;        /* expansion  (standard 2.0; adaptive: 1+2/n) */
    float           beta  = 0.5;        /* contraction/rho (standard 0.5; adaptive: 0.75-0.5/n) */
    float           sigma = 0.5;        /* shrink (standard 0.5; adaptive: 1-1/n) */
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

    // Adaptive Nelder-Mead (Gao & Han 2012): controls the expansion (gamma),
    // contraction (rho/beta), and shrink (sigma) coefficients.
    //   simplex_mode=0 (no):     classical fixed coefficients — gamma=2.0,
    //                            beta=0.5, sigma=0.5 (bit-identical to
    //                            prior DOCK versions).
    //   simplex_mode=1 (yes):    full adaptive — gamma=1+2/n, rho=0.75-0.5/n,
    //                            sigma=1-1/n.
    //   simplex_mode=2 (dim_aware): sigmoid blend from fixed (low n) to
    //                            adaptive (high n), centered at
    //                            simplex_crossover + 6 DOF.
    // Reflection (alpha=1.0) is never adapted.
    // Reference:
    //   Gao, F. & Han, L. (2012), Comput. Optim. Appl. 51(1), 259--277,
    //   DOI: 10.1007/s10589-010-9329-3
    if (simplex_mode == 1) {
        const float n = (float) size;
        gamma = 1.0f + 2.0f / n;
        beta  = 0.75f - 0.5f / n;
        sigma = 1.0f  - 1.0f  / n;
    } else if (simplex_mode == 2) {
        // Dimension-aware blend: sigmoid blends from fixed (w=1 at n_path)
        // to full adaptive (w=0 at high n), centered at crossover+6 DOF.
        const float n    = (float) size;
        const float n0   = (float)(simplex_crossover + 6);
        const float k    = 0.5f;              // sigmoid steepness
        float w = 1.0f - 1.0f / (1.0f + expf(-k * (n - n0)));
        if (w < 0.0f) w = 0.0f;
        if (w > 1.0f) w = 1.0f;
        const float ga = 1.0f + 2.0f / n;     // adaptive gamma
        const float rb = 0.75f - 0.5f / n;    // adaptive rho/beta
        const float sg = 1.0f  - 1.0f  / n;   // adaptive sigma
        gamma = w * 2.0f + (1.0f - w) * ga;
        beta  = w * 0.5f + (1.0f - w) * rb;
        sigma = w * 0.5f + (1.0f - w) * sg;
    }
    // simplex_mode == 0: use fixed coefficients (gamma=2.0, beta=0.5,
    // sigma=0.5) already set at declaration

    /* Ensure torsion_scale_factors is sized for this DOF count —
       scale_vector() accesses it in the GPU batch path via
       gpu_batch_eval_scores().  Base Minimizer::do_minimize()
       does this resize automatically; Simplex_Minimizer subclass
       must do it explicitly. */
    torsion_scale_factors.resize(size, 1);

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

    char dbg_path_seen = '?';

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
                        vertex[j] + 2.0 * (next_rand_01() -
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

            if (getenv("DOCK_POOL_DEBUG")) {
                fprintf(stderr, "DOSX init n=%d", size + 1);
                for (i = 0; i < size + 1; i++) fprintf(stderr, " y%d=%.5f", i, y[i]);
                fprintf(stderr, "\n");
            }

        }

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
               1: expanded point  (prr_exp = gamma*pr + (1-gamma)*pbar)
               2: contracted-pr   (prr_cA  = beta*pr + (1-beta)*pbar)
               3: contracted-orig (prr_cB  = beta*p[ihi] + (1-beta)*pbar)
               With classical coefficients (gamma=2.0) the expansion reduces to
               the original DOCK form (1+alpha)*pr - alpha*pbar since alpha=1.0.
            */
            {
            int use_speculative = use_gpu;
            float batch_scores[4];
            char dbg_path = '?';
            FLOATVec prr_exp(size), prr_cA(size), prr_cB(size);
            for (i = 0; i < size; i++) {
                prr_exp[i] = gamma * pr[i] + (1.0 - gamma) * pbar[i];
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
                    dbg_path = 'E';
                } else {
                    for (i = 0; i < size; i++) p[ihi][i] = pr[i];
                    y[ihi] = ypr;
                    dbg_path = 'R';
                }

            } else if (ypr >= y[inhi]) {
                /* ---- Contraction / Shrink path ---- */
                replace_flag = false;

                /* Save condition BEFORE overwriting p[ihi]/y[ihi].
                   After the overwrite, `(ypr < y[ihi])` would always be
                   false (ypr < ypr), incorrectly selecting the cB variant. */
                bool outer = (ypr < y[ihi]);

                if (outer) {
                    for (i = 0; i < size; i++) p[ihi][i] = pr[i];
                    y[ihi] = ypr;
                    replace_flag = true;
                    dbg_path = 'O';
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
                    /* Pick the right contraction variant based on whether ypr < y[ihi]_original */
                    const FLOATVec& contr_vec = outer ? prr_cA : prr_cB;
                    yprr_contract = outer ? batch_scores[2] : batch_scores[3];
                    /* Copy into prr for compatibility with downstream code */
                    for (i = 0; i < size; i++) prr[i] = contr_vec[i];
                }

                if (yprr_contract < y[ihi]) {
                    for (i = 0; i < size; i++) p[ihi][i] = prr[i];
                    y[ihi] = yprr_contract;
                    replace_flag = true;
                    dbg_path = 'C';
                }

                if (replace_flag == false) {
                    /* SHRINK — can't eliminate high point */
                    dbg_path = 'S';
                    /* Collect shrink vertices and batch-score them */
                    std::vector<FLOATVec> shrink_verts;
                    for (i = 0; i < size + 1; i++) {
                        if (i != ilo) {
                            FLOATVec sv(size);
                            for (j = 0; j < size; j++) {
                                // Shrink toward best vertex:  x_i <- sigma*x_i + (1-sigma)*x_best
                                // Standard sigma=0.5 gives the classic 0.5*(x_i + x_best).
                                sv[j] = p[i][j] = sigma * p[i][j] + (1.0f - sigma) * p[ilo][j];
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
                dbg_path = 'M';
            }

            dbg_path_seen = dbg_path;
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

    if (getenv("DOCK_POOL_DEBUG")) {
            fprintf(stderr, "DOSX step it=%d path=%c delta=%.6f ylo=%.5f\n",
                    iteration, dbg_path_seen, (double)fabs(y[ihi] - y[ilo]), (double)y[ilo]);
        }

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
