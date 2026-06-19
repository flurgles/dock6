#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "master_score.h"
#include "minimizer.h"
#include "conjugate_gradient.h"

using namespace std;

/******************************************************/
void
Conjugate_Gradient_Minimizer::input_parameters(Parameter_Reader & parm,
                                    bool flexible_ligand, bool genetic_algorithm,
                                    bool denovo_design, Master_Score & score)
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

        cout << "\nConjugate Gradient Minimization Parameters" << endl;
        cout <<
            "------------------------------------------------------------------------------------------"
            << endl;

        minimize_ligand =
            (parm.query_param("minimize_ligand", "yes", "yes no") ==
             "yes") ? true : false;

        if (minimize_ligand) {

            // --- Core algorithm parameters ---
            fd_step = atof(parm.query_param("cg_fd_step", "1.0e-4").c_str());
            if (fd_step <= 0.0) {
                cout << "ERROR:  Parameter must be a float greater than zero.  Program will terminate." << endl;
                exit(0);
            }

            line_search_alpha = atof(parm.query_param("cg_line_search_alpha", "1.0").c_str());
            if (line_search_alpha <= 0.0) {
                cout << "ERROR:  Parameter must be a float greater than zero.  Program will terminate." << endl;
                exit(0);
            }

            line_search_tau = atof(parm.query_param("cg_line_search_tau", "0.5").c_str());
            if (line_search_tau <= 0.0 || line_search_tau >= 1.0) {
                cout << "ERROR:  cg_line_search_tau must be in (0,1).  Program will terminate." << endl;
                exit(0);
            }

            max_line_search = atoi(parm.query_param("cg_max_line_search", "20").c_str());
            if (max_line_search <= 0) {
                cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                exit(0);
            }

            // --- Basic minimization parameters ---
            if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                max_iterations = atoi(parm.query_param("cg_max_iterations", "1000").c_str());
                if (max_iterations < 0) {
                    cout << "ERROR:  cg_max_iterations must be >= 0.  Program will terminate." << endl;
                    exit(0);
                }

                torsion_iterations = atoi(parm.query_param("cg_tors_premin_iterations", "0").c_str());
                if (torsion_iterations < 0) {
                    cout << "ERROR:  cg_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                    exit(0);
                }

                max_cycles = atoi(parm.query_param("cg_max_cycles", "1").c_str());
                if (max_cycles <= 0) {
                    cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                    exit(0);
                }

                score_converge = atof(parm.query_param("cg_score_converge", "0.1").c_str());
                cycle_converge = atof(parm.query_param("cg_cycle_converge", "1.0").c_str());
                trans_step_size = atof(parm.query_param("cg_trans_step", "1.0").c_str());
                rot_step_size = atof(parm.query_param("cg_rot_step", "0.1").c_str());
                tors_step_size = atof(parm.query_param("cg_tors_step", "10.0").c_str());
            }

            // --- Advanced minimization parameters ---
            if (flexible_ligand && (use_min_rigid_anchor || use_min_flex_growth)) {
                advanced_min_params =
                    (parm.query_param("use_advanced_simplex_parameters", "no", "yes no") == "yes") ? true : false;
                if (!advanced_min_params) {
                    if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                        max_iterations = atoi(parm.query_param("cg_max_iterations", "1000").c_str());
                        if (max_iterations < 0) {
                            cout << "ERROR:  cg_max_iterations must be >= 0.  Program will terminate." << endl;
                            exit(0);
                        }
                        torsion_iterations = atoi(parm.query_param("cg_tors_premin_iterations", "0").c_str());
                        if (torsion_iterations < 0) {
                            cout << "ERROR:  cg_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                            exit(0);
                        }
                        max_cycles = atoi(parm.query_param("cg_max_cycles", "1").c_str());
                        if (max_cycles <= 0) {
                            cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                            exit(0);
                        }
                        score_converge = atof(parm.query_param("cg_score_converge", "0.1").c_str());
                        cycle_converge = atof(parm.query_param("cg_cycle_converge", "1.0").c_str());
                        trans_step_size = atof(parm.query_param("cg_trans_step", "1.0").c_str());
                        rot_step_size = atof(parm.query_param("cg_rot_step", "0.1").c_str());
                        tors_step_size = atof(parm.query_param("cg_tors_step", "10.0").c_str());
                    }
                }
                if (advanced_min_params) {
                    max_iterations = atoi(parm.query_param("cg_max_iterations", "1000").c_str());
                    torsion_iterations = atoi(parm.query_param("cg_tors_premin_iterations", "0").c_str());
                    max_cycles = atoi(parm.query_param("cg_max_cycles", "1").c_str());
                    score_converge = atof(parm.query_param("cg_score_converge", "0.1").c_str());
                    cycle_converge = atof(parm.query_param("cg_cycle_converge", "1.0").c_str());
                    trans_step_size = atof(parm.query_param("cg_trans_step", "1.0").c_str());
                    rot_step_size = atof(parm.query_param("cg_rot_step", "0.1").c_str());
                    tors_step_size = atof(parm.query_param("cg_tors_step", "10.0").c_str());
                }

                // parameters for anchor minimization
                if (use_min_rigid_anchor) {
                    anchor_min_max_iterations = atoi(parm.query_param("cg_anchor_max_iterations", "500").c_str());
                    if (anchor_min_max_iterations <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(0);
                    }
                    anchor_min_max_cycles = atoi(parm.query_param("cg_anchor_max_cycles", "1").c_str());
                    if (anchor_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(0);
                    }
                    anchor_min_trans_step_size = atof(parm.query_param("cg_anchor_trans_step", "1.0").c_str());
                    anchor_min_rot_step_size = atof(parm.query_param("cg_anchor_rot_step", "0.1").c_str());
                    anchor_min_tors_step_size = atof(parm.query_param("cg_anchor_tors_step", "10.0").c_str());
                }

                // parameters for flexible grow minimization
                if (use_min_flex_growth) {
                    if (use_min_flex_growth_ramp) {
                        flex_min_max_iterations = atoi(parm.query_param("cg_grow_max_iterations", "250").c_str());
                    } else {
                        flex_min_max_iterations = atoi(parm.query_param("cg_grow_max_iterations", "500").c_str());
                    }
                    if (flex_min_max_iterations < 0) {
                        cout << "ERROR:  cg_grow_max_iterations cannot be negative.  Program will terminate." << endl;
                        exit(0);
                    }
                    flex_min_max_cycles = atoi(parm.query_param("cg_grow_max_cycles", "1").c_str());
                    if (flex_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(0);
                    }
                    flex_min_tors_step_size = atof(parm.query_param("cg_grow_tors_step", "10.0").c_str());
                    flex_min_torsion_iterations = atoi(parm.query_param("cg_grow_tors_premin_iterations", "0").c_str());
                    if (use_min_flex_growth_ramp) {
                        flex_min_ramp_max_iterations = atoi(parm.query_param("cg_grow_ramp_max_iterations", "50").c_str());
                        flex_min_ramp_max_cycles = atoi(parm.query_param("cg_grow_ramp_max_cycles", "1").c_str());
                        flex_min_ramp_tors_step_size = atof(parm.query_param("cg_grow_ramp_tors_step", "10.0").c_str());
                        flex_min_ramp_torsion_iterations = atoi(parm.query_param("cg_grow_ramp_tors_premin_iterations", "0").c_str());
                    }
                }

                // parameters for final minimization
                final_min = (parm.query_param("cg_final_min", "no", "yes no") == "yes") ? true : false;
                if (final_min) {
                    final_min_max_iterations = atoi(parm.query_param("cg_final_max_iterations", "500").c_str());
                    if (final_min_max_iterations <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(0);
                    }
                    final_min_max_cycles = atoi(parm.query_param("cg_final_max_cycles", "1").c_str());
                    if (final_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(0);
                    }
                    final_min_trans_step_size = atof(parm.query_param("cg_final_trans_step", "1.0").c_str());
                    final_min_rot_step_size = atof(parm.query_param("cg_final_rot_step", "0.1").c_str());
                    final_min_tors_step_size = atof(parm.query_param("cg_final_tors_step", "10.0").c_str());
                }

                // secondary minimization
                secondary_min_pose = (parm.query_param("cg_secondary_min_pose", "no", "yes no") == "yes") ? true : false;
                if (secondary_min_pose) {
                    secondary_min_max_iterations = atoi(parm.query_param("cg_secondary_max_iterations", "500").c_str());
                    secondary_min_max_cycles = atoi(parm.query_param("cg_secondary_max_cycles", "1").c_str());
                    secondary_min_score_converge = atof(parm.query_param("cg_secondary_score_converge", "0.1").c_str());
                    secondary_min_cycle_converge = atof(parm.query_param("cg_secondary_cycle_converge", "1.0").c_str());
                    secondary_min_trans_step_size = atof(parm.query_param("cg_secondary_trans_step", "1.0").c_str());
                    secondary_min_rot_step_size = atof(parm.query_param("cg_secondary_rot_step", "0.1").c_str());
                    secondary_min_tors_step_size = atof(parm.query_param("cg_secondary_tors_step", "10.0").c_str());
                }

                // parameters for final restraint
                if (final_min or minimize_ligand) {
                    random_seed = atoi(parm.query_param("cg_random_seed", "0").c_str());
                    restrained_min = (parm.query_param("cg_restraint_min", "no", "yes no") == "yes") ? true : false;
                    if (restrained_min) {
                        coefficient_restraint = atof(parm.query_param("cg_coefficient_restraint", "10.0").c_str());
                    }
                }
            }
        }
    }
}

/******************************************************/
void
Conjugate_Gradient_Minimizer::initialize()
{
    //cout << "Initializing Conjugate Gradient" << endl;
    srand(random_seed);
}

/******************************************************/
// Compute gradient via forward finite differences
void
Conjugate_Gradient_Minimizer::compute_gradient(Base_Score & score, DOCKMol & ref_mol,
                                                DOCKMol & tmp_mol, DOCKMol & rmsd_ref,
                                                FLOATVec & x, FLOATVec & g,
                                                float trans_step_size, float rot_step_size,
                                                float tors_step_size, float h)
{
    int i;
    float fx;

    if (!eval_score(score, ref_mol, tmp_mol, x,
                    trans_step_size, rot_step_size, tors_step_size)) {
        for (i = 0; i < n; i++) g[i] = 0.0f;
        return;
    }
    fx = tmp_mol.current_score + tmp_mol.internal_energy;
    if (restrained_min) {
        fx += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
    }

    for (i = 0; i < n; i++) {
        FLOATVec x_plus = x;
        x_plus[i] += h;
        if (eval_score(score, ref_mol, tmp_mol, x_plus,
                       trans_step_size, rot_step_size, tors_step_size)) {
            float f_plus = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                f_plus += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
            g[i] = (f_plus - fx) / h;
        } else {
            g[i] = 0.0f;
        }
    }
}

/******************************************************/
float
Conjugate_Gradient_Minimizer::do_minimize(Base_Score & score, DOCKMol & mol,
                                           FLOATVec & vertex, int max_iterations,
                                           float score_converge, float trans_step_size,
                                           float rot_step_size, float tors_step_size)
{
    n = (int) vertex.size();
    if (n < 1) {
        cerr << "ERROR: CG: zero degrees of freedom" << endl;
        return 1.0e10f;
    }

    DOCKMol ref_mol, tmp_mol, best_mol, rmsd_ref;
    copy_molecule(ref_mol, mol);
    copy_molecule(best_mol, mol);
    copy_molecule(tmp_mol, mol);
    if (restrained_min) copy_molecule(rmsd_ref, mol);

    FLOATVec x = vertex;
    float fopt;

    // Evaluate starting point
    if (eval_score(score, ref_mol, tmp_mol, x,
                   trans_step_size, rot_step_size, tors_step_size)) {
        fopt = tmp_mol.current_score + tmp_mol.internal_energy;
        if (restrained_min) {
            fopt += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
        }
        copy_crds(best_mol, ref_mol);
    } else {
        cerr << "ERROR: CG: initial scoring failed" << endl;
        return 1.0e10f;
    }

    FLOATVec g(n, 0.0f);        // current gradient
    FLOATVec g_prev(n, 0.0f);   // previous gradient
    FLOATVec d(n, 0.0f);        // search direction
    float fd_h = (fd_step > 0.0f) ? fd_step : 1.0e-4f;
    int iter;

    for (iter = 0; iter < max_iterations; iter++) {

        // Store previous gradient
        for (int i = 0; i < n; i++) g_prev[i] = g[i];

        // Compute gradient at current point
        compute_gradient(score, ref_mol, tmp_mol, rmsd_ref, x, g,
                         trans_step_size, rot_step_size, tors_step_size, fd_h);

        // Check gradient norm for convergence
        float norm_g = 0.0f;
        for (int i = 0; i < n; i++) norm_g += g[i] * g[i];
        norm_g = sqrt(norm_g);

        if (norm_g < score_converge) break;

        // Compute conjugate direction (Polak-Ribière)
        if (iter == 0) {
            // First iteration: steepest descent
            for (int i = 0; i < n; i++) d[i] = -g[i];
        } else {
            // Compute beta = max(0, g^T (g - g_prev) / (g_prev^T g_prev))
            float gdg = 0.0f, gpgp = 0.0f;
            for (int i = 0; i < n; i++) {
                gdg += g[i] * (g[i] - g_prev[i]);
                gpgp += g_prev[i] * g_prev[i];
            }
            float beta = 0.0f;
            if (gpgp > 1.0e-20f) {
                beta = gdg / gpgp;
                if (beta < 0.0f) beta = 0.0f;  // reset if negative
            }

            // d = -g + beta * d_prev
            for (int i = 0; i < n; i++) {
                d[i] = -g[i] + beta * d[i];
            }
        }

        // Reset direction to steepest descent every n iterations (Fletcher-Reeves restart)
        if (iter > 0 && iter % n == 0) {
            for (int i = 0; i < n; i++) d[i] = -g[i];
        }

        // Normalize direction
        float norm_d = 0.0f;
        for (int i = 0; i < n; i++) norm_d += d[i] * d[i];
        norm_d = sqrt(norm_d);
        if (norm_d < 1.0e-20f) break;

        // Backtracking line search along d
        float alpha = line_search_alpha;
        float fnew = fopt;
        bool found = false;

        for (int ls = 0; ls < max_line_search; ls++) {
            FLOATVec x_trial(n);
            for (int i = 0; i < n; i++) {
                x_trial[i] = x[i] + alpha * d[i] / norm_d;
                // Clamp to bounds
                if (x_trial[i] < -1.0f) x_trial[i] = -1.0f;
                if (x_trial[i] > 1.0f) x_trial[i] = 1.0f;
            }

            if (eval_score(score, ref_mol, tmp_mol, x_trial,
                           trans_step_size, rot_step_size, tors_step_size)) {
                fnew = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fnew += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }

                if (fnew < fopt) {
                    x = x_trial;
                    fopt = fnew;
                    copy_crds(best_mol, ref_mol);
                    found = true;
                    break;
                }
            }
            alpha *= line_search_tau;
        }

        if (!found) break;  // Line search failed — converged
    }

    // Copy best result back
    copy_molecule(tmp_mol, mol);
    FLOATVec best_vec;
    scale_vector(best_vec, x, trans_step_size, rot_step_size, tors_step_size);
    vector_to_dockmol(tmp_mol, best_vec);
    copy_crds(mol, tmp_mol);

    return fopt;
}
