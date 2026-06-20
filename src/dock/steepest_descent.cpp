#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "master_score.h"
#include "minimizer.h"
#include "steepest_descent.h"

using namespace std;

/******************************************************/
void
Steepest_Descent_Minimizer::input_parameters(Parameter_Reader & parm,
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

        cout << "\nSteepest Descent Minimization Parameters" << endl;
        cout <<
            "------------------------------------------------------------------------------------------"
            << endl;

        minimize_ligand =
            (parm.query_param("minimize_ligand", "yes", "yes no") ==
             "yes") ? true : false;

        if (minimize_ligand) {

            // --- Core algorithm parameters ---
            fd_step = atof(parm.query_param("sd_fd_step", "1.0e-4").c_str());
            if (fd_step <= 0.0) {
                cout << "ERROR:  Parameter must be a float greater than zero.  Program will terminate." << endl;
                exit(1);
            }

            line_search_alpha = atof(parm.query_param("sd_line_search_alpha", "1.0").c_str());
            if (line_search_alpha <= 0.0) {
                cout << "ERROR:  Parameter must be a float greater than zero.  Program will terminate." << endl;
                exit(1);
            }

            line_search_tau = atof(parm.query_param("sd_line_search_tau", "0.5").c_str());
            if (line_search_tau <= 0.0 || line_search_tau >= 1.0) {
                cout << "ERROR:  sd_line_search_tau must be in (0,1).  Program will terminate." << endl;
                exit(1);
            }

            max_line_search = atoi(parm.query_param("sd_max_line_search", "20").c_str());
            if (max_line_search <= 0) {
                cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                exit(1);
            }

            // --- Basic minimization parameters ---
            if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                max_iterations = atoi(parm.query_param("sd_max_iterations", "1000").c_str());
                if (max_iterations < 0) {
                    cout << "ERROR:  sd_max_iterations must be >= 0.  Program will terminate." << endl;
                    exit(1);
                }

                torsion_iterations = atoi(parm.query_param("sd_tors_premin_iterations", "0").c_str());
                if (torsion_iterations < 0) {
                    cout << "ERROR:  sd_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                    exit(1);
                }

                max_cycles = atoi(parm.query_param("sd_max_cycles", "1").c_str());
                if (max_cycles <= 0) {
                    cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                    exit(1);
                }

                score_converge = atof(parm.query_param("sd_score_converge", "0.1").c_str());
                cycle_converge = atof(parm.query_param("sd_cycle_converge", "1.0").c_str());
                trans_step_size = atof(parm.query_param("sd_trans_step", "1.0").c_str());
                rot_step_size = atof(parm.query_param("sd_rot_step", "0.1").c_str());
                tors_step_size = atof(parm.query_param("sd_tors_step", "10.0").c_str());
            }

            // --- Advanced minimization parameters ---
            if (flexible_ligand && (use_min_rigid_anchor || use_min_flex_growth)) {
                advanced_min_params =
                    (parm.query_param("use_advanced_simplex_parameters", "no", "yes no") == "yes") ? true : false;
                if (!advanced_min_params) {
                    if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                        max_iterations = atoi(parm.query_param("sd_max_iterations", "1000").c_str());
                        if (max_iterations < 0) {
                            cout << "ERROR:  sd_max_iterations must be >= 0.  Program will terminate." << endl;
                            exit(1);
                        }
                        torsion_iterations = atoi(parm.query_param("sd_tors_premin_iterations", "0").c_str());
                        if (torsion_iterations < 0) {
                            cout << "ERROR:  sd_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                            exit(1);
                        }
                        max_cycles = atoi(parm.query_param("sd_max_cycles", "1").c_str());
                        if (max_cycles <= 0) {
                            cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                            exit(1);
                        }
                        score_converge = atof(parm.query_param("sd_score_converge", "0.1").c_str());
                        cycle_converge = atof(parm.query_param("sd_cycle_converge", "1.0").c_str());
                        trans_step_size = atof(parm.query_param("sd_trans_step", "1.0").c_str());
                        rot_step_size = atof(parm.query_param("sd_rot_step", "0.1").c_str());
                        tors_step_size = atof(parm.query_param("sd_tors_step", "10.0").c_str());
                    }
                }
                if (advanced_min_params) {
                    max_iterations = atoi(parm.query_param("sd_max_iterations", "1000").c_str());
                    torsion_iterations = atoi(parm.query_param("sd_tors_premin_iterations", "0").c_str());
                    max_cycles = atoi(parm.query_param("sd_max_cycles", "1").c_str());
                    score_converge = atof(parm.query_param("sd_score_converge", "0.1").c_str());
                    cycle_converge = atof(parm.query_param("sd_cycle_converge", "1.0").c_str());
                    trans_step_size = atof(parm.query_param("sd_trans_step", "1.0").c_str());
                    rot_step_size = atof(parm.query_param("sd_rot_step", "0.1").c_str());
                    tors_step_size = atof(parm.query_param("sd_tors_step", "10.0").c_str());
                }

                // parameters for anchor minimization
                if (use_min_rigid_anchor) {
                    anchor_min_max_iterations = atoi(parm.query_param("sd_anchor_max_iterations", "500").c_str());
                    if (anchor_min_max_iterations <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(1);
                    }
                    anchor_min_max_cycles = atoi(parm.query_param("sd_anchor_max_cycles", "1").c_str());
                    if (anchor_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(1);
                    }
                    anchor_min_trans_step_size = atof(parm.query_param("sd_anchor_trans_step", "1.0").c_str());
                    anchor_min_rot_step_size = atof(parm.query_param("sd_anchor_rot_step", "0.1").c_str());
                    anchor_min_tors_step_size = atof(parm.query_param("sd_anchor_tors_step", "10.0").c_str());
                }

                // parameters for flexible grow minimization
                if (use_min_flex_growth) {
                    if (use_min_flex_growth_ramp) {
                        flex_min_max_iterations = atoi(parm.query_param("sd_grow_max_iterations", "250").c_str());
                    } else {
                        flex_min_max_iterations = atoi(parm.query_param("sd_grow_max_iterations", "500").c_str());
                    }
                    if (flex_min_max_iterations < 0) {
                        cout << "ERROR:  sd_grow_max_iterations cannot be negative.  Program will terminate." << endl;
                        exit(1);
                    }
                    flex_min_max_cycles = atoi(parm.query_param("sd_grow_max_cycles", "1").c_str());
                    if (flex_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(1);
                    }
                    flex_min_tors_step_size = atof(parm.query_param("sd_grow_tors_step", "10.0").c_str());
                    flex_min_torsion_iterations = atoi(parm.query_param("sd_grow_tors_premin_iterations", "0").c_str());
                    if (use_min_flex_growth_ramp) {
                        flex_min_ramp_max_iterations = atoi(parm.query_param("sd_grow_ramp_max_iterations", "50").c_str());
                        flex_min_ramp_max_cycles = atoi(parm.query_param("sd_grow_ramp_max_cycles", "1").c_str());
                        flex_min_ramp_tors_step_size = atof(parm.query_param("sd_grow_ramp_tors_step", "10.0").c_str());
                        flex_min_ramp_torsion_iterations = atoi(parm.query_param("sd_grow_ramp_tors_premin_iterations", "0").c_str());
                    }
                }

                // parameters for final minimization
                final_min = (parm.query_param("sd_final_min", "no", "yes no") == "yes") ? true : false;
                if (final_min) {
                    final_min_max_iterations = atoi(parm.query_param("sd_final_max_iterations", "500").c_str());
                    if (final_min_max_iterations <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(1);
                    }
                    final_min_max_cycles = atoi(parm.query_param("sd_final_max_cycles", "1").c_str());
                    if (final_min_max_cycles <= 0) {
                        cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate." << endl;
                        exit(1);
                    }
                    final_min_trans_step_size = atof(parm.query_param("sd_final_trans_step", "1.0").c_str());
                    final_min_rot_step_size = atof(parm.query_param("sd_final_rot_step", "0.1").c_str());
                    final_min_tors_step_size = atof(parm.query_param("sd_final_tors_step", "10.0").c_str());
                }

                // secondary minimization
                secondary_min_pose = (parm.query_param("sd_secondary_min_pose", "no", "yes no") == "yes") ? true : false;
                if (secondary_min_pose) {
                    secondary_min_max_iterations = atoi(parm.query_param("sd_secondary_max_iterations", "500").c_str());
                    secondary_min_max_cycles = atoi(parm.query_param("sd_secondary_max_cycles", "1").c_str());
                    secondary_min_score_converge = atof(parm.query_param("sd_secondary_score_converge", "0.1").c_str());
                    secondary_min_cycle_converge = atof(parm.query_param("sd_secondary_cycle_converge", "1.0").c_str());
                    secondary_min_trans_step_size = atof(parm.query_param("sd_secondary_trans_step", "1.0").c_str());
                    secondary_min_rot_step_size = atof(parm.query_param("sd_secondary_rot_step", "0.1").c_str());
                    secondary_min_tors_step_size = atof(parm.query_param("sd_secondary_tors_step", "10.0").c_str());
                }

                // parameters for final restraint
                if (final_min or minimize_ligand) {
                    random_seed = atoi(parm.query_param("sd_random_seed", "0").c_str());
                    restrained_min = (parm.query_param("sd_restraint_min", "no", "yes no") == "yes") ? true : false;
                    if (restrained_min) {
                        coefficient_restraint = atof(parm.query_param("sd_coefficient_restraint", "10.0").c_str());
                    }
                }
            }
        }
    }
}

/******************************************************/
void
Steepest_Descent_Minimizer::initialize()
{
    //cout << "Initializing Steepest Descent" << endl;
    srand(random_seed);
}

/******************************************************/
// Compute gradient via forward finite differences
// g[i] = (f(x + h*e_i) - f(x)) / h
void
Steepest_Descent_Minimizer::compute_gradient(Base_Score & score, DOCKMol & ref_mol,
                                              DOCKMol & tmp_mol, DOCKMol & rmsd_ref,
                                              FLOATVec & x, FLOATVec & g,
                                              float trans_step_size, float rot_step_size,
                                              float tors_step_size, float h)
{
    int i;
    float fx;

    // Evaluate f(x) using the full penalized score
    if (!eval_score(score, ref_mol, tmp_mol, x,
                    trans_step_size, rot_step_size, tors_step_size)) {
        // Scoring failed — use large gradient to force movement
        for (i = 0; i < n; i++) g[i] = 0.0f;
        return;
    }
    fx = tmp_mol.current_score + tmp_mol.internal_energy;
    if (restrained_min) {
        fx += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
    }

    // Evaluate f(x + h*e_i) for each dimension
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
Steepest_Descent_Minimizer::do_minimize(Base_Score & score, DOCKMol & mol,
                                         FLOATVec & vertex, int max_iterations,
                                         float score_converge, float trans_step_size,
                                         float rot_step_size, float tors_step_size)
{
    n = (int) vertex.size();
    if (n < 1) {
        cerr << "ERROR: SD: zero degrees of freedom" << endl;
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
        cerr << "ERROR: SD: initial scoring failed" << endl;
        return 1.0e10f;
    }

    FLOATVec g(n, 0.0f);
    float fd_h = (fd_step > 0.0f) ? fd_step : 1.0e-4f;
    int iter;

    for (iter = 0; iter < max_iterations; iter++) {

        // Compute gradient
        compute_gradient(score, ref_mol, tmp_mol, rmsd_ref, x, g,
                         trans_step_size, rot_step_size, tors_step_size, fd_h);

        // Check gradient norm for convergence
        float norm_g = 0.0f;
        for (int i = 0; i < n; i++) norm_g += g[i] * g[i];
        norm_g = sqrt(norm_g);

        if (norm_g < score_converge) break;

        // Backtracking line search along -g
        float alpha = line_search_alpha;
        float fnew = fopt;
        bool found = false;

        for (int ls = 0; ls < max_line_search; ls++) {
            FLOATVec x_trial(n);
            for (int i = 0; i < n; i++) {
                x_trial[i] = x[i] - alpha * g[i] / max(norm_g, 1.0e-20f);
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
