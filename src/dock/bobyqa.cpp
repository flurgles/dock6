#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include "master_score.h"
#include "minimizer.h"
#include "bobyqa.h"

using namespace std;

void
BOBYQA_Minimizer::input_parameters(Parameter_Reader & parm,
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

    // minimizer_type is already read by dock.cpp before calling this function.
    // Since this function is only called when minimizer_type == "bobyqa",
    // we skip the check and go straight to reading BOBYQA-specific parameters.
    if (score.primary_min) {

        cout << "\nBOBYQA Minimization Parameters" << endl;
        cout <<
            "------------------------------------------------------------------------------------------"
            << endl;

        minimize_ligand =
            (parm.query_param("minimize_ligand", "yes", "yes no") ==
             "yes") ? true : false;

        if (minimize_ligand) {

            // if anchor and grow is flagged, check which portions of molecule to minimize
            if (flexible_ligand) {
                use_min_rigid_anchor = (parm.query_param("minimize_anchor", "yes", "yes no") == "yes") ? true : false;
                use_min_flex_growth = (parm.query_param("minimize_flexible_growth", "yes", "yes no") == "yes") ? true : false;
                advanced_min_params = (parm.query_param("use_advanced_simplex_parameters", "no", "yes no") == "yes") ? true : false;
                if (!genetic_algorithm && !denovo_design) {
                    use_min_flex_growth_ramp = (parm.query_param("minimize_flexible_growth_ramp", "yes", "yes no") == "yes") ? true : false;
                }
            }

            // BOBYQA-specific parameters for basic minimization
            if (!advanced_min_params) {
                if (!flexible_ligand || (!use_min_rigid_anchor && !use_min_flex_growth)) {
                    const char bobyqa_max_it[] = "bobyqa_max_iterations";
                    max_iterations = atoi(parm.query_param(bobyqa_max_it, "1000").c_str());
                    if (max_iterations < 0) {
                        cout << "ERROR:  Parameter \"" << bobyqa_max_it
                             << "\" must be an integer greater than or equal to zero."
                             << endl
                             << "Program will terminate."
                             << endl;
                        exit(1);
                    }

                    torsion_iterations = atoi(parm.query_param("bobyqa_tors_premin_iterations", "0").c_str());
                    if (torsion_iterations < 0) {
                        cout << "ERROR:  bobyqa_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                        exit(1);
                    }
                }

                max_cycles =
                    atoi(parm.query_param("bobyqa_max_cycles", "1").c_str());
                if (max_cycles <= 0) {
                    cout <<
                        "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                // Initial scoring perturbation attempts (0 = disabled for backward compatibility)
                initial_perturb_attempts =
                    atoi(parm.query_param("bobyqa_initial_perturb_attempts", "0").c_str());
                if (initial_perturb_attempts < 0) {
                    cout <<
                        "ERROR:  bobyqa_initial_perturb_attempts must be non-negative. Program will terminate."
                        << endl;
                    exit(1);
                }

                // Configurable advanced features
                use_rescue = (parm.query_param("bobyqa_use_rescue", "yes", "yes no") == "yes") ? true : false;
                hessian_mode = parm.query_param("bobyqa_hessian_mode", "default");
                if (hessian_mode != "default" && hessian_mode != "block_diag" && hessian_mode != "full_quad") {
                    cout << "ERROR:  bobyqa_hessian_mode must be 'default', 'block_diag', or 'full_quad'.  Program will terminate." << endl;
                    exit(1);
                }
                use_multi_start = (parm.query_param("bobyqa_use_multi_start", "no", "yes no") == "yes") ? true : false;
                multi_start_restarts = atoi(parm.query_param("bobyqa_multi_start_restarts", "3").c_str());
                if (multi_start_restarts < 0) {
                    cout << "ERROR:  bobyqa_multi_start_restarts must be non-negative. Program will terminate." << endl;
                    exit(1);
                }

                // Adaptive restart on stagnation
                use_adaptive_restart = (parm.query_param("bobyqa_use_adaptive_restart", "no", "yes no") == "yes") ? true : false;
                max_restarts = atoi(parm.query_param("bobyqa_max_restarts", "3").c_str());
                if (max_restarts < 0) {
                    cout << "ERROR:  bobyqa_max_restarts must be non-negative. Program will terminate." << endl;
                    exit(1);
                }
                restart_delta_scale = atof(parm.query_param("bobyqa_restart_delta_scale", "1.0").c_str());
                if (restart_delta_scale <= 0.0) {
                    cout << "ERROR:  bobyqa_restart_delta_scale must be positive. Program will terminate." << endl;
                    exit(1);
                }
                stagnation_window = atoi(parm.query_param("bobyqa_stagnation_window", "30").c_str());
                if (stagnation_window < 5) {
                    cout << "ERROR:  bobyqa_stagnation_window must be >= 5. Program will terminate." << endl;
                    exit(1);
                }
                stagnation_tol = atof(parm.query_param("bobyqa_stagnation_tol", "0.001").c_str());
                if (stagnation_tol < 0.0f) {
                    cout << "ERROR:  bobyqa_stagnation_tol must be non-negative. Program will terminate." << endl;
                    exit(1);
                }
                stagnation_abs_tol = atof(parm.query_param("bobyqa_stagnation_abs_tol", "0.1").c_str());
                if (stagnation_abs_tol < 0.0f) {
                    cout << "ERROR:  bobyqa_stagnation_abs_tol must be non-negative. Program will terminate." << endl;
                    exit(1);
                }
                restart_min_delta_ratio = atof(parm.query_param("bobyqa_restart_min_delta_ratio", "0.05").c_str());
                if (restart_min_delta_ratio < 0.0f || restart_min_delta_ratio > 1.0f) {
                    cout << "ERROR:  bobyqa_restart_min_delta_ratio must be in [0,1]. Program will terminate." << endl;
                    exit(1);
                }
                restart_perturbation = atof(parm.query_param("bobyqa_restart_perturbation", "0.05").c_str());
                if (restart_perturbation < 0.0f) {
                    cout << "ERROR:  bobyqa_restart_perturbation must be non-negative. Program will terminate." << endl;
                    exit(1);
                }
                restart_from_best = (parm.query_param("bobyqa_restart_from_best", "yes", "yes no") == "yes") ? true : false;

                if (!use_min_flex_growth_ramp) {
                    score_converge =
                        atof(parm.query_param("bobyqa_score_converge", "0.1").c_str());
                    if (score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                }

                if (use_min_flex_growth_ramp) {
                    score_converge =
                        atof(parm.query_param("bobyqa_score_converge", "0.1").c_str());
                    if (score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    initial_score_converge = atof(parm.query_param("bobyqa_initial_score_converge", "5").c_str());
                    if (initial_score_converge <= score_converge) {
                        cout <<
                            "ERROR:  Parameter must be larger than score converge value.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                }

                cycle_converge =
                    atof(parm.query_param("bobyqa_cycle_converge", "1.0").c_str());
                if (cycle_converge <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                trans_step_size =
                    atof(parm.query_param("bobyqa_trans_step", "1.0").c_str());
                if (trans_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                rot_step_size =
                    atof(parm.query_param("bobyqa_rot_step", "0.1").c_str());
                if (rot_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                tors_step_size =
                    atof(parm.query_param("bobyqa_tors_step", "10.0").c_str());
                if (tors_step_size <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                // BOBYQA-specific algorithm parameters
                rho_beg =
                    atof(parm.query_param("bobyqa_rho_beg", "1.0").c_str());
                if (rho_beg <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                rho_end =
                    atof(parm.query_param("bobyqa_rho_end", "0.001").c_str());
                if (rho_end <= 0.0) {
                    cout <<
                        "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }
                if (rho_beg <= rho_end) {
                    cout <<
                        "ERROR:  bobyqa_rho_beg must be larger than bobyqa_rho_end.  Program will terminate."
                        << endl;
                    exit(1);
                }

                npt =
                    atoi(parm.query_param("bobyqa_npt", "0").c_str());
                if (npt < 0) {
                    cout <<
                        "ERROR:  bobyqa_npt must be non-negative.  Program will terminate."
                        << endl;
                    exit(1);
                }
            }

            // parameters for anchor minimization
            if (use_min_rigid_anchor) {
                anchor_min_max_iterations = atoi(parm.query_param("bobyqa_anchor_max_iterations", "500").c_str());
                if (anchor_min_max_iterations <= 0) {
                    cout <<
                        "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                        << endl;
                    exit(1);
                }

                if (advanced_min_params) {
                    anchor_min_max_cycles =
                        atoi(parm.query_param("bobyqa_anchor_max_cycles", "1").c_str());
                    if (anchor_min_max_cycles <= 0) {
                        cout <<
                            "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    anchor_min_score_converge =
                        atof(parm.query_param("bobyqa_anchor_score_converge", "0.1").c_str());
                    if (anchor_min_score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    anchor_min_cycle_converge =
                        atof(parm.query_param("bobyqa_anchor_cycle_converge", "1.0").c_str());
                    if (anchor_min_cycle_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    anchor_min_trans_step_size =
                        atof(parm.query_param("bobyqa_anchor_trans_step", "1.0").c_str());
                    if (anchor_min_trans_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    anchor_min_rot_step_size =
                        atof(parm.query_param("bobyqa_anchor_rot_step", "0.1").c_str());
                    if (anchor_min_rot_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    anchor_min_tors_step_size =
                        atof(parm.query_param("bobyqa_anchor_tors_step", "10.0").c_str());
                    if (anchor_min_tors_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
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

                if (use_min_flex_growth_ramp) {
                    flex_min_max_iterations = atoi(parm.query_param("bobyqa_grow_max_iterations", "250").c_str());
                } else {
                    flex_min_max_iterations = atoi(parm.query_param("bobyqa_grow_max_iterations", "500").c_str());
                }

                if (flex_min_max_iterations < 0) {
                    cout <<
                        "ERROR:  bobyqa_grow_max_iterations cannot be negative.  Program will terminate."
                        << endl;
                    exit(1);
                }

                flex_min_torsion_iterations = atoi(parm.query_param("bobyqa_grow_tors_premin_iterations", "0").c_str());
                if (flex_min_torsion_iterations < 0) {
                    cout << "ERROR:  bobyqa_grow_tors_premin_iterations cannot be negative. Program will terminate." << endl;
                    exit(1);
                }

                if (advanced_min_params) {
                    flex_min_max_cycles =
                        atoi(parm.query_param("bobyqa_grow_max_cycles", "1").c_str());
                    if (flex_min_max_cycles <= 0) {
                        cout <<
                            "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    flex_min_score_converge =
                        atof(parm.query_param("bobyqa_grow_score_converge", "0.1").c_str());
                    if (flex_min_score_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameter must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    flex_min_cycle_converge =
                        atof(parm.query_param("bobyqa_grow_cycle_converge", "1.0").c_str());
                    if (flex_min_cycle_converge <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    flex_min_trans_step_size =
                        atof(parm.query_param("bobyqa_grow_trans_step", "1.0").c_str());
                    if (flex_min_trans_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    flex_min_rot_step_size =
                        atof(parm.query_param("bobyqa_grow_rot_step", "0.1").c_str());
                    if (flex_min_rot_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
                    }
                    flex_min_tors_step_size =
                        atof(parm.query_param("bobyqa_grow_tors_step", "10.0").c_str());
                    if (flex_min_tors_step_size <= 0.0) {
                        cout <<
                            "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(1);
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

        // final_min
        final_min = (parm.query_param("bobyqa_final_min", "no", "yes no") == "yes") ? true : false;
        if (final_min) {
            final_min_rep_radius_scale = atof(parm.query_param("bobyqa_final_min_rep_rad_scale", "1").c_str());
            if (final_min_rep_radius_scale <= 0.0) {
                cout <<
                    "ERROR:  Parameter must be a float greater than zero. Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_max_iterations =
                atoi(parm.query_param("bobyqa_final_max_iterations", "500").c_str());
            if (final_min_max_iterations <= 0) {
                cout <<
                    "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_max_cycles =
                atoi(parm.query_param("bobyqa_final_max_cycles", "1").c_str());
            if (final_min_max_cycles <= 0) {
                cout <<
                    "ERROR:  Parameters must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_score_converge =
                atof(parm.query_param("bobyqa_final_score_converge", "0.1").c_str());
            if (final_min_score_converge <= 0.0) {
                cout <<
                    "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_cycle_converge =
                atof(parm.query_param("bobyqa_final_cycle_converge", "1.0").c_str());
            if (final_min_cycle_converge <= 0.0) {
                cout <<
                    "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_trans_step_size =
                atof(parm.query_param("bobyqa_final_trans_step", "1.0").c_str());
            if (final_min_trans_step_size <= 0.0) {
                cout <<
                    "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_rot_step_size =
                atof(parm.query_param("bobyqa_final_rot_step", "0.1").c_str());
            if (final_min_rot_step_size <= 0.0) {
                cout <<
                    "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
            final_min_tors_step_size =
                atof(parm.query_param("bobyqa_final_tors_step", "10.0").c_str());
            if (final_min_tors_step_size <= 0.0) {
                cout <<
                    "ERROR:  Parameters must be a float greater than zero.  Program will terminate."
                    << endl;
                exit(1);
            }
        }

        if (final_min or minimize_ligand) {
            // Always read random_seed for reproducible results.
            // Previously this was gated behind use_multi_start, which
            // left random_seed=0 (deterministic srand) when multi-start
            // was disabled.
            random_seed =
                atoi(parm.query_param("bobyqa_random_seed", "0").c_str());
            restrained_min = (parm.query_param("bobyqa_restraint_min", "no", "yes no") == "yes") ? true : false;
            if (restrained_min) {
                coefficient_restraint =
                    atof(parm.query_param("bobyqa_coefficient_restraint", "10.0").c_str());
            }
        }
    }
}
void
BOBYQA_Minimizer::initialize()
{
    //cout << "Initializing BOBYQA" << endl;
    srand(random_seed);
    noise_level = 0.0f;
    noise_threshold = 0.1f;
    // noise_window = 10; (commented out - TODO: implement properly)
    // stagnation_count = 0; (commented out - TODO: implement properly)
    // restart_count = 0; (commented out - TODO: implement properly)
    ratio_history.clear();
    fopt_history.clear();
    restart_count = 0;
    diagnostics = ConvergenceDiagnostics();
}

/**********************************************************************/
//  BOBYQA algorithm: trust-region quadratic model minimizer
//  by M.J.D. Powell (2009)
//
//  This implementation uses:
//    - PRELIM: 2n+1 interpolation points on coordinate axes
//    - TRSBOX: conjugate-gradient trust-region subproblem solver
//    - Model update via farthest-point replacement
//    - Trust region management (eta1=0.1, eta2=0.7)
/**********************************************************************/
float
BOBYQA_Minimizer::do_minimize(Base_Score & score, DOCKMol & mol,
                                FLOATVec & vertex, int max_iter_param,
                                float score_converge, float trans_step_size,
                                float rot_step_size, float tors_step_size)
{
    n = (int) vertex.size();
    if (n < 1) {
        cerr << "ERROR: BOBYQA: zero degrees of freedom" << endl;
        return 1.0e10f;
    }
    cerr << "DEBUG: do_minimize entry, n=" << n << " hessian_mode=" << hessian_mode << " use_multi_start=" << use_multi_start << endl;

    // -- Multi-start wrapper --
    // If enabled, delegate to multi_start_minimize() which calls back into
    // do_minimize() with use_multi_start temporarily disabled.
    if (use_multi_start && multi_start_restarts > 0) {
        bool save_multi_start = use_multi_start;
        use_multi_start = false;
        float result = multi_start_minimize(score, mol, vertex, max_iter_param,
                                             score_converge, trans_step_size,
                                             rot_step_size, tors_step_size);
        use_multi_start = save_multi_start;
        return result;
    }
    // npt: number of interpolation points. If not set by user (<=0), use 2*n+1.
    // NOTE: pybobyqa recommends max(2*n+1, (n+1)*(n+2)/4) for noisy objectives.
    // We tested that formula but it caused regressions in full_quad variants on
    // several DT100 systems (e.g. 1DMP worsened by 5.6 points) because PRELIM
    // only fills 2*n+1 axis points and the remaining duplicate/penalty slots
    // degraded the full quadratic model quality.
    // To try the larger default again, PRELIM must first be extended to fill all
    // np points with distinct well-poised evaluations.
    // int default_npt = max(2 * n + 1, (n + 1) * (n + 2) / 4);
    nptmax = (npt > 0) ? npt : 2 * n + 1;
    if (nptmax < n + 2) nptmax = n + 2;
    int np = nptmax;               // shorthand

    int i, j, iter;
    float f, fnew, ratio, dPred, dAct;
    float norm_g;

    DOCKMol ref_mol, tmp_mol, rmsd_ref;
    copy_molecule(ref_mol, mol);
    copy_molecule(tmp_mol, mol);
    if (restrained_min) copy_molecule(rmsd_ref, mol);

    // Parameters
    const float eta1 = 0.1f;
    const float eta2 = 0.7f;
    const float gamma_up  = 2.0f;
    const float gamma_down = 0.5f;
    const float rho_end_actual = (rho_end > 0.0f) ? rho_end : 0.001f;
    const float rho_beg_actual = (rho_beg > 0.0f) ? rho_beg : 1.0f;

    // Resize interpolation point storage.
    // Initialize every slot to a valid vector so that loops over all np
    // points are safe even when PRELIM only fills the 2n+1 axis stencil.
    // Points beyond the axis stencil start as duplicates of the initial
    // vertex with a penalty score and are replaced during optimization.
    const float PENALTY_SCORE = 1.0e6f;
    xpts.resize(np, vertex);
    fvals.assign(np, PENALTY_SCORE);
    g.resize(n, 0.0f);
    Hdiag.resize(n, 0.0f);
    if (hessian_mode != "default") {
        H.resize(n);
        for (i = 0; i < n; i++) H[i].resize(n, 0.0f);
    }
    xopt.resize(n, 0.0f);

    // ========================================================
    // 1) PRELIM: Initial interpolation set
    //    Place 2n+1 points: x0 and +/- rho_beg along each axis
    // ========================================================

    // Point 0: the initial guess
    xpts[0] = vertex;
    int n_axis = min(n, (np - 1) / 2);

    DOCKMol best_mol;
    copy_molecule(best_mol, mol);

    for (i = 0; i < n_axis; i++) {
        int idx_p = 1 + i;
        xpts[idx_p] = vertex;
        xpts[idx_p][i] += rho_beg_actual;

        int idx_m = 1 + n_axis + i;
        xpts[idx_m] = vertex;
        xpts[idx_m][i] -= rho_beg_actual;
    }

    // Evaluate all axis points - assign large penalty for failed evaluations
    cerr << "DEBUG: PRELIM evaluating axis points, n_axis=" << n_axis << " np=" << np << endl;
    for (i = 0; i < n_axis; i++) {
        int idx_p = 1 + i;
        if (idx_p < np) {
            cerr << "DEBUG: eval axis +" << i << " idx=" << idx_p << endl;
            if (eval_score(score, ref_mol, tmp_mol, xpts[idx_p],
                           trans_step_size, rot_step_size, tors_step_size)) {
                fvals[idx_p] = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fvals[idx_p] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
            } else {
                fvals[idx_p] = PENALTY_SCORE;
            }
        }
        int idx_m = 1 + n_axis + i;
        if (idx_m < np) {
            cerr << "DEBUG: eval axis -" << i << " idx=" << idx_m << endl;
            if (eval_score(score, ref_mol, tmp_mol, xpts[idx_m],
                           trans_step_size, rot_step_size, tors_step_size)) {
                fvals[idx_m] = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fvals[idx_m] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
            } else {
                fvals[idx_m] = PENALTY_SCORE;
            }
        }
    }

    // Evaluate the starting point - try fallback strategies if it fails
    bool start_ok = false;
    FLOATVec best_vertex = vertex;
    float best_score = PENALTY_SCORE;
    int max_attempts = (initial_perturb_attempts > 0) ? initial_perturb_attempts : 1;
    
    // Strategy 1: Try original starting point
    if (eval_score(score, ref_mol, tmp_mol, vertex,
                   trans_step_size, rot_step_size, tors_step_size)) {
        float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
        if (restrained_min) {
            score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
        }
        if (score_val < best_score) {
            best_score = score_val;
            best_vertex = vertex;
            start_ok = true;
        }
    }
    
    // Strategy 2: Try interpolation points (they're at different positions along axes)
    if (!start_ok) {
        for (i = 1; i < np && !start_ok; i++) {
            if (eval_score(score, ref_mol, tmp_mol, xpts[i],
                           trans_step_size, rot_step_size, tors_step_size)) {
                float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
                if (score_val < best_score) {
                    best_score = score_val;
                    best_vertex = xpts[i];
                    start_ok = true;
                }
            }
        }
    }
    
    // Strategy 3: Random perturbations (if enabled)
    if (!start_ok && initial_perturb_attempts > 0) {
        FLOATVec orig_vertex = vertex;
        for (int attempt = 0; attempt < initial_perturb_attempts && !start_ok; attempt++) {
            // Perturb from original vertex
            for (i = 0; i < n; i++) {
                float perturb = ((float)rand() / RAND_MAX - 0.5f) * 0.2f * rho_beg_actual;
                vertex[i] = orig_vertex[i] + perturb;
            }
            if (eval_score(score, ref_mol, tmp_mol, vertex,
                           trans_step_size, rot_step_size, tors_step_size)) {
                float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
                if (score_val < best_score) {
                    best_score = score_val;
                    best_vertex = vertex;
                    start_ok = true;
                }
            }
        }
    }
    
    if (!start_ok) {
        cerr << "ERROR: BOBYQA: initial scoring failed after all fallback strategies" << endl;
        return 1.0e10f;
    }
    
    fvals[0] = best_score;
    copy_crds(best_mol, tmp_mol);
    xopt = best_vertex;
    fopt = fvals[0];
    kopt = 0;

    // Build initial quadratic model (gradient and diagonal Hessian)
    int na = n_axis;
    float inv_rho2 = 1.0f / (rho_beg_actual * rho_beg_actual);
    for (i = 0; i < na; i++) {
        float fp = fvals[1 + i];
        float fm = fvals[1 + na + i];
        g[i] = (fp - fm) / (2.0f * rho_beg_actual);
        Hdiag[i] = (fp + fm - 2.0f * fopt) * inv_rho2;
        if (Hdiag[i] < 1.0e-12f) Hdiag[i] = 1.0e-12f;
    }
    for (i = na; i < n; i++) {
        g[i] = 0.0f;
        Hdiag[i] = 1.0f;
    }

    // Full quadratic model: build off-diagonal elements if enabled
    if (hessian_mode != "default") {
        cerr << "DEBUG: Calling build_full_model at PRELIM" << endl;
        build_full_model(score, ref_mol, tmp_mol, rmsd_ref, best_mol,
                         trans_step_size, rot_step_size, tors_step_size);
        cerr << "DEBUG: build_full_model returned" << endl;
    }

    delta = rho_beg_actual;
    s_step.resize(n, 0.0f);

    // ========================================================
    // 2) Main BOBYQA iteration loop
    // ========================================================

    for (iter = 0; iter < max_iter_param; iter++) {
        cerr << "DEBUG: iter=" << iter << ", delta=" << delta << ", fopt=" << fopt << endl;

        // ---- 2a) TRSBOX: Solve trust-region subproblem ----
        // cerr << "DEBUG: Starting TRSBOX" << endl;
        // Model: m(s) = fopt + g^T s + 0.5 * s^T diag(H) * s
        // Minimize m(s) subject to ||s|| <= delta
        // Use dogleg method between Cauchy point and Newton step

        norm_g = 0.0f;
        for (i = 0; i < n; i++) norm_g += g[i] * g[i];
        norm_g = sqrt(norm_g);
        if (norm_g < 1.0e-20f) break;  // gradient zero: already optimal

        // Compute Cauchy point (steepest descent)
        FLOATVec s(n, 0.0f);
        float gHg = 0.0f;
        if (hessian_mode != "default") {
            // cerr << "DEBUG: gHg full_quad computation, n=" << n << endl;
            // g^T * H * g with full Hessian
            for (i = 0; i < n; i++) {
                if (i >= (int)H.size()) { cerr << "ERROR: H.size=" << H.size() << " i=" << i << endl; break; }
                float sum = 0.0f;
                for (j = 0; j < n; j++) {
                    if (j >= (int)H[i].size()) { cerr << "ERROR: H[" << i << "].size=" << H[i].size() << " j=" << j << endl; break; }
                    sum += H[i][j] * g[j];
                }
                gHg += g[i] * sum;
            }
        } else {
            // Diagonal Hessian approximation
            for (i = 0; i < n; i++) gHg += g[i] * Hdiag[i] * g[i];
        }

        if (gHg <= 0.0f) {
            // Negative curvature direction: go to trust boundary
            for (i = 0; i < n; i++) s[i] = -g[i] * delta / norm_g;
        } else {
            // Cauchy point
            float tau = (norm_g * norm_g * norm_g) / (delta * gHg);
            if (tau > 1.0f) tau = 1.0f;
            for (i = 0; i < n; i++) s[i] = -tau * delta * g[i] / norm_g;
        }

        // Compute Newton step
        FLOATVec s_newt(n, 0.0f);
        float norm_newt = 0.0f;
        if (hessian_mode != "default") {
            // Estimate Hessian eigenvalues. If the smallest eigenvalue is
            // strongly negative, CG will immediately detect indefiniteness
            // and fall back to Cauchy. Skip CG in that case.
            float eig_min = 0.0f, eig_max = 0.0f;
            estimate_hessian_eigenvalues(min(5, n), &eig_min, &eig_max);
            if (eig_min < -1.0e-4f) {
                cerr << "DEBUG: CG skip (indefinite H, eig_min=" << eig_min
                     << "), fall back to Cauchy" << endl;
            } else {
            cerr << "DEBUG: CG solver start, n=" << n << " delta=" << delta << endl;
            // Conjugate gradient: solve H * s = -g
            // H should be SPD; if not, CG may fail, fall back to diagonal
            int cg_iter;
            const int max_cg = n;
            float alpha, beta, rdot, rdot_new;
            FLOATVec r(n, 0.0f), p(n, 0.0f), Hp(n, 0.0f);
            for (i = 0; i < n; i++) {
                r[i] = -g[i];
                p[i] = r[i];
            }
            rdot = 0.0f;
            for (i = 0; i < n; i++) rdot += r[i] * r[i];
            bool cg_converged = false;
            for (cg_iter = 0; cg_iter < max_cg; cg_iter++) {
                // H * p
                // cerr << "DEBUG: CG iter " << cg_iter << " / " << max_cg << endl;
                for (i = 0; i < n; i++) {
                    Hp[i] = 0.0f;
                    if (i >= (int)H.size()) { cerr << "ERROR: H.size=" << H.size() << " i=" << i << endl; break; }
                    for (j = 0; j < n; j++) {
                        if (j >= (int)H[i].size()) { cerr << "ERROR: H[" << i << "].size=" << H[i].size() << " j=" << j << endl; break; }
                        Hp[i] += H[i][j] * p[j];
                    }
                }
                float pHp = 0.0f;
                for (i = 0; i < n; i++) pHp += p[i] * Hp[i];
                // Check for non-positive-definite Hessian.
                // pHp = p^T * H * p.  If pHp < 0, H is not positive definite
                // (negative curvature direction). CG would compute a negative
                // step alpha = rdot / pHp, pushing the solution in the wrong
                // direction. Fall back to steepest descent (Cauchy step).
                if (fabs(pHp) < 1.0e-20f || pHp < 0.0f) {
                    if (pHp < 0.0f) {
                        cerr << "DEBUG: CG indefinite H (pHp=" << pHp
                             << "), fall back to Cauchy" << endl;
                    }
                    break;
                }
                alpha = rdot / pHp;
                for (i = 0; i < n; i++) {
                    s_newt[i] += alpha * p[i];
                    r[i] -= alpha * Hp[i];
                }
                rdot_new = 0.0f;
                for (i = 0; i < n; i++) rdot_new += r[i] * r[i];
                if (rdot_new < 1.0e-20f) { cg_converged = true; break; }
                beta = rdot_new / rdot;
                rdot = rdot_new;
                for (i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
            }
            cerr << "DEBUG: CG done, converged=" << cg_converged << " n_newt=" << norm_newt << endl;
            norm_newt = 0.0f;
            for (i = 0; i < n; i++) norm_newt += s_newt[i] * s_newt[i];
            norm_newt = sqrt(norm_newt);
            // If CG didn't converge or gave a bad step, fall back to Cauchy
            if (!cg_converged || norm_newt > 100.0f * delta || norm_newt < 1.0e-20f) {
                // Fall back to Cauchy step (already in s)
            } else if (norm_newt <= delta) {
                s = s_newt;
            } else {
                // Dogleg method: interpolate between Cauchy and Newton steps.
                // The Cauchy point s_c is the minimizer along steepest descent
                // within the trust region (already in s[]). The Newton step s_n
                // (from CG solve) gives the full-model minimizer but exceeds delta.
                // The dogleg finds alpha in [0,1] where the line
                //   s(alpha) = s_c + alpha * (s_n - s_c)
                // crosses the trust region boundary ||s(alpha)|| = delta.
                // Expanding the norm-squared constraint:
                //   ||s_c + alpha*(s_n-s_c)||^2
                //   = ||s_c||^2 + 2*alpha*s_c^T*(s_n-s_c) + alpha^2*||s_n-s_c||^2
                //   = c + alpha*b + alpha^2*a  =  delta^2
                // where a = ||s_diff||^2, b = 2*s_c^T*s_diff, c = ||s_c||^2 - delta^2.
                // The discriminant b^2 - 4*a*c >= 0 is guaranteed because
                // ||s_c|| <= delta (Cauchy inside TR) and ||s_n|| > delta (Newton outside).
                // We take the smallest alpha in [0,1] as the dogleg step.
                float norm_s_c = 0.0f;
                for (int k = 0; k < n; k++) norm_s_c += s[k] * s[k];
                norm_s_c = sqrt(norm_s_c);
                if (norm_s_c > 1.0e-20f && norm_newt > norm_s_c) {
                    FLOATVec s_diff(n, 0.0f);
                    float a = 0.0f, b = 0.0f, c = 0.0f;
                    for (int k = 0; k < n; k++) {
                        s_diff[k] = s_newt[k] - s[k];
                        a += s_diff[k] * s_diff[k];          // = ||s_diff||^2
                        b += 2.0f * s[k] * s_diff[k];        // = 2 * s_c^T * s_diff
                        c += s[k] * s[k];                    // = ||s_c||^2
                    }
                    c -= delta * delta;                      // = ||s_c||^2 - delta^2
                    float disc = b * b - 4.0f * a * c;
                    if (disc >= 0.0f && a > 1.0e-20f) {
                        float sqrt_disc = sqrt(disc);
                        // Roots of a*alpha^2 + b*alpha + c = 0
                        float alpha1 = (-b + sqrt_disc) / (2.0f * a);
                        float alpha2 = (-b - sqrt_disc) / (2.0f * a);
                        // Take the smaller alpha in [0,1]
                        float alpha = 1.0f;
                        if (alpha1 >= 0.0f && alpha1 <= 1.0f) alpha = alpha1;
                        if (alpha2 >= 0.0f && alpha2 <= 1.0f && alpha2 < alpha) alpha = alpha2;
                        if (alpha < 0.0f) alpha = 0.0f;
                        if (alpha > 1.0f) alpha = 1.0f;
                        for (int k = 0; k < n; k++) s[k] += alpha * s_diff[k];
                    }
                }
                // Fallback: if computation fails, keep Cauchy step
            }
        }
        } else {
            // Diagonal Hessian: s_newt_i = -g_i / H_ii
            for (i = 0; i < n; i++) {
                s_newt[i] = -g[i] / max(Hdiag[i], 1.0e-12f);
                norm_newt += s_newt[i] * s_newt[i];
            }
            norm_newt = sqrt(norm_newt);

            // Dogleg: if Newton step is within trust region, use it
            if (norm_newt <= delta) {
                s = s_newt;
            }
        }

        // ---- 2b) Evaluate trial point ----
        // cerr << "DEBUG: 2b start, n=" << n << " delta=" << delta << endl;
        FLOATVec x_trial(n, 0.0f);
        for (i = 0; i < n; i++) x_trial[i] = xopt[i] + s[i];
        // cerr << "DEBUG: x_trial built, calling eval_score" << endl;

        fnew = fopt;  // default: no improvement
        if (eval_score(score, ref_mol, tmp_mol, x_trial,
                       trans_step_size, rot_step_size, tors_step_size)) {
            fnew = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                fnew += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
        }

        // ---- 2c) Compute actual vs predicted reduction ----
        dAct = fopt - fnew;

        // Predicted reduction = m(0) - m(s)
        dPred = 0.0f;
        if (hessian_mode != "default") {
            // dPred = -[g^T s + 0.5 * s^T * H * s]
            float gs = 0.0f, sHs = 0.0f;
            for (i = 0; i < n; i++) {
                if (i >= (int)H.size()) { cerr << "ERROR: H.size=" << H.size() << " i=" << i << endl; break; }
                gs += g[i] * s[i];
                float sum = 0.0f;
                for (j = 0; j < n; j++) {
                    if (j >= (int)H[i].size()) { cerr << "ERROR: H[" << i << "].size=" << H[i].size() << " j=" << j << endl; break; }
                    sum += H[i][j] * s[j];
                }
                sHs += s[i] * sum;
            }
            dPred = -(gs + 0.5f * sHs);
        } else {
            for (i = 0; i < n; i++) {
                dPred -= g[i] * s[i] + 0.5f * Hdiag[i] * s[i] * s[i];
            }
        }
        if (dPred < 1.0e-12f) dPred = 1.0e-12f;

        ratio = dAct / dPred;

        cerr << "DEBUG: section 2d, ratio=" << ratio << endl;

        // ---- Track sliding-window ratio history ----
        ratio_history.push_back(ratio);
        while ((int)ratio_history.size() > max_ratio_window)
            ratio_history.pop_front();
        float avg_ratio = 0.0f;
        for (size_t ri = 0; ri < ratio_history.size(); ri++)
            avg_ratio += ratio_history[ri];
        avg_ratio /= (float)ratio_history.size();

        // ---- 2d) Accept / reject step ----
        if (ratio > 0.0f) {
            // cerr << "DEBUG: inside 2d accept" << endl;
            // Save state before updating fopt for model update
            if (hessian_mode != "default") {
                // cerr << "DEBUG: copy s_step" << endl;
                fopt_before = fopt;
                s_step = s;
                // cerr << "DEBUG: done copy s_step" << endl;
            }
            // cerr << "DEBUG: copy xopt" << endl;
            xopt = x_trial;
            // cerr << "DEBUG: done copy xopt" << endl;
            fopt = fnew;
            // cerr << "DEBUG: end accept" << endl;
        }

        // cerr << "DEBUG: section 2e" << endl;
        // ---- 2e) Update trust region radius (standard) ----
        bool made_progress = false;
        if (ratio >= eta2) {
            // Good step: expand aggressively
            delta = min(delta * gamma_up, rho_beg_actual * 10.0f);
            made_progress = true;
        } else if (ratio >= eta1) {
            // Acceptable step: expand moderately
            delta = min(delta * 1.5f, rho_beg_actual * 10.0f);
            made_progress = true;
        } else if (ratio > 0.0f) {
            // Acceptable step but not enough to expand
            made_progress = true;
        } else if (ratio < eta1) {
            // Poor step: contract
            // If the model has been consistently poor (low avg_ratio over the
            // sliding window), contract more aggressively to recover faster.
            if (avg_ratio < 0.15f && (int)ratio_history.size() >= max_ratio_window) {
                delta *= gamma_down * gamma_down;  // double contraction
            } else {
                delta *= gamma_down;
            }
            if (delta < rho_end_actual) delta = rho_end_actual;
        }

        // cerr << "DEBUG: stagnation check, count=" << stagnation_count << endl; (commented out - TODO: implement properly)
        // ---- Stagnation detection and adaptive restart ---- (commented out - TODO: implement properly)
        // if (!made_progress) {
        //     stagnation_count++;
        // } else {
        //     stagnation_count = 0;
        // }
        //
        // // Trigger adaptive restart on stagnation
        // if (use_adaptive_restart && stagnation_count >= 5 && restart_count < max_restarts) {
        //     perform_adaptive_restart(score, mol, ref_mol, tmp_mol, rmsd_ref,
        //                              trans_step_size, rot_step_size, tors_step_size);
        //     // After restart, continue to next iteration with new state
        //     continue;
        // }

        cerr << "DEBUG: section 2f, np=" << np << " xpts.size=" << xpts.size() << endl;
        // ---- 2f) Update interpolation set (farthest-point replacement) ----
        // Find the farthest interpolation point from xopt
        float max_dist = -1.0f;
        int farthest = 0;
        for (i = 0; i < np; i++) {
            float dist = 0.0f;
            for (j = 0; j < n; j++) {
                float d = xpts[i][j] - xopt[j];
                dist += d * d;
            }
            if (dist > max_dist) {
                max_dist = dist;
                farthest = i;
            }
        }

        // Replace farthest point with the new trial point
        xpts[farthest] = x_trial;
        fvals[farthest] = fnew;
        
        // ---- 2f.2) RESCUE: check interpolation set quality (if enabled) ----
        if (use_rescue && iter > 0 && iter % 50 == 0) {
            // Check if any points are too close to each other
            bool degenerate = false;
            int too_close = 0;
            for (i = 0; i < np && !degenerate; i++) {
                for (j = i + 1; j < np && !degenerate; j++) {
                    float dist = 0.0f;
                    for (int k = 0; k < n; k++) {
                        float d = xpts[i][k] - xpts[j][k];
                        dist += d * d;
                    }
                    if (dist < 1.0e-8f * delta * delta) {
                        too_close++;
                        if (too_close > np / 3) degenerate = true;
                    }
                }
            }
            if (degenerate) {
                diagnostics.rescue_calls++;
                rescue(score, mol, ref_mol, tmp_mol, rmsd_ref, best_mol,
                       trans_step_size, rot_step_size, tors_step_size,
                       rho_beg_actual);
            }
        }

        // ---- 2g) Update model (gradient and Hessian) ----
        if (ratio > 0.0f) {
            if (hessian_mode != "default") {
                update_model_full(score, ref_mol, tmp_mol, rmsd_ref, best_mol,
                                  trans_step_size, rot_step_size, tors_step_size,
                                  fopt, fnew);
            } else {
                // Diagonal Hessian update
                // The quadratic model m(x) = fopt + g^T*(x-xopt) + 0.5*(x-xopt)^T diag(H)*(x-xopt)
                // has gradient at x_trial: g_new = g + diag(H) * s where s = x_trial - xopt.
                // This satisfies the secant condition g_new[i] - g[i] = H_ii * s_i.
                // The previous blended formula g = 0.5*(g + fd - 0.5*H*s) was non-standard
                // and damped the gradient update, slowing model convergence.
                for (i = 0; i < n; i++) {
                    float diff = x_trial[i] - xopt[i];
                    if (fabs(diff) > 1.0e-10f) {
                        float fd = (fnew - fopt) / diff;
                        float g_orig = g[i];
                        // Standard quadratic model gradient update: g_new = g_old + H_ii * s_i
                        g[i] = g_orig + Hdiag[i] * diff;
                        // Secant condition using ORIGINAL gradient for Hessian diagonal
                        float h_new = (fd - g_orig) / diff;
                        if (h_new > 1.0e-12f) Hdiag[i] = h_new;
                    }
                }
            }
        }

        // ---- 2h) Check convergence ----
        // Standard BOBYQA convergence: trust region has shrunk to minimum radius.
        if (delta <= rho_end_actual && iter > 5) break;

        // Track best-score history for adaptive-restart use.
        while ((int)fopt_history.size() >= stagnation_window) fopt_history.pop_front();
        fopt_history.push_back(fopt);

        // ---- 2i) Adaptive restart on stagnation ----
        // If the best score has not improved meaningfully for a sustained
        // window, reset the trust region and rebuild the model around xopt.
        // This gives BOBYQA a chance to escape a poor local quadratic model.
        if (use_adaptive_restart && restart_count < max_restarts &&
            (int)fopt_history.size() == stagnation_window &&
            delta <= rho_beg_actual * restart_min_delta_ratio) {
            float f_min = *min_element(fopt_history.begin(), fopt_history.end());
            float f_max = *max_element(fopt_history.begin(), fopt_history.end());
            float range = f_max - f_min;
            float scale = max(1.0f, fabs(f_min));
            if (range < stagnation_abs_tol || range / scale < stagnation_tol) {
                perform_adaptive_restart(score, mol, ref_mol, tmp_mol, rmsd_ref, best_mol,
                                         trans_step_size, rot_step_size, tors_step_size,
                                         rho_beg_actual);
                fopt_history.clear();
                diagnostics.restarts = restart_count;
                continue;
            }
        }
    }

    // Record convergence diagnostics
    diagnostics.iterations = iter;
    diagnostics.final_delta = delta;

    // Compute average ratio over the full history for diagnostics
    if (!ratio_history.empty()) {
        float total = 0.0f;
        for (size_t ri = 0; ri < ratio_history.size(); ri++)
            total += ratio_history[ri];
        diagnostics.avg_ratio = total / (float)ratio_history.size();
    }

    // Estimate Hessian eigenvalues for diagnostics
    if (n > 0) {
        float eig_min = 0.0f, eig_max = 0.0f;
        estimate_hessian_eigenvalues(min(5, n), &eig_min, &eig_max);
        diagnostics.hessian_min_eigenvalue = eig_min;
        diagnostics.hessian_max_eigenvalue = eig_max;
    }
    if (delta <= rho_end_actual && diagnostics.iterations > 5) {
        diagnostics.termination_reason = "delta_converged";
    } else if (diagnostics.iterations >= max_iter_param) {
        diagnostics.termination_reason = "max_iterations";
    } else {
        diagnostics.termination_reason = "unknown";
    }

    // ========================================================
    // 3) Return best point found
    // ========================================================

    // Apply best vertex to molecule
    copy_molecule(tmp_mol, mol);
    FLOATVec best_vec;
    scale_vector(best_vec, xopt, trans_step_size, rot_step_size, tors_step_size);
    vector_to_dockmol(tmp_mol, best_vec);
    copy_crds(mol, tmp_mol);

    // Clear state vectors to ensure clean destruction
    xpts.clear();
    fvals.clear();
    g.clear();
    Hdiag.clear();
    xopt.clear();
    s_step.clear();
    if (hessian_mode != "default") {
        H.clear();
    }

    return fopt;
}

/**********************************************************************/
// Build full quadratic model (off-diagonal Hessian elements)
// Uses paired perturbations: evaluate along (i,j) directions and compute
// H[i][j] = [f(x+rho*ei+rho*ej) - f(x+rho*ei) - f(x+rho*ej) + f(x)] / rho^2
// Only called when hessian_mode != "default" and npt >= (n+1)*(n+2)/2
/**********************************************************************/
void
BOBYQA_Minimizer::build_full_model(Base_Score & score, DOCKMol & ref_mol,
                                    DOCKMol & tmp_mol, DOCKMol & rmsd_ref,
                                    DOCKMol & best_mol,
                                    float trans_step_size, float rot_step_size,
                                    float tors_step_size)
{
    int i, j;
    float rho_eff = rho_beg;
    if (rho_eff <= 0.0f) rho_eff = 1.0f;
    
    // Reset Hessian to diagonal
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            H[i][j] = 0.0f;
        }
        H[i][i] = Hdiag[i];
    }
    
    // Build off-diagonal elements by evaluating corner points:
    //   x_corner = xopt + rho_eff * (e_i + e_j) for each pair (i,j)
    //   H[i][j] = (f(corner) - f(xopt+rho*e_i) - f(xopt+rho*e_j) + fopt) / rho^2
    //
    // When hessian_mode == "block_diag", only pairs (i,j) within the same DOF
    // block are computed: translation (0-2), rotation (3-5), torsion (6..n-1).
    // Cross-block coupling is assumed weak and set to zero.
    //
    // Cap at max_corners off-diagonal pairs to balance cost vs accuracy.
    // Scale the cap with DOFs: small systems (< 25 DOF) get all pairs;
    // larger systems get more pairs (10*n) up to a hard cap of 500.
    int max_corners = min(500, max(10 * n, 300));
    int n_pairs = 0;
    int pair_cap = min(n * (n - 1) / 2, max_corners);
    
    // Define DOF block boundaries: translation (0-2), rotation (3-5), torsion (6..n-1)
    int block_start[3] = {0, 3, 6};
    int block_end[3]   = {3, 6, n};
    
    // Helper lambda: evaluate a corner point and set H[i][j] and H[j][i]
    auto eval_corner = [&](int ii, int jj) -> void {
        FLOATVec corner(n, 0.0f);
        for (int k = 0; k < n; k++) {
            corner[k] = xopt[k];
            if (k == ii) corner[k] += rho_eff;
            if (k == jj) corner[k] += rho_eff;
        }
        DOCKMol eval_mol;
        copy_molecule(eval_mol, ref_mol);
        float f_corner = 0.0f;
        if (eval_score(score, ref_mol, eval_mol, corner,
                       trans_step_size, rot_step_size, tors_step_size)) {
            f_corner = eval_mol.current_score + eval_mol.internal_energy;
            if (restrained_min) {
                f_corner += coefficient_restraint * calc_active_rmsd2(rmsd_ref, eval_mol);
            }
            if (1 + ii < nptmax && 1 + jj < nptmax) {
                float f_plus_i = fvals[1 + ii];
                float f_plus_j = fvals[1 + jj];
                if (f_plus_i < 1.0e5f && f_plus_j < 1.0e5f) {
                    float H_ij = (f_corner - f_plus_i - f_plus_j + fopt) / (rho_eff * rho_eff);
                    if (fabs(H_ij) < 1.0e-12f) H_ij = 0.0f;
                    H[ii][jj] = H_ij;
                    H[jj][ii] = H_ij;
                }
            }
        }
    };
    
    if (hessian_mode == "block_diag") {
        // Block-diagonal: iterate within each block separately
        for (int b = 0; b < 3; b++) {
            for (int i = block_start[b]; i < block_end[b] && n_pairs < pair_cap; i++) {
                for (int j = i + 1; j < block_end[b] && n_pairs < pair_cap; j++) {
                    if (Hdiag[i] < 1.0e-10f && Hdiag[j] < 1.0e-10f) continue;
                    n_pairs++;
                    eval_corner(i, j);
                }
            }
        }
        // Zero out cross-block entries (defensive)
        for (int b1 = 0; b1 < 3; b1++) {
            for (int b2 = b1 + 1; b2 < 3; b2++) {
                for (int i = block_start[b1]; i < block_end[b1]; i++) {
                    for (int j = block_start[b2]; j < block_end[b2]; j++) {
                        H[i][j] = 0.0f;
                        H[j][i] = 0.0f;
                    }
                }
            }
        }
    } else {
        // Full Hessian: iterate all pairs (i,j)
        for (int i = 0; i < n && n_pairs < pair_cap; i++) {
            for (int j = i + 1; j < n && n_pairs < pair_cap; j++) {
                if (Hdiag[i] < 1.0e-10f && Hdiag[j] < 1.0e-10f) continue;
                n_pairs++;
                eval_corner(i, j);
            }
        }
    }
}

/**********************************************************************/
// Update full quadratic model using symmetric rank-1 (SR1) update
// Called instead of the diagonal update when hessian_mode != "default"
/**********************************************************************/
void
BOBYQA_Minimizer::update_model_full(Base_Score & score, DOCKMol & ref_mol,
                                     DOCKMol & tmp_mol, DOCKMol & rmsd_ref,
                                     DOCKMol & best_mol,
                                     float trans_step_size, float rot_step_size,
                                     float tors_step_size,
                                     float fopt_before, float fnew)
{
    // SR1 (Symmetric Rank-1) update for the full Hessian
    // Given step s and function values f_old, f_new,
    // estimate y (gradient change) using component-wise secant.
    // SR1: H_new = H_old + (y - H*s)*(y - H*s)^T / ((y - H*s)^T * s)
    
    int i, j;
    std::vector<float> Hs(n, 0.0f);
    std::vector<float> ymHs(n, 0.0f);
    
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (j = 0; j < n; j++) sum += H[i][j] * s_step[j];
        Hs[i] = sum;
    }
    
    float denom = 0.0f;
    for (i = 0; i < n; i++) {
        float yi = 0.0f;
        if (fabs(s_step[i]) > 1.0e-10f) {
            yi = (fnew - fopt_before) / s_step[i];
        }
        ymHs[i] = yi - Hs[i];
        denom += ymHs[i] * s_step[i];
    }
    
    if (fabs(denom) > 1.0e-12f) {
        float inv_denom = 1.0f / denom;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                H[i][j] += ymHs[i] * ymHs[j] * inv_denom;
            }
        }
    }
    
    // Enforce block-diagonal structure after SR1 update
    if (hessian_mode == "block_diag") {
        int block_start[3] = {0, 3, 6};
        int block_end[3]   = {3, 6, n};
        for (int b1 = 0; b1 < 3; b1++) {
            for (int b2 = b1 + 1; b2 < 3; b2++) {
                for (int ii = block_start[b1]; ii < block_end[b1]; ii++) {
                    for (int jj = block_start[b2]; jj < block_end[b2]; jj++) {
                        H[ii][jj] = 0.0f;
                        H[jj][ii] = 0.0f;
                    }
                }
            }
        }
    }
    
    // Gradient update using secant estimate (not model-predicted change)
    // The old code used g[i] += Hs[i], where Hs = H_old * s was computed
    // BEFORE the SR1 update above. This meant g was updated with the old
    // Hessian curvature while H itself was updated to H_new, creating a
    // gradient-model inconsistency: g_new = g_old + H_old * s, but the
    // quadratic model now uses H_new, so the predicted gradient at the new
    // point should be g_old + H_new * s.
    //
    // Instead, we use the component-wise secant estimate:
    //   g_new[i] = (fnew - fopt_before) / s_step[i]
    // This is the actual gradient change observed from the function values,
    // independent of any Hessian approximation. It is always consistent
    // with the true function and requires no extra computation.
    for (i = 0; i < n; i++) {
        if (fabs(s_step[i]) > 1.0e-10f) {
            g[i] = (fnew - fopt_before) / s_step[i];
        }
    }
    
    for (i = 0; i < n; i++) {
        float h_ii = H[i][i];
        if (h_ii < 1.0e-12f) h_ii = 1.0e-12f;
        H[i][i] = 0.9f * h_ii + 0.1f * max(h_ii, Hdiag[i]);
        Hdiag[i] = H[i][i];
    }
}

/**********************************************************************/
// RESCUE: Recover from degenerate interpolation set
// Rebuilds interpolation set when points become too close together
/**********************************************************************/
void
BOBYQA_Minimizer::rescue(Base_Score & score, DOCKMol & mol, DOCKMol & ref_mol,
                          DOCKMol & tmp_mol, DOCKMol & rmsd_ref, DOCKMol & best_mol,
                          float trans_step_size, float rot_step_size, float tors_step_size,
                          float rho_beg_actual)
{
    cout << "BOBYQA RESCUE: rebuilding interpolation set" << endl;
    
    int i, j;
    
    // Reset interpolation points around current xopt
    // Place 2n+1 new points around xopt with smaller radius
    float rescue_rho = max(rho_beg_actual * 0.5f, 1.0e-4f);
    
    xpts[0] = xopt;
    fvals[0] = fopt;
    
    int n_axis_rescue = min(n, (nptmax - 1) / 2);
    
    for (i = 0; i < n_axis_rescue; i++) {
        int idx_p = 1 + i;
        xpts[idx_p] = xopt;
        xpts[idx_p][i] += rescue_rho;
        
        if (eval_score(score, ref_mol, tmp_mol, xpts[idx_p],
                       trans_step_size, rot_step_size, tors_step_size)) {
            fvals[idx_p] = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                fvals[idx_p] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
        } else {
            fvals[idx_p] = fopt + 1000.0f;
        }
        
        int idx_m = 1 + n_axis_rescue + i;
        if (idx_m < nptmax) {
            xpts[idx_m] = xopt;
            xpts[idx_m][i] -= rescue_rho;
            
            if (eval_score(score, ref_mol, tmp_mol, xpts[idx_m],
                           trans_step_size, rot_step_size, tors_step_size)) {
                fvals[idx_m] = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fvals[idx_m] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
            } else {
                fvals[idx_m] = fopt + 1000.0f;
            }
        }
    }
    
    // Rebuild gradient and Hessian
    float inv_rho2 = 1.0f / (rescue_rho * rescue_rho);
    for (i = 0; i < n_axis_rescue; i++) {
        float fp = fvals[1 + i];
        float fm = fvals[1 + n_axis_rescue + i];
        g[i] = (fp - fm) / (2.0f * rescue_rho);
        Hdiag[i] = (fp + fm - 2.0f * fopt) * inv_rho2;
        if (Hdiag[i] < 1.0e-12f) Hdiag[i] = 1.0e-12f;
    }
    for (i = n_axis_rescue; i < n; i++) {
        g[i] = 0.0f;
        Hdiag[i] = 1.0f;
    }
    
    delta = rescue_rho * 2.0f;
    
    // Find best point again
    for (i = 0; i < nptmax; i++) {
        if (fvals[i] < fopt) {
            fopt = fvals[i];
            xopt = xpts[i];
            kopt = i;
        }
    }
    // Rebuild full Hessian if using full quadratic model.
    // The old off-diagonal elements are from the degenerate interpolation set
    // and would mislead the CG solver. Rebuild from fresh corner evaluations.
    if (hessian_mode != "default") {
        build_full_model(score, ref_mol, tmp_mol, rmsd_ref, best_mol,
                         trans_step_size, rot_step_size, tors_step_size);
    }
}

/**********************************************************************/
// Adaptive restart: escape stagnation by resetting the trust region to
// restart_delta_scale * rho_beg and rebuilding the interpolation set
// around the current best point. This gives BOBYQA a fresh quadratic
// model when progress has plateaued.
/**********************************************************************/
void
BOBYQA_Minimizer::perform_adaptive_restart(Base_Score & score, DOCKMol & mol,
                                            DOCKMol & ref_mol, DOCKMol & tmp_mol,
                                            DOCKMol & rmsd_ref, DOCKMol & best_mol,
                                            float trans_step_size, float rot_step_size,
                                            float tors_step_size, float rho_beg_actual)
{
    cout << "BOBYQA ADAPTIVE RESTART " << (restart_count + 1)
         << ": stagnation detected, delta=" << delta
         << ", fopt=" << fopt << endl;

    int i;
    restart_count++;

    // Reset trust region to a multiple of the initial radius.
    // Default restart_delta_scale = 1.0 gives a soft restart at rho_beg.
    delta = rho_beg_actual * restart_delta_scale;
    delta = min(delta, rho_beg_actual * 10.0f);

    // Rebuild interpolation set around current best point with the new radius.
    xpts[0] = xopt;
    fvals[0] = fopt;

    int n_axis_restart = min(n, (nptmax - 1) / 2);

    for (i = 0; i < n_axis_restart; i++) {
        int idx_p = 1 + i;
        xpts[idx_p] = xopt;
        xpts[idx_p][i] += delta;

        if (eval_score(score, ref_mol, tmp_mol, xpts[idx_p],
                       trans_step_size, rot_step_size, tors_step_size)) {
            fvals[idx_p] = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                fvals[idx_p] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
        } else {
            fvals[idx_p] = fopt + 1000.0f;
        }

        int idx_m = 1 + n_axis_restart + i;
        if (idx_m < nptmax) {
            xpts[idx_m] = xopt;
            xpts[idx_m][i] -= delta;

            if (eval_score(score, ref_mol, tmp_mol, xpts[idx_m],
                           trans_step_size, rot_step_size, tors_step_size)) {
                fvals[idx_m] = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fvals[idx_m] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
            } else {
                fvals[idx_m] = fopt + 1000.0f;
            }
        }
    }

    // Find best point among restart points (axis evaluations may have found
    // something better than the previous xopt).
    kopt = 0;
    for (i = 0; i < nptmax; i++) {
        if (fvals[i] < fopt) {
            fopt = fvals[i];
            xopt = xpts[i];
            kopt = i;
        }
    }
    // Make sure best_mol corresponds to the chosen xopt.
    if (eval_score(score, ref_mol, tmp_mol, xopt,
                   trans_step_size, rot_step_size, tors_step_size)) {
        copy_crds(best_mol, tmp_mol);
    }
    xpts[0] = xopt;
    fvals[0] = fopt;
    kopt = 0;

    // Recompute diagonal gradient and Hessian from the new axis points.
    float inv_delta2 = 1.0f / (delta * delta);
    for (i = 0; i < n_axis_restart; i++) {
        float fp = fvals[1 + i];
        float fm = fvals[1 + n_axis_restart + i];
        g[i] = (fp - fm) / (2.0f * delta);
        Hdiag[i] = (fp + fm - 2.0f * fopt) * inv_delta2;
        if (Hdiag[i] < 1.0e-12f) Hdiag[i] = 1.0e-12f;
    }
    for (i = n_axis_restart; i < n; i++) {
        g[i] = 0.0f;
        Hdiag[i] = 1.0f;
    }

    // Clear off-diagonal Hessian before optional rebuild.
    if (hessian_mode != "default") {
        for (i = 0; i < n; i++) {
            std::fill(H[i].begin(), H[i].end(), 0.0f);
        }
    }

    // Rebuild full Hessian if using full quadratic model.
    if (hessian_mode != "default") {
        build_full_model(score, ref_mol, tmp_mol, rmsd_ref, best_mol,
                         trans_step_size, rot_step_size, tors_step_size);
    }
}

/**********************************************************************/
// Lanczos eigenvalue estimation on the quadratic model Hessian.
// For the diagonal model, eigenvalues are just the Hdiag entries.
// For the full n×n model, runs k Lanczos iterations to estimate the
// extreme eigenvalues, which detect indefiniteness for CG and give
// insight into the local landscape curvature.
/**********************************************************************/
void
BOBYQA_Minimizer::estimate_hessian_eigenvalues(int k,
                                                float *eig_min_out,
                                                float *eig_max_out)
{
    if (n <= 0) return;

    // ---- Diagonal model: trivially Hdiag entries ----
    if (hessian_mode == "default" || n == 1) {
        float emin = Hdiag[0];
        float emax = Hdiag[0];
        for (int i = 1; i < n; i++) {
            if (Hdiag[i] < emin) emin = Hdiag[i];
            if (Hdiag[i] > emax) emax = Hdiag[i];
        }
        if (eig_min_out) *eig_min_out = emin;
        if (eig_max_out) *eig_max_out = emax;
        return;
    }

    // ---- Full model: Lanczos iteration on H ----
    if (k <= 0 || k > n) k = min(5, n);
    if (k < 1) k = 1;

    // Lanczos coefficients: T_k = tridiag(beta[i-1], alpha[i], beta[i])
    // Only beta[0..k-2] are valid; beta[k-1] is unused.
    std::vector<float> alpha(k, 0.0f);
    std::vector<float> beta(k, 0.0f);

    // Lanczos vectors: V[j][0..n-1] for j = 0..k
    // We only need V[j] and V[j-1] at each step, but store all for clarity.
    std::vector< std::vector<float> > V(k + 1,
                                         std::vector<float>(n, 0.0f));

    // Initialize V[0] with a deterministic pseudo-random vector.
    // A fixed seed ensures reproducible eigenvalues across runs.
    float norm = 0.0f;
    int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        V[0][i] = (float)seed / (float)0x7fffffff * 2.0f - 1.0f;
        norm += V[0][i] * V[0][i];
    }
    norm = sqrt(norm);
    if (norm < 1.0e-20f) norm = 1.0f;
    for (int i = 0; i < n; i++) V[0][i] /= norm;

    // Lanczos main loop
    std::vector<float> w(n, 0.0f);
    int actual_k = k;
    for (int j = 0; j < k; j++) {
        // w = H * V[j]
        for (int i = 0; i < n; i++) {
            w[i] = 0.0f;
            for (int l = 0; l < n; l++) {
                w[i] += H[i][l] * V[j][l];
            }
        }

        // alpha[j] = V[j]^T * w
        alpha[j] = 0.0f;
        for (int i = 0; i < n; i++)
            alpha[j] += V[j][i] * w[i];

        // w = w - alpha[j] * V[j] - beta[j-1] * V[j-1]
        for (int i = 0; i < n; i++) {
            w[i] -= alpha[j] * V[j][i];
            if (j > 0) w[i] -= beta[j-1] * V[j-1][i];
        }

        // beta[j] = ||w||
        if (j < k - 1) {
            float bdot = 0.0f;
            for (int i = 0; i < n; i++) bdot += w[i] * w[i];
            beta[j] = sqrt(bdot);

            if (beta[j] > 1.0e-15f) {
                for (int i = 0; i < n; i++)
                    V[j+1][i] = w[i] / beta[j];
            } else {
                // Invariant subspace found; no need for more iterations.
                actual_k = j + 1;
                break;
            }
        }
    }

    // ---- Compute eigenvalues of T (actual_k × actual_k tridiagonal) ----
    // Uses the Sturm sequence property: for symmetric tridiagonal T with
    // diagonal a[0..m-1] and sub-diagonal b[0..m-2], the number of sign
    // changes in the Sturm sequence at shift λ equals the number of
    // eigenvalues of T less than λ.  We bisect within Gershgorin bounds
    // to locate each eigenvalue.

    // Gershgorin: |λ - a[i]| ≤ |b[i-1]| + |b[i]|
    float lo = alpha[0], hi = alpha[0];
    for (int i = 0; i < actual_k; i++) {
        float rad = 0.0f;
        if (i > 0)     rad += fabs(beta[i-1]);
        if (i < actual_k-1) rad += fabs(beta[i]);
        float left  = alpha[i] - rad;
        float right = alpha[i] + rad;
        if (left  < lo) lo = left;
        if (right > hi) hi = right;
    }
    // Pad bounds to avoid edge cases
    float pad = (hi - lo) * 0.01f + 1.0e-10f;
    lo -= pad;
    hi += pad;

    // Sturm sequence: count eigenvalues < lambda
    auto sturm_count = [&](float lambda) -> int {
        float p0 = 1.0f;
        float p1 = alpha[0] - lambda;
        int sc = (p0 * p1 < 0.0f) ? 1 : 0;
        for (int i = 1; i < actual_k; i++) {
            float pi = (alpha[i] - lambda) * p1
                       - beta[i-1] * beta[i-1] * p0;
            if (pi * p1 < 0.0f) sc++;
            p0 = p1;
            p1 = pi;
        }
        return sc;
    };

    // Bisect to find each eigenvalue
    std::vector<float> eigs(actual_k, 0.0f);
    for (int idx = 0; idx < actual_k; idx++) {
        float l = lo, r = hi;
        // We want the eigenvalue where exactly idx eigenvalues are below it.
        // sturm_count(r) >= idx+1 (since r is an upper bound on all eigs)
        // sturm_count(l) < idx+1 (since l is a lower bound)
        for (int iter = 0; iter < 80; iter++) {
            float m = (l + r) * 0.5f;
            if (sturm_count(m) >= idx + 1)
                r = m;
            else
                l = m;
        }
        eigs[idx] = (l + r) * 0.5f;
    }

    // The eigenvalues are sorted ascending (by construction of Sturm bisection).
    if (eig_min_out) *eig_min_out = eigs[0];
    if (eig_max_out) *eig_max_out = eigs[actual_k - 1];
}

/**********************************************************************/
// Multi-start minimization: run do_minimize multiple times with random
// starting points and return the best result
/**********************************************************************/
// #pragma GCC optimize("O0") // kept for debugging; remove for production
float
BOBYQA_Minimizer::multi_start_minimize(Base_Score & score, DOCKMol & mol,
                                        FLOATVec & vertex, int max_iter_param,
                                        float score_converge, float trans_step_size,
                                        float rot_step_size, float tors_step_size)
{
    if (multi_start_restarts <= 0) {
        return do_minimize(score, mol, vertex, max_iter_param, score_converge,
                           trans_step_size, rot_step_size, tors_step_size);
    }
    
    cout << "BOBYQA multi-start: " << (multi_start_restarts + 1) << " runs" << endl;
    
    // First, run PRELIM inline to collect valid interpolation points as starting seeds
    DOCKMol ref_mol, tmp_mol, rmsd_ref;
    copy_molecule(ref_mol, mol);
    copy_molecule(tmp_mol, mol);
    if (restrained_min) copy_molecule(rmsd_ref, mol);
    
    struct PrelimPoint { FLOATVec vertex; float score; };
    vector<PrelimPoint> prelim_points;
    const float PENALTY_SCORE = 1.0e6f;
    // Same npt default as do_minimize (see comment there).
    // int default_npt = max(2 * n + 1, (n + 1) * (n + 2) / 4);
    int np = (npt > 0) ? npt : 2 * n + 1;
    if (np < n + 2) np = n + 2;
    int n_axis = min(n, (np - 1) / 2);
    float rho_beg_actual = (rho_beg > 0.0f) ? rho_beg : 1.0f;
    FLOATVec start_vertex;
    if (!xopt.empty()) {
        start_vertex = xopt;
    } else {
        start_vertex.resize(n, 0.0f);
    }
    vector<FLOATVec> xpts_temp(np);
    vector<float> fvals_temp(np, PENALTY_SCORE);
    xpts_temp[0] = start_vertex;
    
    // Evaluate starting point
    if (eval_score(score, ref_mol, tmp_mol, xpts_temp[0],
                   trans_step_size, rot_step_size, tors_step_size)) {
        float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
        if (restrained_min) {
            score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
        }
        fvals_temp[0] = score_val;
        prelim_points.push_back({xpts_temp[0], score_val});
    }
    
    // Axis points: +/- rho_beg along each coordinate
    for (int i = 0; i < n_axis; i++) {
        int idx_p = 1 + i;
        xpts_temp[idx_p] = xpts_temp[0];
        xpts_temp[idx_p][i] += rho_beg_actual;
        if (eval_score(score, ref_mol, tmp_mol, xpts_temp[idx_p],
                       trans_step_size, rot_step_size, tors_step_size)) {
            float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
            fvals_temp[idx_p] = score_val;
            prelim_points.push_back({xpts_temp[idx_p], score_val});
        }
        int idx_m = 1 + n_axis + i;
        xpts_temp[idx_m] = xpts_temp[0];
        xpts_temp[idx_m][i] -= rho_beg_actual;
        if (eval_score(score, ref_mol, tmp_mol, xpts_temp[idx_m],
                       trans_step_size, rot_step_size, tors_step_size)) {
            float score_val = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                score_val += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
            fvals_temp[idx_m] = score_val;
            prelim_points.push_back({xpts_temp[idx_m], score_val});
        }
    }
    
    // Write PRELIM points to temp file - observable side effect compiler can't optimize away
    {
        ofstream prelim_debug("/tmp/bobyqa_prelim_debug.txt");
        prelim_debug << "PRELIM points collected: " << prelim_points.size() << endl;
        for (size_t i = 0; i < prelim_points.size(); ++i) {
            prelim_debug << "Point " << i << ": score=" << prelim_points[i].score << " vertex[0]=" << prelim_points[i].vertex[0] << endl;
        }
        prelim_debug.close();
    }
    cerr << "DEBUG PRELIM: collected " << prelim_points.size() << " valid points" << endl;
    
    // Sort by score (best first)
    sort(prelim_points.begin(), prelim_points.end(),
         [](const PrelimPoint &a, const PrelimPoint &b) { return a.score < b.score; });
    
    cout << "BOBYQA multi-start: collected " << prelim_points.size()
         << " valid PRELIM points, running " << (multi_start_restarts + 1) << " runs" << endl;
    
    float best_final_score = 1.0e10f;
    FLOATVec best_vertex = vertex;
    DOCKMol best_mol;
    copy_molecule(best_mol, mol);
    
    // Run 0: original starting point
    float score_val = do_minimize(score, mol, vertex,
                                   max_iter_param, score_converge,
                                   trans_step_size, rot_step_size, tors_step_size);
    
    cout << "BOBYQA run 0 (original): score = " << score_val << endl;
    
    if (score_val < best_final_score) {
        best_final_score = score_val;
        best_vertex = vertex;
        copy_molecule(best_mol, mol);
    }
    
    // Subsequent runs: use best PRELIM points as starting positions
    for (int run = 1; run <= multi_start_restarts; run++) {
        DOCKMol run_mol;
        copy_molecule(run_mol, mol);
        FLOATVec run_vertex = vertex;
        
        if (run <= (int)prelim_points.size()) {
            // Use PRELIM interpolation point as starting position
            run_vertex = prelim_points[run - 1].vertex;
            cout << "BOBYQA run " << run << " (PRELIM point " << run - 1
                 << ", score=" << prelim_points[run - 1].score << "): ";
        } else {
            // Fallback: small perturbation of original
            const float perturb_scale = 0.10f;
            FLOATVec orig_vertex = vertex;
            for (int i = 6; i < n; i++) {
                float perturb = ((float)rand() / RAND_MAX - 0.5f) * tors_step_size * perturb_scale;
                run_vertex[i] = orig_vertex[i] + perturb;
            }
            cout << "BOBYQA run " << run << " (perturbed): ";
        }
        
        score_val = do_minimize(score, run_mol, run_vertex,
                                   max_iter_param, score_converge,
                                   trans_step_size, rot_step_size, tors_step_size);
        
        cout << score_val << endl;
        
        if (score_val < best_final_score) {
            best_final_score = score_val;
            best_vertex = run_vertex;
            copy_molecule(best_mol, run_mol);
        }
    }
    
    // Restore best result
    vertex = best_vertex;
    copy_molecule(mol, best_mol);
    
    return best_final_score;
}

/**********************************************************************/
// Run PRELIM phase and collect valid interpolation points as multi-start seeds

/**********************************************************************/
// Estimate noise level from ratio history
// Returns standard deviation of ratio values
/**********************************************************************/
// float
// BOBYQA_Minimizer::estimate_noise_level()
// {
//     if (ratio_history.size() < 3) return 0.0f;
    
//     float mean = 0.0f;
//     for (float r : ratio_history) mean += r;
//     mean /= ratio_history.size();
    
//     float var = 0.0f;
//     for (float r : ratio_history) {
//         float diff = r - mean;
//         var += diff * diff;
//     }
    
//     // Keep window bounded
//     if (ratio_history.size() > noise_window) {
//         ratio_history.erase(ratio_history.begin());
//     }
    
//     return sqrt(var / ratio_history.size());
// }

/**********************************************************************/
// Perform adaptive restart when stagnation is detected
// Preserves best point, increases trust region, rebuilds interpolation set
/**********************************************************************/
// void
// BOBYQA_Minimizer::perform_adaptive_restart(Base_Score & score, DOCKMol & mol,
//                                               DOCKMol & ref_mol, DOCKMol & tmp_mol,
//                                               DOCKMol & rmsd_ref,
//                                               float trans_step_size, float rot_step_size,
//                                               float tors_step_size)
// {
//     cout << "BOBYQA ADAPTIVE RESTART " << (restart_count + 1) 
//          << ": stagnation detected, delta=" << delta 
//          << ", noise=" << noise_level << endl;
    
//     // Preserve best point
//     FLOATVec saved_xopt = xopt;
//     float saved_fopt = fopt;
//     DOCKMol saved_best_mol;
//     copy_molecule(saved_best_mol, mol);
    
//     // Increase trust region significantly
//     delta = min(delta * restart_delta_scale, rho_beg * 5.0f);
//     if (delta < rho_beg) delta = rho_beg;
    
//     // Rebuild interpolation set around current best with larger radius
//     float restart_rho = delta;
    
//     xpts[0] = saved_xopt;
//     fvals[0] = saved_fopt;
    
//     int n_axis_restart = min(n, (nptmax - 1) / 2);
    
//     for (int i = 0; i < n_axis_restart; i++) {
//         // Positive direction
//         int idx_p = 1 + i;
//         xpts[idx_p] = saved_xopt;
//         xpts[idx_p][i] += restart_rho;
//         if (eval_score(score, ref_mol, tmp_mol, xpts[idx_p],
//                        trans_step_size, rot_step_size, tors_step_size)) {
//             fvals[idx_p] = tmp_mol.current_score + tmp_mol.internal_energy;
//             if (restrained_min) {
//                 fvals[idx_p] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
//             }
//         } else {
//             fvals[idx_p] = saved_fopt + 1000.0f;
//         }
        
//         // Negative direction
//         int idx_m = 1 + n_axis_restart + i;
//         if (idx_m < nptmax) {
//             xpts[idx_m] = saved_xopt;
//             xpts[idx_m][i] -= restart_rho;
//             if (eval_score(score, ref_mol, tmp_mol, xpts[idx_m],
//                            trans_step_size, rot_step_size, tors_step_size)) {
//                 fvals[idx_m] = tmp_mol.current_score + tmp_mol.internal_energy;
//                 if (restrained_min) {
//                     fvals[idx_m] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
//                 }
//             } else {
//                 fvals[idx_m] = saved_fopt + 1000.0f;
//             }
//         }
//     }
    
//     // Rebuild gradient and diagonal Hessian
//     float inv_rho2 = 1.0f / (restart_rho * restart_rho);
//     for (int i = 0; i < n_axis_restart; i++) {
//         float fp = fvals[1 + i];
//         float fm = fvals[1 + n_axis_restart + i];
//         g[i] = (fp - fm) / (2.0f * restart_rho);
//         Hdiag[i] = (fp + fm - 2.0f * saved_fopt) * inv_rho2;
//         if (Hdiag[i] < 1.0e-12f) Hdiag[i] = 1.0e-12f;
//     }
//     for (int i = n_axis_restart; i < n; i++) {
//         g[i] = 0.0f;
//         Hdiag[i] = 1.0f;
//     }
    
//     // Rebuild full Hessian if enabled
//     if (hessian_mode != "default") {
//         build_full_model(score, ref_mol, tmp_mol, rmsd_ref, saved_best_mol,
//                          trans_step_size, rot_step_size, tors_step_size);
//     }
    
//     // Restore best point
//     xopt = saved_xopt;
//     fopt = saved_fopt;
//     copy_molecule(mol, saved_best_mol);
    
//     // Reset tracking
//     stagnation_count = 0;
//     restart_count++;
//     diagnostics.restarts++;
//     ratio_history.clear();
//     noise_level = 0.0f;
// }

/**********************************************************************/
// Record convergence diagnostics
/**********************************************************************/
// void
// BOBYQA_Minimizer::record_diagnostics(const std::string & reason)
// {
//     diagnostics.termination_reason = reason;
//     diagnostics.final_delta = delta;
    
//     // Compute final gradient norm
//     float norm_g = 0.0f;
//     for (int i = 0; i < n; i++) norm_g += g[i] * g[i];
//     diagnostics.final_gradient_norm = sqrt(norm_g);
    
//     diagnostics.converged_normally = (reason == "delta_converged" || reason == "gradient_converged");
    
//     // Output diagnostics for analysis
//     cout << "BOBYQA DIAGNOSTICS:"
//          << " iter=" << diagnostics.iterations
//          << " evals=" << diagnostics.function_evals
//          << " final_delta=" << diagnostics.final_delta
//          << " grad_norm=" << diagnostics.final_gradient_norm
//          << " avg_ratio=" << diagnostics.avg_ratio
//          << " noise=" << diagnostics.noise_level
//          << " restarts=" << diagnostics.restarts
//          << " rescues=" << diagnostics.rescue_calls
//          << " converged=" << (diagnostics.converged_normally ? "yes" : "no")
//          << " reason=" << diagnostics.termination_reason
//          << endl;
// }
