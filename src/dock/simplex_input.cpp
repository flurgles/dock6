#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include "master_score.h"
#include "minimizer.h"
#include "simplex.h"

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

        // Advanced minimizer parameters (accept both DOCK6.12 and current names)
        if (parm.param_exists("use_advanced_simplex_parameters")) {
            advanced_min_params =
                (parm.query_param("use_advanced_simplex_parameters", "no",
                                  "yes no") == "yes");
        } else {
            advanced_min_params =
                (parm.query_param("advanced_minimizer_parameters", "no",
                                  "yes no") == "yes");
        }

        if (advanced_min_params) {
            cout <<
                "Advanced (Nelder-Mead) Minimizer Parameters (DEPRECATED - use basic parameters)"
                << endl;
            cout <<
                "=================================================================================================="
                << endl;
        } else {
            cout << "Basic Minimizer Parameters" << endl;
            cout <<
                "================================================================================"
                << endl;
        }

        if (advanced_min_params) {
            cout <<
                "==================================================================================================="
                << endl;
            cout << "SIMULATION_OPTIONS_DOCK:" << endl;
            cout <<
                "==================================================================================================="
                << endl;
        }

        // DTM - 05-09-08 - Minimization Parameters
        //               - moving away from basic / advanced to one set of
        //               minimizer parameters
        //               - 'minimize_ligand' triggers min, final min and secondary
        //               min
        //               - separate anchors and growth parameters below
        //               - separate parameters for minimization of final pose

        minimize_ligand =
            (parm.query_param("minimize_ligand", "yes", "yes no") == "yes");

        if (minimize_ligand) {

            cout << "minimize_ligand = " << minimize_ligand << endl;
            cout << endl;

            if (advanced_min_params) {
                max_iterations =
                    atoi(parm.query_param("simplex_max_iterations",
                                          "500").c_str());
                score_converge =
                    atof(parm.query_param("simplex_score_converge",
                                          "0.1").c_str());
                cycle_converge =
                    atof(parm.query_param("simplex_cycle_converge",
                                          "1.0").c_str());
                trans_step_size =
                    atof(parm.query_param("simplex_trans_step_size",
                                          "2.0").c_str());
                rot_step_size =
                    atof(parm.query_param("simplex_rot_step_size",
                                          "0.05").c_str());
                tors_step_size =
                    atof(parm.query_param("simplex_tors_step_size",
                                          "10.0").c_str());
                max_cycles =
                    atoi(parm.query_param("simplex_max_cycles",
                                          "1").c_str());
            } else {
                max_iterations =
                    atoi(parm.query_param("simplex_max_iterations",
                                          "500").c_str());
                score_converge =
                    atof(parm.query_param("simplex_score_converge",
                                          "0.1").c_str());
                cycle_converge =
                    atof(parm.query_param("simplex_cycle_converge",
                                          "1.0").c_str());
                trans_step_size =
                    atof(parm.query_param("simplex_trans_step",
                                          "1.0").c_str());
                rot_step_size =
                    atof(parm.query_param("simplex_rot_step",
                                          "0.1").c_str());
                tors_step_size =
                    atof(parm.query_param("simplex_tors_step",
                                          "10.0").c_str());
                max_cycles =
                    atoi(parm.query_param("simplex_max_cycles",
                                          "1").c_str());
            }

            cout << "Maximum iterations: " << max_iterations << endl;
            cout << "Score convergence: " << score_converge << endl;
            cout << "Cycle convergence: " << cycle_converge << endl;
            cout << "Translation step: " << trans_step_size << endl;
            cout << "Rotation step: " << rot_step_size << endl;
            cout << "Torsion step size: " << tors_step_size << endl;

            // Adaptive Nelder-Mead (Gao & Han 2012): controls whether the
            // expansion/contraction/shrink coefficients are scaled by the
            // problem dimension n to prevent simplex collapse in
            // high-dimensional flexible docking problems.
            //   "no"         — classical fixed coefficients (default)
            //   "yes"        — full adaptive coefficients
            //   "dim_aware"  — sigmoid blend from fixed to adaptive based
            //                  on problem dimensionality
            {
                string mode_str =
                    parm.query_param("simplex_adaptive", "no",
                                     "yes no dim_aware");
                if (mode_str == "yes")
                    simplex_mode = 1;
                else if (mode_str == "dim_aware")
                    simplex_mode = 2;
                else
                    simplex_mode = 0;

                cout << "Adaptive Nelder-Mead (Gao & Han 2012): "
                     << mode_str;
                if (simplex_mode == 2) {
                    simplex_crossover =
                        atoi(parm.query_param("simplex_adaptive_crossover",
                                              "17").c_str());
                    cout << " (crossover: " << simplex_crossover
                         << " bonds)";
                }
                cout << endl;
            }

            // Minimization parameters for rigid anchor
            // If these are not set in the input file, the values previously
            // read are used.
            use_min_rigid_anchor =
                (parm.query_param("minimize_anchor", "yes",
                                  "yes no") == "yes");
            if (use_min_rigid_anchor) {
                cout << "\nAnchor Minimization Parameters" << endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                cout << "minimize_anchor = yes" << endl;
                anchor_min_max_iterations =
                    atoi(parm.query_param("simplex_anchor_max_iterations",
                                          "500").c_str());
                cout << "Maximum iterations: " << anchor_min_max_iterations <<
                    endl;

                if (advanced_min_params) {
                    anchor_min_max_cycles =
                        atoi(parm.query_param("simplex_anchor_max_cycles",
                                              "1").c_str());
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
                        atof(parm.
                             query_param("simplex_anchor_rot_step",
                                         "0.1").c_str());
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
                cout << "Maximum cycles: " << anchor_min_max_cycles << endl;
                cout << "Score convergence: " << anchor_min_score_converge <<
                    endl;
                cout << "Cycle convergence: " << anchor_min_cycle_converge <<
                    endl;
                cout << "Translation step: " << anchor_min_trans_step_size <<
                    endl;
                cout << "Rotation step: " << anchor_min_rot_step_size << endl;
                cout << "Torsion step size: " << anchor_min_tors_step_size <<
                    endl;
            }

            // Minimization parameters for flexible growth
            use_min_flex_growth =
                (parm.query_param("minimize_flexible_growth", "yes",
                                  "yes no") == "yes");
            if (use_min_flex_growth) {
                cout << "\nFlexible Growth Minimization Parameters" << endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                cout << "minimize_flexible_growth = yes" << endl;

                // If ramp is on, default is 250; otherwise 500
                if (use_min_flex_growth_ramp) {
                    flex_min_max_iterations =
                        atoi(parm.query_param("simplex_grow_max_iterations",
                                              "250").c_str());
                } else {
                    flex_min_max_iterations =
                        atoi(parm.query_param("simplex_grow_max_iterations",
                                              "500").c_str());
                }
                if (flex_min_max_iterations < 0) {
                    cout <<
                        "ERROR:  simplex_grow_max_iterations cannot be negative.  Program will terminate."
                        << endl;
                    exit(0);
                }

                cout << "Maximum iterations: " << flex_min_max_iterations <<
                    endl;

                flex_min_torsion_iterations =
                    atoi(parm.query_param("simplex_grow_tors_premin_iterations",
                                          "0").c_str());
                if (flex_min_torsion_iterations < 0) {
                    cout <<
                        "ERROR:  simplex_grow_tors_premin_iterations cannot be negative. Program will terminate."
                        << endl;
                    exit(0);
                }

                cout << "Torsion-only iterations: " <<
                    flex_min_torsion_iterations << endl;

                if (advanced_min_params) {
                    flex_min_max_cycles =
                        atoi(parm.query_param("simplex_grow_max_cycles",
                                              "1").c_str());
                    if (flex_min_max_cycles <= 0) {
                        cout <<
                            "ERROR:  simplex_grow_max_cycles must be an integer greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_score_converge =
                        atof(parm.query_param("simplex_grow_score_converge",
                                              "0.1").c_str());
                    if (flex_min_score_converge <= 0.0) {
                        cout <<
                            "ERROR:  simplex_grow_score_converge must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_cycle_converge =
                        atof(parm.query_param("simplex_grow_cycle_converge",
                                              "1.0").c_str());
                    if (flex_min_cycle_converge <= 0.0) {
                        cout <<
                            "ERROR:  simplex_grow_cycle_converge must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_trans_step_size =
                        atof(parm.query_param("simplex_grow_trans_step",
                                              "1.0").c_str());
                    if (flex_min_trans_step_size <= 0.0) {
                        cout <<
                            "ERROR:  simplex_grow_trans_step must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_rot_step_size =
                        atof(parm.query_param("simplex_grow_rot_step",
                                              "0.1").c_str());
                    if (flex_min_rot_step_size <= 0.0) {
                        cout <<
                            "ERROR:  simplex_grow_rot_step must be a float greater than zero.  Program will terminate."
                            << endl;
                        exit(0);
                    }
                    flex_min_tors_step_size =
                        atof(parm.query_param("simplex_grow_tors_step",
                                              "10.0").c_str());
                    if (flex_min_tors_step_size <= 0.0) {
                        cout <<
                            "ERROR:  simplex_grow_tors_step must be a float greater than zero.  Program will terminate."
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

                cout << "Maximum cycles: " << flex_min_max_cycles << endl;
                cout << "Score convergence: " << flex_min_score_converge <<
                    endl;
                cout << "Cycle convergence: " << flex_min_cycle_converge <<
                    endl;
                cout << "Translation step: " << flex_min_trans_step_size <<
                    endl;
                cout << "Rotation step: " << flex_min_rot_step_size << endl;
                cout << "Torsion step size: " << flex_min_tors_step_size <<
                    endl;
            }

            // ////////////////////////////////////////////////////////////////////
            // Minimization parameters for flexible growth (ramp convergence)
            use_min_flex_growth_ramp =
                (parm.
                 query_param("minimize_flexible_growth_ramp", "yes",
                             "yes no") == "yes");
            if (use_min_flex_growth_ramp) {
                cout << "\nFlexible Growth Minimization Parameters (Ramp)" <<
                    endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                cout << "minimize_flexible_growth_ramp = yes" << endl;

                initial_score_converge =
                    atof(parm.query_param("simplex_initial_score_coverge",
                                          "5").c_str());
                if (initial_score_converge <= flex_min_score_converge) {
                    cout <<
                        "ERROR:  simplex_initial_score_coverge must be larger than score converge value.  Program will terminate."
                        << endl;
                    exit(0);
                }

                cout << "Initial score convergence: " <<
                    initial_score_converge << endl;
            }

            // Minimization parameters for final pose minimization
            // Accept both DOCK6.12 (simplex_final_min) and current name
            if (parm.param_exists("simplex_final_min")) {
                final_min =
                    (parm.query_param("simplex_final_min", "no",
                                      "yes no") == "yes");
            } else {
                final_min =
                    (parm.query_param("final_min_pose", "no",
                                      "yes no") == "yes");
            }

            // General simplex parameters (DOCK6.12 / FLX.sh names)
            random_seed = atoi(parm.query_param("simplex_random_seed", "0").c_str());
            restrained_min = (parm.query_param("simplex_restraint_min", "no", "yes no") == "yes");

            if (final_min) {
                cout << "\nFinal Pose Minimization Parameters" << endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                final_min_max_iterations =
                    atoi(parm.query_param("final_min_max_iterations",
                                          "500").c_str());
                final_min_max_cycles =
                    atoi(parm.query_param("final_min_max_cycles",
                                          "1").c_str());
                final_min_score_converge =
                    atof(parm.query_param("final_min_score_converge",
                                          "0.1").c_str());
                final_min_cycle_converge =
                    atof(parm.query_param("final_min_cycle_converge",
                                          "1.0").c_str());
                final_min_trans_step_size =
                    atof(parm.query_param("final_min_trans_step_size",
                                          "2.0").c_str());
                final_min_rot_step_size =
                    atof(parm.query_param("final_min_rot_step_size",
                                          "0.05").c_str());
                final_min_tors_step_size =
                    atof(parm.query_param("final_min_tors_step_size",
                                          "10.0").c_str());
                final_min_rep_radius_scale =
                    atof(parm.query_param("final_min_rep_radius_scale",
                                          "1.0").c_str());
                cout << "Maximum iterations: " << final_min_max_iterations <<
                    endl;
                cout << "Maximum cycles: " << final_min_max_cycles << endl;
                cout << "Score convergence: " << final_min_score_converge <<
                    endl;
                cout << "Cycle convergence: " << final_min_cycle_converge <<
                    endl;
                cout << "Translation step: " << final_min_trans_step_size <<
                    endl;
                cout << "Rotation step: " << final_min_rot_step_size << endl;
                cout << "Torsion step size: " << final_min_tors_step_size <<
                    endl;
            }

            // Secondary minimization
            secondary_min_pose =
                (parm.query_param("secondary_min_pose", "no",
                                  "yes no") == "yes");
            if (secondary_min_pose) {
                cout << "\nSecondary Minimization Parameters" << endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                cout << "secondary_min_pose = yes" << endl;
                secondary_min_max_iterations =
                    atoi(parm.query_param("secondary_min_max_iterations",
                                          "500").c_str());
                secondary_min_max_cycles =
                    atoi(parm.query_param("secondary_min_max_cycles",
                                          "1").c_str());
                secondary_min_score_converge =
                    atof(parm.query_param("secondary_min_score_converge",
                                          "0.1").c_str());
                secondary_min_cycle_converge =
                    atof(parm.query_param("secondary_min_cycle_converge",
                                          "1.0").c_str());
                secondary_min_trans_step_size =
                    atof(parm.query_param("secondary_min_trans_step_size",
                                          "2.0").c_str());
                secondary_min_rot_step_size =
                    atof(parm.query_param("secondary_min_rot_step_size",
                                          "0.05").c_str());
                secondary_min_tors_step_size =
                    atof(parm.query_param("secondary_min_tors_step_size",
                                          "10.0").c_str());

                cout << "Maximum iterations: " <<
                    secondary_min_max_iterations << endl;
                cout << "Score convergence: " <<
                    secondary_min_score_converge << endl;
                cout << "Cycle convergence: " <<
                    secondary_min_cycle_converge << endl;
                cout << "Translation step: " <<
                    secondary_min_trans_step_size << endl;
                cout << "Rotation step: " << secondary_min_rot_step_size <<
                    endl;
                cout << "Torsion step size: " <<
                    secondary_min_tors_step_size << endl;
            }
        }

        ///////////////////////////////////////////////////////
        //print out trailing end of table
        if (advanced_min_params) {
            cout <<
                "------------------------------------------------------------------------------------------"
                << endl;
        } else {
            cout <<
                "================================================================================"
                << endl;
        }

        cout << endl;

        if (score.c_int.use_primary_score) {
            // if scoring with internal energy in primary score, then it's not used yet
            // in flex growth for pruning — only grid score matters for flex growth.
            // For now we just disable internal energy; once the score is computed after
            // minimization, the (grid_score + internal_energy) is used for scoring.
        }

        if (restrained_min) {
            coefficient_restraint =
                atof(parm.query_param("simplex_coefficient_restraint",
                                      "10.0").c_str());
        }

    }
}
