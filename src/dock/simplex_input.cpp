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

        // Advanced minimizer parameters
        advanced_min_params =
            (parm.query_param("advanced_minimizer_parameters", "no",
                              "yes no") == "yes");

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

        // for now, set the simplex parameters automatically
        // user can override the defaults
        // if advanced parameters are used, individual step sizes override
        // auto values for trans_rot step size and tors step size
        if (advanced_min_params) {
            trans_step_size =
                atof(parm.query_param("trans_step_size", "2.0").c_str());
            rot_step_size =
                atof(parm.query_param("rot_step_size", "0.05").c_str());
            tors_step_size =
                atof(parm.query_param("tors_step_size", "10.0").c_str());
        } else {
            // auto set the step size for rigid anchor
            trans_step_size =
                atof(parm.query_param("trans_step_size", "2.0").c_str());
            rot_step_size =
                atof(parm.query_param("rot_step_size", "0.05").c_str());
            tors_step_size =
                atof(parm.query_param("tors_step_size", "10.0").c_str());
        }

        cout << "trans_step_size = " << trans_step_size << endl;
        cout << "rot_step_size = " << rot_step_size << endl;
        cout << "tors_step_size = " << tors_step_size << endl;

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
                    atof(parm.query_param("simplex_trans_step_size",
                                          "2.0").c_str());
                rot_step_size =
                    atof(parm.query_param("simplex_rot_step_size",
                                          "0.05").c_str());
                tors_step_size =
                    atof(parm.query_param("simplex_tors_step_size",
                                          "10.0").c_str());
            }

            cout << "Maximum iterations: " << max_iterations << endl;
            cout << "Score convergence: " << score_converge << endl;
            cout << "Cycle convergence: " << cycle_converge << endl;
            cout << "Translation step: " << trans_step_size << endl;
            cout << "Rotation step: " << rot_step_size << endl;
            cout << "Torsion step size: " << tors_step_size << endl;

            // Adaptive Nelder-Mead (Gao & Han 2012): scale the expansion,
            // contraction and shrink coefficients by the problem dimension n
            // to prevent simplex collapse in high-dimensional problems.
            // Defaults to OFF for backward compatibility (bit-identical
            // output with the classical fixed coefficients).
            simplex_adaptive =
                (parm.query_param("simplex_adaptive", "no",
                                  "yes no") == "yes");
            cout << "Adaptive Nelder-Mead (Gao & Han 2012): "
                 << (simplex_adaptive ? "yes" : "no") << endl;

            // Minimization parameters for rigid anchor
            // If these are not set in the input file, the values previously
            // read are used.
            use_min_rigid_anchor =
                (parm.query_param("minimize_rigid_anchor", "yes",
                                  "yes no") == "yes");
            if (use_min_rigid_anchor) {
                cout << "\nAnchor Minimization Parameters" << endl;
                cout <<
                    "--------------------------------------------------------------------------------"
                    << endl;
                cout << "minimize_rigid_anchor = yes" << endl;
                anchor_min_max_iterations =
                    atoi(parm.query_param("anchor_min_max_iterations",
                                          "500").c_str());
                anchor_min_max_cycles =
                    atoi(parm.query_param("anchor_min_max_cycles",
                                          "1").c_str());
                anchor_min_score_converge =
                    atof(parm.query_param("anchor_min_score_converge",
                                          "0.1").c_str());
                anchor_min_cycle_converge =
                    atof(parm.query_param("anchor_min_cycle_converge",
                                          "1.0").c_str());
                anchor_min_trans_step_size =
                    atof(parm.query_param("anchor_min_trans_step_size",
                                          "2.0").c_str());
                anchor_min_rot_step_size =
                    atof(parm.query_param("anchor_min_rot_step_size",
                                          "0.05").c_str());
                anchor_min_tors_step_size =
                    atof(parm.query_param("anchor_min_tors_step_size",
                                          "10.0").c_str());
                cout << "Maximum iterations: " << anchor_min_max_iterations <<
                    endl;
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
                flex_min_max_iterations =
                    atoi(parm.query_param("flex_min_max_iterations",
                                          "500").c_str());
                flex_min_torsion_iterations =
                    atoi(parm.query_param("flex_min_torsion_iterations",
                                          "500").c_str());
                flex_min_max_cycles =
                    atoi(parm.query_param("flex_min_max_cycles",
                                          "1").c_str());
                flex_min_score_converge =
                    atof(parm.query_param("flex_min_score_converge",
                                          "0.1").c_str());
                flex_min_cycle_converge =
                    atof(parm.query_param("flex_min_cycle_converge",
                                          "1.0").c_str());
                flex_min_trans_step_size =
                    atof(parm.query_param("flex_min_trans_step_size",
                                          "2.0").c_str());
                flex_min_rot_step_size =
                    atof(parm.query_param("flex_min_rot_step_size",
                                          "0.05").c_str());
                flex_min_tors_step_size =
                    atof(parm.query_param("flex_min_tors_step_size",
                                          "10.0").c_str());
                cout << "Maximum iterations: " << flex_min_max_iterations <<
                    endl;
                cout << "Torsion-only iterations: " <<
                    flex_min_torsion_iterations << endl;
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
                flex_min_max_iterations =
                    atoi(parm.query_param("flex_min_max_iterations",
                                          "500").c_str());
                flex_min_torsion_iterations =
                    atoi(parm.query_param("flex_min_torsion_iterations",
                                          "500").c_str());
                flex_min_max_cycles =
                    atoi(parm.query_param("flex_min_max_cycles",
                                          "1").c_str());
                initial_score_converge =
                    atof(parm.query_param("initial_score_converge",
                                          "10.0").c_str());
                flex_min_score_converge =
                    atof(parm.query_param("flex_min_score_converge",
                                          "0.1").c_str());
                flex_min_cycle_converge =
                    atof(parm.query_param("flex_min_cycle_converge",
                                          "1.0").c_str());
                flex_min_trans_step_size =
                    atof(parm.query_param("flex_min_trans_step_size",
                                          "2.0").c_str());
                flex_min_rot_step_size =
                    atof(parm.query_param("flex_min_rot_step_size",
                                          "0.05").c_str());
                flex_min_tors_step_size =
                    atof(parm.query_param("flex_min_tors_step_size",
                                          "10.0").c_str());
                cout << "Maximum iterations: " << flex_min_max_iterations <<
                    endl;
                cout << "Torsion-only iterations: " <<
                    flex_min_torsion_iterations << endl;
                cout << "Maximum cycles: " << flex_min_max_cycles << endl;
                cout << "Initial score convergence: " <<
                    initial_score_converge << endl;
                cout << "Final score convergence: " <<
                    flex_min_score_converge << endl;
                cout << "Cycle convergence: " << flex_min_cycle_converge <<
                    endl;
                cout << "Translation step: " << flex_min_trans_step_size <<
                    endl;
                cout << "Rotation step: " << flex_min_rot_step_size << endl;
                cout << "Torsion step size: " << flex_min_tors_step_size <<
                    endl;
            }

            // Minimization parameters for final pose minimization
            final_min =
                (parm.query_param("final_min_pose", "no",
                                  "yes no") == "yes");
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
