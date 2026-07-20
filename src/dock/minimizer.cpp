#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "master_score.h"
#include "minimizer.h"
#include "dockmol.h"
using namespace std;

void
Minimizer::id_torsions(DOCKMol & mol, FLOATVec & vertex)
{
    //cout << "Entering id_torsions ..." << endl;
    int             i,
                    j,
                    max_central,
                    central;
    int             a1,
                    a2,
                    a3,
                    a4;
    int             nbr;
    vector < int   >nbrs;
    TORSION         tmp_torsion;

    BREADTH_SEARCH  bfs;

    torsions.clear();

    // loop over bonds- add flex bonds to torsion list
    for (i = 0; i < mol.num_bonds; i++) {
        if (mol.bond_active_flags[i]) {
            // Fochmod Sep 29, 2014 
            // altered to make minimization decisions only if bond is not a
            // rotor and if the mol2 says that it is a single bond
            //cout<< i << "  "<< mol.amber_bt_id[i] << endl;
            if (mol.bond_is_rotor(i)) {
                //cout<< i << "  "<< mol.bond_types[i] << endl;
                if (mol.bond_types[i] == "1") {
                    //cout<< i << "  "<< mol.bond_types[i] << endl;
                    torsions.push_back(tmp_torsion);
                    torsions[torsions.size() - 1].bond_num = i;
                    vertex.push_back(0.0);
                }
            }
        }
    }

    //cout << "torsions.size() = " << torsions.size() <<endl;
    // ID the inter-segment rot-bonds
    for (i = 0; i < torsions.size(); i++) {

        a2 = mol.bonds_origin_atom[torsions[i].bond_num];
        a3 = mol.bonds_target_atom[torsions[i].bond_num];

        max_central = -1;
        // nbrs = mol.get_atom_neighbors(a2);
        nbrs = mol.neighbor_list[a2];

        for (j = 0; j < nbrs.size(); j++) {
            nbr = nbrs[j];
            if (nbr != a3) {
                central = bfs.get_search_radius(mol, nbr, a2);
                if (central > max_central) {
                    a1 = nbr;
                    max_central = central;
                }
            }
        }

        max_central = -1;
        // nbrs = mol.get_atom_neighbors(a3);
        nbrs = mol.neighbor_list[a3];

        for (j = 0; j < nbrs.size(); j++) {
            nbr = nbrs[j];
            if (nbr != a2) {
                central = bfs.get_search_radius(mol, nbr, a3);
                if (central > max_central) {
                    a4 = nbr;
                    max_central = central;
                }
            }
        }

        torsions[i].atom1 = a1;
        torsions[i].atom2 = a2;
        torsions[i].atom3 = a3;
        torsions[i].atom4 = a4;
    }

    // Per-torsion internal-energy cache reset (DISABLED — see minimizer.h).
    // Uncomment for the mask-based approach with a future minimizer.
    // torsion_pair_indices.clear();
    // last_num_torsions = -1;
    // ref_cache_valid = false;

}

void
Minimizer::minimize(Base_Score & score, DOCKMol & mol,
                            FLOATVec & vertex, int max_cycles,
                            float cycle_converge, int max_iterations,
                            float score_converge, float trans_step_size,
                            float rot_step_size, float tors_step_size)
{
    // TEB 2010-03-10
    // only minimize if both max_iterations and max_cycles are more than 0.
    if (max_iterations > 0 && max_cycles > 0 ){

        int             i;
        float           distance;

        // initialize the minimization structures
        current_cycle = 0;
        distance = 0.0;

        // Best-across-cycles tracking (mirrors the GPU ConformerPool
        // behaviour).  The CPU simplex keeps only the LAST cycle's pose,
        // but a prior cycle may have found a better-scoring conformation.
        // Save the best-scoring cycle's pose here and restore it at the end
        // so the caller's molecule reflects the best result, not just the
        // final cycle's.
        float           best_cycle_score = 1e30f;
        float           last_cycle_score = 1e30f;
        bool            has_best_cycle   = false;
        DOCKMol         best_cycle_mol;

        // loop over simplex cycles
        while ((current_cycle < max_cycles)
               && ((distance > cycle_converge) || (current_cycle == 0))) {
            // call simplex minimizer (updates mol to this cycle's best pose)
            last_cycle_score = do_minimize(score, mol, vertex, max_iterations,
                                            score_converge, trans_step_size,
                                            rot_step_size, tors_step_size);

            if (!has_best_cycle || (last_cycle_score < best_cycle_score)) {
                best_cycle_score = last_cycle_score;
                copy_molecule(best_cycle_mol, mol);
                has_best_cycle = true;
            }

            // compute the distance moved, and re-zero the vertex vector
            distance = 0;
            for (i = 0; i < vertex.size(); i++) {
                distance += vertex[i] * vertex[i];
                vertex[i] = 0.0;
            }
            distance = sqrt(distance) / (float) (current_cycle + 1);

            current_cycle++;
        }

        // Restore the best-scoring cycle's pose if it was not the last cycle.
        if (has_best_cycle && (best_cycle_score < last_cycle_score)) {
            copy_molecule(mol, best_cycle_mol);
        }
    }

}

void
Minimizer::minimize_rigid_anchor(DOCKMol & mol, Master_Score & score)
{
    int             i;
    FLOATVec        vertex;

    if (use_min_rigid_anchor) {
        // initialize degrees of freedom as all zeros (rigid DOF only)
        vertex.clear();

        // rigid DOF
        for (i = 0; i < 6; i++)
            vertex.push_back(0.000);

        bond_vectors.clear();
        bond_vectors.resize(mol.num_bonds, -1);

        // OPTIMIZATION: cache internal energy once before minimization.
        // The anchor has no torsional DOFs (only 6 rigid-body DOF:
        // rotation + translation), so inter-atomic distances are
        // invariant — the internal energy never changes during anchor
        // minimization.  Computing it once and reusing the cached value
        // avoids O(nb_int) loop overhead on every eval_score() call.
        score.primary_score->compute_ligand_internal_energy(mol);
        skip_internal_energy = true;

        // mc_premin_override = false;
        minimize(*score.primary_score, mol, vertex, anchor_min_max_cycles,
                 anchor_min_cycle_converge, anchor_min_max_iterations,
                 anchor_min_score_converge, anchor_min_trans_step_size,
                 anchor_min_rot_step_size, anchor_min_tors_step_size);

        skip_internal_energy = false;
    }

}

void
Minimizer::minimize_flexible_ramp_growth(DOCKMol & mol, Master_Score & score, INTVec & bvectors, int current_layer, int num_layers) 
{
     float 	     adjustable_score_converge;
     int             i;
     FLOATVec        vertex;
     //5.0 was used for the b due to Guilherme's preliminary data.  	
     float diff_interval  = initial_score_converge - flex_min_score_converge;
     float adj_unit = diff_interval / (num_layers-1); 
     adj_unit = adj_unit * (float)current_layer;  	
     adjustable_score_converge = initial_score_converge - adj_unit;
 
     if (use_min_flex_growth) {
        vertex.clear();

	for (i = 0; i < 6; i++)
		vertex.push_back(0.000);

  	id_torsions(mol, vertex);
	
  	torsion_scale_factors.resize(torsions.size(), 1);

	bond_vectors.clear();
        bond_vectors = bvectors;
        
        
	minimize(*score.primary_score, mol, vertex, flex_min_max_cycles,
                 flex_min_cycle_converge, flex_min_torsion_iterations,
                 adjustable_score_converge, 0, 0, flex_min_tors_step_size);

	minimize(*score.primary_score, mol, vertex, flex_min_max_cycles,
                 flex_min_cycle_converge, flex_min_max_iterations,
                 adjustable_score_converge, flex_min_trans_step_size,
                 flex_min_rot_step_size, flex_min_tors_step_size);

   }
}

void
Minimizer::minimize_flexible_growth(DOCKMol & mol, Master_Score & score,
                                            INTVec & bvectors)
{
    int             i;
    FLOATVec        vertex;
    if (use_min_flex_growth) {
        // initialize degrees of freedom as all zeros (all DOF)
        vertex.clear();

        // rigid DOF
        for (i = 0; i < 6; i++)
            vertex.push_back(0.000);

        // flex DOF
        id_torsions(mol, vertex);

        torsion_scale_factors.resize(torsions.size(), 1);

        bond_vectors.clear();
        bond_vectors = bvectors;

        // trent & sudipto 01-05-2009
        // just minimize the torsional degrees of freedom
	minimize(*score.primary_score, mol, vertex, flex_min_max_cycles,
                 flex_min_cycle_converge, flex_min_torsion_iterations,
                 flex_min_score_converge, 0, 0, flex_min_tors_step_size);
	
        // NOTE: same pre-min is done in minimize_final_pose
        // sudipto believes that setting trans_step_size=0 && rot_step_size=0
        // in minimize might be using minimization iterations doing 0 size
        // steps of trans or rotational minimization


        // minimize all degrees of freedom for the remainder of the time
        // ceil means we run at least 1 step unless frac_time = 1
        minimize(*score.primary_score, mol, vertex, flex_min_max_cycles,
                 flex_min_cycle_converge, flex_min_max_iterations,
                 flex_min_cycle_converge, flex_min_trans_step_size,
                 flex_min_rot_step_size, flex_min_tors_step_size);	
		 
    }

}

void
Minimizer::minimize_final_pose(DOCKMol & mol, Master_Score & score, AMBER_TYPER &typer)
{
    int             i;
    FLOATVec        vertex;

    //cout << "Entering minimize_final_pose" << endl;
    //cout << minimize_ligand << " " << use_min_rigid_anchor << " " << use_min_flex_growth << endl;

    if (minimize_ligand) {
        // perform minimization if anchor and grow minimization was not called
        if (!use_min_rigid_anchor && !use_min_flex_growth) {
            // initialize degrees of freedom as all zeros (all DOF)
            vertex.clear();

            // rigid DOF
            for (i = 0; i < 6; i++)
                vertex.push_back(0.000);

            // flex DOF
            id_torsions(mol, vertex);

            torsion_scale_factors.resize(torsions.size(), 1);

            bond_vectors.clear();
            bond_vectors.resize(mol.num_bonds, -1);

            // trent & sudipto 03-10-2010
            // if (num of iteration) is zero or num_cycles is zero , 
            // the function minimize() will not call do_minimize()

            // trent & sudipto 01-05-2009
            // just minimize the torsional degrees of freedom
            minimize(*score.primary_score, mol, vertex, max_cycles,
                     cycle_converge, torsion_iterations,
                     score_converge, 0, 0, tors_step_size);

            // NOTE: same pre-min is done in minimize_flexible_growth
            // sudipto believes that setting trans_step_size=0 && rot_step_size=0
            // in minimize might be using minimization iterations doing 0 size 
            // steps of trans or rotational minimization

            minimize(*score.primary_score, mol, vertex, max_cycles,
                     cycle_converge, max_iterations, score_converge,
                     trans_step_size, rot_step_size, tors_step_size);

            // compute the final score for the molecule
            score.compute_primary_score(mol);

        }

/*      // this code has been superseded by global use of internal energy minimization
        if (final_min) {
            cout << "In Minimizer::minimize_final_pose" << endl;
        // initialize degrees of freedom as all zeros (all DOF)
            vertex.clear();

            // rigid DOF
            for (i = 0; i < 6; i++)
                vertex.push_back(0.000);

            // flex DOF
            id_torsions(mol, vertex);

            torsion_scale_factors.resize(torsions.size(), 1);

            bond_vectors.clear();
            bond_vectors.resize(mol.num_bonds, -1);

         // DTM 11-12-08 - fix bug with bad energies in final min - due to re-init vdw energy
            //score.primary_score->rep_radius_scale = final_min_rep_radius_scale;
            //score.primary_score->init_vdw_energy(typer, 6, 12);

            minimize(*score.primary_score, mol, vertex, final_min_max_cycles,
                     final_min_cycle_converge, final_min_max_iterations,
                     final_min_score_converge, final_min_trans_step_size,
                     final_min_rot_step_size, final_min_tors_step_size);

            // compute the final score for the molecule
            score.compute_primary_score(mol);
        }

*/

    }
}

void
Minimizer::minimize_pose_final_min(DOCKMol & mol, Master_Score & score)
{
    int             i;
    FLOATVec        vertex;

    //cout << "Entering minimize_final_pose" << endl;
    //cout << minimize_ligand << " " << use_min_rigid_anchor << " " << use_min_flex_growth << endl;
    //
        if (final_min) {
            cout << "In Minimizer::minimize_final_pose" << endl;
        // initialize degrees of freedom as all zeros (all DOF)
            vertex.clear();

            // rigid DOF
            for (i = 0; i < 6; i++)
                vertex.push_back(0.000);

            // flex DOF
            id_torsions(mol, vertex);

            torsion_scale_factors.resize(torsions.size(), 1);

            bond_vectors.clear();
            bond_vectors.resize(mol.num_bonds, -1);

         // DTM 11-12-08 - fix bug with bad energies in final min - due to re-init vdw energy
            //score.primary_score->rep_radius_scale = final_min_rep_radius_scale;
            //score.primary_score->init_vdw_energy(typer, 6, 12);

            minimize(*score.primary_score, mol, vertex, final_min_max_cycles,
                     final_min_cycle_converge, final_min_max_iterations,
                     final_min_score_converge, final_min_trans_step_size,
                     final_min_rot_step_size, final_min_tors_step_size);

            // compute the final score for the molecule
            score.compute_primary_score(mol);
        }


}

void
Minimizer::secondary_minimize_pose(DOCKMol & mol, Master_Score & score)
{
    int             i;
    FLOATVec        vertex;
        // perform additional round of minimization using secondary scoring
        // function
    if(minimize_ligand){
        if (secondary_min_pose) {

            // initialize degrees of freedom as all zeros (all DOF)
            vertex.clear();

            // rigid DOF
            for (i = 0; i < 6; i++)
                vertex.push_back(0.000);

            // flex DOF
            id_torsions(mol, vertex);

            torsion_scale_factors.resize(torsions.size(), 1);

            bond_vectors.clear();
            bond_vectors.resize(mol.num_bonds, -1);

            minimize(*score.secondary_score, mol, vertex,
                     secondary_min_max_cycles, secondary_min_cycle_converge,
                     secondary_min_max_iterations, secondary_min_score_converge,
                     secondary_min_trans_step_size, secondary_min_rot_step_size,
                     secondary_min_tors_step_size);

            // compute the final score for the molecule
            score.compute_secondary_score(mol);

        }
    }
}

// scale_vector moved to minimizer_transform.cpp

// eval_score moved to minimizer_helpers.cpp


/* ================================================================= */
/*  GPU Batch Scoring — moved to minimizer_gpu.cpp                    */
/* ================================================================= */

// vector_to_dockmol moved to minimizer_transform.cpp
// calc_active_rmsd2 moved to minimizer_transform.cpp


