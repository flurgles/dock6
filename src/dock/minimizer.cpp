#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "master_score.h"
#include "minimizer.h"
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
        // loop over simplex cycles
        while ((current_cycle < max_cycles)
               && ((distance > cycle_converge) || (current_cycle == 0))) {
 
            // call simplex minimizer
            do_minimize(score, mol, vertex, max_iterations, score_converge,
                             trans_step_size, rot_step_size, tors_step_size);
 
            // compute the distance moved, and re-zero the vertex vector
            distance = 0;
            for (i = 0; i < vertex.size(); i++) {
                distance += vertex[i] * vertex[i];
                vertex[i] = 0.0;
            }
            distance = sqrt(distance) / (float) (current_cycle + 1);
 
            current_cycle++;
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

void
Minimizer::scale_vector(FLOATVec & new_vec, FLOATVec & vertex,
                                        float trans_step_size,
                                        float rot_step_size,
                                        float tors_step_size)
{
    int             i;

    new_vec.resize(vertex.size(), 0);

    for (i = 0; i < 3; i++) {
        new_vec[i] =
            (vertex[i] * trans_step_size) / (float) (current_cycle + 1);
        new_vec[i + 3] =
            (vertex[i + 3] * rot_step_size) / (float) (current_cycle + 1);
    }

    for (i = 6; i < vertex.size(); i++) {
        new_vec[i] =
            (vertex[i] * tors_step_size) / ((float) (current_cycle + 1) *
                                            (float) (torsion_scale_factors
                                                     [i - 6]));
    }

}

bool
Minimizer::eval_score(Base_Score & score, DOCKMol & ref_mol,
                                 DOCKMol & tmp_mol, FLOATVec & vertex,
                                 float trans_step_size, float rot_step_size,
                                 float tors_step_size)
{
    FLOATVec        new_vec;
    bool            return_val;

    // cerr << "DEBUG: eval_score start, vertex[0]=" << vertex[0] << endl;
    copy_crds(tmp_mol, ref_mol);
    // cerr << "DEBUG: copy_crds done" << endl;
    scale_vector(new_vec, vertex, trans_step_size, rot_step_size,
                         tors_step_size);
    // cerr << "DEBUG: scale_vector done, new_vec.size=" << new_vec.size() << endl;
    vector_to_dockmol(tmp_mol, new_vec);
    // cerr << "DEBUG: vector_to_dockmol done" << endl;

    // ----- Internal energy (anchor-optimized + distance cutoff) -----
    //
    // Per-torsion mask optimization (DISABLED — see minimizer.h for why):
    // The Nelder-Mead simplex changes ALL torsions per vertex, so a
    // subset-delta path only fires during the initial axis-aligned vertices
    // (< 0.5% of eval calls).  The mask overhead negates the savings at
    // DT100 scale.  Keep the simple full recompute:
    //   - Rigid anchor: skip entirely (skip_internal_energy flag, set by
    //     Minimizer::minimize_rigid_anchor)
    //   - Growth / final min: Base_Score::compute_ligand_internal_energy
    //     has its own 5 A distance cutoff that prunes far-apart pairs.
    //
    // To re-enable the per-torsion mask for a future minimizer:
    //   1. Uncomment the member data in minimizer.h
    //   2. Uncomment build_torsion_pair_indices() call below
    //   3. Uncomment the three-path case below (this block)
    //   4. Uncomment build_torsion_pair_indices() definition
    //   5. Uncomment the cache-reset lines in id_torsions()
    //
    if (!skip_internal_energy) {
        score.compute_ligand_internal_energy(tmp_mol);
    }
    // cerr << "DEBUG: compute_ligand_internal_energy done" << endl;

    return_val = score.compute_score(tmp_mol);
    // cerr << "DEBUG: compute_score done" << endl;

    //cout << "In Minimizer::simplex_score: "
    //     << "score = " << tmp_mol.current_score
    //     << ";int  = " << tmp_mol.internal_energy << endl;

    return return_val;
}


// +++++++++++++++++++++++++++++++++++++++++
// Build per-torsion nb_int pair masks for incremental internal-energy
// optimization (DISABLED — see minimizer.h for rationale).
//
// To re-enable: uncomment this function, the member data in minimizer.h,
// the three-path case in eval_score(), and the cache-reset in id_torsions().
//
// void
// Minimizer::build_torsion_pair_indices(Base_Score & score, DOCKMol & ref_mol)
// {
//     torsion_pair_indices.resize(torsions.size());
//     last_num_torsions = (int)torsions.size();
//
//     for (int t = 0; t < (int)torsions.size(); t++) {
//         int bond_num = torsions[t].bond_num;
//         int a2 = torsions[t].atom2;
//         int a3 = torsions[t].atom3;
//
//         // Directed bond index matching the convention in DOCKMol::set_torsion.
//         int dir_idx = 2 * bond_num;
//         if (a2 >= a3) dir_idx++;  // reverse direction
//
//         // Mark atoms on the moving side of this rotatable bond.
//         std::vector<char> atom_affected(ref_mol.num_atoms, 0);
//         const INTVec& children = ref_mol.atom_child_list[dir_idx];
//         for (int a : children) {
//             atom_affected[a] = 1;
//         }
//
//         // Collect nb_int indices where at least one atom is affected.
//         for (int i = 0; i < (int)score.nb_int.size(); i++) {
//             if (atom_affected[score.nb_int[i].first]
//              || atom_affected[score.nb_int[i].second]) {
//                 torsion_pair_indices[t].push_back(i);
//             }
//         }
//     }
// }


void
Minimizer::vector_to_dockmol(DOCKMol & mol, FLOATVec & v)
{
    DOCKVector      com, // Centre of Mass
                    dv;  // Translation Vector
    int             i;
    float           rmat[3][3];  // Rotational Matrix
    float           quat[3];
    float           current_angle,
                    new_angle;

    // calc COM of active atoms
    com.x = 0;
    com.y = 0;
    com.z = 0;

    for (i = 0; i < mol.num_atoms; i++) {
        if (mol.atom_active_flags[i]) {
            com.x += mol.x[i];
            com.y += mol.y[i];
            com.z += mol.z[i];
        }
    }

    com.x = com.x / mol.num_active_atoms;
    com.y = com.y / mol.num_active_atoms;
    com.z = com.z / mol.num_active_atoms;

    // build a rotation matrix
    quat[0] = v[3];
    quat[1] = v[4];
    quat[2] = v[5];

    get_matrix_from_quaternion(rmat, quat);

    // build translation vector
    dv.x = v[0];
    dv.y = v[1];
    dv.z = v[2];

    // transform mol
    transform(mol, rmat, dv, com);

    // set new torsion angles
    for (i = 6; i < v.size(); i++) {

        if (bond_vectors[torsions[i - 6].bond_num] == -1) {     // if bond
                                                                // directions
                                                                // don't matter

            current_angle =
                mol.get_torsion(torsions[i - 6].atom1, torsions[i - 6].atom2,
                                torsions[i - 6].atom3, torsions[i - 6].atom4);
            new_angle = (PI / 180.0) * (current_angle + v[i]);
            mol.set_torsion(torsions[i - 6].atom1, torsions[i - 6].atom2,
                            torsions[i - 6].atom3, torsions[i - 6].atom4,
                            new_angle);

        } else {                // if bond directions do matter (during flex
                                // growth)

            if (torsions[i - 6].atom2 == bond_vectors[torsions[i - 6].bond_num]) {
                current_angle =
                    mol.get_torsion(torsions[i - 6].atom1,
                                    torsions[i - 6].atom2,
                                    torsions[i - 6].atom3,
                                    torsions[i - 6].atom4);
                new_angle = (PI / 180.0) * (current_angle + v[i]);
                mol.set_torsion(torsions[i - 6].atom1, torsions[i - 6].atom2,
                                torsions[i - 6].atom3, torsions[i - 6].atom4,
                                new_angle);
            }

            if (torsions[i - 6].atom3 == bond_vectors[torsions[i - 6].bond_num]) {
                current_angle =
                    mol.get_torsion(torsions[i - 6].atom4,
                                    torsions[i - 6].atom3,
                                    torsions[i - 6].atom2,
                                    torsions[i - 6].atom1);
                new_angle = (PI / 180.0) * (current_angle + v[i]);
                mol.set_torsion(torsions[i - 6].atom4, torsions[i - 6].atom3,
                                torsions[i - 6].atom2, torsions[i - 6].atom1,
                                new_angle);
            }

        }

        //cout << "torsion:: " << (PI / 180.0) * current_angle << " " << new_angle << endl; 

    }

}

float
Minimizer::calc_active_rmsd2(DOCKMol & ref, DOCKMol & conf)
{
 // This function is used to tether the molecule to prevent the previous growth step.
 // this function calculates rmsd2 between the active atoms of the ref structure and the same atoms of conf.

 // The rmsd2 can be thought of as the mean of the squared distances.

 // only heavy atom rmsd2 is reported
 // the rmsd2 of the active atoms in the reference is reported

//    if (! restrained_min) { // if restrained minimum is not used do not compute the rmsd2
//        return 0;
//    }


    int    i;
    float  rmsd2 = 0.0;
    int    atom_num_total = 0;

    
    for (i = 0; i < ref.num_atoms; i++) {
          if (ref.atom_active_flags[i] && ref.amber_at_heavy_flag[i]){
                    rmsd2 +=
                        ((ref.x[i] - conf.x[i]) * (ref.x[i] - conf.x[i]) +
                         (ref.y[i] - conf.y[i]) * (ref.y[i] - conf.y[i]) +
                         (ref.z[i] - conf.z[i]) * (ref.z[i] - conf.z[i]));

                    atom_num_total += 1;
          }
    }

    if (atom_num_total > 0)
        rmsd2 = rmsd2 / (float) atom_num_total;
    else
        rmsd2 = 0.0;

    return rmsd2;
}

