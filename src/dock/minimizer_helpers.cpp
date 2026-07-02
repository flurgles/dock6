#include <iostream>
#include <vector>
#include <float.h>
#include "dockmol.h"
#include "master_score.h"
#include "minimizer.h"
using namespace std;


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
