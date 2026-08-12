#include <iostream>
#include "amber_typer.h"
#include "base_score.h"
#include "dockmol.h"

using namespace std;


// +++++++++++++++++++++++++++++++++++++++++
// static member initializers

const string Base_Score::DELIMITER    = "########## ";
// 20 is magic
const int    Base_Score::FLOAT_WIDTH  = 20;
// length of names plus length of longest score name, to align with Scores;
// width extened by Yuchen to align all score outputs 10/24/2016.
const int    Base_Score::STRING_WIDTH = 17 + 19;


// +++++++++++++++++++++++++++++++++++++++++
Base_Score::Base_Score()
{
    use_internal_energy = false;
    vdwA = NULL;
    vdwB = NULL;
    ie_vdwA = NULL;
    ie_vdwB = NULL;
    ie_soft_delta = 0.0;
    ie_att_exp = 6;
    ie_rep_exp = 12;
    ie_vdw_cutoff_sq = 1e10f;  //default 1e10f for no effective cutoff. set to 25.0 for nb pair calc.
    nb_int.clear();
    rep_radius_scale = 1.0;
}

// +++++++++++++++++++++++++++++++++++++++++
Base_Score::~Base_Score()
{
    delete[]vdwA;
    delete[]vdwB;
    delete[]ie_vdwA;
    delete[]ie_vdwB;
    nb_int.clear();
}

// +++++++++++++++++++++++++++++++++++++++++
// Calculate Internal Energy of the ligand kxr 010506
// DTM - 11-12-08
// I changed this to only compute interactions between atoms of different segments.
// Also to skip any atoms that are inactive, so
// this can be used during flexible growth.

// Trent & Sudipto 2009-02-12 -- This function is also used for
// single point and rigid minimization. (torsions move during fin min.)
float
Base_Score::compute_ligand_internal_energy(DOCKMol & mol)
{
    float           distancesq;
    unsigned int    i;
    unsigned int    a1,a2;
    float           ligand_internal_energy = 0.0;
    float           int_vdw_rep = 0.0;

    if (use_internal_energy) {
        for (i = 0; i < nb_int.size(); i++) {
            // nb_int is a neighbor list of all non-bonded pair of ligand atoms
               a1 = nb_int[i].first;
               a2 = nb_int[i].second;

               if ( mol.atom_active_flags[a1] && mol.atom_active_flags[a2] ) {
                        distancesq = ((mol.x[a1] - mol.x[a2])*(mol.x[a1] - mol.x[a2]))
                                   + ((mol.y[a1] - mol.y[a2])*(mol.y[a1] - mol.y[a2]))
                                   + ((mol.z[a1] - mol.z[a2])*(mol.z[a1] - mol.z[a2]));

                    // Distance cutoff: pairs beyond ie_vdw_cutoff_sq contribute
                    // negligible 1/r^12 repulsion and are skipped for speed.
                    if (distancesq < ie_vdw_cutoff_sq) {
                        if (ie_soft_delta > 0.0) {
                            // Soft-core Repulsive LJ: E = A / (r^2 + delta)^6
                            // No singularity at r=0, finite repulsive barrier
                            float denom = distancesq + ie_soft_delta;
                            float denom_cubed = denom * denom * denom;
                            int_vdw_rep += (ie_vdwA[a1]*ie_vdwA[a2]) / (denom_cubed * denom_cubed);
                        }
                        else if (ie_rep_exp == 12) {
                            float distance6;
                            distance6 = distancesq*distancesq*distancesq;
                            int_vdw_rep += (ie_vdwA[a1]*ie_vdwA[a2]) / (distance6*distance6);
                        }
                        else
                            int_vdw_rep += (ie_vdwA[a1]*ie_vdwA[a2]) / pow(distancesq, float(ie_rep_exp/2.0));
                    }  // end distance cutoff
                }
        }
    }
    ligand_internal_energy = int_vdw_rep;
    mol.internal_energy = ligand_internal_energy;
    return ligand_internal_energy;
}


// +++++++++++++++++++++++++++++++++++++++++
// Compute VDW repulsion for one non-bonded atom pair.
// Extracted from the inner loop of compute_ligand_internal_energy so that
// Minimizer can sum over a reduced subset when only some torsions change.
// Called by compute_internal_energy_subset() to speed up computation
float
Base_Score::compute_pair_internal_energy(const DOCKMol& mol, int a1, int a2) const
{
    if (!mol.atom_active_flags[a1] || !mol.atom_active_flags[a2])
        return 0.0f;

    float distancesq = ((mol.x[a1] - mol.x[a2]) * (mol.x[a1] - mol.x[a2])
                      + (mol.y[a1] - mol.y[a2]) * (mol.y[a1] - mol.y[a2])
                      + (mol.z[a1] - mol.z[a2]) * (mol.z[a1] - mol.z[a2]));

    // Distance cutoff: pairs beyond ie_vdw_cutoff_sq contribute negligible
    // 1/r^12 repulsion and are skipped for speed.
    if (distancesq >= ie_vdw_cutoff_sq)
        return 0.0f;

    if (ie_soft_delta > 0.0) {
        // Soft-core: E = A / (r^2 + delta)^6
        // No singularity at r=0, finite repulsive barrier
        float denom = distancesq + ie_soft_delta;
        float denom_cubed = denom * denom * denom;
        return (ie_vdwA[a1] * ie_vdwA[a2]) / (denom_cubed * denom_cubed);
    }
    else if (ie_rep_exp == 12) {
        float distance6 = distancesq * distancesq * distancesq;
        return (ie_vdwA[a1] * ie_vdwA[a2]) / (distance6 * distance6);
    }
    else {
        return (ie_vdwA[a1] * ie_vdwA[a2])
               / pow(distancesq, (float)(ie_rep_exp / 2.0));
    }
}


// +++++++++++++++++++++++++++++++++++++++++
// Sum VDW repulsion energy over a specific subset of nb_int pairs.
// The \a indices vector contains indices into the nb_int array.
// The same pair may appear multiple times (caller is responsible for dedup).
// Called by Minimizer to speed up computation when only some torsions change.
float
Base_Score::compute_internal_energy_subset(const DOCKMol& mol,
                                            const std::vector<int>& indices) const
{
    if (!use_internal_energy)
        return 0.0f;

    float total = 0.0f;
    for (int idx : indices) {
        total += compute_pair_internal_energy(mol, nb_int[idx].first,
                                                  nb_int[idx].second);
    }
    return total;
}


// Bad design warning: sudipto
// This function stores data for each specific ligand inside class base_score
// this is bad because this is the parent class of all scoring functions,
// and should be a stateless function wrt each ligand being scored
// However, since the simplex minimizer needs to call this function as well,
// but there is no way out but to store the precalculated VDW A&B as well as
// non-bonded atoms list in here. Let me know you if find a better place
// to store the precalculated data.
// +++++++++++++++++++++++++++++++++++++++++
void
Base_Score::initialize_internal_energy(DOCKMol & mol)
{
    //ie_att_exp, ie_rep_exp, ie_diel must be set before calling this
    //clear the ie_vdw arrays
    delete[]ie_vdwA;
    ie_vdwA = NULL;
    delete[]ie_vdwB;
    ie_vdwB = NULL;

    ie_vdwA = new float[mol.num_atoms];
    ie_vdwB = new float[mol.num_atoms];

    // calculate vdwA and vdwB terms
    for (int i = 0; i < mol.num_atoms; i++) {
        ie_vdwA[i] = sqrt(mol.amber_at_well_depth[i] *
                 (ie_att_exp / (ie_rep_exp - ie_att_exp)) *
                 pow((2 * mol.amber_at_radius[i]), ie_rep_exp));

        ie_vdwB[i] = sqrt(mol.amber_at_well_depth[i] *
                 (ie_rep_exp / (ie_rep_exp - ie_att_exp)) *
                 pow((2 * mol.amber_at_radius[i]), ie_att_exp));
    }

   // neighbor list for internal energy
   nb_int.clear();  //clear list of non-bonded interactions

   // Distance-squared cutoff: pairs farther apart than this contribute
   // negligible 1/r^12 repulsion and are skipped in the inner loop.
   // 25.0 ≈ 5.0 Å — captures all physically meaningful VDW repulsion.
   // Set to 1e10 or larger to disable the cutoff entirely.
   ie_vdw_cutoff_sq = 25.0f;

   int a1,a2;      // atom number counters
   INTPair temp;   // to push_back into array nb_int

   // this double loop makes sure we never check the same atom against itself
   for (a1 = 0; a1 < mol.num_atoms - 1; a1++) {
       for (a2 = a1 + 1; a2 < mol.num_atoms; a2++) {

          // nonbondpair is used in the if statement below.
          // if neither method is set then all are false

          bool nonbondpair = false;

          switch (method) {

          case 0: //Rigid Docking, Single point calcs, Minimization without
                  // orientation -- perform all atom internal.
              // for rigid, segments are not assinged, all segment_id's are -1
              nonbondpair = ((mol.get_bond(a1, a2) == -1) // a1 & a2 are non-bonded
                   && (!mol.atoms_are_one_three(a1, a2)) // not 1-3
                   && (!mol.atoms_are_one_four(a1, a2)) // not 1-4
                   );
              break;

          case 1: //Flex Anchor and Grow -- perform inter-segment internal
              nonbondpair = ( (mol.atom_segment_ids[a1] != mol.atom_segment_ids[a2])
                              //not within same segment
                   && (mol.get_bond(a1, a2) == -1) // a1 & a2 are non-bonded
                   && (!mol.atoms_are_one_three(a1, a2)) // not 1-3
                   && (!mol.atoms_are_one_four(a1, a2)) // not 1-4
                   );
              break;

          case 2: //De novo -- perform inter-segment internal // same as rigid.
              nonbondpair = ((mol.get_bond(a1, a2) == -1) // a1 & a2 are non-bonded
                   && (!mol.atoms_are_one_three(a1, a2))  // not 1-3
                   && (!mol.atoms_are_one_four(a1, a2))   // not 1-4
                   );
              break;

          case 3: //Genetic Algorithm -- perform inter-segment internal // same as rigid.
              nonbondpair = ((mol.get_bond(a1, a2) == -1) // a1 & a2 are non-bonded
                   && (!mol.atoms_are_one_three(a1, a2))  // not 1-3
                   && (!mol.atoms_are_one_four(a1, a2))   // not 1-4
                   );
              break;

          // case 4: //HDB
              // break;
          }

          if ( nonbondpair) // different criteria for different methods
           {
                temp.first = a1;
                temp.second = a2;
                nb_int.push_back(temp);
           }
       }   // for loop a1
   }  // for loop a2

}


// +++++++++++++++++++++++++++++++++++++++++
void
Base_Score::init_vdw_energy(AMBER_TYPER & typer, float att_exp, float rep_exp)
{

    //att_exp and rep_exp are read from the energy grid file
    delete[]vdwA;
    vdwA = NULL;

    delete[]vdwB;
    vdwB = NULL;

    vdwA = new float[typer.atom_typer.types.size()];
    vdwB = new float[typer.atom_typer.types.size()];

    for (int i = 0; i < typer.atom_typer.types.size(); i++) {
        vdwA[i] = sqrt(typer.atom_typer.types[i].well_depth *
                 (att_exp / (rep_exp - att_exp)) *
                 pow((2 * rep_radius_scale * typer.atom_typer.types[i].radius), rep_exp));

        vdwB[i] = sqrt(typer.atom_typer.types[i].well_depth *
                 (rep_exp / (rep_exp - att_exp)) *
                 pow((2 * typer.atom_typer.types[i].radius), att_exp));
    }
}

/*
// +++++++++++++++++++++++++++++++++++++++++
// sudipto & trent - 14-11-08
// Overloaded version of compute_ligand_internal_energy returns the components
// should be exactly the same as function above in algorithm
float
Base_Score::compute_ligand_internal_energy(DOCKMol & mol, float& int_vdw_att, float& int_vdw_rep, float& int_es)
{
    float           distance;
    unsigned int    a1, a2;

    float           ligand_internal_energy = 0.0;
    //float           ie_diel = 2.0;
    //int             ie_rep_exp = 12;
    //int             ie_att_exp = 6;

    int_vdw_att = 0.0;
    int_vdw_rep = 0.0;
    int_es = 0.0;

    if (use_internal_energy) {
        for (a1 = 0; a1 < mol.num_atoms - 1; a1++) {
            for (a2 = a1 + 1; a2 < mol.num_atoms; a2++) {

               if ( (mol.atom_active_flags[a1]) // atom a1 is active
                        && (mol.atom_active_flags[a2]) // atom a2 is active
                        && (mol.get_bond(a1, a2) == -1) // a1 & a2 are non-bonded
                        && (!mol.atoms_are_one_three(a1, a2)) // not 1-3
                        && (!mol.atoms_are_one_four(a1, a2)) // not 1-4
                        && ((mol.atom_segment_ids[a1] != mol.atom_segment_ids[a2])  //not within same segment
                            ||((mol.atom_segment_ids[a1]==-1)&&(mol.atom_segment_ids[a2]==-1)))) {

                        // DTM - 11-12-08 - check that atoms are from different segments, or that no segments have been assigned
                        // && (mol.amber_at_heavy_flag[a1])
                        // && (mol.amber_at_heavy_flag[a2])

                        distance = pow((mol.x[a1] - mol.x[a2]), 2) + pow((mol.y[a1] - mol.y[a2]),
                                         2) + pow((mol.z[a1] - mol.z[a2]), 2);

                        distance = sqrt(distance);
                        int_es += (mol.charges[a1] * mol.charges[a2] * ie_diel) / distance;

                        int_vdw_att -= (vdwB[mol.amber_at_id[a1]] *
                             vdwB[mol.amber_at_id[a2]]) / pow(distance, (ie_att_exp));

                        int_vdw_rep += (vdwA[mol.amber_at_id[a1]] *
                             vdwA[mol.amber_at_id[a2]]) / pow(distance, (ie_rep_exp));
                }
            }
        }
    }
    //ligand_internal_energy = int_vdw_att + int_vdw_rep + int_es;

    // Sudipto - 20-11-08
    // Only return the repulsive component of the vdw energy
    // Since dock does not have a Dihedral energy term, the attractive component of the vdw energy
    // can cause the ligand to fold up on itself to make 'favorable' interactions while still
    // remaining at optimal vdw repulsive distance.
    ligand_internal_energy = int_vdw_rep;
    return ligand_internal_energy;
}

*/