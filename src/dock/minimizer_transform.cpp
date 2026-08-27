#include <iostream>
#include <vector>
#include <math.h>
#include "dockmol.h"
#include "minimizer.h"
using namespace std;


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
