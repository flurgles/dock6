#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <sstream>
#include <time.h>

#include "amber_typer.h"
#include "conf_gen_ag.h"
#include "fingerprint.h"
#include "hungarian.h"
#include "weisfeiler_leman.h"
#include "master_score.h"
#include "score_dock_gpu.h"
#include "simplex.h"
#include "conformer_pool.h"
#include "trace.h"
class Bump_Filter;
class Master_Score;

using namespace std;



// +++++++++++++++++++++++++++++++++++++++++
AG_Conformer_Search::AG_Conformer_Search()
    :   ie_vdwA( NULL ),
        ie_vdwB( NULL ),  
        verbose( false ),
        write_fragment_libraries( false )
{
}


// +++++++++++++++++++++++++++++++++++++++++
AG_Conformer_Search::~AG_Conformer_Search()
{
    delete[]ie_vdwA;
    delete[]ie_vdwB; 
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::input_parameters(Parameter_Reader & parm)
{

    verbose = 0 != parm.verbosity_level();   // -v is for extra scoring info
    string         tmp;
    write_fragment_libraries = (parm.query_param("write_fragment_libraries", "no", "yes no") == "yes");
    if (write_fragment_libraries) {
        fragment_library_prefix = (parm.query_param("fragment_library_prefix", "fraglib"));
        fragment_library_freq_cutoff = atoi(parm.query_param("fragment_library_freq_cutoff", "1").c_str());
        fragment_library_sort_method = parm.query_param("fragment_library_sort_method", "freq", "freq | fingerprint");
        fragment_library_trans_origin = (parm.query_param("fragment_library_trans_origin", "no", "yes no") == "yes");
        user_specified_anchor = false;
        limit_max_anchors = false;
    } else {
    

        cout << "\nAnchor & Grow Parameters" << endl;
        cout <<
        "------------------------------------------------------------------------------------------" 
        << endl;

        user_specified_anchor = false;
        limit_max_anchors = false;
        anchor_size = 40;
        max_anchor_num = 10;

        user_specified_anchor = (parm.query_param("user_specified_anchor", "no", "yes no") == "yes");
        if (user_specified_anchor){
           //cout << "Specify atom name,number seperated by a ','" << endl;
           atom_in_anchor = parm.query_param("atom_in_anchor", "C1,1");
        } else {
            limit_max_anchors = (parm.query_param("limit_max_anchors", "no", "yes no") == "yes");
            if (limit_max_anchors){
            //    anchor_size = 3; // min_anchor_size is set to 3
                max_anchor_num = atoi(parm.query_param("max_anchor_num", "1").c_str());
            }

             anchor_size = atoi(parm.query_param("min_anchor_size", "5").c_str());
             if (anchor_size <= 0) {
                 cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                      << endl;
                 exit(0);
             }
        } 


        cluster = (parm.query_param("pruning_use_clustering", "yes", "yes no") == "yes");
        if(cluster){
            num_anchor_poses =
                atoi(parm.query_param("pruning_max_orients", "1000").c_str());
            if (num_anchor_poses <= 0) {
                cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(0);
            }
            pruning_clustering_cutoff = atoi(parm.query_param(
                    "pruning_clustering_cutoff", "100").c_str());
            if (pruning_clustering_cutoff <= 0) {
                cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(0);
            }

            // RMSD type for pruning clustering during anchor+growth:
            //   "std"        — standard layer-weighted heavy-atom RMSD (default)
            //   "hungarian"  — symmetry-corrected via Hungarian algorithm, layer-weighted
            //   "min"        — one-way minimum RMSD, layer-weighted
            //   "weisfeiler" — WL graph-coloring symmetry RMSD, layer-weighted
            // For final pose clustering, see final_pose_cluster_rmsd_type in library_file.cpp.
            pruning_cluster_rmsd_type = parm.query_param("pruning_cluster_rmsd_type", "std", "std hungarian min weisfeiler");

            anchor_score_cutoff = 1000.0;
            num_growth_poses = INT_MAX;
            anchor_score_cutoff = atof(parm.query_param("pruning_orient_score_cutoff", "1000.0").c_str());
            // DTM-11-10-08
            //growth_cutoff = false;
            //growth_cutoff = true;
            growth_score_cutoff_begin = atof(parm.query_param("pruning_conformer_score_cutoff", "100.0").c_str());
            growth_score_cutoff = growth_score_cutoff_begin;
            growth_score_scaling_factor = atof(parm.query_param("pruning_conformer_score_scaling_factor", "1.0").c_str());

        } else { 
            num_anchor_poses =
                atoi(parm.query_param("pruning_max_orients", "1000").c_str());
            if (num_anchor_poses <= 0) {
                cout << "ERROR:  Parameter must be an integer greater than zero.  Program will terminate."
                    << endl;
                exit(0);
            }
            anchor_score_cutoff = atof(parm.query_param("pruning_orient_score_cutoff", "100.0").c_str());
            num_growth_poses = atoi(parm.query_param("pruning_max_conformers", "75").c_str());
            //growth_cutoff = true;
            growth_score_cutoff_begin = atof(parm.query_param("pruning_conformer_score_cutoff", "100.0").c_str());
            growth_score_cutoff = growth_score_cutoff_begin;
            growth_score_scaling_factor = atof(parm.query_param("pruning_conformer_score_scaling_factor", "1.0").c_str());
        }

        use_clash_penalty = (parm.query_param("use_clash_overlap", "no", "yes no") == "yes");

        if (use_clash_penalty) {
            clash_penalty = atof(parm.query_param("clash_overlap", "0.5").c_str());
            if (clash_penalty <= 0.0) {
                cout << "clash_overlap should be a positive number" << endl;
                exit(0);
            }
        }

        print_growth_tree = (parm.query_param("write_growth_tree", "no", "yes no") == "yes");
        if (print_growth_tree) {
            cout << "Warning: Writing the growth tree increases memory usage and can generate lots of large files." << endl
                 << "Concatenating and compressing growth tree branches is recommended" << endl;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::initialize()
{
    // Separate input section from initializing section.
    cout << endl;
    // I.e., this is the first output for the initializing section.
    cout << "Initializing Conformer Generator Routines...\n";
}


// +++++++++++++++++++++++++++++++++++++++++
// Trent E Balius 2009-02-11
// this function gets the internal energy parms from Master_Conformer_Search.
void
AG_Conformer_Search::initialize_internal_energy_parms(bool uie, int rep_exp, int att_exp, float diel, float soft_delta, float iec)
{
     use_internal_energy = uie;
     ie_rep_exp = rep_exp;
     ie_att_exp = att_exp;
     ie_diel    = diel;
     ie_soft_delta = soft_delta;
     internal_energy_cutoff = iec;
}

// +++++++++++++++++++++++++++++++++++++++++
// Dwight 2016-06-16
// this function simply initializes the internal energy to False when a scoring 
//function is specified but internal energy is set to NO and it gets the internal
//energy value from Master_Conformer_Search
void
AG_Conformer_Search::initialize_internal_energy_null(bool uie)
{
 use_internal_energy = uie;

}

// +++++++++++++++++++++++++++++++++++++++++
// Weisfeiler-Leman (WL) color refinement for molecular graph symmetry detection.
//
// WL iteratively refines atom colors by hashing each atom's current color with
// the sorted colors of its bonded neighbors. After ~3-6 rounds, colors converge
// to stable "WL orbits" — atoms with the same WL color are indistinguishable in
// the graph topology and are symmetric (can be permuted without breaking bonds).
//
// This is strictly stronger than counting atom types: two C.3 atoms bonded to
// different substituents get different WL colors even though they share the same
// DOCK type, correctly ruling out false symmetry. Conversely, D6h benzene ring
// carbons all converge to one WL color, correctly identifying real symmetry.
//
// Returns true if any WL orbit contains 2+ heavy, active atoms — meaning the
// Hungarian algorithm could find a non-trivial optimal matching for RMSD.
// When false, Hungarian/min pruning RMSD falls back to standard RMSD since
// there's no exploitable permutation to find.
//
static bool
mol_has_symmetry(DOCKMol & mol)
{
    // Use shared WL color refinement (active_only=true so only
    // heavy, active atoms are considered — matches pruning logic).
    WL_RMSD wl;
    vector<int> colors;
    wl.wl_color_refine(mol, colors, true);

    // Count atoms per color group. Any group with >=2 atoms
    // indicates symmetry (atoms that could be permuted).
    int max_color = 0;
    for (int c : colors) {
        if (c > max_color) max_color = c;
    }
    vector<int> color_counts(max_color + 1, 0);
    for (int c : colors) {
        if (c >= 0) color_counts[c]++;
    }
    for (int cnt : color_counts) {
        if (cnt >= 2) return true;
    }
    return false;
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::prepare_molecule(DOCKMol & mol)
{
        // DTM - 11-12-08
        int        i;

        copy_molecule(orig, mol);

    bond_list.clear();
    atom_seg_ids.clear();
    bond_seg_ids.clear();
    anchors.clear();
    layers.clear();
    pruned_confs.clear();
    orig_segments.clear();
    layer_segments.clear();

    current_anchor = 0;

    identify_rigid_segments(mol);
    id_anchor_segments();

    // DTM - 11-12-08 - copy segment assignments to new atom_segment_ids array in the DOCKMol object
    for(i=0;i<atom_seg_ids.size();i++) {
            orig.atom_segment_ids[i] = atom_seg_ids[i];
            mol.atom_segment_ids[i] = atom_seg_ids[i];
    }

    // CDS - 04-14 - copy segment assignments to new bond_segment_ids array in the DOCKMol object
    for(i=0;i<bond_seg_ids.size();i++) {
            orig.bond_segment_ids[i] = bond_seg_ids[i];
            mol.bond_segment_ids[i] = bond_seg_ids[i];
    }

    // Write fragments for de novo libraries
    if (write_fragment_libraries){
       bickel_count_fragments(mol);
    }

    anchor_positions.clear();
    anchor_positions.reserve(1000); // can store more than 1000 anchors dynamically

    // Detect graph symmetry via WL color refinement.
    // When no symmetry exists, Hungarian/min fall back to standard RMSD.
    ligand_mol_symmetric = mol_has_symmetry(orig);

    if (verbose) {
        cout << "Ligand graph symmetry detection (WL color refinement): "
             << (ligand_mol_symmetric ? "SYMMETRIC" : "NONE (no exploitable symmetry)")
             << endl;
    }
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::identify_rigid_segments(DOCKMol & mol)
{
    int             i,
                    j,
                    max_central,
                    central;
    int             a1,
                    a2,
                    a3,
                    a4;
    int             t1,
                    t2,
                    nbr;
    vector < int   >nbrs;

    BREADTH_SEARCH  bfs;

    // Identify rigid segments & which segment each atom belongs to
    for (i = 0; i < orig.num_atoms; i++)
        atom_seg_ids.push_back(-1);

    for (i = 0; i < orig.num_bonds; i++)
        bond_seg_ids.push_back(-1);

    extend_segments(0, 0, mol);

    for (i = 0; i < orig.num_bonds; i++) {

        t1 = orig.bonds_origin_atom[i];
        t2 = orig.bonds_target_atom[i];

        if (atom_seg_ids[t1] == atom_seg_ids[t2]) {
            bond_seg_ids[i] = atom_seg_ids[t1];
            orig_segments[bond_seg_ids[i]].bonds.push_back(i);
        }

    }

    // ID the inter-segment rot-bonds
    for (i = 0; i < bond_list.size(); i++) {

        a2 = orig.bonds_origin_atom[bond_list[i].bond_num];
        a3 = orig.bonds_target_atom[bond_list[i].bond_num];

        max_central = -1;
        nbrs = mol.get_atom_neighbors(a2);

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
        nbrs = mol.get_atom_neighbors(a3);

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

        bond_list[i].atom1 = a1;
        bond_list[i].atom2 = a2;
        bond_list[i].atom3 = a3;
        bond_list[i].atom4 = a4;
        bond_list[i].seg1 = atom_seg_ids[bond_list[i].atom2];   // -1
        bond_list[i].seg2 = atom_seg_ids[bond_list[i].atom3];   // -1
        bond_list[i].initial_angle =
            orig.get_torsion(bond_list[i].atom1, bond_list[i].atom2,
                             bond_list[i].atom3, bond_list[i].atom4);
    }

}


// +++++++++++++++++++++++++++++++++++++++++
// is this function growing out a single segment ?
void
AG_Conformer_Search::extend_segments(int atom_num, int segment, DOCKMol & mol)
{
    int             nbr;
    int             bond;
    int             i;
    vector < int   >nbrs;

    SEGMENT         tmp_segment;
    ROT_BOND        tmp_bond;

    if (atom_num == 0)
        orig_segments.push_back(tmp_segment);

    atom_seg_ids[atom_num] = segment;
    orig_segments[segment].atoms.push_back(atom_num);

    if (orig.amber_at_heavy_flag[atom_num])
        orig_segments[segment].num_hvy_atoms++;

    nbrs = orig.get_atom_neighbors(atom_num);

    for (i = 0; i < nbrs.size(); i++) {

        nbr = nbrs[i];
        bond = orig.get_bond(atom_num, nbr);

        if (atom_seg_ids[nbr] == -1) {

            if (mol.bond_is_rotor(bond)) {

                bond_list.push_back(tmp_bond);
                bond_list[bond_list.size() - 1].bond_num = bond;

                orig_segments.push_back(tmp_segment);

                orig_segments[segment].neighbors.push_back(orig_segments.
                                                           size() - 1);
                orig_segments[segment].neighbor_bonds.push_back(bond_list.
                                                                size() - 1);
                orig_segments[segment].neighbor_atoms.push_back(nbr);

                orig_segments[orig_segments.size() -
                              1].neighbors.push_back(segment);
                orig_segments[orig_segments.size() -
                              1].neighbor_bonds.push_back(bond_list.size() - 1);
                orig_segments[orig_segments.size() -
                              1].neighbor_atoms.push_back(atom_num);

                extend_segments(nbr, orig_segments.size() - 1, mol);

            } else {

                extend_segments(nbr, segment, mol);

            }
        }
    }
}


// +++++++++++++++++++++++++++++++++++++++++
// TEB ADD 2010-01-23
// Idenify if atom in string is in the
// segment
// +++++++++++++++++++++++++++++++++++++++++
bool
AG_Conformer_Search::atom_in_anchor_segments(SEGMENT seg)
{
 bool flag = false;
 int i;
 stringstream s;
 string atomname, current_atomname,temp;
 int atomid, current_atomid ;
 s << atom_in_anchor;
 getline(s,atomname,',');
 s >> atomid;
// cout <<"lets see" <<atomname << " " << atomid << endl;

 //cout << atom_in_anchor << " :: " << atomname << "," << atomid <<endl;

 for (i = 0; i < seg.atoms.size(); i++){
     // the atom num is not stored in dockmol.
     current_atomid      = seg.atoms[i];
//     cout << current_atomid << "," ;
     current_atomname = orig.atom_names[current_atomid];
//     cout << current_atomname << endl;

     if (atomname == current_atomname && (current_atomid+1) == atomid){
//         cout << "condition met!" << endl;
         flag = true;
         break;
     }
 }

 if (flag){
   for (i = 0; i < seg.atoms.size(); i++){
     int current_atomid      = seg.atoms[i]; // the atom num is not stored.
     string current_atomname = orig.atom_names[current_atomid];
//     cout << current_atomname << "," << current_atomid << endl;
   }
 }

return flag;
}

// TEB ADDED 2019/06/17
// This function will print out all atoms in segments. 
// It is used to know what atoms are in the anchors.  
// This can be used for runing dock once to get list of anchors 
// and then again for each anchor using the specified anchor parameter. 
void AG_Conformer_Search::print_atom_in_anchor_segments(SEGMENT seg)
{
 int i;
 string  current_atomname;
 int current_atomid ;

  for (i = 0; i < seg.atoms.size(); i++){
  // the atom num is not stored in dockmol.
      current_atomid      = seg.atoms[i];
  //     cout << current_atomid << "," ;
      current_atomname = orig.atom_names[current_atomid];
  //     cout << current_atomname << endl;
      cout << current_atomname << ","<< current_atomid+1 << " ; "  ;

  }
  cout << endl;
}



// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::id_anchor_segments()
{
    Trace trace("AG_Conformer_Search::id_anchor_segments");
    int             i;
    INTPair         tmp;

    for (i = 0; i < orig_segments.size(); i++) {
        tmp.first = orig_segments[i].num_hvy_atoms + orig_segments[i].neighbors.size(); 
        // problem with anchor defs this way
        //  sudipto: who put this comment here and why?
        tmp.second = i;
        anchors.push_back(tmp);
    }

    sort(anchors.begin(), anchors.end());
    reverse(anchors.begin(), anchors.end());

    // TEB ADD 2010-01-23
    // pick only the user spesified anchor for placement in the array.
    if (user_specified_anchor){
//        cout << "user_specified_anchor is true" << endl;
        vector <INTPair> temp_anchors;
        temp_anchors = anchors;
        anchors.clear();
        for (i = 0; i < temp_anchors.size(); i++) {
            if (atom_in_anchor_segments(orig_segments[temp_anchors[i].second])){
//               cout << " user anchor defined." <<endl;
               //anchors.push_back(temp_anchors[i]);
               tmp.first = temp_anchors[i].first;
               tmp.second = temp_anchors[i].second;
               anchors.push_back(tmp);
            }
        }
        if (anchors.size() == 0){
            cout << "atom_in_anchor (" << atom_in_anchor <<") not found no segment" <<endl;
            exit(0);
        }
        if (anchors.size() > 1){
            cout << "atom_in_anchor (" << atom_in_anchor <<") is shared amoug multiple anchors" <<endl;
            cout << "pick new atom and try again" <<endl;
        }
    }

    // TEB ADD 2010-01-25
    // pick only the top N (spesified by the user) anchors
//  if (limit_max_anchors)
    if (limit_max_anchors){
        vector <INTPair> temp_anchors;
        temp_anchors = anchors;
        anchors.clear();
        int num_anchors_val;

        cout << max_anchor_num  << " " << temp_anchors.size() << endl;

        if (max_anchor_num > temp_anchors.size() ){
           cout << "max_anchor_nunum of segments." << endl;
           cout << "Only "<< temp_anchors.size() << " anchors are used." << endl;
           num_anchors_val = temp_anchors.size();
        } else {
           cout << max_anchor_num << " anchors are used." << endl;
           num_anchors_val = max_anchor_num;
        }
        for (i = 0; i < num_anchors_val ; i++) {
           cout << i << endl;
           tmp.first = temp_anchors[i].first;
           tmp.second = temp_anchors[i].second;
           anchors.push_back(tmp);
        }

    }
    // print out the list of anchors with one atom specified: 
    if (verbose) {
       cout << "-----------------------------------" << '\n';
       cout << "ANCHOR ATOMS." << "\n" << endl;
       for (i = 0; i < anchors.size(); i++){
            if ((i != 0) && (anchors[i].first < anchor_size)){ 
               break;
            }
            cout << "ANCHOR #"<< (i+1) << ": ";
            print_atom_in_anchor_segments(orig_segments[anchors[i].second]);
       } 
    } 
}


// +++++++++++++++++++++++++++++++++++++++++
bool
AG_Conformer_Search::next_anchor(DOCKMol & mol)
{
    DOCKMol         tmp_mol,
                    blank_mol;
    CONFORMER       tmp_conf;
    vector < CONFORMER > conf_vec;
    int             i;

    // trent & sudipto Jan 15, 09
    // this makes ure we no longer have the orients of the previous anchor
    anchor_positions.clear();

    // If there are no structs in anchor_confs, generate a new anchor and
    // expand it
    if (anchor_confs.size() == 0) {

        // if there are no more anchors, or you have reached the end of the
        // list 
        // or if there are no more anchors greater than the minium anchor size 
        // (consider moving this anchor size check into id_anchor_segments
        if ((anchors.size() == 0) || (current_anchor == anchors.size())
            || ((anchors[current_anchor].first < anchor_size)
                && (current_anchor > 0))){
            if (verbose) cout << "No more anchor fragments to be docked." << endl;
            return false;
        }

        copy_molecule(tmp_mol, orig);

        setup_growth_anchor(current_anchor);

        current_anchor++;

        // copy the anchor to the anchor_conf list
        anchor_confs.clear();
        anchor_confs.resize(anchor_confs.size() + 1);
        copy_molecule(anchor_confs[anchor_confs.size() - 1], tmp_mol);

    }

/**
        int i,j,k;
        for(i=0;i<layers.size();i++) {
                for(j=0;j<layers[i].segments.size();j++) {
                        for(k=0;k<layer_segments[layers[i].segments[j]].atoms.size();k++) {
                                cout << i << "\t" << j << "\t" << layer_segments[layers[i].segments[j]].atoms[k]+1 << endl;
                        }
                }
        }
**/

    // copy last mol from anchor_conf to mol
    copy_molecule(mol, anchor_confs[anchor_confs.size() - 1]);
    anchor_confs.pop_back();

    if (anchor_confs.size() > 0)
        last_conformer = false;
    else
        last_conformer = true;

    return true;


}

// +++++++++++++++++++++++++++++++++++++++++
// Rebuild the layer/layer_segment state for one anchor, exactly like the
// build step inside next_anchor().  Used by the GPU virtual-screen batch
// driver to replay growth for a previously collected (and already
// minimized) anchor without re-running the whole next_anchor pipeline.
void
AG_Conformer_Search::setup_growth_anchor(int anchor_idx)
{
    // clear bond directionality during minization
    bond_tors_vectors.clear();
    bond_tors_vectors.resize(orig.num_bonds, -1);

    // reset the assigned atoms list
    assigned_atoms.clear();
    assigned_atoms.resize(orig.num_atoms, false);

    // clear and resize the layer_segment list to be the same size as the
    // orig_segments list
    layer_segments.clear();
    layer_segments.resize(orig_segments.size());

    layers.clear();
    extend_layers(anchors[anchor_idx].second,
                  anchors[anchor_idx].second, 0);

    current_anchor = anchor_idx;
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::extend_layers(int previous_segment, int current_segment,
                                   int current_layer)
{
    int             i;
    INTPair         tmp_pair;
    LAYER           tmp_layer;
    LAYER_SEGMENT   tmp_segment;

    if (current_layer >= layers.size())
        layers.push_back(tmp_layer);

    layers[current_layer].segments.push_back(current_segment);
    layers[current_layer].num_segments = layers[current_layer].segments.size();

    // loop over the atoms in each orig_segment and add them to the
    // layer_segment
    // if the atom has not already been assigned to a previous segment
    for (i = 0; i < orig_segments[current_segment].atoms.size(); i++) {
        if (!assigned_atoms[orig_segments[current_segment].atoms[i]]) {
            layer_segments[current_segment].atoms.
                push_back(orig_segments[current_segment].atoms[i]);
            assigned_atoms[orig_segments[current_segment].atoms[i]] = true;
        }
    }

    // copy the bonds from orig_segment to layer_segment
    layer_segments[current_segment].bonds =
        orig_segments[current_segment].bonds;

    for (i = 0; i < orig_segments[current_segment].neighbors.size(); i++) {

        if (orig_segments[current_segment].neighbors[i] != previous_segment) {
            // if this bond leads to the next layer, then add the leading atom
            // and bond to the current segment

            layer_segments[current_segment].bonds.
                push_back(bond_list
                          [orig_segments[current_segment].neighbor_bonds[i]].
                          bond_num);

            if (!assigned_atoms
                [orig.
                 bonds_origin_atom[bond_list
                                   [orig_segments[current_segment].
                                    neighbor_bonds[i]].bond_num]]) {
                layer_segments[current_segment].atoms.push_back(orig.
                                                                bonds_origin_atom
                                                                [bond_list
                                                                 [orig_segments
                                                                  [current_segment].
                                                                  neighbor_bonds
                                                                  [i]].
                                                                 bond_num]);
                assigned_atoms[orig.
                               bonds_origin_atom[bond_list
                                                 [orig_segments
                                                  [current_segment].
                                                  neighbor_bonds[i]].
                                                 bond_num]] = true;

                // set bond direction vector
                bond_tors_vectors[bond_list
                                  [orig_segments[current_segment].
                                   neighbor_bonds[i]].bond_num] =
                    orig.
                    bonds_target_atom[bond_list
                                      [orig_segments[current_segment].
                                       neighbor_bonds[i]].bond_num];
            }

            if (!assigned_atoms
                [orig.
                 bonds_target_atom[bond_list
                                   [orig_segments[current_segment].
                                    neighbor_bonds[i]].bond_num]]) {
                layer_segments[current_segment].atoms.push_back(orig.
                                                                bonds_target_atom
                                                                [bond_list
                                                                 [orig_segments
                                                                  [current_segment].
                                                                  neighbor_bonds
                                                                  [i]].
                                                                 bond_num]);
                assigned_atoms[orig.
                               bonds_target_atom[bond_list
                                                 [orig_segments
                                                  [current_segment].
                                                  neighbor_bonds[i]].
                                                 bond_num]] = true;

                // set bond direction vector
                bond_tors_vectors[bond_list
                                  [orig_segments[current_segment].
                                   neighbor_bonds[i]].bond_num] =
                    orig.
                    bonds_origin_atom[bond_list
                                      [orig_segments[current_segment].
                                       neighbor_bonds[i]].bond_num];
            }

            extend_layers(current_segment,
                          orig_segments[current_segment].neighbors[i],
                          current_layer + 1);

        } else {
            // if this bond comes from the previous layer, then add it as the
            // rot_bond in the segment

            layer_segments[current_segment].rot_bond =
                orig_segments[current_segment].neighbor_bonds[i];
            layer_segments[current_segment].origin_segment = previous_segment;
        }
    }

}


// +++++++++++++++++++++++++++++++++++++++++
bool
AG_Conformer_Search::submit_anchor_orientation(DOCKMol & mol, bool more_orients)
{
    Trace trace("AG_Conformer_Search::submit_anchor_orientation");
    SCOREMol        tmp_mol;
    float           mol_score;
    int             insert_point;
    int             i;

    mol_score = mol.current_score;

    // Loop over existing list of anchors
    insert_point = -1;
    for (i = 0; i < anchor_positions.size(); i++) {

        // finds the first instance of a lower score than the mol, and selects
        // that as the insert point
        if (insert_point == -1) {
            if (anchor_positions[i].first > mol_score)
                insert_point = i;
        }

    }

    //insert mol into list, and pop last member off
    if (insert_point == -1) {
        // if insert_point = -1 (mol to be appended to end of list)
        // only append mol if list is less than number of confs/layer
        if (anchor_positions.size() < num_anchor_poses) {
                anchor_positions.push_back(tmp_mol);
                copy_molecule(anchor_positions[anchor_positions.size() - 1].
                              second, mol);
                anchor_positions[anchor_positions.size() - 1].first = mol_score;
        }

    } else {
        // if molecule is to be inserted within the list

        // if list is smaller than limit, append a blank mol on the end
        if (anchor_positions.size() < num_anchor_poses) {
            anchor_positions.push_back(tmp_mol);
        }
        // start at end of list, and shift all mols one spot to the end of
        // the list, until the insert point is reached
        for (i = anchor_positions.size() - 1; i > insert_point; i--) {
            copy_molecule(anchor_positions[i].second,
                              anchor_positions[i - 1].second);
            anchor_positions[i].first = anchor_positions[i - 1].first;
        }

        // copy mol into insert point
        copy_molecule(anchor_positions[insert_point].second, mol);
        anchor_positions[insert_point].first = mol_score;
   
    }

    // enter into loop if no more orients...
    if (! more_orients){
    // trent e balius 2011-05-30
        sort(anchor_positions.begin(), anchor_positions.end(), less_than_pair);
        if (verbose) cout << "# of anchor positions: " << anchor_positions.size() << endl;
        return true;
    } else {
        //if (verbose) cout << "there are no more orients and not last_conformer" << endl;
        return false;
    }

}


// +++++++++++++++++++++++++++++++++++++++++
float
AG_Conformer_Search::calc_layer_rmsd(CONFORMER & a, CONFORMER & b)
{
    int             i,
                    j,
                    k;
    int             atom;
    float           rmsd;
    int             heavy_total;

    rmsd = 0.0;
    heavy_total = 0;

    for (i = 0; i < layers.size(); i++) {
        for (j = 0; j < layers[i].segments.size(); j++) {
            for (k = 0; k < layer_segments[layers[i].segments[j]].atoms.size();
                 k++) {

                atom = layer_segments[layers[i].segments[j]].atoms[k];

                if (a.structure.atom_active_flags[atom]
                    && a.structure.amber_at_heavy_flag[atom]) {

                    rmsd +=
                        (pow((a.structure.x[atom] - b.structure.x[atom]), 2) +
                         pow((a.structure.y[atom] - b.structure.y[atom]),
                             2) + pow((a.structure.z[atom] -
                                       b.structure.z[atom]), 2)) * (i + 1);

                    heavy_total += i + 1;
                }
            }
        }
    }

    if (heavy_total > 0)
        rmsd = sqrt(rmsd / (float) heavy_total);
    else
        rmsd = 0.0;

    return rmsd;
}


// +++++++++++++++++++++++++++++++++++++++++
// Layer-weighted symmetry-corrected (Hungarian) RMSD for pruning clustering.
// Same layer-weighting scheme as calc_layer_rmsd(), but uses the Hungarian
// algorithm to find the optimal 1-to-1 matching of symmetry-equivalent atoms
// (same DOCK atom type) before computing distances. This prevents artificially
// high RMSD when chemically equivalent atoms (e.g., phenyl ring carbons) are
// permuted between conformers.
//
// Builds a per-atom weight array from growth layer information, then delegates
// to Hungarian_RMSD::calc_Hungarian_RMSD(DOCKMol&, DOCKMol&, const double*)
// which handles the weighted symmetry-corrected matching.
//
float
AG_Conformer_Search::calc_layer_rmsd_hungarian(CONFORMER & a, CONFORMER & b)
{
    int             i, j, k;
    int             atom;

    // Build per-atom weight array: weight[atom_idx] = layer_index + 1
    // Same layer traversal as calc_layer_rmsd() to ensure consistent weighting
    double* weights = new double[a.structure.num_atoms]();

    for (i = 0; i < layers.size(); i++) {
        for (j = 0; j < layers[i].segments.size(); j++) {
            for (k = 0; k < layer_segments[layers[i].segments[j]].atoms.size(); k++) {
                atom = layer_segments[layers[i].segments[j]].atoms[k];
                weights[atom] = (double)(i + 1);
            }
        }
    }

    // Delegate to weighted Hungarian RMSD on the DOCKMol structures
    // The Hungarian_RMSD class handles per-type cost matrix construction,
    // Hungarian algorithm, and weighted normalization internally.
    Hungarian_RMSD hr;
    double rmsd = hr.calc_Hungarian_RMSD(a.structure, b.structure, weights);

    delete[] weights;
    return (float)rmsd;
}


// +++++++++++++++++++++++++++++++++++++++++
// Layer-weighted one-way minimum RMSD for pruning clustering.
// For each heavy, active atom in conformer A, finds the closest same-type atom
// in conformer B (one-way matching). This gives a lower-bound RMSD that is
// not symmetric (calc_layer_rmsd_min(A,B) != calc_layer_rmsd_min(B,A)), but
// is computationally cheaper than the full Hungarian algorithm.
//
// Atoms in later growth layers carry higher weight, matching the weighting
// scheme used by calc_layer_rmsd() and calc_layer_rmsd_hungarian().
//
float
AG_Conformer_Search::calc_layer_rmsd_min(CONFORMER & a, CONFORMER & b)
{
    int             i, j, k;
    int             atom, m;
    float           rmsd = 0.0;
    float           total_weight = 0.0;

    // For each heavy, active atom in A (grouped by layer), find the closest
    // same-DOCK-type atom in B. Weight the contribution by (layer_index + 1)
    // so that atoms added in later growth layers are more influential.
    for (i = 0; i < layers.size(); i++) {
        for (j = 0; j < layers[i].segments.size(); j++) {
            for (k = 0; k < layer_segments[layers[i].segments[j]].atoms.size(); k++) {

                atom = layer_segments[layers[i].segments[j]].atoms[k];

                if (a.structure.atom_active_flags[atom]
                    && a.structure.amber_at_heavy_flag[atom]) {

                    float mindist2 = 1.0e20;

                    // Search all heavy, active atoms in B for the closest
                    // match of the same DOCK atom type
                    for (m = 0; m < b.structure.num_atoms; m++) {
                        if (b.structure.atom_active_flags[m]
                            && b.structure.amber_at_heavy_flag[m]
                            && a.structure.atom_types[atom] == b.structure.atom_types[m]) {

                            float d2 =
                                (a.structure.x[atom] - b.structure.x[m]) *
                                (a.structure.x[atom] - b.structure.x[m]) +
                                (a.structure.y[atom] - b.structure.y[m]) *
                                (a.structure.y[atom] - b.structure.y[m]) +
                                (a.structure.z[atom] - b.structure.z[m]) *
                                (a.structure.z[atom] - b.structure.z[m]);

                            if (d2 < mindist2)
                                mindist2 = d2;
                        }
                    }

                    if (mindist2 < 1.0e20) {
                        rmsd += mindist2 * (float)(i + 1);
                        total_weight += (float)(i + 1);
                    }
                }
            }
        }
    }

    if (total_weight > 0.0)
        rmsd = sqrt(rmsd / total_weight);
    else
        rmsd = 0.0;

    return rmsd;
}


// +++++++++++++++++++++++++++++++++++++++++/
float
AG_Conformer_Search::calc_active_rmsd(CONFORMER & ref, CONFORMER & conf)
{
// this function calculates rmsd between the active atoms of the ref structure and the same atoms of conf.
// the ref structure is usually the anchor
// only heavy atom rmsd is reported
// the rmsd of the active atoms in the reference is reported
    int    i;
    float  rmsd = 0.0;
    int    atom_num_total = 0;

    for (i = 0; i < ref.structure.num_atoms; i++) {
          if (ref.structure.atom_active_flags[i] && ref.structure.amber_at_heavy_flag[i]){
                    rmsd +=
                        (pow((ref.structure.x[i] - conf.structure.x[i]), 2) +
                         pow((ref.structure.y[i] - conf.structure.y[i]), 2) + 
                         pow((ref.structure.z[i] - conf.structure.z[i]), 2));

                    atom_num_total += 1;
          }
    }

    if (atom_num_total > 0)
        rmsd = sqrt(rmsd / (float) atom_num_total);
    else
        rmsd = 0.0;

    return rmsd;
}


// +++++++++++++++++++++++++++++++++++++++++
// Layer-weighted Weisfeiler-Leman symmetry-corrected RMSD for pruning
// clustering. Uses pre-computed WL colors (one refinement per growth layer)
// and a per-atom weight array derived from the growth layer structure, then
// delegates to WL_RMSD::calc_WL_RMSD_weighted which finds the optimal
// within-orbit permutation to minimize the weighted RMSD.
//
// colors must come from wl_color_refine(...,  true) on a representative
// conformer at this growth stage so that only heavy, active atoms get
// WL colors (inactive/hydrogen atoms have colors[i] < 0 and are skipped).
//
// weights[i] = (layer_index + 1) for atoms grown at layer i, 0 otherwise;
// built by the caller (once per layer) using the same traversal as
// calc_layer_rmsd_hungarian().
//
// Falls back to calc_layer_rmsd() if the WL RMSD returns a negative
// sentinel (-1000.0) — this should not happen during normal growth
// (same num_atoms, colors sized to match), but guards defensively.
//
namespace {
    // Pre-compute per-atom weights for WL-RMSD growth pruning.
    // Atoms at layer i get weight (i+1), matching the standard
    // calc_layer_rmsd() layer-weighting scheme. Uses local loop
    // variables to avoid corrupting the outer grow_periphery state.
    void compute_wl_weights(const vector<LAYER>& layers,
                            const vector<LAYER_SEGMENT>& layer_segments,
                            vector<double>& weights)
    {
        fill(weights.begin(), weights.end(), 0.0);
        for (int li = 0; li < (int)layers.size(); li++) {
            const INTVec& segs = layers[li].segments;
            for (int lj = 0; lj < (int)segs.size(); lj++) {
                const LAYER_SEGMENT& seg = layer_segments[segs[lj]];
                for (int lk = 0; lk < (int)seg.atoms.size(); lk++) {
                    weights[seg.atoms[lk]] = (double)(li + 1);
                }
            }
        }
    }
}


float
AG_Conformer_Search::calc_layer_rmsd_weisfeiler(CONFORMER & a,
                                                  CONFORMER & b,
                                                  const vector<int> & colors,
                                                  const double * weights)
{
    if (a.structure.num_atoms != b.structure.num_atoms)
        return calc_layer_rmsd(a, b);

    static WL_RMSD wl;
    double rmsd = wl.calc_WL_RMSD_weighted(a.structure, b.structure,
                                           colors, weights);
    if (rmsd < 0.0)
        return calc_layer_rmsd(a, b);

    return (float)rmsd;
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::grow_periphery(Master_Score & score,
                                    Minimizer & simplex, Bump_Filter & bump,
                                    bool anchors_preminimized)
{ Trace trace("AG_Conformer_Search::grow_periphery()");
    //cout << "AG_Conformer_Search::grow_periphery" << endl;
    int             i,
                    j,
                    k,
                    l;
    CONFORMER       tmp_conf;
    SCOREMol        tmp_scoremol;

    vector < SCOREMol > anchor_positions_b4min;
    vector < CONFORMER > seeds, exp_seeds;
    vector < CONFORMER > b4min_seeds;
    vector < CONFORMER > all_gen_seeds; // store tree with each conformer pointing to its parent
    vector < CONFORMER > all_gen_b4min_seeds; // all unminimized conformers
    float           rmsd;

    // initialize
    bool valid_orient = false;
    bool ie_prune = false; // prune conformers using internal_energy_cutoff
    int num_layers = layers.size();
    clock_t start = clock();

    // GPU virtual-screen: per-ligand LUT slot index for the VS kernel
    // (fixed-cycle over the LUT capacity; -1 = VS path inactive)
    int gpu_lig_idx = -1;
    if (dock_gpu_is_active()) {
        static int gpu_next_lig = 0;
        gpu_lig_idx = (gpu_next_lig++) % dock_gpu_vs_max_ligands();
    }

    //reserve space in vectors
    anchor_positions_b4min.reserve(anchor_positions.size());
    all_gen_seeds.reserve(2000);
    all_gen_b4min_seeds.reserve(2000);

    // check if a scoring function is being used.
    if (score.use_primary_score) {

        //initialize internal energy
        trace.note("AG_conformer::grow_periphery:reinitialize energy ");
        score.primary_score->use_internal_energy = use_internal_energy;
        if (use_internal_energy || score.c_int.use_primary_score) {
           if (use_internal_energy) {
             score.primary_score->ie_att_exp = ie_att_exp;
             score.primary_score->ie_rep_exp = ie_rep_exp;
             score.primary_score->ie_diel = ie_diel;
             score.primary_score->ie_soft_delta = ie_soft_delta;
             ie_prune=true;
           }
           if (!use_internal_energy){
              initialize_internal_energy_parms(use_internal_energy, score.primary_score->ie_rep_exp, score.primary_score->ie_att_exp, score.primary_score->ie_diel, score.primary_score->ie_soft_delta, growth_score_cutoff);
           }
           // need to use DOCKMol with radii and segments assigned
           // it does not matter which atoms are labeled active
           score.primary_score->initialize_internal_energy(anchor_positions[0].second);

/* Upload ligand parameters to GPU for combined grid+IE scoring */
           if (dock_gpu_is_active()) {
                DOCKMol & amol = anchor_positions[0].second;
                int na = amol.num_atoms;
                float *vdwA_arr = new float[na];
                float *vdwB_arr = new float[na];
                float *chg_arr  = new float[na];
                for (int ai = 0; ai < na; ai++) {
                    int type = amol.amber_at_id[ai];
                    vdwA_arr[ai] = score.primary_score->vdwA[type];
                    vdwB_arr[ai] = score.primary_score->vdwB[type];
                    chg_arr[ai]  = amol.charges[ai];
                }
                if (!dock_gpu_set_ligand(vdwA_arr, vdwB_arr, chg_arr, na)) {
                    fprintf(stderr, "GPU-DOCK: set_ligand failed\n");
                }

                /* IE params: nb_int pairs, per-atom ie_vdwA */
                int np = (int)score.primary_score->nb_int.size();
                int *nb_flat = NULL;
                float *ie_vdwA_arr = NULL;
                if (np > 0) {
                    nb_flat = new int[np * 2];
                    for (int pi = 0; pi < np; pi++) {
                        nb_flat[pi * 2]     = score.primary_score->nb_int[pi].first;
                        nb_flat[pi * 2 + 1] = score.primary_score->nb_int[pi].second;
                    }
                    ie_vdwA_arr = new float[na];
                    for (int ai = 0; ai < na; ai++) {
                        ie_vdwA_arr[ai] = score.primary_score->ie_vdwA[ai];
                    }
                    dock_gpu_set_ligand_ie(ie_vdwA_arr, NULL, nb_flat, np,
                                            score.primary_score->ie_soft_delta,
                                            score.primary_score->ie_vdw_cutoff_sq);
                }

                /* Virtual-screen LUT registration: same per-atom params plus
                   the ligand's active flags, so later VS dispatches can score
                   candidates of many ligands against the shared grid. */
                if (gpu_lig_idx >= 0) {
                    int *af = new int[na];
                    for (int ai = 0; ai < na; ai++)
                        af[ai] = amol.atom_active_flags[ai] ? 1 : 0;
                    dock_gpu_vs_register_ligand(gpu_lig_idx,
                                                 vdwA_arr, vdwB_arr, chg_arr, af,
                                                 ie_vdwA_arr, nb_flat, np, na,
                                                 score.primary_score->ie_soft_delta,
                                                 score.primary_score->ie_vdw_cutoff_sq);
                    delete[] af;
                }

                delete[] vdwA_arr;
                delete[] vdwB_arr;
                delete[] chg_arr;
                if (nb_flat) delete[] nb_flat;
                if (ie_vdwA_arr) delete[] ie_vdwA_arr;
            }
        }
    }// else do nothing to internal energy (because it is not used when scoring function is not
     // used

    // /////////////////////////// ANCHORS //////////////////////////////

    if (verbose) cout << "-----------------------------------" << endl 
         << "VERBOSE GROWTH STATS : ANCHOR #" <<  current_anchor << endl << endl;

    // add anchors to seed list

    // moved from main dock loop for flexible docking only
    // sudipto & trent Jan-16-08
    if (!anchors_preminimized && dock_gpu_is_active()
        && simplex.use_min_rigid_anchor
        && gpu_lig_idx >= 0
        && (int)anchor_positions.size() > dock_gpu_recommended_batch_size()) {
        // GPU anchor-pool: batch the rigid-body simplex minimization of ALL
        // anchor orientations into one ConformerPool.  Every anchor shares
        // the same DOF layout (6 rigid DOF, vertex of zeros) and the same
        // atom count / active flags, so a single pool can pack candidates
        // from all orientations into each GPU dispatch instead of issuing
        // one tiny dispatch per orientation.  Only engages when the
        // orientation count is large enough to amortize the pool machinery;
        // small anchor counts keep the sequential path (measured: pool
        // overhead exceeds its win for handfuls of orientations).
        const int saved_anchor_cycle = simplex.current_cycle;
        simplex.current_cycle = 0;
        ConformerPool anchor_pool(dock_gpu_recommended_batch_size(),
                                   &simplex, true,
                                   simplex.simplex_mode,
                                   simplex.simplex_crossover);
        for (i = 0; i < anchor_positions.size(); i++){
            anchor_positions_b4min.push_back(anchor_positions[i]);  // save unmin anchor in anchor_positions_b4min

            // backpressure: drain pool before adding if full
            while (anchor_pool.active_count() >= anchor_pool.capacity()) {
                anchor_pool.step();
                anchor_pool.poll();
            }

            // rigid DOF only — same layout as minimize_rigid_anchor()
            FLOATVec vertex;
            for (int iv = 0; iv < 6; iv++) vertex.push_back(0.0f);

            SimplexStage astage;
            astage.max_iterations = simplex.anchor_min_max_iterations;
            astage.max_cycles     = simplex.anchor_min_max_cycles;
            astage.score_converge = simplex.anchor_min_score_converge;
            astage.trans_step_size = simplex.anchor_min_trans_step_size;
            astage.rot_step_size  = simplex.anchor_min_rot_step_size;
            astage.tors_step_size = simplex.anchor_min_tors_step_size;

            anchor_pool.add(&anchor_positions[i].second, vertex,
                            astage, SimplexStage{0, 0, 0.0f, 0.0f, 0.0f, 0.0f},
                            simplex.anchor_min_cycle_converge,
                            simplex.restrained_min,
                            simplex.coefficient_restraint,
                            nullptr, nullptr,
                            gpu_lig_idx,
                            anchor_positions[i].second.num_atoms,
                            seed_key((unsigned)dock_mol_serial,
                                     (unsigned)current_anchor, (unsigned)i,
                                     0x51u, 0x52u));
        }

        // drain pool — all anchor mols updated in place by fill_slot_from_mol
        while (!anchor_pool.idle()) {
            anchor_pool.step();
            anchor_pool.poll();
        }

// restore cycle counter so the growth-phase scaling below matches
            // the sequential-anchor code path (minimize_rigid_anchor leaves
            // current_cycle at whatever the last cycle count was)
            simplex.current_cycle = saved_anchor_cycle;
        } else if (anchors_preminimized) {
            // Anchors were already minimized by the VS batch driver — keep
            // the unminimized copies for b4min_seeds below.
            // OPTIMIZATION parity: minimize_rigid_anchor() sizes bond_vectors
            // for the anchor mol before each ligand's growth pool runs; the
            // growth pool depends on that sizing, so reproduce it here.
            if (simplex.use_min_rigid_anchor) {
                simplex.bond_vectors.clear();
                simplex.bond_vectors.resize(anchor_positions[0].second.num_bonds, -1);
            }
            for (i = 0; i < anchor_positions.size(); i++){
                anchor_positions_b4min.push_back(anchor_positions[i]);
            }
        } else {
        for (i = 0; i < anchor_positions.size(); i++){
            anchor_positions_b4min.push_back(anchor_positions[i]);  // save unmin anchor in anchor_positions_b4min
            simplex.set_local_rng_seed(
                seed_key((unsigned)dock_mol_serial,
                         (unsigned)current_anchor, (unsigned)i,
                         0x51u, 0x52u));
            simplex.minimize_rigid_anchor(anchor_positions[i].second, score); //minimize anchor
        }
    }

    for (i = 0; i < anchor_positions.size(); i++) {
        // compute the score for the molecule
        // OPTIMIZATION: internal energy was already computed and cached in
        // mol.internal_energy by minimize_rigid_anchor().  The anchor has no
        // torsional DOFs so internal energy is invariant under rigid-body
        // transformations.  Call compute_score() directly instead of
        // compute_primary_score() to avoid redundant O(nb_int) loop work.
        if (score.use_primary_score) {
            valid_orient = score.primary_score->compute_score(anchor_positions[i].second);

            // code to check if there was an error in the scoring function: sudipto 
            // the scoring function stores an error description in dockmol.current_data 
            if (!valid_orient && verbose) cout << "Error scoring Anchor #" << current_anchor 
                 << " Orient #" << i << ": " << anchor_positions[i].second.current_data << endl;

        } else valid_orient = true;

        if (valid_orient) {
            exp_seeds.push_back(tmp_conf);
            exp_seeds[exp_seeds.size() - 1].layer_num = 1;
            exp_seeds[exp_seeds.size() - 1].score = anchor_positions[i].second.current_score;
            exp_seeds[exp_seeds.size() - 1].used = false;
            copy_molecule(exp_seeds[exp_seeds.size() - 1].structure, anchor_positions[i].second);

            b4min_seeds.push_back(tmp_conf);
            b4min_seeds[b4min_seeds.size() - 1].layer_num = 1;

            //b4_min_seeds[b4_min_seeds.size() - 1].score = anchor_positions_b4min[i].second.current_score;
            // we want the b4min score to match the after min score so sorting and pruning the anchors remain in step

            b4min_seeds[b4min_seeds.size() - 1].score = anchor_positions[i].second.current_score;
            b4min_seeds[b4min_seeds.size() - 1].used = false;
            copy_molecule(b4min_seeds[b4min_seeds.size() - 1].structure, anchor_positions_b4min[i].second);
        }
    }

    //assert(exp_seeds.size() == b4min_seeds.size());

    // Pruning section
    seeds.clear(); seeds.reserve(500);
    sort(exp_seeds.begin(), exp_seeds.end(), conformer_less_than);
    sort(b4min_seeds.begin(), b4min_seeds.end(), conformer_less_than);
  
    // make sure we are not missing any unminimized structures
    // remove once we are sure growth tree is bug-free: sudipto
    //assert(exp_seeds.size() == b4min_seeds.size());

    // filter confs by scores
    for (j = 0; j < exp_seeds.size(); j++) {
        if (exp_seeds[j].score > anchor_score_cutoff)  // +1000 for clustering, +25 otherwise
            exp_seeds[j].used = true;
    }

    // remove confs that fail the rank/rmsd test
    if(cluster){
        float RMSD_CUTOFF = pruning_clustering_cutoff;

        // Pre-compute WL colors + per-atom weights for weisfeiler pruning so
        // the O(n^2) pairwise loop pays one wl_color_refine (expensive) and one
        // layer-weight traversal, not one per pair. Colors depend only on the
        // graph (atom types + active bonds), which is shared across all
        // conformers at this growth stage.
        vector<int>  wl_colors;
        vector<double> wl_weights;
        if (pruning_cluster_rmsd_type == "weisfeiler" && ligand_mol_symmetric
            && !exp_seeds.empty()) {
            compute_wl_weights(layers, layer_segments, wl_weights);
            {
                static WL_RMSD wl;
                wl.wl_color_refine(exp_seeds[0].structure, wl_colors, true);
            }
        }

        for (j = 0; j < exp_seeds.size(); j++) {
               if (!exp_seeds[j].used) {
                    for (k = j + 1; k < exp_seeds.size(); k++) {
                       if (!exp_seeds[k].used) {

                            // Dispatch to RMSD function based on pruning_cluster_rmsd_type.
                            // If no graph symmetry detected, Hungarian/min/weisfeiler fall
                            // back to std to avoid paying symmetry cost for no benefit.
                            if ((pruning_cluster_rmsd_type == "hungarian" ||
                                 pruning_cluster_rmsd_type == "min" ||
                                 pruning_cluster_rmsd_type == "weisfeiler") &&
                                (!ligand_mol_symmetric))
                                rmsd = calc_layer_rmsd(exp_seeds[j], exp_seeds[k]);
                            else if (pruning_cluster_rmsd_type == "hungarian")
                                rmsd = calc_layer_rmsd_hungarian(exp_seeds[j], exp_seeds[k]);
                            else if (pruning_cluster_rmsd_type == "min")
                                rmsd = calc_layer_rmsd_min(exp_seeds[j], exp_seeds[k]);
                            else if (pruning_cluster_rmsd_type == "weisfeiler")
                                rmsd = calc_layer_rmsd_weisfeiler(exp_seeds[j], exp_seeds[k],
                                                                  wl_colors, wl_weights.data());
                            else
                                rmsd = calc_layer_rmsd(exp_seeds[j], exp_seeds[k]);
                            rmsd = MAX(rmsd, 0.001);

                            if ((float) k / rmsd > RMSD_CUTOFF) {
                                exp_seeds[k].used = true;
                            }
                       }        
                    }
              }
        }
    }

    // End pruning section
    int anchor_count = 0; // for assigning anchor number 
    conf_anchors.clear(); // remove old anchor orients from previous calls
    conf_anchors.reserve(anchor_positions.size()); //anchor_positions.size() is an upper limit for # of clustered anchors

    // copy pruned confs to seed list
    for (j = 0; j < exp_seeds.size(); j++) {

            // This section is required for the growth tree.
            exp_seeds[j].conformer_num = count_conf_num;
            exp_seeds[j].parent_num   = -1; // when parent is -1, we are at the top of the tree
            b4min_seeds[j].conformer_num = count_conf_num;
            b4min_seeds[j].parent_num   = -1; 
            
            // print anchors to file
            // print_conformer(exp_seeds[j]); 

            if (!exp_seeds[j].used) {
           
                exp_seeds[j].anchor_num   = anchor_count; //anchor position in array conf_anchors 
                b4min_seeds[j].anchor_num = anchor_count; 
                conf_anchors.push_back(b4min_seeds[j]); //unminimized anchors
           
                // Calculate header string for anchor. 
                // The conf_before_min parameter in this function is not used for anchors
                // This section is required for the growth tree.
                // When the growth tree is turned off, computing rmsd for all the conformers
                // is not needed.
                if (print_growth_tree) conf_header(exp_seeds[j], "Minimized Anchor", score);
                if (print_growth_tree) conf_header(b4min_seeds[j], "Unminimized Anchor", score);
                if (print_growth_tree) b4min_seeds[j].structure.simplex_text = "";
           
                seeds.push_back(exp_seeds[j]);
                if (print_growth_tree) all_gen_seeds.push_back(exp_seeds[j]); 
                if (print_growth_tree) all_gen_b4min_seeds.push_back(b4min_seeds[j]); 
                anchor_count++;
            } 
            count_conf_num++;   // this counter also includes pruned anchors
    }
    // /////////////////////////// END ANCHORS //////////////////////////

    // print anchor pruning staticstics
    // why are no anchors getting pruned??
    if (verbose) cout << seeds.size() << "/" <<  anchor_positions.size() 
               << " anchor orients retained (max " << num_anchor_poses << ")" 
               << " t=" << ((double) (clock() - start)) / CLOCKS_PER_SEC << "s" << endl;

    // /////////////////////////// GROWTH ///////////////////////////////

    // counters for printing pruning stats
    int confs_pruned_bad_score = 0;
    int confs_pruned_clash_overlap = 0;
    int confs_pruned_outside_grid = 0;
    int confs_pruned_bump_filter = 0;
    int confs_pruned_clustered = 0;

    if (verbose) {
      for (int layer_count = 1; layer_count < num_layers; layer_count++)
        cout << "Lyr " << layer_count << "-" << layers[layer_count].segments.size() << " Segs|";
      cout << "number of layers:"<< num_layers <<  endl; 
    } 

    // loop over layers
    for (i = 1; i < num_layers; i++) {  // 

        // loop over rot bonds in segment
        for (l = 0; l < layers[i].segments.size(); l++) {

            exp_seeds.clear(); exp_seeds.reserve(1000);
            b4min_seeds.clear(); b4min_seeds.reserve(1000);

            // print details of rotatable bond sampled
            if (verbose && seeds.size() > 0) {
              int bond = layer_segments[layers[seeds[0].layer_num].segments[l]].rot_bond;  
              cout << "Lyr:" << i << " Seg:" << l << " Bond:" << bond_list[bond].bond_num 
                   << " : Sampling " <<  seeds[0].structure.amber_bt_torsion_total[bond_list[bond].bond_num]
                   << " dihedrals " 
                   << seeds[0].structure.atom_names[bond_list[bond].atom1] << "("
                   << seeds[0].structure.atom_types[bond_list[bond].atom1] << ")  "
                   << seeds[0].structure.atom_names[bond_list[bond].atom2] << "("
                   << seeds[0].structure.atom_types[bond_list[bond].atom2] << ")  "
                   << seeds[0].structure.atom_names[bond_list[bond].atom3] << "("
                   << seeds[0].structure.atom_types[bond_list[bond].atom3] << ")  "
                   << seeds[0].structure.atom_names[bond_list[bond].atom4] << "("
                   << seeds[0].structure.atom_types[bond_list[bond].atom4] << ")"
                   //<< " t=" << ((double) (clock() - start)) / CLOCKS_PER_SEC << "s"
                   << endl;
            } 

            // loop over seed conformers
            for (j = 0; j < seeds.size(); j++) {
        // Required for supporting multiple grid docking
                // Duplicates of each exp_seed is made for each grid with the corresponding grid_num
                if (score.ir_ensemble) { 
                        segment_torsion_drive(seeds[j], l, exp_seeds, score.c_mg_nrg.numgrids);
                        //cout << "grow_periphery: " << score.c_mg_nrg.numgrids << " grids" << endl;
                }
                else 
                        segment_torsion_drive(seeds[j], l, exp_seeds, 1);
                // num_grids = 1 when multiple grid docking is turned off

                // score/minimize the new confs

                // ---- C1 Phase 3: Split minimize+score into two passes ----
                // First pass: save b4min, run clash/bump checks (cached for second pass),
                //              minimize (batch queue).
                // Second pass: score + prune after GPU flush.
                //
                // We cache the clash/bump results because the checks run on
                // pre-minimization coordinates, but by the second pass the mol
                // has been updated to the minimized pose.

                std::vector<int> cb_ok(exp_seeds.size(), 0);
                // 0 = clash fail, 1 = bump fail, 2 = both passed

                bool use_gpu = dock_gpu_is_active();
                ConformerPool pool(GPU_POOL_BATCH_MAX,
                                   &simplex, use_gpu,
                                   simplex.simplex_mode,
                                   simplex.simplex_crossover);
                bool last_seed_level = (j == seeds.size() - 1);

                for (k = 0; ((k < exp_seeds.size())&&last_seed_level); k++) {

                    // sudipto & trent - 11-14-08 - save current conf before minimizing
                    CONFORMER conf_before_min = exp_seeds[k];
                    b4min_seeds.push_back(conf_before_min);

                    if (segment_clash_check(exp_seeds[k].structure, i, l)) {

                        if (bump.check_growth_bumps(exp_seeds[k].structure)) {

                            cb_ok[k] = 2;

                            if (use_gpu) {
                                /* Build initial DOF vertex (same layout as
                                   minimize_flexible_growth) */
                                FLOATVec vertex;
                                for (int iv = 0; iv < 6; iv++) vertex.push_back(0.0f);
                                simplex.id_torsions(exp_seeds[k].structure, vertex);
                                simplex.torsion_scale_factors.resize(simplex.torsions.size(), 1);

                                /* mirror Minimizer::minimize_flexible_growth:
                                   with minimize_flexible_growth=no the
                                   growth poses are left unminimized */
                                if (simplex.use_min_flex_growth)
                                {

                                /* Match the CPU ramp path: score_converge ramps
                                   from initial_score_converge down to
                                   flex_min_score_converge across layers. */
                                float pool_score_converge;
                                if (simplex.use_min_flex_growth_ramp) {
                                    float diff_interval =
                                        simplex.initial_score_converge -
                                        simplex.flex_min_score_converge;
                                    float adj_unit =
                                        diff_interval / (float)(num_layers - 1);
                                    adj_unit *= (float)i;
                                    pool_score_converge =
                                        simplex.initial_score_converge - adj_unit;
                                } else {
                                    pool_score_converge =
                                        simplex.flex_min_cycle_converge;
                                }

                                /* Backpressure: drain pool before adding if full */
                                while (pool.active_count() >= pool.capacity()) {
                                    pool.step();
                                    pool.poll();
                                }

                                /* Two stages, mirroring
                                   minimize_flexible_growth: a torsion
                                   pre-min (rigid DOF locked) followed by
                                   the full minimization.  The ramp path
                                   uses the ramped converge for both. */
                                float stageA_converge =
                                    (simplex.use_min_flex_growth_ramp)
                                        ? pool_score_converge
                                        : simplex.flex_min_score_converge;
                                SimplexStage gA;
                                gA.max_iterations =
                                    simplex.flex_min_torsion_iterations;
                                gA.max_cycles     = simplex.flex_min_max_cycles;
                                gA.score_converge = stageA_converge;
                                gA.trans_step_size = 0.0f;
                                gA.rot_step_size  = 0.0f;
                                gA.tors_step_size =
                                    simplex.flex_min_tors_step_size;
                                SimplexStage gB;
                                gB.max_iterations =
                                    simplex.flex_min_max_iterations;
                                gB.max_cycles     = simplex.flex_min_max_cycles;
                                gB.score_converge = pool_score_converge;
                                gB.trans_step_size =
                                    simplex.flex_min_trans_step_size;
                                gB.rot_step_size =
                                    simplex.flex_min_rot_step_size;
                                gB.tors_step_size =
                                    simplex.flex_min_tors_step_size;

                                pool.add(&exp_seeds[k].structure,
                                         vertex,
                                         gA, gB,
                                         simplex.flex_min_cycle_converge,
                                         simplex.restrained_min,
                                         simplex.coefficient_restraint,
                                         nullptr,
                                         reinterpret_cast<void*>((intptr_t)k),
                                         -1, 0,
                                         seed_key((unsigned)dock_mol_serial,
                                                  (unsigned)current_anchor,
                                                  (unsigned)i, (unsigned)l,
                                                  (unsigned)k));
                                }
                            } else {
                                /* CPU fallback: per-conformer minimization */
                                simplex.set_local_rng_seed(
                                    seed_key((unsigned)dock_mol_serial,
                                             (unsigned)current_anchor,
                                             (unsigned)i, (unsigned)l,
                                             (unsigned)k));
                                if (!simplex.use_min_flex_growth_ramp) {
                                    simplex.minimize_flexible_growth(
                                        exp_seeds[k].structure, score,
                                        bond_tors_vectors);
                                } else {
                                    simplex.minimize_flexible_ramp_growth(
                                        exp_seeds[k].structure, score,
                                        bond_tors_vectors,
                                        i, num_layers);
                                }
                            }
                        } else {
                            cb_ok[k] = 1;
                        }
                    }
                } // First pass over new conformers (k)

                // Drain pool — all mols updated in place by fill_slot_from_mol
                if (use_gpu && last_seed_level) {
                    while (!pool.idle()) {
                        pool.step();
                        pool.poll();
                    }
                }

                // Second pass: score and prune
                // GPU-batched path: score every exp_seed of the last seed
                // level in two dispatches (grid-only + grid+IE) instead of
                // one CPU compute_primary_score per conformer.
                bool gpu2 = false;
                std::vector<float> g2_grid;
                std::vector<float> g2_comb;
                std::vector<char>  g2_valid;
                if (use_gpu && score.use_primary_score &&
                    j == (int)seeds.size() - 1 && !exp_seeds.empty()) {
                    const int n2 = (int)exp_seeds.size();
                    const int na2 = exp_seeds[0].structure.num_atoms;
                    if (na2 > 0) {
                        std::vector<float> xyz2((size_t)n2 * na2 * 3);
                        for (int kk = 0; kk < n2; kk++) {
                            DOCKMol & s2 = exp_seeds[kk].structure;
                            for (int a = 0; a < na2; a++) {
                                xyz2[(size_t)kk * na2 * 3 + a * 3 + 0] = s2.x[a];
                                xyz2[(size_t)kk * na2 * 3 + a * 3 + 1] = s2.y[a];
                                xyz2[(size_t)kk * na2 * 3 + a * 3 + 2] = s2.z[a];
                            }
                        }
                        std::vector<int> flags2((size_t)na2);
                        for (int a = 0; a < na2; a++)
                            flags2[a] = exp_seeds[0].structure
                                            .atom_active_flags[a] ? 1 : 0;
                        g2_grid.resize((size_t)n2);
                        g2_comb.resize((size_t)n2);
                        g2_valid.resize((size_t)n2);
                        bool ok = true;
                        const char *dbg2 = getenv("DOCK_GPU2_DEBUG");
                        if (dbg2) fprintf(stderr, "GPU2 gate i=%d l=%d j=%d/%d n2=%d na2=%d use_gpu=%d useprim=%d\n",
                                          i, l, j, (int)seeds.size()-1, n2, na2, (int)use_gpu, (int)score.use_primary_score);
                        for (int off = 0; off < n2; off += GPU_MAX_BATCH_POSES) {
                            int cnt = n2 - off;
                            if (cnt > GPU_MAX_BATCH_POSES)
                                cnt = GPU_MAX_BATCH_POSES;
                            if (!dock_gpu_batch_score(
                                    &xyz2[(size_t)off * na2 * 3], cnt, na2,
                                    flags2.data(), &g2_grid[off])) {
                                if (dbg2) fprintf(stderr, "GPU2 batch_score FAILED off=%d cnt=%d na2=%d\n", off, cnt, na2);
                                ok = false; break;
                            }
                            if (!dock_gpu_batch_score_with_ie_persistent(
                                    &xyz2[(size_t)off * na2 * 3], cnt, na2,
                                    flags2.data(), &g2_comb[off])) {
                                if (dbg2) fprintf(stderr, "GPU2 with_ie FAILED off=%d cnt=%d na2=%d\n", off, cnt, na2);
                                ok = false; break;
                            }
                        }
                        float gminx, gminy, gminz, gmaxx, gmaxy, gmaxz;
                        if (ok && dock_gpu_grid_bounds(&gminx, &gminy, &gminz,
                                                       &gmaxx, &gmaxy, &gmaxz)) {
                            /* kernel clamps out-of-bounds atoms; replicate
                               the CPU out-of-grid rejection */
                            for (int kk = 0; kk < n2; kk++) {
                                DOCKMol & s2 = exp_seeds[kk].structure;
                                bool inb = true;
                                for (int a = 0; a < na2 && inb; a++) {
                                    if (!s2.atom_active_flags[a]) continue;
                                    if (!(s2.x[a] > gminx && s2.x[a] < gmaxx &&
                                          s2.y[a] > gminy && s2.y[a] < gmaxy &&
                                          s2.z[a] > gminz && s2.z[a] < gmaxz))
                                        inb = false;
                                }
                                g2_valid[kk] = inb ? 1 : 0;
                            }
                        } else ok = false;
                        gpu2 = ok;
                    }
                }

                for (k = 0; ((k < exp_seeds.size())&&(j==seeds.size()-1)); k++) {

                    if (cb_ok[k] == 2) {

                            // compute the score for the molecule
                            // now internal energy is computed within compute_primary_score
                            if (score.use_primary_score){
                                if (gpu2) {
                                    valid_orient = (g2_valid[k] != 0);
                                    exp_seeds[k].structure.current_score =
                                        g2_grid[k];
                                    exp_seeds[k].structure.internal_energy =
                                        g2_comb[k] - g2_grid[k];
                                    if (getenv("DOCK_GPU2_DEBUG") && k == 0) {
                                        DOCKMol ref = exp_seeds[k].structure;
                                        bool cpu_ok =
                                            score.compute_primary_score(ref);
                                        fprintf(stderr,
                                            "GPU2 i=%d l=%d j=%d na2=%d "
                                            "gpu_grid=%f gpu_comb=%f "
                                            "cpu_grid=%f cpu_ie=%f "
                                            "cpu_ok=%d cutoff=%f\n",
                                            i, l, j,
                                            exp_seeds[k].structure.num_atoms,
                                            g2_grid[k], g2_comb[k],
                                            ref.current_score,
                                            ref.internal_energy, cpu_ok,
                                            growth_score_cutoff_begin /
                                                pow(growth_score_scaling_factor,
                                                    i));
                                    }
                                } else {
                                    valid_orient = score.compute_primary_score(exp_seeds[k].structure);
                                    valid_orient = score.compute_primary_score(b4min_seeds[k].structure);
                                }
                            } else
                                valid_orient = true;

                            if (valid_orient) {

                                 // internal_energy here is just the repulsive component of the vdw energy
                                 ostringstream text;
                                 text <<  i << ":" << l << ":" << j << ":" << k; 
                                 
                                 // This section is required for the growth tree.
                                 // When the growth tree is turned off, computing rmsd for all the conformers
                                 // is not needed.
                                 if (print_growth_tree) conf_header(exp_seeds[k],   text.str(), score);
                                 if (print_growth_tree) conf_header(b4min_seeds[k], text.str(), score);
                                 if (print_growth_tree) b4min_seeds[k].structure.simplex_text = "";

                                 // exp_seeds.score is used for sorting by energy. Only the top #max_conformers will
                                 // be reported, so internal energy should not be use for sorting. 
                                 // Sorting should be done only on the grid score
                                 exp_seeds[k].score = exp_seeds[k].structure.current_score + 
                                                      exp_seeds[k].structure.internal_energy;

                                 // set the score for the b4min structure to be the same as the minimized structure
                                 // this is a hack to make sure b4min_seeds gets sorted in the same order as exp_seeds
                                 b4min_seeds[k].score = exp_seeds[k].score;
 
                                         growth_score_cutoff = growth_score_cutoff_begin  / ( pow (growth_score_scaling_factor, i) );

if (exp_seeds[k].structure.current_score <= growth_score_cutoff) {
                                              if (ie_prune) {
                                                  if (exp_seeds[k].structure.internal_energy > internal_energy_cutoff) {
                                                      exp_seeds[k].used = true;
                                                      confs_pruned_bad_score++;
                                                  }
                                              }
                                          } else {
                                                exp_seeds[k].used = true;
                                                confs_pruned_bad_score++;
                                          }

                            } else {
                                exp_seeds[k].used = true;  // scoring failed
                                confs_pruned_outside_grid++;
                                //if (verbose) cout << "Error scoring Conformer #" << exp_seeds[k].conformer_num << endl;
                            } 
                    } else if (cb_ok[k] == 1) { 
                         exp_seeds[k].used = true;  // failed bump filter 
                         confs_pruned_bump_filter++;
                         //if (verbose) cout << " Conformer #" << exp_seeds[k].conformer_num << " failed bump filter" << endl;
                    } else { 
                        exp_seeds[k].used = true;  // clash_overlap check 
                        confs_pruned_clash_overlap++;
                        //if (verbose) cout << " Conformer #" << exp_seeds[k].conformer_num << " failed clash check" << endl;
                    }

                } // Second pass over new conformers (k)

            }  // Loop over seed conformers (j)
 
            // Pruning section
            seeds.clear(); seeds.reserve(500);
            sort(exp_seeds.begin(), exp_seeds.end(), conformer_less_than);
            sort(b4min_seeds.begin(), b4min_seeds.end(), conformer_less_than);

            // DTM 11-11-08 - code to prune bad energies has been moved into the inner loop where confs are generated
          
            // remove confs that fail the rank/rmsd test
            if(cluster) {
                float RMSD_CUTOFF = pruning_clustering_cutoff;

                // Pre-compute WL colors + per-atom weights for weisfeiler pruning
                // so the O(n^2) pairwise loop pays one wl_color_refine (expensive)
                // and one layer-weight traversal, not one per pair. Colors depend
                // only on the graph (atom types + active bonds), shared across all
                // conformers at this growth stage. (Same scheme as the anchor
                // pruning block above.)
                vector<int>  wl_colors2;
                vector<double> wl_weights2;
                if (pruning_cluster_rmsd_type == "weisfeiler" && ligand_mol_symmetric
                    && !exp_seeds.empty()) {
                    compute_wl_weights(layers, layer_segments, wl_weights2);
                    {
                        static WL_RMSD wl;
                        wl.wl_color_refine(exp_seeds[0].structure, wl_colors2, true);
                    }
                }

                for (j = 0; j < exp_seeds.size(); j++) {
                    if (!exp_seeds[j].used) {
                        for (k = j + 1; k < exp_seeds.size(); k++) {
                            if (!exp_seeds[k].used) {

                                // Dispatch to RMSD function based on pruning_cluster_rmsd_type.
                                // Short-circuit: no symmetry → fall back to std (avoids symmetry cost).
                                if ((pruning_cluster_rmsd_type == "hungarian" ||
                                     pruning_cluster_rmsd_type == "min" ||
                                     pruning_cluster_rmsd_type == "weisfeiler") &&
                                    (!ligand_mol_symmetric))
                                    rmsd = calc_layer_rmsd(exp_seeds[j], exp_seeds[k]);
                                else if (pruning_cluster_rmsd_type == "hungarian")
                                    rmsd = calc_layer_rmsd_hungarian(exp_seeds[j], exp_seeds[k]);
                                else if (pruning_cluster_rmsd_type == "min")
                                    rmsd = calc_layer_rmsd_min(exp_seeds[j], exp_seeds[k]);
                                else if (pruning_cluster_rmsd_type == "weisfeiler")
                                    rmsd = calc_layer_rmsd_weisfeiler(exp_seeds[j], exp_seeds[k],
                                                                      wl_colors2, wl_weights2.data());
                                else
                                    rmsd = calc_layer_rmsd(exp_seeds[j], exp_seeds[k]);
                                rmsd = MAX(rmsd, 0.001);

                                if ((float) k / rmsd > RMSD_CUTOFF) {
                                    exp_seeds[k].used = true;
                                    confs_pruned_clustered++;
                                }

                            }
                        }
                    }
                }
             }
    
            // End pruning section

            // DTM 11-11-08 - modified this to cut off the list of new seeds at the limit num_growth_poses
            // copy pruned confs to seed list
            for (j = 0; ((j < exp_seeds.size())&&(seeds.size()<num_growth_poses)); j++) {

                if (!exp_seeds[j].used) {
                    seeds.push_back(exp_seeds[j]);
                    if (print_growth_tree) all_gen_seeds.push_back(exp_seeds[j]);
                    if (print_growth_tree) all_gen_b4min_seeds.push_back(b4min_seeds[j]);
                } else {
                    // print pruned branch
                    // note that the current pruned conformer is not present in the all_gen_seeds list,
                    // but that does not matter because the function prints the grown ligand (i.e. current pruned conf)
                    // first and then looks for its parents conformers which are present in this list
                }
            }  // End loop copy pruned confs to seed list (j)

            // print the sizes of vectors to show how many were pruned
            if (verbose) {  
                cout << "Lyr:" << i << " Seg:" << l << " "; 
                cout << seeds.size() << "/" << exp_seeds.size() << " retained, Pruning: "; 
                if (confs_pruned_outside_grid>0) cout << confs_pruned_outside_grid << "-outside grid ";
                if (confs_pruned_bad_score>0) cout << confs_pruned_bad_score << "-score ";
                if (cluster && confs_pruned_clustered>0)           cout << confs_pruned_clustered << "-clustered ";
                if (use_clash_penalty && confs_pruned_clash_overlap>0) cout << confs_pruned_clash_overlap << "-clash overlap ";
                if (bump.bump_filter && confs_pruned_bump_filter>0)  cout << confs_pruned_bump_filter << "-bump filter "; 
                if (print_growth_tree) cout << " (" << all_gen_seeds.size() << " in growth tree)";
                cout << " t=" << ((double) (clock() - start)) / CLOCKS_PER_SEC << "s";
                cout << endl;
            }

            //reset pruning stats counters
            confs_pruned_bad_score = 0;
            confs_pruned_clash_overlap = 0;
            confs_pruned_outside_grid = 0;
            confs_pruned_bump_filter = 0;
            confs_pruned_clustered = 0;

        /* Report GPU thermal + dispatch trend after each segment (requires -v) */
        if (verbose) dock_gpu_monitor(i, l, (int)layers[i].segments.size());

        }  // End loop over segments (l)

    }   // End loop over layers (i)

    if (print_growth_tree)
      for (i = 0; i < seeds.size(); i++)
        print_branch(all_gen_seeds, all_gen_b4min_seeds, seeds[i],score);

    // ///////////////////////// END GROWTH /////////////////////////////

    // copy mols to pruned confs
    pruned_confs.clear(); pruned_confs.reserve(seeds.size());
    for (i = 0; i < seeds.size(); i++) {
        pruned_confs.push_back(tmp_scoremol);
        pruned_confs[i].first = seeds[i].score;
        copy_molecule(pruned_confs[i].second, seeds[i].structure);
    }
    count_conf_num++;// to make sure that the conf_num does not repeat.

    //clear all the vectors: do we really need to do this? (sudipto)
    anchor_positions_b4min.clear();
    seeds.clear();
    exp_seeds.clear();
    b4min_seeds.clear();
    all_gen_seeds.clear(); 
    all_gen_b4min_seeds.clear(); 
}

// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::conf_header(CONFORMER & conf, string text, Master_Score & score)
{
    // sudipto & trent - 11-14-08 a print header for score,etc inside growth loop
    // text = "Anchor" for the anchor and lyr:seg:cnf:tor for growth conformers
    ostringstream conformer_text;
    // For consistency use the formatting for score outputting.
    const string DELIMITER = Base_Score::DELIMITER;

    // Print conformer and parent number afmin 
    conformer_text << DELIMITER << "Conformer #:\t\t" << conf.conformer_num << endl 
                   << DELIMITER << "Parent #:\t\t" << conf.parent_num << endl 
                   << DELIMITER << "Anchor #:\t" << current_anchor << endl; 

    // calculate grid score for the current conformer
    // also, calculate internal score for the current conformer
    score.compute_primary_score(conf.structure);

    // String dockmol.current_data contains the output score summary
    conformer_text << conf.structure.current_data; 

    if (conf.parent_num==-1)  // Anchors 
      conformer_text << DELIMITER << text  << endl;
    else                      // Growth Conformers
      conformer_text << DELIMITER << "lyr:seg:cnf:tor:\t" 
                     << fixed << setprecision(3) << text << endl;

    conformer_text << DELIMITER << "RMSD2Anc:\t" // calculates RMSD to unminimized anchor
                     << calc_active_rmsd(conf_anchors[conf.anchor_num], conf) << endl;
    
    // create a new conformer object for orig as calc_active_rmsd needs a CONFORMER
    // DOCKMol orig = xtal input ligand
    CONFORMER    orig_conf;
    copy_molecule(orig_conf.structure, orig);

    // this RMSD considers only the active atoms in the partially gorwn conformer
    // which is why conf must precede orig when calling calc_active_rmsd
    conformer_text << DELIMITER << "RMSD2Orig:\t" // calculates RMSD to xtal input ligand
                     << calc_active_rmsd(conf, orig_conf) << endl;

    // for optimization we should really stop creating a new conformer for orig
    // every time this function is called. We can either store orig_conf in the class
    // or overload calc_active_rmsd() to work with a DOCKMol as well: sudipto
    
    // save the header text in the current conformer
    conf.header = conformer_text.str();
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::segment_torsion_drive(CONFORMER & conf, int current_bond, vector < CONFORMER > &return_list, int num_rec)
{
    int             num_torsions,
                    bond;
    CONFORMER       new_conf,
                    tmp_conf;
    float           new_angle;
    int             seg_id;

    // bond =
    // layer_segments[layers[conf.layer_num-1].segments[current_bond]].rot_bond;
    bond = layer_segments[layers[conf.layer_num].segments[current_bond]].rot_bond;      // -1

    num_torsions = conf.structure.amber_bt_torsion_total[bond_list[bond].bond_num];

    // sudipto & trent Dec 09, 2008
    // print num of torsions sampled
    //if (verbose) cout << num_torsions << " torsions sampled for bond_num=" << bond_list[bond].bond_num 
    //    << " Conf_#=" << conf.conformer_num << " Score=" << conf.score << endl;

    // ------------------------------------------------------------------
    // OPTIMIZATION: pre-compute the activation pattern once, apply per torsion.
    //
    // WHAT activate_layer_segment() DOES (formerly called once per torsion):
    //   1. reset_active_lists(): zeros ALL atom_active_flags and
    //      bond_active_flags.  O(N) writes.
    //   2. Re-enables flags for all complete layers before layer_num:
    //      for each layer 0..layer_num-1, for each segment, activate its
    //      atoms and bonds.  O(N) total across all layers.
    //   3. Re-enables flags for the current layer up to current_bond:
    //      for segments 0..current_bond, activate atoms and bonds.  O(S)
    //      where S << N.
    //   4. Counts active atoms/bonds by scanning entire arrays.  O(N).
    //   Total per torsion: ~3N operations.
    //
    // WHY THE PATTERN IS CONSTANT:
    //   activate_layer_segment(mol, layer_num, curr_seg) depends only on
    //   layer_num and curr_seg — NOT on coordinates, torsion angles, or
    //   any per-conformer state.  In this function, layer_num =
    //   conf.layer_num and current_bond are set BEFORE the torsion loop
    //   and never change.  The only thing that varies per torsion is the
    //   angle (via set_torsion, which modifies x/y/z only).  Therefore
    //   the activation pattern is identical for every torsion angle.
    //
    // WHAT THIS OPTIMIZATION DOES:
    //   Before the torsion loop, compute the pattern into temporary arrays
    //   (pattern_atom_flags, pattern_bond_flags) with one pass of the
    //   activate logic — same 3N work done once.  Inside the loop, apply
    //   via std::copy (N writes) instead of calling activate_layer_segment
    //   (3N writes).  This makes the loop body cleaner: set_torsion for
    //   coordinates, then a simple memcpy for flags — no hidden O(N) work.
    //
    // WHY THIS IS SAFE:
    //   1. std::copy completely overwrites atom_active_flags and
    //      bond_active_flags — no stale parent flags can leak through.
    //   2. The count (num_active_atoms, num_active_bonds) is assigned
    //      directly from pre-computed values, not accumulated, so it is
    //      always correct regardless of the parent's state.
    //   3. set_torsion() only modifies x/y/z coordinates using the
    //      atom_child_list — it does not touch active_flags.
    //   4. copy_molecule() (called at loop start) copies the parent's
    //      flags, but std::copy immediately overwrites them — the copy
    //      is wasted work but harmless (already optimized separately in
    //      the inplace copy_molecule change).
    //   5. The only other caller of activate_layer_segment (line 593) is
    //      in the anchor setup loop, not in this function, so it is
    //      unaffected.
    //
    // CODE CLEANLINESS:
    //   The torsion loop body now reads linearly: copy → set torsion →
    //   apply pre-computed flags.  The old call to activate_layer_segment
    //   hid O(N) reset + rebuild + count work inside what looks like a
    //   simple flag assignment.  Making the pattern computation explicit
    //   and separate from the application makes the algorithm clearer.
    // ------------------------------------------------------------------

    // Allocate temporary arrays for the pre-computed activation pattern.
    // We use the parent molecule's dimensions (conf.structure) which are
    // the same for all children in this loop.
    const int natoms = conf.structure.num_atoms;
    const int nbonds = conf.structure.num_bonds;
    bool *pattern_atom_flags = new bool[natoms];
    bool *pattern_bond_flags = new bool[nbonds];
    int pattern_num_active_atoms = 0;
    int pattern_num_active_bonds = 0;

    // Step 1: Zero all flags (replicates reset_active_lists [L1610-1625])
    for (int idx = 0; idx < natoms; idx++)
        pattern_atom_flags[idx] = false;
    for (int idx = 0; idx < nbonds; idx++)
        pattern_bond_flags[idx] = false;

    // Step 2: Activate all complete layers before the current one.
    // Replicates activate_layer_segment [L1567-1583]: for each layer
    // 0..layer_num-1, for each segment in that layer, activate its atoms
    // and bonds.
    for (int li = 0; li < conf.layer_num; li++) {
        for (size_t lj = 0; lj < layers[li].segments.size(); lj++) {
            int seg = layers[li].segments[lj];
            for (size_t lk = 0; lk < layer_segments[seg].atoms.size(); lk++)
                pattern_atom_flags[layer_segments[seg].atoms[lk]] = true;
            for (size_t lk = 0; lk < layer_segments[seg].bonds.size(); lk++)
                pattern_bond_flags[layer_segments[seg].bonds[lk]] = true;
        }
    }

    // Step 3: Activate current layer up to current_bond (inclusive).
    // Replicates activate_layer_segment [L1586-1597]: for segments
    // 0..current_bond in layer layer_num, activate atoms and bonds.
    for (int li = 0; li <= current_bond; li++) {
        int seg_num = layers[conf.layer_num].segments[li];
        for (size_t lj = 0; lj < layer_segments[seg_num].atoms.size(); lj++)
            pattern_atom_flags[layer_segments[seg_num].atoms[lj]] = true;
        for (size_t lj = 0; lj < layer_segments[seg_num].bonds.size(); lj++)
            pattern_bond_flags[layer_segments[seg_num].bonds[lj]] = true;
    }

    // Step 4: Count active atoms/bonds.
    // Replicates activate_layer_segment [L1600-1607]: scan all flags and
    // count those set to true.
    for (int idx = 0; idx < natoms; idx++)
        if (pattern_atom_flags[idx]) pattern_num_active_atoms++;
    for (int idx = 0; idx < nbonds; idx++)
        if (pattern_bond_flags[idx]) pattern_num_active_bonds++;

    for (int i = 0; i < num_torsions; i++) {

        copy_molecule(new_conf.structure, conf.structure);
        new_conf.layer_num = conf.layer_num;
        new_conf.score = conf.score;
        new_conf.used = conf.used;
        new_conf.anchor_num = conf.anchor_num;// trent balius 2008-11-17
        new_conf.parent_num = conf.conformer_num; //trent balius 2008-12-03 
        new_conf.conformer_num = count_conf_num; //trent balius 2008-12-03
//        cout << new_conf.conformer_num << endl;

        new_angle =
            new_conf.structure.amber_bt_torsions[bond_list[bond].bond_num][i];
        new_angle = (PI / 180) * new_angle;

        if (bond_list[bond].seg1 == layer_segments[layers[new_conf.layer_num].segments[current_bond]].origin_segment) { // -1
            new_conf.structure.set_torsion(bond_list[bond].atom1,
                                           bond_list[bond].atom2,
                                           bond_list[bond].atom3,
                                           bond_list[bond].atom4, new_angle);
            seg_id = bond_list[bond].seg2;
        }

        if (bond_list[bond].seg2 == layer_segments[layers[new_conf.layer_num].segments[current_bond]].origin_segment) { // -1
            new_conf.structure.set_torsion(bond_list[bond].atom4,
                                           bond_list[bond].atom3,
                                           bond_list[bond].atom2,
                                           bond_list[bond].atom1, new_angle);
            seg_id = bond_list[bond].seg1;
        }

        if (layer_segments[layers[new_conf.layer_num].segments[current_bond]].origin_segment != bond_list[bond].seg1)   // -1
            if (layer_segments[layers[new_conf.layer_num].segments[current_bond]].origin_segment != bond_list[bond].seg2)       // -1
                cout << "Layer growth error!" << endl;

        // Apply pre-computed activation pattern via std::copy.
        // This replaces the call to activate_layer_segment() which would
        // reset O(N) + rebuild O(N) + count O(N) = 3N operations per
        // torsion.  std::copy does 2N writes (atom + bond flags) plus 2
        // integer assignments (counts).  The pattern was computed once
        // before the torsion loop and is identical for all angles.
        std::copy(pattern_atom_flags, pattern_atom_flags + natoms,
                  new_conf.structure.atom_active_flags);
        std::copy(pattern_bond_flags, pattern_bond_flags + nbonds,
                  new_conf.structure.bond_active_flags);
        new_conf.structure.num_active_atoms = pattern_num_active_atoms;
        new_conf.structure.num_active_bonds = pattern_num_active_bonds;

        // Write_Mol2(new_conf.structure,
        // cout);////////////////////////////////////////////////////////////////

        if (current_bond == layers[new_conf.layer_num].segments.size() - 1) {   // -1
            new_conf.layer_num++;
        }

      // Generate a copy for each grid
      // Generate copies of each segment_torsion_drive seed for each individual receptor grid_num
      for (int mg_num=0; mg_num < num_rec; mg_num++) {
              return_list.push_back(tmp_conf);        // ///////////////////////////////////////////////////////////////
              return_list[return_list.size() - 1].layer_num = new_conf.layer_num;
              return_list[return_list.size() - 1].score = new_conf.score;
              return_list[return_list.size() - 1].used = new_conf.used;
              return_list[return_list.size() - 1].anchor_num = new_conf.anchor_num;// trent balius 2008-11-17
              return_list[return_list.size() - 1].conformer_num = new_conf.conformer_num;// trent balius 2008-12-03
              return_list[return_list.size() - 1].parent_num = new_conf.parent_num;// trent balius 2008-12-03
              copy_molecule(return_list[return_list.size() - 1].structure, new_conf.structure);
              return_list[return_list.size() - 1].structure.grid_num = mg_num;
              count_conf_num++; //trent balius 2008-12-03
      }
    }

    // Free temporary pattern arrays allocated before the torsion loop.
    delete[] pattern_atom_flags;
    delete[] pattern_bond_flags;

}


// +++++++++++++++++++++++++++++++++++++++++
// trent balius 2008-12-03
void
AG_Conformer_Search::print_conformer(CONFORMER & conf)
{
       ostringstream file;
       file << "mol." << conf.conformer_num << "_anchor_" << current_anchor << ".mol2";
       fstream fout;
       fout.open (file.str().c_str(), fstream::out|fstream::app);
                            
       fout << conf.header;
       Write_Mol2(conf.structure, fout);
       return;
}


// +++++++++++++++++++++++++++++++++++++++++
// trent balius 2008-12-03
void
AG_Conformer_Search::print_branch(vector <CONFORMER> &list, vector <CONFORMER> &b4min, const CONFORMER & grown_lig, Master_Score & score)
{
  // this function takes vector of conformers with all levels of growth and prints a branch of the tree
  //cout << "start AG_Conformer_Search::print_branch" << endl;
  int parent = grown_lig.parent_num;
  vector < CONFORMER > new_list;

  // set name for branch output file
  fstream foutbranch;
  char fname[80];
  if (!grown_lig.used) sprintf(fname,"%s_anchor%d_branch%d.mol2",
             grown_lig.structure.title.c_str(),current_anchor,grown_lig.conformer_num);
    else sprintf(fname,"pruned_%d_anchor_%d.mol2",grown_lig.conformer_num,current_anchor);
  foutbranch.open (fname, fstream::out);

  // grown_lig is the start of the growth tree  
  new_list.push_back(grown_lig);

  // add the b4min grown lig conformer
  for (int i=b4min.size()-1; i>=0; i--) 
    if (b4min[i].conformer_num == grown_lig.conformer_num) {
      new_list.push_back(b4min[i]);
      break;
    }

  // store the tree in a vector from grown ligand to anchor
  int n = list.size();  // list of seeds from all generations (parent conformers)

  // count_conf_num backwards in parent conformers list searching for parents till anchor is reached
  // when parent is found append to new_list and set new parent
  for (int i = n - 1; i >= 0; i--){
       if (parent == list[i].conformer_num)
       {
           parent = list[i].parent_num;
           new_list.push_back(list[i]);
           new_list.push_back(b4min[i]);
       }
  }

  // Print from anchor to grown ligand
  n = new_list.size();
  for (int i = n - 1; i >=0; i--)
  {
       // calculate anchor rmsd
       float rmsd_anchor_intial=0, rmsd_prev=0;

       // new_list[n-1].structure is the the anchor for this branch
       // calc_active_rmsd() uses the first dockmol object to fix the number of active atoms
       // so the anchor must be the first one
       rmsd_anchor_intial = calc_active_rmsd(new_list[n-1],new_list[i]);

       // calculates the rmsd between the previous growth phase and the current one
       // when i=0 i.e. anchor, previous rmsd is meaningless and so set to zero
       if (i == n-1) rmsd_prev = 0;
       else rmsd_prev = calc_active_rmsd(new_list[i+1],new_list[i]);

       //foutbranch << "########## loop num:\t\t" << i << endl;
       //foutbranch << "########## RMSD Anchor:\t\t" << rmsd_anchor_intial << endl;
       //           << "########## RMSD Previous:\t" << rmsd_prev << endl;
       ostringstream conformer_text;
       conformer_text << new_list[i].header << "########## RMSD2Prev:\t" << rmsd_prev << endl;
       conformer_text << new_list[i].structure.simplex_text << endl;

       foutbranch << conformer_text.str();

       Write_Mol2(new_list[i].structure, foutbranch);
  }
   foutbranch.close();
}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::activate_layer_segment(DOCKMol & mol, int layer_num,
                                            int curr_seg)
{
    int             i,
                    j,
                    k;
    int             seg_num;

    reset_active_lists(mol);

    // activate all layers but last
    for (i = 0; i < layer_num; i++) {
        for (j = 0; j < layers[i].segments.size(); j++) {
            for (k = 0; k < layer_segments[layers[i].segments[j]].atoms.size();
                 k++) {
                mol.atom_active_flags[layer_segments[layers[i].segments[j]].
                                      atoms[k]] = true;
            }

            for (k = 0; k < layer_segments[layers[i].segments[j]].bonds.size();
                 k++) {
                mol.bond_active_flags[layer_segments[layers[i].segments[j]].
                                      bonds[k]] = true;
            }
        }
    }

    // activate last layer up to current segment
    for (i = 0; i <= curr_seg; i++) {

        seg_num = layers[layer_num].segments[i];

        for (j = 0; j < layer_segments[seg_num].atoms.size(); j++)
            mol.atom_active_flags[layer_segments[seg_num].atoms[j]] = true;

        for (j = 0; j < layer_segments[seg_num].bonds.size(); j++)
            mol.bond_active_flags[layer_segments[seg_num].bonds[j]] = true;

    }

    mol.num_active_atoms = 0;
    mol.num_active_bonds = 0;

    for (i = 0; i < mol.num_atoms; i++)
        if (mol.atom_active_flags[i])
            mol.num_active_atoms++;

    for (i = 0; i < mol.num_bonds; i++)
        if (mol.bond_active_flags[i])
            mol.num_active_bonds++;

}


// +++++++++++++++++++++++++++++++++++++++++
void
AG_Conformer_Search::reset_active_lists(DOCKMol & mol)
{
    int             i;

    for (i = 0; i < mol.num_atoms; i++)
        mol.atom_active_flags[i] = 0;

    for (i = 0; i < mol.num_bonds; i++)
        mol.bond_active_flags[i] = 0;

    mol.num_active_atoms = 0;
    mol.num_active_bonds = 0;

}


// +++++++++++++++++++++++++++++++++++++++++
// Return true if the conf is not rejected by a vDW clash check.
bool
AG_Conformer_Search::segment_clash_check(DOCKMol & conf, int layer_num,
                                         int current_bond)
{
    int             a1,
                    a2;
    float           dist_squared,
                    ref;

    int             i,
                    j,
                    k,
                    l;
    int             seg_num,
                    new_seg;

    bool            skip_flag = false;

    if (use_clash_penalty && dock_gpu_is_active()) {
        fprintf(stderr, "DOCK: clash checking (use_clash_overlap yes) is not "
                        "supported with GPU scoring; run on CPU or disable "
                        "use_clash_overlap\n");
        exit(1);
    }

    if (use_clash_penalty) {

        // ID current segment
        seg_num = layers[layer_num].segments[current_bond];

        // loop over atoms in newest segment
        for (i = 0; i < layer_segments[seg_num].atoms.size(); i++) {
            a1 = layer_segments[seg_num].atoms[i];

            // loop from last layer to all inner layers
            for (j = layer_num; j >= 0; j--) {

                // loop over segments in current layer
                for (k = ((j == layer_num) ? (current_bond - 1) :
                         (layers[j].segments.size() - 1)); k >= 0; k--) {

                    new_seg = layers[j].segments[k];

                    // loop over atoms in this segment
                    for (l = 0; l < layer_segments[new_seg].atoms.size(); l++) {
                        a2 = layer_segments[new_seg].atoms[l];

                        if ((conf.get_bond(a1, a2) == -1)
                            && (!conf.atoms_are_one_three(a1, a2))) {

                            dist_squared = pow((conf.x[a1] - conf.x[a2]), 2) +
                                           pow((conf.y[a1] - conf.y[a2]), 2) +
                                           pow((conf.z[a1] - conf.z[a2]), 2);
                            ref = clash_penalty * (conf.amber_at_radius[a1] +
                                                   conf.amber_at_radius[a2]);
                            skip_flag = dist_squared < ref * ref;

                            if ((!conf.atom_active_flags[a1])
                                || (!conf.atom_active_flags[a2]))
                                cout << "Layer ERROR!!!!" << endl;
                        }
                    }
                }
            }
        }
    }

    return !skip_flag;
}


// +++++++++++++++++++++++++++++++++++++++++
bool
AG_Conformer_Search::next_conformer(DOCKMol & mol)
{



    if (pruned_confs.size() > 0) {
        copy_molecule(mol, pruned_confs[pruned_confs.size() - 1].second);
        pruned_confs.pop_back();
        return true;
    } else {
        return false;
    }



}



// +++++++++++++++++++++++++++++++++++++++++
// Store fragment fingerprints, frequencies, and in certain cases the DOCKMol
// object in a map as molecules are dissected at rotatable bonds
void
AG_Conformer_Search::bickel_count_fragments(DOCKMol & mol)
{

//TODO
// Output some statistics about the input library, e.g. max number of layers, max number of
// scaffolds per layer, etc. Stuff that will help us constrict growth space during dn design

    // Make a temporary DOCKMol and Fingerprint object
    DOCKMol tmp_mol;
    Fingerprint finger;

    // Send DOCKMol to fingerprint function and return a list of allowable
    // torsion environments for this molecule only
    vector <string> tmp_torsions;
    finger.write_torsion_environments(mol, tmp_torsions);

    // Add the allowable torsions to a map
    for (int i=0; i<tmp_torsions.size(); i++){
        if (!torsions_map[tmp_torsions[i]]){
            torsions_map[tmp_torsions[i]] = 1;
        } else {
            torsions_map[tmp_torsions[i]]++;
        }
    }
    /*
    for (int i=0; i<tmp_torsions.size(); ++i){
        cout << tmp_torsions[i] << endl;
    }
    */
    tmp_torsions.clear();


    copy_molecule(tmp_mol, orig);
    vector<pair<int,int> > rot_bond_atom_vect;


    //check all bonds in molecule 
    for (int i=0; i<tmp_mol.num_bonds; i++){
        //if bond is rotatable, save origin and target for later search
        if (mol.bond_is_rotor(i)){
            rot_bond_atom_vect.push_back(make_pair(mol.bonds_origin_atom[i],mol.bonds_target_atom[i]));
/*
            cout << "ORIGIN: " << mol.bonds_origin_atom[i] << " TARGET   " << mol.bonds_target_atom[i];
            cout << "  A: " << finger.return_noH_environment(tmp_mol, mol.bonds_origin_atom[i]) <<
            " B: " << finger.return_noH_environment(tmp_mol, mol.bonds_target_atom[i]) << endl << endl;

*/
            if (Parameter_Reader::verbosity_level() > 1 ) {
                cout << "Rot_Bond_Check  " << "Origin: " << mol.bonds_origin_atom[i] 
                << "\tTarget: " << mol.bonds_target_atom[i] << endl;

            }
        }
    }

    vector<pair<int,int> > seg_pairs; //pairs of bound segments
    vector<pair<pair<int,string>,pair<int,string> > > half_tors_pairs;
    
    //for all segments
    for (int i=0; i<orig_segments.size(); i++){
        //for all atoms in each segment
        for (int x=0; x<orig_segments[i].atoms.size(); x++){
            //for all rotable bonds in the molecule
            for (int y=0; y<rot_bond_atom_vect.size(); y++) {
                //if the current atom matches the first atom in the rotatable bond pair
                if (orig_segments[i].atoms[x] == rot_bond_atom_vect[y].first){
                    //for all other segments
                    for (int sec_seg=0; sec_seg<orig_segments.size(); ++sec_seg){
                        //if it's the current segment, skip it (can't be bound to itself)
                        if (sec_seg == i){
                            continue;
                        }
                        //checks all atoms on the second segment for the second half of the rot bond pair
                        for (int atoms_seg2=0; atoms_seg2<orig_segments[sec_seg].atoms.size(); ++atoms_seg2){
                            //if the second atom is in this segment, the segments are bound together
                            if (orig_segments[sec_seg].atoms[atoms_seg2] == rot_bond_atom_vect[y].second){

                                if (Parameter_Reader::verbosity_level() > 1 ) {
                                    cout << "ATOM " << orig_segments[i].atoms[x] << " ON SEGMENT " << i << 
                                        " IS ATTACHED TO ATOM " << orig_segments[sec_seg].atoms[atoms_seg2] <<
                                        " ON SEGMENT " << sec_seg << endl;

                                    cout << "Segment: " << i << "  w/ half torsion " << 
                                        finger.return_noH_environment(tmp_mol, orig_segments[i].atoms[x]) << " is bound to" << 
                                        " Segment: " << sec_seg << " w/ half torsion " << 
                                        finger.return_noH_environment(tmp_mol, orig_segments[sec_seg].atoms[atoms_seg2]) << "CAT" << endl;
                                }

                                half_tors_pairs.push_back(make_pair(
                                                    make_pair(i,finger.return_noH_environment(tmp_mol,orig_segments[i].atoms[x])),
                                                    make_pair(sec_seg,finger.return_noH_environment(tmp_mol,orig_segments[sec_seg].atoms[atoms_seg2]))));        
                                //push bound segment pair together for later use
                                seg_pairs.push_back(make_pair(i,sec_seg));
                            }
                        }//all atoms in second segment 
                    } // all other segments
                } //if current atom matches first in pair
            } //all rotable bonds in molecule
        } // all atoms in each segment
    } // all segments
    if (Parameter_Reader::verbosity_level() > 1 ) {
        for (int i=0; i<half_tors_pairs.size();++i){
            cout << half_tors_pairs[i].first.first << "   " << half_tors_pairs[i].second.first << endl;
            cout << half_tors_pairs[i].first.second << "  " << half_tors_pairs[i].second.second << endl;
        }
    }
    //at this point, all segment pairs for this molecule have been found - fragmentation can happen
    vector<pair<int,string> > seg_with_fingerprints; //pair.first = segment #, .second = fingerprint


    // Loop through each of the segments in tmp_mol and activate those atoms/
    // bond plus neighbor_atoms and neighbor_bonds for the given segment
    for (int i=0; i<orig_segments.size(); i++) {
        // And activate the segment + neighbor atoms / bonds
        copy_molecule(tmp_mol, orig);
        activate_fragment(tmp_mol, i);

        // Compute a fingerprint for just the active segment of the molecule
        string tmp_string;
        tmp_string = finger.compute_unique_string_active(tmp_mol);
        seg_with_fingerprints.push_back(make_pair(i,tmp_string));

        // Check to see if the tmp_string has already been seen as part of the
        // following hash (if it has not yet been seen):
        if (!segment_fingerprints[tmp_string].second){

            // Then add it to the hash with a frequency of '1'
            segment_fingerprints[tmp_string].second = 1;

        } else { // Else it has already been added to the hash,

            // So increment the frequency counter
            segment_fingerprints[tmp_string].second++;
        }


        // If the frequency counter is exactly the number of the 
        // fragment_library_freq_cutoff, then remember the mol object
        // Note: 
        if (segment_fingerprints[tmp_string].second == fragment_library_freq_cutoff){

            // Translate one of the dummy atoms to the origin
            if (fragment_library_trans_origin){

                // Declare a dockvector for the translation
                DOCKVector trans_vec;
    
                // Look through the atoms to identify the first that is active 
                // and a dummy
                for (int j=0; j<tmp_mol.num_atoms; j++){
    
                    if (tmp_mol.atom_active_flags[j] && tmp_mol.atom_types[j].compare("Du")==0){
                        trans_vec.x = -tmp_mol.x[j];
                        trans_vec.y = -tmp_mol.y[j];
                        trans_vec.z = -tmp_mol.z[j];
                        break;    
                    }
                }
    
                // Translate the whole dockmol based on the coordinates of that 
                // dummy atom
                tmp_mol.translate_mol(trans_vec);
            }

            // Add the dockmol object to the hash
            copy_molecule(segment_fingerprints[tmp_string].first, tmp_mol);
        }    
    }
    //frags_with_half_tors == map < string, map < string, map < string, map < string, int >>>>
    //fragment_binding_pairs == map < string, map < string, int> >
    //half_tors_pairs is from up above - vector<pair<pair<int,string>,pair<int,string>>>
    //map<string,map<string,int>>
    //vector<pair<int,int>> seg_pairs; //pairs of bound segments
    //for all bound pairs (segment numbers) 

    for (int entry=0; entry<half_tors_pairs.size(); ++entry) {
    }
    for (int entry=0; entry<seg_pairs.size(); entry++){
        //for all segments with associated footprints
        for (int i=0; i<seg_with_fingerprints.size(); i++) { 
            //if segment # in bonding pairs matches with segment # for fingerprints
            if (seg_with_fingerprints[i].first == seg_pairs[entry].first ){


                //if the footprint hasn't been stored here before
                if (fragment_binding_pairs.count(seg_with_fingerprints[i].second) == 0) {

                    //runs through all other segment fingerprints
                    for (int x=0; x<seg_with_fingerprints.size();x++){\
                        if (x == i){ // can't be bound to self 
                            continue;
                        }
                        //checks if the segment fingerprint matches the second in the pair
                        if (seg_with_fingerprints[x].first == seg_pairs[entry].second){
                            fragment_binding_pairs[seg_with_fingerprints[i].second][seg_with_fingerprints[x].second] =1;


                            for (int half_tor_int=0; half_tor_int < half_tors_pairs.size(); ++half_tor_int){
                                if ((half_tors_pairs[half_tor_int].first.first == seg_pairs[entry].first) 
                                    && (half_tors_pairs[half_tor_int].second.first == seg_pairs[entry].second)) {
                                    frags_with_half_tors[seg_with_fingerprints[i].second]
                                                        [seg_with_fingerprints[x].second]
                                                        [half_tors_pairs[half_tor_int].first.second]
                                                        [half_tors_pairs[half_tor_int].second.second] = 1;

                                    break;
                                }
                            }
                        }
                    }


                } else if (fragment_binding_pairs.count(seg_with_fingerprints[i].second) > 0){
                    for (int x=0; x<seg_with_fingerprints.size();x++){
                        if (x == i){ //can't be bound to self
                            continue;
                        }
                        if (seg_with_fingerprints[x].first == seg_pairs[entry].second){
                            fragment_binding_pairs[seg_with_fingerprints[i].second][seg_with_fingerprints[x].second]+=1;


                            for (int half_tor_int=0; half_tor_int < half_tors_pairs.size(); ++half_tor_int){
                                if ((half_tors_pairs[half_tor_int].first.first == seg_pairs[entry].first) 
                                    && (half_tors_pairs[half_tor_int].second.first == seg_pairs[entry].second)){

                                    frags_with_half_tors[seg_with_fingerprints[i].second]
                                                        [seg_with_fingerprints[x].second]
                                                        [half_tors_pairs[half_tor_int].first.second]
                                                        [half_tors_pairs[half_tor_int].second.second] +=1;
                                    break;

                                }
                            }      
                        }
                    }
                }
            }
        }
    }
    return;
} //end bickel_count_fragments




// +++++++++++++++++++++++++++++++++++++++++
// Store fragment fingerprints, frequencies, and in certain cases the DOCKMol
// object in a map as molecules are dissected at rotatable bonds
void
AG_Conformer_Search::count_fragments(DOCKMol & mol)
{

//TODO
// Output some statistics about the input library, e.g. max number of layers, max number of
// scaffolds per layer, etc. Stuff that will help us constrict growth space during dn design

    // Make a temporary DOCKMol and Fingerprint object
    DOCKMol tmp_mol;
    Fingerprint finger;

    // Send DOCKMol to fingerprint function and return a list of allowable
    // torsion environments for this molecule only
    vector <string> tmp_torsions;
    finger.write_torsion_environments(mol, tmp_torsions);

    // Add the allowable torsions to a map
    for (int i=0; i<tmp_torsions.size(); i++){
        if (!torsions_map[tmp_torsions[i]]){
            torsions_map[tmp_torsions[i]] = 1;
        } else {
            torsions_map[tmp_torsions[i]]++;
        }
        if (torsions_map_ref[tmp_torsions[i]].empty()){
            torsions_map_ref[tmp_torsions[i]] = mol.title;
        }
    }

    tmp_torsions.clear();


    // Loop through each of the segments in tmp_mol and activate those atoms/
    // bond plus neighbor_atoms and neighbor_bonds for the given segment
    for (int i=0; i<orig_segments.size(); i++) {

        // Copy the instance of the molecule from the mol2 (orig) onto tmp_mol
        copy_molecule(tmp_mol, orig);

        // And activate the segment + neighbor atoms / bonds
        activate_fragment(tmp_mol, i);

        // Compute a fingerprint for just the active segment of the molecule
        string tmp_string;
        tmp_string = finger.compute_unique_string_active(tmp_mol);
        //cout << tmp_string << endl;

        // Check to see if the tmp_string has already been seen as part of the
        // following hash (if it has not yet been seen):
        if (!segment_fingerprints[tmp_string].second){

            // Then add it to the hash with a frequency of '1'
            segment_fingerprints[tmp_string].second = 1;

        } else { // Else it has already been added to the hash,

            // So increment the frequency counter
            segment_fingerprints[tmp_string].second++;
        }
        // If the frequency counter is exactly the number of the 
        // fragment_library_freq_cutoff, then remember the mol object
        // Note: 
        if (segment_fingerprints[tmp_string].second == fragment_library_freq_cutoff){
            // Translate one of the dummy atoms to the origin
            if (fragment_library_trans_origin){
                //bypass rigid molecules to be translated 
                bool tmp_bool;
                int tmp_du_counter = 0;
                // characterize a rigid molecule by counting the Du atoms into a int variable
                // by counting the num of du atoms in the active atoms
                for (int j=0; j<tmp_mol.num_atoms; j++){
                    if (tmp_mol.atom_active_flags[j] && tmp_mol.atom_types[j].compare("Du")==0){
                        tmp_du_counter = tmp_du_counter + 1; 
                    }    
                }
                // if tmp_du_counter is 0 then it is a rigid molecule and do nothing
                if (tmp_du_counter == 0) {
                         
                // if tmp_du_counter has some non-zero value, start the linear transformation 
                }else{
                 
                    int du_counter = 0;
                    vector<int> nbrs;

                    // Declare a dockvector for the translation
                    DOCKVector trans_vec;
    
                    // Look through the atoms to identify the first that is active 
                    // and a dummy
                    for (int j=0; j<tmp_mol.num_atoms; j++){
    
                        if (tmp_mol.atom_active_flags[j] && tmp_mol.atom_types[j].compare("Du")==0){
                            trans_vec.x = -tmp_mol.x[j];
                            trans_vec.y = -tmp_mol.y[j];
                            trans_vec.z = -tmp_mol.z[j];

                            du_counter = j;
                            break;    
                        }
                    }
                    // get neighboring atom to that dummy atom
                    nbrs = tmp_mol.get_atom_neighbors(du_counter);

                    // Translate the whole dockmol based on the coordinates of that 
                    // dummy atom
                    tmp_mol.translate_mol(trans_vec);
               
                    // PAK
                    
                    vector <int> list_active_du {};
                    for (int j=0; j<tmp_mol.num_atoms; j++){
                        if (tmp_mol.atom_active_flags[j] && tmp_mol.atom_types[j].compare("Du")==0){
                            list_active_du.push_back(j);
                        }
                    } 
                    // vec1 dummy atom(origin) to the neighboring atom
                    DOCKVector vec1;
 
                    if (list_active_du.size() == 2 ) {
                        for (int j=0; j<list_active_du.size(); j++) {    
                            if (list_active_du[j] != du_counter) { 
                                vec1.x = tmp_mol.x[list_active_du[j]];
                                vec1.y = tmp_mol.y[list_active_du[j]];
                                vec1.z = tmp_mol.z[list_active_du[j]];  
                                break;
                            }
                        }
                    }else {
                        for (int j=0; j<nbrs.size(); j++) {
                            if (tmp_mol.atom_active_flags[nbrs[j]])
                                vec1.x = tmp_mol.x[nbrs[j]];
                                vec1.y = tmp_mol.y[nbrs[j]];
                                vec1.z = tmp_mol.z[nbrs[j]];
                        }
                    } 

                    //vec2 is origin pointing along x axis
                    DOCKVector vec2;
                    vec2.x = 1.00;
                    vec2.y = 0.0;
                    vec2.z = 0.0;
  

                    // Declare some variables
                    float dot;
                    float vec1_magsq;
                    float vec2_magsq;
                    float cos_theta;
                    float sin_theta;

                    //Compute dot product 
                    dot = dot_prod(vec1, vec2);

                    //Compute magnitudes squared
                    vec1_magsq = (vec1.x * vec1.x) + (vec1.y * vec1.y) + (vec1.z * vec1.z);
                    vec2_magsq = (vec2.x * vec2.x) + (vec2.y * vec2.y) + (vec2.z * vec2.z);

                    // Compute cos/sin theta
                    cos_theta = dot / (sqrt (vec1_magsq * vec2_magsq));
 
                    // Some numerical stuff to avoid 
                    if (cos_theta >= 1.0){
                        cos_theta = 1.0;
                        sin_theta = 0.0;
                    }
                    else if (cos_theta <= -1.0){
                        cos_theta = -1.0;
                        sin_theta = 0.0;
                    }
                    else{
                        sin_theta = sqrt (1 - (cos_theta * cos_theta));
                    }

                    // if cos_theta = -1, vectors are parallel in opposite directions
                    if (cos_theta == -1){

                        //Declare the rotation matrix and rotate 
                        double finalmat[3][3] = { { -1, 0, 0}, {0, -1, 0}, {0, 0, -1} };
                        tmp_mol.rotate_mol(finalmat); //LEP do i put the matrix format
                    }

                    //If cos_theta is 1, vec1 and vec2 are perfect
                    //Otherwise, enter this loop and calculate lots of stuff
                    else if (cos_theta != 1) {
                        //Calculate cross product of vec1 and vec2 to get U
                        DOCKVector normalU = cross_prod(vec1, vec2);

                        //Calc cross product of vec2 and U to get ~W
                        DOCKVector normalW = cross_prod(vec2, normalU);

                        // Normalize the vectors
                        vec2 = vec2.normalize_vector();
                        normalU = normalU.normalize_vector();
                        normalW = normalW.normalize_vector();


                        // Make coorinate rotation matrix, which rotates coordinates to normalW, vec2, normalU coordinates
                        float coorRot[3][3];
                        coorRot[0][0] = normalW.x;  
                        coorRot[0][1] = vec2.x;  
                        coorRot[0][2] = normalU.x;
                        coorRot[1][0] = normalW.y;  
                        coorRot[1][1] = vec2.y;  
                        coorRot[1][2] = normalU.y;
                        coorRot[2][0] = normalW.z;  
                        coorRot[2][1] = vec2.z;  
                        coorRot[2][2] = normalU.z;

                        // make inverse matrix of coordRot matrix - since coorRot is an orthonal matrix 
                        // the inverse if its transpose, coorRot^T
                        float invcoorRot[3][3];
                        invcoorRot[0][0] = coorRot[0][0];  
                        invcoorRot[0][1] = coorRot[1][0];
                        invcoorRot[0][2] = coorRot[2][0];

                        invcoorRot[1][0] = coorRot[0][1];  
                        invcoorRot[1][1] = coorRot[1][1];
                        invcoorRot[1][2] = coorRot[2][1];

                        invcoorRot[2][0] = coorRot[0][2];  
                        invcoorRot[2][1] = coorRot[1][2];
                        invcoorRot[2][2] = coorRot[2][2];

                        // Make rotation matrix, which rotates vec2 theta angle on a plane of vec2 and normalW
                        // to the direction of normalW
                        float planeRot[3][3];
                        planeRot[0][0] =  cos_theta; planeRot[0][1] = -sin_theta; planeRot[0][2] = 0;
                        planeRot[1][0] =  sin_theta; planeRot[1][1] =  cos_theta; planeRot[1][2] = 0;
                        planeRot[2][0] =          0; planeRot[2][1] =          0; planeRot[2][2] = 1;

                        //multiply 3 matrices together: coorRot*planeRot*invcoorRot
                        float temp[3][3];
                        double finalmat[3][3];

                        // First multiply CoorRot * planceRot, save as temp
                        for (int i=0; i<3; i++){
                            for (int j=0; j<3; j++){
                                temp[i][j] = 0.0;
                                for (int k=0; k<3; k++){
                                    temp[i][j] += coorRot[i][k]*planeRot[k][j];
                                }
                            }
                        }

                        //Then multiply temp* invcoorRot, save as finalmat
                        for (int i=0; i<3; i++){
                            for (int j=0; j<3; j++){
                                finalmat[i][j] = 0.0;
                                for (int k=0; k<3; k++){
                                    finalmat[i][j] += temp[i][k]*invcoorRot[k][j];
                                }
                            }
                        }

                        //Rotate temp_mol using finalmat3/3
                        tmp_mol.rotate_mol(finalmat);   
                    }
                    //PAK
                    //Now rotate molecule so it could get on the X-Y plane
                    //initializing vectors and variables. Must get three points to make a flat 2D plane.
                    //first point is the Du (0,0,0). second point is the atom attached to Du (x, 0, 0).
                    //third point is the chosen atom that is attached to the neighboring du atom ( x,y,0)
                    
                    //vector<int> next_nbrs {};         
                    //vector<int> tmp_nextnbrs {}; 
                    //vector<string> tmp_vec_atom_strings {};
                    //vector<float> tmp_vec_atom_wt {};
                    int tmp_pos = 0;

                    DOCKVector vec3; 
                    DOCKVector vec4;
                    DOCKVector vec5;
                    
                    //Assign variables to the three vectors
                    vec3.x = 0;
                    vec3.y = 0;
                    vec3.z = 0; 
               
                    //this is the second point, second point is the active atom attached to the Du atom...
                    // Or if it is a linker. the second du atom that is not at the origin is the second point
                    int tmp_counter = 0;
                    if (list_active_du.size() == 2 ) {
                        for (int j=0; j<list_active_du.size(); j++) {
                            if (list_active_du[j] != du_counter) {
                                vec4.x = tmp_mol.x[list_active_du[j]];
                                vec4.y = tmp_mol.y[list_active_du[j]];
                                vec4.z = tmp_mol.z[list_active_du[j]];
                                tmp_counter = j; 
                                break;
                            }
                        }
                    }else{
                        for (int j = 0; j<nbrs.size(); j++){
                            if ( tmp_mol.atom_active_flags[nbrs[j]]) {
                                vec4.x = tmp_mol.x[nbrs[j]];
                                vec4.y = tmp_mol.y[nbrs[j]]; 
                                vec4.z = tmp_mol.z[nbrs[j]];     
                                next_nbrs = tmp_mol.get_atom_neighbors(nbrs[j]);
                                break;
                            }
                        }
                    }
 

                    //the third point is trying to get the heavier atom among the 
                    //list of neighboring atoms of the active du neighboring atom... 
                    //...Or if it is a linker the third atom should be the neighboring at of the origin du atom
                    tmp_counter = 0;
                    if (list_active_du.size() == 2 ) {
                        for (int j=0; j<nbrs.size(); j++) {
                            if (tmp_mol.atom_active_flags[nbrs[j]]){
                                vec5.x = tmp_mol.x[nbrs[j]];
                                vec5.y = tmp_mol.y[nbrs[j]];
                                vec5.z = tmp_mol.z[nbrs[j]];
                                tmp_counter = j;
                                break;
                            } 
                        }
 
                        //So you can compute the direction vector D of X axis (origin to point).
                        //This is the difference of origin and X-axis vector, divided by their distance
                        DOCKVector x_axis_vec {}; 
                        DOCKVector duplicate_vec5 {};
                        
                        duplicate_vec5 = vec5; 
                        float tmp_angle = 0.00;
                        x_axis_vec.x = 1;
                        x_axis_vec.y = 0;
                        x_axis_vec.z = 0;

                        //normalize duplicate_vec5
                        duplicate_vec5.normalize_vector();      
  
                        //calculate dot product between duplicate_vec5 and x_axis_vec 
                        tmp_angle = get_vector_angle(x_axis_vec,duplicate_vec5);  

                        //getting the heaviest atom from a input vector filled with neighboring active atom types 
                        if (tmp_angle <= 0.05) {
                            tmp_nextnbrs = tmp_mol.get_atom_neighbors(nbrs[tmp_counter]); 

                            for (int a=0; a<tmp_nextnbrs.size(); a++){
                                if (tmp_mol.atom_active_flags[tmp_nextnbrs[a]]){
                                    tmp_vec_atom_strings.push_back(tmp_mol.atom_types[tmp_nextnbrs[a]]);
                                }
                            }
                            //calculating atomic weight through an input vector of active atom types.
                            tmp_vec_atom_wt = calc_atoms_wt(tmp_vec_atom_strings, false);

                            // getting the heaviest atom
                            float max_number = *max_element(tmp_vec_atom_wt.begin(), tmp_vec_atom_wt.end());
 
                            // saving the atom position of the active molecule and saving its xyz coordinates
                            // as the third point of rotation
                            for (int j=0; j<tmp_vec_atom_wt.size(); j++){
                                if (tmp_vec_atom_wt[j] == max_number){
                                    tmp_pos = j;
                                }
                            }

                            vec5.x = tmp_mol.x[tmp_nextnbrs[tmp_pos]];
                            vec5.y = tmp_mol.y[tmp_nextnbrs[tmp_pos]];
                            vec5.z = tmp_mol.z[tmp_nextnbrs[tmp_pos]];
 
                            tmp_vec_atom_wt.clear();
                            tmp_vec_atom_strings.clear();
                            tmp_nextnbrs.clear();
                             
                        }  
                    //if not linker, get the heaviest atom in the scaffold/sidechain as the neighboring third point 
                    } else{ 
                        for (int a=0; a<next_nbrs.size(); a++){   
                            if (tmp_mol.atom_active_flags[next_nbrs[a]]){
                                tmp_vec_atom_strings.push_back(tmp_mol.atom_types[next_nbrs[a]]);
                            }    
                        }
                        //calculating atomic weight through an input vector of active atom types.
                        tmp_vec_atom_wt = calc_atoms_wt(tmp_vec_atom_strings, false);

                        // getting the heaviest atom
                        float max_number = *max_element(tmp_vec_atom_wt.begin(), tmp_vec_atom_wt.end());

                        // saving the atom position of the active molecule and saving its xyz coordinates
                        // as the third point of rotation
                        for (int j=0; j<tmp_vec_atom_wt.size(); j++){
                            if (tmp_vec_atom_wt[j] == max_number){
                                tmp_pos = j;
                            }
                        }

                        vec5.x = tmp_mol.x[next_nbrs[tmp_pos]];
                        vec5.y = tmp_mol.y[next_nbrs[tmp_pos]];
                        vec5.z = tmp_mol.z[next_nbrs[tmp_pos]];

                        tmp_vec_atom_wt.clear();
                        tmp_vec_atom_strings.clear();
 
                    }
                    
                    // initializing variables;  
                    float scalar_rot_x;
                    double Xaxi_Rot[3][3]; 
                    double reflection_mat [3][3];
                    float tmpx;
                    float tmpy;
                    float tmpz;
                    float tmpa;
                    float tmpb;
                    float tmpc;
                    float scalar_rot_z;

                    //scalar_rot_x is a scalar to make the first row of the matrix unit length
                    tmpx = vec3.x - vec4.x;
                    tmpy = vec3.y - vec4.y;
                    tmpz = vec3.z - vec4.z;           

                    scalar_rot_x = 1.0 / sqrt ((tmpx * tmpx) + (tmpy * tmpy) + (tmpz * tmpz)); 
 
                    //assigning each position of the top row in the matrix and applying scalar_rot_x to make them unit length.
                    //Points are getting mapped into the x-axis.
                    Xaxi_Rot[0][0] = scalar_rot_x * tmpx;
                    Xaxi_Rot[0][1] = scalar_rot_x * tmpy;
                    Xaxi_Rot[0][2] = scalar_rot_x * tmpz;

                    tmpa = Xaxi_Rot[0][0];
                    tmpb = Xaxi_Rot[0][1]; 
                    tmpc = Xaxi_Rot[0][2]; 
  
                    tmpx = vec5.x - vec3.x;
                    tmpy = vec5.y - vec3.y;
                    tmpz = vec5.z - vec3.z;

                    ////scalar_rot_z is a scalar to make the third row of the matrix unit length
                    scalar_rot_z = 1.0 / sqrt ( (tmpz * tmpb - tmpy * tmpc) * (tmpz * tmpb - tmpy * tmpc) +  
                                                (tmpx * tmpc - tmpz * tmpa) * (tmpx * tmpc - tmpz * tmpa) +  
                                                (tmpy * tmpa - tmpx * tmpb) * (tmpy * tmpa - tmpx * tmpb) ); 
 
                    //assigning each position of the bottom row in the matrix
                    //and applying scalar_rot_z to make the third row of the matrix unit length 
                    Xaxi_Rot[2][0] = scalar_rot_z * ((tmpz * tmpb) - (tmpy * tmpc)); 
                    Xaxi_Rot[2][1] = scalar_rot_z * ((tmpx * tmpc) - (tmpz * tmpa)); 
                    Xaxi_Rot[2][2] = scalar_rot_z * ((tmpy * tmpa) - (tmpx * tmpb));

                    //the middle row of the matrix can be deteremined as the vector
                    //which is perpendicular to both top and bottom rows of the matrix
                    Xaxi_Rot[1][0] = (Xaxi_Rot[2][2] * Xaxi_Rot[0][1]) - (Xaxi_Rot[2][1] * Xaxi_Rot[0][2]);
                    Xaxi_Rot[1][1] = (Xaxi_Rot[2][0] * Xaxi_Rot[0][2]) - (Xaxi_Rot[2][2] * Xaxi_Rot[0][0]);
                    Xaxi_Rot[1][2] = (Xaxi_Rot[2][1] * Xaxi_Rot[0][0]) - (Xaxi_Rot[2][0] * Xaxi_Rot[0][1]);  
               
                    //applying the rotataion matrix 
                    tmp_mol.rotate_mol(Xaxi_Rot);

                    //assigning reflection matrix because molecules will make
                    //mirror images of the molecule, distorting conformations and changing stereochemistry 
                    reflection_mat[0][0] =  1; reflection_mat[0][1] =   0; reflection_mat[0][2] =  0; 
                    reflection_mat[1][0] =  0; reflection_mat[1][1] =  -1; reflection_mat[1][2] =  0; 
                    reflection_mat[2][0] =  0; reflection_mat[2][1] =   0; reflection_mat[2][2] =  1;                
           
                    //applying the reflection matrix 
                    tmp_mol.rotate_mol(reflection_mat);  
 
                    //next phase is to flip fragments 180 degrees IF the cumulative atoms
                    //wts that are calculated BELOW the X-Z plane
                    //is HIGHER than the cumualtive atom wts ABOVE the X-Z plane. 
                    //this is done to ensure some standard of overlapping for the fragments
                    double flip_180 [3][3];  
                    vector<string> above_atom_strings {};
                    vector<string> below_atom_strings {};
                    vector<float> above_atom_wt {};    
                    vector<float> below_atom_wt {};
 
                    for (int j=0; j<tmp_mol.num_atoms; j++) {
                        if (tmp_mol.y[j] == 0.0000 || tmp_mol.y[j] == -0.0000) {
                            continue;
                        }
                        if (tmp_mol.y[j] >= 0.0001 && tmp_mol.atom_active_flags[j]) {
                            above_atom_strings.push_back(tmp_mol.atom_types[j]);
                        } 
                        if (tmp_mol.y[j] <=  -0.0001 && tmp_mol.atom_active_flags[j]) {
                            below_atom_strings.push_back(tmp_mol.atom_types[j]);
                        }
                    } 
  
                    //create a rotataion matrix that flips 180 degrees 
                    flip_180[0][0] =  1; flip_180[0][1] =   0; flip_180[0][2] =  0;
                    flip_180[1][0] =  0; flip_180[1][1] =  -1; flip_180[1][2] =  0;
                    flip_180[2][0] =  0; flip_180[2][1] =   0; flip_180[2][2] =  -1; 
 
                    //calculates the cumulative molwt above and below the X-Z plane  
                    above_atom_wt = calc_atoms_wt(above_atom_strings, true);
                    below_atom_wt = calc_atoms_wt(below_atom_strings, true);                
 
                    //if below atoms are heavier than the above atoms, flip 180 degrees
                    if (below_atom_wt.back() > above_atom_wt.back()) {
                        tmp_mol.rotate_mol(flip_180);
                    }
                }
            }//PAK_END

            // Add the dockmol object to the hash 
            copy_molecule(segment_fingerprints[tmp_string].first, tmp_mol); 


        }
    }


    return;

} // end AG_Conformer_Search::count_fragments()



// +++++++++++++++++++++++++++++++++++++++++
// Set active atom & active bond flags true for each atom and bond in a segment,
// as well as each neighbor_atom and neighbor_bond of that segment
// (Neighbors are remembered as Du - dummy)
void
AG_Conformer_Search::activate_fragment(DOCKMol & mol, int curr_seg)
{
    // Reset all active atoms and bonds
    reset_active_lists(mol);


    // Activate all of the atoms in the current segment
    for (int i=0; i < orig_segments[curr_seg].atoms.size(); i++) {
          mol.atom_active_flags[orig_segments[curr_seg].atoms[i]] = true;
    }

    // Activate all of the atoms that are neighbors to the current segment
    // change the atom type to 'Du' and atom name to 'DuX' where X = 1...N
    int dummy_counter = 1;
    for (int i=0; i<orig_segments[curr_seg].neighbor_atoms.size(); i++){
        mol.atom_active_flags[orig_segments[curr_seg].neighbor_atoms[i]] = true;
        mol.atom_types[orig_segments[curr_seg].neighbor_atoms[i]] = "Du";

        // Also change the partial charge of dummy atoms to 0.0
        mol.charges[orig_segments[curr_seg].neighbor_atoms[i]] = 0.0;

        stringstream ss;
        ss << "Du" << dummy_counter;
        mol.atom_names[orig_segments[curr_seg].neighbor_atoms[i]] = ss.str();
        ss.clear();

        dummy_counter++;
    }


    // Loop over all bonds and check whether the origin and target atoms of
    // those bonds are both active (tried looping over just SEGMENT.neighbor_
    // bonds, but it was not activating the correct bonds - bug?)
    for (int i=0; i<mol.num_bonds; i++){

        // For a given bond, if both target and origin atoms are active...
        if (mol.atom_active_flags[mol.bonds_origin_atom[i]] &&
            mol.atom_active_flags[mol.bonds_target_atom[i]]   ){

            // ...then make the bond active as well.
            mol.bond_active_flags[i] = true;
        }
    }


    // Write_Mol2 needs these to be defined
    mol.num_active_atoms = 0;
    mol.num_active_bonds = 0;

    for (int i=0; i<mol.num_atoms; i++)
        if (mol.atom_active_flags[i])
            mol.num_active_atoms++;

    for (int i=0; i<mol.num_bonds; i++)
        if (mol.bond_active_flags[i])
            mol.num_active_bonds++;

    return;

} // end AG_Conformer_Search::activate_fragment()

// +++++++++++++++++++++++++++++++++++++++++
// Post-processes fragment libraries to make them unique
void
AG_Conformer_Search::bickel_write_unique_fragments()
{
   // These are the same as those in Base_Score.
   const string DELIMITER    = Base_Score::DELIMITER;
   const int    FLOAT_WIDTH  = Base_Score::FLOAT_WIDTH;
   const int    STRING_WIDTH = Base_Score::STRING_WIDTH;

    // Make a vector for sorting the torsion environments
    vector <string> tmp_torsions;
    map <string, int>::iterator iter;
    for ( iter = torsions_map.begin(); iter != torsions_map.end(); iter++ ){
        ostringstream tmp_string;
        tmp_string <<iter->first <<"-" <<iter->second;
        tmp_torsions.push_back(tmp_string.str());
    }
    sort(tmp_torsions.begin(), tmp_torsions.end());

    // Write the torsion environments to file
    ostringstream file_torenv;
    file_torenv <<fragment_library_prefix <<"_torenv.dat";
    fstream fout_torenv;
    fout_torenv.open (file_torenv.str().c_str(), fstream::out|fstream::app);
    for (int i=0; i<tmp_torsions.size(); i++){
        fout_torenv <<tmp_torsions[i] <<"\n";
    }

    // Clear some memory
    fout_torenv.close();
    tmp_torsions.clear();
    torsions_map.clear();



    // Make a vector for sorting the fragments
    vector <pair <string, int> > tmp_vector;

    // given the map, remove elements that don't have the dockmol object attached
    map < string, pair < DOCKMol , int > >::iterator it;
    for( it = segment_fingerprints.begin(); it != segment_fingerprints.end(); it++){
        if (it->second.second >= fragment_library_freq_cutoff){
            pair <string, int> tmp_pair;
            tmp_pair.first = it->first;
            tmp_pair.second = it->second.second;
            tmp_vector.push_back(tmp_pair);
        }
    }

    // Sort the vector by frequency or by fingerprint
    if (fragment_library_sort_method.compare("freq") == 0){
        sort( tmp_vector.begin(), tmp_vector.end(), frequency_sort );
    }
    else if (fragment_library_sort_method.compare("fingerprint") == 0){
        sort( tmp_vector.begin(), tmp_vector.end(), fingerprint_sort );
    }
    else {
        cout <<"Note :: Unrecognized fragment library sorting method" <<endl;
    }


    // Create the output filenames
    ostringstream file_scaffold, file_linker, file_sidechain, file_rigid;

    file_scaffold << fragment_library_prefix << "_scaffold.mol2";
    file_linker << fragment_library_prefix << "_linker.mol2";
    file_sidechain << fragment_library_prefix << "_sidechain.mol2";
    file_rigid << fragment_library_prefix << "_rigid.mol2";

    // Use those filenames to open output filestreams
    fstream fout_scaffold, fout_linker, fout_sidechain, fout_rigid;

    fout_scaffold.open (file_scaffold.str().c_str(), fstream::out|fstream::trunc);
    fout_linker.open (file_linker.str().c_str(), fstream::out|fstream::trunc);
    fout_sidechain.open (file_sidechain.str().c_str(), fstream::out|fstream::trunc);
    fout_rigid.open (file_rigid.str().c_str(), fstream::out|fstream::trunc);
    
    global_frag_index=0;//initialize global_counter for fragments

    //11.15.2020 - tracker vector for the matrix
    std::vector<string> fragment_names_ordered;


    std::map<string,string> frag_name_with_fingerprints;
    // Iterate over the strings in the vector, to print the dockmols from the map
    for (int i=0; i<tmp_vector.size(); i++) {

        int counter = 0;
        for(int j=0; j<segment_fingerprints[tmp_vector[i].first].first.num_atoms; j++) {
            if(segment_fingerprints[tmp_vector[i].first].first.atom_types[j].compare("Du") == 0){
                counter++;
            }
        }


        // And print to the correct filehandle
        if (counter > 2) {

            ostringstream fragment_name;
            fragment_name << "scf." <<global_frag_index << "-" << tmp_vector[i].second;
            frag_name_with_fingerprints[tmp_vector[i].first] = fragment_name.str();
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            fragment_names_ordered.push_back(fragment_name.str());
            global_frag_index++;

            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:" << setw(FLOAT_WIDTH) << "Scaffold" <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH)<<fragment_name.str() <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed << counter <<endl;
            fout_scaffold <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_scaffold);

        } else if (counter == 2){

            ostringstream fragment_name;
            fragment_name << "lnk." <<global_frag_index  << "-" << tmp_vector[i].second;
            frag_name_with_fingerprints[tmp_vector[i].first] = fragment_name.str();
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            fragment_names_ordered.push_back(fragment_name.str());
            global_frag_index++;

            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH) <<"Linker"<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH) << fragment_name.str()<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_linker<<DELIMITER<<endl;
            

            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_linker);

        } else if (counter == 1){

            ostringstream fragment_name;
            fragment_name <<"sid." <<global_frag_index  << "-" << tmp_vector[i].second;
            frag_name_with_fingerprints[tmp_vector[i].first] = fragment_name.str();
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            fragment_names_ordered.push_back(fragment_name.str());
            global_frag_index++;

            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH) <<"Sidechain"<<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH) << fragment_name.str()<<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_sidechain <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_sidechain);

        } else if (counter == 0){

            ostringstream fragment_name;
            fragment_name <<"rig."  <<global_frag_index  << "-" << tmp_vector[i].second;
            frag_name_with_fingerprints[tmp_vector[i].first] = fragment_name.str();
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            fragment_names_ordered.push_back(fragment_name.str());
            global_frag_index++;

            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH)<<"Rigid"<<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH)<< fragment_name.str()<<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_rigid <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_rigid);

        } else {
            cout <<"Warning :: This type of molecule should not exist " << endl;
        }
        
    }

        //for (std::map<std::string,map<std::string,int>>::iterator it = fragment_binding_pairs.begin();
        //        it != fragment_binding_pairs.end(); ++it) {

        //    cout << "Key " << it->first.second <<endl;

    // Clear some memory
    fout_scaffold.close();
    fout_linker.close();
    fout_sidechain.close();
    fout_rigid.close();
    tmp_vector.clear();
    segment_fingerprints.clear();


    std::map < std::string, std::map <std::string, int> > frag_names_with_attachments;

    /*
    for (std::map<std::string,string>::iterator it = frag_name_with_fingerprints.begin();
            it != frag_name_with_fingerprints.end(); ++it) {
        cout << "Fragment name:   " << it->second << endl;
    }
    */


    //fragment_binding_pairs == map < string, map < string, int> >
    //  It contains a map with 

    //std::map<string,string> frag_name_with_fingerprints (first is key, second is name)
    fstream fout_fragment_attachments;
    fout_fragment_attachments.open("zzz.attachments.dat", fstream::out|fstream::trunc);


    //ent1.first is first key, ent2.first is second key, ent2.second is data
    std::vector < std::vector < int> > mat_vector(fragment_names_ordered.size(), vector<int> (fragment_names_ordered.size(), 0));
    //runs through all binding pairs
    for (auto const &ent1 : fragment_binding_pairs) {
        string start_fragment;
        //runs through all fragments by footprint
        for(auto const &frags : frag_name_with_fingerprints){
            if (ent1.first == frags.first){
                start_fragment = frags.second;
                break;
            }
        }
        //If we aren't keeping all fragments, there'll be some empty spots. This ensures they're aren't written.
        if(start_fragment.size() == 0) {
            continue;
        }
        for (auto const &ent2 : ent1.second) {
            for (auto const &frags : frag_name_with_fingerprints ){
                if (ent2.first == frags.first) {
                    frag_names_with_attachments[start_fragment][frags.second] =
                                ent2.second;
                    //cout << ent2.second;
                }
            }
        }
    }
    //runs through all fragments by fragment name (ex. sid-1)
    std::string origin_fingerprint;
    for (auto const &ent1 : frag_names_with_attachments){
        std::vector< std::string > temp_tors_vect;
        for(auto const &frags : frag_name_with_fingerprints){
            if (frags.second == ent1.first){
                origin_fingerprint = frags.first;
                //cout << endl << "ORIGIN " << ent1.first << endl;
                //cout << "ORG TORSIONS START" << endl;
                fout_fragment_attachments << endl << "ORIGIN " << ent1.first << endl;
                fout_fragment_attachments << "ORG TORSIONS START" << endl;
                for (auto const &f0s : frags_with_half_tors) {
                    for(auto const &f1s : f0s.second){
                        for (auto const &t0s : f1s.second){
                            if (f0s.first == frags.first){

                                if (find(temp_tors_vect.begin(), temp_tors_vect.end(), t0s.first) != temp_tors_vect.end()){
                                    continue;
                                } else {
                                    temp_tors_vect.push_back(t0s.first);
                                }
                                

                            
                            }
                        }
                        
                    }
                }
            }
        }
        for (int i=0; i<temp_tors_vect.size(); ++i){
            //cout << temp_tors_vect[i] << endl;
            fout_fragment_attachments << temp_tors_vect[i] << endl;
        }
        temp_tors_vect.clear();
        //cout << "ORG TORSIONS END" << endl << "BOUND TO | FREQUENCY" << endl;
        fout_fragment_attachments << "ORG TORSIONS END" << endl << "BOUND TO | FREQUENCY" << endl;
        //frag_names_with_attachments, ent2 = pair < target fragment name, frequency > 
        for (auto const &ent2 : ent1.second) { //target fragments, by name
            //frag_name_with_fingerprints = map < fingerprint, fragment name (ex sid.1) >
            for(auto const &frags : frag_name_with_fingerprints){
                //if the fragment name in the fingerprint map matches the one in the attachment map
                //this is done to correlate the attachments
                if (frags.second == ent2.first){

                    //this section prepares pieces of the fragment attachment matrix
                    int str_start = ent1.first.find(".")+1;
                    int str_end = ent1.first.find("-");
                    int str2_start = ent2.first.find(".")+1;
                    int str2_end = ent2.first.find("-");

                    mat_vector[stoi(ent1.first.substr(str_start,str_start-str_end))][stoi(ent2.first.substr(str2_start,str2_start-str2_end))] += ent2.second;
                    mat_vector[stoi(ent2.first.substr(str2_start,str2_start-str_end))][stoi(ent1.first.substr(str_start,str_start-str_end))] += ent2.second;


                    for (auto const &t0s : frags_with_half_tors[origin_fingerprint][frags.first] ){
                        for (auto const &t1s : t0s.second) {
                            //cout << frags.second << " " << t0s.first << " " << t1s.first << " " << t1s.second << endl;


                            fout_fragment_attachments << frags.second << " " << t1s.first << " " << t1s.second << endl;
                        }
                    }
                }
            } //end frag_name_with_fingerprints
        } //end second set of attachments
        temp_tors_vect.clear();

        
        //cout << "END" << endl;
        fout_fragment_attachments << "END" << endl;
    }

    fstream fout_fragments_matrix;
    fout_fragments_matrix.open("fraglib_matrix.dat", fstream::out|fstream::trunc);
    //generates the ordered list of fragments for the matrix header
    std::string all_segments_string;
    for(int segment=0; segment<fragment_names_ordered.size(); ++segment){
        int str_start = fragment_names_ordered[segment].find(".") + 1;
        int str_end = fragment_names_ordered[segment].find("-");

        all_segments_string += fragment_names_ordered[segment] + ",";

    }

    //outputs ordered header to matrix file
    fout_fragments_matrix << all_segments_string << endl;
    all_segments_string.clear();

    for(int x=0; x<mat_vector.size(); ++x){
        for (int y=0; y<mat_vector[x].size(); ++y){
            fout_fragments_matrix << mat_vector[x][y] << ",";
        }
        fout_fragments_matrix << endl;
    }
    fout_fragments_matrix.close();



    if (Parameter_Reader::verbosity_level() > 1 ) {
        //ent1.first is first key, ent2.first is second key, ent2.second is data
        //frags_with_half_tors == map < string, map < string, map < string, map < string, int >>>>
        // f0,f1,t0,t1,freq
        for (auto const &ent1 : frags_with_half_tors) {
            for(auto const &ent2 : ent1.second){
                for(auto const &ent3 : ent2.second){
                    for(auto const &ent4 : ent3.second){
                        cout << "First fragment:  " << ent1.first <<endl;
                        cout << "Second fragment: " << ent2.first << endl;
                        cout << "First torsion:   " << ent3.first << endl;
                        cout << "Second torsion:  " << ent4.first << endl;
                        cout << "Frequency:       " << ent4.second << endl << endl;
                    }
                }
            }
        }

    }

    fout_fragment_attachments.close();
    
    return;

} // end AG_Conformer_Search::bickel_write_unique_fragments()


// +++++++++++++++++++++++++++++++++++++++++
// Post-processes fragment libraries to make them unique
void
AG_Conformer_Search::write_unique_fragments()
{
   // These are the same as those in Base_Score.
   const string DELIMITER    = Base_Score::DELIMITER;
   const int    FLOAT_WIDTH  = Base_Score::FLOAT_WIDTH;
   const int    STRING_WIDTH = Base_Score::STRING_WIDTH;

    // Make a vector for sorting the torsion environments
    vector <string> tmp_torsions;
    vector <string> tmp_torsions_ref;
    map <string, int>::iterator iter;

    for ( iter = torsions_map.begin(); iter != torsions_map.end(); iter++ ){
        ostringstream tmp_string;
        tmp_string <<iter->first <<"-" <<iter->second;
        tmp_torsions.push_back(tmp_string.str());
 
        ostringstream tmp_string_ref;
        tmp_string_ref <<iter->first <<"-" <<iter->second <<"-" <<torsions_map_ref[iter->first];
        tmp_torsions_ref.push_back(tmp_string_ref.str());
    }
    sort(tmp_torsions.begin(), tmp_torsions.end());
    sort(tmp_torsions_ref.begin(), tmp_torsions_ref.end());
    // Write the torsion environments to file
    ostringstream file_torenv;
    file_torenv <<fragment_library_prefix <<"_torenv.dat";
    fstream fout_torenv;
    fout_torenv.open (file_torenv.str().c_str(), fstream::out|fstream::app);
    for (int i=0; i<tmp_torsions.size(); i++){
        fout_torenv <<tmp_torsions[i] <<"\n";
    }

    //LEP - spit out spearate torenv with zincid's under debug flag
    if (Parameter_Reader::verbosity_level() > 1) { 
        ostringstream file_torenv_ref;
        file_torenv_ref <<fragment_library_prefix <<"_torenv.ref";
        fstream fout_torenv_ref;
        fout_torenv_ref.open (file_torenv_ref.str().c_str(), fstream::out|fstream::app);
        for (int i=0; i<tmp_torsions_ref.size(); i++){
            fout_torenv_ref <<tmp_torsions_ref[i] <<"\n";
        }
    //CLear some memory
    fout_torenv_ref.close();
    tmp_torsions_ref.clear();
    }

    // Clear some memory
    fout_torenv.close();
    tmp_torsions.clear();
    torsions_map.clear();



    // Make a vector for sorting the fragments
    vector <pair <string, int> > tmp_vector;

    // given the map, remove elements that don't have the dockmol object attached
    map < string, pair < DOCKMol , int > >::iterator it;
    for( it = segment_fingerprints.begin(); it != segment_fingerprints.end(); it++){
        if (it->second.second >= fragment_library_freq_cutoff){
            pair <string, int> tmp_pair;
            tmp_pair.first = it->first;
            tmp_pair.second = it->second.second;
            tmp_vector.push_back(tmp_pair);
        }
    }

    // Sort the vector by frequency or by fingerprint
    if (fragment_library_sort_method.compare("freq") == 0){
        sort( tmp_vector.begin(), tmp_vector.end(), frequency_sort );
    }
    else if (fragment_library_sort_method.compare("fingerprint") == 0){
        sort( tmp_vector.begin(), tmp_vector.end(), fingerprint_sort );
    }
    else {
        cout <<"Note :: Unrecognized fragment library sorting method" <<endl;
    }


    // Create the output filenames
    ostringstream file_scaffold, file_linker, file_sidechain, file_rigid;

    file_scaffold << fragment_library_prefix << "_scaffold.mol2";
    file_linker << fragment_library_prefix << "_linker.mol2";
    file_sidechain << fragment_library_prefix << "_sidechain.mol2";
    file_rigid << fragment_library_prefix << "_rigid.mol2";

    // Use those filenames to open output filestreams
    fstream fout_scaffold, fout_linker, fout_sidechain, fout_rigid;

    fout_scaffold.open (file_scaffold.str().c_str(), fstream::out|fstream::trunc);
    fout_linker.open (file_linker.str().c_str(), fstream::out|fstream::trunc);
    fout_sidechain.open (file_sidechain.str().c_str(), fstream::out|fstream::trunc);
    fout_rigid.open (file_rigid.str().c_str(), fstream::out|fstream::trunc);
    
    global_frag_index=0;//initialize global_counter for fragments
    //cout << "tm_vector size: " << tmp_vector.size() << endl;
    // Iterate over the strings in the vector, to print the dockmols from the map
    for (int i=0; i<tmp_vector.size(); i++) {

        int counter = 0;

        //segment_fingerprints[tmp_vector[i].first].first.prepare_molecule();
        //calc_mol_wt(segment_fingerprints[tmp_vector[i].first].first);
        for(int j=0; j<segment_fingerprints[tmp_vector[i].first].first.num_atoms; j++) {
            //cout << "num atoms: " << segment_fingerprints[tmp_vector[i].first].first.num_atoms << endl;
            if(segment_fingerprints[tmp_vector[i].first].first.atom_types[j].compare("Du") == 0){
                counter++;
            }
        }

        // And print to the correct filehandle
        if (counter > 2) {
            
            calc_mol_wt(segment_fingerprints[tmp_vector[i].first].first);
            ostringstream fragment_name;
            fragment_name <<"scf." <<global_frag_index;
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            // LEP add fragname to substructure name
            for (int k=0; k<segment_fingerprints[tmp_vector[i].first].first.num_atoms; k++){
                segment_fingerprints[tmp_vector[i].first].first.subst_names[k] = 
                    fragment_name.str();
            }
            global_frag_index++;

            fout_scaffold <<DELIMITER<< setw(STRING_WIDTH) <<"TYPE:" << setw(FLOAT_WIDTH) << "Scaffold" <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH)<<fragment_name.str() <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed << counter <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"MW:"<<setw(FLOAT_WIDTH)<< fixed << setprecision(3) <<
                segment_fingerprints[tmp_vector[i].first].first.mol_wt <<endl;
            fout_scaffold <<DELIMITER<<setw(STRING_WIDTH)<<"DB_ID:"<<setw(FLOAT_WIDTH)<<
                                 segment_fingerprints[tmp_vector[i].first].first.title <<endl;
            fout_scaffold <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_scaffold);

        } else if (counter == 2){

            calc_mol_wt(segment_fingerprints[tmp_vector[i].first].first);
            ostringstream fragment_name;
            fragment_name <<"lnk." <<global_frag_index;
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();

            // LEP update substname with fragname
            for (int k=0; k<segment_fingerprints[tmp_vector[i].first].first.num_atoms; k++){
                segment_fingerprints[tmp_vector[i].first].first.subst_names[k] =
                    fragment_name.str();
            }

            global_frag_index++;

            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH) <<"Linker"<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH) << fragment_name.str()<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second<<endl;
            fout_linker<<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_linker <<DELIMITER<<setw(STRING_WIDTH)<<"MW:"<<setw(FLOAT_WIDTH)<< fixed<< setprecision(3) << 
                segment_fingerprints[tmp_vector[i].first].first.mol_wt <<endl;
            fout_linker <<DELIMITER<<setw(STRING_WIDTH)<<"DB_ID:"<<setw(FLOAT_WIDTH)<<
                                 segment_fingerprints[tmp_vector[i].first].first.title <<endl;
            fout_linker<<DELIMITER<<endl;
            

            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_linker);

        } else if (counter == 1){

            calc_mol_wt(segment_fingerprints[tmp_vector[i].first].first);
            ostringstream fragment_name;
            fragment_name <<"sid." <<global_frag_index;
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();

            // LEP update substname with fragname
            for (int k=0; k<segment_fingerprints[tmp_vector[i].first].first.num_atoms; k++){
                segment_fingerprints[tmp_vector[i].first].first.subst_names[k] =
                    fragment_name.str();
            }

            global_frag_index++;

            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH) <<"Sidechain"<<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH) << fragment_name.str()<<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"MW:"<<setw(FLOAT_WIDTH)<< fixed<< setprecision(3) << 
                segment_fingerprints[tmp_vector[i].first].first.mol_wt <<endl;
            fout_sidechain <<DELIMITER<<setw(STRING_WIDTH)<<"DB_ID:"<<setw(FLOAT_WIDTH)<<
                                 segment_fingerprints[tmp_vector[i].first].first.title <<endl;        
            fout_sidechain <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_sidechain);

        } else if (counter == 0){

            calc_mol_wt(segment_fingerprints[tmp_vector[i].first].first);
            ostringstream fragment_name;
            fragment_name <<"rig."  <<global_frag_index;
            segment_fingerprints[tmp_vector[i].first].first.energy = fragment_name.str();
            global_frag_index++;

            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"TYPE:"<<setw(FLOAT_WIDTH)<<"Rigid"<<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"FR_NAME:"<<setw(FLOAT_WIDTH)<< fragment_name.str()<<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"FREQ:"<<setw(FLOAT_WIDTH)<<fixed<<tmp_vector[i].second <<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"CONN_PTS:"<<setw(FLOAT_WIDTH)<<fixed<<counter <<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"MW:"<<setw(FLOAT_WIDTH)<< fixed<< setprecision(3) << 
                segment_fingerprints[tmp_vector[i].first].first.mol_wt <<endl;
            fout_rigid <<DELIMITER<<setw(STRING_WIDTH)<<"DB_ID:"<<setw(FLOAT_WIDTH)<<
                                 segment_fingerprints[tmp_vector[i].first].first.title <<endl;
            fout_rigid <<DELIMITER<<endl;
            Write_Mol2(segment_fingerprints[tmp_vector[i].first].first, fout_rigid);

        } else {
            cout <<"Warning :: This type of molecule should not exist " << endl;
        }
    }


    // Clear some memory
    fout_scaffold.close();
    fout_linker.close();
    fout_sidechain.close();
    fout_rigid.close();
    tmp_vector.clear();
    segment_fingerprints.clear();

    return;

} // end AG_Conformer_Search::write_unique_fragments()



// +++++++++++++++++++++++++++++++++++++++++
// Sort functions called in write_unique_fragments()
int
frequency_sort(pair <string, int> a, pair <string, int> b)
{
    return (a.second > b.second);
}

int
fingerprint_sort(pair <string, int> a, pair <string, int> b)
{
    return (a.first > b.first);
}


// +++++++++++++++++++++++++++++++++++++++++++
// calculate molecular weight of dockmol object of a fragment
//
void AG_Conformer_Search::calc_mol_wt( DOCKMol & mol ) 
{
    float mw = 0.0;
    
    for (int i=0; i<mol.num_atoms; i++) {
        
        if (mol.atom_active_flags[i]){
            string atom = mol.atom_types[i];
            if ( atom == "H")                 
                { mw += ATOMIC_WEIGHT_H; }
        
            else if ( atom == "C.3" || atom == "C.2" || atom == "C.1" || atom == "C.ar" ||
                    atom  == "C.cat" )
                { mw += ATOMIC_WEIGHT_C; }

            else if ( atom == "N.4" || atom == "N.3" || atom == "N.2" || atom == "N.1" ||
                    atom  == "N.ar" || atom == "N.am" || atom == "N.pl3" )
                { mw += ATOMIC_WEIGHT_N; }

            else if ( atom == "O.3" || atom == "O.2" || atom == "O.co2" )
                { mw += ATOMIC_WEIGHT_O; }

            else if ( atom == "S.3" || atom == "S.2" || atom == "S.O" || atom == "S.o" ||
                    atom == "S.O2" || atom == "S.o2" )
                { mw += ATOMIC_WEIGHT_S; }

            else if ( atom == "P.3" )
                { mw += ATOMIC_WEIGHT_P; }

            else if ( atom == "F" )
                { mw += ATOMIC_WEIGHT_F; }

            else if ( atom == "Cl" )
                { mw += ATOMIC_WEIGHT_Cl; }

            else if ( atom == "Br" )
                { mw += ATOMIC_WEIGHT_Br; }

            else if ( atom == "I" )
                { mw += ATOMIC_WEIGHT_I; }

            else if ( atom == "Du" )
                { mw += 0.0; }

            else
                { cout <<"Warning: Did not recognize the atom_type " <<atom
                       <<" in DN_Build::calc_mol_wt()" <<endl; }

        }
    }
    
    mol.mol_wt = mw;

    return;

}//end calc molwt

//PAK added a method to calculate individual atoms in a vector as an input and another vector filled with 
//weight as output. If place a bool cumulative as true, it will calculate the cumaltive weights at the last
//position of the vector. 
vector <float> AG_Conformer_Search::calc_atoms_wt( vector <string> vec_atoms_calc, bool cumulative )
{

    float mw = 0.0;
    float cumulative_mw = 0.0;
    vector <float> return_vec;
    for (int i=0; i<vec_atoms_calc.size(); i++) {
        
        string atom = vec_atoms_calc[i];
        if ( atom == "H")               
            { mw += ATOMIC_WEIGHT_H; }  
            
        else if ( atom == "C.3" || atom == "C.2" || atom == "C.1" || atom == "C.ar" ||
                atom  == "C.cat" ) 
            { mw += ATOMIC_WEIGHT_C; }

        else if ( atom == "N.4" || atom == "N.3" || atom == "N.2" || atom == "N.1" ||
                atom  == "N.ar" || atom == "N.am" || atom == "N.pl3" )
            { mw += ATOMIC_WEIGHT_N; }

        else if ( atom == "O.3" || atom == "O.2" || atom == "O.co2" )
            { mw += ATOMIC_WEIGHT_O; }
            
        else if ( atom == "S.3" || atom == "S.2" || atom == "S.O" || atom == "S.o" ||
                atom == "S.O2" || atom == "S.o2" )
            { mw += ATOMIC_WEIGHT_S; } 

        else if ( atom == "P.3" )
            { mw += ATOMIC_WEIGHT_P; }
            
        else if ( atom == "F" )
            { mw += ATOMIC_WEIGHT_F; }
            
        else if ( atom == "Cl" )
            { mw += ATOMIC_WEIGHT_Cl; }
            
        else if ( atom == "Br" )
            { mw += ATOMIC_WEIGHT_Br; }
            
        else if ( atom == "I" )
            { mw += ATOMIC_WEIGHT_I; }
            
        else if ( atom == "Du" )
            { mw += 0.0; } 
        else
            { mw  = 99999999;
              cout <<"Warning: Did not recognize the atom_type " <<atom
                   <<" in AG_Conformer_Search::calc_atoms_wt()" <<endl; }
        return_vec.push_back(mw);   
        if (cumulative){
            cumulative_mw += mw;
        }
        mw = 0; 
    }
    if (cumulative){
        return_vec.push_back(cumulative_mw);
    }
    return return_vec;


}



