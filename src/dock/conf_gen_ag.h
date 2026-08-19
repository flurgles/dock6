#ifndef CONF_GEN_AG_H
#define CONF_GEN_AG_H 

#include <string>
#include <vector>
#include <map>

#include "dockmol.h"
#include "utils.h"  // INTVec
class ConformerPool;  // batch NM pool (conformer_pool.h) — refs only

#define ATOMIC_WEIGHT_H 1.00794
#define ATOMIC_WEIGHT_C 12.011
#define ATOMIC_WEIGHT_N 14.00647
#define ATOMIC_WEIGHT_O 15.9994
#define ATOMIC_WEIGHT_S 32.066
#define ATOMIC_WEIGHT_P 30.973762
#define ATOMIC_WEIGHT_F 18.9984032
#define ATOMIC_WEIGHT_Cl 35.4527
#define ATOMIC_WEIGHT_Br 79.904
#define ATOMIC_WEIGHT_I 126.90447

class Bump_Filter;
class Master_Score;
class Parameter_Reader;
class Minimizer;

/* ------------------------------------------------------------------ */
/*  Per-pose simplex RNG seed keys                                    */
/* ------------------------------------------------------------------ */

/* Deterministic key mixing the pose's position in the growth tree.
   The sequential path and the GPU pool both call seed_key() with the
   same indices, so every pose draws the same simplex random numbers
   regardless of processing order.  Anchor poses: (serial, anchor, pose).
   Growth poses: (serial, anchor, layer, segment, pose).  Must stay
   identical across conf_gen_ag.cpp and dock.cpp. */
inline unsigned int
seed_key(unsigned int a, unsigned int b, unsigned int c,
         unsigned int d, unsigned int e)
{
    unsigned int x = a * 0x9E3779B1u ^ b * 0x85EBCA77u ^
                     c * 0xC2B2AE3Du ^ d * 0x27D4EB2Fu ^
                     e * 0x165667B1u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}


/********************************************************************/
class           SEGMENT {

  public:

    SEGMENT() {
        num_hvy_atoms = 0;
    };
    // //////////////////////////////////////////////////
    INTVec          atoms;      // list of atoms in the segment
    INTVec          bonds;      // list of bonds completely contained in the
                                // segment
    int             num_hvy_atoms;      // number of atoms in the segment
    INTVec          neighbors;  // neighboring segments
    INTVec          neighbor_bonds;     // rot bonds btwn segments
    INTVec          neighbor_atoms;     // test for segment at a time conf gen

    friend int      operator<(SEGMENT s1, SEGMENT s2) {
        return (s1.atoms.size() > s2.atoms.size());
    };
};


/********************************************************************/
class           LAYER_SEGMENT {

  public:

    LAYER_SEGMENT() {
        num_hvy_atoms = 0;
        rot_bond = -1;
        origin_segment = -1;
    };
    // //////////////////////////////////////////////////
    INTVec          atoms;      // list of atoms in the segment
    INTVec          bonds;      // list of bonds completely contained in the
                                // segment
    int             num_hvy_atoms;      // number of atoms in the segment
    int             rot_bond;   // rot bond to this segment from previous layer
    int             origin_segment;     // which segment this one is derived
                                        // from

    friend int      operator<(LAYER_SEGMENT s1, LAYER_SEGMENT s2) {
        return (s1.atoms.size() > s2.atoms.size());
    };
};


/********************************************************************/
class           LAYER {

  public:

    // //////////////////////////////////////////////////
    INTVec segments;            // List of segments in this layer
    int             num_segments;       // Number of segments in this layer

};


/********************************************************************/
class           ROT_BOND {

  public:

    // //////////////////////////////////////////////////
    int             atom1;      // first atom bound to atom 2 (to define angle)
    int             atom2;      // first atom of rotatable bond
    int             atom3;      // second atom of rotatable bond
    int             atom4;      // first atom bound to atom 3 (to define angle)
    int             seg1;       // Segment containing atom #2
    int             seg2;       // Segment containing atom #3
    int             bond_num;   // bond number in Mol
    float           initial_angle;      // initial angle from database
    float           current_angle;      // current angle

};


/********************************************************************/
class           CONFORMER {

  public:

    // General info
    float           score;      // score of current partial conf
    int             layer_num;  // current layer
    bool            used;       // flag to delete conformer once its used
    int             anchor_num;  // each conformer originated from anchor position.
    int             conformer_num; // independent number;
    int             parent_num;   // number;
    std::string     header;       // text with energy score, rmsd etc. for branch_*.mol2
  
    DOCKMol         structure;  // structure of conformer

};


/********************************************************************/
/* Windowed growth: per-ligand round state for the batch scheduler
   (dock.cpp phase-3b).  One VSGrowState per (job, set) drives the
   same growth algorithm as grow_periphery, but its exp_seeds are
   added to a SHARED ConformerPool whose slots interleave NM
   iterations of many ligands in one GPU dispatch. */
class VSGrowState {

  public:

    int             lig_idx = -1;   // GPU LUT row for this ligand
    int             route   = 0;    // scheduler row index (user_data tag hi bits)

    // Lazy-prep support: rows are stubbed at window start; the full
    // grow_win_init runs when the scheduler first picks the row, so only
    // the in-flight rows' anchor copies + seed structures are live
    // (the eager prep of every row held all ~90MB-per-row states at once).
    int             job_idx = -1;   // owning VSWindowJob
    int             set_idx = -1;   // anchor set within the job
    bool            prepped = false; // grow_win_init ran for this row

    int             i = 1;          // current layer (growth starts at 1)
    int             l = 0;          // current segment within layer
    int             num_layers = 1;

    int             k_add = 0;      // first-pass resume cursor into exp_seeds
    int             inflight = 0;   // slots of this round still in the pool
    bool            adding = true;  // round adds still in progress
    bool            drive_done = false; // torsion drive ran for this round
    bool            drain_done = false; // round fully drained + pruned
    bool            done = false;   // growth complete
    bool            ie_prune = false;   // internal-energy pruning active
    bool            cpu_min_round = false; // poses refined by CPU minimizer
                                          // (keep CPU scores; skip GPU re-score)

    // Ligand state parked here while the shared AG_Conformer_Search is
    // serving another row (grow_win_swap).
    std::vector<SCOREMol>        anchor_positions;
    std::vector<DOCKMol>         anchor_confs;
    std::vector<CONFORMER>       conf_anchors;
    int                          current_anchor = 0;
    int                          dock_mol_serial = 0;
    std::vector<LAYER>           layers;
    std::vector<LAYER_SEGMENT>   layer_segments;
    std::vector<ROT_BOND>        bond_list;
    INTVec                       bond_tors_vectors;
    INTVec                       atom_seg_ids;
    INTVec                       bond_seg_ids;
    std::vector<bool>            assigned_atoms;
    std::vector<int>             next_nbrs;
    std::vector<int>             tmp_nextnbrs;
    std::string                  atom_in_anchor;
    DOCKMol                      orig;
    std::vector<SEGMENT>         orig_segments;
    bool                         ligand_mol_symmetric = false;

    std::vector<CONFORMER> seeds;
    std::vector<CONFORMER> exp_seeds;
    std::vector<CONFORMER> b4min_seeds;
    std::vector<int>    cb_ok;      // 0 clash fail, 1 bump fail, 2 passed
    std::vector<char>   g2_valid;   // GPU2 in-grid per conformer
    std::vector<float>  g2_grid;    // GPU2 grid-only scores (VS LUT)
    std::vector<float>  g2_comb;    // GPU2 grid+IE scores (VS LUT)
    bool                gpu2_ok = false;
    bool                gpu2_pending = false; // async screen in flight (stream2)
    /* Persistent staging for the async GPU2 screen: must outlive the
       stream-ordered batch until dock_gpu_batch_score_sync2(). */
    std::vector<float>  g2_xyz;
    std::vector<int>    g2_pose_lig;
    /* Per-set IE snapshot: primary_score is shared across the window's
       rows and initialize_internal_energy() rebuilds nb_int/ie_vdwA per
       row — the LUT refresh must use THIS row's pair list. */
    std::vector<int>    nb_flat;
    std::vector<float>  ie_vdwA_snap;

    std::vector<CONFORMER> all_gen_seeds;      // growth tree (print_growth_tree)
    std::vector<CONFORMER> all_gen_b4min_seeds;

    /* LBAL round-phase debug timings (DOCK_LBAL_DEBUG) */
    long    dbg_prep_us = 0;
    long    dbg_drive_us = 0;
    long    dbg_add_us = 0;
    long    dbg_prune_us = 0;
    int     dbg_nseeds = 0;
    int     dbg_nexp = 0;

    std::vector<SCOREMol> pruned_confs;        // final output for next_conformer

    int             confs_pruned_bad_score = 0;
    int             confs_pruned_clash_overlap = 0;
    int             confs_pruned_outside_grid = 0;
    int             confs_pruned_bump_filter = 0;
    int             confs_pruned_clustered = 0;
};


/********************************************************************/
class           BRANCH {

  public:

    int             beatm,
                    bmatm,
                    bnhvy,
                    bnhyd,
                    confnum,
                    bi,
                    iconf;
    float           solvat,
                    apol;
    INTVec          level_values,
                    level_counts;

};


/********************************************************************/
class           ATOM_INFO {

  public:

    int             level,
                    vdwtype,
                    flagat,
                    lcolor;
    float           charge,
                    polsolv,
                    apolsolv;
    std::string     type;

    int             branch_num;
    INTVec          level_confs;

};


/********************************************************************/
class           AG_Conformer_Search {

  public:

    // growth tree confromers 
    int             count_conf_num;                //counter - trent balius Dec 13, 2008

    //bool flexible_ligand;                        // no
    bool            verbose;
    int             anchor_size;                   // 10

    // TEB ADD 2010-01-23
    bool            user_specified_anchor;
    std::string     atom_in_anchor;
    bool            limit_max_anchors;
    int             max_anchor_num;

    int             num_anchor_poses;
    float           anchor_score_cutoff;           // energy cutoff value for next layer

    int             num_growth_poses;
    //bool            growth_cutoff;
    float           growth_score_scaling_factor;
    float           growth_score_cutoff;           // energy cutoff value for growth
    float           growth_score_cutoff_begin;

    bool            cluster;
    int             pruning_clustering_cutoff;     // number of confs/layer

    bool            use_clash_penalty;             // flag to use clash penalty or not
    float           clash_penalty;                 // atom clash threshold value

    // move to Master_Conformer_Search 
    bool            use_internal_energy;           // int nrg function superseded by func in base_score
    int             ie_att_exp;
    int             ie_rep_exp;
    float           ie_diel;
    float           ie_soft_delta;
    float          *ie_vdwA;
    float          *ie_vdwB; 
    float           internal_energy_cutoff;        //BCF internal energy cutoff

    bool            print_growth_tree;             // sudipto & trent 30-01-09 

    DOCKMol         orig;                          // original molecule struct

    std::vector < SEGMENT > orig_segments;         // 
    std::vector < LAYER_SEGMENT > layer_segments;  // 
    std::vector < LAYER > layers;                  // 
    std::vector < INTPair > anchors;               // pair list of anchor segments and sizes
    std::vector < SCOREMol > anchor_positions;     // 
    std::vector < DOCKMol > anchor_confs;          // 
    std::vector < ROT_BOND > bond_list;            // 
    std::vector < SCOREMol > pruned_confs;         //

    // list of anchors for current molecule after minimization and pruning
    std::vector < CONFORMER > conf_anchors;        // sudipto & trent - Dec 08, 2008
    std::vector < bool > assigned_atoms;           // flags to track when
                                                   // an atom is assigned
                                                   // to a layer
    bool            last_conformer;                // 

    int             current_anchor;                // current anchor (anchors.size() -> 0)

    // Serial index of the molecule being docked (library order, 0-based).
    // The sequential main loop and the GPU VS driver both set this so the
    // per-pose simplex RNG seeds (seed_key) match across the two paths.
    int             dock_mol_serial = 0;

    bool            return_anchor_value;           // used when no flex ligand is
                                                   // requested    bool            return_periph_value;           // used when no flex ligand is
                                                   // requested

    INTVec          atom_seg_ids;                  // id of which segment each atom
                                                   // belongs to
    INTVec          bond_seg_ids;                  // id of which segment each bond
                                                   // belongs to (flex bonds assigned -1)
    INTVec          bond_tors_vectors;             // id of "start" atom for each bond WRT 
                                                   // layers- for flexible minimization
                                                   // during growth
    //PAK vects
    std::vector < int > next_nbrs;
    std::vector < int > tmp_nextnbrs;
    std::vector < std::string > tmp_vec_atom_strings;
    std::vector < float > tmp_vec_atom_wt;
    // Functions in AG_Conformer_Search //////////////////////////////////////

    AG_Conformer_Search();
    virtual ~ AG_Conformer_Search();
    void            initialize();       // 
    void            initialize_internal_energy_parms(bool uie, int rep_exp, int att_exp, float diel, float soft_delta, float iec); 
    void            initialize_internal_energy_null(bool uie); 
                    // get values from Master_Conformer_Search called in input_parameters of Master
    void            input_parameters(Parameter_Reader & parm);  // 
    void            prepare_molecule(DOCKMol &);        // 
    void            identify_rigid_segments(DOCKMol &); // 
    void            extend_segments(int, int, DOCKMol &);       // 
    void            id_anchor_segments();       // 
    bool            next_anchor(DOCKMol &);     // 
    void            setup_growth_anchor(int anchor_idx); // VS batch replay: rebuild layers for one anchor
    void            extend_layers(int, int, int);       // 
    bool            submit_anchor_orientation(DOCKMol &, bool); // 
    // RMSD type for pruning clustering: "std" (standard layer-weighted heavy-atom),
    // "hungarian" (symmetry-corrected with Hungarian algorithm, layer-weighted),
    // or "min" (one-way minimum RMSD, layer-weighted).
    // RMSD type for pruning clustering: "std" (standard layer-weighted heavy-atom),
    // "hungarian" (symmetry-corrected with Hungarian algorithm, layer-weighted),
    // or "min" (one-way minimum RMSD, layer-weighted).
    std::string     pruning_cluster_rmsd_type;
    // RMSD type for final pose clustering (same values):
    std::string     final_pose_cluster_rmsd_type;
    // Cached flag: true if the ligand has exploitable graph symmetry
    // (computed once in prepare_molecule via WL color refinement).
    // When false, Hungarian/min pruning fall back to standard RMSD.
    bool            ligand_mol_symmetric;
    float           calc_layer_rmsd(CONFORMER &, CONFORMER &);  // standard layer-weighted RMSD
    float           calc_layer_rmsd_hungarian(CONFORMER &, CONFORMER &);  // layer-weighted symmetry-corrected Hungarian RMSD
    float           calc_layer_rmsd_min(CONFORMER &, CONFORMER &);  // layer-weighted one-way min RMSD
    float           calc_layer_rmsd_weisfeiler(CONFORMER &, CONFORMER &,
                                                  const std::vector<int> & colors,
                                                  const double * weights);  // layer-weighted WL symmetry-corrected RMSD
    float           calc_active_rmsd(CONFORMER &, CONFORMER &);  // 2008-11-17 trent balius add  
    void            grow_periphery(Master_Score &, Minimizer &, Bump_Filter &,
                                   bool anchors_preminimized = false);
    void            conf_header(CONFORMER &, std::string, Master_Score &);
    void            segment_torsion_drive(CONFORMER &, int, std::vector < CONFORMER > &, int);
    void            activate_layer_segment(DOCKMol &, int, int);        // 
    void            print_branch(std::vector < CONFORMER > &, std::vector < CONFORMER > &,const CONFORMER &,Master_Score & ); // trent balius 2008-12-03
    void            print_conformer(CONFORMER &); // trent balius 2008-12-08
    void            reset_active_lists(DOCKMol &);      // 
    bool            segment_clash_check(DOCKMol &, int, int);   // 
    bool            next_conformer(DOCKMol &);  // 
    static int      conformer_less_than(CONFORMER a, CONFORMER b) {
        return a.score < b.score;
    };
    bool            atom_in_anchor_segments(SEGMENT); // TEB ADD 2010-01-23

    void            print_atom_in_anchor_segments(SEGMENT); // TEB ADD 2019-06-17

    // For generating fragment libraries and torsion environments for de novo growth
    bool            write_fragment_libraries;       // if true, libraries are written, most of dock.cpp is skipped
    std::string     fragment_library_prefix;        // prefix for libraries (four files total)
    int             fragment_library_freq_cutoff;   // frequency cutoff for writing fragments to library
    std::string     fragment_library_sort_method;       // method for sorting libraries, freq or fingerprint
    bool            fragment_library_trans_origin;  // if true, trans frags to origin before writing libs

    std::map < std::string, std::pair <DOCKMol, int> > segment_fingerprints; // hash for writing fragments
    std::map < std::string, int >                      torsions_map;         // hash for writing tor envs
    std::map < std::string, std::string >              torsions_map_ref;     // hash for writing tor envs


    //JDB - hash for writing fragments w/ associated attachment frequencies
    std::map < std::string, std::map < std::string, int> > fragment_binding_pairs;
    std::map < std::string, //f0
                std::map < std::string, //f1
                std::map < std::string, //t0
                std::map < std::string, int > > > > frags_with_half_tors; //t1,freq

    int             global_frag_index;

    void            count_fragments(DOCKMol &);     // writes fragments to libraries depending on # of attachment points
    void            bickel_count_fragments(DOCKMol &); // JDB
    void            activate_fragment(DOCKMol &, int);  // activates atoms/bonds of a segment + neighbors
    void            bickel_write_unique_fragments();    //JDB 
    void            write_unique_fragments();           // rewrites unique fragment libraries
    void            calc_mol_wt(DOCKMol &);
    std::vector <float>                               calc_atoms_wt(std::vector<std::string>, bool); 

    // ----- Windowed (batch-scheduler) growth, GPU path -----
    // Park the ligand-dependent members of this object into / restore
    // them from a VSGrowState (the scheduler serves many ligands through
    // one AG_Conformer_Search).  park() exchanges (this left empty);
    // restore() deep-copies (state kept in g).
    void            grow_win_park(VSGrowState &);
    void            grow_win_restore(VSGrowState &);
    // Seed build from anchor_positions (anchors_preminimized path) +
    // IE/LUT setup.  Must run with this object's members holding the
    // ligand's state (anchor_positions, dock_mol_serial loaded).
    void            grow_win_init(VSGrowState &, Master_Score &,
                                  Minimizer &, Bump_Filter &);
    // Resumable first pass for the current round: torsion drive + clash/
    // bump + pool.add with backpressure.  Returns when the pool is full
    // or the round is fully added (g.adding == false).
    void            grow_win_prep(VSGrowState &, ConformerPool &,
                                  Master_Score &, Minimizer &, Bump_Filter &);
    // Second pass (GPU2 VS scoring) + prune + seed rebuild; advances the
    // (i, l) cursor; builds pruned_confs when growth ends.  With the async
    // screen, finish splits into an enqueue pass (gpu2_pending set) and a
    // consume pass (results already synced by dock.cpp) that prunes.
    void            grow_win_finish(VSGrowState &, Master_Score &,
                                    Minimizer &, Bump_Filter &);
    // Prune + seed rebuild (the consume half of grow_win_finish).
    void            grow_win_prune(VSGrowState &, Master_Score &,
                                   Bump_Filter &);
    // GPU2 dispatch used by grow_win_finish (kept separate so dock.cpp
    // can run it without holding a swapped-in state).  Async on stream2.
    void            grow_win_score_round(VSGrowState &, Master_Score &);


};


/********************************************************************/
// Sort Functions
int       frequency_sort(std::pair <std::string, int>, std::pair <std::string, int> );
int       fingerprint_sort(std::pair <std::string, int>, std::pair <std::string, int> );

#endif
