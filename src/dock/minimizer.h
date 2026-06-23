//
// minimizer.h — Abstract base class for DOCK minimizers
//
// All minimizers (Simplex, BOBYQA, and future ones like steepest descent)
// inherit from this class and implement the virtual algorithm interface.
// External code passes Minimizer & to enable polymorphic dispatch.
//

#ifndef MINIMIZER_H
#define MINIMIZER_H

#include <string>
#include <vector>
#include "utils.h"  // INTVec, TORSION, FLOATVec

class AMBER_TYPER;
class Base_Score;
class DOCKMol;
class Master_Score;
class Parameter_Reader;


class Minimizer {

  public:

    virtual ~Minimizer() = default;

    // Minimizer selection (read from input file)
    std::string     minimizer_type;  // "simplex" (default) or "bobyqa" (or future types)

    // Core minimization flag
    bool            minimize_ligand;

    // Rotatable bond / torsion data
    std::vector<TORSION> torsions;
    INTVec          torsion_scale_factors;
    INTVec          bond_vectors;

    // State variables
    int             ga_gen;
    int             random_seed;
    int             current_cycle;
    int             current_layer;
    int             num_layers;

    // Tethering (restrained minimization)
    bool            restrained_min;
    float           coefficient_restraint;

    // Advanced / basic minimizer parameters
    bool            advanced_min_params;
    int             max_iterations;
    int             torsion_iterations;
    int             max_cycles;
    float           initial_score_converge;
    float           score_converge;
    float           cycle_converge;
    float           trans_step_size;
    float           rot_step_size;
    float           tors_step_size;

    // Anchor minimization parameters
    bool            use_min_rigid_anchor;
    int             anchor_min_max_iterations;
    int             anchor_min_max_cycles;
    float           anchor_min_score_converge;
    float           anchor_min_cycle_converge;
    float           anchor_min_trans_step_size;
    float           anchor_min_rot_step_size;
    float           anchor_min_tors_step_size;

    // Flexible growth minimization parameters
    bool            use_min_flex_growth_ramp;
    bool            use_min_flex_growth;
    int             flex_min_max_iterations;
    int             flex_min_torsion_iterations;
    int             flex_min_max_cycles;
    float           flex_min_score_converge;
    float           flex_min_cycle_converge;
    float           flex_min_trans_step_size;
    float           flex_min_rot_step_size;
    float           flex_min_tors_step_size;

    // Flexible ramp growth parameters
    bool            use_min_ramp_flex_growth;
    int             flex_min_ramp_max_iterations;
    int             flex_min_ramp_torsion_iterations;
    int             flex_min_ramp_max_cycles;
    float           flex_min_ramp_score_converge;
    float           flex_min_ramp_cycle_converge;
    float           flex_min_ramp_trans_step_size;
    float           flex_min_ramp_rot_step_size;
    float           flex_min_ramp_tors_step_size;

    // Final minimization parameters
    bool            final_min;
    int             final_min_max_iterations;
    int             final_min_max_cycles;
    float           final_min_score_converge;
    float           final_min_cycle_converge;
    float           final_min_trans_step_size;
    float           final_min_rot_step_size;
    float           final_min_tors_step_size;
    float           final_min_rep_radius_scale;

    // Secondary minimization parameters
    bool            secondary_min_pose;
    int             secondary_min_max_iterations;
    int             secondary_min_max_cycles;
    float           secondary_min_score_converge;
    float           secondary_min_cycle_converge;
    float           secondary_min_trans_step_size;
    float           secondary_min_rot_step_size;
    float           secondary_min_tors_step_size;
    bool            secondary_advanced_min_params;

    // GA-specific flag (used by genetic algorithm code externally)
    bool            simplex_ga_flag;

    // ----- Per-torsion internal energy cache (DISABLED) -----
    //
    // BACKGROUND:
    // In the Nelder-Mead (simplex) minimizer used by DOCK, every vertex
    // operation (reflect, expand, contract, shrink) changes ALL torsion
    // angles simultaneously.  The only time a SUBSET of torsions change
    // is during the INITIAL axis-aligned vertex evaluation (one torsion
    // per vertex, N-1 of N+1 vertices).  For a 14-torsion ligand, the
    // per-torsion delta path fires for ~14 out of ~3750 eval calls.
    //
    // PROFILE-DERIVED COST:
    // compute_ligand_internal_energy consumes 17.1% of CPU time in a
    // 45-second 3CCW benchmark.  A per-torsion mask reduces pair
    // evaluations by ~0.6% (14K out of 2.38M pairs) for 14 torsions.
    // For larger ligands (50+ torsions) the proportional savings would
    // be larger, but for DT100-scale systems it was a net-neutral change
    // after fixing a regressed inlined loop.
    //
    // CONCLUSION:
    // The mask-based approach is not cost-effective for simplex-based
    // flexible docking at DT100 scale.  However, it WOULD benefit other
    // minimizer types (e.g. conjugate-gradient, L-BFGS) where each step
    // only modifies a few variables.  If a new minimizer is added, re-
    // enable this logic by:
    //   1. Uncomment the member data below
    //   2. Uncomment build_torsion_pair_indices() in eval_score()
    //   3. Uncomment the three-path case in eval_score()
    //   4. Uncomment build_torsion_pair_indices() definition
    //   5. Uncomment the cache-reset lines in id_torsions()
    //
    // std::vector<std::vector<int> > torsion_pair_indices;
    // float           cached_ref_energy = 0.0f;
    // float           cached_ref_x0 = 0.0f;
    // bool            ref_cache_valid = false;
    // int             last_num_torsions = -1;
    // void            build_torsion_pair_indices(Base_Score &, DOCKMol &);

    // Optimization: skip internal energy recomputation during rigid anchor
    // minimization.  Internal energy is invariant under rigid-body
    // transformations (only 6 DOF: rotation + translation) so caching once
    // at the start of Minimizer::minimize_rigid_anchor() is sufficient.
    bool            skip_internal_energy = false;


    // ----- Virtual algorithm interface -----

    // Read algorithm-specific parameters from the input file.
    virtual void    input_parameters(Parameter_Reader & parm,
                                     bool flexible_ligand, bool genetic_algorithm,
                                     bool denovo_design, Master_Score &) = 0;

    // Initialize per-molecule state (e.g. random seed).
    virtual void    initialize() = 0;


    // ----- Shared wrapper methods (implemented in minimizer.cpp) -----

    // High-level minimization: iterate over cycles calling do_minimize().
    void            minimize(Base_Score &, DOCKMol &, FLOATVec &,
                             int, float, int, float, float, float, float);

    // Wrapper entry points called from dock.cpp and other modules.
    void            minimize_rigid_anchor(DOCKMol &, Master_Score &);
    void            minimize_flexible_growth(DOCKMol &, Master_Score &, INTVec &);
    void            minimize_flexible_ramp_growth(DOCKMol &, Master_Score &,
                                                  INTVec &, int, int);
    void            minimize_final_pose(DOCKMol &, Master_Score &, AMBER_TYPER &);
    void            minimize_pose_final_min(DOCKMol &, Master_Score &);
    void            secondary_minimize_pose(DOCKMol &, Master_Score &);


    // ----- Shared utility methods (implemented in minimizer.cpp) -----

    void            id_torsions(DOCKMol &, FLOATVec &);
    float           calc_active_rmsd2(DOCKMol &, DOCKMol &);
    void            vector_to_dockmol(DOCKMol &, FLOATVec &);


  protected:

    // ----- Algorithm hook (each subclass overrides this) -----

    // Perform one cycle of the minimizer's core algorithm.
    // Returns the best score found.
    virtual float   do_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                int, float, float, float, float) = 0;

    // ----- Shared evaluation helpers (implemented in minimizer.cpp) -----

    // Evaluate the scoring function at a given vertex.
    bool            eval_score(Base_Score &, DOCKMol &, DOCKMol &,
                               FLOATVec &, float, float, float);

    // Scale a normalised vertex vector by the step sizes.
    void            scale_vector(FLOATVec &, FLOATVec &, float, float, float);

};

#endif  // MINIMIZER_H
