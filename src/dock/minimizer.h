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
