//
// conjugate_gradient.h — Conjugate Gradient minimizer for DOCK6
//
// Uses finite-difference gradients with Polak-Ribière conjugate directions.
// At each iteration:
//   1. Compute gradient via forward differences (n+1 evaluations)
//   2. Compute conjugate direction using Polak-Ribière formula
//   3. Line search along conjugate direction
//   4. Reset direction to steepest descent every n iterations
//
// Inherits from Minimizer and follows the same interface as
// Simplex_Minimizer, BOBYQA_Minimizer, and Steepest_Descent_Minimizer.
//

#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#include <string>
#include <vector>
#include "minimizer.h"

class           Conjugate_Gradient_Minimizer : public Minimizer {

  public:

    // Algorithm parameters
    float           fd_step;           // finite-difference step size
    float           line_search_alpha; // initial line search step
    float           line_search_tau;   // backtracking factor
    int             max_line_search;   // max backtracking iterations

    void            input_parameters(Parameter_Reader & parm,
                                     bool flexible_ligand, bool genetic_algorithm,
                                     bool denovo_design, Master_Score &) override;
    void            initialize() override;

  protected:

    float           do_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                int, float, float, float, float) override;

  private:

    int             n;            // number of variables

    // Compute gradient via forward finite differences
    void            compute_gradient(Base_Score &, DOCKMol &, DOCKMol &,
                                     DOCKMol &, FLOATVec &, FLOATVec &,
                                     float, float, float, float);
};

#endif  // CONJUGATE_GRADIENT_H
