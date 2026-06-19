//
// steepest_descent.h — Steepest Descent minimizer for DOCK6
//
// Uses finite-difference gradients for derivative-free optimization.
// At each iteration:
//   1. Compute gradient via forward differences (n+1 evaluations)
//   2. Backtracking line search along -g
//   3. Accept step if function decreases
//
// Inherits from Minimizer and follows the same interface as
// Simplex_Minimizer and BOBYQA_Minimizer.
//

#ifndef STEEPEST_DESCENT_H
#define STEEPEST_DESCENT_H

#include <string>
#include <vector>
#include "minimizer.h"

class           Steepest_Descent_Minimizer : public Minimizer {

  public:

    // Algorithm parameters
    float           fd_step;       // finite-difference step size
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

#endif  // STEEPEST_DESCENT_H
