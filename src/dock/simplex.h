//
#ifndef SIMPLEX_H
#define SIMPLEX_H 

#include <vector>
#include "minimizer.h"

/* C1 Batch queue API (removed in P2 — ConformerPool manages GPU dispatch) */


class           Simplex_Minimizer : public Minimizer {

  public:

    // All data members are inherited from Minimizer.
    // Only algorithm-specific methods are declared here.

  protected:

    // Adaptive Nelder-Mead parameters (Gao & Han 2012).  When false (default),
    // the minimizer uses the classical fixed coefficients (alpha=1.0,
    // gamma=2.0, rho=0.5, sigma=0.5) and results are bit-identical to
    // prior DOCK versions.  When true, the expansion/contraction/shrink
    // coefficients are scaled by the problem dimension n to prevent
    // simplex collapse in high-dimensional problems (15+ DOF), which is
    // common in flexible docking.
    //
    //   gamma = 1.0 + 2.0/n   (expansion)
    //   rho   = 0.75 - 0.5/n  (contraction)
    //   sigma = 1.0  - 1.0/n  (shrink)
    //   alpha = 1.0           (reflection, NOT adapted)
    //
    // Reference:
    //   Gao, F. & Han, L. (2012).  Implementing the Nelder-Mead simplex
    //   algorithm with adaptive parameters.  Computational Optimization
    //   and Applications, 51(1), 259--277.
    //   DOI: 10.1007/s10589-010-9329-3
    bool            simplex_adaptive = false;

  public:

    void            input_parameters(Parameter_Reader & parm,
                                     bool flexible_ligand, bool genetic_algorithm, 
                                     bool denovo_design, Master_Score &) override;
    void            initialize() override;

  protected:

    float           do_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                int, float, float, float, float) override;

};

#endif  // SIMPLEX_H
