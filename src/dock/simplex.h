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

    // Adaptive Nelder-Mead parameters (Gao & Han 2012).
    //
    // simplex_mode controls which coefficient set is used:
    //   0 = no        — classical fixed coefficients (alpha=1.0, gamma=2.0,
    //                   rho=0.5, sigma=0.5).  Bit-identical to prior DOCK.
    //   1 = yes       — full adaptive: coefficients scaled by the problem
    //                   dimension n to prevent simplex collapse in
    //                   high-dimensional problems (15+ DOF).
    //   2 = dim_aware — sigmoid-blended transition between fixed (low n)
    //                   and adaptive (high n), centered at
    //                   simplex_crossover + 6 DOF.
    //
    // Adaptive formulas (Gao & Han 2012):
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
    int             simplex_mode = 0;      // 0=no, 1=yes, 2=dim_aware
    int             simplex_crossover = 17; // bond count for 50/50 blend
                                         // (dim_aware mode only)

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
