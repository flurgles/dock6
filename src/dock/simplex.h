//
#ifndef SIMPLEX_H
#define SIMPLEX_H 

#include <vector>
#include "minimizer.h"

class           Simplex_Minimizer : public Minimizer {

  public:

    // All data members are inherited from Minimizer.
    // Only algorithm-specific methods are declared here.

    void            input_parameters(Parameter_Reader & parm,
                                     bool flexible_ligand, bool genetic_algorithm, 
                                     bool denovo_design, Master_Score &) override;
    void            initialize() override;

  protected:

    float           do_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                int, float, float, float, float) override;

};

#endif  // SIMPLEX_H
