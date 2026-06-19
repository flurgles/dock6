//
#ifndef BOBYQA_H
#define BOBYQA_H

#include <string>
#include <vector>
#include "minimizer.h"

class           BOBYQA_Minimizer : public Minimizer {

  public:

    // BOBYQA-specific algorithm parameters
    float           rho_beg;      // initial trust-region radius
    float           rho_end;      // final trust-region radius
    int             npt;          // number of interpolation points
    int             initial_perturb_attempts;  // number of perturbation attempts for initial scoring (0 = disabled)

    // Configurable advanced features
    bool            use_rescue;          // enable RESCUE for degenerate interpolation sets
    bool            use_full_quadratic;  // enable full quadratic model (off-diagonal Hessian)
    bool            use_multi_start;     // enable multi-start with random restarts
    int             multi_start_restarts; // number of random restarts for multi-start

    void            input_parameters(Parameter_Reader & parm,
                                     bool flexible_ligand, bool genetic_algorithm,
                                     bool denovo_design, Master_Score &) override;
    void            initialize() override;

  protected:

    float           do_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                int, float, float, float, float) override;

  private:
    // Internal data for BOBYQA algorithm
    int             n;            // number of variables
    std::vector<FLOATVec>  xpts;  // interpolation points (each is a vector of coordinates)
    std::vector<float>     fvals; // function values at interpolation points
    FLOATVec         xopt;        // best point found so far
    float            fopt;        // function value at xopt
    FLOATVec         g;           // gradient of model at xopt
    std::vector<float>  Hdiag;    // diagonal of Hessian (approximation)
    std::vector<std::vector<float>> H;  // full Hessian matrix (when use_full_quadratic=true)
    float            delta;       // trust region radius
    int              kopt;        // index of best point in interpolation set
    int              nptmax;      // maximum number of interpolation points

    // State for model update
    FLOATVec         s_step;      // last accepted step vector
    float            fopt_before; // function value at xopt before last step
    float            fnew_val;    // function value at last accepted step

    // Global optimization: noise tracking (kept for compilation, not actively used)
    float           noise_level;         // estimated noise in objective function
    float           noise_threshold;     // threshold for noise classification
    // int             noise_window;        // window size for noise estimation
    // std::vector<float> ratio_history;   // recent ratio values for noise estimation
    // int             stagnation_count;    // consecutive iterations with poor progress
    // int             restart_count;       // number of restarts performed

    // Convergence diagnostics (minimal for compilation)
    struct ConvergenceDiagnostics {
        int iterations = 0;
        int function_evals = 0;
        float final_delta = 0.0f;
        float final_gradient_norm = 0.0f;
        float avg_ratio = 0.0f;
        float noise_level = 0.0f;
        int restarts = 0;
        int rescue_calls = 0;
        bool converged_normally = false;
        std::string termination_reason = "";
    } diagnostics;

    // RESCUE: recover from degenerate interpolation set
    void            rescue(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
                           float, float, float, float);
    // Full quadratic model helpers
    void            build_full_model(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
                                     float, float, float);
    void            update_model_full(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
                                      float, float, float, float, float);
    // Multi-start
    float           multi_start_minimize(Base_Score &, DOCKMol &, FLOATVec &,
                                         int, float, float, float, float);
    // PRELIM-based multi-start helper
    struct PrelimPoint {
        FLOATVec vertex;
        float score;
    };
    void            run_prelim_and_collect_points(Base_Score &, DOCKMol &, DOCKMol &,
                                                   DOCKMol &, DOCKMol &,
                                                   float, float, float,
                                                   std::vector<PrelimPoint> &);
    // Global optimization helpers (commented out - TODO: implement properly)
    // float           estimate_noise_level();
    // void            perform_adaptive_restart(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
    //                                           float, float, float);
    // void            record_diagnostics(const std::string & reason);
};

#endif  // BOBYQA_H
