//
#ifndef BOBYQA_H
#define BOBYQA_H

#include <string>
#include <vector>
#include <deque>
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
    std::string     hessian_mode;      // "default" (diagonal), "block_diag", or "full_quad"
    bool            use_multi_start;     // enable multi-start with random restarts
    int             multi_start_restarts; // number of random restarts for multi-start
    bool            use_adaptive_restart; // enable adaptive restart on stagnation
    int             max_restarts;         // maximum number of adaptive restarts
    float           restart_delta_scale;  // factor to scale rho_beg on restart
    int             stagnation_window;    // iterations without fopt improvement to trigger restart
    float           stagnation_tol;       // relative tolerance for stagnation (default 0.001)
    float           stagnation_abs_tol;   // absolute tolerance for stagnation (default 0.1)
    float           restart_min_delta_ratio; // only restart once delta/rho_beg <= this (default 0.05)
    float           restart_perturbation; // random perturbation scale for restart center (default 0.05)
    bool            restart_from_best;    // restart from best point seen (yes) or current xopt (no)

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
    std::vector<std::vector<float>> H;  // Hessian matrix (when hessian_mode != "default")
    float            delta;       // trust region radius
    int              kopt;        // index of best point in interpolation set
    int              nptmax;      // maximum number of interpolation points

    // State for model update
    FLOATVec         s_step;      // last accepted step vector
    float            fopt_before; // function value at xopt before last step
    float            fnew_val;    // function value at last accepted step
    std::deque<float> fopt_history; // sliding window of best scores for stall detection
    std::deque<float> ratio_history; // sliding window of predicted/actual reduction ratios
    static const int max_ratio_window = 10;  // keep last 10 ratios for avg

    // Global optimization: adaptive restart state
    int             restart_count;       // number of restarts performed
    float           noise_level;         // estimated noise in objective function (placeholder)
    float           noise_threshold;     // threshold for noise classification (placeholder)

    // Convergence diagnostics (minimal for compilation)
    struct ConvergenceDiagnostics {
        int iterations = 0;
        int function_evals = 0;
        float final_delta = 0.0f;
        float final_gradient_norm = 0.0f;
        float avg_ratio = 0.0f;
        float noise_level = 0.0f;
        float hessian_min_eigenvalue = 0.0f;   // Lanczos estimate of smallest eigenvalue
        float hessian_max_eigenvalue = 0.0f;   // Lanczos estimate of largest eigenvalue
        int restarts = 0;
        int rescue_calls = 0;
        bool converged_normally = false;
        std::string termination_reason = "";
    } diagnostics;

    // Lanczos eigenvalue estimation on the quadratic model Hessian
    // Runs k iterations (default min(5,n)) to estimate extreme eigenvalues.
    // For the diagonal model (hessian_mode=="default"), eigenvalues are just Hdiag entries.
    void            estimate_hessian_eigenvalues(int k = 0,
                                                  float *eig_min_out = nullptr,
                                                  float *eig_max_out = nullptr);

    // Adaptive restart on stagnation
    void            perform_adaptive_restart(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
                                              float, float, float, float);
    // RESCUE: recover from degenerate interpolation set
    void            rescue(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
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
    // Global optimization helpers (commented out - TODO: implement properly)
    // float           estimate_noise_level();
    // void            perform_adaptive_restart(Base_Score &, DOCKMol &, DOCKMol &, DOCKMol &, DOCKMol &,
    //                                           float, float, float);
    // void            record_diagnostics(const std::string & reason);
};

#endif  // BOBYQA_H
