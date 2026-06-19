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
    int             restarts_per_torsion;  // multi_start_restarts = 5 + n_tors * this (default 5)
                                          // Controls how many additional restarts each torsional
                                          // DOF adds. Higher values give more thorough exploration
                                          // of high-DOF conformational space at the cost of runtime.
    int             improv_window;        // sliding window of fopt values for improvement detection
                                          // When the range of fopt over this window is below the threshold
                                          // (improv_tol, with 0.01 absolute floor), the minimizer has stalled.
    float           improv_tol;           // improvement detection tolerance (default 0.001 = 0.1% relative)
                                          // Effective threshold = max(0.01, |fopt| * improv_tol)
                                          // Dual relative+absolute check in a single param.
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
    int             stagnation_count;    // consecutive iterations with ratio < eta1 (triggers early restart)
    int             noise_window;        // number of recent ratios for noise estimation
    float           noise_level;         // estimated noise (std dev of ratio)
    float           noise_threshold;     // threshold for noise classification

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
    // Noise estimation: standard deviation of recent ratio values
    float           estimate_noise_level();
};

#endif  // BOBYQA_H
