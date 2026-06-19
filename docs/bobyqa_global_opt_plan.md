# BOBYQA Global Optimization Implementation Plan

**Target**: Implement global optimization features from py_bobyqa into DOCK6's bobyqa.cpp  
**Priority**: #8 in TODO list (highest expected impact)  
**Approach**: Add noise-aware trust region, adaptive restarts, and systematic multi-start seeding  

---

## Current State Analysis (from bobyqa.cpp)

### Existing Features
1. **Multi-start** (`multi_start_minimize`): Runs `do_minimize` multiple times with:
   - Run 0: Original starting point
   - Runs 1..N: PRELIM interpolation points (if available) or small torsional perturbations
2. **RESCUE** (`rescue`): Rebuilds interpolation set when points become degenerate
3. **Full quadratic model** (`use_full_quadratic`): Off-diagonal Hessian via SR1 updates
4. **PRELIM**: 2n+1 interpolation points on coordinate axes at radius `rho_beg`

### Missing Global Optimization Features (from py_bobyqa)
1. **Noise estimation**: Track variance of `ratio = dAct/dPred` to estimate objective noise
2. **Noise-aware trust region**: Adjust `delta` based on noise level, not just ratio
3. **Stagnation detection**: Detect when model is misled (non-smooth landscape) 
4. **Adaptive restarts**: Automatic restart with larger `delta` when stagnation detected
5. **Soft restarts**: Preserve best point and some interpolation info
6. **Systematic seeding**: Better starting point selection (not just PRELIM axis points)

---

## Step-by-Step Implementation Plan

### Phase 1: Add Noise Estimation Infrastructure

#### Step 1.1: Add Noise Tracking Member Variables (bobyqa.h)
```cpp
// Add to BOBYQA_Minimizer class private section:
float           noise_level;         // Estimated noise in objective function
int             noise_window;        // Number of iterations for noise estimation
std::vector<float> ratio_history;   // Track recent ratio values for noise estimation
float           noise_threshold;    // Threshold for "noisy" classification
int             stagnation_count;   // Consecutive iterations with poor progress
int             restart_count;      // Number of restarts performed
int             max_restarts;       // Maximum allowed restarts
bool            use_adaptive_restart; // Enable adaptive restart logic
float           restart_delta_scale;  // Factor to increase delta on restart
```

#### Step 1.2: Add Configuration Parameters (bobyqa.cpp - input_parameters)
Add these parameters to `input_parameters()`:
```cpp
// After existing multi_start parameters:
use_adaptive_restart = (parm.query_param("bobyqa_use_adaptive_restart", "no", "yes no") == "yes");
max_restarts = atoi(parm.query_param("bobyqa_max_restarts", "3").c_str());
restart_delta_scale = atof(parm.query_param("bobyqa_restart_delta_scale", "5.0").c_str());
noise_threshold = atof(parm.query_param("bobyqa_noise_threshold", "0.1").c_str());
```

#### Step 1.3: Initialize Noise Tracking (bobyqa.cpp - initialize + do_minimize)
```cpp
void BOBYQA_Minimizer::initialize() {
    srand(random_seed);
    noise_level = 0.0f;
    noise_window = 10;
    noise_threshold = 0.1f;
    stagnation_count = 0;
    restart_count = 0;
    ratio_history.clear();
}
```

---

### Phase 2: Noise-Aware Trust Region Management

#### Step 2.1: Compute Noise Level from Ratio History
```cpp
// Add method to BOBYQA_Minimizer:
float BOBYQA_Minimizer::estimate_noise_level() {
    if (ratio_history.size() < 3) return 0.0f;
    
    // Compute variance of ratio around 1.0
    float sum = 0.0f, mean = 0.0f;
    for (float r : ratio_history) mean += r;
    mean /= ratio_history.size();
    
    for (float r : ratio_history) {
        float diff = r - mean;
        sum += diff * diff;
    }
    float variance = sum / ratio_history.size();
    
    // Keep window bounded
    if (ratio_history.size() > noise_window) {
        ratio_history.erase(ratio_history.begin());
    }
    
    return sqrt(variance);  // Standard deviation of ratio
}
```

#### Step 2.2: Modify Trust Region Update Logic (in do_minimize main loop)
Replace the simple ratio-based delta update with noise-aware logic:

```cpp
// After computing ratio = dAct / dPred:
ratio_history.push_back(ratio);
noise_level = estimate_noise_level();

// Noise-aware trust region adjustment
if (ratio >= eta2 && noise_level < noise_threshold) {
    // Good step, low noise: expand aggressively
    delta = min(delta * gamma_up, rho_beg_actual * 10.0f);
} else if (ratio >= eta1 && noise_level < noise_threshold) {
    // Acceptable step, low noise: expand moderately
    delta = min(delta * 1.5f, rho_beg_actual * 10.0f);
} else if (ratio > 0.0f && noise_level >= noise_threshold) {
    // Acceptable step but HIGH NOISE: don't expand, just maintain
    delta = delta;  // Keep current delta
} else if (ratio < eta1) {
    // Poor step: contract
    delta *= gamma_down;
    if (delta < rho_end_actual) delta = rho_end_actual;
}

// Stagnation detection
bool made_progress = (ratio > eta1);
if (!made_progress) {
    stagnation_count++;
} else {
    stagnation_count = 0;
}

// Check for stagnation-triggered restart
if (use_adaptive_restart && stagnation_count >= 5 && restart_count < max_restarts) {
    perform_adaptive_restart(score, ref_mol, tmp_mol, rmsd_ref, mol,
                             trans_step_size, rot_step_size, tors_step_size);
}
```

---

### Phase 3: Adaptive Restart Mechanism

#### Step 3.1: Add perform_adaptive_restart Method
```cpp
void BOBYQA_Minimizer::perform_adaptive_restart(Base_Score & score, 
                                                 DOCKMol & mol, DOCKMol & ref_mol,
                                                 DOCKMol & tmp_mol, DOCKMol & rmsd_ref,
                                                 float trans_step_size, float rot_step_size,
                                                 float tors_step_size) {
    cout << "BOBYQA ADAPTIVE RESTART " << (restart_count + 1) 
         << ": stagnation detected, delta=" << delta 
         << ", noise=" << noise_level << endl;
    
    // Preserve best point
    FLOATVec saved_xopt = xopt;
    float saved_fopt = fopt;
    DOCKMol saved_best_mol;
    copy_molecule(saved_best_mol, mol);
    
    // Increase trust region significantly
    delta = min(delta * restart_delta_scale, rho_beg_actual * 5.0f);
    if (delta < rho_beg_actual) delta = rho_beg_actual;
    
    // Rebuild interpolation set around current best with larger radius
    // (similar to RESCUE but with larger radius and preserving best point)
    float restart_rho = delta;
    
    xpts[0] = saved_xopt;
    fvals[0] = saved_fopt;
    
    int n_axis_restart = min(n, (nptmax - 1) / 2);
    for (int i = 0; i < n_axis_restart; i++) {
        // Positive direction
        int idx_p = 1 + i;
        xpts[idx_p] = saved_xopt;
        xpts[idx_p][i] += restart_rho;
        if (eval_score(score, ref_mol, tmp_mol, xpts[idx_p],
                       trans_step_size, rot_step_size, tors_step_size)) {
            fvals[idx_p] = tmp_mol.current_score + tmp_mol.internal_energy;
            if (restrained_min) {
                fvals[idx_p] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
            }
        } else {
            fvals[idx_p] = saved_fopt + 1000.0f;
        }
        
        // Negative direction
        int idx_m = 1 + n_axis_restart + i;
        if (idx_m < nptmax) {
            xpts[idx_m] = saved_xopt;
            xpts[idx_m][i] -= restart_rho;
            if (eval_score(score, ref_mol, tmp_mol, xpts[idx_m],
                           trans_step_size, rot_step_size, tors_step_size)) {
                fvals[idx_m] = tmp_mol.current_score + tmp_mol.internal_energy;
                if (restrained_min) {
                    fvals[idx_m] += coefficient_restraint * calc_active_rmsd2(rmsd_ref, tmp_mol);
                }
            } else {
                fvals[idx_m] = saved_fopt + 1000.0f;
            }
        }
    }
    
    // Rebuild gradient and diagonal Hessian
    float inv_rho2 = 1.0f / (restart_rho * restart_rho);
    for (int i = 0; i < n_axis_restart; i++) {
        float fp = fvals[1 + i];
        float fm = fvals[1 + n_axis_restart + i];
        g[i] = (fp - fm) / (2.0f * restart_rho);
        Hdiag[i] = (fp + fm - 2.0f * saved_fopt) * inv_rho2;
        if (Hdiag[i] < 1.0e-12f) Hdiag[i] = 1.0e-12f;
    }
    for (int i = n_axis_restart; i < n; i++) {
        g[i] = 0.0f;
        Hdiag[i] = 1.0f;
    }
    
    // Rebuild full Hessian if enabled
    if (use_full_quadratic) {
        build_full_model(score, ref_mol, tmp_mol, rmsd_ref, saved_best_mol,
                         trans_step_size, rot_step_size, tors_step_size);
    }
    
    // Reset tracking
    xopt = saved_xopt;
    fopt = saved_fopt;
    copy_molecule(mol, saved_best_mol);
    stagnation_count = 0;
    restart_count++;
    ratio_history.clear();
    noise_level = 0.0f;
}
```

---

### Phase 4: Improved Multi-Start Seeding

#### Step 4.1: Systematic PRELIM Point Collection (Already Exists)
The `run_prelim_and_collect_points` method already collects PRELIM points. Enhance it to also collect:
- Points from previous restarts
- Latin hypercube samples within bounds (if bounds available)
- Points along principal directions from Hessian eigenvectors

#### Step 4.2: Enhance multi_start_minimize with Better Seeds
```cpp
// In multi_start_minimize, replace fallback perturbation with:
else {
    // Use systematic sampling: pick from best PRELIM points, 
    // then use Latin hypercube or Sobol sequence if bounds known
    // For now: use larger perturbations on ALL DOFs (not just torsional)
    const float perturb_scale = 0.5f * restart_delta_scale;  // Scale with restart aggressiveness
    FLOATVec orig_vertex = vertex;
    for (int i = 0; i < n; i++) {
        float perturb = ((float)rand() / RAND_MAX - 0.5f) * 
                        (i < 3 ? trans_step_size : (i < 6 ? rot_step_size : tors_step_size)) 
                        * perturb_scale;
        run_vertex[i] = orig_vertex[i] + perturb;
    }
}
```

#### Step 4.3: Add Cross-Run Information Sharing
```cpp
// After each run in multi_start_minimize, if a new best is found:
// Share interpolation points from the best run to seed subsequent runs
if (score_val < best_final_score) {
    // Store best run's interpolation set
    best_xpts = xpts;
    best_fvals = fvals;
    best_g = g;
    best_Hdiag = Hdiag;
    if (use_full_quadratic) best_H = H;
}

// On subsequent runs, initialize with best known interpolation set
// (partial warm-start instead of cold PRELIM)
```

---

### Phase 5: Convergence Diagnostics Integration (Links to #6)

#### Step 5.1: Add Diagnostics Collection
```cpp
// Add diagnostics struct:
struct ConvergenceDiagnostics {
    int iterations;
    int function_evals;
    float final_delta;
    float final_gradient_norm;
    float avg_ratio;
    float noise_level;
    int restarts;
    int rescue_calls;
    bool converged_normally;
    std::string termination_reason;
};

ConvergenceDiagnostics diagnostics;
```

#### Step 5.2: Populate Diagnostics Throughout
- Track in main loop
- Output at end of `do_minimize`
- Write to log/CSV for analysis

---

### Phase 6: Testing & Validation

#### Step 6.1: Add Test Configuration
Create test input file with:
```ini
minimizer_type = bobyqa
minimize_ligand = yes
bobyqa_max_iterations = 1000
bobyqa_max_cycles = 1
bobyqa_use_adaptive_restart = yes
bobyqa_max_restarts = 3
bobyqa_restart_delta_scale = 5.0
bobyqa_noise_threshold = 0.1
bobyqa_use_multi_start = yes
bobyqa_multi_start_restarts = 5
bobyqa_use_rescue = yes
bobyqa_use_full_quadratic = yes
```

#### Step 6.2: Run on DT100 Flagged Systems
Test specifically on the 45 systems where BOBYQA trails by >3.0:
```bash
# Run test script with new config
./test_bobyqa_global.sh
```

#### Step 6.3: Compare Against Baselines
- Baseline: current best (simplex_10cyc_5000iter)
- Compare: bobyqa_default, bobyqa_full_quad, bobyqa_multi_start, bobyqa_global (new)
- Expected: Global opt should close gap on high-DOF systems (16+ rot bonds)

---

## File Modification Summary

| File | Changes |
|------|---------|
| `src/dock/bobyqa.h` | Add noise tracking members, diagnostics struct, new method declarations |
| `src/dock/bobyqa.cpp` | 1. Add config params in `input_parameters`<br>2. Initialize in `initialize`<br>3. Add `estimate_noise_level()`<br>4. Modify trust region update in `do_minimize` main loop<br>5. Add `perform_adaptive_restart()`<br>6. Enhance `multi_start_minimize` seeding<br>6. Add diagnostics collection |

---

## Key Design Decisions for Weaker LLM Implementation

1. **Minimal API changes**: All new logic in private methods; public interface unchanged
2. **Configurable defaults**: New features OFF by default (`use_adaptive_restart = no`)
3. **Clear separation**: Noise estimation → Trust region adjustment → Restart trigger → Restart execution
4. **Preserve best point**: Restarts never lose the best-found solution
5. **Debug output**: Print restart events with delta/noise for verification
6. **Bounded memory**: `ratio_history` limited to `noise_window` entries

---

## Expected Impact

Based on evidence:
- Non-smooth landscape causes BOBYQA's quadratic model to mislead
- Noise-aware trust region prevents over-contraction on noisy evaluations
- Adaptive restarts escape local minima caused by model error
- Systematic multi-start with larger perturbations explores more basins
- **Target**: Close 50% of the gap on 16+ rotatable bond systems (from +4.45 to ~+2.0)

---

## Verification Checklist

- [ ] Code compiles without warnings
- [ ] `bobyqa_use_adaptive_restart = no` reproduces current behavior exactly
- [ ] Restart events printed to stdout with delta/noise values
- [ ] No memory leaks (vectors cleared properly)
- [ ] Test on 1KV2 (only system where BOBYQA currently wins) - should not regress
- [ ] Test on 1HPS/1MV9 (20 rot bonds, worst gap) - should improve
- [ ] CSV aggregation shows new `bobyqa_test_global` column