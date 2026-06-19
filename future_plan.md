# DOCK6 BOBYQA — Future Plan

Unimplemented ideas and research directions for closing the BOBYQA vs Simplex performance gap.

---

## Short Term: Test What We Already Have

These features are **already implemented** but need systematic evaluation on the full DT100 set.

| Feature | Parameter | Status | Priority |
|---------|-----------|--------|----------|
| Full quadratic model | `bobyqa_use_full_quadratic=yes` | ✅ Implemented | HIGH |
| Multi-start | `bobyqa_use_multi_start=yes` | ✅ Implemented (default: no) | HIGH |
| RESCUE | `bobyqa_use_rescue=yes` | ✅ Implemented (default: yes) | MEDIUM |
| Parameter sweeps | rho_beg, rho_end, npt, score_converge | ✅ Parameters exist | HIGH |

**Action**: Run full DT100 benchmark with:
1. Default BOBYQA (diagonal Hessian)
2. Full quadratic model
3. Multi-start (with conservative perturbation)
4. Full quadratic + multi-start combination
5. Systematic rho_beg/rho_end/npt grid search on systems where gap > 3

---

## Medium Term: Algorithm Improvements

These require **new code** but build on existing infrastructure.

### 1. Parameter Scaling for DOF Types
**Problem**: Hessian approximations conflate Å (translation), radians (rotation), kcal/mol (torsion energy).
**Solution**: Read typical ranges per DOF type; scale variables internally so Hessian operates in isotropic space.
**Files**: `bobyqa.h/cpp` — add scaling in `initialize()`, apply in `vector_to_dockmol()` and `bobyqa_score()`.

### 2. Block-Diagonal Hessian
**Problem**: Full n×n Hessian is O(n²) storage/compute; diagonal ignores all coupling.
**Solution**: 
- Group DOFs: translation (3), rotation (3), torsions (n)
- Full Hessian within each block, diagonal between blocks
- Captures intra-block coupling (most important) at ~O(n) cost
**Files**: `build_full_model()`, `update_model_full()`, TRSBOX solver

### 3. Smarter Model Updates
- Skip update when predicted/actual reduction ratio is poor (η < 0.1)
- Iterative Hessian refinement within update step
- Damped SR1/BFGS updates to maintain positive definiteness

### 4. Better Convergence Diagnostics
- Track predicted/actual reduction ratio over sliding window
- Estimate Hessian eigenvalues (Lanczos on quadratic model)
- Stagnation detection: no improvement in M iterations → trigger recovery

### 5. Advanced TRS Solvers
- Generalized trust region (More-Sorensen)
- 2D subspace minimization
- Lambda-finding using λ trend as model trustworthiness diagnostic

---

## Long Term: Research Directions

### 1. Limited-Memory BFGS (L-BFGS)
Approximate full Hessian from gradient differences over iterations, without storing n×n matrix.

### 2. ML Surrogate Models
Train cheap neural net / GP on visited points to approximate score landscape; use to guide BOBYQA sampling.

### 3. Multi-Fidelity Optimization
- Cheap internal energy / approximate score for pre-screening
- Full grid scoring only for promising candidates
- BOBYQA operates on multi-fidelity model

### 4. Subspace Identification
Use gradient history + Hessian eigenvalue estimates to identify low-dimensional active subspaces; restrict search to those.

### 5. Hybrid Minimizer Scheduling
- Phase 1: Simplex / steepest descent for robust global convergence
- Phase 2: BOBYQA for fine-grained local refinement
- Switch based on convergence metrics

---

## Implementation Notes

All medium-term experiments can be prototyped by modifying:
- `build_full_model()` / `update_model_full()` — Hessian construction & update
- TRSBOX section of `do_minimize()` — trust-region solver experiments  
- `input_parameters()` — new parameter scaffolding
- `initialize()` — parameter scaling setup

The `bobyqa_use_full_quadratic` toggle already gates full vs diagonal Hessian, so A/B testing infrastructure is in place.

---

## Current Benchmark Target

**Goal**: BOBYQA scores within 0.5 units of simplex on ≥90% of DT100 systems (currently 13/99 competitive, 86/99 simplex wins by >0.5).

**Baseline** (2026-06-17):
- Simplex better by >0.5: 85 systems
- Within 0.5: 13 systems  
- BOBYQA better by >0.5: 1 system (1KV2)
- Catastrophic failures: 1 failures: 0 (fixed)

**Target**: Flip the ratio so BOBYQA is competitive or better on the majority.