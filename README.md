### DOCK 6.13.1-flurgles
Forked from [docking-org/dock6](https://github.com/docking-org/dock6) around mid June 2026.
---

## Energy Score Minimization Changes

### Class Refactoring and New Minimizers
- **Minimizer base class** — abstract class makes adding new minimizer types simpler 
- **Conjugate gradient minimizer** — standard Hestenes–Stiefel conjugate gradient method.
- **Steepest descent minimizer** — Cauchy gradient-descent method with Armijo-type backtracking line search.

### BOBYQA Minimizer

- **Derivative-free optimizer** — Bound Optimization BY Quadratic Approximation (BOBYQA) constructs a quadratic model of the objective function from 2n+1 interpolation points and solves a trust-region subproblem at each iteration. No analytical gradients required.
- **adaptive restart** — `bobyqa_restarts_per_torsion` (default 5) scales the number of restarts by rot bond count. Without DOF scaling, BOBYQA converges too greedily for flexible ligands (>12 bonds) and cannot escape poor local minima.
- **Lanczos eigenvalue estimation** — computes eigenvalue estimates of the quadratic model Hessian via Lanczos iteration, enabling curvature diagnostics without building the full n×n matrix.
- **Hessian estimation strategy** — `lanczos` (eigenvalue estimates, slower), `cg` (conjugate gradient approximation, faster), or `off` (disable curvature estimation entirely).
- **Jacobi PCG solver** — Jacobi-preconditioned conjugate gradient solver for the trust-region subproblem, improving convergence for ill-conditioned models.
- **noise-aware trust region management** — dynamic trust-region radius adjustment that detects stagnation due to model noise and triggers restarts with a fresh interpolation set.

Powell, M. J. D. The BOBYQA Algorithm for Bound Constrained Optimization Without Derivatives. Technical Report NA2009/06, Department of Applied Mathematics and Theoretical Physics, University of Cambridge, 2009.

Powell, M. J. D. The NEWUOA Software for Unconstrained Optimization Without Derivatives. In *Large-Scale Nonlinear Optimization*; Di Pillo, G., Roma, M., Eds.; Springer: New York, 2006; pp 255–297.

Powell, M. J. D. On the Use of Quadratic Models in Unconstrained Minimization Without Derivatives. *Optim. Methods Softw.* **2004**, *19* (3–4), 399–411.

### Simplex Minimizer
- **Gao & Han 2012 adaptive coefficients** — `simplex_adaptive yes` scales expansion (γ), contraction (ρ), and shrink (σ) by DOF (rot bonds+6). The standard fixed coefficients (γ=2, ρ=½, σ=½) correspond to n=2 in the adaptive scheme. Adaptive Nelder–Mead samples more for more flexible ligands (slower)
- **dim_aware sigmoid blend** — `simplex_adaptive dim_aware` smoothly transitions from fixed (n=2 behavior) to full adaptive coefficients using a configurable crossover (`simplex_adaptive_crossover`, default 17 rot bonds).

Gao, F.; Han, L. Implementing the Nelder–Mead Simplex Algorithm with Adaptive Parameters. *Comput. Optim. Appl.* **2012**, *51* (1), 259–277.

### Internal Energy
- **Soft-core Lennard-Jones** (`internal_energy_soft_delta`) — softened repulsion at short range
- **5Å repulsive VDW cutoff** — limits non-bonded pair evaluation for internal energy (speedup)
- **Cache internal energy** — precomputed and reused during rigid anchor minimization (speedup)

## GPU Acceleration (Apple/Metal)
- **GPU-accelerated grid generation** — src/grid/score_grid_gpu*
- **Atom-parallel texture-based grid score** — Replace CPU interpolation with GPU trilinear filtering primitive
- **GPU simplex minimizer** — full Nelder-Mead iteration running on the GPU with persistent threadgroup dispatch
- **SIMD-parallel internal energy** — fast IE evaluation on GPU with pairlists
- **Scale with GPU core count** — queries IOKit GPU core count to set optimal batch size
 - **GPU-side ConformerPool** — ConformerPool class manages GPU Simplex minimizer slot state without CPU round trips

## RMSD Symmetry
- **Hungarian RMSD optimizations** — O(N³) Hungarian algorithm is slow (speedup)
- **Short circuit to std rmsd** — check if molecules have no symmetry `mol_has_symmetry()` (speedup)
- **Weisfeiler–Leman graph isomorphism RMSD** — color refinement using WL graph isomorphism test
- **Weisfeiler–Leman per-layer color caching** — avoids O(N²) recomputation during flex growth (speedup)
- **Pruning clustering rmsd type** — Pick RMSD algo used for clustering (hungarian min_rmsd wl std) (slower but improves sampling?)

Weisfeiler, B. Yu.; Leman, A. A. The Reduction of a Graph to Canonical Form and the Algebra Which Appears Therein. *Nauchno-Tekh. Inf.* **1968**, *2* (9), 12–16. [English translation: https://iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf]

## Code Quality & Memory Safety
- **Variable length arrays → `std::vector`** - in `dock.cpp` growth loop
- **`sort_top_X_mol` bookkeeping** — `list_mol_bool[min_ind]` moved outside the inner comparison loop
- **`HDB_Mol::clear_molecule()`** — `name[0] = '\0'` reuse instead of unconditional reallocation
- **Container-overflow** — `ranked_poses[0]` guarded with `!empty()` check
- **Segfault fix** — `torsion_scale_factors.resize()` called after `id_torsions()`

### Build Infrastructure
- **Apple Silicon config templates** — `install/clang` (CPU-only) and `install/clang.gpu` (CPU+Metal) with `-O3 -ffast-math` and no `-flto` on GPU builds

---

Detailed GPU architecture and benchmark results are documented in [`src/dock/GPU_DOCK.md`](src/dock/GPU_DOCK.md).

---
