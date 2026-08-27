# BOBYQA Implementation Code Review — `bobyqa.cpp`

**Reviewer**: pi-agent (non-generating model)
**Date**: 2026-06-17  
**File**: `src/dock/bobyqa.cpp` (1,634 lines) + `src/dock/bobyqa.h` (130 lines)  
**Scope**: Algorithm correctness, memory safety, performance, maintainability

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [Algorithmic Errors](#2-algorithmic-errors)
3. [Memory Safety & Lifetime Issues](#3-memory-safety--lifetime-issues)
4. [Performance Issues](#4-performance-issues)
5. [Code Quality & Maintainability](#5-code-quality--maintainability)
6. [Potential Crashes & Undefined Behavior](#6-potential-crashes--undefined-behavior)
7. [Minor Issues](#7-minor-issues)
8. [Summary Severity Table](#8-summary-severity-table)

---

## 1. Critical Bugs

### 1.1 `xopt` Assigned Twice — Second Write Destroys Fallback Result

**Location**: `do_minimize()`, lines ~230–231 in the current file

```cpp
xopt = best_vertex;     // ← correct: stores best_vertex after fallback
⋮
xopt = vertex;          // ← BUG: immediately overwrites with original vertex!
fopt = fvals[0];
```

The code correctly implements three fallback strategies (original point → interpolation points → random perturbations), finds the best vertex among them, and stores it in `best_vertex`. Then it immediately overwrites `xopt` back to the original `vertex` — discarding all fallback work. The three fallback strategies are therefore **completely ineffective**.

**Impact**: High. If the original starting point fails evaluation (returns PENALTY_SCORE), the fallback might find a valid point but it gets overwritten with the original (failed) vertex. The minimizer then starts from a bad position.

**Fix**: Remove the second `xopt = vertex` assignment, or conditionally assign only if `!start_ok`.

### 1.2 `fopt` Never Updated After Successful Fallback

**Location**: `do_minimize()`, after the fallback block

When `initial_perturb_attempts > 0` and the original point fails, `best_vertex` stores a valid vertex with `best_score`. But `fopt` is always set to `fvals[0]` — which was the `PENALTY_SCORE` (1.0e6) if the original point failed. The model is then built around a penalized starting point, corrupting all gradient and Hessian estimates.

```cpp
fvals[0] = best_score;  // correct: updates fvals[0]
xopt = best_vertex;     // correct
⋮
xopt = vertex;          // BUG: overwrites
fopt = fvals[0];        // BUG: reads fvals[0] AFTER xopt was overwritten
```

**Impact**: High. On systems where the initial scoring fails (e.g., problematic ligand conformations), the minimizer starts with a massively incorrect function value and needs many extra iterations to recover — if it ever does.

### 1.3 `diagnostics.iterations` Never Incremented

**Location**: `ConvergenceDiagnostics` struct usage in `do_minimize()`

The `diagnostics.iterations` and `diagnostics.function_evals` fields are declared and intended to track convergence, but **neither is ever incremented or set**. The diagnostic checks at the end of the main loop compare `diagnostics.iterations` against `max_iterations`, but since it's always 0, the check `diagnostics.iterations >= max_iterations` is always false. The record_diagnostics function is also commented out.

```cpp
// At end of loop:
if (delta <= rho_end_actual && diagnostics.iterations > 5) {  // always false
    // record_diagnostics("delta_converged");
} else if (norm_g < score_converge && delta <= rho_end_actual) {
    // record_diagnostics("gradient_converged");
} else if (diagnostics.iterations >= max_iterations) {  // always false
    // record_diagnostics("max_iterations");
}
```

**Impact**: Medium. Doesn't affect correctness of the minimization itself, but renders the diagnostic subsystem completely non-functional — no termination reason is ever recorded.

---

## 2. Algorithmic Errors

### 2.1 Cauchy Point Computation Does Not Match Powell's BOBYQA

**Location**: TRSBOX section in `do_minimize()`

```cpp
// Current implementation:
float tau = (norm_g * norm_g * norm_g) / (delta * gHg);
if (tau > 1.0f) tau = 1.0f;
for (i = 0; i < n; i++) s[i] = -tau * delta * g[i] / norm_g;
```

This formula has a dimension mismatch. Let's trace the units:
- `norm_g` has units of [g]
- `gHg` has units of [g²/H⁻¹] (energy/length²)
- So `tau` has units of [g³] / [length * g²/H⁻¹] = [g * H / length]
- Then `s[i] = tau * delta * g[i] / norm_g` has units of [... * length * g / g] = [...]

The units don't cancel to give a pure length (which is what `s[i]` should be). 

**Correct Cauchy point for trust-region** (per Powell 2009 and Nocedal & Wright):
```
s_c^C = - (norm_g² / gHg) * g    (if gHg > 0)
s_c^U = - (delta / norm_g) * g    (if gHg ≤ 0, unbounded direction)
```
Then clamp: if ‖s_c‖ > delta, scale by delta/‖s_c‖.

The current code effectively double-scales by delta: once in the denominator of `tau` and again in `s[i] = -tau * delta * ...`.

**Impact**: High. The Cauchy point is the foundation of the trust-region dogleg method. An incorrect Cauchy point means the trust region subproblem is solved incorrectly at every single iteration. This likely contributes to BOBYQA's observed poor performance vs. simplex.

### 2.2 Dogleg Method Is a No-Op

**Location**: TRSBOX, after Newton step computation

```cpp
if (!cg_converged || norm_newt > 100.0f * delta || norm_newt < 1.0e-20f) {
    // Fall back to Cauchy step (already in s)  ← no-op
} else if (norm_newt <= delta) {
    s = s_newt;                                    ← correct
} else {
    // Truncated Newton: combine Cauchy + Newton    ← DOGLEG SHOULD GO HERE
    // But for simplicity, use Cauchy since Newton is outside trust region
    // Simple approach: just use the Cauchy point which is already set.  ← no-op
}
```

When the Newton step exceeds the trust-region boundary (the most common case in early iterations), Powell's BOBYQA uses the **dogleg** method: a convex combination of the Cauchy point and the Newton point that intersects the trust-region boundary. The current code just falls through and uses the Cauchy point unchanged. This is a significant deviation from the BOBYQA algorithm.

**Impact**: High. Without the dogleg, the algorithm cannot take well-scaled steps that mix steepest descent with approximate Newton directions. This severely degrades convergence, especially for ill-conditioned problems.

### 2.3 Gradient Update Uses Arbitrary Blending Coefficient

**Location**: Section 2g, diagonal Hessian update

```cpp
g[i] = 0.5f * (g[i] + fd - 0.5f * Hdiag[i] * diff);
```

This blends the old gradient with a secant-based estimate using a fixed 0.5 factor. In proper quasi-Newton methods (BFGS, SR1), the gradient is updated by the secant equation:
```
g_new = g_old + H * s
```
No arbitrary blending. This heuristic blending means the gradient no longer satisfies the secant condition, so the model becomes inconsistent.

**Impact**: Medium. The gradient gradually diverges from the true function landscape, making the model unreliable. This is partially mitigated by the frequent re-evaluation in later iterations.

### 2.4 `norm_newt` Read Before Assignment in CG Path

**Location**: CG solver completion

```cpp
cerr << "DEBUG: CG done, converged=" << cg_converged << " n_newt=" << norm_newt << endl;
norm_newt = 0.0f;                                    // ← set AFTER debug output
for (i = 0; i < n; i++) norm_newt += s_newt[i] * s_newt[i];
norm_newt = sqrt(norm_newt);
```

The debug output prints `norm_newt` before it's computed. This is harmless in release builds (where debug output is presumably disabled), but causes misleading debugging.

---

## 3. Memory Safety & Lifetime Issues

### 3.1 Repeated `std::vector` Allocations in Hot Loop

**Location**: Main iteration loop (section 2a)

Every iteration allocates:
- `FLOATVec s(n)` — new allocation
- `FLOATVec s_newt(n)` — new allocation
- `FLOATVec x_trial(n)` — new allocation
- Inside CG: `FLOATVec r(n), p(n), Hp(n)` — 3 more allocations

For a 30-DOF system running 1000 iterations, that's **~6000 heap allocations** for temporary vectors alone. Each allocation calls `operator new`, which may involve OS-level memory management.

**Impact**: Performance degradation on high-torsion systems. The gap grows with problem size.

### 3.2 `H[i][j]` Bounds Check Breaks but Doesn't Protect

**Location**: Multiple places where full quadratic Hessian is accessed

```cpp
if (j >= (int)H[i].size()) { cerr << "ERROR: H[" << i << "].size=" << H[i].size() << " j=" << j << endl; break; }
```

The `break` statement only exits the `j` loop. Execution continues with `Hp[i]` partially computed (some terms missing), `sHs` partially computed, etc. This propagates corrupted values through the rest of the iteration.

In a release build without cerr, the bounds are never checked at all — an out-of-bounds access on `H[i][j]` would be silent undefined behavior or a segfault.

### 3.3 `s_step` Used with Stale Data

**Location**: `update_model_full()`

`s_step` is only assigned when `ratio > 0.0f && use_full_quadratic`. But `update_model_full` is called *after* the "Accept step" block, and accesses `s_step[j]` unconditionally. If the step was rejected (`ratio ≤ 0`), `update_model_full` is not called — correct. But if the step was accepted and `use_full_quadratic` is false, the diagonal update path is used (correct), and `s_step` retains data from the last full-quadratic step — which could be many iterations ago. On the next accepted full-quadratic step, `update_model_full` uses a stale `s_step`.

---

## 4. Performance Issues

### 4.1 Verbose Debug Output in Production Code

**Location**: Throughout `do_minimize()`

The code contains extensive `cerr << "DEBUG: ..."` statements:
- Every iteration prints: iter number, delta, fopt
- Every TRSBOX: "Starting TRSBOX"
- Every CG iteration: "CG iter N / M"
- Every function evaluation: "eval_score start", "copy_crds done", etc.
- Section markers: "section 2d", "section 2e", etc.

For 1000 iterations × ~20 debug lines = 20,000 cerr calls per minimization. This is a **massive I/O overhead** and will dominate runtime for small molecules.

**Impact**: BOBYQA appears slower than simplex partially because of these writes. cerr output is line-buffered by default, so each call may also cause a syscall.

### 4.2 `#pragma GCC optimize("O0")` on `multi_start_minimize`

```cpp
#pragma GCC optimize("O0")
float BOBYQA_Minimizer::multi_start_minimize(...)
```

Forces `-O0` (no optimization) on the entire multi_start function. Since multi-start runs `do_minimize` N+1 times, the outer wrapper is cheap — but the function also contains a PRELIM evaluation loop (2n+1 evals) that runs at O0 speed.

**Impact**: Low (wrapper is lightweight), but the pragma is a debugging artifact that should be removed.

### 4.3 `__attribute__((noinline))` on `run_prelim_and_collect_points`

Prevents the compiler from inlining this function. Since it's called only from `multi_start_minimize`, inlining would be beneficial.

### 4.4 Side-Effect File Write for Compiler Barrier

```cpp
{
    ofstream prelim_debug("/tmp/bobyqa_prelim_debug.txt");
    prelim_debug << "PRELIM points collected: " << prelim_points.size() << endl;
}
```

The comment says "observable side effect compiler can't optimize away". This is a very fragile technique:
1. Fails silently if `/tmp` is unwritable
2. Contends for a hard-coded path (parallel runs will overwrite each other's data)
3. Real compiler barriers (`asm volatile`, `std::atomic_signal_fence`) exist

### 4.5 Repeated `restrained_min` Checks

`if (restrained_min)` is checked at every single score evaluation — potentially hundreds of times. Since it's a class member that doesn't change during minimization, a single function pointer or wrapper at the start would eliminate the branch overhead.

---

## 5. Code Quality & Maintainability

### 5.1 Dead Code — Active But Unused Members

The following member variables are set but never read meaningfully:

| Variable | Set in | Read in | Status |
|---|---|---|---|
| `noise_level` | `initialize()` (`= 0.0f`) | Nowhere | Dead |
| `noise_threshold` | `initialize()` (`= 0.1f`) | Nowhere (commented-out code) | Dead |
| `fnew_val` | Nowhere | Nowhere | Dead (only declared) |

### 5.2 Dead Code — Commented-Out Large Blocks

The file contains ~250 lines of commented-out code across 4 functions:
- `estimate_noise_level()` — ~25 lines
- `perform_adaptive_restart()` — ~95 lines  
- `record_diagnostics()` — ~30 lines
- Stagnation detection + adaptive restart in main loop — ~15 lines

Plus dead member variables commented out in the header:
- `noise_window`, `ratio_history`, `stagnation_count`, `restart_count`

This is ~25% of the file as dead code, making maintenance harder and reading more confusing.

### 5.3 Function Parameter Ordering Inconsistency

```cpp
// build_full_model:
(score, ref_mol, tmp_mol, rmsd_ref, best_mol, ...)

// rescue:
(score, mol, ref_mol, tmp_mol, rmsd_ref, ...)

// multi_start_minimize:
(score, mol, vertex, ...)

// run_prelim_and_collect_points:
(score, ref_mol, tmp_mol, rmsd_ref, mol, ...)
```

The order of `DOCKMol` references (`ref_mol`, `tmp_mol`, `rmsd_ref`, `best_mol`, `mol`) is inconsistent across functions. This is an accident waiting to happen — passing arguments in the wrong order would compile, produce incorrect results, and be very hard to debug.

**Suggestion**: Define a `MinimizerContext` struct or consistent parameter ordering convention.

### 5.4 `using namespace std;` at File Scope

While common in older C++ codebases, this pollutes the global namespace and can cause subtle name collisions (e.g., `std::min` vs custom `min`, `size_t` ambiguity). For a 1600+ line file with heavy STL usage, explicit `std::` prefixes would be safer.

### 5.5 No Validation of User-Specified `npt`

If the user sets `bobyqa_npt = 2`, the code silently bumps it to `n+2`. But if the user sets `npt = n+2` (correct minimum for BOBYQA), `n_axis = min(n, (np-1)/2) = min(n, (n+1)/2)`, which means only ~n/2 axes are sampled instead of all n. The remaining DOFs get zero gradient and unit Hessian, which is suboptimal.

There should be a warning when user-specified `npt` leads to incomplete axis sampling.

### 5.6 Integer Types

- `n` is `int` throughout
- `nptmax` is `int`
- But `FLOATVec::size()` returns `size_t`
- Numerous comparisons like `i < n` where `i` is `int` and `n` is `int` — fine for small DOFs
- But `i >= (int)H[i].size()` casts are necessary because `H[i].size()` is `size_t`

For a molecular docking code, DOF count rarely exceeds a few hundred, so this is low-risk.

### 5.7 `exit(0)` on Parameter Errors

The code calls `exit(0)` ~30+ times for parameter validation errors. This:
1. Prevents any cleanup (stack objects are not destructed — though `exit` does call static destructors)
2. Makes the code untestable (any invalid parameter kills the test process)
3. Inconsistent — some errors use `exit(0)`, others print and continue

Standard practice in production C++ is `throw` or return an error code.

### 5.8 `cerr` vs `cout` Inconsistency

Parameter validation errors go to `cout`, but debug output goes to `cerr`. For normal operation, users see errors mixed with output. If they redirect stdout, they miss parameter errors.

---

## 6. Potential Crashes & Undefined Behavior

### 6.1 `RAND_MAX` Platform Dependency

```cpp
float perturb = ((float)rand() / RAND_MAX - 0.5f) * 0.2f * rho_beg_actual;
```

`RAND_MAX` is implementation-defined (typically 2³¹−1 on glibc, 2¹⁵−1 on MSVC). On platforms where it's small, the granularity of `rand()` may cause inadequate perturbation sampling. Also, `rand()` is not thread-safe, though DOCK6 doesn't appear to be multi-threaded.

### 6.2 `H` Reallocation on Every PRELIM + RESCUE + Model Build

Each call to `build_full_model`, `rescue`, and the initial PRELIM resizes `H` (when `use_full_quadratic`). These resize operations invalidate references/iterators, but since no iterators are held across these calls, the risk is low.

### 6.3 Division by Zero in Gradient Update

```cpp
if (fabs(diff) > 1.0e-10f) {
    ...
    float h_new = (fd - g[i]) / diff;
```

If `diff` is exactly 0.0 (step had zero component in this DOF), this is skipped. But the check is `> 1e-10`, not `!= 0`, so there's a guard. Low risk.

### 6.4 `sqrt(norm_newt)` of Negative Value

```cpp
norm_newt += s_newt[i] * s_newt[i];  // always non-negative
norm_newt = sqrt(norm_newt);
```

This is safe — the sum of squares is always non-negative. But using `+=` without explicit zero initialization (remnant from old code) could cause issues if the variable was previously set.

---

## 7. Minor Issues

### 7.1 `volatile` Hack for Compiler Barrier

```cpp
volatile int dummy_force_call = 1; (void)dummy_force_call;
```

This does nothing in practice — `volatile` on a local stack variable doesn't prevent inlining or function elimination. Proper compiler barriers or `__attribute__((noinline))` (already present) are the right approaches.

### 7.2 Magic Numbers

- `1.0e6f` — PENALTY_SCORE (appears 4+ times)
- `0.1f` — eta1, score_converge default
- `0.7f` — eta2
- `2.0f` — gamma_up
- `0.5f` — gamma_down
- `1.0e-12f` — zero/denominator threshold (appears ~10×)
- `1.0e-20f` — gradient zero threshold
- `1.0e-10f` — diff/step significance threshold
- `0.2f` — perturbation scale
- `0.9f` / `0.1f` — blending factors in `update_model_full`
- `300` — max corner evaluations in `build_full_model`
- `1000.0f` — rescue fallback penalty

These should be named constants.

### 7.3 `rescue` Called But Not Fully Used

The rescue function is called and re-evaluates 2n+1 points, but `use_rescue` defaults to `"yes"` in parameter reading. It's enabled by default, adding 2n+1 extra function evaluations every 50 iterations — whether needed or not.

### 7.4 `n_axis_rescue` May Be Less Than `n`

If `npt < 2n+1`, the rescue rebuild only covers `n_axis_rescue < n` dimensions. The remaining DOFs get `g[i]=0, Hdiag[i]=1`, losing model information.

### 7.5 Hard-Coded File Path

`/tmp/bobyqa_prelim_debug.txt` is hard-coded. Multiple concurrent runs will corrupt each other's data.

---

## 8. Severity Summary

| # | Issue | Severity | Category |
|---|---|---|---|
| 1.1 | `xopt` double-assignment discards fallback | **Critical** | Logic bug |
| 1.2 | `fopt` never updated after fallback | **Critical** | Logic bug |
| 2.1 | Cauchy point formula incorrect | **High** | Algorithm bug |
| 2.2 | Dogleg method is no-op | **High** | Algorithm bug |
| 1.3 | `diagnostics.iterations` never incremented | Medium | Logic bug |
| 2.3 | Gradient update blending heuristic | Medium | Algorithm |
| 2.4 | `norm_newt` read before assignment | Low | Debug output |
| 3.1 | Repeated vector allocations in hot loop | Medium | Performance |
| 3.2 | Bounds check breaks but doesn't protect | Medium | Safety |
| 3.3 | `s_step` used with stale data | Medium | Algorithm |
| 4.1 | Excessive debug I/O in production | **High** | Performance |
| 4.2 | `#pragma GCC optimize("O0")` artifact | Low | Code quality |
| 4.4 | File write as compiler barrier | Low | Fragility |
| 5.1 | 4 dead member variables | Low | Dead code |
| 5.2 | ~250 lines commented out | Medium | Dead code |
| 5.3 | Inconsistent parameter ordering | Medium | Maintainability |
| 5.7 | `exit(0)` on parameter errors | Low | Robustness |
| 6.1 | `rand()` platform dependency | Low | Portability |

### Recommendations (Priority Order)

1. **Fix the `xopt` double-assignment** (#1.1) — single line deletion, largest impact
2. **Fix Cauchy point formula** (#2.1) — 3 lines, fixes trust-region subproblem
3. **Add dogleg implementation** (#2.2) — ~15 lines, completes the TRSBOX algorithm
4. **Remove debug `cerr` statements** (#4.1) — guard with `#ifndef NDEBUG` or remove entirely
5. **Remove dead code** (#5.2) — ~250 lines of commented-out functions
6. **Increment `diagnostics.iterations`** (#1.3) — single line, enables diagnostic infrastructure
7. **Consistent DOCKMol parameter ordering** (#5.3) — refactor for safety
8. **Pre-allocate temporary vectors** (#3.1) — reuse across iterations
9. **Fix gradient update** (#2.3) — use proper secant-based formula
10. **Remove `#pragma GCC optimize("O0")`** — debugging artifact
