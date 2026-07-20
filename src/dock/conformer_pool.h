/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Conformer pool — batch multiple simplex minimizations into one GPU dispatch.

Replaces per-conformer minimize_flexible_growth() with a pool of independent
GPU slots.  Each slot tracks one conformer's simplex state (iteration, phase,
vertices, scores).  The pool packs ready candidates from all active slots into
a single GPU dispatch, then distributes the returned scores to each slot's
decision tree.  Fast-converging conformers vacate their slot; new ones fill in.

Not lockstep.  Each slot advances independently — one might be in REFLECT while
another is in SHRINK.  Only candidates ready for GPU scoring are packed; all
decision logic remains on CPU.

The pool calls dock_gpu_batch_score_with_ie() directly.  No GPU kernel changes
needed.  The pool is pure CPU orchestration, portable to any GPU backend that
implements the score_dock_gpu.h API.
*/

#ifndef CONFORMER_POOL_H
#define CONFORMER_POOL_H

#include <vector>
#include "minimizer.h"
#include "score_dock_gpu.h"

class Base_Score;
class DOCKMol;


/* ------------------------------------------------------------------ */
/*  Per-slot phase enum                                               */
/* ------------------------------------------------------------------ */

/* Each active slot cycles through these phases:

       INIT     → score N+1 initial simplex vertices (one GPU dispatch)
       REFLECT  → pack 4 speculative candidates (reflect, expand, contract x2)
       SHRINK   → rescore N vertices (N+1 minus best vertex ilo)
       CONVERGED → done, awaiting poll()

   Phase transitions (all CPU, inside evaluate_slot()):
       INIT     → REFLECT  (after all vertices scored, centroid built)
       REFLECT  → REFLECT  (iteration continues: next reflect/expand/contract)
       REFLECT  → SHRINK   (Nelder-Mead shrink triggered)
       REFLECT  → CONVERGED (score convergence or max_iterations hit)
       SHRINK   → REFLECT  (shrink complete, next reflect)
       SHRINK   → CONVERGED (converged after shrink)
*/
enum class SlotPhase {
    INIT,
    REFLECT,
    SHRINK,
    CONVERGED
};


/* ------------------------------------------------------------------ */
/*  SimplexSlot — per-conformer state                                 */
/* ------------------------------------------------------------------ */

struct SimplexSlot {
    int            id;                 // slot index (0 .. capacity-1)
    SlotPhase      phase;

    // Simplex vertices and scores (same layout as simplex.cpp do_minimize())
    float**        p;                  // p[0..N][0..DOF-1] — N+1 vertex arrays, each length size
    float*         y;                  // y[0..N] — scores
    int            size;               // N = degrees of freedom; num vertices = N+1
    int            iteration;
    bool           converged;

    // Vertex rankings (set by evaluate_slot, used by pack_slot)
    int            ihi;                // index of highest (worst)  score
    int            inhi;               // index of 2nd-highest (2nd-worst) score
    int            ilo;                // index of lowest  (best)  score

    // Working buffers (pre-allocated per slot)
    FLOATVec       centroid;           // pbar — dimension = size
    FLOATVec       reflect_v;          // pr  — dimension = size

    // Per-conformer parameters (from Minimizer)
    float          trans_step_size;
    float          rot_step_size;
    float          tors_step_size;
    float          score_converge;
    int            max_iterations;

    // Cycle support (matches Minimizer::minimize() cycle loop)
    int            current_cycle;
    int            max_cycles;
    float          cycle_converge;

    // Best-across-cycles tracking: save best result so final convergence
    // returns the best cycle's result, not just the last cycle's.
    bool           has_best;
    float          best_score;
    FLOATVec       best_vertex;


    // Molecule state for dof→xyz
    // m_mol points to the caller's DOCKMol (updated in place on convergence).
    // m_refMol is a pre-minimization snapshot (read-only basis for copy_crds).
    // m_tmpMol is scratch for vector_to_dockmol before each scoring call.
    DOCKMol*       m_mol;
    DOCKMol*       m_refMol;
    DOCKMol*       m_tmpMol;

    // Restraint energy (used when restrained_min is active)
    bool           restrained;
    float          coefficient_restraint;
    DOCKMol*       m_rmsd_ref;



    // Caller data (e.g. index into exp_seeds)
    void*          user_data;
};


/* ------------------------------------------------------------------ */
/*  ConformerPool — batch-oriented simplex minimization                */
/* ------------------------------------------------------------------ */

class ConformerPool {
public:
    /* Construct pool with given capacity.
       batch_max: max concurrent slots (from dock_gpu_recommended_batch_size()).
       minimizer: shared Minimizer for vector_to_dockmol / scale_vector / copy_crds.
       use_gpu:   true → pool dispatches GPU work; false → step() is a no-op. */
    ConformerPool(int batch_max, Minimizer* minimizer, bool use_gpu);

    ~ConformerPool();

    /* Add a new conformer to the pool.
       Returns slot id on success, -1 if pool is full.
       The pool stores a pointer to the caller's DOCKMol (mol) and updates
       it in-place on convergence.  mol must remain alive until poll(). */
    int add(DOCKMol* mol,
            const FLOATVec& initial_vertex,
            float trans_step_size, float rot_step_size,
            float tors_step_size, float score_converge,
            int max_iterations,
            int max_cycles = 1,
            float cycle_converge = 1.0f,
            bool restrained = false,
            float coefficient_restraint = 0.0f,
            DOCKMol* rmsd_ref = nullptr,
            void* user_data = nullptr);

    /* Run one pool cycle:
       1. Pack ready candidates from all active slots into m_xyz_buffer.
       2. Submit GPU dispatch via dock_gpu_batch_score_with_ie().
       3. Distribute scores and run each slot's decision tree.
       4. Check convergence.
       Returns the number of newly converged conformers (call poll() for them). */
    int step();

    /* Collect converged conformers since last call.
       The caller's DOCKMol has been updated to the minimized pose. */
    std::vector<SimplexSlot*> poll();

    /* Number of active (non-CONVERGED) slots. */
    int active_count() const;

    /* True if no active slots and no pending GPU work. */
    bool idle() const;

    /* Max capacity (fixed at construction). */
    int capacity() const { return m_batch_max; }

    /* GPU active flag. */
    bool gpu_active() const { return m_use_gpu; }

private:
    int                     m_batch_max;
    Minimizer*              m_minimizer;        // shared for dof→xyz
    bool                    m_use_gpu;
    int                     m_num_atoms;   // ligand atom count, set on first add()

    std::vector<SimplexSlot>  m_slots;
    std::vector<SimplexSlot*> m_converged;       // converged slots pending poll()

    // GPU dispatch buffers (pre-allocated at construction).
    // Sized for worst-case: all slots in SHRINK simultaneously.
    //   max_candidates = m_batch_max * DOF_MAX
    //   xyz:  max_candidates * GPU_MAX_ATOMS * 3 * sizeof(float)
    //   scores: max_candidates * sizeof(float)
    float*                  m_xyz_buffer;
    float*                  m_score_buffer;
    int*                    m_active_flags;  // cached ligand active-atom flags
    int                     m_dispatch_capacity;  // max candidates per dispatch

    // Internal helpers
    int  pack_slot(SimplexSlot& slot, int offset, int capacity);
    void pack_vertex(SimplexSlot& slot, const float* vertex, int idx);
    void evaluate_slot(SimplexSlot& slot, const float* scores);
    bool fill_slot_from_mol(SimplexSlot& slot, int slot_idx);
    bool check_convergence(SimplexSlot& slot);
};

#endif  // CONFORMER_POOL_H
