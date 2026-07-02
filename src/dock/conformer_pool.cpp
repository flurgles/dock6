/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
ConformerPool — CPU-side orchestrator for batch simplex minimization.

Packs ready candidates from active slots into a flat GPU dispatch buffer,
calls dock_gpu_batch_score_with_ie(), distributes returned scores to each
slot's Nelder-Mead decision tree, and manages slot lifecycle (add, converge,
poll).

GPU-independent: no Metal/Vulkan/CUDA types in this file.
All GPU calls go through the score_dock_gpu.h C API.
*/

#include "conformer_pool.h"
#include "dockmol.h"

using namespace std;


/* ------------------------------------------------------------------ */
/*  Internal helpers                                                  */
/* ------------------------------------------------------------------ */

/* Build the initial N+1 simplex vertices via axis-aligned perturbation.
   Matches the initialization in simplex.cpp do_minimize() iteration 0:
       p[0] = vertex (starting point)
       p[1..N] = vertex + random uniform [-1, 1] in each dimension
*/
static void build_initial_simplex(const FLOATVec& vertex, float** p, float* y, int size)
{
    for (int i = 0; i < size; i++) {
        p[0][i] = vertex[i];
    }
    for (int i = 1; i < size + 1; i++) {
        for (int j = 0; j < size; j++) {
            p[i][j] = vertex[j] + 2.0f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        }
    }
    /* y values are filled by GPU scoring in step(), not here */
    (void)y;
}


/* ------------------------------------------------------------------ */
/*  Constructor / Destructor                                          */
/* ------------------------------------------------------------------ */

ConformerPool::ConformerPool(int batch_max, Minimizer* minimizer, bool use_gpu)
    : m_batch_max(batch_max)
    , m_minimizer(minimizer)
    , m_use_gpu(use_gpu)
    , m_num_atoms(0)
    , m_dof(0)
    , m_xyz_buffer(nullptr)
    , m_score_buffer(nullptr)
    , m_dispatch_capacity(0)
{
    m_slots.reserve(batch_max);
}


ConformerPool::~ConformerPool()
{
    for (auto& slot : m_slots) {
        if (slot.p) {
            for (int i = 0; i < slot.size + 1; i++) {
                delete[] slot.p[i];
            }
            delete[] slot.p;
        }
        delete[] slot.y;
        delete slot.m_refMol;
        delete slot.m_tmpMol;
    }
    delete[] m_xyz_buffer;
    delete[] m_score_buffer;
}


/* ------------------------------------------------------------------ */
/*  add — Enqueue a conformer into the pool                           */
/* ------------------------------------------------------------------ */

int
ConformerPool::add(DOCKMol* mol, Base_Score* score,
                   const FLOATVec& initial_vertex,
                   float trans_step_size, float rot_step_size,
                   float tors_step_size, float score_converge,
                   int max_iterations, bool skip_internal_energy,
                   bool restrained, float coefficient_restraint,
                   DOCKMol* rmsd_ref, void* user_data)
{
    if ((int)m_slots.size() >= m_batch_max) return -1;

    int idx = (int)m_slots.size();
    int size = (int)initial_vertex.size();

    /* Lazy buffer allocation on first add().  All conformers in one
       growth layer share the same DOF and atom count, so the first
       conformer's dimensions define buffer sizing for the pool. */
    if (!m_xyz_buffer) {
        m_num_atoms = mol->num_atoms;
        m_dof = size;
        int max_cand = m_batch_max * max(4, size + 1);
        m_xyz_buffer      = new float[max_cand * m_num_atoms * 3];
        m_score_buffer    = new float[max_cand];
        m_dispatch_capacity = max_cand;
    }

    SimplexSlot slot;
    slot.id                 = idx;
    slot.phase              = SlotPhase::INIT;
    slot.size               = size;
    slot.iteration          = 0;
    slot.converged          = false;
    slot.trans_step_size    = trans_step_size;
    slot.rot_step_size      = rot_step_size;
    slot.tors_step_size     = tors_step_size;
    slot.score_converge     = score_converge;
    slot.max_iterations     = max_iterations;
    slot.skip_internal_energy = skip_internal_energy;
    slot.score              = score;
    slot.user_data          = user_data;
    slot.restrained         = restrained;
    slot.coefficient_restraint = coefficient_restraint;
    slot.m_rmsd_ref         = rmsd_ref;

    /* Allocate and build initial simplex vertices (N+1 vertices, each of length size) */
    slot.p = new float*[size + 1];
    for (int i = 0; i < size + 1; i++) {
        slot.p[i] = new float[size];
    }
    slot.y = new float[size + 1];
    slot.centroid.resize(size);
    slot.reflect_v.resize(size);
    build_initial_simplex(initial_vertex, slot.p, slot.y, size);

    /* Molecule state: caller's original (in-place update on convergence),
       ref_mol (read-only copy), tmp_mol (scratch for vector_to_dockmol). */
    slot.m_mol              = mol;
    slot.m_refMol           = new DOCKMol();
    slot.m_tmpMol           = new DOCKMol();
    copy_molecule(*slot.m_refMol, *mol);
    copy_molecule(*slot.m_tmpMol, *mol);

    m_slots.push_back(std::move(slot));
    return idx;
}


/* ------------------------------------------------------------------ */
/*  poll — Collect converged conformers                               */
/* ------------------------------------------------------------------ */

vector<SimplexSlot*>
ConformerPool::poll()
{
    vector<SimplexSlot*> result = std::move(m_converged);
    m_converged.clear();
    return result;
}


/* ------------------------------------------------------------------ */
/*  active_count / idle — Slot status queries                         */
/* ------------------------------------------------------------------ */

int
ConformerPool::active_count() const
{
    int count = 0;
    for (auto& slot : m_slots) {
        if (slot.phase != SlotPhase::CONVERGED) count++;
    }
    return count;
}


bool
ConformerPool::idle() const
{
    return active_count() == 0;
}


/* ------------------------------------------------------------------ */
/*  step — Run one pool cycle (STUB — filled in S5)                   */
/* ------------------------------------------------------------------ */

int
ConformerPool::step()
{
    /* TODO (S5): pack ready candidates, submit GPU dispatch,
       distribute scores, run decision trees, check convergence. */
    if (!m_use_gpu) return 0;

    /* Mark all INIT slots as converged as a placeholder so the caller
       can drain the pool (scores are uninitialized, but this unblocks
       growth-loop integration for testing). */
    for (auto& slot : m_slots) {
        if (slot.phase == SlotPhase::INIT && !slot.converged) {
            slot.phase       = SlotPhase::CONVERGED;
            slot.converged   = true;
            m_converged.push_back(&slot);
        }
    }

    return (int)m_converged.size();
}


/* ------------------------------------------------------------------ */
/*  Private stubs (filled in S3, S4, S5)                              */
/* ------------------------------------------------------------------ */

int
ConformerPool::pack_slot(SimplexSlot& slot, int offset, int capacity)
{
    /* TODO (S3): copy_crds + scale_vector + vector_to_dockmol + xyz extract */
    (void)slot;
    (void)offset;
    (void)capacity;
    return offset;
}


void
ConformerPool::evaluate_slot(SimplexSlot& slot, const float* scores, int count)
{
    /* TODO (S4): Nelder-Mead decision tree — reflect/expand/contract/shrink path */
    (void)slot;
    (void)scores;
    (void)count;
}


bool
ConformerPool::fill_slot_from_mol(SimplexSlot& slot, int slot_idx)
{
    /* TODO (S4): copy molecule state into slot after convergence */
    (void)slot;
    (void)slot_idx;
    return true;
}


bool
ConformerPool::check_convergence(SimplexSlot& slot)
{
    /* TODO (S5): score delta vs score_converge, iteration vs max_iterations */
    (void)slot;
    return false;
}
