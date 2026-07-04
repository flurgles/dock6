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
    , m_xyz_buffer(nullptr)
    , m_score_buffer(nullptr)
    , m_active_flags(nullptr)
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
    delete[] m_active_flags;
}


/* ------------------------------------------------------------------ */
/*  add — Enqueue a conformer into the pool                           */
/* ------------------------------------------------------------------ */

int
ConformerPool::add(DOCKMol* mol,
                   const FLOATVec& initial_vertex,
                   float trans_step_size, float rot_step_size,
                   float tors_step_size, float score_converge,
                   int max_iterations,
                   bool restrained, float coefficient_restraint,
                   DOCKMol* rmsd_ref, void* user_data)
{
    /* Find a recyclable (CONVERGED) slot, or append a new one.
       Caller must poll() before add() so m_converged is drained —
       conf_gen_ag does this in its backpressure / drain loops. */
    int idx = -1;
    for (int i = 0; i < (int)m_slots.size(); i++) {
        if (m_slots[i].phase == SlotPhase::CONVERGED) {
            idx = i;
            /* Free old slot state before reuse */
            if (m_slots[idx].p) {
                for (int j = 0; j < m_slots[idx].size + 1; j++)
                    delete[] m_slots[idx].p[j];
                delete[] m_slots[idx].p;
            }
            delete[] m_slots[idx].y;
            delete m_slots[idx].m_refMol;
            delete m_slots[idx].m_tmpMol;
            m_slots[idx].p = nullptr;
            m_slots[idx].y = nullptr;
            m_slots[idx].m_refMol = nullptr;
            m_slots[idx].m_tmpMol = nullptr;
            break;
        }
    }
    if (idx < 0) {
        if ((int)m_slots.size() >= m_batch_max) return -1;
        idx = (int)m_slots.size();
    }

    int size = (int)initial_vertex.size();

    /* Lazy buffer allocation on first add().  All conformers in one
       growth layer share the same DOF and atom count, so the first
       conformer's dimensions define buffer sizing for the pool. */
    if (!m_xyz_buffer) {
        m_num_atoms = mol->num_atoms;
        int max_cand = m_batch_max * max(4, size + 1);
        m_xyz_buffer      = new float[max_cand * m_num_atoms * 3];
        m_score_buffer    = new float[max_cand];
        m_active_flags    = new int[m_num_atoms];
        for (int a = 0; a < m_num_atoms; a++)
            m_active_flags[a] = mol->atom_active_flags[a] ? 1 : 0;
        m_dispatch_capacity = max_cand;
    }

    SimplexSlot slot;
    slot.id                 = idx;
    slot.phase              = SlotPhase::INIT;
    slot.size               = size;
    slot.iteration          = 0;
    slot.converged          = false;
    slot.ihi                = 0;
    slot.inhi               = 0;
    slot.ilo                = 0;
    slot.trans_step_size    = trans_step_size;
    slot.rot_step_size      = rot_step_size;
    slot.tors_step_size     = tors_step_size;
    slot.score_converge     = score_converge;
    slot.max_iterations     = max_iterations;
    slot.user_data          = user_data;
    slot.restrained         = restrained;
    slot.coefficient_restraint = coefficient_restraint;
    /* Restraint reference: use caller's rmsd_ref if provided, otherwise
       use the internal refMol snapshot (RMSD^2 = 0 for the GPU path, which
       matches the existing do_minimize() GPU behavior). */
    if (restrained) {
        slot.m_rmsd_ref = (rmsd_ref != nullptr) ? rmsd_ref : slot.m_refMol;
    } else {
        slot.m_rmsd_ref = nullptr;
    }

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

    if (idx < (int)m_slots.size()) {
        m_slots[idx] = std::move(slot);   /* recycle converged slot */
    } else {
        m_slots.push_back(std::move(slot));  /* new slot */
    }
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
/*  step — Run one pool cycle                                          */
/* ------------------------------------------------------------------ */

int
ConformerPool::step()
{
    if (!m_use_gpu) return 0;
    if (m_slots.empty()) return 0;

    int na = m_num_atoms;
    int* active_flags = m_active_flags;  // cached at first add()

    /* Phase 1: Pack ready candidates from all active slots */
    int total = 0;
    for (auto& slot : m_slots) {
        if (slot.phase == SlotPhase::CONVERGED || slot.converged) continue;
        total = pack_slot(slot, total, m_dispatch_capacity);
    }

    /* No GPU work pending — check convergence on remaining slots */
    if (total == 0) {
        int newly = 0;
        for (auto& slot : m_slots) {
            if (slot.converged || slot.phase == SlotPhase::CONVERGED) continue;
            if (check_convergence(slot)) {
                fill_slot_from_mol(slot, slot.id);
                m_converged.push_back(&slot);
                newly++;
            }
        }
        return newly;
    }

    /* Phase 2: Submit GPU dispatch */
    int ok = dock_gpu_batch_score_with_ie(
        m_xyz_buffer, total, na, active_flags, m_score_buffer);

    if (!ok) {
        /* GPU scoring failed (e.g. IE data not uploaded).
           Mark all active slots converged with their current best
           pose so the drain loop terminates instead of spinning. */
        int newly = 0;
        for (auto& slot : m_slots) {
            if (slot.converged || slot.phase == SlotPhase::CONVERGED) continue;
            slot.converged = true;
            slot.phase = SlotPhase::CONVERGED;
            fill_slot_from_mol(slot, slot.id);
            m_converged.push_back(&slot);
            newly++;
        }
        return newly;
    }

    /* Phase 3: Distribute scores back to each slot */
    int newly_converged = 0;
    int off = 0;
    for (auto& slot : m_slots) {
        if (slot.phase == SlotPhase::CONVERGED || slot.converged) continue;

        int ncan = 0;
        switch (slot.phase) {
        case SlotPhase::INIT:    ncan = slot.size + 1; break;
        case SlotPhase::REFLECT: ncan = 4;             break;
        case SlotPhase::SHRINK:  ncan = slot.size;     break;
        default: break;
        }
        if (ncan == 0) continue;

        /* Add restraint energy — same Econstraint per slot per iteration */
        if (slot.restrained) {
            float Econstraint = slot.coefficient_restraint *
                m_minimizer->calc_active_rmsd2(*slot.m_rmsd_ref, *slot.m_refMol);
            for (int ci = 0; ci < ncan; ci++) {
                m_score_buffer[off + ci] += Econstraint;
            }
        }

        /* Sentinel check (defense-in-depth: the current kernel
           score_batch_kernel_atom_parallel never writes a sentinel —
           out-of-grid atoms just get 0 grid contribution.  Kept in
           case the kernel is changed to reject out-of-grid poses.) */
        bool sentinel_hit = false;
        for (int ci = 0; ci < ncan; ci++) {
            if (m_score_buffer[off + ci] < -1.0e30f) {
                sentinel_hit = true;
                break;
            }
        }
        if (sentinel_hit) {
            /* Out-of-grid: restore pre-min pose (matches CPU failure_exit) */
            copy_crds(*slot.m_mol, *slot.m_refMol);
            slot.converged = true;
            slot.phase     = SlotPhase::CONVERGED;
            m_converged.push_back(&slot);
            newly_converged++;
            off += ncan;
            continue;
        }

        /* Run decision tree */
        evaluate_slot(slot, m_score_buffer + off);
        off += ncan;

        if (check_convergence(slot)) {
            fill_slot_from_mol(slot, slot.id);
            m_converged.push_back(&slot);
            newly_converged++;
        }
    }

    return newly_converged;
}


/* ------------------------------------------------------------------ */
/*  pack_vertex — Single-vertex dof→xyz pipeline                      */
/* ------------------------------------------------------------------ */

void
ConformerPool::pack_vertex(SimplexSlot& slot, const float* vertex, int idx)
{
    int na = m_num_atoms;
    float* buf = m_xyz_buffer + idx * na * 3;

    /* Restore coordinates from reference molecule */
    copy_crds(*slot.m_tmpMol, *slot.m_refMol);

    /* Convert float* vertex to FLOATVec and apply step sizes */
    FLOATVec vertex_vec(slot.size);
    for (int j = 0; j < slot.size; j++) vertex_vec[j] = vertex[j];

    FLOATVec new_vec;
    m_minimizer->scale_vector(new_vec, vertex_vec,
                               slot.trans_step_size,
                               slot.rot_step_size,
                               slot.tors_step_size);

    /* Convert scaled DOF vector to molecule coordinates */
    m_minimizer->vector_to_dockmol(*slot.m_tmpMol, new_vec);

    /* Extract xyz into flat buffer (row-major: candidate × atom × xyz) */
    for (int a = 0; a < na; a++) {
        buf[a * 3]     = slot.m_tmpMol->x[a];
        buf[a * 3 + 1] = slot.m_tmpMol->y[a];
        buf[a * 3 + 2] = slot.m_tmpMol->z[a];
    }
}


/* ------------------------------------------------------------------ */
/*  pack_slot — Pack all ready candidates for one slot into xyz buffer */
/* ------------------------------------------------------------------ */

int
ConformerPool::pack_slot(SimplexSlot& slot, int offset, int capacity)
{
    int N = slot.size;
    /* Safety: don't overflow the dispatch buffer.  This should never
       trigger with correct sizing; if it does, the slot's candidates
       are deferred to the next step() (no state advance). */
    int need = 0;
    switch (slot.phase) {
    case SlotPhase::INIT:    need = N + 1; break;
    case SlotPhase::REFLECT: need = 4;     break;
    case SlotPhase::SHRINK:  need = N;     break;
    default: return offset;
    }
    if (offset + need > capacity) return offset;

    switch (slot.phase) {

    case SlotPhase::INIT:
        /* Pack all N+1 initial vertices */
        for (int vi = 0; vi < N + 1; vi++) {
            pack_vertex(slot, slot.p[vi], offset + vi);
        }
        return offset + N + 1;

    case SlotPhase::REFLECT: {
        /* Pack 4 speculative candidates:
             0: reflect_v (pr)
             1: expand   = (1+alpha)*pr - alpha*pbar
             2: contract_cA = beta*pr + (1-beta)*pbar
             3: contract_cB = beta*p[ihi] + (1-beta)*pbar
        */
        const float alpha = 1.0f, beta = 0.5f;
        float* pbar = slot.centroid.data();
        float* pr   = slot.reflect_v.data();
        float** p   = slot.p;
        int ihi     = slot.ihi;

        /* Candidate 0: reflected point */
        pack_vertex(slot, pr, offset + 0);

        /* Candidates 1-3 computed on the fly */
        FLOATVec expand_v(N), cA_v(N), cB_v(N);
        for (int j = 0; j < N; j++) {
            expand_v[j] = (1.0f + alpha) * pr[j] - alpha * pbar[j];
            cA_v[j]     = beta * pr[j] + (1.0f - beta) * pbar[j];
            cB_v[j]     = beta * p[ihi][j] + (1.0f - beta) * pbar[j];
        }
        pack_vertex(slot, expand_v.data(), offset + 1);
        pack_vertex(slot, cA_v.data(),     offset + 2);
        pack_vertex(slot, cB_v.data(),     offset + 3);

        return offset + 4;
    }

    case SlotPhase::SHRINK: {
        /* Pack N candidates: all N+1 vertices except p[ilo] */
        int ilo = slot.ilo;
        int ci = 0;
        for (int vi = 0; vi < N + 1; vi++) {
            if (vi == ilo) continue;
            pack_vertex(slot, slot.p[vi], offset + ci);
            ci++;
        }
        return offset + ci;
    }

    case SlotPhase::CONVERGED:
    default:
        return offset;
    }
}


/* ------------------------------------------------------------------ */
/*  Static helpers for the Nelder-Mead decision tree                  */
/* ------------------------------------------------------------------ */

/* Rank vertices: find indices of best (ilo), worst (ihi), and
   2nd-worst (inhi) by score.  Matches simplex.cpp ranking block. */
static void rank_vertices(float* y, int nverts, int& ihi, int& inhi, int& ilo)
{
    ilo = 0;
    if (y[0] > y[1]) {
        ihi  = 0;
        inhi = 1;
    } else {
        ihi  = 1;
        inhi = 0;
    }
    for (int i = 0; i < nverts; i++) {
        if (y[i] < y[ilo]) ilo = i;
        if (y[i] > y[ihi]) {
            inhi = ihi;
            ihi  = i;
        } else if (y[i] > y[inhi] && i != ihi) {
            inhi = i;
        }
    }
}


/* Compute centroid (pbar) = average of all vertices except ihi,
   and reflected point pr = (1+alpha)*pbar - alpha*p[ihi] with alpha = 1.0.
   Matches the per-iteration computation in simplex.cpp. */
static void compute_centroid_reflect(float** p, int N, int ihi,
                                     float* pbar, float* pr)
{
    const float alpha = 1.0f;

    for (int j = 0; j < N; j++) pbar[j] = 0.0f;
    for (int vi = 0; vi < N + 1; vi++) {
        if (vi == ihi) continue;
        for (int j = 0; j < N; j++) pbar[j] += p[vi][j];
    }
    for (int j = 0; j < N; j++) {
        pbar[j] /= (float)N;
        pr[j]    = (1.0f + alpha) * pbar[j] - alpha * p[ihi][j];
    }
}


/* ------------------------------------------------------------------ */
/*  evaluate_slot — Nelder-Mead decision tree for one slot            */
/* ------------------------------------------------------------------ */

/* Process scored candidates and apply the appropriate simplex operation.
   Called from step() after GPU scores are returned.

   Scores array layout depends on slot phase:
       INIT:    scores[0..N]  — one per initial vertex
       REFLECT: scores[0..3]  — reflect, expand, cA, cB
       SHRINK:  scores[0..N-1] — one for each vertex except ilo
*/
void
ConformerPool::evaluate_slot(SimplexSlot& slot, const float* scores)
{
    int     N   = slot.size;
    float** p   = slot.p;
    float*  y   = slot.y;

    switch (slot.phase) {

    /* --- INIT: copy initial vertex scores, rank, compute first centroid --- */
    case SlotPhase::INIT: {
        for (int vi = 0; vi < N + 1; vi++) {
            y[vi] = scores[vi];
        }
        rank_vertices(y, N + 1, slot.ihi, slot.inhi, slot.ilo);
        compute_centroid_reflect(p, N, slot.ihi,
                                  slot.centroid.data(),
                                  slot.reflect_v.data());
        slot.iteration++;
        slot.phase = SlotPhase::REFLECT;
        break;
    }

    /* --- REFLECT: full Nelder-Mead decision tree --- */
    case SlotPhase::REFLECT: {
        const float alpha = 1.0f, beta = 0.5f;
        float* pbar      = slot.centroid.data();
        float* pr        = slot.reflect_v.data();

        float  ypr       = scores[0];   // reflected
        float  yprr_exp  = scores[1];   // expanded
        float  yprr_cA   = scores[2];   // contracted with pr
        float  yprr_cB   = scores[3];   // contracted with p[ihi]

        int    ihi       = slot.ihi;
        int    ilo       = slot.ilo;
        int    inhi      = slot.inhi;

        bool replace_flag = false;

        /* --- Expansion path --- */
        if (ypr <= y[ilo]) {
            if (yprr_exp < y[ilo]) {
                /* Accept expansion point */
                for (int j = 0; j < N; j++) {
                    p[ihi][j] = (1.0f + alpha) * pr[j] - alpha * pbar[j];
                }
                y[ihi] = yprr_exp;
            } else {
                /* Accept reflected point */
                for (int j = 0; j < N; j++) p[ihi][j] = pr[j];
                y[ihi] = ypr;
            }
            replace_flag = true;

        /* --- Contraction / Shrink path --- */
        } else if (ypr >= y[inhi]) {

            if (ypr < y[ihi]) {
                /* Replace worst vertex with reflected point */
                for (int j = 0; j < N; j++) p[ihi][j] = pr[j];
                y[ihi] = ypr;
                replace_flag = true;
            }

            float yprr_contract = (ypr < y[ihi]) ? yprr_cA : yprr_cB;

            if (yprr_contract < y[ihi]) {
                /* Accept the contracted point */
                const float* contr_base = (ypr < y[ihi]) ? pr : p[ihi];
                for (int j = 0; j < N; j++) {
                    p[ihi][j] = beta * contr_base[j] + (1.0f - beta) * pbar[j];
                }
                y[ihi] = yprr_contract;
                replace_flag = true;
            }

            if (!replace_flag) {
                /* SHRINK: move all vertices toward ilo */
                for (int vi = 0; vi < N + 1; vi++) {
                    if (vi == ilo) continue;
                    for (int j = 0; j < N; j++) {
                        p[vi][j] = 0.5f * (p[vi][j] + p[ilo][j]);
                    }
                }
                slot.phase = SlotPhase::SHRINK;
                return;  /* scores are stale — skip ranking */
            }

        } else {
            /* --- Middling: accept reflected point --- */
            for (int j = 0; j < N; j++) p[ihi][j] = pr[j];
            y[ihi] = ypr;
            replace_flag = true;
        }

        /* Re-rank and compute centroid for next iteration */
        rank_vertices(y, N + 1, slot.ihi, slot.inhi, slot.ilo);
        slot.iteration++;
        compute_centroid_reflect(p, N, slot.ihi,
                                  slot.centroid.data(),
                                  slot.reflect_v.data());
        break;
    }

    /* --- SHRINK: copy rescored vertex scores --- */
    case SlotPhase::SHRINK: {
        int ci = 0;
        for (int vi = 0; vi < N + 1; vi++) {
            if (vi == slot.ilo) continue;
            y[vi] = scores[ci++];
        }
        rank_vertices(y, N + 1, slot.ihi, slot.inhi, slot.ilo);
        slot.iteration++;
        compute_centroid_reflect(p, N, slot.ihi,
                                  slot.centroid.data(),
                                  slot.reflect_v.data());
        slot.phase = SlotPhase::REFLECT;
        break;
    }

    case SlotPhase::CONVERGED:
    default:
        break;
    }
}


/* ------------------------------------------------------------------ */
/*  fill_slot_from_mol — Update caller's DOCKMol with minimized pose  */
/* ------------------------------------------------------------------ */

bool
ConformerPool::fill_slot_from_mol(SimplexSlot& slot, int slot_idx)
{
    (void)slot_idx;
    if (!slot.converged) return false;

    int ilo = slot.ilo;

    /* Convert best vertex to molecule coordinates and copy to caller's mol */
    copy_crds(*slot.m_tmpMol, *slot.m_refMol);

    FLOATVec vertex_vec(slot.size);
    for (int j = 0; j < slot.size; j++) vertex_vec[j] = slot.p[ilo][j];

    FLOATVec new_vec;
    m_minimizer->scale_vector(new_vec, vertex_vec,
                               slot.trans_step_size,
                               slot.rot_step_size,
                               slot.tors_step_size);
    m_minimizer->vector_to_dockmol(*slot.m_tmpMol, new_vec);
    copy_crds(*slot.m_mol, *slot.m_tmpMol);
    return true;
}


/* ------------------------------------------------------------------ */
/*  check_convergence — Score delta and iteration-limit test          */
/* ------------------------------------------------------------------ */

bool
ConformerPool::check_convergence(SimplexSlot& slot)
{
    if (slot.converged) return true;
    if (slot.phase == SlotPhase::INIT || slot.phase == SlotPhase::SHRINK) return false;
    if (slot.phase == SlotPhase::CONVERGED) {
        slot.converged = true;
        return true;
    }

    float delta = fabs(slot.y[slot.ihi] - slot.y[slot.ilo]);
    bool done = (delta <= slot.score_converge) || (slot.iteration >= slot.max_iterations);
    if (done) {
        slot.converged = true;
        slot.phase     = SlotPhase::CONVERGED;
    }
    return done;
}
