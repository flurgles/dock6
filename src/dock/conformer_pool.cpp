/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
ConformerPool — CPU-side orchestrator for batch simplex minimization.

Packs ready candidates from active slots into a flat GPU dispatch buffer,
calls dock_gpu_batch_score_with_ie_persistent(), distributes returned scores to each
slot's Nelder-Mead decision tree, and manages slot lifecycle (add, converge,
poll).

GPU-independent: no Metal/Vulkan/CUDA types in this file.
All GPU calls go through the score_dock_gpu.h C API.
*/

#include "conformer_pool.h"
#include "dockmol.h"
#include <cstring>

using namespace std;


/* ------------------------------------------------------------------ */
/*  Internal helpers                                                  */
/* ------------------------------------------------------------------ */

/* Compute Nelder-Mead coefficients from simplex_mode and dimension.
   Exact mirror of simplex.cpp do_minimize() lines 129-149. */
static void compute_nm_coeffs(int size, int simplex_mode, int simplex_crossover,
                               float& gamma, float& beta, float& sigma)
{
    gamma = 2.0f;
    beta  = 0.5f;
    sigma = 0.5f;

    if (simplex_mode == 1) {
        const float n = (float)size;
        gamma = 1.0f + 2.0f / n;
        beta  = 0.75f - 0.5f / n;
        sigma = 1.0f  - 1.0f  / n;
    } else if (simplex_mode == 2) {
        const float n  = (float)size;
        const float n0 = (float)(simplex_crossover + 6);
        const float k  = 0.5f;
        float w = 1.0f - 1.0f / (1.0f + expf(-k * (n - n0)));
        if (w < 0.0f) w = 0.0f;
        if (w > 1.0f) w = 1.0f;
        const float ga = 1.0f + 2.0f / n;
        const float rb = 0.75f - 0.5f / n;
        const float sg = 1.0f  - 1.0f  / n;
        gamma = w * 2.0f + (1.0f - w) * ga;
        beta  = w * 0.5f + (1.0f - w) * rb;
        sigma = w * 0.5f + (1.0f - w) * sg;
    }
}


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

ConformerPool::ConformerPool(int batch_max, Minimizer* minimizer, bool use_gpu,
                              int simplex_mode, int simplex_crossover)
    : m_batch_max(batch_max)
    , m_minimizer(minimizer)
    , m_use_gpu(use_gpu)
    , m_simplex_mode(simplex_mode)
    , m_simplex_crossover(simplex_crossover)
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
    delete[] m_pose_lig;
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
                   int max_cycles,
float cycle_converge,
                    bool restrained, float coefficient_restraint,
                    DOCKMol* rmsd_ref, void* user_data,
                    int lig_idx, int lig_num_atoms)
{
    if (lig_idx >= 0) m_vs_mode = true;
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
       conformer's dimensions define buffer sizing for the pool.
       VS mode (lig_idx >= 0): ligands may have different atom counts,
       so the xyz stride is the LARGEST num_atoms seen so far and
       buffers grow on demand (poses padded with inactive atoms). */
    int first_stride = 0;
    if (m_vs_mode) {
        if (lig_num_atoms <= 0) lig_num_atoms = mol->num_atoms;
        first_stride = lig_num_atoms;
    } else {
        first_stride = mol->num_atoms;
    }

    if (!m_xyz_buffer) {
m_num_atoms = first_stride;
        int max_cand = m_batch_max * max(4, size + 1);
        m_xyz_buffer      = new float[max_cand * m_num_atoms * 3];
        m_score_buffer    = new float[max_cand];
        m_pose_lig        = new int[max_cand];
        m_active_flags    = new int[m_num_atoms];
        for (int a = 0; a < m_num_atoms; a++)
            m_active_flags[a] = 0;  // VS path: flags come from the LUT per ligand
        m_dispatch_capacity = max_cand;
    } else if (m_vs_mode && first_stride > m_num_atoms) {
        /* A larger ligand arrived — grow the stride (and buffers). */
        int max_cand = m_batch_max * max(4, size + 1);
        delete[] m_xyz_buffer;
        delete[] m_score_buffer;
        delete[] m_pose_lig;
        delete[] m_active_flags;
        m_num_atoms = first_stride;
        m_xyz_buffer      = new float[max_cand * m_num_atoms * 3];
        m_score_buffer    = new float[max_cand];
        m_pose_lig        = new int[max_cand];
        m_active_flags    = new int[m_num_atoms];
        for (int a = 0; a < m_num_atoms; a++)
            m_active_flags[a] = 0;  // VS path: flags come from the LUT per ligand
        m_dispatch_capacity = max_cand;
    }

    SimplexSlot slot;
    slot.lig_idx          = lig_idx;
    slot.lig_num_atoms    = (m_vs_mode) ? first_stride
                                       : mol->num_atoms;
    slot.id                 = idx;
    slot.phase              = SlotPhase::INIT;
    slot.size               = size;
    compute_nm_coeffs(size, m_simplex_mode, m_simplex_crossover,
                      slot.nm_gamma, slot.nm_beta, slot.nm_sigma);
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
    slot.current_cycle      = 0;
    slot.max_cycles         = max_cycles;
    slot.cycle_converge     = cycle_converge;
    slot.has_best           = false;
    slot.best_score         = 1e30f;
    slot.best_vertex.resize(size);
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
    int ok;
    if (m_vs_mode) {
        /* Multi-ligand VS path: poses tagged with their ligand LUT slot,
           stride = max atoms across slots, per-pose params from LUT. */
        ok = dock_gpu_batch_score_vs(m_xyz_buffer, total, m_num_atoms,
                                     m_pose_lig, m_score_buffer);
    } else {
        ok = dock_gpu_batch_score_with_ie_persistent(
            m_xyz_buffer, total, na, active_flags, m_score_buffer);
    }

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
    if (slot.lig_idx >= 0) na = slot.lig_num_atoms;  // VS: real atoms only
    float* buf = m_xyz_buffer + idx * m_num_atoms * 3;

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

    /* VS path: tag every packed candidate with its ligand LUT slot.
       Never leave tags uninitialized — the kernel reads them per pose. */
    int vs_tag = slot.lig_idx;
    if (m_vs_mode) {
        for (int ci = 0; ci < need; ci++) {
            m_pose_lig[offset + ci] = (vs_tag >= 0) ? vs_tag : 0;
        }
    }

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
             1: expand   = gamma*pr + (1-gamma)*pbar
             2: contract_cA = beta*pr + (1-beta)*pbar
             3: contract_cB = beta*p[ihi] + (1-beta)*pbar
           Matches simplex.cpp spec_verts layout for GPU batch. */
        float gamma = slot.nm_gamma;
        float beta  = slot.nm_beta;
        float* pbar = slot.centroid.data();
        float* pr   = slot.reflect_v.data();
        float** p   = slot.p;
        int ihi     = slot.ihi;

        /* Candidate 0: reflected point */
        pack_vertex(slot, pr, offset + 0);

        /* Candidates 1-3 computed on the fly */
        FLOATVec expand_v(N), cA_v(N), cB_v(N);
        for (int j = 0; j < N; j++) {
            expand_v[j] = gamma * pr[j] + (1.0f - gamma) * pbar[j];
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
        float gamma    = slot.nm_gamma;
        float beta     = slot.nm_beta;
        float sigma    = slot.nm_sigma;
        float* pbar    = slot.centroid.data();
        float* pr      = slot.reflect_v.data();

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
                /* Accept expansion point: gamma*pr + (1-gamma)*pbar */
                for (int j = 0; j < N; j++) {
                    p[ihi][j] = gamma * pr[j] + (1.0f - gamma) * pbar[j];
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

            /* Save condition BEFORE overwriting p[ihi]/y[ihi].  After the
               overwrite below, `(ypr < y[ihi])` would always be false
               (ypr < ypr), incorrectly selecting the cB variant. */
            bool outer = (ypr < y[ihi]);

            if (outer) {
                /* Replace worst vertex with reflected point */
                for (int j = 0; j < N; j++) p[ihi][j] = pr[j];
                y[ihi] = ypr;
                replace_flag = true;
            }

            float yprr_contract = outer ? yprr_cA : yprr_cB;

            if (yprr_contract < y[ihi]) {
                /* Accept the contracted point: beta*hi + (1-beta)*pbar */
                const float* contr_base = outer ? pr : p[ihi];
                for (int j = 0; j < N; j++) {
                    p[ihi][j] = beta * contr_base[j] + (1.0f - beta) * pbar[j];
                }
                y[ihi] = yprr_contract;
                replace_flag = true;
            }

            if (!replace_flag) {
                /* SHRINK: move all vertices toward ilo: p[ilo] + sigma*(p- p[ilo]) */
                for (int vi = 0; vi < N + 1; vi++) {
                    if (vi == ilo) continue;
                    for (int j = 0; j < N; j++) {
                        p[vi][j] = p[ilo][j] + sigma * (p[vi][j] - p[ilo][j]);
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
    if (!done) return false;

    /* Simplex converged.  Save best result across cycles. */
    {
        float score = slot.y[slot.ilo];
        if (!slot.has_best || score < slot.best_score) {
            slot.best_score = score;
            slot.best_vertex.resize(slot.size);
            for (int j = 0; j < slot.size; j++)
                slot.best_vertex[j] = slot.p[slot.ilo][j];
            slot.has_best = true;
        }
    }

    /* Check if we should restart another cycle. */
    slot.current_cycle++;

    if (slot.current_cycle < slot.max_cycles) {
        /* Compute distance moved (matches minimizer.cpp cycle logic) */
        float distance = 0.0f;
        int ilo = slot.ilo;
        for (int j = 0; j < slot.size; j++)
            distance += slot.p[ilo][j] * slot.p[ilo][j];
        distance = sqrtf(distance) / (float)slot.current_cycle;

        if (distance > slot.cycle_converge) {
            /* Update ref_mol to current best pose, then restart */
            fill_slot_from_mol(slot, slot.id);
            copy_molecule(*slot.m_refMol, *slot.m_mol);

            /* Reset vertex to zeros (start from current best) */
            for (int vi = 0; vi < slot.size + 1; vi++)
                memset(slot.p[vi], 0, slot.size * sizeof(float));

            /* Build new initial simplex from zeros */
            FLOATVec zero_vertex(slot.size, 0.0f);
            build_initial_simplex(zero_vertex, slot.p, slot.y, slot.size);

            /* Reset iteration state */
            slot.iteration = 0;
            slot.phase = SlotPhase::INIT;
            slot.ihi = 0;
            slot.inhi = 0;
            slot.ilo = 0;
            return false;  /* not done yet */
        }
    }

    /* All cycles done.  Restore best vertex across all cycles. */
    if (slot.has_best) {
        for (int j = 0; j < slot.size; j++)
            slot.p[slot.ilo][j] = slot.best_vertex[j];
        slot.y[slot.ilo] = slot.best_score;
    }

    slot.converged = true;
    slot.phase     = SlotPhase::CONVERGED;
    return true;
}
