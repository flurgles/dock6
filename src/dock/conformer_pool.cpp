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
#include <chrono>

static long long lbal_now_ms(void)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
static void build_initial_simplex(Minimizer* min, const FLOATVec& vertex, float** p, float* y, int size)
{
    for (int i = 0; i < size; i++) {
        p[0][i] = vertex[i];
    }
    for (int i = 1; i < size + 1; i++) {
        for (int j = 0; j < size; j++) {
            p[i][j] = vertex[j] + 2.0f * (min->next_rand_01() - 0.5f);
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
                   const SimplexStage& stageA,
                   const SimplexStage& stageB,
                   float cycle_converge,
                   bool restrained, float coefficient_restraint,
                   DOCKMol* rmsd_ref, void* user_data,
                   int lig_idx, int lig_num_atoms,
                   unsigned int rng_seed)
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
        if (m_vs_mode) {
            for (int a = 0; a < m_num_atoms; a++)
                m_active_flags[a] = 0;  // VS path: flags come from the LUT per ligand
        } else {
            for (int a = 0; a < m_num_atoms; a++)
                m_active_flags[a] = (mol->atom_active_flags[a]) ? 1 : 0;
        }
        m_dispatch_capacity = max_cand;
    } else if (m_vs_mode && first_stride > m_num_atoms) {
        /* A larger ligand arrived — grow the stride (and buffers).  Any
           pending async batches reference the OLD buffers (their DtoH
           targets are m_score_buffer + offset), so they must be drained
           BEFORE the buffers are freed.  Dropping the scores of one
           round is benign: the affected slots re-score the same vertices
           on the next enqueue. */
        if (m_npends > 0) {
            dock_gpu_batch_score_sync();
            m_npends = 0;
        }
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
        if (m_vs_mode) {
            for (int a = 0; a < m_num_atoms; a++)
                m_active_flags[a] = 0;  // VS path: flags come from the LUT per ligand
        } else {
            for (int a = 0; a < m_num_atoms; a++)
                m_active_flags[a] = (mol->atom_active_flags[a]) ? 1 : 0;
        }
        m_dispatch_capacity = max_cand;
    } else if (!m_vs_mode) {
        /* Non-VS pools (growth minimization) all carry the same active
           set as the caller's conformer: refresh so flags always match
           the current layer's grown atoms. */
        for (int a = 0; a < m_num_atoms; a++)
            m_active_flags[a] = (mol->atom_active_flags[a]) ? 1 : 0;
    }

    SimplexSlot slot;
    slot.lig_idx          = lig_idx;
    slot.lig_num_atoms    = (m_vs_mode) ? first_stride
                                       : mol->num_atoms;
    /* Snapshot the DOF→xyz mapping so this slot can be packed even when
       the pool holds slots of other ligands (the minimizer's members are
       shared and reflect the most recent id_torsions() call). */
    slot.torsions             = m_minimizer->torsions;
    slot.bond_vectors         = m_minimizer->bond_vectors;
    slot.torsion_scale_factors = m_minimizer->torsion_scale_factors;
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
    slot.trans_step_size    = stageA.trans_step_size;
    slot.rot_step_size      = stageA.rot_step_size;
    slot.tors_step_size     = stageA.tors_step_size;
    slot.score_converge     = stageA.score_converge;
    slot.max_iterations     = stageA.max_iterations;
    slot.current_cycle      = 0;
    slot.max_cycles         = stageA.max_cycles;
    slot.cycle_converge     = cycle_converge;
    slot.stage              = 0;
    slot.max_iterations_B   = stageB.max_iterations;
    slot.max_cycles_B       = stageB.max_cycles;
    slot.score_converge_B   = stageB.score_converge;
    slot.trans_step_size_B  = stageB.trans_step_size;
    slot.rot_step_size_B    = stageB.rot_step_size;
    slot.tors_step_size_B   = stageB.tors_step_size;
    slot.rng_state          = rng_seed ? rng_seed : 1u;
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
    m_minimizer->set_local_rng_state(slot.rng_state);
    build_initial_simplex(m_minimizer, initial_vertex, slot.p, slot.y, size);
    slot.rng_state = m_minimizer->local_rng_state();

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
    int c = step_enqueue();
    if (c) return c;                 /* non-VS path finishes in enqueue */
    return step_finish();
}

void
ConformerPool::adapt_split_k()
{
    /* Governor: compare the GPU's busy time with the host's step time.
       gpu ≫ host  → split into more groups so multiple kernels per step
       keep the GPU fed while the host preps the next step.
       host ≫ gpu   → merge back into one batch (nothing to feed). */
    long long h = m_lbal_host, g = m_lbal_gpu;
    if (h <= 0) h = 1;
    if (g <= 0) g = 1;
    double ratio = (double)g / (double)(h + g);
    if (ratio > 0.55 && m_split_k < 4) m_split_k++;
    else if (ratio < 0.25 && m_split_k > 1) m_split_k--;

    if ((m_steps_since_lbal++ % 96) == 0) {
        static long long last_poses = 0, last_ms = 0;
        long long now = lbal_now_ms();
        double rate = (last_ms > 0)
            ? (double)(m_poses_total - last_poses) * 1000.0 / (double)(now - last_ms)
            : 0.0;
        fprintf(stderr,
                "[LBAL] k=%d host=%lldms gpu=%lldms ratio=%.2f poses=%lld rate=%.0f/s\n",
                m_split_k, h, g, ratio, m_poses_total, rate);
        last_poses = m_poses_total; last_ms = now;
        fflush(stderr);
    }
}

int
ConformerPool::step_enqueue()
{
    if (!m_use_gpu) return 0;
    if (m_slots.empty()) return 0;
    if (!m_vs_mode) return step_legacy();

    m_lbal_t0 = lbal_now_ms();

    int na = m_num_atoms;

    /* Split the active slots into m_split_k groups (round-robin) so one
       group's packing overlaps the previous group's GPU kernel. */
    int need_of[4096], act_list[4096];
    int nact = 0, total = 0;
    for (size_t si = 0; si < m_slots.size(); si++) {
        SimplexSlot& slot = m_slots[si];
        if (slot.phase == SlotPhase::CONVERGED || slot.converged) continue;
        int need = 0;
        switch (slot.phase) {
        case SlotPhase::INIT:    need = slot.size + 1; break;
        case SlotPhase::REFLECT: need = 4;             break;
        case SlotPhase::SHRINK:  need = slot.size;     break;
        default: break;
        }
        need_of[nact] = need;
        act_list[nact] = (int)si;
        total += need;
        nact++;
    }
    m_poses_total += total;

    /* Nothing ready this round — finish() will convergence-check. */
    m_npends = 0;
    if (nact == 0 || total == 0) {
        if (getenv("DOCK_LBAL_DEBUG") != NULL) {
            int c_phase = 0, c_flag = 0, c_other = 0;
            for (size_t si = 0; si < m_slots.size(); si++) {
                if (m_slots[si].phase == SlotPhase::CONVERGED) c_phase++;
                else if (m_slots[si].converged) c_flag++;
                else c_other++;
            }
            int d0 = -1;
            for (size_t si = 0; si < m_slots.size() && d0 < 0; si++) {
                if (m_slots[si].phase != SlotPhase::CONVERGED &&
                    !m_slots[si].converged) d0 = (int)si;
            }
            fprintf(stderr,
                    "[LBALDBG] enqueue: nact=%d total=%d slots=%zu "
                    "conv_phase=%d conv_flag=%d open=%d first_open=%d\n",
                    nact, total, m_slots.size(), c_phase, c_flag, c_other,
                    d0);
        }
        return 0;
    }

    int k = m_split_k;
    int groups = (nact < k) ? nact : k;
    if (groups < 1) groups = 1;

    /* Group base offsets (packing order: group 0, 1, ... round-robin). */
    int base[4] = {0, 0, 0, 0};
    int gtot[4] = {0, 0, 0, 0};
    int gcnt[4] = {0, 0, 0, 0};
    /* Cap per group so the TOTAL packed poses never exceed the
       m_xyz_buffer/m_score_buffer/m_pose_lig buffer capacity
       (m_dispatch_capacity == buffer size).  A per-group cap of the
       full capacity would overflow the buffers when k > 1. */
    int cap = m_dispatch_capacity / groups;
    if (cap < 1) cap = 1;
    for (int si = 0; si < nact; si++) {
        int g = si % groups;
        if (gtot[g] + need_of[si] > cap) continue;   /* deferred */
        gtot[g] += need_of[si];
        gcnt[g]++;
    }
    int off = 0;
    for (int g = 0; g < groups; g++) {
        base[g] = off;
        off += gtot[g];
    }
    if (off == 0) {
        if (getenv("DOCK_LBAL_DEBUG") != NULL)
            fprintf(stderr, "[LBALDBG] enqueue: off==0 nact=%d groups=%d\n",
                    nact, groups);
        return 0;
    }

    /* Default: no slot was enqueued this round (stale bases invalid). */
    for (size_t si = 0; si < m_slots.size(); si++) m_slot_base[si] = -1;

    /* Pack each group, then enqueue its chunks on the background stream. */
    bool fail = false;
    for (int g = 0; g < groups && !fail; g++) {
        int o = base[g];
        int filled = 0;
        for (int si = 0; si < nact; si++) {
            if ((si % groups) != g) continue;
            SimplexSlot& slot = m_slots[act_list[si]];
            m_slot_base[act_list[si]] = o;
            int before = o;
            o = pack_slot(slot, o, base[g] + gtot[g]);
            filled += (o - before);
        }
        if (filled == 0) continue;

        /* Emit bounded chunks (≤ GPU_MAX_BATCH_POSES per call). */
        int dispatched = 0;
        while (dispatched < filled) {
            int n = filled - dispatched;
            if (n > GPU_MAX_BATCH_POSES) n = GPU_MAX_BATCH_POSES;
            const float *xyz_pos = m_xyz_buffer +
                (size_t)(base[g] + dispatched) * m_num_atoms * 3;
            long long t_host = lbal_now_ms();
            int chunk_ok = dock_gpu_batch_score_vs_enqueue(
                xyz_pos, n, na, m_pose_lig + base[g] + dispatched,
                m_score_buffer + base[g] + dispatched, 0);
            if (!chunk_ok) { fail = true; break; }
            m_npends++;
            lbal_now_ms();  /* keep host EMA fed inside the backend */
            (void)t_host;
            dispatched += n;
        }
    }

    if (fail) {
        m_lbal_fail = true;
        return 0;
    }
    if (getenv("DOCK_LBAL_DEBUG") != NULL)
        fprintf(stderr, "[LBALDBG] enqueue: nact=%d total=%d k=%d groups=%d "
                "filled=%d npends=%d\n", nact, total, m_split_k, groups,
                off, m_npends);
    return 0;
}

int
ConformerPool::step_finish()
{
    if (!m_use_gpu) return 0;
    if (m_slots.empty()) return 0;
    if (!m_vs_mode) return 0;

    double host_phase = (double)(lbal_now_ms() - m_lbal_t0);
    if (host_phase < 0.1) host_phase = 0.1;
    m_lbal_host = (long long)(0.85 * (double)m_lbal_host + 0.15 * host_phase);

    if (m_lbal_fail) {
        m_lbal_fail = false;
        m_npends = 0;
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

    if (m_npends > 0) {
        dock_gpu_batch_score_sync();
        dock_gpu_lbal_stats(&m_lbal_host, &m_lbal_gpu);
        adapt_split_k();
        m_npends = 0;
    } else {
        int newly = 0;
        for (auto& slot : m_slots) {
            if (slot.converged || slot.phase == SlotPhase::CONVERGED) continue;
            if (check_convergence(slot)) {
                fill_slot_from_mol(slot, slot.id);
                m_converged.push_back(&slot);
                newly++;
            }
        }
        if (getenv("DOCK_LBAL_DEBUG") != NULL)
            fprintf(stderr, "[LBALDBG] finish: npends=0 newly=%d slots=%zu\n",
                    newly, m_slots.size());
        return newly;
    }

    int newly_converged = 0;
    for (size_t i = 0; i < m_slots.size(); i++) {
        SimplexSlot& slot = m_slots[i];
        if (slot.phase == SlotPhase::CONVERGED || slot.converged) continue;

        int sb = m_slot_base[i];
        if (sb < 0) continue;   /* not enqueued this round — leave for next */

        int ncan = 0;
        switch (slot.phase) {
        case SlotPhase::INIT:    ncan = slot.size + 1; break;
        case SlotPhase::REFLECT: ncan = 4;             break;
        case SlotPhase::SHRINK:  ncan = slot.size;     break;
        default: break;
        }
        if (ncan == 0) continue;
        m_slot_base[i] = -1;   /* consumed */

        /* Add restraint energy — same Econstraint per slot per iteration */
        if (slot.restrained) {
            float Econstraint = slot.coefficient_restraint *
                m_minimizer->calc_active_rmsd2(*slot.m_rmsd_ref, *slot.m_refMol);
            for (int ci = 0; ci < ncan; ci++) {
                m_score_buffer[sb + ci] += Econstraint;
            }
        }

        /* Sentinel check (defense-in-depth: the current kernel
           score_batch_kernel_atom_parallel never writes a sentinel —
           out-of-grid atoms just get 0 grid contribution.  Kept in
           case the kernel is changed to reject out-of-grid poses.) */
        bool sentinel_hit = false;
        for (int ci = 0; ci < ncan; ci++) {
            if (m_score_buffer[sb + ci] < -1.0e30f) {
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
            continue;
        }

        /* Run decision tree */
        evaluate_slot(slot, m_score_buffer + sb);

        if (check_convergence(slot)) {
            fill_slot_from_mol(slot, slot.id);
            m_converged.push_back(&slot);
            newly_converged++;
        }
    }

    return newly_converged;
}

/* Legacy whole-cycle implementation (single synchronous dispatch),
   used when the pool is not in VS mode.  Returns newly converged. */
int
ConformerPool::step_legacy()
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

    /* Phase 2: Submit GPU dispatch — the packed candidate list may
       exceed GPU_MAX_BATCH_POSES (large pools pack several thousand
       poses per step), so emit it in bounded chunks.  Poses are
       row-major in m_xyz_buffer and scores/pose tags are contiguous,
       so each chunk is just a pointer/offset slice of the same
       buffers; chunk dispatches write their scores back at the
       matching offsets. */
    int ok = 1;
    int dispatched = 0;
    while (dispatched < total) {
        int n = total - dispatched;
        if (n > GPU_MAX_BATCH_POSES) n = GPU_MAX_BATCH_POSES;
        const float *xyz_pos = m_xyz_buffer +
                               (size_t)dispatched * m_num_atoms * 3;
        int chunk_ok;
        if (m_vs_mode) {
            /* Multi-ligand VS path: poses tagged with their ligand LUT
               slot, stride = max atoms across slots, per-pose params
               from the LUT. */
            chunk_ok = dock_gpu_batch_score_vs(xyz_pos, n, m_num_atoms,
                                               m_pose_lig + dispatched,
                                               m_score_buffer + dispatched);
        } else {
            chunk_ok = dock_gpu_batch_score_with_ie_persistent(
                xyz_pos, n, m_num_atoms, active_flags,
                m_score_buffer + dispatched);
        }
        if (!chunk_ok) { ok = 0; break; }
        dispatched += n;
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
                               slot.tors_step_size,
                               slot.torsion_scale_factors);

    /* Convert scaled DOF vector to molecule coordinates */
    m_minimizer->vector_to_dockmol(*slot.m_tmpMol, new_vec,
                                   slot.torsions, slot.bond_vectors);

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
        static const bool dbg = (getenv("DOCK_POOL_DEBUG") != NULL);
        if (dbg && slot.id < 4) {
            fprintf(stderr, "POOLDBG init id=%d y=", slot.id);
            for (int vi = 0; vi < N + 1; vi++) fprintf(stderr, " %.5f", y[vi]);
            fprintf(stderr, "\n");
        }
        rank_vertices(y, N + 1, slot.ihi, slot.inhi, slot.ilo);
        compute_centroid_reflect(p, N, slot.ihi,
                                  slot.centroid.data(),
                                  slot.reflect_v.data());
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
                               slot.tors_step_size,
                               slot.torsion_scale_factors);
    m_minimizer->vector_to_dockmol(*slot.m_tmpMol, new_vec,
                                   slot.torsions, slot.bond_vectors);
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
    if (getenv("DOCK_POOL_DEBUG") != NULL && slot.id < 4) {
        fprintf(stderr, "POOLDBG conv id=%d it=%d cyc=%d stg=%d delta=%.6f ylo=%.5f yhi=%.5f done=%d\n",
                slot.id, slot.iteration, slot.current_cycle, slot.stage,
                delta, slot.y[slot.ilo], slot.y[slot.ihi], done);
    }
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
            /* Update ref_mol to the current pose, then restart.
               (fill_slot_from_mol is a no-op here — it requires
               slot.converged — so apply the best vertex inline.) */
            copy_crds(*slot.m_tmpMol, *slot.m_refMol);
            FLOATVec vv(slot.size);
            for (int j = 0; j < slot.size; j++) vv[j] = slot.p[slot.ilo][j];
            FLOATVec nv;
            m_minimizer->scale_vector(nv, vv, slot.trans_step_size,
                                       slot.rot_step_size, slot.tors_step_size,
                                       slot.torsion_scale_factors);
            m_minimizer->vector_to_dockmol(*slot.m_tmpMol, nv,
                                           slot.torsions, slot.bond_vectors);
            copy_crds(*slot.m_mol, *slot.m_tmpMol);
            copy_molecule(*slot.m_refMol, *slot.m_mol);

            /* Reset vertex to zeros (start from current best) */
            for (int vi = 0; vi < slot.size + 1; vi++)
                memset(slot.p[vi], 0, slot.size * sizeof(float));

            /* Build new initial simplex from zeros, continuing the
               pose's deterministic RNG stream (matches the sequential
               path's next cycle draws). */
            FLOATVec zero_vertex(slot.size, 0.0f);
            m_minimizer->set_local_rng_state(slot.rng_state);
            build_initial_simplex(m_minimizer, zero_vertex, slot.p, slot.y, slot.size);
            slot.rng_state = m_minimizer->local_rng_state();

            /* Reset iteration state */
            slot.iteration = 0;
            slot.phase = SlotPhase::INIT;
            slot.ihi = 0;
            slot.inhi = 0;
            slot.ilo = 0;
            return false;  /* not done yet */
        }
    }

    /* All cycles of this stage done.  Transition to stage B (the full
       minimization, mirroring minimize_flexible_growth's second minimize()
       call) if one was requested.  Stage B starts from stage A's best
       pose with a fresh zero vertex, exactly like the sequential path's
       second minimize() call. */
    if (slot.stage == 0 && slot.max_cycles_B > 0) {
        /* Restore best-across-cycles (matches Minimizer::minimize's
           best_cycle_mol restore), then apply it to mol + refMol. */
        if (slot.has_best) {
            for (int j = 0; j < slot.size; j++)
                slot.p[slot.ilo][j] = slot.best_vertex[j];
            slot.y[slot.ilo] = slot.best_score;
        }
        copy_crds(*slot.m_tmpMol, *slot.m_refMol);
        FLOATVec vv(slot.size);
        for (int j = 0; j < slot.size; j++) vv[j] = slot.p[slot.ilo][j];
        FLOATVec nv;
        m_minimizer->scale_vector(nv, vv, slot.trans_step_size,
                                   slot.rot_step_size, slot.tors_step_size,
                                   slot.torsion_scale_factors);
        m_minimizer->vector_to_dockmol(*slot.m_tmpMol, nv,
                                       slot.torsions, slot.bond_vectors);
        copy_crds(*slot.m_mol, *slot.m_tmpMol);
        copy_molecule(*slot.m_refMol, *slot.m_mol);

        for (int vi = 0; vi < slot.size + 1; vi++)
            memset(slot.p[vi], 0, slot.size * sizeof(float));

        FLOATVec zero_vertex(slot.size, 0.0f);
        m_minimizer->set_local_rng_state(slot.rng_state);
        build_initial_simplex(m_minimizer, zero_vertex, slot.p, slot.y, slot.size);
        slot.rng_state = m_minimizer->local_rng_state();

        /* Switch to stage B parameters; best-across-cycles tracking
           restarts (the sequential path discards stage A's best once
           stage B runs). */
        slot.stage              = 1;
        slot.max_iterations     = slot.max_iterations_B;
        slot.max_cycles         = slot.max_cycles_B;
        slot.score_converge     = slot.score_converge_B;
        slot.trans_step_size    = slot.trans_step_size_B;
        slot.rot_step_size      = slot.rot_step_size_B;
        slot.tors_step_size     = slot.tors_step_size_B;
        slot.current_cycle      = 0;
        slot.iteration          = 0;
        slot.has_best           = false;
        slot.best_score         = 1e30f;
        slot.phase              = SlotPhase::INIT;
        slot.ihi                = 0;
        slot.inhi               = 0;
        slot.ilo                = 0;
        return false;  /* stage B still running */
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
