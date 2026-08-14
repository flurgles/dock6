// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// dock.cpp
//
// definition of the main function
//
// The pre version 6.4 history is documented in the manual.
// The version 6.4 changes were primarily from SUNY Stony Brook:
// Code modifications made by the group of
// Robert C Rizzo (Stony Brook University) with
// Sudipto Mukherjee (Stony Brook University), and
// Trent E Balius (Stony Brook University);
// as well as Demetri Moustakas.
// They include:
// rep_vdw ligand clash filter, lig_int_nrg in mol2,
// multi-anchor growth tree,
// torsion pre-minimizer during growth of active atoms bugfix,
// clear anchor_positions array in next_anchor(), question tree update,
// DM's orienting bugfix (updated), last anchor correction, and MW & Charge.
//
// The lists of other past and present authors is in the manual.
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// This software is copyrighted, 2004-2023,
// by the DOCK Developers.
//
// The authors hereby grant permission to use, copy, modify, and re-distribute
// this software and its documentation for any purpose, provided
// that existing copyright notices are retained in all copies and that this
// notice is included verbatim in any distributions. No written agreement,
// license, or royalty fee is required for any of the authorized uses.
// Modifications to this software may be distributed provided that
// the nature of the modifications are clearly indicated.
//
// IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY
// FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
// ARISING OUT OF THE USE OF THIS SOFTWARE, ITS DOCUMENTATION, OR ANY
// DERIVATIVES THEREOF, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.  THIS SOFTWARE
// IS PROVIDED ON AN "AS IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE
// NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
// MODIFICATIONS.
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <time.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdint.h>
#include <map>
#include <vector>
#include "stdlib.h"
#include "string.h"
#include "amber_typer.h"
#include "master_conf.h"
#include "library_file.h"
#include "master_score.h"
#include "orient.h"
#include "simplex.h"
#include "bobyqa.h"
#include "steepest_descent.h"
#include "conjugate_gradient.h"
#include "minimizer.h"
#include "trace.h"
#include "conformer_pool.h"
#include "score_dock_gpu.h"
#include "utils.h"
#include "version.h"
#include "filter.h"
//#include <gzstream.h>
#include "gzstream/gzstream.h"

#ifndef __APPLE__
#include <sys/sysinfo.h>
#endif


// Temporary support on cygwin for calculate_simulation_time vs wall_clock_seconds
// https://stackoverflow.com/questions/71324957/cygwin-c-standard-library-does-not-support-time-utc-and-timespec-get
#if __cplusplus > 199711L
  #define TIME_PRECISION
#endif
#ifdef __CYGWIN__
  #undef TIME_PRECISION
#endif

void            dock_new_handler();
void            print_header( bool USE_MPI, int processes );
//LEP - Preprocesor directive determines if the compiler is >CPP11 for certain functionality
double          wall_clock_seconds();

/* ------------------------------------------------------------------ */
/*  GPU virtual-screen batch driver                                    */
/*                                                                     */
/*  Docks up to `window` ligands against the same receptor grid in     */
/*  one batched pass:  the anchor-minimization phase of every ligand   */
/*  in the window shares a single ConformerPool dispatch stream, with  */
/*  each ligand's atom parameters registered at its own LUT slot so    */
/*  every GPU launch can carry candidates of multiple ligands.         */
/*                                                                     */
/*  Collection phase (CPU): per ligand, run anchor/orientation         */
/*  matching and submit_anchor_orientation as usual, but record the    */
/*  prepared ligand + anchors instead of growing immediately.          */
/*  Batch phase: register all windowed ligands in the LUT, run one     */
/*  anchor pool across every collected anchor+ligand pair.             */
/*  Growth phase (CPU per ligand): replay next_anchor/prepare and        */
/*  call grow_peripheral(..., anchors_preminimized=true) so growth     */
/*  consumes the already-minimized anchors without re-minimizing.      */
/* ------------------------------------------------------------------ */

struct VSAnchorSet {
    vector<SCOREMol>    anchors;    /* collected anchor orientations       */
    int                 lig_idx;   /* LUT slot; assigned in batch phase   */
    int                 np;         /* IE pair count (per-anchor capture)  */
    int                *nb_flat;    /* IE pair list snapshot               */
    float              *ie_vdwA;    /* per-atom IE vdW A params snapshot   */
};

/* Deep-copy anchor positions into dst, reusing dst's existing per-mol
   array storage in place: copy_molecule() retains the destination
   arrays whenever the atom counts match, so steady-state is allocation-
   free (mirrors the sequential submit_anchor_orientation reuse pattern).
   Discarding the old contents via clear() without freeing would orphan
   the previous sets (DOCKMol arrays are not destructor-managed). */
static void
vs_deep_copy_anchors(vector<SCOREMol> & dst, const vector<SCOREMol> & src)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i].first = src[i].first;
        copy_molecule(dst[i].second, src[i].second);
    }
}

struct VSWindowJob {
    DOCKMol               mol;      /* prepared ligand (for growth replay) */
    vector<VSAnchorSet>   sets;     /* one entry per completed anchor      */
    int                   serial;   /* library-order index (per-pose RNG)  */
};

static int
gpu_vs_batch_drive(Library_File & c_library, Master_Conformer_Search & c_master_conf,
                   Orient & c_orient, Bump_Filter & c_bmp_score,
                   Master_Score & c_master_score, AMBER_TYPER & c_typer,
                   Filter & c_filter, Minimizer & active_min,
                   bool USE_MPI)
{
    int window = dock_gpu_recommended_batch_size();
    if (window < 1) window = 1;
    /* Host-RAM budget: this 11 GiB Steam Deck runs earlyoom (kills when
       free memory drops below ~3.4%); the collection footprint is roughly
       0.14 GB per in-flight ligand (~2 K orientation snapshots each), so
       the window is capped well below the 128-slot LUT limit.  Make it a
       runtime/compile-time parameter if the host budget differs. */
    const int vs_window_max = 6;
    const int PREP_BATCH = 1;
    if (window > vs_window_max) window = vs_window_max;
    int maxl = dock_gpu_vs_max_ligands();
    if (window > maxl) window = maxl;
    const float ie_soft = c_master_score.primary_score->ie_soft_delta;
    const float ie_cut  = c_master_score.primary_score->ie_vdw_cutoff_sq;
    const bool can_batch_orient =
        c_orient.orient_ligand && c_master_score.use_score &&
        c_master_score.use_primary_score && !c_library.write_orients;

    /* ---- Phase 1 (collection): fill the window with ligand jobs ---- */
    DOCKMol mol;
    vector<VSWindowJob> jobs;
    int total_written = 0;
    int mol_serial = 0;   /* library-order serial for per-pose RNG keys */
    for (;;) {
    jobs.clear();
    int row_next = 0;   /* LUT rows reserved per anchor set, this window */
    while (c_library.get_mol(mol, c_filter.use_database_filter, USE_MPI,
                             c_master_score.amber, c_typer, c_master_score,
                             active_min)) {
        active_min.initialize();              // seed RNG
        mol.prepare_molecule();
        mol.flag_write_solvation = c_library.write_solv_mol2;
        c_typer.prepare_molecule(mol, c_master_score.read_vdw,
                                 c_orient.use_chemical_matching,
                                 c_master_score.use_ph4,
                                 c_master_score.use_volume);
        c_master_conf.initialize_once = true;
        c_master_conf.prepare_molecule(mol);
        if (c_master_conf.c_ag_conf.write_fragment_libraries) continue;

        // Database filter (same shape as original loop)
        if (c_filter.use_database_filter) {
            c_filter.calc_descriptors(mol);
            if (c_filter.fails_filter(mol)) continue;
        }

        VSWindowJob job;
        job.serial = mol_serial++;

        while (c_master_conf.next_anchor(mol)) {
            c_library.num_anchors++;
            c_orient.match_ligand(mol);

            /* Pass 1: stream this anchor's orientations WITHOUT scoring.
               Orientation scoring (a ~1000-score/anchor CPU hot spot) is
               deferred to one batched LUT dispatch after collection. */
            std::vector<float> ori_xyz;
            std::vector<int>   ori_bump;
            std::vector<int>   ori_more;
            while (c_orient.new_next_orientation(mol)) {
                ori_bump.push_back(
                    c_bmp_score.check_anchor_bumps(mol,
                        c_orient.more_orientations()) ? 1 : 0);
                ori_more.push_back(c_orient.more_orientations() ? 1 : 0);
                for (int ai = 0; ai < mol.num_atoms; ai++) {
                    ori_xyz.push_back(mol.x[ai]);
                    ori_xyz.push_back(mol.y[ai]);
                    ori_xyz.push_back(mol.z[ai]);
                }
            }
            const int n_ori = (int)ori_bump.size();
            const int na_o  = mol.num_atoms;
            int lig_row = -1;
            bool batch_ok = false;
            std::vector<float> ori_score(n_ori > 0 ? (size_t)n_ori : 1, 0.0f);
            std::vector<char>  ori_valid((size_t)n_ori, 0);
            if (n_ori > 0 && can_batch_orient && dock_gpu_is_active() &&
                row_next < maxl) {
                /* Reserve a LUT row for THIS anchor set and register the
                   anchor-atom parameters.  np=0: orientation scoring is
                   grid-only, mirroring the CPU path where nb_int is empty
                   (cleared per orientation and never rebuilt during
                   phase-1 collection). */
                lig_row = row_next++;
                float *vdwA_arr = new float[na_o];
                float *vdwB_arr = new float[na_o];
                float *chg_arr  = new float[na_o];
                int   *af_arr   = new int[na_o];
                for (int ai = 0; ai < na_o; ai++) {
                    int type = mol.amber_at_id[ai];
                    vdwA_arr[ai] = c_master_score.primary_score->vdwA[type];
                    vdwB_arr[ai] = c_master_score.primary_score->vdwB[type];
                    chg_arr[ai]  = mol.charges[ai];
                    af_arr[ai]   = mol.atom_active_flags[ai] ? 1 : 0;
                }
                if (!dock_gpu_vs_register_ligand(lig_row, vdwA_arr, vdwB_arr,
                                                 chg_arr, af_arr, NULL,
                                                 NULL, 0, na_o,
                                                 ie_soft, ie_cut))
                    lig_row = -1;
                delete[] vdwA_arr;
                delete[] vdwB_arr;
                delete[] chg_arr;
                delete[] af_arr;

                if (lig_row >= 0) {
                    std::vector<int> pose_lig((size_t)GPU_MAX_BATCH_POSES,
                                              lig_row);
                    batch_ok = true;
                    for (int off = 0; off < n_ori; off += GPU_MAX_BATCH_POSES) {
                        int cnt = n_ori - off;
                        if (cnt > GPU_MAX_BATCH_POSES) cnt = GPU_MAX_BATCH_POSES;
                        if (!dock_gpu_batch_score_vs(
                                &ori_xyz[(size_t)off * na_o * 3], cnt, na_o,
                                pose_lig.data(), &ori_score[off])) {
                            batch_ok = false;
                            break;
                        }
                    }
                    /* Validity: the kernel clamps out-of-bounds atoms, so
                       replicate the CPU out-of-grid rejection exactly
                       (Base_Grid::is_inside_grid_box). */
                    float gminx, gminy, gminz, gmaxx, gmaxy, gmaxz;
                    if (batch_ok &&
                        !dock_gpu_grid_bounds(&gminx, &gminy, &gminz,
                                              &gmaxx, &gmaxy, &gmaxz))
                        batch_ok = false;
                    if (batch_ok) {
                        for (int oi = 0; oi < n_ori; oi++) {
                            const float *p = &ori_xyz[(size_t)oi * na_o * 3];
                            bool inb = true;
                            for (int ai = 0; ai < na_o && inb; ai++) {
                                if (!mol.atom_active_flags[ai]) continue;
                                float x = p[ai * 3];
                                float y = p[ai * 3 + 1];
                                float z = p[ai * 3 + 2];
                                if (!(x > gminx && x < gmaxx &&
                                      y > gminy && y < gmaxy &&
                                      z > gminz && z < gmaxz))
                                    inb = false;
                            }
                            ori_valid[oi] = inb ? 1 : 0;
                        }
                    }
                }
            }

            /* Pass 2 (replay): identical submission logic to the sequential
               loop; validity comes from the batch (or a CPU fallback). */
            for (int oi = 0; oi < n_ori; oi++) {
                float *dst_x = mol.x;
                float *dst_y = mol.y;
                float *dst_z = mol.z;
                const float *p = &ori_xyz[(size_t)oi * na_o * 3];
                for (int ai = 0; ai < na_o; ai++) {
                    dst_x[ai] = p[ai * 3];
                    dst_y[ai] = p[ai * 3 + 1];
                    dst_z[ai] = p[ai * 3 + 2];
                }
                if (!ori_bump[oi]) continue;
                if (c_master_conf.method == 1) {
                    c_master_score.primary_score->nb_int.clear();
                }
                c_library.num_orients++;
                bool valid_orient;
                if (can_batch_orient && batch_ok && lig_row >= 0) {
                    valid_orient = (ori_valid[oi] != 0);
                    if (valid_orient)
                        mol.current_score = ori_score[oi];
                } else if (can_batch_orient) {
                    valid_orient =
                        c_master_score.compute_primary_score(mol);
                } else {
                    valid_orient = false;
                }
                if (valid_orient || !ori_more[oi]) {
                    if (c_master_conf.submit_anchor_orientation(mol,
                                                    (bool)ori_more[oi])) {
                        /* anchor complete — snapshot its positions */
                        VSAnchorSet as;
                        vs_deep_copy_anchors(as.anchors,
                            c_master_conf.c_ag_conf.anchor_positions);
                        as.lig_idx = lig_row;
                        /* Capture the per-anchor IE pair table NOW:
                           nb_int is cleared per orientation and never rebuilt
                           during phase-1 scoring (the covalent-neighbor
                           structure is untouched, so the method-1 pair rule
                           from Base_Score::initialize_internal_energy gives
                           the exact table the CPU anchor minimization uses). */
                        as.np = 0;
                        as.nb_flat = NULL;
                        as.ie_vdwA = NULL;
                        if (!as.anchors.empty() &&
                            c_master_conf.method == 1) {
                            std::vector<INTPair> pairs;
                            for (int a1 = 0; a1 < mol.num_atoms - 1; a1++)
                                for (int a2 = a1 + 1; a2 < mol.num_atoms; a2++)
                                    if (mol.atom_segment_ids[a1] !=
                                            mol.atom_segment_ids[a2] &&
                                        mol.get_bond(a1, a2) == -1 &&
                                        !mol.atoms_are_one_three(a1, a2) &&
                                        !mol.atoms_are_one_four(a1, a2))
                                        pairs.push_back(INTPair(a1, a2));
                            int np = (int)pairs.size();
                            if (np > 0) {
                                as.np = np;
                                as.nb_flat = new int[np * 2];
                                for (int pi = 0; pi < np; pi++) {
                                    as.nb_flat[pi * 2]     = pairs[pi].first;
                                    as.nb_flat[pi * 2 + 1] = pairs[pi].second;
                                }
                                int na = as.anchors[0].second.num_atoms;
                                DOCKMol & amol = as.anchors[0].second;
                                as.ie_vdwA = new float[na];
                                /* primary_score->ie_vdwA is not allocated
                                   until grow_periphery's initialize_internal_energy
                                   call, so recompute it from the anchor's own
                                   radii/well-depths — the same formula as
                                   Base_Score::initialize_internal_energy. */
                                float ie_att = c_master_score.primary_score
                                                   ->ie_att_exp;
                                float ie_rep = c_master_score.primary_score
                                                   ->ie_rep_exp;
                                for (int ai = 0; ai < na; ai++) {
                                    as.ie_vdwA[ai] = (float)sqrt(
                                        amol.amber_at_well_depth[ai] *
                                        (ie_att / (ie_rep - ie_att)) *
                                        pow(2.0 * amol.amber_at_radius[ai],
                                            ie_rep));
                                }
                            }
                        }
                        job.sets.push_back(std::move(as));
                    }
                }
            }
        }
        if (!job.sets.empty()) {
            copy_molecule(job.mol, mol);
            jobs.push_back(std::move(job));
        }
        if ((int)jobs.size() >= window) break;
    }
    if (jobs.empty()) break;

    /* ---- Phase 2 (batch): minimize ALL window anchors in ONE pool ---- */
    /* LUT rows were reserved per anchor set during collection (as.lig_idx).
       Only sets that could not get a row (orientation batching disabled, or
       the 128-row budget exhausted) are assigned here: they share their
       ligand's first reserved row when one exists, else a fresh per-ligand
       row (≤ window ligands, always within budget). */
    {
        int fallback_rows = 0;
        for (size_t i = 0; i < jobs.size(); i++) {
            int first_reserved = -1;
            for (size_t k = 0; k < jobs[i].sets.size(); k++)
                if (jobs[i].sets[k].anchors.empty()) continue;
                else if (jobs[i].sets[k].lig_idx >= 0) {
                    first_reserved = jobs[i].sets[k].lig_idx;
                    break;
                }
            int fresh_row = -1;
            for (size_t k = 0; k < jobs[i].sets.size(); k++) {
                VSAnchorSet & as = jobs[i].sets[k];
                if (as.anchors.empty() || as.lig_idx >= 0) continue;
                if (first_reserved >= 0) {
                    as.lig_idx = first_reserved;
                } else {
                    /* No row was reserved for this ligand (batching
                       disabled, or budget exhausted before its first
                       set).  Give ONE fresh row to the whole ligand —
                       a per-set counter would wrap across ligands and
                       let a later ligand overwrite this one's row.
                       window <= maxl guarantees the fresh counter can
                       never exhaust the LUT. */
                    if (fresh_row < 0) fresh_row = fallback_rows++;
                    as.lig_idx = fresh_row;
                }
            }
        }
    }

    /* Register every windowed anchor set in the GPU LUT before batching.
       Mirrors the registration block in grow_periphery:  per-atom grid
       vdw/es params from the VDW parm lookup and the atom charges; IE
       params come from the per-anchor snapshot captured at collection
time (the same pair list the sequential path would have fed to
        dock_gpu_set_ligand_ie), plus the run's IE soft-delta and cutoff. */
    for (size_t i = 0; i < jobs.size(); i++) {
        for (size_t k = 0; k < jobs[i].sets.size(); k++) {
        VSAnchorSet & as = jobs[i].sets[k];
        if (as.anchors.empty()) continue;
        DOCKMol & amol = as.anchors[0].second;
        int na = amol.num_atoms;
        float *vdwA_arr = new float[na];
        float *vdwB_arr = new float[na];
        float *chg_arr  = new float[na];
        int   *af_arr   = new int[na];
        for (int ai = 0; ai < na; ai++) {
            int type = amol.amber_at_id[ai];
            vdwA_arr[ai] = c_master_score.primary_score->vdwA[type];
            vdwB_arr[ai] = c_master_score.primary_score->vdwB[type];
            chg_arr[ai]  = amol.charges[ai];
            af_arr[ai]   = amol.atom_active_flags[ai] ? 1 : 0;
        }
        int np = as.np;
        int *nb_flat = as.nb_flat;
        float *ie_vdwA_arr = as.ie_vdwA;
        dock_gpu_vs_register_ligand(as.lig_idx, vdwA_arr, vdwB_arr,
                                    chg_arr, af_arr, ie_vdwA_arr,
                                    nb_flat, np, na, ie_soft, ie_cut);
        delete[] vdwA_arr;
        delete[] vdwB_arr;
        delete[] chg_arr;
        delete[] af_arr;
        }
    }

    /* Anchor minimization: batch pool with per-pose deterministic RNG. */
    {
        ConformerPool anchor_pool(GPU_POOL_BATCH_MAX, &active_min, true,
                                  active_min.simplex_mode,
                                  active_min.simplex_crossover);
        for (size_t i = 0; i < jobs.size(); i++) {
            VSWindowJob & job = jobs[i];
            for (size_t k = 0; k < job.sets.size(); k++) {
                VSAnchorSet & as = job.sets[k];
                if (!active_min.use_min_rigid_anchor) continue;
                for (size_t a = 0; a < as.anchors.size(); a++) {
                    while (anchor_pool.active_count() >= anchor_pool.capacity()) {
                        anchor_pool.step();
                        anchor_pool.poll();
                    }
                    FLOATVec vertex;
                    for (int iv = 0; iv < 6; iv++) vertex.push_back(0.0f);
                    SimplexStage astage;
                    astage.max_iterations = active_min.anchor_min_max_iterations;
                    astage.max_cycles     = active_min.anchor_min_max_cycles;
                    astage.score_converge = active_min.anchor_min_score_converge;
                    astage.trans_step_size = active_min.anchor_min_trans_step_size;
                    astage.rot_step_size  = active_min.anchor_min_rot_step_size;
                    astage.tors_step_size = active_min.anchor_min_tors_step_size;
                    anchor_pool.add(&as.anchors[a].second, vertex,
                                    astage,
                                    SimplexStage{0, 0, 0.0f, 0.0f, 0.0f, 0.0f},
                                    active_min.anchor_min_cycle_converge,
                                    false, 0.0f, nullptr, nullptr,
                                    as.lig_idx,
                                    as.anchors[a].second.num_atoms,
                                    seed_key((unsigned)job.serial,
                                             (unsigned)k, (unsigned)a,
                                             0x51u, 0x52u));
                }
            }
        }
        while (!anchor_pool.idle()) {
            anchor_pool.step();
            anchor_pool.poll();
        }
    }

/* ---- Phase 3 (growth): batch-scheduled replay, anchors preminimized ---- */
    /* All (job,set) rounds share ONE VS-mode pool.  Rounds are prepped
       (torsion drive + clash/bump + pool.add) whenever pool capacity
       allows; NM iterations run for every ligand's slots in the same GPU
       dispatches; each round's second pass (VS grid+IE scoring, prune,
       seed rebuild) runs as soon as its slots converge.  This keeps the
       GPU busy with mixed-ligand batches while the CPU drives/clashes/
       prunes the other ligands' rounds. */
    std::vector<VSGrowState> rows;
    std::vector<std::vector<int> > job_rows(jobs.size());
    VSGrowState base;   /* parked per-job ligand state (prepare_molecule) */
    ConformerPool grow_pool(GPU_POOL_BATCH_MAX, &active_min, true,
                            active_min.simplex_mode,
                            active_min.simplex_crossover);
    for (size_t i = 0; i < jobs.size(); i++) {
        VSWindowJob & job = jobs[i];
        /* Rebuild the AG segment/anchor state for THIS job's ligand —
           the collection loop left it pointing at the last ligand seen. */
        copy_molecule(mol, job.mol);
        c_master_conf.c_ag_conf.dock_mol_serial = job.serial;
        c_master_conf.c_ag_conf.prepare_molecule(mol);
        c_master_conf.c_ag_conf.grow_win_park(base);
        for (size_t k = 0; k < job.sets.size(); k++) {
            /* Rebuild layers for anchor k of this ligand, matching what
               next_anchor() would have built in the sequential path. */
            c_master_conf.c_ag_conf.grow_win_restore(base);
            c_master_conf.c_ag_conf.setup_growth_anchor(k);
            /* restore the anchor positions collected for this job */
            vs_deep_copy_anchors(c_master_conf.c_ag_conf.anchor_positions,
                                 job.sets[k].anchors);
            rows.push_back(VSGrowState());
            VSGrowState & g = rows.back();
            g.route = (int)rows.size() - 1;
            g.lig_idx = job.sets[k].lig_idx;
            c_master_conf.c_ag_conf.grow_win_init(g, c_master_score,
                                                  active_min, c_bmp_score);
            c_master_conf.c_ag_conf.grow_win_park(g);
            job_rows[i].push_back(g.route);
        }
        c_master_conf.c_ag_conf.grow_win_restore(base);
    }

    /* scheduler loop: prep adds while capacity allows, then step the
       shared pool; converged slots unlock their round's second pass.
       Rows are swapped into the AG only when they have CPU work
       (adding or a finished drain); pure drain phases run GPU-only. */
    {
        int cur_swapped = -1;   /* row index currently parked in the AG */
        bool any_pending = true;
        size_t rr = 0;          /* round-robin cursor: one row's CPU work
                                   per iteration so cost stays O(1) in rows */
        while (any_pending) {
            bool any_active = !grow_pool.idle();
            bool any_gpu2 = false;
            for (size_t r = 0; r < rows.size(); r++)
                if (rows[r].gpu2_pending) { any_gpu2 = true; break; }
            if (any_gpu2) {
                /* GPU2 screens are stream-ordered on the secondary stream:
                   sync at the top of the iteration so the row block (prep
                   LUT re-register / finish consume) sees valid results and
                   the LUT is quiescent. */
                dock_gpu_batch_score_sync2();
            }
            bool any_round_open = false;
            if (any_active) {
                /* Enqueue this round's GPU batches first (non-blocking):
                   the row bookkeeping below then overlaps the kernels. */
                grow_pool.step_enqueue();
            }
            {
                /* find the next row with CPU work, starting at rr */
                size_t k = rows.size();
                for (size_t s = 0; s < rows.size(); s++) {
                    size_t r = (rr + s) % rows.size();
                    if (rows[r].done) continue;
                    any_round_open = true;
                    bool needs_state = rows[r].adding ||
                        (rows[r].inflight == 0 && !rows[r].drain_done);
                    if (!needs_state) continue;   /* pure drain: GPU-only */
                    k = r;
                    break;
                }
                rr = (k + 1) % rows.size();
                if (k < rows.size()) {
                    size_t r = k;
                    if (cur_swapped != (int)r) {
                        if (cur_swapped >= 0)
                            c_master_conf.c_ag_conf.grow_win_park(rows[cur_swapped]);
                        c_master_conf.c_ag_conf.grow_win_restore(rows[r]);
                        cur_swapped = (int)r;
                    }
                    if (rows[r].adding)
                        c_master_conf.c_ag_conf.grow_win_prep(rows[r], grow_pool,
                                                              c_master_score,
                                                              active_min,
                                                              c_bmp_score);
                    if (!rows[r].adding && rows[r].inflight == 0 &&
                        !rows[r].drain_done)
                        c_master_conf.c_ag_conf.grow_win_finish(rows[r],
                                                                c_master_score,
                                                                active_min,
                                                                c_bmp_score);
                }
            }
            if (any_active) {
                /* Batch more rows into the pool this round: the extra
                   slots make each GPU dispatch larger, so the GPU stays
                   busier while the host keeps doing row bookkeeping. */
                for (int pb = 1; pb < PREP_BATCH && any_round_open; pb++) {
                    size_t k = rows.size();
                    for (size_t s = 0; s < rows.size(); s++) {
                        size_t r = (rr + s) % rows.size();
                        if (rows[r].done) continue;
                        bool needs_state = rows[r].adding ||
                            (rows[r].inflight == 0 && !rows[r].drain_done);
                        if (!needs_state) continue;
                        k = r;
                        break;
                    }
                    rr = (k + 1) % rows.size();
                    if (k == rows.size()) { any_round_open = false; break; }
                    size_t r = k;
                    if (cur_swapped != (int)r) {
                        if (cur_swapped >= 0)
                            c_master_conf.c_ag_conf.grow_win_park(rows[cur_swapped]);
                        c_master_conf.c_ag_conf.grow_win_restore(rows[r]);
                        cur_swapped = (int)r;
                    }
                    if (rows[r].adding)
                        c_master_conf.c_ag_conf.grow_win_prep(rows[r], grow_pool,
                                                              c_master_score,
                                                              active_min,
                                                              c_bmp_score);
                    if (!rows[r].adding && rows[r].inflight == 0 &&
                        !rows[r].drain_done)
                        c_master_conf.c_ag_conf.grow_win_finish(rows[r],
                                                                c_master_score,
                                                                active_min,
                                                                c_bmp_score);
                }
            }
            if (any_active) {
                grow_pool.step_finish();
                std::vector<SimplexSlot*> done_slots = grow_pool.poll();
                for (size_t s = 0; s < done_slots.size(); s++) {
                    int tag = (int)(intptr_t)done_slots[s]->user_data;
                    int r = tag >> 12;
                    if (r >= 0 && r < (int)rows.size()) rows[r].inflight--;
                }
            }
            /* continue while any round is open or the pool still drains */
            any_pending = any_round_open || any_active || any_gpu2;
        }
        if (cur_swapped >= 0)
            c_master_conf.c_ag_conf.grow_win_park(rows[cur_swapped]);
    }

    /* submission: same per-job, per-set loop as the serial path */
    for (size_t i = 0; i < jobs.size(); i++) {
        VSWindowJob & job = jobs[i];
        for (size_t jr = 0; jr < job_rows[i].size(); jr++) {
            VSGrowState & g = rows[job_rows[i][jr]];
            c_master_conf.c_ag_conf.grow_win_restore(g);
            int nconf = 0;
            while (c_master_conf.next_conformer(mol)) {
                /* nb_int/ie_vdwA are global score state rebuilt per row
                   (grow_win_init) — by submission time they belong to the
                   last-processed row, not this ligand.  Rebuild from the
                   final pose so minimization and the final report use the
                   correct pair list. */
                c_master_score.primary_score->initialize_internal_energy(mol);
                active_min.minimize_final_pose(mol, c_master_score, c_typer);
                /* The GPU growth path sets current_score/internal_energy
                   directly and never refreshes current_data (the ranked
                   text).  Recompute the final pose's primary score once so
                   the reported grid score matches the stored score. */
                c_master_score.compute_primary_score(mol);
                c_library.submit_scored_pose(mol, c_master_score, active_min);
                nconf++;
            }
            c_master_conf.c_ag_conf.grow_win_park(g);
        }
        c_library.submit_conformations(c_master_score);
        c_library.sort_write(false, USE_MPI, c_master_score, active_min);
        c_library.ranked_poses.clear();
    }
    /* The EOF-triggered ranked write inside get_mol() fires during the
       NEXT window's collection, when ranked_list still holds THIS window's
       contributions is not safe to rely on — flush the ranked list per
       window and clear it so no window is written twice. */
    c_library.write_ranked_ligands(false, c_master_score);
    c_library.ranked_list.clear();
    /* Free this window's anchor snapshots (DOCKMol arrays are not
       destructor-managed) before collecting the next window. */
    for (size_t i = 0; i < jobs.size(); i++) {
        VSWindowJob & job = jobs[i];
        job.mol.clear_molecule();
        for (size_t k = 0; k < job.sets.size(); k++) {
            for (size_t a = 0; a < job.sets[k].anchors.size(); a++)
                job.sets[k].anchors[a].second.clear_molecule();
            if (job.sets[k].nb_flat) delete[] job.sets[k].nb_flat;
            if (job.sets[k].ie_vdwA) delete[] job.sets[k].ie_vdwA;
        }
    }
    }
    return total_written;
}
#ifdef TIME_PRECISION
inline double   calculate_simulation_time(timespec t_start, timespec t_end);
#endif
using namespace std;


/************************************************/
int
main(int argc, char **argv)
{

#ifndef __APPLE__
    // set up memory info
    struct sysinfo memInfo;
    sysinfo (&memInfo);
#endif


    // set the function that will be called if new fails.
    std::set_new_handler(dock_new_handler);

    // synchronize C++ stream io and C stdio.
    ios::sync_with_stdio();

#ifdef TRACE
    Trace::traceOn();
#else
    Trace::traceOff();
#endif
    Trace trace( "::main" );

    // DOCK Classes
    // DOCK does not use constructors, but each class has an initialize member.
    // Note that the compiler-generated default constructors effectively
    // do nothing; so that these objects should be considered uninitialized
    // until their initialize members are called below.
    Parameter_Reader         c_parm;
    Library_File             c_library;
    Orient                   c_orient;
    Master_Conformer_Search  c_master_conf;
    Bump_Filter              c_bmp_score;
    Simplex_Minimizer        c_simplex;
    BOBYQA_Minimizer        c_bobyqa;
    Steepest_Descent_Minimizer c_sd;
    Conjugate_Gradient_Minimizer c_cg;
    Minimizer              *active_min = nullptr;
    Master_Score             c_master_score;
    AMBER_TYPER              c_typer;
    Filter                   c_filter;
    DOCKMol                  mol;

    // Local vars
    ofstream        outfile;
    streambuf      *original_cout_buffer;
    char            fname[500];

    bool            USE_MPI;
    // Preprocessor macro MPI is controlled by the platform configuration
    // which is specified during installation.
#ifdef BUILD_DOCK_WITH_MPI
    USE_MPI = true;
#else
    USE_MPI = false;
#endif

    // Initialize mpi and determine if proper number of processors has been called.
    if (USE_MPI){
        // mpi_init requires pointers to argc and argv.
        USE_MPI = c_library.initialize_mpi(&argc, &argv);
    }

    // Direct the output to a file or stdout
    if (check_commandline_argument(argv, argc, "-o") != -1) {
        if (USE_MPI) {
            if (c_library.rank > 0) {
                sprintf(fname, "%s.%d",
                        parse_commandline_argument(argv, argc, "-o").c_str(),
                        c_library.rank);
            } else {
                sprintf(fname, "%s",
                        parse_commandline_argument(argv, argc, "-o").c_str());
            }
        } else {
            sprintf(fname, "%s",
                    parse_commandline_argument(argv, argc, "-o").c_str());
        }
        // Redirect non error output to the -o file.
        // C stdio.
        freopen( fname, "a", stdout );
        // C++ stream io.
        outfile.open(fname);
        original_cout_buffer = cout.rdbuf();
        cout.rdbuf(outfile.rdbuf());

    } else if (USE_MPI) {
        cout << "Error: DOCK must be run with the -o outfile option under MPI"
             << endl;
        c_library.finalize_mpi();
        exit(0);
    }


    // /////////////////////////////////////////////////////////////////////////
    // Begin timing
#ifndef TIME_PRECISION
    double          start_time = wall_clock_seconds();
    //int             wall_clock_nseconds();
#endif
#ifdef TIME_PRECISION
    struct timespec start_time;
    timespec_get(&start_time, TIME_UTC);
#endif
    print_header( USE_MPI, c_library.comm_size );

    // Read input parameters
    c_parm.initialize(argc, argv);
    c_master_conf.input_parameters(c_parm);
    if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3 )
    //if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3  || c_master_conf.method == 5)
        c_library.input_parameters_input(c_parm);
    c_filter.input_parameters(c_parm);   //dbfilter code
    c_orient.input_parameters(c_parm);
    c_bmp_score.input_parameters(c_parm);
    c_master_score.input_parameters(c_parm);
    // Read minimizer_type once, then only configure the selected minimizer.
    // This avoids consuming shared params (minimize_ligand, etc.) from params_in
    // for the unused minimizer, which would produce spurious warnings.
    c_simplex.minimizer_type = c_parm.query_param("minimizer_type", "simplex", "simplex bobyqa steepest_descent conjugate_gradient");
    if (c_simplex.minimizer_type == "bobyqa") {
        c_bobyqa.input_parameters(c_parm, c_master_conf.flexible_ligand, c_master_conf.genetic_algorithm, c_master_conf.denovo_design, c_master_score);
        active_min = static_cast<Minimizer*>(&c_bobyqa);
    } else if (c_simplex.minimizer_type == "steepest_descent") {
        c_sd.input_parameters(c_parm, c_master_conf.flexible_ligand, c_master_conf.genetic_algorithm, c_master_conf.denovo_design, c_master_score);
        active_min = static_cast<Minimizer*>(&c_sd);
    } else if (c_simplex.minimizer_type == "conjugate_gradient") {
        c_cg.input_parameters(c_parm, c_master_conf.flexible_ligand, c_master_conf.genetic_algorithm, c_master_conf.denovo_design, c_master_score);
        active_min = static_cast<Minimizer*>(&c_cg);
    } else {
        c_simplex.input_parameters(c_parm, c_master_conf.flexible_ligand, c_master_conf.genetic_algorithm, c_master_conf.denovo_design, c_master_score);
        active_min = static_cast<Minimizer*>(&c_simplex);
    }



    // check parms compatablity
    if (c_master_conf.method == 3){ // if covalent some minimizer parameters are not compatable. 
         if (active_min->use_min_rigid_anchor){
             cout << "min anchor must be turned off for covalent" << endl;
             active_min->use_min_rigid_anchor = false;
             exit(0);
         }
         if (active_min->flex_min_max_iterations > 0.0){
             cout << "flex_min_max_iterations must be set to 0.0 for covalent" << endl;
             active_min->flex_min_max_iterations = 0.0;
             exit(0);
         }
    } 
     //Put a check to make sure denovo/ga and MPI are not specified together
     //Currently using de novo/ga under MPI causes weird things to happen
    if (USE_MPI && c_master_conf.method ==2){
        cout << "\nError: DOCK does not support the DOCK_DN  option under MPI"
             << endl;
        c_library.finalize_mpi();
        exit(0);
    }
    if (USE_MPI && c_master_conf.method ==3){
        cout << "\nError: DOCK does not support the GA option under MPI"
             << endl;
        c_library.finalize_mpi();
        exit(0);
    }
    if (USE_MPI && c_master_conf.method ==5){
        cout << "\nError: DOCK does not support the HDB option under MPI, for now"
             << endl;
        c_library.finalize_mpi();
        exit(0);
    }


    if (c_master_conf.flexible_ligand   || active_min->minimize_ligand 
        || c_filter.use_database_filter || c_orient.orient_ligand 
        || c_bmp_score.bump_filter      || c_library.calc_rmsd)
        c_master_score.read_vdw = true;

    // we should also read in the vdw.defn if we are calculating xlogp in the 
    // database filter i.e. if c_filter.use_database_filter is true. For now,
    // the xlogp code has been removed, so this is no longer needed
    
    //added this if statement and line so that we have the information from these parm files available for delta_max, the else portion was what was there originally
    if ( (c_master_conf.method==4 && c_master_conf.c_ga_recomb.use_limit_max_change) ){
        bool use_volume = true;
        bool use_ph4 = true;
        bool use_chemical_matching = true;
        bool use_vdw = true;
        c_master_score.use_ph4 = use_ph4;
        c_master_score.use_volume = use_volume; 
        c_orient.use_chemical_matching = use_chemical_matching;
        c_master_score.read_vdw = use_vdw;
        c_typer.input_parameters(c_parm, use_vdw, use_chemical_matching, use_ph4, use_volume);
        cout << "MODIFYING C_TYPER" << endl;
    }
    else{
        c_typer.input_parameters(c_parm, c_master_score.read_vdw,
                             c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume); 
    }
    //if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3 )
    //if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3  || c_master_conf.method == 5)
    if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3  || c_master_conf.method == 5)
        c_library.input_parameters_output(c_parm, c_master_score, USE_MPI);
    mol.input_parameters(c_parm);

    if (!USE_MPI)
        c_parm.write_params();

    // Exit the program if parameterization fails //
    if (!c_parm.parameter_input_successful()) {
        if (USE_MPI)
            c_library.finalize_mpi();
        if (outfile.is_open()) {
            // return non error output to standard output.
            cout.rdbuf(original_cout_buffer);
            outfile.close();
            fclose(stdout);  // C stdio.
        }
        return 1;               // non-zero return indicates error
    }
    // /////////////////////////////////////////////////////////////////////////
    // Initialization routines
    c_master_conf.initialize();
    c_typer.initialize(c_master_score.read_vdw, c_master_score.read_gb_parm,
                       c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);
    //if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3  || c_master_conf.method == 5)
    if (c_master_conf.method == 0 || c_master_conf.method == 1 ||  c_master_conf.method == 3 )
        c_library.initialize(argc, argv, USE_MPI);
    c_orient.initialize(argc, argv);
    c_bmp_score.initialize();
    c_master_score.initialize_all(c_typer, argc, argv);  //removed reference to flex_min_add_internal: not used any more
    // always initialize both minimizers; the actual minimization method
    // is selected at each minimize_* call via minimizer_type.
    c_simplex.initialize();
    c_bobyqa.initialize();
    c_sd.initialize();
    c_cg.initialize();

#ifdef BUILD_DOCK_WITH_MPI
    if ((USE_MPI) && (c_library.rank > 0)) {
        cout << "DOCK is currently running on ";
        char klient[ MPI_MAX_PROCESSOR_NAME ];
        int length;
        MPI_Get_processor_name(klient, &length);
        for ( int i = 0; i < length; ++i )
            cout << klient[i];
        cout << endl;
    }
#endif

    //fstream main_dock_loop_anchors;
    //main_dock_loop_anchors.open ("unmin_anchors.mol2", fstream::out|fstream::app);

    // /////////////////////////////////////////////////////////////////////////
    // Main loop
    //if ligand has been docked
    //      write out and read in next ligand
    //if entire docking is complete
    //      perform final analysis and write out
    //else
    //      read in ligand

    // If you are doing de novo growth, enter this function
    if (c_master_conf.method == 2) {
        //if (c_master_conf.c_dn_build.simple_build_flag)
        //    c_master_conf.c_dn_build.simple_build(c_master_score, c_simplex, c_typer);
        //else
            c_master_conf.c_dn_build.build_molecules(c_master_score, *active_min, c_typer, c_orient);

    } else if (c_master_conf.method == 3) { // this is covalent
    while (c_library.get_mol(mol,false, USE_MPI, c_master_score.amber, c_typer, c_master_score, *active_min)) { 
        // If MPI is used this is done on the compute nodes.
        // filtering must be done here because it needs all prep for docking
        // before the mols can be eliminated.
 
        // keep track of time for individual molecules
        double          mol_start_time = wall_clock_seconds();
        //int             mol_start_ntime = wall_clock_nseconds();

        //seed random number generators
        active_min->initialize();

        //parse ligand atoms into child lists
        mol.prepare_molecule();
        mol.flag_write_solvation = c_library.write_solv_mol2;

        //label ligand atoms with proper vdw, bond, and chem
        //types
        c_typer.prepare_molecule(mol, c_master_score.read_vdw,
                                 c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);

        //parse ligand into rigid and flexible portions
        c_master_conf.prepare_molecule(mol);
 
        // Writing fragment libraries for denovo is done inside c_master_conf.prepare_molecule, 
        // so write some frags then continue to the next molecule
        if (c_master_conf.c_ag_conf.write_fragment_libraries){ continue; }

        // set this bool to true of every new ligand 
        // this is for rigid sampling
        // this ensures that the internal energy is only 
        // initialized once per ligand.
        c_master_conf.initialize_once = true;

        // Database Filter code
        //filter molecules by descriptors
        if (c_filter.use_database_filter)
        {
            //calculate descriptors for the ligand
            // c_filter.calc_descriptors(mol);
            // descriptors are now computed & printed in amber_typer.cpp

            //print out the descriptors
            // cout << c_filter.get_descriptors(mol);

            if (c_filter.fails_filter(mol))  {
                 //ligand failed the filter
                 //move to the next ligand to be docked
                 cout << "\n" "-----------------------------------" "\n";
                 cout << "Molecule: " << mol.title << "\n\n";
                 cout << mol.current_data << endl;
                 continue;
            }
        }

        // we want to run it once to store the orignal
        c_master_conf.next_anchor(mol);
        c_orient.match_ligand_covalent(mol,1.8,1.8,PI,false,false,true); // set the original molecule to the start
        c_master_conf.c_cg_conf.current_anchor = 0;  // reset to start at one.  
        //transform the ligand to covalent bond
        //c_orient.match_ligand_covalent(mol,c_master_conf.c_cg_conf.bondlength,c_master_conf.c_cg_conf.bondlength2,c_master_conf.c_cg_conf.angleval*(PI/180.0));
        //c_master_conf.next_anchor(mol);
        //cout << c_master_conf.c_cg_conf.bondlength << " " << c_master_conf.c_cg_conf.blstep << " " << c_master_conf.c_cg_conf.blstop << endl;

        float bl_val,bl_val2,aval;
        //float bl_val  = c_master_conf.c_cg_conf.bondlength;
        bl_val  = c_master_conf.c_cg_conf.bondlength;
        while ( bl_val <= c_master_conf.c_cg_conf.blstop) {
            //cout << "for debuging ... bl_val  = " << bl_val << endl;
            //float bl_val2 = c_master_conf.c_cg_conf.bondlength2;
            bl_val2 = c_master_conf.c_cg_conf.bondlength2;
            while ( bl_val2 <= c_master_conf.c_cg_conf.blstop2) {
                //cout << "                 bl_val2 = " << bl_val2 << endl;
                //float aval    = c_master_conf.c_cg_conf.angleval;
                aval    = c_master_conf.c_cg_conf.angleval;
                while ( aval <= c_master_conf.c_cg_conf.avstop) {
                    // keep track of time for individual placements
                    double          mol_start_time = wall_clock_seconds();

                    active_min->initialize();

                    //cout << "for debuging ... bl_val  = " << bl_val << endl;
                    //cout << "                 bl_val2 = " << bl_val2 << endl;
                    //cout << "                 aval    = " << aval << endl;

                    c_master_conf.next_anchor(mol);
                    // ajust = true, orient = true, frist = false
                    c_orient.match_ligand_covalent(mol,bl_val,bl_val2,aval*(PI/180.0),true,true,false);

                    float angle = 0.0;
                    //float inc_angle = (10.0 * (PI/180.0));
                    //float inc_angle = (2.0*PI/c_master_conf.c_cg_conf.num_sample_angles); 
                    float inc_angle = (float(c_master_conf.c_cg_conf.dihideral_step) * (PI/180.0)); 
                    float max_angle = 2.0*PI;
                    //while (angle < 2*PI){ // 360 degrees == 2*PI radians
                    //for (angle =0.0; angle < max_angle; angle=angle+inc_angle){ // 360 degrees == 2*PI radians
                    for (angle =0.0; angle - max_angle < -0.001; angle=angle+inc_angle){ // 360 degrees == 2*PI radians
                        //cout << angle << endl;
                        //cout << max_angle * 180.0 / PI << " " << angle * 180.0 / PI << endl;
                        //cout << max_angle << " " << angle << endl;
                        //cout << (350.0*(PI/180.0)) << endl;
                        // rotate about the covalent bond
                        bool tmpflag=true; // more_orients
                        //if (fabs(angle - (350.0*(PI/180.0))) < 0.001) { 
                        if (fabs(angle - (max_angle-inc_angle)) < 0.001) { 
                        //if ((fabs( bl_val - (c_master_conf.c_cg_conf.blstop - c_master_conf.c_cg_conf.blstep))<0.001) and 
                        //   (fabs( bl_val2 - (c_master_conf.c_cg_conf.blstop2 - c_master_conf.c_cg_conf.blstep2))<0.001) and 
                        //   (fabs( aval - (c_master_conf.c_cg_conf.avstop - c_master_conf.c_cg_conf.avstep))<0.001) and  
                        //   (fabs(angle - (max_angle-inc_angle)) < 0.001) ){ 
                             tmpflag=false;
                             //cout << "last angle" << endl;
                        }
                        c_orient.new_next_orientation_covalent(mol,angle);
                            //perform bump check on anchor and filter if fails
                            if (c_bmp_score.check_anchor_bumps(mol, tmpflag)) {
                            //cout << "Entering check_anchor_bumps" << endl; 
                                //Write_Mol2(mol, main_dock_loop_anchors);
                                //anchor minimization for flexible docking is moved to grow_periphery()
                                //score orientation and write out.
                                //submit_orientation returns false only if there is a
                                //problem with the score.
                                //no ligand output is produced outside of this if.
                                if(c_library.submit_orientation(mol, c_master_score, c_orient.orient_ligand) ||
                                   ! tmpflag ){
                                // score the ligands. if outside of grid, returns false. there was an issue if the 
                                // last orient was false then the anchors were not submitted to the growth routine.
                                    //cout << "Entering submit_orientation" << endl; 
                                    //rank orientations but only keep number user cutoff
                                    if(c_master_conf.submit_anchor_orientation(mol, tmpflag)){ 
                                        //cout << "Entering submit_anchor_orientation" << endl; 
                                        //add mol to (orientation) anchor_positions array 
                                        //prune anchors, then perform growth, minimization, and pruning
                                        //until molecule is fully grown
                                        c_master_conf.grow_periphery(c_master_score, *active_min, c_bmp_score);
                                        //for the fully grown conformations, if there are conformations
                                        //remaining
                                        while (c_master_conf.next_conformer(mol)) {
                                                //cout << "Entering while loop next_conformer" << endl; 
                    
                                                //minimize the final pose
                                                active_min->minimize_final_pose(mol, c_master_score, c_typer);
                    
                                                //calculate score and internal
                                                //c_master_score.compute_primary_score(mol); 
                                                //cout << mol.score_text_data << endl;
                                                // add info about angle and bond length to mol2 header
                                                    int FLOAT_WIDTH = 20;
                                                    int STRING_WIDTH = 17 + 19;
                                                    string DELIMITER    = "########## ";
                                               
                                                    ostringstream text;
                                                    //ostringstream blbl2adha ;
                                                    ostringstream blbl2a ;
                                                    //blbl2adha << bl_val << ';' << bl_val2 << ';' << aval << ';' << angle*180/PI ;
                                                    blbl2a << bl_val << ';' << bl_val2 << ';' << aval  ;
                                                    //text << DELIMITER << setw(STRING_WIDTH) << "bl;bl2;a;dha:"      << setw(FLOAT_WIDTH) << fixed << blbl2adha.str()   << endl;
                                                    text << DELIMITER << setw(STRING_WIDTH) << "bl;bl2;a:"      << setw(FLOAT_WIDTH) << fixed << blbl2a.str()   << endl;
                                                    mol.current_data = mol.current_data + text.str();
                                                    //cout << mol.current_data << endl;
                                               
                                                //
                    
                                                //add best scoring pose to list for
                                                //ranking and further analysis
                                                c_library.submit_scored_pose(mol, c_master_score, *active_min);
                                        }
                                        //write out list of final conformations
                                        c_library.submit_conformations(c_master_score);
                                    }
                                }
                            }
                        //angle = angle + (10.0 * (PI/180.0)); // 10 degree incraments
                    }

                    //cout << "\nI AM HERE" << endl;
            
        //print error messages if orienting or growth has failed
        if (c_library.num_orients == 0) {
            double          mol_stop_time = wall_clock_seconds();
//          int             mol_stop_ntime = wall_clock_nseconds();

            cout << "\n" "-----------------------------------" "\n";
            cout << "Molecule: " << mol.title << "\n\n";
            cout << " Elapsed time:\t" << fixed << setprecision(3)
                 << mol_stop_time - mol_start_time << " seconds\n\n";
            //cout << " Elapsed time:\t" << mol_stop_ntime - mol_start_ntime
            //     << " nseconds\n\n";
            cout << " ERROR:  Could not find a valid orientation." << endl;
            cout << "         (For rigid docking the sought orientation is "
                    "the whole ligand;\n"
                    "         for flexible docking the sought orientation is "
                    "the anchor.)\n"
                    "         Confirm that all spheres are inside the grid box"
                    "         \nand that the grid box is big enough to contain "
                    "an orientation.\n";
        } else if (c_library.num_confs == 0) {
            double          mol_stop_time = wall_clock_seconds();
            // int             mol_stop_ntime = wall_clock_nseconds();
            cout << "\n" "-----------------------------------" "\n";
            cout << "Molecule: " << mol.title << "\n\n";
            cout << " Elapsed time:\t" << fixed << setprecision(3)
                 << mol_stop_time - mol_start_time << " seconds\n\n";
            //cout << " Elapsed time:\t" << mol_stop_ntime - mol_start_ntime
            //     << " nseconds\n\n";
            cout << " ERROR:  Could not complete growth." << endl;
            cout << "         Confirm that the grid box is large enough to "
                    "contain the ligand,\n"
                    "         and try increasing max_orientations.\n";
        }
        // End individual molecule timing
        if (c_library.num_orients > 0 && c_library.num_confs > 0) {
            double          mol_stop_time = wall_clock_seconds();
            //int             mol_stop_ntime = wall_clock_nseconds();
            cout << "\n" "-----------------------------------" "\n";
            cout << "Molecule: " << mol.title << "\n\n";
            cout << " Elapsed time for docking:\t" << fixed << setprecision(3)
                 << mol_stop_time - mol_start_time << " seconds\n\n";
            //cout << " Elapsed time for docking:\t" << mol_stop_ntime - mol_start_ntime
            //     << " nseconds\n\n";

        }
                        
                    aval    = aval + c_master_conf.c_cg_conf.avstep;
                }
                bl_val2 = bl_val2 + c_master_conf.c_cg_conf.blstep2;
            }
            bl_val  = bl_val + c_master_conf.c_cg_conf.blstep;

                    //cout << "for debuging ... bl_val  = " << bl_val << endl;
                    //cout << "                 bl_val2 = " << bl_val2 << endl;
                    //cout << "blstop2 = "  << c_master_conf.c_cg_conf.blstop2 << endl;
                    //cout << "                 aval    = " << aval << endl;
        }

    }
  
    // If you are doing genetic algorithm, enter this function
    } else if (c_master_conf.method == 4) {
          c_master_conf.c_ga_recomb.max_breeding(c_master_score, *active_min, c_typer, c_orient);
    
    // If you are doing hierarchical database (HDB) searching enter this function
    } else if (c_master_conf.method == 5) {
          // check if gist is used in combination with HDB search
          // full displacement gist is incompatable.
          if (c_master_score.c_gist.use_primary_score){
             if (c_master_score.c_gist.gist_type == "displace"){
                 cout << " Error... full displacement gist is incompatible with HDB search.\n\tYou can use blurry_displacement or trilinear instead.\n\tExiting the program. " << endl;
                 exit(0);
             }  
          } 
          c_library.db2flag = true;
          c_library.initialize(argc, argv, USE_MPI);
          c_library.initialize_input(); //  TEB 2024/04/11, input from JB 
          // mimic what is done in DOCK3.7 for now this will only work with serial program.
          HDB_Mol db2_data;
          DOCKMol mol_ac; //# all_coords_rigid_seg;
          //igzstream  db2_stream;
          string filename = c_master_conf.c_hdb_conf.db2filename;
          //bool flag_skip_broken = c_master_conf.c_hdb_conf.skip_broken; // if true do not read in broken sets (branches) if false read-in them. 

          string filenamedb2;

          cout << "opening file: " << filename << endl;

          vector<string> list;

          //c_library.read_sdifile("./split_database_file", list);
          c_library.read_sdifile(filename, list);
 
          cout << list.size()<<endl;
          for (int l=0; l<list.size();l++){

               filenamedb2 = list[l];

               
               cout << "opening :" << filenamedb2 <<endl;
               igzstream  db2_stream;
               db2_stream.clear();
               db2_stream.open(filenamedb2.c_str());

               if (db2_stream.fail()) {
                  cout << "\n\nCould not open " << filenamedb2 <<
                          " for reading.  Program will terminate." << endl << endl;
                  exit(0);
               }


               //while (c_library.read_hierarchy_db2( c_master_conf.c_hdb_conf.db2filename, db2_data)){//
               while (c_library.read_hierarchy_db2( db2_stream, db2_data)){
                  double          mol_start_time = wall_clock_seconds();
                  //DOCKMol rigid; //# anchor;
              
                  //mol.clear_molecule();  // this is done in c_hdb_conf.create_mol when allocation happens
                  //mol_ac.clear_molecule();
                  //c_library.read_hierarchy_db2( c_master_conf.c_hdb_conf.db2filename, db2_data); 
                  //ifstream   db2_stream;
                  //c_orient.cliques.clear();
                   
                  active_min->initialize();
                  
                  c_master_conf.c_hdb_conf.all_poses.clear();
              
                  c_master_conf.c_hdb_conf.create_mol(mol,mol_ac,db2_data,0);
              
                  mol.prepare_molecule();
              
                  c_typer.prepare_molecule(mol, c_master_score.read_vdw,
                                         c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);
                  c_master_conf.prepare_molecule(mol);
              
                  c_typer.prepare_molecule(mol_ac, c_master_score.read_vdw,
                                         c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);
                  
                  //copy_molecule(rigid, mol);
                  //cout << "I AM HERE" << endl;
                  c_master_conf.c_hdb_conf.set_rigid_active(mol,db2_data);
                  c_orient.match_ligand(mol);
                  c_library.num_anchors = 1;
              
                  int num = 0;
                  while (c_orient.new_next_orientation(mol, false)){
                    c_master_conf.c_hdb_conf.search(c_master_score,c_orient,c_bmp_score,mol,mol_ac,db2_data,num); 
                    //c_master_conf.c_hdb_conf.set_rigid_active(mol,db2_data);
                    num++;
                  }
              
                  c_library.num_orients = num;
                  
                  //mol_ac.flag_write_solvation = true;
                  //mol.flag_write_solvation = true;
                  mol_ac.flag_write_solvation = c_library.write_solv_mol2;
                  mol.flag_write_solvation = c_library.write_solv_mol2;
                  // sort the poses by energy
                  int number_min = 0;
                  if (c_master_conf.c_hdb_conf.num_per_search < c_master_conf.c_hdb_conf.all_poses.size()) {
                     number_min= c_master_conf.c_hdb_conf.num_per_search;
                  } else{ // 
                     number_min = c_master_conf.c_hdb_conf.all_poses.size();
                  }

                  if (number_min == 0){
                      cout << "Warning... number_min == 0." << endl;
                      //break;
                  }
                  //cout << "Before sorting:  " << c_master_conf.c_hdb_conf.all_poses[0].first << endl;
                  //
                  std::vector<int> top_X(number_min);
                  
                  //c_library.sort_top_X_mol(c_master_conf.c_hdb_conf.all_poses, number_min, top_X);
                  if (! c_library.sort_top_X_mol(c_master_conf.c_hdb_conf.all_poses, number_min, top_X.data())) { 
                      
                      cout << "Warning... no poses found with min score." << endl;
                      //break;
                  }
                  //exit(0);
                  //sort(c_master_conf.c_hdb_conf.all_poses.begin(), c_master_conf.c_hdb_conf.all_poses.end(), less_than_pair);
                  //cout << "Affer sorting:  " << c_master_conf.c_hdb_conf.all_poses[0].first << endl;
                  
                  //Read in db2 file. 
                  //search db2 file and score segments
                  //score viable poses and write out the top scoring poses
                  //
                  //active_min->initialize();

                  //for (int i = 0; i < number_min; i++) {  
                  for (int ii = 0; ii < number_min; ii++) {  
                  //for (int i = 0; i < c_master_conf.c_hdb_conf.all_poses.size(); i++) {  
                         int i = top_X[ii];
                         //cout << i << endl;
                         //mol.flag_write_solvation = true;
                         mol.flag_write_solvation = c_library.write_solv_mol2;
                         //cout << "debug" << i << " " << c_master_conf.c_hdb_conf.all_poses[i].first <<endl;
                         copy_molecule(mol, c_master_conf.c_hdb_conf.all_poses[i].second);
                         //copy_crds(mol, c_master_conf.c_hdb_conf.all_poses[i].second);
                         //c_master_score.compute_primary_score(mol); 
                         //cout << i << " : score before min = " << mol.current_score << endl;  
                         //c_typer.prepare_molecule(mol, c_master_score.read_vdw,
                         //                c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);
                         //c_master_conf.prepare_molecule(mol);
                         active_min->minimize_final_pose(mol, c_master_score, c_typer);
                         c_master_score.compute_primary_score(mol); 
                         //cout << i << " : score affter min = " << mol.current_score << endl;  
                         //mol.score_text_data = mol.score_text_data + mol.hdb_data;
                         mol.current_data = mol.current_data + mol.hdb_data;
                         //add best scoring pose to list for
                         //ranking and further analysis
                         //mol.flag_write_solvation = true;
                         mol.flag_write_solvation = c_library.write_solv_mol2;
                         c_library.submit_scored_pose(mol, c_master_score, *active_min);
                         //write out list of final conformations
                  }

                  //delete top_X;
                  //delete[] top_X;
                  c_library.submit_conformations(c_master_score);
               
                  //cout << "debug all_poses.size() = " << c_master_conf.c_hdb_conf.all_poses.size() << endl;
                  if (c_library.num_orients == 0) {
                      double          mol_stop_time = wall_clock_seconds();
                      cout << "\n" "-----------------------------------" "\n";
                      cout << "Molecule: " << mol.title << "\n\n";
                      cout << " Elapsed time:\t" << mol_stop_time - mol_start_time
                           << " seconds\n\n";
                      cout << " ERROR:  Could not find a valid orientation." << endl;
                      cout << "         (For rigid docking the sought orientation is "
                              "the whole ligand;\n"
                              "         for flexible docking the sought orientation is "
                              "the anchor.)\n"
                              "         Confirm that all spheres are inside the grid box"
                              "         \nand that the grid box is big enough to contain "
                              "an orientation.\n";
                  } else if (c_library.num_confs == 0) {
                      double          mol_stop_time = wall_clock_seconds();
                      cout << "\n" "-----------------------------------" "\n";
                      cout << "Molecule: " << mol.title << "\n\n";
                      cout << " Elapsed time:\t" << mol_stop_time - mol_start_time
                           << " seconds\n\n";
                      cout << " ERROR:  Could not complete growth." << endl;
                      cout << "         Confirm that the grid box is large enough to "
                              "contain the ligand,\n"
                              "         and try increasing max_orientations.\n";
                  }
                  // End individual molecule timing
                  if (c_library.num_orients > 0 && c_library.num_confs > 0) {
                      double          mol_stop_time = wall_clock_seconds();
                      cout << "\n" "-----------------------------------" "\n";
                      cout << "Molecule: " << mol.title << "\n\n";
                      cout << " Elapsed time for docking:\t"
                           << mol_stop_time - mol_start_time << " seconds\n\n";
                  }
                  //c_library.write_scored_poses(USE_MPI, c_master_score);
                  //c_library.sort_write(USE_MPI, c_master_score, *active_min);
                  //c_library.sort_write(USE_FILT,USE_MPI, c_master_score, *active_min);
                  c_library.sort_write(false,USE_MPI, c_master_score, *active_min);
               }
               db2_stream.clear();
               db2_stream.close();
          } // list (of files)
/**/

    // Else if you are doing flexible, rigid, or fixed anchor docking, enter here
    } else {

    if (dock_gpu_is_active() && c_master_conf.method == 1) {
        /* GPU virtual-screen windowed batching */
        gpu_vs_batch_drive(c_library, c_master_conf, c_orient, c_bmp_score,
                           c_master_score, c_typer, c_filter, *active_min,
                           USE_MPI);
    } else {

    int mol_serial_seq = 0;   /* library-order serial for per-pose RNG keys */
    while (c_library.get_mol(mol,c_filter.use_database_filter, USE_MPI, c_master_score.amber, c_typer, c_master_score, *active_min)) { 
        // If MPI is used this is done on the compute nodes.
        // filtering must be done here because it needs all prep for docking
        // before the mols can be eliminated.
 
        // keep track of time for individual molecules
#ifndef TIME_PRECISION
        double          mol_start_time = wall_clock_seconds();
#endif
#ifdef TIME_PRECISION
        struct timespec mol_start_time;
        timespec_get(&mol_start_time, TIME_UTC);
#endif
        //seed random number generators
        active_min->initialize();

        //parse ligand atoms into child lists
        mol.prepare_molecule();
        mol.flag_write_solvation = c_library.write_solv_mol2;

        //label ligand atoms with proper vdw, bond, and chem
        //types
        c_typer.prepare_molecule(mol, c_master_score.read_vdw,
                                 c_orient.use_chemical_matching, c_master_score.use_ph4, c_master_score.use_volume);

        c_master_conf.initialize_once = true;  // this is need for initializing the internal energy fuction for rigid docking and minimization
        //parse ligand into rigid and flexible portions
        c_master_conf.c_ag_conf.dock_mol_serial = mol_serial_seq++;
        c_master_conf.prepare_molecule(mol);
 
        // Writing fragment libraries for denovo is done inside c_master_conf.prepare_molecule, 
        // so write some frags then continue to the next molecule
        if (c_master_conf.c_ag_conf.write_fragment_libraries){ continue; }


        // Database Filter code
        //filter molecules by descriptors
        if (c_filter.use_database_filter)
        {
            //calculate RDKIT-related descriptors for the ligand
            c_filter.calc_descriptors(mol);
            //other descriptors are now computed & printed in amber_typer.cpp
            //print out the descriptors
            //cout << c_filter.get_descriptors(mol);
            #ifdef BUILD_DOCK_WITH_RDKIT
            if (c_filter.fails_filter(mol))  {
                 //ligand failed the filter 
                 mol.fails_filt = true;
                 //move to the next ligand to be docked
                 //cout << "\n" "-----------------------------------" "\n";
                 //cout << "Molecule: " << mol.title << "\n\n";
                 cout << mol.current_data << endl;
                 //continue;
            }else{
                mol.fails_filt = false;
            } 
            #else
            if (c_filter.fails_filter(mol))  {
                 //ligand failed the filter
                 //move to the next ligand to be docked
                 //cout << "\n" "-----------------------------------" "\n";
                 //cout << "Molecule: " << mol.title << "\n\n";
                 cout << mol.current_data << endl;
                 continue;
            }
            #endif
        }

        // sudipto & trent Dec 09, 2008
        // possible problem: Anchors are being generated inside this loop
        // Solution: Generate anchors outside this loop

        //while there is still another anchor fragment to be docked
        // when no anchor & grow is done, this while loop executes only once 
        while (c_master_conf.next_anchor(mol)) {
            trace.note( "Entering while loop next_anchor" );

            c_library.num_anchors++;

            //generate entire list of atom center-sphere center matches
            c_orient.match_ligand(mol);
            
            //transform the ligand to match
            // called only once for singlepoint or minimization only
            while (c_orient.new_next_orientation(mol)) {
                trace.note( "Entering while loop new_next_orientation" );

                //perform bump check on anchor and filter if fails
                trace.boolean("Perform bump_check::", c_bmp_score.check_anchor_bumps(mol, c_orient.more_orientations()));
                if (c_bmp_score.check_anchor_bumps(mol, c_orient.more_orientations())) {
                    trace.note( "Entering check_anchor_bumps if stmt" );

                    //Write_Mol2(mol, main_dock_loop_anchors);

                    //anchor minimization for flexible docking is moved to grow_periphery()

                    //score orientation and write out.
                    //submit_orientation returns false only if there is a
                    //problem with the score.
                    //no ligand output is produced outside of this if.

                    //this ensures that non-bonded pairlist is properly cleared
                    //before scoring orients from a new ligand during flex

                    //if (c_master_conf.method == 1 || c_master_conf.method == 2 || c_master_conf.method == 3 ||  c_master_conf.method == 4){
                    if (c_master_conf.method == 1 ){
                    //if (c_master_conf.method != 0){ // For Rigid, method == 0
                          c_master_score.primary_score->nb_int.clear();
                          trace.note("Clearing the Flex/non-bonded pairlist");
                    }
                    //this ensures that non-bonded pairlist is properly cleared
                    // and rebuilt before scoring orients from a new ligand 
                    //during rigid sampling
                    else {
                          c_master_score.primary_score->nb_int.clear();
                          trace.note("Clearing the Rigid non-bonded pairlist");
                          c_master_conf.initialize_once = true; // reset for internal energy for rigid docking. 
                          c_master_conf.grow_periphery(c_master_score, *active_min, c_bmp_score);
                    }


                    if(c_library.submit_orientation(mol, c_master_score, c_orient.orient_ligand) ||
                       ! c_orient.more_orientations() ){
                    // score the ligands. if outside of grid, returns false. there was an issue if the 
                    // last orient was false then the anchors were not submitted to the growth routine.
                        trace.note( "Entering submit_orientation if stmt" );

                        //rank orientations but only keep number user cutoff
                        if(c_master_conf.submit_anchor_orientation(mol, c_orient.more_orientations())){ 
                            trace.note( "Entering submit_anchor_orientation if stmt" );
                            //add mol to (orientation) anchor_positions array 
                            //prune anchors, then perform growth, minimization, and pruning
                            //until molecule is fully grown
                            c_master_conf.grow_periphery(c_master_score, *active_min, c_bmp_score);

                            //for the fully grown conformations, if there are conformations
                            //remaining
                            while (c_master_conf.next_conformer(mol)) {
                                trace.note( "Entering while loop next_conformer" );

                                //minimize the final pose
                                active_min->minimize_final_pose(mol, c_master_score, c_typer);

                                //calculate score and internal
                                // c_master_score.compute_primary_score(mol); 

                                //add best scoring pose to list for
                                //ranking and further analysis
                                c_library.submit_scored_pose(mol, c_master_score, *active_min);
                            }

                        }
                    }
                }
            }
        }
        //write out list of final conformations
        c_library.submit_conformations(c_master_score);
        
#ifdef TIME_PRECISION
        //print error messages if orienting or growth has failed
            if (c_library.num_orients == 0) {
                //double          mol_stop_time = wall_clock_seconds();
                struct timespec mol_stop_time;
                timespec_get(&mol_stop_time, TIME_UTC);
                double          time_per_mol = calculate_simulation_time(mol_start_time, mol_stop_time);
                cout << "\n" "-----------------------------------" "\n";
                cout << "Molecule: " << mol.title << "\n\n";
                cout << " Elapsed time:\t";
                cout << fixed << setprecision(3) << time_per_mol << " seconds\n\n";
                cout << " ERROR:  Could not find a valid orientation." << endl;
                cout << "         (For rigid docking the sought orientation is "
                        "the whole ligand;\n"
                        "         for flexible docking the sought orientation is "
                        "the anchor.)\n"
                        "         Confirm that all spheres are inside the grid box"
                        "         \nand that the grid box is big enough to contain "
                        "an orientation.\n";
            } else if (c_library.num_confs == 0) {
            //double          mol_stop_time = wall_clock_seconds();
                struct timespec mol_stop_time;
                timespec_get(&mol_stop_time, TIME_UTC);
                double          time_per_mol = calculate_simulation_time(mol_start_time, mol_stop_time);
                cout << "\n" "-----------------------------------" "\n";
                cout << "Molecule: " << mol.title << "\n\n";
                cout << " Elapsed time:\t";
                cout << fixed << setprecision(3) << time_per_mol << " seconds\n\n";
                cout << " ERROR:  Could not complete growth." << endl;
                cout << "         Confirm that the grid box is large enough to "
                        "contain the ligand,\n"
                        "         and try increasing max_orientations.\n";
            }
            // End individual molecule timing
            if (!c_filter.use_database_filter){
                if (c_library.num_orients > 0 && c_library.num_confs > 0) {
                //double          mol_stop_time = wall_clock_seconds();
                    struct timespec mol_stop_time;
                    timespec_get(&mol_stop_time, TIME_UTC);
                    double          time_per_mol = calculate_simulation_time(mol_start_time, mol_stop_time);
                    cout << "\n" "-----------------------------------" "\n";
                    cout << "Molecule: " << mol.title << "\n\n";
                    cout << " Elapsed time for docking:\t";
                    cout << fixed << setprecision(3) << time_per_mol << " seconds\n\n";
                }
            } else {
                if (c_library.num_orients > 0 && c_library.num_confs > 0) {
                    struct timespec mol_stop_time;
                    timespec_get(&mol_stop_time, TIME_UTC);
                    double          time_per_mol = calculate_simulation_time(mol_start_time, mol_stop_time);
                    
                    cout << " Elapsed time for docking:\t";
                    cout << fixed << setprecision(3) << time_per_mol << " seconds\n\n"; 
                }
            }
             
#endif
#ifndef TIME_PRECISION
            //print error messages if orienting or growth has failed
            if (c_library.num_orients == 0) {
                double          mol_stop_time = wall_clock_seconds();

                cout << "\n" "-----------------------------------" "\n";
                cout << "Molecule: " << mol.title << "\n\n";
                cout << " Elapsed time:\t" << fixed << setprecision(3)
                     << mol_stop_time - mol_start_time << " seconds\n\n";
                cout << " ERROR:  Could not find a valid orientation." << endl;
                cout << "         (For rigid docking the sought orientation is "
                        "the whole ligand;\n"
                        "         for flexible docking the sought orientation is "
                        "the anchor.)\n"
                        "         Confirm that all spheres are inside the grid box"
                        "         \nand that the grid box is big enough to contain "
                        "an orientation.\n";
            } else if (c_library.num_confs == 0) {
                double          mol_stop_time = wall_clock_seconds();
                cout << "\n" "-----------------------------------" "\n";
                cout << "Molecule: " << mol.title << "\n\n";
                cout << " Elapsed time:\t" << fixed << setprecision(3)
                     << mol_stop_time - mol_start_time << " seconds\n\n";
                cout << " ERROR:  Could not complete growth." << endl;
                cout << "         Confirm that the grid box is large enough to "
                        "contain the ligand,\n"
                        "         and try increasing max_orientations.\n";
            }
            // End individual molecule timing
            if (c_library.num_orients > 0 && c_library.num_confs > 0) {
                double          mol_stop_time = wall_clock_seconds();
                cout << "\n" "----------------------------------" "\n";
                cout << "Molecule: " << mol.title << "\n\n";
                cout << " Elapsed time for docking:\t" << fixed << setprecision(3)
                     << mol_stop_time - mol_start_time << " seconds\n\n";

            }
#endif

    }
    }
    }
    //main_dock_loop_anchors.close();
    // /////////////////////////////////////////////////////////////////////////


    // Write fragment libraries to file here
    if (c_master_conf.c_ag_conf.write_fragment_libraries)
        //c_master_conf.c_ag_conf.write_unique_fragments();
        c_master_conf.c_ag_conf.write_unique_fragments();
    // Rescore library using secondary scoring
    if (c_master_score.use_secondary_score) {
        c_library.secondary_rescore_poses(c_master_score, *active_min);
        c_library.submit_secondary_pose();
    }
    
    int end_process_mols = 0;
    if ((!USE_MPI) || (c_library.rank == 0)){
        if (c_master_conf.method == 2){
            end_process_mols = c_master_conf.c_dn_build.molecule_counter;
            cout << "\n\n" << c_master_conf.c_dn_build.molecule_counter
                 << " Attachments Processed" << endl;
        } else if (! (c_master_conf.method == 3)){ 
            end_process_mols = c_library.total_mols - c_library.initial_skip;
            cout << "\n\n" << c_library.total_mols - c_library.initial_skip
                 << " Molecules Processed" << endl;
        }
    } else {
        if (c_master_conf.method == 2){
            end_process_mols = c_master_conf.c_dn_build.molecule_counter;
            cout << "\n\n" << c_master_conf.c_dn_build.molecule_counter
                 << " Attachments Processed" << endl;
        } else if (! (c_master_conf.method == 3)){
            end_process_mols = c_library.completed;
            cout << "\n\n" << c_library.completed
                 << " Molecules Processed" << endl;
        }
    }    

    // End timing
#ifndef TIME_PRECISION
    double          stop_time = wall_clock_seconds();
    double          total_time = stop_time - start_time;
    cout << "Total elapsed time:\t";
    cout << fixed << setprecision(3) << total_time << " seconds\n";
#endif
#ifdef TIME_PRECISION
    struct timespec stop_time;
    timespec_get(&stop_time, TIME_UTC);
    double          total_time = calculate_simulation_time(start_time, stop_time); 
    cout << "Total elapsed time:\t";
    cout << fixed << setprecision(3) << total_time << " seconds\n";
#endif
  
    float mol_per_sec = end_process_mols / total_time;

    if (c_master_conf.method == 2) {
        cout << "Number of attachments per second:\t"
             << fixed << setprecision(3) << mol_per_sec << "\n";
    } else if (! (c_master_conf.method == 3)) {
        cout << "Number of molecules per second:\t"
             << fixed << setprecision(3) << mol_per_sec << "\n";
    }    
    
#ifndef __APPLE__

    long long virtmem = getVirtValue();
    std::cout << "Virtual memory used for this process: " << virtmem
              << " kilobytes\n";

    long long physmem = getPhysValue();
    std::cout << "Physical memory used for this process: " << physmem
              << " kilobytes" << std::endl;
#endif


    if (USE_MPI)
        c_library.finalize_mpi();

    if (outfile.is_open()) {
        // return non error output to standard output.
        cout.rdbuf(original_cout_buffer);
        outfile.close();
        fclose(stdout);  // C stdio.
    }

    return 0;                   // zero return indicates success

}

/************************************************/
void
dock_new_handler()
{

    // Define the default behavior for the failure of new.

    cout << "Error: memory exhausted!" << endl
        << "  If this occurs during grid reading then a likely cause\n"
        << "  is a grid that is too large.  Some machines have a very\n"
        << "  restrictive policy on allocating available resources:\n"
        << "  increase the datasize, stacksize, and memoryuse\n"
        << "  using the limit, ulimit, or unlimit commands;\n"
        << "  for many Linuxes this command sequence will work:\n"
        << "  limit; unlimit; limit\n"
        << "  Otherwise read the man pages or apply trial and error\n"
        << "  to find the apt use of these commands.\n"
        << endl;

    cerr << "Error: memory exhausted!" << endl
        << "  If this occurs during grid reading then a likely cause\n"
        << "  is a grid that is too large.  Some machines have a very\n"
        << "  restrictive policy on allocating available resources:\n"
        << "  increase the datasize, stacksize, and memoryuse\n"
        << "  using the limit, ulimit, or unlimit commands;\n"
        << "  for many Linuxes this command sequence will work:\n"
        << "  limit; unlimit; limit\n"
        << "  Otherwise read the man pages or apply trial and error\n"
        << "  to find the apt use of these commands.\n"
        << endl;

    // Throw so that recovery or special error notification can be performed.
    throw           bad_alloc();
}

/************************************************/
void
print_header( bool USE_MPI, int processes )
{
    cout << "\n\n\n--------------------------------------\n"
         << DOCK_VERSION
         << "\n\nReleased " << DOCK_RELEASE_DATE
         << "\nCopyright UCSF"
         << "\n--------------------------------------\n";
    if (USE_MPI) {
        cout << "Parallel dock running "
             << processes
             << " MPI processes"
             << "\n--------------------------------------\n";
    }
    cout << endl;
}

/************************************************/
//make sure c++11 is available - full functionality not guaranteed
#ifdef TIME_PRECISION

double
calculate_simulation_time(timespec t_start, timespec t_end)
{
/***********************************************************************
timespec returns time as a number of seconds, timespec.tv_sec,       
and a number of nanoseconds, timespec.tv_nsec. The correct moment    
in time equals to             
timespec.tv_sec + static_cast<double>(timespec.tv_nsec) / 1000000000.0
***********************************************************************/
    double sec_diff = t_end.tv_sec - t_start.tv_sec;
    long nsec_diff = t_end.tv_nsec - t_start.tv_nsec;
    if (nsec_diff < 0){
        sec_diff = sec_diff - 1.0;
        nsec_diff = 1000000000 + nsec_diff;
    }
    double sec_fraction = static_cast<double>(nsec_diff) / 1000000000.0;
    double simulation_time = sec_diff + sec_fraction;
    return simulation_time;
}
#endif
//Just in case they don't have c++11
////#ifndef TIME_PRECISION
double
wall_clock_seconds()
{
    time_t          t;
    if (static_cast < time_t > (-1) == time(&t)) {
        cout << "Error from time function!  Elapsed time is erroneous." << endl;
    }
    return static_cast < double >(t);
}
/****************************************************/
/*int
wall_clock_nseconds()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return static_cast < int >(ts.tv_nsec);
}*/
////#endif
