#include "dock_threading.h"
#include "score_dock_gpu.h"
#include "utils.h"
#include "trace.h"
#include <thread>
#include <cstdlib>
#include <unistd.h>
#include <sstream>
#include <iostream>

std::mutex g_write_mutex;
std::mutex g_gpu_mutex;
std::mutex WorkerContext::gpu_mutex;
std::mutex g_score_mutex; // protects shared Master_Score nb_int etc for threaded use

int get_num_threads_auto() {
    const char* env = getenv("DOCK_NUM_THREADS");
    if (env && *env) {
        int n = atoi(env);
        if (n > 0) return n;
        if (n == 0) return 1;
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    int gpu_active = dock_gpu_is_active();
    if (gpu_active && hc > 2) hc -= 1;
    if (hc < 1) hc = 1;
    return (int)hc;
}

bool should_use_threads(bool use_mpi, int method) {
    if (use_mpi) return false;
    if (method != 0 && method != 1) return false;
    int n = get_num_threads_auto();
    if (n <= 1) return false;
    const char* env = getenv("DOCK_NUM_THREADS");
    if (env && atoi(env) == 0) return false;
    if (env && atoi(env) == 1) return false;
    return true;
}

DockResult dock_one_ligand(WorkerContext &ctx, DOCKMol mol) {
    DockResult res;
    res.title = mol.title;

    // Capture cout for this ligand (verbose etc) into string
    std::ostringstream buf;
    std::streambuf* old_cout = nullptr;
    bool capture = ctx.master_conf.c_ag_conf.verbose;
    if (capture) old_cout = std::cout.rdbuf(buf.rdbuf());

    // Reset per-lig state
    ctx.lib_local.ranked_poses.clear();
    ctx.lib_local.cluster_assignments.clear();
    ctx.lib_local.high_position = 0;
    ctx.lib_local.high_score = 0;
    ctx.lib_local.num_anchors = 0;
    ctx.lib_local.num_orients = 0;
    ctx.lib_local.num_confs = 0;
    ctx.lib_local.read_success = true;

    // seed RNG
    if (ctx.active_min) ctx.active_min->initialize();

    mol.prepare_molecule();
    mol.flag_write_solvation = ctx.lib_local.write_solv_mol2;

    ctx.typer.prepare_molecule(mol, ctx.master_score->read_vdw,
                               ctx.orient.use_chemical_matching, ctx.master_score->use_ph4, ctx.master_score->use_volume);

    ctx.master_conf.initialize_once = true;
    ctx.master_conf.prepare_molecule(mol);

    if (ctx.master_conf.c_ag_conf.write_fragment_libraries) {
        if (capture) { std::cout.rdbuf(old_cout); res.scored_output = buf.str(); }
        res.success = true;
        return res;
    }

    // DB filter
    if (ctx.filter.use_database_filter) {
        ctx.filter.calc_descriptors(mol);
#ifdef BUILD_DOCK_WITH_RDKIT
        if (ctx.filter.fails_filter(mol)) {
            mol.fails_filt = true;
            std::cout << mol.current_data << std::endl;
        } else {
            mol.fails_filt = false;
        }
#else
        if (ctx.filter.fails_filter(mol)) {
            std::cout << mol.current_data << std::endl;
            if (capture) { std::cout.rdbuf(old_cout); res.scored_output = buf.str(); }
            // still need to return with no poses
            res.num_anchors = ctx.lib_local.num_anchors;
            res.num_orients = ctx.lib_local.num_orients;
            res.num_confs = ctx.lib_local.num_confs;
            return res;
        }
#endif
    }

    // Main anchor loop - protected by score mutex as Master_Score is shared
    {
        std::lock_guard<std::mutex> lk(g_score_mutex);
        while (ctx.master_conf.next_anchor(mol)) {
            ctx.lib_local.num_anchors++;
            ctx.orient.match_ligand(mol);
            while (ctx.orient.new_next_orientation(mol)) {
                if (!ctx.bmp.check_anchor_bumps(mol, ctx.orient.more_orientations())) continue;

                if (ctx.master_conf.method == 1) {
                    ctx.master_score->primary_score->nb_int.clear();
                } else {
                    ctx.master_score->primary_score->nb_int.clear();
                    ctx.master_conf.initialize_once = true;
                    ctx.master_conf.grow_periphery(*ctx.master_score, *ctx.active_min, ctx.bmp);
                }

                if (ctx.lib_local.submit_orientation(mol, *ctx.master_score, ctx.orient.orient_ligand) ||
                    !ctx.orient.more_orientations()) {
                    if (ctx.master_conf.submit_anchor_orientation(mol, ctx.orient.more_orientations())) {
                        ctx.master_conf.grow_periphery(*ctx.master_score, *ctx.active_min, ctx.bmp);
                        while (ctx.master_conf.next_conformer(mol)) {
                            ctx.active_min->minimize_final_pose(mol, *ctx.master_score, ctx.typer);
                            ctx.lib_local.submit_scored_pose(mol, *ctx.master_score, *ctx.active_min);
                        }
                    }
                }
            }
        }
    }
    if (capture) {
        std::cout.rdbuf(old_cout);
        res.scored_output = buf.str();
    }
    ctx.lib_local.submit_conformations(*ctx.master_score);

    // collect results
    res.poses = ctx.lib_local.ranked_poses;
    res.num_anchors = ctx.lib_local.num_anchors;
    res.num_orients = ctx.lib_local.num_orients;
    res.num_confs = ctx.lib_local.num_confs;
    res.success = true;

    // Also collect cluster assignments if needed
    // ranked_list merging will be done by master under lock
    return res;
}
