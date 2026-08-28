#pragma once
// Lean threading for VS - ligand-level parallel, FIFO pull, no stealing.
#include "thread_queue.h"
#include "dockmol.h"
#include "library_file.h"
#include "master_conf.h"
#include "master_score.h"
#include "orient.h"
#include "filter.h"
#include "amber_typer.h"
#include "minimizer.h"
#include "simplex.h"
#include "bobyqa.h"
#include "steepest_descent.h"
#include "conjugate_gradient.h"
#include "score.h"
#include "trace.h"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

struct DockResult {
    std::string title;
    std::vector<RANKMol> poses; // per-lig ranked_poses after sort_write
    std::vector<SCOREMol> ranked_list_entry; // if rank_ligands
    int num_anchors = 0;
    int num_orients = 0;
    int num_confs = 0;
    std::string scored_output; // buffered cout for this lig (verbose)
    bool success = false;
};

// WorkerContext holds thread-local copies of all mutable docking objects.
// File handles remain in master Library_File only.
struct WorkerContext {
    // copies - initialized from master in main
    Library_File lib_local; // only per-lig vectors used, file streams unused
    Orient orient;
    Master_Conformer_Search master_conf;
    Bump_Filter bmp;
    Master_Score* master_score = nullptr; // points to global master (shared, protected by g_score_mutex)
    AMBER_TYPER typer;
    Filter filter;

    // minimizers - owned
    Simplex_Minimizer simplex;
    BOBYQA_Minimizer bobyqa;
    Steepest_Descent_Minimizer sd;
    Conjugate_Gradient_Minimizer cg;
    Minimizer* active_min = nullptr;
    WorkerContext() = default;
    WorkerContext(const WorkerContext&) = delete;
    WorkerContext& operator=(const WorkerContext&) = delete;
    WorkerContext(WorkerContext&&) = default;
    WorkerContext& operator=(WorkerContext&&) = default;

    // GPU serialization for lean phase (no feeder yet)
    static std::mutex gpu_mutex;
};

extern std::mutex g_write_mutex; // protects master Library_File file writes
extern std::mutex g_gpu_mutex;

// helpers
int get_num_threads_auto();
bool should_use_threads(bool use_mpi, int method);
DockResult dock_one_ligand(WorkerContext &ctx, DOCKMol mol);
