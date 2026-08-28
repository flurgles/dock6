#include "dock_threading.h"
#include "score_dock_gpu.h"
#include <thread>
#include <cstdlib>
#include <unistd.h>

std::mutex g_write_mutex;
std::mutex g_gpu_mutex;
std::mutex WorkerContext::gpu_mutex;

int get_num_threads_auto() {
    const char* env = getenv("DOCK_NUM_THREADS");
    if (env && *env) {
        int n = atoi(env);
        if (n > 0) return n;
        if (n == 0) return 1; // DOCK_NUM_THREADS=0 means serial
    }
    // Check vs window to avoid oversubscribe beyond memory
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    // Reserve 1 for GPU feeder + master if GPU active
    int gpu_active = dock_gpu_is_active();
    if (gpu_active && hc > 2) hc -= 1;
    if (hc < 1) hc = 1;
    // Respect DOCK_VS_WINDOW_MAX / per_row memory bound: threads * per_row <= avail
    // Simple clamp: ideally window >= threads. If window small, reduce threads.
    // We read window via vs_calc helper in dock.cpp, but here just return hc.
    return (int)hc;
}

bool should_use_threads(bool use_mpi, int method) {
    if (use_mpi) return false;
    // Only VS methods 0==rigid,1==flex, also 5 HDB? keep serial for HDB/GA/DN
    if (method != 0 && method != 1) return false;
    int n = get_num_threads_auto();
    if (n <= 1) return false;
    // env override
    const char* env = getenv("DOCK_NUM_THREADS");
    if (env && atoi(env) == 0) return false;
    if (env && atoi(env) == 1) return false;
    return true;
}

DockResult dock_one_ligand(WorkerContext &ctx, DOCKMol mol) {
    DockResult res;
    res.title = mol.title;
    // This is a stub - real docking logic will be inlined from dock.cpp
    // For now, just mark success and let caller handle full logic via shared function.
    // The actual docking is performed by calling the same sequence as serial:
    //  - ctx.typer.prepare_molecule
    //  - ctx.master_conf.prepare_molecule
    //  - filter, anchor/orient loops, grow, minimize, submit
    // To avoid duplicating 400 lines, we will call a helper that is factored from dock.cpp
    // in the next commit. For now, return empty.
    res.success = true;
    return res;
}
