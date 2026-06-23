/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
GPU abstraction API for grid generation.

Provides a clean interface that each GPU backend (Metal, Vulkan, CUDA)
implements. The CPU fallback stub returns 0 from init(), causing the
existing CPU code path to run.

Usage in score_grid.c:
    #include "score_grid_gpu.h"

    if (gpu_grid_init(energy, receptor, grid, bump, contact, chemical, label)) {
        gpu_grid_upload(energy, receptor, grid, bump, contact, chemical, label);
        gpu_grid_compute(soft_delta);
        gpu_grid_download(energy, grid, bump, contact, chemical);
        gpu_grid_cleanup();
        return;
    }
    // fall through to CPU path
*/

#ifndef SCORE_GRID_GPU_H
#define SCORE_GRID_GPU_H

/* Forward declarations — the DOCK headers lack include guards so we
   cannot safely include them here.  We only pass pointers, so forward
   declarations suffice. */
typedef struct score_energy_struct   SCORE_ENERGY;
typedef struct score_grid_struct     SCORE_GRID;
typedef struct score_bump_struct     SCORE_BUMP;
typedef struct score_contact_struct  SCORE_CONTACT;
typedef struct score_chemical_struct SCORE_CHEMICAL;
typedef struct molecule_struct       MOLECULE;
typedef struct label_struct          LABEL;

/* ------------------------------------------------------------------ */
/*  GPU abstraction API                                               */
/* ------------------------------------------------------------------ */

/* All functions use C linkage so they can be called from C (score_grid.c)
   even when the backend implementation is compiled as Objective-C++. */
#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU device and allocate all buffers.
   Returns 1 on success, 0 if GPU is unavailable (caller falls back to CPU). */
int gpu_grid_init(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                  SCORE_BUMP *bump, SCORE_CONTACT *contact,
                  SCORE_CHEMICAL *chemical, LABEL *label);

/* Upload receptor atom data and grid parameters to the GPU. */
void gpu_grid_upload(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                     SCORE_BUMP *bump, SCORE_CONTACT *contact,
                     SCORE_CHEMICAL *chemical, LABEL *label);

/* Launch the GPU compute kernel.  Must be called after upload(). */
void gpu_grid_compute(float soft_delta);

/* Download computed grid arrays from GPU back to CPU structs. */
void gpu_grid_download(SCORE_ENERGY *energy, SCORE_GRID *grid,
                       SCORE_BUMP *bump, SCORE_CONTACT *contact,
                       SCORE_CHEMICAL *chemical);

/* Release all GPU resources. */
void gpu_grid_cleanup(void);

/* Returns 1 if the GPU backend is active, 0 if using CPU fallback. */
int gpu_grid_is_active(void);

#ifdef __cplusplus
}
#endif

#endif /* SCORE_GRID_GPU_H */
