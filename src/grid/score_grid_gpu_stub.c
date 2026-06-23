/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
CPU fallback stub — all functions are no-ops.
gpu_grid_init() returns 0, so score_grid.c falls through to the CPU path.
*/

#include "score_grid_gpu.h"

int gpu_grid_init(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                  SCORE_BUMP *bump, SCORE_CONTACT *contact,
                  SCORE_CHEMICAL *chemical, LABEL *label)
{
    (void)energy; (void)receptor; (void)grid;
    (void)bump; (void)contact; (void)chemical; (void)label;
    return 0;  /* GPU not available, use CPU fallback */
}

void gpu_grid_upload(SCORE_ENERGY *energy, MOLECULE *receptor, SCORE_GRID *grid,
                     SCORE_BUMP *bump, SCORE_CONTACT *contact,
                     SCORE_CHEMICAL *chemical, LABEL *label)
{
    (void)energy; (void)receptor; (void)grid;
    (void)bump; (void)contact; (void)chemical; (void)label;
}

void gpu_grid_compute(float soft_delta)
{
    (void)soft_delta;
}

void gpu_grid_download(SCORE_ENERGY *energy, SCORE_GRID *grid,
                       SCORE_BUMP *bump, SCORE_CONTACT *contact,
                       SCORE_CHEMICAL *chemical)
{
    (void)energy; (void)grid;
    (void)bump; (void)contact; (void)chemical;
}

void gpu_grid_cleanup(void)
{
}

int gpu_grid_is_active(void)
{
    return 0;
}
