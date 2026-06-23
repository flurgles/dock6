/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Metal GPU backend for dock6 scoring — internal declarations.
Not included directly by CPU code; the dock_gpu_* API in
score_dock_gpu.h is the public interface.
*/

#ifndef SCORE_DOCK_GPU_METAL_H
#define SCORE_DOCK_GPU_METAL_H

/*
  Grid parameters passed to the Metal batch-score kernel.
  Must match the layout of the 'constant GridParams' struct in
  the Metal shader source inside score_dock_gpu_metal.mm.
*/
typedef struct {
    float  origin_x, origin_y, origin_z;
    int    span_x, span_y, span_z;
    float  spacing;
    int    grid_size;       /* span_x * span_y * span_z */
} DockGridParams;

#endif /* SCORE_DOCK_GPU_METAL_H */
