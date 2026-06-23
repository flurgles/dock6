/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
Metal GPU backend — internal declarations.
Not included directly by score_grid.c; the gpu_grid_* API in
score_grid_gpu.h is the public interface.
*/

#ifndef SCORE_GRID_GPU_METAL_H
#define SCORE_GRID_GPU_METAL_H

/*
  Grid parameters passed to the Metal compute kernel.
  Must match the layout of the 'constant GridParams' struct in
  the Metal shader source (grid_gpu_metal_shader.h).
*/
typedef struct {
    float origin_x, origin_y, origin_z;
    int    span_x, span_y, span_z;
    float spacing;
    float distance;        /* interaction cutoff in Angstroms */
    float dist_sq_min;
    float rep_exponent;    /* float for easier Metal constant buffer alignment */
    float att_exponent;
    int    distance_dielectric;
    float dielectric_factor;
    float soft_delta;
    int    grid_size;      /* span_x * span_y * span_z */
} GridParams;

/* Number of atoms allocated on the GPU (may be > actual count) */
#define GPU_MAX_ATOMS 65536

#endif /* SCORE_GRID_GPU_METAL_H */
