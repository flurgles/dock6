/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
  HIP GPU backend for dock6 scoring — internal declarations.
  Not included directly by CPU code; the dock_gpu_* API in
  score_dock_gpu.h is the public interface.

  Mirrors score_dock_gpu_vulkan.h.  The layout of HipGridParams MUST
  match the kernel parameter struct in score_dock_gpu_hip.cpp.
 */

#ifndef SCORE_DOCK_GPU_HIP_H
#define SCORE_DOCK_GPU_HIP_H

typedef struct {
    float  origin_x, origin_y, origin_z;
    int    span_x, span_y, span_z;
    float  spacing;
    int    grid_size;       /* span_x * span_y * span_z */
} HipGridParams;

#endif /* SCORE_DOCK_GPU_HIP_H */