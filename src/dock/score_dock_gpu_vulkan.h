/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*

/*
  Vulkan GPU backend for dock6 scoring — internal declarations.
  Not included directly by CPU code; the dock_gpu_* API in
  score_dock_gpu.h is the public interface.

  Mirrors score_dock_gpu_metal.h.  The layout of DockGridParams MUST
  match the struct used by the GLSL shader (see push-constant block in
  score_dock_gpu_vulkan.cpp).  Unlike Metal, Vulkan push constants use
  std430-style packing but the compiler pads the same way; keep this in
  sync with the GLSL 'Params' block.
 */

#ifndef SCORE_DOCK_GPU_VULKAN_H
#define SCORE_DOCK_GPU_VULKAN_H

typedef struct {
    float  origin_x, origin_y, origin_z;
    int    span_x, span_y, span_z;
    float  spacing;
    int    grid_size;       /* span_x * span_y * span_z */
} DockGridParams;

#endif /* SCORE_DOCK_GPU_VULKAN_H */
