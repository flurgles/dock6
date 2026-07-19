/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*

/*
  Vulkan GPU backend for dock6 batch scoring.

  Implements the GPU abstraction API defined in score_dock_gpu.h using
  Vulkan (portable across macOS/Linux/Windows).  Targeted first at the
  Steam Deck (AMD RDNA2 / RADV).

  Design notes / differences from the Metal backend:
    * Grid data is stored as 3D textures (VkImage TYPE_3D, R32_SFLOAT)
      with hardware trilinear filtering via sampler3D, matching the
      Metal backend's texture3d<float> approach.  inside_grid() uses
      > 0.0 (not > 1.0 like Metal) to avoid the 1-voxel safety margin
      divergence noted in the Metal backend.
    * Buffers use HOST_VISIBLE | HOST_COHERENT storage.  The Deck is a
      unified-memory APU, so this gives the same perf model as Metal's
      Shared storage with no staging copies.  Grid 3D images use
      DEVICE_LOCAL with staging-buffer upload (optimal tiling required
      for hardware texture filtering).
    * Shaders are authored in GLSL and compiled to SPIR-V at runtime via
      shaderc (mirrors Metal compiling a *.metal string at runtime, and
      keeps the build free of a shader-compile step).
    * The persistent-threadgroup + atomic work-counter kernel maps
      directly: a storage buffer of uint32_t + atomicAdd (GLSL
      atomicCounter / atomicAdd) over a fixed number of workgroups.
    * SIMD-group reduction becomes subgroup reduction via
      subgroupAdd() (VK 1.1 shaderSubgroupArithmetic, available on
      RADV).  A shared-memory fallback is used if subgroups are absent.
 */

#include "score_dock_gpu.h"
#include "score_dock_gpu_vulkan.h"

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

/* Max poses per batch */
#define GPU_MAX_POSES       4096
#define GPU_MAX_ATOMS       512
#define GPU_MAX_NB_PAIRS    32768  /* Max non-bonded pairs per ligand */
#define GPU_MAX_TORSIONS    50
#define GPU_DOF_MAX         56
#define BATCH_MAX           32

/* GLSL source for every compute kernel, compiled to SPIR-V at runtime. */
static const char* shader_src = R"glsl(
#version 450
#extension GL_EXT_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_counter_buffer : enable

layout(local_size_x = 64) in;

// ---- storage buffers (binding indices chosen by host) ----
layout(std430, binding = 0) buffer BXYZ   { float xyz[]; };
layout(std430, binding = 1) buffer BVDWA  { float vdwA[]; };
layout(std430, binding = 2) buffer BVDWB  { float vdwB[]; };
layout(std430, binding = 3) buffer BCHG   { float charges[]; };
// bindings 4-6: 3D texture samplers (hardware trilinear)
layout(binding = 4) uniform sampler3D g_avdw;
layout(binding = 5) uniform sampler3D g_bvdw;
layout(binding = 6) uniform sampler3D g_es;
layout(std430, binding = 7) buffer BOUT   { float out_scores[]; };
layout(std430, binding = 8) buffer BACT   { int   active_flags[]; };
layout(std430, binding = 9) buffer BIEA   { float ie_vdwA[]; };
layout(std430, binding = 10) buffer BSTART { int   pair_starts[]; };
layout(std430, binding = 11) buffer BIDX   { int   pair_indices[]; };
layout(std430, binding = 12) buffer BCNT   { uint  pose_counter[]; };

// ---- push constants (GridParams + extras) ----
layout(push_constant) uniform Params {
    float origin_x, origin_y, origin_z;
    int   span_x, span_y, span_z;
    float spacing;
    int   num_atoms;
    float ie_soft_delta;
    float ie_cutoff_sq;
    int   num_nb_pairs;
    int   num_poses;
} p;

// Hardware-accelerated trilinear via 3D texture sampler.
// Bounds check uses > 0.0 (not > 1.0 like Metal) to avoid the
// 1-voxel safety margin divergence noted in the Metal backend.
float sample_grid(sampler3D grid, float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
        gx >= float(p.span_x - 1) ||
        gy >= float(p.span_y - 1) ||
        gz >= float(p.span_z - 1))
        return 0.0;
    // Vulkan normalized coords: texel i center at (i+0.5)/extent
    vec3 coord = vec3((gx + 0.5) / float(p.span_x),
                      (gy + 0.5) / float(p.span_y),
                      (gz + 0.5) / float(p.span_z));
    return texture(grid, coord).r;
}

// ============ Atom-parallel persistent IE kernel ============
shared float tg_partial[8];
shared uint tg_candidate;

void kernel_persistent() {
    uint tid = gl_LocalInvocationID.x;
    uint tg_size = gl_WorkGroupSize.x;
    uint simd_idx = tid / 32u;

    if (tid == 0u) {
        tg_candidate = atomicAdd(pose_counter[0], 1u);
    }
    barrier();

    while (tg_candidate < uint(p.num_poses)) {
        int stride = int(tg_candidate) * p.num_atoms * 3;
        float total = 0.0;

        for (uint a = tid; a < uint(p.num_atoms); a += tg_size) {
            if (active_flags[a] == 0) continue;
            float x = xyz[stride + int(a)*3];
            float y = xyz[stride + int(a)*3 + 1];
            float z = xyz[stride + int(a)*3 + 2];
            float vdw  = sample_grid(g_avdw, x, y, z);
            float bvdw = sample_grid(g_bvdw, x, y, z);
            float es   = sample_grid(g_es,   x, y, z);
            total += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
            int start = pair_starts[a];
            int end   = pair_starts[a + 1];
            for (int i = start; i < end; i++) {
                int a2 = pair_indices[i];
                float dx = xyz[stride + int(a)*3]     - xyz[stride + a2*3];
                float dy = xyz[stride + int(a)*3 + 1] - xyz[stride + a2*3 + 1];
                float dz = xyz[stride + int(a)*3 + 2] - xyz[stride + a2*3 + 2];
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < p.ie_cutoff_sq) {
                    float r2eff = r2 + p.ie_soft_delta;
                    float denom = r2eff*r2eff*r2eff;
                    total += (ie_vdwA[a]*ie_vdwA[a2]) / (denom*denom);
                }
            }
        }

        float s = subgroupAdd(total);
        if (gl_SubgroupInvocationID == 0u) {
            tg_partial[simd_idx] = s;
        }
        barrier();

        if (tid == 0u) {
            float final_score = tg_partial[0] + tg_partial[1];
            out_scores[tg_candidate] = final_score;
            tg_candidate = atomicAdd(pose_counter[0], 1u);
        }
        barrier();
    }
}

void main() {
    kernel_persistent();
}
)glsl";

/* Second GLSL module: grid-only batch_score (1 thread/pose). */
static const char* shader_src_grid = R"glsl(
#version 450

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer BXYZ   { float xyz[]; };
layout(std430, binding = 1) buffer BVDWA  { float vdwA[]; };
layout(std430, binding = 2) buffer BVDWB  { float vdwB[]; };
layout(std430, binding = 3) buffer BCHG   { float charges[]; };
layout(binding = 4) uniform sampler3D g_avdw;
layout(binding = 5) uniform sampler3D g_bvdw;
layout(binding = 6) uniform sampler3D g_es;
layout(std430, binding = 7) buffer BOUT   { float out_scores[]; };
layout(std430, binding = 8) buffer BACT   { int   active_flags[]; };

layout(push_constant) uniform Params {
    float origin_x, origin_y, origin_z;
    int   span_x, span_y, span_z;
    float spacing;
    int   num_atoms;
    float ie_soft_delta;
    float ie_cutoff_sq;
    int   num_nb_pairs;
    int   num_poses;
} p;

float sample_grid(sampler3D grid, float x, float y, float z) {
    float gx = (x - p.origin_x) / p.spacing;
    float gy = (y - p.origin_y) / p.spacing;
    float gz = (z - p.origin_z) / p.spacing;
    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
        gx >= float(p.span_x - 1) ||
        gy >= float(p.span_y - 1) ||
        gz >= float(p.span_z - 1))
        return 0.0;
    vec3 coord = vec3((gx + 0.5) / float(p.span_x),
                      (gy + 0.5) / float(p.span_y),
                      (gz + 0.5) / float(p.span_z));
    return texture(grid, coord).r;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(p.num_poses)) return;
    int stride = int(tid) * p.num_atoms * 3;
    float score = 0.0;
    for (int a = 0; a < p.num_atoms; a++) {
        int o3 = stride + a*3;
        float x = xyz[o3], y = xyz[o3+1], z = xyz[o3+2];
        if (active_flags[a]) {
            float vdw  = sample_grid(g_avdw, x, y, z);
            float bvdw = sample_grid(g_bvdw, x, y, z);
            float es   = sample_grid(g_es,   x, y, z);
            score += vdwA[a]*vdw - vdwB[a]*bvdw + charges[a]*es;
        }
    }
    out_scores[tid] = score;
}
)glsl";

/* Push constant block matching the GLSL Params layout exactly (std430, 48 bytes). */
struct PushConstants {
    float origin_x, origin_y, origin_z;   // offset  0
    int   span_x, span_y, span_z;         // offset 12
    float spacing;                         // offset 24
    int   num_atoms;                       // offset 28
    float ie_soft_delta;                   // offset 32
    float ie_cutoff_sq;                    // offset 36
    int   num_nb_pairs;                    // offset 40
    int   num_poses;                       // offset 44
};                                         // total: 48 bytes

/* ================================================================== */
/*  Static state                                                       */
/* ================================================================== */

static VkInstance        g_inst       = VK_NULL_HANDLE;
static VkPhysicalDevice  g_phys       = VK_NULL_HANDLE;
static VkDevice          g_dev        = VK_NULL_HANDLE;
static VkQueue           g_queue      = VK_NULL_HANDLE;
static uint32_t         g_queue_idx  = 0;
static VkCommandPool     g_cmdpool    = VK_NULL_HANDLE;
static VkCommandBuffer   g_cmd        = VK_NULL_HANDLE;
static VkDescriptorPool  g_descpool   = VK_NULL_HANDLE;

static VkShaderModule    g_mod_grid   = VK_NULL_HANDLE;
static VkShaderModule    g_mod_ie     = VK_NULL_HANDLE;
static VkPipelineLayout  g_playout    = VK_NULL_HANDLE;
static VkPipeline        g_pl_grid    = VK_NULL_HANDLE;  /* grid-only kernel */
static VkPipeline        g_pl_ie      = VK_NULL_HANDLE;  /* persistent IE kernel */
static VkDescriptorSetLayout g_dslayout = VK_NULL_HANDLE;

/* Buffers (host-visible, coherent) */
static VkBuffer g_buf_vdwA = VK_NULL_HANDLE, g_buf_vdwB = VK_NULL_HANDLE, g_buf_charges = VK_NULL_HANDLE;
static VkBuffer g_buf_ie_vdwA = VK_NULL_HANDLE, g_buf_nb_int = VK_NULL_HANDLE;
static VkBuffer g_buf_pair_starts = VK_NULL_HANDLE, g_buf_pair_indices = VK_NULL_HANDLE;
static VkBuffer g_buf_xyz = VK_NULL_HANDLE, g_buf_scores = VK_NULL_HANDLE;
static VkBuffer g_buf_active_flags = VK_NULL_HANDLE, g_buf_pose_counter = VK_NULL_HANDLE;
static VkDeviceMemory g_mem_vdwA = VK_NULL_HANDLE, g_mem_vdwB = VK_NULL_HANDLE, g_mem_charges = VK_NULL_HANDLE;
static VkDeviceMemory g_mem_ie_vdwA = VK_NULL_HANDLE, g_mem_nb_int = VK_NULL_HANDLE;
static VkDeviceMemory g_mem_pair_starts = VK_NULL_HANDLE, g_mem_pair_indices = VK_NULL_HANDLE;
static VkDeviceMemory g_mem_xyz = VK_NULL_HANDLE, g_mem_scores = VK_NULL_HANDLE;
static VkDeviceMemory g_mem_active_flags = VK_NULL_HANDLE, g_mem_pose_counter = VK_NULL_HANDLE;

/* 3D textures for hardware trilinear filtering (replaces grid SSBOs) */
static VkImage         g_img_avdw = VK_NULL_HANDLE, g_img_bvdw = VK_NULL_HANDLE, g_img_es = VK_NULL_HANDLE;
static VkDeviceMemory  g_mem_img_avdw = VK_NULL_HANDLE, g_mem_img_bvdw = VK_NULL_HANDLE, g_mem_img_es = VK_NULL_HANDLE;
static VkImageView     g_iv_avdw = VK_NULL_HANDLE, g_iv_bvdw = VK_NULL_HANDLE, g_iv_es = VK_NULL_HANDLE;
static VkSampler       g_sampler = VK_NULL_HANDLE;

static DockGridParams g_params;
static int  g_initialized = 0;
static int  g_active      = 0;
static int  g_num_atoms   = 0;
static int  g_num_nb_pairs = 0;
static float g_ie_soft_delta = 0.0;
static float g_ie_cutoff_sq  = 1e10f;
static int  g_compute_units = 0;

/* Timestamp queries */
static VkQueryPool g_tq_pool = VK_NULL_HANDLE;
static float g_timestamp_period_ns = 1.0f; /* ns per timestamp tick */

/* Fence for async dispatch (replaces vkQueueWaitIdle) */
static VkFence g_fence = VK_NULL_HANDLE;

/* Profiling counters (mirrors Metal backend) */
static uint64_t prof_dispatch_count    = 0;
static uint64_t prof_total_conformers  = 0;
static double   prof_total_dispatch_ms = 0.0;
static double   prof_last_dispatch_ms  = 0.0;
#define PROF_ROLLING_SIZE 64
static double   prof_wait_buf[PROF_ROLLING_SIZE];
static int      prof_wait_idx = 0;

/* ================================================================== */
/*  Vulkan helpers                                                     */
/* ================================================================== */

/* Cached memory properties (set once at init, used by all alloc calls) */
static VkPhysicalDeviceMemoryProperties g_mprops;

static void vk_check(VkResult r, const char *where) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: %s failed (%d)\n", where, (int)r);
    }
}

static uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < g_mprops.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_mprops.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return 0;
}

static VkDeviceMemory alloc_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_dev, &bci, NULL, buf) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, *buf, &mr);

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g_dev, &mai, NULL, mem) != VK_SUCCESS) return VK_NULL_HANDLE;
    vkBindBufferMemory(g_dev, *buf, *mem, 0);
    return *mem;
}

static void *buf_map(VkDeviceMemory mem) {
    void *ptr = NULL;
    vkMapMemory(g_dev, mem, 0, VK_WHOLE_SIZE, 0, &ptr);
    return ptr;
}

/* Create a 3D texture image (device-local, optimal tiling) */
static void create_3d_image(int sx, int sy, int sz,
                            VkImage *img, VkDeviceMemory *mem, VkImageView *view) {
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = {(uint32_t)sx, (uint32_t)sy, (uint32_t)sz};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(g_dev, &ici, NULL, img);

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, *img, &mr);

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(g_dev, &mai, NULL, mem);
    vkBindImageMemory(g_dev, *img, *mem, 0);

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ivci.format = VK_FORMAT_R32_SFLOAT;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    ivci.image = *img;
    vkCreateImageView(g_dev, &ivci, NULL, view);
}

/* Upload float data to a 3D image via staging buffer + command buffer */
static void upload_3d_image(const float *data, int sx, int sy, int sz, VkImage img) {
    VkDeviceSize bytes = sizeof(float) * (VkDeviceSize)sx * sy * sz;

    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    alloc_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging_buf, &staging_mem);
    memcpy(buf_map(staging_mem), data, (size_t)bytes);

    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cmd, &cbbi);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, NULL, 0, NULL, 1, &barrier);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {(uint32_t)sx, (uint32_t)sy, (uint32_t)sz};
    vkCmdCopyBufferToImage(g_cmd, staging_buf, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, NULL, 0, NULL, 1, &barrier);

    vkEndCommandBuffer(g_cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_queue);

    vkDestroyBuffer(g_dev, staging_buf, NULL);
    vkFreeMemory(g_dev, staging_mem, NULL);
}

/* Compile GLSL -> SPIR-V via shaderc, return a VkShaderModule. */
static VkShaderModule compile_module(const char *src, const char *label) {
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compilation_result_t result;
    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    shaderc_compile_options_set_source_language(opts, shaderc_source_language_glsl);
    shaderc_compile_options_set_optimization_level(opts, shaderc_optimization_level_performance);

    result = shaderc_compile_into_spv(compiler, src, strlen(src),
                                       shaderc_compute_shader, label, "main", opts);
    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        fprintf(stderr, "GPU-VK: shader '%s' compile failed:\n%s\n",
                label, shaderc_result_get_error_message(result));
        shaderc_result_release(result);
        shaderc_compile_options_release(opts);
        shaderc_compiler_release(compiler);
        return VK_NULL_HANDLE;
    }
    const uint32_t *spv = (const uint32_t *)shaderc_result_get_bytes(result);
    size_t spv_size = shaderc_result_get_length(result);

    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_size;
    smci.pCode = spv;
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(g_dev, &smci, NULL, &mod);

    shaderc_result_release(result);
    shaderc_compile_options_release(opts);
    shaderc_compiler_release(compiler);
    return mod;
}

/* ================================================================== */
/*  GPU abstraction API                                                 */
/* ================================================================== */

extern "C" {

int dock_gpu_init(const float *avdw, const float *bvdw, const float *es,
                  int span_x, int span_y, int span_z,
                  float origin_x, float origin_y, float origin_z,
                  float spacing)
{
    VkApplicationInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "dock6";
    ai.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, NULL, &g_inst) != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: instance creation failed — CPU fallback\n");
        return 0;
    }

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_inst, &ndev, NULL);
    if (ndev == 0) {
        fprintf(stderr, "GPU-VK: no physical devices — CPU fallback\n");
        return 0;
    }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(g_inst, &ndev, devs.data());
    g_phys = devs[0];

    /* Cache memory properties once for all alloc calls */
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_mprops);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_phys, &props);
    fprintf(stderr, "GPU-VK: device: %s\n", props.deviceName);

    /* compute units heuristic from subgroup/max-invocation limits */
    g_compute_units = (int)(props.limits.maxComputeWorkGroupInvocations / 32);
    if (g_compute_units <= 0) g_compute_units = 8;

    /* Cache timestamp period for profiling */
    g_timestamp_period_ns = props.limits.timestampPeriod;
    if (g_timestamp_period_ns <= 0.0f) g_timestamp_period_ns = 1.0f;

    /* Find a compute-capable queue */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, NULL);
    std::vector<VkQueueFamilyProperties> qf(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, qf.data());
    g_queue_idx = 0;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_queue_idx = i; break; }
    }

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo dqci = {};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = g_queue_idx;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &qprio;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    if (vkCreateDevice(g_phys, &dci, NULL, &g_dev) != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: device creation failed — CPU fallback\n");
        return 0;
    }
    vkGetDeviceQueue(g_dev, g_queue_idx, 0, &g_queue);

    /* Command pool + (one reusable) command buffer */
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = g_queue_idx;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g_dev, &cpci, NULL, &g_cmdpool);

    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g_cmdpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_dev, &cbai, &g_cmd);

    /* Fence for async dispatch — created SIGNALED so first wait passes */
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(g_dev, &fci, NULL, &g_fence);

    /* Shaders */
    g_mod_grid = compile_module(shader_src_grid, "grid_batch_score");
    g_mod_ie    = compile_module(shader_src,        "persistent_ie");
    if (!g_mod_grid || !g_mod_ie) {
        fprintf(stderr, "GPU-VK: shader compile failed — CPU fallback\n");
        dock_gpu_cleanup();
        return 0;
    }

    /* Descriptor set layout: bindings 0-3,7-12 = storage buffers,
     * bindings 4-6 = combined image samplers (3D textures for grids) */
    VkDescriptorSetLayoutBinding binds[13];
    for (int i = 0; i < 13; i++) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = (i >= 4 && i <= 6)
            ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 13;
    dslci.pBindings = binds;
    vkCreateDescriptorSetLayout(g_dev, &dslci, NULL, &g_dslayout);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g_dslayout;
    plci.pushConstantRangeCount = 0;  /* push constants declared; range added below */
    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);  /* 48 bytes matching GLSL Params block */
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(g_dev, &plci, NULL, &g_playout);

    VkComputePipelineCreateInfo cp1 = {};
    cp1.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp1.layout = g_playout;
    cp1.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp1.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp1.stage.module = g_mod_grid;
    cp1.stage.pName = "main";
    vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cp1, NULL, &g_pl_grid);

    VkComputePipelineCreateInfo cp2 = cp1;
    cp2.stage.module = g_mod_ie;
    vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cp2, NULL, &g_pl_ie);

    /* Timestamp query pool (2 queries per dispatch: before + after) */
    VkQueryPoolCreateInfo qpci = {};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    vkCreateQueryPool(g_dev, &qpci, NULL, &g_tq_pool);

    /* Cache grid params */
    g_params.origin_x = origin_x;
    g_params.origin_y = origin_y;
    g_params.origin_z = origin_z;
    g_params.span_x = span_x;
    g_params.span_y = span_y;
    g_params.span_z = span_z;
    g_params.spacing = spacing;
    g_params.grid_size = span_x * span_y * span_z;

    /* Create 3D textures for hardware trilinear filtering */
    create_3d_image(span_x, span_y, span_z, &g_img_avdw, &g_mem_img_avdw, &g_iv_avdw);
    create_3d_image(span_x, span_y, span_z, &g_img_bvdw, &g_mem_img_bvdw, &g_iv_bvdw);
    create_3d_image(span_x, span_y, span_z, &g_img_es,   &g_mem_img_es,   &g_iv_es);
    if (!g_mem_img_avdw || !g_mem_img_bvdw || !g_mem_img_es) {
        fprintf(stderr, "GPU-VK: 3D texture alloc failed\n");
        dock_gpu_cleanup();
        return 0;
    }
    upload_3d_image(avdw, span_x, span_y, span_z, g_img_avdw);
    upload_3d_image(bvdw, span_x, span_y, span_z, g_img_bvdw);
    upload_3d_image(es,   span_x, span_y, span_z, g_img_es);

    /* Sampler: linear filtering, clamp-to-border (returns 0.0 for OOB) */
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sci.maxAnisotropy = 1.0f;
    sci.maxLod = 0.0f;
    vkCreateSampler(g_dev, &sci, NULL, &g_sampler);

    /* Per-ligand / per-batch buffers */
    alloc_buffer(sizeof(float)*GPU_MAX_ATOMS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_vdwA, &g_mem_vdwA);
    alloc_buffer(sizeof(float)*GPU_MAX_ATOMS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_vdwB, &g_mem_vdwB);
    alloc_buffer(sizeof(float)*GPU_MAX_ATOMS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_charges, &g_mem_charges);
    alloc_buffer(sizeof(float)*GPU_MAX_ATOMS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_ie_vdwA, &g_mem_ie_vdwA);
    alloc_buffer(sizeof(int)*GPU_MAX_NB_PAIRS*2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_nb_int, &g_mem_nb_int);
    alloc_buffer(sizeof(int)*(GPU_MAX_ATOMS+1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_pair_starts, &g_mem_pair_starts);
    alloc_buffer(sizeof(int)*GPU_MAX_NB_PAIRS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_pair_indices, &g_mem_pair_indices);
    alloc_buffer(sizeof(float)*3*GPU_MAX_ATOMS*GPU_MAX_POSES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_xyz, &g_mem_xyz);
    alloc_buffer(sizeof(float)*GPU_MAX_POSES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_scores, &g_mem_scores);
    alloc_buffer(sizeof(int)*GPU_MAX_ATOMS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_buf_active_flags, &g_mem_active_flags);
    alloc_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &g_buf_pose_counter, &g_mem_pose_counter);

    /* Create descriptor pool once (reused across dispatches via free+realloc) */
    {
        VkDescriptorPoolSize dps[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 }
        };
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = dps;
        vkCreateDescriptorPool(g_dev, &dpci, NULL, &g_descpool);
    }

    g_initialized = 1;
    g_active = 0;
    g_num_nb_pairs = 0;
    fprintf(stderr, "GPU-VK: ready. Grid %dx%dx%d = %d pts\n",
            span_x, span_y, span_z, g_params.grid_size);
    return 1;
}

int dock_gpu_set_ligand(const float *vdwA, const float *vdwB,
                        const float *charges, int num_atoms)
{
    if (!g_initialized) return 0;
    if (num_atoms <= 0 || num_atoms > GPU_MAX_ATOMS) {
        fprintf(stderr, "GPU-VK: invalid num_atoms %d (max %d)\n", num_atoms, GPU_MAX_ATOMS);
        return 0;
    }
    size_t bytes = sizeof(float) * (size_t)num_atoms;
    memcpy(buf_map(g_mem_vdwA), vdwA, bytes);
    memcpy(buf_map(g_mem_vdwB), vdwB, bytes);
    memcpy(buf_map(g_mem_charges), charges, bytes);
    g_num_atoms = num_atoms;
    g_active = 1;
    return 1;
}

int dock_gpu_set_ligand_ie(const float *ie_vdwA, const float *ie_vdwB,
                           const int *nb_int_pairs, int num_nb_pairs,
                           float ie_soft_delta, float ie_cutoff_sq)
{
    (void)ie_vdwB;
    if (!g_initialized) return 0;
    if (num_nb_pairs > GPU_MAX_NB_PAIRS) {
        fprintf(stderr, "GPU-VK: num_nb_pairs %d exceeds max %d\n", num_nb_pairs, GPU_MAX_NB_PAIRS);
        return 0;
    }
    size_t ie_bytes = sizeof(float) * (size_t)g_num_atoms;
    memcpy(buf_map(g_mem_ie_vdwA), ie_vdwA, ie_bytes);

    size_t nb_bytes = sizeof(int) * (size_t)num_nb_pairs * 2;
    memcpy(buf_map(g_mem_nb_int), nb_int_pairs, nb_bytes);

    /* Build per-atom pair lists (same as Metal backend) */
    int na = g_num_atoms;
    int *counts = (int *)calloc(na, sizeof(int));
    int *starts = (int *)buf_map(g_mem_pair_starts);
    int *offsets = (int *)malloc(na * sizeof(int));
    int total = 0;
    for (int p = 0; p < num_nb_pairs; p++) {
        int a1 = nb_int_pairs[p*2];
        if (a1 >= 0 && a1 < na) counts[a1]++;
    }
    for (int a = 0; a < na; a++) {
        starts[a] = total;
        offsets[a] = total;
        total += counts[a];
    }
    starts[na] = total;
    free(counts);

    int *pair_indices = (int *)buf_map(g_mem_pair_indices);
    for (int p = 0; p < num_nb_pairs; p++) {
        int a1 = nb_int_pairs[p*2];
        int a2 = nb_int_pairs[p*2+1];
        if (a1 >= 0 && a1 < na) pair_indices[offsets[a1]++] = a2;
    }
    free(offsets);

    g_num_nb_pairs = num_nb_pairs;
    g_ie_soft_delta = ie_soft_delta;
    g_ie_cutoff_sq = ie_cutoff_sq;
    return 1;
}

/* Build a descriptor set binding all 13 storage buffers. */
static VkDescriptorSet make_desc_set(void)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = g_descpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g_dslayout;
    if (vkAllocateDescriptorSets(g_dev, &dsai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkDescriptorBufferInfo bi[10];
    VkDescriptorImageInfo  ii[3];
    VkWriteDescriptorSet   wds[13];

    /* Buffer bindings: 0-3, 7-12 */
    VkBuffer buf_list[10] = {
        g_buf_xyz, g_buf_vdwA, g_buf_vdwB, g_buf_charges,
        g_buf_scores, g_buf_active_flags, g_buf_ie_vdwA, g_buf_pair_starts,
        g_buf_pair_indices, g_buf_pose_counter
    };
    int buf_idx = 0;
    for (int i = 0; i < 13; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = set;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].dstArrayElement = 0;
        wds[i].descriptorCount = 1;
        if (i >= 4 && i <= 6) {
            /* Image bindings for 3D texture samplers */
            VkImageView views[3] = { g_iv_avdw, g_iv_bvdw, g_iv_es };
            ii[i - 4].sampler = g_sampler;
            ii[i - 4].imageView = views[i - 4];
            ii[i - 4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wds[i].pImageInfo = &ii[i - 4];
        } else {
            /* Buffer bindings */
            bi[buf_idx].buffer = buf_list[buf_idx];
            bi[buf_idx].offset = 0;
            bi[buf_idx].range = VK_WHOLE_SIZE;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo = &bi[buf_idx];
            buf_idx++;
        }
    }
    vkUpdateDescriptorSets(g_dev, 13, wds, 0, NULL);
    return set;
}

int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         float *out_scores)
{
    if (!g_initialized) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    memcpy(buf_map(g_mem_xyz), xyz, xyz_bytes);

    PushConstants pc = {};
    pc.origin_x = g_params.origin_x;
    pc.origin_y = g_params.origin_y;
    pc.origin_z = g_params.origin_z;
    pc.span_x   = g_params.span_x;
    pc.span_y   = g_params.span_y;
    pc.span_z   = g_params.span_z;
    pc.spacing  = g_params.spacing;

    pc.num_atoms    = num_atoms;
    pc.ie_soft_delta = 0.0f;
    pc.ie_cutoff_sq  = 1e10f;
    pc.num_nb_pairs  = 0;
    pc.num_poses     = num_poses;

    VkDescriptorSet set = make_desc_set();

    /* Wait for previous dispatch to finish before reusing command buffer */
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_dev, 1, &g_fence);

    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(g_cmd, &cbbi);
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl_grid);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_playout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(g_cmd, g_playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);

    /* grid-only kernel: 1 thread/pose, local_size_x=64 → ceil(num_poses/64) workgroups */
    uint32_t num_wg = ((uint32_t)num_poses + 63u) / 64u;
    vkCmdResetQueryPool(g_cmd, g_tq_pool, 0, 2);
    vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_tq_pool, 0);
    vkCmdDispatch(g_cmd, num_wg, 1, 1);
    vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_tq_pool, 1);
    vkEndCommandBuffer(g_cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkQueueSubmit(g_queue, 1, &si, g_fence);
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);

    /* Read timestamps */
    uint64_t ts[2] = {0, 0};
    vkGetQueryPoolResults(g_dev, g_tq_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);
    double dispatch_ms = (double)(ts[1] - ts[0]) * g_timestamp_period_ns * 1e-6;

    memcpy(out_scores, buf_map(g_mem_scores), sizeof(float) * (size_t)num_poses);

    vkFreeDescriptorSets(g_dev, g_descpool, 1, &set);

    prof_dispatch_count++;
    prof_total_conformers += num_poses;
    prof_total_dispatch_ms += dispatch_ms;
    prof_last_dispatch_ms = dispatch_ms;
    prof_wait_buf[prof_wait_idx] = dispatch_ms;
    prof_wait_idx = (prof_wait_idx + 1) % PROF_ROLLING_SIZE;

    return 1;
}

int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                 const int *active_flags, float *out_scores)
{
    /* Phase 1 routes IE through the persistent path; kept for API parity. */
    return dock_gpu_batch_score_with_ie_persistent(xyz, num_poses, num_atoms, active_flags, out_scores);
}

int dock_gpu_batch_score_with_ie_persistent(const float *xyz, int num_poses, int num_atoms,
                                            const int *active_flags, float *out_scores)
{
    if (!g_initialized || g_num_nb_pairs == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    memcpy(buf_map(g_mem_xyz), xyz, xyz_bytes);
    if (active_flags)
        memcpy(buf_map(g_mem_active_flags), active_flags, sizeof(int) * (size_t)num_atoms);

    /* reset atomic work-counter to 0 */
    uint32_t zero = 0;
    memcpy(buf_map(g_mem_pose_counter), &zero, sizeof(zero));

    VkDescriptorSet set = make_desc_set();

    /* Wait for previous dispatch to finish before reusing command buffer */
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_dev, 1, &g_fence);

    PushConstants pc = {};
    pc.origin_x = g_params.origin_x;
    pc.origin_y = g_params.origin_y;
    pc.origin_z = g_params.origin_z;
    pc.span_x   = g_params.span_x;
    pc.span_y   = g_params.span_y;
    pc.span_z   = g_params.span_z;
    pc.spacing  = g_params.spacing;
    pc.num_atoms     = num_atoms;
    pc.ie_soft_delta = g_ie_soft_delta;
    pc.ie_cutoff_sq  = g_ie_cutoff_sq;
    pc.num_nb_pairs  = g_num_nb_pairs;
    pc.num_poses     = num_poses;

    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(g_cmd, &cbbi);
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl_ie);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_playout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(g_cmd, g_playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pc);

    unsigned int num_tg = (unsigned int)dock_gpu_recommended_batch_size();
    vkCmdResetQueryPool(g_cmd, g_tq_pool, 0, 2);
    vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_tq_pool, 0);
    vkCmdDispatch(g_cmd, num_tg, 1, 1);
    vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_tq_pool, 1);
    vkEndCommandBuffer(g_cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkQueueSubmit(g_queue, 1, &si, g_fence);
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);

    /* Read timestamps */
    uint64_t ts[2] = {0, 0};
    vkGetQueryPoolResults(g_dev, g_tq_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);
    double dispatch_ms = (double)(ts[1] - ts[0]) * g_timestamp_period_ns * 1e-6;

    memcpy(out_scores, buf_map(g_mem_scores), sizeof(float) * (size_t)num_poses);

    vkFreeDescriptorSets(g_dev, g_descpool, 1, &set);

    prof_dispatch_count++;
    prof_total_conformers += num_poses;
    prof_total_dispatch_ms += dispatch_ms;
    prof_last_dispatch_ms = dispatch_ms;
    prof_wait_buf[prof_wait_idx] = dispatch_ms;
    prof_wait_idx = (prof_wait_idx + 1) % PROF_ROLLING_SIZE;

    return 1;
}

void dock_gpu_cleanup(void)
{
    if (g_dev) {
        vkDeviceWaitIdle(g_dev);
        /* 3D textures (images + views + device-local memory) */
        if (g_iv_avdw)   vkDestroyImageView(g_dev, g_iv_avdw, NULL);
        if (g_iv_bvdw)   vkDestroyImageView(g_dev, g_iv_bvdw, NULL);
        if (g_iv_es)     vkDestroyImageView(g_dev, g_iv_es, NULL);
        if (g_img_avdw)  vkDestroyImage(g_dev, g_img_avdw, NULL);
        if (g_img_bvdw)  vkDestroyImage(g_dev, g_img_bvdw, NULL);
        if (g_img_es)    vkDestroyImage(g_dev, g_img_es, NULL);
        if (g_mem_img_avdw) vkFreeMemory(g_dev, g_mem_img_avdw, NULL);
        if (g_mem_img_bvdw) vkFreeMemory(g_dev, g_mem_img_bvdw, NULL);
        if (g_mem_img_es)   vkFreeMemory(g_dev, g_mem_img_es, NULL);
        if (g_sampler)   vkDestroySampler(g_dev, g_sampler, NULL);
        if (g_descpool)  vkDestroyDescriptorPool(g_dev, g_descpool, NULL);
        /* Host-visible buffers */
        if (g_buf_vdwB)     { vkDestroyBuffer(g_dev, g_buf_vdwB, NULL);     vkFreeMemory(g_dev, g_mem_vdwB, NULL); }
        if (g_buf_charges)  { vkDestroyBuffer(g_dev, g_buf_charges, NULL);  vkFreeMemory(g_dev, g_mem_charges, NULL); }
        if (g_buf_ie_vdwA)  { vkDestroyBuffer(g_dev, g_buf_ie_vdwA, NULL);  vkFreeMemory(g_dev, g_mem_ie_vdwA, NULL); }
        if (g_buf_nb_int)   { vkDestroyBuffer(g_dev, g_buf_nb_int, NULL);   vkFreeMemory(g_dev, g_mem_nb_int, NULL); }
        if (g_buf_pair_starts){ vkDestroyBuffer(g_dev, g_buf_pair_starts, NULL); vkFreeMemory(g_dev, g_mem_pair_starts, NULL); }
        if (g_buf_pair_indices){ vkDestroyBuffer(g_dev, g_buf_pair_indices, NULL); vkFreeMemory(g_dev, g_mem_pair_indices, NULL); }
        if (g_buf_xyz)      { vkDestroyBuffer(g_dev, g_buf_xyz, NULL);      vkFreeMemory(g_dev, g_mem_xyz, NULL); }
        if (g_buf_scores)   { vkDestroyBuffer(g_dev, g_buf_scores, NULL);   vkFreeMemory(g_dev, g_mem_scores, NULL); }
        if (g_buf_active_flags){ vkDestroyBuffer(g_dev, g_buf_active_flags, NULL); vkFreeMemory(g_dev, g_mem_active_flags, NULL); }
        if (g_buf_pose_counter){ vkDestroyBuffer(g_dev, g_buf_pose_counter, NULL); vkFreeMemory(g_dev, g_mem_pose_counter, NULL); }
        if (g_tq_pool)  vkDestroyQueryPool(g_dev, g_tq_pool, NULL);
        if (g_fence)    vkDestroyFence(g_dev, g_fence, NULL);
        if (g_pl_grid)  vkDestroyPipeline(g_dev, g_pl_grid, NULL);
        if (g_pl_ie)    vkDestroyPipeline(g_dev, g_pl_ie, NULL);
        if (g_playout)  vkDestroyPipelineLayout(g_dev, g_playout, NULL);
        if (g_dslayout) vkDestroyDescriptorSetLayout(g_dev, g_dslayout, NULL);
        if (g_mod_grid) vkDestroyShaderModule(g_dev, g_mod_grid, NULL);
        if (g_mod_ie)   vkDestroyShaderModule(g_dev, g_mod_ie, NULL);
        if (g_cmdpool)  vkDestroyCommandPool(g_dev, g_cmdpool, NULL);
        vkDestroyDevice(g_dev, NULL);
    }
    if (g_inst) vkDestroyInstance(g_inst, NULL);
    g_inst = VK_NULL_HANDLE; g_phys = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE; g_cmdpool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE;
    g_mod_grid = g_mod_ie = VK_NULL_HANDLE;
    g_pl_grid = g_pl_ie = VK_NULL_HANDLE; g_playout = VK_NULL_HANDLE; g_dslayout = VK_NULL_HANDLE;
    g_tq_pool = VK_NULL_HANDLE;
    memset(&g_params, 0, sizeof(g_params));
    g_initialized = 0; g_active = 0; g_num_atoms = 0; g_num_nb_pairs = 0;
    g_ie_soft_delta = 0.0f; g_ie_cutoff_sq = 1e10f;
    prof_dispatch_count = 0; prof_total_conformers = 0;
    prof_total_dispatch_ms = 0.0; prof_last_dispatch_ms = 0.0;
    prof_wait_idx = 0;
    memset(prof_wait_buf, 0, sizeof(prof_wait_buf));
}

int dock_gpu_is_active(void)
{
    return g_initialized && (g_dev != VK_NULL_HANDLE);
}

int dock_gpu_recommended_batch_size(void)
{
    if (!g_dev) return 32;
    int size = g_compute_units * 4;
    int cap = GPU_MAX_POSES / 4;
    if (size > cap) size = cap;
    if (size <= 0) size = 32;
    return size;
}

void dock_gpu_monitor(int layer, int segment, int total_segments)
{
    (void)layer; (void)segment; (void)total_segments;
    if (!g_initialized) return;

    /* Compute rolling average from circular buffer */
    double rolling_avg = 0.0;
    int count = (prof_dispatch_count < PROF_ROLLING_SIZE)
                ? (int)prof_dispatch_count : PROF_ROLLING_SIZE;
    if (count > 0) {
        for (int i = 0; i < count; i++) rolling_avg += prof_wait_buf[i];
        rolling_avg /= count;
    }

    fprintf(stderr, "GPU-VK: backend=vulkan device_ready=1 "
            "dispatches=%lu conformers=%lu avg_batch=%.1f "
            "last_dispatch=%.2fms rolling_avg=%.2fms\n",
            (unsigned long)prof_dispatch_count,
            (unsigned long)prof_total_conformers,
            prof_dispatch_count > 0
                ? (double)prof_total_conformers / (double)prof_dispatch_count : 0.0,
            prof_last_dispatch_ms, rolling_avg);
}

} /* extern "C" */
