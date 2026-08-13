/*                                                                    */
/*                        Copyright UCSF, 2026                        */
/*                                                                    */

/*
  Vulkan GPU backend for dock6 batch scoring.

  Implements the GPU abstraction API defined in score_dock_gpu.h using
  Vulkan (portable across macOS/Linux/Windows).  Targeted first at the
  Steam Deck (AMD RDNA2 / RADV).

  Design notes / differences from the Metal backend:
    * Grid data is stored as 3D textures (VkImage TYPE_3D, R32_SFLOAT)
      with hardware trilinear filtering via sampler3D, matching the
      Metal backend's texture3d<float> approach.
    * Buffers use HOST_VISIBLE | HOST_COHERENT storage.  The Deck is a
      unified-memory APU, so this gives the same perf model as Metal's
      Shared storage with no staging copies.  Grid 3D images use
      DEVICE_LOCAL with staging-buffer upload (optimal tiling required
      for hardware texture filtering).
    * Shaders are authored in GLSL and compiled to SPIR-V at runtime via
      shaderc (mirrors Metal compiling a *.metal string at runtime, and
      keeps the build free of a shader-compile step).  SPIR-V is cached
      to disk to avoid recompilation on subsequent runs.
    * The persistent-threadgroup + atomic work-counter kernel maps
      directly: a storage buffer of uint32_t + atomicAdd over a fixed
      number of workgroups.
    * SIMD-group reduction becomes subgroup reduction via
      subgroupAdd() (VK 1.1 shaderSubgroupArithmetic, available on
      RADV).  Requires subgroup size ≤ workgroup size (guaranteed by spec).
    * Memory allocation uses VMA (Vulkan Memory Allocator) for optimal
      sub-allocation and reduced vkAllocateMemory overhead.
    * Descriptors use VK_KHR_push_descriptor to eliminate descriptor
      pool/set allocation; writes are pushed into the command buffer
      at dispatch time.
    * Single command buffer + single fence (synchronous dispatch).
 */

#include "score_dock_gpu.h"
#include "score_dock_gpu_vulkan.h"

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) ({ \
    void *ptr = NULL; \
    posix_memalign(&ptr, (alignment) < sizeof(void*) ? sizeof(void*) : (alignment), (size)); \
    ptr; \
})
#define VMA_SYSTEM_ALIGNED_FREE(ptr) free(ptr)
#include "vk_mem_alloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>
#include <vector>

#define GPU_MAX_POSES       4096
#define GPU_MAX_ATOMS       512
#define GPU_MAX_NB_PAIRS    32768

/* GLSL source for the persistent IE compute kernel. */
static const char* shader_src = R"glsl(
#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

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
layout(std430, binding = 9) buffer BIEA   { float ie_vdwA[]; };
layout(std430, binding = 10) buffer BSTART { int   pair_starts[]; };
layout(std430, binding = 11) buffer BIDX   { int   pair_indices[]; };
layout(std430, binding = 12) buffer BCNT   { uint  pose_counter[]; };

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

shared float tg_partial[64];
shared uint tg_candidate;

void kernel_persistent() {
    uint tid = gl_LocalInvocationID.x;
    uint tg_size = gl_WorkGroupSize.x;
    uint simd_idx = tid / gl_SubgroupSize;

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
                if (a2 < 0 || a2 >= p.num_atoms) continue;
                if (active_flags[a2] == 0) continue;
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
            float final_score = 0.0;
            uint num_subgroups = tg_size / gl_SubgroupSize;
            for (uint i = 0u; i < num_subgroups; i++)
                final_score += tg_partial[i];
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
        if (active_flags[a] != 0) {
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

static VkShaderModule    g_mod_grid   = VK_NULL_HANDLE;
static VkShaderModule    g_mod_ie     = VK_NULL_HANDLE;
static VkPipelineLayout  g_playout    = VK_NULL_HANDLE;
static VkPipeline        g_pl_grid    = VK_NULL_HANDLE;
static VkPipeline        g_pl_ie      = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_dslayout = VK_NULL_HANDLE;

/* VMA allocator */
static VmaAllocator g_vma = VK_NULL_HANDLE;

/* Buffer wrapper: VkBuffer + VmaAllocation + cached mapped pointer */
struct GpuBuf {
    VkBuffer      buf = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void         *map  = NULL;
};

static GpuBuf g_xyz, g_vdwA, g_vdwB, g_charges, g_scores;
static GpuBuf g_active_flags, g_ie_vdwA;
static GpuBuf g_pair_starts, g_pair_indices, g_pose_counter;

/* 3D textures for hardware trilinear filtering */
static VkImage         g_img_avdw = VK_NULL_HANDLE, g_img_bvdw = VK_NULL_HANDLE, g_img_es = VK_NULL_HANDLE;
static VmaAllocation   g_img_alloc_avdw = VK_NULL_HANDLE, g_img_alloc_bvdw = VK_NULL_HANDLE, g_img_alloc_es = VK_NULL_HANDLE;
static VkImageView     g_iv_avdw = VK_NULL_HANDLE, g_iv_bvdw = VK_NULL_HANDLE, g_iv_es = VK_NULL_HANDLE;
static VkSampler       g_sampler = VK_NULL_HANDLE;

/* Pipeline cache for faster reinitialization */
static VkPipelineCache g_pcache = VK_NULL_HANDLE;

/* Push descriptor function pointer (VK_KHR_push_descriptor) */
static PFN_vkCmdPushDescriptorSetKHR fp_vkCmdPushDescriptorSetKHR = NULL;

/* Pre-built push descriptor write array (built once at init, used every dispatch). */
#define NUM_BINDINGS 13
#define NUM_BUF_INFOS 10
#define NUM_IMG_INFOS 3
static VkDescriptorBufferInfo g_pd_bi[NUM_BUF_INFOS];
static VkDescriptorImageInfo  g_pd_ii[NUM_IMG_INFOS];
static VkWriteDescriptorSet   g_pd_w[NUM_BINDINGS];

static DockGridParams g_params;
static int  g_initialized = 0;
static int  g_num_atoms   = 0;
static int  g_num_nb_pairs = 0;
static float g_ie_soft_delta = 0.0;
static float g_ie_cutoff_sq  = 1e10f;
static int  g_compute_units = 0;

/* Timestamp queries */
static VkQueryPool g_tq_pool = VK_NULL_HANDLE;
static float g_timestamp_period_ns = 1.0f;

/* Single fence for synchronous dispatch */
static VkFence g_fence = VK_NULL_HANDLE;

/* Profiling counters */
static uint64_t prof_dispatch_count    = 0;
static uint64_t prof_total_conformers  = 0;
static double   prof_total_dispatch_ms = 0.0;
static double   prof_last_dispatch_ms  = 0.0;
#define PROF_ROLLING_SIZE 64
static double   prof_wait_buf[PROF_ROLLING_SIZE];
static int      prof_wait_idx = 0;

static void fill_push_constants(PushConstants *pc, int num_atoms, int num_poses,
                                float ie_soft_delta, float ie_cutoff_sq, int num_nb_pairs) {
    pc->origin_x    = g_params.origin_x;
    pc->origin_y    = g_params.origin_y;
    pc->origin_z    = g_params.origin_z;
    pc->span_x      = g_params.span_x;
    pc->span_y      = g_params.span_y;
    pc->span_z      = g_params.span_z;
    pc->spacing     = g_params.spacing;
    pc->num_atoms   = num_atoms;
    pc->ie_soft_delta = ie_soft_delta;
    pc->ie_cutoff_sq  = ie_cutoff_sq;
    pc->num_nb_pairs  = num_nb_pairs;
    pc->num_poses     = num_poses;
}

/* ================================================================== */
/*  Vulkan / VMA helpers                                               */
/* ================================================================== */

/* Allocate a host-visible, coherent buffer via VMA. */
static bool alloc_buf(VkDeviceSize size, VkBufferUsageFlags usage, GpuBuf *b) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vci = {};
    vci.usage = VMA_MEMORY_USAGE_AUTO;
    vci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkResult r = vmaCreateBuffer(g_vma, &bci, &vci, &b->buf, &b->alloc, NULL);
    if (r != VK_SUCCESS) return false;

    r = vmaMapMemory(g_vma, b->alloc, &b->map);
    return r == VK_SUCCESS;
}

static void destroy_buf(GpuBuf *b) {
    if (b->alloc) {
        vmaUnmapMemory(g_vma, b->alloc);
        vmaDestroyBuffer(g_vma, b->buf, b->alloc);
    }
    b->buf = VK_NULL_HANDLE;
    b->alloc = VK_NULL_HANDLE;
    b->map = NULL;
}

/* Create a 3D texture image (device-local, optimal tiling) via VMA */
static bool create_3d_image(int sx, int sy, int sz,
                            VkImage *img, VmaAllocation *alloc, VkImageView *view) {
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

    VmaAllocationCreateInfo vci = {};
    vci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult r = vmaCreateImage(g_vma, &ici, &vci, img, alloc, NULL);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: vmaCreateImage failed (%d)\n", (int)r);
        return false;
    }

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    ivci.format = VK_FORMAT_R32_SFLOAT;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    ivci.image = *img;
    if (vkCreateImageView(g_dev, &ivci, NULL, view) != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: vkCreateImageView failed\n");
        vmaDestroyImage(g_vma, *img, *alloc);
        *img = VK_NULL_HANDLE;
        *alloc = VK_NULL_HANDLE;
        *view = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

/* Upload all 3 grid 3D images in a single command buffer submission. */
static bool upload_all_3d_images(const float *data[3], int sx, int sy, int sz,
                                 VkImage imgs[3]) {
    VkDeviceSize per_image = sizeof(float) * (VkDeviceSize)sx * sy * sz;
    VkDeviceSize total = per_image * 3;

    VkBuffer staging_buf = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = total;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vci = {};
    vci.usage = VMA_MEMORY_USAGE_AUTO;
    vci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkResult r = vmaCreateBuffer(g_vma, &bci, &vci, &staging_buf, &staging_alloc, NULL);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: staging buffer alloc failed (%d)\n", (int)r);
        return false;
    }

    void *mapped = NULL;
    r = vmaMapMemory(g_vma, staging_alloc, &mapped);
    if (r != VK_SUCCESS || !mapped) {
        fprintf(stderr, "GPU-VK: staging map failed (%d)\n", (int)r);
        vmaDestroyBuffer(g_vma, staging_buf, staging_alloc);
        return false;
    }
    for (int i = 0; i < 3; i++)
        memcpy((char*)mapped + per_image * i, data[i], (size_t)per_image);
    vmaUnmapMemory(g_vma, staging_alloc);

    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cmd, &cbbi);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {(uint32_t)sx, (uint32_t)sy, (uint32_t)sz};

    for (int i = 0; i < 3; i++) {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = imgs[i];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, NULL, 0, NULL, 1, &barrier);

        region.bufferOffset = per_image * i;
        vkCmdCopyBufferToImage(g_cmd, staging_buf, imgs[i],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, NULL, 0, NULL, 1, &barrier);
    }

    vkEndCommandBuffer(g_cmd);

    vkResetFences(g_dev, 1, &g_fence);
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkQueueSubmit(g_queue, 1, &si, g_fence);
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);

    vmaDestroyBuffer(g_vma, staging_buf, staging_alloc);
    return true;
}

/* Build the static push descriptor write array.
 * Called once after all buffers + images are created. */
static void prepare_push_descriptors(void) {
    g_pd_bi[0]  = {g_xyz.buf,          0, VK_WHOLE_SIZE};
    g_pd_bi[1]  = {g_vdwA.buf,         0, VK_WHOLE_SIZE};
    g_pd_bi[2]  = {g_vdwB.buf,         0, VK_WHOLE_SIZE};
    g_pd_bi[3]  = {g_charges.buf,      0, VK_WHOLE_SIZE};
    g_pd_bi[4]  = {g_scores.buf,       0, VK_WHOLE_SIZE};
    g_pd_bi[5]  = {g_active_flags.buf, 0, VK_WHOLE_SIZE};
    g_pd_bi[6]  = {g_ie_vdwA.buf,      0, VK_WHOLE_SIZE};
    g_pd_bi[7]  = {g_pair_starts.buf,  0, VK_WHOLE_SIZE};
    g_pd_bi[8]  = {g_pair_indices.buf, 0, VK_WHOLE_SIZE};
    g_pd_bi[9]  = {g_pose_counter.buf, 0, VK_WHOLE_SIZE};
    g_pd_ii[0]  = {g_sampler, g_iv_avdw, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    g_pd_ii[1]  = {g_sampler, g_iv_bvdw, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    g_pd_ii[2]  = {g_sampler, g_iv_es,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    int bi_idx = 0, ii_idx = 0;
    for (int i = 0; i < NUM_BINDINGS; i++) {
        g_pd_w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        g_pd_w[i].pNext = NULL;
        g_pd_w[i].dstSet = VK_NULL_HANDLE;
        g_pd_w[i].dstBinding = (uint32_t)i;
        g_pd_w[i].dstArrayElement = 0;
        g_pd_w[i].descriptorCount = 1;
        if (i >= 4 && i <= 6) {
            g_pd_w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            g_pd_w[i].pImageInfo = &g_pd_ii[ii_idx++];
        } else {
            g_pd_w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            g_pd_w[i].pBufferInfo = &g_pd_bi[bi_idx++];
        }
    }
}

/* ================================================================== */
/*  SPIR-V disk cache + shader compilation                             */
/* ================================================================== */

static size_t djb2_hash(const char *src) {
    size_t h = 5381;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++)
        h = ((h << 5) + h) + *p;
    return h;
}

static const char *g_spirv_cache_dir = "/tmp/dock6_spirv_cache";

/* Compile GLSL -> SPIR-V via shaderc, return a VkShaderModule.
 * Caches compiled SPIR-V to disk to avoid recompilation. */
static VkShaderModule compile_module(const char *src, const char *label) {
    /* Ensure cache directory exists */
    mkdir(g_spirv_cache_dir, 0755);

    size_t h = djb2_hash(src);
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path), "%s/%s_%zu.spv",
             g_spirv_cache_dir, label, h);

    /* Try loading cached SPIR-V */
    FILE *f = fopen(cache_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0) {
            void *data = malloc((size_t)sz);
            if (data && (long)fread(data, 1, (size_t)sz, f) == sz) {
                fclose(f);
                VkShaderModuleCreateInfo smci = {};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = (size_t)sz;
                smci.pCode = (const uint32_t *)data;
                VkShaderModule mod = VK_NULL_HANDLE;
                vkCreateShaderModule(g_dev, &smci, NULL, &mod);
                free(data);
                if (mod) {
                    fprintf(stderr, "GPU-VK: loaded cached SPIR-V '%s' (%ld bytes)\n", label, sz);
                    return mod;
                }
            } else {
                free(data);
            }
        }
        fclose(f);
    }

    /* Compile fresh */
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

    /* Cache SPIR-V to disk */
    f = fopen(cache_path, "wb");
    if (f) {
        fwrite(spv, 1, spv_size, f);
        fclose(f);
        fprintf(stderr, "GPU-VK: compiled + cached SPIR-V '%s' (%zu bytes)\n", label, spv_size);
    }

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

static const char *g_pipeline_cache_path = "/tmp/dock6_vulkan_pipeline_cache.bin";
static int g_pipeline_cache_from_disk = 0;

static void load_pipeline_cache(void) {
    FILE *f = fopen(g_pipeline_cache_path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return; }
    fseek(f, 0, SEEK_SET);
    void *data = malloc((size_t)sz);
    if (!data) { fclose(f); return; }
    size_t rd = fread(data, 1, (size_t)sz, f);
    fclose(f);
    if ((long)rd != sz) { free(data); return; }

    VkPipelineCacheCreateInfo pcci = {};
    pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pcci.initialDataSize = (size_t)sz;
    pcci.pInitialData = data;
    vkCreatePipelineCache(g_dev, &pcci, NULL, &g_pcache);
    free(data);
    g_pipeline_cache_from_disk = 1;
    fprintf(stderr, "GPU-VK: loaded pipeline cache (%ld bytes)\n", sz);
}

static void save_pipeline_cache(void) {
    if (!g_pcache || g_pipeline_cache_from_disk) return;
    size_t sz = 0;
    vkGetPipelineCacheData(g_dev, g_pcache, &sz, NULL);
    if (sz == 0) return;
    void *data = malloc(sz);
    if (!data) return;
    vkGetPipelineCacheData(g_dev, g_pcache, &sz, data);
    FILE *f = fopen(g_pipeline_cache_path, "wb");
    if (f) {
        fwrite(data, 1, sz, f);
        fclose(f);
        fprintf(stderr, "GPU-VK: saved pipeline cache (%zu bytes)\n", sz);
    }
    free(data);
}

/* ================================================================== */
/*  Common dispatch + profiling helper                                 */
/* ================================================================== */

static void dispatch_compute(VkPipeline pipeline, uint32_t num_wg,
                             const PushConstants *pc,
                             float *out_scores, int num_poses) {
    vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_dev, 1, &g_fence);

    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(g_cmd, &cbbi);
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    fp_vkCmdPushDescriptorSetKHR(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 g_playout, 0, NUM_BINDINGS, g_pd_w);
    vkCmdPushConstants(g_cmd, g_playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), pc);

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

    uint64_t ts[2] = {0, 0};
    vkGetQueryPoolResults(g_dev, g_tq_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);
    double dispatch_ms = (double)(ts[1] - ts[0]) * g_timestamp_period_ns * 1e-6;

    memcpy(out_scores, g_scores.map, sizeof(float) * (size_t)num_poses);

    prof_dispatch_count++;
    prof_total_conformers += num_poses;
    prof_total_dispatch_ms += dispatch_ms;
    prof_last_dispatch_ms = dispatch_ms;
    prof_wait_buf[prof_wait_idx] = dispatch_ms;
    prof_wait_idx = (prof_wait_idx + 1) % PROF_ROLLING_SIZE;
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

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_phys, &props);
    fprintf(stderr, "GPU-VK: device: %s\n", props.deviceName);
    fflush(stderr);

    g_compute_units = (int)(props.limits.maxComputeWorkGroupInvocations / 32);
    if (g_compute_units <= 0) g_compute_units = 8;

    g_timestamp_period_ns = props.limits.timestampPeriod;
    if (g_timestamp_period_ns <= 0.0f) g_timestamp_period_ns = 1.0f;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, NULL);
    std::vector<VkQueueFamilyProperties> qf(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, qf.data());
    g_queue_idx = 0;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_queue_idx = i; break; }
    }

    const char *dev_exts[] = { VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME };

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
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    if (vkCreateDevice(g_phys, &dci, NULL, &g_dev) != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: device creation failed — CPU fallback\n");
        return 0;
    }
    vkGetDeviceQueue(g_dev, g_queue_idx, 0, &g_queue);

    fp_vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetDeviceProcAddr(g_dev, "vkCmdPushDescriptorSetKHR");
    if (!fp_vkCmdPushDescriptorSetKHR) {
        fprintf(stderr, "GPU-VK: vkCmdPushDescriptorSetKHR not available — CPU fallback\n");
        dock_gpu_cleanup();
        return 0;
    }

    VmaAllocatorCreateInfo vaci = {};
    vaci.flags = 0;
    vaci.vulkanApiVersion = VK_API_VERSION_1_1;
    vaci.physicalDevice = g_phys;
    vaci.device = g_dev;
    vaci.instance = g_inst;
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vaci.pVulkanFunctions = &vulkanFunctions;
    if (vmaCreateAllocator(&vaci, &g_vma) != VK_SUCCESS) {
        fprintf(stderr, "GPU-VK: VMA allocator creation failed — CPU fallback\n");
        dock_gpu_cleanup();
        return 0;
    }

    /* Command pool + single command buffer */
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

    /* Single fence — created SIGNALED so first wait passes */
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(g_dev, &fci, NULL, &g_fence);

    /* Shaders (SPIR-V cached to disk) */
    g_mod_grid = compile_module(shader_src_grid, "grid_batch_score");
    g_mod_ie    = compile_module(shader_src,        "persistent_ie");
    if (!g_mod_grid || !g_mod_ie) {
        fprintf(stderr, "GPU-VK: shader compile failed — CPU fallback\n");
        dock_gpu_cleanup();
        return 0;
    }

    /* Descriptor set layout: bindings 0-3,7-12 = storage buffers,
     * bindings 4-6 = combined image samplers.
     * Created with PUSH_DESCRIPTOR_BIT_KHR. */
    VkDescriptorSetLayoutBinding binds[NUM_BINDINGS];
    for (int i = 0; i < NUM_BINDINGS; i++) {
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
    dslci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    dslci.bindingCount = NUM_BINDINGS;
    dslci.pBindings = binds;
    vkCreateDescriptorSetLayout(g_dev, &dslci, NULL, &g_dslayout);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g_dslayout;
    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(g_dev, &plci, NULL, &g_playout);

    load_pipeline_cache();

    VkComputePipelineCreateInfo cp1 = {};
    cp1.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp1.layout = g_playout;
    cp1.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp1.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp1.stage.module = g_mod_grid;
    cp1.stage.pName = "main";
    vkCreateComputePipelines(g_dev, g_pcache, 1, &cp1, NULL, &g_pl_grid);

    VkComputePipelineCreateInfo cp2 = cp1;
    cp2.stage.module = g_mod_ie;
    vkCreateComputePipelines(g_dev, g_pcache, 1, &cp2, NULL, &g_pl_ie);

    save_pipeline_cache();

    VkQueryPoolCreateInfo qpci = {};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    vkCreateQueryPool(g_dev, &qpci, NULL, &g_tq_pool);

    g_params.origin_x = origin_x;
    g_params.origin_y = origin_y;
    g_params.origin_z = origin_z;
    g_params.span_x = span_x;
    g_params.span_y = span_y;
    g_params.span_z = span_z;
    g_params.spacing = spacing;
    g_params.grid_size = span_x * span_y * span_z;

    /* Create 3D textures for hardware trilinear filtering */
    if (!create_3d_image(span_x, span_y, span_z, &g_img_avdw, &g_img_alloc_avdw, &g_iv_avdw) ||
        !create_3d_image(span_x, span_y, span_z, &g_img_bvdw, &g_img_alloc_bvdw, &g_iv_bvdw) ||
        !create_3d_image(span_x, span_y, span_z, &g_img_es,   &g_img_alloc_es,   &g_iv_es)) {
        fprintf(stderr, "GPU-VK: 3D texture creation failed\n");
        dock_gpu_cleanup();
        return 0;
    }
    const float *grid_data[3] = {avdw, bvdw, es};
    VkImage grid_imgs[3] = {g_img_avdw, g_img_bvdw, g_img_es};
    if (!upload_all_3d_images(grid_data, span_x, span_y, span_z, grid_imgs)) {
        fprintf(stderr, "GPU-VK: 3D texture upload failed\n");
        dock_gpu_cleanup();
        return 0;
    }

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

    /* Per-ligand / per-batch buffers via VMA */
    #define ALLOC_BUF(sz, usage, b) \
        do { if (!alloc_buf(sz, usage, b)) { \
            fprintf(stderr, "GPU-VK: buffer alloc failed\n"); dock_gpu_cleanup(); return 0; \
        } } while(0)
    ALLOC_BUF(sizeof(float)*GPU_MAX_ATOMS,             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_vdwA);
    ALLOC_BUF(sizeof(float)*GPU_MAX_ATOMS,             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_vdwB);
    ALLOC_BUF(sizeof(float)*GPU_MAX_ATOMS,             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_charges);
    ALLOC_BUF(sizeof(float)*GPU_MAX_ATOMS,             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_ie_vdwA);
    ALLOC_BUF(sizeof(int)*(GPU_MAX_ATOMS+1),           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_pair_starts);
    ALLOC_BUF(sizeof(int)*GPU_MAX_NB_PAIRS,            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_pair_indices);
    ALLOC_BUF(sizeof(float)*3*GPU_MAX_ATOMS*GPU_MAX_POSES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_xyz);
    ALLOC_BUF(sizeof(float)*GPU_MAX_POSES,             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_scores);
    ALLOC_BUF(sizeof(int)*GPU_MAX_ATOMS,               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_active_flags);
    ALLOC_BUF(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g_pose_counter);
    #undef ALLOC_BUF

    /* Build push descriptor writes (static — reused every dispatch) */
    prepare_push_descriptors();

    g_initialized = 1;
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
    memcpy(g_vdwA.map, vdwA, bytes);
    memcpy(g_vdwB.map, vdwB, bytes);
    memcpy(g_charges.map, charges, bytes);

    /* Initialize active_flags to all-active (all atoms enabled) */
    int *af = (int *)g_active_flags.map;
    for (int i = 0; i < num_atoms; i++) af[i] = 1;

    g_num_atoms = num_atoms;
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
    memcpy(g_ie_vdwA.map, ie_vdwA, ie_bytes);

    /* Build per-atom pair lists (same as Metal backend) */
    int na = g_num_atoms;
    int *counts = (int *)calloc(na, sizeof(int));
    int *offsets = (int *)malloc(na * sizeof(int));
    if (!counts || !offsets) {
        fprintf(stderr, "GPU-VK: set_ligand_ie alloc failed\n");
        free(counts); free(offsets);
        return 0;
    }
    int *starts = (int *)g_pair_starts.map;
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

    int *pair_indices = (int *)g_pair_indices.map;
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

int dock_gpu_batch_score(const float *xyz, int num_poses, int num_atoms,
                         const int *active_flags, float *out_scores)
{
    if (!g_initialized) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    memcpy(g_xyz.map, xyz, xyz_bytes);
    if (active_flags) {
        memcpy(g_active_flags.map, active_flags, sizeof(int) * (size_t)num_atoms);
    } else {
        int *af = (int *)g_active_flags.map;
        for (int i = 0; i < num_atoms; i++) af[i] = 1;
    }

    PushConstants pc = {};
    fill_push_constants(&pc, num_atoms, num_poses, 0.0f, 1e10f, 0);

    uint32_t num_wg = ((uint32_t)num_poses + 63u) / 64u;
    dispatch_compute(g_pl_grid, num_wg, &pc, out_scores, num_poses);
    return 1;
}

int dock_gpu_batch_score_with_ie(const float *xyz, int num_poses, int num_atoms,
                                 const int *active_flags, float *out_scores)
{
    return dock_gpu_batch_score_with_ie_persistent(xyz, num_poses, num_atoms, active_flags, out_scores);
}

int dock_gpu_batch_score_with_ie_persistent(const float *xyz, int num_poses, int num_atoms,
                                            const int *active_flags, float *out_scores)
{
    if (!g_initialized || g_num_nb_pairs == 0) return 0;
    if (num_poses > GPU_MAX_POSES || num_atoms > GPU_MAX_ATOMS) return 0;

    size_t xyz_bytes = sizeof(float) * (size_t)num_poses * (size_t)num_atoms * 3;
    memcpy(g_xyz.map, xyz, xyz_bytes);
    if (active_flags) {
        memcpy(g_active_flags.map, active_flags, sizeof(int) * (size_t)num_atoms);
    } else {
        int *af = (int *)g_active_flags.map;
        for (int i = 0; i < num_atoms; i++) af[i] = 1;
    }

    /* reset atomic work-counter to 0 */
    uint32_t zero = 0;
    memcpy(g_pose_counter.map, &zero, sizeof(zero));

    PushConstants pc = {};
    fill_push_constants(&pc, num_atoms, num_poses, g_ie_soft_delta, g_ie_cutoff_sq, g_num_nb_pairs);

    unsigned int num_tg = (unsigned int)dock_gpu_recommended_batch_size();
    dispatch_compute(g_pl_ie, num_tg, &pc, out_scores, num_poses);
    return 1;
}

void dock_gpu_cleanup(void)
{
    if (g_dev) {
        vkDeviceWaitIdle(g_dev);

        /* 3D textures */
        if (g_iv_avdw)   vkDestroyImageView(g_dev, g_iv_avdw, NULL);
        if (g_iv_bvdw)   vkDestroyImageView(g_dev, g_iv_bvdw, NULL);
        if (g_iv_es)     vkDestroyImageView(g_dev, g_iv_es, NULL);
        if (g_img_alloc_avdw) vmaDestroyImage(g_vma, g_img_avdw, g_img_alloc_avdw);
        if (g_img_alloc_bvdw) vmaDestroyImage(g_vma, g_img_bvdw, g_img_alloc_bvdw);
        if (g_img_alloc_es)   vmaDestroyImage(g_vma, g_img_es, g_img_alloc_es);
        if (g_sampler)   vkDestroySampler(g_dev, g_sampler, NULL);

        /* Host-visible buffers via VMA */
        destroy_buf(&g_vdwA);
        destroy_buf(&g_vdwB);
        destroy_buf(&g_charges);
        destroy_buf(&g_ie_vdwA);
        destroy_buf(&g_pair_starts);
        destroy_buf(&g_pair_indices);
        destroy_buf(&g_xyz);
        destroy_buf(&g_scores);
        destroy_buf(&g_active_flags);
        destroy_buf(&g_pose_counter);

        /* VMA allocator */
        if (g_vma) { vmaDestroyAllocator(g_vma); g_vma = VK_NULL_HANDLE; }

        if (g_tq_pool)  vkDestroyQueryPool(g_dev, g_tq_pool, NULL);
        if (g_fence)    vkDestroyFence(g_dev, g_fence, NULL);
        if (g_pl_grid)  vkDestroyPipeline(g_dev, g_pl_grid, NULL);
        if (g_pl_ie)    vkDestroyPipeline(g_dev, g_pl_ie, NULL);
        if (g_playout)  vkDestroyPipelineLayout(g_dev, g_playout, NULL);
        if (g_dslayout) vkDestroyDescriptorSetLayout(g_dev, g_dslayout, NULL);
        if (g_mod_grid) vkDestroyShaderModule(g_dev, g_mod_grid, NULL);
        if (g_mod_ie)   vkDestroyShaderModule(g_dev, g_mod_ie, NULL);
        if (g_cmdpool)  vkDestroyCommandPool(g_dev, g_cmdpool, NULL);
        if (g_pcache)   vkDestroyPipelineCache(g_dev, g_pcache, NULL);
        vkDestroyDevice(g_dev, NULL);
    }
    if (g_inst) vkDestroyInstance(g_inst, NULL);

    /* Reset all handles */
    g_inst = VK_NULL_HANDLE; g_phys = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE; g_cmdpool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE;
    g_fence = VK_NULL_HANDLE;
    g_mod_grid = g_mod_ie = VK_NULL_HANDLE;
    g_pl_grid = g_pl_ie = VK_NULL_HANDLE; g_playout = VK_NULL_HANDLE; g_dslayout = VK_NULL_HANDLE;
    g_tq_pool = VK_NULL_HANDLE; g_pcache = VK_NULL_HANDLE;
    g_sampler = VK_NULL_HANDLE;
    g_img_avdw = g_img_bvdw = g_img_es = VK_NULL_HANDLE;
    g_img_alloc_avdw = g_img_alloc_bvdw = g_img_alloc_es = VK_NULL_HANDLE;
    g_iv_avdw = g_iv_bvdw = g_iv_es = VK_NULL_HANDLE;
    g_vma = VK_NULL_HANDLE;
    fp_vkCmdPushDescriptorSetKHR = NULL;
    memset(&g_params, 0, sizeof(g_params));
    g_initialized = 0; g_num_atoms = 0; g_num_nb_pairs = 0;
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
    int size = g_compute_units * 8;
    int cap = GPU_MAX_POSES / 2;
    if (size > cap) size = cap;
    if (size <= 0) size = 32;
    return size;
}

void dock_gpu_monitor(int layer, int segment, int total_segments)
{
    (void)layer; (void)segment; (void)total_segments;
    if (!g_initialized) return;

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

/* ---- Virtual-screening LUT API: not implemented on this backend. ----
   Returning 0 makes the VS driver fall back to CPU scoring for the
   affected ligand (registration failure drops the LUT row, batch calls
   return failure and the pool marks its slots converged), so a Vulkan
   build links and stays correct — just unaccelerated for VS batches. */

int dock_gpu_vs_register_ligand(int lig_idx,
                                const float *vdwA, const float *vdwB,
                                const float *charges, const int *active_flags,
                                const float *ie_vdwA,
                                const int *nb_int_pairs, int num_nb_pairs,
                                int num_atoms,
                                float ie_soft_delta, float ie_cutoff_sq)
{
    (void)lig_idx; (void)vdwA; (void)vdwB; (void)charges;
    (void)active_flags; (void)ie_vdwA; (void)nb_int_pairs;
    (void)num_nb_pairs; (void)num_atoms;
    (void)ie_soft_delta; (void)ie_cutoff_sq;
    return 0;
}

int dock_gpu_vs_max_ligands(void)
{
    return 0;
}

int dock_gpu_batch_score_vs(const float *xyz, int num_poses, int num_atoms,
                            const int *pose_lig, float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores;
    return 0;
}

int dock_gpu_batch_score_vs_grid(const float *xyz, int num_poses,
                                 int num_atoms, const int *pose_lig,
                                 float *out_scores)
{
    (void)xyz; (void)num_poses; (void)num_atoms; (void)pose_lig;
    (void)out_scores;
    return 0;
}

int dock_gpu_grid_bounds(float *minx, float *miny, float *minz,
                         float *maxx, float *maxy, float *maxz)
{
    (void)minx; (void)miny; (void)minz; (void)maxx; (void)maxy; (void)maxz;
    return 0;
}

} /* extern "C" */
