// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#include <array>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <fileformats/stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <fileformats/tiny_obj_loader.h>
#include <nvh/fileoperations.hpp>  // For nvh::loadFile
#define NVVK_ALLOC_DEDICATED
#include <nvvk/allocator_vk.hpp>  // For NVVK memory allocators
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>  // For nvvk::DescriptorSetContainer
#include <nvvk/raytraceKHR_vk.hpp>     // For nvvk::RaytracingBuilderKHR
#include <nvvk/shaders_vk.hpp>         // For nvvk::createShaderModule
#include <nvvk/structs_vk.hpp>         // For nvvk::make

static const uint64_t render_width     = 800;
static const uint64_t render_height    = 600;
static const uint32_t workgroup_width  = 16;
static const uint32_t workgroup_height = 8;
const int NUM_SAMPLES = 64;

[[using spirv: buffer, binding(0)]]
vec3 shader_imageData[];

[[using spirv: uniform, binding(1)]]
accelerationStructure shader_tlas;

[[using spirv: buffer, binding(2)]]
vec3 shader_vertices[];

[[using spirv: buffer, binding(3)]]
uint shader_indices[];

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
inline float stepAndOutputRNGFloat(uint& rngState) {
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = rngState * 747796405 + 1;
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

// Returns the color of the sky in a given direction (in linear color space)
inline vec3 skyColor(vec3 direction) {
  return (direction.y > 0) ?
    mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y) :
    vec3(0.03f);
}

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
};

inline HitInfo getObjectHitInfo(gl_rayQuery& rayQuery)
{
  HitInfo result;
  // Get the ID of the triangle
  const int primitiveID = gl_rayQueryGetIntersectionPrimitiveIndex(rayQuery, true);

  // Get the indices of the vertices of the triangle
  const uint i0 = shader_indices[3 * primitiveID + 0];
  const uint i1 = shader_indices[3 * primitiveID + 1];
  const uint i2 = shader_indices[3 * primitiveID + 2];

  // Get the vertices of the triangle
  const vec3 v0 = shader_vertices[i0];
  const vec3 v1 = shader_vertices[i1];
  const vec3 v2 = shader_vertices[i2];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics;
  barycentrics.yz = gl_rayQueryGetIntersectionBarycentrics(rayQuery, true);
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // For the main tutorial, object space is the same as world space:
  result.worldPosition = objectPos;

  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  result.worldNormal = objectNormal;

  result.color = vec3(0.7f);
  return result;
}

[[using spirv: comp, local_size(workgroup_width, workgroup_height)]]
void compute_shader() {
  const ivec2 resolution(render_width, render_height);
  const ivec2 pixel = ivec2(glcomp_GlobalInvocationID.xy);

  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    return;

  // State of the random number generator.
  uint rngState = resolution.x * pixel.y + pixel.x;  // Initial seed

  const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);
  const float fovVerticalSlope = 1.0 / 5.0;

  // The sum of the colors of all of the samples.
  vec3 summedPixelColor(0.0);

  // Limit the kernel to trace at most 64 samples.
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {

    // Rays always originate at the camera for now. In the future, they'll
    // bounce around the scene.
    vec3 rayOrigin = cameraOrigin;

    const vec2 randomPixelCenter = vec2(pixel) + 
      vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));

    vec2 screenUV = vec2(2 * randomPixelCenter + 1 - vec2(resolution)) / resolution.y;
    screenUV.y = -screenUV.y;

    vec3 rayDirection(fovVerticalSlope * screenUV, -1.0);
    rayDirection = normalize(rayDirection);

    vec3 accumulatedRayColor(1);  // The amount of light that made it to the end of the current ray.


    // Limit the kernel to trace at most 32 segments.
    for(int tracedSegments = 0; tracedSegments < 32; tracedSegments++)
    {
      // Trace the ray and see if and where it intersects the scene!
      // First, initialize a ray query object:
      gl_rayQuery rayQuery;
      gl_rayQueryInitialize(rayQuery,              // Ray query
                            shader_tlas,           // Top-level acceleration structure
                            gl_RayFlagsOpaque,     // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            rayOrigin,             // Ray origin
                            0.0,                   // Minimum t-value
                            rayDirection,          // Ray direction
                            10000.0);              // Maximum t-value
      while(gl_rayQueryProceed(rayQuery));

      // Get the type of committed (true) intersection - nothing, a triangle, or
      // a generated object
      if(gl_rayQueryGetIntersectionType(rayQuery, true) == 
        gl_RayQueryCommittedIntersectionTriangle)
      {
        // Ray hit a triangle
        HitInfo hitInfo = getObjectHitInfo(rayQuery);

        // Apply color absorption
        accumulatedRayColor *= hitInfo.color;

        // Flip the normal so it points against the ray direction:
        hitInfo.worldNormal = faceforward(hitInfo.worldNormal, rayDirection, hitInfo.worldNormal);

        // Start a new ray at the hit position, but offset it slightly along the normal:
        rayOrigin = hitInfo.worldPosition + 0.0001f * hitInfo.worldNormal;

        // For a random diffuse bounce direction, we follow the approach of
        // Ray Tracing in One Weekend, and generate a random point on a sphere
        // of radius 1 centered at the normal. This uses the random_unit_vector
        // function from chapter 8.5:
        const float theta = 2 * M_PIf32 * stepAndOutputRNGFloat(rngState);  // Random in [0, 2pi]
        const float u     = 2 * stepAndOutputRNGFloat(rngState) - 1;  // Random in [-1, 1]
        const float r     = sqrt(1 - u * u);
        rayDirection      = hitInfo.worldNormal + vec3(r * cos(theta), r * sin(theta), u);
        // Then normalize the ray direction:
        rayDirection = normalize(rayDirection);
      }
      else
      {
        // Ray hit the sky
        accumulatedRayColor *= skyColor(rayDirection);

        // Sum this with the pixel's other samples.
        // (Note that we treat a ray that didn't find a light source as if it had
        // an accumulated color of (0, 0, 0)).
        summedPixelColor += accumulatedRayColor;

        break;
      }
    }
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  shader_imageData[linearIndex] =  summedPixelColor / NUM_SAMPLES;
}





////////////////////////////////////////////////////////////////////////////////

VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
  VkCommandBufferAllocateInfo cmdAllocInfo = nvvk::make<VkCommandBufferAllocateInfo>();
  cmdAllocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool                 = cmdPool;
  cmdAllocInfo.commandBufferCount          = 1;
  VkCommandBuffer cmdBuffer;
  NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
  VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
  return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
  NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
  VkSubmitInfo submitInfo       = nvvk::make<VkSubmitInfo>();
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmdBuffer;
  NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
  NVVK_CHECK(vkQueueWaitIdle(queue));
  vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
  VkBufferDeviceAddressInfo addressInfo = nvvk::make<VkBufferDeviceAddressInfo>();
  addressInfo.buffer                    = buffer;
  return vkGetBufferDeviceAddress(device, &addressInfo);
}

int main(int argc, const char** argv)
{
  // Create the Vulkan context, consisting of an instance, device, physical device, and queues.
  nvvk::ContextCreateInfo deviceInfo;  // One can modify this to load different extensions or pick the Vulkan core version
  deviceInfo.apiMajor = 1;             // Specify the version of Vulkan we'll use
  deviceInfo.apiMinor = 2;
  // Required by KHR_acceleration_structure; allows work to be offloaded onto background threads and parallelized
  deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = nvvk::make<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
  deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
  deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

  nvvk::Context context;     // Encapsulates device state in a single object
  context.init(deviceInfo);  // Initialize the context
  // Device must support acceleration structures and ray queries:
  assert(asFeatures.accelerationStructure == VK_TRUE && rayQueryFeatures.rayQuery == VK_TRUE);

  // Create the allocator
  nvvk::AllocatorDedicated allocator;
  allocator.init(context, context.m_physicalDevice);

  // Create a buffer
  VkDeviceSize       bufferSizeBytes  = render_width * render_height * 3 * sizeof(float);
  VkBufferCreateInfo bufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
  bufferCreateInfo.size               = bufferSizeBytes;
  bufferCreateInfo.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this buffer's memory.
  // VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU caches this memory.
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU side of cache management
  // is handled automatically, with potentially slower reads/writes.
  nvvk::BufferDedicated buffer = allocator.createBuffer(bufferCreateInfo,                         //
                                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT       //
                                                            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT  //
                                                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Load the mesh of the first shape from an OBJ file
  std::vector<std::string> searchPaths = {PROJECT_ABSDIRECTORY, PROJECT_ABSDIRECTORY "../", PROJECT_RELDIRECTORY,
                                          PROJECT_RELDIRECTORY "../", PROJECT_NAME};
  tinyobj::ObjReader       reader;  // Used to read an OBJ file
  reader.ParseFromFile(nvh::findFile("scenes/CornellBox-Original-Merged.obj", searchPaths));
  assert(reader.Valid());  // Make sure tinyobj was able to parse this file
  const std::vector<tinyobj::real_t>   objVertices = reader.GetAttrib().GetVertices();
  const std::vector<tinyobj::shape_t>& objShapes   = reader.GetShapes();  // All shapes in the file
  assert(objShapes.size() == 1);                                          // Check that this file has only one shape
  const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape
  // Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
  std::vector<uint32_t> objIndices;
  objIndices.reserve(objShape.mesh.indices.size());
  for(const tinyobj::index_t& index : objShape.mesh.indices)
  {
    objIndices.push_back(index.vertex_index);
  }

  // Create the command pool
  VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
  cmdPoolInfo.queueFamilyIndex        = context.m_queueGCT;
  VkCommandPool cmdPool;
  NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

  // Upload the vertex and index buffers to the GPU.
  nvvk::BufferDedicated vertexBuffer, indexBuffer;
  {
    // Start a command buffer for uploading the buffers
    VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
    // We get these buffers' device addresses, and use them as storage buffers and build inputs.
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
    indexBuffer  = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);

    EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
    allocator.finalizeAndReleaseStaging();
  }

  // Describe the bottom-level acceleration structure (BLAS)
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;
  {
    nvvk::RaytracingBuilderKHR::BlasInput blas;
    // Get the device addresses of the vertex and index buffers
    VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(context, vertexBuffer.buffer);
    VkDeviceAddress indexBufferAddress  = GetBufferDeviceAddress(context, indexBuffer.buffer);
    // Specify where the builder can find the vertices and indices for triangles, and their formats:
    VkAccelerationStructureGeometryTrianglesDataKHR triangles = nvvk::make<VkAccelerationStructureGeometryTrianglesDataKHR>();
    triangles.vertexFormat                                    = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress                        = vertexBufferAddress;
    triangles.vertexStride                                    = 3 * sizeof(float);
    triangles.maxVertex                                       = static_cast<uint32_t>(objVertices.size() - 1);
    triangles.indexType                                       = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress                         = indexBufferAddress;
    triangles.transformData.deviceAddress                     = 0;  // No transform
    // Create a VkAccelerationStructureGeometryKHR object that says it handles opaque triangles and points to the above:
    VkAccelerationStructureGeometryKHR geometry = nvvk::make<VkAccelerationStructureGeometryKHR>();
    geometry.geometry.triangles                 = triangles;
    geometry.geometryType                       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags                              = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blas.asGeometry.push_back(geometry);
    // Create offset info that allows us to say how many triangles and vertices to read
    VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
    offsetInfo.firstVertex     = 0;
    offsetInfo.primitiveCount  = static_cast<uint32_t>(objIndices.size() / 3);  // Number of triangles
    offsetInfo.primitiveOffset = 0;
    offsetInfo.transformOffset = 0;
    blas.asBuildOffsetInfo.push_back(offsetInfo);
    blases.push_back(blas);
  }
  // Create the BLAS
  nvvk::RaytracingBuilderKHR raytracingBuilder;
  raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
  raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  // Create an instance pointing to this BLAS, and build it into a TLAS:
  std::vector<nvvk::RaytracingBuilderKHR::Instance> instances;
  {
    nvvk::RaytracingBuilderKHR::Instance instance;
    instance.transform.identity();  // Set the instance transform to the identity matrix
    instance.instanceCustomId = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
    instance.blasId           = 0;  // The index of the BLAS in `blases` that this instance points to
    instance.hitGroupId = 0;  // Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
    instances.push_back(instance);
  }
  raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  // Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
  // 0 - a storage buffer (the buffer `buffer`)
  // 1 - an acceleration structure (the TLAS)
  // 2 - a storage buffer (the vertex buffer)
  // 3 - a storage buffer (the index buffer)
  nvvk::DescriptorSetContainer descriptorSetContainer(context);
  descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Create a layout from the list of bindings
  descriptorSetContainer.initLayout();
  // Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
  descriptorSetContainer.initPool(1);
  // Create a simple pipeline layout from the descriptor set layout:
  descriptorSetContainer.initPipeLayout();

  // Write values into the descriptor set.
  std::array<VkWriteDescriptorSet, 4> writeDescriptorSets;
  // 0
  VkDescriptorBufferInfo descriptorBufferInfo{};
  descriptorBufferInfo.buffer = buffer.buffer;    // The VkBuffer object
  descriptorBufferInfo.range  = bufferSizeBytes;  // The length of memory to bind; offset is 0.
  writeDescriptorSets[0]      = descriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);
  // 1
  VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
  VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
  descriptorAS.accelerationStructureCount = 1;
  descriptorAS.pAccelerationStructures    = &tlasCopy;
  writeDescriptorSets[1]                  = descriptorSetContainer.makeWrite(0, 1, &descriptorAS);
  // 2
  VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
  vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
  vertexDescriptorBufferInfo.range  = VK_WHOLE_SIZE;
  writeDescriptorSets[2]            = descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);
  // 3
  VkDescriptorBufferInfo indexDescriptorBufferInfo{};
  indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
  indexDescriptorBufferInfo.range  = VK_WHOLE_SIZE;
  writeDescriptorSets[3]           = descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);
  vkUpdateDescriptorSets(context,                                            // The context
                         static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
                         writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
                         0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

  // Shader loading and pipeline creation
  VkShaderModule rayTraceModule = nvvk::createShaderModule(context, 
    __spirv_data, __spirv_size / 4);

  // Describes the entrypoint and the stage to use for this shader module in the pipeline
  VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
  shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module                          = rayTraceModule;
  shaderStageCreateInfo.pName                           = @spirv(compute_shader);

  // Create the compute pipeline
  VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
  pipelineCreateInfo.stage                       = shaderStageCreateInfo;
  pipelineCreateInfo.layout                      = descriptorSetContainer.getPipeLayout();
  // Don't modify flags, basePipelineHandle, or basePipelineIndex
  VkPipeline computePipeline;
  NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
                                      VK_NULL_HANDLE,          // Pipeline cache (uses default)
                                      1, &pipelineCreateInfo,  // Compute pipeline create info
                                      nullptr,                 // Allocator (uses default)
                                      &computePipeline));      // Output

  // Create and start recording a command buffer
  VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

  // Bind the compute shader pipeline
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
  // Bind the descriptor set
  VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
                          &descriptorSet, 0, nullptr);

  // Run the compute shader with enough workgroups to cover the entire buffer:
  vkCmdDispatch(cmdBuffer, (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
                (uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);

  // Add a command that says "Make it so that memory writes by the compute shader
  // are available to read from the CPU." (In other words, "Flush the GPU caches
  // so the CPU can read the data.") To do this, we use a memory barrier.
  // This is one of the most complex parts of Vulkan, so don't worry if this is
  // confusing! We'll talk about pipeline barriers more in the extras.
  VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
  memoryBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;  // Make shader writes
  memoryBarrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;     // Readable by the CPU
  vkCmdPipelineBarrier(cmdBuffer,                              // The command buffer
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // From the compute shader
                       VK_PIPELINE_STAGE_HOST_BIT,             // To the CPU
                       0,                                      // No special flags
                       1, &memoryBarrier,                      // An array of memory barriers
                       0, nullptr, 0, nullptr);                // No other barriers

  // End and submit the command buffer, then wait for it to finish:
  EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);

  // Get the image data back from the GPU
  void* data = allocator.map(buffer);
  stbi_write_hdr("out.hdr", render_width, render_height, 3, reinterpret_cast<float*>(data));
  allocator.unmap(buffer);

  vkDestroyPipeline(context, computePipeline, nullptr);
  vkDestroyShaderModule(context, rayTraceModule, nullptr);
  descriptorSetContainer.deinit();
  raytracingBuilder.destroy();
  allocator.destroy(vertexBuffer);
  allocator.destroy(indexBuffer);
  vkDestroyCommandPool(context, cmdPool, nullptr);
  allocator.destroy(buffer);
  allocator.deinit();
  context.deinit();  // Don't forget to clean up at the end of the program!
}