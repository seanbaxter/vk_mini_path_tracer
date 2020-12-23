// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <random>
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

static const uint32_t workgroup_width  = 16;
static const uint32_t workgroup_height = 8;

enum bindings {
  binding_image     = 0,
  binding_tlas      = 1,
  binding_vertices  = 2,
  binding_indices   = 3,
};

struct PushConstants
{
  int num_samples;
  int sample_batch;
};

[[spirv::push]]
PushConstants shader_push;

[[using spirv: uniform, format(rgba32f), binding(binding_image)]]
image2D shader_image;

[[using spirv: uniform, binding(binding_tlas)]]
accelerationStructure shader_tlas;

[[using spirv: buffer, binding(binding_vertices)]]
vec3 shader_vertices[];

[[using spirv: buffer, binding(binding_indices)]]
uint shader_indices[];

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
inline float stepAndOutputRNGFloat(uint& rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = rngState * 747796405 + 1;
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}


// Uses the Box-Muller transform to return a normally distributed (centered
// at 0, standard deviation 1) 2D point.
vec2 randomGaussian(uint& rngState)
{
  // Almost uniform in (0, 1] - make sure the value is never 0:
  const float u1    = max(1e-38f, stepAndOutputRNGFloat(rngState));
  const float u2    = stepAndOutputRNGFloat(rngState);  // In [0, 1]
  const float r     = sqrt(-2.0 * log(u1));
  const float theta = 2 * M_PIf32 * u2;  // Random in [0, 2pi]
  return r * vec2(cos(theta), sin(theta));
}

// Returns the color of the sky in a given direction (in linear color space)
inline vec3 skyColor(vec3 direction)
{
  // +y in world space is up, so:
  if(direction.y > 0.0f)
  {
    return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
  }
  else
  {
    return vec3(0.03f);
  }
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

  // Transform from object space to world space:
  const mat4x3 objectToWorld = gl_rayQueryGetIntersectionObjectToWorld(rayQuery, true);
  result.worldPosition       = objectToWorld * vec4(objectPos, 1.0f);

  const vec3 objectNormal = cross(v1 - v0, v2 - v0);
  const mat4x3 objectToWorldInverse = gl_rayQueryGetIntersectionWorldToObject(rayQuery, true);
  result.worldNormal                = normalize((objectNormal * objectToWorldInverse).xyz);

  result.color = vec3(0.7f);
  return result;
}

[[using spirv: comp, local_size(workgroup_width, workgroup_height)]]
void compute_shader() {
  const ivec2 resolution = imageSize(shader_image);
  const ivec2 pixel = ivec2(glcomp_GlobalInvocationID.xy);

  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
    return;

  // State of the random number generator.
  uint rngState = (shader_push.sample_batch * resolution.y + pixel.y) * resolution.x + pixel.x; 

  const vec3 cameraOrigin = vec3(-0.001, 0, 53.0);
  const float fovVerticalSlope = 1.0 / 5.0;

  // The sum of the colors of all of the samples.
  vec3 summedPixelColor (0.0);

  // Limit the kernel to trace at most 64 samples.
  for(int sampleIdx = 0; sampleIdx < shader_push.num_samples; sampleIdx++)
  {

    // Rays always originate at the camera for now. In the future, they'll
    // bounce around the scene.
    vec3 rayOrigin = cameraOrigin;

    const vec2 randomPixelCenter = vec2(pixel) + vec2(0.5) + 
      0.375f * randomGaussian(rngState);

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
        const float u     = 2 * stepAndOutputRNGFloat(rngState) - 1;   // Random in [-1, 1]
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

  vec3 averagePixelColor = summedPixelColor / shader_push.num_samples;
  if(shader_push.sample_batch)
  {
    const vec3 previousAverageColor = imageLoad(shader_image, pixel).rgb;
    averagePixelColor = 
      (shader_push.sample_batch * previousAverageColor + averagePixelColor) / 
      (shader_push.sample_batch + 1);
  }

  // Set the color of the pixel `pixel` in the storage image to `averagePixelColor`:
  imageStore(shader_image, pixel, vec4(averagePixelColor, 0.0));
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
  const int render_width = 800;
  const int render_height = 600;

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

  // Initialize the debug utilities:
  nvvk::DebugUtil debugUtil(context);

  // Create the allocator
  nvvk::AllocatorDedicated allocator;
  allocator.init(context, context.m_physicalDevice);

  // Create an image. Images are more complex than buffers - they can have
  // multiple dimensions, different color+depth formats, be arrays of mips,
  // have multisampling, be tiled in memory in e.g. row-linear order or in an
  // implementation-dependent way (and this layout of memory can depend on
  // what the image is being used for), and be shared across multiple queues.
  // Here's how we specify the image we'll use:
  VkImageCreateInfo imageCreateInfo = nvvk::make<VkImageCreateInfo>();
  imageCreateInfo.imageType         = VK_IMAGE_TYPE_2D;
  // RGB32 images aren't usually supported, so we change this to a RGBA32 image.
  imageCreateInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  // Defines the size of the image:
  imageCreateInfo.extent = {render_width, render_height, 1};
  // The image is an array of length 1, and each element contains only 1 mip:
  imageCreateInfo.mipLevels   = 1;
  imageCreateInfo.arrayLayers = 1;
  // We aren't using MSAA (i.e. the image only contains 1 sample per pixel -
  // note that this isn't the same use of the word "sample" as in ray tracing):
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  // The driver controls the tiling of the image for performance:
  imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  // This image is read and written on the GPU, and data can be transferred
  // from it:
  imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // Image is only used by one queue:
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  // The image must be in either VK_IMAGE_LAYOUT_UNDEFINED or VK_IMAGE_LAYOUT_PREINITIALIZED
  // according to the specification; we'll transition the layout shortly,
  // in the same command buffer used to upload the vertex and index buffers:
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  nvvk::ImageDedicated image    = allocator.createImage(imageCreateInfo);
  debugUtil.setObjectName(image.image, "image");

  // Create an image view for the entire image
  // When we create a descriptor for the image, we'll also need an image view
  // that the descriptor will point to. This specifies what part of the image
  // the descriptor views, and how the descriptor views it.
  VkImageViewCreateInfo imageViewCreateInfo = nvvk::make<VkImageViewCreateInfo>();
  imageViewCreateInfo.image                 = image.image;
  imageViewCreateInfo.viewType              = VK_IMAGE_VIEW_TYPE_2D;
  imageViewCreateInfo.format                = imageCreateInfo.format;
  // We could use imageViewCreateInfo.components to make the components of the
  // image appear to be "swizzled", but we don't want to do that. Luckily,
  // all values are set to VK_COMPONENT_SWIZZLE_IDENTITY, which means
  // "don't change anything", by nvvk::make or zero initialization.
  // This says that the ImageView views the color part of the image (since
  // images can contain depth or stencil aspects):
  imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  // This says that we only look at array layer 0 and mip level 0:
  imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
  imageViewCreateInfo.subresourceRange.layerCount     = 1;
  imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
  imageViewCreateInfo.subresourceRange.levelCount     = 1;
  VkImageView imageView;
  NVVK_CHECK(vkCreateImageView(context, &imageViewCreateInfo, nullptr, &imageView));
  debugUtil.setObjectName(imageView, "imageView");

  // Also create an image using linear tiling that can be accessed from the CPU,
  // much like how we created the buffer in the main tutorial. The first image
  // will be entirely local to the GPU for performance, while this image can
  // be mapped to CPU memory. We'll copy data from the first image to this
  // image in order to read the image data back on the CPU.
  // As before, we'll transition the image layout in the same command buffer
  // used to upload the vertex and index buffers.
  imageCreateInfo.tiling           = VK_IMAGE_TILING_LINEAR;
  imageCreateInfo.usage            = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  nvvk::ImageDedicated imageLinear = allocator.createImage(imageCreateInfo,                           //
                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT       //
                                                               | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  //
                                                               | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  debugUtil.setObjectName(imageLinear.image, "imageLinear");

  // Load the mesh of the first shape from an OBJ file
  std::vector<std::string> searchPaths = {
      PROJECT_ABSDIRECTORY,       PROJECT_ABSDIRECTORY "../",    PROJECT_ABSDIRECTORY "../../", PROJECT_RELDIRECTORY,
      PROJECT_RELDIRECTORY "../", PROJECT_RELDIRECTORY "../../", PROJECT_NAME};
  tinyobj::ObjReader reader;  // Used to read an OBJ file
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
  debugUtil.setObjectName(cmdPool, "cmdPool");

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

    // Also, let's transition the layout of `image` to `VK_IMAGE_LAYOUT_GENERAL`,
    // and the layout of `imageLinear` to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`.
    // Although we use `imageLinear` later, we're transferring its layout as
    // early as possible. For more complex applications, tracking images and
    // operations using a graph is a good way to handle these types of images
    // automatically. However, for this tutorial, we'll show how to write
    // image transitions by hand.

    // To do this, we combine both transitions in a single pipeline barrier.
    // This pipeline barrier will say "Make it so that all writes to memory by
    const VkAccessFlags srcAccesses = 0;  // (since image and imageLinear aren't initially accessible)
    // finish and can be read correctly by
    const VkAccessFlags dstImageAccesses       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;  // for image
    const VkAccessFlags dstImageLinearAccesses = VK_ACCESS_TRANSFER_WRITE_BIT;  // for imageLinear
    // "

    // Here's how to do that:
    const VkPipelineStageFlags srcStages = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
    const VkPipelineStageFlags dstStages = nvvk::makeAccessMaskPipelineStageFlags(dstImageAccesses | dstImageLinearAccesses);
    VkImageMemoryBarrier       imageBarriers[2];
    // Image memory barrier for `image` from UNDEFINED to GENERAL layout:
    imageBarriers[0] = nvvk::makeImageMemoryBarrier(image.image,                    // The VkImage
                                                    srcAccesses, dstImageAccesses,  // Source and destination access masks
                                                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,  // Source and destination layouts
                                                    VK_IMAGE_ASPECT_COLOR_BIT);  // Aspects of an image (color, depth, etc.)
    // Image memory barrier for `imageLinear` from UNDEFINED to TRANSFER_DST_OPTIMAL layout:
    imageBarriers[1] = nvvk::makeImageMemoryBarrier(imageLinear.image,                    // The VkImage
                                                    srcAccesses, dstImageLinearAccesses,  // Source and destination access masks
                                                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Source and dst layouts
                                                    VK_IMAGE_ASPECT_COLOR_BIT);  // Aspects of an image (color, depth, etc.)
    // Include the two image barriers in the pipeline barrier:
    vkCmdPipelineBarrier(uploadCmdBuffer,       // The command buffer
                         srcStages, dstStages,  // Src and dst pipeline stages
                         0,                     // Flags for memory dependencies
                         0, nullptr,            // Global memory barrier objects
                         0, nullptr,            // Buffer memory barrier objects
                         2, imageBarriers);     // Image barrier objects

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
  raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

  // Create 441 instances with random rotations pointing to BLAS 0, and build these instances into a TLAS:
  std::vector<nvvk::RaytracingBuilderKHR::Instance> instances;
  std::default_random_engine                        randomEngine;  // The random number generator
  std::uniform_real_distribution<float>             uniformDist(-0.5f, 0.5f);
  for(int x = -10; x <= 10; x++)
  {
    for(int y = -10; y <= 10; y++)
    {
      nvvk::RaytracingBuilderKHR::Instance instance;
      instance.transform.translate(nvmath::vec3f(float(x), float(y), 0.0f));
      instance.transform.scale(1.0f / 2.7f);
      instance.transform.rotate(uniformDist(randomEngine), nvmath::vec3f(0.0f, 1.0f, 0.0f));
      instance.transform.rotate(uniformDist(randomEngine), nvmath::vec3f(1.0f, 0.0f, 0.0f));
      instance.transform.translate(nvmath::vec3f(0.0f, -1.0f, 0.0f));

      instance.instanceCustomId = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
      instance.blasId           = 0;  // The index of the BLAS in `blases` that this instance points to
      instance.hitGroupId = 0;  // Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
      instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
      instances.push_back(instance);
    }
  }
  raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  // Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
  // 0 - a storage image (the image `image`)
  // 1 - an acceleration structure (the TLAS)
  // 2 - a storage buffer (the vertex buffer)
  // 3 - a storage buffer (the index buffer)
  nvvk::DescriptorSetContainer descriptorSetContainer(context);
  descriptorSetContainer.addBinding(binding_image, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(binding_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(binding_vertices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(binding_indices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Create a layout from the list of bindings
  descriptorSetContainer.initLayout();
  // Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
  descriptorSetContainer.initPool(1);
  // Create a push constant range describing the amount of data for the push constants.
  static_assert(sizeof(PushConstants) % 4 == 0, "Push constant size must be a multiple of 4 per the Vulkan spec!");
  VkPushConstantRange pushConstantRange;
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset     = 0;
  pushConstantRange.size       = sizeof(PushConstants);
  // Create a pipeline layout from the descriptor set layout and push constant range:
  descriptorSetContainer.initPipeLayout(1,                    // Number of push constant ranges
                                        &pushConstantRange);  // Pointer to push constant ranges

  // Write values into the descriptor set.
  std::array<VkWriteDescriptorSet, 4> writeDescriptorSets;
  // Color image
  VkDescriptorImageInfo descriptorImageInfo{};
  descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // The image's layout
  descriptorImageInfo.imageView   = imageView;                // How the image should be accessed
  writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, binding_image /*binding*/, &descriptorImageInfo);
  // Top-level acceleration structure (TLAS)
  VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
  VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
  descriptorAS.accelerationStructureCount = 1;
  descriptorAS.pAccelerationStructures    = &tlasCopy;
  writeDescriptorSets[1]                  = descriptorSetContainer.makeWrite(0, binding_tlas, &descriptorAS);
  // Vertex buffer
  VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
  vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
  vertexDescriptorBufferInfo.range  = VK_WHOLE_SIZE;
  writeDescriptorSets[2] = descriptorSetContainer.makeWrite(0, binding_vertices, &vertexDescriptorBufferInfo);
  // Index buffer
  VkDescriptorBufferInfo indexDescriptorBufferInfo{};
  indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
  indexDescriptorBufferInfo.range  = VK_WHOLE_SIZE;
  writeDescriptorSets[3]           = descriptorSetContainer.makeWrite(0, binding_indices, &indexDescriptorBufferInfo);
  vkUpdateDescriptorSets(context,                                            // The context
                         static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
                         writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
                         0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

  // Shader loading and pipeline creation
  VkShaderModule rayTraceModule = nvvk::createShaderModule(context, 
    __spirv_data, __spirv_size / 4);
  debugUtil.setObjectName(rayTraceModule, "rayTraceModule");

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
  debugUtil.setObjectName(computePipeline, "computePipeline");

  PushConstants pushConstants {
    16, 0
  };
  const uint32_t NUM_SAMPLE_BATCHES = 32;
  for(uint32_t sampleBatch = 0; sampleBatch < NUM_SAMPLE_BATCHES; sampleBatch++)
  {
    // Create and start recording a command buffer
    VkCommandBuffer cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

    // Bind the compute shader pipeline
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    // Bind the descriptor set
    VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
                            &descriptorSet, 0, nullptr);

    // Push push constants:
    pushConstants.sample_batch = sampleBatch;
    vkCmdPushConstants(cmdBuffer,                               // Command buffer
                       descriptorSetContainer.getPipeLayout(),  // Pipeline layout
                       VK_SHADER_STAGE_COMPUTE_BIT,             // Stage flags
                       0,                                       // Offset
                       sizeof(PushConstants),                   // Size in bytes
                       &pushConstants);                         // Data

    // Run the compute shader with enough workgroups to cover the entire buffer:
    vkCmdDispatch(cmdBuffer, (render_width + workgroup_width - 1) / workgroup_width,
                  (render_height + workgroup_height - 1) / workgroup_height, 1);

    // On the last sample batch:
    if(sampleBatch == NUM_SAMPLE_BATCHES - 1)
    {
      // Transition `image` from GENERAL to TRANSFER_SRC_OPTIMAL layout. See the
      // code for uploadCmdBuffer above to see a description of what this does:
      const VkAccessFlags        srcAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      const VkAccessFlags        dstAccesses = VK_ACCESS_TRANSFER_READ_BIT;
      const VkPipelineStageFlags srcStages   = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
      const VkPipelineStageFlags dstStages   = nvvk::makeAccessMaskPipelineStageFlags(dstAccesses);
      const VkImageMemoryBarrier barrier =
          nvvk::makeImageMemoryBarrier(image.image,               // The VkImage
                                       srcAccesses, dstAccesses,  // Src and dst access masks
                                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  // Src and dst layouts
                                       VK_IMAGE_ASPECT_COLOR_BIT);
      vkCmdPipelineBarrier(cmdBuffer,             // Command buffer
                           srcStages, dstStages,  // Src and dst pipeline stages
                           0,                     // Dependency flags
                           0, nullptr,            // Global memory barriers
                           0, nullptr,            // Buffer memory barriers
                           1, &barrier);          // Image memory barriers

      // Now, copy the image (which has layout TRANSFER_SRC_OPTIMAL) to imageLinear
      // (which has layout TRANSFER_DST_OPTIMAL).
      {
        VkImageCopy region;
        // We copy the image aspect, layer 0, mip 0:
        region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount     = 1;
        region.srcSubresource.mipLevel       = 0;
        // (0, 0, 0) in the first image corresponds to (0, 0, 0) in the second image:
        region.srcOffset      = {0, 0, 0};
        region.dstSubresource = region.srcSubresource;
        region.dstOffset      = {0, 0, 0};
        // Copy the entire image:
        region.extent = {render_width, render_height, 1};
        vkCmdCopyImage(cmdBuffer,                             // Command buffer
                       image.image,                           // Source image
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  // Source image layout
                       imageLinear.image,                     // Destination image
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Destination image layout
                       1, &region);                           // Regions
      }

      // Add a command that says "Make it so that memory writes by transfers
      // are available to read from the CPU." (In other words, "Flush the GPU caches
      // so the CPU can read the data.") To do this, we use a memory barrier.
      VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
      memoryBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;  // Make transfer writes
      memoryBarrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;       // Readable by the CPU
      vkCmdPipelineBarrier(cmdBuffer,                                // The command buffer
                           VK_PIPELINE_STAGE_TRANSFER_BIT,           // From transfers
                           VK_PIPELINE_STAGE_HOST_BIT,               // To the CPU
                           0,                                        // No special flags
                           1, &memoryBarrier,                        // An array of memory barriers
                           0, nullptr, 0, nullptr);                  // No other barriers
    }

    // End and submit the command buffer, then wait for it to finish:
    EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);

    nvprintf("Rendered sample batch index %d.\n", sampleBatch);
  }

  // Get the image data back from the GPU
  void* data;
  NVVK_CHECK(vkMapMemory(context, imageLinear.allocation, 0, VK_WHOLE_SIZE, 0, &data));
  stbi_write_hdr("out.hdr", render_width, render_height, 4, reinterpret_cast<float*>(data));
  vkUnmapMemory(context, imageLinear.allocation);

  vkDestroyPipeline(context, computePipeline, nullptr);
  vkDestroyShaderModule(context, rayTraceModule, nullptr);
  descriptorSetContainer.deinit();
  raytracingBuilder.destroy();
  allocator.destroy(vertexBuffer);
  allocator.destroy(indexBuffer);
  vkDestroyCommandPool(context, cmdPool, nullptr);
  allocator.destroy(imageLinear);
  vkDestroyImageView(context, imageView, nullptr);
  allocator.destroy(image);
  allocator.deinit();
  context.deinit();  // Don't forget to clean up at the end of the program!
}
