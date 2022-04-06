#define TINYOBJLOADER_IMPLEMENTATION 
#define PI 3.1415926535897932385
#define EPSILON 0.000001

#include <iostream>

#include "aabb.h"
#include "node.h"
#include "triangle.h"
#include "vec3.h"
#include "vec2.h"
#include "raytracer/render_config.h"
#include "raytracer/ray.h"
#include "raytracer/hit.h"
#include "raytracer/camera.h"
#include "raytracer/raytracer.cuh"
#include "image.h"
#include "obj_loader.h"
#include "math_util.h"

#include "lbvh.cu"
#include "sahbvh.cu"

#include "third_party/cuda_helpers/helper_cuda.h" //checkCudaErrors

enum BVH_Type{ none, lbvh, sahbvh };

int main(int argc, char *argv[]){
  std::string filename = "sample_models/large_210.obj";
  int samples_per_pixel = 30;
  int image_height = 512;
  int image_width = 512;
  int max_bounces = 5;
  char* output_filename = "output.ppm";
  BVH_Type bvh_type = lbvh;

  // ----------- CL ARGUMENTS  -----------
  for (size_t i = 2; i < argc; i+=2){
    char* flag = argv[i-1];
    char* parameter = argv[i];
    if(!strcmp(flag, "-i") ||   !strcmp(flag, "--input"))
      filename = std::string(parameter);
    if(!strcmp(flag, "-o") ||   !strcmp(flag, "--image-output"))
      output_filename = parameter;
    if(!strcmp(flag, "-spp") || !strcmp(flag, "--samples-per-pixel"))
      samples_per_pixel = atoi(parameter);
    if(!strcmp(flag, "-iw") ||  !strcmp(flag, "--image-width"))
      image_width = atoi(parameter);
    if(!strcmp(flag, "-ih") ||  !strcmp(flag, "--image-height"))
      image_height = atoi(parameter);
    if(!strcmp(flag, "--max-depth"))
      max_bounces = atoi(parameter);
    if(!strcmp(flag, "-bvh")){
      bvh_type = (BVH_Type)atoi(parameter);
    }
  }

  //Try to read .obj from disk and create necessary geometry buffers on the GPU.
  ObjLoader obj(filename);
  AABB scene_bounding_box        = obj.getSceneBoundingBox();
  Triangle* ptr_device_triangles = obj.createDeviceTriangleBuffer();
  Vec3* ptr_device_vertices      = obj.createDeviceVertexBuffer();
  Vec3* ptr_device_normals       = obj.createDeviceNormalBuffer();

  //BUG/Minor Issue: Calling the constructor for both of these classes will allocate 
  //                   device memory that might not be used. This should ideally be done
  //                   using a strategy pattern or similar.
  LBVH lbvh(ptr_device_triangles, obj.triangle_count, ptr_device_vertices, obj.vertex_count, scene_bounding_box);
  SAHBVH sahbvh(ptr_device_triangles, obj.triangle_count, ptr_device_vertices, obj.vertex_count, scene_bounding_box);

  //Depending on user choice, construct BVH.
  Node* ptr_device_tree;
  if(bvh_type == BVH_Type::lbvh)
    ptr_device_tree = lbvh.construct(); // Construct Karras 2012
  else if(bvh_type == BVH_Type::sahbvh) // Construct Wald   2007 
    ptr_device_tree = sahbvh.construct();

  // ----------- RENDER -----------
  RenderConfig config(image_width, image_height, samples_per_pixel, max_bounces, 1337);
  Camera cam = Camera(config.img_width, config.img_height, 90.0f, 1.0f, Vec3(0,0,0));
  Raytracer raytracer = Raytracer(config, ptr_device_vertices, ptr_device_normals, ptr_device_triangles, obj.index_count);

  printf("Starting rendering...\n");
  //Render with traversal if a BVH is selected.
  Vec3* ptr_device_img = bvh_type == BVH_Type::none ? raytracer.render(cam) : raytracer.render(cam, ptr_device_tree);
  printf("Render complete.\n");

  //Copy framebuffer to host and save to disk.
  Image render_output = Image(config.img_width, config.img_height);
  render_output.copyFromDevice(ptr_device_img, config.img_height * config.img_width);
  render_output.save(output_filename);

  checkCudaErrors(cudaFree(ptr_device_triangles));
  checkCudaErrors(cudaFree(ptr_device_vertices));
  checkCudaErrors(cudaFree(ptr_device_normals));
  return 0;
}