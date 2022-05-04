#define TINYOBJLOADER_IMPLEMENTATION 
#define PI 3.1415926535897932385
#define EPSILON FLT_EPSILON

#include <iostream>
#include <chrono>

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

enum class BVH_Type{NONE = 0, LBVH = 1, SAHBVH = 2};
enum class Render_Type{NONE = 0, NORMAL = 1, HEATMAP = 2};

int main(int argc, char *argv[]){
  std::string filename = "sample_models/large_11k.obj";
  int samples_per_pixel = 30;
  int image_height = 512;
  int image_width = 512;
  int max_bounces = 5;
  float cam_x = 0.0;
  float cam_y = 0.0;
  float cam_z = 0.0;
  Render_Type render_type = Render_Type::NORMAL;
  char* output_filename = "output.ppm";
  BVH_Type bvh_type = BVH_Type::LBVH;

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
    if(!strcmp(flag, "-bvh"))
      bvh_type = (BVH_Type)atoi(parameter);
    if(!strcmp(flag, "-r") || !strcmp(flag, "--render"))
      render_type = (Render_Type)atoi(parameter);
    if(!strcmp(flag, "-x"))
      cam_x = atof(parameter);
    if(!strcmp(flag, "-y"))
      cam_y = atof(parameter);
    if(!strcmp(flag, "-z"))
      cam_z = atof(parameter);
  }

  //Try to read .obj from disk and create necessary geometry buffers on the GPU.
  ObjLoader obj(filename);
  AABB scene_bounding_box        = obj.getSceneBoundingBox();
  Triangle* ptr_device_triangles = obj.createDeviceTriangleBuffer();
  Vec3* ptr_device_vertices      = obj.createDeviceVertexBuffer();
  Vec3* ptr_device_normals       = obj.createDeviceNormalBuffer();

  //BUG/Minor Issue: Calling the constructor for both of these classes will allocate 
  //                 device memory that might not be used.
  LBVH lbvh(ptr_device_triangles, obj.triangle_count, ptr_device_vertices, obj.vertex_count, scene_bounding_box);
  SAHBVH sahbvh(ptr_device_triangles, obj.triangle_count, ptr_device_vertices, obj.vertex_count, scene_bounding_box);

  printf("Constr (us)\tRender (us)\n"); //Print benchmark output table headers

  //Depending on user choice, construct BVH.
  Node* ptr_device_tree = nullptr;
  if(bvh_type == BVH_Type::LBVH)
    ptr_device_tree = lbvh.construct();   // Construct Karras 2012
  else if(bvh_type == BVH_Type::SAHBVH) 
    ptr_device_tree = sahbvh.construct(); // Construct Wald   2007


  // ----------- RENDER -----------
  if(render_type == Render_Type::NONE)
    return 0;

  RenderConfig config(image_width, image_height, samples_per_pixel, max_bounces, 1337);

  Camera cam = Camera(config.img_width, config.img_height, 90.0f, 1.0f, Vec3(cam_x, cam_y, cam_z));
  Raytracer raytracer = Raytracer(config, ptr_device_vertices, ptr_device_normals, ptr_device_triangles, obj.index_count);

  //Benchmark rendering
  std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();

  Vec3* ptr_device_img;
  if(render_type == Render_Type::HEATMAP){
    Vec3* device_traversal_statistics = nullptr;
    ptr_device_img = raytracer.renderTraversalHeatmap(cam, ptr_device_tree, &device_traversal_statistics);
    
    int image_size = config.img_width * config.img_height;
    Vec3* host_traversal_statistics = (Vec3*)malloc(image_size * sizeof(Vec3));
    checkCudaErrors(cudaMemcpy(host_traversal_statistics, device_traversal_statistics, image_size * sizeof(Vec3), cudaMemcpyDeviceToHost));
    FILE *fp = fopen("traversal_steps.txt", "w");
    for (int i = 0; i < config.img_width * config.img_height; i++) {
      int steps = (int)host_traversal_statistics[i].x();
      fprintf(fp, "%i\n", steps);
    }
    fprintf(fp, "\n");
    fclose(fp);
  }
  else{
    if(bvh_type == BVH_Type::NONE)
      ptr_device_img = raytracer.render(cam);
    else
      ptr_device_img = raytracer.render(cam, ptr_device_tree);
  }
  
  std::chrono::steady_clock::time_point stop = std::chrono::high_resolution_clock::now();
  long long duration_us = std::chrono::duration_cast<std::chrono::duration<long long, std::micro>>(stop - start).count();
  printf("\t\t%lli\n", duration_us);

  //Copy framebuffer from device to host and save to disk.
  Image render_output = Image(config.img_width, config.img_height);
  render_output.copyFromDevice(ptr_device_img, config.img_height * config.img_width);

  render_output.save(output_filename);

  cudaFree(ptr_device_triangles);
  cudaFree(ptr_device_vertices);
  cudaFree(ptr_device_normals);
  return 0;
}