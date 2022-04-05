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
#include "main.h"
#include "math_util.h"

#include "lbvh.cu"
#include "sahbvh.cu"

#include "third_party/cuda_helpers/helper_cuda.h"      //checkCudaErrors
#include "third_party/tiny_obj_loader.h"

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

  // ----------- LOAD SCENE  -----------
  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filename, reader_config)) {
    if (!reader.Error().empty()) {
        std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
  }

  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  const tinyobj::attrib_t &attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

  std::cout << "\nFile '" << filename << "' loaded." << std::endl;
  int vertex_count = (int)(attrib.vertices.size()) / 3;
  printf("\t# vertices        = %d\n", vertex_count);
  int indices_count = (int)(shapes[0].mesh.indices.size());
  printf("\t# vertex indices  = %d\n", indices_count);
  int normals_count = (int)(attrib.normals.size()) / 3;
  printf("\t# normals         = %d\n", normals_count);
  int poly_count = indices_count / 3;
  printf("\t# triangles       = %d\n\n", poly_count);

  // ------------ Scene bounding box -----------------
  Vec3 min_bounds = Vec3( 100000000.0, 100000000.0,   100000000.0);
  Vec3 max_bounds = Vec3(-100000000.0,-100000000.0,  -100000000.0);

  for (int i = 0; i < attrib.vertices.size(); i+=3){
    float x = attrib.vertices[i  ];
    float y = attrib.vertices[i+1];
    float z = attrib.vertices[i+2];
    min_bounds.e[0] = min(min_bounds.x(), x);
    min_bounds.e[1] = min(min_bounds.y(), y);
    min_bounds.e[2] = min(min_bounds.z(), z);

    max_bounds.e[0] = max(max_bounds.x(), x);
    max_bounds.e[1] = max(max_bounds.y(), y);
    max_bounds.e[2] = max(max_bounds.z(), z);
  }
  AABB scene_bounding_box(min_bounds, max_bounds);
  printf("Scene bounds calculated...\n\tMin Bounds: (%f, %f, %f)\n", min_bounds.x(), min_bounds.y(), min_bounds.z());
  printf("\tMax Bounds: (%f, %f, %f)\n", max_bounds.x(), max_bounds.y(), max_bounds.z());  


  //The Obj reader does not store vertex indices in contiguous memory.
  //Copy the indices into a block of memory on the host device.
  //This is required beforehand regardless och BVH construction method.
  Triangle *ptr_host_triangles = (Triangle*)malloc(sizeof(Triangle) * poly_count);
  for (int i = 0; i < indices_count; i+=3){
    Triangle tempTri = Triangle();
    int v0_index = shapes[0].mesh.indices[i  ].vertex_index;
    int v1_index = shapes[0].mesh.indices[i+1].vertex_index;
    int v2_index = shapes[0].mesh.indices[i+2].vertex_index;
    tempTri.v0_index = v0_index;
    tempTri.v1_index = v1_index;
    tempTri.v2_index = v2_index;
    ptr_host_triangles[i/3] = tempTri;
  }

  //Allocate and memcpy index, vertex and normal buffers from host to device.
  Triangle *ptr_device_triangles = nullptr;
  cudaMalloc(&ptr_device_triangles, poly_count * sizeof(Triangle));
  cudaMemcpy(ptr_device_triangles, ptr_host_triangles, poly_count * sizeof(Triangle), cudaMemcpyHostToDevice);

  Vec3 *ptr_device_vertices = nullptr;
  cudaMalloc(&ptr_device_vertices, vertex_count * sizeof(Vec3));
  cudaMemcpy(ptr_device_vertices, attrib.vertices.data(), vertex_count * sizeof(Vec3), cudaMemcpyHostToDevice);

  Vec3 *ptr_device_normals = nullptr;
  cudaMalloc(&ptr_device_normals, normals_count * sizeof(Vec3));
  cudaMemcpy(ptr_device_normals, attrib.normals.data(), normals_count * sizeof(Vec3), cudaMemcpyHostToDevice);


  // ----------- CONSTRUCT Karras 2012 -----------
  LBVH lbvh(ptr_device_triangles, poly_count, ptr_device_vertices, vertex_count, scene_bounding_box);
  Node* ptr_device_tree = lbvh.construct();

  // ----------- RENDER -----------
  RenderConfig config(image_width, image_height, samples_per_pixel, max_bounces, 1337);
  Camera cam = Camera(config.img_width, config.img_height, 90.0f, 1.0f, Vec3(0,0,0));
  Raytracer raytracer = Raytracer(config, ptr_device_vertices, ptr_device_normals, ptr_device_triangles, indices_count);
  printf("Starting rendering...\n");
  Vec3* ptr_device_img = nullptr;
  switch(bvh_type){
    case BVH_Type::none:
      ptr_device_img = raytracer.render(cam);
      break;
    case BVH_Type::lbvh || BVH_Type::sahbvh:
      ptr_device_img = raytracer.renderBVH(ptr_device_tree, cam);
      break;
  }
  printf("Render complete.\n");

  //Copy framebuffer to host and save to disk.
  Image render_output = Image(config.img_width, config.img_height);
  render_output.copyFromDevice(ptr_device_img, config.img_height * config.img_width);
  render_output.save(output_filename);

  free(ptr_host_triangles);
  checkCudaErrors(cudaFree(ptr_device_triangles));
  checkCudaErrors(cudaFree(ptr_device_vertices));
  checkCudaErrors(cudaFree(ptr_device_normals));
  return 0;
}