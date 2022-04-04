#define TINYOBJLOADER_IMPLEMENTATION 
#define PI 3.1415926535897932385
#define EPSILON 0.000001

#include <iostream>
#include <thrust/sort.h>

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

#include "third_party/cuda_helpers/helper_cuda.h"      //checkCudaErrors
#include "third_party/tiny_obj_loader.h"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __host__ inline unsigned int expandBits(unsigned int v){
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

//Expects an input Vec3(0..1, 0..1, 0..1)
__device__ __host__ unsigned int mortonCode(Vec3 v){
  //Clamp coordinates to 10 bits.
  float x = min(max(v.x() * 1024.0f, 0.0f), 1023.0f);
  float y = min(max(v.y() * 1024.0f, 0.0f), 1023.0f);
  float z = min(max(v.z() * 1024.0f, 0.0f), 1023.0f);
  //Bit shift componentwise before merging bits into morton code.
  unsigned int xx = expandBits((unsigned int)x) << 2;
  unsigned int yy = expandBits((unsigned int)y) << 1;
  unsigned int zz = expandBits((unsigned int)z);
  return xx | yy | zz;
}

int main(int argc, char *argv[]){
  std::string filename = "sample_models/large_210.obj";
  int samples_per_pixel = 30;
  int image_height = 512;
  int image_width = 512;
  int max_bounces = 5;
  char* output_filename = "output.ppm";

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


  //Calculate scene bounding box. This is not strictly required if we make sure to resolve duplicate codes.
  //Although it likely will lead to better quality structure.
  Vec3 min_bounds = Vec3( 10000000.0, 10000000.0,   10000000.0);
  Vec3 max_bounds = Vec3(-10000000.0,-10000000.0,  -10000000.0);

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
  printf("Scene bounds calculated...\n\tMin Bounds: (%f, %f, %f)\n", min_bounds.x(), min_bounds.y(), min_bounds.z());
  printf("\tMax Bounds: (%f, %f, %f)\n", max_bounds.x(), max_bounds.y(), max_bounds.z());  

  //The Obj reader does not store vertex indices in contiguous memory.
  //Copy the indices into a block of memory on the host device.
  Triangle *ptr_host_triangles = (Triangle*)malloc(sizeof(Triangle) * poly_count);
  for (int i = 0; i < indices_count; i+=3){
    Triangle tempTri = Triangle();
    int v0_index = shapes[0].mesh.indices[i  ].vertex_index;
    int v1_index = shapes[0].mesh.indices[i+1].vertex_index;
    int v2_index = shapes[0].mesh.indices[i+2].vertex_index;
    tempTri.v0_index = v0_index;
    tempTri.v1_index = v1_index;
    tempTri.v2_index = v2_index;

    //TODO @Perf: The morton code generation could easily be done on the GPU instead.
    tinyobj::index_t idx = shapes[0].mesh.indices[i];
    Vec3 v0 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    idx = shapes[0].mesh.indices[i+1];
    Vec3 v1 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    idx = shapes[0].mesh.indices[i+2];
    Vec3 v2 = Vec3( attrib.vertices[3*size_t(idx.vertex_index)+0], 
                    attrib.vertices[3*size_t(idx.vertex_index)+1], 
                    attrib.vertices[3*size_t(idx.vertex_index)+2] );

    Vec3 centroid = (v0 + v1 + v2) / 3;

    // printf("Centroid: (%f, %f, %f)\n", centroid.x(), centroid.y(), centroid.z());           // @debug
    centroid.e[0] = (centroid.x() - min_bounds.x()) / (max_bounds.x() - min_bounds.x());
    centroid.e[1] = (centroid.y() - min_bounds.y()) / (max_bounds.y() - min_bounds.y());
    centroid.e[2] = (centroid.z() - min_bounds.z()) / (max_bounds.z() - min_bounds.z());
    // printf("Centroid: (%f, %f, %f)\n\n", centroid.x(), centroid.y(), centroid.z());         // @debug
    tempTri.morton_code = mortonCode(centroid);
    tempTri.aabb.min_bounds = min(v2, min(v0, v1)); //TODO: This should not happen here? This should be in lbvh::calculateAABB()
    tempTri.aabb.max_bounds = max(v2, max(v0, v1));
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


  // ----------- SORT -----------
  // Sorts the triangle buffer based on the computed morton codes. (using < overloading from the triangle struct).
  thrust::sort(thrust::device, ptr_device_triangles, ptr_device_triangles+poly_count);


  // ----------- CONSTRUCT Karras 2012 -----------
  BVH bvh(ptr_device_triangles, poly_count, ptr_device_vertices, vertex_count);
  Node* ptr_device_tree = bvh.construct();

  // ----------- RENDER -----------
  // RenderConfig config(image_width, image_height, samples_per_pixel, max_bounces, 1337);
  // Camera cam = Camera(config.img_width, config.img_height, 90.0f, 1.0f, Vec3(0,0,0));
  // Raytracer raytracer = Raytracer(config, ptr_device_vertices, ptr_device_normals, ptr_device_triangles, indices_count);
  // printf("Starting rendering...\n");
  // Vec3* ptr_device_img = raytracer.render(cam);
  // printf("Render complete.\n");

  // //Copy framebuffer to host and save to disk.
  // Image render_output = Image(config.img_width, config.img_height);
  // render_output.copyFromDevice(ptr_device_img, config.img_height * config.img_width);
  // render_output.save(output_filename);
  // printf("%s saved to disk.\n", output_filename);

  free(ptr_host_triangles);
  checkCudaErrors(cudaFree(ptr_device_triangles));
  checkCudaErrors(cudaFree(ptr_device_vertices));
  checkCudaErrors(cudaFree(ptr_device_normals));
  return 0;
}