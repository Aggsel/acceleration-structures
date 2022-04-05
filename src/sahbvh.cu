#pragma once

#include "vec3.h"
#include "triangle.h"
#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void constructSAHBVH(Triangle* ptr_device_triangles, Node* ptr_device_internal_nodes, Node* ptr_device_leaf_nodes, int triangle_count);

__global__ void constructSAHBVH(Triangle* ptr_device_triangles, Node* ptr_device_internal_nodes, Node* ptr_device_leaf_nodes, int triangle_count){
  
}

class SAHBVH{
	Node* ptr_device_internal_nodes;
  Node* ptr_device_leaf_nodes;

  Triangle* ptr_device_triangles;
  int triangle_count;
  Vec3* ptr_device_vertices;
  int vertex_count;

  AABB scene_bounds;

  public:
  SAHBVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count, AABB scene_bounds){
    checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_leaf_nodes, (triangle_count)*sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_internal_nodes, 0, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_leaf_nodes, 0, (triangle_count)*sizeof(Node)));

    this->vertex_count = vertex_count;
    this->scene_bounds = scene_bounds;
    this->ptr_device_vertices = ptr_device_vertices;
    this->ptr_device_triangles = ptr_device_triangles;
  }

  ~SAHBVH(){
    checkCudaErrors(cudaFree(ptr_device_internal_nodes));
    checkCudaErrors(cudaFree(ptr_device_leaf_nodes));
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    const int threads_per_block = 512;
    constructSAHBVH<<<(triangle_count-1)/threads_per_block+1, threads_per_block>>>(ptr_device_triangles, ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("SAH BVH Construction completed.\n");
    return ptr_device_internal_nodes;
  }
};