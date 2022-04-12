#pragma once

#include <chrono>
#include <thrust/sort.h>
#include "triangle.h"
#include "node.h"
#include "debug.cu"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void constructLBVH(Triangle *triangles, Node* internal_nodes, Node* leaf_nodes, int primitive_count);
__global__ void calculateAABB(Node* internal_nodes, Triangle* leaf_nodes, int leaf_count, Vec3* vert_buff);
__global__ void generateMortonCodes(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, Vec3 scene_bounds_min, Vec3 inverse_min_max);
__device__ int2 determineRange(Triangle *sorted_morton_codes, int total_primitives, int node_index);
__device__ int findSplit(Triangle *sorted_morton_codes, int first, int last);
__device__ unsigned int mortonCode(Vec3 v);
__device__ inline unsigned int expandBits(unsigned int v);

// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__device__ inline unsigned int expandBits(unsigned int v){
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
//Expects an input Vec3(0..1, 0..1, 0..1)
__device__ unsigned int mortonCode(Vec3 v){
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

__global__ void generateMortonCodes(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, Vec3 scene_bounds_min, Vec3 inverse_min_max){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= triangle_count-1)
    return;

  Triangle tri = triangles[node_index];
  Vec3 v0 = ptr_device_vertex_buffer[tri.v0_index];
  Vec3 v1 = ptr_device_vertex_buffer[tri.v1_index];
  Vec3 v2 = ptr_device_vertex_buffer[tri.v2_index];
  Vec3 centroid = (v0 + v1 + v2) / 3.0;

  centroid.e[0] = (centroid.x() - scene_bounds_min.x()) * inverse_min_max.x();
  centroid.e[1] = (centroid.y() - scene_bounds_min.y()) * inverse_min_max.y();
  centroid.e[2] = (centroid.z() - scene_bounds_min.z()) * inverse_min_max.z();
  triangles[node_index].morton_code = mortonCode(centroid);

  Vec3 vertex_bounds_min = min(min(v0, v1), v2);
  Vec3 vertex_bounds_max = max(max(v0, v1), v2);
  triangles[node_index].aabb = AABB(vertex_bounds_min, vertex_bounds_max);
}

__device__ int commonPrefix(Triangle *morton_codes, int index1, int index2){
  if(index2 < 0 || index1 < 0)
    return 0;
  unsigned int key1 = morton_codes[index1].morton_code;
  unsigned int key2 = morton_codes[index2].morton_code;
  if(key1 != key2)
    // printf("Same: Index: %i Index: %i XOR: %i Prefix l: %i\n", index1, index2, index1 ^ index2, __clz(index1 ^ index2));
    return __clz(key1 ^ key2);
  // printf("Dupe: Index: %i (%u) Index: %i (%u) XOR: %u Prefix l: %i\n", index1, key1, index2, key2, index1 ^ index2, __clz(index1 ^ index2));
  return __clz(index1 ^ index2) + 32;
}

__device__ void determineRange(Triangle *sorted_morton_codes, int total_primitives, int node_index, int* range_min, int* range_max){
  if(node_index == 0){
    *range_min = 0;
    *range_max = total_primitives-1;
  }

  //Determine direction (d).
  //Delta being the number of largest common bits between two keys.
  int next_delta = commonPrefix(sorted_morton_codes, node_index, node_index+1);
  int prev_delta = commonPrefix(sorted_morton_codes, node_index, node_index-1);
  int d = next_delta - prev_delta < 0 ? -1 : 1;

  //Upper bound
  int lmax = 128;
  int delta_min = min(next_delta, prev_delta);
  int delta = -1;
  int i = node_index + d * lmax;
  if(i >= 0 && i < total_primitives){
    delta = commonPrefix(sorted_morton_codes, node_index, i);
  }

  while(delta > delta_min){
    lmax = lmax << 2;
    i = node_index + d * lmax;
    delta = -1;
    if(0 <= i && i < total_primitives)
      delta = commonPrefix(sorted_morton_codes, node_index, i);
  }

  //Binary search for the other end.
  int l = 0;
  int t = lmax >> 1;
  while(t > 0){
    i = node_index + (l + t) * d;
    delta = -1;

    if(0 <= i && i < total_primitives)
      delta = commonPrefix(sorted_morton_codes, node_index, i);

    if(delta > delta_min)
      l += t;

    t = t >> 1;
  }
  unsigned int j = node_index + l * d;
  *range_min = min(node_index, j);
  *range_max = max(node_index, j);
}

//From https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__device__ int findSplit(Triangle *sorted_morton_codes, int first, int last){

  //count leading zeros
  int common_prefix = commonPrefix(sorted_morton_codes, first, last);

  int split = first;
  int step = last - first;

  do{
    step = (step + 1) >> 1;
    int new_split = split + step;

    if (new_split < last){
      int split_prefix = commonPrefix(sorted_morton_codes, first, new_split);
      if (split_prefix > common_prefix)
        split = new_split;
    }
  }while (step > 1);
  
  return split;
}

__global__ void constructLBVH(Triangle *triangles, Node* internal_nodes, Node* leaf_nodes, int primitive_count){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= primitive_count-1)
    return;

  //Determine the range of the current node by performing a binary search on the
  //largest common prefix for neighboring morton codes.
  int first, last;
  determineRange(triangles, primitive_count, node_index, &first, &last);

  // Determine where to split the range.
  int split = findSplit(triangles, first, last);

  // Select left_child.
  Node* left_child;
  if(split == first){
    left_child = &leaf_nodes[split];
    left_child->primitive = &triangles[split];
    left_child->aabb = triangles[split].aabb;
    left_child->is_leaf = true;
  }
  else{
    left_child = &internal_nodes[split];
    left_child->is_leaf = false;
  }
  
  // Select right_child.
  Node* right_child;
  if(split + 1 == last){
    right_child = &leaf_nodes[split + 1];
    right_child->primitive = &triangles[split + 1];
    right_child->aabb = triangles[split + 1].aabb;
    right_child->is_leaf = true;
  }
  else{
    right_child = &internal_nodes[split + 1];
    right_child->is_leaf = false;
  }

  Node *self_ptr = &internal_nodes[node_index];

  assert(self_ptr != nullptr);

  right_child->parent = self_ptr;
  left_child->parent = self_ptr;
  self_ptr->left_child = left_child;
  self_ptr->right_child = right_child;
}

//In parallel, traverse the tree from each leaf upwards.
//The first execution thread to reach a node returns. 
//The second thread joins the nodes AABB and continues traversal.
__global__ void calculateAABB(Node* internal_nodes, Node* leaf_nodes, int leaf_count, Vec3* vert_buff, int* counter){
  int leaf_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(leaf_index >= leaf_count)
      return;

  //Calculate leaf AABB.
  Triangle *leaf_primitive = leaf_nodes[leaf_index].primitive;
  int v0 = leaf_primitive->v0_index;
  int v1 = leaf_primitive->v1_index;
  int v2 = leaf_primitive->v2_index;

  Vec3 min_bounds = min(min(vert_buff[v0], vert_buff[v1]), vert_buff[v2]);
  Vec3 max_bounds = max(max(vert_buff[v0], vert_buff[v1]), vert_buff[v2]);
  AABB leaf_aabb;
  leaf_aabb.min_bounds = min_bounds;
  leaf_aabb.max_bounds = max_bounds;

  Node* parent_node = leaf_nodes[leaf_index].parent;
  assert(parent_node != nullptr);

  parent_node->aabb = leaf_aabb;

  while(true){
    if(parent_node == nullptr)  //Root reached.
      return;

    int parent_index = parent_node - internal_nodes;
    int old = atomicCAS(&counter[parent_index], 0, 1);
    if(old == 0){ //This thread reached the node first. 
      return;
    }

    parent_node->aabb = AABB::join(parent_node->left_child ->aabb,
                                   parent_node->right_child->aabb);

    parent_node = parent_node->parent;
  }
}

class LBVH{
  Node* ptr_device_internal_nodes;
  Node* ptr_device_leaf_nodes;

  Triangle* ptr_device_triangles;
  int triangle_count;
  Vec3* ptr_device_vertices;
  int vertex_count;

  int* ptr_device_visited_node_counters;

  AABB scene_bounds;

  void populateMortonCodes(){
    int threads_per_block = 512;
    Vec3 inverse_min_max = 1.0/(scene_bounds.max_bounds - scene_bounds.min_bounds);
    generateMortonCodes<<<triangle_count/threads_per_block+1, threads_per_block>>>(ptr_device_triangles, triangle_count, ptr_device_vertices, scene_bounds.min_bounds, inverse_min_max);
  }

  public:
  LBVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count, AABB scene_bounds){
    this->ptr_device_triangles = ptr_device_triangles;
    this->triangle_count = triangle_count;
    this->ptr_device_vertices = ptr_device_vertices;

    this->vertex_count = vertex_count;
    this->scene_bounds = scene_bounds;

    // Allocate and initialize memory for:
    // BVH internal nodes, leaf nodes and our AABB bottom up traversal counter buffer.
    checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_leaf_nodes, (triangle_count)*sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_visited_node_counters, (triangle_count-1)));
    //There's no calloc equivalent for cuda. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    checkCudaErrors(cudaMemset(ptr_device_internal_nodes, 0, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_leaf_nodes, 0, (triangle_count)*sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_visited_node_counters, 0, (triangle_count-1)));
  }

  ~LBVH(){
    checkCudaErrors(cudaFree(ptr_device_internal_nodes));
    checkCudaErrors(cudaFree(ptr_device_leaf_nodes));
    checkCudaErrors(cudaFree(ptr_device_visited_node_counters));
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    using namespace std::chrono;

    steady_clock::time_point timer_start = high_resolution_clock::now();
    const int threads_per_block = 512;
    //1. Generate morton codes for each scene primitive.
    populateMortonCodes();

    //2. Sort scene primitives along morton curve.
    thrust::sort(thrust::device, ptr_device_triangles, ptr_device_triangles+triangle_count);
    //3. For each internal node in the tree, calculate it's range and split/children.
    constructLBVH<<<(triangle_count-1)/threads_per_block+1, threads_per_block>>>(ptr_device_triangles, ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count);

    //4. From each leaf node, traverse the tree towards the root of the tree.
    //   If the branch is the first to reach a given node, stop.
    //   If the branch is second, join the childrens AABB and continue traversal.
    calculateAABB<<<triangle_count/threads_per_block+1, threads_per_block>>>(ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count, ptr_device_vertices, ptr_device_visited_node_counters);
    checkCudaErrors(cudaDeviceSynchronize());
    
    steady_clock::time_point timer_end = high_resolution_clock::now();
    long long duration_ms = duration_cast<milliseconds>(timer_end - timer_start).count();
    long long duration_us = duration_cast<microseconds>(timer_end - timer_start).count();
    printf("LBVH Construction completed in %llims (%llius)\n", duration_ms, duration_us);

    return ptr_device_internal_nodes;
  }
};