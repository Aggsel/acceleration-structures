#pragma once

#include <limits>
#include "vec3.h"
#include "triangle.h"
#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"
#include "debug.cu"

struct SubTreeInfo{
  int start;
  int end;

  AABB triangle_bounds;
  AABB centroid_bounds;
};

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, int* ptr_device_triangle_ids);
__global__ void constructSAHBVH(Node* nodes, Triangle* ptr_device_triangles, Vec3 scene_bounds, int start, int end);
__global__ void split(Node* nodes, Triangle* ptr_device_triangles, int* triangle_ids, int* temp_triangle_ids, int start, int end, SubTreeInfo* ptr_device_subtrees_write, SubTreeInfo* ptr_device_subtrees_read, int* subtree_count);

__device__ inline int projectToBin(float k_1, float centroid_bin_axis, float scene_min_axis);
__device__ inline float cost(int N_L, int N_R, float A_L, float A_R);

//Inputs along the selected axis. E.g. tri_centroid for x,y or z depending on the selected axis.
__device__ inline int projectToBin(float k_1, float tri_centroid, float node_min_bounds){
  return (int) (k_1 * (tri_centroid - node_min_bounds));
}

__device__ inline float cost(int N_L, int N_R, float A_L, float A_R){
  //BUG: Might become an issue, *0.0001 to decrease cost (avoiding overflowing for high costs)
  return (A_L * N_L + A_R * N_R) * 0.001;
}

__global__ void split(Node* nodes, 
                      Triangle* ptr_device_triangles, 
                      int* triangle_ids, 
                      int* temp_triangle_ids, 
                      int start, 
                      int end, 
                      SubTreeInfo* ptr_device_subtrees_write, 
                      SubTreeInfo* ptr_device_subtrees_read,
                      int* ptr_device_subtree_counter, 
                      int subtree_count){

  int node_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(node_index >= subtree_count)
    return;

  start = ptr_device_subtrees_read[node_index].start;
  end = ptr_device_subtrees_read[node_index].end;

  // printf("Range min: %i max: %i \n", start, end);
  
  const int number_of_bins = 16;
  const int primitive_count = end - start;

  // //Calculate node AABB
  AABB node_bounds;
  for(int i = start; i < end; i++)
    node_bounds.join(ptr_device_triangles[i].centroid); //TODO: Centroid or aabb?
  nodes[node_index].aabb = node_bounds;

  //Decide which axis to sweep. (the longest side).
  int axis;
  Vec3 size = node_bounds.max_bounds - node_bounds.min_bounds;
  if(size.x() > size.y() && size.x() > size.z())
    axis = 0;
  else if(size.y() > size.x() && size.y() > size.z())
    axis = 1;
  else
    axis = 2;

  //Compute k_1 for the selected axis.
  float k_1 = number_of_bins * (1.0 - FLT_EPSILON) / 
              (node_bounds.max_bounds.e[axis] - node_bounds.min_bounds.e[axis]);

  int bin_triangle_counts[number_of_bins];
  AABB bin_aabbs[number_of_bins];
  for (int i = 0; i < number_of_bins; i++)
    bin_aabbs[i] = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));

  //Calculate N_l, N_r, A_l & A_r for all triangles for all bins.
  //TODO: This could be better parallelized. Atomics? slow though.
  for(int i = start; i < end; i++){
    int bin_index = projectToBin( k_1, 
                                  ptr_device_triangles[i].centroid.e[axis],
                                  node_bounds.min_bounds.e[axis]);

    bin_triangle_counts[bin_index]++;
    bin_aabbs[bin_index].join(ptr_device_triangles[i].centroid);  //TODO: Should this be triangle centroid or aabb?
  }

  //Sweep from left -->>
  int tri_count_l_sweep[number_of_bins];
  AABB aabb_l_sweep[number_of_bins];

  tri_count_l_sweep[0] = bin_triangle_counts[0];
  aabb_l_sweep[0] = bin_aabbs[0];
  for (int i = 1; i < number_of_bins; i++){
    tri_count_l_sweep[i] = tri_count_l_sweep[i-1] + bin_triangle_counts[i];
    aabb_l_sweep[i] = AABB::join(aabb_l_sweep[i-1], bin_aabbs[i]);
  }

  //Sweep from right <<-- and calculate cost.
  AABB aabb_r_sweep[number_of_bins];
  aabb_r_sweep[number_of_bins-1] = bin_aabbs[number_of_bins-1];
  float min_cost = FLT_MAX;
  int split_index = 0;

  for (int i = number_of_bins-2; i >= 0; i--){
    int primitives_left = tri_count_l_sweep[i];
    aabb_r_sweep[i] = AABB::join(aabb_r_sweep[i+1], bin_aabbs[i]);
    float sah_cost = cost(  primitives_left,
                            primitive_count - primitives_left,
                            aabb_l_sweep[i].surfaceArea(),
                            aabb_r_sweep[i].surfaceArea());

    if(sah_cost < min_cost){
      min_cost = sah_cost;
      split_index = i;
    }
  }

  float bin_width = size.e[axis] / number_of_bins;
  float split_position = node_bounds.min_bounds.e[axis] + (bin_width*split_index);

  //Copy triangle ids to temporary array.
  for (int i = start; i < end; i++)
    temp_triangle_ids[i] = triangle_ids[i];

  //Push back triangle ids to the normal array depending on their side of the split position.
  //If triangle centroid is to the "left" of the split position, push to the front of triangle ids
  //Otherwise push to the back.

  //BUG: This will break if we can't guarantee that only ONE thread
  //     is accessing the same indices at a given time. The subtree
  //     ids must be used.  
  int left_i = start;
  int right_i = end-1;
  for (int i = start; i < end; i++){
    if(ptr_device_triangles[temp_triangle_ids[i]].centroid.e[axis] <= split_position){
      triangle_ids[left_i] = temp_triangle_ids[i];
      left_i++;
    }
    else{
      triangle_ids[right_i] = temp_triangle_ids[i];
      right_i--;
    }
  }

  // //This is what we want. 
  // //These values can be used to index into the triangle id array during traversal.
  int node_range_min = start;
  int split = left_i;
  int node_range_end = end;

  // //TODO: How do we handle the case where the split_position == start or end? <- Can this even happen?
  // //      Just check it for now.
  assert(start > 0);
  assert(split != node_range_min);
  assert(split != node_range_end);

  // //Record node relationships.
  // Node left_child = Node();
  // left_child.parent = &nodes[node_index];
  // left_child.start_range = start;
  // left_child.range = split;
  // left_child.aabb = aabb_l_sweep[split_index];
  // nodes[split] = left_child;

  // Node right_child = Node();
  // right_child.parent = &nodes[node_index];
  // right_child.start_range = split;
  // right_child.range = end;
  // right_child.aabb = aabb_r_sweep[split_index];
  // nodes[split+1] = right_child;

  printf("start %i split %i end %i\n", start, split, end);

  SubTreeInfo left_subtree;
  left_subtree.start = start;
  left_subtree.end = split;
  left_subtree.centroid_bounds = aabb_l_sweep[split_index];

  SubTreeInfo right_subtree;
  right_subtree.start = split;
  right_subtree.end = end;
  right_subtree.centroid_bounds = aabb_r_sweep[split_index];

  ptr_device_subtrees_write[node_index*2 + 0] = left_subtree;
  ptr_device_subtrees_write[node_index*2 + 1] = right_subtree;

  atomicAdd(ptr_device_subtree_counter, 2);
}

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, int* ptr_device_triangle_ids){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= triangle_count)
    return;

  ptr_device_triangle_ids[node_index] = node_index; //Initialize triangle ids to [0, 1, 2 .. n]

  Triangle tri = triangles[node_index];
  Vec3 v0 = ptr_device_vertex_buffer[tri.v0_index];
  Vec3 v1 = ptr_device_vertex_buffer[tri.v1_index];
  Vec3 v2 = ptr_device_vertex_buffer[tri.v2_index];
  Vec3 centroid = (v0 + v1 + v2) / 3.0;
  triangles[node_index].centroid = centroid;

  Vec3 vertex_bounds_min = min(min(v0, v1), v2);
  Vec3 vertex_bounds_max = max(max(v0, v1), v2);
  triangles[node_index].aabb = AABB(vertex_bounds_min, vertex_bounds_max);
}

class SAHBVH{
  Node* ptr_device_tree;
  int* ptr_device_triangle_ids;
  int* ptr_device_temp_triangle_ids;

  int host_subtree_counter = 0;
  int *ptr_device_subtree_counter;
  SubTreeInfo* ptr_device_subtrees_0;
  SubTreeInfo* ptr_device_subtrees_1;
  SubTreeInfo* ptr_host_subtrees;

  Triangle* ptr_device_triangles;
  Triangle* ptr_host_triangles;
  int triangle_count;
  Vec3* ptr_device_vertices;
  int vertex_count;

  AABB scene_bounds_triangles;
  AABB scene_bounds_centroids;

  public:
  SAHBVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count, AABB scene_bounds_triangles){
    checkCudaErrors(cudaMalloc(&ptr_device_tree, 2*triangle_count-1 * sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_triangle_ids, triangle_count * sizeof(int)));
    checkCudaErrors(cudaMalloc(&ptr_device_temp_triangle_ids, triangle_count * sizeof(int)));

    checkCudaErrors(cudaMalloc(&ptr_device_subtrees_0, (triangle_count-1) * sizeof(SubTreeInfo)));
    checkCudaErrors(cudaMalloc(&ptr_device_subtrees_1, (triangle_count-1) * sizeof(SubTreeInfo)));

    checkCudaErrors(cudaMalloc(&ptr_device_subtree_counter, sizeof(int)));

    checkCudaErrors(cudaMemset(ptr_device_tree, 0, 2*triangle_count-1 * sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_triangle_ids, 0, triangle_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ptr_device_temp_triangle_ids, 0, triangle_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ptr_device_subtrees_0, 0, (triangle_count-1) * sizeof(SubTreeInfo)));
    checkCudaErrors(cudaMemset(ptr_device_subtrees_1, 0, (triangle_count-1) * sizeof(SubTreeInfo)));

    ptr_host_subtrees = (SubTreeInfo*)malloc( (triangle_count-1) * sizeof(SubTreeInfo));

    this->vertex_count = vertex_count;
    this->triangle_count = triangle_count;
    this->scene_bounds_triangles = scene_bounds_triangles;
    this->ptr_device_vertices = ptr_device_vertices;
    this->ptr_device_triangles = ptr_device_triangles;
  }

  ~SAHBVH(){
    checkCudaErrors(cudaFree(ptr_device_tree));
    checkCudaErrors(cudaFree(ptr_device_triangle_ids));
    checkCudaErrors(cudaFree(ptr_device_temp_triangle_ids));
    checkCudaErrors(cudaFree(ptr_device_subtrees_0));
    checkCudaErrors(cudaFree(ptr_device_subtrees_1));
    checkCudaErrors(cudaFree(ptr_device_subtree_counter));
    free(ptr_host_subtrees);
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    const int threads_per_block = 64;

    computeBoundsAndCentroids<<<triangle_count/threads_per_block+1, threads_per_block>>>(ptr_device_triangles, triangle_count, ptr_device_vertices, ptr_device_triangle_ids);
    checkCudaErrors(cudaDeviceSynchronize());

    //Calculate scene centroid bounds.
    //TODO: Could this be done more efficiently and/or during another step?
    //      Copying the entire buffer back to the host seems excessive.
    ptr_host_triangles = (Triangle*)malloc(triangle_count * sizeof(Triangle)); 
    checkCudaErrors(cudaMemcpy(ptr_host_triangles, ptr_device_triangles, triangle_count * sizeof(Triangle), cudaMemcpyDeviceToHost));
    scene_bounds_centroids = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    for (int i = 0; i < triangle_count; i++)
      scene_bounds_centroids.join(ptr_host_triangles[i].centroid);  //O(n)  :^(
    free(ptr_host_triangles);
    //Initialization complete.


    //TODO: parallelization. Horizontal for upper levels, vertical for lower. (?)
    //The goal of split() is to write to SubTreeInfo. For every entry in subtreeinfo, dispatch a kernel that evaluates
    //that horizontal plane.

    //Split, will write to subtree info and increment subtree counter. (will the subtree counter always be previous * 2)?
    //For every subtree, launch a kernel.
    //Clear subtree list and reset counter.

    host_subtree_counter = 1;
    checkCudaErrors(cudaMemcpy(ptr_device_subtree_counter, &host_subtree_counter, sizeof(int), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++){
      checkCudaErrors(cudaMemcpy(&host_subtree_counter, ptr_device_subtree_counter, sizeof(int), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemset(ptr_device_subtree_counter, 0, sizeof(int)));  //Reset device counter.
      // checkCudaErrors(cudaMemset(ptr_device_subtrees, 0, (triangle_count-1) * sizeof(SubTreeInfo))); //Reset subtrees.
      checkCudaErrors(cudaDeviceSynchronize());

      printf("Subtree_count: %i, 0x%p\t0x%p\n", host_subtree_counter, ptr_device_subtree_counter, &host_subtree_counter);

      int threads = host_subtree_counter/threads_per_block+1;
      if(i%1 == 0){
        split<<<threads, threads_per_block>>>(ptr_device_tree,
                                              ptr_device_triangles,
                                              ptr_device_triangle_ids,
                                              ptr_device_temp_triangle_ids,
                                              0,
                                              triangle_count,
                                              ptr_device_subtrees_0,
                                              ptr_device_subtrees_1,
                                              ptr_device_subtree_counter,
                                              host_subtree_counter);
      }
      else{
        split<<<threads, threads_per_block>>>(ptr_device_tree,
                                              ptr_device_triangles,
                                              ptr_device_triangle_ids,
                                              ptr_device_temp_triangle_ids,
                                              0,
                                              triangle_count,
                                              ptr_device_subtrees_1,
                                              ptr_device_subtrees_0,
                                              ptr_device_subtree_counter,
                                              host_subtree_counter);
      }


      checkCudaErrors(cudaDeviceSynchronize());
    }
    
    checkCudaErrors(cudaMemcpy(ptr_host_subtrees, ptr_device_subtrees_0, (triangle_count-1) * sizeof(SubTreeInfo), cudaMemcpyDeviceToHost));
    //check children of nodes[node_index] and split both.
    // DebugHelper::PrintNodes(ptr_device_tree, triangle_count, ptr_device_triangles);

    checkCudaErrors(cudaDeviceSynchronize());
    printf("Binned SAH BVH construction completed.\n");
    return ptr_device_tree;
  }
};