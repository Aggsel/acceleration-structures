#pragma once

#include <limits>
#include "vec3.h"
#include "triangle.h"
#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"
#include "debug.cu"

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, int* ptr_device_triangle_ids);
__global__ void constructSAHBVH(Node* nodes, Triangle* ptr_device_triangles, Vec3 scene_bounds, int start, int end);
__global__ void split(Node* nodes, int node_index, Triangle* ptr_device_triangles, int* triangle_ids, int* temp_triangle_ids, int start, int end, int* ptr_device_splits_and_ranges);

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

__global__ void split(Node* nodes, int node_index, Triangle* ptr_device_triangles, int* triangle_ids, int* temp_triangle_ids, int start, int end, int* ptr_device_splits_and_ranges){
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

    // printf("Evaluating split %i: %i, %i, %f, %f, %f\tCost: %f\n",
    //   i,
    //   primitives_left, 
    //   primitive_count - primitives_left, 
    //   aabb_l_sweep[i].surfaceArea(), 
    //   aabb_r_sweep[i].surfaceArea(), 
    //   k_1, sah_cost);

    if(sah_cost < min_cost){
      min_cost = sah_cost;
      split_index = i;
    }
  }

  float bin_width = size.e[axis] / number_of_bins;
  float split_position = node_bounds.min_bounds.e[axis] + (bin_width*split_index);

  //@debug Print information regarding the split decision.
  // char* axis_char = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
  // printf("\nSplit found. Axis %s, Split index: %i, SAH Cost: %f\n", axis_char, split_index, min_cost);
  // printf("Bin Width: %f, Split Position: %f\n", bin_width, split_position);
  // printf("Bin Centroid Bounds:\n\tMin Bounds: (%f, %f, %f)\n\tMax Bounds: (%f, %f, %f)\n\n", 
  //   node_bounds.min_bounds.x(),
  //   node_bounds.min_bounds.y(),
  //   node_bounds.min_bounds.z(),
  //   node_bounds.max_bounds.x(),
  //   node_bounds.max_bounds.y(),
  //   node_bounds.max_bounds.z());

  //Copy triangle ids to temporary array.
  for (int i = start; i < end; i++)
    temp_triangle_ids[i] = triangle_ids[i];


  //Push back triangle ids to the normal array depending on their side of the split position.
  //If triangle centroid is to the "left" of the split position, push to the front of triangle ids
  //Otherwise push to the back.
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

  //This is what we want. 
  //These values can be used to index into the triangle id array during traversal.
  int node_range_min = start;
  int split = left_i;
  int node_range_end = end;

  //TODO: Assign a node the ranges, unless it is an internal node (?),
  //      in that case, create two children and perform split on the children.
  //      How can we determine a node index? node_range_min?
  //      How can we determine children node indices? same as karras? node_range_split?
  //      Do we add more fields to the Node type, start & range? Then we could just say split node x and the ranges 
  //      will already be stored.

  //TODO: How do we handle the case where the split_position == start or end? <- Can this even happen?
  //      Just check it for now.
  assert(split != node_range_min);
  assert(split != node_range_end);

  //Record node relationships.
  Node left_child = Node();
  left_child.parent = &nodes[node_index];
  left_child.start_range = start;
  left_child.range = split;
  left_child.aabb = aabb_l_sweep[split_index];  //BUG: Is this correct?
  nodes[split] = left_child;

  Node right_child = Node();
  right_child.parent = &nodes[node_index];
  right_child.start_range = split;
  right_child.range = end;
  right_child.aabb = aabb_r_sweep[split_index]; //BUG: Is this correct?
  nodes[split+1] = right_child;

  ptr_device_splits_and_ranges[node_index*3+0] = node_range_min;
  ptr_device_splits_and_ranges[node_index*3+1] = split;
  ptr_device_splits_and_ranges[node_index*3+2] = node_range_end;

  //@debug - print the entire range of triangles and 
  //         their corresponding side of the splitting plane.
  // for (int i = start; i < end; i++){
  //   if(i < node_range_mid)
  //     printf("Left - Tri: %i, (%f), L: %i, R: %i\n", 
  //       triangle_ids[i], 
  //       ptr_device_triangles[triangle_ids[i]].centroid.e[axis], 
  //       left_i, 
  //       right_i);
  //   else
  //     printf("Right - Tri: %i, (%f), L: %i, R: %i\n", 
  //       triangle_ids[i], 
  //       ptr_device_triangles[triangle_ids[i]].centroid.e[axis], 
  //       left_i, 
  //       right_i);
  // }
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

  int* ptr_device_splits_and_ranges;
  int* ptr_host_splits_and_ranges;

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
    checkCudaErrors(cudaMalloc(&ptr_device_splits_and_ranges, triangle_count-1 * sizeof(int) * 3));

    checkCudaErrors(cudaMemset(ptr_device_tree, 0, 2*triangle_count-1 * sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_triangle_ids, 0, triangle_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ptr_device_temp_triangle_ids, 0, triangle_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ptr_device_splits_and_ranges, 0, triangle_count-1 * sizeof(int) * 3));

    ptr_host_splits_and_ranges = (int*)malloc(triangle_count-1 * sizeof(int) * 3);

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
    checkCudaErrors(cudaFree(ptr_device_splits_and_ranges));
    free(ptr_host_splits_and_ranges);
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    // 1. -- ok --  We need the triangles bounding box (tb_i) as well as the centroids (c_i).
    // 2. -- ok --  The entire scenes triangle bounds (vb) as well as the scenes centroid bounding box (cb)
    //                All coordinates are stored as float4
    // 3. -- ok --  Calculate bin id from each triangles centroid.
    // 4. -- ok --  Assign each triangle centroid to a bin and keep track of the number of assigned triangles. Join the bins AABBs.
    // 5. -- ok --  Evaluate planes by sweeping the bins and accumulate bounds and triangle count left to right. Save values.
    //                Do the same right to left and save the values. While sweeing right to left, evaluate SAH. N_l, N_r, A_l & A_r for all bin pairs.
    // 6. -- ok --  Allocate two arrays, 2N-1 for the nodes and N for the triangle ids.
    // 7. -- ok --  Sweep the triangle id list from left to right and right to left and evaluate the bin ids for each centroid.
    // 8. Horizontal parallelization for the upper levels of the tree, vertical parallelization later.
    const int threads_per_block = 512;

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

    //TODO: parallelization. Horizontal for upper levels, vertical for lower. (?)

    int node_index = 0;
    split<<<1,1>>>(ptr_device_tree, node_index, ptr_device_triangles, ptr_device_triangle_ids, ptr_device_temp_triangle_ids, 0, triangle_count, ptr_device_splits_and_ranges);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(ptr_host_splits_and_ranges, ptr_device_splits_and_ranges, triangle_count -1 * sizeof(int) * 3, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++){
    //   int start = ptr_host_splits_and_ranges[node_index*3 + 0];
    //   int split = ptr_host_splits_and_ranges[node_index*3 + 1];
    //   int end   = ptr_host_splits_and_ranges[node_index*3 + 2];
    //   split<<<1,1>>>(ptr_device_tree, node_index,   ptr_device_triangles, ptr_device_triangle_ids, ptr_device_temp_triangle_ids, start, split, ptr_device_splits_and_ranges);
    //   split<<<1,1>>>(ptr_device_tree, node_index+1, ptr_device_triangles, ptr_device_triangle_ids, ptr_device_temp_triangle_ids, split, end,   ptr_device_splits_and_ranges);
    //   checkCudaErrors(cudaDeviceSynchronize());
    //   checkCudaErrors(cudaMemcpy(ptr_host_splits_and_ranges, ptr_device_splits_and_ranges, triangle_count -1 * sizeof(int) * 3, cudaMemcpyDeviceToHost));
    //   node_index = split;
    // }

    //check children of nodes[node_index] and split both.
    DebugHelper::PrintNodes(ptr_device_tree, triangle_count, ptr_device_triangles);

    printf("Binned SAH BVH construction completed.\n");
    return ptr_device_tree;
  }
};