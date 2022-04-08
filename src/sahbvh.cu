#pragma once

#include <limits>
#include "vec3.h"
#include "triangle.h"
#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"
#include "debug.cu"

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, int* ptr_device_triangle_ids);

//TODO: remove device capabilites.
__device__ __host__ inline int projectToBin(float k_1, float centroid_bin_axis, float scene_min_axis);
__device__ __host__ inline float cost(int N_L, int N_R, float A_L, float A_R);

//Inputs along the selected axis. E.g. tri_centroid for x,y or z depending on the selected axis.
__device__ __host__ inline int projectToBin(float k_1, float tri_centroid, float node_min_bounds){
  return (int) (k_1 * (tri_centroid - node_min_bounds));
}

__device__ __host__ inline float cost(int N_L, int N_R, float A_L, float A_R){
  //BUG: Might become an issue, *0.0001 to decrease cost (avoiding overflowing for high costs)
  return (A_L * N_L + A_R * N_R) * 0.001;
}

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= triangle_count)
    return;

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
  Triangle* triangles;
  Triangle* ptr_device_triangles;
  Vec3* ptr_device_vertex_buffer;
  int triangle_count;
  int vertex_count;

  int nodes_created = 0;

  AABB scene_bounds_triangles;
  AABB scene_bounds_centroids;

  int* triangle_ids;
  int* temp_triangle_ids;
  Node* nodes;
  int nodes_length;

  public:
  SAHBVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count, AABB scene_bounds_triangles){
    this->triangle_ids = (int*)malloc(sizeof(int) * triangle_count);
    for (int i = 0; i < triangle_count; i++)
      triangle_ids[i] = i;
    this->temp_triangle_ids = (int*)malloc(sizeof(int) * triangle_count);
    memcpy(temp_triangle_ids, triangle_ids, triangle_count);

    //TODO: While the maximum number of nodes in the tree is (2n)-1, we do not need leaf nodes (?).
    this->nodes_length = 2*triangle_count-1;
    this->nodes = (Node*)calloc(nodes_length, sizeof(Node));
    this->vertex_count = vertex_count;
    this->triangle_count = triangle_count;
    this->scene_bounds_triangles = scene_bounds_triangles;
    this->ptr_device_triangles = ptr_device_triangles;
    this->ptr_device_vertex_buffer = ptr_device_vertices;
  }

  ~SAHBVH(){
    free(nodes);
    free(triangle_ids);
    free(temp_triangle_ids);
  }

  void splitNode(Node* node, Node* nodes, int start, int end, int* triangle_ids, int* temp_triangle_ids, Triangle* triangles, int depth);

  //Returns device ptr to root of tree.
  Node* construct(){
    printf("Starting SAH Binning construction.\n");

    //TODO: Check that this is actually faster than just computing the centroids on the CPU.
    computeBoundsAndCentroids<<<triangle_count/64+1, 64>>>(ptr_device_triangles, triangle_count, ptr_device_vertex_buffer);
    this->triangles = (Triangle*)malloc(sizeof(Triangle) * triangle_count);
    checkCudaErrors(cudaMemcpy(triangles, ptr_device_triangles, sizeof(Triangle) * triangle_count, cudaMemcpyDeviceToHost));

    Node* root_node = new Node();
    root_node->start_range = 0;
    root_node->range = triangle_count;
    root_node->aabb = scene_bounds_triangles;
    nodes[0] = *root_node;

    //@Debug print all nodes
    for (int i = 0; i < nodes_length; i++)
      printf("Node: %i 0x%p, L: 0x%p, R: 0x%p Start: %i, Range: %i, End: %i\n", i, &nodes[i], nodes[i].left_child, nodes[i].right_child, nodes[i].start_range, nodes[i].range, nodes[i].start_range + nodes[i].range);
    //@Debug print all triangle ids
    for (int i = 0; i < triangle_count; i++)
      printf("%i ", triangle_ids[i]);

    printf("---- STARTING SWEEPING ----\n");
    nodes_created = 0;
    splitNode(nodes, nodes, 0, triangle_count, triangle_ids, temp_triangle_ids, triangles, 0);
    printf("---- SWEEPING DONE ----\n");

    //@Debug print all triangle ids
    for (int i = 0; i < triangle_count; i++)
      printf("%i ", triangle_ids[i]);
    printf("\n");
    //@Debug print all nodes
    for (int i = 0; i < nodes_length; i++)
      printf("Node: %i 0x%p, L: 0x%p, R: 0x%p Start: %i, Range: %i, End: %i, Size: (%f, %f, %f) \n", 
        i, 
        &nodes[i], 
        nodes[i].left_child, 
        nodes[i].right_child, 
        nodes[i].start_range, 
        nodes[i].range, 
        nodes[i].start_range + nodes[i].range,
        nodes[i].aabb.max_bounds.x() - nodes[i].aabb.min_bounds.x(),
        nodes[i].aabb.max_bounds.y() - nodes[i].aabb.min_bounds.y(),
        nodes[i].aabb.max_bounds.z() - nodes[i].aabb.min_bounds.z());
    // for (int i = 0; i < nodes_length; i++)
    //   printf("Node: %i Morton: 0: Min: %i Max: %i Split: 0 %i %i\n", i, nodes[i].start_range, nodes[i].start_range+nodes[i].range, nodes[i].left_child == nullptr, nodes[i].right_child == nullptr);

    printf("SAH Bin construction completed. %i nodes. \n", nodes_created);
    return nodes;
  }
};


void SAHBVH::splitNode(Node* node, Node* nodes, int start, int end, int* triangle_ids, int* temp_triangle_ids, Triangle* triangles, int depth){
    const int number_of_bins = 16;
    const int primitive_count = end - start;

    if(depth > 1000){
      printf("Exiting due to maximum depth.\n");
      return;
    }

    // printf("\n\n\n ------ Splitting node!!  ------ \n");

    //Calculate node AABB
    AABB node_bounds = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    AABB centroid_bounds = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    
    for(int i = start; i < end; i++){
      node_bounds.join(triangles[triangle_ids[i]].aabb); //TODO: Centroid or aabb?
      centroid_bounds.join(triangles[triangle_ids[i]].centroid);
    }
    node->aabb = node_bounds;

    //Decide which axis to sweep. (the longest side).
    int axis;
    Vec3 size = centroid_bounds.max_bounds - centroid_bounds.min_bounds;
    if(size.x() > size.y() && size.x() > size.z())
      axis = 0;
    else if(size.y() > size.x() && size.y() > size.z())
      axis = 1;
    else
      axis = 2;

    //Compute k_1 for the selected axis.
    float k_1 = number_of_bins * (1.0 - FLT_EPSILON) / 
                  (node_bounds.max_bounds.e[axis] - node_bounds.min_bounds.e[axis]);

    //Initialize per bin triangle counts and aabbs.
    int bin_triangle_counts[number_of_bins];
    for (int i = 0; i < number_of_bins; i++)
      bin_triangle_counts[i] = 0;
    AABB bin_aabbs[number_of_bins];
    for (int i = 0; i < number_of_bins; i++)
      bin_aabbs[i] = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));

    //Calculate N_l, N_r, A_l & A_r for all triangles for all bins.
    //TODO: This could be better parallelized?
    for(int i = start; i < end; i++){
      int bin_index = projectToBin( k_1, 
                                    triangles[triangle_ids[i]].centroid.e[axis],
                                    centroid_bounds.min_bounds.e[axis]);

      bin_triangle_counts[bin_index]++;
      bin_aabbs[bin_index].join(triangles[triangle_ids[i]].aabb);  //TODO: Should this be triangle centroid or aabb?
    }

    //@debug Print all bins
    // for (int i = 0; i < number_of_bins; i++)
    //   printf("Bin: %i, Tris: %i, AABB SAH: %f\n", i, bin_triangle_counts[i], bin_aabbs[i].surfaceArea());
    // printf("\n");

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
    for (int i = 0; i < number_of_bins - 1; i++)
      aabb_r_sweep[i] =  AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    
    aabb_r_sweep[number_of_bins-1] = bin_aabbs[number_of_bins-1];
    float min_cost = FLT_MAX;
    int split_index = 0;

    // printf("Operating on range %i - %i\n", start, end);

    float bin_width = size.e[axis] / number_of_bins;
    float split_position = centroid_bounds.min_bounds.e[axis] + (bin_width*split_index);
    // printf("Evaluating splitting %i, \tCost: %f, Prim L: %i Prim R: %i SA L: %f SA R: %f\n", 
    //     number_of_bins-1, 
    //     cost(tri_count_l_sweep[number_of_bins-1],
    //           primitive_count - tri_count_l_sweep[number_of_bins-1],
    //           aabb_l_sweep[number_of_bins-1].surfaceArea(),
    //           aabb_r_sweep[number_of_bins-1].surfaceArea()),
    //     tri_count_l_sweep[number_of_bins-1], 
    //     primitive_count - tri_count_l_sweep[number_of_bins-1], 
    //     aabb_l_sweep[number_of_bins-1].surfaceArea(), 
    //     aabb_r_sweep[number_of_bins-1].surfaceArea());

    for (int i = number_of_bins-2; i >= 0; i--){
      int primitives_left = tri_count_l_sweep[i];
      aabb_r_sweep[i] = AABB::join(aabb_r_sweep[i+1], bin_aabbs[i]);
      float sah_cost = cost(  primitives_left,                    //N_L
                              primitive_count - primitives_left,  //N_R
                              aabb_l_sweep[i].surfaceArea(),      //A_L
                              aabb_r_sweep[i].surfaceArea());     //A_R

      if(sah_cost < min_cost){
        min_cost = sah_cost;
        split_index = i;
      }

      //@Debug
      // split_position = centroid_bounds.min_bounds.e[axis] + (bin_width*split_index);
      // printf("Evaluating splitting %i, \tCost: %f, Prim L: %i Prim R: %i SA L: %f SA R: %f SplitPos: %f\n", 
      //   i, 
      //   sah_cost, 
      //   primitives_left, 
      //   primitive_count - primitives_left, 
      //   aabb_l_sweep[i].surfaceArea(), 
      //   aabb_r_sweep[i].surfaceArea(),
      //   split_position);
    }

    split_position = centroid_bounds.min_bounds.e[axis] + (bin_width*split_index);

    //Copy triangle ids to temporary array.
    for (int i = start; i < end; i++)
      temp_triangle_ids[i] = triangle_ids[i];

    //Update triangle id buffers.
    int left_i = start;
    int right_i = end-1;
    for (int i = start; i < end; i++){
      if(triangles[temp_triangle_ids[i]].centroid.e[axis] <= split_position){
        // printf("moving triangle to left side. %f\n", triangles[temp_triangle_ids[i]].centroid.e[axis]);
        triangle_ids[left_i] = temp_triangle_ids[i];
        left_i++;
      }
      else{
        // printf("moving triangle to right side. %f\n", triangles[temp_triangle_ids[i]].centroid.e[axis]);
        triangle_ids[right_i] = temp_triangle_ids[i];
        right_i--;
      }
    }

    int split = left_i;
    int primitive_count_l = split - start;
    int primitive_count_r = end - split;
    // printf("start: %i split between: %i and %i (index: %i), end: %i, pos: %f, axis: (%f to %f), left_i:%i right_i: %i\n\n", 
    //   start, 
    //   split, 
    //   split+1, 
    //   split_index, 
    //   end, 
    //   split_position,
    //   centroid_bounds.min_bounds.e[axis],
    //   centroid_bounds.max_bounds.e[axis],
    //   left_i,
    //   right_i);

    if(split == start || split == end){
      node->start_range = start;
      node->range = end - start;
      // printf("^ This should be a leaf node. (Start %i, Range %i, End %i)\n", start, end-start, end);
      return;
    }

    //Record node relationships.
    this->nodes_created++;
    Node* left_child = &nodes[nodes_created];
    printf("%i\t", nodes_created);
    left_child->parent = node;
    left_child->start_range = start;
    left_child->range = split - start;
    left_child->aabb = aabb_l_sweep[split_index];
    node->left_child = left_child;

    if(primitive_count_l > 1)
      splitNode(left_child, nodes, start, split, triangle_ids, temp_triangle_ids, triangles, depth+1);

    this->nodes_created++;
    Node* right_child = &nodes[nodes_created];
    right_child->parent = node;
    right_child->start_range = split;
    right_child->range = end - split;
    right_child->aabb = aabb_r_sweep[split_index];
    node->right_child = right_child;

    if(primitive_count_r > 1)
      splitNode(right_child, nodes, split, end, triangle_ids, temp_triangle_ids, triangles, depth+1);
  }