#pragma once

#include "third_party/shared_queue.h"
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include "vec3.h"
#include "triangle.h"
#include "node.h"
#include "debug.cu"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void computeBoundsAndCentroids(Triangle* triangles, int triangle_count, Vec3* ptr_device_vertex_buffer, int* ptr_device_triangle_ids);

inline int projectToBin(float k_1, float centroid_bin_axis, float scene_min_axis);
inline float cost(int N_L, int N_R, float A_L, float A_R);

//Inputs along the selected axis. E.g. tri_centroid for x,y or z depending on the selected axis.
inline int projectToBin(float k_1, float tri_centroid, float node_min_bounds){
  return min(max((int) (k_1 * (tri_centroid - node_min_bounds)), 0), 16);
}

inline float cost(int N_L, int N_R, float A_L, float A_R){
  return (A_L * N_L + A_R * N_R);
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

//See comment in SAHBVH constructor regarding the use of this function.
__global__ void deepCopyTreeToGPU(Node* input_nodes, int nodes_length, Node* output_internal_nodes, Node* output_leaf_nodes, Triangle* triangles, int internal_nodes_count, int leaf_nodes_count, int* triangle_ids){
  //Traverse tree in reverse order and copy nodes to internal_nodes and leaf_nodes buffers.
  int internal_nodes = internal_nodes_count;
  int leaf_nodes = leaf_nodes_count-1;

  for (int i = nodes_length; i >= 0; i--){
    Node node = input_nodes[i];
    if(node.is_leaf){
      // printf("Leaf: Start range: %i, Leaf nodes: %i Nodes left: %i, Internal Nodes %i \n", node.start_range, leaf_nodes, i, internal_nodes);
      node.primitive = &triangles[triangle_ids[node.start_range]];
      output_leaf_nodes[leaf_nodes] = node;
      input_nodes[i].parent = &output_leaf_nodes[leaf_nodes]; //leave a trail for the parent to pick up.
      leaf_nodes--;
    }
    else{
      // printf("Left Child %i, Right Child: %i, Internal Nodes %i, Index: %i\n", node.left_child_i, node.right_child_i, internal_nodes, i);
      node.left_child = input_nodes[node.left_child_i].parent;
      node.right_child = input_nodes[node.right_child_i].parent;
      output_internal_nodes[internal_nodes] = node;
      input_nodes[i].parent = &output_internal_nodes[internal_nodes]; //leave a trail for the parent to pick up.
      internal_nodes--;
    }
  }
}

class SAHBVH{
  Triangle* triangles;
  Triangle* ptr_device_triangles;
  Vec3* ptr_device_vertex_buffer;
  int triangle_count;
  int vertex_count;

  std::atomic_int nodes_created;
  std::atomic_int leaf_nodes_created;

  AABB scene_bounds_triangles;
  AABB scene_bounds_centroids;

  int* triangle_ids;
  int* temp_triangle_ids;
  Node* ptr_host_internal_nodes;

  Node* ptr_device_internal_nodes;
  Node* ptr_device_leaf_nodes;
  Node* ptr_device_temp_nodes;
  int nodes_length;

  SharedQueue<Node*> work_queue;
  std::vector<std::thread> workers;

  public:
  SAHBVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count, AABB scene_bounds_triangles){
    this->triangle_ids = (int*)malloc(sizeof(int) * triangle_count);
    for (int i = 0; i < triangle_count; i++)
      triangle_ids[i] = i;
    this->temp_triangle_ids = (int*)malloc(sizeof(int) * triangle_count);
    memcpy(temp_triangle_ids, triangle_ids, triangle_count);

    this->nodes_length = 2*triangle_count-1;
    this->ptr_host_internal_nodes = (Node*)calloc(nodes_length, sizeof(Node));
    //Initialize indices to -1, indicating no child is present.
    for (int i = 0; i < nodes_length; i++){
      this->ptr_host_internal_nodes[i].right_child_i = -1;
      this->ptr_host_internal_nodes[i].left_child_i = -1;
    }

    this->nodes_created.store(0);
    this->leaf_nodes_created.store(0);
    
    this->vertex_count = vertex_count;
    this->triangle_count = triangle_count;
    this->scene_bounds_triangles = scene_bounds_triangles;
    this->ptr_device_triangles = ptr_device_triangles;
    this->ptr_device_vertex_buffer = ptr_device_vertices;

    checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1) * sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_leaf_nodes, triangle_count * sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_temp_nodes, nodes_length * sizeof(Node)));
  }

  ~SAHBVH(){
    free(ptr_host_internal_nodes);
    free(triangle_ids);
    free(temp_triangle_ids);
    checkCudaErrors(cudaFree(ptr_device_temp_nodes));
  }

  void splitNode(SAHBVH *bvh, Node* node, Node* nodes, int start, int end, int* triangle_ids, int* temp_triangle_ids, Triangle* triangles, int depth);
  void creationThread(SAHBVH* bvh, int thread_id);

  //Returns device ptr to root of tree.
  Node* construct(){
    using namespace std::chrono;
    // printf("Starting SAH Binning construction.\n");

    computeBoundsAndCentroids<<<triangle_count/64+1, 64>>>(ptr_device_triangles, triangle_count, ptr_device_vertex_buffer);
    checkCudaErrors(cudaDeviceSynchronize());
    this->triangles = (Triangle*)malloc(sizeof(Triangle) * triangle_count);
    checkCudaErrors(cudaMemcpy(triangles, ptr_device_triangles, sizeof(Triangle) * triangle_count, cudaMemcpyDeviceToHost));

    Node* root_node = new Node();
    root_node->start_range = 0;
    root_node->depth = 0;
    root_node->range = triangle_count;
    root_node->aabb = scene_bounds_triangles;
    root_node->left_child_i = 1;
    root_node->right_child_i = 2;
    ptr_host_internal_nodes[0] = *root_node;

    work_queue.push(root_node);
    steady_clock::time_point start = high_resolution_clock::now();
    int num_threads = std::thread::hardware_concurrency();

    for (int i = 0; i < num_threads; i++){
      workers.push_back(std::thread([this, i] {
        this->creationThread(this, i);
      }));
      while(work_queue.size() < num_threads); //HACK: Busy-wait until theres work available for all threads.
    }
    for(int i = 0; i < workers.size(); i++){
      if (workers[i].joinable()){
        workers[i].join();
      }
    }

    steady_clock::time_point stop = high_resolution_clock::now();
    long long duration_ms = duration_cast<milliseconds>(stop - start).count();
    long long duration_us = duration_cast<microseconds>(stop - start).count();
    printf("Binned SAH\t%lli\tms\t%lli\tus\n", duration_ms, duration_us);

    //This memcpy is inevitable if we construct tree on the cpu and render on the gpu.
    checkCudaErrors(cudaMemcpy(ptr_device_temp_nodes, ptr_host_internal_nodes, nodes_length * sizeof(Node), cudaMemcpyHostToDevice));

    int* ptr_device_triangle_ids;
    checkCudaErrors(cudaMalloc(&ptr_device_triangle_ids, sizeof(int) * triangle_count));
    checkCudaErrors(cudaMemcpy(ptr_device_triangle_ids, triangle_ids, sizeof(int) * triangle_count, cudaMemcpyHostToDevice));
    
    //This could be omitted if the tracing traversal is modified to be compliant with our CPU tree structure.
    //As of right now we're using the same traversal algorithm for both lbvh and binned sah bvh, but the structure is slightly different. 
    //We must therefore first copy the data to the GPU and then update all pointers in the tree to be device, not host.
    deepCopyTreeToGPU<<<1,1>>>(ptr_device_temp_nodes,
      nodes_created.load(),
      ptr_device_internal_nodes,
      ptr_device_leaf_nodes,
      ptr_device_triangles,
      nodes_created.load() - leaf_nodes_created.load(),
      leaf_nodes_created.load(),
      ptr_device_triangle_ids);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(ptr_device_triangle_ids));

    return ptr_device_internal_nodes;
  }
};

void SAHBVH::creationThread(SAHBVH* bvh, int thread_id){
  // int nodes_processed = 0;

  while(!bvh->work_queue.empty()){
    Node* active_node = bvh->work_queue.pop_front();
    splitNode(this,
      active_node,
      this->ptr_host_internal_nodes,
      active_node->start_range,
      active_node->start_range + active_node->range,
      this->triangle_ids,
      this->temp_triangle_ids,
      this->triangles,
      0);
    // nodes_processed++;
  }
  // printf("Thread: %i, Nodes processed: %i\n", thread_id, nodes_processed);
}

void SAHBVH::splitNode(SAHBVH *bvh, Node* node, Node* nodes, int start, int end, int* triangle_ids, int* temp_triangle_ids, Triangle* triangles, int depth){
  const int number_of_bins = 16;
  const int primitive_count = end - start;

  node->aabb           = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
  AABB centroid_bounds = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
  
  for(int i = start; i < end; i++){
    node->aabb.join(triangles[triangle_ids[i]].aabb);
    centroid_bounds.join(triangles[triangle_ids[i]].centroid);
  }

  //Decide which axis to sweep. (the longest side).
  int axis;
  Vec3 size = centroid_bounds.max_bounds - centroid_bounds.min_bounds;
  if(size.x() >= size.y() && size.x() >= size.z())
    axis = 0;
  else if(size.y() >= size.x() && size.y() >= size.z())
    axis = 1;
  else
    axis = 2;

  //Compute k_1 for the selected axis.
  float k_1 = number_of_bins * (1.0 - FLT_EPSILON) / 
                (node->aabb.max_bounds.e[axis] - node->aabb.min_bounds.e[axis]);

  //Initialize per bin triangle counts and aabbs.
  int bin_triangle_counts[number_of_bins];
  for (int i = 0; i < number_of_bins; i++)
    bin_triangle_counts[i] = 0;
  AABB bin_aabbs[number_of_bins];
  for (int i = 0; i < number_of_bins; i++)
    bin_aabbs[i] = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));

  //Calculate N_l, N_r, A_l & A_r for all triangles for all bins.
  for(int i = start; i < end; i++){
    int bin_index = projectToBin( k_1, 
                                  triangles[triangle_ids[i]].centroid.e[axis],
                                  centroid_bounds.min_bounds.e[axis]);

    bin_triangle_counts[bin_index]++;
    bin_aabbs[bin_index].join(triangles[triangle_ids[i]].aabb);
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
  for (int i = 0; i < number_of_bins - 1; i++)
    aabb_r_sweep[i] =  AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));

  aabb_r_sweep[number_of_bins-1] = bin_aabbs[number_of_bins-1];
  float min_cost = FLT_MAX;
  int split_index = 0;

  float bin_width = size.e[axis] / number_of_bins;
  float split_position = centroid_bounds.min_bounds.e[axis] + (bin_width*split_index);

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
      triangle_ids[left_i] = temp_triangle_ids[i];
      left_i++;
    }
    else{
      triangle_ids[right_i] = temp_triangle_ids[i];
      right_i--;
    }
  }
  int split = left_i;

  if(split == start || split == end){
    node->start_range = start;
    node->range = end - start;
    leaf_nodes_created++;
    return;
  }
  node->is_leaf = false;

  //Record node relationships.
  int node_index = ++nodes_created;
  Node* left_child = &nodes[node_index];
  left_child->start_range = start;
  left_child->range = split - start;
  left_child->depth = node->depth + 1;
  left_child->is_leaf = true;
  node->left_child_i = left_child - nodes;
  work_queue.push(left_child);

  node_index = ++nodes_created;
  Node* right_child = &nodes[node_index];
  right_child->start_range = split;
  right_child->range = end - split;
  right_child->is_leaf = true;
  right_child->depth = node->depth + 1;
  node->right_child_i = right_child - nodes;
  work_queue.push(right_child);
}