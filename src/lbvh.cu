#pragma once

#include "triangle.h"
#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void constructLBVH(Triangle *triangles, Node* internal_nodes, Node* leaf_nodes, int primitive_count);
__device__ int2 determineRange(Triangle *sorted_morton_codes, int total_primitives, int node_index);
__device__ int findSplit(Triangle *sorted_morton_codes, int first, int last);
__global__ void calculateAABB(Node* internal_nodes, Triangle* leaf_nodes, int leaf_count, Vec3* vert_buff);

__device__ int commonPrefix(Triangle *morton_codes, int index1, int index2){
  unsigned int key1 = morton_codes[index1].morton_code;
  unsigned int key2 = morton_codes[index2].morton_code;
  if(key1 != key2)
    return __clz(key1 ^ key2);
  return __clz(index1 ^ index2);
}

__device__ int2 determineRange(Triangle *sorted_morton_codes, int total_primitives, int node_index){
  //Time complexity of the algorithm is proportional to the number of keys covered by the nodes.
  //The widest node is also one that we know in advance:
  if(node_index == 0){
    int2 range;
    range.x = 0;
    range.y = total_primitives-1;
    return range;
  }

  //Determine direction (d).
  //Delta being the number of largest common bits between two keys.
  int next_delta = commonPrefix(sorted_morton_codes, node_index, node_index+1);
  int prev_delta = commonPrefix(sorted_morton_codes, node_index, node_index-1);
  int d = next_delta - prev_delta < 0 ? -1 : 1;

  //Compute upper bound for the length of the range.
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

  int2 min_max;
  min_max.x = min(node_index, j);
  min_max.y = max(node_index, j);
  return min_max;
}

//From https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__device__ int findSplit(Triangle *sorted_morton_codes, int first, int last){
  int first_morton = sorted_morton_codes[first].morton_code;
  int last_morton = sorted_morton_codes[last].morton_code;

  if(first_morton == last_morton)
    return (first + last) >> 1;

  //count leading zeros
  int common_prefix = commonPrefix(sorted_morton_codes, first, last);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than common_prefix bits with the first one.
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

  //binary search morton codes.
  int2 range = determineRange(triangles, primitive_count, node_index);
  int first = range.x;
  int last = range.y;

  // Determine where to split the range.
  int split = findSplit(triangles, first, last);

  // Select left_child.
  Node* left_child;
  if(split == first){
    left_child = &leaf_nodes[split];
    left_child->primitive = &triangles[split];
    left_child->aabb = triangles[split].aabb;
    left_child->isLeaf = true;
  }
  else{
    left_child = &internal_nodes[split];
    left_child->isLeaf = false;
  }
  
  // Select right_child.
  Node* right_child;
  if(split + 1 == last){
    right_child = &leaf_nodes[split + 1];
    right_child->primitive = &triangles[split + 1];
    right_child->aabb = triangles[split + 1].aabb;
    right_child->isLeaf = true;
  }
  else{
    right_child = &internal_nodes[split + 1];
    right_child->isLeaf = false;
  }

  // printf("Node: %i, \tMorton: %i, \tMin: %i, \tMax: %i, \tSplit: %i, \t%i, \t%i\n", node_index, triangles[node_index].morton_code, first, last, split, split == first, split + 1 == last); // @debug
  Node *self_ptr = &internal_nodes[node_index];

  assert(self_ptr != nullptr);

  right_child->parent = self_ptr;
  left_child->parent = self_ptr;
  self_ptr->left_child = left_child;
  self_ptr->right_child = right_child;

  // Node 0 is always the root of the tree, the pointer supplied (Node* internalNodes) will be the address of the root.
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

  parent_node->aabb = leaf_aabb; //BUG: <- sometimes cudaErrorIllegalAddress, due to duplicate morton codes.

  while(true){
    if(parent_node == nullptr)  //Root reached.
      return;

    int parent_index = parent_node - internal_nodes;
    int old = atomicCAS(&counter[parent_index], 0, 1);
    if(old == 0){ //This thread reached the node first. 
      return;
    }

    parent_node->aabb = AABB::join(parent_node->left_child->aabb,
                                        parent_node->right_child->aabb);

    parent_node = parent_node->parent;
  }
  
  //  Relevant resources:
  //  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  //  https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
}

// Modified from 
// https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
// TODO: This is not optimized AT ALL.
__global__ void traverseTree(Node* root){
  printf("\nTraversing tree...\n");
  Node* stack[128];
  int stack_index = -1;
  stack_index++;
  stack[stack_index] = nullptr;

  int i = 0;
  Node* node = root;
  do{
    Node* left_child = node->left_child;
    Node* right_child = node->right_child;

    //Only continue traversing children if they're no leaves.
    //TODO: This is where we can check for ray aabb intersections. 
    bool traverse_left  = !left_child ->isLeaf;
    bool traverse_right = !right_child->isLeaf;

    if(!traverse_left && !traverse_right){
      node = stack[stack_index];
      stack_index--;
    }
    else{
      //Prioritize traversing left branch.
      node = traverse_left ? left_child : right_child;

      //Push right child onto the stack if both branches should be traversed.
      if (traverse_left && traverse_right){
        stack_index++;
        stack[stack_index] = right_child;
      }
    }
    i++;
  }while(node != nullptr);
}

__global__ void printInternalNodes(Node* internal_nodes, int primitive_count, Triangle* leaf_nodes){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= primitive_count-1)
    return;
  Node* node = &internal_nodes[node_index];
  printf("Node: %i\tNode: 0x%p\tParent: 0x%p, \tLeft Child: 0x%p, \tRight Child: 0x%p\tMorton: %i\n", node_index, node, node->parent, node->left_child, node->right_child, leaf_nodes[node_index].morton_code);
}

__global__ void printLeafNodes(Node* leaf_nodes, int primitive_count, Triangle* triangles){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= primitive_count)
    return;
  Node* node = &leaf_nodes[node_index];
  printf("Leaf: \tNode: %i\tNode: 0x%p\tParent: 0x%p, \tLeft Child: 0x%p, \tRight Child: 0x%p\tMorton: %i\n", node_index, node, node->parent, node->left_child, node->right_child, triangles[node_index].morton_code);
}

class BVH{
  Node* ptr_device_internal_nodes;
  Node* ptr_device_leaf_nodes;

  Triangle* ptr_device_triangles;
  int triangle_count;
  Vec3* ptr_device_vertices;
  int vertex_count;

  int* ptr_device_visited_node_counters;

  public:
  BVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count){
    this->ptr_device_triangles = ptr_device_triangles;
    this->triangle_count = triangle_count;

    this->ptr_device_vertices = ptr_device_vertices;
    this->vertex_count = vertex_count;

    checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_leaf_nodes, (triangle_count)*sizeof(Node)));

    checkCudaErrors(cudaMemset(ptr_device_internal_nodes, 0, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMemset(ptr_device_leaf_nodes, 0, (triangle_count)*sizeof(Node)));

    checkCudaErrors(cudaMalloc(&ptr_device_visited_node_counters, (triangle_count-1)));
    //There's no calloc equivalent for cuda. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    checkCudaErrors(cudaMemset(ptr_device_visited_node_counters, 0, (triangle_count-1)));
  }

  ~BVH(){
    checkCudaErrors(cudaFree(ptr_device_internal_nodes));
    checkCudaErrors(cudaFree(ptr_device_leaf_nodes));
    checkCudaErrors(cudaFree(ptr_device_visited_node_counters));
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    //TODO: Move morton code generation, scene bounding box calculation etc to this function.
    // <<<x,y>>> Launches x thread blocks with y threads per block.
    const int threads_per_block = 512;

    constructLBVH<<<(triangle_count-1)/threads_per_block+1, threads_per_block>>>(ptr_device_triangles, ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());

    calculateAABB<<<triangle_count/threads_per_block+1, threads_per_block>>>(ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count, ptr_device_vertices, ptr_device_visited_node_counters);
    checkCudaErrors(cudaDeviceSynchronize());

    // --- @Debug purposes ---
    // printInternalNodes<<<(triangle_count-1)/threads_per_block+1, threads_per_block>>>(ptr_device_internal_nodes, triangle_count, ptr_device_triangles);
    // checkCudaErrors(cudaDeviceSynchronize());
    // printLeafNodes<<<triangle_count/threads_per_block+1, threads_per_block>>>(ptr_device_leaf_nodes, triangle_count, ptr_device_triangles);
    // checkCudaErrors(cudaDeviceSynchronize());
    // traverseTree<<<1,1>>>(ptr_device_internal_nodes);
    // checkCudaErrors(cudaDeviceSynchronize());

    printf("LBVH Construction completed.\n");
    return ptr_device_internal_nodes;
  }
};