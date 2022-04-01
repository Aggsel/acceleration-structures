#pragma once

#include "triangle.h"
#include "node.h"
#include "cuda_helpers/helper_cuda.h"

__global__ void constructLBVH(Triangle *triangles, Node* internal_nodes, Node* leaf_nodes, int primitive_count);
__device__ int2 determineRange(Triangle *sorted_morton_codes, int total_primitives, int node_index);
__device__ int findSplit(Triangle *sorted_morton_codes, int first, int last);
__global__ void calculateAABB(Node* internal_nodes, Triangle* leaf_nodes, int leaf_count, Vec3* vert_buff);

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
  int current_code = sorted_morton_codes[node_index].morton_code;
  //BUG: Properly handle out of bounds exceptions. (branchless? :O)
  int prev_code = sorted_morton_codes[node_index-1].morton_code;
  int next_code = sorted_morton_codes[node_index+1].morton_code;
  //BUG: Prevent duplicate morton codes.
  // if(prev_code == current_code)
  //   prev_code = prev_code ^ node_index-1;
  // if(next_code == current_code)
  //   next_code = next_code ^ node_index+1;
  int next_delta = __clz(current_code ^ next_code);
  int prev_delta = __clz(current_code ^ prev_code);
  int d = next_delta - prev_delta < 0 ? -1 : 1;

  //Compute upper bound for the length of the range.
  //TODO: __Note from Karras 2012__:
  //      When searching for lmax on
  //      lines 5–8, we have found that it is beneficial to start from a
  //      larger number, e.g. 128, and multiply the value by 4 instead
  //      of 2 after each iteration to reduce the total amount of work.

  int lmax = 2;
  int delta_min = min(next_delta, prev_delta);
  int delta = -1;
  int i = node_index + d * lmax;
  if(i >= 0 && i < total_primitives){
      delta = __clz(current_code ^ sorted_morton_codes[i].morton_code);
  }

  while(delta > delta_min){
      lmax = lmax << 1;
      i = node_index + d * lmax;
      delta = -1;
      if(0 <= i && i < total_primitives)
          delta = __clz(current_code ^ sorted_morton_codes[i].morton_code);
  }

  //Binary search for the other end.
  int l = 0;
  int t = lmax >> 1;
  while(t > 0){
    i = node_index + (l + t) * d;
    delta = -1;
    if(0 <= i && i < total_primitives){
        delta = __clz(current_code ^ sorted_morton_codes[i].morton_code);
    }
    if(delta > delta_min){
        l += t;
    }
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
  int common_prefix = __clz(first_morton ^ last_morton);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than common_prefix bits with the first one.
  int split = first;
  int step = last - first;

  do{
    step = (step + 1) >> 1;
    int new_split = split + step;

    if (new_split < last){
      int split_morton = sorted_morton_codes[new_split].morton_code;
      int split_prefix = __clz(first_morton ^ split_morton);
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

  // Find out which range of objects the node corresponds to.
  // (This is where the magic happens!)

  //binary search morton codes.
  int2 range = determineRange(triangles, primitive_count, node_index);
  int first = range.x;
  int last = range.y;

  // Determine where to split the range.
  int split = findSplit(triangles, first, last);

  // Select left_child.
  Node* left_child; // = &internal_nodes[split];
  if(split == first){
    // printf("@BVH::constructLBVH() \tLeaf created during construction.\n"); //@debug
    left_child = &leaf_nodes[split];
    left_child->primitive = &triangles[split];
    left_child->aabb = triangles[split].aabb;
    left_child->isLeaf = true;
  }
  else{
    left_child = &internal_nodes[split];
  }
  left_child->parent = &internal_nodes[node_index];

  // Select right_child.
  Node* right_child; // = &internal_nodes[split + 1];
  if(split + 1 == last){
    // printf("@BVH::constructLBVH() \tLeaf created during construction.\n");  //@debug
    right_child = &leaf_nodes[split + 1];
    right_child->primitive = &triangles[split + 1];
    right_child->aabb = triangles[split + 1].aabb;
    right_child->isLeaf = true;
  }
  else{
    right_child = &internal_nodes[split + 1];
  }
  right_child->parent = &internal_nodes[node_index];

  printf("Node: %i, \tMorton: %i, \tMin: %i, \tMax: %i, \tSplit: %i, \t%i, \t%i\n", node_index, triangles[node_index].morton_code, first, last, split, split == first, split + 1 == last); // @debug

  // Record parent-child relationships.
  internal_nodes[node_index].left_child = left_child;
  internal_nodes[node_index].right_child = right_child;
  // Node 0 is always the root of the tree, the pointer supplied (Node* internalNodes) will be the address of the root.
}

__global__ void calculateAABB(Node* internal_nodes, Node* leaf_nodes, int leaf_count, Vec3* vert_buff, int* counter){
  int leaf_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(leaf_index >= leaf_count)
      return;

  int v0 = leaf_nodes[leaf_index].primitive->v0_index;
  int v1 = leaf_nodes[leaf_index].primitive->v1_index;
  int v2 = leaf_nodes[leaf_index].primitive->v2_index;

  Vec3 min_bounds = min(min(vert_buff[v0], vert_buff[v1]), vert_buff[v2]);
  Vec3 max_bounds = max(max(vert_buff[v0], vert_buff[v1]), vert_buff[v2]);
  AABB leaf_aabb;
  leaf_aabb.min_bounds = min_bounds;
  leaf_aabb.max_bounds = max_bounds;

  Node* current_node_ptr = leaf_nodes[leaf_index].parent;

  if(!current_node_ptr){
    printf("\nTHIS SHOULD NEVER HAPPEN OH GOD WHY\n");
    return;
  }

  current_node_ptr->aabb = leaf_aabb; //BUG: <- sometimes cudaErrorIllegalAddress

  while(true){
    if(!current_node_ptr){
      AABB aabb = internal_nodes[0].aabb;
      printf("@BVH::constructAABB() \t internal_nodes[0]: 0x%p internal_nodes: 0x%p Min Bounds: (%f, %f, %f)\n", internal_nodes, internal_nodes, aabb.min_bounds.x(), aabb.min_bounds.y(), aabb.min_bounds.z());  //@debug
      printf("@BVH::constructAABB() \t internal_nodes[0]: 0x%p internal_nodes: 0x%p Max Bounds: (%f, %f, %f)\n", internal_nodes, internal_nodes, aabb.max_bounds.x(), aabb.max_bounds.y(), aabb.max_bounds.z());  //@debug
      return; //Root reached, return.
    }

    int parent_index = current_node_ptr - internal_nodes;
    int old = atomicCAS(&counter[parent_index], 0, 1);
    if(old == 0){ //This thread reached the node first. 
      return;
    }

    // if(current_node_ptr->left_child != nullptr)
    //   current_node_ptr->aabb.join(current_node_ptr->left_child->aabb);
    // if(current_node_ptr->right_child != nullptr)
    //   current_node_ptr->aabb.join(current_node_ptr->right_child->aabb);
    current_node_ptr->aabb = AABB::join(current_node_ptr->left_child->aabb,
                                        current_node_ptr->right_child->aabb);

    // printf("Joined min(%f, %f, %f) max(%f, %f, %f) \t&& min(%f, %f, %f) max(%f, %f, %f) \t = min(%f, %f, %f) max(%f, %f, %f)\n",
    // current_node_ptr->left_child->aabb.min_bounds.x(),
    // current_node_ptr->left_child->aabb.min_bounds.y(),
    // current_node_ptr->left_child->aabb.min_bounds.z(),
    // current_node_ptr->left_child->aabb.max_bounds.x(),
    // current_node_ptr->left_child->aabb.max_bounds.y(),
    // current_node_ptr->left_child->aabb.max_bounds.z(),
    // current_node_ptr->right_child->aabb.min_bounds.x(),
    // current_node_ptr->right_child->aabb.min_bounds.y(),
    // current_node_ptr->right_child->aabb.min_bounds.z(),
    // current_node_ptr->right_child->aabb.max_bounds.x(),
    // current_node_ptr->right_child->aabb.max_bounds.y(),
    // current_node_ptr->right_child->aabb.max_bounds.z(),
    // current_node_ptr->aabb.min_bounds.x(),
    // current_node_ptr->aabb.min_bounds.y(),
    // current_node_ptr->aabb.min_bounds.z(),
    // current_node_ptr->aabb.max_bounds.x(),
    // current_node_ptr->aabb.max_bounds.y(),
    // current_node_ptr->aabb.max_bounds.z() );

    if(!current_node_ptr->parent)   //Parent does not exist, we should be at the root.
      printf("\nCurrent node (root): %p\n", current_node_ptr);

    current_node_ptr = current_node_ptr->parent;
  }
  
  //  Relevant resources:
  //  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  //  https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
}

//https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
//TODO: This is not optimized AT ALL.
__global__ void traverseTree(Node* root){
  printf("\nTraversing tree...\n");
  Node* stack[64];
  Node** stackPtr = stack;

  stackPtr++;
  *stackPtr = (Node*)NULL;

  int i = 0;
  int leaf_count = 0;
  Node* node = root;
  do{
    Node* left_child = node->left_child;
    Node* right_child = node->left_child;

    //Only continue traversing children if they're no leaves.
    //TODO: This is where we can check for ray aabb intersections. 
    bool traverseL = !left_child->isLeaf;
    bool traverseR = !right_child->isLeaf;

    if (!traverseL && !traverseR){
      leaf_count++;
      node = *--stackPtr;
    }
    else{
      node = (traverseL) ? left_child : right_child;
      if (traverseL && traverseR)
        *stackPtr++ = right_child;
    }
    i++;
  }while (node != nullptr);

  printf("Traversed %i nodes, %i leaves found.\n", i, leaf_count);
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
  //BUG: We have no guarantee that the triangle or vertex pointer can be dereferenced 
  //     safely for the entire lifetime of this object.
  BVH(Triangle* ptr_device_triangles, int triangle_count, Vec3* ptr_device_vertices, int vertex_count){
    this->ptr_device_triangles = ptr_device_triangles;
    this->triangle_count = triangle_count;

    this->ptr_device_vertices = ptr_device_vertices;
    this->vertex_count = vertex_count;

    checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1)*sizeof(Node)));
    checkCudaErrors(cudaMalloc(&ptr_device_leaf_nodes, (triangle_count)*sizeof(Node)));

    checkCudaErrors(cudaMalloc(&ptr_device_visited_node_counters, (triangle_count-1)));
    //There's no calloc equivalent for cuda. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    checkCudaErrors(cudaMemset(ptr_device_visited_node_counters, 0, (triangle_count-1)));
  }

  ~BVH(){
    //BUG: We might want to leave the responsibility of deallocation to the caller of BVH::construct().
    //     This could result in a nullpt when trying to access the tree.
    checkCudaErrors(cudaFree(ptr_device_internal_nodes));
    checkCudaErrors(cudaFree(ptr_device_leaf_nodes));
    checkCudaErrors(cudaFree(ptr_device_visited_node_counters));
  }

  //Returns device ptr to root of tree.
  Node* construct(){
    //TODO: Move morton code generation, scene bounding box calculation etc to this function.
    // <<<x,y>>> Launches x thread blocks with y threads per block.
    constructLBVH<<<(triangle_count-1)/64+1,64>>>(ptr_device_triangles, ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());

    calculateAABB<<<triangle_count/64+1,64>>>(ptr_device_internal_nodes, ptr_device_leaf_nodes, triangle_count, ptr_device_vertices, ptr_device_visited_node_counters);
    checkCudaErrors(cudaDeviceSynchronize());

    traverseTree<<<1,1>>>(ptr_device_internal_nodes);
    checkCudaErrors(cudaDeviceSynchronize());

    return ptr_device_internal_nodes;
  }
};