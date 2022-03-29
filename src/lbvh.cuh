#pragma once

#include "triangle.h"
#include "node.h"
#include "cuda_helpers/helper_cuda.h"

__global__ void constructLBVH(Triangle *triangles, Node* internalNodes, int primitive_count);

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
  //TODO: When searching for lmax on
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

__global__ void constructLBVH(Triangle *triangles, Node* internalNodes, int primitive_count){
  int node_index = blockIdx.x *blockDim.x + threadIdx.x;
  if(node_index >= primitive_count)
    return;

  // Find out which range of objects the node corresponds to.
  // (This is where the magic happens!)

  //binary search morton codes.
  int2 range = determineRange(triangles, primitive_count, node_index);
  int first = range.x;
  int last = range.y;

  // Determine where to split the range.
  int split = findSplit(triangles, first, last);

//   printf("Node: %i, \tMorton: %i, \tMin: %i, \tMax: %i, \tSplit: %i\n", node_index, triangles[node_index].morton_code, first, last, split); // @debug

  // Select left_child.
  Node* left_child = &internalNodes[split];
  if (split == first){
    left_child->primitive = &triangles[split];
    left_child->isLeaf = true;
  }
  else{
    left_child = &internalNodes[split];
  }

  // Select rightChild.
  Node* right_child = &internalNodes[split + 1];
  if (split + 1 == last){
    right_child->primitive = &triangles[split];
    right_child->isLeaf = true;
  }
  else{
    right_child = &internalNodes[split + 1];
  }

  // Record parent-child relationships.
  internalNodes[node_index].leftChild = left_child;
  internalNodes[node_index].rightChild = right_child;
  left_child->parent = &internalNodes[node_index];
  right_child->parent = &internalNodes[node_index];

  // Node 0 is always the root of the tree, the pointer supplied (Node* internalNodes) will be the address of the root.
}

__host__ class BVH{
    Node* ptr_device_internal_nodes;
    
    Triangle* ptr_device_triangles;
    int triangle_count;

    public:
    //BUG: We have no guarantee that the triangle pointer can be dereferenced 
    //     safely for the entire lifetime of this object.
    BVH(Triangle* ptr_device_triangles, int triangle_count){
        this->ptr_device_triangles = ptr_device_triangles;
        this->triangle_count = triangle_count;
        checkCudaErrors(cudaMalloc(&ptr_device_internal_nodes, (triangle_count-1)*sizeof(Node)));
    }

    ~BVH(){
        checkCudaErrors(cudaFree(ptr_device_internal_nodes));
    }

    __host__ void construct(){
        //TODO: Move morton code generation, scene bounding box calculation etc to this function.
        // <<<x,y>>> Launches x thread blocks with y threads per block.
        constructLBVH<<<triangle_count/64+1,64>>>(ptr_device_triangles, ptr_device_internal_nodes, triangle_count);
        checkCudaErrors(cudaDeviceSynchronize());

        //TODO: Calculate AABB for each node. 
        //      1. Traverse the tree from the nodes, performing a union of previously visited nodes and the nurrent nodes AABB.
        //      2. Using atomic counter intructions, increment the counter for each node visited.
        //         Since we've constructed a binary tree, a maximum of two branches will ever reach each node.
        //      3. If the current thread is first, perform union and terminate.
        // calculateBounds<<<primitive_count/64+1,64>>>();
    }
};