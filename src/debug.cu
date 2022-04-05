#pragma once

#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void d_printNodes(Node* nodes, int size, Triangle* triangles){
    int node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(node_index >= size)
        return;
    Node* node = &nodes[node_index];
    printf("Node: %i\tNode: 0x%p\tParent: 0x%p, \tLeft Child: 0x%p, \tRight Child: 0x%p\tMorton: %i\n", node_index, node, node->parent, node->left_child, node->right_child, triangles[node_index].morton_code);
}

namespace DebugHelper{
    void PrintNodes(Node* node_array, int size, Triangle* leaf_nodes){
        int threads_per_block = 512;
        d_printNodes<<<size/threads_per_block+1, threads_per_block>>>(node_array, size, leaf_nodes);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}