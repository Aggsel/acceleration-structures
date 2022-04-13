#pragma once

#include "node.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void d_printNodes(Node* nodes, int size, Triangle* triangles){
    for (int i = 0; i < size; i++){
        Node* node = &nodes[i];
        printf("Node: %i\tNode: 0x%p\tParent: 0x%p, \tLeft Child: 0x%p, \tRight Child: 0x%p\tMorton: %u Depth: %i IsLeaf: %i AABB: (%f, %f, %f)\n", 
            i, 
            node, 
            node->parent, 
            node->left_child, 
            node->right_child, 
            triangles[i].morton_code, 
            node->depth, 
            node->is_leaf,
            node->aabb.max_bounds.x() - node->aabb.min_bounds.x(),
            node->aabb.max_bounds.y() - node->aabb.min_bounds.y(),
            node->aabb.max_bounds.z() - node->aabb.min_bounds.z());
    }
}

namespace DebugHelper{
    void PrintNodes(Node* node_array, int size, Triangle* leaf_nodes){
        d_printNodes<<<1, 1>>>(node_array, size, leaf_nodes);
        cudaDeviceSynchronize();
    }
}