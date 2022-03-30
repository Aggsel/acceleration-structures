#pragma once
#include "triangle.h"
#include "aabb.h"

struct Node{
    Node* leftChild;
    Node* rightChild;
    Node* parent;

    AABB aabb;

    bool isLeaf;
    Triangle* primitive;
};