#pragma once
#include "triangle.h"

struct Node{
    Node* leftChild;
    Node* rightChild;
    Node* parent;

    bool isLeaf;
    Triangle* primitive;
};