#pragma once
#include "triangle.h"
#include "aabb.h"

struct Node{
    Node* left_child;
    Node* right_child;
    Node* parent;

    AABB aabb;

    Triangle* primitive;
};