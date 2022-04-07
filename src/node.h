#pragma once
#include "triangle.h"
#include "aabb.h"

struct Node{
  Node* left_child;
  Node* right_child;
  Node* parent;
  bool isLeaf;

  AABB aabb;
  int start_range;
  int range;

  Triangle* primitive;
};