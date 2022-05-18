#pragma once
#include "triangle.h"
#include "aabb.h"
#include "macros.h"

struct ALIGN(16) Node{
  Node* left_child;
  Node* right_child;
  Node* parent;
  bool is_leaf;

  AABB aabb;
  AABB centroid_aabb;
  int start_range;
  int range;
  int left_child_i;
  int right_child_i;

  int depth;

  Triangle* primitive;
};