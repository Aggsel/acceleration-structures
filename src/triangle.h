#pragma once
#include "vec3.h"

struct Triangle{
  public:
  unsigned int morton_code;
  int v0_index;
  int v1_index;
  int v2_index;
  AABB aabb;
  Vec3 centroid;
};
__host__ __device__ inline bool operator< (const Triangle& lhs, const Triangle& rhs){ return lhs.morton_code < rhs.morton_code; }
__host__ __device__ inline bool operator> (const Triangle& lhs, const Triangle& rhs){ return rhs < lhs; }
__host__ __device__ inline bool operator<=(const Triangle& lhs, const Triangle& rhs){ return !(lhs > rhs); }
__host__ __device__ inline bool operator>=(const Triangle& lhs, const Triangle& rhs){ return !(lhs < rhs); }