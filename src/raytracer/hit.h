#pragma once
#include "vec3.h"
#include "macros.h"

struct ALIGN(16) RayHit{
  Vec3 pos;
  Vec3 normal;
  float dist;
  Vec2 uv;
  __host__ __device__ RayHit();
};

__host__ __device__ RayHit::RayHit() {dist = FLT_MAX;}