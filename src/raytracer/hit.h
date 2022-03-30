#pragma once
#include "vec3.h"

struct RayHit{
    Vec3 pos;
    Vec3 normal;
    float dist;
    Vec2 uv;
    __host__ __device__ RayHit();
};

__host__ __device__ RayHit::RayHit() {dist = 99999999.0f;}