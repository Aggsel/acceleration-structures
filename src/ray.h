#pragma once
#include "vec3.h"

class Ray{
  Vec3 org;
  Vec3 dir;
  public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3 a, const Vec3 b) { org = a; dir = b; }
    __host__ __device__ Vec3 origin() const { return org; }
    __host__ __device__ Vec3 direction() const { return dir; }
    __host__ __device__ Vec3 point_along_ray(float t) const { return org + t*dir; }
};