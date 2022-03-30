#pragma once
#include "vec3.h"
#include "raytracer/ray.h"

class AABB{
    public:
    Vec3 min_bounds;
    Vec3 max_bounds;
    __device__ __host__ void join(AABB otherAABB);
    __device__ __host__ bool intersect_ray(Ray ray);
};