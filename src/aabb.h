#pragma once
#include "vec3.h"
#include "raytracer/ray.h"

class AABB{
    public:
    Vec3 min_bounds;
    Vec3 max_bounds;
    __device__ __host__ AABB(Vec3 min_bounds, Vec3 max_bounds){
        this->min_bounds = min_bounds;
        this->max_bounds = max_bounds;
    }
    AABB() = default;
    __device__ __host__ void join(AABB otherAABB);
    __device__ __host__ static AABB join(AABB aabb_1, AABB aabb_2);
    __device__ __host__ bool intersect_ray(Ray ray);
};


__device__ __host__ void AABB::join(AABB otherAABB){
    this->min_bounds = min(this->min_bounds, otherAABB.min_bounds);
    this->max_bounds = max(this->max_bounds, otherAABB.max_bounds);
}

__device__ __host__ AABB AABB::join(AABB aabb_1, AABB aabb_2){
    AABB aabb;
    aabb.min_bounds = min(aabb_1.min_bounds, aabb_2.min_bounds);
    aabb.max_bounds = max(aabb_1.max_bounds, aabb_2.max_bounds);
    return aabb;
}

//https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
__device__ __host__ bool AABB::intersect_ray(Ray ray){
    float tmin = (min_bounds.x() - ray.org.x()) / ray.dir.x(); 
    float tmax = (max_bounds.x() - ray.org.x()) / ray.dir.x(); 

    if (tmin > tmax){
        float temp = tmax;
        tmax = tmin;
        tmin = temp;
    }

    float tymin = (min_bounds.y() - ray.org.y()) / ray.dir.y(); 
    float tymax = (max_bounds.y() - ray.org.y()) / ray.dir.y(); 

    if (tymin > tymax){
        float temp = tymax;
        tymax = tymin;
        tymin = temp;
    }

    if ((tmin > tymax) || (tymin > tmax)) 
        return false; 

    if (tymin > tmin) 
        tmin = tymin; 

    if (tymax < tmax) 
        tmax = tymax; 

    float tzmin = (min_bounds.z() - ray.org.z()) / ray.dir.z(); 
    float tzmax = (max_bounds.z() - ray.org.z()) / ray.dir.z(); 

    if (tzmin > tzmax){
        float temp = tzmax;
        tzmax = tzmin;
        tzmin = temp;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false; 

    if (tzmin > tmin) 
        tmin = tzmin; 

    if (tzmax < tmax) 
        tmax = tzmax; 

    return true; 
}