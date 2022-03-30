#pragma once
#include "vec3.h"
#include "raytracer/ray.h"

class AABB{
    public:
    Vec3 min_bounds;
    Vec3 max_bounds;
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

//https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__device__ __host__ bool AABB::intersect_ray(Ray ray){
    Vec3 dirfrac = ray.dir / 1.0;

    float t1 = (this->min_bounds.x() - ray.org.x())*dirfrac.x();
    float t2 = (this->max_bounds.x() - ray.org.x())*dirfrac.x();
    float t3 = (this->min_bounds.y() - ray.org.y())*dirfrac.y();
    float t4 = (this->max_bounds.y() - ray.org.y())*dirfrac.y();
    float t5 = (this->min_bounds.z() - ray.org.z())*dirfrac.z();
    float t6 = (this->max_bounds.z() - ray.org.z())*dirfrac.z();

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0){
        // t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax){
        // t = tmax;
        return false;
    }

    // t = tmin;
    return true;
}