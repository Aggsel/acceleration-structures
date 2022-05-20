#pragma once
#include "vec3.h"
#include "raytracer/ray.h"
#include "macros.h"

class ALIGN(16) AABB{
    public:
    Vec3 min_bounds;
    Vec3 max_bounds;
    __device__ __host__ AABB(Vec3 min_bounds, Vec3 max_bounds){
        this->min_bounds = min_bounds;
        this->max_bounds = max_bounds;
    }
    AABB() = default;
    __device__ __host__ void join(AABB otherAABB);
    __device__ __host__ void join(Vec3 vec);
    __device__ __host__ float surfaceArea();
    __device__ __host__ AABB extend();
    __device__ __host__ static AABB join(AABB aabb_1, AABB aabb_2);
    __device__ bool intersectRay(Ray ray);
};

namespace AABB_CONST{
const AABB inv_aabb = AABB(Vec3(FLT_MAX, FLT_MAX, FLT_MAX), Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
}

//TODO: Rewrite this to use SSE instructions.
__device__ __host__ AABB AABB::extend(){
    AABB extended = *this;
    float scale[3];
    Vec3 scene_scale;
    float eps = 0.0001;
    for (int i = 0; i < 3; i++) {
        scale[i] = this->max_bounds[i] - this->min_bounds[i];
        if (scale[i] < eps) {
            scene_scale[i] = eps;
        } else {
            scene_scale[i] = eps * scale[i];
        }
    }
    extended.min_bounds = extended.min_bounds - scene_scale;
    extended.max_bounds = extended.max_bounds + scene_scale;

    return extended;
}

__device__ __host__ void AABB::join(AABB otherAABB){
    this->min_bounds = min(this->min_bounds, otherAABB.min_bounds);
    this->max_bounds = max(this->max_bounds, otherAABB.max_bounds);
}

__device__ __host__ void AABB::join(Vec3 vec){
    this->min_bounds = min(this->min_bounds, vec);
    this->max_bounds = max(this->max_bounds, vec);
}

__device__ __host__ AABB AABB::join(AABB aabb_1, AABB aabb_2){
    AABB aabb;
    aabb.min_bounds = min(aabb_1.min_bounds, aabb_2.min_bounds);
    aabb.max_bounds = max(aabb_1.max_bounds, aabb_2.max_bounds);
    return aabb;
}

__device__ __host__ float AABB::surfaceArea(){
    Vec3 size = this->max_bounds - this->min_bounds;
    return max( (size.x() * size.y() + size.x() * size.z() + size.y() * size.z()) * 0.01, 0.0);
}

//https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__device__ bool AABB::intersectRay(Ray ray){
    Vec3 dirfrac;
    dirfrac[0] = 1.0f / ray.dir.x();
    dirfrac[1] = 1.0f / ray.dir.y();
    dirfrac[2] = 1.0f / ray.dir.z();
    float t1 = (this->min_bounds.x() - ray.org.x())*dirfrac.x();
    float t2 = (this->max_bounds.x() - ray.org.x())*dirfrac.x();
    float t3 = (this->min_bounds.y() - ray.org.y())*dirfrac.y();
    float t4 = (this->max_bounds.y() - ray.org.y())*dirfrac.y();
    float t5 = (this->min_bounds.z() - ray.org.z())*dirfrac.z();
    float t6 = (this->max_bounds.z() - ray.org.z())*dirfrac.z();

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if (tmax < 0)
        return false;
    if (tmin > tmax)
        return false;
    return true;
}