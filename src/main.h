#pragma once

__host__ __device__ Ray createCameraRay(Vec2 uv);
__host__ __device__ RayHit trace(Ray *ray);
__host__ __device__ float magnitude(Vec3 a);
__host__ __device__ Vec3 normalize(Vec3 v);
__host__ __device__ Vec3 abs(Vec3 a);
__host__ __device__ float dot(Vec3 a, Vec3 b);
__host__ __device__ void intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius);