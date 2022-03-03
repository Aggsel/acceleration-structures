#pragma once

__device__ RayHit trace(Ray *ray);
__global__ void init_kernels(int image_width, int image_height, curandState *rand);
__global__ void render(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner, curandState *rand);

__device__ Vec3 shade(Ray *ray, RayHit hit);
__device__ void intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius);
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2);
__host__ __device__ float magnitude(Vec3 a);
__host__ __device__ Vec3 normalize(Vec3 v);
__host__ __device__ Vec3 abs(Vec3 a);
__host__ __device__ float dot(Vec3 a, Vec3 b);