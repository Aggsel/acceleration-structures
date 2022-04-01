#pragma once

__global__ void render(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__global__ void init_kernels(int image_width, int image_height, curandState *rand);
__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__device__ Vec3 randomInUnitSphere(curandState *rand);
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2);
__device__ bool intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius);

int serializeImageBuffer(Vec3 *ptr_img, const char *fileName, int image_width, int image_height);

__device__ __host__ unsigned int mortonCode(Vec3 v);
__device__ __host__ inline unsigned int expandBits(unsigned int v);