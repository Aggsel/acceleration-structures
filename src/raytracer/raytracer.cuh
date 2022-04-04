#pragma once

#include <curand_kernel.h>  //CUDA random
#include "vec3.h"
#include "math_util.h"
#include "raytracer/render_config.h"
#include "third_party/cuda_helpers/helper_cuda.h"

__global__ void render_kernel(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__global__ void init_kernels(int image_width, int image_height, curandState *rand);
__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals);
__device__ Vec3 randomInUnitSphere(curandState *rand);
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2);
__device__ bool intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius);

__global__ void render_kernel(Vec3 *output_image, Camera cam, curandState *rand, RenderConfig config, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= config.img_width) || (pixel_y >= config.img_height)) return;
  int pixel_index = pixel_y*config.img_width + pixel_x;

  curandState local_rand = rand[pixel_index];

  Vec3 result = Vec3(0.0, 0.0, 0.0);
  for (int i = 0; i < config.samples_per_pixel; i++){
    Vec2 uv = Vec2((pixel_x + curand_uniform(&local_rand)) / (config.img_width-1), (pixel_y+ curand_uniform(&local_rand)) / (config.img_height-1));
    Ray ray = Ray(Vec3(0,0,0), normalize(cam.lower_left_corner + uv.x()*cam.horizontal + uv.y()*cam.vertical - Vec3(0,0,0)) );
    Vec3 out_col = color(&ray, &local_rand, config.max_bounces, vertices, triangles, vertex_count, normals);

    float r = clamp01(out_col.x());
    float g = clamp01(out_col.y());
    float b = clamp01(out_col.z());
    result = result + Vec3(r,g,b);
  }
  
  //Gamma correction
  float scale = 1.0 / config.samples_per_pixel;
  float r = sqrt(scale * result.x());
  float g = sqrt(scale * result.y());
  float b = sqrt(scale * result.z());
  result = Vec3(r,g,b);

  output_image[pixel_index] = result;
}

//As mentioned in Accelerated Ray Tracing in One Weekend (https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
//It's a good idea to seperate initialization and actual rendering if we want accurate performance numbers. 
__global__ void initKernels(int image_width, int image_height, unsigned long long rand_seed, curandState *rand){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= image_width) || (pixel_y >= image_height))
    return;
  int pixel_index = pixel_y*image_width + pixel_x;

  curand_init(rand_seed, pixel_index, 0, &rand[pixel_index]);
}

__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth, Vec3 *vertices, Triangle *triangles, int vertex_count, Vec3 *normals) {
  float cur_attenuation = 1.0f;
  for(int i = 0; i < max_depth; i++) {
    RayHit hit;
    bool was_hit = false;
    for (int j = 0; j < vertex_count/3; j++){ //TODO: Traverse acceleration structure.
      RayHit tempHit;

      if (!intersectTri(ray, &tempHit,  vertices[triangles[j].v0_index],
                                        vertices[triangles[j].v1_index],
                                        vertices[triangles[j].v2_index],
                                        normals [triangles[j].v0_index],
                                        normals [triangles[j].v1_index],
                                        normals [triangles[j].v2_index]))
        continue; //Did not hit triangle.
      
      if(tempHit.dist > hit.dist)
        continue; //Hit triangle but not closest intersection so far.

      hit.dist = tempHit.dist;
      hit.normal = tempHit.normal;
      hit.pos = tempHit.pos;
      hit.uv = tempHit.uv;
      was_hit = true;
    }

    if(was_hit){
      Vec3 target = hit.pos + hit.normal + randomInUnitSphere(rand);
      cur_attenuation *= 0.5f;
      ray->org = hit.pos;
      ray->dir = normalize(target - hit.pos);
      continue;
    }
    else {
      Vec3 unit_direction = normalize(ray->direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      Vec3 c = (1.0f-t)*Vec3(1.0, 1.0, 1.0) + t*Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0);
}

__device__ Vec3 randomInUnitSphere(curandState *rand){
  while(true){
    float x = (curand_uniform(rand) * 2.0) - 1.0;
    float y = (curand_uniform(rand) * 2.0) - 1.0;
    float z = (curand_uniform(rand) * 2.0) - 1.0;
    Vec3 p = Vec3(x, y, z);
    if(sqrMagnitude(p) >= 1)
      continue;
    return p;
  }
}

/* From Möller & Trumbore, Fast, Minimum Storage Ray/Triangle Intersection */
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2, Vec3 n0, Vec3 n1, Vec3 n2){
  Vec3 edge1 = v1 - v0;
  Vec3 edge2 = v2 - v0;

  Vec3 pvec = cross(ray->direction(), edge2);
  float det = dot(edge1, pvec);
  //Culling implementation
  if(det < EPSILON)
    return false;
  
  Vec3 tvec = ray->origin() - v0;
  bestHit->uv.e[0] = dot(tvec, pvec);
  if(bestHit->uv.x() < 0.0 || bestHit->uv.x() > det)
    return false;

  Vec3 qvec = cross(tvec, edge1);
  bestHit->uv.e[1] = dot(ray->direction(), qvec);
  if(bestHit->uv.y() < 0.0 || bestHit->uv.x() + bestHit->uv.y() > det)
    return false;

  float inv_det = 1.0 / det;
  bestHit->dist = dot(edge2, qvec) * inv_det;
  bestHit->uv.e[0] *= inv_det;
  bestHit->uv.e[1] *= inv_det;
  bestHit->normal = normalize(cross(edge1, edge2));
  bestHit->pos = ray->point_along_ray(bestHit->dist);

  //BUG: There's something funky with the normals when interpolating...
  // bestHit->normal = normalize(bestHit->uv.x()*n1 + bestHit->uv.y() * n2 + (1.0 - bestHit->uv.x() - bestHit->uv.y()) * n0);
  return true;
}

//From Shirleys Ray Tracing in One Weekend.
__device__ bool intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius){
  Vec3 oc = ray->origin() - point;
  float a = sqrMagnitude(ray->direction());
  float half_b = dot(oc, ray->direction());
  float c = sqrMagnitude(oc) - radius*radius;

  float discriminant = half_b*half_b - a*c;
  if (discriminant < 0) return false;
  float sqrtd = sqrt(discriminant);

  float root = (-half_b - sqrtd) / a;
  if (root < 0.00001 || 999999.0 < root) {
      root = (-half_b + sqrtd) / a;
      if (root < 0.00001 || 999999.0 < root)
          return false;
  }
  bestHit->dist = root;
  bestHit->pos = ray->point_along_ray(bestHit->dist);
  bestHit->normal = (bestHit->pos - point) / radius;
  return true;
}

class Raytracer{
    curandState *d_rand_state;
    Vec3 *ptr_device_img;
    RenderConfig config;

    dim3 threads, blocks;

    Vec3 *ptr_device_vertices;
    Vec3 *ptr_device_normals;
    Triangle *ptr_device_triangles;
    int index_count;

    public:
    Raytracer(RenderConfig render_config, Vec3 *device_vertex_buffer, Vec3 *normal_buffer, Triangle *triangle_buffer, int index_count){
        config = render_config;
        ptr_device_vertices = device_vertex_buffer;
        ptr_device_normals = normal_buffer;
        ptr_device_triangles = triangle_buffer;
        this->index_count = index_count;

        cudaMalloc(&d_rand_state, config.img_width * config.img_height*sizeof(curandState));
        cudaMalloc(&ptr_device_img, config.img_width * config.img_height*sizeof(Vec3));

        int threads_x = 8;
        int threads_y = 8;
        threads = dim3(threads_x,threads_y);
        blocks = dim3(config.img_width/threads_x+1,config.img_height/threads_y+1);

        initKernels<<<blocks, threads>>>(config.img_width, config.img_height, config.seed, d_rand_state);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~Raytracer(){
        cudaFree(ptr_device_img);
        cudaFree(d_rand_state);
    }

    Vec3* render(Camera cam){
        render_kernel<<<blocks, threads>>>(ptr_device_img, cam, d_rand_state, config, ptr_device_vertices, ptr_device_triangles, index_count, ptr_device_normals);
        checkCudaErrors(cudaDeviceSynchronize());
        return ptr_device_img;
    }
};