#include <iostream>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "vec2.h"
#include "ray.h"
#include "hit.h"
#include "main.h"
#include "math_util.h"

#define PI 3.1415926535897932385
#define EPSILON 0.000001

__global__ void render(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner, curandState *rand, int max_depth, int spp){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= image_width) || (pixel_y >= image_height)) return;
  int pixel_index = pixel_y*image_width + pixel_x;

  curandState local_rand = rand[pixel_index];
  // Usage: curand_uniform(&local_rand) returns a float 0..1
  // https://docs.nvidia.com/cuda/curand/device-api-overview.html

  Vec3 result = Vec3(0.0, 0.0, 0.0);
  for (int i = 0; i < spp; i++){
    Vec2 uv = Vec2((pixel_x + curand_uniform(&local_rand)) / (image_width-1), (pixel_y+ curand_uniform(&local_rand)) / (image_height-1));
    // Vec2 uv = Vec2( float(i) / (image_width-1), float(j) / (image_height-1));
    Ray ray = Ray(Vec3(0,0,0), lower_left_corner + uv.x()*horizontal + uv.y()*vertical - Vec3(0,0,0));
    Vec3 out_col = color(&ray, &local_rand, max_depth);

    float r = clamp01(out_col.x());
    float g = clamp01(out_col.y());
    float b = clamp01(out_col.z());
    result = result + Vec3(r,g,b);
  }
  
  //Gamma correction
  float scale = 1.0 / spp;
  float r = sqrt(scale * result.x());
  float g = sqrt(scale * result.y());
  float b = sqrt(scale * result.z());
  result = Vec3(r,g,b);

  image[pixel_index] = result;
}

//As mentioned in Accelerated Ray Tracing in One Weekend (https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
//It's a good idea to seperate initialization and actual rendering if we want accurate performance numbers. 
__global__ void initKernels(int image_width, int image_height, unsigned long long rand_seed, curandState *rand){
  int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
  if((pixel_x >= image_width) || (pixel_y >= image_height)) return;
  int pixel_index = pixel_y*image_width + pixel_x;

  curand_init(rand_seed, pixel_index, 0, &rand[pixel_index]);
}

__device__ Vec3 color(Ray *ray, curandState *rand, int max_depth) {
  float cur_attenuation = 1.0f;

  for(int i = 0; i < max_depth; i++) {
    RayHit hit;
    //TODO: Seperate into a world object, containing an array of hittables.
    //      Also, only use the closest hit.
    if (intersectSphere(ray, &hit, Vec3(-0.8, 0, -1), 0.05f) || 
        intersectSphere(ray, &hit, Vec3(0, -0.8, -1), 0.05f) || 
        intersectSphere(ray, &hit, Vec3(0.8, 0, -1), 0.05f) || 
        intersectTri(ray, &hit, Vec3(-0.7, 0.2, -0.6), Vec3(0, -0.8, -1), Vec3(0.8, 0, -1))) {
      Vec3 target = hit.pos + hit.normal + randomInUnitSphere(rand);
      cur_attenuation *= 0.5f;
      ray->org = hit.pos;
      ray->dir = target-hit.pos;
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
__device__ bool intersectTri(Ray *ray, RayHit *bestHit, Vec3 v0, Vec3 v1, Vec3 v2){
  Vec3 edge1 = v1 - v0;
  Vec3 edge2 = v2 - v0;

  Vec3 pvec = cross(ray->direction(), edge2);
  float det = dot(edge1, pvec);
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

  bestHit->dist = dot(edge2, qvec);;
  float inv_det = 1.0 / det;
  bestHit->dist *= inv_det;
  bestHit->uv.e[0] *= inv_det;
  bestHit->uv.e[1] *= inv_det;
  bestHit->normal = cross(edge1, edge2);  //TODO: Lerp vertex normals instead.
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

int serializeImageBuffer(Vec3 *ptr_img, const char *fileName, int image_width, int image_height){
  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);

  for (int j = image_height-1; j >= 0; j--) {
    for (int i = 0; i < image_width; i++) {
      size_t pixel_index = j*image_width + i;
      float r = clamp01(abs(ptr_img[pixel_index].x()));
      float g = clamp01(abs(ptr_img[pixel_index].y()));
      float b = clamp01(abs(ptr_img[pixel_index].z()));
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      fprintf(fp, "%d %d %d\n", ir, ig, ib);
    }
  }

  fclose(fp);
  return 0;
}

void println(const char* str){
  std::cout << str << std::endl;
}

int main(int argc, char *argv[]){
  //Set default values for filename and image size.
  char* output_filename = "output.ppm";
  int image_width = 1280;
  int image_height = 720;

  int max_depth = 10;
  int samples_per_pixel = 500;

  float aspect_ratio = image_width / image_height;
  float viewport_height = 2.0;
  float viewport_width = aspect_ratio * viewport_height;
  float focal_length = 1.0;

  Vec3 origin = Vec3(0, 0, 0);
  Vec3 horizontal = Vec3(viewport_width, 0, 0);
  Vec3 vertical = Vec3(0, viewport_height, 0);
  Vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length);

  int threads_x = 8;
  int threads_y = 8;
  dim3 blocks(image_width/threads_x+1,image_height/threads_y+1);
  dim3 threads(threads_x,threads_y);

  curandState *d_rand_state;
  Vec3 *ptr_img;
  cudaMallocManaged(&d_rand_state, image_width * image_height*sizeof(curandState));
  cudaMallocManaged(&ptr_img, image_width * image_height*sizeof(Vec3));

  println("Initializing kernels...");
  initKernels<<<blocks, threads>>>(image_width, image_height, 1337, d_rand_state);
  println("Initialization complete. Starting Rendering...");
  render<<<blocks, threads>>>(ptr_img, image_width, image_height, horizontal, vertical, lower_left_corner, d_rand_state, max_depth, samples_per_pixel);
  cudaDeviceSynchronize();

  println("Render complete, writing to disk...");
  serializeImageBuffer(ptr_img, output_filename, image_width, image_height);
  println("Saved to disk.");

  cudaFree(ptr_img);
  cudaFree(d_rand_state);
  return 0;
}