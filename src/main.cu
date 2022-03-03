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

__global__ void render(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner, curandState *rand){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= image_width) || (j >= image_height)) return;
  int pixel_index = j*image_width + i;

  curandState local_rand = rand[pixel_index];
  // Usage: curand_uniform(&local_rand) returns a float 0..1
  // https://docs.nvidia.com/cuda/curand/device-api-overview.html

  Vec2 uv = Vec2(float(i) / (image_width-1), float(j) / (image_height-1));
  Ray ray = Ray(Vec3(0,0,0), lower_left_corner + uv.x()*horizontal + uv.y()*vertical - Vec3(0,0,0));
  Vec3 result = Vec3(0.0, 0.0, 0.0);
  RayHit hit = trace(&ray);
  result = result + shade(&ray, hit);

  image[pixel_index] = result;
}

//As mentioned in Accelerated Ray Tracing in One Weekend (https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
//It's a good idea to seperate initialization and actual rendering if we want accurate performance numbers. 
__global__ void initKernels(int image_width, int image_height, unsigned long long rand_seed, curandState *rand){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= image_width) || (j >= image_height)) return;
  int pixel_index = j*image_width + i;

  curand_init(rand_seed, pixel_index, 0, &rand[pixel_index]);
}

__device__ Vec3 shade(Ray *ray, RayHit hit){
  if(hit.dist < 999999.0)
    return hit.normal;
  else
    return Vec3(0.0, 0.0, 0.0);
}

__device__ RayHit trace(Ray *ray){
  RayHit hit;
  hit.dist = 9999999.0f; //TODO: This should be float.max.
  hit.normal = Vec3(1,0,0);
  hit.pos = Vec3(1,0,0);
  intersectSphere(ray, &hit, Vec3( 0, 0,-1), 0.5f);
  return hit;
}

/* From Möller & Trumbore, Fast, Minimum Storage Ray/Triangle Intersection*/
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

  bestHit->dist = dot(edge2, qvec);
  float inv_det = 1.0 / det;
  bestHit->dist = bestHit->dist * inv_det;
  bestHit->uv.e[0] = bestHit->uv.e[0] * inv_det;
  bestHit->uv.e[1] = bestHit->uv.e[1] * inv_det;
  return true;
}

__device__ void intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius){
  Vec3 oc = ray->origin() - point;
  float a = dot(ray->direction(), ray->direction());
  float b = 2.0 * dot(oc, ray->direction());
  float c = dot(oc,oc) - radius*radius;
  float discriminant = b*b - 4*a*c;
  if(discriminant > 0.0){
    float t = (-b - sqrt(discriminant)) / (2.0*a);
    if(t < bestHit->dist && t > 0){
      bestHit->dist = t;
      bestHit->pos = ray->point_along_ray(t);

      bestHit->normal = normalize(bestHit->pos - point);
    }
  }
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
  int image_width = 256;
  int image_height = 256;

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
  render<<<blocks, threads>>>(ptr_img, image_width, image_height, horizontal, vertical, lower_left_corner, d_rand_state);
  cudaDeviceSynchronize();

  println("Render complete, writing to disk...");
  serializeImageBuffer(ptr_img, output_filename, image_width, image_height);
  println("Saved to disk.");

  cudaFree(ptr_img);
  return 0;
}