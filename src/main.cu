#include <iostream>
#include <math.h>
#include <cmath>
#include <stdio.h>

#include "vec3.h"
#include "vec2.h"
#include "ray.h"
#include "hit.h"
#include "main.h"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define PI 3.14159265359

__global__
void render(Vec3 *image){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= IMAGE_WIDTH) || (j >= IMAGE_HEIGHT)) return;
  int pixel_index = j*IMAGE_WIDTH + i;

  Vec2 uv = Vec2(float(i) / IMAGE_WIDTH * 2.0 - 1.0, float(j) / IMAGE_HEIGHT * 2.0 - 1.0);
  Ray ray = createCameraRay(uv);
  RayHit hit = trace(&ray);
  image[pixel_index] = normalize(abs(hit.normal));
}

__device__ RayHit trace(Ray *ray){
  RayHit hit;
  hit.dist = 100000.0f; //TODO: This should be float.max.
  hit.normal = Vec3(1,0,0);
  hit.pos = Vec3(1,0,0);

  intersectSphere(ray, &hit, Vec3( 0, 0,-5), 2.0f);
  intersectSphere(ray, &hit, Vec3( 0, 2,-5), 2.0f);
  intersectSphere(ray, &hit, Vec3( 0,-2,-5), 2.0f);
  intersectSphere(ray, &hit, Vec3( 2, 0,-6), 2.0f);
  intersectSphere(ray, &hit, Vec3(-2, 0,-6), 2.0f);
  return hit;
}

__device__ void intersectSphere(Ray *ray, RayHit *bestHit, Vec3 point, float radius){
  Vec3 oc = ray->origin() - point;
  float a = dot(ray->direction(), ray->direction());
  float b = 2.0 * dot(oc, ray->direction());
  float c = dot(oc,oc) - radius*radius;
  float discriminant = b*b - 4*a*c;
  if(discriminant > 0.0){
    float t = (-b - sqrt(discriminant)) / (2.0*a);
    if(t < bestHit->dist){
      bestHit->dist = t;
      bestHit->pos = ray->point_along_ray(t);
      bestHit->normal = normalize(bestHit->pos - point);
    }
  }
}

__device__ Ray createCameraRay(Vec2 uv){
  //TODO: Implement field of view, focal length etc..
  Vec3 origin(0,0,0); 
  Vec3 dir = Vec3(uv.x(), uv.y(), -1);
  dir = normalize(dir);
  return Ray(origin, dir);
}

int serializeImageBuffer(Vec3 *ptr_img, const char *fileName){
  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

  for (int j = IMAGE_HEIGHT-1; j >= 0; j--) {
    for (int i = 0; i < IMAGE_WIDTH; i++) {
      size_t pixel_index = j*IMAGE_WIDTH + i;
      //BUG: Make sure ptr_img values are 0..1.
      float r = abs(ptr_img[pixel_index].x());
      float g = abs(ptr_img[pixel_index].y());
      float b = abs(ptr_img[pixel_index].z());
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
  const char* output_filename = "output.ppm";
  if(argc > 0)
    output_filename = argv[1];
  
  const int img_size = IMAGE_HEIGHT * IMAGE_WIDTH;
  Vec3 *ptr_img;

  int threads_x = 8;
  int threads_y = 8;

  dim3 blocks(IMAGE_WIDTH/threads_x+1,IMAGE_HEIGHT/threads_y+1);
  dim3 threads(threads_x,threads_y);

  println("Initialization complete. Starting Rendering.");

  cudaMallocManaged(&ptr_img, img_size*sizeof(Vec3));
  render<<<blocks, threads>>>(ptr_img);
  cudaDeviceSynchronize();

  println("Render complete, writing to disk.");
  serializeImageBuffer(ptr_img, output_filename);
  println("Saved to disk.");

  cudaFree(ptr_img);
  return 0;
}