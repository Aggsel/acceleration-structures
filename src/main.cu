#include <iostream>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "vec3.h"
#include "vec2.h"
#include "ray.h"
#include "hit.h"
#include "main.h"

#define PI 3.14159265359
#define EPSILON 0.000001

__global__
void render(Vec3 *image, int image_width, int image_height, Vec3 horizontal, Vec3 vertical, Vec3 lower_left_corner){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= image_width) || (j >= image_height)) return;
  int pixel_index = j*image_width + i;

  Vec2 uv = Vec2(float(i) / (image_width-1), float(j) / (image_height-1));
  Ray ray = Ray(Vec3(0,0,0), lower_left_corner + uv.x()*horizontal + uv.y()*vertical - Vec3(0,0,0));
  Vec3 result = Vec3(0.0, 0.0, 0.0);
  RayHit hit = trace(&ray);
  result = result + shade(&ray, hit);

  image[pixel_index] = result;
}

__device__ Vec3 shade(Ray *ray, RayHit hit){
  if(hit.dist < 999999.0)
    return Vec3(hit.uv.x(), hit.uv.y(), 1-hit.uv.x()-hit.uv.y());
  else
    return Vec3(0.0, 0.0, 0.0);
}

__device__ RayHit trace(Ray *ray){
  RayHit hit;
  hit.dist = 9999999.0f; //TODO: This should be float.max.
  hit.normal = Vec3(1,0,0);
  hit.pos = Vec3(1,0,0);
  intersectTri(ray, &hit, Vec3(-1, 0, -5), Vec3(1, 0, -5), Vec3(0, 2, -5));
  // intersectSphere(ray, &hit, Vec3( 0, 0,-5), 2.0f);
  // intersectSphere(ray, &hit, Vec3( 0, 2,-5), 2.0f);
  // intersectSphere(ray, &hit, Vec3( 0,-2,-5), 2.0f);
  // intersectSphere(ray, &hit, Vec3( 2, 0,-6), 2.0f);
  // intersectSphere(ray, &hit, Vec3(-2, 0,-6), 2.0f);
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
    if(t < bestHit->dist){
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
  //Set default values for filename and image size.
  char* output_filename = "output.ppm";
  int image_width = 512;
  int image_height = 512;

  float aspect_ratio = image_width / image_height;
  float viewport_height = 2.0;
  float viewport_width = aspect_ratio * viewport_height;
  float focal_length = 1.0;

  Vec3 origin = Vec3(0, 0, 0);
  Vec3 horizontal = Vec3(viewport_width, 0, 0);
  Vec3 vertical = Vec3(0, viewport_height, 0);
  Vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length);

  int img_size = image_width * image_height;
  Vec3 *ptr_img;

  int threads_x = 8;
  int threads_y = 8;

  dim3 blocks(image_width/threads_x+1,image_height/threads_y+1);
  dim3 threads(threads_x,threads_y);

  println("Initialization complete. Starting Rendering.");

  cudaMallocManaged(&ptr_img, img_size*sizeof(Vec3));
  render<<<blocks, threads>>>(ptr_img, image_width, image_height, horizontal, vertical, lower_left_corner);
  cudaDeviceSynchronize();

  println("Render complete, writing to disk.");
  serializeImageBuffer(ptr_img, output_filename, image_width, image_height);
  println("Saved to disk.");

  cudaFree(ptr_img);
  return 0;
}