#include <iostream>
#include <math.h>
#include <stdio.h>
#include "vec3.h"
#include "vec2.h"
#include "ray.h"

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256

__global__
void render(Vec3 *image){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= IMAGE_WIDTH) || (j >= IMAGE_HEIGHT)) return;
  int pixel_index = j*IMAGE_WIDTH + i;

  Vec2 uv = Vec2(float(i) / IMAGE_WIDTH, float(j) / IMAGE_WIDTH);
  image[pixel_index] = Vec3(uv.x(), uv.y(), 0.2);
}

__host__ __device__ Ray createCameraRay(Vec2 uv){
  return Ray(Vec3(0,0,0), Vec3(0,0,1));
}

int serializeImageBuffer(Vec3 *ptr_img, const char *fileName){
  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

  for (int j = IMAGE_HEIGHT-1; j >= 0; j--) {
    for (int i = 0; i < IMAGE_WIDTH; i++) {
      size_t pixel_index = j*IMAGE_WIDTH + i;
      float r = ptr_img[pixel_index].x();
      float g = ptr_img[pixel_index].y();
      float b = ptr_img[pixel_index].z();
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

  cudaFree(ptr_img);
  return 0;
}