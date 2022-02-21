#include <iostream>
#include <math.h>
#include <stdio.h>

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define PIXEL_STRIDE 3
#define PIXEL_TYPE unsigned char

__global__
void render(float *image){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= IMAGE_WIDTH) || (j >= IMAGE_HEIGHT)) return;
  int pixel_index = j*IMAGE_WIDTH*3 + i*3;
  image[pixel_index + 0] = float(i) / IMAGE_WIDTH;
  image[pixel_index + 1] = float(j) / IMAGE_HEIGHT;
  image[pixel_index + 2] = 0.2;
}

int write_image_buffer_to_disk(float *ptr_img, const char *fileName){
  // Write image to disk
  FILE *fp = fopen(fileName, "w");
  fprintf(fp, "P3\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

  for (int j = IMAGE_HEIGHT-1; j >= 0; j--) {
    for (int i = 0; i < IMAGE_WIDTH; i++) {
      size_t pixel_index = j*PIXEL_STRIDE*IMAGE_WIDTH + i*PIXEL_STRIDE;
      float r = ptr_img[pixel_index + 0];
      float g = ptr_img[pixel_index + 1];
      float b = ptr_img[pixel_index + 2];
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      fprintf(fp, "%d %d %d\n", ir, ig, ib);
    }
  }

  fclose(fp);
  return 0;
}

int main(void){
  const int img_pixels = IMAGE_HEIGHT * IMAGE_WIDTH;
  float *ptr_img;
  int img_size = img_pixels*sizeof(float)*PIXEL_STRIDE;

  int tx = 8;
  int ty = 8;

  dim3 blocks(IMAGE_WIDTH/tx+1,IMAGE_HEIGHT/ty+1);
  dim3 threads(tx,ty);

  // Allocate 3ints per channel per pixel.
  cudaMallocManaged(&ptr_img, img_size);
  render<<<blocks, threads>>>(ptr_img);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  write_image_buffer_to_disk(ptr_img, "first.ppm");

  // Free memory
  cudaFree(ptr_img);
  return 0;
} 