#include <iostream>
#include <math.h>
#include <stdio.h>

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define PIXEL_STRIDE 3

class Vec3{
  float e[3];
  public:
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
};

__host__ __device__ Vec3 operator*(float lhs, Vec3 rhs){
  return Vec3(lhs*rhs.x(), lhs*rhs.y(), lhs*rhs.z());
}
__host__ __device__ Vec3 operator+(Vec3 lhs, Vec3 rhs){
  return Vec3(lhs.x()+rhs.x(), lhs.y()+rhs.y(), lhs.z()+rhs.z());
}

class Ray{
  Vec3 org;
  Vec3 dir;
  public:
    __device__ Ray() {}
    __device__ Ray(const Vec3 a, const Vec3 b) { org = a; dir = b; }
    __device__ Vec3 origin() const       { return org; }
    __device__ Vec3 direction() const    { return dir; }
    __device__ Vec3 point_at_parameter(float t) const { return org + t*dir; }
};

__global__
void render(Vec3 *image){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= IMAGE_WIDTH) || (j >= IMAGE_HEIGHT)) return;
  int pixel_index = j*IMAGE_WIDTH + i;

  float u = float(i) / IMAGE_WIDTH;
  float v = float(j) / IMAGE_WIDTH;

  image[pixel_index] = Vec3(u, v, 0.2);
}

//TODO: Make "Vec3* ptr_img" instead.
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
  Vec3 *ptr_img;
  int img_size = img_pixels*sizeof(Vec3);

  int threads_x = 8;
  int threads_y = 8;

  dim3 blocks(IMAGE_WIDTH/threads_x+1,IMAGE_HEIGHT/threads_y+1);
  dim3 threads(threads_x,threads_y);

  // Allocate 3ints per channel per pixel.
  cudaMallocManaged(&ptr_img, img_size);
  render<<<blocks, threads>>>(ptr_img);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  write_image_buffer_to_disk((float*)ptr_img, "output.ppm");

  // Free memory
  cudaFree(ptr_img);
  return 0;
}