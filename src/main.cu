#include <iostream>
#include <math.h>
#include <stdio.h>

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100


__global__
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

void fileio(void){
  const int dimx = IMAGE_WIDTH, dimy = IMAGE_HEIGHT;
  int i, j;
  FILE *fp = fopen("first.ppm", "wb");
  fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
  for (j = 0; j < dimy; ++j){
    for (i = 0; i < dimx; ++i){
      static unsigned char color[3];
      color[0] = i % 256;  /* red */
      color[1] = j % 256;  /* green */
      color[2] = (i * j) % 256;  /* blue */
      fwrite(color, 1, 3, fp);
    }
  }
  fclose(fp);
}

int main(void){
  fileio();

  // int N = 1<<20;
  // float *x, *y;

  // // Allocate Unified Memory – accessible from CPU or GPU
  // cudaMallocManaged(&x, N*sizeof(float));
  // cudaMallocManaged(&y, N*sizeof(float));

  // // initialize x and y arrays on the host
  // for (int i = 0; i < N; i++) {
  //   x[i] = 1.0f;
  //   y[i] = 2.0f;
  // }

  // // Run kernel on 1M elements on the GPU
  // add<<<1, 1>>>(N, x, y);

  // // Wait for GPU to finish before accessing on host
  // cudaDeviceSynchronize();

  // // Check for errors (all values should be 3.0f)
  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = fmax(maxError, fabs(y[i]-3.0f));
  // std::cout << "Max error: " << maxError << std::endl;

  // // Free memory
  // cudaFree(x);
  // cudaFree(y);
  
  return 0;
}