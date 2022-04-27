#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include "math_util.h"
#include "vec3.h"

#include "third_party/cuda_helpers/helper_cuda.h"

class Image{
    int width, height;
    Vec3 *frame_buffer;

    public:
    Image(int width, int height){
        this->width = width;
        this->height = height;
    }

    ~Image(){
        free(frame_buffer);
    }

    bool copyFromDevice(Vec3* device_ptr, int size){
        assert(size == width * height);

        frame_buffer = (Vec3*)malloc(sizeof(Vec3)*size);
        if(!frame_buffer)
            return false;
        checkCudaErrors(cudaMemcpy(frame_buffer, device_ptr, sizeof(Vec3) * size, cudaMemcpyDeviceToHost));
        return true;
    }

    // Modified from https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file#C
    bool save(const char* filename){
        FILE *fp = fopen(filename, "w");
        fprintf(fp, "P3\n%d %d\n255\n", this->width, this->width);
        for (int j = this->height-1; j >= 0; j--) {
            for (int i = 0; i < this->width; i++) {
            size_t pixel_index = j*this->width + i;
            float r = clamp01(abs(frame_buffer[pixel_index].x()));
            float g = clamp01(abs(frame_buffer[pixel_index].y()));
            float b = clamp01(abs(frame_buffer[pixel_index].z()));
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            fprintf(fp, "%d %d %d\n", ir, ig, ib);
            }
        }
        fclose(fp);
        printf("%s saved to disk.\n", filename);
        return true;
    }
};