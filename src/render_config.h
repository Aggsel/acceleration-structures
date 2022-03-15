#pragma once

__device__ __host__ struct RenderConfig{
    RenderConfig(int w, int h, int spp, int max_bounces){
        this->img_width = w;
        this->img_height = h;
        this->samples_per_pixel = spp;
        this->max_bounces = max_bounces;
    }

    int img_width;
    int img_height;
    int samples_per_pixel;
    int max_bounces;
};