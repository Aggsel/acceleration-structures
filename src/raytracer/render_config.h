#pragma once

struct RenderConfig{
    public:
    int img_width;
    int img_height;
    int samples_per_pixel;
    int max_bounces;
    int seed;
    
    RenderConfig(int w, int h, int spp, int max_bounces, int seed){
        this->img_width = w;
        this->img_height = h;
        this->samples_per_pixel = spp;
        this->max_bounces = max_bounces;
        this->seed = seed;
    }

    RenderConfig() = default;

};